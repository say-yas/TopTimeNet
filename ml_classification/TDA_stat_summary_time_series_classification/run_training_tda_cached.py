"""
run_training_tda_cached.py
Training runner for TDAEnd2EndNet using precomputed TDA feature caches.

Two modes
---------
MODE A (end2end): pass X_seg=(N,1,T), y_seg. Model is TDAEnd2EndNet, which
    computes the full TDA pipeline from raw segments on every forward pass.
MODE B (cached): pass features=(N,feat_dim), labels. Model is
    CachedGroupMLP, which consumes precomputed feature vectors directly.

Robustness and calibration features
------------------------------------
- Noise robustness sweep: run_cached_tda_training() accepts noise_levels
  (a list of sigma values, or None to disable). When provided, after each
  run's testing_step(), the test set is evaluated at each noise level. In
  cached mode, noise is added to the flat feature vector and
  _cached_robustness_sweep() (implemented in this file) is used; in
  end2end mode, noise is added to the raw time series and
  net.evaluate_robustness() is called directly. Files saved:
    robustness_sweep_run{N}.csv   per-sigma rows for run N
    robustness_all_runs.csv       all runs stacked, saved after all runs

- IQR feature clipping: if the feature cache contains clip bounds
  (clip_q_lo / clip_q_hi), they are loaded and applied to noisy feature
  vectors inside _cached_robustness_sweep(), matching the clipping
  applied at precompute time.

- Noise augmentation during training: when noise_aug_sigma > 0 (default
  0.05), the training split is augmented by concatenating a noisy copy of
  x_tr with Gaussian noise of that standard deviation. This teaches the
  GroupProjector's BatchNorm layers what shifted feature distributions
  look like, improving robustness at low noise levels.

- Temperature scaling (post-hoc calibration): after training, a scalar
  temperature T is fitted on the validation set via LBFGS to minimise
  cross-entropy loss on scaled logits (logits / T). T > 1 shrinks logit
  magnitudes, lowering confidence for better alignment with accuracy. T is
  saved in the CSV record and used inside the robustness sweep (logits
  are divided by T before softmax).

- ECE (Expected Calibration Error): each per-sigma record in the
  robustness sweep CSV includes "ece" (M=10 equal-width bins by default).
"""

from __future__ import annotations

import gc
import glob
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

try:
    import torchinfo
    _HAS_TORCHINFO = True
except ImportError:
    _HAS_TORCHINFO = False


from ml_classification.TDA_stat_summary_time_series_classification.nn_tda_stat_summary_model import (
    TDAEnd2EndNet,
    GroupProjector,
    ClassHead,
    EarlyStopper,
    _get_activation,
    _count,
    _init_weights,
    build_fusion,
    LifetimeStatsLayer,
    TDABettiExtractor,
    TDAPIExtractor,
)
import ml_classification.TDA_stat_summary_time_series_classification.nn_tda_stat_summary_train as tda_train
from ml_classification.TDA_stat_summary_time_series_classification.precompute_tda import (
    apply_clip_bounds,
)


# ============================================================================
# CachedGroupMLP
# ============================================================================

class CachedGroupMLP(nn.Module):
    """
    Accepts a flat (B, feat_dim) cached feature vector, splits it into the
    same 5 group tensors that TDAEnd2EndNet produces internally, then runs
    them through the identical learnable stack:

        GroupProjector -> fusion layer -> ClassHead
    """

    def __init__(
        self,
        n_hom_dims:    int             = 2,
        n_classes:     int             = 2,
        embed_dim:     int             = 32,
        fusion:        str             = "low_rank",
        n_heads:       int             = 4,
        n_attn_layers: int             = 1,
        ffn_dim:       int             = 0,
        rank:          int             = 8,
        dropout:       float           = 0.1,
        activation:    str             = "gelu",
        head_hidden:   Tuple[int, ...] = (64, 32),
    ):
        super().__init__()
        self.n_hom_dims = n_hom_dims

        group_dims = [
            4,
            n_hom_dims,
            n_hom_dims * LifetimeStatsLayer._S,
            n_hom_dims * TDABettiExtractor._S,
            n_hom_dims * TDAPIExtractor._S,
        ]
        self._group_dims  = group_dims
        self._feat_dim    = sum(group_dims)
        self._fusion_name = fusion

        starts = np.cumsum([0] + group_dims)
        self._slices = [(int(starts[i]), int(starts[i+1]))
                        for i in range(len(group_dims))]

        self.group_proj = GroupProjector(
            group_dims=group_dims, embed_dim=embed_dim,
            dropout=dropout, activation=activation,
        )

        fkw: dict = {"dropout": dropout}
        if fusion in ("mgta", "linear_attn"):
            fkw["n_heads"] = n_heads; fkw["n_layers"] = n_attn_layers
        if fusion == "mgta":
            fkw["ffn_dim"] = ffn_dim
        if fusion == "linear_attn":
            fkw["dim_ff"]  = ffn_dim
        if fusion == "low_rank":
            fkw["rank"] = rank; fkw["n_layers"] = n_attn_layers
            fkw["activation"] = activation

        n_groups = len(group_dims)
        self.fusion_layer = build_fusion(fusion, n_groups=n_groups, embed_dim=embed_dim, **fkw)
        self.classifier = ClassHead(
            in_dim=self.fusion_layer.out_dim, n_classes=n_classes,
            head_hidden=head_hidden, dropout=dropout, activation=activation,
        )
        self.group_proj.apply(_init_weights)
        self.classifier.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        groups = [x[:, s:e] for s, e in self._slices]
        tokens = self.group_proj(groups)
        fused  = self.fusion_layer(tokens)
        return self.classifier(fused)

    def count_parameters(self, verbose: bool = False) -> int:
        parts = {
            "GroupProjector  ": self.group_proj,
            f"FusionLayer({self._fusion_name:<10})": self.fusion_layer,
            "ClassHead       ": self.classifier,
        }
        total = 0
        if verbose:
            print(f"\n  {'Component':<36} {'Params':>8}")
            print("  " + "-" * 48)
            print(f"  feat_dim={self._feat_dim}  group_dims={self._group_dims}")
            print()
        for name, mod in parts.items():
            n = _count(mod); total += n
            if verbose:
                print(f"  {name:<36}  {n:>8,}")
        if verbose:
            print("  " + "-" * 48)
            print(f"  {'TOTAL TRAINABLE':<36}  {total:>8,}")
        return total

    @classmethod
    def from_config(cls, cfg: dict, n_classes: int) -> "CachedGroupMLP":
        return cls(
            n_hom_dims    = cfg.get("n_hom_dims",      2),
            n_classes     = n_classes,
            embed_dim     = cfg.get("embed_dim",       32),
            fusion        = cfg.get("fusion",    "low_rank"),
            n_heads       = cfg.get("n_heads",          4),
            n_attn_layers = cfg.get("n_attn_layers",    1),
            ffn_dim       = cfg.get("ffn_dim",          0),
            rank          = cfg.get("rank",             8),
            dropout       = cfg.get("dropout",        0.1),
            activation    = cfg.get("activation",   "gelu"),
            head_hidden   = tuple(cfg.get("head_hidden", [64, 32])),
        )


# ============================================================================
# Helpers
# ============================================================================

def _free_memory(objects, pathsave=None):
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    for p in glob.glob("checkpoint*.pt"):
        try: os.remove(p)
        except OSError: pass
    if pathsave:
        for p in glob.glob(os.path.join(pathsave, "checkpoint*.pt")):
            try: os.remove(p)
            except OSError: pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect(); gc.collect()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalise_features(x_train, x_val, x_test, norm_type):
    if norm_type in ("none", "None", None, "per-timestep"):
        return x_train, x_val, x_test
    if norm_type == "global":
        mu  = x_train.mean()
        std = x_train.std().clamp(min=1e-8)
    elif norm_type in ("per-channel", "per-feature"):
        mu  = x_train.mean(dim=0, keepdim=True)
        std = x_train.std(dim=0,  keepdim=True).clamp(min=1e-8)
    else:
        raise ValueError(f"Unknown norm_type '{norm_type}'.")
    return (x_train - mu) / std, (x_val - mu) / std, (x_test - mu) / std


def _fit_temperature(
    net:       nn.Module,
    loader_va: torch.utils.data.DataLoader,
    device:    torch.device,
    max_iter:  int = 50,
) -> float:
    """
    Post-hoc temperature scaling: fit scalar T on the validation set so
    that cross-entropy of (logits / T) is minimised. T > 1 gives softer
    confidences.

    Args
    ----
    net       : trained model (eval mode, any architecture with .forward())
    loader_va : validation DataLoader
    device    : torch device
    max_iter  : LBFGS iterations

    Returns
    -------
    T : float, optimal temperature (1.0 = no change)
    """
    net.eval()
    logits_all = []
    labels_all = []

    with torch.no_grad():
        for xb, yb in loader_va:
            logits_all.append(net(xb.to(device)).cpu())
            labels_all.append(yb.cpu())

    logits_all = torch.cat(logits_all)   # (N_val, n_classes)
    labels_all = torch.cat(labels_all)   # (N_val,)

    T   = nn.Parameter(torch.ones(1))
    opt = optim.LBFGS([T], lr=0.1, max_iter=max_iter)
    loss_fn = nn.CrossEntropyLoss()

    def _eval():
        opt.zero_grad()
        loss = loss_fn(logits_all / T.clamp(min=0.5, max=5.0), labels_all)
        loss.backward()
        return loss

    opt.step(_eval)
    t_val = T.clamp(min=0.5, max=5.0).item()
    print(f"  Calibration temperature T = {t_val:.4f}")
    return t_val


def _compute_ece(
    preds: torch.Tensor,
    confs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) with equal-width confidence bins.

    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args
    ----
    preds  : (N,) int tensor of predicted class indices
    confs  : (N,) float tensor of max softmax probabilities
    labels : (N,) int tensor of true labels
    n_bins : number of equal-width bins over [0, 1]

    Returns
    -------
    ece : float in [0, 1]
    """
    N          = len(labels)
    bin_edges  = torch.linspace(0.0, 1.0, n_bins + 1)
    ece        = 0.0

    for i in range(n_bins):
        lo   = bin_edges[i].item()
        hi   = bin_edges[i + 1].item()
        mask = (confs >= lo) & (confs < hi)
        if i == n_bins - 1:              # include right edge in last bin
            mask = (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = (preds[mask] == labels[mask]).float().mean().item()
        bin_conf = confs[mask].mean().item()
        ece     += (mask.float().sum().item() / N) * abs(bin_conf - bin_acc)

    return ece


def _cached_robustness_sweep(
    net:          CachedGroupMLP,
    x_test:       torch.Tensor,
    y_test:       torch.Tensor,
    noise_levels: list,
    batch_size:   int,
    seed:         int,
    device:       torch.device,
    norm_mu:      Optional[torch.Tensor] = None,
    norm_std:     Optional[torch.Tensor] = None,
    temperature:  float = 1.0,
    clip_q_lo:    Optional[torch.Tensor] = None,
    clip_q_hi:    Optional[torch.Tensor] = None,
    ece_bins:     int = 10,
) -> pd.DataFrame:
    """
    Noise robustness sweep for CachedGroupMLP.

    Noise is added to the flat (unnormalised, unclipped) feature vectors,
    then IQR clipping is applied, then z-score normalisation (if used
    during training), then temperature-scaled softmax.

    Args
    ----
    net          : trained CachedGroupMLP (already in eval mode)
    x_test       : (N, feat_dim) raw (pre-clip, pre-norm) float tensor on CPU
    y_test       : (N,) long tensor on CPU
    noise_levels : list of sigma values
    batch_size   : forward-pass batch size
    seed         : RNG seed for reproducible noise
    device       : torch device
    norm_mu      : training mean for re-normalisation (or None)
    norm_std     : training std  for re-normalisation (or None)
    temperature  : calibration temperature (default 1.0, disabled)
    clip_q_lo    : (feat_dim,) lower clip bound (or None)
    clip_q_hi    : (feat_dim,) upper clip bound (or None)
    ece_bins     : number of ECE calibration bins

    Returns
    -------
    pd.DataFrame indexed by sigma with columns:
        sigma, snr_db, accuracy, mean_confidence, pct_low_conf, ece
    """
    net.eval()
    rng = torch.Generator()
    rng.manual_seed(seed)

    records   = []
    n_levels  = len(noise_levels)
    n_samples = x_test.shape[0]

    clip_active = (clip_q_lo is not None) and (clip_q_hi is not None)
    print(f"  Cached robustness sweep: {n_levels} sigma levels  N={n_samples}")
    print(f"  IQR clip: {'ON' if clip_active else 'OFF'}  "
          f"temperature T={temperature:.4f}  ECE bins={ece_bins}")

    for i, sigma in enumerate(noise_levels):
        if sigma == 0.0:
            x_noisy = x_test.clone()
        else:
            noise   = torch.zeros_like(x_test).normal_(0.0, sigma, generator=rng)
            x_noisy = x_test + noise

        if clip_active:
            x_noisy = apply_clip_bounds(x_noisy, clip_q_lo, clip_q_hi)

        if norm_mu is not None and norm_std is not None:
            x_noisy = (x_noisy - norm_mu) / norm_std

        all_preds  = []
        all_confs  = []
        all_logits = []

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                xb     = x_noisy[start: start + batch_size].to(device)
                logits = net(xb)
                probs  = torch.softmax(logits / temperature, dim=1)
                conf, pred = probs.max(dim=1)
                all_preds.append(pred.cpu())
                all_confs.append(conf.cpu())
                all_logits.append(logits.cpu())

        preds = torch.cat(all_preds)
        confs = torch.cat(all_confs)

        acc       = (preds == y_test).float().mean().item()
        mean_conf = confs.mean().item()
        pct_low   = (confs < 0.5).float().mean().item() * 100.0
        snr_db    = 10.0 * np.log10(1.0 / (sigma ** 2)) if sigma > 0 else float("inf")
        ece       = _compute_ece(preds, confs, y_test, n_bins=ece_bins)

        records.append({
            "sigma"           : sigma,
            "snr_db"          : snr_db,
            "accuracy"        : acc,
            "mean_confidence" : mean_conf,
            "pct_low_conf"    : pct_low,
            "ece"             : ece,
        })
        print(f"  [{i+1:2d}/{n_levels}]  sigma={sigma:.3f}  SNR={snr_db:+.1f} dB  "
              f"acc={acc:.4f}  conf={mean_conf:.4f}  "
              f"low_conf={pct_low:.1f}%  ECE={ece:.4f}")

    return pd.DataFrame(records).set_index("sigma")


# ============================================================================
# Main runner
# ============================================================================

def run_cached_tda_training(
    features             = None,
    labels               = None,
    X_seg                = None,
    y_seg                = None,
    classes              = None,
    device               = None,
    n_classes:     int   = 2,
    cfg:           dict  = None,
    test_size:     float = 0.3,
    val_size:      float = 0.1,
    batch_size:    int   = 128,
    num_cpus:      int   = 1,
    lr:            float = 1e-3,
    num_epochs:    int   = 900,
    patience:      int   = 20,
    opt:           str   = "adam",
    muon_lr:       float = 0.02,
    verbose:       bool  = False,
    pathsave:      str   = "./",
    weights              = None,
    norm_type:     str   = "none",
    num_training:  int   = 10,
    embed_dim:     int   = 32,
    fusion:        str   = "low_rank",
    n_heads:       int   = 4,
    n_attn_layers: int   = 1,
    ffn_dim:       int   = 0,
    rank:          int   = 8,
    head_hidden:   Tuple = (64, 32),
    dropout:       float = 0.1,
    activation:    str   = "gelu",
    n_hom_dims:    int   = 2,
    seg_len:       int   = 200,
    reliability_threshold: float = 0.6,
    label_smoothing:       float = 0.1,
    grad_clip_norm:        float = 1.0,
    base_seed:             int   = 42,
    cache_path:            str   = None,
    layout:                dict  = None,
    noise_levels:          list  = None,
    noise_batch_size:      int   = 32,
    noise_aug_sigma:       float = 0.05,
    _clip_q_lo:            Optional[torch.Tensor] = None,
    _clip_q_hi:            Optional[torch.Tensor] = None,
    use_temperature_scaling: bool = True,
    ece_bins:              int   = 10,
    # When True, return a list of per-run artifacts (state_dict, idx_te,
    # norm stats, run_seed, clean test accuracy) instead of None. Purely
    # additive: default False preserves existing behaviour for every
    # other caller of this function.
    return_run_artifacts:  bool  = False,
):
    """
    Train num_training independent runs.

    MODE A (end2end): pass X_seg=(N,1,T), y_seg. Model=TDAEnd2EndNet.
    MODE B (cached) : pass features=(N,feat_dim), labels. Model=CachedGroupMLP.

    Robustness and calibration behaviour
    -------------------------------------
    IQR clip bounds are loaded from the cache payload (if present) and
    applied inside the robustness sweep to noisy feature vectors.

    noise_aug_sigma > 0 (default 0.05): training data is augmented with a
    noisy copy of x_tr. Teaches the GroupProjector's BatchNorm layers what
    shifted feature distributions look like. Set to 0.0 to disable.

    use_temperature_scaling=True: after training each run, a scalar T is
    fitted on the validation set via LBFGS. Logits are divided by T in the
    robustness sweep. T is saved in the CSV record.

    ECE is added to every per-sigma row in the robustness sweep CSV.

    return_run_artifacts=True: also returns a list (one entry per
    completed run) of dicts with keys "model_state_dict" (CPU, detached
    copy), "idx_te", "norm_mu", "norm_std", "run_seed", "run_idx",
    "n_classes", "clean_test_accuracy", enough to reconstruct the exact
    trained model and its preprocessing state for further use (e.g. a
    raw-signal noise sweep) without reimplementing any part of this
    function's training logic elsewhere.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _opt     = cfg.get("optimizer", opt)        if cfg else opt
    _muon_lr = float(cfg.get("muon_lr", muon_lr)) if cfg else muon_lr

    if noise_levels is None and cfg is not None:
        noise_levels = cfg.get("noise_levels", None)

    if cfg is not None:
        noise_aug_sigma = float(cfg.get("noise_aug_sigma", noise_aug_sigma))

    # determine mode
    if features is not None and labels is not None:
        mode     = "cached"
        data_x   = (features.float() if isinstance(features, torch.Tensor)
                    else torch.tensor(features, dtype=torch.float32))
        data_y   = np.array(labels)
        feat_dim = data_x.shape[1]
        print(f"[run_cached_tda] MODE B (cached)  feat_dim={feat_dim}  N={len(data_x)}")
    elif X_seg is not None and y_seg is not None:
        mode     = "end2end"
        data_x   = X_seg
        data_y   = np.array(y_seg) if not isinstance(y_seg, np.ndarray) else y_seg
        feat_dim = None
        print(f"[run_cached_tda] MODE A (end2end)  shape={tuple(X_seg.shape)}")
    else:
        raise ValueError(
            "Provide either (features, labels) for cached mode "
            "or (X_seg, y_seg) for end2end mode."
        )

    # load cache layout and clip bounds
    clip_q_lo = _clip_q_lo
    clip_q_hi = _clip_q_hi

    if mode == "cached" and cache_path and os.path.exists(cache_path):
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            if layout is None:
                layout = payload.get("layout", {})
                print("[run_cached_tda] Feature layout from cache:")
                for grp, (s, e) in layout.items():
                    print(f"  [{s:>3}:{e:>3}]  {grp}  ({e-s} dims)")
            if clip_q_lo is None and "clip_q_lo" in payload:
                clip_q_lo = payload["clip_q_lo"]
                clip_q_hi = payload["clip_q_hi"]
                print(f"[run_cached_tda] IQR clip bounds loaded from cache")
            elif clip_q_lo is None:
                print("[run_cached_tda] No clip bounds in cache, "
                      "re-run precompute_tda with force=True to add them")
        except Exception as exc:
            print(f"[run_cached_tda] Could not load cache payload: {exc}")
            layout = {}

    run_records      = []
    arr_acc, arr_f1  = [], []
    arr_gm, arr_pr   = [], []
    arr_re           = []
    computation_time = 0.0
    csv_path         = os.path.join(pathsave, "results_cached_tda.csv")

    all_sweep_dfs = []
    run_artifacts = []

    for idx in range(num_training):
        print(f"\n{'='*60}")
        print(f"  Run {idx+1}/{num_training}  [{mode}]  fusion={fusion}")
        print(f"{'='*60}")

        run_seed = base_seed + idx
        _set_seed(run_seed)
        print(f"  seed={run_seed}  opt={_opt}  "
              f"label_smoothing={label_smoothing}  "
              f"noise_aug_sigma={noise_aug_sigma if mode == 'cached' else 'n/a (end2end)'}")

        params = dict(
            test_size             = test_size,
            val_size              = val_size,
            batch_size            = batch_size,
            num_cpus              = num_cpus,
            lr                    = lr,
            num_epochs            = num_epochs,
            patience              = patience,
            n_classes             = n_classes,
            norm_type             = norm_type,
            verbose               = verbose,
            num_training          = num_training,
            reliability_threshold = reliability_threshold,
            label_smoothing       = label_smoothing,
            grad_clip_norm        = grad_clip_norm,
        )

        start = time.time()
        net = optimizer = loss_fn = early_stopper = scheduler = trainer = None
        x_tr = x_va = x_te = y_tr = y_va = y_te = None
        ds_tr = ds_va = ds_te = None
        loader_tr = loader_va = loader_te = None
        cm = cm_test = None
        norm_mu_saved = norm_std_saved = None
        x_te_raw = y_te_raw = None
        temperature = 1.0   # default: no scaling

        try:
            # split
            idx_all = np.arange(len(data_y))
            idx_tr, idx_te, y_tr, y_te = train_test_split(
                idx_all, data_y,
                test_size=test_size, random_state=run_seed,
                stratify=data_y, shuffle=True,
            )
            idx_tr, idx_va, y_tr, y_va = train_test_split(
                idx_tr, y_tr,
                test_size=val_size, random_state=run_seed,
                stratify=y_tr, shuffle=True,
            )

            x_tr = data_x[idx_tr]
            x_va = data_x[idx_va]
            x_te = data_x[idx_te]

            # keep raw (pre-clip, pre-norm) test features for the
            # robustness sweep
            if noise_levels is not None and mode == "cached":
                x_te_raw = x_te.clone()
                y_te_raw = torch.tensor(y_te, dtype=torch.long)

            if mode == "cached":
                # apply IQR clip bounds to all splits (fitted on the full
                # dataset at precompute time, so applying here is safe)
                if clip_q_lo is not None:
                    x_tr = apply_clip_bounds(x_tr, clip_q_lo, clip_q_hi)
                    x_va = apply_clip_bounds(x_va, clip_q_lo, clip_q_hi)
                    x_te = apply_clip_bounds(x_te, clip_q_lo, clip_q_hi)

                # noise augmentation: concatenate a noisy copy of the
                # training split
                if noise_aug_sigma > 0.0:
                    rng_aug = torch.Generator()
                    rng_aug.manual_seed(run_seed + 9999)
                    noise_aug = torch.zeros_like(x_tr).normal_(
                        0.0, noise_aug_sigma, generator=rng_aug
                    )
                    x_tr_noisy = x_tr + noise_aug
                    if clip_q_lo is not None:
                        x_tr_noisy = apply_clip_bounds(x_tr_noisy, clip_q_lo, clip_q_hi)
                    x_tr = torch.cat([x_tr, x_tr_noisy], dim=0)
                    y_tr = np.concatenate([y_tr, y_tr])
                    print(f"  Noise augmentation: "
                          f"sigma={noise_aug_sigma}  train size {len(y_tr)//2} -> {len(y_tr)}")

                # capture normalisation stats before applying
                if norm_type not in ("none", "None", None, "per-timestep"):
                    if norm_type == "global":
                        norm_mu_saved  = x_tr.mean()
                        norm_std_saved = x_tr.std().clamp(min=1e-8)
                    else:
                        norm_mu_saved  = x_tr.mean(dim=0, keepdim=True)
                        norm_std_saved = x_tr.std(dim=0, keepdim=True).clamp(min=1e-8)
                x_tr, x_va, x_te = _normalise_features(x_tr, x_va, x_te, norm_type)

            # datasets and loaders
            ds_tr = tda_train.data_to_tensor(x_tr.numpy(), y_tr)
            ds_va = tda_train.data_to_tensor(x_va.numpy(), y_va)
            ds_te = tda_train.data_to_tensor(x_te.numpy(), y_te)

            del x_tr, x_va, x_te, y_tr, y_va, y_te
            x_tr = x_va = x_te = y_tr = y_va = y_te = None

            loader_tr = torch.utils.data.DataLoader(
                ds_tr, batch_size=batch_size, shuffle=True,  num_workers=0)
            loader_va = torch.utils.data.DataLoader(
                ds_va, batch_size=batch_size, shuffle=False, num_workers=0)
            loader_te = torch.utils.data.DataLoader(
                ds_te, batch_size=batch_size, shuffle=False, num_workers=0)

            # build model
            arch_kwargs = dict(
                n_classes     = n_classes,
                embed_dim     = embed_dim,
                fusion        = fusion,
                n_heads       = n_heads,
                n_attn_layers = n_attn_layers,
                ffn_dim       = ffn_dim,
                rank          = rank,
                head_hidden   = head_hidden,
                dropout       = dropout,
                activation    = activation,
            )
            if mode == "cached":
                net = CachedGroupMLP(n_hom_dims=n_hom_dims, **arch_kwargs).to(device)
            else:
                if cfg is not None:
                    net = TDAEnd2EndNet.from_config(cfg, n_classes).to(device)
                else:
                    net = TDAEnd2EndNet(seg_len=seg_len, **arch_kwargs).to(device)

            if _HAS_TORCHINFO:
                try:
                    in_shape = (
                        (batch_size, feat_dim)        if mode == "cached"
                        else (batch_size, 1, seg_len)
                    )
                    print(torchinfo.summary(net, input_size=in_shape, device=device))
                except Exception:
                    net.count_parameters(verbose=True)
            else:
                net.count_parameters(verbose=True)

            n_params = net.count_parameters()
            strcomplexity = (
                f"idx={idx+1}: {mode}  fusion={fusion}  "
                f"embed={embed_dim}  rank={rank}  "
                f"ls={label_smoothing}  clip={grad_clip_norm}  "
                f"params={n_params:,}"
            )
            print(f"  {strcomplexity}")

            # optimizer, loss, scheduler, early stopper
            optimizer = (
                optim.Adam(net.parameters(),  lr=lr, weight_decay=1e-4)
                if _opt == "adam"
                else optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
            )

            loss_fn = nn.CrossEntropyLoss(
                weight          = weights.to(device) if weights is not None else None,
                label_smoothing = label_smoothing,
            )

            ckpt = os.path.join(pathsave, f"checkpoint_cached_idx{idx+1}.pt")
            early_stopper = EarlyStopper(verbose=verbose, path=ckpt, patience=patience)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", factor=0.9,
                patience=max(1, patience - 4), threshold=1e-8,
            )

            trainer = tda_train.TrainTDAStat(
                net, optimizer, loss_fn,
                early_stopper, scheduler, device, params,
            )

            # training
            (train_losses, val_losses,
             train_accs,   val_accs,
             cm,
             train_f1s,    val_f1s,
             train_gmeans, val_gmeans,
             train_pres,   val_pres,
             train_recs,   val_recs,
             train_rels,   val_rels,
             train_neus,   val_neus,
             val_cls_rel) = trainer.run_training(loader_tr, loader_va)

            for metric, tr_v, vl_v in [
                ("loss",        train_losses, val_losses),
                ("accuracy",    train_accs,   val_accs),
                ("f1",          train_f1s,    val_f1s),
                ("reliability", train_rels,   val_rels),
            ]:
                tda_train.plot_progress(
                    strcomplexity + f"  {metric}", metric, tr_v, vl_v,
                    save_path=os.path.join(pathsave, f"{metric}_cached_idx{idx+1}.png"),
                )
                plt.close("all")

            # temperature scaling
            if use_temperature_scaling and mode == "cached":
                temperature = _fit_temperature(net, loader_va, device)
            else:
                temperature = 1.0

            # test
            (correct, total, acc_test, f1_test, gm_test,
             pr_test, re_test, rel_test, neu_test,
             cls_rel_test, cm_test) = trainer.testing_step(loader_te)

            end = time.time()
            computation_time = end - start

            print(
                f"\nRun {idx+1} | acc={acc_test:.4f}  f1={f1_test:.4f}  "
                f"gmean={gm_test:.4f}  rel={rel_test:.4f}  "
                f"neutral={neu_test:.1f}%  time={computation_time:.1f}s  "
                f"T={temperature:.3f}"
            )

            arr_acc.append(acc_test); arr_f1.append(f1_test)
            arr_gm.append(gm_test);   arr_pr.append(pr_test)
            arr_re.append(re_test)

            tda_train.visualize_confusion_matrix(
                cm_test.numpy().astype(float), classes, correct, total,
                path=os.path.join(pathsave, f"confusion_matrix_cached_idx{idx+1}.png"),
            )
            plt.close("all")

            # noise robustness sweep
            if noise_levels is not None:
                print(f"\n  Noise robustness sweep (run {idx + 1}) ...")
                net.eval()

                if mode == "cached":
                    df_sweep = _cached_robustness_sweep(
                        net          = net,
                        x_test       = x_te_raw,        # raw features
                        y_test       = y_te_raw,
                        noise_levels = noise_levels,
                        batch_size   = noise_batch_size,
                        seed         = run_seed,
                        device       = device,
                        norm_mu      = norm_mu_saved,
                        norm_std     = norm_std_saved,
                        temperature  = temperature,
                        clip_q_lo    = clip_q_lo,
                        clip_q_hi    = clip_q_hi,
                        ece_bins     = ece_bins,
                    )
                else:
                    # TDAEnd2EndNet: noise on raw time series
                    x_test_all = torch.cat([b[0] for b in loader_te], dim=0)
                    y_test_all = torch.cat([b[1] for b in loader_te], dim=0)
                    df_sweep = net.evaluate_robustness(
                        x_test       = x_test_all,
                        y_test       = y_test_all,
                        noise_levels = noise_levels,
                        batch_size   = noise_batch_size,
                        seed         = run_seed,
                    )
                    del x_test_all, y_test_all

                df_sweep["run_idx"]     = idx + 1
                df_sweep["mode"]        = mode
                df_sweep["fusion"]      = fusion
                df_sweep["opt"]         = _opt
                df_sweep["temperature"] = temperature
                all_sweep_dfs.append(df_sweep.reset_index())

                sweep_path = os.path.join(pathsave, f"robustness_sweep_run{idx + 1}.csv")
                df_sweep.to_csv(sweep_path)
                print(f"  Robustness sweep saved -> {sweep_path}")
                plt.close("all")

                del df_sweep

            # CSV record
            layout_str = str({grp: f"{s}:{e}" for grp, (s, e) in (layout or {}).items()})
            record = {
                "run_idx"              : idx + 1,
                "run_seed"             : run_seed,
                "mode"                 : mode,
                "modeltype"            : (
                    "CachedGroupMLP" if mode == "cached" else "TDAEnd2EndNet"
                ),
                "computation_time_s"   : round(computation_time, 2),
                "n_params"             : n_params,
                "fusion"               : fusion,
                "embed_dim"            : embed_dim,
                "rank"                 : rank,
                "n_attn_layers"        : n_attn_layers,
                "n_heads"              : n_heads,
                "feat_dim"             : feat_dim or "n/a",
                "feature_layout"       : layout_str,
                "n_hom_dims"           : n_hom_dims,
                "n_classes"            : n_classes,
                "head_hidden"          : str(head_hidden),
                "dropout"              : dropout,
                "activation"           : activation,
                "label_smoothing"      : label_smoothing,
                "grad_clip_norm"       : grad_clip_norm,
                "norm_type"            : norm_type,
                "lr"                   : lr,
                "muon_lr"              : _muon_lr if _opt == "muon" else None,
                "batch_size"           : batch_size,
                "num_epochs"           : num_epochs,
                "patience"             : patience,
                "opt"                  : _opt,
                "test_size"            : test_size,
                "val_size"             : val_size,
                "reliability_threshold": reliability_threshold,
                "noise_levels_tested"  : str(noise_levels) if noise_levels else None,
                "noise_aug_sigma"      : noise_aug_sigma,
                "temperature"          : round(temperature, 4),
                "iqr_clip_applied"     : clip_q_lo is not None,
                "accuracy_test"        : round(acc_test, 4),
                "f1_test"              : round(f1_test,  4),
                "gmean_test"           : round(gm_test,  4),
                "precision_test"       : round(pr_test,  4),
                "recall_test"          : round(re_test,  4),
                "reliability_test"     : round(rel_test, 4),
                "neutral_pct_test"     : round(neu_test, 4),
                "final_train_loss"     : round(train_losses[-1], 4),
                "final_val_loss"       : round(val_losses[-1],   4),
                "final_train_acc"      : round(train_accs[-1],   4),
                "final_val_acc"        : round(val_accs[-1],     4),
                "final_train_f1"       : round(train_f1s[-1],    4),
                "final_val_f1"         : round(val_f1s[-1],      4),
                "final_train_rel"      : round(train_rels[-1],   4),
                "final_val_rel"        : round(val_rels[-1],     4),
                "final_val_neutral"    : round(val_neus[-1],     4),
                "n_epochs_trained"     : len(train_losses),
                **{f"rel_class_{(classes or list(range(n_classes)))[c]}":
                   round(r, 4) for c, r in cls_rel_test.items()},
            }
            run_records.append(record)
            pd.DataFrame(run_records).to_csv(csv_path, index=False)
            print(f"  Saved -> {csv_path}")

            # capture this run's artifact before the finally block's
            # _free_memory() call discards net/optimizer/etc. Clone and
            # move every tensor to CPU so nothing here can be invalidated
            # by whatever _free_memory does to the original objects.
            if return_run_artifacts:
                run_artifacts.append({
                    "model_state_dict": {
                        k: v.clone().detach().cpu() for k, v in net.state_dict().items()
                    },
                    "idx_te": idx_te,
                    "norm_mu": norm_mu_saved.clone().detach().cpu() if norm_mu_saved is not None else None,
                    "norm_std": norm_std_saved.clone().detach().cpu() if norm_std_saved is not None else None,
                    "run_seed": run_seed,
                    "run_idx": idx + 1,
                    "n_classes": n_classes,
                    "clean_test_accuracy": float(acc_test),
                    "temperature": float(temperature),
                })

        finally:
            _free_memory(
                [net, optimizer, loss_fn, early_stopper,
                 scheduler, trainer,
                 x_tr, x_va, x_te, y_tr, y_va, y_te,
                 ds_tr, ds_va, ds_te,
                 loader_tr, loader_va, loader_te,
                 cm, cm_test,
                 x_te_raw, y_te_raw],
                pathsave=pathsave,
            )
            plt.close("all")

    # combined robustness CSV
    if all_sweep_dfs:
        combined_path = os.path.join(pathsave, "robustness_all_runs.csv")
        pd.concat(all_sweep_dfs, ignore_index=True).to_csv(combined_path, index=False)
        print(f"\n  Combined robustness CSV -> {combined_path}")

    # summary
    print(f"\n{'-'*55}")
    print(f"  Summary over {num_training} runs  [{mode}  fusion={fusion}]:")
    for name, arr in [
        ("accuracy",  arr_acc), ("f1",        arr_f1),
        ("gmean",     arr_gm),  ("precision",  arr_pr),
        ("recall",    arr_re),
    ]:
        if arr:
            print(f"  {name:12s}: {np.mean(arr):.4f} +/- {np.std(arr):.4f}")
    print(f"{'-'*55}")

    if run_records:
        summary = {
            "run_idx"       : "MEAN +/- STD",
            "mode"          : mode,
            "fusion"        : fusion,
            "accuracy_test" : f"{np.mean(arr_acc):.4f} +/- {np.std(arr_acc):.4f}",
            "f1_test"       : f"{np.mean(arr_f1):.4f}  +/- {np.std(arr_f1):.4f}",
            "gmean_test"    : f"{np.mean(arr_gm):.4f}  +/- {np.std(arr_gm):.4f}",
            "precision_test": f"{np.mean(arr_pr):.4f}  +/- {np.std(arr_pr):.4f}",
            "recall_test"   : f"{np.mean(arr_re):.4f}  +/- {np.std(arr_re):.4f}",
        }
        pd.concat(
            [pd.DataFrame(run_records), pd.DataFrame([summary])],
            ignore_index=True,
        ).to_csv(csv_path, index=False)
    print(f"Final CSV -> {csv_path}")

    # additive-only: every other caller ignores this function's return
    # value (None) already, so returning run_artifacts only when
    # explicitly requested changes nothing for them.
    if return_run_artifacts:
        return run_artifacts
    return computation_time
