"""
raw_signal_robustness_sweep.py
Evaluate TopTimeNet's robustness to noise injected into the raw time
series, as opposed to the feature-level sweep in run_training_tda_cached.py
(_cached_robustness_sweep), which perturbs the already-computed 42-dim
feature vectors directly.

Why this script exists
-----------------------
The feature-level sweep only tests the robustness of the learnable stage
(GroupProjector, fusion, classifier) to perturbed topological summary
statistics. It never touches the Takens embedding or persistent homology
computation, so it cannot establish:
  (a) robustness of the complete raw-signal to feature to classification
      pipeline, or
  (b) whether that robustness is attributable to the stability properties
      of persistent homology itself.

This script closes that gap: for each noise level sigma, Gaussian noise is
added directly to the raw (B, 1, T) test segments, and the full feature
pipeline is recomputed from scratch: Takens embedding, persistent
homology (ripser), and all five feature branches (geometric, entropy,
lifetime, Betti, persistence image), using the same layer classes that
precompute_tda.py uses to build the training cache. Only after this full
recomputation is the trained classifier evaluated.

For direct comparison, the script also re-runs the existing feature-level
sweep on the same trained model and test set, so both robustness curves
(raw-signal vs. feature-level) appear together in the output.

Three modes
-----------
--mode all     (default) Single-process behaviour: for each of --num_runs
               independent runs, train a fresh model, then run both
               sweeps over all noise levels sequentially, in one process.
               Simplest option; fine for a quick local check.

--mode train   Train one model (run_seed = base_seed + --run_idx) and
               save a persistent "run artifact" (weights, normalization
               stats, test-split indices) to --artifact. Does not run any
               sweep. This is the expensive, parallelizable-across-runs
               half of the pipeline.

--mode sweep   Load a run artifact saved by --mode train and evaluate
               exactly one noise level (--sigma), for both the raw-signal
               and feature-level sweeps, appending the result to
               --sweep_out. This is the expensive, parallelizable-
               across-(run, sigma)-pairs half: since raw-signal
               recomputation is the true bottleneck (a full ripser pass
               per sample per sigma), splitting sigma values across
               separate SLURM array tasks, not just runs, gives much
               better wall-clock parallelism than --mode all alone.

train and sweep are meant to be driven by a two-stage SLURM pipeline (see
submit_raw_signal_robustness.sh): stage 1 submits one array task per run
(--mode train), stage 2 submits one array task per (run, sigma) pair
(--mode sweep), and a final merge job stacks all the small per-(run,sigma)
CSVs into the same combined format --mode all would have produced.

Usage
-----
    # single-process, everything in one run:
    python raw_signal_robustness_sweep.py --config best_config_small.json \
        --num_runs 5 --noise_levels 0.0,0.05,0.1,0.2,0.5,1.0 --out_dir ./out

    # two-stage (see submit_raw_signal_robustness.sh for the SLURM wrapper):
    python raw_signal_robustness_sweep.py --config best_config_small.json \
        --mode train --run_idx 1 --base_seed 12345 --artifact ./run1.pt

    python raw_signal_robustness_sweep.py --config best_config_small.json \
        --mode sweep --artifact ./run1.pt --run_idx 1 --sigma 0.1 \
        --sweep_out ./run1_sigma0.1.csv

Notes
-----
- Recomputing persistent homology from scratch for every sample at every
  noise level is far more expensive than the feature-level sweep alone.
  Use --mode train / --mode sweep with the SLURM wrapper for anything
  beyond a quick local sanity check.
- The script trains its own model instance(s) from the given config,
  since the existing training pipeline does not persist trained weights
  after each run. Training calls run_cached_tda_training() directly (the
  same function used by the main results pipeline), and reproduces its
  actual optimizer-selection behaviour: that function's real branch is
  `optim.Adam if _opt == "adam" else optim.AdamW`, with no separate Muon
  path, so any config with "optimizer": "muon" is, in practice, trained
  with AdamW. This script intentionally matches that behaviour so the
  model trained here for the robustness sweep matches the model whose
  numbers are reported elsewhere. If the main results are later retrained
  with a genuine Muon path, update this script's optimizer branch to
  match.
- IQR clip bounds are loaded from the existing TDA cache (clip_q_lo,
  clip_q_hi), matching precompute_tda.py's convention: the same bounds
  fit on the clean training set are applied to recomputed noisy features,
  exactly as for the feature-level sweep. This keeps the two sweeps
  comparable; only the noise injection point differs.
- Training uses the cache's own label array (payload["labels"]) rather
  than re-deriving labels from a fresh raw H5 reload, so --mode train
  never needs to touch the raw dataset (only --mode sweep and --mode all
  do, since only they need the actual raw signal for noise injection).
  This also guarantees --mode train and --mode sweep agree exactly on
  which label corresponds to which row index.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# project imports (same package layout as the rest of the pipeline)
from ml_classification.TDA_stat_summary_time_series_classification.nn_tda_stat_summary_model import (
    TakensLayer,
    PointCloudStatsLayer,
    RipserPHLayer,
    PersistenceEntropyLayer,
    LifetimeStatsLayer,
    BettiCurveLayer,
    PILayer,
    TDABettiExtractor,
    TDAPIExtractor,
    EarlyStopper,
)
from ml_classification.TDA_stat_summary_time_series_classification.run_training_tda_cached import (
    CachedGroupMLP,
    _normalise_features,
    _compute_ece,
    _cached_robustness_sweep,
    # The actual training entry point, called directly rather than
    # reimplemented, so the trained model is guaranteed produced by the
    # same code path as the main results pipeline. Requires
    # run_training_tda_cached.py's return_run_artifacts support.
    run_cached_tda_training,
)
import ml_classification.TDA_stat_summary_time_series_classification.nn_tda_stat_summary_train as tda_train
from ml_classification.TDA_stat_summary_time_series_classification.precompute_tda import (
    apply_clip_bounds,
)

import ml_classification.utils.read_data_from_h5 as read_data_from_h5
import ml_classification.utils.pad_truncate_tensor as pad_truncate_tensor
import ml_classification.utils.segment_time_series as segmenting_data
# Imported directly (not reimplemented) so class balancing is bit-for-bit
# identical to what main_train_time_series_tda_stat_summary.py runs before
# calling precompute_and_cache(). This matters because balance_dataset()
# ends with an in-place rng.shuffle(), so the cache's row order is a
# shuffled permutation of the raw, unbalanced segment order: an index
# computed from splitting the cache's row order only lines up with the
# same row when this exact function and random_seed are reused on a
# freshly re-segmented raw signal array.
from ml_classification.TDA_stat_summary_time_series_classification.main_train_time_series_tda_stat_summary import (
    balance_dataset,
)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Raw-signal-level robustness sweep for a TopTimeNet best_config.json"
    )
    p.add_argument("--config", required=True, help="Path to best_config.json / best_config_small.json")
    p.add_argument("--mode", choices=["all", "train", "sweep"], default="all",
                    help="'all': single-process, everything. 'train': train one model and save an "
                         "artifact. 'sweep': load an artifact and evaluate exactly one sigma.")
    p.add_argument("--out_dir", default="./raw_robustness_results",
                    help="Used only in --mode all")
    p.add_argument("--num_runs", type=int, default=5,
                    help="Independent training runs. Used only in --mode all "
                         "(in --mode train/sweep, one run = one --run_idx invocation)")
    p.add_argument("--noise_levels", default="0.0,0.025,0.05,0.075,0.1,0.2,0.5,1.0",
                    help="Used only in --mode all")
    p.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for TDA feature recomputation (keep small; ripser is costly)")
    p.add_argument("--base_seed", type=int, default=None,
                    help="Override cfg['random_seed'] if set")
    p.add_argument("--noise_aug_sigma", type=float, default=None,
                    help="Override cfg['noise_aug_sigma'] if set. best_config*.json's "
                         "own noise_aug_sigma may not match the value actually used to "
                         "train the reported main results; pass the value your training "
                         "run actually used (e.g. --noise_aug_sigma 0.05) to reproduce "
                         "that model. Otherwise this script may train a less robust "
                         "model than the one being reported.")
    p.add_argument("--use_temperature_scaling", default=None, choices=["true", "false"],
                    help="Override cfg['use_temperature_scaling'] if set (pass 'true' "
                         "or 'false'). best_config*.json's own value may not match what "
                         "actually trained the reported main results. Without the "
                         "correct value here, ECE and mean_confidence are computed with "
                         "temperature=1.0 (no scaling) instead of the true fitted value; "
                         "accuracy is unaffected either way, since argmax is "
                         "temperature-invariant.")
    # --mode train / --mode sweep
    p.add_argument("--artifact", default=None,
                    help="[train] path to save the trained run artifact. "
                         "[sweep] path to load it from.")
    p.add_argument("--run_idx", type=int, default=None,
                    help="[train] run_seed = base_seed + run_idx. "
                         "[sweep] copied into the output CSV for bookkeeping.")
    p.add_argument("--sigma", type=float, default=None,
                    help="[sweep] the single noise level to evaluate")
    p.add_argument("--sweep_out", default=None,
                    help="[sweep] CSV path to write this (run, sigma) result to")
    args = p.parse_args()

    if args.mode == "train" and (args.artifact is None or args.run_idx is None):
        p.error("--mode train requires --artifact and --run_idx")
    if args.mode == "sweep" and (args.artifact is None or args.sigma is None or args.sweep_out is None):
        p.error("--mode sweep requires --artifact, --sigma, and --sweep_out")
    return args


# ============================================================================
# Data loading (mirrors load_and_preprocess() in search_best_config.py)
# ============================================================================

def load_raw_segments(cfg: dict, target_n: Optional[int] = None) -> Tuple[torch.Tensor, np.ndarray, list]:
    """Reproduce the exact raw (B, 1, T) segmented dataset used at precompute
    time, so that indices line up 1:1 with the cached feature matrix.
    Only needed when raw signal access is required (--mode all / --mode
    sweep); --mode train never calls this.

    Self-detects whether the cache was built from the balanced or
    unbalanced segment array, rather than assuming one or the other:
    precompute_and_cache() caches whatever X_seg/y_seg it is given by its
    caller, and whether that caller applied class balancing first depends
    on which script built the cache. A cache built before
    cfg['balance_strategy'] was added or changed can stay stale and
    unbalanced even if the current config says otherwise. This function
    tries the unbalanced segmentation first, then (if balancing is
    configured) the balanced version, and returns whichever one's length
    matches target_n (the cache's actual row count), raising a clear,
    actionable error if neither matches rather than a downstream shape
    mismatch.

    If target_n is None, no matching is attempted and class balancing is
    applied whenever cfg['balance_strategy'] != 'none' (only safe if you
    already know the cache's provenance).
    """
    h5_path = os.path.join(cfg["path_data"], cfg["h5_filename"])
    df = read_data_from_h5.read_data(h5_path)

    for state in cfg.get("exclude_states", []):
        df = df[df["state"] != state]
    keep_states = cfg.get("keep_states", [])
    if keep_states:
        df = df[df["state"].isin(keep_states)]
        if len(df) == 0:
            raise ValueError(f"No rows remain after keep_states={keep_states}")

    class_labels = sorted(df["state"].unique())
    state_to_label = {s: i for i, s in enumerate(class_labels)}
    df["label"] = df["state"].map(state_to_label)

    out = pad_truncate_tensor.make_tensors(df, seq_len=cfg["length_series"])
    X_full = out["X"].unsqueeze(1)
    y_full = out["y"]

    seg_dur = cfg["segmentation_duration"]
    seg_out = segmenting_data.segment_data(X_full.squeeze(1), y_full, segment_duration=seg_dur)
    X_seg_unbal = seg_out["X"].unsqueeze(1)     # (N, 1, T), before balancing
    y_seg_unbal = seg_out["y"]                  # (N,)
    y_unbal_np = y_seg_unbal.numpy() if isinstance(y_seg_unbal, torch.Tensor) else np.array(y_seg_unbal)

    balance_strategy = cfg.get("balance_strategy", "none")

    if target_n is None:
        if balance_strategy != "none":
            X_seg, ynumpy = balance_dataset(
                X_seg_unbal, y_unbal_np, strategy=balance_strategy,
                random_seed=cfg.get("random_seed", 42),
            )
        else:
            X_seg, ynumpy = X_seg_unbal, y_unbal_np
        print(f"[load_raw_segments] X_seg={tuple(X_seg.shape)}  classes={class_labels}  "
              f"balance_strategy={balance_strategy}  (target_n not given, no auto-detection)")
        return X_seg, np.array(ynumpy), class_labels

    # target_n given: try unbalanced first, then balanced (if configured),
    # and use whichever matches the cache's actual length.
    if len(X_seg_unbal) == target_n:
        print(f"[load_raw_segments] X_seg={tuple(X_seg_unbal.shape)}  classes={class_labels}  "
              f"matched target_n={target_n} using unbalanced segments "
              f"(cache was built without class balancing, despite "
              f"balance_strategy='{balance_strategy}' in the config).")
        return X_seg_unbal, np.array(y_unbal_np), class_labels

    if balance_strategy != "none":
        X_seg_bal, y_bal_np = balance_dataset(
            X_seg_unbal, y_unbal_np, strategy=balance_strategy,
            random_seed=cfg.get("random_seed", 42),
        )
        if len(X_seg_bal) == target_n:
            print(f"[load_raw_segments] X_seg={tuple(X_seg_bal.shape)}  classes={class_labels}  "
                  f"matched target_n={target_n} using balanced segments "
                  f"(balance_strategy='{balance_strategy}').")
            return X_seg_bal, np.array(y_bal_np), class_labels
    else:
        X_seg_bal, y_bal_np = None, None

    # Neither matched: fail with enough detail to diagnose by hand.
    msg = (
        f"[load_raw_segments] Could not match cache length target_n={target_n} to "
        f"either the unbalanced (N={len(X_seg_unbal)}) or "
    )
    if X_seg_bal is not None:
        msg += f"balanced (N={len(X_seg_bal)}, strategy='{balance_strategy}') "
    else:
        msg += f"balanced (balance_strategy='none', not attempted) "
    msg += (
        f"segmentation of the raw H5 data. The cache may have been built with a "
        f"different segmentation_duration, exclude_states/keep_states, length_series, "
        f"or balance_strategy than the current config, or from a different dataset "
        f"file. Re-run precompute_tda.py with force_recompute=True, or manually "
        f"inspect tda_cache.pt's provenance before proceeding."
    )
    raise ValueError(msg)


def load_cache(cfg: dict):
    """Load the cached (clean) feature matrix, labels, IQR clip bounds, and
    layout. This is the only data source --mode train needs."""
    cache_path = cfg.get("tda_cache_path", os.path.join(cfg["path_save"], "tda_cache.pt"))
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    features = payload["features"].float()
    labels = np.asarray(payload["labels"])
    layout = payload.get("layout", {})
    clip_q_lo = payload.get("clip_q_lo", None)
    clip_q_hi = payload.get("clip_q_hi", None)
    if clip_q_lo is None:
        print("[load_cache] WARNING: no IQR clip bounds in cache, "
              "recomputed features will not be clipped, unlike the feature-level sweep.")
    print(f"[load_cache] features={tuple(features.shape)}  "
          f"clip_bounds={'yes' if clip_q_lo is not None else 'no'}")
    return features, labels, layout, clip_q_lo, clip_q_hi


# ============================================================================
# Full TDA feature recomputation (mirrors precompute_tda.py exactly)
# ============================================================================

class FullTDAPipeline:
    """Bundles the same non-parametric TDA layers used by precompute_tda.py,
    so that features recomputed here are, by construction, computed
    identically to how the training cache was built. The only difference
    is the (possibly noisy) input signal."""

    def __init__(self, cfg: dict, batch_size: int = 32):
        n_hom = cfg.get("n_hom_dims", 2)
        self.n_hom = n_hom
        self.batch_size = batch_size

        self.takens = TakensLayer(dim=cfg.get("takens_dim", 2), delay=cfg.get("takens_delay", 5))
        self.pc_stats = PointCloudStatsLayer()
        self.ph = RipserPHLayer(maxdim=n_hom - 1, max_workers=cfg.get("ph_workers", 4))
        self.ph_ent = PersistenceEntropyLayer(n_hom_dims=n_hom)
        self.lt_stats = LifetimeStatsLayer(n_hom_dims=n_hom)
        self.betti = BettiCurveLayer(n_hom_dims=n_hom, n_bins=cfg.get("n_betti_bins", 50))
        self.pi_layer = PILayer(
            n_hom_dims=n_hom, n_pi_bins=cfg.get("n_pi_bins", 20), sigma=cfg.get("pi_sigma", 0.1)
        )
        self.b_ext = TDABettiExtractor(n_hom_dims=n_hom, n_betti_bins=cfg.get("n_betti_bins", 50))
        self.p_ext = TDAPIExtractor(n_hom_dims=n_hom, n_pi_bins=cfg.get("n_pi_bins", 20))
        self._calibrated = False

    def calibrate(self, X_seg_full: torch.Tensor, warmup_batch_size: int) -> None:
        """Fit the persistence-image imager once, using the same warm-up
        procedure precompute_tda.py uses to build the training cache: the
        first `warmup_batch_size` samples of the full, clean (pre-split,
        pre-noise) segmented dataset. This calibration is then reused,
        unchanged, across every subsequent call to compute(), across all
        sigma levels and across all runs if the same FullTDAPipeline
        instance is reused.

        This must be called once, on clean data, before any sweep.
        PILayer's PersistenceImager.fit() sets the persistence-image pixel
        grid (birth_range / pers_range) from whatever diagrams it is first
        given. Refitting the imager fresh at every sigma level would mean
        every sigma level, including sigma=0 with no injected noise at
        all, is evaluated on a different pixel grid than the one the model
        was actually trained on, since the cached-feature training
        pipeline calibrates once, upfront, on clean training data
        (precompute_tda.py's own warm-up). Calibrating once here, with
        clean data, before any sweep, ensures every sigma level (including
        sigma=0) is evaluated on the same fixed pixel grid the model was
        trained and evaluated on.
        """
        dev = torch.device("cpu")
        warmup_x = X_seg_full[: min(warmup_batch_size, len(X_seg_full))]
        with torch.no_grad():
            pcs_w = self.takens(warmup_x)
            pc_w = self.pc_stats(pcs_w, dev)
            diams_w = pc_w[:, 0].cpu().numpy()
            dgms_w = self.ph(pcs_w)
            # Betti layer has no fitted state (stateless per call); kept
            # here only for exact fidelity with precompute_tda.py's warm-up
            # sequence. The call that actually matters is the PI layer below.
            _ = self.betti(dgms_w, dev, diameters=diams_w)
            _ = self.pi_layer(dgms_w, dev)
        self._calibrated = True
        print(f"  [FullTDAPipeline] Calibrated PI imager on the first {len(warmup_x)} "
              f"clean (pre-split, pre-noise) samples, matching precompute_tda.py's "
              f"own warm-up procedure. This calibration is now fixed for every "
              f"sigma level.")

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """x : (N, 1, T) raw (possibly noisy) segments -> (N, feat_dim) features."""
        if not self._calibrated:
            raise RuntimeError(
                "FullTDAPipeline.compute() called before calibrate(). Call "
                "calibrate(X_seg_full, warmup_batch_size) once with clean, "
                "pre-noise, pre-split data before computing features for any "
                "sigma level, so every sigma level (including sigma=0) shares "
                "the same persistence-image calibration the model was "
                "actually trained on."
            )
        dev = torch.device("cpu")
        N = x.shape[0]
        all_feats = []
        for start in range(0, N, self.batch_size):
            xb = x[start: start + self.batch_size]
            with torch.no_grad():
                pcs = self.takens(xb)
                pc_t = self.pc_stats(pcs, dev)
                diameters = pc_t[:, 0].cpu().numpy()
                diagrams = self.ph(pcs)
                ent_t = self.ph_ent(diagrams, dev)
                lt_t = self.lt_stats(diagrams, dev)
                betti_t = self.betti(diagrams, dev, diameters=diameters)
                pi_t = self.pi_layer(diagrams, dev)
                b_stats = self.b_ext(betti_t)
                p_stats = self.p_ext(pi_t)
                feats = torch.cat([pc_t, ent_t, lt_t, b_stats, p_stats], dim=1)
            all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)


# ============================================================================
# Model construction (shared by training and sweep-only reconstruction)
# ============================================================================

def build_model_from_cfg(cfg: dict, n_classes: int, device: torch.device) -> CachedGroupMLP:
    return CachedGroupMLP(
        n_hom_dims=cfg.get("n_hom_dims", 2), n_classes=n_classes,
        embed_dim=cfg.get("embed_dim", 32), fusion=cfg.get("fusion", "low_rank"),
        n_heads=cfg.get("n_heads", 4), n_attn_layers=cfg.get("n_attn_layers", 1),
        ffn_dim=cfg.get("ffn_dim", 0), rank=cfg.get("rank", 8),
        dropout=cfg.get("dropout", 0.1), activation=cfg.get("activation", "gelu"),
        head_hidden=tuple(cfg.get("head_hidden", [64, 32])),
    ).to(device)


# ============================================================================
# Raw-signal-level robustness sweep
# ============================================================================

def raw_signal_sweep(
    net: nn.Module, tda: FullTDAPipeline, X_test_raw: torch.Tensor, y_test: torch.Tensor,
    noise_levels: List[float], seed: int, device: torch.device,
    norm_mu: Optional[torch.Tensor], norm_std: Optional[torch.Tensor], norm_type: str,
    clip_q_lo: Optional[torch.Tensor], clip_q_hi: Optional[torch.Tensor],
    temperature: float = 1.0,
    ece_bins: int = 10,
) -> pd.DataFrame:
    net.eval()
    rng = torch.Generator(); rng.manual_seed(seed)
    records = []
    n_levels = len(noise_levels)

    print(f"  Raw-signal robustness sweep: {n_levels} sigma level(s)  N={len(X_test_raw)}  "
          f"temperature T={temperature:.4f}")
    for i, sigma in enumerate(noise_levels):
        t0 = time.time()
        if sigma == 0.0:
            x_noisy_raw = X_test_raw.clone()
        else:
            noise = torch.zeros_like(X_test_raw).normal_(0.0, sigma, generator=rng)
            x_noisy_raw = X_test_raw + noise

        # Full recomputation of the 42-dim feature vector from the noisy
        # raw signal: this is the step the feature-level sweep skips. The
        # PI imager is not reset or refit here; it was calibrated once,
        # upfront, on clean data (see FullTDAPipeline.calibrate()), and
        # that same fixed calibration is reused for every sigma level so
        # the pixel grid never changes with the noise level.
        feats_noisy = tda.compute(x_noisy_raw)

        if clip_q_lo is not None:
            feats_noisy = apply_clip_bounds(feats_noisy, clip_q_lo, clip_q_hi)
        if norm_mu is not None and norm_type not in ("none", "None", None, "per-timestep"):
            feats_noisy = (feats_noisy - norm_mu) / norm_std

        with torch.no_grad():
            logits = net(feats_noisy.to(device))
            # Temperature scaling, matching _cached_robustness_sweep's own
            # softmax(logits / temperature) exactly. Does not affect
            # accuracy (argmax is temperature-invariant for T>0), only
            # confidence and ECE.
            probs = torch.softmax(logits / temperature, dim=1)
            conf, pred = probs.max(dim=1)

        preds, confs = pred.cpu(), conf.cpu()
        acc = (preds == y_test).float().mean().item()
        mean_conf = confs.mean().item()
        pct_low = (confs < 0.5).float().mean().item() * 100.0
        snr_db = 10.0 * np.log10(1.0 / (sigma ** 2)) if sigma > 0 else float("inf")
        ece = _compute_ece(preds, confs, y_test, n_bins=ece_bins)

        elapsed = time.time() - t0
        records.append({
            "sigma": sigma, "snr_db": snr_db, "accuracy": acc,
            "mean_confidence": mean_conf, "pct_low_conf": pct_low, "ece": ece,
        })
        print(f"  [{i+1:2d}/{n_levels}]  sigma={sigma:.3f}  SNR={snr_db:+.1f} dB  "
              f"acc={acc:.4f}  conf={mean_conf:.4f}  ECE={ece:.4f}  ({elapsed:.1f}s)")

    return pd.DataFrame(records).set_index("sigma")


# ============================================================================
# --mode train
# ============================================================================

def run_train_mode(cfg: dict, args, base_seed: int, device: torch.device) -> None:
    """Calls run_cached_tda_training() directly, rather than a
    hand-reimplemented training loop. This guarantees the model trained
    here is produced by the same code path (optimizer selection,
    clip-bounds ordering, noise augmentation, class-weight computation,
    seeding) as the pipeline that generates the main reported results.

    Seed convention: run_cached_tda_training's internal loop computes
    run_seed = base_seed + idx for idx in range(num_training). Since this
    is called here with num_training=1 (idx is always 0), passing
    base_seed = cfg['random_seed'] + (args.run_idx - 1) reproduces exactly
    the run_seed that the (args.run_idx)-th run of a single sequential
    num_training=N call (using cfg['random_seed'] as its base_seed) would
    have used, matching the convention the main results pipeline itself
    uses.
    """
    features, labels, layout, clip_q_lo, clip_q_hi = load_cache(cfg)
    n_classes = int(len(np.unique(labels)))

    # Class-name labels for confusion-matrix plotting inside
    # run_cached_tda_training (visualize_confusion_matrix requires a real
    # sequence, not None). Derived from cfg['keep_states'], sorted the
    # same way main_train_time_series_tda_stat_summary.py derives them
    # (sorted(df['state'].unique())): keep_states already filters to
    # exactly the classes present in this dataset, so this reproduces the
    # same class-index to name mapping without re-reading the raw H5 file
    # just for label plotting.
    classes = sorted(cfg.get("keep_states", [])) or [f"class_{i}" for i in range(n_classes)]

    # Class weights computed once from the full cache's label
    # distribution, matching main_train_time_series_tda_stat_summary.py's
    # class_weights_cache computation exactly, rather than recomputing
    # per-split weights inside a reimplemented training loop.
    present = np.unique(labels)
    w_present = class_weight.compute_class_weight("balanced", classes=present, y=labels)
    w_full = np.zeros(n_classes, dtype=np.float32)
    w_full[present] = w_present
    weights = torch.tensor(w_full, dtype=torch.float)

    run_seed_base = cfg.get("random_seed", 42) + (args.run_idx - 1)

    artifact_dir = os.path.dirname(os.path.abspath(args.artifact)) or "."
    # Must be unique per run_idx: this directory (and therefore the
    # EarlyStopper checkpoint file inside it) would otherwise be shared
    # across every parallel SLURM array task, since artifact_dir is the
    # same directory for all tasks (only the artifact filename differs),
    # causing concurrent tasks to read and write the identical checkpoint
    # path simultaneously and risking silent cross-task corruption.
    ckpt_dir = os.path.join(artifact_dir, f"tmp_run_cached_output_{args.run_idx}")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"[train] run_idx={args.run_idx}  base_seed(passed)={run_seed_base}  "
          f"-> internal run_seed={run_seed_base}  n_classes={n_classes}  classes={classes}")

    artifacts = run_cached_tda_training(
        features=features, labels=labels, classes=classes, device=device,
        n_classes=n_classes, cfg=cfg,
        test_size=cfg.get("test_size", 0.1), val_size=cfg.get("val_size", 0.1),
        batch_size=cfg.get("batch_size", 128), num_cpus=1,
        lr=cfg.get("lr", 1e-3), num_epochs=cfg.get("num_epochs", 500),
        patience=cfg.get("patience", 50), opt=cfg.get("optimizer", "adam"),
        muon_lr=cfg.get("muon_lr", 0.02), verbose=False, pathsave=ckpt_dir,
        weights=weights, norm_type=cfg.get("norm_type", "none"),
        num_training=1,
        embed_dim=cfg.get("embed_dim", 32), fusion=cfg.get("fusion", "low_rank"),
        n_heads=cfg.get("n_heads", 4), n_attn_layers=cfg.get("n_attn_layers", 1),
        ffn_dim=cfg.get("ffn_dim", 0), rank=cfg.get("rank", 8),
        head_hidden=tuple(cfg.get("head_hidden", [64, 32])),
        dropout=cfg.get("dropout", 0.1), activation=cfg.get("activation", "gelu"),
        n_hom_dims=cfg.get("n_hom_dims", 2),
        reliability_threshold=cfg.get("reliability_threshold", 0.6),
        label_smoothing=cfg.get("label_smoothing", 0.1),
        grad_clip_norm=cfg.get("grad_clip_norm", 1.0),
        base_seed=run_seed_base,
        cache_path=cfg.get("tda_cache_path", os.path.join(cfg["path_save"], "tda_cache.pt")),
        layout=layout,
        # Internal robustness sweep disabled; this script runs its own
        # (both feature-level and raw-signal) sweeps separately, in
        # --mode sweep.
        noise_levels=None,
        noise_aug_sigma=cfg.get("noise_aug_sigma", 0.05),
        _clip_q_lo=clip_q_lo, _clip_q_hi=clip_q_hi,
        use_temperature_scaling=cfg.get("use_temperature_scaling", True),
        return_run_artifacts=True,
    )

    assert artifacts and len(artifacts) == 1, (
        f"Expected exactly 1 artifact from num_training=1, got {len(artifacts) if artifacts else 0}"
    )
    art = artifacts[0]

    os.makedirs(artifact_dir, exist_ok=True)
    torch.save({
        "model_state_dict": art["model_state_dict"],
        "idx_te": art["idx_te"],
        "norm_mu": art["norm_mu"],
        "norm_std": art["norm_std"],
        "run_seed": art["run_seed"],
        "run_idx": args.run_idx,   # external, SLURM-facing run index (for filenames/bookkeeping)
        "n_classes": art["n_classes"],
        "clean_test_accuracy": art["clean_test_accuracy"],
        "temperature": art["temperature"],
    }, args.artifact)
    print(f"[train] Saved run artifact -> {args.artifact}  "
          f"(internal run_seed={art['run_seed']}, clean_test_accuracy={art['clean_test_accuracy']:.4f})")


# ============================================================================
# --mode sweep
# ============================================================================

def run_sweep_mode(cfg: dict, args, tda: FullTDAPipeline, device: torch.device) -> None:
    # load_cache() first: we need the cache's actual row count before
    # load_raw_segments() can self-detect whether the cache was built from
    # balanced or unbalanced segments (see load_raw_segments docstring).
    features, labels, layout, clip_q_lo, clip_q_hi = load_cache(cfg)
    X_seg, y_seg_raw, class_labels = load_raw_segments(cfg, target_n=len(features))

    if len(features) != len(X_seg):
        raise ValueError(
            f"Cached features (N={len(features)}) and freshly segmented raw data "
            f"(N={len(X_seg)}) do not match in length. The cache may have been built "
            f"with a different segmentation_duration/balance_strategy than the current "
            f"config. Re-run precompute_tda.py or check cfg['segmentation_duration']."
        )
    if not np.array_equal(labels, y_seg_raw):
        print("[sweep] WARNING: cache labels and freshly re-segmented raw labels differ "
              "in value. This should not happen if the cache was built from this exact "
              "config. Proceeding using the cache's labels (matching what --mode train "
              "used), but double-check the cache is up to date.")

    # Calibrate the PI imager once, on clean data, matching
    # precompute_tda.py's own warm-up procedure (see FullTDAPipeline.
    # calibrate() docstring). precompute_batch_size (not the --batch_size
    # CLI arg, which controls the sweep's own recomputation batch size) is
    # the parameter precompute_tda.py itself uses for this warm-up, so it
    # is read here to match exactly.
    tda.calibrate(X_seg, warmup_batch_size=cfg.get("precompute_batch_size", 64))

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    n_classes = artifact["n_classes"]
    net = build_model_from_cfg(cfg, n_classes, device)
    net.load_state_dict(artifact["model_state_dict"])
    net.to(device)

    idx_te = artifact["idx_te"]
    norm_mu, norm_std = artifact["norm_mu"], artifact["norm_std"]
    run_seed = artifact["run_seed"]
    run_idx = args.run_idx if args.run_idx is not None else artifact["run_idx"]
    norm_type = cfg.get("norm_type", "none")
    # Fitted calibration temperature, matching the original pipeline's own
    # use of it in both the raw-signal and feature-level sweeps. Falls
    # back to 1.0 (no scaling) for artifacts saved before this field was
    # added.
    temperature = float(artifact.get("temperature", 1.0))

    X_test_raw = X_seg[idx_te]
    y_test = torch.tensor(labels[idx_te], dtype=torch.long)

    print(f"[sweep] run_idx={run_idx}  run_seed={run_seed}  sigma={args.sigma}  "
          f"clean_test_accuracy(train-time)={artifact['clean_test_accuracy']:.4f}  "
          f"temperature={temperature:.4f}")

    # (1) raw-signal-level, this single sigma
    df_raw = raw_signal_sweep(
        net, tda, X_test_raw, y_test, [args.sigma], seed=run_seed, device=device,
        norm_mu=norm_mu, norm_std=norm_std, norm_type=norm_type,
        clip_q_lo=clip_q_lo, clip_q_hi=clip_q_hi, temperature=temperature,
    ).reset_index()
    df_raw["level"] = "raw_signal"
    df_raw["run_idx"] = run_idx

    # (2) feature-level, same trained model and test set, same single sigma
    x_te_features_raw = features[idx_te].clone()
    df_feat = _cached_robustness_sweep(
        net=net, x_test=x_te_features_raw, y_test=y_test,
        noise_levels=[args.sigma], batch_size=args.batch_size,
        seed=run_seed, device=device, norm_mu=norm_mu, norm_std=norm_std,
        clip_q_lo=clip_q_lo, clip_q_hi=clip_q_hi, temperature=temperature,
    ).reset_index()
    df_feat["level"] = "feature"
    df_feat["run_idx"] = run_idx

    combined = pd.concat([df_raw, df_feat], ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.sweep_out)) or ".", exist_ok=True)
    combined.to_csv(args.sweep_out, index=False)
    print(f"[sweep] Saved -> {args.sweep_out}")


# ============================================================================
# --mode all (single-process behaviour)
# ============================================================================

def run_all_mode(cfg: dict, args, base_seed: int, device: torch.device) -> None:
    noise_levels = [float(s) for s in args.noise_levels.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device: {device}   Runs: {args.num_runs}   Noise levels: {noise_levels}")

    # load_cache() first: needed before load_raw_segments() can
    # self-detect whether the cache was built from balanced or unbalanced
    # segments (see load_raw_segments docstring).
    features, labels, layout, clip_q_lo, clip_q_hi = load_cache(cfg)
    X_seg, y_seg_raw, class_labels = load_raw_segments(cfg, target_n=len(features))

    if len(features) != len(X_seg):
        raise ValueError(
            f"Cached features (N={len(features)}) and freshly segmented raw data "
            f"(N={len(X_seg)}) do not match in length. The cache may have been built "
            f"with a different segmentation_duration/balance_strategy than the current "
            f"config. Re-run precompute_tda.py or check cfg['segmentation_duration']."
        )
    if not np.array_equal(labels, y_seg_raw):
        print("WARNING: cache labels and freshly re-segmented raw labels differ in "
              "value, proceeding using the cache's labels.")

    n_classes = int(len(np.unique(labels)))
    tda = FullTDAPipeline(cfg, batch_size=args.batch_size)
    # Calibrate the PI imager once, on clean data, matching
    # precompute_tda.py's own warm-up procedure (see FullTDAPipeline.
    # calibrate() docstring). Done once here (not per run), since
    # calibration depends only on the clean data and cfg, not on which
    # model or run is being evaluated.
    tda.calibrate(X_seg, warmup_batch_size=cfg.get("precompute_batch_size", 64))

    # Class weights computed once from the full cache's label
    # distribution, matching main_train_time_series_tda_stat_summary.py's
    # class_weights_cache exactly (see run_train_mode's docstring for
    # detail).
    present = np.unique(labels)
    w_present = class_weight.compute_class_weight("balanced", classes=present, y=labels)
    w_full = np.zeros(n_classes, dtype=np.float32)
    w_full[present] = w_present
    weights = torch.tensor(w_full, dtype=torch.float)

    ckpt_dir = os.path.join(args.out_dir, "tmp_run_cached_output")
    os.makedirs(ckpt_dir, exist_ok=True)

    all_raw, all_feat = [], []

    for run_idx in range(1, args.num_runs + 1):
        print(f"\n{'='*60}\n  Run {run_idx}/{args.num_runs}\n{'='*60}")
        # Same seed convention as run_train_mode: see that function's
        # docstring for why this reproduces the main pipeline's own
        # sequential-run seeding exactly.
        run_seed_base = cfg.get("random_seed", 42) + (run_idx - 1)

        artifacts = run_cached_tda_training(
            features=features, labels=labels, classes=class_labels, device=device,
            n_classes=n_classes, cfg=cfg,
            test_size=cfg.get("test_size", 0.1), val_size=cfg.get("val_size", 0.1),
            batch_size=cfg.get("batch_size", 128), num_cpus=1,
            lr=cfg.get("lr", 1e-3), num_epochs=cfg.get("num_epochs", 500),
            patience=cfg.get("patience", 50), opt=cfg.get("optimizer", "adam"),
            muon_lr=cfg.get("muon_lr", 0.02), verbose=False, pathsave=ckpt_dir,
            weights=weights, norm_type=cfg.get("norm_type", "none"),
            num_training=1,
            embed_dim=cfg.get("embed_dim", 32), fusion=cfg.get("fusion", "low_rank"),
            n_heads=cfg.get("n_heads", 4), n_attn_layers=cfg.get("n_attn_layers", 1),
            ffn_dim=cfg.get("ffn_dim", 0), rank=cfg.get("rank", 8),
            head_hidden=tuple(cfg.get("head_hidden", [64, 32])),
            dropout=cfg.get("dropout", 0.1), activation=cfg.get("activation", "gelu"),
            n_hom_dims=cfg.get("n_hom_dims", 2),
            reliability_threshold=cfg.get("reliability_threshold", 0.6),
            label_smoothing=cfg.get("label_smoothing", 0.1),
            grad_clip_norm=cfg.get("grad_clip_norm", 1.0),
            base_seed=run_seed_base,
            cache_path=cfg.get("tda_cache_path", os.path.join(cfg["path_save"], "tda_cache.pt")),
            layout=layout,
            noise_levels=None,   # this script runs its own sweeps below
            noise_aug_sigma=cfg.get("noise_aug_sigma", 0.05),
            _clip_q_lo=clip_q_lo, _clip_q_hi=clip_q_hi,
            use_temperature_scaling=cfg.get("use_temperature_scaling", True),
            return_run_artifacts=True,
        )
        art = artifacts[0]
        net = build_model_from_cfg(cfg, art["n_classes"], device)
        net.load_state_dict(art["model_state_dict"])
        net.to(device)
        idx_te = art["idx_te"]
        norm_mu, norm_std = art["norm_mu"], art["norm_std"]
        run_seed = art["run_seed"]
        temperature = float(art["temperature"])
        print(f"  [run {run_idx}] clean test accuracy = {art['clean_test_accuracy']:.4f}  "
              f"(internal run_seed={run_seed}, T={temperature:.4f})")

        X_test_raw = X_seg[idx_te]
        y_test = torch.tensor(labels[idx_te], dtype=torch.long)
        norm_type = cfg.get("norm_type", "none")

        df_raw = raw_signal_sweep(
            net, tda, X_test_raw, y_test, noise_levels, seed=run_seed, device=device,
            norm_mu=norm_mu, norm_std=norm_std, norm_type=norm_type,
            clip_q_lo=clip_q_lo, clip_q_hi=clip_q_hi, temperature=temperature,
        )
        df_raw["run_idx"] = run_idx
        df_raw["level"] = "raw_signal"
        all_raw.append(df_raw.reset_index())

        x_te_features_raw = features[idx_te].clone()
        df_feat = _cached_robustness_sweep(
            net=net, x_test=x_te_features_raw, y_test=y_test,
            noise_levels=noise_levels, batch_size=args.batch_size,
            seed=run_seed, device=device, norm_mu=norm_mu, norm_std=norm_std,
            clip_q_lo=clip_q_lo, clip_q_hi=clip_q_hi, temperature=temperature,
        )
        df_feat["run_idx"] = run_idx
        df_feat["level"] = "feature"
        all_feat.append(df_feat.reset_index())

    df_raw_all = pd.concat(all_raw, ignore_index=True)
    df_feat_all = pd.concat(all_feat, ignore_index=True)
    df_combined = pd.concat([df_raw_all, df_feat_all], ignore_index=True)

    combined_path = os.path.join(args.out_dir, "raw_vs_feature_robustness.csv")
    df_combined.to_csv(combined_path, index=False)
    print(f"\nCombined results -> {combined_path}")

    summary = df_combined.groupby(["level", "sigma"])["accuracy"].agg(["mean", "std"]).reset_index()
    print("\nSummary (mean accuracy +/- std across runs):")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(args.out_dir, "raw_vs_feature_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    for level, color in [("raw_signal", "#D85A30"), ("feature", "#378ADD")]:
        sub = summary[summary["level"] == level].sort_values("sigma")
        ax.plot(sub["sigma"], sub["mean"], marker="o", lw=2, color=color, label=level)
        ax.fill_between(sub["sigma"], sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                         color=color, alpha=0.2)
    ax.set_xlabel("Gaussian noise sigma")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Raw-signal vs. feature-level noise robustness")
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, "raw_vs_feature_robustness.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Comparison plot -> {fig_path}")


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    if args.base_seed is not None:
        cfg["random_seed"] = args.base_seed
    base_seed = cfg.get("random_seed", 42)

    if args.noise_aug_sigma is not None:
        cfg["noise_aug_sigma"] = args.noise_aug_sigma
    elif float(cfg.get("noise_aug_sigma", 0.0)) == 0.0:
        print("WARNING: cfg['noise_aug_sigma'] is 0.0 and --noise_aug_sigma was not "
              "set. Confirm whether the config's noise_aug_sigma actually matches "
              "the value used to train the reported main results before proceeding, "
              "and pass it explicitly via --noise_aug_sigma if it differs from 0.0, "
              "or this script will train a materially less robust model than the "
              "one reported.")

    if args.use_temperature_scaling is not None:
        cfg["use_temperature_scaling"] = (args.use_temperature_scaling == "true")
    elif not cfg.get("use_temperature_scaling", True):
        print("WARNING: cfg['use_temperature_scaling'] is false and "
              "--use_temperature_scaling was not set. Confirm whether the config's "
              "value actually matches what trained the reported main results before "
              "proceeding. Pass --use_temperature_scaling true to match if it does, "
              "or ECE/confidence values computed here will use temperature=1.0 (no "
              "scaling) even though accuracy is unaffected either way.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Config: {args.config}   Mode: {args.mode}   Device: {device}   "
          f"noise_aug_sigma: {cfg.get('noise_aug_sigma', 0.0)}   "
          f"use_temperature_scaling: {cfg.get('use_temperature_scaling', True)}")

    if args.mode == "train":
        run_train_mode(cfg, args, base_seed, device)
    elif args.mode == "sweep":
        tda = FullTDAPipeline(cfg, batch_size=args.batch_size)
        run_sweep_mode(cfg, args, tda, device)
    else:
        run_all_mode(cfg, args, base_seed, device)


if __name__ == "__main__":
    main()
