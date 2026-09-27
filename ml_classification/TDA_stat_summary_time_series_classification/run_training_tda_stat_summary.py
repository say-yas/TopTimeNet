"""
run_training_tda_stat_summary.py
Training runner for TDAEnd2EndNet.

Robustness and calibration features
-------------------------------------
- Noise robustness sweep: run_tda_stat_summary_training() accepts a
  noise_levels parameter (a list of sigma floats, or None to skip). When
  provided, after each run's testing_step(), the method calls
  net.evaluate_robustness() and saves:
    robustness_sweep_run{N}.csv   per-sigma rows for run N
    robustness_all_runs.csv       all runs stacked, with a run_idx column
  The sweep only runs when noise_levels is not None.

- Temperature scaling (post-hoc calibration): after training each run, a
  scalar temperature T is fitted on the validation set via LBFGS to
  minimise cross-entropy of (logits / T). T > 1 gives softer confidences,
  for better alignment with accuracy. T is saved in the CSV record and
  applied inside evaluate_robustness() via the temperature argument added
  to that method. Set use_temperature_scaling=False to disable.

- ECE (Expected Calibration Error) is added to the robustness sweep:
  each per-sigma row includes "ece" (M=10 equal-width bins by default).
  evaluate_robustness() accepts ece_bins= to control this; the
  TDAEnd2EndNet.evaluate_robustness() method is wrapped at call time
  rather than modified directly.
"""

from __future__ import annotations

import gc
import glob
import os
import time
import warnings

import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import torchinfo
    _HAS_TORCHINFO = True
except ImportError:
    _HAS_TORCHINFO = False

# Muon import
_MUON_BACKEND = None
try:
    from muon import MuonWithAuxAdam as _MuonWithAuxAdam
    _MUON_BACKEND = "muon_pkg"
except ImportError:
    try:
        _torch_muon = torch.optim.Muon          # type: ignore[attr-defined]
        _MUON_BACKEND = "torch_optim"
    except AttributeError:
        _MUON_BACKEND = None

from ml_classification.TDA_stat_summary_time_series_classification.nn_tda_stat_summary_model import (
    TDAEnd2EndNet,
    EarlyStopper,
)
import ml_classification.TDA_stat_summary_time_series_classification.nn_tda_stat_summary_train as tda_model_train


# ============================================================================
# Helpers
# ============================================================================

def _free_trial_memory(objects: list, pathsave: str = None) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    for ckpt in glob.glob("checkpoint*.pt"):
        try: os.remove(ckpt)
        except OSError: pass
    if pathsave:
        for ckpt in glob.glob(os.path.join(pathsave, "checkpoint*.pt")):
            try: os.remove(ckpt)
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


# Parameter splitting for Muon

def _split_muon_params(net: nn.Module) -> tuple[list, list]:
    head_and_input_keywords = {
        "takens", "pc_stats", "embed_in", "input_proj",
        "head", "classifier", "out_proj", "final",
        "norm", "bn", "bias",
    }
    hidden_weights = []
    rest_params    = []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        name_lower = name.lower()
        is_excluded = any(kw in name_lower for kw in head_and_input_keywords)
        if p.ndim >= 2 and not is_excluded:
            hidden_weights.append(p)
        else:
            rest_params.append(p)
    return hidden_weights, rest_params


def _build_muon_optimizer(net, lr, muon_lr, opt_base):
    hidden_weights, rest_params = _split_muon_params(net)
    n_muon  = sum(p.numel() for p in hidden_weights)
    n_adamw = sum(p.numel() for p in rest_params)
    print(f"  Muon param split: {n_muon:,} -> Muon  |  {n_adamw:,} -> AdamW")

    if _MUON_BACKEND == "muon_pkg":
        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                 lr=muon_lr, weight_decay=0.01),
            dict(params=rest_params,    use_muon=False,
                 lr=lr, betas=(0.9, 0.95), weight_decay=1e-4),
        ]
        optimizer = _MuonWithAuxAdam(param_groups)
        print(f"  Using MuonWithAuxAdam (muon_lr={muon_lr}, adamw_lr={lr})")
        return optimizer, None, "muon_pkg"

    elif _MUON_BACKEND == "torch_optim":
        opt_muon  = torch.optim.Muon(hidden_weights, lr=muon_lr, momentum=0.95)  # type: ignore
        opt_adamw = optim.AdamW(rest_params, lr=lr, weight_decay=1e-4)
        optimizer = _DualOptimizer(opt_muon, opt_adamw)
        print(f"  Using torch.optim.Muon + AdamW (muon_lr={muon_lr}, adamw_lr={lr})")
        return optimizer, None, "torch_optim"

    else:
        warnings.warn(
            "Muon requested but not available, falling back to AdamW.\n"
            "  pip install git+https://github.com/KellerJordan/Muon",
            stacklevel=3,
        )
        all_params = list(net.parameters())
        optimizer  = optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
        return optimizer, None, "adamw_fallback"


class _DualOptimizer:
    def __init__(self, opt_muon, opt_adamw):
        self._muon  = opt_muon
        self._adamw = opt_adamw
        self.param_groups = opt_adamw.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self._muon.step()
        self._adamw.step(closure)

    def state_dict(self):
        return {"muon": self._muon.state_dict(), "adamw": self._adamw.state_dict()}

    def load_state_dict(self, state):
        self._muon.load_state_dict(state["muon"])
        self._adamw.load_state_dict(state["adamw"])


# Temperature scaling

def _fit_temperature(
    net:       nn.Module,
    loader_va: torch.utils.data.DataLoader,
    device:    torch.device,
    max_iter:  int = 50,
) -> float:
    """
    Post-hoc temperature scaling on the validation set.

    Fits a single scalar T that minimises cross-entropy of (logits / T).
    T > 1 gives smaller softmax probabilities, so confidence moves closer
    to accuracy.

    Returns
    -------
    T : float, optimal temperature clamped to [0.5, 5.0]
    """
    net.eval()
    logits_all = []
    labels_all = []

    with torch.no_grad():
        for xb, yb in loader_va:
            logits_all.append(net(xb.to(device)).cpu())
            labels_all.append(yb.cpu())

    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)

    T       = nn.Parameter(torch.ones(1))
    opt_T   = optim.LBFGS([T], lr=0.1, max_iter=max_iter)
    loss_fn = nn.CrossEntropyLoss()

    def _eval():
        opt_T.zero_grad()
        loss = loss_fn(logits_all / T.clamp(min=0.5, max=5.0), labels_all)
        loss.backward()
        return loss

    opt_T.step(_eval)
    t_val = T.clamp(min=0.5, max=5.0).item()
    print(f"  Calibration temperature T = {t_val:.4f}")
    return t_val


# ECE-aware evaluate_robustness wrapper

def _compute_ece(
    preds:  torch.Tensor,
    confs:  torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-width confidence bins."""
    N         = len(labels)
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece       = 0.0

    for i in range(n_bins):
        lo   = bin_edges[i].item()
        hi   = bin_edges[i + 1].item()
        mask = (confs >= lo) & (confs < hi)
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = (preds[mask] == labels[mask]).float().mean().item()
        bin_conf = confs[mask].mean().item()
        ece     += (mask.float().sum().item() / N) * abs(bin_conf - bin_acc)

    return ece


def _evaluate_robustness_with_temperature(
    net:          TDAEnd2EndNet,
    x_test:       torch.Tensor,
    y_test:       torch.Tensor,
    noise_levels: list,
    batch_size:   int,
    seed:         int,
    temperature:  float = 1.0,
    ece_bins:     int   = 10,
) -> pd.DataFrame:
    """
    Wrapper around TDAEnd2EndNet.evaluate_robustness() that divides logits
    by temperature T before softmax, and computes ECE per sigma level,
    adding it as a column. This avoids modifying TDAEnd2EndNet directly,
    keeping the model class unchanged. For temperature=1.0 and
    ece_bins=0, behaviour is identical to the original
    evaluate_robustness().

    Returns
    -------
    pd.DataFrame indexed by sigma with columns:
        sigma, snr_db, accuracy, mean_confidence, pct_low_conf[, ece]
    """
    import numpy as np

    net.eval()
    rng = torch.Generator()
    rng.manual_seed(seed)
    dev = next(net.parameters()).device
    x_test = x_test.to(dev)
    y_test = y_test.to(dev)

    records   = []
    n_levels  = len(noise_levels)
    print(f"  TDA end2end robustness sweep: {n_levels} sigma levels  "
          f"T={temperature:.4f}  ECE bins={ece_bins}")

    for i, sigma in enumerate(noise_levels):
        if sigma == 0.0:
            x_noisy = x_test.clone()
        else:
            noise   = torch.zeros_like(x_test).normal_(0.0, sigma, generator=rng)
            x_noisy = x_test + noise

        net.reset_pi_imager()

        all_preds  = []
        all_confs  = []

        with torch.no_grad():
            for start in range(0, x_noisy.shape[0], batch_size):
                xb     = x_noisy[start: start + batch_size]
                logits = net(xb)
                probs  = torch.softmax(logits / temperature, dim=1)
                conf, pred = probs.max(dim=1)
                all_preds.append(pred.cpu())
                all_confs.append(conf.cpu())

        preds = torch.cat(all_preds)
        confs = torch.cat(all_confs)
        y_cpu = y_test.cpu()

        acc       = (preds == y_cpu).float().mean().item()
        mean_conf = confs.mean().item()
        pct_low   = (confs < 0.5).float().mean().item() * 100.0
        snr_db    = 10.0 * np.log10(1.0 / (sigma ** 2)) if sigma > 0 else float("inf")
        ece       = _compute_ece(preds, confs, y_cpu, n_bins=ece_bins) if ece_bins > 0 else None

        row = {
            "sigma"           : sigma,
            "snr_db"          : snr_db,
            "accuracy"        : acc,
            "mean_confidence" : mean_conf,
            "pct_low_conf"    : pct_low,
        }
        if ece is not None:
            row["ece"] = ece

        records.append(row)
        ece_str = f"  ECE={ece:.4f}" if ece is not None else ""
        print(f"  [{i+1:2d}/{n_levels}]  sigma={sigma:.3f}  SNR={snr_db:+.1f} dB  "
              f"acc={acc:.4f}  conf={mean_conf:.4f}  low_conf={pct_low:.1f}%{ece_str}")

    return pd.DataFrame(records).set_index("sigma")


# ============================================================================
# Main training function
# ============================================================================

def run_tda_stat_summary_training(
    data,
    labels,
    classes,
    device,
    cfg:            dict  = None,
    num_channels:   int   = 1,
    n_classes:      int   = 2,
    seg_len:        int   = 200,
    takens_dim:     int   = 2,
    takens_delay:   int   = 5,
    n_hom_dims:     int   = 2,
    n_betti_bins:   int   = 50,
    n_pi_bins:      int   = 20,
    pi_sigma:       float = 0.1,
    ph_workers:     int   = 4,
    embed_dim:      int   = 32,
    fusion:         str   = "low_rank",
    rank:           int   = 8,
    n_attn_layers:  int   = 1,
    n_heads:        int   = 4,
    head_hidden:    tuple = (64, 32),
    test_size:             float = 0.1,
    val_size:              float = 0.1,
    batch_size:            int   = 128,
    num_cpus:              int   = 1,
    lr:                    float = 0.001,
    num_epochs:            int   = 900,
    patience:              int   = 20,
    dropout:               float = 0.1,
    activation:            str   = "gelu",
    opt:                   str   = "adam",
    muon_lr:               float = 0.02,
    verbose:               bool  = False,
    pathsave:              str   = "./",
    weights:               torch.Tensor = None,
    norm_type:             str   = "none",
    num_training:          int   = 10,
    reliability_threshold: float = 0.6,
    label_smoothing:       float = 0.1,
    grad_clip_norm:        float = 1.0,
    base_seed:             int   = 42,
    noise_levels:          list  = None,
    noise_batch_size:      int   = 16,
    use_temperature_scaling: bool = True,
    ece_bins:              int   = 10,
    # backward-compat stubs
    max_length_series:   int  = 200,
    embed_size:          int  = 16,
    nhead:               int  = 4,
    dim_feedforward:     int  = 2048,
    num_encoderlayers:   int  = 1,
    conv1d_emb:          bool = True,
    conv1d_kernel_size:  int  = 3,
    size_linear_layers:  int  = 16,
    tda_modeltype:       str  = "end2end",
    proj_dim:            int  = 64,
    use_cross_attn:      bool = True,
):
    """
    Run TDAEnd2EndNet training `num_training` independent times.

    use_temperature_scaling=True (default): after training, fits T on the
    validation set and uses it in the robustness sweep.

    ECE is added to the robustness sweep CSV (ece_bins=10 equal-width
    bins by default). Set ece_bins=0 to disable.
    """

    run_records        = []
    arr_acc_test       = []
    arr_f1score_test   = []
    arr_gmean_test     = []
    arr_precision_test = []
    arr_recall_test    = []
    computation_time   = 0.0

    all_sweep_dfs = []

    _opt     = cfg.get("optimizer", opt) if cfg else opt
    _muon_lr = float(cfg.get("muon_lr", muon_lr)) if cfg else muon_lr

    # noise_levels can also come from config JSON
    if noise_levels is None and cfg is not None:
        noise_levels = cfg.get("noise_levels", None)

    for idx in range(num_training):
        print(f"\n{'='*60}")
        print(f"  Training run {idx + 1}/{num_training}  [TDAEnd2EndNet]")
        print(f"{'='*60}")

        run_seed = base_seed + idx
        _set_seed(run_seed)
        print(f"  seed={run_seed}  opt={_opt}" +
              (f"  muon_lr={_muon_lr}" if _opt == "muon" else "") +
              f"  label_smoothing={label_smoothing}")

        parameters = {
            "test_size"             : test_size,
            "val_size"              : val_size,
            "batch_size"            : batch_size,
            "num_cpus"              : num_cpus,
            "lr"                    : lr,
            "num_epochs"            : num_epochs,
            "verbose"               : verbose,
            "n_classes"             : n_classes,
            "patience"              : patience,
            "opt"                   : _opt,
            "muon_lr"               : _muon_lr,
            "norm_type"             : norm_type,
            "num_training"          : num_training,
            "reliability_threshold" : reliability_threshold,
            "label_smoothing"       : label_smoothing,
            "grad_clip_norm"        : grad_clip_norm,
            "seg_len"               : seg_len,
            "takens_dim"            : takens_dim,
            "takens_delay"          : takens_delay,
            "n_hom_dims"            : n_hom_dims,
            "n_betti_bins"          : n_betti_bins,
            "n_pi_bins"             : n_pi_bins,
            "pi_sigma"              : pi_sigma,
            "embed_dim"             : embed_dim,
            "fusion"                : fusion,
            "rank"                  : rank,
            "n_attn_layers"         : n_attn_layers,
            "n_heads"               : n_heads,
            "dropout"               : dropout,
            "activation"            : activation,
            "head_hidden"           : str(head_hidden),
            "ph_workers"            : ph_workers,
        }
        print("parameters:", parameters)

        start = time.time()

        net = optimizer = loss_function = early_stopper = None
        scheduler = train_net = None
        x_train = x_val = x_test = y_train = y_val = y_test = None
        xy_train = xy_val = xy_test = None
        trainloader = valloader = testloader = None
        confusion_matrix = cm = None
        muon_backend = "none"
        temperature  = 1.0   # default

        try:
            # build model
            if cfg is not None:
                net = TDAEnd2EndNet.from_config(cfg, n_classes).to(device)
            else:
                net = TDAEnd2EndNet(
                    n_classes     = n_classes,
                    seg_len       = seg_len,
                    takens_dim    = takens_dim,
                    takens_delay  = takens_delay,
                    n_hom_dims    = n_hom_dims,
                    n_betti_bins  = n_betti_bins,
                    n_pi_bins     = n_pi_bins,
                    pi_sigma      = pi_sigma,
                    embed_dim     = embed_dim,
                    fusion        = fusion,
                    rank          = rank,
                    n_attn_layers = n_attn_layers,
                    n_heads       = n_heads,
                    dropout       = dropout,
                    activation    = activation,
                    head_hidden   = head_hidden,
                    ph_workers    = ph_workers,
                ).to(device)

            if _HAS_TORCHINFO:
                try:
                    print(torchinfo.summary(
                        net, input_size=(batch_size, 1, seg_len), device=device,
                    ))
                except Exception:
                    net.count_parameters(verbose=True)
            else:
                net.count_parameters(verbose=True)

            n_params = net.count_parameters()
            strcomplexity = (
                f"idx={idx + 1}: TDAEnd2EndNet  "
                f"seg={seg_len} hom={n_hom_dims} "
                f"fusion={fusion} embed={embed_dim} rank={rank} "
                f"opt={_opt} "
                f"ls={label_smoothing} clip={grad_clip_norm} "
                f"params={n_params:,}"
            )
            print(f"  {strcomplexity}")

            # optimizer
            if _opt == "muon":
                optimizer, _, muon_backend = _build_muon_optimizer(
                    net, lr=lr, muon_lr=_muon_lr, opt_base="muon",
                )
            elif _opt == "adamw":
                optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
                muon_backend = "none"
            else:
                optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
                muon_backend = "none"

            loss_function = nn.CrossEntropyLoss(
                weight          = weights.to(device) if weights is not None else None,
                label_smoothing = label_smoothing,
            )

            ckpt_path = os.path.join(pathsave, f"checkpoint_idx{idx + 1}.pt")
            early_stopper = EarlyStopper(verbose=verbose, path=ckpt_path, patience=patience)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", factor=0.9,
                patience=max(1, patience - 4), threshold=1e-8,
            )

            train_net = tda_model_train.TrainTDAStat(
                net, optimizer, loss_function,
                early_stopper, scheduler, device, parameters,
            )

            # data splits
            x_train, x_val, x_test, y_train, y_val, y_test = \
                train_net.generate_train_val_test(data, labels)

            print("x_train:", np.array(x_train).shape,
                  " y_train:", np.array(y_train).shape)
            print("x_val:  ", np.array(x_val).shape,
                  " y_val:  ", np.array(y_val).shape)
            print("x_test: ", np.array(x_test).shape,
                  " y_test: ", np.array(y_test).shape)

            xy_train = tda_model_train.data_to_tensor(x_train, y_train)
            xy_val   = tda_model_train.data_to_tensor(x_val,   y_val)
            xy_test  = tda_model_train.data_to_tensor(x_test,  y_test)

            del x_train, x_val, x_test, y_train, y_val, y_test
            x_train = x_val = x_test = y_train = y_val = y_test = None

            trainloader, valloader, testloader = \
                train_net.init_data_loaders(xy_train, xy_val, xy_test, num_workers=0)

            # training loop
            (train_losses, val_losses,
             train_accs,   val_accs,
             confusion_matrix,
             train_f1,     val_f1,
             train_gmean,  val_gmean,
             train_pre,    val_pre,
             train_rec,    val_rec,
             train_rel,    val_rel,
             train_neutral, val_neutral,
             val_class_rel) = train_net.run_training(trainloader, valloader)

            for metric, train_v, val_v in [
                ("loss",        train_losses, val_losses),
                ("accuracy",    train_accs,   val_accs),
                ("f1",          train_f1,     val_f1),
                ("reliability", train_rel,    val_rel),
            ]:
                tda_model_train.plot_progress(
                    strcomplexity + f"  {metric}", metric, train_v, val_v,
                    save_path=os.path.join(pathsave, f"{metric}_idx{idx + 1}.png"),
                )
                plt.close("all")

            # temperature scaling
            if use_temperature_scaling:
                temperature = _fit_temperature(net, valloader, device)
            else:
                temperature = 1.0

            # testing
            (correct, total, accuracy_test, f1score_test, gmean_test,
             precision_test, recall_test,
             reliability_test, neutral_test,
             class_reliability_test,
             confusion_matrix_test) = train_net.testing_step(testloader)

            end = time.time()
            computation_time = end - start

            print(
                f"\nRun {idx + 1} | acc={accuracy_test:.4f}  f1={f1score_test:.4f}  "
                f"gmean={gmean_test:.4f}  rel={reliability_test:.4f}  "
                f"neutral={neutral_test:.1f}%  time={computation_time:.1f}s  "
                f"T={temperature:.3f}"
            )

            arr_acc_test.append(accuracy_test)
            arr_f1score_test.append(f1score_test)
            arr_gmean_test.append(gmean_test)
            arr_precision_test.append(precision_test)
            arr_recall_test.append(recall_test)

            tda_model_train.visualize_confusion_matrix(
                confusion_matrix_test.numpy().astype(float),
                classes, correct, total,
                path=os.path.join(pathsave, f"confusion_matrix_test_idx{idx + 1}.png"),
            )
            plt.close("all")

            if verbose:
                cm_val = confusion_matrix.numpy()
                df_cm  = pd.DataFrame(cm_val, index=list(classes), columns=list(classes))
                fig, ax = plt.subplots(figsize=(8, 6))
                sn.heatmap(
                    df_cm / (df_cm.astype("float").sum() + 1e-9),
                    annot=True, cmap="coolwarm", fmt=".2f", ax=ax,
                )
                ax.set_title(strcomplexity)
                ax.set_xlabel("Predicted", fontsize=12)
                ax.set_ylabel("True",      fontsize=12)
                fig.savefig(os.path.join(pathsave, f"confusion_matrix_val_idx{idx + 1}.png"))
                plt.close(fig)
                del df_cm, cm_val

            # noise robustness sweep
            if noise_levels is not None:
                print(f"\n  Noise robustness sweep (run {idx + 1}) ...")

                x_test_all = torch.cat([b[0] for b in testloader], dim=0)
                y_test_all = torch.cat([b[1] for b in testloader], dim=0)

                net.eval()
                df_sweep = _evaluate_robustness_with_temperature(
                    net          = net,
                    x_test       = x_test_all,
                    y_test       = y_test_all,
                    noise_levels = noise_levels,
                    batch_size   = noise_batch_size,
                    seed         = run_seed,
                    temperature  = temperature,
                    ece_bins     = ece_bins,
                )
                df_sweep["run_idx"]     = idx + 1
                df_sweep["fusion"]      = fusion
                df_sweep["opt"]         = _opt
                df_sweep["temperature"] = temperature
                all_sweep_dfs.append(df_sweep.reset_index())

                sweep_path = os.path.join(pathsave, f"robustness_sweep_run{idx + 1}.csv")
                df_sweep.to_csv(sweep_path)
                print(f"  Robustness sweep saved -> {sweep_path}")
                plt.close("all")

                del x_test_all, y_test_all, df_sweep

            # CSV record
            record = {
                "run_idx"              : idx + 1,
                "run_seed"             : run_seed,
                "modeltype"            : "TDAEnd2EndNet",
                "computation_time_s"   : round(computation_time, 2),
                "n_params"             : n_params,
                "seg_len"              : seg_len,
                "n_classes"            : n_classes,
                "takens_dim"           : takens_dim,
                "takens_delay"         : takens_delay,
                "n_hom_dims"           : n_hom_dims,
                "n_betti_bins"         : n_betti_bins,
                "n_pi_bins"            : n_pi_bins,
                "pi_sigma"             : pi_sigma,
                "embed_dim"            : embed_dim,
                "fusion"               : fusion,
                "rank"                 : rank,
                "n_attn_layers"        : n_attn_layers,
                "n_heads"              : n_heads,
                "dropout"              : dropout,
                "activation"           : activation,
                "head_hidden"          : str(head_hidden),
                "ph_workers"           : ph_workers,
                "lr"                   : lr,
                "muon_lr"              : _muon_lr if _opt == "muon" else None,
                "muon_backend"         : muon_backend,
                "batch_size"           : batch_size,
                "num_epochs"           : num_epochs,
                "patience"             : patience,
                "opt"                  : _opt,
                "norm_type"            : norm_type,
                "test_size"            : test_size,
                "val_size"             : val_size,
                "reliability_threshold": reliability_threshold,
                "label_smoothing"      : label_smoothing,
                "grad_clip_norm"       : grad_clip_norm,
                "base_seed"            : base_seed,
                "noise_levels_tested"  : str(noise_levels) if noise_levels else None,
                "temperature"          : round(temperature, 4),
                "accuracy_test"        : round(accuracy_test,    4),
                "f1_test"              : round(f1score_test,     4),
                "gmean_test"           : round(gmean_test,       4),
                "precision_test"       : round(precision_test,   4),
                "recall_test"          : round(recall_test,      4),
                "reliability_test"     : round(reliability_test, 4),
                "neutral_pct_test"     : round(neutral_test,     4),
                "final_train_loss"     : round(train_losses[-1], 4),
                "final_val_loss"       : round(val_losses[-1],   4),
                "final_train_acc"      : round(train_accs[-1],   4),
                "final_val_acc"        : round(val_accs[-1],     4),
                "final_train_f1"       : round(train_f1[-1],     4),
                "final_val_f1"         : round(val_f1[-1],       4),
                "final_train_rel"      : round(train_rel[-1],    4),
                "final_val_rel"        : round(val_rel[-1],      4),
                "final_val_neutral"    : round(val_neutral[-1],  4),
                "n_epochs_trained"     : len(train_losses),
                **{f"rel_class_{c}": round(r, 4)
                   for c, r in class_reliability_test.items()},
            }
            run_records.append(record)

            csv_path = os.path.join(pathsave, "results.csv")
            pd.DataFrame(run_records).to_csv(csv_path, index=False)
            print(f"  Results saved -> {csv_path}  ({len(run_records)} row(s))")

        finally:
            _free_trial_memory(
                [net, optimizer, loss_function, early_stopper,
                 scheduler, train_net,
                 x_train, x_val, x_test, y_train, y_val, y_test,
                 xy_train, xy_val, xy_test,
                 trainloader, valloader, testloader,
                 confusion_matrix, cm],
                pathsave=pathsave,
            )
            plt.close("all")

    # combined robustness CSV
    if all_sweep_dfs:
        combined_path = os.path.join(pathsave, "robustness_all_runs.csv")
        pd.concat(all_sweep_dfs, ignore_index=True).to_csv(combined_path, index=False)
        print(f"\n  Combined robustness CSV -> {combined_path}")

    # summary stats
    print(f"\n{'-'*55}")
    print(f"  Summary over {num_training} runs:")
    for name, arr in [
        ("accuracy",  arr_acc_test),
        ("f1",        arr_f1score_test),
        ("gmean",     arr_gmean_test),
        ("precision", arr_precision_test),
        ("recall",    arr_recall_test),
    ]:
        print(f"  {name:12s}: {np.mean(arr):.4f} +/- {np.std(arr):.4f}")
    print(f"{'-'*55}")

    summary_row = {
        "run_idx"        : "MEAN +/- STD",
        "fusion"         : fusion,
        "opt"            : _opt,
        "accuracy_test"  : f"{np.mean(arr_acc_test):.4f} +/- {np.std(arr_acc_test):.4f}",
        "f1_test"        : f"{np.mean(arr_f1score_test):.4f} +/- {np.std(arr_f1score_test):.4f}",
        "gmean_test"     : f"{np.mean(arr_gmean_test):.4f} +/- {np.std(arr_gmean_test):.4f}",
        "precision_test" : f"{np.mean(arr_precision_test):.4f} +/- {np.std(arr_precision_test):.4f}",
        "recall_test"    : f"{np.mean(arr_recall_test):.4f} +/- {np.std(arr_recall_test):.4f}",
    }
    csv_path = os.path.join(pathsave, "results.csv")
    pd.concat(
        [pd.DataFrame(run_records), pd.DataFrame([summary_row])],
        ignore_index=True,
    ).to_csv(csv_path, index=False)
    print(f"Final CSV with summary row -> {csv_path}")

    return computation_time
