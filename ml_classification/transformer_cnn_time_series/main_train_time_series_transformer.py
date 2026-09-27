"""
main_train_time_series_transformer.py
Standalone training script. All parameters are read from config.json
(created by create_config.py).

Usage
-----
    # create config (once, then edit as needed)
    python create_config.py

    # run training
    python main_train_time_series_transformer.py                          # uses ./config.json
    python main_train_time_series_transformer.py --config my_config.json  # custom config path

Output
------
    <path_save>/results.csv                   one row per run, plus a summary row
    <path_save>/all_trials.csv                one row per script invocation,
                                               with model parameter counts and
                                               total wall-clock time
    <path_save>/robustness_sweep_run{N}.csv   per-sigma metrics for run N
    <path_save>/robustness_all_runs.csv       all runs stacked

Muon optimizer
--------------
Set in config.json:
    "optimizer": "muon",
    "muon_lr":   0.02
muon_lr is the learning rate for Muon's hidden-weight group; the AdamW
group (biases, norms, head) still uses "lr". Ignored (and not required in
config) when optimizer is "adam" or "adamW".
Install:  pip install git+https://github.com/KellerJordan/Muon

Noise robustness sweep
-------------------------
Set in config.json:
    "noise_levels":            [0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
    "use_temperature_scaling": true,
    "ece_bins":                10,
    "noise_batch_size":        32
noise_levels=null (or the key absent) disables the sweep entirely; the
pipeline behaves exactly as before. See run_training_transformer_multiset.py
for the sweep implementation; it mirrors the sweep already used for
TopTimeNet in run_training_tda_cached.py, so results from both pipelines
share the same CSV schema and can be plotted with the same script.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.utils import class_weight

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transformer / CNN training pipeline")
    parser.add_argument(
        "--config", type=str, default="config.json",
        help="Path to JSON config file (default: ./config.json)",
    )
    return parser.parse_args()


# config loader

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Run  python create_config.py  to generate it first."
        )
    with open(path) as f:
        cfg = json.load(f)
    print(f"Config loaded from: {path}")
    for k, v in cfg.items():
        print(f"  {k:28s}: {v}")
    return cfg


# imports that depend on the project structure

import ml_classification.utils.read_data_from_h5 as read_data_from_h5
import ml_classification.utils.pad_truncate_tensor as pad_truncate_tensor
import ml_classification.transformer_cnn_time_series.run_training_transformer_multiset as run_training_transformer_multiset
import ml_classification.utils.segment_time_series as segmenting_data

# Model lib is needed only for the parameter count; import lazily so the
# rest of the script still runs if the model module is unavailable.
try:
    import ml_classification.transformer_cnn_time_series.nn_transformer_model as model_lib
    _MODEL_LIB_AVAILABLE = True
except ImportError:
    _MODEL_LIB_AVAILABLE = False


# defaults for transformer-specific keys (absent in CNN-only configs)
_TRANS_DEFAULTS: dict = {
    "embed_size":        64,
    "nhead_encoder":     4,
    "dim_feedforward":   128,
    "num_encoderlayers": 2,
    "conv1d_emb":        True,
    "conv1d_kernel_size":3,
    "size_linear_layers":128,
    "dropout":           0.1,
}


def _cfg_get(cfg: dict, key: str):
    """Return cfg[key] if present, else the transformer default, else raise."""
    if key in cfg:
        return cfg[key]
    if key in _TRANS_DEFAULTS:
        return _TRANS_DEFAULTS[key]
    raise KeyError(f"Required key '{key}' missing from config and has no default.")


# helpers

def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params) for a model.

    Args:
        model: Any ``nn.Module``.

    Returns:
        ``(total, trainable)`` integer parameter counts.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _build_probe_model(cfg: dict, n_classes: int, device: torch.device):
    """Instantiate a throw-away model purely to count parameters.

    Uses the same factory logic as the training runner so the count is
    always consistent with what actually gets trained.

    Args:
        cfg:       Config dict (same keys as passed to ``run_transformer_training``).
        n_classes: Number of output classes.
        device:    Unused, the probe always stays on CPU to avoid
                   allocating GPU memory just for a parameter count.

    Returns:
        ``nn.Module`` on CPU, or ``None`` if the model lib is unavailable.
    """
    if not _MODEL_LIB_AVAILABLE:
        return None

    modeltype = cfg.get("modeltype", "trans1")
    try:
        if modeltype == "trans1":
            return model_lib.TransformerI(
                input_channels    = cfg["num_channels"],
                output_size       = n_classes,
                seq_len           = cfg["segmentation_duration"],
                embed_size        = _cfg_get(cfg, "embed_size"),
                nhead             = _cfg_get(cfg, "nhead_encoder"),
                dim_feedforward   = _cfg_get(cfg, "dim_feedforward"),
                dropout           = _cfg_get(cfg, "dropout"),
                conv1d_emb        = _cfg_get(cfg, "conv1d_emb"),
                conv1d_kernel_size= _cfg_get(cfg, "conv1d_kernel_size"),
                size_linear_layers= _cfg_get(cfg, "size_linear_layers"),
                num_encoderlayers = _cfg_get(cfg, "num_encoderlayers"),
            )
        elif modeltype == "cnn1":
            return model_lib.CNNI(
                input_channels      = cfg["num_channels"],
                output_size         = n_classes,
                base_channels       = cfg.get("cnn_base_channels", 32),
                channel_multipliers = tuple(cfg.get("cnn_channel_multipliers", [1, 2, 4])),
                kernel_size         = _cfg_get(cfg, "conv1d_kernel_size"),
                dropout             = _cfg_get(cfg, "dropout"),
                pooling             = cfg.get("cnn_pooling", "mean"),
                size_linear_layers  = _cfg_get(cfg, "size_linear_layers"),
            )
    except Exception as exc:
        print(f"[WARNING] Could not build probe model for parameter count: {exc}")
    return None


def _append_trial_row(csv_path: str, row: dict) -> None:
    """Append one row to ``all_trials.csv``, creating it if necessary.

    Uses a CSV-safe append pattern: read existing file (if any), concat,
    and overwrite, so column order stays stable even across config changes.

    Args:
        csv_path: Absolute path to the trials CSV.
        row:      Dict of column to value for this trial.
    """
    df_new = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(csv_path, index=False)
    print(f"Trial row appended -> {csv_path}")


# ============================================================================
# Class balancing
# ============================================================================

def balance_dataset(
    X: torch.Tensor,
    y: np.ndarray,
    strategy: str,
    random_seed: int = 42,
) -> tuple:
    """
    Balance X (N, 1, T) and y (N,) according to strategy.

    Strategies
    ----------
    "none"        : return unchanged
    "undersample" : randomly drop majority classes, so all classes equal
                    the minority count
    "oversample"  : randomly repeat minority classes (with replacement)
    "hybrid"      : all classes brought to the median count
    """
    rng = np.random.default_rng(random_seed)

    if strategy == "none":
        return X, y

    classes, counts = np.unique(y, return_counts=True)
    count_map = dict(zip(classes, counts))

    print(f"\n{'-'*55}")
    print(f"  Balance strategy: '{strategy}'")
    print("  Before balancing:")
    for c, n in count_map.items():
        print(f"    class {c}: {n} segments")

    if strategy == "undersample":
        target = int(counts.min())
    elif strategy == "oversample":
        target = int(counts.max())
    elif strategy == "hybrid":
        target = int(np.median(counts).round())
    else:
        raise ValueError(
            f"Unknown balance_strategy '{strategy}'. "
            "Choose: 'none' | 'undersample' | 'oversample' | 'hybrid'."
        )

    kept_indices = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        n_c   = len(idx_c)
        if n_c > target:
            chosen = rng.choice(idx_c, size=target, replace=False)
        elif n_c < target:
            deficit = target - n_c
            extra   = rng.choice(idx_c, size=deficit, replace=True)
            chosen  = np.concatenate([idx_c, extra])
        else:
            chosen = idx_c
        kept_indices.append(chosen)

    all_idx = np.concatenate(kept_indices)
    rng.shuffle(all_idx)

    X_bal = X[all_idx]
    y_bal = y[all_idx]

    classes_after, counts_after = np.unique(y_bal, return_counts=True)
    print(f"  After balancing (target per class: {target}):")
    for c, n in zip(classes_after, counts_after):
        print(f"    class {c}: {n} segments")
    print(f"  Total: {len(y_bal)} segments  (was {len(y)})")
    print(f"{'-'*55}")

    return X_bal, y_bal


# ============================================================================
def main() -> None:
    wall_start = time.time()

    args = parse_args()
    cfg  = load_config(args.config)

    # reproducibility
    seed = cfg["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # device
    if cfg["force_cpu"]:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    num_cpus = os.cpu_count()
    print(f"\nDevice   : {device}")
    print(f"CPU cores: {num_cpus}")

    # load data
    filepath = os.path.join(cfg["path_data"], cfg["h5_filename"])
    print(f"\nLoading data from: {filepath}")
    df_timeseries = read_data_from_h5.read_data(filepath)
    print(f"{df_timeseries.shape[0]} time series read")

    # exclude unwanted states
    for state in cfg["exclude_states"]:
        before = len(df_timeseries)
        df_timeseries = df_timeseries[df_timeseries["state"] != state]
        print(f"Excluded state '{state}': {before} -> {len(df_timeseries)} rows")

    # class labels and integer encoding
    class_labels   = sorted(df_timeseries["state"].unique())   # sorted for stability
    state_to_label = {s: i for i, s in enumerate(class_labels)}
    df_timeseries["label"] = df_timeseries["state"].map(state_to_label)
    n_classes = len(class_labels)

    print(f"\nClass labels  : {class_labels}")
    print(f"State to label : {state_to_label}")

    # pad / truncate
    length_series = cfg["length_series"]
    print(f"\nPad/truncate to {length_series} samples ...")
    out_uni = pad_truncate_tensor.make_tensors(df_timeseries, seq_len=length_series)
    X_full  = out_uni["X"].unsqueeze(1)   # (n_rows, 1, seq_len)
    y_full  = out_uni["y"]                # (n_rows,)
    print(f"X shape: {tuple(X_full.shape)}   y shape: {tuple(y_full.shape)}")

    # segment
    seg_dur = cfg["segmentation_duration"]
    print(f"\nSegmenting with window={seg_dur} ...")
    seg_out = segmenting_data.segment_data(
        X_full.squeeze(1), y_full, segment_duration=seg_dur,
    )
    X_seg = seg_out["X"].unsqueeze(1)     # (n_segments, 1, seg_dur)
    y_seg = seg_out["y"]                  # (n_segments,)
    print(f"After segmentation: X={tuple(X_seg.shape)}  y={tuple(y_seg.shape)}")

    unique_cls, counts = torch.unique(y_seg, return_counts=True)
    for cls, cnt in zip(unique_cls.tolist(), counts.tolist()):
        print(f"  Class {cls} ({class_labels[cls]}): {cnt} segments")

    # class balancing
    ynumpy           = y_seg.numpy() if isinstance(y_seg, torch.Tensor) else np.array(y_seg)
    balance_strategy = cfg.get("balance_strategy", "none")

    if balance_strategy != "none":
        X_seg, ynumpy = balance_dataset(
            X_seg, ynumpy,
            strategy    = balance_strategy,
            random_seed = seed,
        )
        y_seg = torch.tensor(ynumpy, dtype=torch.long)

    # class weights
    present_classes = np.unique(ynumpy)
    all_classes     = np.arange(n_classes)
    missing         = np.setdiff1d(all_classes, present_classes)

    if missing.size > 0:
        print(f"WARNING: classes {missing} missing after segmentation, weights set to 0")

    weights_present = class_weight.compute_class_weight(
        class_weight="balanced", classes=present_classes, y=ynumpy,
    )
    weights_full = np.zeros(n_classes, dtype=np.float32)
    weights_full[present_classes] = weights_present
    class_weights = torch.tensor(weights_full, dtype=torch.float)

    print(f"Class weights : {class_weights}")
    print(f"Weights sum   : {class_weights.sum().item():.4f}")

    # model parameter count (before training)
    probe = _build_probe_model(cfg, n_classes, device)
    if probe is not None:
        total_params, trainable_params = _count_parameters(probe)
        del probe   # free immediately, just needed for the count
    else:
        total_params = trainable_params = -1   # sentinel: could not count
    print(f"\nModel parameters : {total_params:,}  (trainable: {trainable_params:,})")

    # output directory
    os.makedirs(cfg["path_save"], exist_ok=True)

    # optimizer: read from config; muon_lr only required for muon
    optimizer_name = cfg.get("optimizer", "adamW")
    muon_lr        = float(cfg["muon_lr"]) if optimizer_name == "muon" and "muon_lr" in cfg \
                     else 0.02   # passed to runner but ignored when optimizer != muon

    print(f"\nOptimizer : {optimizer_name}" +
          (f"  (muon_lr={muon_lr})" if optimizer_name == "muon" else ""))

    # resolve transformer keys (with defaults for CNN-only configs)
    embed_size         = _cfg_get(cfg, "embed_size")
    nhead_encoder      = _cfg_get(cfg, "nhead_encoder")
    dim_feedforward    = _cfg_get(cfg, "dim_feedforward")
    num_encoderlayers  = _cfg_get(cfg, "num_encoderlayers")
    dropout            = _cfg_get(cfg, "dropout")
    conv1d_emb         = _cfg_get(cfg, "conv1d_emb")
    conv1d_kernel_size = _cfg_get(cfg, "conv1d_kernel_size")
    size_linear_layers = _cfg_get(cfg, "size_linear_layers")

    # robustness sweep config
    noise_levels             = cfg.get("noise_levels", None)
    use_temperature_scaling  = cfg.get("use_temperature_scaling", True)
    ece_bins                 = int(cfg.get("ece_bins", 10))
    noise_batch_size         = int(cfg.get("noise_batch_size", 32))

    print(f"\nnoise_levels             : {noise_levels if noise_levels else 'disabled'}")
    if noise_levels:
        print(f"use_temperature_scaling  : {use_temperature_scaling}")
        print(f"ece_bins                 : {ece_bins}")
        print(f"noise_batch_size         : {noise_batch_size}")

    # training
    # Derive a globally unique run identifier for this process, so that
    # when multiple SLURM array tasks share the same cfg["path_save"] (as
    # they do for the CNN/Transformer baseline runs), each task's
    # robustness_sweep_run{N}.csv gets a distinct N instead of every task
    # writing "robustness_sweep_run1.csv" and colliding with each other.
    # Prefers SLURM_ARRAY_TASK_ID (set automatically by sbatch --array);
    # falls back to cfg["random_seed"] for local/non-SLURM runs, and
    # finally to None (internal loop index) if neither is available.
    external_run_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if external_run_id is not None:
        external_run_id = int(external_run_id)
        print(f"external_run_id = {external_run_id}  (from SLURM_ARRAY_TASK_ID)")
    else:
        external_run_id = cfg.get("random_seed")
        print(f"external_run_id = {external_run_id}  "
              f"(SLURM_ARRAY_TASK_ID not set, falling back to cfg['random_seed'])")

    print("\nStarting training ...\n")
    last_run_time = run_training_transformer_multiset.run_transformer_training(
        data               = X_seg,
        labels             = ynumpy,
        classes            = class_labels,
        device             = device,
        num_channels       = cfg["num_channels"],
        n_classes          = n_classes,
        test_size          = cfg["test_size"],
        val_size           = cfg["val_size"],
        batch_size         = cfg["batch_size"],
        num_cpus           = num_cpus,
        lr                 = cfg["lr"],
        muon_lr            = muon_lr,
        num_epochs         = cfg["num_epochs"],
        patience           = cfg["patience"],
        modeltype          = cfg["modeltype"],
        max_length_series  = cfg["segmentation_duration"],
        embed_size         = embed_size,
        nhead              = nhead_encoder,
        dim_feedforward    = dim_feedforward,
        num_encoderlayers  = num_encoderlayers,
        dropout            = dropout,
        conv1d_emb         = conv1d_emb,
        conv1d_kernel_size = conv1d_kernel_size,
        size_linear_layers = size_linear_layers,
        cnn_base_channels  = cfg.get("cnn_base_channels", 32),
        cnn_channel_multipliers = tuple(cfg.get("cnn_channel_multipliers", [1, 2, 4])),
        cnn_pooling        = cfg.get("cnn_pooling", "mean"),
        opt                = optimizer_name,
        verbose            = cfg["verbose"],
        pathsave           = cfg["path_save"],
        weights            = class_weights,
        norm_type          = cfg["norm_type"],
        num_training       = cfg["num_training"],
        noise_levels             = noise_levels,
        noise_batch_size         = noise_batch_size,
        use_temperature_scaling  = use_temperature_scaling,
        ece_bins                 = ece_bins,
        external_run_id          = external_run_id,
    )

    total_wall_time = time.time() - wall_start

    # read back per-run results to compute summary stats
    results_csv = os.path.join(cfg["path_save"], "results.csv")
    mean_acc = mean_f1 = mean_gmean = mean_pre = mean_rec = float("nan")
    std_acc  = std_f1  = std_gmean  = std_pre  = std_rec  = float("nan")

    if os.path.exists(results_csv):
        df_res = pd.read_csv(results_csv)
        # Drop the summary row written by run_training (non-numeric run_idx)
        df_numeric = df_res[pd.to_numeric(df_res["run_idx"], errors="coerce").notna()]
        for col in ("accuracy_test", "f1_test", "gmean_test",
                    "precision_test", "recall_test"):
            df_numeric = df_numeric.copy()   # silence SettingWithCopyWarning
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")

        def _mean(col): return df_numeric[col].mean() if col in df_numeric else float("nan")
        def _std(col):  return df_numeric[col].std()  if col in df_numeric else float("nan")

        mean_acc   = _mean("accuracy_test");   std_acc   = _std("accuracy_test")
        mean_f1    = _mean("f1_test");         std_f1    = _std("f1_test")
        mean_gmean = _mean("gmean_test");      std_gmean = _std("gmean_test")
        mean_pre   = _mean("precision_test");  std_pre   = _std("precision_test")
        mean_rec   = _mean("recall_test");     std_rec   = _std("recall_test")

    # all_trials.csv row
    trial_row: dict = {
        # bookkeeping
        "timestamp":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_file":             os.path.abspath(args.config),
        "path_save":               cfg["path_save"],
        "device":                  str(device),
        "random_seed":             cfg["random_seed"],
        # data
        "h5_filename":             cfg["h5_filename"],
        "n_classes":               n_classes,
        "class_labels":            str(class_labels),
        "n_segments":              int(X_seg.shape[0]),
        "segmentation_duration":   seg_dur,
        "num_channels":            cfg["num_channels"],
        "norm_type":               cfg["norm_type"],
        "balance_strategy":        balance_strategy,
        # model architecture
        "modeltype":               cfg["modeltype"],
        "total_params":            total_params,
        "trainable_params":        trainable_params,
        "embed_size":              embed_size,
        "nhead":                   nhead_encoder,
        "dim_feedforward":         dim_feedforward,
        "num_encoderlayers":       num_encoderlayers,
        "conv1d_emb":              conv1d_emb,
        "conv1d_kernel_size":      conv1d_kernel_size,
        "size_linear_layers":      size_linear_layers,
        "dropout":                 dropout,
        "cnn_base_channels":       cfg.get("cnn_base_channels", ""),
        "cnn_channel_multipliers": str(cfg.get("cnn_channel_multipliers", "")),
        "cnn_pooling":             cfg.get("cnn_pooling", ""),
        # training hyperparameters
        "optimizer":               optimizer_name,
        "lr":                      cfg["lr"],
        "muon_lr":                 muon_lr if optimizer_name == "muon" else None,
        "batch_size":              cfg["batch_size"],
        "num_epochs":              cfg["num_epochs"],
        "patience":                cfg["patience"],
        "num_training":            cfg["num_training"],
        "test_size":               cfg["test_size"],
        "val_size":                cfg["val_size"],
        # robustness sweep config
        "noise_levels":            str(noise_levels) if noise_levels else None,
        "use_temperature_scaling": use_temperature_scaling if noise_levels else None,
        "ece_bins":                ece_bins if noise_levels else None,
        # timing
        "total_wall_time_s":       round(total_wall_time, 1),
        "last_run_time_s":         round(last_run_time,   1),
        # aggregate metrics (mean +/- std across runs)
        "mean_accuracy":           round(mean_acc,   4),
        "std_accuracy":            round(std_acc,    4),
        "mean_f1":                 round(mean_f1,    4),
        "std_f1":                  round(std_f1,     4),
        "mean_gmean":              round(mean_gmean, 4),
        "std_gmean":               round(std_gmean,  4),
        "mean_precision":          round(mean_pre,   4),
        "std_precision":           round(std_pre,    4),
        "mean_recall":             round(mean_rec,   4),
        "std_recall":              round(std_rec,    4),
    }

    trials_csv = os.path.join(cfg["path_save"], "all_trials.csv")
    _append_trial_row(trials_csv, trial_row)

    print(f"\n{'='*55}")
    print(f"  Total wall time : {total_wall_time:.1f}s")
    print(f"  Parameters      : {total_params:,}  (trainable: {trainable_params:,})")
    print(f"  Optimizer       : {optimizer_name}" +
          (f"  (muon_lr={muon_lr})" if optimizer_name == "muon" else ""))
    print(f"  Mean accuracy   : {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Mean F1         : {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  Mean geometric mean : {mean_gmean:.4f} +/- {std_gmean:.4f}")
    print(f"  Mean precision    : {mean_pre:.4f} +/- {std_pre:.4f}")
    print(f"  Mean recall       : {mean_rec:.4f} +/- {std_rec:.4f}")
    if noise_levels:
        print(f"  Noise sweep       : {len(noise_levels)} sigma levels "
              f"-> robustness_all_runs.csv")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
