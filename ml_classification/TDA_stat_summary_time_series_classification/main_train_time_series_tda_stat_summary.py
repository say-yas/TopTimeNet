"""
main_train_time_series_tda_stat_summary.py
Single entry point for TDA-based time-series classification.

Two modes, selected by "use_cache" in config.json:

  use_cache = false  (default)
    Runs TDA inside every forward() call. Slow but no pre-processing step.
    Calls run_tda_stat_summary_training() (uses TDAEnd2EndNet)

  use_cache = true
    Pre-computes the full enriched TDA feature vector once, caches to disk.
    On subsequent runs with the same config, loads the cache; no TDA rerun.
    Calls run_cached_tda_training() (uses CachedGroupMLP)
    Training is roughly 100x faster because no TDA runs during the loop.

Cache behaviour:
  - If cache file does not exist, it is computed and saved automatically.
  - If cache file already exists, it is loaded immediately (no recompute).
  - To force recompute: delete the cache file or set "force_recompute": true.
"""

import argparse
import json
import logging
import os
import random
import time
import warnings

import numpy as np
import torch
from sklearn.utils import class_weight

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)


# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="TDA time-series classification pipeline"
    )
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
        if not k.startswith("_"):
            print(f"  {k:35s}: {v}")
    return cfg


# project imports
import ml_classification.utils.read_data_from_h5  as read_data_from_h5
import ml_classification.utils.pad_truncate_tensor as pad_truncate_tensor
import ml_classification.utils.segment_time_series as segmenting_data

import ml_classification.TDA_stat_summary_time_series_classification.run_training_tda_stat_summary \
    as run_training_tda_stat_summary
import ml_classification.TDA_stat_summary_time_series_classification.run_training_tda_cached \
    as run_training_tda_cached

# precompute_and_cache is the single canonical TDA feature computer for the
# cached path.
from ml_classification.TDA_stat_summary_time_series_classification.precompute_tda import (
    precompute_and_cache,
)


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
def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # reproducibility
    seed = cfg["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # device
    if cfg.get("force_cpu", False):
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

    # exclude / keep states
    for state in cfg.get("exclude_states", []):
        before = len(df_timeseries)
        df_timeseries = df_timeseries[df_timeseries["state"] != state]
        print(f"Excluded '{state}': {before} -> {len(df_timeseries)} rows")

    keep_states = cfg.get("keep_states", [])
    if keep_states:
        df_timeseries = df_timeseries[df_timeseries["state"].isin(keep_states)]
        print(f"Kept states: {keep_states}  ->  {len(df_timeseries)} rows")
        if len(df_timeseries) == 0:
            raise ValueError(
                f"No rows after keep_states={keep_states}. "
                "Check state names match the data exactly."
            )
        missing_states = set(keep_states) - set(df_timeseries["state"].unique())
        if missing_states:
            print(f"WARNING: states not found in data: {missing_states}")

    # class labels and integer encoding
    class_labels   = sorted(df_timeseries["state"].unique())
    state_to_label = {s: i for i, s in enumerate(class_labels)}
    df_timeseries["label"] = df_timeseries["state"].map(state_to_label)
    print(f"\nClass labels : {class_labels}")
    print(f"State to label : {state_to_label}")

    # pad / truncate
    length_series = cfg["length_series"]
    print(f"\nPad/truncate to {length_series} samples ...")
    out_uni = pad_truncate_tensor.make_tensors(df_timeseries, seq_len=length_series)
    X_full  = out_uni["X"].unsqueeze(1)
    y_full  = out_uni["y"]
    print(f"X: {tuple(X_full.shape)}   y: {tuple(y_full.shape)}")

    # segment
    seg_dur = cfg["segmentation_duration"]
    print(f"\nSegmenting with window={seg_dur} ...")
    seg_out = segmenting_data.segment_data(
        X_full.squeeze(1), y_full, segment_duration=seg_dur,
    )
    X_seg = seg_out["X"].unsqueeze(1)
    y_seg = seg_out["y"]
    print(f"After segmentation: X={tuple(X_seg.shape)}  y={tuple(y_seg.shape)}")

    unique_cls, counts = torch.unique(y_seg, return_counts=True)
    for cls, cnt in zip(unique_cls.tolist(), counts.tolist()):
        print(f"  Class {cls} ({class_labels[cls]}): {cnt} segments")

    # class balancing
    balance_strategy = cfg.get("balance_strategy", "none")
    ynumpy = (y_seg.numpy() if isinstance(y_seg, torch.Tensor)
              else np.array(y_seg))

    if balance_strategy != "none":
        X_seg, ynumpy = balance_dataset(
            X_seg, ynumpy,
            strategy    = balance_strategy,
            random_seed = seed,
        )
        y_seg = torch.tensor(ynumpy, dtype=torch.long)

    # class weights
    present_classes = np.unique(ynumpy)
    missing_cls     = np.setdiff1d(np.arange(len(class_labels)), present_classes)
    if missing_cls.size > 0:
        print(f"WARNING: classes {missing_cls} missing, weights set to 0")

    weights_present               = class_weight.compute_class_weight(
        "balanced", classes=present_classes, y=ynumpy,
    )
    weights_full                  = np.zeros(len(class_labels), dtype=np.float32)
    weights_full[present_classes] = weights_present
    class_weights                 = torch.tensor(weights_full, dtype=torch.float)
    print(f"Class weights: {class_weights.tolist()}")

    os.makedirs(cfg["path_save"], exist_ok=True)

    # shared training knobs
    label_smoothing = float(cfg.get("label_smoothing", 0.1))
    grad_clip_norm  = cfg.get("grad_clip_norm", 1.0)
    base_seed       = int(cfg.get("random_seed", 42))

    # muon_lr is only used when cfg["optimizer"] == "muon"
    muon_lr = float(cfg.get("muon_lr", 0.02))

    # noise_levels: None disables the robustness sweep; a list triggers it
    # after each run
    noise_levels = cfg.get("noise_levels", None)

    use_temperature_scaling = cfg.get("use_temperature_scaling", True)

    # noise augmentation sigma for cached mode (0.0 disables it)
    noise_aug_sigma = float(cfg.get("noise_aug_sigma", 0.05))

    # ========================================================================
    # Mode selection
    # ========================================================================
    use_cache = cfg.get("use_cache", False)

    if use_cache:
        # fast path (cached)
        cache_path      = cfg.get(
            "tda_cache_path",
            os.path.join(cfg["path_save"], "tda_cache.pt"),
        )
        force_recompute = cfg.get("force_recompute", False)

        print(f"\n{'-'*55}")
        print(f"  Cache path : {cache_path}")
        print(f"  Force recompute : {force_recompute}")
        print(f"{'-'*55}")

        _ = precompute_and_cache(
            X_seg       = X_seg,
            cfg         = cfg,
            cache_path  = cache_path,
            force       = force_recompute,
            y_seg       = torch.tensor(ynumpy, dtype=torch.long),
            clip_lo_pct = float(cfg.get("clip_lo_pct", 1.0)),
            clip_hi_pct = float(cfg.get("clip_hi_pct", 99.0)),
        )

        # reload full payload so we have labels, layout, clip bounds, etc.
        payload      = torch.load(cache_path, map_location="cpu", weights_only=True)
        features     = payload["features"]
        labels_t     = payload["labels"]
        feat_dim     = int(payload["feat_dim"])
        layout       = payload.get("layout", {})
        ynumpy_cache = labels_t.numpy()

        print(f"\n  features : {tuple(features.shape)}  feat_dim={feat_dim}")
        if layout:
            print("  Layout:")
            for grp, (s, e) in layout.items():
                print(f"    [{s:>3}:{e:>3}]  {grp}  ({e-s} dims)")
        if "clip_q_lo" in payload:
            print("  IQR clip bounds: present in cache")

        # recompute class weights from cached labels (may differ after balancing)
        present_cache = np.unique(ynumpy_cache)
        w_present     = class_weight.compute_class_weight(
            "balanced", classes=present_cache, y=ynumpy_cache,
        )
        w_full                = np.zeros(len(class_labels), dtype=np.float32)
        w_full[present_cache] = w_present
        class_weights_cache   = torch.tensor(w_full, dtype=torch.float)

        print("\nStarting cached training (CachedGroupMLP) ...\n")
        run_training_tda_cached.run_cached_tda_training(
            features     = features,
            labels       = ynumpy_cache,
            classes      = class_labels,
            device       = device,
            cfg          = cfg,
            n_classes    = len(class_labels),
            test_size    = cfg["test_size"],
            val_size     = cfg["val_size"],
            batch_size   = cfg["batch_size"],
            num_cpus     = num_cpus,
            lr           = cfg["lr"],
            num_epochs   = cfg["num_epochs"],
            patience     = cfg["patience"],
            opt          = cfg["optimizer"],
            muon_lr      = muon_lr,
            verbose      = cfg["verbose"],
            pathsave     = cfg["path_save"],
            weights      = class_weights_cache,
            norm_type    = cfg.get("norm_type",          "none"),
            num_training = cfg["num_training"],
            embed_dim    = cfg.get("embed_dim",            32),
            fusion       = cfg.get("fusion",        "low_rank"),
            rank         = cfg.get("rank",                  8),
            n_attn_layers= cfg.get("n_attn_layers",         1),
            n_heads      = cfg.get("n_heads",               4),
            head_hidden  = tuple(cfg.get("head_hidden", [64, 32])),
            dropout      = cfg.get("dropout",             0.1),
            activation   = cfg.get("activation",         "gelu"),
            n_hom_dims   = cfg.get("n_hom_dims",            2),
            reliability_threshold   = cfg.get("reliability_threshold", 0.6),
            label_smoothing         = label_smoothing,
            grad_clip_norm          = grad_clip_norm,
            base_seed               = base_seed,
            cache_path              = cache_path,
            layout                  = layout,
            noise_levels            = noise_levels,
            noise_aug_sigma         = noise_aug_sigma,
            use_temperature_scaling = use_temperature_scaling,
        )

    else:
        # slow path (end2end)
        print("\nStarting end-to-end training (TDAEnd2EndNet) ...\n")
        run_training_tda_stat_summary.run_tda_stat_summary_training(
            data         = X_seg,
            labels       = ynumpy,
            classes      = class_labels,
            device       = device,
            cfg          = cfg,
            n_classes    = len(class_labels),
            num_channels = cfg["num_channels"],
            test_size    = cfg["test_size"],
            val_size     = cfg["val_size"],
            batch_size   = cfg["batch_size"],
            num_cpus     = num_cpus,
            lr           = cfg["lr"],
            num_epochs   = cfg["num_epochs"],
            patience     = cfg["patience"],
            opt          = cfg["optimizer"],
            muon_lr      = muon_lr,
            verbose      = cfg["verbose"],
            pathsave     = cfg["path_save"],
            weights      = class_weights,
            norm_type    = cfg.get("norm_type",          "none"),
            num_training = cfg["num_training"],
            seg_len      = cfg.get("segmentation_duration", seg_dur),
            takens_dim   = cfg.get("takens_dim",             2),
            takens_delay = cfg.get("takens_delay",           5),
            n_hom_dims   = cfg.get("n_hom_dims",             2),
            n_betti_bins = cfg.get("n_betti_bins",          50),
            n_pi_bins    = cfg.get("n_pi_bins",             20),
            pi_sigma     = cfg.get("pi_sigma",            0.1),
            embed_dim    = cfg.get("embed_dim",             32),
            fusion       = cfg.get("fusion",        "low_rank"),
            rank         = cfg.get("rank",                  8),
            n_attn_layers= cfg.get("n_attn_layers",         1),
            n_heads      = cfg.get("n_heads",               4),
            head_hidden  = tuple(cfg.get("head_hidden", [64, 32])),
            ph_workers   = cfg.get("ph_workers",            4),
            dropout      = cfg.get("dropout",             0.1),
            activation   = cfg.get("activation",         "gelu"),
            reliability_threshold   = cfg.get("reliability_threshold", 0.6),
            label_smoothing         = label_smoothing,
            grad_clip_norm          = grad_clip_norm,
            base_seed               = base_seed,
            noise_levels            = noise_levels,
            use_temperature_scaling = use_temperature_scaling,
        )


if __name__ == "__main__":
    main()
