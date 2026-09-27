"""
search_best_config.py
TDA hyperparameter search driver: samples trial configurations from a
search space, trains each trial, tracks results, and rebuilds
best_config.json / best_config_small.json from the merged results.
"""

from __future__ import annotations

import argparse
import gc
import glob
import itertools
import json
import logging
import math
import os
import random
import tempfile
import time
import warnings
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.utils import class_weight

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import ml_classification.utils.read_data_from_h5  as read_data_from_h5
import ml_classification.utils.pad_truncate_tensor as pad_truncate_tensor
import ml_classification.utils.segment_time_series as segmenting_data

import ml_classification.TDA_stat_summary_time_series_classification.run_training_tda_stat_summary \
    as run_training_tda_stat_summary
import ml_classification.TDA_stat_summary_time_series_classification.run_training_tda_cached \
    as run_training_tda_cached

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
)


_DEFAULTS = {
    "label_smoothing" : 0.05,
    "grad_clip_norm"  : 1.0,
    "embed_dim"       : 32,
    "fusion"          : "low_rank",
    "rank"            : 8,
    "n_attn_layers"   : 1,
    "n_heads"         : 4,
    "ffn_dim"         : 0,
}

_PARETO_SCORE_TOL = 0.0


def parse_args():
    p = argparse.ArgumentParser(description="TDA hyperparameter search")
    p.add_argument("--config",           default="search_config.json")
    p.add_argument("--generate-configs", action="store_true")
    p.add_argument("--trial-idx",        type=int, default=None)
    p.add_argument("--merge-results",    action="store_true")
    return p.parse_args()


def _sanitize_for_json(obj):
    if isinstance(obj, np.bool_):   return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray): return [_sanitize_for_json(x) for x in obj.tolist()]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):  return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_sanitize_for_json(v) for v in obj]
    return obj


def _atomic_json_dump(data, path):
    clean   = _sanitize_for_json(data)
    dir_    = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(clean, f, indent=4)
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise


def _restore_type(key, val):
    """Cast a value read from CSV back to its correct Python type."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    int_keys = {
        "takens_dim", "takens_delay", "n_hom_dims", "n_betti_bins", "n_pi_bins",
        "ph_workers", "precompute_batch_size",
        "embed_dim", "rank", "n_attn_layers", "n_heads", "ffn_dim",
        "batch_size", "patience", "num_epochs", "num_training", "n_classes",
        "n_params", "n_epochs_trained", "trial_idx", "segmentation_duration",
    }
    bool_keys = {"force_cpu", "use_cache", "force_recompute", "verbose"}
    list_keys = {"head_hidden", "keep_states", "exclude_states"}
    float_keys = {
        "lr", "dropout", "pi_sigma", "test_size", "val_size",
        "label_smoothing", "grad_clip_norm", "reliability_threshold",
    }
    str_keys = {"fusion", "activation", "optimizer", "norm_type"}

    if key in int_keys:
        try: return int(float(val))
        except (ValueError, TypeError): return val
    if key in bool_keys:
        if isinstance(val, bool): return val
        if isinstance(val, (int, float)): return bool(val)
        if isinstance(val, str): return val.strip().lower() in ("true", "1", "yes")
        return val
    if key in list_keys:
        if isinstance(val, list): return val
        s = str(val).strip()
        try:
            parsed = json.loads(s.replace("'", '"'))
            return parsed if isinstance(parsed, list) else val
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            parts = s.strip("[]").replace(",", " ").split()
            return [int(p) for p in parts] if parts else []
        except ValueError:
            return val
    if key in float_keys:
        try: return float(val)
        except (ValueError, TypeError): return val
    if key in str_keys:
        return str(val)
    return val


def sample_param(spec, rng):
    t = spec["type"]
    if t in ("choice", "int_choice"):  return rng.choice(spec["values"])
    if t == "uniform":                 return rng.uniform(spec["low"], spec["high"])
    if t == "log_uniform":
        return math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"])))
    raise ValueError(f"Unknown param type: {t!r}")


def sample_config(search_space, rng):
    return {name: sample_param(spec, rng) for name, spec in search_space.items()}


def grid_configs(search_space):
    for name, spec in search_space.items():
        if spec["type"] not in ("choice", "int_choice"):
            raise ValueError(
                f"Grid search requires choice type but '{name}' has '{spec['type']}'."
            )
    keys   = list(search_space.keys())
    values = [search_space[k]["values"] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def is_valid(hparams, enforce_head):
    fusion    = str(hparams.get("fusion", "low_rank"))
    embed_dim = int(hparams.get("embed_dim", 32))
    n_heads   = int(hparams.get("n_heads",    4))
    if not enforce_head:
        return True
    if fusion in ("linear_attn", "mgta") and embed_dim % n_heads != 0:
        return False
    return True


def build_trial_list(search_space, strategy, n_trials, enforce_head, seed):
    rng = random.Random(seed)
    if strategy == "grid":
        all_h = [h for h in grid_configs(search_space) if is_valid(h, enforce_head)]
        print(f"Grid search: {len(all_h)} valid combinations")
    else:
        all_h, attempts = [], 0
        while len(all_h) < n_trials and attempts < n_trials * 20:
            h = sample_config(search_space, rng)
            if is_valid(h, enforce_head):
                all_h.append(h)
            attempts += 1
        print(f"Random search: {len(all_h)} trials ({attempts} attempts)")
    return all_h


def log_mem(label):
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
        print(f"  [MEM] {label}: {rss:.2f} GB RSS")
    except ImportError:
        pass


def aggressive_cleanup(trial_dir=None):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    for ckpt in glob.glob("checkpoint*.pt"):
        try: os.remove(ckpt)
        except OSError: pass
    if trial_dir:
        for ckpt in glob.glob(os.path.join(trial_dir, "checkpoint*.pt")):
            try: os.remove(ckpt)
            except OSError: pass
    gc.collect(); gc.collect()


def balance_dataset(X, y, strategy, random_seed=42):
    if strategy == "none":
        return X, y
    rng = np.random.default_rng(random_seed)
    classes, counts = np.unique(y, return_counts=True)
    if strategy == "undersample":  target = int(counts.min())
    elif strategy == "oversample": target = int(counts.max())
    elif strategy == "hybrid":     target = int(np.median(counts).round())
    else: raise ValueError(f"Unknown balance_strategy '{strategy}'.")
    kept = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        if len(idx_c) > target:
            chosen = rng.choice(idx_c, size=target, replace=False)
        elif len(idx_c) < target:
            extra  = rng.choice(idx_c, size=target - len(idx_c), replace=True)
            chosen = np.concatenate([idx_c, extra])
        else:
            chosen = idx_c
        kept.append(chosen)
    all_idx = np.concatenate(kept)
    rng.shuffle(all_idx)
    return X[all_idx], y[all_idx]


def _build_layout(n_hom):
    pc_dim  = 4
    ent_dim = n_hom
    lt_dim  = n_hom * LifetimeStatsLayer._S
    b_dim   = n_hom * TDABettiExtractor._S
    p_dim   = n_hom * TDAPIExtractor._S
    starts  = np.cumsum([0, pc_dim, ent_dim, lt_dim, b_dim, p_dim])
    return {
        "pc_stats"    : (int(starts[0]), int(starts[1])),
        "entropy"     : (int(starts[1]), int(starts[2])),
        "lifetime"    : (int(starts[2]), int(starts[3])),
        "betti_stats" : (int(starts[3]), int(starts[4])),
        "pi_stats"    : (int(starts[4]), int(starts[5])),
    }


def _compute_tda_cache(X_seg, y_seg, fixed, cache_path):
    takens_dim   = fixed.get("takens_dim",            2)
    takens_delay = fixed.get("takens_delay",           5)
    n_hom_dims   = fixed.get("n_hom_dims",             2)
    n_betti_bins = fixed.get("n_betti_bins",          50)
    n_pi_bins    = fixed.get("n_pi_bins",             15)
    pi_sigma     = fixed.get("pi_sigma",            0.05)
    ph_workers   = fixed.get("ph_workers",             4)
    batch_sz     = fixed.get("precompute_batch_size", 64)

    layout   = _build_layout(n_hom_dims)
    feat_dim = layout["pi_stats"][1]

    print(f"\n  Computing TDA cache  ({len(X_seg)} segments, feat_dim={feat_dim}) ...")

    takens     = TakensLayer(dim=takens_dim, delay=takens_delay)
    pc_stats_l = PointCloudStatsLayer()
    ph         = RipserPHLayer(maxdim=n_hom_dims - 1, max_workers=ph_workers)
    ph_ent_l   = PersistenceEntropyLayer(n_hom_dims=n_hom_dims)
    lt_stats_l = LifetimeStatsLayer(n_hom_dims=n_hom_dims)
    betti_l    = BettiCurveLayer(n_hom_dims=n_hom_dims, n_bins=n_betti_bins)
    pi_l       = PILayer(n_hom_dims=n_hom_dims, n_pi_bins=n_pi_bins, sigma=pi_sigma)
    b_ext      = TDABettiExtractor(n_hom_dims=n_hom_dims, n_betti_bins=n_betti_bins)
    p_ext      = TDAPIExtractor(n_hom_dims=n_hom_dims, n_pi_bins=n_pi_bins)

    dev = torch.device("cpu")
    N   = len(X_seg)
    all_feats = []
    t0  = time.time()

    x_w = X_seg[:min(batch_sz, N)]
    with torch.no_grad():
        pcs_w   = takens(x_w)
        pc_w    = pc_stats_l(pcs_w, dev)
        diams_w = pc_w[:, 0].cpu().numpy()
        dgms_w  = ph(pcs_w)
        _       = betti_l(dgms_w, dev, diameters=diams_w)
        _       = pi_l(dgms_w, dev)

    for start in range(0, N, batch_sz):
        end = min(start + batch_sz, N)
        x   = X_seg[start:end]
        with torch.no_grad():
            pcs       = takens(x)
            pc_stat_t = pc_stats_l(pcs, dev)
            diameters = pc_stat_t[:, 0].cpu().numpy()
            diagrams  = ph(pcs)
            ent_t     = ph_ent_l(diagrams, dev)
            lt_t      = lt_stats_l(diagrams, dev)
            betti_t   = betti_l(diagrams, dev, diameters=diameters)
            pi_t      = pi_l(diagrams, dev)
            b_stats   = b_ext(betti_t)
            p_stats   = p_ext(pi_t)
            feats     = torch.cat([pc_stat_t, ent_t, lt_t, b_stats, p_stats], dim=1)
        all_feats.append(feats.cpu())
        print(f"    {end:>6}/{N}  [{time.time()-t0:5.1f}s]", end="\r")

    print(f"\n  Done in {time.time()-t0:.1f}s")

    features = torch.cat(all_feats, dim=0)
    labels   = (y_seg.long() if isinstance(y_seg, torch.Tensor)
                else torch.tensor(y_seg, dtype=torch.long))

    payload = {
        "features"     : features,
        "labels"       : labels,
        "feat_dim"     : feat_dim,
        "layout"       : layout,
        "cfg_snapshot" : {
            "takens_dim"    : takens_dim, "takens_delay" : takens_delay,
            "n_hom_dims"    : n_hom_dims, "n_betti_bins" : n_betti_bins,
            "n_pi_bins"     : n_pi_bins,  "pi_sigma"     : pi_sigma,
            "_layer_dims"   : {
                "LifetimeStatsLayer._S" : LifetimeStatsLayer._S,
                "TDABettiExtractor._S"  : TDABettiExtractor._S,
                "TDAPIExtractor._S"     : TDAPIExtractor._S,
            },
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    torch.save(payload, cache_path)
    print(f"  Saved -> {cache_path}  ({os.path.getsize(cache_path)/1e6:.1f} MB)")
    return features, labels, feat_dim, layout


def load_and_preprocess(fixed, device):
    filepath = os.path.join(fixed["path_data"], fixed["h5_filename"])
    print(f"Loading: {filepath}")
    log_mem("before read_data")

    df = read_data_from_h5.read_data(filepath)

    for state in fixed.get("exclude_states", []):
        df = df[df["state"] != state]
    keep_states = fixed.get("keep_states", [])
    if keep_states:
        df = df[df["state"].isin(keep_states)]
        print(f"Kept states: {keep_states}  ->  {len(df)} rows")
        if len(df) == 0:
            raise ValueError(f"No rows after keep_states={keep_states}")

    class_labels   = sorted(df["state"].unique())
    state_to_label = {s: i for i, s in enumerate(class_labels)}
    df["label"]    = df["state"].map(state_to_label)
    print(f"Classes: {state_to_label}")
    log_mem("after read_data")

    out_uni = pad_truncate_tensor.make_tensors(df, seq_len=fixed["length_series"])
    X_full  = out_uni["X"].unsqueeze(1)
    y_full  = out_uni["y"]
    del df, out_uni
    gc.collect()

    print(f"X_full: {tuple(X_full.shape)}  y_full: {tuple(y_full.shape)}")

    ynumpy = y_full.numpy() if isinstance(y_full, torch.Tensor) else np.array(y_full)

    balance_strategy = fixed.get("balance_strategy", "none")
    if balance_strategy != "none":
        X_full, ynumpy = balance_dataset(
            X_full, ynumpy,
            strategy    = balance_strategy,
            random_seed = fixed.get("random_seed", 42),
        )
        y_full = torch.tensor(ynumpy, dtype=torch.long)
        print(f"After balancing ({balance_strategy}): {len(ynumpy)} series")

    present = np.unique(ynumpy)
    wp      = class_weight.compute_class_weight("balanced", classes=present, y=ynumpy)
    wf      = np.zeros(len(class_labels), dtype=np.float32)
    wf[present] = wp
    class_weights_tensor = torch.tensor(wf, dtype=torch.float)

    tda_features = tda_labels = feat_dim = layout = None
    _cache_path  = fixed.get(
        "tda_cache_path",
        os.path.join(fixed["path_save"], "tda_cache.pt"),
    )
    if fixed.get("use_cache", False) and os.path.exists(_cache_path):
        # map_location="cpu" so a GPU-saved cache loads on CPU-only machines
        _payload     = torch.load(_cache_path, map_location="cpu", weights_only=True)
        tda_features = _payload["features"]
        feat_dim     = int(_payload["feat_dim"])
        layout       = _payload.get("layout", {})
        if "labels" in _payload:
            tda_labels = _payload["labels"]
            print(f"[load_and_preprocess] Cache loaded: "
                  f"{tuple(tda_features.shape)}  feat_dim={feat_dim}")
        else:
            print("[load_and_preprocess] ERROR: cache missing 'labels' key.")
            print(f"  Delete and rebuild:  rm {_cache_path}")
            tda_features = tda_labels = feat_dim = layout = None
    elif fixed.get("use_cache", False):
        print("[load_and_preprocess] WARNING: use_cache=true but cache not found:")
        print(f"  {_cache_path}")

    return (X_full, y_full, class_labels, class_weights_tensor,
            tda_features, tda_labels, feat_dim, layout)


def run_trial(
    trial_idx, hparams, fixed,
    X_full, y_full, class_labels, class_weights,
    device, num_cpus, alpha, path_save,
    tda_features=None, tda_labels=None, feat_dim=None,
    layout=None,
):
    trial_dir = os.path.join(path_save, f"trial_{trial_idx:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Trial {trial_idx:4d}  |  {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'-'*60}")
    for k, v in hparams.items():
        print(f"  {k:30s}: {v}")
    print(f"{'-'*60}")
    log_mem("trial start")

    use_cache = fixed.get("use_cache", False)
    n_classes = len(class_labels)

    seg_len = int(hparams.get(
        "segmentation_duration",
        fixed.get("segmentation_duration", 500),
    ))
    print(f"  segmentation_duration : {seg_len}")

    seg_out = segmenting_data.segment_data(
        X_full.squeeze(1), y_full, segment_duration=seg_len,
    )
    X_seg  = seg_out["X"].unsqueeze(1)
    y_seg  = seg_out["y"]
    ynumpy = y_seg.numpy() if isinstance(y_seg, torch.Tensor) else np.array(y_seg)
    print(f"  X_seg : {tuple(X_seg.shape)}")

    present_seg = np.unique(ynumpy)
    wp_seg      = class_weight.compute_class_weight("balanced", classes=present_seg, y=ynumpy)
    wf_seg               = np.zeros(n_classes, dtype=np.float32)
    wf_seg[present_seg]  = wp_seg
    class_weights        = torch.tensor(wf_seg, dtype=torch.float)

    label_smoothing = float(hparams.get(
        "label_smoothing", fixed.get("label_smoothing", _DEFAULTS["label_smoothing"])
    ))
    grad_clip_norm  = float(hparams.get(
        "grad_clip_norm",  fixed.get("grad_clip_norm",  _DEFAULTS["grad_clip_norm"])
    ))

    embed_dim    = int(hparams.get("embed_dim",    fixed.get("embed_dim",    _DEFAULTS["embed_dim"])))
    fusion       = str(hparams.get("fusion",       fixed.get("fusion",       _DEFAULTS["fusion"])))
    rank         = int(hparams.get("rank",         fixed.get("rank",         _DEFAULTS["rank"])))
    n_attn_layers= int(hparams.get("n_attn_layers",fixed.get("n_attn_layers",_DEFAULTS["n_attn_layers"])))
    n_heads      = int(hparams.get("n_heads",      fixed.get("n_heads",      _DEFAULTS["n_heads"])))
    ffn_dim      = int(hparams.get("ffn_dim",      fixed.get("ffn_dim",      _DEFAULTS["ffn_dim"])))
    head_hidden  = tuple(hparams.get("head_hidden", fixed.get("head_hidden", [64, 32])))
    dropout      = float(hparams.get("dropout",    fixed.get("dropout",      0.1)))
    activation   = str(hparams.get("activation",  fixed.get("activation",   "gelu")))

    try:
        if use_cache and tda_features is not None and tda_labels is not None:
            ynumpy_cache = (tda_labels.numpy() if isinstance(tda_labels, torch.Tensor)
                            else np.array(tda_labels))
            present_c   = np.unique(ynumpy_cache)
            wp_c        = class_weight.compute_class_weight("balanced", classes=present_c, y=ynumpy_cache)
            wf_c             = np.zeros(n_classes, dtype=np.float32)
            wf_c[present_c]  = wp_c
            cw_cache         = torch.tensor(wf_c, dtype=torch.float)

            computation_time = run_training_tda_cached.run_cached_tda_training(
                features      = tda_features,
                labels        = ynumpy_cache,
                classes       = class_labels,
                device        = device,
                cfg           = fixed,
                n_classes     = n_classes,
                test_size     = fixed["test_size"],
                val_size      = fixed["val_size"],
                batch_size    = int(hparams.get("batch_size", 128)),
                num_cpus      = num_cpus,
                lr            = float(hparams.get("lr",       1e-3)),
                num_epochs    = fixed["num_epochs"],
                patience      = int(hparams.get("patience",    20)),
                opt           = str(hparams.get("optimizer",   fixed.get("optimizer", "adam"))),
                verbose       = fixed.get("verbose", False),
                pathsave      = trial_dir + "/",
                weights       = cw_cache,
                norm_type     = str(hparams.get("norm_type",  "none")),
                num_training  = fixed["num_training"],
                embed_dim     = embed_dim,
                fusion        = fusion,
                rank          = rank,
                n_attn_layers = n_attn_layers,
                n_heads       = n_heads,
                ffn_dim       = ffn_dim,
                head_hidden   = head_hidden,
                dropout       = dropout,
                activation    = activation,
                n_hom_dims    = fixed.get("n_hom_dims", 2),
                reliability_threshold = fixed.get("reliability_threshold", 0.6),
                label_smoothing       = label_smoothing,
                grad_clip_norm        = grad_clip_norm,
                base_seed             = fixed.get("random_seed", 42),
                cache_path            = fixed.get("tda_cache_path"),
                layout                = layout,
            )
            csv_name = "results_cached_tda.csv"

        else:
            computation_time = run_training_tda_stat_summary.run_tda_stat_summary_training(
                data          = X_seg,
                labels        = ynumpy,
                classes       = class_labels,
                device        = device,
                cfg           = fixed,
                n_classes     = n_classes,
                num_channels  = fixed.get("num_channels", 1),
                test_size     = fixed["test_size"],
                val_size      = fixed["val_size"],
                batch_size    = int(hparams.get("batch_size", 128)),
                num_cpus      = num_cpus,
                lr            = float(hparams.get("lr",        1e-3)),
                num_epochs    = fixed["num_epochs"],
                patience      = int(hparams.get("patience",     20)),
                opt           = str(hparams.get("optimizer",    fixed.get("optimizer", "adam"))),
                verbose       = fixed.get("verbose", False),
                pathsave      = trial_dir + "/",
                weights       = class_weights,
                norm_type     = str(hparams.get("norm_type",   "none")),
                num_training  = fixed["num_training"],
                seg_len       = seg_len,
                takens_dim    = int(hparams.get("takens_dim",   fixed.get("takens_dim",   2))),
                takens_delay  = int(hparams.get("takens_delay", fixed.get("takens_delay", 5))),
                n_hom_dims    = fixed.get("n_hom_dims", 2),
                n_betti_bins  = int(hparams.get("n_betti_bins", fixed.get("n_betti_bins", 50))),
                n_pi_bins     = int(hparams.get("n_pi_bins",    fixed.get("n_pi_bins",    15))),
                pi_sigma      = float(hparams.get("pi_sigma",   fixed.get("pi_sigma",   0.05))),
                embed_dim     = embed_dim,
                fusion        = fusion,
                rank          = rank,
                n_attn_layers = n_attn_layers,
                n_heads       = n_heads,
                ffn_dim       = ffn_dim,
                head_hidden   = head_hidden,
                ph_workers    = fixed.get("ph_workers", 4),
                dropout       = dropout,
                activation    = activation,
                reliability_threshold = fixed.get("reliability_threshold", 0.6),
                label_smoothing       = label_smoothing,
                grad_clip_norm        = grad_clip_norm,
                base_seed             = fixed.get("random_seed", 42),
            )
            csv_name = "results.csv"

        csv_path = os.path.join(trial_dir, csv_name)
        df_trial = pd.read_csv(csv_path)
        df_runs  = df_trial[df_trial["run_idx"] != "MEAN +/- STD"].copy()

        score_cols = [
            "accuracy_test", "f1_test", "gmean_test",
            "precision_test", "recall_test",
            "reliability_test", "neutral_pct_test",
            "final_train_loss", "final_val_loss",
            "final_train_acc",  "final_val_acc",
            "final_train_f1",   "final_val_f1",
            "final_train_rel",  "final_val_rel",
            "n_epochs_trained",
            "n_params",
        ]

        for col in score_cols:
            if col in df_runs.columns:
                df_runs[col] = pd.to_numeric(df_runs[col], errors="coerce")

        means   = {col: float(df_runs[col].mean())
                   for col in score_cols if col in df_runs.columns}
        f1_mean = means.get("f1_test",    float("nan"))
        gm_mean = means.get("gmean_test", float("nan"))
        score   = alpha * f1_mean + (1.0 - alpha) * gm_mean

        n_params_mean = means.get("n_params", float("nan"))
        n_params_int  = (
            int(round(n_params_mean))
            if not math.isnan(n_params_mean)
            else None
        )

        rel_class_cols  = [c for c in df_runs.columns if c.startswith("rel_class_")]
        rel_class_means = {col: round(float(df_runs[col].mean()), 4) for col in rel_class_cols}
        del df_trial, df_runs

        result = {
            "trial_idx"        : trial_idx,
            "score"            : round(score,   4),
            "f1_test"          : round(f1_mean, 4),
            "gmean_test"       : round(gm_mean, 4),
            "accuracy_test"    : round(means.get("accuracy_test",    float("nan")), 4),
            "precision_test"   : round(means.get("precision_test",   float("nan")), 4),
            "recall_test"      : round(means.get("recall_test",      float("nan")), 4),
            "reliability_test" : round(means.get("reliability_test", float("nan")), 4),
            "neutral_pct_test" : round(means.get("neutral_pct_test", float("nan")), 4),
            "final_train_loss" : round(means.get("final_train_loss", float("nan")), 4),
            "final_val_loss"   : round(means.get("final_val_loss",   float("nan")), 4),
            "final_val_acc"    : round(means.get("final_val_acc",    float("nan")), 4),
            "final_val_f1"     : round(means.get("final_val_f1",     float("nan")), 4),
            "final_val_rel"    : round(means.get("final_val_rel",    float("nan")), 4),
            "n_epochs_trained" : round(means.get("n_epochs_trained", float("nan")), 1),
            "computation_time" : round(computation_time, 1),
            "n_params"         : n_params_int,
            "status"           : "ok",
            **rel_class_means,
            **{f"hp_{k}": v for k, v in hparams.items()},
            "hp_label_smoothing"  : label_smoothing,
            "hp_grad_clip_norm"   : grad_clip_norm,
            "hp_embed_dim"        : embed_dim,
            "hp_fusion"           : fusion,
            "hp_rank"             : rank,
            "hp_n_attn_layers"    : n_attn_layers,
            "hp_n_heads"          : n_heads,
        }

    except Exception as e:
        print(f"  Trial {trial_idx} FAILED: {e}")
        import traceback; traceback.print_exc()
        result = {
            "trial_idx"           : trial_idx,
            "score"               : -1.0,
            "f1_test"             : float("nan"),
            "gmean_test"          : float("nan"),
            "accuracy_test"       : float("nan"),
            "reliability_test"    : float("nan"),
            "computation_time"    : float("nan"),
            "n_params"            : None,
            "status"              : f"failed: {e}",
            **{f"hp_{k}": v for k, v in hparams.items()},
            "hp_label_smoothing"  : label_smoothing,
            "hp_grad_clip_norm"   : grad_clip_norm,
            "hp_embed_dim"        : embed_dim,
            "hp_fusion"           : fusion,
            "hp_rank"             : rank,
        }

    finally:
        aggressive_cleanup(trial_dir)
        log_mem("after cleanup")

    pd.DataFrame([result]).to_csv(
        os.path.join(trial_dir, "trial_summary.csv"), index=False
    )
    return result


def _write_best_config(best_row, fixed, out_path):
    best_config = deepcopy(fixed)
    for k, v in best_row.items():
        if str(k).startswith("hp_"):
            key = k[3:]
            best_config[key] = _restore_type(key, v)
    for key, default_val in _DEFAULTS.items():
        if key not in best_config:
            best_config[key] = default_val
    _atomic_json_dump(best_config, out_path)
    print(f"  Written -> {out_path}")


def _find_pareto_small(df_ok: pd.DataFrame, tol: float = 0.0) -> pd.Series:
    df = df_ok.copy()
    df["_score"]   = pd.to_numeric(df["score"],    errors="coerce")
    df["_nparams"] = pd.to_numeric(df["n_params"], errors="coerce")
    df = df.dropna(subset=["_score", "_nparams"]).reset_index(drop=True)

    if df.empty:
        print("  WARNING: no trials with both score and n_params, "
              "falling back to best score.")
        return df_ok.sort_values("score", ascending=False).iloc[0]

    scores  = df["_score"].values
    nparams = df["_nparams"].values
    n       = len(df)

    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            j_score_ok  = scores[j]  >= scores[i]  - tol
            j_params_ok = nparams[j] <= nparams[i]
            j_better    = (scores[j] > scores[i] + tol) or (nparams[j] < nparams[i])
            if j_score_ok and j_params_ok and j_better:
                is_dominated[i] = True
                break

    pareto_idx = np.where(~is_dominated)[0]
    df_pareto  = df.iloc[pareto_idx].copy()
    df_pareto  = df_pareto.sort_values(
        ["_score", "_nparams"], ascending=[False, True]
    ).reset_index(drop=True)

    chosen = df_pareto.iloc[0]
    print(f"\n  Pareto front ({len(pareto_idx)} trials):")
    for _, row in df_pareto.iterrows():
        marker = " <- CHOSEN" if row["trial_idx"] == chosen["trial_idx"] else ""
        print(f"    trial={int(row['trial_idx']):4d}  "
              f"score={row['_score']:.4f}  "
              f"n_params={int(row['_nparams']):,}{marker}")

    orig_idx = df_ok.index[
        df_ok["trial_idx"] == chosen["trial_idx"]
    ].tolist()
    return df_ok.loc[orig_idx[0]] if orig_idx else df_ok.iloc[
        df_ok["score"].astype(float).idxmax()
    ]


def merge_results(path_save, alpha, fixed):
    summaries = sorted(glob.glob(os.path.join(path_save, "trial_*/trial_summary.csv")))
    if not summaries:
        print("No trial_summary.csv files found.")
        return

    df_final = pd.concat(
        [pd.read_csv(f) for f in summaries], ignore_index=True
    ).sort_values("score", ascending=False)

    summary_csv = os.path.join(path_save, "all_trials.csv")
    df_final.to_csv(summary_csv, index=False)
    print(f"Merged {len(summaries)} trials -> {summary_csv}")

    df_ok = df_final[df_final["status"] == "ok"]
    if df_ok.empty:
        print("All trials failed.")
        return

    best = df_ok.iloc[0]
    print(f"\n  BEST (score)  trial={int(best['trial_idx'])}  "
          f"score={best['score']:.4f}  "
          f"f1={best.get('f1_test', float('nan')):.4f}  "
          f"gmean={best.get('gmean_test', float('nan')):.4f}  "
          f"n_params={best.get('n_params', 'n/a')}")
    for k, v in best.items():
        if str(k).startswith("hp_"):
            print(f"    {k[3:]:30s}: {v}")

    _write_best_config(best, fixed, os.path.join(path_save, "best_config.json"))

    best_small = _find_pareto_small(df_ok, tol=_PARETO_SCORE_TOL)
    small_path = os.path.join(path_save, "best_config_small.json")

    if int(best_small["trial_idx"]) == int(best["trial_idx"]):
        print(f"\n  BEST SMALL = BEST (same trial {int(best['trial_idx'])}), "
              f"no separate small config written.")
    else:
        print(f"\n  BEST SMALL (Pareto)  "
              f"trial={int(best_small['trial_idx'])}  "
              f"score={float(best_small['score']):.4f}  "
              f"n_params={best_small.get('n_params', 'n/a')}")
        _write_best_config(best_small, fixed, small_path)
        print(f"  Small config -> {small_path}")

    seg_col = ("hp_segmentation_duration" if "hp_segmentation_duration" in df_ok.columns
               else "segmentation_duration" if "segmentation_duration" in df_ok.columns
               else None)
    seg_durs = (sorted(df_ok[seg_col].dropna().unique()) if seg_col is not None
                else [fixed.get("segmentation_duration", None)])

    for seg_dur in seg_durs:
        if seg_dur is None:
            continue
        seg_dur_int = int(seg_dur)
        df_seg = df_ok[df_ok[seg_col] == seg_dur] if seg_col is not None else df_ok
        if df_seg.empty:
            continue
        best_seg = df_seg.iloc[0]
        out_path = os.path.join(path_save, f"best_config_seg{seg_dur_int}.json")
        fixed_with_seg = deepcopy(fixed)
        fixed_with_seg["segmentation_duration"] = seg_dur_int
        _write_best_config(best_seg, fixed_with_seg, out_path)
        print(f"    seg_dur={seg_dur_int}  ->  {out_path}")


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    with open(args.config) as f:
        cfg = json.load(f)

    fixed        = cfg["fixed"]
    search_space = cfg["search_space"]
    strategy     = cfg.get("strategy",                  "random")
    n_trials     = cfg.get("n_trials",                  50)
    alpha        = cfg.get("objective_alpha",            0.6)
    enforce_head = cfg.get("enforce_head_divisibility",  True)
    seed         = fixed.get("random_seed",              42)
    path_save    = fixed["path_save"]

    os.makedirs(path_save, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.merge_results:
        merge_results(path_save, alpha, fixed)
        return

    if fixed.get("force_cpu", False):       device = torch.device("cpu")
    elif torch.cuda.is_available():         device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else:                                   device = torch.device("cpu")

    num_cpus    = os.cpu_count()
    configs_dir = os.path.join(path_save, "trial_configs")

    if args.generate_configs:
        all_hparams = build_trial_list(search_space, strategy, n_trials, enforce_head, seed)
        os.makedirs(configs_dir, exist_ok=True)
        for i, h in enumerate(all_hparams, start=1):
            _atomic_json_dump(h, os.path.join(configs_dir, f"trial_{i:04d}.json"))
        print(f"Trial configs -> {configs_dir}  ({len(all_hparams)} files)")
        print("Config generation complete.")
        return

    if args.trial_idx is not None:
        cfg_path = os.path.join(configs_dir, f"trial_{args.trial_idx:04d}.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"Trial config not found: {cfg_path}\nRun --generate-configs first."
            )
        with open(cfg_path) as f:
            hparams = json.load(f)
        print(f"Device: {device}  |  CPUs: {num_cpus}")
        (X_full, y_full, class_labels, cw,
         tda_features, tda_labels, feat_dim, layout) = load_and_preprocess(fixed, device)
        run_trial(args.trial_idx, hparams, fixed,
                  X_full, y_full, class_labels, cw,
                  device, num_cpus, alpha, path_save,
                  tda_features, tda_labels, feat_dim, layout)
        return

    all_hparams = build_trial_list(search_space, strategy, n_trials, enforce_head, seed)
    os.makedirs(configs_dir, exist_ok=True)
    for i, h in enumerate(all_hparams, start=1):
        _atomic_json_dump(h, os.path.join(configs_dir, f"trial_{i:04d}.json"))
    print(f"Trial configs -> {configs_dir}  ({len(all_hparams)} files)")
    print(f"Device: {device}  |  CPUs: {num_cpus}")

    (X_full, y_full, class_labels, cw,
     tda_features, tda_labels, feat_dim, layout) = load_and_preprocess(fixed, device)
    log_mem("data ready")

    all_results = []
    summary_csv = os.path.join(path_save, "all_trials.csv")

    for trial_idx, hparams in enumerate(all_hparams, start=1):
        result = run_trial(
            trial_idx, hparams, fixed,
            X_full, y_full, class_labels, cw,
            device, num_cpus, alpha, path_save,
            tda_features, tda_labels, feat_dim, layout,
        )
        all_results.append(result)

        df_r = pd.DataFrame(all_results).sort_values("score", ascending=False)
        df_r.to_csv(summary_csv, index=False)
        del df_r

        df_top = (pd.DataFrame(all_results)
                  .pipe(lambda d: d[d["status"] == "ok"])
                  .sort_values("score", ascending=False)
                  .head(5))
        print(f"\n-- Leaderboard after trial {trial_idx} --")
        if not df_top.empty:
            cols = [c for c in ["trial_idx", "score", "f1_test", "gmean_test",
                                 "reliability_test", "n_params",
                                 "hp_fusion", "hp_embed_dim", "hp_rank",
                                 "hp_label_smoothing", "hp_grad_clip_norm",
                                 "hp_segmentation_duration"]
                    if c in df_top.columns]
            print(df_top[cols].to_string(index=False))
        del df_top

    merge_results(path_save, alpha, fixed)


if __name__ == "__main__":
    main()

