"""
search_best_config.py
Hyperparameter search for TransformerI / CNNI time-series classifiers.

Modes
-----
    # 1. Generate per-trial JSON configs (no training)
    python hparam_search.py --config search_config.json --generate-configs

    # 2. Run all trials sequentially
    python hparam_search.py --config search_config.json

    # 3. Run a single trial (for SLURM array jobs)
    python hparam_search.py --config search_config.json --trial-idx 7

    # 4. Merge per-trial CSVs into all_trials.csv and best configs
    python hparam_search.py --config search_config.json --merge-results
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
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.utils import class_weight

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import ml_classification.utils.read_data_from_h5 as read_data_from_h5
import ml_classification.utils.pad_truncate_tensor as pad_truncate_tensor
import ml_classification.transformer_cnn_time_series.run_training_transformer_multiset as run_training_transformer_multiset
import ml_classification.utils.segment_time_series as segmenting_data


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperparameter search")
    p.add_argument("--config",           default="search_config.json",
                   help="Path to search_config.json")
    p.add_argument("--generate-configs", action="store_true",
                   help="Write trial JSON configs then exit (no training)")
    p.add_argument("--trial-idx",        type=int, default=None,
                   help="Run a single trial by index (for SLURM array jobs)")
    p.add_argument("--merge-results",    action="store_true",
                   help="Merge trial_summary.csv files and write best configs")
    return p.parse_args()


# ============================================================================
# JSON helpers
# ============================================================================

def _sanitize_for_json(obj: Any) -> Any:
    """Recursively make an object JSON-serialisable.

    Converts numpy scalars, non-finite floats, and numpy arrays to their
    Python equivalents. ``NaN`` and ``Inf`` become ``None``.

    Args:
        obj: Any Python / numpy value.

    Returns:
        A JSON-serialisable value.
    """
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _atomic_json_dump(data: dict, path: str) -> None:
    """Write *data* as JSON atomically via a temp file and ``os.replace``.

    Args:
        data: Dict to serialise (``_sanitize_for_json`` applied automatically).
        path: Destination file path.
    """
    clean = _sanitize_for_json(data)
    dir_  = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(clean, f, indent=4)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ============================================================================
# Type restoration (mirrors rebuild_best_config.py)
# ============================================================================

_INT_KEYS: frozenset[str] = frozenset({
    "batch_size", "embed_size", "nhead_encoder", "nhead",
    "dim_feedforward", "num_encoderlayers", "size_linear_layers",
    "patience", "n_params", "num_channels",
    "cnn_base_channels", "random_seed", "num_training",
    "num_epochs", "segmentation_duration", "length_series",
})
_BOOL_KEYS: frozenset[str] = frozenset({"conv1d_emb", "force_cpu", "verbose"})
_LIST_KEYS: frozenset[str] = frozenset({"cnn_channel_multipliers", "exclude_states"})


def _restore_type(key: str, val: Any) -> Any:
    """Cast a raw pandas / CSV value back to the correct Python type.

    Args:
        key: Hyperparameter name (without any ``hp_`` prefix).
        val: Raw value (may be numpy scalar, string, or float).

    Returns:
        Value cast to ``int``, ``bool``, ``list``, or ``float`` as appropriate.
    """
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass

    if key in _INT_KEYS:
        try:
            return int(val)
        except (ValueError, TypeError):
            return val

    if key in _BOOL_KEYS:
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if isinstance(val, str):
            return val.strip().lower() in ("true", "1", "yes")
        return bool(val)

    if key in _LIST_KEYS:
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            val = val.strip()
            if val.startswith("["):
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    pass
            parts = [p.strip() for p in val.strip("[]").split(",")]
            try:
                return [int(p) for p in parts if p]
            except ValueError:
                return parts
        return val

    return val


# ============================================================================
# Search space sampling
# ============================================================================

def sample_param(spec: dict, rng: random.Random) -> Any:
    """Draw one sample from a search-space specification.

    Args:
        spec: Dict with ``'type'`` and type-specific keys
              (``'values'``, or ``'low'`` / ``'high'``).
        rng:  Seeded ``random.Random`` instance.

    Returns:
        A single sampled value.

    Raises:
        ValueError: If ``spec['type']`` is not recognised.
    """
    t = spec["type"]
    if t in ("choice", "int_choice"):
        return rng.choice(spec["values"])
    if t == "uniform":
        return rng.uniform(spec["low"], spec["high"])
    if t == "log_uniform":
        return math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"])))
    raise ValueError(f"Unknown param type: {t!r}")


def sample_config(search_space: dict, rng: random.Random) -> dict:
    """Sample one full hyperparameter configuration.

    Args:
        search_space: Search-space dict from the config file.
        rng:          Seeded ``random.Random`` instance.

    Returns:
        Dict mapping each parameter name to its sampled value.
    """
    return {name: sample_param(spec, rng) for name, spec in search_space.items()}


def grid_configs(search_space: dict) -> list[dict]:
    """Enumerate every combination in the search space (grid search).

    Args:
        search_space: Search-space dict, all entries must be
                      ``'choice'`` or ``'int_choice'``.

    Returns:
        List of hyperparameter dicts, one per grid point.

    Raises:
        ValueError: If any entry has a continuous type.
    """
    for name, spec in search_space.items():
        if spec["type"] not in ("choice", "int_choice"):
            raise ValueError(
                f"Grid search requires choice/int_choice, "
                f"got '{name}': '{spec['type']}'"
            )
    keys   = list(search_space.keys())
    values = [search_space[k]["values"] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def is_valid(hparams: dict, enforce_head: bool) -> bool:
    """Return True if *hparams* satisfies all hard constraints.

    Constraints checked:
    * ``embed_size`` divisible by ``nhead_encoder`` (when ``enforce_head=True``).
    * ``conv1d_kernel_size`` must be odd when ``conv1d_emb=True``.

    Args:
        hparams:      Candidate hyperparameter dict.
        enforce_head: Whether to apply the divisibility constraint.

    Returns:
        ``True`` if the config is valid.
    """
    if enforce_head:
        embed = hparams.get("embed_size", 1)
        nhead = hparams.get("nhead_encoder", 1)
        if embed % nhead != 0:
            return False
    if hparams.get("conv1d_emb", False):
        if hparams.get("conv1d_kernel_size", 3) % 2 == 0:
            return False
    return True


def build_trial_list(
    search_space: dict,
    strategy: str,
    n_trials: int,
    enforce_head: bool,
    seed: int,
) -> list[dict]:
    """Build the ordered list of hyperparameter dicts to evaluate.

    Args:
        search_space:  Search-space dict.
        strategy:      ``'random'`` or ``'grid'``.
        n_trials:      Number of random trials (ignored for grid).
        enforce_head:  Apply ``embed_size % nhead_encoder == 0`` filter.
        seed:          RNG seed for reproducible random search.

    Returns:
        List of valid hyperparameter dicts.
    """
    rng = random.Random(seed)
    if strategy == "grid":
        all_h = [h for h in grid_configs(search_space) if is_valid(h, enforce_head)]
        print(f"Grid search: {len(all_h)} valid combinations")
    else:
        all_h, attempts = [], 0
        max_attempts = n_trials * 50   # generous budget for constrained spaces
        while len(all_h) < n_trials and attempts < max_attempts:
            h = sample_config(search_space, rng)
            if is_valid(h, enforce_head):
                all_h.append(h)
            attempts += 1
        if len(all_h) < n_trials:
            print(
                f"WARNING: only {len(all_h)}/{n_trials} valid trials found "
                f"after {attempts} attempts, constraints may be too tight."
            )
        else:
            print(f"Random search: {len(all_h)} trials sampled")
    return all_h


# ============================================================================
# Memory helpers
# ============================================================================

def log_mem(label: str) -> None:
    """Log current RSS memory and (optionally) GPU memory.

    Args:
        label: Description printed alongside the memory figures.
    """
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
        print(f"  [MEM] {label}: {rss:.2f} GB RSS")
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024 ** 3
            resv  = torch.cuda.memory_reserved()  / 1024 ** 3
            print(f"  [GPU] allocated={alloc:.2f} GB  reserved={resv:.2f} GB")
    except ImportError:
        pass


def aggressive_cleanup(trial_dir: str | None = None) -> None:
    """Release GPU / CPU memory and delete checkpoint files.

    Args:
        trial_dir: If given, also removes checkpoint files in that directory.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    for ckpt in glob.glob("checkpoint*.pt"):
        try:
            os.remove(ckpt)
        except OSError:
            pass
    if trial_dir is not None:
        for ckpt in glob.glob(os.path.join(trial_dir, "checkpoint*.pt")):
            try:
                os.remove(ckpt)
            except OSError:
                pass
    gc.collect()
    gc.collect()


# ============================================================================
# Data loading (series-level only; segmentation deferred to run_trial)
# ============================================================================

def load_and_preprocess(
    fixed: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list, torch.Tensor]:
    """Load, filter, and pad/truncate at the series level.

    Segmentation is intentionally deferred to ``run_trial`` so each trial
    can use its own ``segmentation_duration`` (whether fixed or tuned).

    Args:
        fixed:  The ``fixed`` block from search_config.json.
        device: Target device (used only for logging; data stays on CPU).

    Returns:
        ``(X_full, y_full, class_labels, class_weights_tensor)`` where
        ``X_full`` has shape ``(N_series, 1, length_series)``.
    """
    filepath = os.path.join(fixed["path_data"], fixed["h5_filename"])
    print(f"Loading data: {filepath}")
    log_mem("before read_data")

    df = read_data_from_h5.read_data(filepath)

    for state in fixed.get("exclude_states", []):
        df = df[df["state"] != state]

    keep_states = fixed.get("keep_states", [])
    if keep_states:
        df = df[df["state"].isin(keep_states)]
        print(f"Kept states: {keep_states}  ->  {len(df)} rows")
        if len(df) == 0:
            raise ValueError(f"No rows remain after keep_states={keep_states}")

    class_labels   = sorted(df["state"].unique())
    state_to_label = {s: i for i, s in enumerate(class_labels)}
    df["label"]    = df["state"].map(state_to_label)
    print(f"Classes: {class_labels}  ->  {state_to_label}")
    log_mem("after read_data")

    out_uni = pad_truncate_tensor.make_tensors(df, seq_len=fixed["length_series"])
    X_full  = out_uni["X"].unsqueeze(1)   # (N_series, 1, length_series)
    y_full  = out_uni["y"]
    del df, out_uni
    gc.collect()
    log_mem("after pad/truncate")

    print(f"X_full: {tuple(X_full.shape)}   y_full: {tuple(y_full.shape)}")

    # Series-level class weights; recomputed more accurately per trial
    ynumpy  = y_full.numpy() if isinstance(y_full, torch.Tensor) else np.array(y_full)
    present = np.unique(ynumpy)
    wp      = class_weight.compute_class_weight("balanced", classes=present, y=ynumpy)
    wf      = np.zeros(len(class_labels), dtype=np.float32)
    wf[present] = wp
    class_weights_tensor = torch.tensor(wf, dtype=torch.float)

    log_mem("data ready")
    return X_full, y_full, class_labels, class_weights_tensor


# ============================================================================
# Single trial
# ============================================================================

def run_trial(
    trial_idx:     int,
    hparams:       dict,
    fixed:         dict,
    X_full:        torch.Tensor,
    y_full:        torch.Tensor,
    class_labels:  list,
    class_weights: torch.Tensor,
    device:        torch.device,
    num_cpus:      int,
    alpha:         float,
    path_save:     str,
) -> dict:
    """Run one hyperparameter trial and return a result dict.

    Segmentation uses ``hparams['segmentation_duration']`` if present,
    otherwise ``fixed['segmentation_duration']``, otherwise 500.

    ``modeltype`` is resolved the same way, so it can be either fixed or tuned.

    ``optimizer`` is resolved from ``hparams`` first, then ``fixed``, defaulting
    to ``'adamW'``. ``muon_lr`` is taken from ``hparams`` when the sampled
    optimizer is ``'muon'``; it is ignored (and not forwarded) otherwise.

    Args:
        trial_idx:     1-based trial number (matches the JSON config filename).
        hparams:       Sampled hyperparameter dict for this trial.
        fixed:         Fixed settings block from search_config.json.
        X_full:        Unsegmented data, shape ``(N_series, 1, length_series)``.
        y_full:        Series-level labels, shape ``(N_series,)``.
        class_labels:  Ordered list of class-name strings.
        class_weights: Series-level class-weight tensor (refined post-segmentation).
        device:        Torch device.
        num_cpus:      DataLoader worker count.
        alpha:         Objective weight for accuracy vs reliability.
        path_save:     Root results directory.

    Returns:
        Result dict (also written to ``<trial_dir>/trial_summary.csv``).
    """
    trial_dir = os.path.join(path_save, f"trial_{trial_idx:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Trial {trial_idx:4d}  |  {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'-'*60}")
    for k, v in hparams.items():
        print(f"  {k:26s}: {v}")
    print(f"{'-'*60}")
    log_mem("trial start")

    # resolve per-trial settings (hparams take priority over fixed)
    seg_dur   = int(hparams.get(
        "segmentation_duration",
        fixed.get("segmentation_duration", 500),
    ))
    modeltype = str(hparams.get("modeltype", fixed.get("modeltype", "trans1")))

    # optimizer is searchable; fall back to fixed then default
    optimizer_name = str(hparams.get("optimizer", fixed.get("optimizer", "adamW")))

    # muon_lr is only meaningful when optimizer == "muon"
    muon_lr = float(hparams.get("muon_lr", fixed.get("muon_lr", 0.02))) \
              if optimizer_name == "muon" else 0.02   # passed but ignored by runner

    print(f"  modeltype             : {modeltype}")
    print(f"  segmentation_duration : {seg_dur}")
    print(f"  optimizer             : {optimizer_name}" +
          (f"  (muon_lr={muon_lr:.6f})" if optimizer_name == "muon" else ""))

    try:
        # per-trial segmentation
        # Kept inside try/except: segmentation or class-weight computation
        # can raise (e.g. a segment_duration/hparams combo that leaves only
        # one class present after segmentation), and if that happens
        # outside the try block, the trial crashes uncaught,
        # trial_summary.csv never gets written for it, and the failure is
        # invisible both to merge_results() (silently missing from
        # all_trials.csv) and to sacct/Slurm (an array wrapper script
        # without `set -e` still reports the job as COMPLETED 0:0).
        seg_out = segmenting_data.segment_data(
            X_full.squeeze(1), y_full, segment_duration=seg_dur,
        )
        X_seg  = seg_out["X"].unsqueeze(1)   # (N_seg, 1, seg_dur)
        y_seg  = seg_out["y"]
        ynumpy = y_seg.numpy() if isinstance(y_seg, torch.Tensor) else np.array(y_seg)
        print(f"  X_seg : {tuple(X_seg.shape)}")

        # Recompute class weights on actual segments
        n_classes   = len(class_labels)
        present_seg = np.unique(ynumpy)
        wp_seg      = class_weight.compute_class_weight(
            "balanced", classes=present_seg, y=ynumpy
        )
        wf_seg              = np.zeros(n_classes, dtype=np.float32)
        wf_seg[present_seg] = wp_seg
        trial_class_weights = torch.tensor(wf_seg, dtype=torch.float)

        computation_time = run_training_transformer_multiset.run_transformer_training(
            data               = X_seg,
            labels             = ynumpy,
            classes            = class_labels,
            device             = device,
            num_channels       = fixed["num_channels"],
            n_classes          = n_classes,
            test_size          = fixed["test_size"],
            val_size           = fixed["val_size"],
            batch_size         = int(hparams["batch_size"]),
            num_cpus           = num_cpus,
            lr                 = float(hparams["lr"]),
            muon_lr            = muon_lr,
            num_epochs         = fixed["num_epochs"],
            patience           = int(hparams["patience"]),
            modeltype          = modeltype,
            max_length_series  = seg_dur,
            embed_size         = int(hparams.get("embed_size",         32)),
            nhead              = int(hparams.get("nhead_encoder",       4)),
            dim_feedforward    = int(hparams.get("dim_feedforward",     64)),
            num_encoderlayers  = int(hparams.get("num_encoderlayers",   1)),
            dropout            = float(hparams.get("dropout",           0.0)),
            conv1d_emb         = bool(hparams.get("conv1d_emb",         True)),
            conv1d_kernel_size = fixed["conv1d_kernel_size"],
            size_linear_layers = int(hparams.get("size_linear_layers",  16)),
            # CNN-specific: from hparams if tuned, otherwise from fixed
            cnn_base_channels  = int(hparams.get(
                "cnn_base_channels",
                fixed.get("cnn_base_channels", 32),
            )),
            cnn_channel_multipliers = _restore_type(
                "cnn_channel_multipliers",
                hparams.get(
                    "cnn_channel_multipliers",
                    fixed.get("cnn_channel_multipliers", [1, 2, 4]),
                ),
            ),
            cnn_pooling        = str(hparams.get(
                "cnn_pooling",
                fixed.get("cnn_pooling", "mean"),
            )),
            opt                = optimizer_name,
            verbose            = fixed["verbose"],
            pathsave           = trial_dir + os.sep,
            weights            = trial_class_weights,
            norm_type          = str(hparams.get("norm_type", fixed.get("norm_type", "per-channel"))),
            num_training       = fixed["num_training"],
        )

        # read back per-run CSV
        csv_path = os.path.join(trial_dir, "results.csv")
        df_trial = pd.read_csv(csv_path)
        # Drop summary row written by run_training
        df_runs  = df_trial[
            pd.to_numeric(df_trial["run_idx"], errors="coerce").notna()
        ].copy()

        score_cols = [
            "accuracy_test", "f1_test", "gmean_test",
            "precision_test", "recall_test",
            "reliability_test", "neutral_pct_test",
            "final_train_loss", "final_val_loss",
            "final_train_acc",  "final_val_acc",
            "final_train_rel",  "final_val_rel",
            "n_epochs_trained",
            "total_params",     # written by run_training via all_trials.csv
        ]
        for col in score_cols:
            if col in df_runs.columns:
                df_runs[col] = pd.to_numeric(df_runs[col], errors="coerce")

        means = {
            col: float(df_runs[col].mean())
            for col in score_cols if col in df_runs.columns
        }

        acc_mean = means.get("accuracy_test",    float("nan"))
        rel_mean = means.get("reliability_test", float("nan"))
        score    = alpha * acc_mean + (1.0 - alpha) * rel_mean

        rel_class_cols  = [c for c in df_runs.columns if c.startswith("rel_class_")]
        rel_class_means = {
            col: round(float(df_runs[col].mean()), 4)
            for col in rel_class_cols
        }
        del df_trial, df_runs

        result: dict = {
            "trial_idx":             trial_idx,
            "score":                 round(score,    4),
            "modeltype":             modeltype,
            "optimizer":             optimizer_name,
            "muon_lr":               muon_lr if optimizer_name == "muon" else None,
            "segmentation_duration": seg_dur,
            "accuracy_test":         round(acc_mean, 4),
            "f1_test":               round(means.get("f1_test",          float("nan")), 4),
            "gmean_test":            round(means.get("gmean_test",        float("nan")), 4),
            "precision_test":        round(means.get("precision_test",    float("nan")), 4),
            "recall_test":           round(means.get("recall_test",       float("nan")), 4),
            "reliability_test":      round(rel_mean,                                     4),
            "neutral_pct_test":      round(means.get("neutral_pct_test",  float("nan")), 4),
            "final_train_loss":      round(means.get("final_train_loss",  float("nan")), 4),
            "final_val_loss":        round(means.get("final_val_loss",    float("nan")), 4),
            "final_val_acc":         round(means.get("final_val_acc",     float("nan")), 4),
            "final_val_rel":         round(means.get("final_val_rel",     float("nan")), 4),
            "n_epochs_trained":      round(means.get("n_epochs_trained",  float("nan")), 1),
            "total_params":          int(means.get("total_params", -1))
                                     if not math.isnan(means.get("total_params", float("nan")))
                                     else -1,
            "computation_time_s":    round(computation_time, 1),
            "status":                "ok",
            **rel_class_means,
            **{f"hp_{k}": v for k, v in hparams.items()},
        }

    except Exception as exc:
        import traceback
        print(f"  Trial {trial_idx} FAILED: {exc}")
        traceback.print_exc()
        result = {
            "trial_idx":             trial_idx,
            "score":                 -1.0,
            "modeltype":             modeltype,
            "optimizer":             optimizer_name,
            "muon_lr":               muon_lr if optimizer_name == "muon" else None,
            "segmentation_duration": seg_dur,
            "accuracy_test":         float("nan"),
            "reliability_test":      float("nan"),
            "computation_time_s":    float("nan"),
            "status":                f"failed: {exc}",
            **{f"hp_{k}": v for k, v in hparams.items()},
        }

    finally:
        aggressive_cleanup(trial_dir)
        log_mem("after cleanup")

    pd.DataFrame([result]).to_csv(
        os.path.join(trial_dir, "trial_summary.csv"), index=False
    )
    return result


# ============================================================================
# Merge results and write best configs (one per modeltype and overall)
# ============================================================================

def _build_best_config(best_row: pd.Series, fixed: dict) -> dict:
    """Merge fixed settings with hyperparameters from the best trial row.

    Args:
        best_row: Best-trial Series (from the merged DataFrame).
        fixed:    Fixed settings block.

    Returns:
        Complete config dict suitable for passing to ``run_transformer_training``.
    """
    cfg = deepcopy(fixed)

    # Overlay hp_* columns (with proper type restoration)
    for col in best_row.index:
        if str(col).startswith("hp_"):
            key = col[3:]
            cfg[key] = _restore_type(key, best_row[col])

    # Bare non-hp_ columns that are still hyperparameter values
    # (some search setups don't add the hp_ prefix)
    bare_keys = {
        "lr", "batch_size", "embed_size", "nhead_encoder", "dim_feedforward",
        "num_encoderlayers", "dropout", "size_linear_layers", "conv1d_emb",
        "norm_type", "patience", "cnn_base_channels", "cnn_channel_multipliers",
        "cnn_pooling", "optimizer", "muon_lr",
    }
    for key in bare_keys:
        if key in best_row.index and key not in cfg:
            cfg[key] = _restore_type(key, best_row[key])

    # Always store the resolved segmentation_duration
    if "segmentation_duration" in best_row.index:
        cfg["segmentation_duration"] = _restore_type(
            "segmentation_duration", best_row["segmentation_duration"]
        )

    # Ensure modeltype is explicit in the output
    if "modeltype" in best_row.index and not pd.isna(best_row["modeltype"]):
        cfg["modeltype"] = str(best_row["modeltype"])

    # Ensure optimizer is explicit; strip muon_lr when not applicable
    if "optimizer" in best_row.index and not pd.isna(best_row["optimizer"]):
        cfg["optimizer"] = str(best_row["optimizer"])
    if cfg.get("optimizer") != "muon":
        cfg.pop("muon_lr", None)

    return cfg


def merge_results(path_save: str, alpha: float, fixed: dict) -> None:
    """Aggregate per-trial CSVs, print leaderboard, write per-modeltype best configs.

    Produces:
    * ``all_trials.csv``               merged, sorted by score
    * ``best_config_<modeltype>.json`` one per model type found in results
    * ``best_config.json``             overall best (backward compatibility)

    Args:
        path_save: Root results directory.
        alpha:     Objective weight for accuracy (displayed only; scores from CSV).
        fixed:     Fixed settings block (used as base for best configs).
    """
    summaries = sorted(glob.glob(
        os.path.join(path_save, "trial_*/trial_summary.csv")
    ))
    if not summaries:
        print("No trial_summary.csv files found.")
        return

    df_all = pd.concat(
        [pd.read_csv(f) for f in summaries], ignore_index=True
    ).sort_values("score", ascending=False)

    summary_csv = os.path.join(path_save, "all_trials.csv")
    df_all.to_csv(summary_csv, index=False)
    print(f"Merged {len(summaries)} trials -> {summary_csv}")

    df_ok = df_all[df_all["status"] == "ok"].copy()
    if df_ok.empty:
        print("All trials failed.")
        for _, row in df_all[["trial_idx", "status"]].iterrows():
            print(f"  Trial {int(row['trial_idx']):4d}: {row['status']}")
        return

    print(f"\nSuccessful trials: {len(df_ok)} / {len(df_all)}")

    written: list[str] = []

    # per-modeltype best configs
    modeltypes_in_results: list[str] = []
    if "modeltype" in df_ok.columns:
        modeltypes_in_results = [
            str(m) for m in df_ok["modeltype"].dropna().unique()
        ]

    for modeltype in modeltypes_in_results:
        subset = df_ok[df_ok["modeltype"] == modeltype]
        if subset.empty:
            continue
        best_row = subset.iloc[0]   # already sorted by score descending

        best_opt    = best_row.get("optimizer", best_row.get("hp_optimizer", "?"))
        best_muonlr = best_row.get("muon_lr",   best_row.get("hp_muon_lr",   None))

        print(f"\n{'='*60}")
        print(
            f"  Best [{modeltype}]  trial={int(best_row['trial_idx'])}  "
            f"score={best_row['score']:.4f}  "
            f"acc={best_row['accuracy_test']:.4f}  "
            f"rel={best_row['reliability_test']:.4f}  "
            f"optimizer={best_opt}" +
            (f"  muon_lr={best_muonlr:.6f}" if best_opt == "muon" and best_muonlr is not None else "")
        )
        print(f"{'-'*60}")
        for k, v in best_row.items():
            if str(k).startswith("hp_"):
                print(f"    {k[3:]:26s}: {v}")
        print(f"{'='*60}")

        fixed_for_model = deepcopy(fixed)
        fixed_for_model["modeltype"] = modeltype

        cfg = _build_best_config(best_row, fixed_for_model)
        out_path = os.path.join(path_save, f"best_config_{modeltype}.json")
        _atomic_json_dump(cfg, out_path)
        print(f"Best config [{modeltype}] -> {out_path}")
        written.append(out_path)

    # overall best config (backward compatibility)
    overall_best = df_ok.iloc[0]
    cfg_overall  = _build_best_config(overall_best, fixed)
    out_overall  = os.path.join(path_save, "best_config.json")
    _atomic_json_dump(cfg_overall, out_overall)
    print(f"Overall best config -> {out_overall}")
    written.append(out_overall)

    print(f"\n{'-'*60}")
    print(f"  Written {len(written)} config file(s):")
    for p in written:
        print(f"    {p}")
    print(f"{'-'*60}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    with open(args.config) as f:
        cfg = json.load(f)

    fixed        = cfg["fixed"]
    search_space = cfg["search_space"]
    strategy     = cfg["strategy"]
    n_trials     = cfg["n_trials"]
    alpha        = cfg["objective_alpha"]
    enforce_head = cfg.get("enforce_head_divisibility", True)
    seed         = fixed["random_seed"]
    path_save    = fixed["path_save"]

    os.makedirs(path_save, exist_ok=True)

    # Seed all RNGs for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # merge-only mode
    if args.merge_results:
        merge_results(path_save, alpha, fixed)
        return

    # build trial list and write JSON configs
    all_hparams = build_trial_list(search_space, strategy, n_trials, enforce_head, seed)

    configs_dir = os.path.join(path_save, "trial_configs")
    os.makedirs(configs_dir, exist_ok=True)
    for i, h in enumerate(all_hparams, start=1):
        with open(os.path.join(configs_dir, f"trial_{i:04d}.json"), "w") as f:
            json.dump(_sanitize_for_json(h), f, indent=2)
    print(f"Trial configs -> {configs_dir}  ({len(all_hparams)} files)")

    if args.generate_configs:
        print("Config generation complete, exiting.")
        return

    # device and worker count
    if fixed.get("force_cpu", False):
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Respect the config's num_cpus if set; fall back to 1 (safe default)
    num_cpus = fixed.get("num_cpus", 1)
    print(f"Device: {device}  |  CPUs: {num_cpus}")

    # single-trial mode (SLURM array)
    if args.trial_idx is not None:
        cfg_path = os.path.join(configs_dir, f"trial_{args.trial_idx:04d}.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"Trial config not found: {cfg_path}\n"
                "Run  --generate-configs  first."
            )
        with open(cfg_path) as f:
            hparams = json.load(f)

        X_full, y_full, class_labels, cw = load_and_preprocess(fixed, device)
        run_trial(
            args.trial_idx, hparams, fixed,
            X_full, y_full, class_labels, cw,
            device, num_cpus, alpha, path_save,
        )
        return

    # sequential mode
    X_full, y_full, class_labels, cw = load_and_preprocess(fixed, device)
    log_mem("data ready, starting trials")

    all_results: list[dict] = []
    summary_csv = os.path.join(path_save, "all_trials.csv")

    for trial_idx, hparams in enumerate(all_hparams, start=1):
        result = run_trial(
            trial_idx, hparams, fixed,
            X_full, y_full, class_labels, cw,
            device, num_cpus, alpha, path_save,
        )
        all_results.append(result)

        # Incremental save
        df_r = pd.DataFrame(all_results).sort_values("score", ascending=False)
        df_r.to_csv(summary_csv, index=False)
        del df_r

        # Rolling leaderboard (top 5 successful trials so far)
        df_top = (
            pd.DataFrame(all_results)
            .pipe(lambda d: d[d["status"] == "ok"])
            .sort_values("score", ascending=False)
            .head(5)
        )
        print(f"\n-- Leaderboard after trial {trial_idx} --")
        if not df_top.empty:
            show_cols = [
                c for c in [
                    "trial_idx", "score", "modeltype", "optimizer",
                    "accuracy_test", "reliability_test", "segmentation_duration",
                ]
                if c in df_top.columns
            ]
            print(df_top[show_cols].to_string(index=False))
        del df_top

    merge_results(path_save, alpha, fixed)


if __name__ == "__main__":
    main()

