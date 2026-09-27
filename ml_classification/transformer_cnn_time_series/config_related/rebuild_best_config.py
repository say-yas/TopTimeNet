#!/usr/bin/env python3
"""
rebuild_best_config.py
Rebuild best_config.json from all_trials.csv.
Produces one best-config file per model type found in the results.

Output files (written to --search-dir):
    best_config_trans1.json   best TransformerI trial
    best_config_cnn1.json     best CNNI trial
    best_config.json          overall best (any modeltype), for backward compatibility

Usage:
    python rebuild_best_config.py --search-dir /path/to/hparam_search_results
    python rebuild_best_config.py --search-dir ./results --alpha 0.8
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd


# JSON helpers

def sanitize(obj: Any) -> Any:
    """Recursively convert numpy types and non-finite floats to JSON-safe values.

    Args:
        obj: Any Python / numpy object.

    Returns:
        A JSON-serialisable equivalent (``None`` replaces NaN / Inf).
    """
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if (math.isnan(float(obj)) or math.isinf(float(obj))) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [sanitize(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def atomic_write(data: Any, path: str) -> None:
    """Write *data* as JSON to *path* atomically via a temp file.

    Uses ``os.replace`` so the destination is never left half-written if the
    process is interrupted.

    Args:
        data: JSON-serialisable object (``sanitize`` is applied automatically).
        path: Destination file path.
    """
    clean = sanitize(data)
    dir_  = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(clean, f, indent=4)
        os.replace(tmp, path)
        print(f"  Written -> {path}")
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# type restoration

# Keys that pandas may widen to float but should be int in the config.
_INT_KEYS: frozenset[str] = frozenset({
    "batch_size", "embed_size", "nhead_encoder", "nhead",
    "dim_feedforward", "num_encoderlayers", "size_linear_layers",
    "patience", "n_params", "num_channels", "num_encoderlayers",
    "cnn_base_channels", "random_seed", "num_training", "num_epochs",
    "segmentation_duration", "length_series",
})

# Keys that should be bool.
_BOOL_KEYS: frozenset[str] = frozenset({"conv1d_emb", "force_cpu", "verbose"})

# Keys that should be parsed back as a Python list (stored as string in CSV).
_LIST_KEYS: frozenset[str] = frozenset({"cnn_channel_multipliers", "exclude_states"})


def _restore_type(key: str, val: Any) -> Any:
    """Restore the correct Python type for a hyperparameter value.

    Pandas reads every CSV column as either float64 or object, so int, bool,
    and list values need to be converted back.

    Args:
        key: Hyperparameter name (without the ``hp_`` prefix).
        val: Raw value from the DataFrame cell.

    Returns:
        Value cast to the appropriate type.
    """
    if pd.isna(val) if not isinstance(val, (list, dict)) else False:
        return None

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
            # Stored as "[1, 2, 4]" or "1,2,4"
            val = val.strip()
            if val.startswith("["):
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    pass
            # Fallback: split on comma
            parts = [p.strip() for p in val.strip("[]").split(",")]
            try:
                return [int(p) for p in parts if p]
            except ValueError:
                return parts
        return val

    return val


# core logic

def _load_trials(search_dir: str) -> pd.DataFrame:
    """Load ``all_trials.csv``, rebuilding it from per-trial files if absent.

    Args:
        search_dir: Root directory of the hyperparameter search results.

    Returns:
        DataFrame with one row per trial.

    Raises:
        SystemExit: If no trial data can be found at all.
    """
    summary_csv = os.path.join(search_dir, "all_trials.csv")

    if not os.path.exists(summary_csv):
        print("all_trials.csv not found, rebuilding from trial_summary.csv files ...")
        summaries = sorted(glob.glob(
            os.path.join(search_dir, "trial_*/trial_summary.csv")
        ))
        if not summaries:
            print(f"[ERROR] No trial_summary.csv files found in: {search_dir}")
            raise SystemExit(1)
        df = pd.concat([pd.read_csv(f) for f in summaries], ignore_index=True)
        df.to_csv(summary_csv, index=False)
        print(f"Rebuilt all_trials.csv from {len(summaries)} trial summaries")
    else:
        df = pd.read_csv(summary_csv)
        print(f"Loaded all_trials.csv  ({len(df)} rows)")

    return df


def _load_fixed_settings(search_dir: str) -> dict:
    """Load the ``fixed`` block from the search config, if available.

    Tries several candidate filenames so the script works even when the config
    was named non-standardly.

    Args:
        search_dir: Directory to search for the config file.

    Returns:
        The ``fixed`` sub-dict, or an empty dict if no config is found.
    """
    candidates = [
        "search_config.json",
        "search_config_dual_branch.json",
    ]
    for name in candidates:
        path = os.path.join(search_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            fixed = raw.get("fixed", {})
            print(f"Fixed settings loaded from: {path}")
            return fixed

    print("WARNING: no search_config.json found, best_config will contain "
          "hyperparameters only (no fixed settings).")
    return {}


def _best_trial_for(
    df_ok: pd.DataFrame,
    modeltype: str | None,
) -> pd.Series | None:
    """Return the row with the highest score for a given modeltype.

    Args:
        df_ok:     DataFrame of successful trials (status == 'ok').
        modeltype: Filter value (e.g. ``'trans1'``), or ``None`` for all rows.

    Returns:
        The best row as a Series, or ``None`` if no matching rows exist.
    """
    if modeltype is not None:
        # hp_modeltype column if tuned; fixed.modeltype column otherwise
        for col in ("hp_modeltype", "modeltype"):
            if col in df_ok.columns:
                subset = df_ok[df_ok[col] == modeltype]
                if not subset.empty:
                    return subset.sort_values("score", ascending=False).iloc[0]
        return None

    # No filter, overall best
    return df_ok.sort_values("score", ascending=False).iloc[0]


def _build_config(
    best_row: pd.Series,
    fixed: dict,
    modeltype_override: str | None = None,
) -> dict:
    """Merge fixed settings with the hyperparameters from a best-trial row.

    Args:
        best_row:          Row from the trials DataFrame.
        fixed:             Fixed settings dict from search_config.json.
        modeltype_override: If provided, force ``modeltype`` in the output
                            (useful when the model type comes from ``fixed``
                            rather than a tuned column).

    Returns:
        Complete config dict ready to be passed to ``run_transformer_training``.
    """
    cfg = dict(fixed)   # start with fixed settings

    # Overlay hyperparameters (columns prefixed with "hp_")
    for col in best_row.index:
        if str(col).startswith("hp_"):
            key = col[3:]
            cfg[key] = _restore_type(key, best_row[col])

    # Also promote any bare columns that look like config keys
    # (some search scripts write them without the hp_ prefix)
    known_bare = {
        "lr", "batch_size", "embed_size", "nhead_encoder", "dim_feedforward",
        "num_encoderlayers", "dropout", "size_linear_layers", "conv1d_emb",
        "norm_type", "patience", "cnn_base_channels", "cnn_channel_multipliers",
        "cnn_pooling",
    }
    for key in known_bare:
        if key in best_row.index and key not in cfg:
            cfg[key] = _restore_type(key, best_row[key])

    if modeltype_override is not None:
        cfg["modeltype"] = modeltype_override

    return cfg


def _print_summary(best_row: pd.Series, label: str) -> None:
    """Print a formatted summary of the best trial to stdout.

    Args:
        best_row: Best-trial Series.
        label:    Short description (e.g. ``'trans1'`` or ``'overall'``).
    """
    idx = int(best_row["trial_idx"]) if "trial_idx" in best_row.index else "?"
    print(f"\n{'='*55}")
    print(f"  Best trial [{label}]: #{idx}")
    print(f"  Score       : {best_row['score']:.4f}")
    for col in ("accuracy_test", "reliability_test", "f1_test",
                "gmean_test", "total_wall_time_s", "total_params"):
        if col in best_row.index and not pd.isna(best_row[col]):
            print(f"  {col:<20}: {best_row[col]}")
    print(f"{'-'*55}")
    print("  Hyperparameters:")
    for col in best_row.index:
        if str(col).startswith("hp_"):
            print(f"    {col[3:]:26s}: {best_row[col]}")
    print(f"{'='*55}")


# main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebuild per-modeltype best configs from all_trials.csv."
    )
    p.add_argument(
        "--search-dir", required=True,
        help="Path to the hparam_search_results directory.",
    )
    p.add_argument(
        "--alpha", type=float, default=0.7,
        help="Objective weight: alpha*accuracy + (1-alpha)*reliability "
             "(default: 0.7). Used only for display, scores are read from CSV.",
    )
    p.add_argument(
        "--modeltypes", nargs="+", default=["trans1", "cnn1"],
        help="Model types to produce separate best-config files for "
             "(default: trans1 cnn1).",
    )
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    search_dir = args.search_dir.rstrip("/\\")

    # load trials
    df = _load_trials(search_dir)

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df_ok = df[df["status"] == "ok"].copy()

    if df_ok.empty:
        print("\n[ERROR] No successful trials (status == 'ok') found.")
        print("Status counts:")
        print(df["status"].value_counts().to_string())
        raise SystemExit(1)

    print(f"\nSuccessful trials: {len(df_ok)} / {len(df)}")

    # load fixed settings
    fixed = _load_fixed_settings(search_dir)

    # per-modeltype best configs
    written: list[str] = []

    for modeltype in args.modeltypes:
        best_row = _best_trial_for(df_ok, modeltype)
        if best_row is None:
            print(f"\n[SKIP] No successful trials found for modeltype='{modeltype}'")
            continue

        _print_summary(best_row, label=modeltype)

        # Build fixed settings for this modeltype: if fixed came from a
        # shared search config, override modeltype so the per-model file
        # is self-contained.
        fixed_for_model = dict(fixed)
        fixed_for_model["modeltype"] = modeltype

        cfg = _build_config(best_row, fixed_for_model, modeltype_override=modeltype)

        out_path = os.path.join(search_dir, f"best_config_{modeltype}.json")
        print(f"\nWriting best config for '{modeltype}':")
        atomic_write(cfg, out_path)
        written.append(out_path)

        # Quick re-parse validation
        with open(out_path) as f:
            check = json.load(f)
        print(f"  Validation: re-parsed OK  ({len(check)} keys)")

    # overall best config (backward compatibility)
    overall_best = _best_trial_for(df_ok, modeltype=None)
    if overall_best is not None:
        overall_modeltype = None
        # try to determine model type from columns
        for col in ("hp_modeltype", "modeltype"):
            if col in overall_best.index and not pd.isna(overall_best[col]):
                overall_modeltype = str(overall_best[col])
                break
        if overall_modeltype is None and "modeltype" in fixed:
            overall_modeltype = fixed["modeltype"]

        _print_summary(overall_best, label="overall")
        cfg_overall = _build_config(
            overall_best, fixed, modeltype_override=overall_modeltype
        )
        out_overall = os.path.join(search_dir, "best_config.json")
        print("\nWriting overall best config:")
        atomic_write(cfg_overall, out_overall)
        written.append(out_overall)

        with open(out_overall) as f:
            check = json.load(f)
        print(f"  Validation: re-parsed OK  ({len(check)} keys)")

    # summary
    print(f"\n{'-'*55}")
    print(f"  Wrote {len(written)} config file(s):")
    for p in written:
        print(f"    {p}")
    print(f"{'-'*55}")


if __name__ == "__main__":
    main()
