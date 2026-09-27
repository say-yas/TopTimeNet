#!/usr/bin/env python3
"""
rebuild_best_config.py
Rebuilds best_config.json (and optionally best_config_small.json) from
all_trials.csv produced by hparam_search_tda.py.

Usage
-----
    python rebuild_best_config.py --search-dir /path/to/hparam_search_results/
    python rebuild_best_config.py --search-dir ./results/ --alpha 0.6
    python rebuild_best_config.py --search-dir ./results/ --metric f1_test
    python rebuild_best_config.py --search-dir ./results/ --small
    python rebuild_best_config.py --search-dir ./results/ --small --tol 0.01

Notes
-----
- IQR clip bounds (clip_lo_pct, clip_hi_pct) are pipeline constants
  injected from the "fixed" section, not hyperparameters, so they always
  come from search_config["fixed"]. They are listed in _DEFAULTS so
  rebuild still produces a complete config when running against an older
  search_config.json that lacks them.
- noise_aug_sigma is disabled (0.0) during search; the written
  best_config.json carries 0.0 from the fixed section, and
  submit_best_training.sh patches it to the real value (e.g. 0.05) for
  best-model training.
- use_temperature_scaling is disabled during search and enabled in
  best-model training.
- noise_levels and noise_batch_size come from the fixed section during
  search (null / 32 respectively).
"""

import argparse
import glob
import json
import math
import os
import tempfile

import numpy as np
import pandas as pd


# helpers

def sanitize(obj):
    """Recursively convert obj to a JSON-serialisable Python type."""
    if isinstance(obj, np.bool_):   return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray): return [sanitize(x) for x in obj.tolist()]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):  return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [sanitize(v) for v in obj]
    return obj


def atomic_write(data: dict, path: str) -> None:
    """Write JSON atomically via a temp file and os.replace."""
    clean = sanitize(data)
    dir_  = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(clean, f, indent=4)
        os.replace(tmp, path)
        print(f"Written -> {path}")
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise


def _restore_type(key: str, val) -> object:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None

    int_keys = {
        "takens_dim", "takens_delay", "n_hom_dims",
        "n_betti_bins", "n_pi_bins", "ph_workers",
        "precompute_batch_size",
        "embed_dim", "rank", "n_attn_layers", "n_heads", "ffn_dim",
        "batch_size", "patience", "num_epochs",
        "num_training", "n_classes",
        "n_params", "n_epochs_trained", "trial_idx",
        "noise_batch_size",
    }
    bool_keys = {
        "force_cpu", "use_cache", "force_recompute", "verbose",
        "use_temperature_scaling",
    }
    list_keys = {
        "head_hidden",
        "keep_states",
        "exclude_states",
        # noise_levels is a list of floats, e.g. [0.0, 0.05, 0.1, ...]
        "noise_levels",
    }
    float_keys = {
        "lr", "dropout", "pi_sigma",
        "test_size", "val_size",
        "label_smoothing", "grad_clip_norm",
        "reliability_threshold",
        "clip_lo_pct", "clip_hi_pct",
        "noise_aug_sigma",
        "muon_lr",
    }
    str_keys = {
        "fusion", "activation", "optimizer", "norm_type",
    }

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
        # handle None / "null" as None (noise_levels disabled)
        if val is None: return None
        s = str(val).strip()
        if s.lower() in ("none", "null", ""): return None
        try:
            parsed = json.loads(s.replace("'", '"'))
            return parsed if isinstance(parsed, list) else val
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            parts = s.strip("[]").replace(",", " ").split()
            # try float first (noise_levels), fall back to int (head_hidden)
            try:
                return [float(p) for p in parts] if parts else []
            except ValueError:
                return [int(p) for p in parts] if parts else []
        except ValueError:
            return val

    if key in float_keys:
        try: return float(val)
        except (ValueError, TypeError): return val

    if key in str_keys:
        return str(val)

    return val


_DEFAULTS = {
    # architecture
    "embed_dim"      : 32,
    "fusion"         : "low_rank",
    "rank"           : 8,
    "n_attn_layers"  : 1,
    "n_heads"        : 4,
    "ffn_dim"        : 0,
    "head_hidden"    : [64, 32],
    "activation"     : "gelu",
    "dropout"        : 0.1,
    # regularisation
    "label_smoothing": 0.1,
    "grad_clip_norm" : 1.0,
    "norm_type"      : "none",
    # IQR clipping: pipeline constants, not hyperparameters
    "clip_lo_pct"    : 1.0,
    "clip_hi_pct"    : 99.0,
    # noise augmentation: disabled in search, enabled in best-model training
    "noise_aug_sigma": 0.0,
    # temperature scaling: disabled in search, enabled in best-model training
    "use_temperature_scaling": False,
    # robustness sweep: disabled in search, enabled via SLURM script
    "noise_levels"    : None,
    "noise_batch_size": 32,
    "muon_lr"         : 0.02,
}


# Pareto-efficiency helper

def _find_pareto_small(df_ok: pd.DataFrame, tol: float = 0.0) -> pd.Series:
    """
    Return the best Pareto-efficient trial on the (score up, n_params
    down) front.

    A trial i is dominated if there exists j with:
        score[j]  >= score[i]  - tol    (j is at least as good in score)
        params[j] <= params[i]           (j is at least as small)
        at least one strict inequality

    Among non-dominated trials we pick: highest score, ties broken by
    fewest params. Falls back to the best-score row if n_params is
    missing for all trials.
    """
    df = df_ok.copy()
    df["_score"]   = pd.to_numeric(df["score"],    errors="coerce")
    df["_nparams"] = pd.to_numeric(df["n_params"], errors="coerce")

    valid = df.dropna(subset=["_score", "_nparams"]).reset_index(drop=True)

    if valid.empty:
        print("  WARNING: n_params missing in all trials, "
              "cannot compute Pareto front; falling back to best score.")
        return df_ok.sort_values("score", ascending=False).iloc[0]

    scores  = valid["_score"].values
    nparams = valid["_nparams"].values
    n       = len(valid)

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
    df_pareto  = valid.iloc[pareto_idx].sort_values(
        ["_score", "_nparams"], ascending=[False, True]
    ).reset_index(drop=True)

    chosen = df_pareto.iloc[0]

    print(f"\n  Pareto front ({len(pareto_idx)} trials on score up / n_params down):")
    for _, row in df_pareto.iterrows():
        marker = " <- CHOSEN" if row["trial_idx"] == chosen["trial_idx"] else ""
        print(f"    trial={int(row['trial_idx']):4d}  "
              f"score={row['_score']:.4f}  "
              f"n_params={int(row['_nparams']):>10,}{marker}")

    orig = df_ok.index[df_ok["trial_idx"] == chosen["trial_idx"]].tolist()
    return df_ok.loc[orig[0]] if orig else df_ok.sort_values(
        "score", ascending=False
    ).iloc[0]


# ============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Rebuild best_config.json from hparam_search_tda.py results"
    )
    p.add_argument("--search-dir", required=True,
                   help="Path to hparam_search_results directory")
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Objective weight: alpha*f1 + (1-alpha)*gmean "
                        "(ignored when --metric is set)")
    p.add_argument("--metric", default=None,
                   help="Rank directly by this column instead of composite score "
                        "(e.g. f1_test, gmean_test, accuracy_test)")
    p.add_argument("--out", default=None,
                   help="Output path for best_config.json "
                        "(default: <search-dir>/best_config.json)")
    p.add_argument("--no-inject-defaults", action="store_true",
                   help="Do not inject defaults for missing keys")
    p.add_argument("--small", action="store_true",
                   help="Also write best_config_small.json: the Pareto-optimal "
                        "model on the (score up, n_params down) front.")
    p.add_argument("--tol", type=float, default=0.0,
                   help="Score tolerance for Pareto dominance (default 0.0). "
                        "E.g. --tol 0.01 accepts a 1% score drop for fewer params.")
    args = p.parse_args()

    search_dir = args.search_dir.rstrip("/")

    # Step 1: load or rebuild all_trials.csv
    summary_csv = os.path.join(search_dir, "all_trials.csv")

    if not os.path.exists(summary_csv):
        print("all_trials.csv not found, rebuilding from trial_summary.csv ...")
        summaries = sorted(glob.glob(
            os.path.join(search_dir, "trial_*/trial_summary.csv")
        ))
        if not summaries:
            print(f"ERROR: no trial_summary.csv files found in {search_dir}")
            return
        df = pd.concat([pd.read_csv(f) for f in summaries], ignore_index=True)
        df.to_csv(summary_csv, index=False)
        print(f"Rebuilt all_trials.csv from {len(summaries)} trial summaries")
    else:
        df = pd.read_csv(summary_csv)
        print(f"Loaded all_trials.csv  ({len(df)} rows, {len(df.columns)} cols)")

    # Step 2: compute ranking score
    df_ok = df[df["status"] == "ok"].copy()

    if df_ok.empty:
        print("\nERROR: no successful trials found.")
        print("Status counts:")
        print(df["status"].value_counts().to_string())
        return

    if args.metric:
        rank_col = args.metric
        if rank_col not in df_ok.columns:
            print(f"ERROR: --metric '{rank_col}' not found in CSV.")
            print(f"  Available columns: {list(df_ok.columns)}")
            return
        df_ok["_rank_score"] = pd.to_numeric(df_ok[rank_col], errors="coerce")
        score_label = rank_col
    else:
        alpha  = args.alpha
        f1_col = next((c for c in ["f1_test", "final_val_f1"]
                       if c in df_ok.columns), None)
        gm_col = next((c for c in ["gmean_test", "final_val_gmean"]
                       if c in df_ok.columns), None)

        if f1_col and gm_col:
            df_ok["_rank_score"] = (
                alpha       * pd.to_numeric(df_ok[f1_col], errors="coerce") +
                (1 - alpha) * pd.to_numeric(df_ok[gm_col], errors="coerce")
            )
            score_label = f"{alpha:.1f}x{f1_col} + {1-alpha:.1f}x{gm_col}"
        elif "score" in df_ok.columns:
            df_ok["_rank_score"] = pd.to_numeric(df_ok["score"], errors="coerce")
            score_label = "score (pre-computed)"
        else:
            print("ERROR: could not find f1_test / gmean_test / score columns.")
            print(f"  Available: {list(df_ok.columns)}")
            return

    df_ok = df_ok.sort_values("_rank_score", ascending=False).reset_index(drop=True)
    best  = df_ok.iloc[0]

    # Step 3: print summary
    print(f"\n{'='*60}")
    print(f"  Best trial index : {int(best.get('trial_idx', -1))}")
    print(f"  Ranking by       : {score_label}")
    print(f"  Score            : {best['_rank_score']:.4f}")

    n_params_best = best.get("n_params")
    if n_params_best is not None and not (
        isinstance(n_params_best, float) and math.isnan(n_params_best)
    ):
        print(f"  n_params         : {int(float(n_params_best)):,}")

    print(f"{'-'*60}")

    metric_cols = [c for c in best.index
                   if any(c.startswith(pfx) for pfx in
                          ["accuracy", "f1", "gmean", "precision",
                           "recall", "reliability", "neutral"])]
    if metric_cols:
        print("  Test metrics:")
        for c in metric_cols:
            v = best.get(c)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                print(f"    {c:32s}: {float(v):.4f}")

    rel_cols = [c for c in best.index if c.startswith("rel_class_")]
    if rel_cols:
        print("  Per-class reliability:")
        for c in rel_cols:
            v = best.get(c)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                print(f"    {c:32s}: {float(v):.4f}")

    print(f"{'-'*60}")

    hp_cols = [c for c in best.index if str(c).startswith("hp_")]

    arch_keys  = {"embed_dim", "fusion", "rank", "n_attn_layers",
                  "n_heads", "ffn_dim", "head_hidden", "activation", "dropout"}
    reg_keys   = {"label_smoothing", "grad_clip_norm"}
    train_keys = {"lr", "batch_size", "patience", "optimizer", "norm_type",
                  "segmentation_duration", "takens_dim", "takens_delay",
                  "n_betti_bins", "n_pi_bins", "pi_sigma"}

    hp_arch  = [c for c in hp_cols if c[3:] in arch_keys]
    hp_reg   = [c for c in hp_cols if c[3:] in reg_keys]
    hp_train = [c for c in hp_cols if c[3:] in train_keys]
    hp_other = [c for c in hp_cols
                if c[3:] not in arch_keys | reg_keys | train_keys]

    def _print_hp(cols, label):
        if not cols:
            return
        print(f"  {label}:")
        for col in cols:
            key = col[3:]
            val = _restore_type(key, best[col])
            print(f"    {key:30s}: {val}")

    _print_hp(hp_arch,  "Architecture")
    _print_hp(hp_reg,   "Regularisation")
    _print_hp(hp_train, "Training")
    _print_hp(hp_other, "Other")
    print(f"{'='*60}\n")

    # Step 4: load fixed settings from search_config.json
    config_candidates = [
        os.path.join(search_dir, "search_config.json"),
        os.path.join(search_dir, "search_config_tda.json"),
        os.path.join(search_dir, "search_config_dual_branch.json"),
    ]
    fixed = {}
    for c in config_candidates:
        if os.path.exists(c):
            with open(c) as f:
                raw_cfg = json.load(f)
            fixed = raw_cfg.get("fixed", {})
            print(f"Fixed settings loaded from: {c}")
            break

    if not fixed:
        print("WARNING: no search_config.json found, best_config will contain "
              "only hyperparameters, no fixed pipeline settings.")

    # Step 5: assemble best_config
    best_config = dict(fixed)

    for col in hp_cols:
        key = col[3:]
        best_config[key] = _restore_type(key, best[col])

    if "n_classes" not in best_config and "n_classes" in best.index:
        try:
            best_config["n_classes"] = int(float(best["n_classes"]))
        except (ValueError, TypeError):
            pass

    if not args.no_inject_defaults:
        injected = []
        for key, default_val in _DEFAULTS.items():
            if key not in best_config:
                best_config[key] = default_val
                injected.append(f"{key}={default_val!r}")
        if injected:
            print(f"Injected defaults for missing keys:")
            for item in injected:
                print(f"  {item}")
            print("  Use --no-inject-defaults to suppress.")

    # Step 6: write best_config atomically
    out_path = args.out or os.path.join(search_dir, "best_config.json")
    atomic_write(best_config, out_path)

    with open(out_path) as f:
        check = json.load(f)
    print(f"Validation: re-parsed OK  ({len(check)} keys)")

    missing = [k for k in _DEFAULTS if k not in check]
    if missing:
        print(f"WARNING: keys still missing after injection: {missing}")
    else:
        present = {k: check[k] for k in _DEFAULTS}
        print("Completeness check OK:")
        for k, v in present.items():
            print(f"  {k:30s}: {v}")

    # Step 7: optionally write best_config_small.json
    if args.small:
        print(f"\n{'-'*60}")
        print(f"  Computing Pareto-small model  (tol={args.tol})")

        best_small = _find_pareto_small(df_ok, tol=args.tol)
        small_trial = int(best_small["trial_idx"])
        best_trial  = int(best.get("trial_idx", -1))

        if small_trial == best_trial:
            print(f"\n  BEST SMALL = BEST (same trial {small_trial})")
            print("  best_config_small.json not written "
                  "(would be identical to best_config.json).")
        else:
            n_params_small = best_small.get("n_params")
            np_str = (
                f"{int(float(n_params_small)):,}"
                if n_params_small is not None
                   and not (isinstance(n_params_small, float)
                            and math.isnan(n_params_small))
                else "n/a"
            )
            print(f"\n  BEST SMALL  trial={small_trial}  "
                  f"score={float(best_small['score']):.4f}  "
                  f"n_params={np_str}")

            small_config = dict(fixed)
            hp_cols_small = [c for c in best_small.index if str(c).startswith("hp_")]
            for col in hp_cols_small:
                key = col[3:]
                small_config[key] = _restore_type(key, best_small[col])

            if "n_classes" not in small_config and "n_classes" in best_small.index:
                try:
                    small_config["n_classes"] = int(float(best_small["n_classes"]))
                except (ValueError, TypeError):
                    pass

            if not args.no_inject_defaults:
                for key, default_val in _DEFAULTS.items():
                    if key not in small_config:
                        small_config[key] = default_val

            small_out = os.path.join(
                os.path.dirname(out_path), "best_config_small.json"
            )
            atomic_write(small_config, small_out)

            with open(small_out) as f:
                check_small = json.load(f)
            print(f"Validation (small): re-parsed OK  ({len(check_small)} keys)")

    print(f"\nTo train with best config:")
    print(f"  python main_train_time_series_tda_stat_summary.py "
          f"--config {out_path}")
    if args.small and "small_out" in dir():
        print(f"\nTo train with best small config:")
        print(f"  python main_train_time_series_tda_stat_summary.py "
              f"--config {small_out}")


if __name__ == "__main__":
    main()

