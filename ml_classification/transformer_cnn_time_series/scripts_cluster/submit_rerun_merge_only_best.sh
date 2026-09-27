#!/bin/bash
# Standalone re-merge script: use this if training already finished and you
# just need to regenerate all_trials_merged.csv with the run_idx fix applied.
#
# Usage: bash rerun_merge_only.sh /path/to/hparam_search_results

set -euo pipefail
ROOT="${1:?Usage: bash rerun_merge_only.sh /path/to/hparam_search_results}"

python3 - "$ROOT" << 'PYEOF'
import os, sys, pandas as pd

root = sys.argv[1]

config_dirs = []
for dirpath, dirnames, filenames in os.walk(root):
    if any(f.startswith("best_config_") and f.endswith(".json") for f in filenames):
        config_dirs.append(dirpath)

config_dirs.sort()
print(f"Found {len(config_dirs)} config directorie(s):")
for d in config_dirs:
    print(f"  {d}")

dfs = []
for cdir in config_dirs:
    modeltype = "unknown"
    for f in os.listdir(cdir):
        if f.startswith("best_config_") and f.endswith(".json"):
            modeltype = f[len("best_config_"):-len(".json")]
            break

    rel = os.path.relpath(cdir, root)

    for fname in ("all_trials.csv", "results.csv"):
        csv = os.path.join(cdir, fname)
        if os.path.exists(csv):
            try:
                df = pd.read_csv(csv)
                if "run_idx" in df.columns:
                    mask = pd.to_numeric(df["run_idx"], errors="coerce").notna()
                    df = df[mask]
                df["modeltype"] = modeltype
                df["config_dir"] = rel
                df["source_file"] = csv
                dfs.append(df)
                print(f"  {rel:30s}: {len(df):4d} rows  <-  {fname}")
            except Exception as e:
                print(f"  WARNING: could not read {csv}: {e}")
            break

if not dfs:
    print("No result files found - nothing to merge.")
    raise SystemExit(0)

merged = pd.concat(dfs, ignore_index=True)
out = os.path.join(root, "all_trials_merged.csv")
merged.to_csv(out, index=False)
print(f"\nMerged {len(merged)} rows -> {out}")

metric_cols = [c for c in (
    "accuracy_test", "f1_test", "gmean_test",
    "precision_test", "recall_test", "reliability_test",
    "total_wall_time_s", "total_params",
) if c in merged.columns]

if metric_cols:
    for col in metric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    print("\nSummary by modeltype and config_dir:")
    print(merged.groupby(["modeltype", "config_dir"])[metric_cols]
          .agg(["mean", "std"])
          .round(4)
          .to_string())
PYEOF
