"""
merge_robustness_sweeps.py
Combines per-run robustness_sweep_run{N}.csv files, each written with a
globally unique N per run/task, into a single robustness_all_runs.csv.

This replaces an earlier in-process "combined CSV" write, which silently
overwrote itself whenever multiple SLURM array tasks shared the same
pathsave (only the last task's single run ever survived). Run this once,
after every task writing to a given pathsave has finished.

Usage
-----
    python merge_robustness_sweeps.py /path/to/pathsave

    # or merge multiple pathsave directories into their own combined files:
    python merge_robustness_sweeps.py /path/a /path/b /path/c
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def merge_one(pathsave: str) -> None:
    pattern = os.path.join(pathsave, "robustness_sweep_run*.csv")
    files = sorted(glob.glob(pattern))

    print(f"\n{'='*60}\n  {pathsave}\n{'='*60}")
    print(f"Found {len(files)} file(s) matching robustness_sweep_run*.csv")

    if not files:
        print("Nothing to merge.")
        return

    dfs = []
    seen_run_ids = set()
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            print(f"  WARNING: could not read {f}: {exc}")
            continue
        if "run_idx" in df.columns:
            ids = set(df["run_idx"].unique().tolist())
            dupes = ids & seen_run_ids
            if dupes:
                print(f"  WARNING: {f} contains run_idx value(s) {dupes} "
                      f"already seen in another file, check for a stale "
                      f"file with a colliding run_idx.")
            seen_run_ids |= ids
        dfs.append(df)

    if not dfs:
        print("No readable files, nothing to merge.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    combined_path = os.path.join(pathsave, "robustness_all_runs.csv")
    combined.to_csv(combined_path, index=False)

    n_runs = combined["run_idx"].nunique() if "run_idx" in combined.columns else "?"
    print(f"Merged {len(combined)} rows from {len(dfs)} file(s), "
          f"covering {n_runs} distinct run(s) -> {combined_path}")

    if "sigma" in combined.columns and "accuracy" in combined.columns:
        summary = combined.groupby("sigma")["accuracy"].agg(["mean", "std", "count"]).reset_index()
        print("\nSummary (mean accuracy +/- std across runs, per sigma):")
        print(summary.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pathsave", nargs="+", help="One or more pathsave directories to merge")
    args = p.parse_args()

    for path in args.pathsave:
        merge_one(path)


if __name__ == "__main__":
    main()
