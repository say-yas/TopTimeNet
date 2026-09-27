"""
plot_robustness.py
Plots robustness sweep results from robustness_all_runs.csv (or any CSV
with the same schema).

Usage
-----
    python plot_robustness.py /path/to/robustness_all_runs.csv
    python plot_robustness.py /path/to/robustness_all_runs.csv --dpi 200
    python plot_robustness.py /path/to/robustness_all_runs.csv --out ./figs/

Plots produced (saved alongside the CSV unless --out is given)
----------------------------------------------------------
  1. accuracy_vs_sigma.png
       Mean +/- std accuracy across runs as a function of sigma.
       Individual run traces shown as thin lines in the background.

  2. confidence_vs_sigma.png
       Mean +/- std mean_confidence across runs.
       Accuracy mean overlaid as a dashed line for direct comparison
       (shows the confidence-accuracy gap, i.e. overconfidence region).

  3. calibration_gap.png
       confidence - accuracy per run per sigma, showing how
       overconfidence evolves with noise. Horizontal line at 0 is
       perfect calibration.

  4. accuracy_vs_snr.png
       Same data as plot 1 but on the SNR (dB) x-axis instead of sigma.
       Excludes the sigma=0 / SNR=inf point (plotted as a separate
       marker).

  5. per_run_heatmap.png
       Heatmap: rows = run index, columns = sigma level, color = accuracy.
       Shows which runs are consistently robust and which are fragile.

  6. ece_vs_sigma.png (only if an 'ece' column is present)
       Mean +/- std Expected Calibration Error across runs.

  7. summary_table.png
       Rendered table of mean +/- std for accuracy, confidence, and
       (if present) ECE at each sigma.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# style constants
C_ACC    = "#0967db"   # blue         - accuracy
C_CONF   = "#b8ef2e"   # lime         - confidence
C_GAP    = "#f75d2e"   # orange-red   - gap / warning
C_GRID   = "#e1e0d9"
C_TICK   = "#898781"
C_TEXT   = "#aaa9a9"
C_TRACE  = "#78b0f4"   # individual run traces (light)
ALPHA_TR = 0.18        # opacity of individual run traces
LW_MEAN  = 2.5
LW_TRACE = 0.8
FONT_TITLE = 13
FONT_LABEL = 11
FONT_TICK  = 10


def _style_ax(ax, xlabel="", ylabel="", title=""):
    """Apply shared axis styling: transparent background, light grid, title."""
    ax.set_facecolor("none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(C_GRID)
    ax.tick_params(colors=C_TICK, labelsize=FONT_TICK)
    ax.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL, color=C_TICK)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL, color=C_TICK)
    ax.set_title(title,   fontsize=FONT_TITLE, fontweight="500", pad=10, color=C_TEXT)


def _save(fig, path, dpi):
    """Save a figure with a transparent background and close it."""
    fig.patch.set_alpha(0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  Saved -> {path}")


def _pct(v):
    """Format a fraction as a percentage string."""
    return f"{v * 100:.1f}%"


def main():
    parser = argparse.ArgumentParser(
        description="Plot noise robustness sweep results."
    )
    parser.add_argument(
        "csv_path",
        help="Path to robustness_all_runs.csv (or any compatible CSV)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory for figures (default: same directory as CSV)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Figure DPI (default: 150)",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        print(f"ERROR: file not found: {csv_path}")
        sys.exit(1)

    out_dir = args.out or os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns : {list(df.columns)}")

    required = {"sigma", "accuracy", "mean_confidence", "run_idx"}
    missing  = required - set(df.columns)
    if missing:
        print(f"ERROR: missing required columns: {missing}")
        sys.exit(1)

    df["sigma"]           = pd.to_numeric(df["sigma"],           errors="coerce")
    df["accuracy"]        = pd.to_numeric(df["accuracy"],        errors="coerce")
    df["mean_confidence"] = pd.to_numeric(df["mean_confidence"], errors="coerce")
    df["run_idx"]         = pd.to_numeric(df["run_idx"],         errors="coerce")
    df = df.dropna(subset=["sigma", "accuracy", "mean_confidence", "run_idx"])

    sigmas   = sorted(df["sigma"].unique())
    run_ids  = sorted(df["run_idx"].unique())
    n_runs   = len(run_ids)
    has_snr  = "snr_db" in df.columns
    has_ece  = "ece"    in df.columns

    print(f"sigma levels : {sigmas}")
    print(f"Runs : {n_runs}  ({run_ids[0]:.0f} to {run_ids[-1]:.0f})")

    agg = (
        df.groupby("sigma")[["accuracy", "mean_confidence"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["sigma", "acc_mean", "acc_std", "conf_mean", "conf_std"]
    if has_ece:
        ece_agg = df.groupby("sigma")["ece"].agg(["mean", "std"]).reset_index()
        ece_agg.columns = ["sigma", "ece_mean", "ece_std"]
        agg = agg.merge(ece_agg, on="sigma")

    sigma_arr    = agg["sigma"].values
    acc_mean     = agg["acc_mean"].values
    acc_std      = agg["acc_std"].values
    conf_mean    = agg["conf_mean"].values
    conf_std     = agg["conf_std"].values

    meta_parts = []
    for col in ("fusion", "opt", "mode"):
        if col in df.columns:
            vals = df[col].dropna().unique()
            if len(vals) == 1:
                meta_parts.append(f"{col}={vals[0]}")
    meta_str = "  |  ".join(meta_parts) if meta_parts else ""

    dpi = args.dpi

    # Plot 1: accuracy vs sigma
    fig, ax = plt.subplots(figsize=(7, 4))

    for rid in run_ids:
        sub = df[df["run_idx"] == rid].sort_values("sigma")
        ax.plot(sub["sigma"], sub["accuracy"],
                color=C_TRACE, lw=LW_TRACE, alpha=ALPHA_TR, zorder=1)

    ax.fill_between(sigma_arr,
                    acc_mean - acc_std,
                    acc_mean + acc_std,
                    color=C_ACC, alpha=0.12, zorder=2)
    ax.plot(sigma_arr, acc_mean,
            color=C_ACC, lw=LW_MEAN, marker="o", ms=6,
            markeredgecolor="white", markeredgewidth=1.5,
            zorder=3, label=f"Mean accuracy  (n={n_runs})")

    clean_acc = acc_mean[sigma_arr == 0.0]
    if len(clean_acc):
        ax.axhline(clean_acc[0], color=C_ACC, lw=0.8,
                   linestyle="--", alpha=0.4, zorder=1)
        ax.text(sigma_arr[-1], clean_acc[0] + 0.005,
                f"Clean: {_pct(clean_acc[0])}",
                ha="right", va="bottom", fontsize=9, color=C_ACC)

    ax.set_xlim(sigma_arr[0] - 0.02, sigma_arr[-1] + 0.02)
    ax.set_ylim(max(0, acc_mean.min() - acc_std.max() - 0.05), 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, frameon=True)
    _style_ax(ax,
              xlabel="Gaussian noise sigma",
              ylabel="Accuracy",
              title=f"Accuracy vs Noise Level\n{meta_str}")
    _save(fig, os.path.join(out_dir, "accuracy_vs_sigma.png"), dpi)

    # Plot 2: confidence vs sigma with accuracy overlay
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.fill_between(sigma_arr,
                    conf_mean - conf_std,
                    conf_mean + conf_std,
                    color=C_CONF, alpha=0.12, zorder=2)
    ax.plot(sigma_arr, conf_mean,
            color=C_CONF, lw=LW_MEAN, marker="s", ms=6,
            markeredgecolor="white", markeredgewidth=1.5,
            zorder=3, label="Mean confidence")

    ax.fill_between(sigma_arr,
                    acc_mean - acc_std,
                    acc_mean + acc_std,
                    color=C_ACC, alpha=0.10, zorder=2)
    ax.plot(sigma_arr, acc_mean,
            color=C_ACC, lw=LW_MEAN, linestyle="--",
            marker="o", ms=5,
            markeredgecolor="white", markeredgewidth=1.5,
            zorder=3, label="Mean accuracy")

    ax.fill_between(sigma_arr, acc_mean, conf_mean,
                    where=(conf_mean > acc_mean),
                    color=C_GAP, alpha=0.10, label="Overconfidence gap",
                    zorder=1)

    ax.set_xlim(sigma_arr[0] - 0.02, sigma_arr[-1] + 0.02)
    ax.set_ylim(max(0, min(acc_mean.min(), conf_mean.min()) - 0.05), 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, frameon=True)
    _style_ax(ax,
              xlabel="Gaussian noise sigma",
              ylabel="Value",
              title=f"Confidence vs Accuracy, Calibration View\n{meta_str}")
    _save(fig, os.path.join(out_dir, "confidence_vs_sigma.png"), dpi)

    # Plot 3: calibration gap (confidence - accuracy)
    fig, ax = plt.subplots(figsize=(7, 4))

    gap_per_run = []
    for rid in run_ids:
        sub  = df[df["run_idx"] == rid].sort_values("sigma")
        s    = sub["sigma"].values
        gap  = sub["mean_confidence"].values - sub["accuracy"].values
        ax.plot(s, gap, color=C_GAP, lw=LW_TRACE, alpha=ALPHA_TR + 0.05, zorder=1)
        gap_per_run.append(gap)

    gap_arr  = np.array(gap_per_run)
    gap_mean = gap_arr.mean(axis=0)
    gap_std  = gap_arr.std(axis=0)

    ax.fill_between(sigma_arr, gap_mean - gap_std, gap_mean + gap_std,
                    color=C_GAP, alpha=0.15, zorder=2)
    ax.plot(sigma_arr, gap_mean,
            color=C_GAP, lw=LW_MEAN, marker="D", ms=6,
            markeredgecolor="white", markeredgewidth=1.5,
            zorder=3, label="Mean gap (conf - acc)")
    ax.axhline(0, color=C_TICK, lw=1.0, linestyle="--", zorder=4,
               label="Perfect calibration (gap = 0)")

    ax.set_xlim(sigma_arr[0] - 0.02, sigma_arr[-1] + 0.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.legend(fontsize=9, frameon=True)
    _style_ax(ax,
              xlabel="Gaussian noise sigma",
              ylabel="Confidence - Accuracy",
              title=f"Calibration Gap vs Noise\n{meta_str}")
    _save(fig, os.path.join(out_dir, "calibration_gap.png"), dpi)

    # Plot 4: accuracy vs SNR (dB)
    if has_snr:
        df_finite = df[np.isfinite(df["snr_db"].astype(float))].copy()
        df_finite["snr_db"] = df_finite["snr_db"].astype(float)

        snr_agg = (
            df_finite.groupby("snr_db")["accuracy"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("snr_db", ascending=False)
        )
        snr_arr      = snr_agg["snr_db"].values
        snr_acc_mean = snr_agg["mean"].values
        snr_acc_std  = snr_agg["std"].values

        clean_row = df[~np.isfinite(df["snr_db"].astype(float))]
        has_clean = not clean_row.empty
        if has_clean:
            ca      = clean_row["accuracy"].mean()
            ca_std  = clean_row["accuracy"].std()
            step       = snr_arr[0] - snr_arr[1] if len(snr_arr) > 1 else 6
            clean_x    = snr_arr[0] + step
            x_plot     = np.concatenate([[clean_x], snr_arr])
            acc_plot   = np.concatenate([[ca],       snr_acc_mean])
            std_plot   = np.concatenate([[ca_std],   snr_acc_std])
        else:
            x_plot   = snr_arr
            acc_plot = snr_acc_mean
            std_plot = snr_acc_std

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.fill_between(x_plot,
                        acc_plot - std_plot,
                        acc_plot + std_plot,
                        color=C_ACC, alpha=0.12, zorder=2)
        ax.plot(x_plot, acc_plot,
                color=C_ACC, lw=LW_MEAN, marker="o", ms=6,
                markeredgecolor="white", markeredgewidth=1.5,
                zorder=3, label=f"Mean accuracy  (n={n_runs})")

        if has_clean:
            ax.scatter([clean_x], [ca], s=110, color=C_ACC,
                       zorder=5, marker="*", label=f"sigma=0 (inf dB, clean): {_pct(ca)}")
            ax.axvline(clean_x - step * 0.5, color=C_GRID,
                       lw=1.2, linestyle=":", zorder=1)
            ax.text(clean_x, ca + 0.008, "sigma=0\ninf dB",
                    ha="center", va="bottom", fontsize=8, color=C_ACC)

        ax.invert_xaxis()

        if has_clean:
            ticks_sorted = sorted(list(snr_arr) + [clean_x], reverse=True)
            labels = []
            for t in ticks_sorted:
                if t == clean_x:
                    labels.append("inf dB")
                else:
                    labels.append(f"{int(round(t))} dB")
            ax.set_xticks(ticks_sorted)
            ax.set_xticklabels(labels, fontsize=FONT_TICK, color=C_TICK)

        ax.set_ylim(max(0, acc_plot.min() - std_plot.max() - 0.05), 1.02)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=9, frameon=True)
        _style_ax(ax,
                  xlabel="SNR (dB), less noise on the left, more noise on the right",
                  ylabel="Accuracy",
                  title=f"Accuracy vs SNR\n{meta_str}")
        _save(fig, os.path.join(out_dir, "accuracy_vs_snr.png"), dpi)

    # Plot 5: per-run heatmap
    pivot = df.pivot_table(index="run_idx", columns="sigma",
                           values="accuracy", aggfunc="mean")
    pivot = pivot.reindex(sorted(pivot.index))
    pivot = pivot[sorted(pivot.columns)]

    fig_h = max(4, len(run_ids) * 0.32)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0.5, vmax=1.0, interpolation="nearest")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{s:.2f}" for s in pivot.columns],
                       fontsize=8, color=C_TICK)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"run {int(r)}" for r in pivot.index],
                       fontsize=8, color=C_TICK)
    ax.set_xlabel("Gaussian noise sigma", fontsize=FONT_LABEL, color=C_TICK)
    ax.set_ylabel("Run",              fontsize=FONT_LABEL, color=C_TICK)
    ax.set_title(f"Accuracy Heatmap, per Run x sigma\n{meta_str}",
                 fontsize=FONT_TITLE, fontweight="500", pad=10, color=C_TEXT)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if v < 0.72 else "#222")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Accuracy", fontsize=9, color=C_TICK)
    cbar.ax.tick_params(labelsize=8, colors=C_TICK)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "per_run_heatmap.png"), dpi)

    # Plot 6: ECE vs sigma (only if ECE column present)
    if has_ece:
        ece_per_run = []
        fig, ax = plt.subplots(figsize=(7, 4))

        for rid in run_ids:
            sub = df[df["run_idx"] == rid].sort_values("sigma")
            ax.plot(sub["sigma"], sub["ece"],
                    color=C_GAP, lw=LW_TRACE, alpha=ALPHA_TR, zorder=1)
            ece_per_run.append(sub["ece"].values)

        ece_arr  = np.array(ece_per_run)
        ece_mean = ece_arr.mean(axis=0)
        ece_std  = ece_arr.std(axis=0)

        ax.fill_between(sigma_arr, ece_mean - ece_std, ece_mean + ece_std,
                        color=C_GAP, alpha=0.15, zorder=2)
        ax.plot(sigma_arr, ece_mean,
                color=C_GAP, lw=LW_MEAN, marker="D", ms=6,
                markeredgecolor="white", markeredgewidth=1.5,
                zorder=3, label="Mean ECE")
        ax.axhline(0, color=C_TICK, lw=0.8, linestyle="--",
                   label="Perfect calibration (ECE = 0)")

        ax.set_xlim(sigma_arr[0] - 0.02, sigma_arr[-1] + 0.02)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, frameon=True)
        _style_ax(ax,
                  xlabel="Gaussian noise sigma",
                  ylabel="ECE (Expected Calibration Error)",
                  title=f"Calibration Error vs Noise\n{meta_str}")
        _save(fig, os.path.join(out_dir, "ece_vs_sigma.png"), dpi)

    # Plot 7: summary table
    rows = []
    for _, r in agg.iterrows():
        row = {
            "sigma"       : f"{r['sigma']:.2f}",
            "Accuracy"    : f"{r['acc_mean']*100:.1f} +/- {r['acc_std']*100:.1f}%",
            "Confidence"  : f"{r['conf_mean']*100:.1f} +/- {r['conf_std']*100:.1f}%",
            "Gap"         : f"{(r['conf_mean']-r['acc_mean'])*100:.1f}pp",
        }
        if has_ece:
            row["ECE"] = f"{r['ece_mean']:.3f} +/- {r['ece_std']:.3f}"
        rows.append(row)

    tbl_df  = pd.DataFrame(rows)
    n_cols  = len(tbl_df.columns)
    n_rows  = len(tbl_df)
    fig_w   = max(6, n_cols * 1.6)
    fig_hh  = 0.5 + n_rows * 0.38

    fig, ax = plt.subplots(figsize=(fig_w, fig_hh))
    ax.axis("off")

    tbl = ax.table(
        cellText   = tbl_df.values,
        colLabels  = tbl_df.columns,
        cellLoc    = "center",
        loc        = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#2a78d6")
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(1, n_rows + 1):
        bg = "#f4f4f2" if i % 2 == 0 else "white"
        for j in range(n_cols):
            tbl[i, j].set_facecolor(bg)
            tbl[i, j].set_edgecolor(C_GRID)

    ax.set_title(f"Robustness Summary  (mean +/- std over {n_runs} runs)\n{meta_str}",
                 fontsize=FONT_TITLE, fontweight="500", pad=6, color=C_TEXT, y=1.02)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "summary_table.png"), dpi)

    print(f"\nSummary  ({n_runs} runs, {len(sigmas)} sigma levels)")
    print(f"  {'sigma':>5}  {'Accuracy':>16}  {'Confidence':>16}  {'Gap':>8}")
    for _, r in agg.iterrows():
        gap = (r["conf_mean"] - r["acc_mean"]) * 100
        print(
            f"  {r['sigma']:>5.2f}  "
            f"{r['acc_mean']*100:>6.2f} +/- {r['acc_std']*100:<6.2f}  "
            f"{r['conf_mean']*100:>6.2f} +/- {r['conf_std']*100:<6.2f}  "
            f"{gap:>+6.2f}pp"
        )
    print(f"\n  Figures written to: {out_dir}/")


if __name__ == "__main__":
    main()
