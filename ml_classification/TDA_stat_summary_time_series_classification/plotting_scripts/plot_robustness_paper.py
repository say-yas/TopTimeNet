"""
plot_robustness_paper.py
Publication-quality plots of a noise-robustness sweep, styled for a
white-background journal figure.

Usage
-----
    python plot_robustness_paper.py /path/to/robustness_all_runs.csv
    python plot_robustness_paper.py /path/to/robustness_all_runs.csv --out ./figs/

Plots produced (PDF + PNG, saved alongside the CSV unless --out is given)
--------------------------------------------------------------------
  1. accuracy_vs_snr        Mean accuracy vs SNR (dB), individual runs as
                            thin background traces, +/- 1 std band.
  2. ece_vs_sigma           Mean Expected Calibration Error vs Gaussian
                            noise sigma, same run-trace/band treatment.
  3. accuracy_ece_combined  Two-panel figure combining (1) and (2) side
                            by side.
  4. per_run_heatmap        Accuracy heatmap, rows = run, columns = sigma.

Design choices
--------------
  * White figure/axes background, not transparent, so the figure renders
    correctly regardless of viewer or embedding context.
  * Serif font family and STIX math font, to match a LaTeX document's
    body text and inline math in axis labels.
  * Colorblind-safe, print-safe palette (Wong 2011).
  * Vector PDF as the primary output; PNG at 300 DPI as a secondary
    preview output.
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


# palette
C_ACC     = "#0072B2"   # blue         - accuracy
C_CONF    = "#009E73"   # bluish-green - confidence
C_GAP     = "#D55E00"   # vermillion   - gap / ECE / warning
C_TRACE   = "#0072B2"   # individual run traces (same hue as mean, low alpha)
C_GRID    = "#D9D9D9"
C_AXIS    = "#333333"
C_TEXT    = "#000000"
ALPHA_TR  = 0.10
LW_MEAN   = 2.0
LW_TRACE  = 0.6
MARKER_SZ = 5.5

FONT_LABEL = 10
FONT_TICK  = 9
FONT_LEG   = 8.5
FONT_TITLE = 11


def _set_rcparams():
    """Journal-style rcParams: serif fonts, white background, thin spines."""
    matplotlib.rcParams.update({
        "font.family":       "serif",
        "mathtext.fontset":  "stix",
        "axes.edgecolor":    C_AXIS,
        "axes.labelcolor":   C_TEXT,
        "text.color":        C_TEXT,
        "xtick.color":       C_AXIS,
        "ytick.color":       C_AXIS,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "axes.linewidth":    0.8,
        "legend.frameon":    True,
        "legend.edgecolor":  C_GRID,
        "legend.framealpha": 1.0,
    })


def _style_ax(ax, xlabel="", ylabel="", title=None):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_AXIS)
    ax.spines["bottom"].set_color(C_AXIS)
    ax.tick_params(colors=C_AXIS, labelsize=FONT_TICK, length=3)
    ax.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL, color=C_TEXT)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL, color=C_TEXT)
    if title:
        ax.set_title(title, fontsize=FONT_TITLE, color=C_TEXT, pad=8)


def _save(fig, path_noext, dpi):
    fig.savefig(path_noext + ".pdf", bbox_inches="tight")
    fig.savefig(path_noext + ".png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path_noext}.pdf / .png")


def _plot_accuracy_vs_snr(ax, df, sigma_arr, acc_mean, acc_std, run_ids, n_runs):
    """Draw the accuracy-vs-SNR curve onto a given axis. Reused standalone
    and inside the combined two-panel figure."""
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
        ca     = clean_row["accuracy"].mean()
        ca_std = clean_row["accuracy"].std()
        step   = snr_arr[0] - snr_arr[1] if len(snr_arr) > 1 else 6
        clean_x  = snr_arr[0] + step
        x_plot   = np.concatenate([[clean_x], snr_arr])
        acc_plot = np.concatenate([[ca],       snr_acc_mean])
        std_plot = np.concatenate([[ca_std],   snr_acc_std])
    else:
        x_plot, acc_plot, std_plot = snr_arr, snr_acc_mean, snr_acc_std

    for rid in run_ids:
        sub = df_finite[df_finite["run_idx"] == rid].sort_values(
            "snr_db", ascending=False)
        xs, ys = list(sub["snr_db"].values), list(sub["accuracy"].values)
        if has_clean:
            crow = df[(df["run_idx"] == rid) & (~np.isfinite(df["snr_db"].astype(float)))]
            if not crow.empty:
                xs = [clean_x] + xs
                ys = [crow["accuracy"].values[0]] + ys
        ax.plot(xs, ys, color=C_TRACE, lw=LW_TRACE, alpha=ALPHA_TR, zorder=1)

    ax.fill_between(x_plot, acc_plot - std_plot, acc_plot + std_plot,
                    color=C_ACC, alpha=0.15, zorder=2, linewidth=0)
    ax.plot(x_plot, acc_plot, color=C_ACC, lw=LW_MEAN, marker="o",
            ms=MARKER_SZ, markeredgecolor="white", markeredgewidth=1.0,
            zorder=3, label=f"Mean accuracy ($n={n_runs}$)")

    ax.invert_xaxis()
    if has_clean:
        ticks_sorted = sorted(list(snr_arr) + [clean_x], reverse=True)
        labels = ["$\\infty$" if t == clean_x else f"{int(round(t))}" for t in ticks_sorted]
        ax.set_xticks(ticks_sorted)
        ax.set_xticklabels(labels, fontsize=FONT_TICK, color=C_AXIS)

    ax.set_ylim(max(0, acc_plot.min() - std_plot.max() - 0.05), 1.01)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=FONT_LEG, loc="lower left")
    _style_ax(ax, xlabel="SNR (dB)", ylabel="Accuracy")


def _plot_ece_vs_sigma(ax, df, sigma_arr, run_ids):
    ece_per_run = []
    for rid in run_ids:
        sub = df[df["run_idx"] == rid].sort_values("sigma")
        ax.plot(sub["sigma"], sub["ece"], color=C_GAP, lw=LW_TRACE,
                alpha=ALPHA_TR, zorder=1)
        ece_per_run.append(sub["ece"].values)

    ece_arr  = np.array(ece_per_run)
    ece_mean = ece_arr.mean(axis=0)
    ece_std  = ece_arr.std(axis=0)

    ax.fill_between(sigma_arr, ece_mean - ece_std, ece_mean + ece_std,
                    color=C_GAP, alpha=0.15, zorder=2, linewidth=0)
    ax.plot(sigma_arr, ece_mean, color=C_GAP, lw=LW_MEAN, marker="D",
            ms=MARKER_SZ - 1, markeredgecolor="white", markeredgewidth=1.0,
            zorder=3, label="Mean ECE")
    ax.axhline(0, color=C_AXIS, lw=0.8, linestyle="--", alpha=0.6,
               label="Perfect calibration")

    ax.set_xlim(sigma_arr[0] - 0.02, sigma_arr[-1] + 0.02)
    ax.set_ylim(-0.006, max(ece_mean + ece_std) * 1.08)
    ax.legend(fontsize=FONT_LEG, loc="upper left")
    _style_ax(ax, xlabel=r"Gaussian noise $\sigma$", ylabel="ECE")


def main():
    parser = argparse.ArgumentParser(
        description="Publication-styled plots of a noise-robustness sweep."
    )
    parser.add_argument("csv_path", help="Path to robustness_all_runs.csv")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI (default 300)")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"ERROR: file not found: {args.csv_path}")
        sys.exit(1)

    out_dir = args.out or os.path.dirname(os.path.abspath(args.csv_path))
    os.makedirs(out_dir, exist_ok=True)

    _set_rcparams()

    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} rows from {args.csv_path}")

    required = {"sigma", "accuracy", "run_idx"}
    missing  = required - set(df.columns)
    if missing:
        print(f"ERROR: missing required columns: {missing}")
        sys.exit(1)

    df["sigma"]    = pd.to_numeric(df["sigma"],    errors="coerce")
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["run_idx"]  = pd.to_numeric(df["run_idx"],  errors="coerce")
    df = df.dropna(subset=["sigma", "accuracy", "run_idx"])

    sigmas  = sorted(df["sigma"].unique())
    run_ids = sorted(df["run_idx"].unique())
    n_runs  = len(run_ids)
    has_snr = "snr_db" in df.columns
    has_ece = "ece"    in df.columns

    print(f"sigma levels : {sigmas}")
    print(f"runs         : {n_runs}")

    agg = df.groupby("sigma")["accuracy"].agg(["mean", "std"]).reset_index()
    sigma_arr = agg["sigma"].values
    acc_mean  = agg["mean"].values
    acc_std   = agg["std"].values

    dpi = args.dpi

    if has_snr:
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        _plot_accuracy_vs_snr(ax, df, sigma_arr, acc_mean, acc_std, run_ids, n_runs)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, "accuracy_vs_snr"), dpi)

    if has_ece:
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        _plot_ece_vs_sigma(ax, df, sigma_arr, run_ids)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, "ece_vs_sigma"), dpi)

    if has_snr and has_ece:
        fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7))
        _plot_accuracy_vs_snr(axs[0], df, sigma_arr, acc_mean, acc_std, run_ids, n_runs)
        _plot_ece_vs_sigma(axs[1], df, sigma_arr, run_ids)
        axs[0].text(-0.18, 1.05, "(a)", transform=axs[0].transAxes,
                    fontsize=FONT_LABEL, fontweight="bold")
        axs[1].text(-0.18, 1.05, "(b)", transform=axs[1].transAxes,
                    fontsize=FONT_LABEL, fontweight="bold")
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, "accuracy_ece_combined"), dpi)

    pivot = df.pivot_table(index="run_idx", columns="sigma",
                           values="accuracy", aggfunc="mean")
    pivot = pivot.reindex(sorted(pivot.index))
    pivot = pivot[sorted(pivot.columns)]

    fig_h = max(2.6, len(run_ids) * 0.16)
    fig, ax = plt.subplots(figsize=(4.2, fig_h))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues",
                   vmin=0.75, vmax=1.0, interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{s:.2f}" for s in pivot.columns],
                       fontsize=FONT_TICK - 1, color=C_AXIS)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{int(r)}" for r in pivot.index],
                       fontsize=FONT_TICK - 2, color=C_AXIS)
    ax.set_xlabel(r"Gaussian noise $\sigma$", fontsize=FONT_LABEL, color=C_TEXT)
    ax.set_ylabel("Run index", fontsize=FONT_LABEL, color=C_TEXT)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Accuracy", fontsize=FONT_LABEL - 1, color=C_TEXT)
    cbar.ax.tick_params(labelsize=FONT_TICK - 1, colors=C_AXIS)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "per_run_heatmap"), dpi)

    print(f"Figures written to: {out_dir}/")


if __name__ == "__main__":
    main()
