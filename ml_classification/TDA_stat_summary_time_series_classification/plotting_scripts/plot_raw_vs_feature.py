"""
plot_raw_vs_feature.py
Two-panel plot comparing feature-level vs. raw-signal noise injection:
(a) accuracy vs sigma, (b) ECE vs sigma.

Usage
-----
    python plot_raw_vs_feature.py /path/to/raw_vs_feature_robustness_merged.csv
    python plot_raw_vs_feature.py /path/to/merged.csv --out ./figs/
    python plot_raw_vs_feature.py /path/to/merged.csv --accuracy_only

Input format
------------
CSV with columns: level ('feature' or 'raw_signal'), sigma, accuracy, ece,
run_idx (extra columns such as snr_db, mean_confidence are ignored).

Output
------
raw_vs_feature.pdf / .png (two-panel), saved alongside the CSV unless
--out is given. With --accuracy_only, produces the single-panel
raw_vs_feature_accuracy.pdf / .png instead.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


# palette
C_ACC   = "#0072B2"   # blue   - feature-level curves
C_RAW   = "#EE8F0A"   # orange - raw-signal curves
C_GRID  = "#D9D9D9"
C_AXIS  = "#333333"
C_TEXT  = "#000000"
ALPHA_TR  = 0.10
LW_MEAN   = 2.0
LW_TRACE  = 0.6
MARKER_SZ = 5.5

FONT_LABEL = 10
FONT_TICK  = 9
FONT_LEG   = 8.5


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


def _style_ax(ax, xlabel="", ylabel=""):
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


def _plot_metric_vs_sigma(ax, raw_df: pd.DataFrame, metric: str,
                           ylabel: str, as_percent: bool, marker_feat="o", marker_raw="s"):
    """Draw `metric` vs sigma for both levels on a single axis: individual-run
    traces, mean line, +/- 1 std band per level. Shared by both panels."""
    level_style = {
        "feature":    dict(color=C_ACC, marker=marker_feat, label="Feature-level noise"),
        "raw_signal": dict(color=C_RAW, marker=marker_raw, label="Raw-signal noise"),
    }

    for level, style in level_style.items():
        sub_all = raw_df[raw_df["level"] == level]
        if sub_all.empty:
            print(f"  WARNING: no rows for level='{level}', skipping.")
            continue

        run_ids = sorted(sub_all["run_idx"].unique())
        for rid in run_ids:
            sub = sub_all[sub_all["run_idx"] == rid].sort_values("sigma")
            ax.plot(sub["sigma"], sub[metric], color=style["color"],
                    lw=LW_TRACE, alpha=ALPHA_TR, zorder=1)

        agg = sub_all.groupby("sigma")[metric].agg(["mean", "std"]).reset_index()
        agg = agg.sort_values("sigma")
        ax.fill_between(agg["sigma"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        color=style["color"], alpha=0.15, zorder=2, linewidth=0)
        ax.plot(agg["sigma"], agg["mean"], color=style["color"], lw=LW_MEAN,
                marker=style["marker"], ms=MARKER_SZ - 1, markeredgecolor="white",
                markeredgewidth=1.0, zorder=3,
                label=f"{style['label']} ($n={len(run_ids)}$)")

    all_sigmas = sorted(raw_df["sigma"].unique())
    ax.set_xlim(all_sigmas[0] - 0.02, all_sigmas[-1] + 0.02)
    if as_percent:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    _style_ax(ax, xlabel=r"Gaussian noise $\sigma$", ylabel=ylabel)


def plot_raw_vs_feature_accuracy(ax, raw_df: pd.DataFrame) -> None:
    """Panel (a): accuracy vs sigma, feature-level vs raw-signal."""
    _plot_metric_vs_sigma(ax, raw_df, metric="accuracy", ylabel="Accuracy",
                          as_percent=True, marker_feat="o", marker_raw="s")
    ax.set_ylim(0.45, 1.01)
    ax.legend(fontsize=FONT_LEG, loc="center right")


def plot_raw_vs_feature_ece(ax, raw_df: pd.DataFrame) -> None:
    """Panel (b): ECE vs sigma, feature-level vs raw-signal."""
    _plot_metric_vs_sigma(ax, raw_df, metric="ece", ylabel="ECE",
                          as_percent=True, marker_feat="D", marker_raw="^")
    ax.set_ylim(-0.02, 0.55)
    ax.legend(fontsize=FONT_LEG, loc="center right")


def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"level", "sigma", "accuracy", "run_idx"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing required columns: {missing}")
        sys.exit(1)
    for col in ("sigma", "accuracy", "run_idx"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "ece" in df.columns:
        df["ece"] = pd.to_numeric(df["ece"], errors="coerce")
    df = df.dropna(subset=["sigma", "accuracy", "run_idx"])
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to raw_vs_feature_robustness_merged.csv")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI (default 300)")
    parser.add_argument("--accuracy_only", action="store_true",
                         help="Produce only the single-panel accuracy figure "
                              "(raw_vs_feature_accuracy.pdf/.png) instead of the "
                              "two-panel accuracy+ECE figure.")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"ERROR: file not found: {args.csv_path}")
        sys.exit(1)

    out_dir = args.out or os.path.dirname(os.path.abspath(args.csv_path))
    os.makedirs(out_dir, exist_ok=True)

    _set_rcparams()
    df = _load(args.csv_path)

    print(f"Loaded {len(df)} rows from {args.csv_path}")
    print(f"  levels       : {sorted(df['level'].unique())}")
    print(f"  sigma levels : {sorted(df['sigma'].unique())}")
    print(f"  runs         : {df['run_idx'].nunique()}")
    print(f"  has ece      : {'ece' in df.columns}")

    if args.accuracy_only:
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        plot_raw_vs_feature_accuracy(ax, df)
        fig.tight_layout()
        out_path = os.path.join(out_dir, "raw_vs_feature_accuracy")
        fig.savefig(out_path + ".pdf", bbox_inches="tight")
        fig.savefig(out_path + ".png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {out_path}.pdf / .png")
        return

    if "ece" not in df.columns:
        print("ERROR: --accuracy_only was not given, but the CSV has no 'ece' "
              "column, cannot build panel (b). Re-run with --accuracy_only, "
              "or supply a CSV that includes ece.")
        sys.exit(1)

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7))
    plot_raw_vs_feature_accuracy(axs[0], df)
    plot_raw_vs_feature_ece(axs[1], df)
    axs[0].text(-0.18, 1.05, "(a)", transform=axs[0].transAxes,
                fontsize=FONT_LABEL, fontweight="bold")
    axs[1].text(-0.18, 1.05, "(b)", transform=axs[1].transAxes,
                fontsize=FONT_LABEL, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "raw_vs_feature")
    fig.savefig(out_path + ".pdf", bbox_inches="tight")
    fig.savefig(out_path + ".png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}.pdf / .png")


if __name__ == "__main__":
    main()
