"""
plot_raw_vs_feature_all_models.py
Two-panel plot comparing noise robustness across three models: TopTimeNet
(feature-level and raw-signal sweeps), CNN (raw-signal only), and
Transformer (raw-signal only, converged runs). Panel (a): accuracy vs
sigma. Panel (b): ECE vs sigma.

The CNN and Transformer have no intermediate feature representation, so
they only have a raw-signal curve. Four curves are plotted in total:
TopTimeNet feature-level, TopTimeNet raw-signal, CNN raw-signal, and
Transformer raw-signal.

Usage
-----
    python plot_raw_vs_feature_all_models.py \
        --toptimenet /path/to/raw_vs_feature_robustness_merged.csv \
        --cnn /path/to/cnn_robustness_all_runs.csv \
        --transformer /path/to/transformer_robustness_all_runs.csv \
        --transformer_exclude_seeds 51,53,60 \
        --out ./figs/

Input formats
-------------
--toptimenet : columns level ('feature'/'raw_signal'), sigma, accuracy,
               ece, run_idx.
--cnn, --transformer : columns sigma, accuracy, ece, run_idx (single
               level, always raw-signal for these two models). Rows for
               non-converged runs can be dropped via
               --cnn_exclude_run_idx / --transformer_exclude_run_idx
               (comma-separated run_idx values) or
               --transformer_exclude_seeds (comma-separated random_seed
               values, if the CSV has a random_seed column instead).

Output
------
raw_vs_feature_all_models.pdf / .png, saved to --out.
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
C_FEAT  = "#3B81F3"   # blue      - TopTimeNet feature-level
C_RAW   = "#07A8F3"   # cyan      - TopTimeNet raw-signal
C_CNN   = "#E06DEB"   # magenta   - CNN raw-signal
C_TRANS = "#11B508"   # green     - Transformer raw-signal
C_GRID  = "#D9D9D9"
C_AXIS  = "#333333"
C_TEXT  = "#000000"
ALPHA_TR  = 0.10
LW_MEAN   = 2.0
LW_TRACE  = 0.6
MARKER_SZ = 5.5

FONT_LABEL = 10
FONT_TICK  = 9
FONT_LEG   = 8


def _set_rcparams():
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


def _draw_curve(ax, df, metric, color, marker, label):
    """Draw one model/level curve: individual-run traces, mean line, std band."""
    run_ids = sorted(df["run_idx"].unique())
    for rid in run_ids:
        sub = df[df["run_idx"] == rid].sort_values("sigma")
        ax.plot(sub["sigma"], sub[metric], color=color, lw=LW_TRACE,
                alpha=ALPHA_TR, zorder=1)

    agg = df.groupby("sigma")[metric].agg(["mean", "std"]).reset_index().sort_values("sigma")
    ax.fill_between(agg["sigma"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                    color=color, alpha=0.15, zorder=2, linewidth=0)
    ax.plot(agg["sigma"], agg["mean"], color=color, lw=LW_MEAN, marker=marker,
            ms=MARKER_SZ - 1, markeredgecolor="white", markeredgewidth=1.0,
            zorder=3, label=f"{label} ($n={len(run_ids)}$)")


def _load_single_level(path, exclude_run_idx=None, exclude_seeds=None):
    """Load a single-level (raw-signal only) robustness CSV, optionally
    dropping non-converged runs by run_idx or random_seed."""
    df = pd.read_csv(path)
    required = {"sigma", "accuracy", "run_idx"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: {path} missing required columns: {missing}")
        sys.exit(1)
    for col in ("sigma", "accuracy", "run_idx"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "ece" in df.columns:
        df["ece"] = pd.to_numeric(df["ece"], errors="coerce")
    df = df.dropna(subset=["sigma", "accuracy", "run_idx"])

    if exclude_seeds and "random_seed" in df.columns:
        before = df["run_idx"].nunique()
        df = df[~df["random_seed"].isin(exclude_seeds)]
        print(f"  Excluded seeds {exclude_seeds}: {before} -> {df['run_idx'].nunique()} runs")
    if exclude_run_idx:
        before = df["run_idx"].nunique()
        df = df[~df["run_idx"].isin(exclude_run_idx)]
        print(f"  Excluded run_idx {exclude_run_idx}: {before} -> {df['run_idx'].nunique()} runs")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toptimenet", required=True,
                         help="raw_vs_feature_robustness_merged.csv (level, sigma, accuracy, ece, run_idx)")
    parser.add_argument("--cnn", required=True,
                         help="CNN robustness_all_runs.csv (sigma, accuracy, ece, run_idx)")
    parser.add_argument("--transformer", required=True,
                         help="Transformer robustness_all_runs.csv (sigma, accuracy, ece, run_idx)")
    parser.add_argument("--cnn_exclude_run_idx", default=None,
                         help="Comma-separated run_idx values to exclude from --cnn")
    parser.add_argument("--transformer_exclude_run_idx", default=None,
                         help="Comma-separated run_idx values to exclude from --transformer")
    parser.add_argument("--transformer_exclude_seeds", default=None,
                         help="Comma-separated random_seed values to exclude from --transformer")
    parser.add_argument("--out", default=".", help="Output directory")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    _set_rcparams()

    ttn = pd.read_csv(args.toptimenet)
    for col in ("sigma", "accuracy", "run_idx"):
        ttn[col] = pd.to_numeric(ttn[col], errors="coerce")
    if "ece" in ttn.columns:
        ttn["ece"] = pd.to_numeric(ttn["ece"], errors="coerce")
    ttn = ttn.dropna(subset=["sigma", "accuracy", "run_idx"])
    ttn_feat = ttn[ttn["level"] == "feature"]
    ttn_raw  = ttn[ttn["level"] == "raw_signal"]
    print(f"TopTimeNet: feature n={ttn_feat['run_idx'].nunique()}, "
          f"raw n={ttn_raw['run_idx'].nunique()}")

    cnn_exclude = [int(x) for x in args.cnn_exclude_run_idx.split(",")] if args.cnn_exclude_run_idx else None
    print("CNN:")
    cnn = _load_single_level(args.cnn, exclude_run_idx=cnn_exclude)

    trans_exclude_run = [int(x) for x in args.transformer_exclude_run_idx.split(",")] if args.transformer_exclude_run_idx else None
    trans_exclude_seeds = [int(x) for x in args.transformer_exclude_seeds.split(",")] if args.transformer_exclude_seeds else None
    print("Transformer:")
    trans = _load_single_level(args.transformer, exclude_run_idx=trans_exclude_run,
                                exclude_seeds=trans_exclude_seeds)

    has_ece = all("ece" in d.columns for d in (ttn_feat, ttn_raw, cnn, trans))

    ncols = 2 if has_ece else 1
    fig, axs = plt.subplots(1, ncols, figsize=(7.3, 3.2) if has_ece else (3.6, 3.2))
    if ncols == 1:
        axs = [axs]

    ax = axs[0]
    _draw_curve(ax, ttn_feat, "accuracy", C_FEAT, "o", "TopTimeNet (feature-level)")
    _draw_curve(ax, ttn_raw,  "accuracy", C_RAW,  "s", "TopTimeNet (raw-signal)")
    _draw_curve(ax, cnn,      "accuracy", C_CNN,  "^", "CNN (raw-signal)")
    _draw_curve(ax, trans,    "accuracy", C_TRANS,"D", "Transformer (raw-signal)")
    all_sigmas = sorted(ttn["sigma"].unique())
    ax.set_xlim(all_sigmas[0] - 0.02, all_sigmas[-1] + 0.02)
    ax.set_ylim(0.45, 1.01)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=FONT_LEG, loc="center right", framealpha=0.6)
    _style_ax(ax, xlabel=r"Gaussian noise $\sigma$", ylabel="Accuracy")
    if has_ece:
        ax.text(-0.18, 1.05, "(a)", transform=ax.transAxes, fontsize=FONT_LABEL, fontweight="bold")

    if has_ece:
        ax = axs[1]
        _draw_curve(ax, ttn_feat, "ece", C_FEAT, "o", "TopTimeNet (feature-level)")
        _draw_curve(ax, ttn_raw,  "ece", C_RAW,  "s", "TopTimeNet (raw-signal)")
        _draw_curve(ax, cnn,      "ece", C_CNN,  "^", "CNN (raw-signal)")
        _draw_curve(ax, trans,    "ece", C_TRANS,"D", "Transformer (raw-signal)")
        ax.set_xlim(all_sigmas[0] - 0.02, all_sigmas[-1] + 0.02)
        ax.set_ylim(-0.02, 0.55)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        _style_ax(ax, xlabel=r"Gaussian noise $\sigma$", ylabel="ECE")
        ax.text(-0.18, 1.05, "(b)", transform=ax.transAxes, fontsize=FONT_LABEL, fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(args.out, "raw_vs_feature_all_models")
    fig.savefig(out_path + ".pdf", bbox_inches="tight")
    fig.savefig(out_path + ".png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}.pdf / .png")


if __name__ == "__main__":
    main()
