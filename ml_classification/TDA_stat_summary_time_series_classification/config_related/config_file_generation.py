"""
create_config.py
Interactively generates config.json for TDAEnd2EndNet / CachedGroupMLP.

Usage
-----
    python create_config.py                          # writes ./config.json
    python create_config.py --output my_cfg.json      # custom output path
    python create_config.py --no-interactive          # all defaults, no prompts

The script asks one question per parameter group, shows the default,
and accepts an empty Enter to keep it.
"""

import argparse
import json
import os


# CLI
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate config.json for TDAEnd2EndNet / CachedGroupMLP"
    )
    p.add_argument("--output", default="config.json",
                   help="Output path (default: ./config.json)")
    p.add_argument("--no-interactive", action="store_true",
                   help="Write all defaults without asking questions")
    return p.parse_args()


# prompt helpers

def ask(prompt: str, default, cast=str, choices=None):
    """Print prompt with default, read input, cast to type. Empty input returns default."""
    if isinstance(default, bool):
        hint = "true/false"
    elif choices:
        hint = " | ".join(str(c) for c in choices)
    else:
        hint = str(default)

    while True:
        try:
            raw = input(f"  {prompt} [{hint}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default

        if raw == "":
            return default

        if cast is bool:
            if raw.lower() in ("true", "yes", "y", "1"):  return True
            if raw.lower() in ("false", "no", "n", "0"):  return False
            print("    Enter true or false.")
            continue

        if cast is list:
            try:
                raw2 = raw.strip("[]").replace(",", " ").split()
                return [float(v) for v in raw2] if raw2 else []
            except ValueError:
                print("    Enter space- or comma-separated numbers, e.g. 0.0 0.05 0.1")
                continue

        if cast is list_int:
            try:
                raw2 = raw.strip("[]").replace(",", " ").split()
                return [int(v) for v in raw2] if raw2 else []
            except ValueError:
                print("    Enter space- or comma-separated integers, e.g. 64 32")
                continue

        if choices and raw not in [str(c) for c in choices]:
            print(f"    Choose one of: {choices}")
            continue

        try:
            return cast(raw)
        except (ValueError, TypeError):
            print(f"    Expected {cast.__name__}, got: {raw!r}")


def list_int(x):
    """Cast helper for integer lists."""
    parts = str(x).strip("[]").replace(",", " ").split()
    return [int(v) for v in parts]


def section(title: str):
    print(f"\n{'-'*55}")
    print(f"  {title}")
    print(f"{'-'*55}")


# ============================================================================
def build_config(interactive: bool) -> dict:

    def q(prompt, default, cast=str, choices=None):
        if not interactive:
            return default
        return ask(prompt, default, cast, choices)

    cfg = {}

    # paths
    section("Paths")
    cfg["path_data"]   = q("Path to data directory",    "/path/to/your/data/")
    cfg["h5_filename"] = q("HDF5 filename",              "your_data.h5")
    cfg["path_save"]   = q("Output / results directory", "./results/")

    # reproducibility
    section("Reproducibility and device")
    cfg["random_seed"] = q("Random seed",              42,    int)
    cfg["force_cpu"]   = q("Force CPU (disable GPU)?", False, bool)

    # data preprocessing
    section("Data preprocessing")
    cfg["length_series"]         = q("Full series length (samples)",     100_000, int)
    cfg["segmentation_duration"] = q("Segment window length (samples)",      500, int)
    cfg["num_channels"]          = q("Number of channels",                     1, int)

    if interactive:
        raw_excl = input(
            "  States to exclude (comma-separated, or Enter for none): "
        ).strip()
        cfg["exclude_states"] = [s.strip() for s in raw_excl.split(",") if s.strip()]
        raw_keep = input(
            "  States to keep (comma-separated, or Enter for all): "
        ).strip()
        cfg["keep_states"] = [s.strip() for s in raw_keep.split(",") if s.strip()]
    else:
        cfg["exclude_states"] = []
        cfg["keep_states"]    = ["periodic", "chaotic"]

    cfg["balance_strategy"] = q(
        "Class balance strategy",
        "undersample",
        choices=["none", "undersample", "oversample", "hybrid"],
    )

    # TDA
    section("TDA hyperparameters")
    cfg["takens_dim"]            = q("Takens embedding dimension (2 or 3)",    2,     int)
    cfg["takens_delay"]          = q("Takens time delay (samples)",           20,     int)
    cfg["n_hom_dims"]            = q("Homology dimensions (1=H0, 2=H0+H1)",   2,     int)
    cfg["n_betti_bins"]          = q("Betti curve resolution",                30,     int)
    cfg["n_pi_bins"]             = q("Persistence image grid size (n x n)",   15,     int)
    cfg["pi_sigma"]              = q("PI Gaussian bandwidth",               0.005,  float)
    cfg["ph_workers"]            = q("Ripser parallel workers",                4,     int)
    cfg["precompute_batch_size"] = q("Batch size for TDA pre-compute",        64,     int)

    # cache
    section("TDA feature cache")
    cfg["use_cache"] = q(
        "Pre-compute TDA features once and cache? (recommended for speed)",
        True, bool,
    )
    default_cache = os.path.join(cfg["path_save"], "tda_cache.pt")
    cfg["tda_cache_path"]  = q("Cache file path", default_cache)
    cfg["force_recompute"] = q(
        "Force recompute cache even if it already exists?",
        False, bool,
    )

    # IQR clipping
    section("IQR feature clipping")
    if interactive:
        print("  Clips each feature to [lo, hi] percentile bounds computed")
        print("  at precompute time, then re-applied to noisy features in")
        print("  the robustness sweep. 1/99 is a safe default.")
    cfg["clip_lo_pct"] = q("Lower clip percentile", 1.0,  float)
    cfg["clip_hi_pct"] = q("Upper clip percentile", 99.0, float)

    # model architecture
    section("Model architecture")
    cfg["embed_dim"] = q(
        "Shared token embedding dim D (GroupProjector output)",
        32, int,
    )
    cfg["fusion"] = q(
        "Fusion strategy",
        "low_rank",
        choices=["bilinear", "gated", "linear_attn", "low_rank", "mgta"],
    )
    cfg["rank"] = q(
        "Bottleneck rank r (low_rank fusion only; ignored otherwise)",
        8, int,
    )
    cfg["n_attn_layers"] = q(
        "Number of fusion layers (low_rank / linear_attn / mgta)",
        1, int,
    )
    cfg["n_heads"] = q(
        "Attention heads (linear_attn / mgta only; ignored otherwise)",
        4, int,
    )
    cfg["ffn_dim"] = q(
        "FFN hidden dim (mgta / linear_attn; 0 = auto 2x/4x embed_dim)",
        0, int,
    )
    cfg["head_hidden"] = q(
        "ClassHead MLP hidden widths (e.g. 64 32; empty = single Linear)",
        [64, 32], list_int,
    )
    cfg["activation"] = q(
        "Activation function",
        "gelu",
        choices=["relu", "gelu", "leaky_relu"],
    )
    cfg["dropout"] = q("Dropout rate", 0.1, float)

    # training
    section("Training hyperparameters")
    cfg["n_classes"]    = q("Number of output classes",        2,       int)
    cfg["num_training"] = q("Number of independent runs",      10,      int)
    cfg["lr"]           = q("Learning rate",                   2.55e-3, float)
    cfg["num_epochs"]   = q("Max epochs",                      900,     int)
    cfg["patience"]     = q("Early stopping patience",         20,      int)
    cfg["batch_size"]   = q("Batch size",                      128,     int)
    cfg["optimizer"]    = q(
        "Optimizer", "adam", choices=["adam", "adamw", "muon"],
    )
    cfg["muon_lr"] = q(
        "Muon learning rate (only used when optimizer=muon)",
        0.02, float,
    )

    if interactive:
        print("  NOTE: 'none' is recommended for both modes.")
        print("        GroupProjector has 5 per-group BatchNorm1d layers.")
        print("        External normalisation is redundant and disrupts point-cloud geometry.")
    cfg["norm_type"] = q(
        "Normalisation type",
        "none",
        choices=["none", "global", "per-channel", "per-timestep"],
    )

    cfg["test_size"]  = q("Test fraction",                       0.1,  float)
    cfg["val_size"]   = q("Validation fraction (of train+val)",  0.1,  float)
    cfg["verbose"]    = q("Verbose logging?",                    False, bool)
    cfg["reliability_threshold"] = q(
        "Reliability threshold (neutral zone cutoff)", 0.6, float,
    )

    # regularisation
    section("Regularisation")
    cfg["label_smoothing"] = q(
        "Label smoothing (0.0 = off; 0.1 recommended for calibration)",
        0.1, float,
    )
    cfg["grad_clip_norm"] = q(
        "Gradient clip max norm (0.0 = off, 1.0 recommended)",
        1.0, float,
    )

    # calibration
    section("Calibration")
    if interactive:
        print("  Temperature scaling fits a scalar T on the val set after")
        print("  training via LBFGS so confidence aligns with accuracy.")
        print("  T > 1 gives softer confidences. Adds about 1s per run.")
    cfg["use_temperature_scaling"] = q(
        "Fit temperature scaling after each run?",
        True, bool,
    )

    # robustness sweep
    section("Noise robustness sweep")
    if interactive:
        print("  After each training run, the model is evaluated on the")
        print("  test set with Gaussian noise added at each sigma level.")
        print("  Results saved to robustness_sweep_run{N}.csv and")
        print("  robustness_all_runs.csv. Set to [] to disable.")

    if interactive:
        raw_nl = input(
            "  Noise sigma levels (comma-separated floats, or Enter for default): "
        ).strip()
        if raw_nl:
            try:
                cfg["noise_levels"] = [float(v) for v in raw_nl.replace(",", " ").split()]
            except ValueError:
                print("    Could not parse, using default")
                cfg["noise_levels"] = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
        else:
            cfg["noise_levels"] = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    else:
        cfg["noise_levels"] = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

    cfg["noise_batch_size"] = q(
        "Batch size for robustness sweep forward passes",
        32, int,
    )

    # noise augmentation
    section("Noise augmentation")
    if interactive:
        print("  During training, a noisy copy of x_train is concatenated")
        print("  (sigma = noise_aug_sigma). Teaches GroupProjector BatchNorm")
        print("  what shifted feature distributions look like.")
        print("  0.0 = disabled.")
    cfg["noise_aug_sigma"] = q(
        "Training noise augmentation sigma (0.0 = off)",
        0.05, float,
    )

    return cfg


# comment annotations

COMMENTS = {
    "_file"  : "config.json, generated by create_config.py for TDAEnd2EndNet / CachedGroupMLP",
    "_usage" : "python main_train_time_series_tda_stat_summary.py --config config.json",

    "___S_PATHS___"  : "--- paths ---",
    "___S_REPRO___"  : "--- reproducibility and device ---",
    "___S_DATA___"   : "--- data preprocessing ---",
    "_exclude_states"    : "List of state names to drop before training.",
    "_keep_states"       : "Whitelist: only these states are kept. [] = all.",
    "_balance_strategy"  : "'none'|'undersample'(recommended)|'oversample'|'hybrid'.",

    "___S_TDA___"    : "--- TDA hyperparameters ---",
    "_takens_dim"    : "Takens embedding dimension. 2 = (x[t], x[t+tau]).",
    "_takens_delay"  : "Time delay tau (samples). Rule of thumb: first minimum of mutual information.",
    "_n_hom_dims"    : "Homology dims: 1=H0 only, 2=H0+H1. H1 (loops) discriminates chaos vs periodic.",
    "_n_betti_bins"  : "Betti curve resolution. More bins gives finer filtration scale but is slower.",
    "_n_pi_bins"     : "Persistence image grid size (n x n pixels). 15-20 is a good default.",
    "_pi_sigma"      : "Gaussian kernel bandwidth for PI. Smaller is sharper.",
    "_ph_workers"    : "Ripser parallel worker threads. CPUS_PER_TASK in SLURM = ph_workers+2.",
    "_precompute_batch_size" : "Segments per chunk during TDA pre-computation. Lower if out of memory.",

    "___S_CACHE___"  : "--- TDA feature cache ---",
    "_use_cache"     : "true = pre-compute enriched TDA features once (feat_dim=42 for n_hom=2), train CachedGroupMLP (about 100x faster). false = TDA in every forward() (TDAEnd2EndNet, slow but no pre-processing).",
    "_tda_cache_path": "Where to save/load the pre-computed feature cache (.pt file).",
    "_force_recompute": "Set true to delete and regenerate the cache (e.g. after changing TDA params).",

    "___S_CLIP___"   : "--- IQR feature clipping ---",
    "_clip_lo_pct"   : "Lower percentile bound for IQR clipping at precompute time. 1.0 = 1st percentile.",
    "_clip_hi_pct"   : "Upper percentile bound for IQR clipping. 99.0 = 99th percentile. Bounds are saved in cache and re-applied to noisy features in the robustness sweep.",

    "___S_MODEL___"  : "--- model architecture ---",
    "_embed_dim"     : "Shared token embedding dim D. Each of the 5 TDA feature groups is projected to this dim. Fusion output = 5*D.",
    "_fusion"        : "'bilinear' (about 330 params) | 'low_rank' (recommended) | 'linear_attn' | 'gated' (about 10k) | 'mgta' (full transformer).",
    "_rank"          : "Bottleneck rank r for low_rank fusion. params = 2*5*embed_dim*r. r=8 gives 2560 params.",
    "_n_attn_layers" : "Number of stacked fusion layers (low_rank, linear_attn, mgta).",
    "_n_heads"       : "Attention heads for linear_attn and mgta. Must divide embed_dim.",
    "_ffn_dim"       : "FFN hidden dim for mgta / linear_attn. 0 = auto (2x or 4x embed_dim).",
    "_head_hidden"   : "ClassHead MLP hidden widths after fusion. [64,32] gives two hidden layers.",
    "_activation"    : "'gelu' (recommended) | 'relu' | 'leaky_relu'.",
    "_dropout"       : "Dropout in GroupProjector, fusion layers, and ClassHead.",

    "___S_TRAIN___"  : "--- training ---",
    "_norm_type"     : "'none' recommended. GroupProjector has 5 per-group BatchNorm1d layers. External normalisation disrupts point-cloud geometry.",
    "_optimizer"     : "'adam' | 'adamw' | 'muon'. Muon uses a separate lr (muon_lr) for hidden weight matrices.",
    "_muon_lr"       : "Muon learning rate for hidden weight matrices. Only active when optimizer=muon.",
    "_reliability_threshold" : "Entropy reliability threshold in [0,1]. Samples below this are the neutral zone.",

    "___S_REG___"    : "--- regularisation ---",
    "_label_smoothing": "CrossEntropyLoss label smoothing. 0.1 recommended (raised from an earlier default of 0.05); reduces overconfident logits and improves ECE.",
    "_grad_clip_norm" : "Gradient clipping max norm. 1.0 recommended.",

    "___S_CAL___"    : "--- calibration ---",
    "_use_temperature_scaling" : "Fit a scalar temperature T on the val set after each run. T>1 gives softer confidences, closer to accuracy. A cheap post-hoc step.",

    "___S_ROB___"    : "--- noise robustness sweep ---",
    "_noise_levels"  : "List of Gaussian noise sigma values for the post-training robustness sweep. [] or null disables it. Results in robustness_sweep_run{N}.csv and robustness_all_runs.csv.",
    "_noise_batch_size" : "Batch size for robustness sweep forward passes. Keep small for end2end mode, since TDA is costly per sample.",

    "___S_AUG___"    : "--- noise augmentation ---",
    "_noise_aug_sigma" : "Sigma of Gaussian noise added to a copy of x_train during training. Teaches GroupProjector BatchNorm what shifted feature distributions look like, closing the accuracy cliff at low sigma. 0.0 = disabled.",
}


def annotated_dict(cfg: dict) -> dict:
    """Interleave comment keys into config dict for readability in JSON."""
    out = {}
    out["_file"]  = COMMENTS["_file"]
    out["_usage"] = COMMENTS["_usage"]

    out["___SECTION_PATHS___"] = COMMENTS["___S_PATHS___"]
    for k in ["path_data", "h5_filename", "path_save"]:
        if k in cfg: out[k] = cfg[k]

    out["___SECTION_REPRO___"] = COMMENTS["___S_REPRO___"]
    for k in ["random_seed", "force_cpu"]:
        if k in cfg: out[k] = cfg[k]

    out["___SECTION_DATA___"] = COMMENTS["___S_DATA___"]
    for k in ["length_series", "segmentation_duration", "num_channels",
              "exclude_states"]:
        if k in cfg: out[k] = cfg[k]
    out["_exclude_states"] = COMMENTS["_exclude_states"]
    if "keep_states"      in cfg: out["keep_states"]      = cfg["keep_states"]
    out["_keep_states"]    = COMMENTS["_keep_states"]
    if "balance_strategy" in cfg: out["balance_strategy"] = cfg["balance_strategy"]
    out["_balance_strategy"] = COMMENTS["_balance_strategy"]

    out["___SECTION_TDA___"] = COMMENTS["___S_TDA___"]
    for k in ["takens_dim", "takens_delay", "n_hom_dims",
              "n_betti_bins", "n_pi_bins", "pi_sigma",
              "ph_workers", "precompute_batch_size"]:
        if k in cfg:
            out[k] = cfg[k]
            if f"_{k}" in COMMENTS: out[f"_{k}"] = COMMENTS[f"_{k}"]

    out["___SECTION_CACHE___"] = COMMENTS["___S_CACHE___"]
    for k in ["use_cache", "tda_cache_path", "force_recompute"]:
        if k in cfg:
            out[k] = cfg[k]
            if f"_{k}" in COMMENTS: out[f"_{k}"] = COMMENTS[f"_{k}"]

    out["___SECTION_CLIP___"] = COMMENTS["___S_CLIP___"]
    for k in ["clip_lo_pct", "clip_hi_pct"]:
        if k in cfg:
            out[k] = cfg[k]
            if f"_{k}" in COMMENTS: out[f"_{k}"] = COMMENTS[f"_{k}"]

    out["___SECTION_MODEL___"] = COMMENTS["___S_MODEL___"]
    for k in ["embed_dim", "fusion", "rank", "n_attn_layers", "n_heads",
              "ffn_dim", "head_hidden", "activation", "dropout"]:
        if k in cfg:
            out[k] = cfg[k]
            if f"_{k}" in COMMENTS: out[f"_{k}"] = COMMENTS[f"_{k}"]

    out["___SECTION_TRAINING___"] = COMMENTS["___S_TRAIN___"]
    for k in ["n_classes", "num_training", "lr", "num_epochs", "patience",
              "batch_size", "optimizer", "muon_lr", "norm_type",
              "test_size", "val_size", "verbose", "reliability_threshold"]:
        if k in cfg:
            out[k] = cfg[k]
            if f"_{k}" in COMMENTS: out[f"_{k}"] = COMMENTS[f"_{k}"]

    out["___SECTION_REG___"] = COMMENTS["___S_REG___"]
    for k in ["label_smoothing", "grad_clip_norm"]:
        if k in cfg:
            out[k] = cfg[k]
            if f"_{k}" in COMMENTS: out[f"_{k}"] = COMMENTS[f"_{k}"]

    out["___SECTION_CALIBRATION___"] = COMMENTS["___S_CAL___"]
    if "use_temperature_scaling" in cfg:
        out["use_temperature_scaling"] = cfg["use_temperature_scaling"]
        out["_use_temperature_scaling"] = COMMENTS["_use_temperature_scaling"]

    out["___SECTION_ROBUSTNESS___"] = COMMENTS["___S_ROB___"]
    for k in ["noise_levels", "noise_batch_size"]:
        if k in cfg:
            out[k] = cfg[k]
            if f"_{k}" in COMMENTS: out[f"_{k}"] = COMMENTS[f"_{k}"]

    out["___SECTION_AUGMENTATION___"] = COMMENTS["___S_AUG___"]
    if "noise_aug_sigma" in cfg:
        out["noise_aug_sigma"] = cfg["noise_aug_sigma"]
        out["_noise_aug_sigma"] = COMMENTS["_noise_aug_sigma"]

    return out


# ============================================================================
def main():
    args        = parse_args()
    interactive = not args.no_interactive

    if interactive:
        print("=" * 58)
        print("   TDAEnd2EndNet config generator")
        print("   Press Enter to accept the default shown in [...]")
        print("=" * 58)

    cfg     = build_config(interactive)
    out     = annotated_dict(cfg)
    outpath = args.output

    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Config written -> {outpath}")
    print(f"{'='*55}")
    print("\nKey settings:")
    for k in ["path_data", "h5_filename", "keep_states", "balance_strategy",
              "use_cache", "takens_dim", "takens_delay", "n_hom_dims",
              "embed_dim", "fusion", "rank", "n_attn_layers",
              "head_hidden", "activation", "dropout",
              "label_smoothing", "grad_clip_norm",
              "clip_lo_pct", "clip_hi_pct",
              "use_temperature_scaling",
              "noise_levels", "noise_aug_sigma", "noise_batch_size",
              "n_classes", "num_training", "lr", "num_epochs",
              "batch_size", "optimizer", "muon_lr", "norm_type"]:
        if k in cfg:
            print(f"  {k:30s}: {cfg[k]}")
    print(f"\nRun training:")
    print(f"  python main_train_time_series_tda_stat_summary.py --config {outpath}")
    print(f"\nPrecompute cache only (optional, runs automatically otherwise):")
    print(f"  python precompute_tda.py --config {outpath}")


if __name__ == "__main__":
    main()

