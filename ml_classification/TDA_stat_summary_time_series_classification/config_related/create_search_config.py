"""
create_search_config.py
Run once to generate search_config.json, which defines:
  - the fixed (non-tuned) pipeline settings
  - the hyperparameter search space for TDAEnd2EndNet / CachedGroupMLP
  - search strategy settings

Then run:
    python hparam_search_tda.py
    python hparam_search_tda.py --config search_config.json
"""

import json
import os

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Make numpy scalar/array types JSON-serialisable."""
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


search_config = {

    # Fixed pipeline settings: edit these to match your environment. Never
    # sampled; passed verbatim to run_tda_stat_summary_training() /
    # run_cached_tda_training() as the cfg= dict, and as fallbacks for any
    # search-space key not present in a sampled trial.
    "fixed": {

        # paths
        "path_data"   : "/path/to/your/data/",
        "h5_filename" : "your_data.h5",
        "path_save"   : "./results/hparam_search/",

        # reproducibility
        "random_seed" : 42,
        "force_cpu"   : False,

        # data
        "length_series"         : 100_000,
        "segmentation_duration" : 500,
        "num_channels"          : 1,
        "exclude_states"        : [],
        "keep_states"           : ["periodic", "chaotic"],
        "balance_strategy"      : "undersample",

        # cache
        # feat_dim=42 for n_hom=2:
        #   [pc_stats(4) | entropy(2) | lifetime(10) | betti(12) | pi(14)]
        # set use_cache=false if TDA params are in search_space (slow but correct)
        "use_cache"       : True,
        "tda_cache_path"  : "./results/hparam_search/tda_cache.pt",
        "force_recompute" : False,

        # TDA fixed params
        "ph_workers"            : 4,
        "precompute_batch_size" : 64,
        "n_hom_dims"            : 2,

        # training fixed params
        "num_epochs"            : 900,
        "num_training"          : 1,    # 1 run per trial for speed
        "test_size"             : 0.10,
        "val_size"              : 0.10,
        "verbose"               : False,
        "reliability_threshold" : 0.6,

        # architecture fallback defaults
        "embed_dim"     : 32,
        "fusion"        : "low_rank",
        "rank"          : 8,
        "n_attn_layers" : 1,
        "n_heads"       : 4,
        "ffn_dim"       : 0,
        "head_hidden"   : [64, 32],
        "activation"    : "gelu",
        "dropout"       : 0.1,
        "norm_type"     : "none",

        # regularisation fallback defaults
        "label_smoothing" : 0.1,
        "grad_clip_norm"  : 1.0,

        # optimizer
        # muon_lr is forwarded to both runners; only active when
        # optimizer="muon". Defaults to adam during search for stability.
        "optimizer" : "adam",
        "muon_lr"   : 0.02,

        # IQR clipping
        # Applied at precompute time and re-applied to noisy features in
        # the robustness sweep. Not a hyperparameter; keep fixed.
        "clip_lo_pct" : 1.0,
        "clip_hi_pct" : 99.0,

        # noise augmentation
        # Disabled during search (0.0) so each trial evaluates the clean
        # model without augmentation overhead. The best-model training
        # script (submit_best_training.sh) re-enables it via a config patch.
        "noise_aug_sigma" : 0.0,

        # temperature scaling
        # Disabled during search: post-hoc fitting adds about 1s per run
        # and does not affect which config wins (T scales confidence, not
        # accuracy). Enabled in best-model training.
        "use_temperature_scaling" : False,

        # robustness sweep
        # Null during search for speed. submit_best_training.sh injects
        # the real noise_levels list into each per-task config JSON.
        "noise_levels"    : None,
        "noise_batch_size": 32,
    },

    # Search space. Each entry is one of:
    #   {"type": "choice",      "values": [...]}      pick one from list
    #   {"type": "int_choice",  "values": [...]}      pick one int
    #   {"type": "log_uniform", "low": x, "high": y}   float, log scale
    #   {"type": "uniform",     "low": x, "high": y}   float, linear scale
    "search_space": {

        # training
        "lr": {
            "type": "log_uniform", "low": 1e-5, "high": 1e-2,
        },
        "batch_size": {
            "type": "int_choice", "values": [32, 64, 128, 256],
        },
        "optimizer": {
            "type": "choice", "values": ["adam", "adamw"],
        },
        "patience": {
            "type": "int_choice", "values": [10, 20, 30],
        },

        "norm_type": {
            "type"  : "choice",
            "values": ["none", "global", "per-channel"],
        },

        # regularisation
        "label_smoothing": {
            "type"  : "choice",
            "values": [0.0, 0.05, 0.1],
        },
        "grad_clip_norm": {
            "type"  : "choice",
            "values": [0.5, 1.0, 2.0],
        },

        # TDA
        # WARNING: if use_cache=true in "fixed", changing these has no
        # effect on the cached features. Either remove them from
        # search_space, or set use_cache=false (slow).
        "takens_dim": {
            "type": "int_choice", "values": [2, 3],
        },
        "takens_delay": {
            "type"  : "int_choice",
            "values": [5, 10, 15, 20],
        },
        "n_betti_bins": {
            "type": "int_choice", "values": [30, 50, 75],
        },
        "n_pi_bins": {
            "type": "int_choice", "values": [10, 15, 20],
        },
        "pi_sigma": {
            "type"  : "choice",
            "values": [0.005, 0.05, 0.1],
        },

        # model architecture
        "embed_dim": {
            "type"  : "int_choice",
            "values": [16, 32, 64],
        },
        "fusion": {
            "type"  : "choice",
            "values": ["bilinear", "low_rank", "linear_attn", "gated", "mgta"],
        },
        "rank": {
            "type"  : "int_choice",
            "values": [4, 8, 16],
            # Bottleneck rank for low_rank fusion only.
        },
        "n_attn_layers": {
            "type"  : "int_choice",
            "values": [1, 2],
        },
        "n_heads": {
            "type"  : "int_choice",
            "values": [2, 4, 8],
            # constraint embed_dim % n_heads == 0, enforced only for
            # fusion in ["linear_attn", "mgta"]
        },
        "head_hidden": {
            "type"  : "choice",
            "values": [
                [],
                [32],
                [64, 32],
                [128, 64],
                [64, 32, 16],
            ],
        },
        "activation": {
            "type"  : "choice",
            "values": ["relu", "gelu", "leaky_relu"],
        },
        "dropout": {
            "type"  : "choice",
            "values": [0.0, 0.05, 0.1, 0.2],
        },
    },

    # Search strategy
    "strategy" : "random",   # "random" | "grid"
    "n_trials" : 50,

    # Objective: combined score = alpha * f1 + (1-alpha) * gmean
    "objective_alpha": 0.6,

    # Constraints enforced before each trial is run
    # embed_dim % n_heads == 0, for linear_attn and mgta only.
    "enforce_head_divisibility": True,
}


# write
out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "search_config.json"
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w") as f:
    json.dump(search_config, f, indent=4, cls=NumpyEncoder)


# summary
n_tunable = len(search_config["search_space"])
fixed     = search_config["fixed"]
ss        = search_config["search_space"]

print(f"Search config written -> {out_path}")
print(f"\n  Strategy   : {search_config['strategy']}")
print(f"  Trials     : {search_config['n_trials']}")
print(f"  Objective  : {search_config['objective_alpha']:.1f} x F1  +  "
      f"{1 - search_config['objective_alpha']:.1f} x G-mean")

# split tunable params into sections for readability
arch_keys  = {"embed_dim", "fusion", "rank", "n_attn_layers",
              "n_heads", "ffn_dim", "head_hidden", "activation", "dropout"}
reg_keys   = {"label_smoothing", "grad_clip_norm"}
tda_keys   = {"takens_dim", "takens_delay", "n_betti_bins", "n_pi_bins", "pi_sigma"}
train_keys = {"lr", "batch_size", "optimizer", "patience", "norm_type"}

def _print_section(label, keys):
    items = [(n, s) for n, s in ss.items() if n in keys]
    if not items:
        return
    print(f"\n  {label}:")
    for name, spec in items:
        if spec["type"] in ("choice", "int_choice"):
            print(f"    {name:25s}: {spec['values']}")
        else:
            print(f"    {name:25s}: [{spec['low']:.2e}, {spec['high']:.2e}]"
                  f"  ({spec['type']})")

print(f"\n  Params to tune ({n_tunable}):")
_print_section("Architecture", arch_keys)
_print_section("Regularisation", reg_keys)
_print_section("TDA", tda_keys)
_print_section("Training", train_keys)

print(f"\n  Fixed settings (search disabled, best-model training enables these):")
for k, label in [
    ("noise_aug_sigma",        "noise augmentation sigma"),
    ("use_temperature_scaling","temperature scaling"),
    ("noise_levels",           "robustness sweep"),
    ("clip_lo_pct",            "IQR clip lo percentile"),
    ("clip_hi_pct",            "IQR clip hi percentile"),
    ("label_smoothing",        "label smoothing (fallback)"),
    ("muon_lr",                "muon learning rate"),
]:
    if k in fixed:
        print(f"    {label:40s}: {fixed[k]}")

print(f"\n  Other fixed settings:")
for k in ["keep_states", "balance_strategy", "use_cache",
          "length_series", "segmentation_duration",
          "num_epochs", "num_training", "test_size", "val_size",
          "n_hom_dims", "embed_dim", "fusion", "rank",
          "norm_type", "grad_clip_norm", "reliability_threshold"]:
    if k in fixed:
        print(f"    {k:25s}: {fixed[k]}")

# cache-vs-TDA-search warning
if fixed.get("use_cache") and any(
    k in ss for k in ["takens_dim", "takens_delay", "n_betti_bins",
                      "n_pi_bins", "pi_sigma"]
):
    print(
        "\n  WARNING: use_cache=True but TDA params are in search_space.\n"
        "     The cache will NOT change between trials; TDA params\n"
        "     in search_space will have no effect on the features.\n"
        "     Options:\n"
        "       (a) Set use_cache=False in 'fixed' to recompute TDA each trial\n"
        "           (slow but correct if you want to tune TDA params).\n"
        "       (b) Remove TDA params from search_space and fix them in\n"
        "           'fixed' to use a single cached feature set (fast)."
    )

print(f"\nNext step:")
print(f"  python hparam_search_tda.py --config {out_path}")
