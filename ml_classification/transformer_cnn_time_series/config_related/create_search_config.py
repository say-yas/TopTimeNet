"""
create_search_config.py
Generate search_config.json, which defines:
  - fixed (non-tuned) pipeline settings
  - the hyperparameter search space
  - search strategy settings

Usage
-----
    python create_search_config.py                          # writes ./search_config.json
    python create_search_config.py --out my_search.json     # custom output path
    python create_search_config.py --validate search_config.json

Then run:
    python hparam_search.py                                 # random search (default)
    python hparam_search.py --config search_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


# JSON encoder

class NumpyEncoder(json.JSONEncoder):
    """Make numpy scalar / array types JSON-serialisable."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# validation

# Types expected for every key in the "fixed" block.
_FIXED_REQUIRED: dict[str, type | tuple] = {
    "path_data":             str,
    "h5_filename":           str,
    "path_save":             str,
    "length_series":         int,
    "segmentation_duration": int,
    "exclude_states":        list,
    "force_cpu":             bool,
    "modeltype":             str,
    "num_channels":          int,
    "conv1d_kernel_size":    int,
    "test_size":             (int, float),
    "val_size":              (int, float),
    "num_epochs":            int,
    "optimizer":             str,
    "verbose":               bool,
    "num_training":          int,
    "random_seed":           int,
    "reliability_threshold": (int, float),
}

_FIXED_ALLOWED: dict[str, list] = {
    "modeltype": ["trans1", "cnn1"],
    "optimizer": ["adam", "adamW"],
}

# Valid sampling types and their required keys.
_SPACE_TYPES: dict[str, set[str]] = {
    "choice":       {"values"},
    "int_choice":   {"values"},
    "uniform":      {"low", "high"},
    "log_uniform":  {"low", "high"},
}


def _validate_search_config(cfg: dict) -> list[str]:
    """Return a list of error strings (empty = valid).

    Validates:
    * Top-level structure (fixed, search_space, strategy, n_trials, etc.).
    * All required fixed keys with correct types and allowed values.
    * Numeric constraints on fixed keys.
    * Each search-space entry has a recognised type and its required sub-keys.
    * CNN-specific keys present when ``modeltype`` is ``'cnn1'``.
    * ``embed_size`` / ``nhead_encoder`` divisibility when
      ``enforce_head_divisibility`` is True.

    Args:
        cfg: Parsed search config dict.

    Returns:
        List of human-readable error strings.
    """
    errors: list[str] = []

    # top-level keys
    for key in ("fixed", "search_space", "strategy", "n_trials", "objective_alpha"):
        if key not in cfg:
            errors.append(f"Missing top-level key: '{key}'")

    if errors:
        return errors   # can't proceed without basic structure

    fixed  = cfg["fixed"]
    space  = cfg["search_space"]

    # strategy
    if cfg.get("strategy") not in ("random", "grid"):
        errors.append(
            f"'strategy' must be 'random' or 'grid', got {cfg.get('strategy')!r}"
        )
    if not isinstance(cfg.get("n_trials"), int) or cfg["n_trials"] < 1:
        errors.append("'n_trials' must be a positive int")

    alpha = cfg.get("objective_alpha")
    if not isinstance(alpha, (int, float)) or not (0.0 <= alpha <= 1.0):
        errors.append("'objective_alpha' must be a float in [0, 1]")

    # fixed: presence and type
    for key, expected in _FIXED_REQUIRED.items():
        if key not in fixed:
            errors.append(f"fixed.'{key}' is missing")
            continue
        if not isinstance(fixed[key], expected):
            errors.append(
                f"fixed.'{key}' must be {expected}, "
                f"got {type(fixed[key]).__name__} ({fixed[key]!r})"
            )

    # fixed: allowed values
    for key, allowed in _FIXED_ALLOWED.items():
        if key in fixed and fixed[key] not in allowed:
            errors.append(
                f"fixed.'{key}' must be one of {allowed}, got {fixed[key]!r}"
            )

    # fixed: numeric constraints
    num_checks = [
        ("conv1d_kernel_size", lambda v: v % 2 == 1,
         "must be odd"),
        ("test_size",          lambda v: 0 < v < 1,  "must be in (0, 1)"),
        ("val_size",           lambda v: 0 < v < 1,  "must be in (0, 1)"),
        ("num_epochs",         lambda v: v >= 1,      "must be >= 1"),
        ("num_training",       lambda v: v >= 1,      "must be >= 1"),
        ("reliability_threshold", lambda v: 0 < v < 1, "must be in (0, 1)"),
        ("segmentation_duration", lambda v: v > 0,   "must be > 0"),
        ("length_series",      lambda v: v > 0,       "must be > 0"),
    ]
    for key, pred, msg in num_checks:
        if key in fixed:
            try:
                if not pred(fixed[key]):
                    errors.append(f"fixed.'{key}' {msg} (got {fixed[key]!r})")
            except Exception:
                pass

    # CNN-specific fixed keys required when modeltype == "cnn1"
    if fixed.get("modeltype") == "cnn1":
        cnn_keys = ("cnn_base_channels", "cnn_channel_multipliers", "cnn_pooling")
        for k in cnn_keys:
            if k not in fixed and k not in space:
                errors.append(
                    f"modeltype='cnn1' requires '{k}' in fixed or search_space"
                )
        if "cnn_channel_multipliers" in fixed:
            mults = fixed["cnn_channel_multipliers"]
            if not isinstance(mults, list) or len(mults) == 0:
                errors.append("fixed.'cnn_channel_multipliers' must be a non-empty list")
            elif not all(isinstance(m, int) and m > 0 for m in mults):
                errors.append(
                    "fixed.'cnn_channel_multipliers' must be a list of positive ints"
                )

    # search space entries
    if not isinstance(space, dict):
        errors.append("'search_space' must be a dict")
    else:
        for param, spec in space.items():
            if not isinstance(spec, dict):
                errors.append(f"search_space.'{param}' must be a dict")
                continue
            stype = spec.get("type")
            if stype not in _SPACE_TYPES:
                errors.append(
                    f"search_space.'{param}'.type must be one of "
                    f"{list(_SPACE_TYPES)}, got {stype!r}"
                )
                continue
            for sub_key in _SPACE_TYPES[stype]:
                if sub_key not in spec:
                    errors.append(
                        f"search_space.'{param}' (type={stype!r}) "
                        f"is missing required sub-key '{sub_key}'"
                    )
            if stype in ("uniform", "log_uniform"):
                lo, hi = spec.get("low"), spec.get("high")
                if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                    if lo >= hi:
                        errors.append(
                            f"search_space.'{param}': low ({lo}) must be < high ({hi})"
                        )

    # embed_size / nhead_encoder divisibility (when both are choices)
    if cfg.get("enforce_head_divisibility", False):
        embed_values = (
            space.get("embed_size", {}).get("values")
            or ([fixed["embed_size"]] if "embed_size" in fixed else None)
        )
        nhead_values = (
            space.get("nhead_encoder", {}).get("values")
            or ([fixed["nhead_encoder"]] if "nhead_encoder" in fixed else None)
        )
        if embed_values and nhead_values:
            bad_pairs = [
                (e, n) for e in embed_values for n in nhead_values
                if e % n != 0
            ]
            if bad_pairs:
                errors.append(
                    f"enforce_head_divisibility=True but these "
                    f"(embed_size, nhead) pairs are invalid: {bad_pairs[:5]}"
                    + (" ..." if len(bad_pairs) > 5 else "")
                )

    return errors


# search config definition

search_config: dict = {

    # fixed pipeline settings
    "fixed": {
        "path_data": (
            "/Users/shararehsayyad/Documents/Projects/"
            "topological_data_analysis/Computational_topology/"
            "dataset/extended_teaspoon_dataset/"
        ),
        "h5_filename":           "all_extended_teaspoon_datasets.h5",
        "path_save": (
            "/Users/shararehsayyad/Documents/Projects/"
            "topological_data_analysis/Computational_topology/"
            "ml_classification/test/hparam_search_results/"
        ),
        "length_series":          1200,
        "segmentation_duration":  300,
        "exclude_states":         ["default"],
        "force_cpu":              True,

        # model, shared fixed values (architecture params are in search_space below)
        "modeltype":              "trans1",   # "trans1" | "cnn1"
        "num_channels":           1,
        "conv1d_kernel_size":     3,          # must be odd, not tuned here

        # CNN fixed values (used when modeltype="cnn1")
        # Move any of these into search_space to tune them.
        "cnn_base_channels":        32,
        "cnn_channel_multipliers":  [1, 2, 4],
        "cnn_pooling":              "mean",   # "mean" | "max" | "last"

        # training fixed values
        "test_size":              0.20,
        "val_size":               0.20,
        "num_epochs":             100,
        "optimizer":              "adamW",    # "adam" | "adamW"
        "verbose":                False,
        "num_training":           1,          # >1 averages over multiple runs
        "random_seed":            43,
        "reliability_threshold":  0.6,
    },

    # hyperparameter search space
    # Sampling types:
    #   {"type": "choice",      "values": [...]}    pick one (any type)
    #   {"type": "int_choice",  "values": [...]}    pick one int
    #   {"type": "uniform",     "low": x, "high": y} float in [low, high)
    #   {"type": "log_uniform", "low": x, "high": y} float in log scale
    "search_space": {
        # optimisation
        "lr": {
            "type": "log_uniform", "low": 1e-5, "high": 1e-1,
        },
        "batch_size": {
            "type": "int_choice", "values": [16, 32, 64, 128],
        },
        "patience": {
            "type": "int_choice", "values": [5, 10, 20],
        },
        "norm_type": {
            "type": "choice",
            "values": ["per-channel", "per-timestep", "global"],
        },

        # Transformer architecture (trans1)
        "embed_size": {
            "type": "int_choice", "values": [16, 32, 64, 128],
        },
        "nhead_encoder": {
            "type": "int_choice", "values": [1, 2, 4, 8],
        },
        "dim_feedforward": {
            "type": "int_choice", "values": [16, 32, 64, 128, 256],
        },
        "num_encoderlayers": {
            "type": "int_choice", "values": [1, 2, 3],
        },
        "conv1d_emb": {
            "type": "choice", "values": [True, False],
        },

        # CNN architecture (cnn1), uncomment to tune
        # "cnn_base_channels":       {"type": "int_choice", "values": [16, 32, 64]},
        # "cnn_channel_multipliers": {"type": "choice",
        #                             "values": [[1, 2], [1, 2, 4], [1, 2, 4, 8]]},
        # "cnn_pooling":             {"type": "choice", "values": ["mean", "max", "last"]},

        # shared
        "dropout": {
            "type": "choice", "values": [0.0, 0.1],
        },
        "size_linear_layers": {
            "type": "int_choice", "values": [16, 32, 64, 128],
        },
    },

    # search strategy
    "strategy": "random",   # "random" | "grid"
    "n_trials": 30,         # number of random trials (ignored for grid search)

    # objective: combined score
    # score = alpha * accuracy + (1 - alpha) * reliability
    "objective_alpha": 0.7,

    # constraints
    # Skip trials where embed_size % nhead_encoder != 0
    "enforce_head_divisibility": True,
}


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or validate a hyperparameter search config."
    )
    parser.add_argument(
        "--out", type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "search_config.json"
        ),
        help="Output path for the config (default: ./search_config.json)",
    )
    parser.add_argument(
        "--validate", type=str, metavar="CONFIG",
        help="Validate an existing JSON search config and exit.",
    )
    return parser.parse_args()


# main

def main() -> None:
    args = parse_args()

    # validate-only mode
    if args.validate:
        path = args.validate
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}", file=sys.stderr)
            sys.exit(1)
        with open(path) as f:
            cfg = json.load(f)
        errors = _validate_search_config(cfg)
        if errors:
            print(f"Search config '{path}' has {len(errors)} error(s):")
            for e in errors:
                print(f"  x {e}")
            sys.exit(1)
        else:
            print(f"Search config '{path}' is valid")
            sys.exit(0)

    # validate the in-memory config before writing
    errors = _validate_search_config(search_config)
    if errors:
        print(f"Search config has {len(errors)} error(s), fix before writing:\n")
        for e in errors:
            print(f"  x {e}")
        sys.exit(1)

    # write
    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(search_config, f, indent=4, cls=NumpyEncoder)

    fixed  = search_config["fixed"]
    space  = search_config["search_space"]
    alpha  = search_config["objective_alpha"]

    print(f"Search config written to: {out_path}")
    print(f"  Strategy   : {search_config['strategy']}")
    print(f"  Trials     : {search_config['n_trials']}")
    print(f"  Modeltype  : {fixed['modeltype']}")
    print(f"  Objective  : {alpha} * accuracy + {1 - alpha:.2f} * reliability")
    print(f"  Tuned params ({len(space)}): {list(space.keys())}")
    print(f"  Head divisibility enforced: {search_config['enforce_head_divisibility']}")


if __name__ == "__main__":
    main()
