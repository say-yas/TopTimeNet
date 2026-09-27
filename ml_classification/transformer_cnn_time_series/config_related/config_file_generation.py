"""
create_config.py
Generate config.json with all tunable parameters for train.py.

Usage
-----
    python create_config.py                        # writes ./config.json
    python create_config.py --out my_config.json   # custom output path
    python create_config.py --validate             # validate an existing config

Edit the values in the ``config`` dict below, then re-run to overwrite.
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


# required keys and their expected Python types
# Used by _validate_config to catch typos and missing keys early.

_REQUIRED: dict[str, type | tuple] = {
    # paths
    "path_data":                str,
    "h5_filename":              str,
    "path_save":                str,
    # data
    "length_series":            int,
    "segmentation_duration":    int,
    "exclude_states":           list,
    # device
    "force_cpu":                bool,
    # model
    "modeltype":                str,
    "num_channels":             int,
    "embed_size":               int,
    "nhead_encoder":            int,
    "dim_feedforward":          int,
    "num_encoderlayers":        int,
    "dropout":                  (int, float),
    "conv1d_emb":               bool,
    "conv1d_kernel_size":       int,
    "size_linear_layers":       int,
    # CNN-specific (required even for trans1 so the key always exists)
    "cnn_base_channels":        int,
    "cnn_channel_multipliers":  list,
    "cnn_pooling":              str,
    # training
    "lr":                       (int, float),
    "batch_size":               int,
    "num_epochs":               int,
    "patience":                 int,
    "optimizer":                str,
    "norm_type":                str,
    "test_size":                (int, float),
    "val_size":                 (int, float),
    "num_training":             int,
    "verbose":                  bool,
    "random_seed":              int,
    # reliability
    "reliability_threshold":    (int, float),
}

_ALLOWED: dict[str, list] = {
    "modeltype":   ["trans1", "cnn1"],
    "optimizer":   ["adam", "adamW"],
    "norm_type":   ["global", "per-channel", "per-timestep"],
    "cnn_pooling": ["mean", "max", "last"],
}


def _validate_config(cfg: dict) -> list[str]:
    """Return a list of validation error strings (empty = valid).

    Checks:
    * All required keys are present.
    * Values have the expected Python type.
    * Enumerated fields contain a recognised value.
    * Common numeric constraints (odd kernel, positive patience, etc.).

    Args:
        cfg: Config dict to validate.

    Returns:
        List of human-readable error strings; empty list means no errors.
    """
    errors: list[str] = []

    # presence and type
    for key, expected_type in _REQUIRED.items():
        if key not in cfg:
            errors.append(f"Missing required key: '{key}'")
            continue
        if not isinstance(cfg[key], expected_type):
            errors.append(
                f"'{key}' must be {expected_type}, "
                f"got {type(cfg[key]).__name__} ({cfg[key]!r})"
            )

    # enumerated values
    for key, allowed in _ALLOWED.items():
        if key in cfg and cfg[key] not in allowed:
            errors.append(
                f"'{key}' must be one of {allowed}, got {cfg[key]!r}"
            )

    # numeric constraints
    checks = [
        ("conv1d_kernel_size", lambda v: v % 2 == 1,
         "must be odd (to preserve sequence length)"),
        ("nhead_encoder",      lambda v: v > 0,   "must be > 0"),
        ("embed_size",         lambda v: v > 0,   "must be > 0"),
        ("embed_size",
         lambda v: cfg.get("nhead_encoder", 1) > 0
                   and v % cfg.get("nhead_encoder", 1) == 0,
         "must be divisible by nhead_encoder"),
        ("test_size",  lambda v: 0 < v < 1,  "must be in (0, 1)"),
        ("val_size",   lambda v: 0 < v < 1,  "must be in (0, 1)"),
        ("dropout",    lambda v: 0 <= v < 1, "must be in [0, 1)"),
        ("patience",   lambda v: v >= 1,      "must be >= 1"),
        ("num_training", lambda v: v >= 1,    "must be >= 1"),
        ("reliability_threshold", lambda v: 0 < v < 1, "must be in (0, 1)"),
        ("segmentation_duration", lambda v: v > 0, "must be > 0"),
        ("length_series", lambda v: v > 0,    "must be > 0"),
    ]
    for key, predicate, msg in checks:
        if key in cfg:
            try:
                if not predicate(cfg[key]):
                    errors.append(f"'{key}' {msg} (got {cfg[key]!r})")
            except Exception:
                pass   # type errors already caught above

    # CNN multipliers must be non-empty positive ints
    if "cnn_channel_multipliers" in cfg:
        mults = cfg["cnn_channel_multipliers"]
        if isinstance(mults, list):
            if len(mults) == 0:
                errors.append("'cnn_channel_multipliers' must not be empty")
            elif not all(isinstance(m, int) and m > 0 for m in mults):
                errors.append(
                    "'cnn_channel_multipliers' must be a list of positive ints"
                )

    return errors


# config definition

config: dict = {
    # paths
    "path_data": (
        "/Users/shararehsayyad/Documents/Projects/topological_data_analysis/"
        "Computational_topology/dataset/extended_teaspoon_dataset/"
    ),
    "h5_filename": "all_extended_teaspoon_datasets.h5",
    "path_save": (
        "/Users/shararehsayyad/Documents/Projects/topological_data_analysis/"
        "Computational_topology/ml_classification/test/"
    ),

    # data preprocessing
    "length_series":          100_000,   # pad/truncate length before segmentation
    "segmentation_duration":  350,       # window length in samples
    "exclude_states":         ["default"],

    # device
    "force_cpu": True,

    # model, shared
    "modeltype":           "trans1",     # "trans1" | "cnn1"
    "num_channels":        1,
    "conv1d_kernel_size":  3,            # must be odd; used by both models
    "size_linear_layers":  16,
    "dropout":             0.0,

    # model, TransformerI (trans1)
    "embed_size":          32,           # must be divisible by nhead_encoder
    "nhead_encoder":       4,
    "dim_feedforward":     16,
    "num_encoderlayers":   1,
    "conv1d_emb":          True,

    # model, CNNI (cnn1)
    "cnn_base_channels":        32,
    "cnn_channel_multipliers":  [1, 2, 4],
    "cnn_pooling":              "mean",  # "mean" | "max" | "last"

    # training
    "lr":           0.0001,
    "batch_size":   64,
    "num_epochs":   5,
    "patience":     5,
    "optimizer":    "adamW",             # "adam" | "adamW"
    "norm_type":    "per-timestep",      # "global" | "per-channel" | "per-timestep"
    "test_size":    0.20,
    "val_size":     0.20,
    "num_training": 1,
    "verbose":      True,
    "random_seed":  43,

    # reliability
    "reliability_threshold": 0.6,
}


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or validate a training config file."
    )
    parser.add_argument(
        "--out", type=str,
        default=os.path.join(os.path.dirname(__file__), "config.json"),
        help="Output path for the config (default: ./config.json)",
    )
    parser.add_argument(
        "--validate", type=str, metavar="CONFIG",
        help="Validate an existing JSON config file and exit.",
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
        errors = _validate_config(cfg)
        if errors:
            print(f"Config '{path}' has {len(errors)} error(s):")
            for e in errors:
                print(f"  x {e}")
            sys.exit(1)
        else:
            print(f"Config '{path}' is valid")
            sys.exit(0)

    # validate the in-memory config before writing
    errors = _validate_config(config)
    if errors:
        print(f"Config has {len(errors)} error(s), fix them before writing:\n")
        for e in errors:
            print(f"  x {e}")
        sys.exit(1)

    # write
    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(config, f, indent=4, cls=NumpyEncoder)

    print(f"Config written to: {out_path}\n")
    for k, v in config.items():
        print(f"  {k:28s}: {v}")


if __name__ == "__main__":
    main()
