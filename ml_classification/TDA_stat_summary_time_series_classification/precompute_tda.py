"""
precompute_tda.py
Run once before training to compute and cache the full enriched TDA
feature vector for every segment in the dataset.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

# single source of truth: all TDA layers come from the merged model file
from ml_classification.TDA_stat_summary_time_series_classification.nn_tda_stat_summary_model import (
    TakensLayer,
    PointCloudStatsLayer,
    RipserPHLayer,
    PersistenceEntropyLayer,
    LifetimeStatsLayer,
    BettiCurveLayer,
    PILayer,
    TDABettiExtractor,
    TDAPIExtractor,
)


# ============================================================================
# Layout helpers
# ============================================================================

def _build_layout(n_hom: int) -> dict:
    """
    Derive slice layout directly from the layer classes so it can never go
    out of sync with the model code.

    For n_hom=2:
      pc_stats    : 4      (always fixed)
      entropy     : n_hom  (1 per hom dim)
      lifetime    : n_hom * LifetimeStatsLayer._S     = 10
      betti_stats : n_hom * TDABettiExtractor._S      = 12
      pi_stats    : n_hom * TDAPIExtractor._S         = 14
      TOTAL                                            = 42
    """
    pc_dim  = 4
    ent_dim = n_hom
    lt_dim  = n_hom * LifetimeStatsLayer._S      # 5 per dim
    b_dim   = n_hom * TDABettiExtractor._S       # 6 per dim
    p_dim   = n_hom * TDAPIExtractor._S          # 7 per dim
    starts  = np.cumsum([0, pc_dim, ent_dim, lt_dim, b_dim, p_dim])
    return {
        "pc_stats"    : (int(starts[0]), int(starts[1])),
        "entropy"     : (int(starts[1]), int(starts[2])),
        "lifetime"    : (int(starts[2]), int(starts[3])),
        "betti_stats" : (int(starts[3]), int(starts[4])),
        "pi_stats"    : (int(starts[4]), int(starts[5])),
    }


def _print_layout(layout: dict) -> None:
    print("  Feature layout:")
    for name, (s, e) in layout.items():
        print(f"    [{s:>3}:{e:>3}]  {name}  ({e-s} dims)")


# ============================================================================
# IQR clipping helper
# ============================================================================

def compute_clip_bounds(
    features: torch.Tensor,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-feature percentile clip bounds from a feature matrix.

    Args
    ----
    features : (N, feat_dim) float tensor
    lo_pct   : lower percentile (default 1.0, i.e. 1st percentile)
    hi_pct   : upper percentile (default 99.0, i.e. 99th percentile)

    Returns
    -------
    q_lo : (feat_dim,) tensor, lower clip bound per feature
    q_hi : (feat_dim,) tensor, upper clip bound per feature
    """
    q_lo = torch.quantile(features, lo_pct / 100.0, dim=0)
    q_hi = torch.quantile(features, hi_pct / 100.0, dim=0)
    # safety: ensure q_lo < q_hi (degenerate features with zero variance)
    degenerate = q_lo >= q_hi
    if degenerate.any():
        n_deg = degenerate.sum().item()
        print(f"  [clip] {n_deg} degenerate features (q_lo >= q_hi), bounds widened by +/-1e-3")
        q_lo[degenerate] = q_lo[degenerate] - 1e-3
        q_hi[degenerate] = q_hi[degenerate] + 1e-3
    return q_lo, q_hi


def apply_clip_bounds(
    features: torch.Tensor,
    q_lo: torch.Tensor,
    q_hi: torch.Tensor,
) -> torch.Tensor:
    """
    Apply pre-computed per-feature clip bounds to a feature matrix.
    Safe to call on both training and test/noisy features.

    Args
    ----
    features : (N, feat_dim)
    q_lo     : (feat_dim,)
    q_hi     : (feat_dim,)

    Returns
    -------
    clipped  : (N, feat_dim), values clamped to [q_lo, q_hi]
    """
    return features.clamp(min=q_lo, max=q_hi)


# ============================================================================
# Main precomputation function
# ============================================================================

def precompute_and_cache(
    X_seg:      torch.Tensor,
    cfg:        dict,
    cache_path: str  = "tda_cache.pt",
    batch_size: Optional[int] = None,
    force:      bool = False,
    y_seg:      Optional[torch.Tensor] = None,
    clip_lo_pct: float = 1.0,
    clip_hi_pct: float = 99.0,
) -> torch.Tensor:
    """
    Compute TDA features for every segment in X_seg and cache to disk.

    After computing all features, per-feature IQR clipping is applied:
    features are clamped to [clip_lo_pct, clip_hi_pct] percentile bounds.
    The bounds are saved in the cache payload under the keys "clip_q_lo"
    and "clip_q_hi", so downstream code (training and the robustness
    sweep) can apply the identical transformation to any new or noisy
    feature vectors. Clipping is applied before saving. Existing caches
    without clip bounds are still loadable; load_cache() warns if bounds
    are absent.

    Args
    ----
    X_seg        : (N, 1, T) float tensor of segmented time series
    cfg          : config dict (from JSON)
    cache_path   : where to write the .pt cache
    batch_size   : override precompute_batch_size from cfg
    force        : recompute even if cache_path already exists
    y_seg        : (N,) int tensor of labels (stored in cache, not used here)
    clip_lo_pct  : lower percentile for IQR clipping (default 1.0)
    clip_hi_pct  : upper percentile for IQR clipping (default 99.0)

    Returns
    -------
    all_feats : (N, feat_dim) clipped feature tensor
    """
    import os

    if not force and os.path.exists(cache_path):
        print(f"[precompute_tda] Loading existing cache: {cache_path}")
        payload  = torch.load(cache_path, weights_only=True)
        features = payload["features"]
        print(f"  Loaded {tuple(features.shape)}  feat_dim={payload['feat_dim']}")
        _print_layout(payload["layout"])
        if "clip_q_lo" in payload:
            print(f"  Clip bounds present: lo/hi percentiles saved in cache")
        else:
            print(f"  WARNING: no clip bounds in this cache, consider force=True to rebuild")
        return features

    n_hom      = cfg.get("n_hom_dims",                2)
    takens_dim = cfg.get("takens_dim",                 2)
    delay      = cfg.get("takens_delay",               5)
    n_b_bins   = cfg.get("n_betti_bins",              50)
    n_pi_bins  = cfg.get("n_pi_bins",                 20)
    pi_sigma   = cfg.get("pi_sigma",                 0.1)
    ph_workers = cfg.get("ph_workers",                 4)
    batch_sz   = batch_size or cfg.get("precompute_batch_size", 64)
    dev        = torch.device("cpu")

    # layout derived from live layer classes, never hard-coded
    layout   = _build_layout(n_hom)
    feat_dim = layout["pi_stats"][1]   # last slice end = total dim

    print(f"[precompute_tda] n_hom={n_hom}  feat_dim={feat_dim}  "
          f"batch_size={batch_sz}  N={len(X_seg)}")
    _print_layout(layout)

    # instantiate all non-parametric TDA layers
    takens     = TakensLayer(dim=takens_dim, delay=delay)
    pc_stats_l = PointCloudStatsLayer()
    ph         = RipserPHLayer(maxdim=n_hom - 1, max_workers=ph_workers)
    ph_ent_l   = PersistenceEntropyLayer(n_hom_dims=n_hom)
    lt_stats_l = LifetimeStatsLayer(n_hom_dims=n_hom)
    betti_l    = BettiCurveLayer(n_hom_dims=n_hom, n_bins=n_b_bins)
    pi_l       = PILayer(n_hom_dims=n_hom, n_pi_bins=n_pi_bins, sigma=pi_sigma)
    b_ext      = TDABettiExtractor(n_hom_dims=n_hom, n_betti_bins=n_b_bins)
    p_ext      = TDAPIExtractor(n_hom_dims=n_hom, n_pi_bins=n_pi_bins)

    # Warm-up: fit the PI imager using the exact same pipeline as the main
    # loop (including diameter-normalised relative Betti filtration), so
    # the imager grid is fitted on diagrams from the same filtration range
    # used for all cached features.
    print("[precompute_tda] Warming up PI imager on first batch...")
    warmup_x = X_seg[: min(batch_sz, len(X_seg))]
    with torch.no_grad():
        pcs_w    = takens(warmup_x)
        pc_w     = pc_stats_l(pcs_w, dev)
        diams_w  = pc_w[:, 0].cpu().numpy()
        dgms_w   = ph(pcs_w)
        _        = betti_l(dgms_w, dev, diameters=diams_w)   # relative filtration
        _        = pi_l(dgms_w, dev)                          # fits imager
    print("  PI imager fitted.")

    # main loop
    all_feats = []
    N         = len(X_seg)
    t0        = time.time()

    for i in range(0, N, batch_sz):
        x = X_seg[i: i + batch_sz]

        with torch.no_grad():
            pcs       = takens(x)
            pc_stat_t = pc_stats_l(pcs, dev)
            diameters = pc_stat_t[:, 0].cpu().numpy()   # for relative Betti
            diagrams  = ph(pcs)
            ent_t     = ph_ent_l(diagrams, dev)
            lt_t      = lt_stats_l(diagrams, dev)
            betti_t   = betti_l(diagrams, dev, diameters=diameters)
            pi_t      = pi_l(diagrams, dev)
            b_stats   = b_ext(betti_t)
            p_stats   = p_ext(pi_t)
            feats     = torch.cat(
                [pc_stat_t, ent_t, lt_t, b_stats, p_stats], dim=1
            )

        all_feats.append(feats.cpu())

        done    = min(i + batch_sz, N)
        elapsed = time.time() - t0
        eta     = elapsed / done * (N - done) if done > 0 else 0
        print(
            f"  {done:>6}/{N}  "
            f"[{'#' * int(30 * done / N):<30}]  "
            f"elapsed {elapsed:5.0f}s  ETA {eta:5.0f}s",
            end="\r",
        )

    print()

    all_feats = torch.cat(all_feats, dim=0)

    # assertion uses derived feat_dim, never a hard-coded number
    assert all_feats.shape == (N, feat_dim), (
        f"Shape mismatch: got {tuple(all_feats.shape)}, expected ({N}, {feat_dim}). "
        f"Check that LifetimeStatsLayer._S={LifetimeStatsLayer._S}, "
        f"TDABettiExtractor._S={TDABettiExtractor._S}, "
        f"TDAPIExtractor._S={TDAPIExtractor._S} match expectations."
    )

    # IQR clipping
    print(f"\n[precompute_tda] Computing IQR clip bounds "
          f"(lo={clip_lo_pct}%  hi={clip_hi_pct}%) ...")
    q_lo, q_hi = compute_clip_bounds(all_feats, lo_pct=clip_lo_pct, hi_pct=clip_hi_pct)
    all_feats_clipped = apply_clip_bounds(all_feats, q_lo, q_hi)

    # report how many values were actually clipped
    n_clipped = (all_feats != all_feats_clipped).sum().item()
    frac      = n_clipped / all_feats.numel() * 100
    print(f"  Clipped {n_clipped:,} values ({frac:.2f}% of all feature entries)")

    all_feats = all_feats_clipped

    # cfg_snapshot records everything that affects cached values, plus
    # model-architecture keys (fusion etc.) for self-documentation, even
    # though those don't affect precomputation itself
    _cfg_feature_keys = (
        "n_hom_dims", "takens_dim", "takens_delay",
        "n_betti_bins", "n_pi_bins", "pi_sigma", "ph_workers",
    )
    _cfg_model_keys = (
        "embed_dim", "fusion", "rank", "n_attn_layers", "n_heads",
        "dropout", "head_hidden", "activation",
    )
    cfg_snapshot = {k: cfg[k] for k in _cfg_feature_keys if k in cfg}
    cfg_snapshot["_model_keys"] = {k: cfg[k] for k in _cfg_model_keys if k in cfg}
    cfg_snapshot["_layer_dims"] = {
        "LifetimeStatsLayer._S" : LifetimeStatsLayer._S,
        "TDABettiExtractor._S"  : TDABettiExtractor._S,
        "TDAPIExtractor._S"     : TDAPIExtractor._S,
    }
    cfg_snapshot["_clip"] = {
        "lo_pct" : clip_lo_pct,
        "hi_pct" : clip_hi_pct,
        "n_clipped_values" : n_clipped,
        "clipped_fraction_pct" : round(frac, 4),
    }

    labels = (
        y_seg.long() if isinstance(y_seg, torch.Tensor)
        else torch.tensor(y_seg, dtype=torch.long) if y_seg is not None
        else torch.zeros(N, dtype=torch.long)
    )

    payload = {
        "features"     : all_feats,
        "labels"       : labels,
        "feat_dim"     : feat_dim,
        "layout"       : layout,
        "cfg_snapshot" : cfg_snapshot,
        # clip bounds: applied to all new/noisy feature vectors at inference
        "clip_q_lo"    : q_lo,    # (feat_dim,)
        "clip_q_hi"    : q_hi,    # (feat_dim,)
    }

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    torch.save(payload, cache_path)

    elapsed_total = time.time() - t0
    print(f"\n[precompute_tda] Saved {tuple(all_feats.shape)} -> {cache_path}")
    print(f"  Total time : {elapsed_total:.1f}s  "
          f"({elapsed_total / N * 1000:.1f} ms/sample)")
    print(f"  feat_dim   : {feat_dim}  (derived from live layer classes)")
    print(f"  clip bounds: saved (apply_clip_bounds() for inference)")
    _print_layout(layout)

    return all_feats


# ============================================================================
# Load / ablation helpers
# ============================================================================

def load_cache(cache_path: str) -> tuple:
    """
    Load a precomputed TDA feature cache from disk.

    Returns
    -------
    features : (N, feat_dim) clipped feature tensor
    payload  : full dict (includes clip_q_lo, clip_q_hi, layout, cfg_snapshot)
    """
    payload  = torch.load(cache_path, weights_only=True)
    features = payload["features"]
    print(f"[load_cache] {cache_path}")
    print(f"  features : {tuple(features.shape)}  feat_dim={payload['feat_dim']}")
    _print_layout(payload["layout"])

    # warn if cached dims differ from current code
    stored = payload["cfg_snapshot"].get("_layer_dims", {})
    live   = {
        "LifetimeStatsLayer._S" : LifetimeStatsLayer._S,
        "TDABettiExtractor._S"  : TDABettiExtractor._S,
        "TDAPIExtractor._S"     : TDAPIExtractor._S,
    }
    mismatches = {k: (stored.get(k), live[k]) for k in live if stored.get(k) != live[k]}
    if mismatches:
        import warnings
        warnings.warn(
            f"[load_cache] Layer dim mismatch between cache and current code: "
            f"{mismatches}. The cached feature slices may be misaligned. "
            f"Re-run precompute_and_cache(..., force=True) to rebuild.",
            UserWarning, stacklevel=2,
        )

    if "clip_q_lo" in payload:
        clip_cfg = payload["cfg_snapshot"].get("_clip", {})
        lo  = clip_cfg.get("lo_pct", "?")
        hi  = clip_cfg.get("hi_pct", "?")
        frac = clip_cfg.get("clipped_fraction_pct", "?")
        print(f"  clip bounds: [{lo}%, {hi}%] percentile  "
              f"({frac}% values clipped at precompute time)")
    else:
        import warnings
        warnings.warn(
            "[load_cache] No clip bounds found in this cache. "
            "The robustness sweep will not apply IQR clipping to noisy features. "
            "Re-run precompute_and_cache(..., force=True) to add clip bounds.",
            UserWarning, stacklevel=2,
        )

    return features, payload


def ablate_feature_group(
    features: torch.Tensor,
    layout:   dict,
    group:    str,
) -> torch.Tensor:
    assert group in layout, f"Unknown group '{group}'. Available: {list(layout)}"
    out = features.clone()
    s, e = layout[group]
    out[:, s:e] = 0.0
    print(f"[ablate] zeroed '{group}' (dims {s}:{e})")
    return out


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import argparse, json, os, sys

    parser = argparse.ArgumentParser(
        description="Precompute enriched TDA features and save to cache."
    )
    parser.add_argument("--config",      required=True,
                        help="Path to search_config.json or best_config.json")
    parser.add_argument("--cache_path",  default=None,
                        help="Override tda_cache_path in config")
    parser.add_argument("--force",       action="store_true",
                        help="Recompute even if cache already exists")
    parser.add_argument("--batch_size",  type=int, default=None,
                        help="Override precompute_batch_size in config")
    parser.add_argument("--clip_lo_pct", type=float, default=1.0,
                        help="Lower percentile for IQR clipping (default 1.0)")
    parser.add_argument("--clip_hi_pct", type=float, default=99.0,
                        help="Upper percentile for IQR clipping (default 99.0)")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = json.load(f)

    cfg = raw.get("fixed", raw)

    cache_path = (
        args.cache_path
        or cfg.get("tda_cache_path")
        or os.path.join(cfg["path_save"], "tda_cache.pt")
    )

    if os.path.exists(cache_path) and not args.force:
        payload  = torch.load(cache_path, weights_only=True)
        feat_dim = int(payload["feat_dim"])
        n_seg    = len(payload["features"])
        has_clip = "clip_q_lo" in payload
        print(f"Cache already exists: {n_seg} segments, feat_dim={feat_dim}  "
              f"clip_bounds={'yes' if has_clip else 'NO, rebuild with --force'}")
        print(f"  -> {cache_path}")
        print("Nothing to do. Use --force to recompute.")
        sys.exit(0)

    import ml_classification.utils.read_data_from_h5  as read_data_from_h5
    import ml_classification.utils.pad_truncate_tensor as pad_truncate_tensor
    import ml_classification.utils.segment_time_series as segmenting_data

    h5_path = os.path.join(cfg["path_data"], cfg["h5_filename"])
    print(f"Loading dataset: {h5_path}")

    df = read_data_from_h5.read_data(h5_path)

    for state in cfg.get("exclude_states", []):
        df = df[df["state"] != state]

    keep_states = cfg.get("keep_states", [])
    if keep_states:
        df = df[df["state"].isin(keep_states)]
        print(f"Kept states: {keep_states}  ->  {len(df)} series")
        if len(df) == 0:
            print("ERROR: no rows remain after keep_states filter.")
            sys.exit(1)

    class_labels   = sorted(df["state"].unique())
    state_to_label = {s: i for i, s in enumerate(class_labels)}
    df["label"]    = df["state"].map(state_to_label)
    print(f"Classes: {state_to_label}")

    out    = pad_truncate_tensor.make_tensors(df, seq_len=cfg["length_series"])
    X_full = out["X"].unsqueeze(1)
    y_full = out["y"]
    print(f"Series tensor: {tuple(X_full.shape)}")

    seg_dur = cfg["segmentation_duration"]
    seg_out = segmenting_data.segment_data(
        X_full.squeeze(1), y_full,
        segment_duration=seg_dur,
    )
    X_seg = seg_out["X"].unsqueeze(1)
    y_seg = seg_out["y"]
    print(f"Segmented: {tuple(X_seg.shape)}  seg_dur={seg_dur}")

    feats = precompute_and_cache(
        X_seg       = X_seg,
        cfg         = cfg,
        cache_path  = cache_path,
        batch_size  = args.batch_size,
        force       = args.force,
        y_seg       = y_seg,
        clip_lo_pct = args.clip_lo_pct,
        clip_hi_pct = args.clip_hi_pct,
    )

    print(f"\nDone.")
    print(f"  Feature matrix : {tuple(feats.shape)}")
    print(f"  Labels         : {tuple(y_seg.shape)}")
    print(f"  Cache path     : {cache_path}")
