"""run_training_transformer_multiset.py - Launch repeated training runs for TransformerI or CNNI.

Typical usage
-------------
from run_training_transformer_multiset import run_transformer_training
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_transformer_training(
    data, labels, classes, device,
    modeltype="trans1",   # or "cnn1"
    num_training=10,
    pathsave="./results/",
)

Muon optimizer
--------------
Pass opt="muon" to use Muon for hidden 2-D weights and AdamW for everything
else (biases, norms, input embedding, classifier head). muon_lr controls
the Muon learning rate (default 0.02); the AdamW group still uses lr.

    run_transformer_training(
        ...,
        opt="muon",
        muon_lr=0.02,
        lr=3e-4,
    )

Install:  pip install git+https://github.com/KellerJordan/Muon

Note on single-process / single-GPU Muon:
The KellerJordan `muon` package's `MuonWithAuxAdam.step()` unconditionally
calls `torch.distributed.get_world_size()` internally (it was written for
multi-GPU orthogonalization sync). On a plain single-process SLURM job this
raises:

    ValueError: Default process group has not been initialized, please
    make sure to call init_process_group.

This module works around that by initializing a dummy single-process
(``world_size=1``) ``gloo`` process group before building the Muon
optimizer; see ``_ensure_dist_initialized_for_muon()``. No real distributed
training happens; this is purely to satisfy Muon's internal API assumptions.

As a second line of defense (in case this optimizer object ever gets handed
to a `TrainTransformer` instance from somewhere else, or a stale copy of
this module is imported), `nn_transformer_train.TrainTransformer.__init__`
also independently detects a Muon-family optimizer and ensures the group
exists before any `.step()` call. Both guards are safe to run redundantly.

Noise robustness sweep
-------------------------
Pass noise_levels=[0.0, 0.05, 0.1, ...] to run a post-training robustness
sweep after every run, mirroring the sweep already implemented for
TopTimeNet in run_training_tda_cached.py. Since TransformerI/CNNI operate
directly on the raw (post train-time-normalization) time series with no
separate feature-cache stage, noise here is injected directly into the
model's input tensor: the direct analogue of TDAEnd2EndNet's raw-signal
sweep, not the TDA cached-feature sweep (there is no cached-feature stage
in this pipeline). Also adds post-hoc temperature-scaling calibration and
Expected Calibration Error to match the metrics already reported for
TopTimeNet, so the two pipelines' robustness results are directly
comparable.
"""

from __future__ import annotations

import gc
import glob
import os
import re
import sys
import time
import warnings
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe on HPC / headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo

# Local imports; adjust paths to match your project layout
import ml_classification.transformer_cnn_time_series.nn_transformer_model as model_lib
import ml_classification.transformer_cnn_time_series.nn_transformer_train as train_lib

print(f"[run_training_transformer_multiset] loaded from: {__file__}")
print(f"[run_training_transformer_multiset] nn_transformer_train loaded from: {train_lib.__file__}")


# ============================================================================
# Muon import: try the KellerJordan package first, then torch.optim (>= 2.12)
# ============================================================================

_MUON_BACKEND = None
try:
    from muon import MuonWithAuxAdam as _MuonWithAuxAdam
    _MUON_BACKEND = "muon_pkg"
except ImportError:
    try:
        _torch_muon = torch.optim.Muon          # type: ignore[attr-defined]
        _MUON_BACKEND = "torch_optim"
    except AttributeError:
        _MUON_BACKEND = None


# ============================================================================
# Distributed-group shim for single-process Muon
# ============================================================================

def _ensure_dist_initialized_for_muon() -> bool:
    """Initialize a dummy single-process torch.distributed group.

    The KellerJordan `muon` package's internal `dist.get_world_size()` /
    `all_gather` calls assume a process group always exists (it's written
    for multi-GPU sync of the Newton-Schulz orthogonalization step). A
    world_size=1 `gloo` group satisfies that requirement without requiring
    actual distributed training, safe to use on a single-GPU or CPU-only
    SLURM task.

    The port is derived from ``SLURM_ARRAY_TASK_ID`` / ``SLURM_JOB_ID`` (or
    the process PID as a last resort) so that concurrent array tasks
    co-located on the same node don't collide on the same TCP port.

    Returns:
        True if a process group is now initialized (either it already was,
        or this call just created one); False if initialization failed.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return True

    if not dist.is_available():
        warnings.warn(
            "torch.distributed is not available in this PyTorch build; "
            "cannot initialize the dummy process group Muon requires.",
            stacklevel=2,
        )
        return False

    task_id = (
        os.environ.get("SLURM_ARRAY_TASK_ID")
        or os.environ.get("SLURM_JOB_ID")
        or str(os.getpid())
    )
    try:
        port = 20000 + (int(task_id) % 20000)
    except ValueError:
        port = 20000 + (os.getpid() % 20000)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    try:
        dist.init_process_group(
            backend="gloo",   # gloo works regardless of CUDA availability
            rank=0,
            world_size=1,
        )
        print(
            f"  Initialized single-process torch.distributed group "
            f"(world_size=1, port={os.environ['MASTER_PORT']}) for Muon."
        )
        return True
    except Exception as exc:
        warnings.warn(
            f"Could not initialize a dummy process group for Muon "
            f"(required even for single-GPU runs): {exc}\n"
            "Falling back to AdamW.",
            stacklevel=2,
        )
        return False


def _maybe_destroy_dist_group() -> None:
    """Tear down the dummy process group created for Muon, if any exists.

    Safe to call unconditionally, no-ops if nothing was initialized.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


# ============================================================================
# Filename sanitization
# ============================================================================

def _sanitize_filename_part(s: str) -> str:
    """Make a string safe for use in a filename: word chars and dashes only.

    Any run of non-alphanumeric characters (spaces, commas, parens, '=',
    quotes, etc.) collapses to a single '-'; leading/trailing separators
    are stripped.

    Args:
        s: Raw string, possibly containing spaces/punctuation.

    Returns:
        A string containing only ``[A-Za-z0-9_-]`` characters.
    """
    s = re.sub(r"[^\w\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-_")


def _build_run_label(
    modeltype: str,
    n_classes: int,
    num_channels: int,
    **arch_kwargs,
) -> str:
    """Build a short, filesystem-safe label used as a filename prefix.

    Unlike slicing a human-readable description string (which can cut
    mid-token and leave stray punctuation such as commas/parens/quotes in
    filenames, e.g. ``CNNI__in=1_out=2_base_ch=64_mults=(1,_2,``), this
    builds the label directly from the key parameters that actually
    distinguish one run's architecture from another.

    Args:
        modeltype:    ``'trans1'`` or ``'cnn1'``.
        n_classes:    Output class count.
        num_channels: Input channel count.
        **arch_kwargs: Architecture-specific values, ``embed_size``,
            ``nhead``, ``dim_feedforward``, ``num_encoderlayers`` for
            ``trans1``; ``cnn_base_channels``, ``cnn_pooling`` for ``cnn1``.

    Returns:
        A string containing only word characters, dashes, and underscores,
        safe to use directly in a filename.
    """
    parts = [modeltype, f"in{num_channels}", f"out{n_classes}"]

    if modeltype == "trans1":
        parts += [
            f"emb{arch_kwargs.get('embed_size')}",
            f"heads{arch_kwargs.get('nhead')}",
            f"ff{arch_kwargs.get('dim_feedforward')}",
            f"L{arch_kwargs.get('num_encoderlayers')}",
        ]
    elif modeltype == "cnn1":
        # NOTE: cnn_channel_multipliers deliberately excluded from the label
        # (it's a tuple like (1, 2, 4); even sanitized it made filenames
        # long and hard to read). It's still recorded in full in results.csv.
        parts += [
            f"base{arch_kwargs.get('cnn_base_channels')}",
            f"pool{arch_kwargs.get('cnn_pooling')}",
        ]

    return "_".join(_sanitize_filename_part(p) for p in parts)


# ============================================================================
# Memory helper
# ============================================================================

def _free_trial_memory(objects: list, pathsave: Optional[str] = None) -> None:
    """Delete PyTorch / Python objects and flush all caches.

    Called in the ``finally`` block of every run so GPU / CPU memory is
    released even when a run crashes mid-way.

    Args:
        objects:  List of variables to delete (``None`` entries are skipped).
        pathsave: If given, also removes ``checkpoint*.pt`` files written by
                  ``EarlyStopper`` in that directory.
    """
    for obj in objects:
        try:
            del obj
        except Exception:
            pass

    # Remove EarlyStopper checkpoint files from cwd and pathsave
    for pattern in ["checkpoint*.pt"]:
        for ckpt in glob.glob(pattern):
            try:
                os.remove(ckpt)
            except OSError:
                pass
    if pathsave:
        for ckpt in glob.glob(os.path.join(pathsave, "checkpoint*.pt")):
            try:
                os.remove(ckpt)
            except OSError:
                pass

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    gc.collect()
    gc.collect()   # second pass breaks reference cycles


# ============================================================================
# Muon parameter splitting and optimizer builder
# ============================================================================

def _split_muon_params(model: nn.Module) -> tuple[list, list]:
    """Split parameters into Muon-eligible and AdamW groups.

    Muon rules:
      hidden_weights : ndim >= 2, not first-layer inputs or the classifier head.
      rest_params    : biases, norms, 1-D params, input embedding, head.

    Keywords that force a parameter to AdamW:
      "embed", "input", "head", "classifier", "norm", "bn", "bias", "out_proj"

    For TransformerI this means:
      Muon  -> encoder Linear weights, FF Linear weights, Conv1d weights (hidden)
      AdamW -> positional / token embeddings, final linear head, all biases,
               LayerNorm params

    For CNNI this means:
      Muon  -> conv weight tensors in hidden blocks (all ndim=4, flattened by Muon)
      AdamW -> BatchNorm params, final linear head, all biases
    """
    head_and_input_keywords = {
        "embed", "input_proj", "pos_enc",
        "head", "classifier", "fc_out", "linear_out", "out_proj",
        "norm", "bn", "bias",
    }

    hidden_weights: list = []
    rest_params:    list = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        name_lower = name.lower()
        is_excluded = any(kw in name_lower for kw in head_and_input_keywords)
        if p.ndim >= 2 and not is_excluded:
            hidden_weights.append(p)
        else:
            rest_params.append(p)

    return hidden_weights, rest_params


class _DualOptimizer:
    """Thin wrapper that presents a single optimizer interface over two
    separate torch.optim instances (torch.optim.Muon + AdamW path).

    Exposes .zero_grad(), .step(), .state_dict(), and .param_groups so
    ReduceLROnPlateau and TrainTransformer can treat it as a single object.
    """

    def __init__(self, opt_muon, opt_adamw):
        self._muon  = opt_muon
        self._adamw = opt_adamw
        # Expose AdamW param_groups so ReduceLROnPlateau can read the lr
        self.param_groups = opt_adamw.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self._muon.step()
        self._adamw.step(closure)

    def state_dict(self):
        return {"muon": self._muon.state_dict(), "adamw": self._adamw.state_dict()}

    def load_state_dict(self, state):
        self._muon.load_state_dict(state["muon"])
        self._adamw.load_state_dict(state["adamw"])


def _build_muon_optimizer(
    model:    nn.Module,
    lr:       float,
    muon_lr:  float,
) -> tuple:
    """Build a Muon-based optimizer (or pair).

    Initializes a dummy single-process torch.distributed group first (see
    ``_ensure_dist_initialized_for_muon``), since both the ``muon_pkg`` and
    ``torch_optim`` backends assume one exists internally. If that
    initialization fails, falls back to plain AdamW over all parameters.

    Returns
    -------
    optimizer    : single combined optimizer or _DualOptimizer
    muon_backend : str, "muon_pkg" | "torch_optim" | "adamw_fallback"
    n_muon       : int, number of parameters updated by Muon
    n_adamw      : int, number of parameters updated by AdamW
    """
    hidden_weights, rest_params = _split_muon_params(model)
    n_muon  = sum(p.numel() for p in hidden_weights)
    n_adamw = sum(p.numel() for p in rest_params)

    print(
        f"  Muon param split: {n_muon:,} -> Muon  |  {n_adamw:,} -> AdamW"
    )

    if _MUON_BACKEND in ("muon_pkg", "torch_optim"):
        if not _ensure_dist_initialized_for_muon():
            optimizer = optim.AdamW(
                list(model.parameters()), lr=lr, weight_decay=1e-4,
            )
            print(f"  Muon unavailable (no process group), using AdamW (lr={lr})")
            return optimizer, "adamw_fallback", 0, n_muon + n_adamw

    if _MUON_BACKEND == "muon_pkg":
        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                 lr=muon_lr, weight_decay=0.01),
            dict(params=rest_params,    use_muon=False,
                 lr=lr, betas=(0.9, 0.95), weight_decay=1e-4),
        ]
        optimizer = _MuonWithAuxAdam(param_groups)
        print(f"  Using MuonWithAuxAdam  (muon_lr={muon_lr}, adamw_lr={lr})")
        return optimizer, "muon_pkg", n_muon, n_adamw

    elif _MUON_BACKEND == "torch_optim":
        opt_muon  = torch.optim.Muon(       # type: ignore[attr-defined]
            hidden_weights, lr=muon_lr, momentum=0.95,
        )
        opt_adamw = optim.AdamW(rest_params, lr=lr, weight_decay=1e-4)
        print(f"  Using torch.optim.Muon + AdamW  (muon_lr={muon_lr}, adamw_lr={lr})")
        return _DualOptimizer(opt_muon, opt_adamw), "torch_optim", n_muon, n_adamw

    else:
        warnings.warn(
            "Muon optimizer requested (opt='muon') but neither the 'muon' "
            "package nor torch.optim.Muon (PyTorch >= 2.12) is available. "
            "Falling back to AdamW.\n"
            "  Install:  pip install git+https://github.com/KellerJordan/Muon",
            stacklevel=3,
        )
        optimizer = optim.AdamW(
            list(model.parameters()), lr=lr, weight_decay=1e-4,
        )
        print(f"  Muon unavailable, using AdamW (lr={lr})")
        return optimizer, "adamw_fallback", 0, n_muon + n_adamw


# ============================================================================
# Post-hoc temperature scaling
# ============================================================================

def _fit_temperature(
    net:       nn.Module,
    loader_va: torch.utils.data.DataLoader,
    device:    torch.device,
    max_iter:  int = 50,
) -> float:
    """
    Post-hoc temperature scaling: fit a scalar T on the validation set so
    that cross-entropy of (logits / T) is minimised. T > 1 gives softer
    (less overconfident) predicted probabilities.

    Mirrors _fit_temperature() in run_training_tda_cached.py exactly, so
    calibration results are directly comparable across both pipelines.

    Returns
    -------
    T : float, optimal temperature, clamped to [0.5, 5.0] (1.0 = no change)
    """
    net.eval()
    logits_all = []
    labels_all = []

    with torch.no_grad():
        for xb, yb in loader_va:
            logits_all.append(net(xb.to(device)).cpu())
            labels_all.append(yb.cpu())

    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)

    T   = nn.Parameter(torch.ones(1))
    opt_T = optim.LBFGS([T], lr=0.1, max_iter=max_iter)
    loss_fn = nn.CrossEntropyLoss()

    def _eval():
        opt_T.zero_grad()
        loss = loss_fn(logits_all / T.clamp(min=0.5, max=5.0), labels_all)
        loss.backward()
        return loss

    opt_T.step(_eval)
    t_val = T.clamp(min=0.5, max=5.0).item()
    print(f"  Calibration temperature T = {t_val:.4f}")
    return t_val


# ============================================================================
# Expected Calibration Error
# ============================================================================

def _compute_ece(
    preds:  torch.Tensor,
    confs:  torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) with equal-width confidence bins.

        ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Identical implementation to run_training_tda_cached.py's _compute_ece,
    duplicated here so this module has no dependency on the TDA package.

    Returns
    -------
    ece : float in [0, 1]
    """
    N          = len(labels)
    bin_edges  = torch.linspace(0.0, 1.0, n_bins + 1)
    ece        = 0.0

    for i in range(n_bins):
        lo   = bin_edges[i].item()
        hi   = bin_edges[i + 1].item()
        mask = (confs >= lo) & (confs < hi)
        if i == n_bins - 1:              # include right edge in last bin
            mask = (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = (preds[mask] == labels[mask]).float().mean().item()
        bin_conf = confs[mask].mean().item()
        ece     += (mask.float().sum().item() / N) * abs(bin_conf - bin_acc)

    return ece


# ============================================================================
# Noise robustness sweep
# ============================================================================

def _robustness_sweep(
    net:          nn.Module,
    x_test:       torch.Tensor,
    y_test:       torch.Tensor,
    noise_levels: List[float],
    batch_size:   int,
    seed:         int,
    device:       torch.device,
    temperature:  float = 1.0,
    ece_bins:     int = 10,
) -> pd.DataFrame:
    """
    Noise-robustness sweep for TransformerI / CNNI.

    Gaussian noise of standard deviation sigma is injected directly into
    the model's input tensor (the same, already train-time-normalized
    representation the model was evaluated on in testing_step) for each
    sigma in noise_levels, and the model is re-evaluated from scratch at
    every level. There is no separate feature-extraction stage to
    recompute for these architectures; the model itself is the feature
    extractor, so this sweep is the direct analogue of TDAEnd2EndNet's
    raw-signal sweep (net.evaluate_robustness), not the TDA cached-feature
    sweep, which has no counterpart here.

    Same schema as the TDA pipeline's robustness sweeps, so results can be
    concatenated and plotted with the same plot_robustness_paper.py script:

        sigma, snr_db, accuracy, mean_confidence, pct_low_conf, ece

    Args
    ----
    net          : trained model (eval mode set internally)
    x_test       : (N, C, T) test input tensor, CPU, in whatever
                   preprocessing state the model expects
    y_test       : (N,) long tensor of labels, CPU
    noise_levels : list of sigma values
    batch_size   : forward-pass batch size
    seed         : RNG seed for reproducible noise
    device       : torch device
    temperature  : calibration temperature (default 1.0, disabled)
    ece_bins     : number of ECE calibration bins

    Returns
    -------
    pd.DataFrame indexed by sigma
    """
    net.eval()
    rng = torch.Generator()
    rng.manual_seed(seed)

    records   = []
    n_levels  = len(noise_levels)
    n_samples = x_test.shape[0]

    print(f"  Robustness sweep: {n_levels} sigma levels  N={n_samples}  "
          f"temperature T={temperature:.4f}  ECE bins={ece_bins}")

    for i, sigma in enumerate(noise_levels):
        if sigma == 0.0:
            x_noisy = x_test.clone()
        else:
            noise   = torch.zeros_like(x_test).normal_(0.0, sigma, generator=rng)
            x_noisy = x_test + noise

        all_preds, all_confs = [], []

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                xb     = x_noisy[start: start + batch_size].to(device)
                logits = net(xb)
                probs  = torch.softmax(logits / temperature, dim=1)
                conf, pred = probs.max(dim=1)
                all_preds.append(pred.cpu())
                all_confs.append(conf.cpu())

        preds = torch.cat(all_preds)
        confs = torch.cat(all_confs)

        acc       = (preds == y_test).float().mean().item()
        mean_conf = confs.mean().item()
        pct_low   = (confs < 0.5).float().mean().item() * 100.0
        snr_db    = 10.0 * np.log10(1.0 / (sigma ** 2)) if sigma > 0 else float("inf")
        ece       = _compute_ece(preds, confs, y_test, n_bins=ece_bins)

        records.append({
            "sigma"           : sigma,
            "snr_db"          : snr_db,
            "accuracy"        : acc,
            "mean_confidence" : mean_conf,
            "pct_low_conf"    : pct_low,
            "ece"             : ece,
        })
        print(f"  [{i+1:2d}/{n_levels}]  sigma={sigma:.3f}  SNR={snr_db:+.1f} dB  "
              f"acc={acc:.4f}  conf={mean_conf:.4f}  "
              f"low_conf={pct_low:.1f}%  ECE={ece:.4f}")

    return pd.DataFrame(records).set_index("sigma")


# ============================================================================
# Model factory
# ============================================================================

def _build_model(
    modeltype: str,
    num_channels: int,
    n_classes: int,
    max_length_series: int,
    embed_size: int,
    nhead: int,
    dim_feedforward: int,
    num_encoderlayers: int,
    dropout: float,
    conv1d_emb: bool,
    conv1d_kernel_size: int,
    size_linear_layers: int,
    device: torch.device,
    # CNN-specific
    cnn_base_channels: int = 32,
    cnn_channel_multipliers: tuple = (1, 2, 4),
    cnn_pooling: str = "mean",
) -> tuple[nn.Module, str, str]:
    """Instantiate a model by name and return it with metadata strings.

    Args:
        modeltype: ``'trans1'`` for TransformerI, ``'cnn1'`` for CNNI.
        (all other args): forwarded to the model constructor.

    Returns:
        ``(model, arch_str, label_str)`` where ``arch_str`` is a
        human-readable architecture description (may contain spaces,
        commas, parens, used only in plot *titles*), and ``label_str`` is
        a short, filesystem-safe prefix built directly from the model's
        key parameters (safe to use in *filenames*).

    Raises:
        ValueError: If ``modeltype`` is not recognised.
    """
    if modeltype == "trans1":
        model = model_lib.TransformerI(
            input_channels=num_channels,
            output_size=n_classes,
            seq_len=max_length_series,
            embed_size=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            conv1d_emb=conv1d_emb,
            conv1d_kernel_size=conv1d_kernel_size,
            size_linear_layers=size_linear_layers,
            num_encoderlayers=num_encoderlayers,
            # device kwarg removed; TransformerI no longer accepts it,
            # placement is handled by model.to(device) below.
        )
        arch_str = (
            f"TransformerI  in={num_channels} out={n_classes} "
            f"seq={max_length_series} emb={embed_size} heads={nhead} "
            f"ff={dim_feedforward} layers={num_encoderlayers} "
            f"conv1d={conv1d_emb}"
        )
        label_str = _build_run_label(
            modeltype, n_classes, num_channels,
            embed_size=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoderlayers=num_encoderlayers,
        )

    elif modeltype == "cnn1":
        model = model_lib.CNNI(
            input_channels=num_channels,
            output_size=n_classes,
            base_channels=cnn_base_channels,
            channel_multipliers=cnn_channel_multipliers,
            kernel_size=conv1d_kernel_size,
            dropout=dropout,
            pooling=cnn_pooling,
            size_linear_layers=size_linear_layers,
        )
        arch_str = (
            f"CNNI  in={num_channels} out={n_classes} "
            f"base_ch={cnn_base_channels} mults={cnn_channel_multipliers} "
            f"kernel={conv1d_kernel_size} pool={cnn_pooling}"
        )
        label_str = _build_run_label(
            modeltype, n_classes, num_channels,
            cnn_base_channels=cnn_base_channels,
            cnn_pooling=cnn_pooling,
        )

    else:
        raise ValueError(
            f"Unknown modeltype '{modeltype}'. "
            f"Supported values: 'trans1', 'cnn1'."
        )

    return model, arch_str, label_str


# ============================================================================
# Main training function
# ============================================================================

def run_transformer_training(
    data,
    labels,
    classes: list,
    device: torch.device,
    # data
    num_channels: int = 6,
    n_classes: int = 2,
    test_size: float = 0.3,
    val_size: float = 0.1,
    norm_type: str = "per-channel",
    # DataLoader
    batch_size: int = 32,
    num_cpus: int = 1,
    # model
    modeltype: str = "trans1",
    max_length_series: int = 200,
    embed_size: int = 16,
    nhead: int = 4,
    dim_feedforward: int = 2048,
    num_encoderlayers: int = 1,
    dropout: float = 0.0,
    conv1d_emb: bool = True,
    conv1d_kernel_size: int = 3,
    size_linear_layers: int = 16,
    # CNN-specific (ignored for trans1)
    cnn_base_channels: int = 32,
    cnn_channel_multipliers: tuple = (1, 2, 4),
    cnn_pooling: str = "mean",
    # optimisation
    opt: str = "adamW",       # "adam" | "adamW" | "muon"
    lr: float = 1e-3,
    muon_lr: float = 0.02,    # Muon hidden-layer lr; ignored when opt != "muon"
    num_epochs: int = 100,
    patience: int = 5,
    weights=None,
    # misc
    verbose: bool = False,
    pathsave: str = "./",
    num_training: int = 20,
    # robustness sweep
    noise_levels: Optional[List[float]] = None,
    noise_batch_size: int = 32,
    # temperature scaling
    use_temperature_scaling: bool = True,
    # ECE bins
    ece_bins: int = 10,
    # External, globally unique run identifier (e.g. SLURM array task ID),
    # used instead of the internal per-call loop index for anything
    # written to pathsave. Required when separate process invocations
    # (e.g. one per SLURM array task) share the same pathsave directory,
    # see the note in the docstring below for why this matters.
    external_run_id: Optional[int] = None,
) -> float:
    """Run ``num_training`` independent training runs and aggregate results.

    Each run re-splits the data, re-initialises the model, trains with early
    stopping, evaluates on the test set, and writes one row to a CSV. A
    summary row (mean +/- std) is appended at the end.

    If noise_levels is given, after each run's test evaluation a
    noise-robustness sweep is also run (see _robustness_sweep), with
    optional post-hoc temperature-scaling calibration and per-sigma
    Expected Calibration Error. Per-run sweep CSVs are saved as
    robustness_sweep_run{N}.csv, where N is external_run_id when given,
    else the internal loop index, using the same schema as the TDA
    pipeline's sweep CSVs, produced by the same plot_robustness_paper.py
    script.

    Multi-process accumulation: robustness_all_runs.csv is not safely
    accumulated across separate process invocations (e.g. separate SLURM
    array tasks sharing one pathsave). Each call to this function only
    ever sees its own in-memory all_sweep_dfs, so writing a "combined"
    file here would silently overwrite whatever a concurrent or prior
    task already wrote, with no read-back. This function therefore only
    writes robustness_sweep_run{N}.csv (per call, uniquely named via
    external_run_id) and does not write robustness_all_runs.csv itself.
    Combine all robustness_sweep_run*.csv files across every task after
    the full set of runs/tasks has completed, e.g. via
    merge_robustness_sweeps.py (a separate, one-time merge step run once
    the whole batch is done), mirroring the per-task-isolation and
    aggregation pattern already used for all_trials.csv.

    Args:
        data:                Input tensor / array of shape ``(N, channels, time)``.
        labels:              Integer label array of shape ``(N,)``.
        classes:             Ordered list of class-name strings (for plots).
        device:              ``torch.device`` for model and data.
        num_channels:        Input channel count.
        n_classes:           Output class count.
        test_size:           Fraction of data held out for testing.
        val_size:            Fraction of the remaining data held out for validation.
        norm_type:           Passed to ``normalize_data``
                             (``'global'``, ``'per-channel'``, ``'per-timestep'``).
        batch_size:          DataLoader batch size.
        num_cpus:            DataLoader worker count.
        modeltype:           ``'trans1'`` (TransformerI) or ``'cnn1'`` (CNNI).
        max_length_series:   Sequence length expected by the model.
        embed_size:          Transformer embedding dimension (trans1 only).
        nhead:               Attention heads (trans1 only).
        dim_feedforward:     Transformer FF dimension (trans1 only).
        num_encoderlayers:   Encoder layer count (trans1 only).
        dropout:             Dropout rate.
        conv1d_emb:          Use Conv1D input embedding (trans1 only).
        conv1d_kernel_size:  Kernel size, must be odd.
        size_linear_layers:  MLP head hidden size.
        cnn_base_channels:   Base channel width (cnn1 only).
        cnn_channel_multipliers: Per-block channel multipliers (cnn1 only).
        cnn_pooling:         Temporal pooling strategy for cnn1 (``'mean'``,
                             ``'max'``, ``'last'``).
        opt:                 Optimiser, ``'adam'``, ``'adamW'``, or ``'muon'``.
        lr:                  Learning rate (AdamW group when opt='muon').
        muon_lr:             Learning rate for Muon hidden-weight group
                             (default 0.02, ignored when opt != 'muon').
        num_epochs:          Maximum epochs per run.
        patience:            Early-stopping patience (epochs).
        weights:             Optional class-weight tensor for ``CrossEntropyLoss``.
        verbose:             Print per-epoch detail and save CM plots.
        pathsave:            Directory for CSV, plots, and checkpoints.
        num_training:        Number of independent runs.
        noise_levels:        list of sigma values, or None to disable
                             the robustness sweep entirely.
        noise_batch_size:    batch size used during the sweep.
        use_temperature_scaling: fit a calibration temperature on the
                             validation set after training and use it in
                             the sweep's softmax.
        ece_bins:            number of equal-width bins for ECE.

    Returns:
        Wall-clock time (seconds) of the *last* completed run.
    """
    os.makedirs(pathsave, exist_ok=True)

    # per-run accumulators
    run_records: list[dict] = []
    arr_acc_test:       list[float] = []
    arr_f1score_test:   list[float] = []
    arr_gmean_test:     list[float] = []
    arr_precision_test: list[float] = []
    arr_recall_test:    list[float] = []

    # accumulate sweep DataFrames across all runs
    all_sweep_dfs: list[pd.DataFrame] = []

    computation_time = 0.0   # defined even when num_training == 0

    for idx in range(num_training):
        print(f"\n{'='*55}")
        print(f"  Training run {idx + 1} / {num_training}  opt={opt}" +
              (f"  muon_lr={muon_lr}" if opt == "muon" else ""))
        print(f"{'='*55}")

        parameters = {
            "test_size": test_size, "val_size": val_size,
            "batch_size": batch_size, "num_cpus": num_cpus,
            "modeltype": modeltype, "lr": lr, "muon_lr": muon_lr,
            "num_epochs": num_epochs,
            "verbose": verbose, "num_channels": num_channels,
            "n_classes": n_classes, "patience": patience,
            "max_length_series": max_length_series,
            "nhead": nhead, "dim_feedforward": dim_feedforward,
            "embed_size": embed_size, "dropout": dropout,
            "conv1d_emb": conv1d_emb, "conv1d_kernel_size": conv1d_kernel_size,
            "size_linear_layers": size_linear_layers,
            "num_encoderlayers": num_encoderlayers,
            "opt": opt, "weights": weights,
            "norm_type": norm_type, "num_training": num_training,
        }
        if verbose:
            print("parameters:", parameters)

        start = time.time()

        # Initialise all names so the finally block never hits NameError
        model = optimizer = loss_function = early_stopper = scheduler = None
        train_model = None
        x_train = x_val = x_test = y_train = y_val = y_test = None
        xy_train = xy_val = xy_test = None
        trainloader = valloader = testloader = None
        confusion_matrix_val = confusion_matrix_test = None
        muon_backend = "none"
        temperature = 1.0   # default: no scaling
        run_seed = idx  # local fallback if not otherwise seeded upstream

        try:
            # build model
            model, arch_str, label_str = _build_model(
                modeltype=modeltype,
                num_channels=num_channels,
                n_classes=n_classes,
                max_length_series=max_length_series,
                embed_size=embed_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_encoderlayers=num_encoderlayers,
                dropout=dropout,
                conv1d_emb=conv1d_emb,
                conv1d_kernel_size=conv1d_kernel_size,
                size_linear_layers=size_linear_layers,
                device=device,
                cnn_base_channels=cnn_base_channels,
                cnn_channel_multipliers=cnn_channel_multipliers,
                cnn_pooling=cnn_pooling,
            )

            # Use external_run_id (e.g. SLURM array task ID) when given,
            # not the internal loop index idx+1: this label feeds
            # directly into ckpt_path below, which every parallel SLURM
            # task writes to inside the same shared pathsave directory.
            # Since each task calls this with num_training=1, idx is
            # always 0 for every task, so idx+1 alone would make every
            # parallel task compute the identical checkpoint path and
            # write/read it concurrently, a race condition that risks
            # both hard failures (a torch.load on a checkpoint corrupted
            # by a concurrent write) and silent corruption (a task
            # reading a different task's complete-but-wrong checkpoint
            # with no error at all). This single value drives the
            # checkpoint path, plot filenames, and titles below, so all
            # per-run artifacts are consistently and uniquely labeled.
            this_run_id = external_run_id if external_run_id is not None else (idx + 1)
            run_label = f"run{this_run_id}_{label_str}"   # used for file names

            # torchinfo expects (batch, channels, seq_len)
            print(torchinfo.summary(
                model,
                input_size=(1, num_channels, max_length_series),
                device=device,
            ))

            # initialise weights
            model_lib.init_weights(model, "kaiming")
            model = model.to(device)

            # Count parameters on the actual trained instance (not a
            # separate probe model), so the count is guaranteed correct
            # for exactly what was trained, and saved per-run below, not
            # just once per script invocation in all_trials.csv.
            total_params     = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters()
                                    if p.requires_grad)
            print(f"  total_params={total_params:,}  trainable={trainable_params:,}")

            # optimiser
            if opt == "muon":
                optimizer, muon_backend, n_muon, n_adamw = _build_muon_optimizer(
                    model, lr=lr, muon_lr=muon_lr,
                )
            elif opt == "adam":
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            elif opt in ("adamW", "adamw"):
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            else:
                raise ValueError(
                    f"Unknown optimizer '{opt}'. Use 'adam', 'adamW', or 'muon'."
                )

            # loss
            loss_function = nn.CrossEntropyLoss(
                weight=weights.to(device) if weights is not None else None
            )

            # early stopper
            ckpt_path = os.path.join(pathsave, f"checkpoint_run{this_run_id}.pt")
            early_stopper = model_lib.EarlyStopper(
                verbose=verbose,
                path=ckpt_path,
                patience=patience,
            )

            # LR scheduler
            # For Muon, ReduceLROnPlateau adjusts the AdamW param group only.
            # _DualOptimizer.param_groups is already wired to AdamW.
            # MuonWithAuxAdam exposes all param_groups; the scheduler will
            # only meaningfully affect the use_muon=False group since Muon
            # manages its own step size internally.
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.9,
                patience=max(1, patience - 4), threshold=1e-8,
            )

            # data splits
            train_model = train_lib.TrainTransformer(
                model, optimizer, loss_function,
                early_stopper, scheduler, device, parameters,
            )

            x_train, x_val, x_test, y_train, y_val, y_test = \
                train_model.generate_train_val_test(data, labels)

            print(f"  x_train {tuple(x_train.shape)}  y_train {tuple(y_train.shape)}")
            print(f"  x_val   {tuple(x_val.shape)}  y_val   {tuple(y_val.shape)}")
            print(f"  x_test  {tuple(x_test.shape)}  y_test  {tuple(y_test.shape)}")

            # Keep the test-split tensors (as returned, i.e. already in
            # whatever normalized state generate_train_val_test produced)
            # for the robustness sweep below, since testloader only
            # exposes them batch-by-batch.
            if noise_levels is not None:
                x_test_for_sweep = (
                    x_test.clone() if isinstance(x_test, torch.Tensor)
                    else torch.tensor(np.array(x_test), dtype=torch.float32)
                )
                y_test_for_sweep = torch.tensor(np.array(y_test), dtype=torch.long)
            else:
                x_test_for_sweep = y_test_for_sweep = None

            xy_train = train_lib.data_to_tensor(x_train, y_train)
            xy_val   = train_lib.data_to_tensor(x_val,   y_val)
            xy_test  = train_lib.data_to_tensor(x_test,  y_test)

            # Free split arrays; DataLoader copies internally
            del x_train, x_val, x_test, y_train, y_val, y_test
            x_train = x_val = x_test = y_train = y_val = y_test = None

            # num_workers=0 prevents persistent worker processes that hold
            # separate copies of the dataset across runs
            trainloader, valloader, testloader = \
                train_model.init_data_loaders(
                    xy_train, xy_val, xy_test, num_workers=0
                )

            # training loop
            (train_losses, val_losses,
             train_accs,   val_accs,
             confusion_matrix_val,
             train_f1,     val_f1,
             train_gmean,  val_gmean,
             train_pre,    val_pre,
             train_rec,    val_rec,
             train_rel,    val_rel,
             train_neutral, val_neutral,
             val_class_rel) = train_model.run_training(trainloader, valloader)

            # learning curves
            for metric_name, train_v, val_v in (
                ("loss",        train_losses, val_losses),
                ("accuracy",    train_accs,   val_accs),
                ("f1",          train_f1,     val_f1),
                ("reliability", train_rel,    val_rel),
            ):
                train_lib.plot_progress(
                    title=f"{arch_str} - {metric_name} (run {this_run_id})",
                    label=metric_name,
                    train_results=train_v,
                    val_results=val_v,
                    save_path=os.path.join(
                        pathsave, f"{metric_name}_{run_label}.png"
                    ),
                )
                plt.close("all")

            # test evaluation
            (correct, total, accuracy_test,
             f1score_test, gmean_test,
             precision_test, recall_test,
             reliability_test, neutral_test,
             class_reliability_test,
             confusion_matrix_test) = train_model.testing_step(testloader)

            computation_time = time.time() - start

            print(
                f"  Run {this_run_id} | "
                f"acc={accuracy_test:.4f}  f1={f1score_test:.4f}  "
                f"gmean={gmean_test:.4f}  rel={reliability_test:.4f}  "
                f"neutral={neutral_test:.1f}%  "
                f"time={computation_time:.1f}s"
            )

            # accumulate test scores
            arr_acc_test.append(accuracy_test)
            arr_f1score_test.append(f1score_test)
            arr_gmean_test.append(gmean_test)
            arr_precision_test.append(precision_test)
            arr_recall_test.append(recall_test)

            # confusion matrix plots
            for cm_tensor, cm_tag in (
                (confusion_matrix_test, "test"),
                *( [(confusion_matrix_val, "val")] if verbose else [] ),
            ):
                if cm_tensor is None:
                    continue
                cm_np   = cm_tensor.numpy().astype(float)
                cm_norm = cm_np / (cm_np.sum(axis=1, keepdims=True) + 1e-9)
                df_cm   = pd.DataFrame(cm_norm,
                                       index=list(classes),
                                       columns=list(classes))
                fig, ax = plt.subplots(figsize=(max(5, n_classes), max(5, n_classes)))
                sn.heatmap(
                    df_cm, annot=True, cmap="coolwarm",
                    fmt=".2f", vmin=0.0, vmax=1.0,
                    ax=ax, linewidths=0.5, linecolor="white",
                )
                ax.set_title(
                    f"{arch_str}  ({cm_tag}, run {this_run_id})  "
                    f"acc={accuracy_test:.4f}",
                    fontsize=10,
                )
                ax.set_xlabel("Predicted", fontsize=12)
                ax.set_ylabel("True",      fontsize=12)
                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        pathsave,
                        f"confusion_matrix_{cm_tag}_{run_label}.png",
                    ),
                    dpi=150,
                )
                plt.close(fig)
                del df_cm, cm_np, cm_norm

            # temperature scaling
            if noise_levels is not None and use_temperature_scaling:
                temperature = _fit_temperature(model, valloader, device)
            elif noise_levels is not None:
                temperature = 1.0

            # noise robustness sweep
            if noise_levels is not None:
                print(f"\n  Robustness sweep (run {this_run_id}) ...")
                model.eval()
                df_sweep = _robustness_sweep(
                    net          = model,
                    x_test       = x_test_for_sweep,
                    y_test       = y_test_for_sweep,
                    noise_levels = noise_levels,
                    batch_size   = noise_batch_size,
                    seed         = run_seed,
                    device       = device,
                    temperature  = temperature,
                    ece_bins     = ece_bins,
                )
                # this_run_id was already computed once, near run_label
                # above, reused here (not recomputed) so every per-run
                # artifact (checkpoint, plots, sweep CSV, titles) is
                # guaranteed to use the exact same identifier.
                df_sweep["run_idx"]         = this_run_id
                df_sweep["modeltype"]       = modeltype
                df_sweep["opt"]             = opt
                df_sweep["temperature"]     = temperature
                # carried alongside the sweep so robustness results can
                # be related to model size without a separate join
                df_sweep["total_params"]     = total_params
                df_sweep["trainable_params"] = trainable_params
                all_sweep_dfs.append(df_sweep.reset_index())

                sweep_path = os.path.join(
                    pathsave, f"robustness_sweep_run{this_run_id}.csv")
                df_sweep.to_csv(sweep_path)
                print(f"  Robustness sweep saved -> {sweep_path}")

                del df_sweep, x_test_for_sweep, y_test_for_sweep
                plt.close("all")

            # CSV record (one row per run)
            record: dict = {
                "run_idx":             idx + 1,
                "modeltype":           modeltype,
                "computation_time_s":  round(computation_time, 2),
                # actual parameter count of the trained model instance
                "total_params":        total_params,
                "trainable_params":    trainable_params,
                "num_channels":        num_channels,
                "n_classes":           n_classes,
                "max_length_series":   max_length_series,
                "embed_size":          embed_size,
                "nhead":               nhead,
                "dim_feedforward":     dim_feedforward,
                "num_encoderlayers":   num_encoderlayers,
                "dropout":             dropout,
                "conv1d_emb":          conv1d_emb,
                "conv1d_kernel_size":  conv1d_kernel_size,
                "size_linear_layers":  size_linear_layers,
                "cnn_base_channels":   cnn_base_channels,
                "cnn_channel_multipliers": str(cnn_channel_multipliers),
                "cnn_pooling":         cnn_pooling,
                "lr":                  lr,
                "muon_lr":             muon_lr if opt == "muon" else None,
                "muon_backend":        muon_backend,
                "batch_size":          batch_size,
                "num_epochs":          num_epochs,
                "patience":            patience,
                "opt":                 opt,
                "norm_type":           norm_type,
                "test_size":           test_size,
                "val_size":            val_size,
                # sweep bookkeeping
                "noise_levels_tested": str(noise_levels) if noise_levels else None,
                "temperature":         round(temperature, 4),
                # test metrics
                "accuracy_test":       round(accuracy_test,    4),
                "f1_test":             round(f1score_test,     4),
                "gmean_test":          round(gmean_test,       4),
                "precision_test":      round(precision_test,   4),
                "recall_test":         round(recall_test,      4),
                "reliability_test":    round(reliability_test, 4),
                "neutral_pct_test":    round(neutral_test,     2),
                # final-epoch training metrics
                "final_train_loss":    round(train_losses[-1],  4),
                "final_val_loss":      round(val_losses[-1],    4),
                "final_train_acc":     round(train_accs[-1],    4),
                "final_val_acc":       round(val_accs[-1],      4),
                "final_train_f1":      round(train_f1[-1],      4),
                "final_val_f1":        round(val_f1[-1],        4),
                "final_train_rel":     round(train_rel[-1],     4),
                "final_val_rel":       round(val_rel[-1],       4),
                "final_val_neutral":   round(val_neutral[-1],   2),
                "n_epochs_trained":    len(train_losses),
                # per-class reliability
                **{
                    f"rel_class_{c}": round(r, 4)
                    for c, r in class_reliability_test.items()
                },
            }
            run_records.append(record)

            # Incremental save, safe against mid-loop crash
            csv_path = os.path.join(pathsave, "results.csv")
            pd.DataFrame(run_records).to_csv(csv_path, index=False)
            print(f"  Results saved -> {csv_path}  ({len(run_records)} rows)")

        except Exception as exc:
            print(f"\n[ERROR] Run {idx + 1} failed: {exc}")
            import traceback
            traceback.print_exc()

        finally:
            # always release memory, even on failure
            _free_trial_memory(
                [
                    model, optimizer, loss_function, early_stopper,
                    scheduler, train_model,
                    x_train, x_val, x_test, y_train, y_val, y_test,
                    xy_train, xy_val, xy_test,
                    trainloader, valloader, testloader,
                    confusion_matrix_val, confusion_matrix_test,
                ],
                pathsave=pathsave,
            )
            plt.close("all")

    # Intentionally not writing a combined robustness_all_runs.csv here:
    # this function's all_sweep_dfs only ever contains the run(s) from
    # this call, so a "combined" write would silently overwrite any rows
    # already written by a concurrent or prior process invocation sharing
    # the same pathsave (exactly what caused only one run's data to
    # survive when this was called once per SLURM array task). Each
    # call's robustness_sweep_run{N}.csv (written above, uniquely named
    # via external_run_id) is now the safe source of truth; combine them
    # across all tasks in a separate, one-time merge step after the full
    # batch completes, e.g. via merge_robustness_sweeps.py.
    if all_sweep_dfs:
        print(f"\n  Wrote {len(all_sweep_dfs)} robustness_sweep_run*.csv "
              f"file(s) to {pathsave}. robustness_all_runs.csv is not "
              f"written per-call; run merge_robustness_sweeps.py once all "
              f"tasks finish to produce the combined file.")

    # summary stats
    if not run_records:
        print("\nNo runs completed, no summary to write.")
        _maybe_destroy_dist_group()
        return computation_time

    print(f"\n{'-' * 55}")
    print(f"  Summary over {len(arr_acc_test)} completed run(s)  opt={opt}")
    print(f"{'-' * 55}")

    metric_arrays = {
        "accuracy":  arr_acc_test,
        "f1":        arr_f1score_test,
        "gmean":     arr_gmean_test,
        "precision": arr_precision_test,
        "recall":    arr_recall_test,
    }
    summary_row: dict = {"run_idx": "SUMMARY", "modeltype": modeltype, "opt": opt}
    # architecture is identical across runs, so report it once here too
    if run_records and "total_params" in run_records[-1]:
        summary_row["total_params"]     = run_records[-1]["total_params"]
        summary_row["trainable_params"] = run_records[-1]["trainable_params"]
    for name, arr in metric_arrays.items():
        if not arr:
            continue
        mean, std = float(np.mean(arr)), float(np.std(arr))
        print(f"  {name:<12}: {mean:.4f} +/- {std:.4f}")
        summary_row[f"{name}_test_mean"] = round(mean, 4)
        summary_row[f"{name}_test_std"]  = round(std,  4)
    print(f"{'-' * 55}")

    csv_path   = os.path.join(pathsave, "results.csv")
    df_runs    = pd.DataFrame(run_records)
    df_summary = pd.DataFrame([summary_row])
    pd.concat([df_runs, df_summary], ignore_index=True).to_csv(csv_path, index=False)
    print(f"Final CSV with summary row -> {csv_path}")

    _maybe_destroy_dist_group()

    return computation_time
