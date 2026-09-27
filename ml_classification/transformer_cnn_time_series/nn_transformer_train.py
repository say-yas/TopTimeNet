"""nn_transformer_train.py - Training harness for TransformerI / CNNI.

Public API
----------
TrainTransformer      - main training / evaluation class
normalize_data        - fit-on-train, apply-to-all normalisation
data_to_tensor        - numpy to TensorDataset
accuracy              - correct / total
plot_progress         - epoch-curve plot
visualize_confusion_matrix
check_precision_recall_accuracy
compute_reliability   - confidence / entropy / margin from logits
weighted_geometric_mean_score
"""


from __future__ import annotations

import os
import time
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torcheval.metrics.functional import (
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Muon dist-group guard
# ---------------------------------------------------------------------------
#
# The KellerJordan `muon` package's `MuonWithAuxAdam.step()` (and, on newer
# PyTorch, `torch.optim.Muon.step()`) unconditionally calls
# `torch.distributed.get_world_size()` internally: it was written assuming
# multi-GPU orthogonalization sync. On a plain single-process job with no
# process group ever initialized, that raises:
#
#     ValueError: Default process group has not been initialized, please
#     make sure to call init_process_group.
#
# TrainTransformer doesn't build the optimizer itself (it's handed one by
# the caller), so it can't rely on the caller having set this up. Instead,
# __init__ detects a Muon-family optimizer and, if no process group exists
# yet, initializes a dummy single-process (world_size=1) `gloo` group. No
# real distributed training happens; this only satisfies Muon's internal
# API assumptions.


def _is_muon_optimizer(optimizer) -> bool:
    """Best-effort detection of a Muon-family optimizer (or a wrapper over
    one), by walking the object's class name and any child optimizers it
    exposes (e.g. a `_DualOptimizer`-style wrapper with `_muon`/`_adamw`
    attributes, or a `MuonWithAuxAdam` instance).
    """
    if optimizer is None:
        return False

    def _name_has_muon(obj) -> bool:
        return "muon" in type(obj).__name__.lower()

    if _name_has_muon(optimizer):
        return True

    # Common wrapper pattern: object holds sub-optimizers as attributes
    for attr in ("_muon", "muon", "optimizer_muon"):
        sub = getattr(optimizer, attr, None)
        if sub is not None and _name_has_muon(sub):
            return True

    # Or a list/tuple of param_groups tagged with use_muon=True
    param_groups = getattr(optimizer, "param_groups", None)
    if param_groups:
        for g in param_groups:
            if isinstance(g, dict) and g.get("use_muon"):
                return True

    return False


def _ensure_dist_initialized_for_muon() -> bool:
    """Initialize a dummy single-process torch.distributed group, if needed.

    Safe to call unconditionally and repeatedly, no-ops if a group already
    exists. Returns True if a process group is initialized afterwards
    (either it already was, or this call just created one); False if
    initialization failed (caller should expect Muon to error and may want
    to fall back to a different optimizer).
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
            f"  [TrainTransformer] Initialized single-process "
            f"torch.distributed group (world_size=1, "
            f"port={os.environ['MASTER_PORT']}) for Muon."
        )
        return True
    except Exception as exc:
        warnings.warn(
            f"Could not initialize a dummy process group for Muon "
            f"(required even for single-GPU runs): {exc}",
            stacklevel=2,
        )
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_reliability_batch(
    logits: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    """Return per-sample entropy-based reliability in [0, 1] (high = reliable).

    Extracted so training_step, validation_step, and testing_step all call the
    same code path instead of repeating it.

    Args:
        logits: (batch, n_classes) raw model outputs.
        n_classes: number of output classes.

    Returns:
        Tensor of shape (batch,) on CPU.
    """
    probs = torch.softmax(logits.detach().cpu(), dim=1)
    entropy = -(probs * probs.clamp(min=1e-9).log()).sum(dim=1)
    max_entropy = torch.log(torch.tensor(float(n_classes)))
    return 1.0 - (entropy / max_entropy)


def _reliability_summary(
    all_reliability: torch.Tensor,
    all_labels: torch.Tensor,
    n_classes: int,
    threshold: float,
) -> tuple[float, float, dict[int, float]]:
    """Aggregate reliability tensors into scalar summaries.

    Args:
        all_reliability: (N,) reliability scores.
        all_labels: (N,) true integer labels.
        n_classes: number of classes.
        threshold: samples below this are counted as "neutral zone".

    Returns:
        mean_reliability, pct_neutral, per_class_reliability
    """
    mean_reliability = all_reliability.mean().item()
    pct_neutral = (all_reliability < threshold).float().mean().item() * 100.0

    per_class: dict[int, float] = {}
    for c in range(n_classes):
        mask = all_labels == c
        per_class[c] = (
            all_reliability[mask].mean().item() if mask.sum() > 0 else float("nan")
        )

    return mean_reliability, pct_neutral, per_class


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def accuracy(correct: int | float, total: int | float) -> float:
    """Fraction of correctly classified samples."""
    return float(correct) / float(total)


def data_to_tensor(
    data: np.ndarray,
    labels: np.ndarray,
    label_dtype: torch.dtype = torch.long,
) -> torch.utils.data.TensorDataset:
    """Convert numpy arrays to a TensorDataset.

    Args:
        data: Feature array of shape (N, ...).
        labels: Integer label array of shape (N,).
        label_dtype: dtype for the label tensor. ``torch.long`` (default) is
            required by cross-entropy loss; use ``torch.float`` for regression.

    Returns:
        TensorDataset wrapping ``(data_tensor, label_tensor)``.
    """
    torch_data = torch.tensor(np.array(data, dtype=np.float32))
    torch_lbls = torch.tensor(np.array(labels)).to(label_dtype)
    return torch.utils.data.TensorDataset(torch_data, torch_lbls)


def normalize_data(
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    x_test: torch.Tensor,
    norm_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalise tensors using statistics computed on the training set only.

    All statistics are fitted on ``x_train`` and then applied to every split,
    preventing data leakage from validation / test sets.

    Expected shape: ``(samples, channels, time)``, PyTorch convention.
    2-D input ``(samples, time)`` is auto-promoted to ``(samples, 1, time)``.

    Args:
        x_train: Training tensor.
        x_val:   Validation tensor.
        x_test:  Test tensor.
        norm_type: One of:

            * ``'global'``       - single mean / std across all dims.
            * ``'per-channel'``  - one mean / std per channel (dim 1).
            * ``'per-timestep'`` - one mean / std per time step (dim 2).

    Returns:
        ``(x_train_scaled, x_val_scaled, x_test_scaled)`` as float tensors.
    """
    def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1) if x.dim() == 2 else x

    # Accept numpy arrays gracefully
    def _to_tensor(x) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.array(x, dtype=np.float32))
        return x.float()

    x_train = _ensure_3d(_to_tensor(x_train))
    x_val   = _ensure_3d(_to_tensor(x_val))
    x_test  = _ensure_3d(_to_tensor(x_test))

    reduce: dict[str, tuple[int, ...]] = {
        "global":       (0, 1, 2),   # scalar
        "per-channel":  (0, 2),      # -> [1, C, 1]
        "per-timestep": (0, 1),      # -> [1, 1, T]
    }
    if norm_type not in reduce:
        raise ValueError(
            f"Unsupported norm_type '{norm_type}'. "
            f"Choose from: {list(reduce)}."
        )

    dims = reduce[norm_type]
    train_mean = x_train.mean(dim=dims, keepdim=True)
    train_std  = x_train.std(dim=dims,  keepdim=True).clamp(min=1e-8)

    x_train_s = (x_train - train_mean) / train_std
    x_val_s   = (x_val   - train_mean) / train_std
    x_test_s  = (x_test  - train_mean) / train_std

    print(f"[normalize_data]  norm_type={norm_type!r}")
    for name, t in (("train", x_train_s), ("val", x_val_s), ("test", x_test_s)):
        print(f"  {name:<6}: shape={tuple(t.shape)}  "
              f"mean={t.mean().item():.4f}  std={t.std().item():.4f}")

    return x_train_s, x_val_s, x_test_s


def compute_reliability(logits: torch.Tensor) -> dict:
    """Compute per-sample reliability from raw model logits.

    Three complementary measures are returned so callers can pick the one
    most appropriate for their task.

    Args:
        logits: ``(batch, n_classes)`` raw output before softmax.

    Returns:
        Dict with keys:

        * ``confidence``          - max probability; shape ``(batch,)``
        * ``reliability_entropy`` - ``1 - normalised_entropy``; shape ``(batch,)``
        * ``margin``              - top-1 minus top-2 probability; shape ``(batch,)``
        * ``predicted``           - argmax class indices; shape ``(batch,)``
        * ``probs``               - full softmax distribution; shape ``(batch, C)``

        All values are on CPU. High values mean a more reliable prediction.
    """
    probs = F.softmax(logits.detach().cpu(), dim=1)
    n_classes = probs.shape[1]

    confidence, predicted = probs.max(dim=1)

    entropy     = -(probs * probs.clamp(min=1e-9).log()).sum(dim=1)
    max_entropy = torch.log(torch.tensor(float(n_classes)))
    rel_entropy = 1.0 - (entropy / max_entropy)

    top2   = probs.topk(2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]

    return {
        "confidence":          confidence,
        "reliability_entropy": rel_entropy,
        "margin":              margin,
        "predicted":           predicted,
        "probs":               probs,
    }


def weighted_geometric_mean_score(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
) -> float:
    """Weighted geometric mean of per-class (sensitivity x specificity)^0.5.

    Uses a vectorised confusion matrix instead of a Python loop.

    Args:
        y_true: 1-D integer ground-truth labels.
        y_pred: 1-D integer predicted labels.

    Returns:
        Scalar float in [0, 1].
    """
    y_true = torch.as_tensor(y_true, dtype=torch.long)
    y_pred = torch.as_tensor(y_pred, dtype=torch.long)

    n_classes = int(y_true.max().item()) + 1
    cm = torch.zeros(n_classes, n_classes, dtype=torch.float64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Vectorised per-class TP, FP, FN, TN
    tp = cm.diag()
    fn = cm.sum(dim=1) - tp      # row sums minus diagonal
    fp = cm.sum(dim=0) - tp      # col sums minus diagonal
    tn = cm.sum() - (tp + fp + fn)

    sensitivity = tp / (tp + fn).clamp(min=1e-9)
    specificity = tn / (tn + fp).clamp(min=1e-9)
    gmean = (sensitivity * specificity).sqrt().mean().item()
    return gmean


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_progress(
    title: str,
    label: str,
    train_results: list[float],
    val_results: list[float],
    yscale: str = "linear",
    save_path: Optional[str] = None,
    extra_pt=None,
    extra_pt_label: Optional[str] = None,
) -> None:
    """Plot train vs validation metric curves over epochs.

    Args:
        title: Plot title.
        label: Y-axis label.
        train_results: Per-epoch training metric values.
        val_results: Per-epoch validation metric values.
        yscale: Matplotlib y-axis scale (``'linear'``, ``'log'``, etc.).
        save_path: If given, save the figure to this path.
        extra_pt: Optional ``(x, y)`` point to highlight (e.g. best epoch).
        extra_pt_label: Legend label for ``extra_pt``.
    """
    epoch_array = np.arange(1, len(train_results) + 1)

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(epoch_array, train_results,
            color="#378ADD", lw=2, marker="o", ms=5, label="Train")
    ax.plot(epoch_array, val_results,
            color="#D85A30", lw=2, marker="o", ms=5,
            linestyle="--", dashes=(6, 3), label="Validation")

    if extra_pt is not None:
        ax.scatter(*extra_pt, s=120, color="black", zorder=5,
                   label=extra_pt_label)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#888780")

    ax.grid(True, color="#C0C0BCA2", linewidth=0.5, zorder=0)
    ax.set_xlabel("Epoch",  color="#888780", fontsize=12)
    ax.set_ylabel(label,    color="#888780", fontsize=12)
    ax.set_yscale(yscale)
    ax.tick_params(colors="#888780", labelsize=10)
    ax.set_title(title, fontsize=13, fontweight="500", pad=12)
    ax.legend(frameon=True, fontsize=10, labelcolor="#5F5E5A", loc="best")

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def get_metrics_from_confusion_matrix(
    cm: np.ndarray, chosen_index: int
) -> tuple[float, float, float]:
    """Extract recall, precision, and accuracy for one class from a confusion matrix.

    Args:
        cm: Square numpy confusion matrix (rows = true, cols = predicted).
        chosen_index: Class index to inspect.

    Returns:
        ``(recall, precision, accuracy)`` as floats in [0, 1].
    """
    total = cm.sum()
    i = chosen_index
    tp = cm[i, i]
    recall    = tp / (cm[i, :].sum() + 1e-9)   # TP / (TP + FN), row i
    precision = tp / (cm[:, i].sum() + 1e-9)   # TP / (TP + FP), col i
    tn = total - cm[i, :].sum() - cm[:, i].sum() + tp
    acc = (tp + tn) / (total + 1e-9)
    return float(recall), float(precision), float(acc)


def check_precision_recall_accuracy(
    cm: np.ndarray, all_classes: list
) -> None:
    """Print recall, precision, and accuracy for every class.

    Args:
        cm: Confusion matrix (numpy array).
        all_classes: Ordered list of class names.
    """
    for i, cls in enumerate(all_classes):
        recall, precision, acc = get_metrics_from_confusion_matrix(cm, i)
        print(f"{cls}: recall={recall:.4f}  precision={precision:.4f}  accuracy={acc:.4f}")


def visualize_confusion_matrix(
    cm: np.ndarray,
    classes: list,
    correct: int,
    total: int,
    path: Optional[str] = None,
) -> None:
    """Plot a row-normalised confusion matrix heatmap.

    Args:
        cm: Raw (unnormalised) confusion matrix, shape (C, C).
        classes: Ordered list of class-name strings.
        correct: Number of correctly classified samples (for title).
        total: Total number of samples (for title).
        path: If provided, save the figure here.
    """
    cm_float = cm.astype(float)
    cm_norm  = cm_float / (cm_float.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        vmin=0.0, vmax=1.0, cmap="coolwarm",
        ax=ax, linewidths=0.5, linecolor="white",
    )

    overall_acc = correct / (total + 1e-9)
    ax.set_xlabel("Predicted",  size=18)
    ax.set_ylabel("True",       size=18)
    ax.set_title(f"Confusion Matrix  (accuracy = {overall_acc:.4f})", size=18)
    ax.xaxis.set_ticklabels(classes, size=13, rotation=45, ha="right")
    ax.yaxis.set_ticklabels(classes, size=13, rotation=0)

    plt.tight_layout()
    if path:
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Accuracy: {correct}/{total} = {overall_acc:.4f}")


# ---------------------------------------------------------------------------
# TrainTransformer
# ---------------------------------------------------------------------------

class TrainTransformer:
    """Training harness for time-series classification models.

    Handles train / validation / test loops, early stopping, LR scheduling,
    and all metric collection in one place.

    Args:
        model: ``nn.Module`` to train.
        optimizer: PyTorch optimiser.
        criterion: Loss function (e.g. ``nn.CrossEntropyLoss``).
        early_stopper: ``EarlyStopper`` instance (or ``None`` to disable).
        scheduler: LR scheduler (or ``None``).
        device: ``torch.device`` for model and data.
        params: Configuration dict.  Required keys:

            * ``n_classes``    (int)
            * ``num_epochs``   (int)
            * ``batch_size``   (int)
            * ``test_size``    (float), fraction held out for test
            * ``val_size``     (float), fraction of train held out for val
            * ``norm_type``    (str), passed to ``normalize_data``
            * ``num_cpus``     (int), DataLoader worker count
            * ``verbose``      (bool)

            Optional keys:

            * ``reliability_threshold`` (float, default 0.6)
            * ``early_stop_warmup``     (int,   default 10), epochs before
              early stopping is evaluated

    Note:
        If ``optimizer`` is a Muon-family optimizer (detected by class name
        or ``use_muon`` param-group tags), ``__init__`` ensures a dummy
        single-process ``torch.distributed`` group exists. Muon's
        internals call ``dist.get_world_size()`` unconditionally even for
        single-GPU / single-process runs, which otherwise raises
        ``ValueError: Default process group has not been initialized``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        early_stopper,
        scheduler,
        device: torch.device,
        params: dict,
    ):
        self.model         = model
        self.optimizer     = optimizer
        self.criterion     = criterion
        self.early_stopper = early_stopper
        self.scheduler     = scheduler
        self.device        = device
        self.params        = params

        # Muon requires a torch.distributed process group even for
        # single-process runs: set up a dummy one now, before the first
        # optimizer.step() call, regardless of which script built
        # `optimizer`.
        if _is_muon_optimizer(self.optimizer):
            if not _ensure_dist_initialized_for_muon():
                warnings.warn(
                    "Optimizer looks like Muon but the dummy "
                    "torch.distributed process group could not be "
                    "initialized. optimizer.step() will likely raise "
                    "ValueError: Default process group has not been "
                    "initialized.",
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def generate_train_val_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple:
        """Split, stratify, and normalise raw arrays.

        Statistics for normalisation are always fitted on the training split
        only to prevent leakage.

        Args:
            x: Feature array of shape ``(N, ...)``.
            y: Label array of shape ``(N,)``.

        Returns:
            ``(x_train, x_val, x_test, y_train, y_val, y_test)`` as tensors.
        """
        test_size = self.params["test_size"]
        val_size  = self.params["val_size"]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=43,
            stratify=y, shuffle=True,
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=val_size, random_state=43,
            stratify=y_train, shuffle=True,
        )

        for split, labels in (("train", y_train), ("val", y_val), ("test", y_test)):
            counts = pd.Series(labels).value_counts().to_dict()
            print(f"  {split}: {counts}")

        norm_type = self.params["norm_type"]
        x_train_s, x_val_s, x_test_s = normalize_data(
            x_train, x_val, x_test, norm_type
        )

        return x_train_s, x_val_s, x_test_s, y_train, y_val, y_test

    def init_data_loaders(
        self,
        trainset: torch.utils.data.Dataset,
        valset:   torch.utils.data.Dataset,
        testset:  torch.utils.data.Dataset,
        num_workers: Optional[int] = None,
    ) -> tuple:
        """Wrap datasets in DataLoaders.

        Args:
            trainset: Training ``Dataset``.
            valset:   Validation ``Dataset``.
            testset:  Test ``Dataset``.
            num_workers: Override ``params['num_cpus']`` if provided.

        Returns:
            ``(trainloader, valloader, testloader)``
        """
        batch_size = self.params["batch_size"]
        n_workers  = num_workers if num_workers is not None else self.params["num_cpus"]

        make = lambda ds, shuffle: torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        return make(trainset, True), make(valset, False), make(testset, False)

    # ------------------------------------------------------------------
    # Training / evaluation steps
    # ------------------------------------------------------------------

    def _eval_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        n_classes: int,
    ) -> tuple[float, float, float, float]:
        """Return (f1, precision, recall, gmean) from flat label tensors."""
        f1   = float(multiclass_f1_score(y_true, y_pred, num_classes=n_classes, average="macro"))
        pre  = float(multiclass_precision(y_true, y_pred, num_classes=n_classes, average="macro"))
        rec  = float(multiclass_recall(y_true, y_pred, num_classes=n_classes, average="macro"))
        gm   = float(weighted_geometric_mean_score(y_true, y_pred))
        return f1, pre, rec, gm

    def training_step(self, dataloader, _master_bar=None) -> tuple:
        """Run one full training epoch.

        Args:
            dataloader: Training ``DataLoader``.
            _master_bar: Unused (kept for API compatibility).

        Returns:
            ``(loss, accuracy, f1, precision, recall, gmean,
               mean_reliability, pct_neutral)``
        """
        n_classes  = self.params["n_classes"]
        threshold  = self.params.get("reliability_threshold", 0.6)

        losses, y_all, ypred_all   = [], [], []
        correct = total            = 0
        reliability_all            = []

        self.model.train()
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(x)                              # (B, C)
            loss   = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += len(y)
            losses.append(loss.item())

            reliability_all.append(_compute_reliability_batch(logits, n_classes))
            y_all.extend(y.cpu().tolist())
            ypred_all.extend(preds.cpu().tolist())

        y_t     = torch.tensor(y_all)
        ypred_t = torch.tensor(ypred_all)
        rel_all = torch.cat(reliability_all)

        mean_rel, pct_neutral, _ = _reliability_summary(
            rel_all, y_t, n_classes, threshold
        )
        f1, pre, rec, gm = self._eval_metrics(y_t, ypred_t, n_classes)

        return (
            float(np.mean(losses)),
            accuracy(correct, total),
            f1, pre, rec, gm,
            mean_rel, pct_neutral,
        )

    def validation_step(self, dataloader, _master_bar=None) -> tuple:
        """Evaluate the model on a validation (or test) dataloader.

        Args:
            dataloader: Validation ``DataLoader``.
            _master_bar: Unused (kept for API compatibility).

        Returns:
            ``(loss, accuracy, confusion_matrix, f1, precision, recall, gmean,
               mean_reliability, pct_neutral, per_class_reliability)``
        """
        n_classes = self.params["n_classes"]
        threshold = self.params.get("reliability_threshold", 0.6)

        losses, y_all, ypred_all = [], [], []
        correct = total          = 0
        cm        = torch.zeros(n_classes, n_classes)
        rel_all   = []
        labels_all = []

        self.model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                preds  = logits.argmax(dim=1)

                loss = self.criterion(logits, y)
                losses.append(loss.item())

                correct += (preds == y).sum().item()
                total   += len(y)

                # vectorised confusion-matrix update, no per-sample Python loop
                cm.index_put_(
                    (y.cpu().long(), preds.cpu().long()),
                    torch.ones(len(y)),
                    accumulate=True,
                )

                rel = _compute_reliability_batch(logits, n_classes)
                rel_all.append(rel)
                labels_all.append(y.cpu())

                y_all.extend(y.cpu().tolist())
                ypred_all.extend(preds.cpu().tolist())

        y_t      = torch.tensor(y_all)
        ypred_t  = torch.tensor(ypred_all)
        rel_cat  = torch.cat(rel_all)
        lbl_cat  = torch.cat(labels_all)

        mean_rel, pct_neutral, per_class_rel = _reliability_summary(
            rel_cat, lbl_cat, n_classes, threshold
        )
        f1, pre, rec, gm = self._eval_metrics(y_t, ypred_t, n_classes)

        return (
            float(np.mean(losses)),
            accuracy(correct, total),
            cm,
            f1, pre, rec, gm,
            mean_rel, pct_neutral, per_class_rel,
        )

    # Keep the old misspelled name as an alias so existing call sites don't break
    def validatation_step(self, dataloader, master_bar=None) -> tuple:
        """Deprecated alias for :meth:`validation_step`."""
        return self.validation_step(dataloader, master_bar)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def run_training(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader:   torch.utils.data.DataLoader,
    ) -> tuple:
        """Train for ``params['num_epochs']`` epochs with early stopping.

        Early stopping is skipped for the first ``params.get('early_stop_warmup', 10)``
        epochs so the model has time to leave its random initialisation.

        Returns:
            A tuple of history lists, one entry per epoch:

            ``(train_losses, val_losses,
               train_accs,   val_accs,
               confusion_matrix,
               train_f1s,    val_f1s,
               train_gmeans, val_gmeans,
               train_pres,   val_pres,
               train_recs,   val_recs,
               train_rels,   val_rels,
               train_neutrals, val_neutrals,
               val_class_rel)``
        """
        num_epochs = self.params["num_epochs"]
        verbose    = self.params["verbose"]
        threshold  = self.params.get("reliability_threshold", 0.6)
        warmup     = self.params.get("early_stop_warmup", 10)

        # metric history
        train_losses, val_losses     = [], []
        train_accs,   val_accs       = [], []
        train_f1s,    val_f1s        = [], []
        train_gmeans, val_gmeans     = [], []
        train_pres,   val_pres       = [], []
        train_recs,   val_recs       = [], []
        train_rels,   val_rels       = [], []
        train_neutrals, val_neutrals = [], []
        confusion_matrix             = None
        val_class_rel                = {}

        start = time.time()
        epoch_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")

        for epoch in epoch_bar:

            # train
            (tr_loss, tr_acc, tr_f1, tr_pre, tr_rec, tr_gm,
             tr_rel, tr_neut) = self.training_step(train_dataloader)

            # validate
            (vl_loss, vl_acc, confusion_matrix, vl_f1, vl_pre, vl_rec,
             vl_gm, vl_rel, vl_neut, val_class_rel) = self.validation_step(val_dataloader)

            # record
            train_losses.append(tr_loss);  val_losses.append(vl_loss)
            train_accs.append(tr_acc);     val_accs.append(vl_acc)
            train_f1s.append(tr_f1);       val_f1s.append(vl_f1)
            train_gmeans.append(tr_gm);    val_gmeans.append(vl_gm)
            train_pres.append(tr_pre);     val_pres.append(vl_pre)
            train_recs.append(tr_rec);     val_recs.append(vl_rec)
            train_rels.append(tr_rel);     val_rels.append(vl_rel)
            train_neutrals.append(tr_neut); val_neutrals.append(vl_neut)

            # LR scheduler
            if self.scheduler is not None:
                self.scheduler.step(vl_loss)
            lr = self.optimizer.param_groups[0]["lr"]

            # progress bar
            epoch_bar.set_postfix(
                tr_loss=f"{tr_loss:.4f}", vl_loss=f"{vl_loss:.4f}",
                tr_acc =f"{tr_acc:.3f}",  vl_acc =f"{vl_acc:.3f}",
                tr_rel =f"{tr_rel:.3f}",  vl_rel =f"{vl_rel:.3f}",
                neutral=f"{vl_neut:.1f}%",
                lr=f"{lr:.2e}",
            )

            if verbose:
                print(
                    f"Epoch {epoch+1:4d} | "
                    f"loss {tr_loss:.4f}/{vl_loss:.4f} | "
                    f"acc {tr_acc:.3f}/{vl_acc:.3f} | "
                    f"f1 {tr_f1:.3f}/{vl_f1:.3f} | "
                    f"rel {tr_rel:.3f}/{vl_rel:.3f} | "
                    f"neutral {vl_neut:.1f}% | lr {lr:.2e}"
                )
                low = {c: r for c, r in val_class_rel.items()
                       if not np.isnan(r) and r < threshold}
                if low:
                    print(f"  Low-reliability classes: {low}")

            # early stopping (after warmup)
            if epoch >= warmup and self.early_stopper is not None:
                self.early_stopper.update(vl_loss, self.model)
                if self.early_stopper.early_stop:
                    print(f"\nEarly stopping at epoch {epoch + 1}.")
                    self.model = self.early_stopper.load_checkpoint(self.model)
                    break

        elapsed = int(np.round(time.time() - start))
        print(f"\nFinished training after {elapsed} seconds.")

        return (
            train_losses, val_losses,
            train_accs,   val_accs,
            confusion_matrix,
            train_f1s,    val_f1s,
            train_gmeans, val_gmeans,
            train_pres,   val_pres,
            train_recs,   val_recs,
            train_rels,   val_rels,
            train_neutrals, val_neutrals,
            val_class_rel,
        )

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------

    def testing_step(self, test_loader: torch.utils.data.DataLoader) -> tuple:
        """Evaluate on the held-out test set and print a summary report.

        Delegates to :meth:`validation_step` to avoid duplicated logic, then
        adds the formatted test report.

        Args:
            test_loader: Test ``DataLoader``.

        Returns:
            ``(correct, total, accuracy, f1, gmean, precision, recall,
               mean_reliability, pct_neutral, per_class_reliability,
               confusion_matrix)``
        """
        threshold = self.params.get("reliability_threshold", 0.6)

        (loss, acc, cm, f1, pre, rec, gm,
         mean_rel, pct_neut, per_class_rel) = self.validation_step(test_loader)

        # Back-calculate correct / total from accuracy and confusion matrix
        total   = int(cm.sum().item())
        correct = int(cm.diag().sum().item())

        print(f'\n{"-"*55}')
        print(f'  TEST RESULTS')
        print(f'{"-"*55}')
        print(f'  Accuracy  : {acc:.4f}  ({correct}/{total})')
        print(f'  F1        : {f1:.4f}')
        print(f'  Precision : {pre:.4f}')
        print(f'  Recall    : {rec:.4f}')
        print(f'  G-Mean    : {gm:.4f}')
        print(f'  Reliability (mean) : {mean_rel:.4f}')
        print(f'  Neutral zone       : {pct_neut:.1f}%  (threshold={threshold})')
        print(f'  Per-class reliability:')
        for c, rel in per_class_rel.items():
            flag = "  <- LOW" if (not np.isnan(rel) and rel < threshold) else ""
            print(f'    class {c:2d}: {rel:.4f}{flag}')
        print(f'{"-"*55}')

        return (
            correct, total, acc,
            f1, gm, pre, rec,
            mean_rel, pct_neut, per_class_rel,
            cm,
        )

    def get_confusion_matrix(
        self,
        test_loader: torch.utils.data.DataLoader,
        view_cm: bool = False,
    ) -> np.ndarray:
        """Return the raw confusion matrix for ``test_loader`` as a numpy array.

        For full metrics use :meth:`testing_step` instead; this method is
        provided as a quick stand-alone CM helper.

        Args:
            test_loader: DataLoader to evaluate.
            view_cm: If ``True``, print the matrix before returning.

        Returns:
            Confusion matrix as a numpy array of shape ``(C, C)``.
        """
        # validation_step already builds the CM; reuse it
        _, _, cm, *_ = self.validation_step(test_loader)
        cm_np = cm.numpy()
        if view_cm:
            print(cm_np)
        return cm_np

