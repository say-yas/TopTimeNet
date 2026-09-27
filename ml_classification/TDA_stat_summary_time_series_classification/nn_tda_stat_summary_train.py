"""
nn_tda_stat_summary_train.py
Training harness for TDAEnd2EndNet.

Notes on model.train() / model.eval() discipline
-------------------------------------------------
train() and eval() are each called once before their respective loop,
never inside it. This is important for the 6 BatchNorm1d layers present
in TDAEnd2EndNet (5 in GroupProjector, one per feature group, plus 1 in
ClassHead): calling train() mid-epoch would reset all running mean/var
estimates.

Notes on normalisation (norm_type)
------------------------------------
TDAEnd2EndNet has 5 per-group BatchNorm1d layers in GroupProjector, each
normalising its own feature group independently. This makes external
normalisation largely redundant and potentially harmful, since it changes
the raw signal geometry that TakensLayer and PointCloudStatsLayer rely
on. The recommended setting is "norm_type": "none".

Notes on the PI imager
------------------------
TDAEnd2EndNet.pi_layer is fitted lazily on the first forward pass. If
training is re-run (e.g. after a crash) or the model is reused, the old
imager may have been fitted on a different data distribution.
reset_pi_imager() is called at the start of run_training() to guarantee a
clean fit on the first training batch of each run.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torcheval.metrics.functional import (
    multiclass_f1_score, multiclass_recall, multiclass_precision,
)
from tqdm import tqdm


# ============================================================================
# TrainTDAStat
# ============================================================================

class TrainTDAStat:
    def __init__(self, model, optimizer, criterion,
                 early_stopper, scheduler, device, params):
        self.model         = model
        self.optimizer     = optimizer
        self.criterion     = criterion
        self.early_stopper = early_stopper
        self.scheduler     = scheduler
        self.device        = device
        self.params        = params

        # label smoothing
        ls = float(params.get("label_smoothing", 0.05))
        if ls > 0.0 and isinstance(criterion, nn.CrossEntropyLoss):
            if criterion.label_smoothing == 0.0:
                self.criterion = nn.CrossEntropyLoss(
                    weight          = criterion.weight,
                    label_smoothing = ls,
                )
                print(f"[TrainTDAStat] label_smoothing={ls} applied to CrossEntropyLoss")

        # gradient clipping
        self._grad_clip = params.get("grad_clip_norm", 1.0)

        # max entropy, on the correct device, for the reliability metric
        n_classes = params["n_classes"]
        self._max_entropy = torch.log(
            torch.tensor(float(n_classes), device=device)
        )

    # data preparation

    def generate_train_val_test(self, x, y):
        test_size = self.params["test_size"]
        val_size  = self.params["val_size"]
        norm_type = self.params["norm_type"]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=43,
            stratify=y, shuffle=True,
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=val_size, random_state=43,
            stratify=y_train, shuffle=True,
        )

        print(
            "counts, train:", pd.Series(y_train).value_counts().to_dict(),
            " val:",  pd.Series(y_val).value_counts().to_dict(),
            " test:", pd.Series(y_test).value_counts().to_dict(),
        )

        if norm_type == "per-timestep":
            warnings.warn(
                "norm_type='per-timestep' normalises each time step independently "
                "and distorts the temporal structure required by TakensLayer. "
                "It also corrupts the point-cloud geometry that "
                "PointCloudStatsLayer relies on (diameter, correlation dimension, "
                "and nearest-neighbour distances all change under per-timestep "
                "scaling). Use norm_type='none' with TDAEnd2EndNet, since the "
                "model's GroupProjector contains 5 per-group BatchNorm1d layers "
                "that handle normalisation of each TDA feature group independently.",
                UserWarning,
                stacklevel=2,
            )
        elif norm_type in ("none", "None", None):
            print(
                "[TrainTDAStat] norm_type='none', recommended for TDAEnd2EndNet. "
                "GroupProjector's per-group BatchNorm1d layers normalise each "
                "feature group internally, preserving raw signal geometry for "
                "TakensLayer and PointCloudStatsLayer."
            )

        x_tr, x_va, x_te = normalize_data(x_train, x_val, x_test, norm_type)
        print("shapes:", x_tr.shape, x_va.shape, x_te.shape)
        return x_tr, x_va, x_te, y_train, y_val, y_test

    def init_data_loaders(self, trainset, valset, testset, num_workers=None):
        batch_size = self.params["batch_size"]
        n_workers  = num_workers if num_workers is not None \
                     else self.params["num_cpus"]
        trainloader = DataLoader(trainset, batch_size=batch_size,
                                 shuffle=True,  num_workers=n_workers)
        valloader   = DataLoader(valset,   batch_size=batch_size,
                                 shuffle=False, num_workers=n_workers)
        testloader  = DataLoader(testset,  batch_size=batch_size,
                                 shuffle=False, num_workers=n_workers)
        return trainloader, valloader, testloader

    # training step

    def training_step(self, dataloader, master_bar):
        """
        One full training epoch.

        model.train() is called once before the loop. TDAEnd2EndNet
        contains 6 BatchNorm1d layers total (5 in GroupProjector, one per
        TDA feature group, plus 1 prepended to ClassHead), all outside the
        internal torch.no_grad() block, so they participate in the
        forward pass and accumulate running stats only when in train
        mode. Calling train() inside the loop would reset those stats
        mid-epoch.
        """
        self.model.train()   # once, before the loop

        epoch_loss, epoch_y, epoch_ypred = [], [], []
        epoch_correct, epoch_total       = 0, 0
        epoch_reliability                = []

        threshold = self.params.get("reliability_threshold", 0.6)
        n_classes = self.params["n_classes"]

        for x, y in dataloader:
            self.optimizer.zero_grad()

            y_pred = self.model(x.to(self.device))

            with torch.no_grad():
                probs   = torch.softmax(y_pred, dim=1)
                entropy = -(probs * probs.clamp(min=1e-9).log()).sum(dim=1)
                rel     = 1.0 - entropy / self._max_entropy
                epoch_reliability.append(rel.cpu())

            epoch_y.append(y.tolist())
            epoch_ypred.append(y_pred.argmax(dim=1).cpu().tolist())
            epoch_correct += (y.to(self.device) == y_pred.argmax(dim=1)).sum()
            epoch_total   += len(y)

            loss = self.criterion(y_pred, y.to(self.device))
            loss.backward()

            if self._grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)

            self.optimizer.step()
            epoch_loss.append(loss.item())

        epoch_y     = torch.tensor(np.array(sum(epoch_y,     [])).flatten())
        epoch_ypred = torch.tensor(np.array(sum(epoch_ypred, [])).flatten())

        all_rel          = torch.cat(epoch_reliability)
        mean_reliability = all_rel.mean().item()
        pct_neutral      = (all_rel < threshold).float().mean().item() * 100

        f1score   = multiclass_f1_score(epoch_y, epoch_ypred,
                                        num_classes=n_classes, average="macro")
        recall    = multiclass_recall(epoch_y, epoch_ypred,
                                      num_classes=n_classes, average="macro")
        precision = multiclass_precision(epoch_y, epoch_ypred,
                                         num_classes=n_classes, average="macro")
        gmean     = weighted_geometric_mean_score(epoch_y, epoch_ypred)

        return (
            float(np.mean(epoch_loss)),
            accuracy(epoch_correct, epoch_total),
            float(f1score.cpu()),
            float(precision.cpu()),
            float(recall.cpu()),
            float(gmean),
            mean_reliability,
            pct_neutral,
        )

    # validation step

    def validatation_step(self, dataloader, master_bar):
        """
        Validation epoch. model.eval() is called once before the loop.
        All 6 BatchNorm1d layers switch to using stored running
        statistics (not batch statistics), giving stable, consistent
        normalisation.
        """
        n_classes = self.params["n_classes"]
        threshold = self.params.get("reliability_threshold", 0.6)

        epoch_loss, epoch_y, epoch_ypred = [], [], []
        epoch_correct, epoch_total       = 0, 0
        confusion_matrix                 = torch.zeros(n_classes, n_classes)
        epoch_reliability, epoch_rel_lbl = [], []

        self.model.eval()   # once, before the loop
        with torch.no_grad():
            for x, y in dataloader:
                y_pred = self.model(x.to(self.device))

                probs   = torch.softmax(y_pred, dim=1)
                entropy = -(probs * probs.clamp(min=1e-9).log()).sum(dim=1)
                rel     = 1.0 - entropy / self._max_entropy
                epoch_reliability.append(rel.cpu())
                epoch_rel_lbl.append(y.cpu())

                epoch_y.append(y.tolist())
                preds = y_pred.argmax(dim=1)
                epoch_ypred.append(preds.cpu().tolist())
                epoch_correct += (y.to(self.device) == preds).sum()
                epoch_total   += len(y)

                # vectorised confusion-matrix update, no per-sample Python loop
                confusion_matrix.index_put_(
                    (y.cpu().long(), preds.cpu().long()),
                    torch.ones(len(y)),
                    accumulate=True,
                )

                loss = self.criterion(y_pred, y.to(self.device))
                epoch_loss.append(loss.item())

        epoch_y     = torch.tensor(np.array(sum(epoch_y,     [])).flatten())
        epoch_ypred = torch.tensor(np.array(sum(epoch_ypred, [])).flatten())

        all_rel      = torch.cat(epoch_reliability)
        all_rel_lbl  = torch.cat(epoch_rel_lbl)
        mean_rel     = all_rel.mean().item()
        pct_neutral  = (all_rel < threshold).float().mean().item() * 100

        per_class_rel = {}
        for c in range(n_classes):
            mask = all_rel_lbl == c
            per_class_rel[c] = (
                all_rel[mask].mean().item() if mask.sum() > 0 else float("nan")
            )

        f1score   = multiclass_f1_score(epoch_y, epoch_ypred,
                                        num_classes=n_classes, average="macro")
        recall    = multiclass_recall(epoch_y, epoch_ypred,
                                      num_classes=n_classes, average="macro")
        precision = multiclass_precision(epoch_y, epoch_ypred,
                                         num_classes=n_classes, average="macro")
        gmean     = weighted_geometric_mean_score(epoch_y, epoch_ypred)

        return (
            float(np.mean(epoch_loss)),
            accuracy(epoch_correct, epoch_total),
            confusion_matrix,
            float(f1score.cpu()),
            float(precision.cpu()),
            float(recall.cpu()),
            float(gmean),
            mean_rel,
            pct_neutral,
            per_class_rel,
        )

    # full training loop

    def run_training(self, train_dataloader, val_dataloader):
        num_epochs = self.params["num_epochs"]
        verbose    = self.params["verbose"]
        threshold  = self.params.get("reliability_threshold", 0.6)

        # reset the PI imager so it refits on this run's training data
        if hasattr(self.model, "reset_pi_imager"):
            self.model.reset_pi_imager()
            print("[TrainTDAStat] PILayer imager reset, will refit on first batch.")

        fusion_name = getattr(self.model, "_fusion_name", None)
        if fusion_name:
            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(
                f"[TrainTDAStat] fusion='{fusion_name}'  "
                f"trainable_params={n_params:,}  "
                f"epochs={num_epochs}"
            )

        start_time = time.time()

        train_losses, val_losses = [], []
        train_accs,   val_accs   = [], []
        train_f1s,    val_f1s    = [], []
        train_gmeans, val_gmeans = [], []
        train_pres,   val_pres   = [], []
        train_recs,   val_recs   = [], []
        train_rels,   val_rels   = [], []
        train_neus,   val_neus   = [], []

        epoch_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")

        for epoch in epoch_bar:
            (tr_loss, tr_acc, tr_f1, tr_pre, tr_rec,
             tr_gm, tr_rel, tr_neu) = self.training_step(train_dataloader, epoch_bar)

            (vl_loss, vl_acc, confusion_matrix, vl_f1, vl_pre, vl_rec,
             vl_gm, vl_rel, vl_neu, val_cls_rel) = self.validatation_step(
                 val_dataloader, epoch_bar)

            train_losses.append(tr_loss);  val_losses.append(vl_loss)
            train_accs.append(tr_acc);     val_accs.append(vl_acc)
            train_f1s.append(tr_f1);       val_f1s.append(vl_f1)
            train_gmeans.append(tr_gm);    val_gmeans.append(vl_gm)
            train_pres.append(tr_pre);     val_pres.append(vl_pre)
            train_recs.append(tr_rec);     val_recs.append(vl_rec)
            train_rels.append(tr_rel);     val_rels.append(vl_rel)
            train_neus.append(tr_neu);     val_neus.append(vl_neu)

            if self.scheduler:
                self.scheduler.step(vl_loss)
            lr = self.optimizer.param_groups[0]["lr"]

            epoch_bar.set_postfix(
                tr_loss=f"{tr_loss:.4f}", vl_loss=f"{vl_loss:.4f}",
                tr_acc=f"{tr_acc:.3f}",  vl_acc=f"{vl_acc:.3f}",
                tr_rel=f"{tr_rel:.3f}",  vl_rel=f"{vl_rel:.3f}",
                neutral=f"{vl_neu:.1f}%", lr=f"{lr:.2e}",
            )

            if verbose:
                print(
                    f"Epoch {epoch+1:4d} | "
                    f"loss {tr_loss:.4f}/{vl_loss:.4f} | "
                    f"acc {tr_acc:.3f}/{vl_acc:.3f} | "
                    f"f1 {tr_f1:.3f}/{vl_f1:.3f} | "
                    f"rel {tr_rel:.3f}/{vl_rel:.3f} | "
                    f"neutral {vl_neu:.1f}% | lr {lr:.2e}"
                )
                low = {c: r for c, r in val_cls_rel.items()
                       if r < threshold and not np.isnan(r)}
                if low:
                    print(f"  Low-reliability classes: {low}")

            if epoch > 10 and self.early_stopper:
                self.early_stopper.update(vl_loss, self.model)
                if self.early_stopper.early_stop:
                    print(f"\nEarly stopping at epoch {epoch+1}.")
                    self.model = self.early_stopper.load_checkpoint(self.model)
                    break

        elapsed = int(np.round(time.time() - start_time))
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
            train_neus,   val_neus,
            val_cls_rel,
        )

    # test evaluation

    def testing_step(self, test_loader):
        n_classes = self.params["n_classes"]
        threshold = self.params.get("reliability_threshold", 0.6)

        correct, total           = 0, 0
        y_all, ypred_all         = [], []
        epoch_rel, epoch_rel_lbl = [], []
        confusion_matrix         = torch.zeros(n_classes, n_classes)

        self.model.eval()   # once, before the loop
        with torch.no_grad():
            for x, labels in test_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                outputs   = self.model(x)
                predicted = outputs.argmax(dim=1)

                probs   = torch.softmax(outputs, dim=1)
                entropy = -(probs * probs.clamp(min=1e-9).log()).sum(dim=1)
                rel     = 1.0 - entropy / self._max_entropy
                epoch_rel.append(rel.cpu())
                epoch_rel_lbl.append(labels.cpu())

                y_all.append(labels.cpu().tolist())
                ypred_all.append(predicted.cpu().tolist())
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

                # vectorised confusion-matrix update, no per-sample Python loop
                confusion_matrix.index_put_(
                    (labels.cpu().long(), predicted.cpu().long()),
                    torch.ones(len(labels)),
                    accumulate=True,
                )

        y     = torch.tensor(np.array(sum(y_all,     [])).flatten())
        ypred = torch.tensor(np.array(sum(ypred_all, [])).flatten())

        all_rel     = torch.cat(epoch_rel)
        all_rel_lbl = torch.cat(epoch_rel_lbl)
        mean_rel    = all_rel.mean().item()
        pct_neutral = (all_rel < threshold).float().mean().item() * 100

        per_class_rel = {}
        for c in range(n_classes):
            mask = all_rel_lbl == c
            per_class_rel[c] = (
                all_rel[mask].mean().item() if mask.sum() > 0 else float("nan")
            )

        f1score   = multiclass_f1_score(y, ypred, num_classes=n_classes, average="macro")
        recall    = multiclass_recall(y, ypred,   num_classes=n_classes, average="macro")
        precision = multiclass_precision(y, ypred, num_classes=n_classes, average="macro")
        gmean     = weighted_geometric_mean_score(y, ypred)

        fusion_info = ""
        if hasattr(self.model, "_fusion_name"):
            fusion_info = f"  Fusion     : {self.model._fusion_name}\n"

        print(f'\n{"-"*55}')
        print("  TEST RESULTS")
        print(f'{"-"*55}')
        print(fusion_info, end="")
        print(f"  Accuracy  : {accuracy(correct, total):.4f}  ({correct}/{total})")
        print(f"  F1        : {float(f1score):.4f}")
        print(f"  Precision : {float(precision):.4f}")
        print(f"  Recall    : {float(recall):.4f}")
        print(f"  G-Mean    : {float(gmean):.4f}")
        print(f"  Reliability (mean) : {mean_rel:.4f}")
        print(f"  Neutral zone       : {pct_neutral:.1f}%  (threshold={threshold})")
        print("  Per-class reliability:")
        for c, r in per_class_rel.items():
            flag = "  <- LOW" if r < threshold else ""
            print(f"    class {c:2d}: {r:.4f}{flag}")
        print(f'{"-"*55}')

        return (
            correct, total,
            accuracy(correct, total),
            float(f1score.cpu()),
            float(gmean),
            float(precision.cpu()),
            float(recall.cpu()),
            mean_rel,
            pct_neutral,
            per_class_rel,
            confusion_matrix,
        )


# ============================================================================
# Standalone helper functions
# ============================================================================

def normalize_data(x_train, x_val, x_test, norm_type):
    """
    Normalise using training-set statistics only.
    Input shape: (N, 1, T); output shape identical.

    norm_type options
    ------------------
    "none"         : recommended for TDAEnd2EndNet. The model's
                     GroupProjector contains 5 per-group BatchNorm1d
                     layers (one per TDA feature group) that normalise
                     each group independently, preserving the raw signal
                     geometry that TakensLayer and PointCloudStatsLayer
                     depend on. External normalisation on top of this is
                     redundant and shifts the point-cloud geometry.

    "global"       : single mean/std over all elements of x_train.
                     Safe if you want a light scale correction before the
                     Takens embedding.

    "per-channel"  : mean/std per channel (averaged over samples and
                     time). Appropriate for multi-channel inputs where
                     channels have very different physical units.

    "per-timestep" : mean/std per time step. Avoid with TDAEnd2EndNet,
                     since it distorts temporal structure and point-cloud
                     geometry.
    """
    def _ensure_3d(x):
        return x.unsqueeze(1) if x.dim() == 2 else x

    x_train = _ensure_3d(x_train)
    x_val   = _ensure_3d(x_val)
    x_test  = _ensure_3d(x_test)

    if norm_type in ("none", "None", None):
        print(f"[normalize_data] norm_type=none (skipping, BN inside model)")
        print(f"  train {tuple(x_train.shape)}  val {tuple(x_val.shape)}  test {tuple(x_test.shape)}")
        return x_train, x_val, x_test

    if norm_type == "global":
        mu  = x_train.mean()
        std = x_train.std().clamp(min=1e-8)
    elif norm_type == "per-channel":
        mu  = x_train.mean(dim=(0, 2), keepdim=True)
        std = x_train.std(dim=(0, 2),  keepdim=True).clamp(min=1e-8)
    elif norm_type == "per-timestep":
        mu  = x_train.mean(dim=(0, 1), keepdim=True)
        std = x_train.std(dim=(0, 1),  keepdim=True).clamp(min=1e-8)
    else:
        raise ValueError(
            f"Unknown norm_type '{norm_type}'. "
            "Choose: 'none', 'global', 'per-channel', 'per-timestep'."
        )

    tr = (x_train - mu) / std
    va = (x_val   - mu) / std
    te = (x_test  - mu) / std

    print(f"[normalize_data] norm_type={norm_type}")
    print(f"  train {tuple(tr.shape)}  mean={tr.mean():.4f}  std={tr.std():.4f}")
    print(f"  val   {tuple(va.shape)}  test {tuple(te.shape)}")
    return tr, va, te


def data_to_tensor(data, labels):
    data   = np.array(data,   dtype=np.float32)
    labels = np.array(labels)
    return torch.utils.data.TensorDataset(
        torch.tensor(data),
        torch.tensor(labels),
    )


def accuracy(correct, total):
    return float(correct) / total


def plot_progress(title, label, train_results, val_results, yscale="linear",
                  save_path=None, extra_pt=None, extra_pt_label=None):
    epoch_array = np.arange(len(train_results)) + 1
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(epoch_array, train_results,
            color="#378ADD", lw=2, marker="o", ms=5, label="Train")
    ax.plot(epoch_array, val_results,
            color="#D85A30", lw=2, marker="o", ms=5,
            linestyle="--", dashes=(6, 3), label="Validation")

    if extra_pt is not None:
        ax.scatter(*extra_pt, s=120, color="black", zorder=5, label=extra_pt_label)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#888780")
    ax.grid(True, color="#C0C0BCA2", linewidth=0.5, zorder=0)
    ax.set_xlabel("Epoch", color="#888780", fontsize=12)
    ax.set_ylabel(label,   color="#888780", fontsize=12)
    ax.set_yscale(yscale)
    ax.tick_params(colors="#888780", labelsize=10)
    ax.set_title(title, fontsize=13, fontweight="500", pad=12)
    ax.legend(frameon=True, fontsize=10, labelcolor="#5F5E5A", loc="upper right")
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_confusion_matrix(cm, classes, correct, total, path=None):
    cm_float = cm.astype("float")
    cm_norm  = cm_float / (cm_float.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax  = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", vmin=0, vmax=1,
                cmap="coolwarm", ax=ax, linewidths=0.5, linecolor="white")
    overall_acc = correct / total if total > 0 else 0.0
    ax.set_xlabel("Predicted", size=18)
    ax.set_ylabel("True",      size=18)
    ax.set_title(f"Confusion Matrix  (accuracy = {overall_acc:.4f})", size=18)
    ax.xaxis.set_ticklabels(classes, size=13, rotation=45, ha="right")
    ax.yaxis.set_ticklabels(classes, size=13, rotation=0)
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Accuracy: {correct}/{total} = {overall_acc:.4f}")


def weighted_geometric_mean_score(y_true, y_pred):
    # safe conversion: works whether inputs are tensors or numpy arrays
    if isinstance(y_true, torch.Tensor): y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.numpy()
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    classes = np.unique(y_true)
    n       = len(classes)
    cm      = np.zeros((n, n), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1
    sensitivity, specificity = [], []
    for i in range(n):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        sensitivity.append(TP / (TP + FN) if (TP + FN) else 0)
        specificity.append(TN / (TN + FP) if (TN + FP) else 0)
    return float(np.mean(np.sqrt(
        np.array(sensitivity) * np.array(specificity)
    )))


def compute_reliability(logits: torch.Tensor) -> dict:
    probs     = F.softmax(logits, dim=1)
    n_classes = probs.shape[1]
    conf, predicted = probs.max(dim=1)
    entropy   = -(probs * probs.clamp(min=1e-9).log()).sum(dim=1)
    max_ent   = torch.log(torch.tensor(float(n_classes), device=logits.device))
    rel_ent   = 1.0 - entropy / max_ent
    top2      = probs.topk(2, dim=1).values
    margin    = top2[:, 0] - top2[:, 1]
    return {
        "confidence"          : conf,
        "reliability_entropy" : rel_ent,
        "margin"              : margin,
        "predicted"           : predicted,
        "probs"               : probs,
    }
