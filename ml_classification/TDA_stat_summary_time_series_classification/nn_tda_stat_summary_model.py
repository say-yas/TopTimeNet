"""
nn_tda_stat_summary_model.py
End-to-end TDA model: raw segmented time series -> TDA features -> classify.

All fusion strategies and the full model live in this single file.

Architecture overview
----------------------
  (B, 1, T)
    -> TakensLayer              (no params) -> list of (N, d) point clouds
         -> PointCloudStatsLayer            -> (B, 4)           [group 0]
         -> RipserPHLayer       (no params) -> persistence diagrams
              -> PersistenceEntropyLayer    -> (B, n_hom)       [group 1]
              -> LifetimeStatsLayer         -> (B, n_hom*5)     [group 2]
              -> BettiCurveLayer (relative, scale-invariant)
                   -> TDABettiExtractor     -> (B, n_hom*6)     [group 3]
              -> PILayer
                   -> TDAPIExtractor        -> (B, n_hom*7)     [group 4]
    -> GroupProjector: per-group BatchNorm1d + Linear -> 5 x (B, D)
    -> Fusion layer (selectable, see below)
         All methods: List[Tensor(B, D)] -> Tensor(B, 5*D)
    -> ClassHead (BatchNorm1d -> MLP) -> (B, n_classes)

Fusion strategies (fusion= kwarg or config.json "fusion" key)
----------------------------------------------------------------
  "bilinear"     BilinearPairwiseFusion    ~330 params    cheapest
  "gated"        GatedResidualFusion       ~10k params
  "linear_attn"  LinearAttentionFusion     ~3-7k params   no softmax
  "low_rank"     LowRankCrossGroupMLP      ~640-5k params recommended
  "mgta"         MultiGroupTokenAttention  ~17k params    most expressive

  All give every group full access to every other group.

Quick start
-----------
  from nn_tda_end2end_model import TDAEnd2EndNet

  model = TDAStatSummaryNet.standard(n_classes=2)                  # low_rank
  model = TDAStatSummaryNet.standard(n_classes=2, fusion="mgta")   # transformer
  model = TDAStatSummaryNet.from_config(cfg, n_classes=2)

  logits = model(x)   # x: (B, 1, T)

Parameter counts (N=5 groups, D=32, r=8, n_heads=4, n_layers=2)
------------------------------------------------------------------
  All TDA layers (fixed)    :      0
  GroupProjector            :  ~2 000
  bilinear fusion           :    330
  linear_attn fusion (1L)   :  ~3 500
  low_rank fusion   (r=8)   :  ~2 560
  gated fusion              : ~10 500
  mgta fusion       (2L)    : ~17 700
  ClassHead (64->32->n)     :  ~7 000
  TOTAL standard/low_rank   : ~12 000
  TOTAL standard/mgta       : ~27 000

Dependencies: pip install ripser persim
"""

from __future__ import annotations

import os
import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ripser import ripser
from persim import PersistenceImager

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================================
# Shared utilities
# ============================================================================

def _get_activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "leaky_relu": nn.LeakyReLU(0.1)}[name]

def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.zero_()


# ============================================================================
# Fusion layers
# All accept: tokens: List[Tensor(B, D)] -> Tensor(B, N*D)
# ============================================================================

# A. Bilinear Pairwise Fusion

class BilinearPairwiseFusion(nn.Module):
    """
    For every ordered pair (i, j) with i < j, learn a scalar interaction
    weight projected onto a D-dimensional interaction vector, scattered
    back to both tokens as a residual.

        s_ij  = tanh( sum(token_i * w_ij * token_j) )      scalar
        delta = s_ij * v_ij                                (D,)
        token_i += delta ; token_j += delta

    Total params: N*(N-1)/2 * 2*D   (N=5, D=32 -> 10 * 64 = 640)

    Cheapest true pairwise interaction. No softmax, no QKV, no
    positional embedding. LayerNorm after the residual add stabilises
    training.
    """

    def __init__(self, n_groups: int, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.n_groups  = n_groups
        self.embed_dim = embed_dim
        pairs          = [(i, j) for i in range(n_groups) for j in range(i + 1, n_groups)]
        self.pairs     = pairs
        n_pairs        = len(pairs)
        self.pair_weights = nn.Parameter(torch.randn(n_pairs, embed_dim) * 0.01)
        self.pair_values  = nn.Parameter(torch.randn(n_pairs, embed_dim) * 0.01)
        self.norms        = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_groups)])
        self.drop         = nn.Dropout(dropout)
        self.out_dim      = n_groups * embed_dim

    def forward(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        residuals = [torch.zeros_like(t) for t in tokens]
        for idx, (i, j) in enumerate(self.pairs):
            w     = self.pair_weights[idx]                              # (D,)
            v     = self.pair_values[idx]                               # (D,)
            s     = torch.tanh((tokens[i] * w * tokens[j]).sum(-1, keepdim=True))  # (B,1)
            delta = self.drop(s * v)                                    # (B, D)
            residuals[i] = residuals[i] + delta
            residuals[j] = residuals[j] + delta
        return torch.cat(
            [self.norms[k](tokens[k] + residuals[k]) for k in range(self.n_groups)], dim=1
        )   # (B, N*D)


# B. Gated Residual Fusion

class GatedResidualFusion(nn.Module):
    """
    Each token reads a learned summary of all other tokens via a sigmoid
    gate.

        context_k = mean({token_j : j != k})          parameter-free
        gate_k    = sigmoid(W_gate_k . context_k)      (D x D params)
        value_k   = tanh(W_val_k . context_k)          (D x D params)
        out_k     = LayerNorm(token_k + gate_k * value_k)

    Total params: N * 2 * (D^2 + D)   (N=5, D=32 -> 10 560)

    The gate is soft: it closes to about 0 when context carries no useful
    signal, letting the token pass through unchanged. Interpretable via
    the gate_magnitudes() diagnostic.
    """

    def __init__(self, n_groups: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n_groups   = n_groups
        self.gate_proj  = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_groups)])
        self.value_proj = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_groups)])
        self.norms      = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_groups)])
        self.drop       = nn.Dropout(dropout)
        self.out_dim    = n_groups * embed_dim

    def forward(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(tokens, dim=1)   # (B, N, D)
        out = []
        for k in range(self.n_groups):
            mask    = [j for j in range(self.n_groups) if j != k]
            context = stacked[:, mask, :].mean(dim=1)                  # (B, D)
            gate    = torch.sigmoid(self.gate_proj[k](context))
            value   = torch.tanh(self.value_proj[k](context))
            out.append(self.norms[k](tokens[k] + self.drop(gate * value)))
        return torch.cat(out, dim=1)           # (B, N*D)

    def gate_magnitudes(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        """Diagnostic: gate magnitudes (B, N, D). High = strongly peer-modulated."""
        stacked = torch.stack(tokens, dim=1)
        gates   = []
        for k in range(self.n_groups):
            mask    = [j for j in range(self.n_groups) if j != k]
            context = stacked[:, mask, :].mean(dim=1)
            gates.append(torch.sigmoid(self.gate_proj[k](context)))
        return torch.stack(gates, dim=1)       # (B, N, D)


# C. Linear Attention Fusion

class LinearAttentionFusion(nn.Module):
    """
    Full QKV attention without the O(N^2) softmax, using the kernel
    trick:

        phi(x) = ELU(x) + 1     (positive, smooth, approximates exp)
        Attn   = phi(Q) (phi(K)^T V)  (compute K^T V first, O(N*D^2))

    For N=5 tokens this is mathematically equivalent to softmax attention
    but numerically simpler (no scaling, no stability tricks).

    Includes a pre-norm residual attention block plus a feedforward
    sublayer per layer.

    Total params approximately 4*D^2 (QKV + out) + 2*D*dim_ff (FFN) per
    layer. D=32, dim_ff=64, 1 layer gives about 7 000 params.
    """

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def __init__(self, n_groups: int, embed_dim: int,
                 n_heads: int = 4, n_layers: int = 1,
                 dim_ff: int = 0, dropout: float = 0.1):
        super().__init__()
        dim_ff  = dim_ff or 2 * embed_dim
        n_heads = max(h for h in range(1, n_heads + 1) if embed_dim % h == 0)
        self.n_groups  = n_groups
        self.embed_dim = embed_dim
        self.n_heads   = n_heads
        self.head_dim  = embed_dim // n_heads
        self.n_layers  = n_layers

        self.q_projs    = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(n_layers)])
        self.k_projs    = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(n_layers)])
        self.v_projs    = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(n_layers)])
        self.o_projs    = nn.ModuleList([nn.Linear(embed_dim, embed_dim)              for _ in range(n_layers)])
        self.ffn        = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, dim_ff), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(dim_ff, embed_dim), nn.Dropout(dropout))
            for _ in range(n_layers)
        ])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.ffn_norms  = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.drop       = nn.Dropout(dropout)
        self.out_dim    = n_groups * embed_dim

    def _layer(self, x: torch.Tensor, i: int) -> torch.Tensor:
        B, N, D = x.shape
        H, d    = self.n_heads, self.head_dim
        xn      = self.attn_norms[i](x)
        Q = self._phi(self.q_projs[i](xn).view(B, N, H, d).transpose(1, 2))  # (B,H,N,d)
        K = self._phi(self.k_projs[i](xn).view(B, N, H, d).transpose(1, 2))
        V =           self.v_projs[i](xn).view(B, N, H, d).transpose(1, 2)
        KtV  = torch.matmul(K.transpose(-2, -1), V)                   # (B,H,d,d)
        attn = torch.matmul(Q, KtV)                                    # (B,H,N,d)
        norm = (Q * K.sum(-2, keepdim=True)).sum(-1, keepdim=True).clamp(1e-6)
        attn = (attn / norm).transpose(1, 2).contiguous().view(B, N, D)
        x    = x + self.drop(self.o_projs[i](attn))
        x    = x + self.ffn[i](self.ffn_norms[i](x))
        return x

    def forward(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        x = torch.stack(tokens, dim=1)        # (B, N, D)
        for i in range(self.n_layers):
            x = self._layer(x, i)
        return x.flatten(1)                   # (B, N*D)


# D. Low-Rank Cross-Group MLP

class LowRankCrossGroupMLP(nn.Module):
    """
    Full cross-group interaction via a factored bottleneck MLP.

    Concatenates all tokens into (B, N*D), projects down to bottleneck
    rank r, back up to (B, N*D), and adds as a pre-norm residual.

        h   = GELU(W_down . x)     W_down: (N*D -> r)
        d   = W_up . h             W_up  : (r -> N*D)
        out = LayerNorm(x + d)

    Total params: 2 * N*D * r   (N=5, D=32, r=8 -> 2 560)
    Stack n_layers blocks for more depth (linear parameter growth).

    Recommended default: r=8, n_layers=1, giving 2 560 params.
    r=4 gives 1 280 params. r=16 gives 5 120 params (about linear-attn
    level).
    """

    def __init__(self, n_groups: int, embed_dim: int,
                 rank: int = 8, n_layers: int = 1,
                 dropout: float = 0.05, activation: str = "gelu"):
        super().__init__()
        flat_dim    = n_groups * embed_dim
        act         = _get_activation(activation)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(flat_dim, rank), act,
                          nn.Dropout(dropout), nn.Linear(rank, flat_dim), nn.Dropout(dropout))
            for _ in range(n_layers)
        ])
        self.norms   = nn.ModuleList([nn.LayerNorm(flat_dim) for _ in range(n_layers)])
        self.out_dim = flat_dim

    def forward(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(tokens, dim=1)          # (B, N*D)
        for block, norm in zip(self.blocks, self.norms):
            x = norm(x + block(x))
        return x                               # (B, N*D)


# E. Multi-Group Token Attention (full Transformer)

class MultiGroupTokenAttention(nn.Module):
    """
    Full Transformer encoder (pre-norm) over N group tokens.

    Learned positional embeddings distinguish group identities. The most
    expressive option; use it when data is sufficient to justify the
    added parameters.

    Total params (N=5, D=32, n_heads=4, dim_ff=128, n_layers=2) about
    17 700.
    """

    def __init__(self, n_groups: int, embed_dim: int,
                 n_heads: int = 4, n_layers: int = 2,
                 ffn_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        ffn_dim = ffn_dim or 4 * embed_dim
        n_heads = max(h for h in range(1, n_heads + 1) if embed_dim % h == 0)
        self.pos_embed = nn.Parameter(torch.randn(1, n_groups, embed_dim) * 0.02)
        enc_layer      = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_dim = n_groups * embed_dim

    def forward(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        x = torch.stack(tokens, dim=1) + self.pos_embed   # (B, N, D)
        return self.encoder(x).flatten(1)                  # (B, N*D)

    def attention_weights(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        """Last-layer attention weights (B, n_heads, N, N) for interpretability."""
        x = torch.stack(tokens, dim=1) + self.pos_embed
        for layer in self.encoder.layers[:-1]:
            x = layer(x)
        last = self.encoder.layers[-1]
        _, w = last.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
        return w   # (B, n_heads, N, N)


# Factory

_FUSION_REGISTRY = {
    "bilinear":    BilinearPairwiseFusion,
    "gated":       GatedResidualFusion,
    "linear_attn": LinearAttentionFusion,
    "low_rank":    LowRankCrossGroupMLP,
    "mgta":        MultiGroupTokenAttention,
}

def build_fusion(name: str, n_groups: int, embed_dim: int, **kwargs) -> nn.Module:
    """
    Instantiate a fusion layer by name.

    Args
    ----
    name      : "bilinear" | "gated" | "linear_attn" | "low_rank" | "mgta"
    n_groups  : number of feature group tokens (5 in this model)
    embed_dim : shared token embedding dim D
    **kwargs  : forwarded to the chosen class (rank, n_layers, n_heads, etc.)
    """
    if name not in _FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion '{name}'. Choose from: {list(_FUSION_REGISTRY)}")
    return _FUSION_REGISTRY[name](n_groups=n_groups, embed_dim=embed_dim, **kwargs)


# ============================================================================
# Non-parametric TDA layers
# ============================================================================

class TakensLayer(nn.Module):
    """
    Takens delay embedding: 1-D series -> point cloud in phase space.
        point_i = [x[i], x[i+tau], ..., x[i+(d-1)*tau]]
    Input : (B, 1, T)
    Output: list of B arrays (N, d)
    """
    def __init__(self, dim: int = 2, delay: int = 5):
        super().__init__()
        self.dim = dim; self.delay = delay

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        T = len(x); N = T - (self.dim - 1) * self.delay
        if N <= 0:
            raise ValueError(f"Segment too short: T={T}, dim={self.dim}, delay={self.delay}")
        return np.stack([x[i*self.delay: i*self.delay+N] for i in range(self.dim)],
                        axis=1).astype(np.float32)

    def forward(self, x: torch.Tensor) -> List[np.ndarray]:
        xn = x.squeeze(1).detach().cpu().numpy()
        return [self.forward_numpy(xn[b]) for b in range(x.shape[0])]


class PointCloudStatsLayer(nn.Module):
    """
    4 geometry stats from the Takens point cloud (before persistent
    homology): diameter, correlation_dim_proxy, mean_nn_dist, std_nn_dist
    Input : list of B arrays (N, d)
    Output: (B, 4)
    """
    def _stats_one(self, pc: np.ndarray) -> np.ndarray:
        diff  = pc[:, None, :] - pc[None, :, :]
        dists = np.sqrt((diff**2).sum(-1))
        diam  = dists.max()
        upper = dists[np.triu_indices(len(pc), k=1)]
        if len(upper) > 0 and upper.max() > 0:
            rs, rl   = np.quantile(upper, [0.10, 0.50])
            ns, nl   = (upper < rs).sum() + 1, (upper < rl).sum() + 1
            corr_d   = np.log(nl / ns) / (np.log(rl / (rs + 1e-10)) + 1e-10)
        else:
            corr_d   = 0.0
        np.fill_diagonal(dists, np.inf)
        nn_d = dists.min(axis=1)
        return np.array([diam, corr_d, nn_d.mean(), nn_d.std()], dtype=np.float32)

    def forward(self, pcs: List[np.ndarray], device: torch.device) -> torch.Tensor:
        return torch.tensor(np.stack([self._stats_one(p) for p in pcs]), device=device)


class RipserPHLayer(nn.Module):
    """
    Vietoris-Rips persistent homology via ripser (parallelised).
    Input : list of B arrays (N, d)
    Output: list of B diagram-lists [H0_arr, H1_arr, ...], each (n_pairs, 2)
    """
    def __init__(self, maxdim: int = 1, max_workers: int = 4):
        super().__init__()
        self.maxdim = maxdim; self.max_workers = max_workers

    @staticmethod
    def _clean(dgm: np.ndarray, mv: float) -> np.ndarray:
        dgm = dgm.copy()
        dgm[np.isinf(dgm[:, 1]), 1] = mv
        return dgm[(dgm[:, 1] - dgm[:, 0]) > 0]

    def _one(self, pc: np.ndarray) -> List[np.ndarray]:
        r  = ripser(pc, maxdim=self.maxdim)["dgms"]
        mv = max((d[~np.isinf(d[:, 1]), 1].max()
                  if (~np.isinf(d[:, 1])).any() else 1.0) for d in r)
        return [self._clean(d, mv) for d in r]

    def forward(self, pcs: List[np.ndarray]) -> List[List[np.ndarray]]:
        out = [None] * len(pcs)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            fs = {pool.submit(self._one, pc): i for i, pc in enumerate(pcs)}
            for f in as_completed(fs):
                out[fs[f]] = f.result()
        return out


class PersistenceEntropyLayer(nn.Module):
    """
    Shannon entropy of normalised bar lifetimes per homology dim.
    Periodic tends toward low entropy (few dominant bars).
    Chaotic tends toward high entropy (many bars of similar length).
    Output: (B, n_hom_dims)
    """
    def __init__(self, n_hom_dims: int = 2):
        super().__init__()
        self.n_hom_dims = n_hom_dims

    @staticmethod
    def _ent(dgm: np.ndarray) -> float:
        if not len(dgm): return 0.0
        lt = (dgm[:, 1] - dgm[:, 0]); lt = lt[lt > 0]
        if not len(lt): return 0.0
        p = lt / (lt.sum() + 1e-10)
        return float(-(p * np.log(p + 1e-10)).sum())

    def forward(self, diags: List[List[np.ndarray]], dev: torch.device) -> torch.Tensor:
        B   = len(diags)
        out = np.zeros((B, self.n_hom_dims), dtype=np.float32)
        for b, dl in enumerate(diags):
            for k in range(self.n_hom_dims):
                if k < len(dl): out[b, k] = self._ent(dl[k])
        return torch.tensor(out, device=dev)


class LifetimeStatsLayer(nn.Module):
    """
    5 lifetime statistics per homology dim, directly from persistence
    diagrams: max_lifetime, sum_lifetime, dominance_ratio,
    coeff_variation, n_significant
    Output: (B, n_hom_dims * 5)
    """
    _S = 5
    def __init__(self, n_hom_dims: int = 2):
        super().__init__()
        self.n_hom_dims = n_hom_dims
        self.out_dim    = n_hom_dims * self._S

    def _one(self, dgm: np.ndarray) -> np.ndarray:
        z = np.zeros(self._S, dtype=np.float32)
        if not len(dgm): return z
        lt = (dgm[:, 1] - dgm[:, 0]); lt = lt[lt > 0]
        if not len(lt): return z
        mx = lt.max(); sm = lt.sum()
        return np.array([mx, sm, mx / (sm + 1e-10),
                         lt.std() / (lt.mean() + 1e-10),
                         float((lt > 0.05 * mx).sum())], dtype=np.float32)

    def forward(self, diags: List[List[np.ndarray]], dev: torch.device) -> torch.Tensor:
        B   = len(diags)
        out = np.zeros((B, self.out_dim), dtype=np.float32)
        for b, dl in enumerate(diags):
            for k in range(self.n_hom_dims):
                dgm = dl[k] if k < len(dl) else np.empty((0, 2))
                out[b, k*self._S: (k+1)*self._S] = self._one(dgm)
        return torch.tensor(out, device=dev)


class BettiCurveLayer(nn.Module):
    """
    Per-sample relative Betti curves (scale-invariant: normalised by
    diameter).
    Output: (B, n_hom_dims, n_bins)
    """
    def __init__(self, n_hom_dims: int = 2, n_bins: int = 50):
        super().__init__()
        self.n_hom_dims = n_hom_dims; self.n_bins = n_bins

    def _curve(self, dgm: np.ndarray, t: np.ndarray) -> np.ndarray:
        if not len(dgm): return np.zeros(len(t), dtype=np.float32)
        return ((dgm[:, 0:1] <= t) & (t < dgm[:, 1:2])).sum(0).astype(np.float32)

    def forward(self, diags: List[List[np.ndarray]], dev: torch.device,
                diameters: Optional[np.ndarray] = None) -> torch.Tensor:
        B   = len(diags)
        out = np.zeros((B, self.n_hom_dims, self.n_bins), dtype=np.float32)
        for b, dl in enumerate(diags):
            if diameters is not None and diameters[b] > 0:
                t = np.linspace(0.0, float(diameters[b]), self.n_bins)
            else:
                vals = [dl[k].ravel() for k in range(self.n_hom_dims)
                        if k < len(dl) and len(dl[k]) > 0]
                c = np.concatenate(vals) if vals else np.array([0., 1.])
                t = np.linspace(c.min(), c.max(), self.n_bins)
            for k in range(self.n_hom_dims):
                if k < len(dl): out[b, k] = self._curve(dl[k], t)
        return torch.tensor(out, device=dev)


class PILayer(nn.Module):
    """
    Persistence images via Gaussian kernel (lazily fitted on first batch).
    Output: (B, n_hom_dims, n_pi_bins, n_pi_bins)
    """
    def __init__(self, n_hom_dims: int = 2, n_pi_bins: int = 20, sigma: float = 0.1):
        super().__init__()
        self.n_hom_dims = n_hom_dims; self.n_pi_bins = n_pi_bins; self.sigma = sigma
        self._im: List[Optional[PersistenceImager]] = [None] * n_hom_dims

    def _imager(self, k: int, dgms_k: List[np.ndarray]) -> PersistenceImager:
        if self._im[k] is None:
            valid   = [d for d in dgms_k if len(d) > 0]
            im      = PersistenceImager(pixel_size=self.sigma, kernel_params={"sigma": self.sigma})
            im.fit(valid if valid else [np.array([[0., 1.]])])
            self._im[k] = im
        return self._im[k]

    def reset(self) -> None:
        self._im = [None] * self.n_hom_dims

    def forward(self, diags: List[List[np.ndarray]], dev: torch.device) -> torch.Tensor:
        B   = len(diags)
        out = np.zeros((B, self.n_hom_dims, self.n_pi_bins, self.n_pi_bins), dtype=np.float32)
        for k in range(self.n_hom_dims):
            dgms_k = [dl[k] if k < len(dl) else np.empty((0, 2)) for dl in diags]
            im     = self._imager(k, dgms_k)
            for b, dgm in enumerate(dgms_k):
                if not len(dgm): continue
                try:
                    out[b, k] = F.interpolate(
                        torch.tensor(im.transform([dgm])[0]).unsqueeze(0).unsqueeze(0),
                        size=(self.n_pi_bins, self.n_pi_bins),
                        mode="bilinear", align_corners=False,
                    ).squeeze().numpy()
                except Exception:
                    pass
        return torch.tensor(out, device=dev)


class TDABettiExtractor(nn.Module):
    """
    6 statistics per homology dim from Betti curves: max, mean, std,
    peak_position, zero_crossings, bimodality_coeff
    Output: (B, n_hom_dims * 6)
    """
    _S = 6
    def __init__(self, n_hom_dims: int = 2, n_betti_bins: int = 50):
        super().__init__()
        self.n_hom_dims   = n_hom_dims
        self.n_betti_bins = n_betti_bins
        self.out_dim      = n_hom_dims * self._S

    @staticmethod
    def _bim(c: torch.Tensor) -> torch.Tensor:
        mu   = c.mean(1, keepdim=True); s = c - mu
        std  = s.std(1).clamp(1e-8)
        skew = (s**3).mean(1) / std**3
        kurt = ((s**4).mean(1) / std**4).clamp(1e-8)
        return (skew**2 + 1) / kurt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for k in range(self.n_hom_dims):
            c = x[:, k, :]
            d = c[:, 1:] - c[:, :-1]
            feats += [
                c.max(1).values,
                c.mean(1),
                c.std(1),
                c.argmax(1).float() / (self.n_betti_bins - 1),
                ((d[:, :-1] * d[:, 1:]) < 0).float().sum(1),
                self._bim(c),
            ]
        return torch.stack(feats, dim=1)   # (B, n_hom*6)


class TDAPIExtractor(nn.Module):
    """
    7 statistics per homology dim from persistence images: total_mass,
    max_pixel, active_fraction, birth_centroid, death_centroid,
    90th_percentile, entropy
    Output: (B, n_hom_dims * 7)
    """
    _S = 7
    def __init__(self, n_hom_dims: int = 2, n_pi_bins: int = 20):
        super().__init__()
        self.n_hom_dims = n_hom_dims; self.n_pi_bins = n_pi_bins
        self.out_dim    = n_hom_dims * self._S
        idx = torch.arange(n_pi_bins, dtype=torch.float32) / max(n_pi_bins - 1, 1)
        self.register_buffer("bg", idx)
        self.register_buffer("dg", idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for k in range(self.n_hom_dims):
            pi  = x[:, k]
            ms  = pi.sum((-2, -1)).clamp(1e-9)
            pn  = pi.flatten(1) / (pi.flatten(1).sum(1, keepdim=True) + 1e-9)
            ent = -(pn * (pn + 1e-9).log()).sum(1) / np.log(self.n_pi_bins**2 + 1e-9)
            feats += [
                pi.sum((-2, -1)),
                pi.flatten(1).max(1).values,
                (pi > pi.mean((-2, -1), keepdim=True)).float().mean((-2, -1)),
                (pi * self.bg.view(1, -1, 1)).sum((-2, -1)) / ms,
                (pi * self.dg.view(1, 1, -1)).sum((-2, -1)) / ms,
                torch.quantile(pi.flatten(1), 0.9, dim=1),
                ent,
            ]
        return torch.stack(feats, dim=1)   # (B, n_hom*7)


# ============================================================================
# Learnable infrastructure
# ============================================================================

class GroupProjector(nn.Module):
    """
    Per-group BatchNorm1d + Linear + activation + dropout -> shared embed
    dim D.

    Each group gets its own BatchNorm because TDA feature scales differ
    by orders of magnitude across groups (diameter vs entropy vs Betti
    counts). A single global BN over the concatenated 46-dim vector would
    be wrong.
    """
    def __init__(self, group_dims: List[int], embed_dim: int,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(d),
                nn.Linear(d, embed_dim),
                _get_activation(activation),
                nn.Dropout(dropout),
            ) for d in group_dims
        ])

    def forward(self, groups: List[torch.Tensor]) -> List[torch.Tensor]:
        return [p(g) for p, g in zip(self.projectors, groups)]


class ClassHead(nn.Module):
    """BatchNorm1d prepended to a configurable MLP. head_hidden=() -> single Linear."""
    def __init__(self, in_dim: int, n_classes: int,
                 head_hidden: Tuple[int, ...] = (64, 32),
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        layers: List[nn.Module] = [nn.BatchNorm1d(in_dim)]
        cur = in_dim
        for h in head_hidden:
            layers += [nn.Linear(cur, h), _get_activation(activation)]
            if dropout > 0: layers.append(nn.Dropout(dropout))
            cur = h
        layers.append(nn.Linear(cur, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ============================================================================
# EarlyStopper
# ============================================================================

class EarlyStopper:
    """Saves best checkpoint and sets early_stop flag after `patience` non-improvements."""
    def __init__(self, verbose: bool = False, path: str = "checkpoint.pt", patience: int = 5):
        self.patience = patience; self.verbose = verbose
        self.counter = 0; self.best_loss = None
        self._early_stop = False; self.path = path

    @property
    def early_stop(self) -> bool:
        return self._early_stop

    def update(self, val_loss: float, model: nn.Module) -> None:
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose: print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience: self._early_stop = True

    def load_checkpoint(self, model: nn.Module) -> nn.Module:
        if not os.path.exists(self.path):
            print(f"WARNING: checkpoint not found at {self.path}"); return model
        # map_location="cpu" so a GPU checkpoint loads cleanly on CPU-only machines
        model.load_state_dict(torch.load(self.path, map_location="cpu", weights_only=True))
        return model


# ============================================================================
# TDAEnd2EndNet
# ============================================================================

class TDAEnd2EndNet(nn.Module):
    """
    End-to-end TDA classification network with selectable full-group
    fusion.

    All five feature groups are projected to a shared embedding dim D via
    per-group BatchNorm + Linear, then fused by one of five strategies:

      "bilinear"     cheapest, pairwise scalar interactions
      "gated"        sigmoid-gated residual per token
      "linear_attn"  kernel-based attention without softmax
      "low_rank"     bottleneck MLP (recommended default)
      "mgta"         full Transformer encoder (most expressive)

    All fusion methods implement:
        tokens: List[Tensor(B, D)] -> Tensor(B, 5*D)

    Parameters (n_hom=2, D=32)
    ---------------------------
      TDA layers (all fixed)  :      0
      GroupProjector          :  ~2 000
      bilinear fusion         :    640
      low_rank  fusion (r=8)  :  2 560
      linear_attn (1L)        :  3 500
      gated     fusion        : 10 560
      mgta      (2L)          : 17 700
      ClassHead (64->32->n)   :  7 000
    """

    def __init__(
        self,
        n_classes:     int             = 2,
        seg_len:       int             = 200,
        takens_dim:    int             = 2,
        takens_delay:  int             = 5,
        n_hom_dims:    int             = 2,
        n_betti_bins:  int             = 50,
        n_pi_bins:     int             = 20,
        pi_sigma:      float           = 0.1,
        embed_dim:     int             = 32,
        fusion:        str             = "low_rank",
        # fusion-specific
        n_heads:       int             = 4,
        n_attn_layers: int             = 1,
        ffn_dim:       int             = 0,
        rank:          int             = 8,
        # general
        dropout:       float           = 0.1,
        activation:    str             = "gelu",
        head_hidden:   Tuple[int, ...] = (64, 32),
        ph_workers:    int             = 4,
        # device kwarg removed; use model.to(device) after construction.
        # Kept as **kwargs to avoid breaking existing call sites that pass
        # device= as a keyword argument.
        **kwargs,
    ):
        super().__init__()

        # non-parametric TDA layers
        self.takens     = TakensLayer(dim=takens_dim, delay=takens_delay)
        self.pc_stats   = PointCloudStatsLayer()
        self.ph         = RipserPHLayer(maxdim=n_hom_dims - 1, max_workers=ph_workers)
        self.ph_entropy = PersistenceEntropyLayer(n_hom_dims=n_hom_dims)
        self.lt_stats   = LifetimeStatsLayer(n_hom_dims=n_hom_dims)
        self.betti      = BettiCurveLayer(n_hom_dims=n_hom_dims, n_bins=n_betti_bins)
        self.pi_layer   = PILayer(n_hom_dims=n_hom_dims, n_pi_bins=n_pi_bins, sigma=pi_sigma)
        self.betti_ext  = TDABettiExtractor(n_hom_dims=n_hom_dims, n_betti_bins=n_betti_bins)
        self.pi_ext     = TDAPIExtractor(n_hom_dims=n_hom_dims, n_pi_bins=n_pi_bins)

        # group dims: [pc | entropy | lifetime | betti | pi]
        group_dims = [
            4,                        # pc_stats
            n_hom_dims,               # entropy
            self.lt_stats.out_dim,    # n_hom * 5
            self.betti_ext.out_dim,   # n_hom * 6
            self.pi_ext.out_dim,      # n_hom * 7
        ]
        n_groups = len(group_dims)    # always 5

        # per-group projection to shared dim D
        self.group_proj = GroupProjector(
            group_dims=group_dims, embed_dim=embed_dim,
            dropout=dropout, activation=activation,
        )

        # fusion layer (route method-specific kwargs)
        fkw: dict = {"dropout": dropout}
        if fusion in ("mgta", "linear_attn"):
            fkw["n_heads"] = n_heads; fkw["n_layers"] = n_attn_layers
        if fusion == "mgta":
            fkw["ffn_dim"] = ffn_dim
        if fusion == "linear_attn":
            fkw["dim_ff"]  = ffn_dim
        if fusion == "low_rank":
            fkw["rank"] = rank; fkw["n_layers"] = n_attn_layers; fkw["activation"] = activation

        self.fusion_layer = build_fusion(fusion, n_groups=n_groups, embed_dim=embed_dim, **fkw)
        self._fusion_name = fusion

        # classification head
        self.classifier = ClassHead(
            in_dim=self.fusion_layer.out_dim, n_classes=n_classes,
            head_hidden=head_hidden, dropout=dropout, activation=activation,
        )
        self._head_in = self.fusion_layer.out_dim
        self._gd      = group_dims

        self.group_proj.apply(_init_weights)
        self.classifier.apply(_init_weights)

    # forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 1, T) -> logits (B, n_classes)"""
        dev = x.device
        with torch.no_grad():
            pcs     = self.takens(x)
            pc_t    = self.pc_stats(pcs, dev)
            diams   = pc_t[:, 0].cpu().numpy()
            diags   = self.ph(pcs)
            ent_t   = self.ph_entropy(diags, dev)
            lt_t    = self.lt_stats(diags, dev)
            betti_t = self.betti(diags, dev, diameters=diams)
            pi_t    = self.pi_layer(diags, dev)
            b_stats = self.betti_ext(betti_t)
            p_stats = self.pi_ext(pi_t)

        tokens = self.group_proj([pc_t, ent_t, lt_t, b_stats, p_stats])
        fused  = self.fusion_layer(tokens)
        return self.classifier(fused)

    # introspection

    def count_parameters(self, verbose: bool = False) -> int:
        fixed = {
            "TakensLayer           ": self.takens,
            "PointCloudStatsLayer  ": self.pc_stats,
            "RipserPHLayer         ": self.ph,
            "PersistenceEntropyLyr ": self.ph_entropy,
            "LifetimeStatsLayer    ": self.lt_stats,
            "BettiCurveLayer       ": self.betti,
            "PILayer               ": self.pi_layer,
            "TDABettiExtractor     ": self.betti_ext,
            "TDAPIExtractor        ": self.pi_ext,
        }
        learned = {
            "GroupProjector        ": self.group_proj,
            f"FusionLayer({self._fusion_name:<11})": self.fusion_layer,
            "ClassHead             ": self.classifier,
        }
        total = 0
        if verbose:
            print(f"\n  {'Component':<40} {'Params':>8}  Note")
            print("  " + "-" * 62)
            print(f"  Group dims : {self._gd}")
            print(f"  Head input : {self._head_in}  ({len(self._gd)} groups x embed_dim)")
            print()
        for name, mod in {**fixed, **learned}.items():
            n = _count(mod); total += n
            if verbose:
                note = "<- fixed  " if mod in fixed.values() else "<- trained"
                print(f"  {name:<40}  {n:>8,}  {note}")
        if verbose:
            print("  " + "-" * 62)
            print(f"  {'TOTAL TRAINABLE':<40}  {total:>8,}")
        return total

    def reset_pi_imager(self) -> None:
        self.pi_layer.reset()

    # robustness evaluation

    def evaluate_robustness(
        self,
        x_test:       torch.Tensor,
        y_test:       torch.Tensor,
        noise_levels: Optional[List[float]] = None,
        batch_size:   int = 16,
        seed:         int = 42,
    ) -> "pd.DataFrame":
        """Sweep Gaussian noise levels over the test set without retraining.

        Implemented directly on the model because TDAEnd2EndNet.forward
        owns the full TDA pipeline (Takens, persistent homology,
        features, classify), so there is no separate validation_step to
        delegate to.

        Important: noise is added to the raw time-series input before
        the Takens embedding. This is intentional: it tests whether the
        TDA features themselves are robust to sensor noise, not just the
        MLP head. The PI imager is reset before each sigma level so it is
        refitted on the noisy data, matching realistic inference
        conditions.

        Args:
            x_test:       Normalised test tensor, shape (N, 1, T).
            y_test:       Integer label tensor, shape (N,).
            noise_levels: List of sigma values to evaluate.
                          Defaults to [0, 0.05, 0.1, ..., 1.0] (21 levels).
            batch_size:   Samples per forward pass (keep small; persistent
                          homology is costly).
            seed:         RNG seed for reproducible noise draws.

        Returns:
            pd.DataFrame indexed by sigma with columns:
            sigma, accuracy, mean_confidence, pct_low_conf.
        """
        import pandas as pd

        if noise_levels is None:
            noise_levels = [round(i * 0.05, 2) for i in range(21)]  # 0 to 1.0

        self.eval()
        rng    = torch.Generator(); rng.manual_seed(seed)
        dev    = next(self.parameters()).device
        x_test = x_test.to(dev)
        y_test = y_test.to(dev)

        records = []
        n_levels = len(noise_levels)

        print(f"TDA robustness sweep: {n_levels} noise levels, "
              f"sigma in [{min(noise_levels):.2f}, {max(noise_levels):.2f}]")
        print(f"Test set: {x_test.shape[0]} samples  batch_size={batch_size}")
        print()

        for i, sigma in enumerate(noise_levels):
            if sigma == 0.0:
                x_noisy = x_test.clone()
            else:
                noise   = torch.zeros_like(x_test).normal_(0.0, sigma, generator=rng)
                x_noisy = x_test + noise

            # Reset PI imager: noisy data shifts the persistence image
            # range, so the imager must be refitted to avoid silent
            # extrapolation.
            self.reset_pi_imager()

            all_preds  = []
            all_confs  = []

            with torch.no_grad():
                for start in range(0, x_noisy.shape[0], batch_size):
                    xb     = x_noisy[start: start + batch_size]
                    logits = self(xb)                               # (B, C)
                    probs  = torch.softmax(logits, dim=1)
                    conf, pred = probs.max(dim=1)
                    all_preds.append(pred.cpu())
                    all_confs.append(conf.cpu())

            preds = torch.cat(all_preds)
            confs = torch.cat(all_confs)

            acc      = (preds == y_test.cpu()).float().mean().item()
            mean_conf = confs.mean().item()
            pct_low   = (confs < 0.5).float().mean().item() * 100.0
            snr_db    = 10.0 * np.log10(1.0 / (sigma ** 2)) if sigma > 0 else float("inf")

            records.append({
                "sigma":         sigma,
                "snr_db":        snr_db,
                "accuracy":      acc,
                "mean_confidence": mean_conf,
                "pct_low_conf":  pct_low,
            })

            print(f"  [{i+1:2d}/{n_levels}]  sigma={sigma:.3f}  "
                  f"SNR={snr_db:+.1f} dB  "
                  f"acc={acc:.4f}  conf={mean_conf:.4f}  "
                  f"low_conf={pct_low:.1f}%")

        df = pd.DataFrame(records).set_index("sigma")
        print(f"\nSweep complete.  Clean accuracy = "
              f"{df.loc[df.index.min(), 'accuracy']:.4f}")
        return df

    # named constructors

    @classmethod
    def tiny(cls, n_classes: int = 2, seg_len: int = 200,
             fusion: str = "bilinear", **kwargs) -> "TDAEnd2EndNet":
        """About 700 params. Bilinear fusion, no hidden layers."""
        return cls(n_classes=n_classes, seg_len=seg_len, embed_dim=16,
                   fusion=fusion, dropout=0.0, head_hidden=(), **kwargs)

    @classmethod
    def small(cls, n_classes: int = 2, seg_len: int = 200,
              fusion: str = "low_rank", **kwargs) -> "TDAEnd2EndNet":
        """About 4-6k params. Low-rank fusion, one hidden layer."""
        return cls(n_classes=n_classes, seg_len=seg_len, embed_dim=32,
                   fusion=fusion, rank=8, n_attn_layers=1,
                   dropout=0.05, head_hidden=(32,), **kwargs)

    @classmethod
    def standard(cls, n_classes: int = 2, seg_len: int = 200,
                 fusion: str = "low_rank", **kwargs) -> "TDAEnd2EndNet":
        """About 10-12k params. Low-rank fusion, two hidden layers."""
        return cls(n_classes=n_classes, seg_len=seg_len, embed_dim=32,
                   fusion=fusion, rank=8, n_attn_layers=2, n_heads=4,
                   dropout=0.1, head_hidden=(64, 32), **kwargs)

    @classmethod
    def from_config(cls, cfg: dict, n_classes: int, **kwargs) -> "TDAEnd2EndNet":
        """Build from a config dict (e.g. loaded from config.json)."""
        return cls(
            n_classes     = n_classes,
            seg_len       = cfg.get("segmentation_duration", 200),
            takens_dim    = cfg.get("takens_dim",      2),
            takens_delay  = cfg.get("takens_delay",    5),
            n_hom_dims    = cfg.get("n_hom_dims",      2),
            n_betti_bins  = cfg.get("n_betti_bins",   50),
            n_pi_bins     = cfg.get("n_pi_bins",      20),
            pi_sigma      = cfg.get("pi_sigma",       0.1),
            embed_dim     = cfg.get("embed_dim",       32),
            fusion        = cfg.get("fusion",    "low_rank"),
            n_heads       = cfg.get("n_heads",          4),
            n_attn_layers = cfg.get("n_attn_layers",    1),
            ffn_dim       = cfg.get("ffn_dim",          0),
            rank          = cfg.get("rank",             8),
            dropout       = cfg.get("dropout",        0.1),
            activation    = cfg.get("activation",   "gelu"),
            head_hidden   = tuple(cfg.get("head_hidden", [64, 32])),
            ph_workers    = cfg.get("ph_workers",       4),
            **kwargs,
        )


# ============================================================================
# Integration test and fusion comparison
# ============================================================================

if __name__ == "__main__":
    import time

    B, T = 8, 200
    x    = torch.randn(B, 1, T)

    print("=" * 72)
    print("  TDAEnd2EndNet: fusion strategy comparison")
    print("=" * 72)
    print(f"\n  {'Config':<30} {'Trainable':>10}  {'ms/batch':>10}  out")
    print("  " + "-" * 65)

    rows = [
        ("tiny  / bilinear   ", TDAEnd2EndNet.tiny(fusion="bilinear")),
        ("small / low_rank   ", TDAEnd2EndNet.small(fusion="low_rank")),
        ("small / linear_attn", TDAEnd2EndNet.small(fusion="linear_attn")),
        ("small / gated      ", TDAEnd2EndNet.small(fusion="gated")),
        ("small / mgta       ", TDAEnd2EndNet.small(fusion="mgta")),
        ("std   / low_rank   ", TDAEnd2EndNet.standard(fusion="low_rank")),
        ("std   / mgta       ", TDAEnd2EndNet.standard(fusion="mgta")),
    ]

    for label, model in rows:
        model.eval()
        with torch.no_grad(): _ = model(x)          # warmup (fits PI imager)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(10): out = model(x)
        ms = (time.perf_counter() - t0) / 10 * 1000
        print(f"  {label:<30} {model.count_parameters():>10,}  {ms:>10.1f}  {tuple(out.shape)}")

    print("\n" + "-" * 72)
    print("  Detailed breakdown, standard / low_rank:")
    TDAEnd2EndNet.standard(fusion="low_rank").count_parameters(verbose=True)

    print("\n" + "-" * 72)
    print("  Robustness sweep, standard / low_rank:")
    model_rob = TDAEnd2EndNet.standard(fusion="low_rank", n_classes=2)
    y_dummy   = torch.randint(0, 2, (B,))
    df_rob    = model_rob.evaluate_robustness(x, y_dummy, noise_levels=[0.0, 0.1, 0.5, 1.0])
    print(df_rob.to_string())

    print("\n" + "-" * 72)
    print("""  config.json reference:
  {
    "fusion"        : "low_rank",   // "bilinear"|"gated"|"linear_attn"|"low_rank"|"mgta"
    "embed_dim"     : 32,
    "rank"          : 8,            // low_rank only
    "n_attn_layers" : 1,            // low_rank / linear_attn / mgta: depth
    "n_heads"       : 4,            // linear_attn / mgta only
    "ffn_dim"       : 0,            // mgta / linear_attn (0 = auto)
    "dropout"       : 0.1,
    "activation"    : "gelu",
    "head_hidden"   : [64, 32],
    "takens_dim"    : 2,
    "takens_delay"  : 5,
    "n_hom_dims"    : 2,
    "n_betti_bins"  : 50,
    "n_pi_bins"     : 20,
    "pi_sigma"      : 0.1,
    "ph_workers"    : 4
  }""")


