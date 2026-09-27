import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_weights(m, init_type="kaiming"):
    if isinstance(m, nn.Linear):
        if init_type == "kaiming":
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif init_type == "xavier":
            nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        if init_type == "kaiming":
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif init_type == "xavier":
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


# ---------------------------------------------------------------------------
# Positional Encoding
# Source: https://github.com/pytorch/pytorch/issues/51551
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Operates in (seq_len, batch, d_model) space but also accepts
    batch_first tensors when ``batch_first=True``.

    Args:
        d_model (int): Embedding dimension.
        dropout (float): Dropout probability. Default: 0.1.
        max_len (int): Maximum sequence length. Default: 5000.
        batch_first (bool): If True, input/output shape is
            (batch, seq_len, d_model). Default: False.
    """

    def __init__(self, d_model: int, dropout: float = 0.1,
                 max_len: int = 5000, batch_first: bool = False):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)                        # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)                                        # (max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) if batch_first else (seq_len, batch, d_model)
        """
        if self.batch_first:
            # pe is (max_len, 1, d_model), need (1, seq_len, d_model)
            x = x + self.pe[:x.size(1)].permute(1, 0, 2)
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Model
# ---------------------------------------------------------------------------

class TransformerI(nn.Module):
    """Transformer encoder model for time-series regression / classification.

    Input shape:  (batch, input_channels, seq_len)   [PyTorch-standard]
    Output shape: (batch, output_size)

    Design notes:
    * A single stacked ``TransformerEncoder`` is used (no duplicate
      encoder pass).
    * The causal mask is pre-computed once in ``__init__`` and moved to
      the correct device on first use, avoiding per-step allocation.
    * ``PositionalEncoding`` runs in batch-first mode.
    * ``batch_first=True`` is consistent throughout.
    * ``use_causal_mask`` lets you disable the causal mask for tasks that
      do not need autoregressive masking (e.g. classification).
    * ``num_encoderlayers`` stacks that many encoder layers.
    * ``activation`` ('relu' | 'gelu') is forwarded to the encoder.
    * ``norm_first`` (Pre-LN) is forwarded to the encoder layer, often
      more stable for deeper stacks.
    * ``pooling`` ('flatten' | 'mean' | 'last') controls how the encoder
      output is summarised before the head.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_size: int = 5,
        seq_len: int = 200,
        embed_size: int = 16,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        conv1d_emb: bool = True,
        conv1d_kernel_size: int = 3,
        size_linear_layers: int = 16,
        num_encoderlayers: int = 1,
        activation: str = "relu",        # "relu" | "gelu"
        norm_first: bool = False,        # Pre-LN: more stable for deep stacks
        use_causal_mask: bool = False,   # True only for autoregressive tasks; False for classification
        pooling: str = "flatten",        # "flatten" | "mean" | "last"
    ):
        super().__init__()

        if embed_size % nhead != 0:
            raise ValueError(f"embed_size ({embed_size}) must be divisible by nhead ({nhead}).")
        if conv1d_emb and conv1d_kernel_size % 2 == 0:
            raise ValueError("conv1d_kernel_size must be odd to preserve sequence length.")
        if pooling not in ("flatten", "mean", "last"):
            raise ValueError(f"pooling must be 'flatten', 'mean', or 'last', got '{pooling}'.")

        self.seq_len = seq_len
        self.embed_size = embed_size
        self.conv1d_emb = conv1d_emb
        self.use_causal_mask = use_causal_mask
        self.pooling = pooling
        # Note: do not store device as self.device, nn.Module uses that
        # internally. Use model.parameters().__next__().device at runtime,
        # or call model.to(device) and let PyTorch handle placement.

        # input embedding
        if conv1d_emb:
            pad = (conv1d_kernel_size - 1) // 2
            self.input_embedding = nn.Conv1d(
                input_channels, embed_size,
                kernel_size=conv1d_kernel_size, padding=pad
            )
        else:
            self.input_embedding = nn.Linear(input_channels, embed_size)

        # positional encoding (batch-first)
        self.position_encoder = PositionalEncoding(
            d_model=embed_size,
            dropout=dropout,
            max_len=seq_len,
            batch_first=True,
        )

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True,             # keeps (batch, seq, feat) throughout
        )
        encoder_norm = nn.LayerNorm(embed_size)
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoderlayers,
            norm=encoder_norm,
        )

        # regression / classification head
        if pooling == "flatten":
            head_in = seq_len * embed_size
        else:                             # "mean" or "last" -> (batch, embed_size)
            head_in = embed_size

        self.linear1 = nn.Linear(head_in, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, size_linear_layers)
        self.outlayer = nn.Linear(size_linear_layers, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # causal mask (pre-computed, device-agnostic buffer)
        # Registered as a non-persistent buffer so it moves with .to(device)
        # but is not saved in state_dict.
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf')), diagonal=1
        )
        self.register_buffer('_causal_mask', mask, persistent=False)

    # forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_channels, seq_len)
        Returns:
            (batch, output_size)
        """
        # embedding
        if self.conv1d_emb:
            # Conv1d: (batch, channels, seq_len) -> (batch, embed_size, seq_len)
            x = self.input_embedding(x)
            x = x.permute(0, 2, 1)           # -> (batch, seq_len, embed_size)
        else:
            x = x.permute(0, 2, 1)           # -> (batch, seq_len, channels)
            x = self.input_embedding(x)       # -> (batch, seq_len, embed_size)

        # positional encoding
        x = self.position_encoder(x)          # (batch, seq_len, embed_size)

        # transformer
        # Slice the pre-computed mask to the actual sequence length so the
        # model handles inputs shorter than seq_len without a size mismatch.
        if self.use_causal_mask:
            T = x.size(1)
            mask = self._causal_mask[:T, :T]
        else:
            mask = None
        x = self.transformer(x, mask=mask)    # (batch, seq_len, embed_size)

        # pooling
        if self.pooling == "flatten":
            x = x.reshape(x.size(0), -1)     # (batch, seq_len * embed_size)
        elif self.pooling == "mean":
            x = x.mean(dim=1)                # (batch, embed_size)
        else:  # "last"
            x = x[:, -1, :]                  # (batch, embed_size)

        # head
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        return self.outlayer(x)


# ---------------------------------------------------------------------------
# CNN Model
# ---------------------------------------------------------------------------

class ResidualBlock1d(nn.Module):
    """Residual block for 1-D signals.

    Architecture (per block):
        Conv1d -> BN -> ReLU -> Dropout -> Conv1d -> BN
        + skip connection (1x1 conv if channels change)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2   # 'same' padding for odd kernels

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=pad, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=pad, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.relu_inplace = nn.ReLU(inplace=True)   # safe before the add
        self.relu = nn.ReLU(inplace=False)           # after add: must not be inplace

        # 1x1 projection so residual dims always match
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.relu_inplace(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)          # non-inplace after addition


class CNNI(nn.Module):
    """Dilated residual CNN for time-series regression / classification.

    Shares the same input/output contract as ``TransformerI``:
        Input:  (batch, input_channels, seq_len)
        Output: (batch, output_size)

    Architecture:
        Stem conv -> N residual blocks (with exponentially growing dilation)
        -> global pooling -> MLP head

    Dilation grows as 2^layer_idx (1, 2, 4, 8, ...), giving the network an
    exponentially large receptive field without any pooling that would
    discard temporal resolution before the head.

    Args:
        input_channels (int): Number of input channels. Default: 1.
        output_size (int): Number of outputs. Default: 5.
        base_channels (int): Channel width after the stem. Default: 32.
        channel_multipliers (tuple): Per-block channel multiplier relative to
            ``base_channels``. E.g. ``(1, 2, 4)`` gives 32, 64, 128 channels.
            Default: (1, 2, 4).
        kernel_size (int): Kernel size for all conv layers (must be odd).
            Default: 3.
        dropout (float): Dropout probability. Default: 0.0.
        pooling (str): 'mean' | 'max' | 'last'. Default: 'mean'.
        size_linear_layers (int): Hidden size of the MLP head. Default: 128.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_size: int = 5,
        base_channels: int = 32,
        channel_multipliers: tuple = (1, 2, 4),
        kernel_size: int = 3,
        dropout: float = 0.0,
        pooling: str = "mean",
        size_linear_layers: int = 128,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if pooling not in ("mean", "max", "last"):
            raise ValueError(f"pooling must be 'mean', 'max', or 'last', got '{pooling}'.")

        self.pooling = pooling

        # stem
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        # residual blocks (exponential dilation)
        blocks = []
        in_ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            dilation = 2 ** i              # 1, 2, 4, 8, ...
            blocks.append(
                ResidualBlock1d(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        final_channels = base_channels * channel_multipliers[-1]

        # MLP head
        self.linear1 = nn.Linear(final_channels, size_linear_layers)
        self.linear2 = nn.Linear(size_linear_layers, size_linear_layers // 2)
        self.outlayer = nn.Linear(size_linear_layers // 2, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_channels, seq_len)
        Returns:
            (batch, output_size)
        """
        x = self.stem(x)                  # (batch, base_channels, seq_len)
        x = self.blocks(x)                # (batch, final_channels, seq_len)

        # pooling
        if self.pooling == "mean":
            x = x.mean(dim=-1)            # (batch, final_channels)
        elif self.pooling == "max":
            x = x.max(dim=-1).values      # (batch, final_channels)
        else:  # "last"
            x = x[:, :, -1]              # (batch, final_channels)

        # head
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        return self.outlayer(x)


# ---------------------------------------------------------------------------
# Early Stopper
# ---------------------------------------------------------------------------

class EarlyStopper:
    """Stop training when validation loss stops decreasing.

    Args:
        verbose (bool): Print counter updates. Default: False.
        path (str): Checkpoint file path. Default: 'checkpoint.pt'.
        patience (int): Epochs to wait before stopping. Default: 1.
        min_delta (float): Minimum decrease in loss to qualify as an
            improvement. Default: 0.0.
    """

    def __init__(
        self,
        verbose: bool = False,
        path: str = 'checkpoint.pt',
        patience: int = 1,
        min_delta: float = 0.0,
    ):
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self._early_stop = False
        self.val_loss_min = np.inf
        self.path = path

    @property
    def early_stop(self) -> bool:
        """True once the early-stopping criterion has been met."""
        return self._early_stop

    def update(self, val_loss: float, model: nn.Module):
        """Call once per epoch after computing validation loss.

        Args:
            val_loss: Validation loss for this epoch.
            model: Model whose weights should be checkpointed on improvement.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_checkpoint(model, val_loss)
            return

        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = val_loss
            self._save_checkpoint(model, val_loss)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self._early_stop = True

    def _save_checkpoint(self, model: nn.Module, val_loss: float):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def load_checkpoint(self, model: nn.Module) -> nn.Module:
        """Load best weights back into *model* (in-place) and return it."""
        if not os.path.exists(self.path):
            print(f"WARNING: checkpoint not found at '{self.path}'")
            return model
        model.load_state_dict(
            torch.load(self.path, map_location="cpu", weights_only=True)
        )
        return model
