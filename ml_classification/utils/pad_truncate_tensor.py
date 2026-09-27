import torch
import pandas as pd
import numpy as np

def _pad_or_truncate(arr: np.ndarray, seq_len: int) -> torch.Tensor:
    """1-D array to fixed-length 1-D FloatTensor (truncate or zero-pad right)."""
    t = torch.tensor(arr[:seq_len], dtype=torch.float32)   # truncate
    if len(t) < seq_len:                                    # pad right
        t = torch.nn.functional.pad(t, (0, seq_len - len(t)))
    return t                                                # shape: (seq_len,)

def make_tensors(
        df:         pd.DataFrame,
        seq_len:    int
) -> dict:
    """
    Convert the array-cell DataFrame into padded/truncated PyTorch tensors.
    One sample per row of df (univariate).

    Parameters
    ----------
    df      : DataFrame produced by read_data() (has columns ts, label, variable)
    seq_len : fixed sequence length; longer series are truncated,
              shorter ones are zero-padded on the right

    Returns
    -------
    dict with keys:
        'X'    : FloatTensor (n_rows, seq_len), input features
        'y'    : LongTensor (n_rows,), class labels
        'meta' : DataFrame, one row per sample, for traceability
    """
    tensors = [_pad_or_truncate(row['ts'], seq_len) for _, row in df.iterrows()]

    X = torch.stack(tensors)                                # (n_rows, seq_len)
    y = torch.tensor(df['label'].values, dtype=torch.long)  # (n_rows,)

    meta = df[['dataset', 'state', 'variable', 'label']].reset_index(drop=True)

    print(f'X shape: {tuple(X.shape)}   y shape: {tuple(y.shape)}')
    return {'X': X, 'y': y, 'meta': meta}
