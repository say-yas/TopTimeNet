import torch


def segment_data(
        X:                torch.Tensor,
        y:                torch.Tensor,
        segment_duration: int,
        drop_zero:        bool = True,
) -> dict:
    """
    Segment a univariate tensor into fixed-length windows, then drop
    all-zero segments (padding artefacts).

    Parameters
    ----------
    X                : (n_rows, seq_len), output of make_tensors (univariate)
    y                : (n_rows,), class labels
    segment_duration : window length in samples
    drop_zero        : if True, remove segments where every value == 0

    Returns
    -------
    dict with keys
        'X'    : (n_valid_segments, segment_duration) FloatTensor
        'y'    : (n_valid_segments,) LongTensor
        'info' : dict, counts and shapes for quick inspection
    """
    n_rows, seq_len = X.shape

    # how many complete windows fit?
    num_segments = seq_len // segment_duration
    usable_len   = num_segments * segment_duration   # drop the last partial window

    print(f'seq_len={seq_len}  segment_duration={segment_duration}  '
          f'num_segments={num_segments}  '
          f'(discarding last {seq_len - usable_len} samples per row)')

    # reshape (n_rows, seq_len) into (n_rows, num_segments, segment_duration)
    X_seg = X[:, :usable_len].reshape(n_rows, num_segments, segment_duration)
    print(f'Before zero-filter : {X_seg.shape[0] * num_segments} segments  '
          f'shape ({n_rows * num_segments}, {segment_duration})')

    # expand labels to match every segment of each row
    # y shape: (n_rows,) -> (n_rows, num_segments) -> flatten -> (n_rows * num_segments,)
    y_seg = y.unsqueeze(1).expand(n_rows, num_segments).reshape(-1)  # (n_rows*num_segs,)
    X_flat = X_seg.reshape(-1, segment_duration)                      # (n_rows*num_segs, seg_dur)

    # drop all-zero segments (pure padding)
    if drop_zero:
        # a segment is valid if at least one value is non-zero
        valid_mask  = X_flat.abs().sum(dim=1) > 0          # (n_rows*num_segs,)
        X_flat = X_flat[valid_mask]
        y_seg  = y_seg[valid_mask]
        n_dropped = (~valid_mask).sum().item()
        print(f'Dropped {n_dropped} all-zero segments')

    print(f'After  zero-filter : X={tuple(X_flat.shape)}  y={tuple(y_seg.shape)}')

    return {
        'X'   : X_flat,
        'y'   : y_seg,
        'info': {
            'n_rows'          : n_rows,
            'num_segments'    : num_segments,
            'segment_duration': segment_duration,
            'total_segments'  : X_flat.shape[0],
        }
    }
