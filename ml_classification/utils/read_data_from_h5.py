from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
from src.dataset_generation.load_dataset_extended_teaspoon import load_extended_teaspoon_dataset_from_h5

def _make_var_names(n: int) -> list[str]:
    """
    Auto-generate variable labels from the number of signals in ts_data.

    Examples
    --------
    n=1 gives ['X']
    n=3 gives ['X', 'Y', 'Z']
    n=5 gives ['X', 'Y', 'Z', 'V3', 'V4']  (overflow uses V{i})
    """
    base = list('XYZUVWPQRST')  # up to 20 variables
    return [base[i] if i < len(base) else f'V{i}' for i in range(n)]

def read_data(filepath: str, datasets: list[str] | None = None) -> pd.DataFrame:
    """
    Read every (dataset, state) group from the HDF5 file and return a
    tidy DataFrame where each row is one variable from one simulation.

    Schema
    ------
    dataset   : str          dataset name (e.g. 'lorenz')
    state     : str          dynamic state (e.g. 'chaotic')
    variable  : str          auto-generated label ('X', 'Y', 'Z', 'V3', ...)
    var_index : int          0-based position inside ts_data
    n_samples : int          length of the arrays
    t         : np.ndarray   full time axis stored as a single cell
    ts        : np.ndarray   full signal stored as a single cell

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file produced by the collection pipeline.
    datasets : list[str] | None
        Dataset names to include. None processes every group in the file.

    Returns
    -------
    pd.DataFrame (object dtype for the array columns t and ts)

    Example output
    --------------
    dataset    state     variable  var_index  n_samples  t          ts
    lorenz     chaotic   X         0          70000      array(...) array(...)
    lorenz     chaotic   Y         1          70000      array(...) array(...)
    lorenz     chaotic   Z         2          70000      array(...) array(...)
    lorenz     periodic  X         0          70000      array(...) array(...)
    rossler    chaotic   X         0          70000      array(...) array(...)
    ...
    """
    # collect all (dataset, state) keys present in the file
    with h5py.File(filepath, 'r') as f:
        all_keys = [
            (ds, st)
            for ds in f.keys()
            for st in f[ds].keys()
        ]

    if datasets is not None:
        all_keys = [(ds, st) for ds, st in all_keys if ds in datasets]

    print(f'Processing {len(all_keys)} (dataset, state) pairs from {filepath}\n')

    # build records list (one dict per variable)
    records: list[dict] = []
    skipped: list[tuple] = []

    for dataset, state in all_keys:
        print(f'  {dataset} [{state}]')

        result = load_extended_teaspoon_dataset_from_h5(filepath, dataset, state)
        if result is None:
            print(f'    Skipped.')
            skipped.append((dataset, state))
            continue

        t_data, ts_data = result

        n_vars    = len(ts_data)
        var_names = _make_var_names(n_vars)     # derived at runtime from length
        n_samples = len(t_data)

        print(f'    Variables : {var_names}  ({n_vars} total)')
        print(f'    Samples   : {n_samples}')

        t_arr = np.asarray(t_data, dtype=float)   # share across variables

        for idx, (var_label, ts_arr) in enumerate(zip(var_names, ts_data)):
            records.append({
                'dataset'  : dataset,
                'state'    : state,
                'variable' : var_label,
                'var_index': idx,
                'n_samples': n_samples,
                't'        : t_arr,                           # same object, no copy
                'ts'       : np.asarray(ts_arr, dtype=float),
            })

    # assemble DataFrame
    df = pd.DataFrame(records)

    # ensure correct dtypes for scalar columns
    if not df.empty:
        df['var_index'] = df['var_index'].astype(int)
        df['n_samples'] = df['n_samples'].astype(int)

    # summary
    print(f'\n{"-" * 60}')
    print(f'DataFrame shape : {df.shape}')
    print(f'Rows (signals)  : {len(df)}')
    print(f'Unique datasets : {df["dataset"].nunique() if not df.empty else 0}')
    print(f'Unique states   : {df["state"].unique().tolist() if not df.empty else []}')
    if skipped:
        print(f'Skipped         : {skipped}')
    print(f'{"-" * 60}\n')

    if not df.empty:
        print(df[['dataset', 'state', 'variable', 'var_index',
                   'n_samples']].to_string(index=False))

    return df
