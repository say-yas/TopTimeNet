import h5py

def load_extended_teaspoon_dataset_from_h5(filepath: str, dataset: str, state: str):
    """
    Read one (dataset, state) entry from the HDF5 file.
    Returns (t, ts) where ts is a list of 1-D numpy arrays.
    """
    key = f'{dataset}/{state}'
    with h5py.File(filepath, 'r') as f:
        if key not in f:
            print(f'  Key not found in HDF5: {key}')
            return None
        grp     = f[key]
        t_data  = grp['t'][:]                          # (N,)
        ts_data = grp['ts'][:]                         # (n_signals, N)
    return t_data, list(ts_data)                       # list so ts[0] always works
