"""
generate_datasets.py
Simulate every (dataset, state) in DATASET_REGISTRY and save to a single
HDF5 file. Run this once; it may take several minutes.

Output
------
    PATH_DATA/all_extended_teaspoon_datasets.h5
        {dataset_name}/
            {state}/
                t   : 1-D float64 time array
                ts  : 2-D float64 array (n_signals x SampleSize)

Usage
-----
    python generate_datasets.py
"""

import traceback
import numpy as np
import h5py
from src.dataset_generation.dataset_registry_extended_teaspoon import DATASET_REGISTRY, load_dataset_long


def generate_all(filepath: str) -> None:
    print(f'Saving all datasets to {filepath}\n')

    with h5py.File(filepath, 'w') as f:
        for dataset, (_, states, _cat) in DATASET_REGISTRY.items():
            states_to_run = states if states != [None] else [None]

            for state in states_to_run:
                key = f'{dataset}/{state or "default"}'
                print(f'  Simulating  {key} ...')

                data = load_dataset_long(dataset, state)
                if data is None:
                    print(f'    SKIP: load failed')
                    continue

                t_data, ts_data = data

                # normalise to numpy
                t_data = np.asarray(t_data, dtype=float)

                if isinstance(ts_data, (list, tuple)):
                    ts_data = np.array([np.asarray(s, dtype=float)
                                        for s in ts_data])   # (n_signals, N)
                else:
                    ts_data = np.asarray(ts_data, dtype=float)
                    if ts_data.ndim == 1:
                        ts_data = ts_data[np.newaxis, :]      # (1, N)

                grp = f.create_group(key)
                grp.create_dataset('t',  data=t_data,  compression='gzip')
                grp.create_dataset('ts', data=ts_data, compression='gzip')
                grp.attrs['dataset']  = dataset
                grp.attrs['state']    = state or 'default'
                grp.attrs['category'] = _cat
                print(f'    saved  t={t_data.shape}  ts={ts_data.shape}')

    print(f'\nDone. File: {filepath}')


if __name__ == '__main__':
    import sys

    DEFAULT_PATH = (
        '/Users/shararehsayyad/Documents/Projects/topological_data_analysis/'
        'Computational_topology/dataset/extended_teaspoon_dataset/'
    )

    # accept path from terminal: python generate_datasets.py /my/custom/path/
    # fall back to DEFAULT_PATH if nothing is passed
    PATH_DATA = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH

    # ensure trailing slash
    PATH_DATA = PATH_DATA.rstrip('/') + '/'

    print(f'Output directory : {PATH_DATA}')
    generate_all(PATH_DATA + 'all_extended_teaspoon_datasets.h5')
