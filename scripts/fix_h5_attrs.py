"""
fix_h5_attrs.py — repair corrupted h5 file attributes (after AutoDL hard shutdown).

Diagnoses: shape is OK but attrs['nt'] / etc. raise `Unsupported integer size (0)`.
Cause: hard shutdown during a previous run left attribute metadata not flushed.
Fix:  delete each attr (delete works even if reading fails), then re-write with
      known-correct values.

We hard-code attribute values for the standard paper-FOPC config:
    nt   = 11        (number of time steps)
    dt   = 0.1       (time step)
    nx   = 128       (spatial grids)
    dx   = 1/129     (spatial step)
    tmin = 0.0
    tmax = 1.0
    x    = linspace(dx, 1-dx, 128)  (spatial grid points)

Usage:
    python scripts/fix_h5_attrs.py data/free_u_f_paper_fopc/burgers_train.h5
    python scripts/fix_h5_attrs.py data/free_u_f_paper_fopc/burgers_test.h5
"""
import sys
import h5py
import numpy as np


# Paper FOPC config — these are fixed.
CONFIG = {
    'nt':   np.int64(11),
    'dt':   np.float64(0.1),
    'nx':   np.int64(128),
    'dx':   np.float64(1.0 / 129),
    'tmin': np.float64(0.0),
    'tmax': np.float64(1.0),
}
# x grid: linspace from dx to xmax-dx, 128 points
xmin, xmax = 0.0, 1.0
s = 128
delta_x = (xmax - xmin) / (s + 1)
CONFIG['x'] = np.linspace(xmin + delta_x, xmax - delta_x, s, dtype=np.float64)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/fix_h5_attrs.py <h5_file>")
        sys.exit(1)
    path = sys.argv[1]
    print(f"Repairing {path}")

    with h5py.File(path, 'r+') as f:
        # Find the dataset to repair. Walk both train/test groups.
        for mode in ['train', 'test']:
            if mode not in f:
                continue
            grp = f[mode]
            for ds_name in list(grp.keys()):
                if ds_name.endswith('_f'):
                    continue  # only repair main pde_*-* dataset, not the _f variant
                ds = grp[ds_name]
                print(f"  [{mode}/{ds_name}] shape={ds.shape}")
                for key in list(ds.attrs.keys()):
                    try:
                        del ds.attrs[key]
                        print(f"    deleted attr '{key}'")
                    except Exception as e:
                        print(f"    could not delete '{key}': {e}")
                for key, val in CONFIG.items():
                    ds.attrs[key] = val
                    print(f"    set    attr '{key}' = {val if not hasattr(val, 'shape') else val.shape}")
    print("✅ Done. Verify with: python -c \"import h5py; f=h5py.File('"
          + path + "','r'); ds=f['train' if 'train' in f else 'test'][list(f[list(f.keys())[0]].keys())[0]];"
          " print('nt:', ds.attrs['nt'], 'shape:', ds.shape); f.close()\"")


if __name__ == "__main__":
    main()
