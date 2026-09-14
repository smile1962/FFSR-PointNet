import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import h5py
import numpy as np


def verify_coordinate_consistency(low_h5_path, high_h5_path, tol=1e-5):
    """
    Verify that the LR [6, N] and HR [4, N] datasets have identical spatial coordinates (channels 0 and 1).
    """
    low_h5 = h5py.File(low_h5_path, 'r')
    high_h5 = h5py.File(high_h5_path, 'r')

    low_keys = sorted(list(low_h5.keys()))
    high_keys = sorted(list(high_h5.keys()))

    assert len(low_keys) == len(high_keys), "sample counts do not match!"
    print(f"Verifying coordinate consistency for {len(low_keys)} samples...")

    all_consistent = True

    for idx, (lk, hk) in enumerate(zip(low_keys, high_keys)):
        ld = low_h5[lk][:]  # expected shape (6, N) or (N, 6)
        hd = high_h5[hk][:]  # expected shape (4, N) or (N, 4)

        # Convert everything to the (channels, sample points) format
        if ld.shape[0] != 6 and ld.shape[1] == 6: ld = ld.T
        if hd.shape[0] != 4 and hd.shape[1] == 4: hd = hd.T

        # Extract the X, Y coordinates of LR and HR (channels 0 and 1)
        coords_low = ld[0:2, :]
        coords_high = hd[0:2, :]

        if not np.allclose(coords_low, coords_high, atol=tol):
            print(f"Error: spatial coordinates (X, Y) of sample {lk} differ between LR and HR!")
            all_consistent = False
            break

    low_h5.close()
    high_h5.close()

    if all_consistent:
        print("[OK] All samples in the LR and HR datasets share identical spatial coordinates (X, Y)!")


if __name__ == "__main__":
    # Replace with your actual path
    verify_coordinate_consistency(rf"../{ROOT}\Data\dataset/Point_LR_geo.h5", rf"../{ROOT}\Data\dataset/Point_HR.h5")