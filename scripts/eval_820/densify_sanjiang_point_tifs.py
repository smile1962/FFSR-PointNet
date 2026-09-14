"""Densify sparse Sanjiang point-cloud GeoTIFFs for visualization.

The PointNet Sanjiang outputs were trained on a 3 m point lattice but were
written directly onto the 1.5 m HR raster.  Consequently, only cells whose
coordinates coincide with the 3 m lattice contain values; the surrounding
cells remain zero and appear as a checkerboard/scatter pattern.

This utility performs a nearest-neighbour expansion of the sparse lattice onto
the complete 1.5 m raster.  It does not rerun or modify any model.  Values at
the sparse lattice nodes are preserved exactly, including zero-valued dry
nodes.  The Sanjiang mask is used only to confirm the target 1.5 m geometry;
source values are not clipped to it, because the historical point-cloud ROI is
shifted by one 1.5 m cell at part of its boundary.
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MASK = (
    ROOT / "Data" / "dataset" / "geo" / "Shancha_shuixi_1_5_mask_polyfilled.tif"
)
SKIP_PARTS = {"D_coeff", "U_coeff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def read_mask_shape(path):
    with rasterio.open(path) as src:
        return src.shape


def infer_phase(arr):
    """Return the single 2-pixel lattice phase represented by non-zero cells."""
    nz = arr != 0
    if not nz.any():
        return None
    row_mods = set(np.unique(np.where(nz.any(axis=1))[0] % 2).tolist())
    col_mods = set(np.unique(np.where(nz.any(axis=0))[0] % 2).tolist())
    if len(row_mods) != 1 or len(col_mods) != 1:
        return None
    return int(next(iter(row_mods))), int(next(iter(col_mods)))


def is_lattice_field(arr, phase):
    nz = arr != 0
    if not nz.any():
        return True
    row_mod, col_mod = phase
    lattice = np.zeros(arr.shape, dtype=bool)
    lattice[row_mod::2, col_mod::2] = True
    return not bool(np.any(nz & ~lattice))


def densify(arr, phase):
    row_mod, col_mod = phase
    lattice = np.zeros(arr.shape, dtype=bool)
    lattice[row_mod::2, col_mod::2] = True
    _, indices = distance_transform_edt(~lattice, return_indices=True)
    dense = arr[indices[0], indices[1]].astype(np.float32, copy=False)
    return dense


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_tif(path, arr, profile):
    profile = profile.copy()
    profile.update(count=1, dtype="float32", nodata=0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32, copy=False), 1)


def main():
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else input_root.with_name(input_root.name + "-Dense")
    )
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_root} is not empty; pass --overwrite to replace its contents."
        )
    target_shape = read_mask_shape(args.mask)
    paths = sorted(input_root.rglob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"No TIF files under {input_root}")

    candidates = []
    for path in paths:
        rel = path.relative_to(input_root)
        if SKIP_PARTS.intersection(rel.parts):
            continue
        with rasterio.open(path) as src:
            if src.count != 1 or src.shape != target_shape:
                continue
            arr = src.read(1)
        phase = infer_phase(arr)
        if phase is not None and is_lattice_field(arr, phase):
            candidates.append((path, rel, phase))

    if not candidates:
        raise RuntimeError("No sparse 1.5 m lattice TIFs were detected.")
    global_phase = Counter(phase for _, _, phase in candidates).most_common(1)[0][0]
    print(f"Detected Sanjiang lattice phase: row%2={global_phase[0]}, "
          f"col%2={global_phase[1]}")
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")

    processed = 0
    copied = 0
    for path in paths:
        rel = path.relative_to(input_root)
        dst = output_root / rel
        if SKIP_PARTS.intersection(rel.parts):
            copy_file(path, dst)
            copied += 1
            continue
        with rasterio.open(path) as src:
            profile = src.profile.copy()
            if src.count != 1 or src.shape != target_shape:
                copy_file(path, dst)
                copied += 1
                continue
            arr = src.read(1)
        phase = infer_phase(arr)
        if phase is None or not is_lattice_field(arr, phase):
            copy_file(path, dst)
            copied += 1
            continue
        dense = densify(arr, global_phase)
        write_tif(dst, dense, profile)
        processed += 1

    print(f"Densified {processed} TIFs; copied {copied} non-sparse/auxiliary files.")
    print("Original files were left untouched.")


if __name__ == "__main__":
    main()
