"""Create one shared train/val/test split for Sanjiang point and grid models.

The grid h5 is ordered real2-first then real4. The point-cloud h5 keys are
interleaved by timestamp (real2 and real4 with the same time are adjacent), so
using identical numeric split indices does not select the same flood states.
This script derives point indices from the authoritative grid split and saves a
mapping that both point-cloud training and evaluation should consume.
"""

import json
import sys
from pathlib import Path

import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT / "Unet" / "roi"))

import roi_data  # noqa: E402


def point_index_for_grid(grid_idx):
    grid_idx = int(grid_idx)
    if grid_idx < 144:
        return 2 * grid_idx
    return 2 * (grid_idx - 144) + 1


def main():
    cfg = roi_data.get_config("village")
    split = roi_data.get_sample_indices(cfg)
    records = sio.loadmat(
        str(ROOT / "Data" / "dataset" / "Unet" / "SanJiang" / "metadata_val.mat"),
        squeeze_me=True,
        struct_as_record=False,
    )["metadata"]

    grid_to_point = {}
    point_split = {}
    for subset in ("train", "val", "test"):
        point_split[subset] = [
            point_index_for_grid(g) for g in split[subset]
        ]
    for gidx, rec in enumerate(records):
        grid_to_point[gidx] = point_index_for_grid(gidx)

    out = {
        "grid_train": split["train"],
        "grid_val": split["val"],
        "grid_test": split["test"],
        "point_train": point_split["train"],
        "point_val": point_split["val"],
        "point_test": point_split["test"],
        "grid_to_point": grid_to_point,
        "note": (
            "grid indices follow LR_San.h5 event-blocked order; point indices "
            "follow the natural-key order of SanJiang_LR.h5/SanJiang_HR.h5."
        ),
    }
    dest = ROOT / "Data" / "dataset" / "sanjiang_grid_to_point_split.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {dest}")
    print("point_test:", out["point_test"])


if __name__ == "__main__":
    main()
