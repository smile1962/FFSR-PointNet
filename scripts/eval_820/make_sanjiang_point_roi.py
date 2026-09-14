"""Create a point-level ROI raster for Sanjiang point-cloud evaluations.

The Sanjiang point-cloud model outputs are fixed to the 130,618 grid cells
used when the h5 dataset was generated. This helper rasterizes those cells on
the same 1.5 m geometry used by the grid CNN outputs so ROI metrics can be
computed on a shared mask.
"""

from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HR_H5 = ROOT / "Data" / "dataset" / "SanJiang_HR.h5"
OUT_REAL2 = ROOT / "Data" / "dataset" / "geo" / "sanjiang_point_roi_real2.tif"
OUT_REAL4 = ROOT / "Data" / "dataset" / "geo" / "sanjiang_point_roi_real4.tif"

ROWS, COLS = 2010, 1094
XMIN, YMAX = 11503199.581122924, 3623692.5104884803
DXY = 1.5
CRS_WKT = """PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",
GEOGCS["GCS_WGS_1984",
    DATUM["D_WGS_1984",
        SPHEROID["WGS_1984",6378137.0,298.257223563]],
    PRIMEM["Greenwich",0.0],
    UNIT["Degree",0.0174532925199433]],
PROJECTION["Mercator_Auxiliary_Sphere"],
PARAMETER["False_Easting",0.0],
PARAMETER["False_Northing",0.0],
PARAMETER["Central_Meridian",0.0],
PARAMETER["Standard_Parallel_1",0.0],
PARAMETER["Auxiliary_Sphere_Type",0.0],
UNIT["Meter",1.0]]"""


def raster_mask(xy, out_tif):
    col = np.floor((xy[:, 0] - XMIN) / DXY).astype(np.int64)
    row = np.floor((YMAX - xy[:, 1]) / DXY).astype(np.int64)
    valid = (row >= 0) & (row < ROWS) & (col >= 0) & (col < COLS)
    if valid.sum() != 130618:
        raise RuntimeError(f"Expected 130618 valid cells, got {int(valid.sum())}")
    mask = np.zeros((ROWS, COLS), dtype=np.uint8)
    mask[row[valid], col[valid]] = 1
    transform = from_origin(XMIN, YMAX, DXY, DXY)
    with rasterio.open(
        str(out_tif),
        "w",
        driver="GTiff",
        height=ROWS,
        width=COLS,
        count=1,
        dtype="uint8",
        crs=CRS_WKT,
        transform=transform,
    ) as dst:
        dst.write(mask, 1)
    print(f"Saved point ROI with {int(mask.sum())} cells: {out_tif}")


def main():
    with h5py.File(str(HR_H5), "r") as f:
        keys = list(f.keys())
        key_real2 = next(k for k in keys if "(" not in k)
        key_real4 = next(k for k in keys if "(" in k)
        xy_real2 = f[key_real2][:, :2]
        xy_real4 = f[key_real4][:, :2]
    raster_mask(xy_real2, OUT_REAL2)
    raster_mask(xy_real4, OUT_REAL4)


if __name__ == "__main__":
    main()
