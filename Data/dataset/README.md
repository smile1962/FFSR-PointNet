# Dataset

## ⚠️ About this folder

**The full datasets are NOT included in this repository** — they are far too large
(tens of GB) and the underlying hydrodynamic simulation results are subject to
data-sharing restrictions.

To let you **run and verify the code end-to-end**, this folder ships a
**dummy dataset**: synthetic data with **exactly the same structure** as the
real one (same HDF5 key naming, per-sample shape, dtype and attributes), but
only **1–2 samples per file**.

> The numbers in the dummy files are **synthetic and physically meaningless**.
> They are only meant for smoke-testing the pipeline (shape checks, normalization,
> model forward/backward, plotting). **Do not report any accuracy metric computed
> on this dummy data.**

Total size: ~27 MB. Coordinates are on a regular grid, terrain is a smooth
analytic surface, and the flow fields are sparse (≈6 % wet cells) so that the
files compress well.

---

## Point-cloud datasets (`Data/dataset/*.h5`)

Each file stores one HDF5 **dataset per time step**; every dataset has shape
`(N_points, C)` with `dtype=float64`.

| File | Case | Samples | Per-sample shape | Columns |
|---|---|---|---|---|
| `Point_HR.h5` | Shouxi | 272 → **2** | (126460, 4) | X, Y, depth, velocity |
| `Point_HR_Val.h5` | Shouxi (8.20 event) | 72 → **2** | (126460, 4) | X, Y, depth, velocity |
| `Point_LR_geo.h5` | Shouxi | 272 → **2** | (126460, 6) | X, Y, depth, velocity, DEM, slope |
| `Point_LR_geo_val.h5` | Shouxi (8.20 event) | 72 → **2** | (126460, 6) | X, Y, depth, velocity, DEM, slope |
| `Point_LR.h5` | Shouxi (non-interpolated LR) | 272 → **2** | (4347, 4) | X, Y, depth, velocity |
| `Point_LR_Val.h5` | Shouxi (non-interpolated LR) | 72 → **2** | (4347, 4) | X, Y, depth, velocity |
| `SanJiang_HR.h5` | Sanjiang | 279 → **2** | (130618, 4) | X, Y, depth, velocity |
| `SanJiang_LR.h5` | Sanjiang | 279 → **2** | (130618, 6) | X, Y, depth, velocity, DEM, slope |
| `SanJiang_HR_val.h5` | Sanjiang | 1 → **1** | (490746, 4) | X, Y, depth, velocity |
| `SanJiang_LR_val.h5` | Sanjiang | 1 → **1** | (4177, 4) | X, Y, depth, velocity |
| `Yuhuo_HR.h5` | Yuhuo (transfer) | 72 → **2** | (79835, 4) | X, Y, depth, velocity |
| `Yuhuo_LR.h5` | Yuhuo (transfer) | 72 → **2** | (79835, 6) | X, Y, depth, velocity, DEM, slope |
| `Yuhuo_HR_val.h5` | Yuhuo (transfer) | 72 → **2** | (79835, 4) | X, Y, depth, velocity |
| `Yuhuo_LR_val.h5` | Yuhuo (transfer) | 72 → **2** | (79835, 6) | X, Y, depth, velocity, DEM, slope |
| `LaoYangCun_HR.h5` | LaoYangCun (transfer) | 110 → **2** | (481233, 4) | X, Y, depth, velocity |
| `LaoYangCun_LR.h5` | LaoYangCun (transfer) | 110 → **2** | (481233, 6) | X, Y, depth, velocity, DEM, slope |
| `LaoYangCun_HR_val.h5` | LaoYangCun (transfer) | 66 → **2** | (481233, 4) | X, Y, depth, velocity |
| `LaoYangCun_LR_val.h5` | LaoYangCun (transfer) | 66 → **2** | (481233, 6) | X, Y, depth, velocity, DEM, slope |

The `meta` attribute is a JSON string describing each time step, e.g.
`[{"event": "2year", "time": "2025-08-06 00:30"}, ...]`.

## Grid datasets (`Data/dataset/Unet/*.h5`)

A single dataset named `data` with shape `(N, C, H, W)`, `dtype=float32`.

| File | Case | Samples | Shape | Channels |
|---|---|---|---|---|
| `HR_Shouxi.h5` | Shouxi | 272 → **2** | (N, 2, 1200, 1800) | depth, velocity |
| `LR_Shouxi.h5` | Shouxi | 272 → **2** | (N, 4, 1200, 1800) | depth, velocity, DEM, slope |
| `HR_Shouxi_val.h5` | Shouxi | 72 → **2** | (N, 2, 1200, 1800) | depth, velocity |
| `LR_Shouxi_val.h5` | Shouxi | 72 → **2** | (N, 4, 1200, 1800) | depth, velocity, DEM, slope |
| `SanJiang_HR.h5` | Sanjiang | 279 → **2** | (N, 2, 2048, 1152) | depth, velocity |
| `SanJiang_LR.h5` | Sanjiang | 279 → **2** | (N, 4, 2048, 1152) | depth, velocity, DEM, slope |
| `SanJiang_HR_val.h5` | Sanjiang | 1 → **1** | (N, 2, 2048, 1152) | depth, velocity |
| `SanJiang_LR_val.h5` | Sanjiang | 1 → **1** | (N, 4, 2048, 1152) | depth, velocity, DEM, slope |

`LR_*` is the low-resolution input; `HR_*` is the high-resolution target.
Invalid cells are stored as `0`.

---

## Using the real dataset

Replace the files in this folder with your own data **keeping the same names,
shapes and channel order**. Nothing else in the code needs to change.

To build the point-cloud HDF5 files from HEC-RAS GeoTIFF outputs, see
`SR-PointNet/main/utils/h5Gen.py`; the grid HDF5 files are produced by
`Unet/utils/datasetGen_h5.py`.
