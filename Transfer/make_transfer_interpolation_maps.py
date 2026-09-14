"""Create 20 m coarse fields and 1.5 m interpolation maps for transfer cases.

For every transfer-result sample:

1. Coarsen the existing 1.5 m D_ori/U_ori field to a 20 m grid with area
   averaging.
2. Bilinearly resample the 20 m field back to the original 1.5 m HR grid.
3. Write the 20 m field back to D_ori/U_ori and the interpolated field to
   D_Int/U_Int.

The script is idempotent: if D_ori/U_ori is already a 20 m raster, it is used
directly as the coarse input.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import argparse
from math import ceil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject


DEFAULT_RESULT_ROOT = Path(rf"{ROOT}\Results\Transfer")
CASES = ("yuhua_light", "laoyangcun_light")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--result_root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
    )
    parser.add_argument(
        "--coarse_res",
        type=float,
        default=20.0,
        help="Target coarse cell size in metres.",
    )
    parser.add_argument(
        "--downsample",
        choices=("nearest", "average"),
        default="nearest",
        help="Resampling method used to create the 20 m field.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing D_Int/U_Int files.",
    )
    return parser.parse_args()


def read_profile_and_array(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.profile.copy(), src.transform


def reference_hr_geometry(case_dir):
    """Read the HR grid from a model result so all outputs share one geometry."""
    for sub in ("D_pred", "D_true"):
        files = sorted((case_dir / sub).glob("*.tif"))
        if files:
            with rasterio.open(files[0]) as src:
                return (
                    src.height,
                    src.width,
                    src.transform,
                    src.crs,
                    src.profile.copy(),
                )
    raise FileNotFoundError(f"No D_pred/D_true reference TIFF under {case_dir}")


def coarse_geometry(hr_height, hr_width, hr_transform, coarse_res):
    width = int(ceil(abs(hr_transform.a) * hr_width / coarse_res))
    height = int(ceil(abs(hr_transform.e) * hr_height / coarse_res))
    transform = from_origin(
        hr_transform.c,
        hr_transform.f,
        coarse_res,
        coarse_res,
    )
    return height, width, transform


def downsample_coarse(
    values,
    src_transform,
    src_crs,
    dst_height,
    dst_width,
    dst_transform,
    method,
):
    out = np.zeros((dst_height, dst_width), dtype=np.float32)
    reproject(
        source=values,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=None,
        dst_transform=dst_transform,
        dst_crs=src_crs,
        dst_nodata=0.0,
        resampling=Resampling[method],
    )
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, 0.0, None)


def bilinear_upsample(
    values,
    src_transform,
    src_crs,
    dst_height,
    dst_width,
    dst_transform,
):
    out = np.zeros((dst_height, dst_width), dtype=np.float32)
    reproject(
        source=values,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=None,
        dst_transform=dst_transform,
        dst_crs=src_crs,
        dst_nodata=0.0,
        resampling=Resampling.bilinear,
    )
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, 0.0, None)


def write_tif(path, values, profile, replace_existing=False):
    profile = profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=0.0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not replace_existing:
            raise FileExistsError(path)
        path.unlink()
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype(np.float32, copy=False), 1)


def process_case(case_dir, coarse_res, downsample_method, overwrite):
    hr_height, hr_width, hr_transform, hr_crs, hr_profile = (
        reference_hr_geometry(case_dir)
    )
    coarse_h, coarse_w, coarse_transform = coarse_geometry(
        hr_height,
        hr_width,
        hr_transform,
        coarse_res,
    )

    processed = 0
    for variable in ("D", "U"):
        ori_dir = case_dir / f"{variable}_ori"
        int_dir = case_dir / f"{variable}_Int"
        files = sorted(ori_dir.glob("*.tif"))
        if not files:
            raise FileNotFoundError(f"No TIFFs under {ori_dir}")

        for src_path in files:
            write_coarse = False
            with rasterio.open(src_path) as src:
                src_res = abs(src.transform.a)
                if abs(src_res - coarse_res) < 1e-6:
                    coarse = src.read(1).astype(np.float32)
                    coarse_transform_used = src.transform
                    coarse_profile = src.profile.copy()
                else:
                    values = src.read(1).astype(np.float32)
                    coarse = downsample_coarse(
                        values,
                        src.transform,
                        src.crs,
                        coarse_h,
                        coarse_w,
                        coarse_transform,
                        downsample_method,
                    )
                    coarse_transform_used = coarse_transform
                    coarse_profile = hr_profile.copy()
                    coarse_profile.update(
                        height=coarse_h,
                        width=coarse_w,
                        transform=coarse_transform,
                        crs=hr_crs,
                    )
                    write_coarse = True

            if write_coarse:
                write_tif(
                    src_path,
                    coarse,
                    coarse_profile,
                    replace_existing=overwrite,
                )

            dense = bilinear_upsample(
                coarse,
                coarse_transform_used,
                hr_crs,
                hr_height,
                hr_width,
                hr_transform,
            )
            threshold = 0.01 if variable == "D" else 0.0
            dense[dense < threshold] = 0.0

            out_path = int_dir / src_path.name
            if out_path.exists() and not overwrite:
                raise FileExistsError(
                    f"{out_path} exists; pass --overwrite to replace it."
                )
            write_tif(out_path, dense, hr_profile, replace_existing=overwrite)
            processed += 1

    return processed, (coarse_h, coarse_w), (hr_height, hr_width)


def main():
    args = parse_args()
    root = args.result_root.resolve()
    total = 0
    for case in CASES:
        case_dir = root / case
        if not case_dir.is_dir():
            raise FileNotFoundError(case_dir)
        count, coarse_shape, hr_shape = process_case(
            case_dir,
            args.coarse_res,
            args.downsample,
            args.overwrite,
        )
        total += count
        print(
            f"{case}: {count} files, "
            f"coarse={coarse_shape}, HR={hr_shape}"
        )
    print(f"Done. Processed {total} transfer interpolation files.")


if __name__ == "__main__":
    main()
