import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path


class Config:
    """ROI-weighted patch SR-Unet configuration for Sanjiang town."""

    case_name = "Sanjiang"

    project_root = Path(rf"{ROOT}")
    unet_root = project_root / "Unet"
    h5_dir = project_root / "Data" / "dataset" / "Unet"

    lr_train_h5 = "SanJiang_LR.h5"
    hr_train_h5 = "SanJiang_HR.h5"
    lr_test_h5 = "SanJiang_LR_val.h5"
    hr_test_h5 = "SanJiang_HR_val.h5"
    meta_path = None

    roi_path = project_root / "Data" / "dataset" / "geo" / "Shancha_shuixi_1_5_mask_polyfilled.tif"
    expected_roi_count = 490746

    original_shape = (2010, 1094)
    padded_shape = (2048, 1152)
    pad_origin = (38, 58)  # rows added above, columns added left
    patch_size = (64, 64)

    split_mode = "sample"
    separate_test_h5 = True
    sample_split_seed = 42
    split_ratios = (0.7, 0.1, 0.2)

    weight_dir = project_root / "Weights" / "Unet" / "roi_village"
    result_dir = project_root / "Results" / "Unet" / "srunet_roi_sanjiang"
    norm_path = weight_dir / "roi_train_norm_village_new.npz"
    cache_path = weight_dir / "roi_patch_cache_village_new.h5"
    loss_log_path = weight_dir / "roi_loss_log_village.txt"
    checkpoint_path = weight_dir / "checkpoint_last_village.ckpt"
    best_weight_path = weight_dir / "best_roi_village.pth"

    num_epochs = 200
    batch_size = 32
    patch_batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-5
    lr_patience = 10
    early_stop_patience = 40
    zero_threshold = 1e-6
    value_threshold = 0.01
    error_threshold = 0.015
    seed = 42

    tiff_rows, tiff_cols = 2010, 1094
    cellsize_x, cellsize_y = 1.5, 1.5
    xmin, ymax = 11503199.581122924, 3623692.5104884803
    crs_wkt = """PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",
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
