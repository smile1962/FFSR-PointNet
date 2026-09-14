import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path


class Config:
    """ROI-weighted patch SR-Unet configuration for Shouxi watershed."""

    case_name = "Shouxi"

    project_root = Path(rf"{ROOT}")
    unet_root = project_root / "Unet"
    h5_dir = project_root / "Data" / "dataset" / "Unet"

    lr_train_h5 = "LR_Shouxi.h5"
    hr_train_h5 = "HR_Shouxi.h5"
    lr_test_h5 = "LR_Shouxi_val.h5"
    hr_test_h5 = "HR_Shouxi_val.h5"
    meta_path = h5_dir / "geo_inside" / "metadata_val.mat"

    roi_path = project_root / "Data" / "dataset" / "geo" / "shouxi_mask_30_Point.tif"
    expected_roi_count = 126460

    original_shape = (1075, 1779)
    padded_shape = (1200, 1800)
    pad_origin = (125, 21)  # rows added above, columns added left
    patch_size = (120, 120)

    split_mode = "events"
    # Inclusive ranges [start, end) inside LR_Shouxi.h5 / HR_Shouxi.h5.
    train_event_ranges = [(0, 120), (144, 272)]
    val_event_ranges = [(120, 144)]
    test_event_ranges = None  # None means all samples in the test h5

    sample_split_seed = 42
    split_ratios = (0.7, 0.1, 0.2)

    weight_dir = project_root / "Weights" / "Unet" / "roi_watershed"
    result_dir = project_root / "Results" / "Unet" / "srunet_roi_shouxi_820"
    norm_path = weight_dir / "roi_train_norm_watershed.npz"
    cache_path = weight_dir / "roi_patch_cache_watershed.h5"
    loss_log_path = weight_dir / "roi_loss_log_watershed.txt"
    checkpoint_path = weight_dir / "checkpoint_last_watershed.ckpt"
    best_weight_path = weight_dir / "best_roi_watershed.pth"

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

    tiff_rows, tiff_cols = 1075, 1779
    cellsize_x, cellsize_y = 30.0, 30.0
    xmin, ymax = 11469243.372897469, 3638166.6798295653
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
