import os


class Config:
    """Super-resolution configuration parameters (all adjustable here)"""

    # -------------------------- Model parameters --------------------------
    # patch_size = (64, 64)  # patch size
    # patch_batch_size = 32  # batch size for patch prediction
    # model_weight_path = "sr_model_San.pth"  # model weight file
    patch_size = (120, 120)  # patch size
    patch_batch_size = 32  # batch size for patch prediction
    threshold_D = 0.01  # depth values below this are set to 0
    threshold_U = 0.01  # velocity values below this are set to 0
    model_weight_path = "sr_model_NoThreshold.pth"  # model weight file

    # -------------------------- Path parameters --------------------------
    data_dir = r"G:\Projects\ShouXi\python\dataset"  # root data directory
    result_dir = r"G:\Projects\ShouXi\python\resulting"  # root result directory
    result_dir_shouxi = r"G:\Projects\ShouXi\python\debug"
    weight_dir = r"G:\Projects\ShouXi\python\weights"

    low_h5 = "LR_val.h5"  # low-resolution data file (already bilinearly interpolated)
    high_h5 = "HR_val.h5"  # high-resolution data file
    meta_path = "metadata_val.mat"  # metadata file

    # -------------------------- Dataset split --------------------------
    split_ratios = [0.7, 0.1, 0.2]  # [train, test, val] ratio
    random_seed = 1  # random seed (keeps the split consistent)

    # -------------------------- Output control --------------------------
    save_fig = False  # whether to save the visualization
    output_subdirs = [  # result subdirectories to create
        "U_pred", "U_ori", "U_true", "U_error",
        "U_coeff_bicbuic", "D_coeff_bicbuic",
        "U_coeff", "D_coeff",
        "D_pred", "D_ori", "D_true", "D_error"
    ]

    # -------------------------- TIFF georeferencing (kept consistent with PointNet) --------------------------
#     tiff_rows = 2010  # number of TIFF rows
#     tiff_cols = 1094  # number of TIFF columns
#     cellsize_x = 1.5  # X resolution (m)
#     cellsize_y = 1.5  # Y resolution (m)
#     xmin = 11503199.581122924  # top-left X coordinate
#     ymin = 3620677.5104884803  # bottom-left Y coordinate
#     xmax = 11504840.581122924  # bottom-right X coordinate
#     ymax = 3623692.5104884803  # top-left Y coordinate
#     crs_wkt = """PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",
# GEOGCS["GCS_WGS_1984",
#     DATUM["D_WGS_1984",
#         SPHEROID["WGS_1984",6378137.0,298.257223563]],
#     PRIMEM["Greenwich",0.0],
#     UNIT["Degree",0.0174532925199433]],
# PROJECTION["Mercator_Auxiliary_Sphere"],
# PARAMETER["False_Easting",0.0],
# PARAMETER["False_Northing",0.0],
# PARAMETER["Central_Meridian",0.0],
# PARAMETER["Standard_Parallel_1",0.0],
# PARAMETER["Auxiliary_Sphere_Type",0.0],
# UNIT["Meter",1.0]]"""  # projection information
    tiff_rows, tiff_cols = 1075, 1779
    cellsize_x, cellsize_y = 30.0, 30.0
    xmin, ymin, xmax, ymax = (
        11469243.372897469,
        3605916.6798295653,
        11522613.372897469,
        3638166.6798295653,
    )
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

    # -------------------------- Channel index configuration (adjust to your data) --------------------------
    lr_d_channel = 0  # channel holding the low-resolution depth data
    lr_u_channel = 1  # channel holding the low-resolution velocity data
    lr_feat_channels = [2, 3]  # low-resolution feature channels (model input)
    hr_d_channel = 0  # channel holding the high-resolution depth data
    hr_u_channel = 1  # channel holding the high-resolution velocity data


# Instantiate the configuration object (for import by other files)
config = Config()
