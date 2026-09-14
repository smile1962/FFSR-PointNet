import argparse
import os


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "Data", "dataset")
WEIGHTS = os.path.join(ROOT, "Weights", "FFSR_PointNet_Light")

CASE_PROFILES = {
    "shouxi": {
        "label": "Shouxi",
        "use_samples": 272,
        "low_res_folder": os.path.join(DATA, "Point_LR_geo.h5"),
        "high_res_folder": os.path.join(DATA, "Point_HR.h5"),
        "val_low_res_folder": os.path.join(DATA, "Point_LR_geo_val.h5"),
        "val_high_res_folder": os.path.join(DATA, "Point_HR_Val.h5"),
        "mean_std_file": os.path.join(
            os.path.dirname(__file__), "norm", "mean_std_case_Shouxi.npz"
        ),
        "teacher_path": os.path.join(
            ROOT, "Weights", "FFSR_PointNet", "best_shouxi_geo.pth"
        ),
        "save_dir": os.path.join(WEIGHTS, "full_rank128"),
        "training_time_file": "training_time_shouxi.txt",
    },
    "sanjiang": {
        "label": "Sanjiang",
        "use_samples": 279,
        "low_res_folder": os.path.join(DATA, "SanJiang_LR.h5"),
        "high_res_folder": os.path.join(DATA, "SanJiang_HR.h5"),
        "val_low_res_folder": os.path.join(DATA, "SanJiang_LR.h5"),
        "val_high_res_folder": os.path.join(DATA, "SanJiang_HR.h5"),
        "mean_std_file": os.path.join(
            os.path.dirname(__file__), "norm", "mean_std_case_Sanjiang.npz"
        ),
        "teacher_path": os.path.join(
            ROOT, "Weights", "FFSR_PointNet", "best_sanjiang_geo.pth"
        ),
        "save_dir": os.path.join(WEIGHTS, "sanjiang_rank128"),
        "training_time_file": "training_time_sanjiang.txt",
        "split_map_file": os.path.join(DATA, "sanjiang_grid_to_point_split.json"),
    },
}


def _case_name(value):
    case = str(value).strip().lower()
    aliases = {
        "watershed": "shouxi",
        "shouxi": "shouxi",
        "village": "sanjiang",
        "sanjiang": "sanjiang",
    }
    if case not in aliases:
        raise argparse.ArgumentTypeError("case must be shouxi or sanjiang")
    return aliases[case]


def get_config():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--case", type=_case_name, default="shouxi")
    known, _ = bootstrap.parse_known_args()
    profile = CASE_PROFILES[known.case]

    parser = argparse.ArgumentParser(description="FFSR-Light training")
    parser.add_argument("--case", type=_case_name, default=known.case)
    parser.add_argument("--batch_size", "--batch-size", "--batch", type=int, default=4)
    parser.add_argument("--num_epochs", "--epoch", "--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_samples", "--samples", type=int, default=profile["use_samples"])

    parser.add_argument("--low_res_folder", type=str, default=profile["low_res_folder"])
    parser.add_argument("--high_res_folder", type=str, default=profile["high_res_folder"])
    parser.add_argument("--val_low_res_folder", type=str, default=profile["val_low_res_folder"])
    parser.add_argument("--val_high_res_folder", type=str, default=profile["val_high_res_folder"])
    parser.add_argument("--mean_std_file", type=str, default=profile["mean_std_file"])
    parser.add_argument("--save_dir", type=str, default=profile["save_dir"])
    parser.add_argument(
        "--best_save_dir",
        type=str,
        default=os.path.join(profile["save_dir"], "compressed_best.pth"),
    )
    parser.add_argument(
        "--final_save_dir",
        type=str,
        default=os.path.join(profile["save_dir"], "compressed_final.pth"),
    )
    parser.add_argument(
        "--loss_log_path",
        type=str,
        default=os.path.join(profile["save_dir"], "loss_compressed.csv"),
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=os.path.join(profile["save_dir"], "resume_light.ckpt"),
    )
    parser.add_argument("--prior_dim", type=int, default=6)
    parser.add_argument("--model_input_dim", type=int, default=4)
    parser.add_argument("--output_dim", type=int, default=2)
    parser.add_argument("--teacher_path", type=str, default=profile["teacher_path"])
    parser.add_argument("--teacher_weight", type=float, default=0.0)
    parser.add_argument("--resume_checkpoint_interval", type=int, default=5)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--context_points", type=int, default=8192)
    parser.add_argument(
        "--model_type",
        choices=["subsampled"],
        default="subsampled",
        help="Official Light architecture (subsampled T-Net + rank-limited decoder)",
    )
    config = parser.parse_args()
    config.case_name = profile["label"].lower()
    config.training_time_file = profile["training_time_file"]
    config.split_map_file = profile.get("split_map_file")
    return config
