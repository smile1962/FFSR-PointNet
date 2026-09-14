import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse


"""
This part is used for the Shouxi River Watershed example training and verification
"""
def get_config():
    parser = argparse.ArgumentParser(description="Flow Field Prediction Training")

    # Basic training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--global_feat_dim", type=int, default=512, help="Global feature dimension for the encoder")
    parser.add_argument("--use_samples", type=int, default=272, help="Number of samples to be used")
    parser.add_argument("--use_samples_val", type=int, default=72, help="Number of samples to be used")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Proportion of training set") # default 0.7
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Proportion of test set") # default 0.1
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Proportion of validation set") # default 0.2
    parser.add_argument("--model_input_dim", type=int, default=4, help="Encoder hydraulic input channels")
    parser.add_argument("--prior_dim", type=int, default=4, help="HR prior channels for the implicit decoder")
    parser.add_argument("--output_dim", type=int, default=2, help="Predicted hydraulic output channels")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Adam weight decay")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--early_stop_patience", type=int, default=1000000, help="Validation patience; set a finite value to enable early stopping")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--use_amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CUDA automatic mixed precision",
    )
    parser.add_argument(
        "--amp_dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Automatic mixed precision dtype",
    )
    parser.add_argument(
        "--use_event_split",
        dest="use_event_split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use event-aware train/validation split",
    )
    parser.add_argument(
        "--train_events",
        nargs="+",
        default=["2year", "5year", "10year", "20year", "50year", "100year", "realRain", "real430"],
        help="Training events used when event-aware splitting is enabled",
    )
    parser.add_argument(
        "--val_events",
        nargs="+",
        default=["100year"],
        help="Held-out validation event",
    )

    # Data directories and normalization configuration
    parser.add_argument("--val_high_res_folder", type=str, default=rf"{ROOT}\Data\dataset/Point_HR_Val.h5", help="Folder for low-resolution data")
    parser.add_argument("--val_low_res_folder", type=str, default=rf"{ROOT}\Data\dataset/Point_LR_geo_val.h5",
                        help="Folder for validation low-resolution data")
    parser.add_argument("--low_res_folder", type=str, default=rf"{ROOT}\Data\dataset/Point_LR_geo.h5", help="Folder for validation low-resolution data")
    parser.add_argument("--high_res_folder", type=str, default=rf"{ROOT}\Data\dataset/Point_HR.h5", help="Folder for high-resolution data")
    parser.add_argument("--mean_std_file", type=str, default=r"./norm/mean_std_case_Shouxi.npz", help="File name to save/load mean and std values")

    # Model and weight saving paths
    parser.add_argument("--save_dir", type=str, default=rf"{ROOT}\Weights\FFSR_PointNet", help="Directory for saving model weights")
    parser.add_argument("--best_save_dir", type=str, default=rf"{ROOT}\Weights\FFSR_PointNet/best_shouxi_geo.pth", help="Directory for saving the best model weights")
    parser.add_argument("--final_save_dir", type=str, default=rf"{ROOT}\Weights\FFSR_PointNet/final_shouxi_v2_backbone.pth", help="Directory for saving the final model weights")
    parser.add_argument("--resume_checkpoint", type=str, default=rf"{ROOT}\Weights\FFSR_PointNet/resume_shouxi_v2_backbone.ckpt", help="Checkpoint for resuming interrupted training")
    parser.add_argument("--resume_checkpoint_interval", type=int, default=10, help="Save a resumable checkpoint every N epochs")
    parser.add_argument(
        "--save_optimizer_state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include optimizer and scheduler state in resumable checkpoints",
    )
    parser.add_argument("--loss_log_path", type=str, default=rf"{ROOT}\Weights\FFSR_PointNet/loss_log_shouxi_v2_backbone.txt", help="Loss history file")
    parser.add_argument("--save_root", type=str, default=rf"{ROOT}\Results\FFSR_PointNet\Shouxi_820", help="Directory for saving the validation results")

    return parser.parse_args()


"""
This part is used for the SanJiang Town example training and verification
"""
# import argparse
#
# def get_config():
#     parser = argparse.ArgumentParser(description="Flow Field Prediction Training")
#
#     # Basic training parameters
#     parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
#     parser.add_argument("--num_epochs", type=int, default=2000, help="Number of training epochs")
#     parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
#     parser.add_argument("--global_feat_dim", type=int, default=512, help="Global feature dimension for the encoder")
#     parser.add_argument("--use_samples", type=int, default=278, help="Number of samples to be used")
#     parser.add_argument("--train_ratio", type=float, default=0.7, help="Proportion of training set")
#     parser.add_argument("--test_ratio", type=float, default=0.1, help="Proportion of test set")
#     parser.add_argument("--val_ratio", type=float, default=0.2, help="Proportion of validation set")
#
#     # Data directories and normalization configuration
#     parser.add_argument("--low_res_folder", type=str, default=r"Data\dataset/SanJiang_LR.h5", help="Folder for low-resolution data")
#     parser.add_argument("--val_low_res_folder", type=str, default=r"Data\dataset/SanJiang_LR.h5", help="Folder for validation low-resolution data")
#     parser.add_argument("--high_res_folder", type=str, default=r"Data\dataset/SanJiang_HR.h5", help="Folder for high-resolution data")
#     parser.add_argument("--mean_std_file", type=str, default=r"./norm/mean_std_case_sanjiang.npz", help="File name to save/load mean and std values")
#
#     # Model and weight saving paths
#     parser.add_argument("--save_dir", type=str, default="Weights\FFSR_PointNet", help="Directory for saving model weights")
#     parser.add_argument("--best_save_dir", type=str, default="Weights\FFSR_PointNet/best_sanjiang_geo.pth", help="Directory for saving the best model weights")
#     parser.add_argument("--final_save_dir", type=str, default="Weights\FFSR_PointNet/final_sanjiang_geo.pth", help="Directory for saving the final model weights")
#     parser.add_argument("--save_root", type=str, default="../resultSaving_SanJiang", help="Directory for saving the validation results")
#
#     return parser.parse_args()

if __name__ == "__main__":
    config = get_config()
    print(config)
