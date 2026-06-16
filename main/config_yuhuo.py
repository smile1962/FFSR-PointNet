import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Flow Field Prediction Training")

    # Basic training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs") # default 200
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--global_feat_dim", type=int, default=512, help="Global feature dimension for the encoder")
    parser.add_argument("--use_samples", type=int, default=36, help="Number of samples to be used")
    parser.add_argument("--use_samples_val", type=int, default=72, help="Number of samples to be used")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion of training set") # default 0.7
    parser.add_argument("--test_ratio", type=float, default=0, help="Proportion of test set") # default 0.1
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Proportion of validation set") # default 0.2

    # Data directories and normalization configuration
    parser.add_argument("--val_high_res_folder", type=str, default=r"../dataset/Yuhuo_HR_val.h5", help="Folder for low-resolution data")
    parser.add_argument("--val_low_res_folder", type=str, default=r"../dataset/Yuhuo_LR_val.h5",
                        help="Folder for validation low-resolution data")
    parser.add_argument("--low_res_folder", type=str, default=r"../dataset/Yuhuo_LR.h5", help="Folder for validation low-resolution data")
    parser.add_argument("--high_res_folder", type=str, default=r"../dataset/Yuhuo_HR.h5", help="Folder for high-resolution data")
    parser.add_argument("--mean_std_file", type=str, default=r"./norm/mean_std_case_yuhuo.npz", help="File name to save/load mean and std values")

    # Model and weight saving paths
    parser.add_argument("--save_dir", type=str, default="../PointnetWeights", help="Directory for saving model weights")
    parser.add_argument("--pretrained_files", type=str, default="../PointnetWeights/best_sanjiang_pretrained.pth", help="Directory for saving model weights")
    parser.add_argument("--best_save_dir", type=str, default="../PointnetWeights/best_yuhuo_transfer_Sp36_Epoch200.pth", help="Directory for saving the best model weights")
    parser.add_argument("--final_save_dir", type=str, default="../PointnetWeights/final_yuhuo_transfer_Sp36_Epoch200.pth", help="Directory for saving the final model weights")
    parser.add_argument("--final_save_loss_dir", type=str, default="../PointnetWeights/loss_log_Sp36_Epoch200.txt", help="Directory for saving the final loss function values")
    parser.add_argument("--save_root", type=str, default="../resultSaving_yuhuo_transfer", help="Directory for saving the validation results")

    return parser.parse_args()


if __name__ == "__main__":
    config = get_config()
    print(config)
