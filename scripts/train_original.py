import sys

from runner import ROOT, run


if __name__ == "__main__":
    defaults = [
        "--save_dir", rf"{ROOT}\Weights\FFSR_PointNet\retrain_1000",
        "--best_save_dir", rf"{ROOT}\Weights\FFSR_PointNet\retrain_1000\best_shouxi_geo.pth",
        "--final_save_dir", rf"{ROOT}\Weights\FFSR_PointNet\retrain_1000\final_shouxi_v2_backbone.pth",
        "--resume_checkpoint", rf"{ROOT}\Weights\FFSR_PointNet\retrain_1000\resume_shouxi_v2_backbone.ckpt",
        "--loss_log_path", rf"{ROOT}\Weights\FFSR_PointNet\retrain_1000\loss_log_shouxi_v2_backbone.txt",
    ]
    sys.exit(
        run(
            r"run.py",
            rf"{ROOT}\FFSR-PointNet\main",
            defaults=defaults,
            extra=sys.argv[1:],
        )
    )
