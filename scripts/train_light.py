import sys

from runner import ROOT, run


if __name__ == "__main__":
    defaults = [
        "--model_type", "subsampled",
        "--rank", "128",
        "--context_points", "8192",
        "--teacher_weight", "0",
        "--batch_size", "8",
        "--use_amp",
    ]
    sys.exit(
        run(
            r"train_compressed.py",
            rf"{ROOT}\FFSR-PointNet-Light\main",
            defaults=defaults,
            extra=sys.argv[1:],
        )
    )
