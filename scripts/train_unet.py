import sys

from runner import ROOT, run


if __name__ == "__main__":
    args = sys.argv[1:]
    defaults = []
    if "--case" not in args:
        defaults.extend(["--case", "watershed"])
    if "--model" not in args and "--num_epochs" not in args:
        defaults.extend(["--model", "srunet", "--num_epochs", "100"])
    sys.exit(
        run(
            r"train_roi_patch.py",
            rf"{ROOT}\Unet\roi",
            defaults=defaults,
            extra=args,
        )
    )
