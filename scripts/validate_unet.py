import sys

from runner import ROOT, run


if __name__ == "__main__":
    defaults = ["--case", "watershed"]
    if "--model" not in sys.argv[1:]:
        defaults.extend(["--model", "srunet"])
    sys.exit(
        run(
            r"validation_roi.py",
            rf"{ROOT}\Unet\roi",
            defaults=defaults,
            extra=sys.argv[1:],
        )
    )
