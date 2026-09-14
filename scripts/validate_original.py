import sys

from runner import ROOT, run


if __name__ == "__main__":
    sys.exit(
        run(
            r"validate_v2.py",
            rf"{ROOT}\FFSR-PointNet\main",
            extra=sys.argv[1:],
        )
    )

