import sys

import validation_roi


def main():
    sys.argv = [sys.argv[0], "--case", "watershed"] + sys.argv[1:]
    validation_roi.main()


if __name__ == "__main__":
    main()

