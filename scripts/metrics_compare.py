from runner import ROOT, run


if __name__ == "__main__":
    raise SystemExit(
        run(
            r"metrics_roi_three_models.py",
            rf"{ROOT}\Unet\roi",
        )
    )

