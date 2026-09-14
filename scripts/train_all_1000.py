import os
import shutil
import subprocess
import sys
import time

from runner import PYTHON, ROOT


def run_step(name, args):
    cmd = [PYTHON, "-u"] + args
    print("\n" + "=" * 80)
    print(f"Starting: {name}")
    print(" ".join(cmd))
    print("=" * 80)
    t0 = time.time()
    ret = subprocess.call(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    if ret != 0:
        print(f"\nFAILED: {name} (return {ret})")
        sys.exit(ret)
    print(f"\nFinished: {name} in {elapsed / 60:.1f} minutes")


def archive_if_fresh(path):
    if not os.path.exists(path):
        return
    backup_root = os.path.join(ROOT, "Archive_UnusedModels", "before_all_1000")
    os.makedirs(backup_root, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(
        backup_root,
        os.path.basename(path) + "_before_all_1000_" + stamp,
    )
    shutil.move(path, dst)
    print(f"Moved existing {path} -> {dst}")


def main():
    fresh = "--fresh" in sys.argv
    skip_original = "--skip_original" in sys.argv
    if fresh:
        sys.argv.remove("--fresh")
    if skip_original:
        sys.argv.remove("--skip_original")

    light_dir = os.path.join(ROOT, "Weights", "FFSR_PointNet_Light", "full_rank128")
    flosr_dir = os.path.join(ROOT, "Weights", "Unet", "flosr_watershed")
    original_dir = os.path.join(ROOT, "Weights", "FFSR_PointNet", "retrain_1000")
    if fresh:
        archive_if_fresh(light_dir)
        archive_if_fresh(flosr_dir)
        archive_if_fresh(original_dir)

    run_step(
        "Light FFSR-PointNet 1000 epochs",
        [os.path.join(ROOT, "scripts", "train_light.py"), "--num_epochs", "1000"],
    )
    run_step(
        "FLO-SR 1000 epochs",
        [
            os.path.join(ROOT, "scripts", "train_unet.py"),
            "--model", "flosr",
            "--num_epochs", "1000",
            "--use_amp",
            "--disable_early_stop",
        ],
    )
    if not skip_original:
        run_step(
            "Full-parameter FFSR-PointNet 1000 epochs",
            [os.path.join(ROOT, "scripts", "train_original.py"), "--num_epochs", "1000"],
        )
    print("\nRequested 1000-epoch runs completed.")


if __name__ == "__main__":
    main()
