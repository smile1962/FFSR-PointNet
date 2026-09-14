"""
Laoyangcun batch rendering
===================

Iterate over all samples under BASE_DIR, calling the six-panel renderer in Laoyangcun_export.py for each
Plotting function; results are saved uniformly to BASE_DIR\\Arcgis.

The plotting style (layout, palette, colorbar ranges) all comes from Laoyangcun_export.py,
It is not redefined here; change it there and the batch version follows automatically.

Output filename: Figure_<Depth|Velocity>_<sample name>.png
e.g.        Figure_Depth_2026-09-07_00-10.png

Usage:
    python Laoyangcun_export_batch.py                 # all samples, 4 processes by default
    python Laoyangcun_export_batch.py --workers 8     # specify the number of processes
    python Laoyangcun_export_batch.py --limit 2       # run only the first 2 samples (test)
    python Laoyangcun_export_batch.py --resume        # skip images that already exist
    python Laoyangcun_export_batch.py --out-dir D:\\tmp\\out
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# The backend must be set before importing pyplot so figures render correctly under multiprocessing
os.environ.setdefault("MPLBACKEND", "Agg")

# Let subprocesses import Laoyangcun_export.py from the same directory
sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

from Laoyangcun_export import render_6panel_figure


# ============================================================
# Configuration
# ============================================================
BASE_DIR = (
    rf"{ROOT}"
    r"\Results\Transfer\laoyangcun_light"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "Arcgis"
)

SHP_PATH = (
    rf"{ROOT}"
    r"\Data\dataset\geo\laoyangcun"
    r"\boundary.shp"
)

HILLSHADE_PATH = (
    rf"{ROOT}"
    r"\Data\dataset\geo\laoyangcun"
    r"\HillShade.tif"
)

ROI_PATH = (
    rf"{ROOT}"
    r"\Data\dataset\geo"
    r"\Shancha_shuixi_1_5_mask_polyfilled.tif"
)

# Variable type
VAR_TYPES = ['Depth', 'Velocity']

# The four input subfolders required per variable (relative to BASE_DIR)
SUB_FOLDERS = ['ori', 'true', 'pred', 'Int']

# Number of parallel processes (tune to your machine; set to 1 if memory is tight)
MAX_WORKERS = 4


# ============================================================
# Collect samples
# ============================================================
def collect_samples():

    """Return [(var_type, stem, filename), ...] sorted by variable and sample name."""

    tasks = []

    for var_type in VAR_TYPES:

        prefix = 'D' if var_type == 'Depth' else 'U'

        # Use the fine-grid CFD results (true) as the reference and list all samples
        ref_dir = os.path.join(
            BASE_DIR,
            f'{prefix}_true'
        )

        if not os.path.isdir(ref_dir):
            print(f"[WARN] directory does not exist; skipping {var_type}: {ref_dir}")
            continue

        stems = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(ref_dir)
            if f.lower().endswith('.tif')
        )

        for stem in stems:

            fname = f'{stem}.tif'

            missing = [
                sub for sub in SUB_FOLDERS
                if not os.path.exists(
                    os.path.join(
                        BASE_DIR,
                        f'{prefix}_{sub}',
                        fname
                    )
                )
            ]

            if missing:
                lack = ", ".join(
                    f'{prefix}_{sub}' for sub in missing
                )
                print(f"[WARN] {var_type} {stem} is missing {lack}; skipping")
                continue

            tasks.append((var_type, stem, fname))

    return tasks


# ============================================================
# Single task (executed in a subprocess)
# ============================================================
def render_one(task):

    var_type, stem, fname, out_dir = task

    out_name = f'Figure_{var_type}_{stem}.png'

    t0 = time.time()

    render_6panel_figure(
        var_type=var_type,
        base_path=BASE_DIR,
        filename=fname,
        shp_path=SHP_PATH,
        hillshade_path=HILLSHADE_PATH,
        roi_path=ROI_PATH,
        out_dir=out_dir,
        out_name=out_name
    )

    return out_name, time.time() - t0


# ============================================================
# Used by --resume: check whether an image has already been generated
# ============================================================
def already_done(task, out_dir):

    path = os.path.join(
        out_dir,
        f'Figure_{task[0]}_{task[1]}.png'
    )

    return (
        os.path.exists(path)
        and os.path.getsize(path) > 10_000
    )


# ============================================================
# Main program
# ============================================================
if __name__ == "__main__":

    # ---------------- Command-line arguments ----------------
    workers = MAX_WORKERS
    limit = None
    resume = False
    out_dir = OUT_DIR

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--workers' and i + 1 < len(argv):
            workers = max(1, int(argv[i + 1]))
            i += 2
        elif a == '--limit' and i + 1 < len(argv):
            limit = int(argv[i + 1])
            i += 2
        elif a == '--out-dir' and i + 1 < len(argv):
            out_dir = argv[i + 1]
            i += 2
        elif a == '--resume':
            resume = True
            i += 1
        else:
            print(f"[WARN] unknown argument: {a}")
            i += 1

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("BASE_DIR : %s" % BASE_DIR)
    print("OUT_DIR  : %s" % out_dir)
    print("workers  : %d" % workers)
    print("=" * 78)

    tasks = collect_samples()

    if limit is not None:
        tasks = tasks[:limit]

    if resume:
        before = len(tasks)
        tasks = [
            t for t in tasks
            if not already_done(t, out_dir)
        ]
        print(
            f"[resume] skipped {before - len(tasks)} existing images,"
            f"{len(tasks)} images remaining"
        )

    # Bundle the output directory into the task so subprocesses can access it
    tasks = [
        (var_type, stem, fname, out_dir)
        for var_type, stem, fname in tasks
    ]

    print(f"{len(tasks)} images to generate")

    if not tasks:
        print("No tasks to process.")
        sys.exit(0)

    # ---------------- Start batch rendering ----------------
    t_start = time.time()

    done = 0
    failed = []
    finished = set()

    def run_sequential(task_list, offset=0):
        """Run sequentially and return (number completed, list of failures)."""
        ok_n = 0
        bad = []
        for task in task_list:
            try:
                out_name, dt = render_one(task)
                ok_n += 1
                print(
                    f"[{offset + ok_n}/{len(tasks)}] {out_name}  ({dt:.1f} s)",
                    flush=True
                )
            except Exception as e:
                bad.append((task, repr(e)))
                print(f"[FAIL] {task}: {e}", flush=True)
        return ok_n, bad

    # Probe whether multiprocessing is available using an empty task (restricted environments may forbid pipe creation)
    pool_ok = False

    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as probe:
                probe.submit(int, 1).result(timeout=120)
            pool_ok = True
        except Exception as e:
            print(
                f"[WARN] multiprocessing unavailable ({type(e).__name__}: {e}),"
                f"switched to single-process sequential execution",
                flush=True
            )

    if pool_ok:

        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:

                futures = {
                    pool.submit(render_one, task): task
                    for task in tasks
                }

                for fut in as_completed(futures):

                    task = futures[fut]

                    try:
                        out_name, dt = fut.result()
                        done += 1
                        finished.add(task)
                        print(
                            f"[{done}/{len(tasks)}] {out_name}  ({dt:.1f} s)",
                            flush=True
                        )
                    except Exception as e:
                        failed.append((task, repr(e)))
                        finished.add(task)
                        print(f"[FAIL] {task}: {e}", flush=True)

        except Exception as e:
            # The process pool crashed midway: run the remaining tasks sequentially
            print(
                f"[WARN] process pool error ({type(e).__name__}: {e}),"
                f"remaining tasks switched to sequential execution",
                flush=True
            )
            leftover = [t for t in tasks if t not in finished]
            if leftover:
                ok_n, bad = run_sequential(leftover, done)
                done += ok_n
                failed += bad

    else:

        ok_n, bad = run_sequential(tasks)
        done += ok_n
        failed += bad

    # ---------------- Summary ----------------
    total = time.time() - t_start

    print("\n" + "=" * 78)
    print(
        "Completed %d / %d images in %.1f minutes"
        % (done, len(tasks), total / 60.0)
    )
    print("Output directory: %s" % out_dir)

    if failed:
        print("\n%d failed:" % len(failed))
        for task, err in failed:
            print("  %s -> %s" % (task, err))

    print("=" * 78)
