#!/usr/bin/env python
"""Batch ScatterDraw/IntervalDraw for retained Shouxi and Transfer samples.

The plotting code intentionally mirrors scripts\\
ScatterDraw.py and IntervalDraw.py, including their fixed axis ranges and
field-specific color schemes. It only generalizes paths and labels.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import pearsonr


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
GLOBAL_FONT_SIZE = 10

RESULTS_ROOT = Path(rf"{ROOT}\Results")


def read_arrays(pred_path, true_path):
    def read_one(path):
        with rasterio.open(str(path)) as src:
            arr = src.read(1)
            nodata = src.nodata
        if nodata is not None:
            arr = arr.copy()
            arr[arr == nodata] = np.nan
        return arr.astype(np.float64)

    return read_one(pred_path), read_one(true_path)


def wet_mask(pred, true):
    return (
        ~np.isnan(pred)
        & ~np.isnan(true)
        & (pred > 0)
        & (true > 0)
    )


def make_scatter_figure(pred_path, true_path, field, out_path, source_key="AI"):
    """Reproduce the field-specific layout of scripts/ScatterDraw.py."""
    pred, true = read_arrays(pred_path, true_path)
    mask = wet_mask(pred, true)
    x = pred[mask].ravel()
    y = true[mask].ravel()
    if x.size < 3:
        raise RuntimeError(f"Too few wet points for scatter: {pred_path}")
    source_label = "Interpolation" if source_key == "Int" else "AI"

    if field == "D":
        upper = 15.0
        x = np.clip(x, 0, upper)
        y = np.clip(y, 0, upper)
        bounds = np.linspace(-4, 0, 9)
        cmap = plt.get_cmap("jet", len(bounds) - 1)
        ticks = [0, 5, 10, 15]
        xlabel = f"Depth - {source_label} (m)"
        ylabel = "Depth - Ground Truth (m)"
    elif field == "U":
        upper = 10.0
        x = np.clip(x, 0, upper)
        y = np.clip(y, 0, upper)
        bounds = np.linspace(-4, 0, 11)
        cmap = plt.get_cmap("RdBu_r", len(bounds) - 1)
        ticks = [0, 2, 4, 6, 8, 10]
        xlabel = f"Vel. Mag. - {source_label} (m/s)"
        ylabel = "Vel. Mag. - Ground Truth (m/s)"
    else:
        raise ValueError(f"Unknown field {field}")
    norm = BoundaryNorm(bounds, cmap.N)

    bins = 70
    counts, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[[0, upper], [0, upper]]
    )
    x_idx = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
    y_idx = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)
    density = counts[x_idx, y_idx].astype(float)
    density[density < 1] = 1.0
    density_log = np.log10(density / density.max())
    density_log = np.clip(density_log, -4, 0)
    rho = float(pearsonr(x, y)[0])

    fig, ax = plt.subplots(figsize=(3, 3), dpi=600)
    ax.set_box_aspect(1)
    sc = ax.scatter(
        x,
        y,
        c=density_log,
        s=4,
        cmap=cmap,
        norm=norm,
        edgecolors="white",
        linewidths=0.25,
    )
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(xlabel, fontsize=GLOBAL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=GLOBAL_FONT_SIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", width=0.8, length=3, labelsize=GLOBAL_FONT_SIZE)

    cax = inset_axes(
        ax,
        width=0.15,
        height=0.8,
        bbox_to_anchor=(0.15, 0.9),
        bbox_transform=ax.transAxes,
        loc="upper left",
        borderpad=0,
    )
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(
        sm,
        cax=cax,
        ticks=[0, -1, -2, -3, -4],
        boundaries=bounds,
        spacing="proportional",
    )
    cb.outline.set_linewidth(1.0)
    cb.ax.tick_params(labelsize=8, width=0.8, length=3)
    cax.set_title(
        "Density (log$_{10}$)",
        fontsize=8,
        pad=8,
        fontfamily="Times New Roman",
    )
    ax.text(
        0.62,
        0.16,
        f"\u03c1={rho:.4f}",
        transform=ax.transAxes,
        fontsize=GLOBAL_FONT_SIZE,
    )
    plt.tight_layout(pad=0.6)
    plt.savefig(str(out_path), format="tif", dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_interval_figure(pred_path, true_path, field, out_path, source_key="AI"):
    """Reproduce scripts/IntervalDraw.py matrix style."""
    pred, true = read_arrays(pred_path, true_path)
    mask = wet_mask(pred, true)
    x = pred[mask].ravel()
    y = true[mask].ravel()
    if x.size < 1:
        raise RuntimeError(f"No wet points for interval plot: {pred_path}")
    source_label = "Interpolation" if source_key == "Int" else "AI"

    bin_edges = [0, 0.5, 1.0, 2.0, 1000]
    n_bins = len(bin_edges) - 1
    counts, _, _ = np.histogram2d(x, y, bins=[bin_edges, bin_edges])
    percentages = counts / x.size * 100.0

    color_bounds = np.array([0, 1, 2, 5, 10, 20, 30, 50])
    n_colors = len(color_bounds) - 1
    if field == "D":
        cmap = plt.get_cmap("jet", n_colors)
        field_name = "Depth"
        unit = "(m)"
    elif field == "U":
        cmap = plt.get_cmap("RdBu_r", n_colors)
        field_name = "Vel. Mag."
        unit = "(m/s)"
    else:
        raise ValueError(f"Unknown field {field}")
    norm = BoundaryNorm(color_bounds, cmap.N)
    xlabel = f"{field_name} - {source_label} {unit}"
    ylabel = f"{field_name} - Ground Truth {unit}"

    fig, ax = plt.subplots(figsize=(3, 3), dpi=600)
    ax.set_box_aspect(1)
    plot_edges = np.arange(n_bins + 1)
    ax.pcolormesh(
        plot_edges,
        plot_edges,
        percentages.T,
        cmap=cmap,
        norm=norm,
        edgecolors="white",
        linewidth=1,
    )
    for i in range(n_bins):
        for j in range(n_bins):
            val = percentages[i, j]
            t_color = "white" if (val > 30 or val < 5) else "black"
            ax.text(
                i + 0.5,
                j + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="center",
                color=t_color,
                fontsize=8,
            )
    ax.set_xticks(plot_edges)
    ax.set_yticks(plot_edges)
    ax.set_xticklabels(["0", "0.5", "1", "2", ""])
    ax.set_yticklabels(["0", "0.5", "1", "2", ""])
    ax.set_xlabel(xlabel, fontsize=GLOBAL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=GLOBAL_FONT_SIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", width=0.8, length=3, labelsize=GLOBAL_FONT_SIZE)
    plt.tight_layout(pad=0.6)
    plt.savefig(str(out_path), format="tif", dpi=600, bbox_inches="tight")
    plt.close(fig)


def add_jobs(jobs, case_root, out_dir, sample_name, source_key="AI"):
    for field in ("D", "U"):
        pred = case_root / f"{field}_pred" / sample_name
        true = case_root / f"{field}_true" / sample_name
        if pred.exists() and true.exists():
            jobs.append((field, pred, true, out_dir, sample_name, source_key))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT / "PublicationFigures")
    parser.add_argument("--dry-run", action="store_true", help="List jobs without drawing")
    args = parser.parse_args()
    jobs = []

    for model in (
        "FFSR-PointNet-Light",
        "SR-PointNet",
        "FLO-SR",
        "SR-Unet",
    ):
        case_root = RESULTS_ROOT / model
        sample_name = "2019-08-20_04_00.tif"
        add_jobs(
            jobs,
            case_root,
            args.output_root / "Shouxi_820" / model,
            sample_name,
        )

    interp_root = RESULTS_ROOT / "Linear-Interpolation-Shouxi-820"
    add_jobs(
        jobs,
        interp_root,
        args.output_root / "Shouxi_820" / "Linear-Interpolation",
        "2019-08-20_04_00.tif",
        source_key="Int",
    )

    sanjiang_root = RESULTS_ROOT / "Sanjiang"
    for model in (
        "FFSR-PointNet-Light",
        "FLO-SR",
        "Linear-Interpolation",
        "SR-PointNet",
        "SR-Unet",
    ):
        case_root = sanjiang_root / model
        if not (case_root / "D_pred").is_dir():
            print(f"Skipping missing Sanjiang folder: {case_root}")
            continue
        source_key = "Int" if model == "Linear-Interpolation" else "AI"
        for pred_file in sorted((case_root / "D_pred").glob("*.tif")):
            add_jobs(
                jobs,
                case_root,
                args.output_root / "Sanjiang" / model,
                pred_file.name,
                source_key,
            )

    transfer_root = RESULTS_ROOT / "Transfer"
    for case_dir in sorted(transfer_root.iterdir()):
        if not case_dir.is_dir():
            continue
        for pred_file in sorted((case_dir / "D_pred").glob("*.tif")):
            add_jobs(
                jobs,
                case_dir,
                args.output_root / "Transfer" / case_dir.name,
                pred_file.name,
            )

    print(f"Total pred/true pairs to draw: {len(jobs)}")
    if args.dry_run:
        for field, pred_path, _, out_dir, sample_name, source_key in jobs:
            print(f"{out_dir.relative_to(args.output_root)} | {field} | {source_key} | {sample_name}")
        return
    for field, pred_path, true_path, out_dir, sample_name, source_key in jobs:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(sample_name).stem
        scatter_out = out_dir / f"scatter_{field}_{source_key}_{stem}.tif"
        interval_out = out_dir / f"interval_{field}_{source_key}_{stem}.tif"
        make_scatter_figure(pred_path, true_path, field, scatter_out, source_key)
        make_interval_figure(pred_path, true_path, field, interval_out, source_key)
        print(f"Saved: {scatter_out}")
        print(f"Saved: {interval_out}")


if __name__ == "__main__":
    main()
