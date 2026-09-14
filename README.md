# FFSR-PointNet

Official implementation of:

> **High-Resolution Hydrodynamic Downscaling in Mountainous Catchments via a Transferable Point-Cloud Deep Learning Framework**
>
> Jiahui Wang, Huaiqi Yang, Rundong Liu, Xinhua Zhang, Wenrui Huang, Hong Xiao*
>
> State Key Laboratory of Hydraulics and Mountain River Engineering, Sichuan University

## Abstract

Flash floods in mountainous regions pose severe threats, making high-resolution (HR) flow dynamics simulation critical for precise early warning. However, real-time forecasting in topographically complex basins remains challenging due to the prohibitive computational costs of hydrodynamic models. To address this, we propose FFSR-PointNet, a downscaling framework incorporating a context-subsampled T-Net and rank-constrained factorized decoder, to reconstruct HR water depth and velocity fields from coarse simulations. To mitigate severe zero-inflation inherent to CNN-based methods, it processes only active grid cells within a dilated river network region of interest (ROI). Comparative benchmarking against masked-CNN baselines (FLO-SR and SR-Unet) within a common ROI demonstrates its superiority: FFSR-PointNet effectively bypasses zero-inflation, reducing flow field RMSE by 43.6–68.6% and cutting computational demands by two orders of magnitude (27.64 vs. 2928.82 GFLOPs), thereby accelerating per-epoch training time to 3.66 s. Crucially, it simulates an 18-h flood sequence in 4.21 s, offering a 2848× speedup over fine-grid hydrodynamic analysis. Applied across two spatial scales—Shouxi River watershed (30 m) and Sanjiang Town (1.5 m)—the model achieves exceptional fidelity (R ≥ 0.98). Furthermore, an adaptive transfer learning strategy leveraging truncated-SVD decoder initialization enables rapid adaptation to new geometries (Yuhua and Laoyang Towns) using merely 36 target samples, yielding depth/velocity correlations of 0.92/0.90 and 0.94/0.93. This framework delivers an accurate, highly efficient, and transferable solution for real-time flood hazard mapping in complex mountainous catchments.

**Keywords:** Numerical simulation · Deep learning · Hydrodynamic analysis · Super-resolution reconstruction

---

## 1. Repository structure

```text
FFSR-PointNet/
├── SR-PointNet/                      # Full-parameter FFSR-PointNet (point cloud)
│   └── main/
│       ├── config.py                 # CLI config for training / validation
│       ├── run.py                    # entry called by scripts/train_original.py
│       ├── modelTraining.py          # trains FFSRPointNetV2 (resumable)
│       ├── validate_v2.py            # numerical 8.20 metrics (RMSE, R, recall, precision)
│       ├── Validation_*.py           # GeoTIFF visual validation per model variant
│       ├── speed_test.py             # single-sample inference timing
│       ├── calculate_indices.py      # reusable classification / regression metrics
│       ├── compare_820_models.py     # two-folder comparison -> comparison_820.xlsx
│       ├── model/                    # FFSRP.py (original SR-PointNet), FFSRP_v2.py (FFSR-PointNet)
│       ├── norm/                     # saved normalization statistics (*.npz)
│       └── utils/                    # dataset generation, plotting, raster tools
├── FFSR-PointNet-Light/              # Lightweight model (8192 context pts, rank-128 decoder)
│   └── main/
│       ├── config_light.py           # CLI config (--model_type subsampled only)
│       ├── train_compressed.py       # official training entry (resumable)
│       ├── evaluate_compressed.py    # external 820 validation entry
│       ├── model/                    # FFSRP_subsampled_compressed.py, FFSRP_original.py
│       ├── norm/
│       └── utils/
├── Unet/                             # CNN baselines: SR-Unet, FLO-SR
│   ├── main.py, config_*.py          # original (pre-ROI) SR-Unet workflow
│   ├── validation_shouxi.py, validation_sanjiang.py
│   ├── calculate_indices.py
│   ├── model/                        # SRUnet.py, FLO_SR.py, SRCNN.py, model_factory.py
│   ├── utils/                        # HDF5 generation, visualization
│   └── roi/                          # ROI-aware patch workflow (used in the paper)
│       ├── config_roi_watershed.py, config_roi_village.py
│       ├── roi_data.py               # ROI loading, patch extraction, normalization
│       ├── train_roi_patch.py        # ROI-masked MSE training (early stopping)
│       ├── validation_roi.py         # core ROI validation + GeoTIFF reconstruction
│       ├── metrics_roi*.py           # ROI-aware metrics and comparison tables
│       └── plot_roi_shouxi_peak.py
├── Transfer/                         # Transfer learning (Yuhuo, LaoYangCun)
│   ├── train_transfer_sanjiang.py
│   ├── validate_transfer_sanjiang.py
│   ├── make_transfer_interpolation_maps.py
│   ├── make_laoyangcun_transfer_datasets.py
│   └── transfer_config.example.json
├── Data/dataset/                     # h5 datasets (see §2)
├── Weights/                          # trained checkpoints (see §2)
├── scripts/                          # entry points and evaluation utilities (see §6)
│   └── eval_820/                     # 8.20 evaluation pipeline
├── requirements.txt
└── .gitignore
```

Large result rasters are written at runtime to `Results/` (git-ignored) to keep the repository light.

---

## 2. Installation

```bash
conda create -n ffsr python=3.11 -y
conda activate ffsr
pip install -r requirements.txt
```

All commands in this document are run **from the repository root** unless stated otherwise.

### Data (`Data/dataset/`)

> **⚠️ The full datasets are NOT included in this repository** — they are far too
> large and the underlying hydrodynamic simulation results are subject to
> data-sharing restrictions.
>
> Instead, `Data/dataset/` ships a **dummy dataset**: synthetic data with
> **exactly the same structure** (HDF5 key naming, per-sample shape, dtype and
> attributes) but only **1–2 samples per file**, so that you can run and verify
> the whole pipeline end-to-end. See [`Data/dataset/README.md`](Data/dataset/README.md)
> for the full per-file specification and instructions on plugging in the real data.
>
> The dummy values are **physically meaningless** — do not report metrics computed on them.

| File | Case | Description |
|---|---|---|
| `Point_HR.h5` / `Point_LR_geo.h5` | Shouxi | Training, high- / low-resolution point clouds |
| `Point_HR_Val.h5` / `Point_LR_geo_val.h5` | Shouxi | Validation on the 8.20 flood event |
| `SanJiang_HR.h5` / `SanJiang_LR.h5` | Sanjiang | Training |
| `SanJiang_HR_val.h5` / `SanJiang_LR_val.h5` | Sanjiang | Validation |
| `Yuhuo_HR.h5` / `Yuhuo_LR.h5` | Yuhuo | Transfer learning, training |
| `LaoYangCun_HR.h5` / `LaoYangCun_LR.h5` | LaoYangCun | Transfer learning, training |

Grid datasets for the CNN baselines live in `Data/dataset/Unet/`. Normalization statistics are stored in each model's `main/norm/`.

### Pretrained weights (`Weights/`)

> **⚠️ Pretrained checkpoints are NOT included in this repository** (they exceed the
> hosting limits). The folder only contains a placeholder `readme.txt`.
> Run the training commands in §3–§5 to produce them, or place your own checkpoints
> at the paths below.

| Path | Model |
|---|---|
| `Weights/FFSR_PointNet/best_shouxi_geo.pth` | FFSR-PointNet (Shouxi) |
| `Weights/FFSR_PointNet/best_sanjiang_geo.pth` | FFSR-PointNet (Sanjiang) |
| `Weights/FFSR_PointNet_Light/full_rank128/` | FFSR-PointNet-Light (Shouxi) |
| `Weights/FFSR_PointNet_Light/sanjiang_rank128/` | FFSR-PointNet-Light (Sanjiang) |
| `Weights/Unet/roi_watershed/`, `roi_village/`, `flosr_watershed/`, `flosr_village/` | SR-Unet / FLO-SR |

---

## 3. Shouxi River watershed

### 3.1 Training

**Point-cloud models**

```bash
# FFSR-PointNet (full parameter). Re-running the same command resumes automatically.
python scripts/train_original.py --num_epochs 2000 --batch_size 8

# FFSR-PointNet-Light (subsampled T-Net, rank-128 decoder). Resumes automatically.
python scripts/train_light.py --case shouxi --num_epochs 1000 --batch_size 8
```

**Grid models (ROI-aware patch workflow)**

```bash
python scripts/train_unet.py --model srunet --num_epochs 1000 --use_amp
python scripts/train_unet.py --model flosr  --num_epochs 1000 --use_amp
```

Early stopping is enabled by default (`early_stop_patience=40`). Add `--resume` to continue an interrupted run, and `--disable_early_stop` only if you intentionally want all epochs. Weights and logs go to `Weights/Unet/roi_watershed` and `Weights/Unet/flosr_watershed`.

### 3.2 Validation on the 8.20 event

**Step 1 — Point-cloud models: export GeoTIFFs and metrics**

```bash
# SR-PointNet
python -u scripts/eval_820/validate_point.py --model original \
  --weight  "Weights/FFSR_PointNet/best_shouxi_geo.pth" \
  --mean_std "FFSR-PointNet/main/norm/mean_std_case_Shouxi.npz" \
  --save_root "Results/SR-PointNet" \
  --limit 0

# FFSR-PointNet-Light
python -u scripts/eval_820/validate_point.py --model light \
  --weight  "Weights/FFSR_PointNet_Light/full_rank128/compressed_best.pth" \
  --mean_std "FFSR-PointNet-Light/main/norm/mean_std_case_Shouxi.npz" \
  --save_root "Results/FFSR-PointNet-Light" \
  --limit 0
```

Each output folder then contains `D_pred / D_true / D_ori / D_error / U_pred / U_true / U_ori / U_error` (GeoTIFF), `D_coeff / U_coeff` (density scatter) and `quantitative_metrics_820.xlsx`.

**Step 2 — Grid models**

```bash
cd Unet/roi
python -u validation_roi.py --case watershed --model srunet --result_dir "Results/SR-Unet"
python -u validation_roi.py --case watershed --model flosr  --result_dir "Results/FLO-SR"
cd ../..
```

**Step 3 — Metrics, audit and combined summary**

```bash
# Point-cloud models
python -u scripts/eval_820/analyze_results.py --result_dir "Results/SR-PointNet" --model_label "SR-PointNet"
python -u scripts/eval_820/audit_results.py   --result_dir "Results/SR-PointNet" --model_label "SR-PointNet"
python -u scripts/eval_820/analyze_results.py --result_dir "Results/FFSR-PointNet-Light" --model_label "FFSR-PointNet-Light"
python -u scripts/eval_820/audit_results.py   --result_dir "Results/FFSR-PointNet-Light" --model_label "FFSR-PointNet-Light"

# Grid models (mask to ROI first, then scatter + metrics + audit)
python -u scripts/eval_820/mask_results_to_roi.py --result_dir "Results/SR-Unet"
python -u scripts/eval_820/make_scatter.py        --result_dir "Results/SR-Unet"
python -u scripts/eval_820/analyze_results.py     --result_dir "Results/SR-Unet" --model_label "ROI-aware Patch SR-Unet"
python -u scripts/eval_820/audit_results.py       --result_dir "Results/SR-Unet" --model_label "ROI-aware Patch SR-Unet"

# repeat the four commands above for FLO-SR

# refresh the combined workbook
python -u scripts/eval_820/make_summary.py --results_root "Results"
```

> **Metric ROI convention.** All quantitative metrics are computed inside
> `Data/dataset/geo/shouxi_mask_30_Point.tif` (126460 positive cells), which
> matches the 1075 x 1779 result rasters. The `mask_30.tif` in the same folder is
> a different 895 x 1514 raster and must not be applied to these outputs.

---

## 4. Sanjiang Town

Same ordering as the Shouxi case: point-cloud models first, then grid models.

### 4.1 Training

```bash
# FFSR-PointNet-Light
python scripts/train_light.py --case sanjiang --num_epochs 1000 --batch_size 8

# SR-Unet / FLO-SR
python scripts/train_unet.py --model srunet --case sanjiang --num_epochs 1000 --use_amp
python scripts/train_unet.py --model flosr  --case sanjiang --num_epochs 1000 --use_amp
```

Sanjiang follows the 70/10/20 split used by the grid baselines; the held-out 27 test indices are saved to `Weights/FFSR_PointNet_Light/sanjiang_rank128/sanjiang_test_indices.json`. Grid training reads `Data/dataset/Unet/SanJiang_LR.h5` and `SanJiang_HR.h5`.

### 4.2 Validation (aligned 27-sample test)

```bash
# SR-PointNet
python -u scripts/eval_820/validate_point.py --case sanjiang --model original \
  --weight  "Weights/FFSR_PointNet/best_sanjiang_geo.pth" \
  --mean_std "FFSR-PointNet/main/norm/mean_std_case_Sanjiang.npz" \
  --split_json "Weights/FFSR_PointNet_Light/sanjiang_rank128/sanjiang_test_indices.json" \
  --save_root "Results/Sanjiang/SR-PointNet" --write_metrics

# FFSR-PointNet-Light
python -u scripts/eval_820/validate_point.py --case sanjiang --model light \
  --weight  "Weights/FFSR_PointNet_Light/sanjiang_rank128/compressed_best.pth" \
  --mean_std "FFSR-PointNet-Light/main/norm/mean_std_case_Sanjiang.npz" \
  --split_json "Weights/FFSR_PointNet_Light/sanjiang_rank128/sanjiang_test_indices.json" \
  --save_root "Results/Sanjiang/FFSR-PointNet-Light" --write_metrics

# SR-Unet / FLO-SR on the same 27 timestamps (timestamp filenames match SR-PointNet)
python -u scripts/validate_unet.py --model srunet --case sanjiang \
  --index_json "Weights/FFSR_PointNet_Light/sanjiang_rank128/sanjiang_test_indices.json" \
  --name_source_h5 "Data/dataset/SanJiang_HR.h5" \
  --result_dir "Results/Sanjiang/SR-Unet-27"

python -u scripts/validate_unet.py --model flosr --case sanjiang \
  --index_json "Weights/FFSR_PointNet_Light/sanjiang_rank128/sanjiang_test_indices.json" \
  --name_source_h5 "Data/dataset/SanJiang_HR.h5" \
  --result_dir "Results/Sanjiang/FLO-SR-27"
```

Compute the matching ROI metrics (peak file `2019-08-19_23-10(1).tif`):

```bash
python -u scripts/eval_820/analyze_results.py \
  --result_dir "Results/Sanjiang/SR-Unet-27" \
  --model_label "SR-Unet (Sanjiang aligned 27)" \
  --roi_path "Data/dataset/geo/Shancha_shuixi_1_5_mask_polyfilled.tif" \
  --peak_name "2019-08-19_23-10(1).tif" \
  --output_name "quantitative_metrics_sanjiang.xlsx"
```

> The rebuilt grid h5 files keep the same sample order as the point-cloud h5
> files, so the same numeric index refers to the same timestamp. Legacy folders
> containing `sample_orig*` results come from the older event-blocked dataset and
> must not be mixed into the aligned four-model comparison.

---

## 5. Transfer learning

Two target domains are provided: **Yuhuo Town** (default) and **LaoYangCun**.

```bash
cd Transfer

# Train. Defaults: --case yuhua --use-samples 36 --num-epochs 200 --rank 128
# --context-points 8192 --batch-size 8. Encoder/transform are frozen; only the
# decoder is fine-tuned. Target samples are drawn randomly with --seed.
python train_transfer_sanjiang.py
python train_transfer_sanjiang.py --case laoyangcun

# Validate (writes to Results/Transfer)
python validate_transfer_sanjiang.py
python validate_transfer_sanjiang.py --case laoyangcun

cd ..
```

`--use-samples 0` (or negative) uses all target samples. `mean_std_<case>_<samples>_<seed>.npz` is written alongside the subset so the same command reproduces exactly. Command-line arguments override `--config transfer_config.example.json`.

Outputs per case: `D_pred / D_true / D_ori / D_Int / D_error` and `U_*` (GeoTIFF), `D_coeff / U_coeff` (density scatter), `quantitative_metrics_transfer.xlsx`, `model_profile.json`.

To regenerate the 20 m coarse inputs and the bilinear `D_Int / U_Int` maps:

```bash
python Transfer/make_transfer_interpolation_maps.py \
  --result_root "Results/Transfer" \
  --coarse_res 20 --downsample nearest --overwrite
```

---

## 6. Scripts and visualization

**Entry points (`scripts/`)**

| Script | Purpose |
|---|---|
| `train_original.py`, `validate_original.py` | FFSR-PointNet full model (Shouxi) |
| `train_light.py`, `validate_light.py` | FFSR-PointNet-Light (`--case shouxi\|sanjiang`) |
| `train_unet.py`, `validate_unet.py` | SR-Unet / FLO-SR (`--model srunet\|flosr`, `--case watershed\|sanjiang`) |
| `metrics_compare.py` | Three-model quantitative comparison |
| `build_sanjiang_unified_dataset.py` | Rebuild the Sanjiang unified grid h5 datasets |
| `train_all_1000.py`, `runner.py` | Batch training helpers |

**Figures and visualization**

| Script | Purpose |
|---|---|
| `scripts/make_comparison_videos.py` | **One-click comparison videos** (HR / Pred / Int / Error, 2x2) for both water depth and velocity. Edit the config block at the top, then run. |
| `scripts/draw_publication_figs.py` | Publication figures |
| `scripts/ScatterDraw.py`, `scripts/IntervalDraw.py` | Density scatter and interval-distribution plots |
| `scripts/Shouxi_export.py`, `SanJiang_export.py`, `Yuhuo_export.py`, `Laoyangcun_export.py` | Per-case result export |
| `scripts/eval_820/make_scatter.py` | Density scatter for evaluated models |
| `scripts/eval_820/flood_area_interval_curves.py` | Flood-area interval curves |
| `scripts/eval_820/profile_models.py` | Parameter / GFLOPs profile (`model_profile.csv`) |
| `Data/dataset/Unet/vis_depth_sample.py` | Side-by-side LR vs HR depth sample viewer |

**In-model utilities**

`FFSR-PointNet/main/utils/` and `FFSR-PointNet-Light/main/utils/` provide dataset generation (`datasetGen.py`), raster tooling (`changeCellSize*.py`, `ROIExtraction.py`, `TIFMerge.py`, `R.py`), h5 generation (`h5Gen.py`), metrics (`Calculate_MSE.py`, `calculateMAE.py`, `ResultAnalysis/IndicesCalculate.py`), renaming (`changeNameBatch.py`) and video generation (`videoGen.py`).

---
