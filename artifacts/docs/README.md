# MSCG IVC Artifact Notes

This directory contains supplementary notes for the trimmed public artifact release. The authoritative instructions are in the repository-root `README.md`.

## Current release

The public package contains:

- model, training, inference, and analysis code under `methods/` and `analysis/`;
- processed evaluation outputs under `artifacts/newdata/`;
- figure/table generation scripts under `artifacts/code/`.

The raw MSCG images, masks, fold manifests, SAM checkpoint, trained checkpoints, experiment caches, and manuscript source are not included.

## Directory layout

```text
artifacts/
├── code/                 # Figure and table generation scripts
├── docs/                 # This note and the reproduction guide
└── newdata/              # Current processed JSON, CSV, and heatmap outputs
```

## Quick start

Run the following commands from the repository root:

```bash
python -m pip install numpy matplotlib seaborn
python artifacts/code/generate_three_way_figures.py
python artifacts/code/generate_fullft_vs_lora_figure.py
```

The generated figures are written to `figures_new/`, and the generated comparison table is written to `tables/table_three_way_results.tex`.

The point-versus-box script `generate_figure5_box_vs_point.py` is retained for reference, but its input JSON is not part of this release and is not a runnable quick-start component.

## Current processed result files

```text
artifacts/newdata/ws1_kfold/three_way_comparison.json
artifacts/newdata/ws1_kfold/fold{0..4}/{sam,deeplabv3,xgb}/summary.json
artifacts/newdata/ws1_kfold/fold{0..4}/{sam,deeplabv3,xgb}/per_cell.csv
artifacts/newdata/ws1_kfold/fold{0..4}/{sam,deeplabv3,xgb}/per_image.csv
artifacts/newdata/fullft_lora_comparison.json
artifacts/newdata/inference_benchmarks_rtx4090.json
```

The three-way five-fold mIoU results are 86.28% ± 0.69% for SAM+LoRA (r8), 76.54% ± 1.06% for DeepLabV3+, and 60.17% ± 0.48% for VGG16+XGBoost. The test sets contain 7,032 images in folds 0–3 and 7,008 images in fold 4.

The current Full FT/LoRA aggregate reports 86.57% ± 0.90% for Full FT, 86.41% ± 0.61% for LoRA r16, 86.28% ± 0.69% for LoRA r8, and 86.16% ± 1.32% for LoRA r4.

## Re-running training

The original shell runners require a Linux shell, CUDA-enabled PyTorch, an external MSCG dataset, and an external SAM ViT-H checkpoint. Set the local paths before running them, for example:

```bash
PROJECT_ROOT=/path/to/MSCG_Code \
DATA_ROOT=/path/to/MSCG_dataset \
SAM_CKPT=/path/to/sam_vit_h_4b8939.pth \
bash run_experiment.sh --phase eval
```

See [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md) for the aggregation conventions and the distinction between processed-result regeneration and full training reproduction.
