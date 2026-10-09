# MSCG IVC Reproducibility Package

This repository contains the training and analysis code, together with processed evaluation outputs, for the manuscript:

**“MSCG: A Scale-Stratified Benchmark for Robust Segmentation under Controlled Corruptions.”**

It is the trimmed public artifact package for the paper data link. The repository documents the reported experiments and regenerates the main result figures and comparison tables.

## Scope of this release

Included:

- training and inference code for SAM+LoRA, DeepLabV3+, and the classical VGG16+XGBoost pipeline;
- analysis scripts for fold-level evaluation, robustness, and inference benchmarking;
- processed metrics in `artifacts/newdata/` (JSON, CSV, and selected heatmap outputs);
- figure and table generation scripts in `artifacts/code/`;
- the original experiment runner scripts `run_experiment.sh` and `run_fullft_lora.sh`.

Not included:

- the original MSCG images, masks, and fold manifests;
- the SAM ViT-H checkpoint or trained model checkpoints;
- experiment caches;
- the manuscript source and compiled manuscript.

The released JSON/CSV files are sufficient to regenerate the packaged summary figures and tables. Re-running training or evaluation requires access to the external dataset and checkpoints.

## Repository layout

```text
.
├── analysis/                         # Evaluation and analysis scripts
├── artifacts/
│   ├── code/                         # Figure/table generation scripts
│   ├── docs/                         # Historical artifact notes
│   └── newdata/                      # Current processed results
├── methods/                          # Model, dataset, and training code
├── run_experiment.sh                 # Three-way five-fold experiment runner
├── run_fullft_lora.sh                # Full fine-tuning/LoRA experiment runner
├── EXPERIMENT_FULL_RECORD.txt        # Detailed experiment record
└── README.md
```

`artifacts/docs/` contains notes inherited from the manuscript workspace. This root README and the files under `artifacts/newdata/` define the current trimmed release.

## Reproducing the processed figures and table

From the repository root, install the plotting dependencies:

```bash
python -m pip install numpy matplotlib seaborn
```

Then run:

```bash
python artifacts/code/generate_three_way_figures.py
python artifacts/code/generate_fullft_vs_lora_figure.py
```

The scripts read the current processed results and create:

```text
figures_new/
├── figure_overall_comparison_1.png/pdf
├── figure_scale_performance.png/pdf
├── figure_robustness_comparison.png/pdf
├── figure_fullft_vs_lora_comparison.png/pdf
└── figure_pareto_frontier.png/pdf

tables/
└── table_three_way_results.tex
```

`generate_figure5_box_vs_point.py` is retained as a legacy script, but its required input file is not part of this trimmed release and is therefore not included in the quick-start commands.

## Current processed results

### Three-way comparison

Results below are the five-fold values in `artifacts/newdata/ws1_kfold/three_way_comparison.json`.

| Method | mIoU | Dice | Accuracy | Brittleness |
|---|---:|---:|---:|---:|
| **SAM+LoRA (r8)** | **86.28% ± 0.69%** | 92.57% ± 0.40% | 93.55% ± 0.40% | 0.48% |
| DeepLabV3+ | 76.54% ± 1.06% | 86.47% ± 0.69% | 88.48% ± 0.63% | 5.40% |
| VGG16+XGBoost | 60.17% ± 0.48% | 74.66% ± 0.37% | 76.99% ± 0.45% | 4.39% |

The test-set sizes are 7,032 images for folds 0–3 and 7,008 images for fold 4, for 35,136 test images in total.

All three pairwise mIoU comparisons are significant after Bonferroni correction. The corrected p-values and effect sizes are recorded in `three_way_comparison.json`.

### Full fine-tuning and LoRA

The current aggregate file is `artifacts/newdata/fullft_lora_comparison.json`.

| Configuration | Folds | Test mIoU | Trainable parameters | Checkpoint size |
|---|---:|---:|---:|---:|
| Full FT | 5 | 86.57% ± 0.90% | 641M* | 7.16 GiB** |
| LoRA r16 | 5 | 86.41% ± 0.61% | 8M (1.25%) | 90.7 MB |
| LoRA r8 | 5 | 86.28% ± 0.69% | 6M (0.94%) | 53.1 MB |
| LoRA r4 | 5 | 86.16% ± 1.32% | 5M (0.79%) | 57 MB |

\* The Full FT parameter count is available in the fold-level metadata for the released Full FT outputs.

\** The Full FT checkpoint size is reported by the available fold-level metadata; the aggregate JSON does not contain a complete resource summary for every configuration. The `0` training-time entries for LoRA r8 indicate missing timing metadata, not zero-hour training.

### Inference benchmark

The benchmark is stored in `artifacts/newdata/inference_benchmarks_rtx4090.json` and uses 1024×1024 inputs, batch size 1, 10 warm-up runs, and 100 measured runs on an RTX 4090.

| Model | Mean latency | FPS |
|---|---:|---:|
| SAM+LoRA | 241.8 ms | 4.13 |
| DeepLabV3+ | 13.4 ms | 74.84 |

## Running the original training/evaluation pipelines

The shell scripts were used in the original Linux/GPU environment. Before running them, replace the dataset, checkpoint, cache, and output paths with local paths:

```text
DATA_ROOT   = external MSCG dataset root
SAM_CKPT    = SAM ViT-H checkpoint
CACHE_DIR   = local experiment cache
OUTPUT_DIR  = local result directory
```

The scripts require a Linux shell, CUDA-enabled PyTorch, the external MSCG dataset, and the SAM checkpoint. The processed outputs in `artifacts/newdata/` can be inspected without those external resources.

## Data and privacy note

The raw MSCG images, masks, fold manifests, and model weights are not redistributed in this repository. The CSV files contain evaluation summaries and per-image metrics, not the source images. Check the applicable dataset licenses before redistributing any omitted materials.

## Citation

Please cite the associated manuscript when using this code or the processed results. Use its final bibliographic details when available:

```text
MSCG: A Scale-Stratified Benchmark for Robust Segmentation under Controlled Corruptions.
```

For the exact experimental settings, aggregation conventions, and result provenance, see `EXPERIMENT_FULL_RECORD.txt`.
