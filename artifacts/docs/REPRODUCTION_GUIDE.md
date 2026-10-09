# Reproduction Guide

This guide describes how to regenerate the processed figures and table from the current public result files. It does not claim that the full training pipeline is self-contained: the MSCG data and model checkpoints are external resources.

## 1. Environment

For figure and table generation, Python 3.10 or newer is sufficient with:

```bash
python -m pip install numpy matplotlib seaborn
```

The original training and evaluation environment used Ubuntu 22.04, Python 3.12.9, CUDA 12.6, PyTorch 2.6.0+cu126, and torchvision 0.21.0+cu126. Training additionally requires the external dataset, fold manifests, SAM checkpoint, and the method-specific dependencies.

## 2. Generate the three-way comparison figures and table

Run from the repository root:

```bash
python artifacts/code/generate_three_way_figures.py
```

Inputs:

```text
artifacts/newdata/ws1_kfold/three_way_comparison.json
artifacts/newdata/ws1_kfold/fold{0..4}/{sam,deeplabv3,xgb}/summary.json
artifacts/newdata/ws1_kfold/fold{0..4}/{sam,deeplabv3,xgb}/per_cell.csv
```

Outputs:

```text
figures_new/figure_overall_comparison_1.png
figures_new/figure_overall_comparison_1.pdf
figures_new/figure_scale_performance.png
figures_new/figure_scale_performance.pdf
figures_new/figure_robustness_comparison.png
figures_new/figure_robustness_comparison.pdf
figures_new/effective_robustness.json
figures_new/robustness_fold_metrics.csv
tables/table_three_way_results.tex
tables/table_robustness_k_levels.tex
```

The overall mIoU means and standard deviations are taken from `three_way_comparison.json`. Scale plots use the `per_scale_miou` values in each fold summary and average them across the five folds. Use `python artifacts/code/generate_three_way_figures.py --robustness-only` to regenerate only the robustness figure, Table 8, and their statistics.

## 3. Generate the Full FT/LoRA figures

Run:

```bash
python artifacts/code/generate_fullft_vs_lora_figure.py
```

Input:

```text
artifacts/newdata/fullft_lora_comparison.json
```

Outputs:

```text
figures_new/figure_fullft_vs_lora_comparison.png
figures_new/figure_fullft_vs_lora_comparison.pdf
figures_new/figure_pareto_frontier.png
figures_new/figure_pareto_frontier.pdf
```

The figures use the values in the aggregate JSON. They do not require the raw images or checkpoints.

## 4. Generate three-method, five-fold robustness outputs

The robustness analysis reads all 15 `per_cell.csv` files: SAM+LoRA, DeepLabV3+, and VGG16+XGBoost in folds 0--4. Each file contains the complete 120-cell grid. At each severity within a fold, the 20 condition--stratum cell mIoUs are weighted by evaluated-pixel `count`:

```bash
python analysis/compute_effective_robustness.py \
  --stats-dir artifacts/newdata/ws1_kfold \
  --output-dir figures_new \
  --table-output tables/table_robustness_k_levels.tex
```

This writes `effective_robustness.json`, `robustness_fold_metrics.csv`, and `figure_robustness_comparison.png/pdf` under `figures_new/`, plus the Table 8 LaTeX file. The JSON preserves the input hashes, all five fold trajectories, fold-level endpoint statistics, and their means and population standard deviations.

Absolute endpoint drop, k5/k0 retention, and ER-AUS are calculated within each fold first. Their unweighted means and population SDs (`ddof=0`) are then computed across folds. Figure 6 shows the five-fold mean trajectories; the population SDs are recorded in the JSON and Table 8. Ratios are not calculated from cross-fold mean endpoints.

This update extends method and fold coverage while preserving the complete-grid cell-weighted definition. No-transform cells are retained, including the clean condition at k1--k5. These endpoint results are not corrupted-only pooled-confusion mIoUs. The SDs describe variation across shared folds, not confidence intervals or independent training repetitions.

## 5. Full training/evaluation reproduction

Before using the shell runners, set local paths through environment variables:

```bash
export PROJECT_ROOT=/path/to/MSCG_Code
export DATA_ROOT=/path/to/MSCG_dataset
export SAM_CKPT=/path/to/sam_vit_h_4b8939.pth
export CACHE_DIR=/path/to/experiment_cache
export OUTPUT_DIR=/path/to/output_directory
```

Then run, for example:

```bash
bash run_experiment.sh --phase eval
bash run_fullft_lora.sh --phase eval
```

`run_experiment.sh` evaluates the three-way five-fold experiment. `run_fullft_lora.sh` handles the Full FT/LoRA comparison. The scripts require the external dataset and checkpoints and may take substantial GPU time.

## 6. Aggregation and release limitations

- Folds 0–3 contain 7,032 test images and fold 4 contains 7,008, for 35,136 images total.
- Overall comparison statistics are read from `three_way_comparison.json`.
- Full FT/LoRA comparison statistics are read from `fullft_lora_comparison.json`.
- Robustness endpoint statistics use all three methods and folds 0--4 from `per_cell.csv`, with ratios formed within folds before averaging.
- Full FT resource metadata is available only for the released metadata files; LoRA r8 timing metadata is missing and must not be interpreted as zero-hour training.
- The point-versus-box input JSON required by `generate_figure5_box_vs_point.py` is not included in this trimmed release.
