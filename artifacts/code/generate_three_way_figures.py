#!/usr/bin/env python3
"""
Generate three-way comparison figures and LaTeX table.
"""

import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Set publication-quality style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['figure.dpi'] = 300

# Color scheme
COLORS = {
    'sam': '#2ECC71',        # Green (best)
    'deeplabv3': '#3498DB',  # Blue (middle)
    'xgb': '#E74C3C',        # Red (baseline)
}

LABELS = {
    'sam': 'SAM+LoRA',
    'deeplabv3': 'DeepLabV3+',
    'xgb': 'VGG16+XGBoost',
}

def load_data(stats_dir: Path) -> Dict:
    """Load three-way comparison results."""
    with open(stats_dir / 'three_way_comparison.json', 'r') as f:
        data = json.load(f)
    
    # Add per-scale data from fold summaries (point plot style)
    stats_dir_path = Path(stats_dir)
    per_scale_data = {}
    for method in ['sam', 'deeplabv3', 'xgb']:
        per_scale_data[method] = {}
        for fold in range(data.get('num_folds', 5)):
            summary_file = stats_dir_path / f'fold{fold}' / method / 'summary.json'
            if summary_file.exists():
                with open(summary_file) as f:
                    fold_data = json.load(f)
                    if 'per_scale_miou' in fold_data:
                        for scale, val in fold_data['per_scale_miou'].items():
                            if scale not in per_scale_data[method]:
                                per_scale_data[method][scale] = []
                            per_scale_data[method][scale].append(float(val))
    
    # Average across folds for point plot
    for method in per_scale_data:
        for scale in per_scale_data[method]:
            vals = per_scale_data[method][scale]
            if vals:
                per_scale_data[method][scale] = float(np.mean(vals))
    
    # Add per-severity data for robustness trajectory
    per_severity_data = {}
    for method in data['methods']:
        per_severity_data[method] = {}
        for fold in range(data.get('num_folds', 5)):
            summary_file = stats_dir_path / f'fold{fold}' / method / 'summary.json'
            if summary_file.exists():
                with open(summary_file) as f:
                    fold_data = json.load(f)
                    if 'per_severity_miou' in fold_data:
                        for sev, val in fold_data['per_severity_miou'].items():
                            if sev not in per_severity_data[method]:
                                per_severity_data[method][sev] = []
                            per_severity_data[method][sev].append(float(val))
    
    # Average across folds
    for method in per_severity_data:
        for sev in per_severity_data[method]:
            vals = per_severity_data[method][sev]
            if vals:
                per_severity_data[method][sev] = float(np.mean(vals))
    
    data['per_scale_data'] = per_scale_data
    data['per_severity_data'] = per_severity_data
    return data


def figure1_overall_comparison(data: Dict, output_dir: Path):
    """Figure 1: Overall three-way comparison with fold-level points."""

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Use the same descending mIoU order as the manuscript results table.
    methods = data.get('rankings', data['methods'])
    method_names = [LABELS[m] for m in methods]

    # mIoU data
    means = [data['comparison'][m]['miou']['mean'] * 100 for m in methods]
    stds = [data['comparison'][m]['miou']['std'] * 100 for m in methods]
    colors = [COLORS[m] for m in methods]

    # Create bars
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=10, color=colors,
                   alpha=0.85, edgecolor='black', linewidth=1.5, width=0.6,
                   zorder=2)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 1.5,
                f'{mean:.2f}%\n±{std:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Show the five fold-level values directly without inferential annotations.
    fold_values_by_method = {
        method: np.asarray(data['comparison'][method]['miou']['values']) * 100
        for method in methods
    }
    offsets = np.linspace(-0.10, 0.10, len(next(iter(fold_values_by_method.values()))))

    # Connect the same held-out fold across methods to make the paired design visible.
    for fold_idx, offset in enumerate(offsets):
        ax.plot(
            x + offset,
            [fold_values_by_method[method][fold_idx] for method in methods],
            color='0.45',
            linewidth=0.8,
            alpha=0.40,
            zorder=1,
        )

    for index, method in enumerate(methods):
        ax.scatter(
            np.full(len(offsets), x[index]) + offsets,
            fold_values_by_method[method],
            s=28,
            facecolors='white',
            edgecolors='black',
            linewidths=0.9,
            zorder=4,
        )

    ax.set_ylabel('mIoU (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Patch(facecolor='0.75', edgecolor='black', label='Five-fold mean'),
            Line2D([0], [0], color='black', linewidth=1.5, label='Population SD'),
            Line2D([0], [0], marker='o', color='black', markerfacecolor='white',
                   linestyle='None', label='Fold score'),
        ],
        loc='upper right',
        framealpha=0.95,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_overall_comparison_1.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure_overall_comparison_1.pdf', bbox_inches='tight')
    plt.close()

    print("Figure 1: Overall comparison saved")


def figure2_per_scale_comparison(data: Dict, output_dir: Path):
    """Figure 2: Performance across complexity scales for all three methods (point plot style)"""

    fig, ax = plt.subplots(figsize=(12, 6))

    scales = ['s1', 's2', 's3', 's4', 's5']
    scale_labels = ['s1\n(<1%)', 's2\n(1-3%)', 's3\n(3-10%)', 's4\n(10-30%)', 's5\n(>30%)']

    x = np.arange(len(scales))

    # Use per_scale_data from loaded data (point plot style)
    per_scale_data = data.get('per_scale_data', {})
    
    # For point plot style, we use markers and lines
    markers = {'sam': 'o', 'deeplabv3': 's', 'xgb': '^'}
    
    for method in ['sam', 'deeplabv3', 'xgb']:
        method_data = per_scale_data.get(method, {})
        y_vals = []
        for scale in scales:
            y_val = method_data.get(scale, np.nan)
            y_vals.append(y_val * 100 if not np.isnan(y_val) else np.nan)
        
        color = COLORS[method]
        marker = markers.get(method, 'o')
        ax.plot(x, y_vals, marker=marker, markersize=8, linewidth=2,
                color=color, label=LABELS[method], alpha=0.8, markeredgecolor='black', markeredgewidth=1)

    ax.set_xlabel('Geometric Complexity Scale (% of image area)', fontweight='bold')
    ax.set_ylabel('mIoU (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['s1\n(<1%)', 's2\n(1-3%)', 's3\n(3-10%)', 's4\n(10-30%)', 's5\n(>30%)'])
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_scale_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure_scale_performance.pdf', bbox_inches='tight')
    plt.close()

    print("Figure 2: Per-scale comparison (point plot) saved")


def figure3_robustness_trajectory(data: Dict, output_dir: Path):
    """Manuscript Figure 6 and Table 8 share all three methods and five folds."""
    base_dir = Path(__file__).resolve().parents[2]
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))
    from analysis.compute_effective_robustness import build_summary, write_outputs

    summary = build_summary(base_dir / 'artifacts' / 'newdata' / 'ws1_kfold')
    write_outputs(summary, output_dir, base_dir / 'tables' / 'table_robustness_k_levels.tex')
    print("Robustness figure, table, and fold statistics saved (three methods, five folds)")


def generate_results_table(data: Dict, output_dir: Path):
    """Generate the three-panel LaTeX comparison table used in the manuscript."""

    latex = r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Matched five-fold comparison of the three systems under the MSCG complete-grid protocol. Panel A reports binary metrics computed from pooled pixels, with mIoU and Dice macro-averaged over background and foreground, using all 24 records per source (including nine no-transform records). Panel B reports paired within-fold mIoU differences, and Panel C summarizes the accuracy-evaluation operating points.}
\label{tab:three_way_results}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}
\toprule
\multicolumn{5}{@{}l}{\textbf{Panel A. Overall performance}} \\
\midrule
\textbf{Method} & \textbf{mIoU (\%)} & \textbf{Dice (\%)} & \textbf{Accuracy (\%)} & \textbf{Rank} \\
\midrule
"""

    methods = data['rankings']
    for rank, method in enumerate(methods, 1):
        miou = data['comparison'][method]['miou']
        dice = data['comparison'][method]['dice']
        acc = data['comparison'][method]['accuracy']

        label = LABELS[method]
        miou_text = f"{miou['mean']*100:.2f} $\\pm$ {miou['std']*100:.2f}"
        dice_text = f"{dice['mean']*100:.2f} $\\pm$ {dice['std']*100:.2f}"
        acc_text = f"{acc['mean']*100:.2f} $\\pm$ {acc['std']*100:.2f}"
        rank_text = str(rank)
        if rank == 1:
            label = rf"\textbf{{{label}}}"
            miou_text = rf"\textbf{{{miou_text}}}"
            dice_text = rf"\textbf{{{dice_text}}}"
            acc_text = rf"\textbf{{{acc_text}}}"
            rank_text = rf"\textbf{{{rank_text}}}"
        latex += f"{label} & {miou_text} & {dice_text} & {acc_text} & {rank_text} \\\\\n"

    latex += r"""\bottomrule
\end{tabular*}

\vspace{0.15cm}

\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
\multicolumn{7}{@{}l}{\textbf{Panel B. Fold-level paired mIoU differences (percentage points)}} \\
\midrule
\textbf{Comparison} & \textbf{Fold 0} & \textbf{Fold 1} & \textbf{Fold 2} & \textbf{Fold 3} & \textbf{Fold 4} & \textbf{Mean $\Delta$} \\
\midrule
"""

    pairs = [
        ('sam', 'deeplabv3', 'SAM+LoRA $-$ DeepLabV3+'),
        ('sam', 'xgb', 'SAM+LoRA $-$ VGG16+XGBoost'),
        ('deeplabv3', 'xgb', 'DeepLabV3+ $-$ VGG16+XGBoost'),
    ]

    for first, second, desc in pairs:
        first_values = np.asarray(data['comparison'][first]['miou']['values'])
        second_values = np.asarray(data['comparison'][second]['miou']['values'])
        differences = (first_values - second_values) * 100
        fold_text = " & ".join(f"{value:+.2f}" for value in differences)
        mean_text = f"{np.mean(differences):+.2f}"
        latex += f"{desc} & {fold_text} & \\textbf{{{mean_text}}} \\\\\n"

    latex += r"""\bottomrule
\end{tabular*}

\vspace{0.15cm}

\begin{tabularx}{\textwidth}{@{}lX@{}}
\toprule
\multicolumn{2}{@{}l}{\textbf{Panel C. Accuracy-evaluation operating points}} \\
\midrule
\textbf{System} & \textbf{Operating point} \\
\midrule
SAM+LoRA
& $1024\times1024$; pre-trained SAM ViT-H with rank-8 LoRA and a trainable mask decoder; \texttt{P1} mask-derived two-point prompt; candidate with the highest predicted IoU; bilinear upsampling; and threshold $\tau=0.5$. \\

DeepLabV3+
& $512\times512$; ImageNet-pretrained ResNet-101; no prompt; sigmoid threshold $\tau=0.5$; and nearest-neighbor resizing. \\

VGG16+XGBoost
& $64\times64$ patches with a stride of 32, resized to $224\times224$ for VGG16; fixed ImageNet-pretrained VGG16 plus XGBoost; patch reconstruction; and Dense CRF. \\
\bottomrule
\end{tabularx}

\vspace{0.2cm}

\scriptsize
\textit{Note.} Positive values favor the first method listed in each comparison. Fold-level differences were computed from the unrounded mIoU values and rounded to two decimal places for display. Boldface highlights the best-performing row in Panel A and the mean differences in Panel B.
\end{table}
"""

    with open(output_dir / 'table_three_way_results.tex', 'w') as f:
        f.write(latex)

    print("Table 1: Three-way results table (LaTeX) saved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robustness-only', action='store_true', help='Regenerate only Figure 6, Table 8, and their statistics')
    args = parser.parse_args()
    # Project root is two levels up from artifacts/code/
    base_dir = Path(__file__).resolve().parents[2]
    stats_dir = base_dir / 'artifacts' / 'newdata' / 'ws1_kfold'
    figures_dir = base_dir / 'figures_new'
    tables_dir = base_dir / 'tables'
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if args.robustness_only:
        figure3_robustness_trajectory({}, figures_dir)
        return

    print("=" * 70)
    print("GENERATING THREE-WAY COMPARISON FIGURES")
    print("=" * 70)
    print()

    # Load data
    print("Loading three-way comparison data...")
    data = load_data(stats_dir)
    print(f"Data loaded ({len(data['methods'])} methods)")
    print()

    # Generate figures
    print("Generating figures...")
    figure1_overall_comparison(data, figures_dir)
    figure2_per_scale_comparison(data, figures_dir)
    figure3_robustness_trajectory(data, figures_dir)
    print()

    # Generate table
    print("Generating LaTeX table...")
    generate_results_table(data, tables_dir)
    print()

    print("=" * 70)
    print(f"ALL FIGURES SAVED TO: {figures_dir.absolute()}")
    print("=" * 70)
    print()
    print("Files generated:")
    print("  - figure_overall_comparison_1.png/pdf")
    print("  - figure_scale_performance.png/pdf")
    print("  - figure_robustness_comparison.png/pdf")
    print(f"  - table_three_way_results.tex (in {tables_dir})")
    print()


if __name__ == '__main__':
    main()
