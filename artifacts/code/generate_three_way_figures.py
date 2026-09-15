#!/usr/bin/env python3
"""
Generate three-way comparison figures and LaTeX table.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict

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

N_PAIRWISE_COMPARISONS = 3


def format_p_value(p):
    """Format adjusted p-values with enough precision for the figure labels."""
    if p < 0.001:
        return f"{p:.2e}"
    else:
        return f"{p:.3f}"


def format_p_value_latex(p):
    """Format adjusted p-values for use inside LaTeX math mode."""
    if p < 0.001:
        coefficient, exponent = f"{p:.2e}".split("e")
        return rf"{coefficient} \times 10^{{{int(exponent)}}}"
    return f"{p:.3f}"


def bonferroni_adjust(p, n_comparisons=N_PAIRWISE_COMPARISONS):
    """Return the Bonferroni-adjusted p-value, capped at one."""
    return min(float(p) * n_comparisons, 1.0)


def significance_marker(p):
    """Mark significance using the adjusted p-value."""
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


def load_data(stats_dir: Path) -> Dict:
    """Load three-way comparison results."""
    with open(stats_dir / 'three_way_comparison.json', 'r') as f:
        data = json.load(f)
    
    # Add per-scale data from fold summaries (point plot style)
    stats_dir_path = Path(stats_dir)
    per_scale_data = {}
    for method in data['methods']:
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
    """Figure 1: Overall three-way method comparison with error bars"""

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = data['methods']
    method_names = [LABELS[m] for m in methods]

    # mIoU data
    means = [data['comparison'][m]['miou']['mean'] * 100 for m in methods]
    stds = [data['comparison'][m]['miou']['std'] * 100 for m in methods]
    colors = [COLORS[m] for m in methods]

    # Create bars
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=10, color=colors,
                   alpha=0.85, edgecolor='black', linewidth=1.5, width=0.6)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 1.5,
                f'{mean:.2f}%\n±{std:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Add significance brackets from pairwise tests
    y_max = max(means) + max(stds) + 10

    # Get p-values from pairwise tests
    pairwise = data.get('pairwise_tests', {})
    
    # The JSON stores raw p-values. Adjust them before plotting and use the
    # actual method positions so the annotations cannot be swapped when the
    # data order changes.
    p1 = bonferroni_adjust(pairwise.get('sam_vs_deeplabv3', {}).get('miou', {}).get('t_test', {}).get('p_value', 1))
    p2 = bonferroni_adjust(pairwise.get('sam_vs_xgb', {}).get('miou', {}).get('t_test', {}).get('p_value', 1))
    p3 = bonferroni_adjust(pairwise.get('xgb_vs_deeplabv3', {}).get('miou', {}).get('t_test', {}).get('p_value', 1))

    positions = {method: index for index, method in enumerate(methods)}

    def add_significance_bracket(method_a, method_b, y, p_value):
        x_a, x_b = positions[method_a], positions[method_b]
        ax.plot([x_a, x_b], [y, y], 'k-', linewidth=1.5)
        ax.text((x_a + x_b) / 2, y + 1,
                f'{significance_marker(p_value)}\np_adj={format_p_value(p_value)}',
                ha='center', va='bottom', fontsize=8)

    # SAM vs XGBoost, SAM vs DeepLabV3+, and XGBoost vs DeepLabV3+
    # are placed according to their actual x coordinates.
    add_significance_bracket('sam', 'xgb', y_max, p2)
    add_significance_bracket('sam', 'deeplabv3', y_max + 7, p1)
    add_significance_bracket('xgb', 'deeplabv3', y_max - 7, p3)

    ax.set_ylabel('mIoU (%)', fontweight='bold')
    ax.set_title('MSCG 5-Fold Cross-Validation: Overall Method Comparison (n=35136 images)',
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.set_ylim(0, y_max + 15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

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
    
    for method in data['methods']:
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
    ax.set_title('Per-Scale Performance Comparison (Point Plot)', fontweight='bold', pad=15)
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
    """Figure 3: Severity-endpoint robustness trajectory (line plot) - matches trajectory_severity.png style"""
    
    # Load effective robustness data for endpoints
    eff_robustness_path = Path(__file__).resolve().parents[2] / 'figures' / 'effective_robustness.json'
    eff_data = {}
    if eff_robustness_path.exists():
        with open(eff_robustness_path) as f:
            eff_data = json.load(f)
    
    # Use per_severity_data from fold summaries for trajectory
    per_severity_data = data.get('per_severity_data', {})
    
    # Plot SAM+LoRA trajectory
    sam_data = per_severity_data.get('sam', {})
    severities = sorted([int(k) for k in sam_data.keys()])
    sam_miou = [sam_data[str(k)] for k in severities]  # 0-1 scale
    
    # Plot Classical (XGBoost) trajectory  
    xgb_data = per_severity_data.get('xgb', {})
    xgb_miou = [xgb_data.get(str(k), np.nan) for k in severities]
    
    # Match trajectory_severity.png style exactly
    plt.figure(figsize=(6.5, 3.6))
    plt.plot(severities, sam_miou, label='SAM+LoRA', color='#1f77b4', marker='o')
    plt.plot(severities, xgb_miou, label='Classical', color='#d62728', marker='o')
    plt.xlabel('Severity (k)')
    plt.ylabel('Two-class mIoU')
    plt.title('Performance Trajectory vs Severity')
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'figure_robustness_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure_robustness_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print("Figure 3: Robustness trajectory saved (trajectory_severity style)")


def generate_results_table(data: Dict, output_dir: Path):
    """Generate LaTeX table for three-way comparison"""

    latex = r"""\begin{table*}[ht]
\centering
\caption{MSCG matched five-fold comparison of the three segmentation methods. Metrics are averaged over the 5 shared test folds.}
\label{tab:three_way_results}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.24\textwidth}@{}>{\centering\arraybackslash}p{0.18\textwidth}@{}>{\centering\arraybackslash}p{0.20\textwidth}@{}>{\centering\arraybackslash}p{0.30\textwidth}@{}>{\centering\arraybackslash}p{0.08\textwidth}@{}}
\toprule
\textbf{Method} & \textbf{mIoU (\%)} & \textbf{Dice (\%)} & \textbf{Accuracy (\%)} & \textbf{Rank} \\
\midrule
"""

    methods = data['rankings']
    for rank, method in enumerate(methods, 1):
        miou = data['comparison'][method]['miou']
        dice = data['comparison'][method]['dice']
        acc = data['comparison'][method]['accuracy']

        symbol = r"\textbf{*}" if rank == 1 else ""
        latex += f"{LABELS[method]}{symbol} & {miou['mean']*100:.2f} $\\pm$ {miou['std']*100:.2f} & "
        latex += f"{dice['mean']*100:.2f} $\\pm$ {dice['std']*100:.2f} & "
        latex += f"{acc['mean']*100:.2f} $\\pm$ {acc['std']*100:.2f} & {rank} \\\\\n"

    latex += r"""\midrule
\multicolumn{5}{l}{\textbf{Pairwise Comparisons (paired $t$-tests; Bonferroni-adjusted $p$-values):}} \\
"""

    pairs = [
        ('sam_vs_deeplabv3', 'SAM+LoRA vs DeepLabV3+'),
        ('sam_vs_xgb', 'SAM+LoRA vs VGG16+XGBoost'),
        ('xgb_vs_deeplabv3', 'DeepLabV3+ vs VGG16+XGBoost'),
    ]

    for pair_key, desc in pairs:
        raw_p = data['pairwise_tests'][pair_key]['miou']['t_test']['p_value']
        p_val = bonferroni_adjust(raw_p)
        display_sign = -1 if pair_key == 'xgb_vs_deeplabv3' else 1
        cohen_d = data['pairwise_tests'][pair_key]['miou']['cohen_d'] * display_sign
        mean_diff = data['pairwise_tests'][pair_key]['miou']['mean_diff'] * 100 * display_sign

        sig = significance_marker(p_val)
        p_str = format_p_value_latex(p_val)
        latex += (
            "\\multicolumn{5}{l}{\\quad "
            f"{desc}: $\\Delta$={mean_diff:+.2f}\\%, "
            f"$p_{{\\mathrm{{adj}}}}={p_str}$ {sig}, "
            f"$d={cohen_d:.2f}$"
            "} \\\\\n"
        )

    latex += r"""\bottomrule
\end{tabular}
\vspace{0.2cm}

\footnotesize
* Winner (best overall mIoU). Significance levels for Bonferroni-adjusted $p$-values: *** $p_{\mathrm{adj}}<0.001$, ** $p_{\mathrm{adj}}<0.01$, * $p_{\mathrm{adj}}<0.05$. \\
All pairwise comparisons use paired t-tests with n=5 folds. Cohen's d: small (0.2), medium (0.5), large (0.8).
\end{table*}
"""

    with open(output_dir / 'table_three_way_results.tex', 'w') as f:
        f.write(latex)

    print("Table 1: Three-way results table (LaTeX) saved")


def main():
    # Project root is two levels up from artifacts/code/
    base_dir = Path(__file__).resolve().parents[2]
    stats_dir = base_dir / 'artifacts' / 'newdata' / 'ws1_kfold'
    figures_dir = base_dir / 'figures_new'
    tables_dir = base_dir / 'tables'
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

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
