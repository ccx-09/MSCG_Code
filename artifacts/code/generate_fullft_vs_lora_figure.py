#!/usr/bin/env python3
"""
Generate figures comparing full fine-tuning and LoRA configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['figure.dpi'] = 300


def load_comparison_data(json_path: Path):
    """Load full FT vs LoRA comparison data from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract data in the format expected by plotting functions
    comparison = data.get('comparison', {})
    
    full_ft_data = {
        'mean': comparison.get('full_ft', {}).get('test_miou', {}).get('mean', 0) * 100,
        'std': comparison.get('full_ft', {}).get('test_miou', {}).get('std', 0) * 100,
        'trainable_params_m': comparison.get('full_ft', {}).get('trainable_params_m', 641),
        'trainable_pct': comparison.get('full_ft', {}).get('trainable_param_pct', 100),
    }
    
    lora_data = {}
    for rank in ['4', '8', '16']:
        key = f'lora_r{rank}'
        if key in comparison:
            lora_data[f'r{rank}'] = {
                'mean': comparison[key].get('test_miou', {}).get('mean', 0) * 100,
                'std': comparison[key].get('test_miou', {}).get('std', 0) * 100,
                'trainable_params_m': comparison[key].get('trainable_params_m', 0),
                'trainable_pct': comparison[key].get('trainable_param_pct', 0),
            }
    
    return full_ft_data, lora_data


def generate_method_comparison_figure(output_dir: Path, full_ft_data: dict, lora_data: dict):
    """
    Generate a Full FT vs LoRA comparison figure with two subplots:
    (a) mIoU comparison and (b) parameter efficiency.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Prepare data
    methods = ['Full FT', 'SAM+LoRA r=16', 'SAM+LoRA r=8', 'SAM+LoRA r=4']
    means = [
        full_ft_data['mean'],
        lora_data['r16']['mean'],
        lora_data['r8']['mean'],
        lora_data['r4']['mean']
    ]
    stds = [
        full_ft_data['std'],
        lora_data['r16']['std'],
        lora_data['r8']['std'],
        lora_data['r4']['std']
    ]
    trainable_params = [
        full_ft_data['trainable_params_m'],
        lora_data['r16']['trainable_params_m'],
        lora_data['r8']['trainable_params_m'],
        lora_data['r4']['trainable_params_m']
    ]
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']  # Red, Blue, Green, Orange

    # Subplot 1: mIoU Comparison with Error Bars
    x_pos = np.arange(len(methods))
    bars1 = ax1.bar(x_pos, means, yerr=stds, capsize=8,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars1, means, stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                f'{mean:.2f}%\n±{std:.2f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_ylabel('Held-out mIoU (%)', fontweight='bold')
    ax1.set_title('(a) Performance Comparison', fontweight='bold', loc='left', pad=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.set_ylim(0, max(means) + max(stds) + 6)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.axhline(full_ft_data['mean'], color='red', linestyle='--',
                linewidth=1, alpha=0.5, label='Full FT mean')
    ax1.legend(loc='lower left', fontsize=8)

    # Subplot 2: Parameter Efficiency (log scale)
    bars2 = ax2.bar(x_pos, trainable_params, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, params) in enumerate(zip(bars2, trainable_params)):
        height = bar.get_height()
        if params >= 10:
            label = f'{int(params)}M'
        else:
            label = f'{params}M'
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                label, ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2.set_ylabel('Trainable Parameters (M)', fontweight='bold')
    ax2.set_title('(b) Parameter Efficiency', fontweight='bold', loc='left', pad=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    ax2.set_axisbelow(True)

    # Add efficiency annotations slightly above each LoRA bar to avoid overlap
    for i in range(1, len(methods)):
        reduction = (1 - trainable_params[i] / trainable_params[0]) * 100
        params = trainable_params[i]
        ax2.text(
            i,
            params * 1.3,  # place above the bar and its value label (log scale)
            f'{reduction:.1f}%\nreduction',
            ha='center',
            va='bottom',
            fontsize=7,
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor=colors[i],
                alpha=0.9
            )
        )

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_fullft_vs_lora_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure_fullft_vs_lora_comparison.pdf', bbox_inches='tight')
    plt.close()

    print("Full FT vs LoRA comparison figure saved")


def generate_pareto_frontier_figure(output_dir: Path, full_ft_data: dict, lora_data: dict):
    """
    Generate Pareto frontier plot: Performance vs Parameters
    Shows the efficiency-performance tradeoff
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # Data points
    methods = ['Full FT', 'SAM+LoRA r=16', 'SAM+LoRA r=8', 'SAM+LoRA r=4']
    means = [
        full_ft_data['mean'],
        lora_data['r16']['mean'],
        lora_data['r8']['mean'],
        lora_data['r4']['mean']
    ]
    stds = [
        full_ft_data['std'],
        lora_data['r16']['std'],
        lora_data['r8']['std'],
        lora_data['r4']['std']
    ]
    trainable_pct = [
        full_ft_data['trainable_pct'],
        lora_data['r16']['trainable_pct'],
        lora_data['r8']['trainable_pct'],
        lora_data['r4']['trainable_pct']
    ]

    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    markers = ['s', 'o', '^', 'D']

    # Plot points with error bars
    for i, (method, mean, std, pct, color, marker) in enumerate(zip(
            methods, means, stds, trainable_pct, colors, markers)):
        ax.errorbar(pct, mean, yerr=std, fmt=marker, markersize=12,
                   color=color, ecolor=color, elinewidth=2, capsize=5,
                   capthick=2, label=method, alpha=0.8, markeredgecolor='black',
                   markeredgewidth=1.5)

        # Add annotations - place at bottom of segments
        offset_x = 5 if i < 2 else -15
        offset_y = -15  # Position all labels at bottom
        ax.annotate(method, (pct, mean), xytext=(offset_x, offset_y),
                   textcoords='offset points', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor=color, alpha=0.9))

    # Connect points to show frontier
    sorted_idx = np.argsort(trainable_pct)
    ax.plot(np.array(trainable_pct)[sorted_idx], np.array(means)[sorted_idx],
           'k--', alpha=0.3, linewidth=1, zorder=0)

    ax.set_xlabel('Trainable Parameters (% of SAM)', fontweight='bold')
    ax.set_ylabel('Validation mIoU (%)', fontweight='bold')
    ax.set_title('Performance-Efficiency Pareto Frontier', fontweight='bold', pad=15)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', framealpha=0.95)

    # Add an efficiency annotation for LoRA r=16.
    # It retains 99.8% of the updated five-fold Full FT mean.
    lora_r16_mean = lora_data['r16']['mean']
    ft_mean = full_ft_data['mean']
    pct_of_ft = (lora_r16_mean / ft_mean) * 100
    ax.annotate(
        f'Sweet spot:\n{pct_of_ft:.1f}% of Full FT\n1.25% parameters',
        xy=(lora_data['r16']['trainable_pct'], lora_r16_mean),
        xytext=(20, 79),
        textcoords='data',
        arrowprops=dict(
            arrowstyle='->',
            lw=2,
            color='#3498DB',
            connectionstyle='arc3,rad=-0.2'
        ),
        fontsize=9,
        ha='left',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='lightyellow',
            edgecolor='#3498DB',
            alpha=0.9,
            linewidth=2
        )
    )

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_pareto_frontier.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure_pareto_frontier.pdf', bbox_inches='tight')
    plt.close()

    print("Pareto frontier figure saved")


def main():
    # Use the non-resolved path to avoid symlink issues
    base_dir = Path(__file__).parents[2]
    json_path = base_dir / 'artifacts' / 'newdata' / 'fullft_lora_comparison.json'
    output_dir = base_dir / 'figures_new'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING FULL FT VS LORA FIGURES")
    print("=" * 70)
    print()

    # Load data from JSON
    print("Loading full FT vs LoRA comparison data...")
    full_ft_data, lora_data = load_comparison_data(json_path)
    print(f"Full FT: mIoU={full_ft_data['mean']:.2f}±{full_ft_data['std']:.2f}%")
    for k, v in lora_data.items():
        print(f"LoRA {k}: mIoU={v['mean']:.2f}±{v['std']:.2f}%")
    print()

    print("Generating figures...")
    generate_method_comparison_figure(output_dir, full_ft_data, lora_data)
    generate_pareto_frontier_figure(output_dir, full_ft_data, lora_data)
    print()

    print("=" * 70)
    print(f"FIGURES SAVED TO: {output_dir.absolute()}")
    print("=" * 70)
    print()
    print("Files generated:")
    print("  - figure_fullft_vs_lora_comparison.png/pdf (2-panel comparison)")
    print("  - figure_pareto_frontier.png/pdf (efficiency-performance tradeoff)")


if __name__ == '__main__':
    main()
