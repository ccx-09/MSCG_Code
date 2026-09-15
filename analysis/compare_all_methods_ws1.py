#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path
from scipy import stats
from tabulate import tabulate

def load_method_results(stats_dir: Path, method: str, num_folds: int=5):
    results = []
    for fold in range(num_folds):
        summary_file = stats_dir / f'fold{fold}' / method / 'summary.json'
        if summary_file.exists():
            with open(summary_file) as f:
                data = json.load(f)
                results.append({'fold': fold, 'miou': data['overall']['miou'], 'dice': data['overall']['dice'], 'accuracy': data['overall']['accuracy'], 'per_scale_miou': data.get('per_scale_miou', {}), 'per_severity_miou': data.get('per_severity_miou', {}), 'per_corruption_miou': data.get('per_corruption_miou', {}), 'brittleness': data.get('brittleness_coefficient', np.nan)})
    return results

def compute_statistics(values):
    return {'mean': float(np.mean(values)), 'std': float(np.std(values)), 'min': float(np.min(values)), 'max': float(np.max(values)), 'values': [float(v) for v in values]}

def paired_t_test(values1, values2):
    t_stat, p_value = stats.ttest_rel(values1, values2)
    return {'t_statistic': float(t_stat), 'p_value': float(p_value), 'significant_p05': bool(p_value < 0.05), 'significant_p01': bool(p_value < 0.01), 'significant_p001': bool(p_value < 0.001)}

def cohen_d(values1, values2):
    diff = np.array(values1) - np.array(values2)
    return float(np.mean(diff) / np.std(diff, ddof=1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats_dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    methods = {'sam': load_method_results(args.stats_dir, 'sam'), 'xgb': load_method_results(args.stats_dir, 'xgb'), 'deeplabv3': load_method_results(args.stats_dir, 'deeplabv3')}
    for method, results in methods.items():
        pass
    complete_methods = {name: res for name, res in methods.items() if len(res) == 5}
    if len(complete_methods) < 2:
        return 1
    comparison = {}
    for method, results in complete_methods.items():
        comparison[method] = {'miou': compute_statistics([r['miou'] for r in results]), 'dice': compute_statistics([r['dice'] for r in results]), 'accuracy': compute_statistics([r['accuracy'] for r in results]), 'brittleness': compute_statistics([r['brittleness'] for r in results if not np.isnan(r['brittleness'])])}
    table_data = []
    for metric in ['miou', 'dice', 'accuracy']:
        row = [metric.upper()]
        for method in complete_methods.keys():
            stats_dict = comparison[method][metric]
            row.append(f"{stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}")
        table_data.append(row)
    headers = ['Metric'] + [m.upper() for m in complete_methods.keys()]
    method_list = list(complete_methods.keys())
    pairwise_tests = {}
    for i, method1 in enumerate(method_list):
        for method2 in method_list[i + 1:]:
            pair_key = f'{method1}_vs_{method2}'
            pairwise_tests[pair_key] = {}
            for metric in ['miou', 'dice', 'accuracy']:
                values1 = comparison[method1][metric]['values']
                values2 = comparison[method2][metric]['values']
                t_test = paired_t_test(values1, values2)
                effect = cohen_d(values1, values2)
                pairwise_tests[pair_key][metric] = {'t_test': t_test, 'cohen_d': effect, 'mean_diff': float(np.mean(values1) - np.mean(values2))}
                sig = '***' if t_test['significant_p001'] else '**' if t_test['significant_p01'] else '*' if t_test['significant_p05'] else 'ns'
    rankings = sorted(complete_methods.keys(), key=lambda m: comparison[m]['miou']['mean'], reverse=True)
    for rank, method in enumerate(rankings, 1):
        miou_stats = comparison[method]['miou']
    scales = ['s1', 's2', 's3', 's4', 's5']
    scale_table = []
    for scale in scales:
        row = [scale]
        for method in complete_methods.keys():
            results = complete_methods[method]
            scale_mious = [r['per_scale_miou'].get(scale, np.nan) for r in results if r['per_scale_miou']]
            if scale_mious:
                mean_miou = np.nanmean(scale_mious)
                row.append(f'{mean_miou:.4f}')
            else:
                row.append('N/A')
        scale_table.append(row)
    output_data = {'methods': list(complete_methods.keys()), 'num_folds': 5, 'comparison': comparison, 'pairwise_tests': pairwise_tests, 'rankings': rankings, 'summary': {'best_method': rankings[0], 'best_miou': comparison[rankings[0]]['miou']['mean']}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    return 0
if __name__ == '__main__':
    import sys
    sys.exit(main())
