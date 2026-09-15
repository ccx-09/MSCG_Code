#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_per_cell(sam_csv: Path, cls_csv: Path):
    sam = pd.read_csv(sam_csv)
    cls = pd.read_csv(cls_csv)
    sam['severity'] = sam['severity'].astype(int)
    cls['severity'] = cls['severity'].astype(int)
    return (sam, cls)

def aggregate_by_severity(df: pd.DataFrame) -> pd.Series:
    g = df.groupby('severity').apply(lambda d: (d['miou'] * d['count']).sum() / d['count'].sum())
    return g.sort_index()

def compute_er(sev_series: pd.Series) -> dict:
    m0 = float(sev_series.loc[0]) if 0 in sev_series.index else float('nan')
    m5 = float(sev_series.loc[5]) if 5 in sev_series.index else float('nan')
    aus = float(sev_series.values.sum())
    er_end = float(m5 / m0) if np.isfinite(m0) and m0 > 0 else float('nan')
    er_aus = float(aus / len(sev_series) / m0) if np.isfinite(m0) and m0 > 0 else float('nan')
    return {'miou_k0': m0, 'miou_k5': m5, 'er_end': er_end, 'er_aus': er_aus, 'maintains_60pct_at_k5': bool(m5 >= 0.6 if np.isfinite(m5) else False)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam-csv', required=True, type=Path, help='Path to SAM+LoRA per_cell.csv')
    parser.add_argument('--cls-csv', required=True, type=Path, help='Path to Classical per_cell.csv')
    parser.add_argument('--output-dir', required=True, type=Path, help='Output directory for JSON and PNG')
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sam, cls = load_per_cell(args.sam_csv, args.cls_csv)
    sam_sev = aggregate_by_severity(sam)
    cls_sev = aggregate_by_severity(cls)
    sam_er = compute_er(sam_sev)
    cls_er = compute_er(cls_sev)
    summary = {'SAM+LoRA': sam_er, 'Classical': cls_er}
    (out_dir / 'effective_robustness.json').write_text(json.dumps(summary, indent=2))
    plt.figure(figsize=(6.5, 3.6))
    plt.plot(sam_sev.index, sam_sev.values, label='SAM+LoRA', color='#1f77b4', marker='o')
    plt.plot(cls_sev.index, cls_sev.values, label='Classical', color='#d62728', marker='o')
    plt.xlabel('Severity (k)')
    plt.ylabel('Two-class mIoU')
    plt.title('Performance Trajectory vs Severity')
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'trajectory_severity.png', dpi=150)
    plt.close()
if __name__ == '__main__':
    main()
