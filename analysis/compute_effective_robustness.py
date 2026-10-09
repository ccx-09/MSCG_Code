#!/usr/bin/env python3
"""Reproduce Table 8 and Figure 6 from all three methods and five test folds."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev


METHODS = {"sam": "SAM+LoRA", "deeplabv3": "DeepLabV3+", "xgb": "VGG16+XGBoost"}
FOLDS = tuple(range(5))
SEVERITIES = tuple(range(6))
CONDITIONS = ("clean", "fog", "gaussian_blur", "shot_noise")
SCALES = tuple(f"s{i}" for i in range(1, 6))
COLORS = {"sam": "#1f77b4", "deeplabv3": "#2ca02c", "xgb": "#d62728"}
METRICS = ("miou_k0_pct", "miou_k5_pct", "absolute_drop_pp", "retention_pct", "er_aus_pct")


def load_trajectory(path: Path) -> list[float]:
    """Keep the published pixel-count-weighted cell-mIoU definition."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(c, s, k) for c in CONDITIONS for s in SCALES for k in SEVERITIES}
    cells = {}
    for row in rows:
        key = (row["corruption"], row["scale_bin"], int(row["severity"]))
        if key in cells:
            raise ValueError(f"Duplicate cell {key} in {path}")
        count, score = int(row["count"]), float(row["miou"])
        if count <= 0 or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid count or mIoU for {key} in {path}")
        cells[key] = (count, score)
    if set(cells) != expected:
        raise ValueError(f"Expected the complete 120-cell grid in {path}")
    return [
        math.fsum(n * score for (c, s, k), (n, score) in cells.items() if k == severity)
        / sum(n for (c, s, k), (n, score) in cells.items() if k == severity)
        for severity in SEVERITIES
    ]


def fold_metrics(trajectory: list[float]) -> dict:
    m0, m5 = trajectory[0], trajectory[5]
    if m0 <= 0:
        raise ValueError("Endpoint normalization requires a positive k0 mIoU")
    return {
        "miou_k0_pct": 100 * m0,
        "miou_k5_pct": 100 * m5,
        "absolute_drop_pp": 100 * (m0 - m5),
        "retention_pct": 100 * m5 / m0,
        "er_aus_pct": 100 * mean(value / m0 for value in trajectory),
    }


def describe(values) -> dict:
    values = list(values)
    return {"mean": mean(values), "population_sd": pstdev(values), "fold_values": values}


def build_summary(stats_dir: Path) -> dict:
    summary = {
        "protocol": {
            "folds": list(FOLDS),
            "severities": list(SEVERITIES),
            "conditions": list(CONDITIONS),
            "within_fold": "Pixel-count-weighted mean of the 20 condition-stratum cell mIoUs at each severity.",
            "across_folds": "Unweighted mean and population SD (ddof=0); ratios are computed within each fold first.",
            "no_transform_cells_retained": True,
            "scope": "Complete-grid descriptive trajectory; not corrupted-only pooled-confusion mIoU.",
            "input_precision": "Cell mIoUs read from the released six-decimal CSV values.",
        },
        "methods": {},
    }
    for method, label in METHODS.items():
        folds = []
        for fold in FOLDS:
            path = stats_dir / f"fold{fold}" / method / "per_cell.csv"
            trajectory = load_trajectory(path)
            folds.append({
                "fold": fold,
                "input": path.relative_to(stats_dir).as_posix(),
                "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "trajectory_miou_pct": [100 * value for value in trajectory],
                "metrics": fold_metrics(trajectory),
            })
        summary["methods"][method] = {
            "label": label,
            "folds": folds,
            "trajectory_miou_pct": {
                str(k): describe(f["trajectory_miou_pct"][k] for f in folds) for k in SEVERITIES
            },
            "metrics": {metric: describe(f["metrics"][metric] for f in folds) for metric in METRICS},
        }
    return summary


def plot_trajectory(summary: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    style = {"pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        # Keep Figure 6 curves, colors, numeric severity ticks, and grid.
        # Show mIoU as percentage ticks and place the legend above the plot.
        # The plotted points are the five-fold means for all three methods.
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        markers = {"sam": "o", "deeplabv3": "o", "xgb": "o"}
        for method, label in METHODS.items():
            trajectory = summary["methods"][method]["trajectory_miou_pct"]
            ax.plot(
                SEVERITIES,
                [trajectory[str(k)]["mean"] / 100 for k in SEVERITIES],
                label=label,
                color=COLORS[method],
                marker=markers[method],
            )
        ax.set_xlabel("Severity (k)")
        ax.set_ylabel("Two-class mIoU")
        ax.set_ylim(0.5, 0.9)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9], ["50%", "60%", "70%", "80%", "90%"])
        ax.set_xticks(SEVERITIES)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
        fig.tight_layout()
        fig.savefig(output_dir / "figure_robustness_comparison.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / "figure_robustness_comparison.pdf", bbox_inches="tight")
        plt.close(fig)


def render_table(summary: dict) -> str:
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
        r"\setlength{\tabcolsep}{1.5pt}", r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Endpoint statistics for all three systems across the five shared held-out folds. Entries are the mean $\pm$ population standard deviation of the fold-level statistics. Endpoint retention and ER-AUS are normalized within each fold before averaging across folds.}",
        r"\label{tab:robustness_k_levels}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{k0 mIoU} & \textbf{k5 mIoU} & \textbf{Drop (pp)} & \textbf{Retention (\%)} & \textbf{ER-AUS (\%)} \\",
        r"\midrule",
    ]
    percent_metrics = {"miou_k0_pct", "miou_k5_pct", "retention_pct", "er_aus_pct"}
    for method, label in METHODS.items():
        cells = []
        for metric in METRICS:
            item = summary["methods"][method]["metrics"][metric]
            suffix = r"\%" if metric in percent_metrics else ""
            cells.append(f"{item['mean']:.2f} $\\pm$ {item['population_sd']:.2f}{suffix}")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule", r"\end{tabular*}", r"\vspace{0.15cm}",
        r"\begin{minipage}{\textwidth}\scriptsize",
        r"\textit{Note.} At each severity, fold-level mIoU is the pixel-count-weighted mean of the 20 condition--stratum cell mIoUs. The complete grid retains all no-transform cells, including the clean condition at k1--k5. Absolute drop is k0 mIoU minus k5 mIoU; retention is k5/k0; and ER-AUS averages the six k0-normalized scores. Standard deviations quantify variation across the shared folds.",
        r"\end{minipage}", r"\end{table}", "",
    ])
    return "\n".join(lines)


def write_outputs(summary: dict, output_dir: Path, table_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "effective_robustness.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "robustness_fold_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "fold", *METRICS])
        writer.writeheader()
        for method, result in summary["methods"].items():
            for fold in result["folds"]:
                writer.writerow({"method": method, "fold": fold["fold"], **fold["metrics"]})
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(render_table(summary), encoding="utf-8")
    plot_trajectory(summary, output_dir)


def main():
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-dir", type=Path, default=base / "artifacts/newdata/ws1_kfold")
    parser.add_argument("--output-dir", type=Path, default=base / "figures_new")
    parser.add_argument("--table-output", type=Path, default=base / "tables/table_robustness_k_levels.tex")
    args = parser.parse_args()
    summary = build_summary(args.stats_dir)
    write_outputs(summary, args.output_dir, args.table_output)
    for method, result in summary["methods"].items():
        print(result["label"])
        for metric, value in result["metrics"].items():
            print(f"  {metric}: {value['mean']:.6f} +/- {value['population_sd']:.6f}")


if __name__ == "__main__":
    main()
