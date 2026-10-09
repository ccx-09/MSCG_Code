#!/usr/bin/env python3
"""
Generate the prompt ablation figure for point versus box prompts.
"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Any
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def generate(output_path: str, data_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = load_results(data_path)


    pairs = [(r["iou_before"], r["iou_after"]) for r in data.get("results", [])]
    mean_before = float(data.get("mean_before", 0.0))
    mean_after = float(data.get("mean_after", 0.0))

    # Figure layout
    fig = plt.figure(figsize=(11.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    # Left: paired improvements for each example
    ax_left.set_title("Per‑example IoU (Point → Box)", pad=8)
    y_positions = list(range(len(pairs)))[::-1]  # Ex 1 on top
    for i, (b, a) in enumerate(pairs):
        y = y_positions[i]
        # Connection with arrow to emphasize improvement
        ax_left.annotate(
            "",
            xy=(a, y),
            xytext=(b, y),
            arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=2.5, shrinkA=0, shrinkB=0),
            zorder=1,
        )
        ax_left.scatter([b], [y], color="#b91c1c", s=80, marker="o", edgecolor="#7f1d1d", linewidth=0.6, label="Point" if i == 0 else "", zorder=2)
        ax_left.scatter([a], [y], color="#065f46", s=80, marker="o", edgecolor="#064e3b", linewidth=0.6, label="Box" if i == 0 else "", zorder=2)
    ax_left.set_yticks(y_positions)
    ax_left.set_yticklabels([f"Ex {i+1}" for i in range(len(pairs))])
    ax_left.set_xlim(0, 1)
    ax_left.set_xlabel("IoU")
    ax_left.grid(axis="x", color="#e5e7eb")
    ax_left.legend(frameon=False, loc="lower right")

    # Right: mean before vs after
    ax_right.set_title("Mean IoU", pad=8)
    bars = ax_right.bar(["Point", "Box"], [mean_before, mean_after], color=["#fecaca", "#bbf7d0"], edgecolor="#374151")
    for rect, val in zip(bars, [mean_before, mean_after]):
        ax_right.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    ax_right.set_ylim(0, 1)
    ax_right.set_ylabel("IoU")
    ax_right.grid(axis="y", color="#e5e7eb")


    fig.subplots_adjust(top=0.82, left=0.08, right=0.98, bottom=0.16)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    figures_dir = base_dir / "figures"
    data_dir = base_dir / "data" / "ws1_full_val"

    out = figures_dir / "figure5_box_vs_point.png"
    src = data_dir / "box_prompt_vs_point_results.json"

    generate(str(out), str(src))
    print(f"Saved: {out}")
