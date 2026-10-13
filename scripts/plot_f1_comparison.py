"""Plot retained methods using saved best-checkpoint validation metrics."""
from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "figures" / "f1_comparison"
METHODS = [
    ("cnn_softmax", "CNN Softmax", "results"),
    ("atenet", "ATENet", "results"),
    ("invconv1d", "InvConv1D", "results"),
    ("mambasl", "MambaSL", "results"),
    ("modern_tcn", "ModernTCN", "results"),
    ("mptsnet1d", "MPTSNet1D", "results"),
    ("multigrain_infonce", "Multi-grain CNN", "results"),
    ("pulse_hypothesis_fused", "Hypothesis-fused CNN", "results"),
    ("modular_multigrain_iq", "Modular IQ ResNet1D", "results"),
]
DATASETS = [("tsrd", "TSRD"), ("real_radar", "Real radar")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-channels", type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    suffix = "_3ch" if args.input_channels == 3 else ""
    global OUTPUT
    OUTPUT = ROOT / "results/figures" / ("f1_comparison" + suffix)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, _ in DATASETS:
        for key, label, folder in METHODS:
            source = ROOT / folder / f"{key}_{dataset}{suffix}" / "final_metrics.json"
            if not source.exists():
                raise FileNotFoundError(f"No {args.input_channels}-channel training result: {source}; "
                                        "train it first, or use --input-channels 2 for historical results")
            metrics = json.loads(source.read_text(encoding="utf-8"))
            config = json.loads(source.with_name("resolved_pipeline.json").read_text(encoding="utf-8"))
            train = config.get("train", {})
            assert 0 <= metrics["macro_f1"] <= 1
            rows.append({
                "dataset": dataset, "method": label, "experiment": f"{key}_{dataset}",
                "input_channels": args.input_channels,
                "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"],
                "best_epoch": metrics.get("best_epoch"), "seed": train.get("seed"),
                "sampler": train.get("sampler"), "batch_size": train.get("batch_size"),
                "lr": train.get("lr"), "source": source.relative_to(ROOT).as_posix(),
            })
    with (OUTPUT / "f1_source_data.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.4))
    x = np.arange(len(METHODS))
    labels = [label.replace("Multi-grain CNN", "Multi-grain\nCNN")
              .replace("Hypothesis-fused CNN", "Hypothesis-fused\nCNN")
              .replace("Modular IQ ResNet1D", "Modular IQ\nResNet1D")
              for _, label, _ in METHODS]
    for ax, (dataset, title), letter in zip(axes, DATASETS, "ab"):
        data = [r for r in rows if r["dataset"] == dataset]
        values = [r["macro_f1"] for r in data]
        colors = ["#92ADC6"] * 6 + ["#B5ACCF", "#B5ACCF", "#D69864"]
        bars = ax.bar(x, values, width=0.65, color=colors, edgecolor="white", linewidth=0.7)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=5, fontsize=10)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylim(0, 1.04)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylabel("Macro-F1")
        ax.set_title(title, loc="left", fontsize=13, pad=12)
        ax.text(-0.055, 1.04, letter, transform=ax.transAxes, weight="bold", fontsize=14)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E7E7E7", linewidth=0.6)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.12, hspace=0.48)
    fig.suptitle("F1 comparison across retained methods", fontsize=16, y=0.98)
    fig.text(0.075, 0.035,
             "Saved best-checkpoint validation results; one run per method. Training settings vary.\n"
             "Orange: complete Modular IQ method. No uncertainty intervals available.",
             fontsize=9, color="#555555")
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT / f"macro_f1_comparison.{suffix}", dpi=300)
    plt.close(fig)
    note = """# F1 figure notes

Contract: compare the recorded Macro-F1 of nine retained methods separately on TSRD and real-radar data; do not presume the complete method is best.
Archetype: quantitative grid, two equally sized dataset panels, a common 0–1 axis, direct numeric labels.
Source: final_metrics.json from each experiment, best-checkpoint validation metrics; see f1_source_data.csv.
Macro-F1 uses each saved macro_f1 field without recomputation; it is distinct from weighted_f1 and all_class_macro_f1. Legacy metric conventions may differ, so use a common evaluation script for a final controlled benchmark.
Each bar represents one saved run, not a mean across seeds. No confidence intervals or significance claims are shown.
Learning rates, samplers, batch sizes and model-specific training settings vary; the CSV includes these settings. Scores are descriptive results, not a controlled ablation.
Real-radar data config names its val directory as the test set; these are the val-split results used to choose checkpoints, not a fresh independent held-out evaluation.
Exports: 300-dpi white-background PNG, editable-text SVG and PDF. Python/Matplotlib only.
"""
    (OUTPUT / "figure_notes.md").write_text(note, encoding="utf-8")
    for dataset, title in DATASETS:
        best = max((r for r in rows if r["dataset"] == dataset), key=lambda r: r["macro_f1"])
        print(f"{title}: highest saved Macro-F1 {best['method']} = {best['macro_f1']:.4f}")
    print(OUTPUT / "macro_f1_comparison.png")


if __name__ == "__main__":
    main()
