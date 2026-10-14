from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    active_labels: list[int] | None = None,
) -> dict[str, float]:
    if active_labels is None:
        active_labels = sorted(set(labels.tolist()) | set(predictions.tolist()))
    all_labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_precision": float(
            precision_score(
                labels,
                predictions,
                labels=active_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                labels,
                predictions,
                labels=active_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=active_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                labels=active_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "all_class_macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=all_labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def save_classification_outputs(
    output_dir: str | Path,
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    prefix: str = "val",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_labels = list(range(num_classes))
    label_names = [str(index) for index in all_labels]

    matrix = confusion_matrix(labels, predictions, labels=all_labels)
    pd.DataFrame(matrix, index=label_names, columns=label_names).to_csv(
        output_dir / f"{prefix}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    support = matrix.sum(axis=1)
    normalized = np.divide(matrix, support[:, None],
                           out=np.zeros_like(matrix, dtype=np.float64), where=support[:, None] > 0)
    pd.DataFrame(normalized, index=label_names, columns=label_names).to_csv(
        output_dir / f"{prefix}_confusion_matrix_normalized.csv", encoding="utf-8-sig")

    report = classification_report(
        labels,
        predictions,
        labels=all_labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    for index, name in enumerate(label_names):
        # Per-class accuracy is correct / true samples, i.e. class recall.
        report[name]["class_accuracy"] = float(normalized[index, index]) if support[index] else None
        report[name]["correct_count"] = int(matrix[index, index])
    with (output_dir / f"{prefix}_classification_report.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    pd.DataFrame(report).transpose().to_csv(
        output_dir / f"{prefix}_classification_report.csv",
        encoding="utf-8-sig",
    )

