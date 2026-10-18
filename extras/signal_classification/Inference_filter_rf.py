"""Inference with a radar-frequency (RF) coarse filter.

Typical multi-modal RF regions are learned from known labeled data. Model
probabilities for RF-incompatible classes are then removed (hard mode) or
down-weighted (soft mode) before the final prediction is selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data_process.dataset import (
    NormalizationStats,
    SliceDataset,
    load_cut_source,
)
from data_process.csv_features import read_feature_csv
from train import FUSION_CHOICES, MODEL_CHOICES, extract_logits, metrics


def natural_key(value: str) -> tuple:
    parts = re.split(r"(\d+)", str(value))
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts)


def softmax_numpy(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


@dataclass
class RFProfile:
    class_name: str
    point_count: int
    histogram: np.ndarray
    active_bins: np.ndarray
    median: float
    q01: float
    q99: float


RF_CACHE_VERSION = 2

def default_cache_path(reference_root: str | Path) -> Path:
    reference_path = Path(reference_root)
    return reference_path / "_rf_profile_cache.npz"


def reference_signature(reference_root: str | Path) -> dict[str, object]:
    """Build a cheap signature for a raw ``class/*.csv`` reference tree."""
    reference_path = Path(reference_root).resolve()
    if not reference_path.is_dir():
        raise FileNotFoundError(f"Raw RF reference directory does not exist: {reference_path}")
    class_directories = sorted(
        (
            {
                "name": path.name,
                "mtime_ns": path.stat().st_mtime_ns,
                "csv_count": sum(1 for item in path.glob("*.csv") if item.is_file()),
            }
            for path in reference_path.iterdir()
            if path.is_dir()
        ),
        key=lambda item: natural_key(str(item["name"])),
    )
    return {
        "kind": "csv_directory",
        "path": str(reference_path),
        "class_directories": class_directories,
    }


def cache_metadata(
    args: argparse.Namespace,
    class_names: list[str],
    no_header_order: str | None = None,
) -> dict[str, object]:
    return {
        "version": RF_CACHE_VERSION,
        "reference": reference_signature(args.reference_root),
        "rf_bin_width": args.rf_bin_width,
        "rf_coverage": args.rf_coverage,
        "rf_tolerance_mhz": args.rf_tolerance_mhz,
        "no_header_order": args.no_header_order if no_header_order is None else no_header_order,
        "class_names": class_names,
    }


def save_profile_cache(
    path: Path,
    profiles: list[RFProfile | None],
    bin_edges: np.ndarray,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bin_count = len(bin_edges) - 1
    missing = np.asarray([profile is None for profile in profiles], dtype=bool)
    histograms = np.zeros((len(profiles), bin_count), dtype=np.float64)
    active_bins = np.zeros((len(profiles), bin_count), dtype=bool)
    point_counts = np.zeros(len(profiles), dtype=np.int64)
    medians = np.full(len(profiles), np.nan, dtype=np.float64)
    q01 = np.full(len(profiles), np.nan, dtype=np.float64)
    q99 = np.full(len(profiles), np.nan, dtype=np.float64)
    for index, profile in enumerate(profiles):
        if profile is None:
            continue
        histograms[index] = profile.histogram
        active_bins[index] = profile.active_bins
        point_counts[index] = profile.point_count
        medians[index] = profile.median
        q01[index] = profile.q01
        q99[index] = profile.q99

    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            bin_edges=np.asarray(bin_edges, dtype=np.float64),
            missing=missing,
            histograms=histograms,
            active_bins=active_bins,
            point_counts=point_counts,
            medians=medians,
            q01=q01,
            q99=q99,
        )
    os.replace(temporary_path, path)


def load_profile_cache(
    path: Path,
    expected_metadata: dict[str, object],
) -> tuple[list[RFProfile | None], np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        stored_metadata = json.loads(str(cache["metadata"].item()))
        if stored_metadata != expected_metadata:
            raise ValueError("cache metadata does not match the current reference data or RF parameters")
        bin_edges = np.asarray(cache["bin_edges"], dtype=np.float64)
        missing = np.asarray(cache["missing"], dtype=bool)
        histograms = np.asarray(cache["histograms"], dtype=np.float64)
        active_bins = np.asarray(cache["active_bins"], dtype=bool)
        point_counts = np.asarray(cache["point_counts"], dtype=np.int64)
        medians = np.asarray(cache["medians"], dtype=np.float64)
        q01 = np.asarray(cache["q01"], dtype=np.float64)
        q99 = np.asarray(cache["q99"], dtype=np.float64)

    class_names = list(expected_metadata["class_names"])
    expected_shape = (len(class_names), len(bin_edges) - 1)
    if histograms.shape != expected_shape or active_bins.shape != expected_shape:
        raise ValueError("cache array shape is invalid")
    profiles: list[RFProfile | None] = []
    for index, class_name in enumerate(class_names):
        if missing[index]:
            profiles.append(None)
        else:
            profiles.append(
                RFProfile(
                    class_name=str(class_name),
                    point_count=int(point_counts[index]),
                    histogram=histograms[index],
                    active_bins=active_bins[index],
                    median=float(medians[index]),
                    q01=float(q01[index]),
                    q99=float(q99[index]),
                )
            )
    return profiles, bin_edges


def _read_raw_rf_file(
    path: Path,
    bin_width: float,
    no_header_order: str | None = None,
) -> tuple[Path, Counter[int], int, float, float, str | None]:
    """Read only the RF column of one unsliced raw CSV."""
    try:
        frame = read_feature_csv(path, no_header_order=no_header_order).features
        values = frame["CF_MHz"].to_numpy(dtype=np.float64, copy=False)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("RF column contains no finite values")
        bin_ids = np.floor(values / bin_width).astype(np.int64)
        unique, counts = np.unique(bin_ids, return_counts=True)
        return (
            path,
            Counter({int(key): int(value) for key, value in zip(unique, counts)}),
            int(values.size),
            float(values.min()),
            float(values.max()),
            None,
        )
    except (OSError, ValueError, KeyError, UnicodeError, pd.errors.ParserError) as exc:
        return path, Counter(), 0, np.nan, np.nan, str(exc)


def _histogram_quantile(histogram: np.ndarray, edges: np.ndarray, quantile: float) -> float:
    total = float(histogram.sum())
    if total <= 0:
        return float("nan")
    target = max(0.0, min(1.0, quantile)) * total
    index = int(np.searchsorted(np.cumsum(histogram), target, side="left"))
    index = min(index, len(histogram) - 1)
    return float((edges[index] + edges[index + 1]) / 2.0)


def _iter_raw_rf_results(
    files: list[Path], bin_width: float, workers: int, no_header_order: str | None = None
):
    """Read files with a bounded number of in-flight futures."""
    if not files:
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        file_iterator = iter(files)
        pending = {
            executor.submit(_read_raw_rf_file, next(file_iterator), bin_width, no_header_order)
            for _ in range(min(len(files), workers * 2))
        }
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                try:
                    pending.add(
                        executor.submit(_read_raw_rf_file, next(file_iterator), bin_width, no_header_order)
                    )
                except StopIteration:
                    pass


def build_raw_rf_profiles(
    reference_root: str | Path,
    class_names: list[str],
    bin_width: float,
    coverage: float,
    tolerance_mhz: float,
    workers: int,
    error_output: Path | None = None,
    no_header_order: str | None = None,
) -> tuple[list[RFProfile | None], np.ndarray]:
    """Learn RF profiles from original CSVs without loading PDW sequences."""
    root = Path(reference_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw RF reference directory does not exist: {root}")
    class_dirs = {path.name: path for path in root.iterdir() if path.is_dir()}
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found below raw RF reference root: {root}")

    class_counts: dict[str, Counter[int]] = {}
    point_counts: dict[str, int] = {}
    errors: list[tuple[str, str]] = []
    min_bin: int | None = None
    max_bin: int | None = None
    workers = max(1, int(workers))
    for class_name in sorted(class_dirs, key=natural_key):
        files = sorted(
            (path for path in class_dirs[class_name].glob("*.csv") if path.is_file()),
            key=lambda path: natural_key(str(path)),
        )
        counts = Counter()
        points = 0
        results = _iter_raw_rf_results(files, bin_width, workers, no_header_order)
        for path, file_counts, file_points, _minimum, _maximum, error in tqdm(
            results, total=len(files), desc=f"RF reference {class_name}", leave=False
        ):
            if error is not None:
                errors.append((str(path), error))
                continue
            counts.update(file_counts)
            points += file_points
            if file_counts:
                file_min = min(file_counts)
                file_max = max(file_counts)
                min_bin = file_min if min_bin is None else min(min_bin, file_min)
                max_bin = file_max if max_bin is None else max(max_bin, file_max)
        class_counts[class_name] = counts
        point_counts[class_name] = points

    if errors:
        destination = error_output or root / "rf_reference_read_errors.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "error"])
            writer.writerows(errors)
        print(f"RF reference files skipped: {len(errors)}; details saved to {destination}")
    if min_bin is None or max_bin is None:
        raise ValueError("Raw RF reference data contains no finite RF values")

    bin_edges = np.arange(min_bin, max_bin + 2, dtype=np.float64) * bin_width
    tolerance_bins = int(np.ceil(tolerance_mhz / bin_width))
    profiles: list[RFProfile | None] = []
    bin_count = len(bin_edges) - 1
    for class_name in class_names:
        counts = class_counts.get(str(class_name), Counter())
        if not counts:
            profiles.append(None)
            continue
        histogram_counts = np.zeros(bin_count, dtype=np.float64)
        for bin_id, count in counts.items():
            index = bin_id - min_bin
            if 0 <= index < bin_count:
                histogram_counts[index] = count
        total = histogram_counts.sum()
        histogram = histogram_counts / total
        order = np.argsort(histogram)[::-1]
        selected_count = int(np.searchsorted(np.cumsum(histogram[order]), coverage, side="left")) + 1
        active = np.zeros(bin_count, dtype=bool)
        active[order[:selected_count]] = True
        active = dilate_mask(active, tolerance_bins)
        profiles.append(
            RFProfile(
                class_name=str(class_name),
                point_count=int(point_counts.get(str(class_name), int(total))),
                histogram=histogram,
                active_bins=active,
                median=_histogram_quantile(histogram_counts, bin_edges, 0.5),
                q01=_histogram_quantile(histogram_counts, bin_edges, 0.01),
                q99=_histogram_quantile(histogram_counts, bin_edges, 0.99),
            )
        )
    return profiles, bin_edges


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = np.ones(radius * 2 + 1, dtype=np.int32)
    return np.convolve(mask.astype(np.int32), kernel, mode="same") > 0


def mask_to_intervals(mask: np.ndarray, edges: np.ndarray) -> list[tuple[float, float]]:
    intervals = []
    start = None
    for index, active in enumerate(np.append(mask, False)):
        if active and start is None:
            start = index
        elif not active and start is not None:
            intervals.append((float(edges[start]), float(edges[index])))
            start = None
    return intervals


def save_rf_profiles(
    profiles: list[RFProfile | None],
    class_names: list[str],
    edges: np.ndarray,
    output_dir: Path,
) -> None:
    csv_path = output_dir / "rf_profiles.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_name", "rf_points", "median_mhz", "q01_mhz", "q99_mhz", "typical_intervals_mhz"])
        for index, profile in enumerate(profiles):
            if profile is None:
                writer.writerow([class_names[index], 0, "", "", "", "UNAVAILABLE"])
                continue
            intervals = ";".join(f"{lower:.3f}-{upper:.3f}" for lower, upper in mask_to_intervals(profile.active_bins, edges))
            writer.writerow([profile.class_name, profile.point_count, profile.median, profile.q01, profile.q99, intervals])

    available = [profile for profile in profiles if profile is not None]
    if not available:
        return
    matrix = np.asarray([profile.histogram for profile in available])
    height = min(30.0, max(7.0, len(available) * 0.22))
    figure, axis = plt.subplots(figsize=(14, height), constrained_layout=True)
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        extent=(edges[0], edges[-1], len(available) - 0.5, -0.5),
    )
    axis.set_yticks(np.arange(len(available)), [profile.class_name for profile in available])
    axis.set_xlabel("RF (MHz)")
    axis.set_ylabel("Radar type")
    axis.set_title("Typical RF distributions learned from reference data")
    figure.colorbar(image, ax=axis, label="Probability")
    figure.savefig(output_dir / "rf_profile_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def collect_model_logits(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    groups: list[str] = []
    model.eval()
    with torch.no_grad():
        for inputs, labels, batch_groups in loader:
            outputs = extract_logits(model(inputs.float().to(device)))
            logits_parts.append(outputs.detach().cpu().numpy())
            label_parts.append(labels.numpy())
            groups.extend(str(group) for group in batch_groups)
    return np.concatenate(logits_parts), np.concatenate(label_parts), groups


def fuse_model_outputs(
    logits: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    fusion: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[np.ndarray]]:
    if fusion == "none":
        probabilities = softmax_numpy(logits)
        names = [f"{group}#slice{index:06d}" for index, group in enumerate(groups)]
        return probabilities, labels, names, [np.asarray([index]) for index in range(len(labels))]

    grouped: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        grouped.setdefault(group, []).append(index)
    ordered_groups = sorted(grouped, key=natural_key)
    probabilities = []
    fused_labels = []
    item_indices = []
    slice_probabilities = softmax_numpy(logits)
    slice_predictions = logits.argmax(axis=1)
    for group in ordered_groups:
        indices = np.asarray(grouped[group], dtype=np.int64)
        unique_labels = np.unique(labels[indices])
        if len(unique_labels) != 1:
            raise ValueError(f"Source group {group!r} contains inconsistent labels")
        if fusion == "vote":
            scores = np.bincount(slice_predictions[indices], minlength=logits.shape[1]).astype(np.float64)
            scores /= scores.sum()
        elif fusion == "score":
            scores = softmax_numpy(logits[indices].mean(axis=0, keepdims=True))[0]
        else:
            scores = slice_probabilities[indices].mean(axis=0)
        probabilities.append(scores)
        fused_labels.append(int(unique_labels[0]))
        item_indices.append(indices)
    return np.asarray(probabilities), np.asarray(fused_labels), ordered_groups, item_indices


def rf_compatibility(values: np.ndarray, profile: RFProfile, edges: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0
    inside = (values >= edges[0]) & (values < edges[-1])
    compatible = np.zeros(len(values), dtype=bool)
    indices = np.searchsorted(edges, values[inside], side="right") - 1
    compatible[inside] = profile.active_bins[indices]
    return float(compatible.mean())


def compute_compatibility_matrix(
    test_data: np.ndarray,
    item_indices: list[np.ndarray],
    profiles: list[RFProfile | None],
    edges: np.ndarray,
) -> np.ndarray:
    result = np.ones((len(item_indices), len(profiles)), dtype=np.float64)
    for item_index, indices in enumerate(item_indices):
        rf = test_data[indices, 0, :].reshape(-1)
        for class_index, profile in enumerate(profiles):
            if profile is not None:
                result[item_index, class_index] = rf_compatibility(rf, profile, edges)
    return result


def correct_predictions(
    model_probabilities: np.ndarray,
    compatibility: np.ndarray,
    profiles: list[RFProfile | None],
    mode: str,
    threshold: float,
    weight: float,
    empty_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corrected = model_probabilities.copy()
    eligible = compatibility >= threshold
    for class_index, profile in enumerate(profiles):
        if profile is None:
            eligible[:, class_index] = True

    if mode == "hard":
        for row in range(len(corrected)):
            if eligible[row].any():
                corrected[row, ~eligible[row]] = 0.0
            elif empty_policy == "closest":
                keep = int(np.argmax(compatibility[row]))
                corrected[row, :] = 0.0
                corrected[row, keep] = model_probabilities[row, keep]
    else:
        corrected *= np.power(np.clip(compatibility, 1e-6, 1.0), weight)
        for class_index, profile in enumerate(profiles):
            if profile is None:
                corrected[:, class_index] = model_probabilities[:, class_index]

    row_sums = corrected.sum(axis=1, keepdims=True)
    empty_rows = row_sums[:, 0] <= 0
    corrected[empty_rows] = model_probabilities[empty_rows]
    corrected /= corrected.sum(axis=1, keepdims=True)
    return corrected.argmax(axis=1), corrected, eligible


def save_results(
    output_dir: Path,
    output_name: str,
    class_names: list[str],
    item_groups: list[str],
    model_probabilities: np.ndarray,
    corrected_probabilities: np.ndarray,
    compatibility: np.ndarray,
    eligible: np.ndarray,
    corrected_predictions: np.ndarray,
) -> None:
    raw_predictions = model_probabilities.argmax(axis=1)
    prediction_path = output_dir / output_name
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "predicted_label"])
        for index, prediction in enumerate(corrected_predictions):
            writer.writerow([index, class_names[int(prediction)]])

    with (output_dir / "rf_filter_details.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "index",
                "source_group",
                "model_prediction",
                "model_confidence",
                "corrected_prediction",
                "corrected_confidence",
                "prediction_changed",
                "rf_candidates",
                "best_rf_match",
                "best_rf_match_ratio",
            ]
        )
        for index in range(len(corrected_predictions)):
            raw = int(raw_predictions[index])
            corrected = int(corrected_predictions[index])
            candidates = [class_names[i] for i in np.flatnonzero(eligible[index])]
            best_rf = int(np.argmax(compatibility[index]))
            writer.writerow(
                [
                    index,
                    item_groups[index],
                    class_names[raw],
                    model_probabilities[index, raw],
                    class_names[corrected],
                    corrected_probabilities[index, corrected],
                    int(raw != corrected),
                    ";".join(candidates),
                    class_names[best_rf],
                    compatibility[index, best_rf],
                ]
            )
    print(f"RF-filtered predictions saved to: {prediction_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model inference corrected by known radar RF distributions")
    parser.add_argument(
        "--reference_root",
        default="./raw_data",
        help="已知型号原始数据根目录，结构为 reference_root/<类别>/*.csv；仅读取RF列",
    )
    parser.add_argument("--test_root", "--data_root", dest="test_root", default="./dataset/cut_data/test", help="待测试数据目录或H5路径")
    parser.add_argument("--test_format", choices=["auto", "h5", "csv"], default="auto")
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--normalization", choices=["none", "standard", "minmax"], default="standard")
    parser.add_argument("--dataset", default="cut_data")
    parser.add_argument("--outdir", default="./result")
    parser.add_argument("--session_name", default="radar_1")
    parser.add_argument("--model", default="ResNet18", choices=MODEL_CHOICES)
    parser.add_argument("--weight_name", default="model_best")
    parser.add_argument("--output_csv", default="predictions_rf_filtered.csv")
    parser.add_argument("--output_dir", default=None, help="RF分析结果目录；默认模型实验目录")
    parser.add_argument("--rf_profile_cache", default=None, help="RF分布缓存NPZ路径；默认保存在reference_root旁")
    parser.add_argument("--rebuild_rf_profile", action="store_true", help="忽略已有缓存并重新统计RF分布")
    parser.add_argument("--rf_workers", type=int, default=8, help="读取原始RF CSV的线程数")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--device_type", choices=["auto", "npu", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fusion", choices=FUSION_CHOICES, default="probability")
    parser.add_argument("--rf_bin_width", type=float, default=50.0, help="RF直方图箱宽，单位MHz")
    parser.add_argument("--rf_coverage", type=float, default=0.99, help="典型RF频段覆盖的已知数据比例")
    parser.add_argument("--rf_tolerance_mhz", type=float, default=100.0, help="典型频段两侧容差，单位MHz")
    parser.add_argument("--rf_match_threshold", type=float, default=0.5, help="测试信号落入典型频段的最低比例")
    parser.add_argument(
        "--no-header-order",
        default=None,
        help="无表头CSV的物理列顺序，例如 CF_MHz,PW_us,PRI_us；有表头CSV自动按列名识别",
    )
    parser.add_argument("--rf_mode", choices=["hard", "soft"], default="hard", help="hard排除不匹配型号；soft按RF匹配度降低概率")
    parser.add_argument("--rf_weight", type=float, default=2.0, help="soft模式的RF权重指数")
    parser.add_argument("--empty_policy", choices=["model", "closest"], default="model", help="没有型号通过RF阈值时保留模型结果或使用最接近型号")
    args = parser.parse_args()
    if args.rf_bin_width <= 0 or args.rf_tolerance_mhz < 0 or args.rf_weight <= 0 or args.rf_workers < 1:
        parser.error("rf_bin_width、rf_weight和rf_workers必须大于0，rf_tolerance_mhz不能小于0")
    if not 0 < args.rf_coverage <= 1 or not 0 <= args.rf_match_threshold <= 1:
        parser.error("rf_coverage必须在(0,1]，rf_match_threshold必须在[0,1]")
    return args


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = utils.resolve_device(args.device, args.device_type)
    print(f"Using device: {device}")

    experiment_dir = Path(args.outdir) / args.dataset / args.model / args.session_name
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = experiment_dir / "weights" / f"{args.weight_name}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    if not class_names:
        raise ValueError("Checkpoint does not contain class_names")
    no_header_order = args.no_header_order or checkpoint.get("no_header_order")

    cache_path = Path(args.rf_profile_cache) if args.rf_profile_cache else default_cache_path(args.reference_root)
    expected_metadata = cache_metadata(args, class_names, no_header_order)
    profiles = None
    bin_edges = None
    if cache_path.is_file() and not args.rebuild_rf_profile:
        try:
            profiles, bin_edges = load_profile_cache(cache_path, expected_metadata)
            print(f"Loaded cached RF profiles: {cache_path}")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"RF profile cache is not reusable ({exc}); rebuilding it")
    if profiles is None:
        print("No reusable RF profile cache found; calculating RF distributions")
        profiles, bin_edges = build_raw_rf_profiles(
            args.reference_root,
            class_names,
            args.rf_bin_width,
            args.rf_coverage,
            args.rf_tolerance_mhz,
            args.rf_workers,
            output_dir / "rf_reference_read_errors.csv",
            no_header_order,
        )
        save_profile_cache(cache_path, profiles, bin_edges, expected_metadata)
        print(f"RF profiles cached at: {cache_path}")

    test_data, test_raw_labels, test_groups = load_cut_source(
        args.test_root, args.sequence_length, args.test_format, no_header_order
    )
    label_map = {name: index for index, name in enumerate(class_names)}
    unknown_test = sorted({str(value) for value in test_raw_labels} - set(label_map), key=natural_key)
    if unknown_test:
        raise ValueError(f"Test data contains labels absent from checkpoint: {unknown_test}")
    test_labels = np.asarray([label_map[str(value)] for value in test_raw_labels], dtype=np.int64)

    if "normalization_center" in checkpoint and "normalization_scale" in checkpoint:
        normalization_stats = NormalizationStats(
            checkpoint.get("normalization", args.normalization),
            np.asarray(checkpoint["normalization_center"]),
            np.asarray(checkpoint["normalization_scale"]),
        )
    else:
        raise ValueError(
            "Checkpoint lacks normalization statistics. Raw RF reference CSVs are used only "
            "for RF profiling and cannot fit model-input normalization; retrain/save a checkpoint "
            "with normalization_center and normalization_scale."
        )
    normalized_test = normalization_stats.transform(test_data)
    dataset = SliceDataset(normalized_test, test_labels, test_groups)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = utils.load_model(
        args.model,
        len(class_names),
        dataset.data.shape[1],
        dropout=args.dropout,
        length=args.sequence_length,
        device=device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    logits, labels, groups = collect_model_logits(model, loader, device)
    model_probabilities, fused_labels, item_groups, item_indices = fuse_model_outputs(
        logits, labels, groups, args.fusion
    )

    missing_profiles = [class_names[index] for index, profile in enumerate(profiles) if profile is None]
    if missing_profiles:
        print(f"Warning: no reference RF profile for {len(missing_profiles)} classes; they will not be filtered")
    save_rf_profiles(profiles, class_names, bin_edges, output_dir)

    compatibility = compute_compatibility_matrix(test_data, item_indices, profiles, bin_edges)
    corrected_predictions, corrected_probabilities, eligible = correct_predictions(
        model_probabilities,
        compatibility,
        profiles,
        args.rf_mode,
        args.rf_match_threshold,
        args.rf_weight,
        args.empty_policy,
    )
    save_results(
        output_dir,
        args.output_csv,
        class_names,
        item_groups,
        model_probabilities,
        corrected_probabilities,
        compatibility,
        eligible,
        corrected_predictions,
    )

    raw_predictions = model_probabilities.argmax(axis=1)
    raw_metrics = metrics(fused_labels, raw_predictions, len(class_names))
    corrected_metrics = metrics(fused_labels, corrected_predictions, len(class_names))
    changed = int(np.count_nonzero(raw_predictions != corrected_predictions))
    print(
        f"Before RF filter: acc={raw_metrics['acc']:.4f}, macro_f1={raw_metrics['macro_f1']:.4f}"
    )
    print(
        f"After RF filter:  acc={corrected_metrics['acc']:.4f}, "
        f"macro_f1={corrected_metrics['macro_f1']:.4f}, changed={changed}/{len(raw_predictions)}"
    )


if __name__ == "__main__":
    main()
