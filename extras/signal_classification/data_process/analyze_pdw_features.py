"""Statistical comparison of CF, PW and PRI distributions across radar types.

Input data is organized as ``input_dir/class_name/*.csv``. Exact streaming
moments are computed for every analyzed row, while a bounded uniform sample is
kept per class for quantiles and plots so memory use does not grow with the
dataset size.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    from data_process.csv_features import (
        FEATURE_COLS,
        feature_cli_help,
        parse_no_header_order,
        read_feature_csv as read_feature_table,
    )
except ModuleNotFoundError:
    from csv_features import (
        FEATURE_COLS,
        feature_cli_help,
        parse_no_header_order,
        read_feature_csv as read_feature_table,
    )


FEATURES = FEATURE_COLS
FEATURE_LABELS = {
    "CF_MHz": "Carrier frequency (MHz)",
    "PW_us": "Pulse width (us)",
    "PRI_us": "PRI (us)",
}
COLORS = ("#1769aa", "#d95f02", "#238b45")


def natural_key(value: str) -> tuple:
    parts = re.split(r"(\d+)", str(value))
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts)


def configure_fonts() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def read_feature_csv(
    path: Path,
    no_header_order: str | None = None,
) -> tuple[Path, np.ndarray | None, str | None]:
    try:
        values = read_feature_table(path, no_header_order=no_header_order).features.to_numpy(
            dtype=np.float64,
            copy=False,
        )
        if not len(values):
            raise ValueError("no finite PDW rows")
        return path, values, None
    except Exception as exc:  # Keep a full error report instead of aborting a large run.
        return path, None, f"{type(exc).__name__}: {exc}"


@dataclass
class RunningMoments:
    count: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURES), dtype=np.float64))
    m2: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURES), dtype=np.float64))
    minimum: np.ndarray = field(default_factory=lambda: np.full(len(FEATURES), np.inf))
    maximum: np.ndarray = field(default_factory=lambda: np.full(len(FEATURES), -np.inf))

    def update(self, values: np.ndarray) -> None:
        if not len(values):
            return
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = np.square(values - batch_mean).sum(axis=0)
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean += delta * batch_count / total
            self.m2 += batch_m2 + np.square(delta) * self.count * batch_count / total
        self.count += batch_count
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    @property
    def std(self) -> np.ndarray:
        if self.count < 2:
            return np.zeros(len(FEATURES), dtype=np.float64)
        return np.sqrt(self.m2 / (self.count - 1))


class PrioritySample:
    """Uniform row sample using random priorities with bounded memory."""

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.values = np.empty((0, len(FEATURES)), dtype=np.float64)
        self.priorities = np.empty(0, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        if self.capacity == 0 or not len(values):
            return
        priorities = self.rng.random(len(values))
        combined_values = np.concatenate((self.values, values), axis=0)
        combined_priorities = np.concatenate((self.priorities, priorities))
        if len(combined_values) > self.capacity:
            indices = np.argpartition(combined_priorities, -self.capacity)[-self.capacity :]
            combined_values = combined_values[indices]
            combined_priorities = combined_priorities[indices]
        self.values = combined_values
        self.priorities = combined_priorities


@dataclass
class ClassAnalysis:
    name: str
    files_available: int
    files_selected: int
    files_read: int
    files_failed: int
    moments: RunningMoments
    sample: np.ndarray
    sequence: np.ndarray


def list_class_files(class_dir: Path) -> list[Path]:
    with os.scandir(class_dir) as entries:
        return [
            Path(entry.path)
            for entry in entries
            if entry.is_file() and entry.name.lower().endswith(".csv") and entry.name != "_index.csv"
        ]


def choose_files(paths: list[Path], maximum: int, rng: np.random.Generator) -> list[Path]:
    if maximum <= 0 or len(paths) <= maximum:
        return paths
    indices = rng.choice(len(paths), size=maximum, replace=False)
    return [paths[int(index)] for index in indices]


def analyze_class(
    class_dir: Path,
    max_files: int,
    sample_size: int,
    sequence_points: int,
    workers: int,
    rng: np.random.Generator,
    errors: list[tuple[str, str, str]],
    no_header_order: str | None = None,
) -> ClassAnalysis:
    paths = list_class_files(class_dir)
    selected = choose_files(paths, max_files, rng)
    moments = RunningMoments()
    reservoir = PrioritySample(sample_size, rng)
    sequence_parts: list[np.ndarray] = []
    sequence_count = 0
    files_read = 0

    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    reader = lambda path: read_feature_csv(path, no_header_order)
    results = executor.map(reader, selected) if executor else map(reader, selected)
    try:
        for path, values, error in tqdm(
            results,
            total=len(selected),
            desc=f"型号 {class_dir.name}",
            unit="file",
            leave=False,
        ):
            if error is not None:
                errors.append((class_dir.name, str(path), error))
                continue
            files_read += 1
            moments.update(values)
            reservoir.update(values)
            if sequence_count < sequence_points:
                part = values[: sequence_points - sequence_count]
                sequence_parts.append(part)
                sequence_count += len(part)
    finally:
        if executor is not None:
            executor.shutdown()

    sequence = (
        np.concatenate(sequence_parts, axis=0)
        if sequence_parts
        else np.empty((0, len(FEATURES)), dtype=np.float64)
    )
    return ClassAnalysis(
        name=class_dir.name,
        files_available=len(paths),
        files_selected=len(selected),
        files_read=files_read,
        files_failed=len(selected) - files_read,
        moments=moments,
        sample=reservoir.values,
        sequence=sequence,
    )


def write_statistics(analyses: list[ClassAnalysis], output_path: Path) -> None:
    quantiles = (0.05, 0.25, 0.50, 0.75, 0.95)
    columns = [
        "class_name",
        "feature",
        "files_available",
        "files_analyzed",
        "files_failed",
        "valid_points",
        "mean",
        "std",
        "min",
        "sample_q05",
        "sample_q25",
        "sample_median",
        "sample_q75",
        "sample_q95",
        "max",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for analysis in analyses:
            sampled_quantiles = np.quantile(analysis.sample, quantiles, axis=0)
            for feature_index, feature in enumerate(FEATURES):
                writer.writerow(
                    {
                        "class_name": analysis.name,
                        "feature": feature,
                        "files_available": analysis.files_available,
                        "files_analyzed": analysis.files_read,
                        "files_failed": analysis.files_failed,
                        "valid_points": analysis.moments.count,
                        "mean": analysis.moments.mean[feature_index],
                        "std": analysis.moments.std[feature_index],
                        "min": analysis.moments.minimum[feature_index],
                        "sample_q05": sampled_quantiles[0, feature_index],
                        "sample_q25": sampled_quantiles[1, feature_index],
                        "sample_median": sampled_quantiles[2, feature_index],
                        "sample_q75": sampled_quantiles[3, feature_index],
                        "sample_q95": sampled_quantiles[4, feature_index],
                        "max": analysis.moments.maximum[feature_index],
                    }
                )


def plot_sequences(analyses: list[ClassAnalysis], output_path: Path) -> None:
    usable = [analysis for analysis in analyses if len(analysis.sequence)]
    lengths = [len(analysis.sequence) for analysis in usable]
    boundaries = np.cumsum([0, *lengths])
    centers = (boundaries[:-1] + boundaries[1:]) / 2
    width = min(36.0, max(12.0, len(usable) * 0.55))
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(width, 10), constrained_layout=True)
    for feature_index, (feature, color) in enumerate(zip(FEATURES, COLORS)):
        axis = axes[feature_index]
        values = np.concatenate([analysis.sequence[:, feature_index] for analysis in usable])
        axis.plot(np.arange(len(values)), values, color=color, linewidth=0.65)
        for boundary in boundaries[1:-1]:
            axis.axvline(boundary, color="#777777", linewidth=0.6, linestyle="--", alpha=0.7)
        axis.set_ylabel(FEATURE_LABELS[feature])
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(centers, [analysis.name for analysis in usable], rotation=90)
    axes[-1].set_xlabel("Radar type (concatenated representative PDW samples)")
    figure.suptitle("PDW feature sequences by radar type")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_range(analyses: list[ClassAnalysis], feature_index: int) -> tuple[float, float]:
    values = np.concatenate([analysis.sample[:, feature_index] for analysis in analyses])
    lower, upper = np.quantile(values, (0.005, 0.995))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower, upper = float(values.min()), float(values.max())
    if lower == upper:
        lower, upper = lower - 0.5, upper + 0.5
    return float(lower), float(upper)


def plot_distribution_heatmaps(analyses: list[ClassAnalysis], bins: int, output_path: Path) -> None:
    width = 15.0
    height = min(32.0, max(9.0, len(analyses) * 0.22))
    figure, axes = plt.subplots(1, 3, figsize=(width, height), constrained_layout=True)
    for feature_index, feature in enumerate(FEATURES):
        lower, upper = _plot_range(analyses, feature_index)
        edges = np.linspace(lower, upper, bins + 1)
        rows = []
        for analysis in analyses:
            clipped = np.clip(analysis.sample[:, feature_index], lower, upper)
            counts, _ = np.histogram(clipped, bins=edges)
            rows.append(counts / max(counts.sum(), 1))
        matrix = np.asarray(rows)
        axis = axes[feature_index]
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            extent=(lower, upper, len(analyses) - 0.5, -0.5),
        )
        axis.set_title(FEATURE_LABELS[feature])
        axis.set_xlabel("Feature value (0.5%-99.5% sample range)")
        axis.set_yticks(np.arange(len(analyses)), [analysis.name for analysis in analyses])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Probability")
    axes[0].set_ylabel("Radar type")
    figure.suptitle("PDW distribution comparison by radar type")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_boxplots(analyses: list[ClassAnalysis], output_path: Path) -> None:
    height = min(32.0, max(9.0, len(analyses) * 0.25))
    figure, axes = plt.subplots(1, 3, figsize=(16, height), constrained_layout=True)
    labels = [analysis.name for analysis in analyses]
    for feature_index, feature in enumerate(FEATURES):
        axes[feature_index].boxplot(
            [analysis.sample[:, feature_index] for analysis in analyses],
            vert=False,
            tick_labels=labels,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": COLORS[feature_index], "alpha": 0.65},
            medianprops={"color": "black"},
        )
        axes[feature_index].set_title(FEATURE_LABELS[feature])
        axes[feature_index].grid(axis="x", alpha=0.25)
    axes[0].set_ylabel("Radar type")
    figure.suptitle("PDW feature boxplots by radar type (sampled quantiles)")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _js_distance(first: np.ndarray, second: np.ndarray) -> float:
    epsilon = np.finfo(np.float64).eps
    first = (first + epsilon) / (first.sum() + epsilon * len(first))
    second = (second + epsilon) / (second.sum() + epsilon * len(second))
    midpoint = (first + second) / 2.0
    divergence = 0.5 * np.sum(first * np.log(first / midpoint))
    divergence += 0.5 * np.sum(second * np.log(second / midpoint))
    return float(np.sqrt(max(divergence, 0.0) / np.log(2.0)))


def distribution_distances(
    analyses: list[ClassAnalysis], bins: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    count = len(analyses)
    feature_matrices = []
    for feature_index in range(len(FEATURES)):
        lower, upper = _plot_range(analyses, feature_index)
        edges = np.linspace(lower, upper, bins + 1)
        feature_matrices.append(
            np.asarray(
                [
                    np.histogram(np.clip(a.sample[:, feature_index], lower, upper), bins=edges)[0]
                    for a in analyses
                ],
                dtype=np.float64,
            )
        )

    combined = np.zeros((count, count), dtype=np.float64)
    rows: list[dict[str, object]] = []
    for first in range(count):
        for second in range(first + 1, count):
            distances = [
                _js_distance(matrix[first], matrix[second]) for matrix in feature_matrices
            ]
            average = float(np.mean(distances))
            combined[first, second] = combined[second, first] = average
            rows.append(
                {
                    "class_a": analyses[first].name,
                    "class_b": analyses[second].name,
                    **{f"{feature}_js_distance": value for feature, value in zip(FEATURES, distances)},
                    "mean_js_distance": average,
                }
            )
    return combined, rows


def write_distances(rows: list[dict[str, object]], output_path: Path) -> None:
    columns = ["class_a", "class_b", *[f"{feature}_js_distance" for feature in FEATURES], "mean_js_distance"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: float(row["mean_js_distance"]), reverse=True))


def plot_distance_heatmap(analyses: list[ClassAnalysis], matrix: np.ndarray, output_path: Path) -> None:
    size = min(28.0, max(9.0, len(analyses) * 0.24))
    figure, axis = plt.subplots(figsize=(size, size), constrained_layout=True)
    image = axis.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
    labels = [analysis.name for analysis in analyses]
    axis.set_xticks(np.arange(len(labels)), labels, rotation=90)
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_title("Radar-type PDW distribution distance")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Mean Jensen-Shannon distance")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_errors(errors: list[tuple[str, str, str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_name", "file", "error"])
        writer.writerows(errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CF, PW and PRI distributions for every radar type")
    parser.add_argument("--input_dir", "-i", default=r'G:\competition\2026第五届天智杯\赛前准备\雷达型号识别\test_competition_lengyan-main\test_competition_lengyan-main\datasets\synthetic_radar_overlap', help="数据根目录，直接子文件夹名称作为型号")
    parser.add_argument("--output_dir", "-o", default="./data_process/pdw_analysis", help="统计表和图片输出目录")
    parser.add_argument("--classes", default="", help="仅分析指定型号，使用英文逗号分隔；默认全部")
    parser.add_argument(
        "--max_files_per_class",
        type=int,
        default=10,
        help="每个型号随机分析的最大CSV数；默认10，设为0表示全量",
    )
    parser.add_argument("--sample_size", type=int, default=20000, help="每个型号用于分位数和分布图的均匀抽样点数")
    parser.add_argument("--sequence_points", type=int, default=256, help="连续样本图中每个型号保留的PDW点数")
    parser.add_argument("--bins", type=int, default=80, help="分布直方图箱数")
    parser.add_argument("--workers", type=int, default=4, help="并行读取CSV的线程数；机械硬盘建议1-2")
    parser.add_argument("--no-header-order", default=None, help=feature_cli_help())
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.max_files_per_class < 0 or args.sample_size < 1 or args.sequence_points < 1:
        parser.error("max_files_per_class不能小于0，sample_size和sequence_points必须大于0")
    if args.bins < 2 or args.workers < 1:
        parser.error("bins必须至少为2，workers必须至少为1")
    if args.no_header_order:
        try:
            parse_no_header_order(args.no_header_order)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    configure_fonts()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = {item.strip() for item in args.classes.split(",") if item.strip()}
    class_dirs = sorted(
        [path for path in input_dir.iterdir() if path.is_dir() and (not requested or path.name in requested)],
        key=lambda path: natural_key(path.name),
    )
    if not class_dirs:
        raise ValueError(f"No matching radar-type directories found below {input_dir}")
    missing = requested - {path.name for path in class_dirs}
    if missing:
        raise ValueError(f"Requested classes do not exist: {sorted(missing, key=natural_key)}")

    rng = np.random.default_rng(args.seed)
    errors: list[tuple[str, str, str]] = []
    analyses = []
    for class_dir in class_dirs:
        analysis = analyze_class(
            class_dir,
            args.max_files_per_class,
            args.sample_size,
            args.sequence_points,
            args.workers,
            rng,
            errors,
            args.no_header_order,
        )
        if analysis.moments.count and len(analysis.sample):
            analyses.append(analysis)
            print(
                f"型号 {analysis.name}: 文件 {analysis.files_read}/{analysis.files_selected}，"
                f"有效PDW点 {analysis.moments.count}"
            )
        else:
            print(f"警告: 型号 {analysis.name} 没有可分析的有效数据")
    if not analyses:
        raise ValueError("No valid PDW data was read")

    statistics_path = output_dir / "class_feature_statistics.csv"
    distances_path = output_dir / "pairwise_distribution_distance.csv"
    write_statistics(analyses, statistics_path)
    distance_matrix, distance_rows = distribution_distances(analyses, args.bins)
    write_distances(distance_rows, distances_path)
    write_errors(errors, output_dir / "read_errors.csv")
    plot_sequences(analyses, output_dir / "pdw_sequence_comparison.png")
    plot_distribution_heatmaps(analyses, args.bins, output_dir / "feature_distribution_heatmaps.png")
    plot_boxplots(analyses, output_dir / "feature_boxplots.png")
    plot_distance_heatmap(analyses, distance_matrix, output_dir / "distribution_distance_heatmap.png")

    print(f"分析完成，结果保存至: {output_dir.resolve()}")
    print("mean/std/min/max为所分析文件的精确统计；sample_q*和图形基于每型号均匀抽样点。")


if __name__ == "__main__":
    main()
