"""Dataset utilities for CSV or HDF5 slices produced by ``cut_data.py``.

The loader keeps source-file groups so training can score individual slices,
while validation and test code can aggregate all slices from one source signal.
Normalization statistics are fitted on the training split only.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset

from data_process.h5_storage import find_h5_file
from data_process.csv_features import FEATURE_COLS, read_feature_csv


SPLIT_NAMES = ("train", "val", "valid", "validation", "test")
_SLICE_SUFFIX = re.compile(r"_slice\d+$", re.IGNORECASE)


def _natural_path_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", str(path).replace("\\", "/"))
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts)


@dataclass(frozen=True)
class SliceRecord:
    path: Path
    group: str
    raw_label: str


@dataclass(frozen=True)
class NormalizationStats:
    method: str
    center: np.ndarray
    scale: np.ndarray

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return data.astype(np.float32, copy=False)
        return ((data - self.center[None, :, None]) / self.scale[None, :, None]).astype(
            np.float32, copy=False
        )


def _natural_label_key(value: str) -> tuple[int, object]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _metadata(path: Path, root: Path) -> tuple[str, str]:
    relative = path.relative_to(root)
    if len(relative.parts) < 2:
        raise ValueError(f"Slice CSV must be inside a class directory: {path}")
    raw_label = relative.parts[0]

    # New slices encode the original source stem in the filename:
    # <source_stem>_slice0000.csv. Keep discovery path-only so loading
    # metadata does not open every CSV before the actual data pass.
    source = _SLICE_SUFFIX.sub("", path.stem) or path.stem

    # Prefix the voting group with the folder label so same-named source files
    # in different classes can never be merged into one group.
    return f"{raw_label}/{source}", raw_label

def discover_records(root: str | Path) -> list[SliceRecord]:
    root = Path(root)
    paths = sorted((path for path in root.rglob("*.csv") if path.name != "_index.csv"), key=_natural_path_key)
    if not paths:
        raise FileNotFoundError(f"No slice CSV files found below {root}")
    return [SliceRecord(path, *_metadata(path, root)) for path in paths]


def _split_records(records: list[SliceRecord], train_ratio: float, seed: int) -> tuple[list[SliceRecord], list[SliceRecord], list[SliceRecord]]:
    if not 0 < train_ratio <= 1:
        raise ValueError("train_ratio must be in (0, 1]")
    grouped: dict[str, list[SliceRecord]] = {}
    for record in records:
        grouped.setdefault(record.group, []).append(record)

    rng = np.random.default_rng(seed)
    by_label: dict[str, list[str]] = {}
    for group, group_records in grouped.items():
        labels = {record.raw_label for record in group_records}
        if len(labels) != 1:
            raise ValueError(f"Group {group!r} contains multiple labels: {sorted(labels)}")
        by_label.setdefault(next(iter(labels)), []).append(group)

    train_groups: set[str] = set()
    val_groups: set[str] = set()
    test_groups: set[str] = set()
    for label_groups in by_label.values():
        rng.shuffle(label_groups)
        count = len(label_groups)
        train_count = count if train_ratio >= 1 else max(1, int(math.floor(count * train_ratio)))
        remaining = count - train_count
        val_count = 0 if remaining == 0 else max(1, remaining // 2)
        train_groups.update(label_groups[:train_count])
        val_groups.update(label_groups[train_count:train_count + val_count])
        test_groups.update(label_groups[train_count + val_count:])

    def collect(groups: set[str]) -> list[SliceRecord]:
        return [record for record in records if record.group in groups]

    return collect(train_groups), collect(val_groups), collect(test_groups)


def _find_split_root(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def resolve_splits(root: str | Path, train_ratio: float, seed: int) -> tuple[list[SliceRecord], list[SliceRecord], list[SliceRecord]]:
    root = Path(root)
    train_root = _find_split_root(root, ("train",))
    val_root = _find_split_root(root, ("val", "valid", "validation"))
    test_root = _find_split_root(root, ("test",))
    if train_root and val_root and test_root:
        return discover_records(train_root), discover_records(val_root), discover_records(test_root)
    return _split_records(discover_records(root), train_ratio, seed)


def _resize(sample: np.ndarray, length: int) -> np.ndarray:
    if sample.shape[0] == length:
        return sample
    if sample.shape[0] > length:
        return sample[:length]
    repeats = (length + sample.shape[0] - 1) // sample.shape[0]
    return np.tile(sample, (repeats, 1))[:length]


def load_records(
    records: Iterable[SliceRecord],
    sequence_length: int,
    no_header_order: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = list(records)
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    data = np.empty((len(records), len(FEATURE_COLS), sequence_length), dtype=np.float32)
    groups = np.empty(len(records), dtype=object)
    labels = np.empty(len(records), dtype=np.int64)
    label_values = sorted({record.raw_label for record in records}, key=_natural_label_key)
    label_map = {value: index for index, value in enumerate(label_values)}
    for index, record in enumerate(tqdm(records,desc='Loading data...')):
        frame = read_feature_csv(record.path, no_header_order=no_header_order).features
        values = frame.to_numpy(dtype=np.float32, copy=False)
        if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
            raise ValueError(f"Invalid numeric data in {record.path}")
        values = _resize(values, sequence_length)
        data[index] = values.T
        groups[index] = record.group
        labels[index] = label_map[record.raw_label]
    return data, labels, groups


def load_h5_records(
    path: str | Path,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bulk-load one H5 split as data, raw string labels, and source groups."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as handle:
        required = {"data", "raw_labels", "groups"}
        missing = required - set(handle.keys())
        if missing:
            raise ValueError(f"{path} is missing H5 datasets: {sorted(missing)}")
        data = np.asarray(handle["data"], dtype=np.float32)
        raw_labels = np.asarray(handle["raw_labels"].asstr()[:], dtype=object)
        groups = np.asarray(handle["groups"].asstr()[:], dtype=object)
        stored_columns = handle.attrs.get("feature_cols")

    if data.ndim != 3 or data.shape[1] != len(FEATURE_COLS):
        raise ValueError(
            f"Expected H5 data [samples, {len(FEATURE_COLS)}, length], got {data.shape}"
        )
    if len(raw_labels) != len(data) or len(groups) != len(data):
        raise ValueError(f"H5 metadata length does not match data in {path}")
    if stored_columns is not None:
        columns = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in stored_columns
        )
        if columns != FEATURE_COLS:
            raise ValueError(
                f"H5 feature columns {columns} do not match expected {FEATURE_COLS}"
            )
    if data.shape[2] > sequence_length:
        data = data[:, :, :sequence_length]
    elif data.shape[2] < sequence_length:
        repeats = (sequence_length + data.shape[2] - 1) // data.shape[2]
        data = np.tile(data, (1, 1, repeats))[:, :, :sequence_length]
    if data.size and not np.isfinite(data).all():
        raise ValueError(f"H5 data contains non-finite values: {path}")
    return data, raw_labels, groups


def load_cut_source(
    root: str | Path,
    sequence_length: int,
    data_format: str = "auto",
    no_header_order: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one split using auto, H5, or legacy CSV format."""
    data_format = str(data_format).lower()
    if data_format not in {"auto", "h5", "csv"}:
        raise ValueError("data_format must be one of: auto, h5, csv")
    h5_path = find_h5_file(root)
    if data_format in {"auto", "h5"} and h5_path is not None:
        print(f"Loading H5 slices: {h5_path}")
        return load_h5_records(h5_path, sequence_length)
    if data_format == "h5":
        raise FileNotFoundError(f"No H5 slice file found below {root}")

    records = discover_records(root)
    data, _, groups = load_records(records, sequence_length, no_header_order)
    raw_labels = np.asarray([record.raw_label for record in records], dtype=object)
    return data, raw_labels, groups


def fit_normalization(train_data: np.ndarray, method: str = "standard") -> NormalizationStats:
    """Fit one center and scale per feature channel using training data only."""
    method = str(method).lower()
    if method not in {"none", "standard", "minmax"}:
        raise ValueError("normalization must be one of: none, standard, minmax")
    if train_data.ndim != 3:
        raise ValueError(f"Expected [samples, channels, sequence], got {train_data.shape}")
    if method == "none":
        return NormalizationStats(method, np.zeros(train_data.shape[1], dtype=np.float32), np.ones(train_data.shape[1], dtype=np.float32))
    flat = train_data.transpose(0, 2, 1).reshape(-1, train_data.shape[1]).astype(np.float64, copy=False)
    if method == "standard":
        center = flat.mean(axis=0)
        scale = flat.std(axis=0)
    else:
        center = flat.min(axis=0)
        scale = flat.max(axis=0) - center
    scale = np.where(scale < 1e-6, 1.0, scale)
    return NormalizationStats(method, center.astype(np.float32), scale.astype(np.float32))


def load_cut_splits(
    root: str | Path,
    sequence_length: int = 128,
    train_ratio: float = 0.8,
    seed: int = 2026,
    normalization: str = "standard",
    data_format: str = "auto",
    no_header_order: str | None = None,
) -> dict[str, object]:
    root = Path(root)
    train_root = _find_split_root(root, ("train",))
    val_root = _find_split_root(root, ("val", "valid", "validation"))
    test_root = _find_split_root(root, ("test",))
    if train_root and val_root and test_root:
        train_data, train_raw_labels, train_groups = load_cut_source(train_root, sequence_length, data_format, no_header_order)
        val_data, val_raw_labels, val_groups = load_cut_source(val_root, sequence_length, data_format, no_header_order)
        test_data, test_raw_labels, test_groups = load_cut_source(test_root, sequence_length, data_format, no_header_order)
    else:
        if str(data_format).lower() == "h5":
            raise FileNotFoundError(f"H5 mode requires train/val/test split directories below {root}")
        train_records, val_records, test_records = resolve_splits(root, train_ratio, seed)

        def load_csv(records: list[SliceRecord]):
            data, _, groups = load_records(records, sequence_length, no_header_order)
            labels = np.asarray([record.raw_label for record in records], dtype=object)
            return data, labels, groups

        train_data, train_raw_labels, train_groups = load_csv(train_records)
        val_data, val_raw_labels, val_groups = load_csv(val_records)
        test_data, test_raw_labels, test_groups = load_csv(test_records)

    raw_labels = sorted(
        set(train_raw_labels) | set(val_raw_labels) | set(test_raw_labels),
        key=_natural_label_key,
    )
    label_map = {value: index for index, value in enumerate(raw_labels)}
    train_labels = np.asarray([label_map[value] for value in train_raw_labels], dtype=np.int64)
    val_labels = np.asarray([label_map[value] for value in val_raw_labels], dtype=np.int64)
    test_labels = np.asarray([label_map[value] for value in test_raw_labels], dtype=np.int64)
    stats = fit_normalization(train_data, normalization)
    return {
        "train": (stats.transform(train_data), train_labels, train_groups),
        "val": (stats.transform(val_data), val_labels, val_groups),
        "test": (stats.transform(test_data), test_labels, test_groups),
        "class_names": raw_labels,
        "stats": stats,
    }


def load_cut_test(
    root: str | Path,
    sequence_length: int = 128,
    class_names: list[str] | tuple[str, ...] | None = None,
    normalization_stats: NormalizationStats | None = None,
    normalization: str = "standard",
    data_format: str = "auto",
    no_header_order: str | None = None,
) -> dict[str, object]:
    """Load one test directory without attempting any train/val split."""
    data, raw_labels, groups = load_cut_source(root, sequence_length, data_format, no_header_order)
    inferred_names = sorted(set(raw_labels), key=_natural_label_key)
    names = list(class_names) if class_names is not None else inferred_names
    label_map = {str(name): index for index, name in enumerate(names)}
    unknown = sorted(
        {str(value) for value in raw_labels if str(value) not in label_map},
        key=_natural_label_key,
    )
    if unknown:
        raise ValueError(f"Test data contains labels absent from checkpoint/classes: {unknown}")
    labels = np.asarray([label_map[str(value)] for value in raw_labels], dtype=np.int64)
    stats = normalization_stats or fit_normalization(data, normalization)
    return {
        "test": (stats.transform(data), labels, groups),
        "class_names": names,
        "stats": stats,
    }

class SliceDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, groups: np.ndarray, transform=None):
        self.data = torch.from_numpy(np.asarray(data, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.groups = np.asarray(groups, dtype=object)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        sample = self.data[index].clone()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index], str(self.groups[index])
