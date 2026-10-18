from __future__ import annotations

import time
import csv
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset, WeightedRandomSampler

from pulsecls.cfg import normalize_dataset_path, remap_repo_path
from pulsecls.data.cache import cache_directory
from pulsecls.data.sources import input_config, require_class_folders
from pulsecls.utils.progress import Progress


def _sort_class_dir(path: Path) -> tuple[int, str]:
    try:
        return (0, f"{int(path.name):06d}")
    except ValueError:
        return (1, path.name)


def find_samples(split_dir: Path) -> list[tuple[Path, int]]:
    if split_dir.suffix == ".json" and split_dir.is_file():
        manifest = json.loads(split_dir.read_text(encoding="utf-8"))
        root = (split_dir.parent / manifest.get("root", ".")).resolve()
        records = []
        for row in manifest["samples"]:
            path = (root / row["path"]).resolve()
            if not path.is_relative_to(root) or path.suffix != ".csv" or not path.is_file():
                raise ValueError(f"Invalid sample path in {split_dir}: {row['path']}")
            label = int(row["label"])
            if label < 0 or path.parent.name != str(label):
                raise ValueError(f"Sample class directory does not match label: {path}")
            records.append((path, label))
        if not records:
            raise FileNotFoundError(f"No samples in {split_dir}")
        if len({path for path, _ in records}) != len(records):
            raise ValueError(f"Duplicate sample paths in {split_dir}")
        return records
    records: list[tuple[Path, int]] = []
    class_dirs = sorted(
        [path for path in split_dir.iterdir() if path.is_dir() and path.name.isdecimal()],
        key=_sort_class_dir,
    )
    for class_dir in class_dirs:
        label = int(class_dir.name)
        records.extend(
            (path, label)
            for path in sorted(class_dir.glob("*.csv"), key=lambda p: (0, int(p.stem)) if p.stem.isdecimal() else (1, p.name))
        )
    if not records:
        raise FileNotFoundError(f"No sample CSV files found in {split_dir}")
    return records


DEFAULT_FEATURES = ("CF_MHz", "PW_us", "PRI_us")


def resize_sequence(sample: np.ndarray, length: int, length_policy: dict | None = None) -> np.ndarray:
    """Crop long streams; append raw zeros or cyclic repeats to short streams."""
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 1:
        raise ValueError("sequence_length must be a positive integer")
    if sample.ndim != 2 or len(sample) == 0:
        raise ValueError("Cannot resize an empty or non-matrix sequence")
    policy = {"short": "error", "long": "error", **(length_policy or {})}
    if set(policy) - {"short", "long"} or policy["short"] not in {"error", "zero", "repeat"} or policy["long"] not in {"error", "head", "center"}:
        raise ValueError(f"Invalid length_policy: {policy}")
    size = len(sample)
    if size == length:
        return sample
    if size > length:
        if policy["long"] == "error":
            raise ValueError(f"Sequence has {size} pulses; expected {length}")
        start = (size - length) // 2 if policy["long"] == "center" else 0
        return sample[start:start + length]
    if policy["short"] == "error":
        raise ValueError(f"Sequence has {size} pulses; expected {length}")
    if policy["short"] == "repeat":
        return np.tile(sample, (int(np.ceil(length / size)), 1))[:length]
    return np.pad(sample, ((0, length - size), (0, 0)))


def zero_padding_enabled(data_cfg: dict) -> bool:
    return data_cfg.get("length_policy", {}).get("short") == "zero"


def validate_label_range(labels, num_classes, context):
    """Validate metadata before loading large arrays or launching accelerator kernels."""
    if num_classes is None or not len(labels):
        return
    invalid = (labels < 0) | (labels >= int(num_classes))
    if invalid.any():
        label = int(labels[np.flatnonzero(invalid)[0]])
        raise ValueError(
            f"Class label {label} in {context} is outside num_classes={num_classes} "
            f"(valid labels: 0..{int(num_classes) - 1}). Adjust train.num_classes or select the matching dataset; "
            "changing the class count does not filter dataset labels."
        )


def feature_column_indices(header, features, csv_path):
    """Resolve the same column aliases for conversion and training."""
    aliases = {"CF_MHz": ("CF_MHz", "载频(MHz)"), "PW_us": ("PW_us", "脉宽(us)"),
               "PRI_us": ("PRI_us", "脉冲重复周期(us)", "PRI"),
               "脉冲到达时间": ("脉冲到达时间", "脉冲到达时间(ms)", "TOA"),
               "脉冲到达时间(ms)": ("脉冲到达时间(ms)", "脉冲到达时间", "TOA")}
    if header is None:
        raise ValueError(f"{csv_path} is an empty CSV")
    indices = []
    for feature in features:
        match = next((name for name in aliases.get(feature, (feature,)) if name in header), None)
        if match is None:
            raise ValueError(f"{csv_path} missing feature {feature!r}; columns: {header}. "
                             "Expected CF/PW/PRI columns for three-channel inputs.")
        indices.append(header.index(match))
    return indices


def read_sequence_csv(csv_path: Path,
                      features: list[str] | tuple[str, ...] = DEFAULT_FEATURES) -> np.ndarray:
    """Read raw feature columns, preserving sequence length and physical units."""
    csv_path = Path(csv_path)
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        indices = feature_column_indices(next(csv.reader(stream), None), features, csv_path)
        sample = np.loadtxt(stream, delimiter=",", usecols=tuple(indices), dtype=np.float64, ndmin=2)
    sample = np.asarray(sample, dtype=np.float64)
    if sample.ndim != 2 or sample.shape[1] != len(features) or not len(sample):
        raise ValueError(
            f"{csv_path} has shape {sample.shape}; "
            f"expected a nonempty sequence with {len(features)} feature columns"
        )
    if not np.isfinite(sample).all():
        raise ValueError(f"{csv_path} contains NaN or infinite values")
    return sample


def load_sample(csv_path: Path, sequence_length: int,
                features: list[str] | tuple[str, ...] = DEFAULT_FEATURES,
                length_policy: dict | None = None) -> np.ndarray:
    sample = read_sequence_csv(csv_path, features)
    # Crop before relative-TOA anchoring; pad only after subtracting the origin.
    if len(sample) >= sequence_length:
        sample = resize_sequence(sample, sequence_length, length_policy)
    # Legacy datasets explicitly requesting TOA retain origin anchoring; PRI is never anchored.
    for index, feature in enumerate(features):
        if feature in ("脉冲到达时间", "脉冲到达时间(ms)", "TOA"):
            sample[:, index] -= sample[0, index]
    sample = resize_sequence(sample, sequence_length, length_policy)
    return sample.astype(np.float32)


def load_split_arrays(
    data_cfg: dict,
    split_key: str,
    *, selection_plan: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    require_class_folders(data_cfg)
    root = remap_repo_path(data_cfg["root"])
    selection_info = None
    if "selection" in data_cfg:
        from pulsecls.data.selection import select_records, selection_config
        if split_key not in {"train", "val", "all"}:
            raise ValueError("Pipeline dataset sampling supports train/val only; no test split")
        partitions, selection_info = selection_plan if selection_plan is not None else select_records(data_cfg)
        if selection_info["settings"] != selection_config(data_cfg["selection"]) or selection_info.get("data_root") != str(root):
            raise ValueError("Selection plan does not match the current dataset/settings")
        if selection_info.get("input", input_config()) != input_config(data_cfg.get("input")):
            raise ValueError("Selection plan does not match the current data input format/options")
        records = partitions[split_key]
        split_name = f"pipeline:{split_key}"
    else:
        split_name = normalize_dataset_path(data_cfg[split_key])
        records = None
    sequence_length = int(data_cfg["sequence_length"])
    sliced_length = data_cfg.get("slicing", {}).get("length")
    if sliced_length is not None and sequence_length != sliced_length:
        raise ValueError(f"Sliced data has length {sliced_length}; set sequence_length={sliced_length} in the entry")
    cache_dir = cache_directory(root, data_cfg)
    features = data_cfg.get("features", list(DEFAULT_FEATURES[:2]))
    length_policy = data_cfg.get("length_policy")
    signature = hashlib.sha256(json.dumps([features, length_policy, selection_info and selection_info["settings"], selection_info and selection_info["source_fingerprint"], 6], ensure_ascii=True, sort_keys=True).encode()).hexdigest()[:12]
    filename = (f"{split_key}_seq{sequence_length}_c{len(features)}_{signature}.npz"
                if "features" in data_cfg or length_policy or selection_info else f"{split_key}_seq{sequence_length}.npz")
    cache_file = cache_dir / filename if cache_dir is not None else None

    if cache_file is not None and cache_file.exists():
        with Progress(f"Read {split_key} cache", unit="steps", enabled=data_cfg.get("progress", True)):
            with np.load(cache_file, allow_pickle=False) as cached:
                labels = cached["labels"].astype(np.int64, copy=False)
                validate_label_range(labels, data_cfg.get("num_classes"), str(cache_file))
                data = cached["data"].astype(np.float32, copy=False)
                if data.ndim != 3 or data.shape[1:] != (sequence_length, len(features)):
                    raise ValueError(f"Incompatible feature cache: {cache_file}; regenerate this cache")
                paths = []
                for stored in cached["paths"].astype(str).tolist():
                    normalized = normalize_dataset_path(stored)
                    if normalized.lower().startswith("f:/competition/"):
                        paths.append(str(remap_repo_path(normalized)))
                    elif Path(normalized).is_absolute():
                        paths.append(normalized)
                    else:
                        paths.append(str(root / normalized))
        print(f"Loaded cache {cache_file}: {len(labels)} samples")
        return data, labels, paths

    if records is None:
        with Progress(f"Find {split_key} samples", unit="steps", enabled=data_cfg.get("progress", True)):
            records = find_samples(root / split_name)
    labels = np.fromiter((label for _, label in records), dtype=np.int64, count=len(records))
    validate_label_range(labels, data_cfg.get("num_classes"), split_name)
    data = np.empty((len(records), sequence_length, len(features)), dtype=np.float32)
    paths: list[str] = []

    started = time.perf_counter()
    with Progress(f"Load {split_key} CSV", len(records), enabled=data_cfg.get("progress", True)) as progress:
        for index, (path, label) in enumerate(records):
            if label < 0 or ("num_classes" in data_cfg and label >= int(data_cfg["num_classes"])):
                raise ValueError(f"Class label {label} outside num_classes={data_cfg['num_classes']}")
            data[index] = load_sample(path, sequence_length, features, length_policy)
            labels[index] = label
            paths.append(str(path))
            progress.update()
    print(
        f"Loaded {len(records)} samples from {split_name} "
        f"in {time.perf_counter() - started:.1f}s"
    )

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        relative_paths = [Path(path).relative_to(root).as_posix() for path in paths]
        temporary = cache_file.with_name(f".{cache_file.stem}.{uuid4().hex}.npz")
        try:
            with Progress(f"Write {split_key} cache", unit="steps", enabled=data_cfg.get("progress", True)):
                np.savez_compressed(temporary, data=data, labels=labels, paths=np.array(relative_paths),
                                    features=np.array(features))
                temporary.replace(cache_file)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Saved cache: {cache_file}")
    return data, labels, paths


def feature_stats(train_data: np.ndarray, ignore_zero_padding: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if ignore_zero_padding:
        valid = train_data[~np.all(train_data == 0, axis=-1)]
        if not len(valid):
            raise ValueError("No valid pulses for normalization")
        mean = valid.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = valid.std(axis=0, dtype=np.float64).astype(np.float32)
        return mean, np.maximum(std, 1e-6).astype(np.float32)
    mean = train_data.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = train_data.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6).astype(np.float32)


class PulseArrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        zero_padding: bool = False,
        normalize: bool = True,
        normalizer=None,
    ) -> None:
        channels = data.shape[-1]
        if mean.size != channels or std.size != channels:
            raise ValueError(f"Normalization has {mean.size} channels; data has {channels}")
        if not normalize:
            normalized = data
        elif normalizer is not None:
            normalized = normalizer.transform_numpy(data, zero_padding=zero_padding)
        else:
            normalized = (data - mean.reshape(1, 1, channels)) / std.reshape(1, 1, channels)
        if zero_padding and normalize:
            normalized[np.all(data == 0, axis=-1)] = 0
        self.features = torch.from_numpy(normalized.transpose(0, 2, 1).astype(np.float32, copy=True))
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.features[index], self.labels[index], index


class MergeLastSingletonBatchSampler(BatchSampler):
    """Keep every sampled sequence while avoiding a last batch of one for BatchNorm."""
    def __iter__(self):
        pending = None
        for batch in super().__iter__():
            if pending is not None:
                if len(batch) == 1:
                    yield pending + batch
                    pending = None
                    continue
                yield pending
            pending = batch
        if pending is not None:
            yield pending

    def __len__(self):
        count = super().__len__()
        return count - 1 if len(self.sampler) > 1 and len(self.sampler) % self.batch_size == 1 else count


def make_balanced_sampler(
    labels: np.ndarray,
    num_classes: int,
    power: float = 1.0,
) -> WeightedRandomSampler:
    """Sample with class weights proportional to count**(-power).

    ``power=0`` is ordinary random sampling, ``power=1`` is fully
    class-balanced sampling, and ``power=0.5`` is a milder square-root
    correction that is usually better behaved for long-tailed data.
    """
    if power < 0:
        raise ValueError(f"Sampler power must be non-negative, got {power}")
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.zeros_like(counts, dtype=np.float64)
    active = counts > 0
    weights[active] = np.power(counts[active], -float(power))
    sample_weights = weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def class_weights(
    labels: np.ndarray,
    num_classes: int,
    mode: str = "balanced",
) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes, dtype=np.float32)
    active = counts > 0
    if mode == "balanced":
        weights[active] = len(labels) / (active.sum() * counts[active])
    elif mode == "sqrt_balanced":
        weights[active] = np.sqrt(len(labels) / (active.sum() * counts[active]))
    elif mode == "log_balanced":
        weights[active] = np.log1p(len(labels) / counts[active])
    else:
        raise ValueError(
            f"Unknown class-weight mode {mode!r}; expected "
            "'balanced', 'sqrt_balanced', or 'log_balanced'."
        )
    if active.any():
        weights[active] /= weights[active].mean()
    return torch.from_numpy(weights)
