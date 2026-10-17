"""Flat CSV folder input, optional filename-keyed labels, and lazy preprocessing."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pulsecls.cfg import remap_repo_path
from pulsecls.data.dataset import load_sample, zero_padding_enabled
from pulsecls.utils.normalization import FeatureNormalizer
from pulsecls.utils.progress import Progress


def find_inference_csvs(source, *, progress=True):
    root = remap_repo_path(source)
    if not root.is_dir():
        raise FileNotFoundError(f"Inference folder does not exist: {root}")
    with Progress("Find inference CSVs", unit="steps", enabled=progress):
        paths = sorted((p for p in root.iterdir() if p.suffix.lower() == ".csv" and p.is_file()),
                       key=lambda p: (0, int(p.stem), p.name) if p.stem.isdecimal() else (1, 0, p.name))
    if not paths:
        raise FileNotFoundError(f"No CSV sequences directly inside {root}")
    return paths


def read_labels(labels_file, paths, num_classes):
    """TXT rows: filename.csv<TAB>integer_label; row order is immaterial."""
    if labels_file is None or str(labels_file).strip() == "":
        return None
    file = remap_repo_path(labels_file)
    mapping = {}
    for number, line in enumerate(file.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.rsplit(None, 1)
        if len(fields) != 2:
            raise ValueError(f"{file}:{number}: expected filename.csv and integer label")
        name, value = fields
        if Path(name).name != name or name in mapping:
            raise ValueError(f"Duplicate or invalid filename in labels: {name}")
        try:
            label = int(value)
        except ValueError as error:
            raise ValueError(f"Invalid label at {file}:{number}") from error
        if not 0 <= label < num_classes:
            raise ValueError(f"Label {label} outside num_classes={num_classes}")
        mapping[name] = label
    names = {p.name for p in paths}
    if set(mapping) != names:
        missing = sorted(names - set(mapping))[:5]
        extra = sorted(set(mapping) - names)[:5]
        raise ValueError(f"Labels must match input CSVs exactly: missing={missing}, extra={extra}")
    return np.array([mapping[p.name] for p in paths], dtype=np.int64)


class InferenceCSVDataset(Dataset):
    """Read each sequence when its batch is needed, using training normalization."""
    def __init__(self, paths, data_cfg, mean, std, *, normalizer=None):
        self.paths, self.data_cfg = paths, data_cfg
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.on_loaded = None
        if self.mean.shape != (len(data_cfg["features"]),) or self.std.shape != self.mean.shape or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Checkpoint normalization must match the feature count, with finite positive std")
        self.normalizer = normalizer if normalizer is not None else FeatureNormalizer.from_checkpoint(
            {"mean": self.mean, "std": self.std}, len(data_cfg["features"]))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        cfg = self.data_cfg
        raw = load_sample(self.paths[index], int(cfg["sequence_length"]), cfg["features"],
                          length_policy=cfg.get("length_policy"))
        values = self.normalizer.transform_numpy(raw, zero_padding=zero_padding_enabled(cfg))
        if self.on_loaded is not None:
            self.on_loaded()
        return torch.from_numpy(values.T.copy()), torch.tensor(0), index
