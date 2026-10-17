"""Sample whole sequences from the total pool, then split train/validation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from pulsecls.cfg import remap_repo_path
from pulsecls.data.file_index import load_inventory, record_at
from pulsecls.data.sources import input_config
from pulsecls.utils.progress import Progress

DEFAULT_SELECTION = {"sample_ratio": 1.0, "train_ratio": .8, "val_ratio": .2,
                     "seed": 2026, "stratified": True}


def selection_config(config):
    if not isinstance(config, dict) or set(config) - set(DEFAULT_SELECTION):
        raise ValueError("dataset must contain sample_ratio, train_ratio, val_ratio, seed, stratified")
    cfg = {**DEFAULT_SELECTION, **config}
    for key in ("sample_ratio", "train_ratio", "val_ratio"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"dataset.{key} must be a finite ratio")
    if not 0 < cfg["sample_ratio"] <= 1 or not 0 < cfg["train_ratio"] <= 1 or not 0 <= cfg["val_ratio"] < 1:
        raise ValueError("sample_ratio/train_ratio must be in (0, 1], val_ratio in [0, 1)")
    if not np.isclose(cfg["train_ratio"] + cfg["val_ratio"], 1, atol=1e-9, rtol=0):
        raise ValueError("dataset.train_ratio + dataset.val_ratio must equal 1")
    if isinstance(cfg["seed"], bool) or not isinstance(cfg["seed"], int) or cfg["seed"] < 0:
        raise ValueError("dataset.seed must be a nonnegative integer")
    if not isinstance(cfg["stratified"], bool):
        raise ValueError("dataset.stratified must be true/false")
    return cfg


def validation_enabled(data_cfg):
    if "selection" in data_cfg:
        return selection_config(data_cfg["selection"])["val_ratio"] > 0
    return bool(data_cfg.get("val", "val"))


def _allocate(capacity, budget, rng, minimum=0):
    """Largest-remainder allocation, bounded by per-class capacity, exact total."""
    capacity = np.asarray(capacity, dtype=int)
    if budget < 0 or budget > capacity.sum():
        raise ValueError("Insufficient per-class capacity for split")
    if not capacity.sum():
        return np.zeros_like(capacity)
    expected = budget * capacity / capacity.sum()
    counts = np.floor(expected).astype(int)
    tie_order = rng.permutation(len(capacity))
    order = np.lexsort((tie_order, -(expected - counts)))
    for i in order[:budget - counts.sum()]:
        counts[i] += 1
    lower = np.minimum(capacity, minimum)
    if lower.sum() <= budget:
        for i in np.flatnonzero(counts < lower):
            while counts[i] < lower[i]:
                donor = int(np.argmax(counts - lower))
                counts[donor] -= 1
                counts[i] += 1
    return counts


def select_records(data_cfg):
    cfg = selection_config(data_cfg["selection"])
    root = remap_repo_path(data_cfg["root"])
    with Progress("Read dataset index", unit="steps", enabled=data_cfg.get("progress", True)):
        inventory, inventory_fingerprint, index_source = load_inventory(root, data_cfg)
    total_count = sum(group["count"] for group in inventory)
    sample_count = int(np.floor(total_count * cfg["sample_ratio"] + 1e-9))
    if sample_count == 0:
        raise ValueError("dataset.sample_ratio selects zero sequences; increase it")
    with Progress("Sample and split sequences", unit="steps", enabled=data_cfg.get("progress", True)):
        sample_rng = np.random.default_rng(np.random.SeedSequence(cfg["seed"], spawn_key=(0,)))
        split_rng = np.random.default_rng(np.random.SeedSequence(cfg["seed"], spawn_key=(1,)))
        if cfg["stratified"]:
            labels = sorted({group["label"] for group in inventory})
            groups = {label: [g for g in inventory if g["label"] == label] for label in labels}
            capacities = np.array([sum(g["count"] for g in groups[label]) for label in labels])
            minimum = 2 if cfg["val_ratio"] > 0 and sample_count >= 2 * len(labels) else 1
            counts = _allocate(capacities, sample_count, sample_rng, minimum)
            sampled_groups = [[record_at(groups[label], int(i)) for i in sample_rng.choice(int(capacity), size=int(count), replace=False)]
                              for label, capacity, count in zip(labels, capacities, counts)]
            sampled = [row for rows in sampled_groups for row in rows]
        else:
            indices = sample_rng.choice(total_count, size=sample_count, replace=False)
            offsets = np.r_[0, np.cumsum([g["count"] for g in inventory])]
            group_indices = np.searchsorted(offsets[1:], indices, side="right")
            sampled = [record_at([inventory[int(group)]], int(index - offsets[group])) for index, group in zip(indices, group_indices)]
        train_count = sample_count if cfg["val_ratio"] == 0 else int(np.floor(sample_count * cfg["train_ratio"] + 1e-9))
        val_count = sample_count - train_count
        if train_count == 0 or (cfg["val_ratio"] > 0 and val_count == 0):
            raise ValueError("Too few sampled sequences for requested train/val ratios")
        if cfg["stratified"]:
            sizes = np.array([len(rows) for rows in sampled_groups])
            active = np.count_nonzero(sizes)
            if train_count < active:
                raise ValueError("Train split cannot cover sampled classes; increase sample_ratio or train_ratio")
            capacity = sizes - 1 if val_count >= active and (sizes[sizes > 0] >= 2).all() else sizes
            capacity = np.maximum(capacity, 0)
            train_sizes = _allocate(capacity, train_count, split_rng, minimum=1)
            train, val = [], []
            for rows, size in zip(sampled_groups, train_sizes):
                order = split_rng.permutation(len(rows))
                train.extend(rows[i] for i in order[:size])
                val.extend(rows[i] for i in order[size:])
        else:
            order = split_rng.permutation(sample_count)
            train = [sampled[i] for i in order[:train_count]]
            val = [sampled[i] for i in order[train_count:]]
    source_digest = hashlib.sha256()
    source_digest.update(inventory_fingerprint.encode())
    with Progress("Check sampled files", len(sampled), unit="file", enabled=data_cfg.get("progress", True)) as progress:
        for path, label in sampled:
            try:
                stat = path.stat()
            except FileNotFoundError as error:
                raise FileNotFoundError(f"Selected CSV is missing: {path}; finish regenerating the dataset before training") from error
            source_digest.update(f"{path.relative_to(root).as_posix()}:{label}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
            progress.update()
    info = {"settings": cfg, "input": input_config(data_cfg.get("input")),
            "total_count": total_count, "sampled_count": sample_count,
            "train_count": len(train), "val_count": len(val),
            "source_fingerprint": source_digest.hexdigest(), "index_source": index_source,
            "algorithm_version": 2, "data_root": str(root),
            "sampled": [{"path": p.relative_to(root).as_posix(), "label": y} for p, y in sampled],
            "train": [{"path": p.relative_to(root).as_posix(), "label": y} for p, y in train],
            "val": [{"path": p.relative_to(root).as_posix(), "label": y} for p, y in val]}
    return {"train": train, "val": val, "all": sampled}, info
