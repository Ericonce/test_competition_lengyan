"""Small class inventories: use generated counts or cache ordinary file names."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from pulsecls.cfg import load_yaml


def _class_dirs(directory):
    if not directory.is_dir():
        return []
    return sorted((Path(e.path) for e in os.scandir(directory) if e.name.isdecimal() and e.is_dir()), key=lambda p: int(p.name))


def _stamp(path):
    stat = path.stat()
    return [str(path), stat.st_mtime_ns, stat.st_size]


def load_inventory(root, data_cfg):
    """Do not open sequence CSVs. Numbered synthetic files need no enumeration."""
    from pulsecls.data.sources import input_config, load_source_inventory
    if input_config(data_cfg.get("input"))["format"] != "class_folders":
        return load_source_inventory(root, data_cfg)
    root = Path(root).resolve()
    explicit = data_cfg.get("all")
    direct = _class_dirs(root)
    manifests = []
    # A root with numeric classes is a total pool, even if old split JSONs vanished.
    if explicit is None and direct:
        directories = direct
    elif explicit is not None:
        source = root / explicit
        if source.is_file() and source.suffix == ".json":
            directories, manifests = [], [source]
        else:
            directories = _class_dirs(source)
    else:
        directories = []
        for name in ("train", "val", "test"):
            if not data_cfg.get(name):
                continue
            source = root / data_cfg[name]
            if source.is_file() and source.suffix == ".json":
                manifests.append(source)
            elif source.is_dir():
                directories.extend(_class_dirs(source))
            else:
                raise FileNotFoundError(f"Dataset source does not exist: {source}; point data.yaml.all at the total class directory")
    directories = sorted(set(p.resolve() for p in directories), key=lambda p: (int(p.name), str(p)))
    manifests = sorted(set(p.resolve() for p in manifests))
    metadata = [root / name for name in ("generation_config.yaml", "summary.json", "emitters.json", "dataset_index.json") if (root / name).is_file()]
    stamps = [_stamp(p) for p in directories + manifests + metadata]
    cache_dir = root / ".pulsecls_cache"
    source_key = hashlib.sha256(json.dumps([str(p) for p in directories + manifests]).encode()).hexdigest()[:12]
    cache_path = cache_dir / f"file_index_{source_key}.json"
    cached = None
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("version") == 1 and cached.get("stamps") == stamps:
                return cached["classes"], cached["fingerprint"], "cached file index"
        except (ValueError, OSError):
            cached = None
    classes = None
    gen_path, summary_path = root / "generation_config.yaml", root / "summary.json"
    if directories == direct and not manifests and gen_path.is_file() and summary_path.is_file():
        # A changed class directory with unchanged metadata means files were edited.
        old_meta = {row[0]: row[1:] for row in (cached or {}).get("stamps", []) if row[0] in {str(p) for p in metadata}}
        new_meta = {str(p): _stamp(p)[1:] for p in metadata}
        edited = cached is not None and old_meta == new_meta
        if not edited:
            config, summary = load_yaml(gen_path), json.loads(summary_path.read_text(encoding="utf-8"))
            count = config.get("num_emitters", 0)
            sizes = config.get("sequences_per_emitter", 0)
            sizes = sizes if isinstance(sizes, list) else [sizes] * count
            if len(sizes) == count and [int(p.name) for p in directories] == list(range(count)) and sum(sizes) == summary.get("num_sequences"):
                classes = [{"label": i, "directory": str(root / str(i)), "count": int(n), "names": None} for i, n in enumerate(sizes)]
    origin = "generator counts" if classes is not None else "directory names (first/index-refresh scan)"
    if classes is None:
        groups = {}
        for directory in directories:
            names = sorted(e.name for e in os.scandir(directory) if e.name.lower().endswith(".csv") and e.is_file())
            groups[(int(directory.name), str(directory))] = set(names)
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            base = (manifest_path.parent / manifest.get("root", ".")).resolve()
            for row in manifest["samples"]:
                relative = Path(row["path"])
                if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".csv":
                    raise ValueError(f"Invalid sample path in {manifest_path}: {relative}")
                directory = (base / relative.parent).resolve()
                label = int(row["label"])
                if directory.name != str(label):
                    raise ValueError(f"Class directory does not match label: {relative}")
                groups.setdefault((label, str(directory)), set()).add(relative.name)
        classes = [{"label": label, "directory": directory, "count": len(names), "names": sorted(names)}
                   for (label, directory), names in sorted(groups.items()) if names]
    if not classes or not sum(group["count"] for group in classes):
        raise FileNotFoundError(f"No CSV sequences found under dataset root: {root}")
    fingerprint = hashlib.sha256(json.dumps([stamps, classes], sort_keys=True).encode()).hexdigest()
    payload = {"version": 1, "stamps": stamps, "classes": classes, "fingerprint": fingerprint}
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return classes, fingerprint, origin


def record_at(groups, index):
    """Materialize a path only after its integer index has been sampled."""
    for group in groups:
        if index < group["count"]:
            if "records" in group:
                row = group["records"][index]
                if "table" in group:
                    from pulsecls.data.sources import TableSample
                    return TableSample(Path(group["directory"]) / group["table"], row["id"],
                                       tuple(tuple(span) for span in row["spans"]), group["delimiter"],
                                       group["encoding"], group["order_column"]), group["label"]
                return Path(group["directory"]) / row["path"], group["label"]
            name = f"{index}.csv" if group["names"] is None else group["names"][index]
            return Path(group["directory"]) / name, group["label"]
        index -= group["count"]
    raise IndexError("Sequence index exceeds class inventory")
