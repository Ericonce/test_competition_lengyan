"""Selectable labeled inputs without copying sequences into class directories."""
from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import quote
from uuid import uuid4


INPUT_FORMATS = ("class_folders", "label_file", "filename", "table")
_DEFAULTS = {"format": "class_folders", "files": ".", "pattern": "*.csv",
             "labels_file": "labels.csv", "filename_column": "filename", "label_column": "label",
             "header": True, "delimiter": ",", "encoding": "utf-8-sig",
             "filename_regex": r"class(?P<label>\d+)_.*\.csv", "label_offset": 0, "label_map": {},
             "table_file": "samples.csv", "sample_id_columns": ["sample_id"], "order_column": None}


def input_config(options=None):
    if isinstance(options, str):
        options = {"format": options}
    if options is not None and not isinstance(options, dict):
        raise TypeError("data_input / data.input must be a format name or mapping")
    cfg = {**deepcopy(_DEFAULTS), **(options or {})}
    if set(cfg) - set(_DEFAULTS):
        raise ValueError(f"Unknown data input options: {sorted(set(cfg) - set(_DEFAULTS))}")
    if cfg["format"] not in INPUT_FORMATS:
        raise ValueError(f"input.format must be one of {INPUT_FORMATS}")
    for key in ("files", "pattern", "labels_file", "filename_column", "label_column", "encoding", "table_file"):
        if not isinstance(cfg[key], str) or not cfg[key]:
            raise ValueError(f"input.{key} must be a nonempty string")
    if not isinstance(cfg["header"], bool):
        raise ValueError("input.header must be true/false")
    if cfg["delimiter"] != "whitespace" and (not isinstance(cfg["delimiter"], str) or len(cfg["delimiter"]) != 1):
        raise ValueError("input.delimiter must be one character, or 'whitespace' for a label TXT")
    if cfg["format"] == "table" and (not cfg["header"] or cfg["delimiter"] == "whitespace"):
        raise ValueError("Table input requires a CSV/TSV header and a one-character delimiter")
    if isinstance(cfg["label_offset"], bool) or not isinstance(cfg["label_offset"], int):
        raise ValueError("input.label_offset must be an integer")
    if not isinstance(cfg["label_map"], dict) or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                                                   for v in cfg["label_map"].values()):
        raise ValueError("input.label_map must map original labels to nonnegative integer class IDs")
    if len({str(key) for key in cfg["label_map"]}) != len(cfg["label_map"]):
        raise ValueError("input.label_map contains duplicate original label names")
    cfg["label_map"] = {str(k): v for k, v in cfg["label_map"].items()}
    ids = cfg["sample_id_columns"]
    if not isinstance(ids, list) or not ids or any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("input.sample_id_columns must be a nonempty list of distinct column names")
    if cfg["order_column"] is not None and (not isinstance(cfg["order_column"], str) or not cfg["order_column"]):
        raise ValueError("input.order_column must be a column name or null")
    if not isinstance(cfg["filename_regex"], str):
        raise ValueError("input.filename_regex must be a regular expression")
    if cfg["format"] == "filename":
        try:
            expression = re.compile(cfg["filename_regex"])
        except re.error as error:
            raise ValueError(f"Invalid input.filename_regex: {error}") from error
        if "label" not in expression.groupindex:
            raise ValueError("input.filename_regex must contain the named group (?P<label>...)")
    return cfg


def _inside(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Input path must be inside data root: {value}")
    return path


def _label(value, cfg, num_classes):
    text = str(value).strip()
    if cfg["label_map"]:
        if text not in cfg["label_map"]:
            raise ValueError(f"Label {text!r} is missing from input.label_map")
        label = cfg["label_map"][text]
    else:
        if not re.fullmatch(r"[+-]?\d+", text):
            raise ValueError(f"Label {text!r} must be an integer, or configured in input.label_map")
        label = int(text) + cfg["label_offset"]
    if label < 0 or (num_classes is not None and label >= int(num_classes)):
        raise ValueError(f"Label {text!r} maps to {label}, outside num_classes={num_classes}; check label_offset/label_map")
    return label


def _columns(header, names, path):
    if len(header) != len(set(header)):
        raise ValueError(f"Duplicate column names in {path}")
    missing = [name for name in names if name not in header]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return [header.index(name) for name in names]


@dataclass(frozen=True)
class TableSample:
    """One logical sequence, located by CSV byte spans rather than a new file."""
    source: Path
    sample_id: str
    spans: tuple[tuple[int, int], ...]
    delimiter: str = ","
    encoding: str = "utf-8-sig"
    order_column: str | None = None

    def __str__(self):
        return f"{self.source}#sample={quote(self.sample_id, safe='')}"

    def __fspath__(self):
        return str(self)

    def relative_to(self, root):
        return Path(f"{self.source.relative_to(root)}#sample={quote(self.sample_id, safe='')}")

    def is_relative_to(self, root):
        return self.source.is_relative_to(root)

    def stat(self):
        return self.source.stat()

    def read_rows(self):
        rows = []
        with self.source.open("rb") as stream:
            reader = csv.reader((line.decode(self.encoding) for line in stream), delimiter=self.delimiter)
            header = next(reader)
            for start, end in self.spans:
                stream.seek(start)
                rows.extend(csv.reader(io.StringIO(stream.read(end - start).decode(self.encoding)), delimiter=self.delimiter))
        if not rows or any(len(row) != len(header) for row in rows):
            raise ValueError(f"Malformed or empty table sample: {self}")
        if self.order_column is not None:
            import math
            column = _columns(header, [self.order_column], self.source)[0]
            try:
                order = [float(row[column]) for row in rows]
            except ValueError as error:
                raise ValueError(f"Non-numeric pulse order in {self}") from error
            if not all(math.isfinite(v) for v in order) or len(set(order)) != len(order):
                raise ValueError(f"Pulse order must contain distinct finite numbers in {self}")
            rows = [row for _, row in sorted(zip(order, rows), key=lambda pair: pair[0])]
        return header, rows


def _table_records(path, cfg, num_classes):
    samples = {}
    with path.open("rb") as stream:
        reader = csv.reader((line.decode(cfg["encoding"]) for line in stream), delimiter=cfg["delimiter"])
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty input table: {path}")
        ids = _columns(header, cfg["sample_id_columns"], path)
        label_column = _columns(header, [cfg["label_column"]], path)[0]
        if cfg["order_column"] is not None:
            _columns(header, [cfg["order_column"]], path)
        start = stream.tell()
        for number, row in enumerate(reader, 2):
            end = stream.tell()
            if not row:
                start = end
                continue
            if len(row) != len(header):
                raise ValueError(f"{path}: record {number} has an inconsistent column count")
            values = [row[i].strip() for i in ids]
            if any(not value for value in values):
                raise ValueError(f"Missing sample ID in {path}: record {number}")
            key = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
            label = _label(row[label_column], cfg, num_classes)
            sample = samples.setdefault(key, {"id": key, "label": label, "spans": []})
            if sample["label"] != label:
                raise ValueError(f"Conflicting labels for sample ID {values}; use composite sample_id_columns when IDs repeat per class")
            spans = sample["spans"]
            if spans and spans[-1][1] == start:
                spans[-1][1] = end
            else:
                spans.append([start, end])
            start = end
    return sorted(samples.values(), key=lambda sample: sample["id"])


def _flat_records(root, files, cfg, num_classes, labels_path):
    paths = sorted((p.resolve() for p in files.glob(cfg["pattern"]) if p.is_file() and p.suffix.lower() == ".csv"
                    and p.resolve() != labels_path), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"No input CSVs matching {cfg['pattern']!r} in {files}")
    if any(p.parent != files or not p.is_relative_to(root) for p in paths):
        raise ValueError("Flat input must contain direct CSV files inside data root")
    if cfg["format"] == "filename":
        expression = re.compile(cfg["filename_regex"])
        records = []
        for path in paths:
            match = expression.fullmatch(path.name)
            if match is None:
                raise ValueError(f"Filename {path.name!r} does not match input.filename_regex")
            records.append({"path": path.relative_to(root).as_posix(), "label": _label(match['label'], cfg, num_classes)})
        return records
    labels = {}
    with labels_path.open(encoding=cfg["encoding"], newline="") as stream:
        reader = ((line.split() for line in stream) if cfg["delimiter"] == "whitespace"
                  else csv.reader(stream, delimiter=cfg["delimiter"]))
        if cfg["header"]:
            header = next(reader, [])
            name_column, label_column = _columns(header, [cfg["filename_column"], cfg["label_column"]], labels_path)
        else:
            name_column, label_column = 0, 1
        for number, row in enumerate(reader, 2 if cfg["header"] else 1):
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) <= max(name_column, label_column) or (not cfg["header"] and len(row) != 2):
                raise ValueError(f"Invalid label row in {labels_path}:{number}")
            name = row[name_column].strip()
            if Path(name).name != name or name in labels:
                raise ValueError(f"Duplicate or invalid filename in labels: {name!r}")
            labels[name] = _label(row[label_column], cfg, num_classes)
    names = {p.name for p in paths}
    if set(labels) != names:
        raise ValueError(f"Label/file mismatch: missing={sorted(names-set(labels))[:5]}, extra={sorted(set(labels)-names)[:5]}")
    return [{"path": p.relative_to(root).as_posix(), "label": labels[p.name]} for p in paths]


def load_source_inventory(root, data_cfg):
    """Cache an inventory; table contents are scanned once, sample reads use seeks."""
    root = Path(root).resolve()
    cfg = input_config(data_cfg.get("input"))
    table = cfg["format"] == "table"
    source = _inside(root, cfg["table_file"] if table else cfg["files"])
    labels_path = _inside(root, cfg["labels_file"]) if cfg["format"] == "label_file" else None
    sources = [source] + ([labels_path] if labels_path is not None else [])
    cache_dir = root / ".pulsecls_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamps = [[str(p), p.stat().st_mtime_ns, p.stat().st_size] for p in sources]
    signature = hashlib.sha256(json.dumps([cfg, data_cfg.get("num_classes"), stamps], sort_keys=True).encode()).hexdigest()
    cache = cache_dir / f"input_index_{cfg['format']}_{hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:12]}.json"
    if cache.exists():
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved.get("version") == 1 and saved.get("fingerprint") == signature:
                return saved["classes"], signature, f"cached {cfg['format']} index"
        except (OSError, ValueError):
            pass
    rows = _table_records(source, cfg, data_cfg.get("num_classes")) if table else _flat_records(
        root, source, cfg, data_cfg.get("num_classes"), labels_path)
    if not rows:
        raise ValueError(f"No labeled samples found in {source}")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    groups = [{"label": label, "directory": str(root), "count": len(records), "records": records,
               **({"table": source.relative_to(root).as_posix(), "delimiter": cfg["delimiter"],
                   "encoding": cfg["encoding"], "order_column": cfg["order_column"]} if table else {})}
              for label, records in sorted(grouped.items())]
    if stamps != [[str(p), p.stat().st_mtime_ns, p.stat().st_size] for p in sources]:
        raise ValueError("Input files changed while building the index; finish preparing the data and retry")
    temporary = cache.with_name(f".{cache.stem}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps({"version": 1, "fingerprint": signature, "classes": groups}), encoding="utf-8")
        temporary.replace(cache)
    finally:
        temporary.unlink(missing_ok=True)
    return groups, signature, f"{cfg['format']} index"
