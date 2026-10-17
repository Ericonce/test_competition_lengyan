"""Prepare external competition data once, then train on standard class folders."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
from tempfile import mkdtemp

import yaml

from pulsecls.cfg import load_yaml, remap_repo_path, validate_num_classes
from pulsecls.data.dataset import DEFAULT_FEATURES, feature_column_indices, read_sequence_csv
from pulsecls.data.sources import TableSample, _flat_records, _inside, _table_records, input_config
from pulsecls.utils.progress import Progress


def _stamp(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def convert_dataset(data, output, *, progress=True):
    """Copy labeled CSVs or split a long table into a new, standalone dataset.

    The output must be a new directory outside the source tree. Original files
    are untouched. A failed conversion never publishes a partial dataset.
    """
    if isinstance(data, dict):
        description, cfg = None, deepcopy(data)
    else:
        description = Path(data).resolve()
        cfg = load_yaml(description)
    base = description.parent if description and cfg.get("root_relative_to_yaml", False) else None
    root = remap_repo_path(cfg["root"], base).resolve()
    options = input_config(cfg.get("input"))
    if options["format"] == "class_folders":
        raise ValueError("Data already uses class_folders; train with its existing data.yaml")
    features = cfg.get("features", list(DEFAULT_FEATURES))
    if (not isinstance(features, (list, tuple)) or not features
            or any(not isinstance(name, str) or not name for name in features)
            or len(set(features)) != len(features)):
        raise ValueError("features must be a nonempty list of distinct column names")
    num_classes = validate_num_classes(cfg.get("num_classes"))
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Output must be outside the source tree; choose a sibling directory")

    table = options["format"] == "table"
    source = _inside(root, options["table_file"] if table else options["files"])
    labels = _inside(root, options["labels_file"]) if options["format"] == "label_file" else None
    watched = [source] + ([labels] if labels else [])
    before = [_stamp(path) for path in watched]
    with Progress("Read source labels", unit="steps", enabled=progress):
        rows = (_table_records(source, options, num_classes) if table
                else _flat_records(root, source, options, num_classes, labels))
    if not rows:
        raise ValueError(f"No labeled samples found in {source}")
    counts = Counter(row["label"] for row in rows)
    output_cfg = {key: value for key, value in cfg.items()
                  if key not in {"root", "root_relative_to_yaml", "input", "all", "train", "val", "test"}}
    output_cfg.update(root=".", root_relative_to_yaml=True, all=".",
                      num_classes=num_classes or max(counts) + 1, features=list(features),
                      input={"format": "class_folders"})
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{output.name}.converting-", dir=output.parent)).resolve()
    try:
        for label in counts:
            (stage / str(label)).mkdir()
        emitted = Counter()
        with (stage / "conversion_manifest.csv").open("w", encoding="utf-8-sig", newline="") as manifest:
            writer = csv.writer(manifest)
            writer.writerow(["source", "label", "output_path", "num_pulses"])
            with Progress("Convert to class folders", len(rows), unit="sample", enabled=progress) as bar:
                for row in rows:
                    label = row["label"]
                    if table:
                        ref = TableSample(source, row["id"], tuple(tuple(span) for span in row["spans"]),
                                          options["delimiter"], options["encoding"], options["order_column"])
                        header, pulses = ref.read_rows()
                        columns = feature_column_indices(header, features, ref)
                        relative = Path(str(label)) / f"{emitted[label]}.csv"
                        with (stage / relative).open("w", encoding="utf-8", newline="") as stream:
                            sample_writer = csv.writer(stream)
                            sample_writer.writerow(features)
                            sample_writer.writerows([pulse[i] for i in columns] for pulse in pulses)
                        identity = str(ref.relative_to(root))
                    else:
                        ref = _inside(root, row["path"])
                        stamp = _stamp(ref)
                        relative = Path(str(label)) / ref.name
                        shutil.copy2(ref, stage / relative)
                        if _stamp(ref) != stamp:
                            raise ValueError(f"Source changed during conversion: {ref}")
                        identity = ref.relative_to(root).as_posix()
                    # Use the very same reader as training, before publishing any files.
                    values = read_sequence_csv(stage / relative, features)
                    writer.writerow([identity, label, relative.as_posix(), len(values)])
                    emitted[label] += 1
                    bar.update()
        if before != [_stamp(path) for path in watched]:
            raise ValueError("Source changed during conversion; finish preparing the data and retry")
        summary = {"source_root": str(root), "input": options, "num_samples": len(rows),
                   "class_counts": dict(sorted(counts.items())), "features": list(features)}
        (stage / "conversion_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (stage / "data.yaml").write_text(yaml.safe_dump(output_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Output appeared during conversion: {output}")
        stage.rename(output)
    finally:
        # Only our newly created staging directory can be removed here.
        if stage.exists() and stage.parent == output.parent and stage.name.startswith(f".{output.name}.converting-"):
            shutil.rmtree(stage)
    return output / "data.yaml"
