"""Repair invalid PDW pulses before slicing; write a separate CSV dataset."""
from __future__ import annotations

import argparse
import csv
import json
from numbers import Real
from pathlib import Path
import shutil
import sys
from tempfile import mkdtemp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from pulsecls.cfg import load_yaml, validate_num_classes
from pulsecls.data.dataset import DEFAULT_FEATURES, feature_column_indices, find_samples
from pulsecls.data.inference import find_inference_csvs
from pulsecls.utils.progress import Progress


def _valid_pulses(values):
    # Check the float32 representation actually used by model inputs, too.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        model_values = values.astype(np.float32)
    return (np.isfinite(model_values).all(axis=1) & (model_values > 0).all(axis=1)
            & (model_values[:, 1] < model_values[:, 2]))


def _clean_file(path, iqr):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        columns = feature_column_indices(next(reader, None), DEFAULT_FEATURES, path)
        rows = []
        for row in reader:
            if not row:
                continue
            values = []
            for index in columns:
                try:
                    values.append(float(row[index]))
                except (ValueError, IndexError):
                    values.append(float("nan"))
            rows.append(values)
    if not rows:
        raise ValueError(f"{path} contains no pulses")
    values = np.asarray(rows, dtype=np.float64)
    valid = _valid_pulses(values)
    invalid_count = int((~valid).sum())
    outliers = np.zeros(len(values), dtype=bool)
    if iqr and valid.sum() >= 4:
        lower, upper = np.percentile(values[valid], [25, 75], axis=0)
        margin = iqr * (upper - lower)
        outliers = valid & ((values < lower - margin) | (values > upper + margin)).any(axis=1)
        valid &= ~outliers
    if not valid.any():
        raise ValueError(f"{path} has no valid pulses to interpolate; check its data/units or disable IQR")
    bad, good = np.flatnonzero(~valid), np.flatnonzero(valid)
    # Repair the whole pulse together so interpolation preserves PW < PRI.
    for column in range(len(DEFAULT_FEATURES)):
        values[bad, column] = np.interp(bad, good, values[good, column])
    if not _valid_pulses(values).all():
        raise ValueError(f"{path} cannot be repaired within float32 PDW constraints")
    return values, {"rows": len(values), "invalid_rows": invalid_count,
                    "outlier_rows": int(outliers.sum()), "repaired_rows": len(bad)}


def clean_csv(source, output, *, iqr=0., progress=True):
    """Keep sample identity, pulse count/order and units; interpolate bad rows."""
    if isinstance(iqr, bool) or not isinstance(iqr, Real) or not np.isfinite(iqr) or iqr < 0:
        raise ValueError("iqr must be finite and nonnegative (0 disables statistical filtering)")
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be outside the source tree")
    if (source / "_slicing").exists():
        raise ValueError("Clean original CSVs before slicing; padding zeros are not missing pulses")
    labeled = any(p.is_dir() and p.name.isdecimal() for p in source.iterdir())
    records = (find_samples(source) if labeled else
               [(p, None) for p in find_inference_csvs(source, progress=progress)])
    description = source / "data.yaml"
    original = load_yaml(description) if labeled and description.is_file() else {}
    num_classes = validate_num_classes(original.get("num_classes")) if labeled else None
    if labeled and num_classes is None:
        num_classes = max(label for _, label in records) + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{output.name}.cleaning-", dir=output.parent)).resolve()
    repaired = 0
    try:
        metadata = stage / "_cleaning"
        metadata.mkdir()
        with (metadata / "report.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["file", "rows", "invalid_rows", "outlier_rows", "repaired_rows"])
            writer.writeheader()
            with Progress("Clean CSVs", len(records), unit="file", enabled=progress) as bar:
                for path, label in records:
                    if not path.resolve().is_relative_to(source):
                        raise ValueError(f"CSV must be inside source folder: {path}")
                    if labeled and (path.parent.name != str(label) or not 0 <= label < num_classes):
                        raise ValueError(f"Class directory outside 0..{num_classes - 1}: {path}")
                    before = path.stat()
                    values, counts = _clean_file(path, float(iqr))
                    relative = path.relative_to(source)
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    np.savetxt(target, values, delimiter=",", fmt="%.17g",
                               header=",".join(DEFAULT_FEATURES), comments="", encoding="utf-8")
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError(f"Source changed while cleaning: {path}")
                    writer.writerow({"file": relative.as_posix(), **counts})
                    repaired += counts["repaired_rows"]
                    bar.update()
        if labeled:
            cfg = {key: value for key, value in original.items()
                   if key not in {"root", "root_relative_to_yaml", "input", "all", "train", "val", "test", "selection"}}
            cfg.update(root=".", root_relative_to_yaml=True, all=".", cache=True,
                       num_classes=num_classes, features=list(DEFAULT_FEATURES))
            (stage / "data.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        for name in ("labels.txt", "conversion_manifest.csv", "conversion_summary.json"):
            if (source / name).is_file():
                shutil.copy2(source / name, stage / name)
        summary = {"source_root": str(source), "files": len(records), "repaired_rows": repaired, "iqr": float(iqr)}
        (metadata / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Output appeared while cleaning: {output}")
        stage.rename(output)
    finally:
        if stage.exists() and stage.parent == output.parent and stage.name.startswith(f".{output.name}.cleaning-"):
            shutil.rmtree(stage)
    print(f"Cleaned {len(records)} CSVs; repaired {repaired} pulses. Report: {output / '_cleaning/report.csv'}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Class-folder training pool or flat test CSV folder")
    parser.add_argument("--output", required=True, help="New output directory outside the source tree")
    parser.add_argument("--iqr", type=float, default=0., help="Per-CSV IQR factor, e.g. 3; default 0 disables it")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    clean_csv(args.source, args.output, iqr=args.iqr, progress=not args.no_progress)


if __name__ == "__main__":
    main()
