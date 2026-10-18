"""Slice class-folder training CSVs or flat test CSVs without changing sources."""
from __future__ import annotations

import argparse
import csv
import json
from numbers import Integral, Real
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
from pulsecls.data.dataset import DEFAULT_FEATURES, find_samples, read_sequence_csv, resize_sequence
from pulsecls.data.inference import find_inference_csvs
from pulsecls.utils.progress import Progress


def slice_csv(source, output, *, length=256, overlap=0.5, short="zero", progress=True):
    """Export fixed-stride windows, padding only the final incomplete window.

    Row offsets exclude the header and use [start_row, stop_row). At most one
    padded window is emitted per source; already covered tails are not repeated.
    """
    if isinstance(length, bool) or not isinstance(length, Integral) or length < 1:
        raise ValueError("length must be a positive integer")
    if (isinstance(overlap, bool) or not isinstance(overlap, Real)
            or not np.isfinite(overlap) or not 0 <= overlap < 1):
        raise ValueError("overlap must be a finite ratio in [0, 1)")
    if short not in {"zero", "repeat"}:
        raise ValueError("short must be zero or repeat")
    length = int(length)
    stride = length - int(length * overlap)
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be outside the source tree")
    if (source / "_slicing").exists():
        raise ValueError("Already sliced; choose the original CSV folder")
    labeled = any(p.is_dir() and p.name.isdecimal() for p in source.iterdir())
    records = (find_samples(source) if labeled else
               [(path, None) for path in find_inference_csvs(source, progress=progress)])
    description = source / "data.yaml"
    original = load_yaml(description) if labeled and description.is_file() else {}
    features = original.get("features", list(DEFAULT_FEATURES))
    num_classes = validate_num_classes(original.get("num_classes")) if labeled else None
    if labeled and num_classes is None:
        num_classes = max(label for _, label in records) + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{output.name}.slicing-", dir=output.parent)).resolve()
    count = 0
    settings = {"length": length, "overlap": float(overlap), "stride": stride, "short": short}
    try:
        metadata = stage / "_slicing"
        metadata.mkdir()
        with (metadata / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["slice_csv", "source_csv", "slice_index", "start_row", "stop_row",
                             "valid_length", "source_length", "label"])
            with Progress("Slice CSVs", len(records), unit="file", enabled=progress) as bar:
                for ref, label in records:
                    if not ref.resolve().is_relative_to(source):
                        raise ValueError(f"CSV must be inside source folder: {ref}")
                    if labeled and (ref.parent.name != str(label) or not 0 <= label < num_classes):
                        raise ValueError(f"Expected numeric class directory in range 0..{num_classes - 1}: {ref}")
                    before = ref.stat()
                    values = read_sequence_csv(ref, features)
                    relative = ref.relative_to(source)
                    (stage / relative.parent).mkdir(parents=True, exist_ok=True)
                    for index, start in enumerate(range(0, len(values), stride)):
                        stop = min(start + length, len(values))
                        window = resize_sequence(values[start:stop], length, {"short": short})
                        target = relative.with_name(f"{ref.stem}__slice{index:06d}.csv")
                        np.savetxt(stage / target, window, delimiter=",", fmt="%.17g",
                                   header=",".join(features), comments="", encoding="utf-8")
                        writer.writerow([target.as_posix(), relative.as_posix(), index, start, stop,
                                         stop - start, len(values), label])
                        count += 1
                        if stop == len(values):
                            break
                    after = ref.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError(f"Source changed while slicing: {ref}")
                    bar.update()
        if labeled:
            cfg = {"root": ".", "root_relative_to_yaml": True, "all": ".", "cache": True,
                   "num_classes": num_classes, "features": list(features),
                   "sequence_length": length, "length_policy": {"short": short, "long": "head"},
                   "slicing": {**settings, "manifest": "_slicing/manifest.csv", "group_by_source": True}}
            if "class_names" in original:
                cfg["class_names"] = original["class_names"]
            (stage / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        summary = {"source_root": str(source), "source_files": len(records), "slices": count,
                   "labeled": labeled, "features": list(features), **settings}
        (metadata / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Output appeared while slicing: {output}")
        stage.rename(output)
    finally:
        if stage.exists() and stage.parent == output.parent and stage.name.startswith(f".{output.name}.slicing-"):
            shutil.rmtree(stage)
    print(f"Sliced {len(records)} CSVs into {count} windows; length={length}, stride={stride}.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Class-folder training pool or flat test CSV folder")
    parser.add_argument("--output", required=True, help="New output folder outside the source tree")
    parser.add_argument("--length", type=int, default=256, help="Pulses per slice (default: 256)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap ratio in [0, 1), e.g. 0.5 = 50%%")
    parser.add_argument("--short", choices=("zero", "repeat"), default="zero", help="Padding for short CSVs and final windows")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    output = slice_csv(args.source, args.output, length=args.length, overlap=args.overlap,
                       short=args.short, progress=not args.no_progress)
    print(f"Sliced CSVs: {output}\nSource mapping: {output / '_slicing/manifest.csv'}")


if __name__ == "__main__":
    main()
