"""Slice radar PDW CSV files and optionally append derived features.

CF/PW/PRI are always retained. Every selected derived channel has exactly the
same sequence length as the original channels; the first difference/rate value
of each source sequence is filled with zero before windowing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from data_process.csv_features import (
        FEATURE_COLS,
        feature_cli_help,
        parse_no_header_order,
        read_feature_csv,
    )
except ModuleNotFoundError:
    from csv_features import FEATURE_COLS, feature_cli_help, parse_no_header_order, read_feature_csv

try:
    from data_process.h5_storage import H5SliceWriter, resolve_h5_path
except ModuleNotFoundError:
    from h5_storage import H5SliceWriter, resolve_h5_path


DERIVED_FEATURE_COLUMNS = {
    "duty_cycle": "Duty_Cycle",
    "delta_rf": "Delta_CF_MHz",
    "delta_pw": "Delta_PW_us",
    "delta_pri": "Delta_PRI_us",
    "pw_change_rate": "PW_Change_Rate",
}
DERIVED_FEATURE_ALIASES = {
    "duty": "duty_cycle",
    "delta_cf": "delta_rf",
}


def normalize_derived_features(features: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Validate feature keys and return them in a stable channel order."""
    if features is None:
        return ()
    raw_values = [features] if isinstance(features, str) else list(features)
    tokens = [
        token.strip().lower()
        for value in raw_values
        for token in str(value).split(",")
        if token.strip()
    ]
    if not tokens or tokens == ["none"]:
        return ()
    if "none" in tokens:
        raise ValueError("'none' cannot be combined with other derived features")
    if "all" in tokens:
        if len(tokens) != 1:
            raise ValueError("'all' cannot be combined with other derived features")
        return tuple(DERIVED_FEATURE_COLUMNS)

    normalized = [DERIVED_FEATURE_ALIASES.get(token, token) for token in tokens]
    unknown = sorted(set(normalized) - set(DERIVED_FEATURE_COLUMNS))
    if unknown:
        available = ", ".join(DERIVED_FEATURE_COLUMNS)
        raise ValueError(f"Unknown derived features {unknown}; available: {available}, all, none")
    selected = set(normalized)
    return tuple(name for name in DERIVED_FEATURE_COLUMNS if name in selected)


def output_feature_cols(derived_features: tuple[str, ...] | list[str] | str | None = None) -> tuple[str, ...]:
    """Return original and selected derived column names in persisted order."""
    selected = normalize_derived_features(derived_features)
    return FEATURE_COLS + tuple(DERIVED_FEATURE_COLUMNS[name] for name in selected)


def add_derived_features(
    data: np.ndarray,
    derived_features: tuple[str, ...] | list[str] | str | None = None,
) -> np.ndarray:
    """Append the selected aligned derived channels.

    ``data`` must use the canonical ``CF_MHz, PW_us, PRI_us`` order. Division
    by zero is mapped to zero so the generated data remains finite.
    """
    selected = normalize_derived_features(derived_features)
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_COLS):
        raise ValueError(
            f"Expected data shaped [length, {len(FEATURE_COLS)}], got {values.shape}"
        )
    if len(values) == 0:
        return np.empty(
            (0, len(output_feature_cols(selected))), dtype=np.float32
        )
    if not np.isfinite(values).all():
        raise ValueError("Source features contain NaN or infinite values")

    feature_values: dict[str, np.ndarray] = {}
    pw = values[:, 1]
    if "duty_cycle" in selected:
        pri = values[:, 2]
        feature_values["duty_cycle"] = np.divide(
            pw,
            pri,
            out=np.zeros_like(pw),
            where=pri != 0,
        )

    difference_features = {"delta_rf", "delta_pw", "delta_pri", "pw_change_rate"}
    if difference_features.intersection(selected):
        differences = np.zeros_like(values)
        differences[1:] = values[1:] - values[:-1]
        feature_values["delta_rf"] = differences[:, 0]
        feature_values["delta_pw"] = differences[:, 1]
        feature_values["delta_pri"] = differences[:, 2]

    if "pw_change_rate" in selected:
        pw_change_rate = np.zeros_like(pw)
        previous_pw = pw[:-1]
        np.divide(
            differences[1:, 1],
            np.abs(previous_pw),
            out=pw_change_rate[1:],
            where=previous_pw != 0,
        )
        feature_values["pw_change_rate"] = pw_change_rate

    channels = [values]
    channels.extend(feature_values[name][:, None] for name in selected)
    result = np.concatenate(channels, axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Derived features contain NaN or infinite values")
    return result.astype(np.float32, copy=False)


def pad_by_repeat(data: np.ndarray, target_len: int) -> np.ndarray:
    """Repeat complete aligned rows until ``target_len`` is reached."""
    if len(data) >= target_len:
        return data[:target_len]
    if len(data) == 0:
        raise ValueError("Cannot pad an empty signal")
    repeats = (target_len + len(data) - 1) // len(data)
    return np.tile(data, (repeats, 1))[:target_len]


def sliding_window_cut(data: np.ndarray, slice_len: int, overlap: float) -> list[np.ndarray]:
    """Cut aligned feature rows using the same semantics as ``cut_data.py``."""
    n_rows = len(data)
    if n_rows < slice_len:
        return [pad_by_repeat(data, slice_len)]

    step = max(1, int(slice_len * (1.0 - overlap)))
    windows: list[np.ndarray] = []
    start = 0
    while start + slice_len <= n_rows:
        windows.append(data[start : start + slice_len])
        start += step
    if start < n_rows:
        windows.append(pad_by_repeat(data[start:], slice_len))
    return windows


def process_file(
    filepath: str | Path,
    slice_len: int,
    overlap: float,
    no_header_order: str | None = None,
    derived_features: tuple[str, ...] | list[str] | str | None = None,
) -> np.ndarray:
    """Return one file as ``[num_slices, length, channels]``."""
    frame = read_feature_csv(filepath, no_header_order=no_header_order).features
    data = frame.to_numpy(dtype=np.float64, copy=False)
    enriched = add_derived_features(data, derived_features)
    windows = sliding_window_cut(enriched, slice_len, overlap)
    return np.asarray(windows, dtype=np.float32)


def read_file_list(file_list: str | Path) -> list[tuple[Path, str]]:
    """Read ``path,label`` lines generated by ``split_data.py``."""
    entries: list[tuple[Path, str]] = []
    file_list_path = Path(file_list).resolve()
    with file_list_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.rsplit(",", 1)
            filepath = Path(fields[0].strip()).expanduser()
            if not filepath.is_absolute():
                filepath = file_list_path.parent / filepath
            label = fields[1].strip() if len(fields) == 2 else filepath.parent.name
            if not filepath.is_file():
                raise FileNotFoundError(f"File list line {line_number} does not exist: {filepath}")
            if not label:
                raise ValueError(f"File list line {line_number} has no class label: {line}")
            entries.append((filepath.resolve(), label))
    if not entries:
        raise ValueError(f"File list is empty: {file_list_path}")
    return entries


def discover_files(input_dir: Path) -> list[tuple[Path, str]]:
    """Find CSV files stored below immediate class directories."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    csv_files = sorted(
        path for path in input_dir.rglob("*.csv") if path.name != "_index.csv"
    )
    root_files = [path for path in csv_files if len(path.relative_to(input_dir).parts) < 2]
    if root_files:
        raise ValueError("Input CSV files must be inside class directories")
    return [(path, path.relative_to(input_dir).parts[0]) for path in csv_files]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSV sliding-window slicing with aligned derived PDW features"
    )
    parser.add_argument("--input_dir", "-d", default="", help="Input CSV root directory")
    parser.add_argument(
        "--output_dir", "-o", default="./dataset/cut_data_features/val", help="Output path"
    )
    parser.add_argument(
        "--file_list",
        "--txt_path",
        dest="file_list",
        default=r"dataset\split_data\val.txt",
        help="CSV path and label list generated by split_data.py; use an empty value to scan input_dir",
    )
    parser.add_argument("--slice_len", "-l", type=int, default=256, help="Slice length")
    parser.add_argument("--overlap", "-p", type=float, default=0.0, help="Overlap ratio [0, 1)")
    parser.add_argument(
        "--derived_features",
        "--derived-features",
        nargs="*",
        default=(),
        metavar="FEATURE",
        help=(
            "Optional derived features: duty_cycle, delta_rf, delta_pw, delta_pri, "
            "pw_change_rate; use 'all' for every feature (comma-separated values are also accepted)"
        ),
    )
    parser.add_argument(
        "--include_pw_change_rate",
        "--pw-change-rate",
        action="store_true",
        help="Backward-compatible alias that adds pw_change_rate",
    )
    parser.add_argument(
        "--output_format", choices=("h5", "csv"), default="csv", help="Output format"
    )
    parser.add_argument("--h5_name", default="val.h5", help="H5 filename when output_dir is a directory")
    parser.add_argument("--h5_compression", choices=("none", "lzf", "gzip"), default="lzf")
    parser.add_argument("--h5_buffer_size", type=int, default=2048)
    parser.add_argument("--no-header-order", default=None, help=feature_cli_help())
    args = parser.parse_args()

    if args.slice_len <= 0:
        parser.error("--slice_len must be greater than zero")
    if not 0 <= args.overlap < 1:
        parser.error("--overlap must satisfy 0 <= overlap < 1")
    if args.h5_buffer_size <= 0:
        parser.error("--h5_buffer_size must be greater than zero")
    try:
        if args.no_header_order:
            parse_no_header_order(args.no_header_order)
        derived_features = list(normalize_derived_features(args.derived_features))
        if args.include_pw_change_rate and "pw_change_rate" not in derived_features:
            derived_features.append("pw_change_rate")
        args.derived_features = normalize_derived_features(derived_features)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    columns = output_feature_cols(args.derived_features)
    h5_path = resolve_h5_path(output_dir, args.h5_name)

    if args.file_list:
        file_entries = read_file_list(args.file_list)
    else:
        try:
            file_entries = discover_files(input_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

    print(
        f"Found {len(file_entries)} CSV files, slice length={args.slice_len}, "
        f"overlap={args.overlap:.0%}, channels={len(columns)}"
    )
    print("Output features: " + ", ".join(columns))

    if args.output_format == "h5":
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with H5SliceWriter(
            h5_path,
            args.slice_len,
            columns,
            compression=args.h5_compression,
            buffer_size=args.h5_buffer_size,
        ) as writer:
            for filepath, class_name in tqdm(file_entries, desc="Slicing"):
                windows = process_file(
                    filepath,
                    args.slice_len,
                    args.overlap,
                    args.no_header_order,
                    args.derived_features,
                )
                writer.add(windows, class_name, filepath.stem)
        print(f"Done: {writer.count} slices written to {h5_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    global_id = 0
    for filepath, class_name in tqdm(file_entries, desc="Slicing"):
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        windows = process_file(
            filepath,
            args.slice_len,
            args.overlap,
            args.no_header_order,
            args.derived_features,
        )
        for slice_index, values in enumerate(windows):
            filename = f"{filepath.stem}_slice{slice_index:04d}.csv"
            pd.DataFrame(values, columns=columns).to_csv(
                class_output_dir / filename, index=False, encoding="utf-8-sig"
            )
            index_rows.append(
                {
                    "global_slice_id": global_id,
                    "filename": (Path(class_name) / filename).as_posix(),
                    "class_name": class_name,
                    "source_file": filepath.stem,
                    "slice_index": slice_index,
                }
            )
            global_id += 1

    pd.DataFrame(index_rows).to_csv(
        output_dir / "_index.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Done: {global_id} slices written to {output_dir}")


if __name__ == "__main__":
    main()
