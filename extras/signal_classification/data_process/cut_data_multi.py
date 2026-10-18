"""
Parallel CSV signal slicer.

This is the accelerated counterpart of ``cut_data.py``. It keeps the same
feature columns and sliding-window semantics while processing independent
source CSV files in parallel. HDF5 is written only by the main process.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from data_process.csv_features import FEATURE_COLS, feature_cli_help, parse_no_header_order, read_feature_csv
except ModuleNotFoundError:
    from csv_features import FEATURE_COLS, feature_cli_help, parse_no_header_order, read_feature_csv

try:
    from data_process.h5_storage import H5SliceWriter, resolve_h5_path
except ModuleNotFoundError:
    from h5_storage import H5SliceWriter, resolve_h5_path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False

def _natural_path_key(path: Path) -> tuple:
    """Sort numeric path components naturally (2 before 10)."""
    import re
    parts = re.split(r"(\d+)", str(path).replace("\\", "/"))
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts)


def pad_by_repeat(data: np.ndarray, target_len: int) -> np.ndarray:
    """Repeat rows until ``data`` has exactly ``target_len`` rows."""
    if len(data) >= target_len:
        return data[:target_len]
    if len(data) == 0:
        raise ValueError("Cannot pad an empty signal")
    repeats = (target_len + len(data) - 1) // len(data)
    return np.tile(data, (repeats, 1))[:target_len]


def sliding_window_array(
    data: np.ndarray,
    slice_len: int,
    overlap: float,
) -> np.ndarray:
    """Return all windows as one ``[num_windows, slice_len, features]`` array."""
    n_rows = data.shape[0]
    if n_rows < slice_len:
        return pad_by_repeat(data, slice_len)[None, ...]

    step = max(1, int(slice_len * (1.0 - overlap)))
    starts = np.arange(0, n_rows - slice_len + 1, step, dtype=np.int64)
    windows = data[starts[:, None] + np.arange(slice_len, dtype=np.int64)]

    # Match cut_data.py: after the last complete window, retain and repeat-pad
    # the remaining tail whenever the next start is still inside the signal.
    next_start = int(starts[-1]) + step
    if next_start < n_rows:
        tail = pad_by_repeat(data[next_start:], slice_len)
        windows = np.concatenate((windows, tail[None, ...]), axis=0)
    return windows


def _write_slice(path: Path, values: np.ndarray) -> None:
    frame = pd.DataFrame(values, columns=FEATURE_COLS)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def process_file(
    task: tuple[int, str, str, str, int, float, str, str | None]
) -> tuple[int, Any, str, str, str | None]:
    """Read, slice, and write one source file inside a worker process."""
    file_index, filepath, output_dir, class_name, slice_len, overlap, output_format, no_header_order = task
    source_name = Path(filepath).stem
    frame = None
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            frame = read_feature_csv(filepath, no_header_order=no_header_order).features
            break
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))
    if frame is None:
        error = f"{filepath}\t{type(last_error).__name__}: {last_error}"
        return file_index, None, class_name, source_name, error
    frame = frame.dropna(how="all")
    if frame.empty:
        empty = np.empty((0, slice_len, len(FEATURE_COLS)), dtype=np.float32)
        return file_index, empty if output_format == "h5" else [], class_name, source_name, None

    data = frame.to_numpy(dtype=np.float32, copy=False)
    windows = sliding_window_array(data, slice_len, overlap)
    if output_format == "h5":
        return file_index, windows, class_name, source_name, None

    output_path = Path(output_dir) / class_name
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for slice_index, values in enumerate(windows):
        filename = f"{source_name}_slice{slice_index:04d}.csv"
        _write_slice(output_path / filename, values)
        rows.append(
            {
                "filename": (Path(class_name) / filename).as_posix(),
                "class_name": class_name,
                "source_file": source_name,
                "slice_index": slice_index,
            }
        )
    return file_index, rows, class_name, source_name, None


def read_file_list(file_list: str | Path) -> list[tuple[Path, str]]:
    """Read split_data.py lines: absolute_csv_path,label."""
    entries: list[tuple[Path, str]] = []
    with Path(file_list).open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.rsplit(",", 1)
            filepath = Path(fields[0].strip()).expanduser()
            if not filepath.is_absolute():
                filepath = Path(file_list).resolve().parent / filepath
            label = fields[1].strip() if len(fields) == 2 else filepath.parent.name
            if not filepath.is_file():
                raise FileNotFoundError(f"清单第{line_number}行文件不存在: {filepath}")
            if not label:
                raise ValueError(f"清单第{line_number}行缺少类别标签: {line}")
            entries.append((filepath.resolve(), label))
    if not entries:
        raise ValueError(f"清单为空: {file_list}")
    return entries

def _worker_count(requested: int) -> int:
    if requested > 0:
        return requested
    cpu_count = os.cpu_count() or 1
    # CSV output is I/O-heavy; an unbounded process count usually hurts speed.
    return max(1, min(cpu_count, 8))


def main() -> None:
    parser = argparse.ArgumentParser(description="并行CSV滑动窗口切片并保存为H5或CSV")
    parser.add_argument("--input_dir", "-d", default=r'G:\competition\2026第五届天智杯\赛前准备\雷达型号识别\test_competition_lengyan-main\test_competition_lengyan-main\datasets\synthetic_radar_overlap', help="输入CSV目录")
    parser.add_argument("--output_dir", "-o", default='./dataset/cut_data/val', help="输出切片目录")

    parser.add_argument("--file_list", "--txt_path", dest="file_list", default=r'dataset\split_data\val.txt', help="split_data.py生成的CSV路径清单")
    parser.add_argument("--slice_len", "-l", type=int, default=256, help="切片长度(默认256)")
    parser.add_argument("--overlap", "-p", type=float, default=0, help="重叠比例[0,1)(默认0)")
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=8,
        help="进程数，0表示自动选择，默认最多使用8个进程",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="进程池任务批大小；文件大小差异较大时建议保持1",
    )
    parser.add_argument(
        "--output_format",
        choices=["h5", "csv"],
        default="h5",
        help="保存格式；h5由主进程批量写入，csv由工作进程分别写入",
    )
    parser.add_argument("--h5_name", default="val.h5", help="output_dir为目录时的H5文件名")
    parser.add_argument("--h5_compression", choices=["none", "lzf", "gzip"], default="lzf")
    parser.add_argument("--h5_buffer_size", type=int, default=2048, help="H5批量写入的切片数")
    parser.add_argument("--no-header-order", default=None, help=feature_cli_help())
    args = parser.parse_args()

    if args.slice_len <= 0:
        parser.error("--slice_len必须大于0")
    if not 0 <= args.overlap < 1:
        parser.error("--overlap必须满足0 <= overlap < 1")
    if args.workers < 0:
        parser.error("--workers不能小于0")
    if args.chunksize <= 0:
        parser.error("--chunksize必须大于0")
    if args.h5_buffer_size <= 0:
        parser.error("--h5_buffer_size必须大于0")
    try:
        if args.no_header_order:
            parse_no_header_order(args.no_header_order)
    except ValueError as exc:
        parser.error(str(exc))

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    h5_path = resolve_h5_path(output_dir, args.h5_name)
    if args.output_format == "h5":
        h5_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.file_list:
        file_entries = read_file_list(args.file_list)
    else:
        if not input_dir.is_dir():
            parser.error(f"输入目录不存在: {input_dir}")
        output_resolved = output_dir.resolve()
        csv_files = sorted(
            (
                path for path in input_dir.rglob("*.csv")
                if path.name != "_index.csv"
                and not _is_relative_to(path.resolve(), output_resolved)
            ),
            key=_natural_path_key,
        )
        root_files = [path for path in csv_files if len(path.relative_to(input_dir).parts) < 2]
        if root_files:
            parser.error("输入CSV必须放在input_dir下的类别文件夹中")
        file_entries = [(path, path.relative_to(input_dir).parts[0]) for path in csv_files]
    class_names = sorted({class_name for _, class_name in file_entries})
    if args.output_format == "csv":
        for class_name in class_names:
            (output_dir / class_name).mkdir(parents=True, exist_ok=True)
    print(
        f"找到 {len(file_entries)} 个CSV, 切片长度={args.slice_len}, "
        f"重叠={args.overlap:.0%}, 进程数={_worker_count(args.workers)}, "
        f"输出格式={args.output_format}"
    )
    if not file_entries:
        if args.output_format == "h5":
            with H5SliceWriter(
                h5_path,
                args.slice_len,
                FEATURE_COLS,
                compression=args.h5_compression,
                buffer_size=args.h5_buffer_size,
            ):
                pass
            print(f"完成! 共 0 个切片保存至 {h5_path}")
        else:
            pd.DataFrame(columns=["global_slice_id", "filename", "class_name", "source_file", "slice_index"]).to_csv(
                output_dir / "_index.csv", index=False, encoding="utf-8-sig"
            )
            print(f"完成! 共 0 个切片保存至 {output_dir}")
        return

    tasks = [
        (
            index,
            str(filepath),
            str(output_dir),
            class_name,
            args.slice_len,
            args.overlap,
            args.output_format,
            args.no_header_order,
        )
        for index, (filepath, class_name) in enumerate(file_entries)
    ]
    index_rows: list[dict[str, Any]] = []
    error_rows: list[str] = []
    max_workers = _worker_count(args.workers)
    global_id = 0
    writer = None
    if args.output_format == "h5":
        writer = H5SliceWriter(
            h5_path,
            args.slice_len,
            FEATURE_COLS,
            compression=args.h5_compression,
            buffer_size=args.h5_buffer_size,
        )

    # executor.map preserves input order, so global IDs remain identical to the
    # serial script even though files are processed concurrently.
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            result_iter = executor.map(process_file, tasks, chunksize=args.chunksize)
            for _, payload, class_name, source_name, error in tqdm(
                result_iter, total=len(tasks), desc="切片处理中"
            ):
                if error:
                    error_rows.append(error)
                    continue
                if args.output_format == "h5":
                    writer.add(payload, class_name, source_name)
                    global_id += len(payload)
                else:
                    for row in payload:
                        index_rows.append({"global_slice_id": global_id, **row})
                        global_id += 1
    finally:
        if writer is not None:
            writer.close()

    if error_rows:
        error_root = h5_path.parent if args.output_format == "h5" else output_dir
        error_path = error_root / "_errors.txt"
        error_path.write_text("\n".join(error_rows) + "\n", encoding="utf-8")
        print(f"警告: {len(error_rows)} 个文件读取失败，详情见 {error_path}")

    if args.output_format == "csv":
        pd.DataFrame(
            index_rows,
            columns=["global_slice_id", "filename", "class_name", "source_file", "slice_index"],
        ).to_csv(output_dir / "_index.csv", index=False, encoding="utf-8-sig")
        destination = output_dir
    else:
        destination = h5_path
    print(f"完成! 共 {global_id} 个切片保存至 {destination}")


if __name__ == "__main__":
    main()
