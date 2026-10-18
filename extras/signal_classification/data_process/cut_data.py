"""
cut_data.py
将目录下长度不定的CSV(载频/脉宽/PRI)按滑动窗口切分为统一长度切片。
不足切片长度的数据通过循环复制补充。
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
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


def pad_by_repeat(data: np.ndarray, target_len: int) -> np.ndarray:
    """若数据行数不足target_len，则循环复制填充至target_len。"""
    n = len(data)
    if n >= target_len:
        return data
    repeats = target_len // n + 1
    padded = np.tile(data, (repeats, 1))[:target_len]
    return padded


def sliding_window_cut(data: np.ndarray, slice_len: int, overlap: float):
    """滑动窗口切分，返回切片列表。"""
    n = len(data)
    if n < slice_len:
        return [pad_by_repeat(data, slice_len)]

    step = max(1, int(slice_len * (1 - overlap)))
    slices = []
    start = 0
    while start + slice_len <= n:
        slices.append(data[start:start + slice_len])
        start += step

    # 尾部不足一个切片时，循环复制补充
    if start < n:
        tail = data[start:]
        slices.append(pad_by_repeat(tail, slice_len))

    return slices


# def read_file_list(file_list):
#     """Read split_data.py lines: absolute_csv_path,label."""
#     entries: list[tuple[Path, str]] = []
#     file_list_path = Path(file_list).resolve()
#     with file_list_path.open("r", encoding="utf-8-sig") as f:
#         for line_number, raw_line in enumerate(f, 1):
#             line = raw_line.strip()
#             if not line or line.startswith("#"):
#                 continue
#             fields = line.rsplit(",", 1)
#             filepath = Path(fields[0].strip()).expanduser()
#             if not filepath.is_absolute():
#                 filepath = file_list_path.parent / filepath
#             label = fields[1].strip() if len(fields) == 2 else filepath.parent.name
#             if not filepath.is_file():
#                 raise FileNotFoundError(f"清单第{line_number}行文件不存在: {filepath}")
#             if not label:
#                 raise ValueError(f"清单第{line_number}行缺少类别标签: {line}")
#             entries.append((filepath.resolve(), label))
#     if not entries:
#         raise ValueError(f"清单为空: {file_list}")
#     return entries
def read_file_list(file_list):
    """Read split_data.py lines: absolute_csv_path,label."""
    entries: list[tuple[Path, str]] = []
    with Path(file_list).open("r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        for line_number, raw_line in enumerate(tqdm(lines), 1):
        # for line_number, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            # if not line or line.startswith("#"):
            #     continue
            fields = line.rsplit(",", 1)
            filepath = Path(fields[0].strip()).expanduser()
            # if not filepath.is_absolute():
            #     filepath = Path(file_list).resolve().parent / filepath
            label = fields[1].strip() if len(fields) == 2 else filepath.parent.name
            # if not filepath.is_file():
            #     raise FileNotFoundError(f"清单第{line_number}行文件不存在: {filepath}")
            # if not label:
            #     raise ValueError(f"清单第{line_number}行缺少类别标签: {line}")
            entries.append((filepath.resolve(), label))
    if not entries:
        raise ValueError(f"清单为空: {file_list}")
    return entries

def process_file(
    filepath: str,
    slice_len: int,
    overlap: float,
    no_header_order: str | None = None,
):
    """处理单个CSV，返回[num_slices, length, channels]数组。"""
    df = read_feature_csv(filepath, no_header_order=no_header_order).features
    if len(df) == 0:
        return np.empty((0, slice_len, len(FEATURE_COLS)), dtype=np.float32)
    raw_slices = sliding_window_cut(
        df.to_numpy(dtype=np.float32, copy=False), slice_len, overlap
    )
    return np.asarray(raw_slices, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="CSV滑动窗口切片并保存为H5或CSV")
    parser.add_argument("--input_dir", "-d", default=r'', help="输入CSV目录")
    parser.add_argument("--output_dir", "-o", default='./dataset/cut_data/val', help="输出切片目录")

    parser.add_argument("--file_list", "--txt_path", dest="file_list", default=r'dataset\split_data\val.txt', help="split_data.py生成的CSV路径清单")
    parser.add_argument("--slice_len", "-l", type=int, default=256, help="切片长度(默认256)")
    parser.add_argument("--overlap", "-p", type=float, default=0, help="重叠比例[0,1)(默认0)")
    parser.add_argument(
        "--output_format",
        choices=["h5", "csv"],
        default="csv",
        help="保存格式；h5速度更快，csv用于兼容旧流程",
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
    if args.h5_buffer_size <= 0:
        parser.error("--h5_buffer_size必须大于0")

    try:
        no_header_order = args.no_header_order
        if no_header_order:
            # Validate once before processing the file list.
            parse_no_header_order(no_header_order)
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
        csv_files = sorted(path for path in input_dir.rglob("*.csv") if path.name != "_index.csv")
        root_files = [path for path in csv_files if len(path.relative_to(input_dir).parts) < 2]
        if root_files:
            parser.error("输入CSV必须放在input_dir下的类别文件夹中")
        file_entries = [(path, path.relative_to(input_dir).parts[0]) for path in csv_files]
    print(f"找到 {len(file_entries)} 个CSV, 切片长度={args.slice_len}, 重叠={args.overlap:.0%}")
     
    idx = np.random.choice(len(file_entries), size=10000, replace=False)
    file_entries = [file_entries[i] for i in idx]

    if args.output_format == "h5":
        with H5SliceWriter(
            h5_path,
            args.slice_len,
            FEATURE_COLS,
            compression=args.h5_compression,
            buffer_size=args.h5_buffer_size,
        ) as writer:
            for filepath, class_name in tqdm(file_entries, desc="切片处理中"):
                windows = process_file(
                    str(filepath), args.slice_len, args.overlap, no_header_order
                )
                writer.add(windows, class_name, filepath.stem)
        print(f"完成! 共 {writer.count} 个切片保存至 {h5_path}")
        return

    index_rows = []
    global_id = 0
    for filepath, class_name in tqdm(file_entries, desc="切片处理中"):
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        windows = process_file(
            str(filepath), args.slice_len, args.overlap, no_header_order
        )
        source = filepath.stem
        for slice_index, values in enumerate(windows):
            filename = f"{source}_slice{slice_index:04d}.csv"
            pd.DataFrame(values, columns=FEATURE_COLS).to_csv(
                class_output_dir / filename, index=False, encoding="utf-8-sig"
            )
            index_rows.append({
                "global_slice_id": global_id,
                "filename": (Path(class_name) / filename).as_posix(),
                "class_name": class_name,
                "source_file": source,
                "slice_index": slice_index,
            })
            global_id += 1

    pd.DataFrame(index_rows).to_csv(
        output_dir / "_index.csv", index=False, encoding="utf-8-sig"
    )
    print(f"完成! 共 {global_id} 个切片保存至 {output_dir}")


if __name__ == "__main__":
    main()
