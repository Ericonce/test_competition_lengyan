"""Shared HDF5 storage helpers for sliced radar PDW signals."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


H5_FILENAMES = ("data.h5", "slices.h5", "data.hdf5", "slices.hdf5")


def resolve_h5_path(output: str | Path, filename: str = "data.h5") -> Path:
    """Resolve either an explicit H5 path or a directory plus filename."""
    output = Path(output)
    if output.suffix.lower() in {".h5", ".hdf5"}:
        return output
    return output / filename


def find_h5_file(path: str | Path) -> Path | None:
    """Find one unambiguous H5 file at ``path`` or directly below it."""
    path = Path(path)
    if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}:
        return path
    if not path.is_dir():
        return None
    for name in H5_FILENAMES:
        candidate = path / name
        if candidate.is_file():
            return candidate
    candidates = sorted((*path.glob("*.h5"), *path.glob("*.hdf5")))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple H5 files found below {path}; specify one explicitly: {candidates}"
        )
    return candidates[0]


class H5SliceWriter:
    """Append sliced signals to one chunked HDF5 file through a write buffer."""

    def __init__(
        self,
        path: str | Path,
        slice_len: int,
        feature_cols: Iterable[str],
        compression: str = "lzf",
        buffer_size: int = 2048,
    ) -> None:
        if slice_len < 1:
            raise ValueError("slice_len must be positive")
        if buffer_size < 1:
            raise ValueError("buffer_size must be positive")
        if compression not in {"none", "lzf", "gzip"}:
            raise ValueError("compression must be one of: none, lzf, gzip")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.slice_len = int(slice_len)
        self.feature_cols = tuple(str(column) for column in feature_cols)
        self.buffer_size = int(buffer_size)
        self._file = h5py.File(self.path, "w")
        self._file.attrs["format_version"] = 1
        self._file.attrs["layout"] = "samples,channels,length"
        self._file.attrs["feature_cols"] = np.asarray(
            self.feature_cols, dtype=h5py.string_dtype("utf-8")
        )
        self._file.attrs["slice_len"] = self.slice_len

        chunk_samples = min(self.buffer_size, 1024)
        compression_value = None if compression == "none" else compression
        compression_options = 4 if compression == "gzip" else None
        self._data = self._file.create_dataset(
            "data",
            shape=(0, len(self.feature_cols), self.slice_len),
            maxshape=(None, len(self.feature_cols), self.slice_len),
            chunks=(chunk_samples, len(self.feature_cols), self.slice_len),
            dtype=np.float32,
            compression=compression_value,
            compression_opts=compression_options,
            shuffle=compression_value is not None,
        )
        string_dtype = h5py.string_dtype("utf-8")
        self._raw_labels = self._file.create_dataset(
            "raw_labels", shape=(0,), maxshape=(None,), dtype=string_dtype
        )
        self._groups = self._file.create_dataset(
            "groups", shape=(0,), maxshape=(None,), dtype=string_dtype
        )
        self._source_files = self._file.create_dataset(
            "source_files", shape=(0,), maxshape=(None,), dtype=string_dtype
        )
        self._slice_indices = self._file.create_dataset(
            "slice_indices", shape=(0,), maxshape=(None,), dtype=np.int32
        )
        self._buffer_data: list[np.ndarray] = []
        self._buffer_labels: list[str] = []
        self._buffer_groups: list[str] = []
        self._buffer_sources: list[str] = []
        self._buffer_indices: list[int] = []
        self._buffer_count = 0
        self.count = 0

    def add(self, windows: np.ndarray, class_name: str, source_name: str) -> None:
        """Add ``[num_slices, length, channels]`` windows for one source file."""
        windows = np.asarray(windows, dtype=np.float32)
        if windows.size == 0:
            return
        expected = (self.slice_len, len(self.feature_cols))
        if windows.ndim != 3 or tuple(windows.shape[1:]) != expected:
            raise ValueError(
                f"Expected windows [N, {expected[0]}, {expected[1]}], got {windows.shape}"
            )
        count = windows.shape[0]
        self._buffer_data.append(windows.transpose(0, 2, 1).copy())
        self._buffer_labels.extend([str(class_name)] * count)
        self._buffer_groups.extend([f"{class_name}/{source_name}"] * count)
        self._buffer_sources.extend([str(source_name)] * count)
        self._buffer_indices.extend(range(count))
        self._buffer_count += count
        if self._buffer_count >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer_count:
            return
        data = np.concatenate(self._buffer_data, axis=0)
        start = self.count
        end = start + len(data)
        for dataset in (
            self._data,
            self._raw_labels,
            self._groups,
            self._source_files,
            self._slice_indices,
        ):
            dataset.resize(end, axis=0)
        self._data[start:end] = data
        self._raw_labels[start:end] = self._buffer_labels
        self._groups[start:end] = self._buffer_groups
        self._source_files[start:end] = self._buffer_sources
        self._slice_indices[start:end] = self._buffer_indices
        self.count = end
        self._buffer_data.clear()
        self._buffer_labels.clear()
        self._buffer_groups.clear()
        self._buffer_sources.clear()
        self._buffer_indices.clear()
        self._buffer_count = 0

    def close(self) -> None:
        if self._file is None:
            return
        self.flush()
        self._file.attrs["num_slices"] = self.count
        self._file.close()
        self._file = None

    def __enter__(self) -> "H5SliceWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

