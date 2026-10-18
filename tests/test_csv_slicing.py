"""Window coverage, source tracking, leak-free splitting, and real model input."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pulsecls import PulseCLS
from pulsecls.cfg import REPO_ROOT, get_cfg, load_yaml
from pulsecls.data.dataset import load_split_arrays, read_sequence_csv
from pulsecls.data.inference import find_inference_csvs
from pulsecls.data.selection import select_records
from tools.slice_csv import slice_csv


def write_sample(path, length):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.column_stack((3000 + np.arange(length), 10 + np.arange(length) * .125,
                              500 + np.arange(length) * .5))
    # Column aliases and ordering are shared with the existing data reader.
    np.savetxt(path, values[:, [2, 0, 1]], delimiter=",", fmt="%.17g",
               header="PRI,载频(MHz),脉宽(us)", comments="", encoding="utf-8-sig")
    return values


def manifest(output):
    with (output / "_slicing/manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("size", [1, 255, 256, 257, 512, 1300, 20001])
@pytest.mark.parametrize("overlap", [0., .5])
def test_windows_cover_original_and_track_offsets(tmp_path, size, overlap):
    source = tmp_path / "raw"
    values = write_sample(source / "信号.csv", size)
    digest = hashlib.sha256((source / "信号.csv").read_bytes()).hexdigest()
    output = slice_csv(source, tmp_path / "sliced", length=256, overlap=overlap, progress=False)
    rows = manifest(output)
    stride = 256 - int(256 * overlap)
    assert len(rows) == max(1, math.ceil((size - 256) / stride) + 1)
    covered = np.zeros(size, dtype=bool)
    for index, row in enumerate(rows):
        start, stop = int(row["start_row"]), int(row["stop_row"])
        assert start == index * stride and stop == min(start + 256, size)
        assert row["source_csv"] == "信号.csv" and int(row["slice_index"]) == index
        assert int(row["source_length"]) == size and int(row["valid_length"]) == stop - start
        window = read_sequence_csv(output / row["slice_csv"])
        assert window.shape == (256, 3)
        np.testing.assert_array_equal(window[:stop-start], values[start:stop])
        assert not window[stop-start:].any()
        covered[start:stop] = True
    assert covered.all()
    assert len(find_inference_csvs(output, progress=False)) == len(rows)
    assert hashlib.sha256((source / "信号.csv").read_bytes()).hexdigest() == digest


def test_repeat_padding_and_fractional_overlap(tmp_path):
    source = tmp_path / "raw"
    values = write_sample(source / "a.csv", 13)
    output = slice_csv(source, tmp_path / "sliced", length=8, overlap=.3, short="repeat", progress=False)
    rows = manifest(output)
    assert [int(r["start_row"]) for r in rows] == [0, 6]
    tail = read_sequence_csv(output / rows[-1]["slice_csv"])
    np.testing.assert_array_equal(tail, np.concatenate([values[6:], values[6:7]]))


def class_pool(root):
    for label in range(2):
        for sample, length in enumerate([9, 19, 36, 51]):
            write_sample(root / str(label) / f"sample{sample}.csv", length)
    return root


@pytest.mark.parametrize("stratified", [True, False])
def test_sampling_and_split_keep_all_slices_of_a_source_together(tmp_path, stratified):
    output = slice_csv(class_pool(tmp_path / "raw"), tmp_path / "sliced", length=16, progress=False)
    cfg = get_cfg(REPO_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
                  overrides={"data": output / "data.yaml", "sequence_length": 16,
                             "dataset": {"sample_ratio": .75, "train_ratio": 2/3,
                                         "val_ratio": 1/3, "stratified": stratified}})["data_cfg"]
    rows = manifest(output)
    origin = {row["slice_csv"]: row["source_csv"] for row in rows}
    parts, info = select_records(cfg)
    sources = {split: {origin[path.relative_to(output).as_posix()] for path, _ in records}
               for split, records in parts.items()}
    assert info["source_counts"] == {"total": 8, "sampled": 6, "train": 4, "val": 2}
    assert not sources["train"] & sources["val"]
    assert sources["train"] | sources["val"] == sources["all"]
    assert select_records(cfg)[0] == parts
    for split in parts:
        expected = {r["slice_csv"] for r in rows if r["source_csv"] in sources[split]}
        assert {p.relative_to(output).as_posix() for p, _ in parts[split]} == expected
    assert load_split_arrays(cfg, "train")[0].shape[1:] == (16, 3)
    cfg["sequence_length"] = 8
    with pytest.raises(ValueError, match="set sequence_length=16"):
        load_split_arrays(cfg, "train")


@pytest.mark.parametrize("options", [{"length": 0}, {"length": True}, {"length": 1.5},
    {"overlap": -1}, {"overlap": 1}, {"overlap": float("nan")}, {"overlap": True}, {"short": "drop"}])
def test_invalid_options_do_not_create_output(tmp_path, options):
    with pytest.raises(ValueError):
        slice_csv(tmp_path / "raw", tmp_path / "sliced", progress=False, **options)
    assert not (tmp_path / "sliced").exists()


def test_failure_preserves_sources_and_existing_output(tmp_path):
    source = tmp_path / "raw"
    write_sample(source / "a.csv", 9)
    (source / "b.csv").write_text("CF_MHz,PW_us,PRI_us\nNaN,10,500\n")
    with pytest.raises(ValueError, match="NaN"):
        slice_csv(source, tmp_path / "sliced", length=8, progress=False)
    assert not (tmp_path / "sliced").exists() and not list(tmp_path.glob(".sliced.slicing-*"))
    with pytest.raises(ValueError, match="outside"):
        slice_csv(source, source / "nested", progress=False)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        slice_csv(source, existing, progress=False)
    assert (existing / "keep.txt").read_text() == "keep" and (source / "a.csv").exists()


def test_cli_runs_outside_repository(tmp_path):
    source = tmp_path / "raw input"
    write_sample(source / "a.csv", 27)
    output = tmp_path / "sliced output"
    run = subprocess.run([sys.executable, str(REPO_ROOT / "tools/slice_csv.py"),
        "--source", str(source), "--output", str(output), "--length", "16", "--overlap", "0.5",
        "--no-progress"], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert len(manifest(output)) == 3


def test_sliced_data_train_validation_and_test_folder_inference(tmp_path):
    output = slice_csv(class_pool(tmp_path / "raw"), tmp_path / "sliced", length=16, progress=False)
    device = os.environ.get("CSV_SLICING_TEST_DEVICE", "cpu")
    api = PulseCLS()
    api.train(pipeline=REPO_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
              data=output / "data.yaml", sequence_length=16, num_classes=2, output_dir=tmp_path / "run",
              device=device, epochs=1, batch_size=4, use_amp=False, visualize=False,
              dataset={"sample_ratio": 1., "train_ratio": .75, "val_ratio": .25},
              augment={"enabled": False}, data_cfg={"progress": False},
              model_cfg={"hidden_channels": [4, 8], "stream_embedding_dim": 8, "contrast_dim": 4})
    saved = json.loads((tmp_path / "run/dataset_selection.json").read_text())
    assert saved["group_by"] == "source_csv" and saved["source_counts"]["sampled"] == 8
    metrics = api.val(pipeline=api.trainer.pipeline, device=device,
                      checkpoint=tmp_path / "run/last_model.pt", save_dir=tmp_path / "val")
    assert np.isfinite(metrics["macro_f1"])
    write_sample(tmp_path / "test/a.csv", 29)
    write_sample(tmp_path / "test/b.csv", 9)
    test = slice_csv(tmp_path / "test", tmp_path / "test_sliced", length=16, progress=False)
    result = api.infer(pipeline=api.trainer.pipeline, source=test, device=device,
                       checkpoint=tmp_path / "run/last_model.pt", save_dir=tmp_path / "predictions",
                       labels_file="", inference={"progress": False})
    predictions = pd.read_csv(result["predictions_csv"])
    source_by_slice = {r["slice_csv"]: r["source_csv"] for r in manifest(test)}
    assert result["num_sequences"] == len(source_by_slice)
    assert set(predictions.filename.map(source_by_slice)) == {"a.csv", "b.csv"}
