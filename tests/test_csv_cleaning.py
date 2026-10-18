"""Invalid pulse repair, source preservation, and compatibility with training."""
import csv
import json
import subprocess
import sys

import numpy as np
import pytest
import yaml

from pulsecls import PulseCLS
from pulsecls.cfg import REPO_ROOT, get_cfg, load_yaml
from pulsecls.data.dataset import DEFAULT_FEATURES, load_split_arrays, read_sequence_csv
from pulsecls.data.inference import find_inference_csvs
from tools.clean_csv import clean_csv
from tools.slice_csv import slice_csv


def write_sample(path, rows, header=DEFAULT_FEATURES):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def report(output):
    with (output / "_cleaning/report.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("bad", [
    ["", 10, 100], [1000, "bad", 100], [1000, 10, "NaN"],
    ["inf", 10, 100], [1000, 10, "-inf"],
    [0, 10, 100], [-1, 10, 100], [1000, 0, 100], [1000, 10, -1],
    [1000, 200, 100], [1000, 100, 100], [1e100, 10, 100], [1e-100, 10, 100],
    [1000],
])
def test_repair_invalid_whole_pulse_without_changing_order_or_sources(tmp_path, bad):
    source = tmp_path / "raw"
    path = source / "sample.csv"
    write_sample(path, [[1000, 10, 100], bad, [3000, 30, 300]])
    original = path.read_bytes()
    output = clean_csv(source, tmp_path / "clean", progress=False)
    np.testing.assert_array_equal(read_sequence_csv(output / path.name),
                                  [[1000, 10, 100], [2000, 20, 200], [3000, 30, 300]])
    assert path.read_bytes() == original
    assert report(output) == [{"file": "sample.csv", "rows": "3", "invalid_rows": "1",
                               "outlier_rows": "0", "repaired_rows": "1"}]


def test_aliases_boundary_fill_and_labels_file(tmp_path):
    source = tmp_path / "raw"
    write_sample(source / "信号.csv", [[100, "NaN", 1], [200, 2000, 20],
                                       [300, 3000, 30], [400, 4000, ""]],
                 header=["PRI", "载频(MHz)", "脉宽(us)"])
    labels = b"\xef\xbb\xbf\xe4\xbf\xa1\xe5\x8f\xb7.csv\t0\n"
    (source / "labels.txt").write_bytes(labels)
    output = clean_csv(source, tmp_path / "clean", progress=False)
    np.testing.assert_array_equal(read_sequence_csv(output / "信号.csv"),
                                  [[2000, 20, 200], [2000, 20, 200], [3000, 30, 300], [3000, 30, 300]])
    assert (output / "labels.txt").read_bytes() == labels
    assert [p.name for p in find_inference_csvs(output, progress=False)] == ["信号.csv"]


def test_iqr_is_optional_and_ignores_invalid_pulses(tmp_path):
    source = tmp_path / "raw"
    values = np.tile([3000., 10., 100.], (13, 1))
    values[6, 0] = 1e12
    values[8] = [1e20, -1, 100]
    write_sample(source / "sample.csv", values)
    plain = clean_csv(source, tmp_path / "plain", progress=False)
    assert read_sequence_csv(plain / "sample.csv")[6, 0] == 1e12
    filtered = clean_csv(source, tmp_path / "filtered", iqr=3, progress=False)
    np.testing.assert_array_equal(read_sequence_csv(filtered / "sample.csv"),
                                  np.tile([3000., 10., 100.], (13, 1)))
    assert report(filtered)[0]["invalid_rows"] == "1"
    assert report(filtered)[0]["outlier_rows"] == "1"
    assert report(filtered)[0]["repaired_rows"] == "2"


@pytest.mark.parametrize("content", ["", "CF_MHz,PW_us,PRI_us\n", "CF_MHz,PW_us\n1,2\n",
                                      "CF_MHz,PW_us,PRI_us\nNaN,0,0\n"])
def test_unrecoverable_file_rolls_back_without_skipping_a_sample(tmp_path, content):
    source = tmp_path / "raw"
    write_sample(source / "a.csv", [[3000, 10, 100]])
    (source / "b.csv").write_text(content, encoding="utf-8")
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    with pytest.raises(ValueError):
        clean_csv(source, tmp_path / "clean", progress=False)
    assert not (tmp_path / "clean").exists()
    assert not list(tmp_path.glob(".clean.cleaning-*"))
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original


@pytest.mark.parametrize("iqr", [-1, True, "3", float("nan"), float("inf")])
def test_invalid_options_create_no_output(tmp_path, iqr):
    with pytest.raises(ValueError, match="iqr"):
        clean_csv(tmp_path / "raw", tmp_path / "clean", iqr=iqr, progress=False)
    assert not (tmp_path / "clean").exists()


def test_existing_output_nested_output_and_sliced_sources_are_protected(tmp_path):
    source = tmp_path / "raw"
    write_sample(source / "a.csv", [[3000, 10, 100]])
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        clean_csv(source, output, progress=False)
    assert (output / "keep.txt").read_text() == "keep"
    with pytest.raises(ValueError, match="outside"):
        clean_csv(source, source / "nested", progress=False)
    (source / "_slicing").mkdir()
    with pytest.raises(ValueError, match="before slicing"):
        clean_csv(source, tmp_path / "clean", progress=False)


def test_class_metadata_and_clean_slice_train_validation_inference(tmp_path):
    source = tmp_path / "raw"
    for label in range(2):
        for index in range(4):
            rows = np.column_stack([3000 + label * 100 + np.arange(24 + index),
                                    np.full(24 + index, 10), np.full(24 + index, 100)]).astype(float)
            rows[3, 0] = np.nan
            write_sample(source / str(label) / f"{index}.csv", rows)
    (source / "data.yaml").write_text(yaml.safe_dump({"root": "wrong/old/path", "all": ".",
        "num_classes": 2, "class_names": ["A", "B"], "sequence_length": 16,
        "length_policy": {"short": "zero", "long": "head"}}), encoding="utf-8")
    clean = clean_csv(source, tmp_path / "clean", progress=False)
    assert load_yaml(clean / "data.yaml")["root"] == "."
    assert load_yaml(clean / "data.yaml")["class_names"] == ["A", "B"]
    assert len(report(clean)) == 8
    cfg = get_cfg(REPO_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
                  overrides={"data": clean / "data.yaml", "sequence_length": 16,
                             "dataset": {"sample_ratio": 1., "train_ratio": .75, "val_ratio": .25},
                             "data_cfg": {"cache": False, "progress": False}})["data_cfg"]
    assert load_split_arrays(cfg, "train")[0].shape == (6, 16, 3)
    sliced = slice_csv(clean, tmp_path / "sliced", length=16, progress=False)
    api = PulseCLS()
    api.train(pipeline=REPO_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
              data=sliced / "data.yaml", sequence_length=16, num_classes=2, output_dir=tmp_path / "run",
              device="cpu", epochs=1, batch_size=4, use_amp=False, visualize=False,
              dataset={"sample_ratio": 1., "train_ratio": .75, "val_ratio": .25},
              augment={"enabled": False}, data_cfg={"cache": False, "progress": False},
              model_cfg={"hidden_channels": [4, 8], "stream_embedding_dim": 8, "contrast_dim": 4})
    metrics = api.val(pipeline=api.trainer.pipeline, device="cpu",
                      checkpoint=tmp_path / "run/last_model.pt", save_dir=tmp_path / "val")
    assert np.isfinite(metrics["macro_f1"])
    result = api.infer(pipeline=api.trainer.pipeline, source=clean / "0", device="cpu",
                       checkpoint=tmp_path / "run/last_model.pt", save_dir=tmp_path / "predictions",
                       labels_file="", inference={"progress": False})
    assert result["num_sequences"] == 4


def test_cli_runs_outside_repository_and_keeps_valid_values(tmp_path):
    source = tmp_path / "raw input"
    values = np.array([[3000.123456789, 10.123456789, 100.123456789]])
    write_sample(source / "sample.csv", values)
    output = tmp_path / "clean output"
    run = subprocess.run([sys.executable, str(REPO_ROOT / "tools/clean_csv.py"),
                          "--source", str(source), "--output", str(output), "--no-progress"],
                         cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    np.testing.assert_array_equal(read_sequence_csv(output / "sample.csv"), values)
    assert json.loads((output / "_cleaning/summary.json").read_text())["repaired_rows"] == 0
