import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from pulsecls.data.dataset import load_sample, read_sequence_csv
from tools.visualize_dataset import _sample_records, visualize_dataset


@pytest.fixture
def raw_dataset(tmp_path):
    original = {}
    for label in range(3):
        folder = tmp_path / "sequences" / str(label)
        folder.mkdir(parents=True)
        for number in range(5):
            path = folder / f"sample_{number}.csv"
            length = 1 + label + number
            raw = np.column_stack((3500 + label * 100 + np.arange(length),
                                   10 + number + np.arange(length) * .1,
                                   500 + label * 10 + np.arange(length) * .5))
            np.savetxt(path, raw, delimiter=",", header="CF_MHz,PW_us,PRI_us", comments="")
            original[path] = raw
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({"name": "test_radar", "root": ".", "root_relative_to_yaml": True,
                                   "num_classes": 3, "features": ["CF_MHz", "PW_us", "PRI_us"],
                                   "all": "sequences", "cache": True}), encoding="utf-8")
    return data, original


def test_raw_reader_preserves_units_length_feature_order_and_training_padding(tmp_path):
    path = tmp_path / "sequence.csv"
    path.write_text("PRI_us,label,PW_us,CF_MHz\n500,0,10,3500\n501,0,11,3510\n", encoding="utf-8")
    raw = read_sequence_csv(path)
    np.testing.assert_array_equal(raw, [[3500, 10, 500], [3510, 11, 501]])
    np.testing.assert_array_equal(read_sequence_csv(path, ("PRI_us", "CF_MHz", "PW_us")), raw[:, [2, 0, 1]])
    resized = load_sample(path, 4, length_policy={"short": "zero", "long": "head"})
    np.testing.assert_array_equal(resized[:2], raw)
    np.testing.assert_array_equal(resized[2:], 0)
    assert raw.shape == (2, 3) and raw.dtype == np.float64


def test_visualization_samples_before_reading_and_exports_full_raw_statistics(raw_dataset, tmp_path):
    data, original = raw_dataset
    output = tmp_path / "visualization"
    with patch("tools.visualize_dataset.read_sequence_csv", wraps=read_sequence_csv) as reader:
        result = visualize_dataset(data, output, sequences_per_class=3, max_sequences=4,
                                   plot_classes=2, examples_per_class=1, pulses_per_sequence=2,
                                   max_plot_points=5, bins=5, dpi=60,
                                   formats=("png", "svg", "pdf"), progress=False)
    assert reader.call_count == 4 < len(original)
    assert result["total_sequences"] == 15 and result["sampled_sequences"] == 4
    assert result["plotted_pulses"] <= 5 and result["sampled_class_ids"] == [0, 1, 2]
    assert result["plotted_class_ids"] == [0, 1]
    with (output / "sequence_stats.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    for row in rows:
        raw = original[data.parent / row["path"]]
        assert int(row["num_pulses"]) == len(raw)
        for i, feature in enumerate(("CF_MHz", "PW_us", "PRI_us")):
            assert float(row[f"{feature}_mean"]) == pytest.approx(raw[:, i].mean())
            assert float(row[f"{feature}_std"]) == pytest.approx(raw[:, i].std(ddof=0))
    with (output / "sampled_pulses.csv").open(encoding="utf-8-sig") as stream:
        pulses = list(csv.DictReader(stream))
    assert len(pulses) == result["plotted_pulses"]
    for pulse in pulses:
        source = rows[int(pulse["sequence_id"])]
        expected = original[data.parent / source["path"]][int(pulse["pulse_index"]) - 1]
        np.testing.assert_allclose([float(pulse[k]) for k in ("CF_MHz", "PW_us", "PRI_us")], expected)
    assert json.loads((output / "analysis_summary.json").read_text(encoding="utf-8"))["data_root"] == str(data.parent)
    for figure in result["figures"]:
        for extension in ("png", "svg", "pdf"):
            assert (output / figure["image"]).with_suffix(f".{extension}").stat().st_size > 1000
    report = (output / "index.html").read_text(encoding="utf-8")
    assert "sequence_stats.csv" in report and "sampled_pulses.csv" in report


def test_record_sampling_is_reproducible_filters_classes_and_never_duplicates():
    inventory = [{"label": label, "directory": f"/data/{label}", "count": 10,
                  "names": [f"sample_{i}.csv" for i in range(10)]} for label in range(4)]
    left, labels = _sample_records(inventory, [1, 3], 4, 5, np.random.default_rng(42))
    right, _ = _sample_records(inventory, [1, 3], 4, 5, np.random.default_rng(42))
    assert left == right and len(left) == len(set(left)) == 5
    assert labels == [1, 3] and {label for _, label in left} == {1, 3}
    with pytest.raises(ValueError, match="cover every selected class"):
        _sample_records(inventory, None, 4, 3, np.random.default_rng(42))
    with pytest.raises(ValueError, match="no CSV sequences"):
        _sample_records(inventory, [5], 4, 5, np.random.default_rng(42))


def test_reader_rejects_nonfinite_csv_values(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text("CF_MHz,PW_us,PRI_us\nNaN,10,500\n", encoding="utf-8")
    with pytest.raises(ValueError, match="NaN or infinite"):
        read_sequence_csv(path)
