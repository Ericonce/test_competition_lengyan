import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from pulsecls.data.dataset import find_samples
from pulsecls.data.file_index import load_inventory
from tools.postcheck_overlap import _find_aligned, postcheck_overlap


FEATURES = ["CF_MHz", "PW_us", "PRI_us"]


def write_csv(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, delimiter=",", header=",".join(FEATURES), comments="", fmt="%.17g")
    return path


def read_report(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "training"
    a = np.column_stack((3500 + np.arange(8), np.arange(8) / 4, 500 + np.arange(8)))
    b = np.column_stack((4500 + np.arange(4), 10 + np.arange(4), 600 + np.arange(4)))
    train_a = write_csv(root / "pool/0/a.csv", a)
    train_b = write_csv(root / "pool/1/b.csv", b)
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump({"name": "radar", "root": ".", "root_relative_to_yaml": True,
                                    "all": "pool", "num_classes": 3, "features": FEATURES}), encoding="utf-8")
    source = tmp_path / "test"
    source.mkdir()
    return data, source, train_a, train_b, a, b


def test_exact_numeric_equality_containment_and_unmatched_export(dataset, tmp_path):
    data, source, train_a, train_b, a, b = dataset
    shutil.copy2(train_a, source / "0.csv")
    # Aliased headers, changed column order, signed zero, CRLF, BOM and an ignored column.
    formatted = "label,PRI,脉宽(us),载频(MHz)\r\n" + "".join(
        f"123,{row[2]:.6f},{'-0.0' if row[1] == 0 else f'{row[1]:.6f}'},{row[0]:.6f}\r\n" for row in a)
    (source / "1.csv").write_bytes(b"\xef\xbb\xbf" + formatted.encode("utf-8"))
    write_csv(source / "2.csv", a[2:5])
    write_csv(source / "3.csv", np.vstack(([1, 2, 3], b, [4, 5, 6])))
    write_csv(source / "4.csv", [[9000, 30, 1000], [9001, 31, 1001]])
    original = {p: p.read_bytes() for p in [train_a, train_b, *source.glob("*.csv")]}
    output = tmp_path / "result"
    result = postcheck_overlap(data, source, output, progress=False)
    assert result["matched_files"] == 4 and result["unmatched_files"] == 1 and result["conflict_files"] == 0
    rows = {row["filename"]: row for row in read_report(output / "assignments.csv")}
    for name, label, rule in (("0.csv", 0, "sha256_equal"), ("1.csv", 0, "sequence_equal"),
                              ("2.csv", 0, "test_in_train"), ("3.csv", 1, "train_in_test")):
        assert rows[name]["prediction"] == str(label) and rows[name]["strongest_rule"] == rule
        assert (output / rows[name]["output_path"]).read_bytes() == (source / name).read_bytes()
    matches = {row["filename"]: row for row in read_report(output / "matches.csv")}
    assert matches["2.csv"]["train_start_row"] == "2" and matches["2.csv"]["overlap_rows"] == "3"
    assert matches["3.csv"]["test_start_row"] == "1" and matches["3.csv"]["overlap_rows"] == "4"
    assert rows["4.csv"]["status"] == "unmatched" and rows["4.csv"]["prediction"] == ""
    assert read_report(output / "conflicts.csv") == []
    assert [row["filename"] for row in read_report(output / "unmatched.csv")] == ["4.csv"]
    assert (output / "matched_labels.txt").read_text(encoding="utf-8") == "0.csv\t0\n1.csv\t0\n2.csv\t0\n3.csv\t1\n"
    counts = read_report(output / "class_counts.csv")
    assert [int(row["matched_test_files"]) for row in counts] == [3, 1, 0]
    assert [int(row["reference_files"]) for row in counts] == [1, 1, 0]
    assert len(read_report(output / "reference_files.csv")) == 2
    classified = output / "classified"
    exported = yaml.safe_load((classified / "data.yaml").read_text(encoding="utf-8"))
    assert exported["features"] == FEATURES and exported["root_relative_to_yaml"] is True
    inventory, _, _ = load_inventory(classified, exported)
    assert sum(group["count"] for group in inventory) == 4
    assert len(find_samples(classified)) == 4
    assert all(p.read_bytes() == contents for p, contents in original.items())
    assert not list(output.rglob("*.sqlite3"))
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["reference_scope"] == "data_yaml_inventory" and saved["matched_files"] == 4


def test_multiple_same_class_references_are_kept_but_test_is_copied_once(dataset, tmp_path):
    data, source, train_a, _, a, _ = dataset
    write_csv(train_a.parent / "duplicate.csv", a)
    shutil.copy2(train_a, source / "same.csv")
    output = tmp_path / "result"
    summary = postcheck_overlap(data, source, output, progress=False)
    assert summary["matched_files"] == 1 and len(read_report(output / "matches.csv")) == 2
    assignment = read_report(output / "assignments.csv")[0]
    assert assignment["match_count"] == "2" and assignment["candidate_labels"] == "[0]"
    assert len(list((output / "classified/0").glob("*.csv"))) == 1


@pytest.mark.parametrize("reference_kind", ["equal", "shorter"])
def test_cross_class_conflicts_block_assignment_even_with_exact_sha(dataset, tmp_path, reference_kind):
    data, source, train_a, train_b, a, _ = dataset
    write_csv(train_b, a if reference_kind == "equal" else a[1:4])
    shutil.copy2(train_a, source / "conflict.csv")
    output = tmp_path / "result"
    summary = postcheck_overlap(data, source, output, progress=False)
    assert summary["conflict_files"] == 1 and summary["matched_files"] == 0
    assert not list((output / "classified").rglob("*.csv"))
    row = read_report(output / "conflicts.csv")[0]
    assert row["candidate_labels"] == "[0, 1]" and row["prediction"] == ""
    assert row["strongest_rule"] == "sha256_equal" and row["match_count"] == "2"
    assert (output / "matched_labels.txt").read_text(encoding="utf-8") == ""


def test_partial_overlap_gaps_reversal_and_float32_rounding_are_not_exact(dataset, tmp_path):
    data, source, _, _, a, _ = dataset
    write_csv(source / "partial.csv", np.vstack((a[-3:], [9000, 50, 1000])))
    write_csv(source / "gapped.csv", a[[1, 3, 5]])
    write_csv(source / "reversed.csv", a[::-1])
    changed = a.copy()
    changed[3, 0] += 1e-9
    assert changed[3, 0] != a[3, 0] and np.float32(changed[3, 0]) == np.float32(a[3, 0])
    write_csv(source / "rounded.csv", changed)
    summary = postcheck_overlap(data, source, tmp_path / "result", progress=False)
    assert summary["unmatched_files"] == 4 and summary["matched_files"] == 0


def test_minimum_overlap_applies_to_containment_only(dataset, tmp_path):
    data, source, _, train_b, a, _ = dataset
    write_csv(source / "one_row.csv", a[3:4])
    shutil.copy2(train_b, source / "exact.csv")
    first = postcheck_overlap(data, source, tmp_path / "all_lengths", progress=False)
    second = postcheck_overlap(data, source, tmp_path / "minimum5", min_overlap_rows=5, progress=False)
    assert first["matched_files"] == 2
    assert second["matched_files"] == 1 and second["unmatched_files"] == 1
    assert read_report(tmp_path / "minimum5/assignments.csv")[1]["filename"] == "one_row.csv"


def test_selection_file_checks_only_actual_train_records_and_rebases_root(dataset, tmp_path):
    data, source, train_a, train_b, _, _ = dataset
    shutil.copy2(train_a, source / "train.csv")
    shutil.copy2(train_b, source / "val.csv")
    manifest = tmp_path / "dataset_selection.json"
    manifest.write_text(json.dumps({"data_root": "F:/old_machine/datasets/radar",
                                   "train": [{"path": "pool/0/a.csv", "label": 0}],
                                   "val": [{"path": "pool/1/b.csv", "label": 1}]}), encoding="utf-8")
    summary = postcheck_overlap(data, source, tmp_path / "result", selection_file=manifest, progress=False)
    assert summary["reference_scope"] == "selected_training_only" and summary["reference_files"] == 1
    rows = {row["filename"]: row for row in read_report(tmp_path / "result/assignments.csv")}
    assert rows["train.csv"]["prediction"] == "0" and rows["val.csv"]["status"] == "unmatched"


@pytest.mark.parametrize("manifest", [[], {"train": ["pool/0/a.csv"]},
                                      {"train": [{"path": None, "label": 0}]},
                                      {"train": [{"path": "pool/0/a.csv"}]}])
def test_malformed_training_selection_reports_a_clear_error(dataset, tmp_path, manifest):
    data, source, train_a, _, _, _ = dataset
    shutil.copy2(train_a, source / "same.csv")
    selection = tmp_path / "dataset_selection.json"
    selection.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="(train list|Invalid training record)"):
        postcheck_overlap(data, source, tmp_path / "result", selection_file=selection, progress=False)
    assert not (tmp_path / "result").exists()


def test_external_manifest_reference_folder_cannot_receive_output(dataset, tmp_path):
    data, source, train_a, _, a, _ = dataset
    reference = write_csv(tmp_path / "external/0/a.csv", a)
    manifest = data.parent / "manifest.json"
    manifest.write_text(json.dumps({"root": "../external", "samples": [{"path": "0/a.csv", "label": 0}]}), encoding="utf-8")
    cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
    cfg["all"] = "manifest.json"
    data.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    shutil.copy2(train_a, source / "same.csv")
    output = reference.parent / "result"
    with pytest.raises(ValueError, match="reference CSV class folder"):
        postcheck_overlap(data, source, output, progress=False)
    assert not output.exists()


def test_data_paths_and_exported_dataset_work_from_another_cwd(dataset, tmp_path, monkeypatch):
    data, source, train_a, _, _, _ = dataset
    shutil.copy2(train_a, source / "same.csv")
    elsewhere = tmp_path / "different_cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    output = tmp_path / "result"
    result = postcheck_overlap(data, source, output, progress=False)
    assert result["matched_files"] == 1
    assert Path(result["classified_dir"]).is_absolute()
    exported_cfg = yaml.safe_load((output / "classified/data.yaml").read_text(encoding="utf-8"))
    assert exported_cfg["root"] == "." and exported_cfg["all"] == "."
    assert len(find_samples(output / "classified")) == 1


def test_data_yaml_manifest_inventory_is_supported(dataset, tmp_path):
    data, source, train_a, _, _, _ = dataset
    manifest = data.parent / "manifest.json"
    manifest.write_text(json.dumps({"root": "pool", "samples": [{"path": "0/a.csv", "label": 0}]}), encoding="utf-8")
    cfg = yaml.safe_load(data.read_text())
    cfg["all"] = "manifest.json"
    data.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    shutil.copy2(train_a, source / "same.CSV")
    (source / "nested").mkdir()
    (source / "nested/ignored.csv").write_text("invalid CSV", encoding="utf-8")
    summary = postcheck_overlap(data, source, tmp_path / "result", progress=False)
    assert summary["reference_files"] == 1 and summary["test_files"] == 1 and summary["matched_files"] == 1


def test_output_is_not_overwritten_and_cannot_be_inside_inputs(dataset, tmp_path):
    data, source, train_a, _, _, _ = dataset
    shutil.copy2(train_a, source / "same.csv")
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        postcheck_overlap(data, source, existing, progress=False)
    assert (existing / "keep.txt").read_text() == "keep"
    for output in (source / "result", data.parent / "result"):
        with pytest.raises(ValueError, match="outside"):
            postcheck_overlap(data, source, output, progress=False)
        assert not output.exists()


def test_invalid_reference_fails_without_publishing_partial_result(dataset, tmp_path):
    data, source, train_a, train_b, _, _ = dataset
    shutil.copy2(train_a, source / "good.csv")
    train_b.write_text("CF_MHz,PW_us,PRI_us\n1,inf,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="NaN or infinite"):
        postcheck_overlap(data, source, tmp_path / "result", progress=False)
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".postcheck-*"))


def test_empty_flat_test_folder_is_rejected(dataset, tmp_path):
    data, source, _, _, _, _ = dataset
    with pytest.raises(FileNotFoundError, match="No CSV sequences"):
        postcheck_overlap(data, source, tmp_path / "result", progress=False)
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("content", ["CF_MHz,PW_us\n1,2\n", "CF_MHz,PW_us,PRI_us\nNaN,2,3\n",
                                     "CF_MHz,PW_us,PRI_us\n", ""])
def test_invalid_csv_fails_without_publishing_partial_result(dataset, tmp_path, content):
    data, source, _, _, _, _ = dataset
    (source / "bad.csv").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        postcheck_overlap(data, source, tmp_path / "result", progress=False)
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".postcheck-*"))


def test_indexed_matches_agree_with_exhaustive_contiguous_numpy_search(tmp_path):
    rng = np.random.default_rng(2026)
    root = tmp_path / "training"
    references, test_values = {}, {}
    for index in range(15):
        values = rng.integers(0, 4, size=(int(rng.integers(1, 8)), 3)).astype(float)
        path = write_csv(root / str(index % 3) / f"{index}.csv", values)
        references[str(path)] = (index % 3, values)
        test_values[f"equal_{index}.csv"] = values
        test_values[f"subset_{index}.csv"] = values[len(values) // 2:]
        test_values[f"superset_{index}.csv"] = np.vstack(([[99, 99, 99]], values, [[88, 88, 88]]))
    source = tmp_path / "test"
    for name, values in test_values.items():
        write_csv(source / name, values)
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump({"root": ".", "root_relative_to_yaml": True,
                                    "all": ".", "num_classes": 3, "features": FEATURES}), encoding="utf-8")
    expected = {}
    for name, test in test_values.items():
        for path, (label, train) in references.items():
            short, long = (test, train) if len(test) <= len(train) else (train, test)
            for offset in range(len(long) - len(short) + 1):
                if np.array_equal(short, long[offset:offset + len(short)]):
                    expected[(name, path)] = (label, offset if len(test) < len(train) else 0,
                                             offset if len(test) > len(train) else 0, len(short))
                    break
    output = tmp_path / "result"
    postcheck_overlap(data, source, output, progress=False)
    actual = {(row["filename"], row["train_path"]): (int(row["label"]), int(row["train_start_row"]),
                                                   int(row["test_start_row"]), int(row["overlap_rows"]))
              for row in read_report(output / "matches.csv")}
    assert actual == expected
    for row in read_report(output / "assignments.csv"):
        labels = sorted({value[0] for (name, _), value in expected.items() if name == row["filename"]})
        assert json.loads(row["candidate_labels"]) == labels
        assert row["status"] == ("matched" if len(labels) == 1 else "conflict" if labels else "unmatched")


def test_byte_search_rejects_unaligned_matches_and_finds_first_valid_row():
    assert _find_aligned(b"xabcabcabc", b"abc", 3) == -1
    assert _find_aligned(b"xabcxxabc", b"abc", 3) == 2


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_overlap_threshold_rejects_invalid_values(dataset, tmp_path, value):
    data, source, _, _, _, _ = dataset
    with pytest.raises(ValueError, match="positive integer"):
        postcheck_overlap(data, source, tmp_path / "result", min_overlap_rows=value, progress=False)
