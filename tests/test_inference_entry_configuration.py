"""Check entrypoint configuration without loading project weights or large datasets."""
import ast
import json
import runpy
from pathlib import Path
from unittest.mock import patch

import pytest

from pulsecls import PulseCLS
from pulsecls.cfg import REPO_ROOT, get_cfg


@pytest.mark.parametrize("entry", sorted((REPO_ROOT / "infer").rglob("*.py")))
def test_inference_entries_keep_paths_and_common_settings_in_calls(entry, monkeypatch):
    monkeypatch.setattr("sys.argv", [str(entry)])
    tree = ast.parse(entry.read_text(encoding="utf-8"))
    assignments = {target.id for node in tree.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    assert assignments == {"PROJECT_ROOT"}
    seen = []

    def capture(self, **kwargs):
        assert {"pipeline", "source", "checkpoint", "save_dir", "labels_file", "num_classes",
                "device", "sequence_length", "length_policy", "inference", "augment"} <= set(kwargs)
        assert Path(kwargs["checkpoint"]).suffix == ".pt"
        assert Path(kwargs["augment"]["preset"]).is_file()
        overrides = {key: value for key, value in kwargs.items() if key not in ("pipeline", "num_classes")}
        overrides["inference.num_classes"] = kwargs["num_classes"]
        cfg = get_cfg(kwargs["pipeline"], overrides=overrides)
        assert cfg["inference"]["num_classes"] == kwargs["num_classes"]
        assert cfg["inference"]["batch_size"] == 256
        assert cfg["data_cfg"]["sequence_length"] == kwargs["sequence_length"]
        assert cfg["augment"]["methods"]
        if entry.name == "infer_cnn_softmax_synthetic.py":
            assert Path(kwargs["labels_file"]) == Path(kwargs["source"]) / "labels.txt"
        seen.append(kwargs)

    def capture_vote(self, config, **kwargs):
        assert Path(config).is_file()
        assert kwargs["num_classes"] == 100 and kwargs["batch_size"] == 256
        assert Path(kwargs["augment"]["preset"]).is_file()
        assert kwargs["main_model"] in [m["name"] for m in kwargs["models"] if m["enabled"]]
        for member in kwargs["models"]:
            assert Path(member["pipeline"]).is_file() and Path(member["checkpoint"]).suffix == ".pt"
        seen.append(kwargs)

    with patch.object(PulseCLS, "infer", capture), patch.object(PulseCLS, "infer_vote", capture_vote):
        runpy.run_path(str(entry), run_name="__main__")
    assert len(seen) == 1


def test_custom_pipeline_cli_keeps_its_defaults_and_applies_inference_options(monkeypatch):
    entry = REPO_ROOT / "infer/infer.py"
    pipeline = REPO_ROOT / "configs/pipeline/mambasl_synthetic.yaml"
    monkeypatch.setattr("sys.argv", [str(entry), "--pipeline", str(pipeline), "--batch-size", "3",
                        "--num-classes", "10", "--no-progress", "--short-policy", "repeat"])
    def capture(self, **kwargs):
        assert kwargs["checkpoint"] is None
        assert kwargs["inference"] == {"batch_size": 3, "progress": False}
        assert kwargs["length_policy"] == {"short": "repeat", "long": "head"}
        assert kwargs["num_classes"] == 10
        assert "augment" not in kwargs and "save_dir" not in kwargs
    with patch.object(PulseCLS, "infer", capture):
        runpy.run_path(str(entry), run_name="__main__")


def test_cli_output_directory_changes_automatic_checkpoint_and_predictions(monkeypatch):
    entry = REPO_ROOT / "infer/infer.py"
    monkeypatch.setattr("sys.argv", [str(entry), "--output-dir", "results/another_training_run"])
    def capture(self, **kwargs):
        assert kwargs["checkpoint"] is None and kwargs["save_dir"] is None
        assert kwargs["output_dir"] == "results/another_training_run"
    with patch.object(PulseCLS, "infer", capture):
        runpy.run_path(str(entry), run_name="__main__")


def test_cli_data_override_resolves_length_policy_without_original_dataset(tmp_path, monkeypatch):
    entry = REPO_ROOT / "infer/infer.py"
    data = tmp_path / "data.yaml"
    data.write_text(json.dumps({"root": ".", "root_relative_to_yaml": True,
        "num_classes": 2, "features": ["CF_MHz", "PW_us", "PRI_us"]}))
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(json.dumps({"task": "classify", "data": "missing/data.yaml",
        "model": str(REPO_ROOT / "configs/model/cnn1d.yaml"),
        "output_dir": str(tmp_path / "out"), "sequence_length": 32,
        "length_policy": {"short": "zero", "long": "head"}}))
    monkeypatch.setattr("sys.argv", [str(entry), "--pipeline", str(pipeline),
        "--data", str(data), "--short-policy", "repeat", "--device", "cpu"])
    seen = []
    def capture(self, **kwargs):
        assert kwargs["data"] == str(data)
        assert kwargs["length_policy"] == {"short": "repeat", "long": "head"}
        cfg = get_cfg(kwargs["pipeline"], overrides={"data": kwargs["data"], "device": kwargs["device"]})
        assert cfg["data_cfg"]["num_classes"] == 2
        seen.append(kwargs)
    with patch.object(PulseCLS, "infer", capture):
        runpy.run_path(str(entry), run_name="__main__")
    assert len(seen) == 1
