"""Regression checks for removal of dataset-specific experiment configurations."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_dataset_pipeline_and_entrypoints_are_removed():
    for path in (ROOT / "configs/pipeline").rglob("*.yaml"):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "synthetic" in path.stem
        assert "tsrd" not in cfg["data"] and "real_radar" not in cfg["data"]
    for section in ("train", "infer"):
        for path in (ROOT / section).rglob("*.py"):
            assert "tsrd" not in path.as_posix() and "real_radar" not in path.as_posix()
    assert not (ROOT / "configs/data/tsrd_random256.yaml").exists()
    assert not (ROOT / "configs/data/real_radar_random128.yaml").exists()
