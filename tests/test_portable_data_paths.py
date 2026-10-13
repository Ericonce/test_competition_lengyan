from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from pulsecls import cfg
from pulsecls.data.dataset import load_split_arrays
from pulsecls.engine.predictor import BasePredictor


def test_dataset_configs_are_relative_and_cwd_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = cfg.REPO_ROOT / cfg.DEFAULT_CFG_DICT["data"]
    raw = cfg.load_yaml(path)
    assert not Path(raw["root"]).is_absolute()
    pipeline = cfg.get_cfg(cfg.REPO_ROOT / cfg.DEFAULT_CFG_DICT["pipeline"])
    assert Path(pipeline["data_cfg"]["root"]) == path.parent
    portable = cfg.portable_data_cfg(pipeline["data_cfg"])
    assert not Path(portable["root"]).is_absolute()
    assert portable["features"] == ["CF_MHz", "PW_us", "PRI_us"]


def test_legacy_and_relative_paths_follow_new_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "REPO_ROOT", tmp_path)
    expected = tmp_path / "datasets/example"
    assert cfg.remap_repo_path("F:/competition/datasets/example") == expected
    assert cfg.remap_repo_path("datasets/example") == expected
    assert cfg.portable_data_cfg({"root": str(expected)})["root"] == "datasets/example"


def test_chinese_legacy_dataset_paths_follow_ascii_names(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "REPO_ROOT", tmp_path)
    old = "F:/competition/datasets/真实雷达脉冲流数据集/划分后数据/训练集/0/sample_1.csv"
    assert cfg.remap_repo_path(old) == tmp_path / "datasets/real_radar/split_random128/train/0/sample_1.csv"
    assert cfg.normalize_dataset_path("验证集") == "val"


def test_cached_paths_relocate(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "REPO_ROOT", tmp_path)
    root = tmp_path / "datasets/example"
    cache = root / ".pulsecls_cache"
    cache.mkdir(parents=True)
    np.savez_compressed(cache / "val_seq2.npz", data=np.zeros((2, 2, 2)), labels=[0, 0],
                        paths=["val/0/sample_1.csv", "F:/competition/datasets/example/val/0/sample_2.csv"])
    _, _, paths = load_split_arrays({"root": "datasets/example", "val": "val", "sequence_length": 2}, "val")
    assert paths == [str(root / f"val/0/sample_{i}.csv") for i in (1, 2)]


def test_predictor_remaps_checkpoint_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "REPO_ROOT", tmp_path)
    model = torch.nn.Linear(2, 2)
    checkpoint = {"data_cfg": {"root": "F:/competition/datasets/example", "num_classes": 2},
                  "model_cfg": {}, "model_state_dict": model.state_dict(), "mean": [0, 0], "std": [1, 1]}
    pipeline = {"output_dir": str(tmp_path), "train": {"device": "cpu"}, "data_cfg": {}, "model_cfg": {}}
    with patch("pulsecls.engine.predictor.torch.load", return_value=checkpoint), \
         patch.object(BasePredictor, "get_model", return_value=model):
        predictor = BasePredictor(pipeline, tmp_path / "model.pt")
    assert predictor.data_cfg["root"] == str(tmp_path / "datasets/example")
    assert checkpoint["data_cfg"]["root"].startswith("F:/")
