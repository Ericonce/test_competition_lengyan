import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from pulsecls import PulseCLS
from pulsecls.cfg import REPO_ROOT, get_cfg
from pulsecls.data.dataset import load_split_arrays
from pulsecls.data.selection import select_records
from pulsecls.data.synthetic import SyntheticRadarGenerator
from pulsecls.engine.trainer import BaseTrainer


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "raw"
    SyntheticRadarGenerator(num_emitters=2, sequences_per_emitter=20,
        overlap={"enabled": False}, observation={"enabled": False},
        noise={"enabled": False}, sequence={"random_length": False, "length": 16}).generate(root)
    cfg = get_cfg(REPO_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml", overrides={"data": root / "data.yaml",
        "dataset": {"sample_ratio": 1., "train_ratio": .8, "val_ratio": .2, "seed": 2026, "stratified": True}})["data_cfg"]
    cfg["sequence_length"] = 16
    return cfg


def test_sample_before_split_exact_counts_and_reproducibility(data):
    data["selection"].update(sample_ratio=.5, train_ratio=.8, val_ratio=.2)
    parts, info = select_records(data)
    assert (info["total_count"], info["sampled_count"], info["train_count"], info["val_count"]) == (40, 20, 16, 4)
    assert len(parts["all"]) == 20
    assert not set(parts["train"]) & set(parts["val"])
    assert set(parts["train"]) | set(parts["val"]) == set(parts["all"])
    assert select_records(data)[0] == parts
    # Changing only the split ratio must preserve the first-stage sample pool.
    data["selection"].update(train_ratio=.6, val_ratio=.4)
    changed, _ = select_records(data)
    assert changed["all"] == parts["all"]
    assert len(changed["train"]) == 12 and len(changed["val"]) == 8


def test_stratified_preserves_class_proportions_and_sampling_seed(data):
    data["selection"].update(sample_ratio=.5)
    parts, _ = select_records(data)
    for name in ("train", "val", "all"):
        counts = np.bincount([y for _, y in parts[name]])
        assert counts[0] == counts[1]
    data["selection"]["seed"] += 1
    assert select_records(data)[0]["all"] != parts["all"]


def test_legacy_synthetic_metadata_uses_total_pool_instead_of_old_indices(data):
    data.pop("all")
    data.update(train="splits/old_train.json", val="splits/old_val.json", test="splits/old_test.json")
    parts, info = select_records(data)
    assert info["total_count"] == 40 and len(parts["train"]) + len(parts["val"]) == 40


def test_numbered_synthetic_sampling_does_not_enumerate_or_stat_unselected_csvs(data):
    import pulsecls.data.file_index as index_module
    data["selection"].update(sample_ratio=.25)
    checked = []
    original = Path.stat
    def stat(path, *args, **kwargs):
        if path.suffix == ".csv":
            checked.append(path)
        return original(path, *args, **kwargs)
    # Metadata counts suffice: a CSV-directory scan would fail this test.
    scan = index_module.os.scandir
    def scandir(path):
        if Path(path).name.isdecimal():
            raise AssertionError("Unselected class directory was enumerated")
        return scan(path)
    with patch.object(Path, "stat", stat), patch.object(index_module.os, "scandir", scandir):
        parts, info = select_records(data)
    assert info["total_count"] == 40 and info["sampled_count"] == 10
    assert set(checked) == {path for path, _ in parts["all"]}


def test_loading_only_selected_csvs_and_reusing_the_same_selection_plan(data):
    import pulsecls.data.dataset as dataset_module
    data["selection"].update(sample_ratio=.5)
    data["cache"] = False
    plan = select_records(data)
    with patch.object(dataset_module, "load_sample", wraps=dataset_module.load_sample) as reader, \
         patch("pulsecls.data.selection.select_records", side_effect=AssertionError("Resampling during loading")):
        a, _, _ = load_split_arrays(data, "train", selection_plan=plan)
        b, _, _ = load_split_arrays(data, "val", selection_plan=plan)
    assert len(a) == 16 and len(b) == 4
    assert reader.call_count == 20
    assert {call.args[0] for call in reader.call_args_list} == {path for path, _ in plan[0]["all"]}


def test_ordinary_data_reuses_filename_index_without_rescanning_classes(data):
    import pulsecls.data.file_index as index_module
    root = Path(data["root"])
    (root / "generation_config.yaml").unlink()
    (root / "summary.json").unlink()
    data["selection"].update(sample_ratio=.5)
    first, _ = select_records(data)
    scan = index_module.os.scandir
    def scandir(path):
        if Path(path).name.isdecimal():
            raise AssertionError("Cached ordinary data should not scan CSV directories again")
        return scan(path)
    with patch.object(index_module.os, "scandir", scandir):
        second, info = select_records(data)
    assert second == first and info["index_source"] == "cached file index"


def test_trainer_samples_once_for_both_loaders(data, tmp_path):
    import pulsecls.engine.trainer as trainer_module
    with patch.object(trainer_module, "select_records", wraps=select_records) as selector:
        trainer = BaseTrainer(training_cfg(data, tmp_path / "sample_once"))
        trainer._build_loaders()
    assert selector.call_count == 1


def test_cache_changes_with_sampling_split_or_source_changes(data):
    data["selection"].update(sample_ratio=.5)
    a, _, first_paths = load_split_arrays(data, "train")
    assert len(a) == 16
    data["selection"]["sample_ratio"] = 1.
    b, _, second_paths = load_split_arrays(data, "train")
    assert len(b) == 32
    assert set(first_paths) != set(second_paths)
    assert len(list((Path(data["root"]) / ".pulsecls_cache").glob("train*.npz"))) == 2
    file = Path(data["root"]) / "0/0.csv"
    file.write_text(file.read_text() + "\n")
    load_split_arrays(data, "train")
    assert len(list((Path(data["root"]) / ".pulsecls_cache").glob("train*.npz"))) == 3


def training_cfg(data, folder, epochs=1):
    return {"task": "classify", "data_cfg": data, "output_dir": str(folder),
        "model_cfg": {"type": "cnn1d", "in_channels": 3, "hidden_channels": [8, 16],
                      "stream_embedding_dim": 8, "pulse_embedding_dim": 8, "contrast_dim": 4, "dropout": .2},
        "train": {"epochs": epochs, "device": "cpu", "batch_size": 4, "use_amp": False,
                  "early_stopping_patience": 0}, "augment": {"enabled": False}, "loss": {"ce": {"weight": 1.}}}


def test_train_only_skips_validation_scheduler_and_early_stop(data, tmp_path):
    data["selection"].update(sample_ratio=.5, train_ratio=1., val_ratio=0.)
    folder = tmp_path / "train_only"
    folder.mkdir()
    (folder / "best_model.pt").write_bytes(b"stale validation weight")
    cfg = training_cfg(data, folder, epochs=3)
    calls = []
    original = BaseTrainer._run_epoch
    def run(self, model, loader, *args, **kwargs):
        calls.append(loader)
        return original(self, model, loader, *args, **kwargs)
    with patch.object(BaseTrainer, "_run_epoch", run), patch.object(torch.optim.lr_scheduler.ReduceLROnPlateau, "step") as scheduler_step:
        result = PulseCLS().train(pipeline=cfg)
    assert scheduler_step.call_count == 0
    assert len(calls) == 4 and all(loader is calls[0] for loader in calls)
    assert result["epoch"] == 3 and result["validation_enabled"] is False
    assert result["evaluation_split"] == "train"
    assert not (folder / "best_model.pt").exists()
    assert (folder / "last_model.pt").exists() and (folder / "final_model.pt").exists()
    history = pd.read_csv(folder / "training_history.csv")
    assert len(history) == 3 and not any(key.startswith("val_") for key in history.columns)
    membership = json.loads((folder / "dataset_selection.json").read_text())
    assert membership["sampled_count"] == membership["train_count"] == 20 and membership["val_count"] == 0
    assert sum(torch.load(folder / "final_model.pt", weights_only=False)["train_class_counts"]) == 20
    # Automatic inference chooses final_model.pt and the train split in this mode.
    metrics = PulseCLS().predict(pipeline=cfg)
    assert np.isfinite(metrics["macro_f1"])
    with pytest.raises(ValueError, match="Validation is disabled"):
        PulseCLS().predict(pipeline=cfg, split="val")


def test_train_only_resume_keeps_exact_continuation(data, tmp_path):
    data["selection"].update(sample_ratio=.5, train_ratio=1., val_ratio=0.)
    continuous = training_cfg(data, tmp_path / "continuous", 3)
    PulseCLS().train(pipeline=continuous)
    initial = training_cfg(data, tmp_path / "initial", 1)
    PulseCLS().train(pipeline=initial)
    resumed = training_cfg(data, tmp_path / "resumed", 2)
    PulseCLS().train(pipeline=resumed, resume=True, resume_checkpoint=tmp_path / "initial/last_model.pt")
    expected = torch.load(tmp_path / "continuous/final_model.pt", weights_only=False)
    actual = torch.load(tmp_path / "resumed/final_model.pt", weights_only=False)
    assert actual["epoch"] == 3
    for key in expected["model_state_dict"]:
        torch.testing.assert_close(actual["model_state_dict"][key], expected["model_state_dict"][key], rtol=0, atol=0)


def test_sampling_singleton_last_batch_trains_without_dropping_data(data, tmp_path):
    data["selection"].update(sample_ratio=.325, train_ratio=1., val_ratio=0.)
    cfg = training_cfg(data, tmp_path / "singleton")
    trainer = BaseTrainer(cfg)
    loader, val, labels, _, _, _ = trainer._build_loaders()
    sizes = [len(y) for _, y, _ in loader]
    assert sum(sizes) == len(labels) == 13 and min(sizes) >= 2 and val is None
    result = trainer.fit()
    assert result["epoch"] == 1 and result["validation_enabled"] is False


def test_code_override_pipeline_selection_priority(data):
    source = REPO_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml"
    cfg = get_cfg(source, overrides={"data": Path(data["root"]) / "data.yaml", "dataset": {"sample_ratio": .5}, "dataset.train_ratio": 1., "dataset.val_ratio": 0.})
    assert cfg["data_cfg"]["selection"]["sample_ratio"] == .5
    assert cfg["data_cfg"]["selection"]["train_ratio"] == 1.
    assert cfg["data_cfg"]["selection"]["val_ratio"] == 0.
    assert get_cfg(cfg)["data_cfg"]["selection"] == cfg["data_cfg"]["selection"]


@pytest.mark.parametrize("override", [{"sample_ratio": 0}, {"sample_ratio": 2}, {"train_ratio": .5},
    {"val_ratio": -1}, {"seed": -1}, {"sample_ratio": True}, {"unknown": 1}])
def test_invalid_selection_is_rejected(data, override):
    data["selection"].update(override)
    with pytest.raises(ValueError):
        select_records(data)


def test_zero_sample_clear_error_and_test_split_removed(data):
    data["selection"]["sample_ratio"] = .001
    with pytest.raises(ValueError, match="zero sequences"):
        select_records(data)
    with pytest.raises(ValueError, match="no test split"):
        load_split_arrays(data, "test")
