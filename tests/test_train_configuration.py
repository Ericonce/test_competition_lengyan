import ast
import runpy
from pathlib import Path
from unittest.mock import patch

import pytest

from pulsecls import PulseCLS
from pulsecls.cfg import REPO_ROOT, deep_update, get_cfg
from pulsecls.engine.trainer import BaseTrainer

PIPELINE = REPO_ROOT / "configs/pipeline/mptsnet1d_synthetic.yaml"


@pytest.fixture
def alternative_data(tmp_path):
    import yaml
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({"root": ".", "root_relative_to_yaml": True, "all": ".",
                    "num_classes": 3, "features": ["CF_MHz", "PW_us", "PRI_us"], "cache": True}), encoding="utf-8")
    return path


def test_all_sections_override_yaml_without_losing_defaults(alternative_data):
    baseline = get_cfg(PIPELINE)
    cfg = get_cfg(PIPELINE, overrides={
        "data": alternative_data,
        "epochs": 7, "train": {"epochs": 4, "optimizer_foreach": True},
        "model": {"dropout": .7}, "data_cfg": {"cache": False},
        "augment": {"methods": {"jittering": {"snr_db": 25}}},
        "loss": {"ce": {"class_weights": None}},
        "inference": {"split": "train"},
        "model.dropout": .8,
    })
    assert cfg["train"]["epochs"] == 7
    assert cfg["train"]["optimizer_foreach"] is True
    assert cfg["model_cfg"]["dropout"] == .8
    assert cfg["model_cfg"]["depth"] == baseline["model_cfg"]["depth"]
    assert cfg["data_cfg"]["num_classes"] == 3
    assert cfg["data_cfg"]["cache"] is False
    assert cfg["augment"]["methods"]["jittering"]["snr_db"] == 25
    assert cfg["augment"]["methods"]["scaling"] == baseline["augment"]["methods"]["scaling"]
    assert cfg["loss"]["ce"]["class_weights"] is None
    assert cfg["inference"]["split"] == "train"


def test_data_reference_preserves_resolved_overrides(alternative_data):
    cfg = get_cfg(PIPELINE, overrides={"model": {"dropout": .65}})
    cfg = get_cfg(cfg, overrides={"data": alternative_data})
    assert cfg["model_cfg"]["dropout"] == .65


def test_trainer_accepts_paths_and_overrides():
    trainer = BaseTrainer(PIPELINE, epochs=3, device="cpu", model={"dropout": .6})
    assert trainer.train_cfg["epochs"] == 3
    assert trainer.model_cfg["dropout"] == .6
    assert trainer.train_cfg["resume"] is False


def test_reject_unknown_parameter():
    with pytest.raises(ValueError, match="epohcs"):
        get_cfg(PIPELINE, overrides={"epohcs": 3})


def test_visualization_defaults_and_code_priority():
    assert get_cfg(PIPELINE)["train"]["visualize"] is True
    baseline = get_cfg(PIPELINE, {"train": {"visualize": False}})
    assert get_cfg(baseline, {"visualize": None})["train"]["visualize"] is False
    assert get_cfg(baseline, {"visualize": True})["train"]["visualize"] is True
    assert get_cfg(PIPELINE, {"train.visualize": True, "visualize": False})["train"]["visualize"] is False


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_visualization_requires_a_boolean(value):
    with pytest.raises(TypeError, match="visualize"):
        get_cfg(PIPELINE, {"visualize": value})


def test_optional_none_retains_yaml_and_nested_none_clears():
    cfg = get_cfg(PIPELINE, overrides={"epochs": None, "train": {"resume_checkpoint": None}})
    assert cfg["train"]["epochs"] == get_cfg(PIPELINE)["train"]["epochs"]
    assert cfg["train"]["resume_checkpoint"] is None


@pytest.mark.parametrize("overrides,expected", [
    ({"patience": 10}, 10),
    ({"train": {"patience": 11}}, 11),
    ({"train.patience": 12}, 12),
    ({"early_stopping_patience": 6, "patience": 10}, 10),
    ({"train": {"early_stopping_patience": 6, "patience": 10}}, 10),
    ({"train.early_stopping_patience": 6, "train.patience": 10}, 10),
    ({"train": {"patience": 8}, "train.patience": 9, "patience": 10}, 10),
    ({"patience": None, "early_stopping_patience": 9}, 9),
])
def test_patience_alias_reaches_effective_training_setting_without_mutating_input(overrides, expected):
    from copy import deepcopy
    original = deepcopy(overrides)
    cfg = get_cfg(PIPELINE, overrides=overrides)
    assert cfg["train"]["early_stopping_patience"] == expected
    assert "patience" not in cfg["train"]
    assert overrides == original
    trainer = BaseTrainer(PIPELINE, **overrides)
    assert trainer.train_cfg["early_stopping_patience"] == expected


def test_yaml_patience_and_train_overrides_obey_code_priority(tmp_path):
    import yaml
    source = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))
    source["train"]["patience"] = 13
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    assert get_cfg(path)["train"]["early_stopping_patience"] == 13
    assert get_cfg(path, overrides={"early_stopping_patience": 7})["train"]["early_stopping_patience"] == 7
    source["train_overrides"] = {"early_stopping_patience": 8}
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    assert get_cfg(path)["train"]["early_stopping_patience"] == 8
    source["train_overrides"] = {"patience": 14}
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    cfg = get_cfg(path)
    assert cfg["train"]["early_stopping_patience"] == 14
    assert "patience" not in cfg["train"]
    assert get_cfg(cfg, overrides={"patience": 15})["train"]["early_stopping_patience"] == 15


@pytest.mark.parametrize("entry", sorted((REPO_ROOT / "train").rglob("*.py")))
def test_every_training_entry_exposes_common_parameters_with_code_priority(entry, monkeypatch):
    # Generated datasets can be absent while the user regenerates them; this test
    # checks entrypoint wiring rather than depending on a large local data artifact.
    import pulsecls.cfg as config_module
    load_yaml = config_module.load_yaml
    def yaml_source(path):
        if Path(path).name == "data.yaml" and not Path(path).exists():
            return {"root": ".", "root_relative_to_yaml": True, "all": ".",
                    "num_classes": 10 if "_small" in str(path) else 100,
                    "features": ["CF_MHz", "PW_us", "PRI_us"], "cache": True}
        return load_yaml(path)
    monkeypatch.setattr(config_module, "load_yaml", yaml_source)
    tree = ast.parse(entry.read_text(encoding="utf-8"))
    settings = {target.id for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)}
    assert settings == {"PROJECT_ROOT"}
    seen = []
    def capture(self, **kwargs):
        assert {"pipeline", "data", "output_dir", "num_classes", "epochs", "batch_size", "lr",
                "device", "resume", "resume_checkpoint", "sequence_length", "length_policy",
                "dataset", "augment", "labelsmooth", "visualize"} <= set(kwargs)
        cfg = self._resolve_pipeline(**kwargs)
        baseline = get_cfg(kwargs["pipeline"], overrides={"data": kwargs["data"]})
        assert cfg["train"]["num_classes"] == cfg["data_cfg"]["num_classes"] == kwargs["num_classes"]
        assert cfg["train"]["epochs"] == kwargs["epochs"]
        assert cfg["train"]["visualize"] is kwargs["visualize"] is True
        assert cfg["train"]["early_stopping_patience"] == kwargs.get(
            "patience", kwargs.get("early_stopping_patience", baseline["train"]["early_stopping_patience"]))
        for key in ("sampler", "use_amp"):
            assert cfg["train"][key] == kwargs.get(key, kwargs.get("train", {}).get(key, baseline["train"][key]))
        expected_loss = deep_update(baseline["loss"], kwargs.get("loss", {}))
        expected_loss["ce"]["label_smoothing"] = kwargs["labelsmooth"]
        assert cfg["loss"] == expected_loss
        assert cfg["data_cfg"]["selection"] == deep_update(baseline["data_cfg"]["selection"], kwargs.get("dataset", {}))
        augment = kwargs["augment"]
        preset = augment.get("preset")
        expected_augment = deep_update(config_module.load_yaml(preset) if preset is not None else baseline["augment"],
                                       {key: value for key, value in augment.items() if key != "preset"})
        assert cfg["augment"] == expected_augment
        assert cfg["train"]["resume"] is False
        assert kwargs["resume_checkpoint"] == Path(kwargs["output_dir"]) / "last_model.pt"
        assert cfg["model_cfg"]["in_channels"] == 3
        if entry.name == "train_small.py":
            assert kwargs["epochs"] == 50 and kwargs["num_classes"] == 10
            assert Path(kwargs["data"]).parent.name == "synthetic_radar_overlap_small"
        seen.append(cfg)
    with patch.object(PulseCLS, "train", capture):
        runpy.run_path(str(entry), run_name="__main__")
    assert len(seen) == 1


def test_edit_yaml_is_effective_and_explicit_values_win(tmp_path):
    import yaml
    pipeline = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))
    path = tmp_path / "pipeline.yaml"
    pipeline["train"]["epochs"] = 17
    path.write_text(yaml.safe_dump(pipeline), encoding="utf-8")
    assert PulseCLS()._resolve_pipeline(pipeline=path)["train"]["epochs"] == 17
    pipeline["train"]["epochs"] = 19
    path.write_text(yaml.safe_dump(pipeline), encoding="utf-8")
    assert PulseCLS()._resolve_pipeline(pipeline=path)["train"]["epochs"] == 19
    assert PulseCLS()._resolve_pipeline(pipeline=path, epochs=3)["train"]["epochs"] == 3


def test_augment_preset_replacement_with_local_threshold(tmp_path):
    preset = tmp_path / "augment.yaml"
    preset.write_text("enabled: false\nmethods:\n  jittering:\n    enabled: true\n    snr_db: 28\n", encoding="utf-8")
    cfg = get_cfg(PIPELINE, overrides={"augment": {"preset": preset, "methods": {"jittering": {"snr_db": 20}}}})
    assert cfg["augment"]["enabled"] is False
    assert cfg["augment"]["methods"]["jittering"]["snr_db"] == 20


@pytest.mark.parametrize("enabled", [False, True])
def test_training_augment_overrides_yaml_and_master_switch_gates_all_methods(alternative_data, tmp_path, enabled):
    import torch
    import yaml
    from pulsecls.data.augment import PulseAugment

    preset = tmp_path / "augment.yaml"
    preset.write_text(yaml.safe_dump({
        "enabled": not enabled,
        "order": ["scaling", "jittering", "time_shift", "mixup"],
        "methods": {
            "jittering": {"enabled": False, "prob": 1.0, "snr_db": 50.0},
            "time_shift": {"enabled": True, "prob": 1.0, "ratio": 0.1},
            "scaling": {"enabled": True, "prob": 1.0, "std": 0.15},
            "mixup": {"enabled": False, "prob": 1.0, "alpha": 0.2},
        },
    }), encoding="utf-8")
    cfg = get_cfg(PIPELINE, overrides={
        "data": alternative_data,
        "augment": {
            "preset": preset,
            "enabled": enabled,
            "methods": {
                "jittering": {"enabled": True, "snr_db": 25.0},
                "time_shift": {"ratio": 0.25},
                "scaling": {"enabled": False},
                "mixup": {"enabled": True},
            },
            "order": ["time_shift", "jittering", "scaling", "mixup"],
        },
    })
    settings = cfg["augment"]
    assert settings["enabled"] is enabled
    assert settings["methods"]["jittering"] == {"enabled": True, "prob": 1.0, "snr_db": 25.0}
    assert settings["methods"]["time_shift"]["enabled"] is True  # Inherited from YAML.
    assert settings["methods"]["time_shift"]["ratio"] == 0.25
    assert settings["methods"]["scaling"]["enabled"] is False
    assert settings["methods"]["scaling"]["std"] == 0.15  # Unspecified thresholds are inherited.

    augment = PulseAugment(**settings)
    inputs = torch.arange(4 * 3 * 32, dtype=torch.float32).view(4, 3, 32) + 1
    original = inputs.clone()
    labels = torch.tensor([0, 0, 1, 1])
    random_state = torch.random.get_rng_state().clone()
    with patch.object(augment.runner, "run", wraps=augment.runner.run) as run:
        output, mixup = augment.apply(inputs, labels)
    torch.testing.assert_close(inputs, original, rtol=0, atol=0)
    if enabled:
        assert [call.args[0] for call in run.call_args_list] == ["time_shift", "jittering"]
        assert mixup is not None
        assert not torch.equal(output, original)
    else:
        run.assert_not_called()
        assert mixup is None
        torch.testing.assert_close(output, original, rtol=0, atol=0)
        assert torch.equal(torch.random.get_rng_state(), random_state)
