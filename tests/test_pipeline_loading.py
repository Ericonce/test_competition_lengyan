from pathlib import Path

import pytest

from pulsecls.cfg import DEFAULT_CFG_DICT, load_yaml
from pulsecls.utils.config import load_pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "configs/pipeline").glob("*.yaml")))
def test_all_current_pipelines_are_complete_synthetic_configs(path):
    pipeline = load_pipeline(path)
    assert "synthetic" in path.stem
    assert pipeline["data_cfg"]["features"] == ["CF_MHz", "PW_us", "PRI_us"]
    assert pipeline["model_cfg"]["in_channels"] == 3
    assert pipeline["data_cfg"]["sequence_length"] == pipeline["sequence_length"]
    assert pipeline["data_cfg"]["length_policy"] == pipeline["length_policy"]
    assert pipeline["data_cfg"]["selection"] == pipeline["dataset"]
    assert pipeline["inference"]["progress"] is True
    assert pipeline["inference"]["labels_file"] is None
    assert pipeline["train"]["device"] == "auto"
    source = load_yaml(path)
    assert pipeline["train"]["num_classes"] == source["train"]["num_classes"]
    assert pipeline["inference"]["num_classes"] == source["inference"]["num_classes"]
    assert (ROOT / "train" / path.stem / "train.py").is_file()
    assert (ROOT / "infer" / path.stem / "infer.py").is_file()


def test_complete_modular_method_keeps_iq_and_contrastive_objectives():
    pipeline = load_pipeline(ROOT / "configs/pipeline/modular_multigrain_iq_synthetic.yaml")
    assert pipeline["model_cfg"]["type"] == "modular_resnet1d_hypothesis"
    assert pipeline["model_cfg"]["hypothesis_for_classification"] is True
    assert pipeline["loss"]["objective"] == "multigrain"
    assert pipeline["loss"]["stream_infonce"]["weight"] > 0
    assert pipeline["loss"]["pulse_infonce"]["weight"] > 0


def test_default_and_voting_references_point_to_current_synthetic_pipelines():
    assert DEFAULT_CFG_DICT["pipeline"] == "configs/pipeline/cnn_softmax_synthetic.yaml"
    vote = load_yaml(ROOT / "configs/inference/vote.yaml")
    for member in vote["models"]:
        assert "synthetic" in member["pipeline"]
        assert (ROOT / member["pipeline"]).is_file()
