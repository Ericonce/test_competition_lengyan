from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from pulsecls.cfg import REPO_ROOT, get_cfg
from pulsecls.engine.trainer import BaseTrainer
from pulsecls.losses.classification import build_classification_loss


PIPELINE = REPO_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml"


@pytest.mark.parametrize("smoothing", [0, 0.1, 1.0])
def test_direct_parameter_overrides_yaml_nested_and_dotted_loss(smoothing):
    baseline = get_cfg(PIPELINE)
    baseline["loss"]["ce"]["label_smoothing"] = 0.3
    overrides = {"loss": {"ce": {"label_smoothing": 0.4}},
                 "loss.ce.label_smoothing": 0.5, "labelsmooth": smoothing}
    original = deepcopy(overrides)
    trainer = BaseTrainer(baseline, **overrides)
    assert trainer.loss_cfg["ce"]["label_smoothing"] == smoothing
    assert baseline["loss"]["ce"]["label_smoothing"] == 0.3
    assert overrides == original


def test_omitted_or_none_parameter_inherits_yaml(tmp_path):
    import yaml
    pipeline = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))
    pipeline["loss"]["ce"]["label_smoothing"] = 0.2
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(pipeline), encoding="utf-8")
    assert get_cfg(path)["loss"]["ce"]["label_smoothing"] == 0.2
    assert get_cfg(path, {"labelsmooth": None})["loss"]["ce"]["label_smoothing"] == 0.2
    assert get_cfg(path, {"labelsmooth": 0})["loss"]["ce"]["label_smoothing"] == 0


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_smoothing_fails_before_loading_training_data(value):
    with pytest.raises(ValueError, match="label_smoothing"):
        get_cfg(PIPELINE, {"labelsmooth": value})
    with pytest.raises(ValueError, match="label_smoothing"):
        get_cfg(PIPELINE, {"loss": {"ce": {"label_smoothing": value}}})


@pytest.mark.parametrize("smoothing", [0.0, 0.1])
@pytest.mark.parametrize("prior_alpha", [0.0, 0.7])
@pytest.mark.parametrize("class_weights", [None, "balanced"])
def test_smoothing_reaches_both_cross_entropy_paths_and_backpropagates(smoothing, prior_alpha, class_weights):
    labels = np.array([0, 0, 0, 1, 1, 2])
    logits = torch.tensor([[3.0, -0.5, 1.0], [0.5, 2.0, -1.0], [1.0, 0.0, 3.0]], requires_grad=True)
    targets = torch.tensor([0, 1, 2])
    criterion = build_classification_loss(
        {"label_smoothing": smoothing, "prior_alpha": prior_alpha, "class_weights": class_weights},
        labels, 3, torch.device("cpu"))
    counts = torch.tensor([3.0, 2.0, 1.0])
    adjusted = logits + prior_alpha * counts.log()
    expected = F.cross_entropy(adjusted, targets, weight=criterion.weight, label_smoothing=smoothing)
    actual = criterion(logits, targets)
    torch.testing.assert_close(actual, expected)
    if smoothing == 0:
        torch.testing.assert_close(actual, F.cross_entropy(adjusted, targets, weight=criterion.weight))
    else:
        assert not torch.isclose(actual, F.cross_entropy(adjusted, targets, weight=criterion.weight))
    actual.backward()
    assert torch.isfinite(logits.grad).all()
