from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pulsecls import PulseCLS
from pulsecls.cfg import REPO_ROOT, get_cfg, load_yaml
from pulsecls.data.inference import InferenceCSVDataset, find_inference_csvs
from pulsecls.data.synthetic import SyntheticRadarGenerator
from pulsecls.nn.classifiers.resnet_softmax import ResNetSoftmaxClassifier
from pulsecls.nn.tasks import build_model


PIPELINE = REPO_ROOT / "configs/pipeline/resnet_softmax_synthetic.yaml"


@pytest.mark.parametrize("model_type", ["resnet_softmax", "resnet1d_softmax"])
@pytest.mark.parametrize("length", [32, 128, 257])
def test_resnet_softmax_supports_three_channels_lengths_and_cross_entropy(model_type, length):
    model = build_model({"type": model_type, "base_channels": 4, "dropout": 0.},
                        {"num_classes": 5, "features": ["CF_MHz", "PW_us", "PRI_us"]})
    assert isinstance(model, ResNetSoftmaxClassifier)
    inputs = torch.randn(2, 3, length, requires_grad=True)
    outputs = model(inputs)
    assert outputs["logits"].shape == (2, 5)
    assert outputs["stream_embedding"].shape == (2, 32)
    assert "stream_contrast" not in outputs and "pulse_contrast" not in outputs
    loss = torch.nn.functional.cross_entropy(outputs["logits"], torch.tensor([0, 4]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(inputs.grad).all()
    assert model.backbone.stem[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    assert model(torch.randn(1, 3, length))["logits"].shape == (1, 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_default_resnet_softmax_backpropagates_on_local_cuda():
    cfg = load_yaml(REPO_ROOT / "configs/model/resnet_softmax.yaml")
    model = build_model(cfg, {"num_classes": 4, "features": ["CF_MHz", "PW_us", "PRI_us"]}).to("cuda")
    logits = model(torch.randn(2, 3, 256, device="cuda"))["logits"]
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3], device="cuda"))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_resnet_train_resume_folder_inference_and_voting(tmp_path):
    train_root, test_root = tmp_path / "train_pool", tmp_path / "test_pool"
    generator = SyntheticRadarGenerator(output_dir=str(train_root), test_output_dir=str(test_root),
        num_emitters=3, sequences_per_emitter=4, test_num_sequences=7,
        overlap={"enabled": False}, observation={"enabled": False}, noise={"enabled": False},
        sequence={"random_length": False, "length": 32})
    generator.generate()
    generator.generate_test()
    cfg = get_cfg(PIPELINE, overrides={
        "data": train_root / "data.yaml", "num_classes": 3,
        "output_dir": tmp_path / "initial", "device": "cpu", "epochs": 1,
        "batch_size": 4, "use_amp": False, "patience": 10, "sequence_length": 32,
        "dataset": {"sample_ratio": 1.0, "train_ratio": .75, "val_ratio": .25},
        "model": {"base_channels": 4, "dropout": 0.},
        "data_cfg": {"cache": False, "progress": False},
        "inference": {"progress": False},
    })
    model = PulseCLS()
    initial = model.train(pipeline=cfg)
    assert np.isfinite(initial["macro_f1"])
    last = tmp_path / "initial/last_model.pt"
    saved = torch.load(last, map_location="cpu", weights_only=False)
    assert saved["model_cfg"]["type"] == "resnet_softmax"
    assert saved["model_state_dict"]["classifier.weight"].shape == (3, 32)
    assert saved["epoch"] == 1
    model.train(pipeline=cfg, resume=True, resume_checkpoint=last,
                output_dir=tmp_path / "resumed", epochs=1)
    resumed = torch.load(tmp_path / "resumed/last_model.pt", map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2 and len(resumed["history"]) == 2
    weights = tmp_path / "resumed/best_model.pt"
    result = model.infer(pipeline=cfg, source=test_root, checkpoint=weights,
                         num_classes=3, save_dir=tmp_path / "inference",
                         labels_file=test_root / "labels.txt", inference={"batch_size": 2, "progress": False})
    frame = pd.read_csv(result["predictions_csv"])
    assert result["evaluated"] and result["num_sequences"] == 7 and result["num_classes"] == 3
    assert frame["confidence"].between(1 / 3 - 1e-6, 1).all()
    paths = find_inference_csvs(test_root, progress=False)
    predictor = model.predictor
    sample = InferenceCSVDataset(paths, predictor.data_cfg, predictor.mean, predictor.std)[0][0]
    with torch.no_grad():
        probabilities = predictor.model(sample[None])["logits"].softmax(dim=1)
    score, label = probabilities.max(dim=1)
    assert frame.iloc[0]["confidence"] == pytest.approx(score.item(), rel=1e-5)
    assert int(frame.iloc[0]["prediction"]) == label.item()
    vote = model.infer_vote({"source": str(test_root), "labels_file": str(test_root / "labels.txt"),
        "output_dir": str(tmp_path / "vote"), "main_model": "resnet_softmax", "num_classes": 3,
        "device": "cpu", "sequence_length": 32, "progress": False,
        "models": [{"name": "resnet_softmax", "pipeline": str(PIPELINE), "checkpoint": str(weights)}]})
    voted = pd.read_csv(vote["predictions_csv"])
    assert vote["evaluated"] and voted["prediction"].tolist() == frame["prediction"].tolist()
