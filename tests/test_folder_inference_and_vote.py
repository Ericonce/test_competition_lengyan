import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from pulsecls import PulseCLS
from pulsecls.cfg import REPO_ROOT, get_cfg
from pulsecls.data.dataset import PulseArrayDataset, load_sample
from pulsecls.data.inference import InferenceCSVDataset, find_inference_csvs, read_labels
from pulsecls.data.synthetic import SyntheticRadarGenerator
from pulsecls.engine.voter import vote_predictions
from pulsecls.nn.tasks import build_model


@pytest.fixture
def generated(tmp_path):
    engine = SyntheticRadarGenerator(output_dir=str(tmp_path / "train_pool"),
        test_output_dir=str(tmp_path / "test_pool"), num_emitters=3, sequences_per_emitter=4,
        test_num_sequences=11, save_clean=True, sequence={"length_range": [12, 25]},
        overlap={"enabled": False}, observation={"enabled": False})
    engine.generate()
    engine.generate_test()
    return engine, tmp_path / "train_pool", tmp_path / "test_pool"


def checkpoint(tmp_path, train_root, kind="multigrain_cnn", prediction=0):
    model_cfg = {"type": kind, "in_channels": 3, "hidden_channels": [8, 16],
                 "stream_embedding_dim": 8, "contrast_dim": 4, "dropout": 0.,
                 "d_model": 4, "depth": 1, "large_kernel": 5, "small_kernel": 3,
                 "patch_size": 4, "patch_stride": 2}
    cfg = get_cfg(REPO_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml", overrides={
        "data": train_root / "data.yaml", "sequence_length": 16, "device": "cpu",
        "model_cfg": model_cfg, "output_dir": tmp_path / f"output_{kind}"})
    model = build_model(model_cfg, cfg["data_cfg"])
    # Deterministic checkpoint predictions allow exact evaluation of voting/CSV alignment.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        head = model.classifier if kind == "multigrain_cnn" else model.core.classifier
        head.bias[prediction] = 4.
    path = tmp_path / f"{kind}_{prediction}.pt"
    torch.save({"model_cfg": model_cfg, "data_cfg": cfg["data_cfg"],
                "mean": np.array([4000., 20., 600.]), "std": np.array([1000., 10., 200.]),
                "model_state_dict": model.state_dict()}, path)
    return cfg, path


def test_test_set_is_flat_reuses_identities_and_independent_rng(generated):
    engine, train, test = generated
    paths = find_inference_csvs(test)
    assert len(paths) == 11 and not (test / "data.yaml").exists()
    assert not any(p.is_dir() for p in test.iterdir())
    assert (test / "emitters.json").read_bytes() == (train / "emitters.json").read_bytes()
    labels = read_labels(test / "labels.txt", paths, 3)
    assert sorted(np.bincount(labels).tolist()) == [3, 4, 4]
    seen = {p.read_bytes() for p in train.glob("*/*.csv")}
    assert all(p.read_bytes() not in seen for p in paths)
    before = {p.name: p.read_bytes() for p in paths}
    (test / "stale.csv").write_text("old")
    engine.generate_test()
    assert not (test / "stale.csv").exists()
    assert {p.name: p.read_bytes() for p in find_inference_csvs(test)} == before


def test_generator_rejects_mismatched_identity_config_and_training_target(generated):
    engine, train, test = generated
    engine.config["seed"] += 1
    with pytest.raises(ValueError, match="snapshot differs"):
        engine.generate_test()
    engine.config["seed"] -= 1
    with pytest.raises(ValueError, match="separate"):
        engine.generate_test(train)
    with pytest.raises(ValueError, match="ASCII"):
        engine.generate_test(test.parent / "中文目录")


def test_failed_test_regeneration_preserves_old_data(generated, monkeypatch):
    engine, _, test = generated
    original = (test / "0.csv").read_bytes()
    def fail(*args):
        raise RuntimeError("generation failure")
    monkeypatch.setattr(engine, "_make_sequence", fail)
    with pytest.raises(RuntimeError, match="generation failure"):
        engine.generate_test()
    assert (test / "0.csv").read_bytes() == original


def test_lazy_loader_matches_training_preprocessing_without_pri_anchoring(tmp_path):
    sample = tmp_path / "0.csv"
    sample.write_text("CF_MHz,PW_us,PRI_us\n4000,20,600\n4200,30,900\n", encoding="utf-8")
    cfg = {"features": ["CF_MHz", "PW_us", "PRI_us"], "sequence_length": 4,
           "length_policy": {"short": "zero", "long": "head"}}
    raw = load_sample(sample, 4, cfg["features"], cfg["length_policy"])
    np.testing.assert_array_equal(raw[:, 2], [600, 900, 0, 0])
    mean, std = np.array([4000, 20, 600]), np.array([1000, 10, 200])
    lazy = InferenceCSVDataset([sample], cfg, mean, std)
    eager = PulseArrayDataset(raw[None], np.array([0]), mean, std, zero_padding=True)
    torch.testing.assert_close(lazy[0][0], eager[0][0])


def test_labels_are_filename_keyed_and_validate_exact_coverage(tmp_path):
    paths = [tmp_path / "10.csv", tmp_path / "2.csv"]
    file = tmp_path / "labels.txt"
    file.write_text("2.csv\t1\n10.csv\t0\n")
    np.testing.assert_array_equal(read_labels(file, paths, 2), [0, 1])
    assert read_labels(None, paths, 2) is None
    for text in ("2.csv 1\n", "2.csv 1\n10.csv 0\n10.csv 1\n", "2.csv 1\n10.csv 2\n"):
        file.write_text(text)
        with pytest.raises(ValueError):
            read_labels(file, paths, 2)


def test_folder_inference_csv_optional_evaluation_and_no_training_selection(generated, tmp_path, monkeypatch, capsys):
    _, train, test = generated
    cfg, weights = checkpoint(tmp_path, train)
    def unexpected(*args, **kwargs):
        raise AssertionError("Folder inference must not sample/split training data")
    monkeypatch.setattr("pulsecls.data.selection.select_records", unexpected)
    output = tmp_path / "predictions"
    model = PulseCLS()
    labeled = model.infer(pipeline=cfg, checkpoint=weights, source=test,
                          labels_file=test / "labels.txt", save_dir=output)
    frame = pd.read_csv(output / "predictions.csv")
    assert labeled["num_sequences"] == 11 and labeled["evaluated"]
    assert "macro_f1" in labeled and "correct" in frame and "label" in frame
    assert frame.filename.tolist() == [f"{i}.csv" for i in range(11)]
    assert (frame.prediction == 0).all()
    terminal = capsys.readouterr().out
    assert "Read input CSV" in terminal and "Inference" in terminal
    assert "11/11" in terminal and "Write predictions.csv" in terminal
    unlabeled = model.infer(pipeline=cfg, checkpoint=weights, source=test, save_dir=output)
    assert not unlabeled["evaluated"] and "macro_f1" not in unlabeled
    assert "label" not in pd.read_csv(output / "predictions.csv")
    assert not (output / "metrics.json").exists()
    assert not (output / "test_confusion_matrix.csv").exists()


def test_vote_plurality_and_main_model_tie_rules():
    chosen, counts, tied = vote_predictions([[0, 0, 2], [1, 1, 1], [1, 2, 0]], 0, 3)
    np.testing.assert_array_equal(chosen, [1, 0, 2])
    np.testing.assert_array_equal(counts, [2, 1, 1])
    np.testing.assert_array_equal(tied, [False, True, True])
    # Main isn't a highest-vote candidate: preserve plurality, use smallest tied class.
    chosen, _, _ = vote_predictions([[2], [1], [1], [0], [0]], 0, 3)
    assert chosen.tolist() == [0]


def test_folder_inference_uses_checkpoint_classes_when_training_data_was_overridden(generated, tmp_path):
    _, train, test = generated
    cfg, weights = checkpoint(tmp_path, train)
    # A train(data=small_data_yaml) run can use the original larger-dataset pipeline.
    cfg["data_cfg"]["num_classes"] = 100
    cfg["data_cfg"]["features"] = ["old_cf", "old_pw", "old_toa"]
    cfg["inference"]["progress"] = False
    model = PulseCLS()
    result = model.infer(pipeline=cfg, checkpoint=weights, source=test,
                         labels_file=test / "labels.txt", save_dir=tmp_path / "small_inference")
    assert result["evaluated"] and result["num_sequences"] == 11
    assert model.predictor.num_classes == 3
    assert model.predictor.data_cfg["features"] == ["CF_MHz", "PW_us", "PRI_us"]
    assert model.predictor.data_cfg["sequence_length"] == 16
    assert (pd.read_csv(result["predictions_csv"]).prediction == 0).all()
    with pytest.raises(ValueError, match="pipeline classes=100, checkpoint classes=3"):
        model.predict(pipeline=cfg, checkpoint=weights, split="val")


def test_two_different_model_architectures_vote_and_evaluate(generated, tmp_path, monkeypatch):
    _, train, test = generated
    pipelines, models = [], []
    for kind, prediction in (("multigrain_cnn", 0), ("modern_tcn", 1)):
        cfg, weights = checkpoint(tmp_path, train, kind, prediction)
        # Voting also accepts resolved pipeline dictionaries through its normal loader.
        path = tmp_path / f"{kind}.yaml"
        source = {"data": str(train / "data.yaml"), "model": str(REPO_ROOT / "configs/model/cnn1d.yaml"),
                  "model_overrides": cfg["model_cfg"], "sequence_length": 16,
                  "length_policy": {"short": "zero", "long": "head"}, "train": {"device": "cpu"},
                  "output_dir": str(tmp_path / kind)}
        path.write_text(yaml.safe_dump(source), encoding="utf-8")
        models.append({"name": kind, "pipeline": str(path), "checkpoint": str(weights)})
    preset = tmp_path / "augment.yaml"
    preset.write_text("enabled: true\nmethods:\n  jittering:\n    enabled: true\n    std: 0.08\n", encoding="utf-8")
    from pulsecls.engine.predictor import BasePredictor
    predict_paths = BasePredictor.predict_paths
    seen = []
    def checked_prediction(self, paths):
        assert self.pipeline["inference"]["batch_size"] == 2
        assert self.pipeline["train"]["batch_size"] != 2
        assert self.pipeline["augment"]["methods"]["jittering"]["std"] == .08
        seen.append(self.pipeline)
        return predict_paths(self, paths)
    monkeypatch.setattr(BasePredictor, "predict_paths", checked_prediction)
    config = {"source": str(test), "labels_file": str(test / "labels.txt"),
              "output_dir": str(tmp_path / "vote"), "main_model": "modern_tcn", "device": "cpu", "models": models,
              "batch_size": 2, "augment": {"preset": preset}}
    result = PulseCLS().infer_vote(config)
    frame = pd.read_csv(tmp_path / "vote/predictions.csv")
    assert (frame.multigrain_cnn_prediction == 0).all()
    assert (frame.modern_tcn_prediction == 1).all()
    assert (frame.prediction == 1).all() and frame.tied.all()
    assert (frame.vote_count == 1).all() and (frame.vote_fraction == .5).all()
    assert result["evaluated"] and "macro_f1" in result
    assert len(seen) == 2


@pytest.mark.parametrize("kind", ["multigrain_cnn", "modern_tcn"])
@pytest.mark.parametrize("target", [2, 5])
def test_pipeline_class_count_controls_output_without_changing_learned_weights(generated, tmp_path, kind, target):
    _, train, test = generated
    cfg, weights = checkpoint(tmp_path, train, kind=kind, prediction=2)
    cfg["inference"].update(num_classes=target, progress=False)
    model = PulseCLS()
    result = model.infer(pipeline=cfg, checkpoint=weights, source=test,
                         save_dir=tmp_path / f"adapted_{kind}_{target}")
    assert result["num_classes"] == model.predictor.data_cfg["num_classes"] == target
    assert model.predictor.checkpoint_num_classes == 3
    saved = torch.load(weights, map_location="cpu", weights_only=False)["model_state_dict"]
    for name, value in model.predictor.model.state_dict().items():
        torch.testing.assert_close(value, saved[name])
    frame = pd.read_csv(result["predictions_csv"])
    if target == 2:
        assert (frame.prediction == 0).all()
        np.testing.assert_allclose(frame.confidence, .5)
    else:
        assert (frame.prediction == 2).all()
        raw = torch.tensor([[0., 0., 4.]])
        probs = torch.softmax(model.predictor._adapt_logits(raw), 1)
        torch.testing.assert_close(probs[:, :3], torch.softmax(raw, 1))
        assert (probs[:, 3:] == 0).all()


@pytest.mark.parametrize("target", [2, 5])
def test_class_prior_correction_matches_configured_output_size(generated, tmp_path, target):
    _, train, _ = generated
    cfg, weights = checkpoint(tmp_path, train, prediction=2)
    saved = torch.load(weights, map_location="cpu", weights_only=False)
    saved["train_class_counts"] = [1, 2, 100]
    torch.save(saved, weights)
    cfg["inference"].update(num_classes=target, prior_alpha=.6, progress=False)
    model = PulseCLS()
    model.infer(pipeline=cfg, checkpoint=weights, source=train / "0",
                save_dir=tmp_path / f"prior_{target}")
    assert len(model.predictor.log_priors) == target
    expected = torch.tensor([[0., 0., 4.]])
    expected = model.predictor._adapt_logits(expected) - .6 * model.predictor.log_priors
    expected_score, expected_label = expected.softmax(1).max(1)
    frame = pd.read_csv(tmp_path / f"prior_{target}/predictions.csv")
    np.testing.assert_allclose(frame.confidence, expected_score.item(), rtol=1e-6)
    assert (frame.prediction == expected_label.item()).all()


def test_voting_uses_pipeline_output_space_with_different_checkpoint_widths(generated, tmp_path):
    _, train, test = generated
    members = []
    for kind, width, prediction in (("multigrain_cnn", 3, 2), ("modern_tcn", 4, 3)):
        cfg, path = checkpoint(tmp_path, train, kind=kind, prediction=2)
        if width == 4:
            data_cfg = {**cfg["data_cfg"], "num_classes": width}
            network = build_model(cfg["model_cfg"], data_cfg)
            with torch.no_grad():
                for parameter in network.parameters():
                    parameter.zero_()
                network.core.classifier.bias[prediction] = 4.
            torch.save({"data_cfg": data_cfg, "model_cfg": cfg["model_cfg"],
                        "model_state_dict": network.state_dict(), "mean": [4000., 20., 600.],
                        "std": [1000., 10., 200.]}, path)
        pipeline_path = tmp_path / f"vote_{kind}.yaml"
        pipeline_path.write_text(yaml.safe_dump({
            "data": str(train / "data.yaml"), "model": str(REPO_ROOT / "configs/model/cnn1d.yaml"),
            "sequence_length": 16, "length_policy": {"short": "zero", "long": "head"},
            "train": {"device": "cpu"}, "output_dir": str(tmp_path / kind)}), encoding="utf-8")
        members.append({"name": kind, "pipeline": str(pipeline_path), "checkpoint": str(path)})
    result = PulseCLS().infer_vote({"source": str(test), "output_dir": str(tmp_path / "mixed_vote"),
        "main_model": "modern_tcn", "num_classes": 5, "progress": False, "device": "cpu", "models": members})
    assert result["num_classes"] == 5
    frame = pd.read_csv(result["predictions_csv"])
    assert (frame.multigrain_cnn_prediction == 2).all()
    assert (frame.modern_tcn_prediction == 3).all()
    assert (frame.prediction == 3).all()
