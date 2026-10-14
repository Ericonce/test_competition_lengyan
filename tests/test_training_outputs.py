import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn.metrics import f1_score

from pulsecls import PulseCLS
from pulsecls.data.synthetic import SyntheticRadarGenerator
from pulsecls.utils.metrics import save_classification_outputs
from pulsecls.utils.training_outputs import select_tsne_indices


def test_class_accuracy_and_normalized_matrix_handle_absent_classes(tmp_path):
    labels, predictions = np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 2, 2])
    save_classification_outputs(tmp_path, labels, predictions, 4)
    report = json.loads((tmp_path / "val_classification_report.json").read_text())
    assert report["0"]["class_accuracy"] == .5 and report["2"]["class_accuracy"] == 1.
    assert report["3"]["class_accuracy"] is None
    expected = f1_score(labels, predictions, labels=range(4), average=None, zero_division=0)
    np.testing.assert_allclose([report[str(i)]["f1-score"] for i in range(4)], expected)
    matrix = pd.read_csv(tmp_path / "val_confusion_matrix_normalized.csv", index_col=0).to_numpy()
    np.testing.assert_allclose(matrix.sum(axis=1), [1., 1., 1., 0.])
    assert report["1"]["correct_count"] == 1


def test_tsne_sampling_is_balanced_repeatable_and_does_not_change_global_rng():
    labels = np.repeat([0, 1, 2], [50, 3, 10])
    state = np.random.get_state()
    indices = select_tsne_indices(labels, 12, 2026)
    np.testing.assert_array_equal(indices, select_tsne_indices(labels, 12, 2026))
    assert len(set(indices)) == 12
    np.testing.assert_array_equal(np.bincount(labels[indices]), [5, 3, 4])
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    np.testing.assert_array_equal(select_tsne_indices(labels, 0, 2026), np.arange(len(labels)))
    with pytest.raises(ValueError, match="all represented classes"):
        select_tsne_indices(labels, 2, 2026)


@pytest.fixture
def small_pipeline(tmp_path):
    root = tmp_path / "dataset"
    SyntheticRadarGenerator(output_dir=str(root), num_emitters=3, sequences_per_emitter=8,
        noise={"enabled": False}, observation={"enabled": False}, overlap={"enabled": False},
        sequence={"random_length": False, "length": 32}).generate()
    return {
        "task": "classify", "output_dir": str(tmp_path / "run"),
        "data_cfg": {"root": str(root), "all": ".", "num_classes": 3,
                     "features": ["CF_MHz", "PW_us", "PRI_us"], "sequence_length": 32,
                     "length_policy": {"short": "zero", "long": "head"}, "cache": False, "progress": False,
                     "selection": {"sample_ratio": 1., "train_ratio": .75, "val_ratio": .25, "seed": 2026, "stratified": True}},
        "model_cfg": {"type": "cnn1d", "in_channels": 3, "hidden_channels": [8, 16],
                      "stream_embedding_dim": 8, "contrast_dim": 4, "dropout": .1},
        "train": {"device": "cpu", "epochs": 2, "batch_size": 4, "use_amp": False, "seed": 2026,
                  "early_stopping_patience": 20},
        "augment": {"enabled": False}, "loss": {"ce": {"weight": 1.}},
    }


def test_training_outputs_use_best_validation_and_resume_history(small_pipeline, capsys):
    model = PulseCLS()
    result = model.train(pipeline=small_pipeline, reporting={"dpi": 75, "formats": ["png", "svg", "pdf"],
                         "tsne": {"max_samples": 4, "max_iter": 300, "perplexity": 30.}})
    root = Path(small_pipeline["output_dir"])
    history = pd.read_csv(root / "training_history.csv")
    assert history["epoch"].tolist() == [1, 2]
    assert history[["train_accuracy", "val_accuracy", "train_macro_f1", "val_macro_f1"]].notna().all().all()
    frame = pd.read_csv(root / "val_predictions.csv")
    assert len(frame) == 6 and frame["sample_index"].tolist() == list(range(6))
    assert frame["path"].nunique() == 6 and frame["confidence"].between(1 / 3 - 1e-6, 1).all()
    assert result["macro_f1"] == pytest.approx(f1_score(frame.label, frame.prediction, average="macro", zero_division=0))
    assert result["best_epoch"] == torch.load(root / "best_model.pt", weights_only=False)["epoch"]
    embedded = np.load(root / "tsne/val_embeddings.npz")
    points = pd.read_csv(root / "tsne/val_tsne.csv")
    assert embedded["embeddings"].shape == (4, 8) and len(points) == 4
    np.testing.assert_array_equal(embedded["labels"], frame.iloc[embedded["sample_indices"]].label)
    embedded.close()
    settings = json.loads((root / "tsne/settings.json").read_text())
    assert settings["effective_perplexity"] == 3. and settings["source_split"] == "val"
    for name in ("accuracy_curve", "macro_f1_curve", "training_curves", "val_per_class_metrics", "val_confusion_matrix", "val_tsne"):
        for extension in ("png", "svg", "pdf"):
            assert (root / "figures" / f"{name}.{extension}").stat().st_size > 0
    cfg = yaml.safe_load((root / "configs/pipeline.yaml").read_text(encoding="utf-8"))
    assert cfg["train"]["reporting"]["dpi"] == 75 and cfg["runtime"]["initial_epoch"] == 0
    for name in ("data", "augment", "model"):
        assert (root / "configs" / f"{name}.yaml").is_file()
    assert "Effective augment.yaml" in capsys.readouterr().out
    assert "val macro-F1" in (root / "training.log").read_text(encoding="utf-8")
    model.train(pipeline=small_pipeline, epochs=1, resume=True, resume_checkpoint=root / "last_model.pt",
                reporting={"dpi": 60, "formats": ["png"], "tsne": {"max_samples": 4, "max_iter": 300}})
    assert pd.read_csv(root / "training_history.csv").epoch.tolist() == [1, 2, 3]
    assert (root / "training.log").read_text(encoding="utf-8").count("Training started:") == 2
    assert not (root / "figures/val_tsne.svg").exists()  # Previous exports are not mistaken for this run.


def test_no_validation_skips_analysis_and_evaluates_each_train_record_once(small_pipeline):
    root = Path(small_pipeline["output_dir"])
    (root / "figures").mkdir(parents=True)
    (root / "figures/val_tsne.png").write_bytes(b"old plot")
    (root / "tsne").mkdir()
    (root / "tsne/val_embeddings.npz").write_bytes(b"old embedding")
    small_pipeline["data_cfg"]["selection"].update(train_ratio=1., val_ratio=0.)
    result = PulseCLS().train(pipeline=small_pipeline, epochs=1, sampler="class_balanced",
                             reporting={"dpi": 60, "formats": ["png"]})
    assert result["evaluation_split"] == "train" and (root / "final_model.pt").exists()
    history = pd.read_csv(root / "training_history.csv")
    assert history[["val_accuracy", "val_macro_f1"]].isna().all().all()
    predictions = pd.read_csv(root / "train_predictions.csv")
    assert len(predictions) == 24 and predictions.path.nunique() == 24
    assert (root / "figures/training_curves.png").exists()
    assert not list((root / "figures").glob("val_*")) and not list((root / "tsne").glob("val_*"))
    assert (root / "train_classification_report.csv").exists()
    assert not (root / "val_classification_report.csv").exists()


@pytest.mark.parametrize("has_val", [True, False])
def test_disable_visualization_skips_plots_and_feature_sampling_but_keeps_results(small_pipeline, monkeypatch, has_val):
    import pulsecls.engine.trainer as trainer_module

    def unexpected_visualization(*args, **kwargs):
        raise AssertionError("Disabled visualization must not sample embeddings or generate figures")

    for name in ("select_tsne_indices", "save_training_figures", "save_tsne"):
        monkeypatch.setattr(trainer_module, name, unexpected_visualization)
    original_collect = trainer_module.BaseTrainer._collect_predictions

    def check_indices(self, model, loader, indices, log_priors, metric_prior_alpha):
        assert len(indices) == 0
        result = original_collect(self, model, loader, indices, log_priors, metric_prior_alpha)
        assert result[3] is None
        return result

    monkeypatch.setattr(trainer_module.BaseTrainer, "_collect_predictions", check_indices)
    root = Path(small_pipeline["output_dir"])
    for path in (root / "figures/training_curves.png", root / "figures/val_tsne.svg",
                 root / "tsne/val_embeddings.npz", root / "tsne/val_tsne.csv", root / "tsne/settings.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"previous visualization")
    if not has_val:
        small_pipeline["data_cfg"]["selection"].update(train_ratio=1., val_ratio=0.)
    model = PulseCLS()
    result = model.train(pipeline=small_pipeline, epochs=1, visualize=False,
                         reporting={"tsne": {"max_samples": 1}})
    split = "val" if has_val else "train"
    assert result["evaluation_split"] == split
    assert model.trainer.train_cfg["visualize"] is False
    assert not list((root / "figures").iterdir())
    assert not list((root / "tsne").iterdir())
    for filename in ("last_model.pt", "best_model.pt" if has_val else "final_model.pt",
                     "training_history.csv", "final_metrics.json", "training.log", "resolved_pipeline.json",
                     f"{split}_predictions.csv", f"{split}_classification_report.csv",
                     f"{split}_confusion_matrix.csv", f"{split}_confusion_matrix_normalized.csv"):
        assert (root / filename).stat().st_size > 0
    cfg = yaml.safe_load((root / "configs/pipeline.yaml").read_text(encoding="utf-8"))
    assert cfg["train"]["visualize"] is False
    predictions = pd.read_csv(root / f"{split}_predictions.csv")
    assert len(predictions) == (6 if has_val else 24)
    assert result["macro_f1"] == pytest.approx(f1_score(predictions.label, predictions.prediction, average="macro", zero_division=0))
