import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pulsecls import PulseCLS
from pulsecls.cfg import REPO_ROOT, get_cfg
from pulsecls.data.dataset import (
    PulseArrayDataset, feature_stats, find_samples, load_sample, load_split_arrays, resize_sequence,
)
from pulsecls.data.synthetic import (
    CF_MODES, FEATURES, PRI_MODES, SyntheticRadarGenerator, add_gaussian_noise,
)


def small_generator(**overrides):
    base = {"num_emitters": 3, "sequences_per_emitter": 6,
            "sequence": {"length_range": [8, 20]}, "noise": {"snr_db_range": [30, 45], "correlation_range": [0, 0]},
            "overlap": {"enabled": False}, "observation": {"enabled": False},
            "save_clean": True}
    from pulsecls.cfg import deep_update
    return SyntheticRadarGenerator(**deep_update(base, overrides))


def test_exact_full_sequence_energy_snr_including_constant_channels():
    clean = np.tile([4000., 20., 500.], (256, 1))
    noisy, meta = add_gaussian_noise(clean, 25., [10000, 50, 1000], np.random.default_rng(42))
    energy = np.sum((clean / [10000, 50, 1000]) ** 2)
    noise_energy = np.sum(((noisy - clean) / [10000, 50, 1000]) ** 2)
    assert meta["signal_energy"] == pytest.approx(energy)
    assert 10 * np.log10(energy / noise_energy) == pytest.approx(25., abs=1e-10)
    assert np.all(np.std(noisy, axis=0) > 0)


@pytest.mark.parametrize("pri,cf", list(itertools.product(PRI_MODES, CF_MODES)))
def test_all_twenty_pri_cf_combinations(pri, cf):
    engine = small_generator(num_emitters=1, transmitter={"templates": [
        {"pri_mode": pri, "cf_mode": cf, "pw_mode": "fixed"}]})
    _, emitters = engine.create_emitters()
    emitter = emitters[0]
    x = engine.clean_sequence(emitter, 256, 17, np.random.default_rng(7))
    assert x.shape == (256, 3) and np.isfinite(x).all() and (x > 0).all()
    assert (x[:, 1] < x[:, 2] * engine.config["parameters"]["max_duty_ratio"]).all()
    if pri == "fixed":
        assert np.ptp(x[:, 2]) == 0
    else:
        assert np.ptp(x[:, 2]) > 0
    assert np.ptp(x[:, 0]) > 0
    assert np.ptp(x[:, 1]) == 0


def test_generation_format_reproducibility_metadata_and_splits(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    a = small_generator().generate(first)
    b = small_generator().generate(second)
    assert a["num_sequences"] == b["num_sequences"] == 18
    assert a["length_min"] < a["length_max"]
    templates = json.loads((first / "emitters.json").read_text())
    assert len(templates["emitters"]) == 3
    metadata = [json.loads(line) for line in (first / "sequences.jsonl").read_text().splitlines()]
    assert len({row["snr_db"] for row in metadata}) > 1
    assert len({row["length"] for row in metadata}) > 1
    for row in metadata:
        p = first / row["path"]
        assert p.read_bytes() == (second / row["path"]).read_bytes()
        assert p.read_text().splitlines()[0] == "CF_MHz,PW_us,PRI_us"
        noisy = np.loadtxt(p, delimiter=",", skiprows=1)
        clean = np.load(first / "clean" / str(row["label"]) / (p.stem + ".npy"))
        assert len(noisy) == row["length"]
        es = np.sum((clean / [10000, 50, 1000]) ** 2)
        en = np.sum(((noisy - clean) / [10000, 50, 1000]) ** 2)
        assert 10 * np.log10(es / en) == pytest.approx(row["snr_db"], abs=1e-8)
    assert not (first / "splits").exists()
    from pulsecls.data.selection import select_records
    from pulsecls.cfg import load_yaml
    data_cfg = load_yaml(first / "data.yaml")
    data_cfg.update(root=str(first), selection={"sample_ratio": 1., "train_ratio": .8, "val_ratio": .2})
    partitions, info = select_records(data_cfg)
    train_paths = {p for p, _ in partitions["train"]}
    val_paths = {p for p, _ in partitions["val"]}
    assert not train_paths & val_paths
    assert train_paths | val_paths == {first / row["path"] for row in metadata}
    assert len(find_samples(first)) == 18  # clean and metadata directories are ignored
    # A second run replaces the old dataset and removes stale class files/caches.
    (first / "0" / "stale.csv").write_text("old data")
    cache = first / ".pulsecls_cache"
    cache.mkdir(exist_ok=True)
    (cache / "old.npz").write_bytes(b"stale cache")
    small_generator().generate(first)
    assert not (first / "0" / "stale.csv").exists() and not cache.exists()
    assert (first / metadata[0]["path"]).read_bytes() == (second / metadata[0]["path"]).read_bytes()


def test_fixed_length_no_noise_and_per_emitter_counts(tmp_path):
    root = tmp_path / "data"
    result = small_generator(sequences_per_emitter=[3, 4, 5],
                             sequence={"random_length": False, "length": 10},
                             noise={"enabled": False}).generate(root)
    assert result["num_sequences"] == 12
    assert result["length_min"] == result["length_max"] == 10
    for label, count in enumerate([3, 4, 5]):
        assert len(list((root / str(label)).glob("*.csv"))) == count
        for path in (root / str(label)).glob("*.csv"):
            np.testing.assert_array_equal(np.loadtxt(path, delimiter=",", skiprows=1),
                                          np.load(root / "clean" / str(label) / (path.stem + ".npy")))


def test_low_snr_small_pw_completes_and_records_actual_snr(tmp_path):
    root = tmp_path / "low_snr"
    engine = small_generator(num_emitters=2, sequences_per_emitter=3,
        sequence={"random_length": False, "length": 512},
        parameters={"pw_us": [0.1, 0.1]}, noise={"snr_db_range": [10, 10]})
    engine.generate(root)
    for line in (root / "sequences.jsonl").read_text().splitlines():
        row = json.loads(line)
        observed = np.loadtxt(root / row["path"], delimiter=",", skiprows=1)
        clean = np.load(root / "clean" / str(row["label"]) / (Path(row["path"]).stem + ".npy"))
        assert (observed > 0).all()
        assert (observed[:, 1] < observed[:, 2] * .5).all()
        assert row["noise_attempts"] == 1 and row["snr_db"] == 10
        es = np.sum((clean / [10000, 50, 1000]) ** 2)
        en = np.sum(((observed - clean) / [10000, 50, 1000]) ** 2)
        assert row["achieved_snr_db"] == pytest.approx(10 * np.log10(es / en))


def test_each_channel_noise_power_is_proportional_to_its_signal():
    clean = np.tile([10000., .1, 1000.], (512, 1))
    observed, _ = add_gaussian_noise(clean, 10, [10000, 50, 1000], np.random.default_rng(4))
    per_channel_snr = 10 * np.log10(np.sum(clean ** 2, axis=0) / np.sum((observed - clean) ** 2, axis=0))
    np.testing.assert_allclose(per_channel_snr, 10, atol=1e-10)


def test_correlated_noise_keeps_target_energy_and_temporal_correlation():
    clean = np.tile([5000., 20., 600.], (4096, 1))
    observed, metadata = add_gaussian_noise(clean, 20, [10000, 50, 1000], np.random.default_rng(44), correlation=.8)
    noise = observed - clean
    for channel in range(3):
        assert np.corrcoef(noise[:-1, channel], noise[1:, channel])[0, 1] == pytest.approx(.8, abs=.05)
    assert metadata["achieved_snr_db"] == pytest.approx(20, abs=1e-9)


def test_overlap_groups_share_template_and_have_close_parameter_centers():
    engine = small_generator(num_emitters=10, overlap={"enabled": True, "emitters_per_group": 5})
    _, emitters = engine.create_emitters()
    for start in (0, 5):
        members = emitters[start:start + 5]
        assert len({e["group_id"] for e in members}) == 1
        assert len({e["template_id"] for e in members}) == 1
        for key, ratio in zip(("cf_mhz", "pw_us", "pri_us"), engine.config["overlap"]["base_jitter_ratio"]):
            values = np.array([e[key] for e in members])
            assert values.max() / values.min() <= (1 + ratio) / (1 - ratio) + 1e-10
        assert len({e["period_pulses"] for e in members}) > 1


def test_missing_pulses_recompute_pri_instead_of_merely_deleting_rows():
    from unittest.mock import Mock
    engine = small_generator(observation={"enabled": True, "session_bias_std_ratio": [0, 0, 0],
        "missing_pulse_rate_range": [.5, .5]})
    clean = np.array([[100., 10., 100.], [200., 10., 200.], [300., 10., 300.], [400., 10., 400.]])
    rng = Mock()
    rng.normal.return_value = np.zeros(3)
    rng.uniform.return_value = .5
    rng.random.return_value = np.array([1., 0., 1., 1.])
    result, metadata = engine.observation_sequence(clean, rng)
    np.testing.assert_array_equal(result[:, 0], [100, 300, 400])
    np.testing.assert_allclose(result[:, 2], [300, 300, 400])
    assert result[:, 2].sum() == clean[:, 2].sum()
    assert metadata["missing_pulses"] == 1


def test_failed_regeneration_keeps_old_dataset(tmp_path):
    from unittest.mock import patch
    root = tmp_path / "dataset"
    engine = small_generator()
    engine.generate(root)
    before = (root / "0/0.csv").read_bytes()
    with patch.object(engine, "_write_dataset", side_effect=RuntimeError("generation failed")):
        with pytest.raises(RuntimeError, match="generation failed"):
            engine.generate(root)
    assert (root / "0/0.csv").read_bytes() == before
    assert not list(tmp_path.glob(".dataset.generating-*"))


def test_replacement_failure_rolls_back_old_dataset(tmp_path):
    from unittest.mock import patch
    root = tmp_path / "dataset"
    engine = small_generator()
    engine.generate(root)
    before = (root / "0/0.csv").read_bytes()
    original = Path.rename
    def rename(path, target):
        if path.name.startswith(".dataset.generating-"):
            raise OSError("replacement failed")
        return original(path, target)
    with patch.object(Path, "rename", rename):
        with pytest.raises(OSError, match="replacement failed"):
            engine.generate(root)
    assert (root / "0/0.csv").read_bytes() == before
    assert not list(tmp_path.glob(".dataset.*"))


def test_refuses_to_replace_project_root_or_unrelated_directory(tmp_path):
    root = tmp_path / "unrelated"
    root.mkdir()
    (root / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="unrelated"):
        small_generator().generate(root)
    assert (root / "keep.txt").read_text() == "keep"
    with pytest.raises(ValueError, match="project/drive root"):
        small_generator().generate(REPO_ROOT)


def test_zero_repeat_and_crop_policies_and_padding_normalization(tmp_path):
    sample = np.array([[100., 2., 20.], [200., 4., 40.]])
    np.testing.assert_array_equal(resize_sequence(sample, 5, {"short": "repeat"}), sample[[0, 1, 0, 1, 0]])
    padded = resize_sequence(sample, 5, {"short": "zero"})
    np.testing.assert_array_equal(padded[2:], 0)
    np.testing.assert_array_equal(resize_sequence(np.tile(sample, (3, 1)), 2, {"long": "center"}), sample)
    with pytest.raises(ValueError):
        resize_sequence(sample, 5)
    mean, std = feature_stats(padded[None], ignore_zero_padding=True)
    np.testing.assert_allclose(mean, sample.mean(axis=0))
    data = PulseArrayDataset(padded[None], np.array([0]), mean, std, zero_padding=True)
    torch.testing.assert_close(data[0][0][:, 2:], torch.zeros(3, 3))


def test_reader_generated_data_and_cache_policy(tmp_path):
    root = tmp_path / "data"
    small_generator().generate(root)
    cfg = get_cfg(REPO_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml", overrides={
        "data": root / "data.yaml", "sequence_length": 16,
        "dataset": {"sample_ratio": 1., "train_ratio": .8, "val_ratio": .2}})
    assert Path(cfg["data_cfg"]["root"]) == root
    a, labels, paths = load_split_arrays(cfg["data_cfg"], "train")
    assert a.shape[1:] == (16, 3)
    caches = list((root / ".pulsecls_cache").glob("*.npz"))
    cfg["data_cfg"]["length_policy"]["short"] = "repeat"
    b, _, _ = load_split_arrays(cfg["data_cfg"], "train")
    assert len(list((root / ".pulsecls_cache").glob("*.npz"))) == len(caches) + 1
    assert not np.all(b == 0, axis=-1).any()
    assert labels.shape == (len(paths),)


def test_pipeline_owns_length_and_padding_with_code_overrides(tmp_path):
    from pulsecls.cfg import load_pipeline, load_yaml
    root = tmp_path / "generated"
    small_generator().generate(root)
    raw_data = load_yaml(root / "data.yaml")
    assert "sequence_length" not in raw_data and "length_policy" not in raw_data
    source = REPO_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml"
    loaded = load_pipeline(source, data_override=root / "data.yaml")
    assert loaded["data_cfg"]["sequence_length"] == 256
    assert loaded["data_cfg"]["length_policy"] == {"short": "zero", "long": "head"}
    # Replacing data in model.train(data=...) must retain pipeline length settings.
    cfg = get_cfg(source, overrides={"data": root / "data.yaml"})
    assert cfg["data_cfg"]["sequence_length"] == 256
    overridden = get_cfg(source, overrides={"data": root / "data.yaml", "sequence_length": 32,
        "length_policy": {"short": "repeat", "long": "center"}})
    # BaseTrainer calls get_cfg again on resolved configs; explicit choices must survive.
    assert get_cfg(overridden)["data_cfg"]["sequence_length"] == 32
    assert get_cfg(overridden)["data_cfg"]["length_policy"]["short"] == "repeat"


def test_train_resume_and_infer_different_lengths(tmp_path):
    root = tmp_path / "data"
    small_generator().generate(root)
    source = REPO_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml"
    checkpoint = tmp_path / "results" / "last_model.pt"
    overrides = {"data": root / "data.yaml", "output_dir": tmp_path / "results",
                 "device": "cpu", "epochs": 1, "batch_size": 3,
                 "data_cfg": {"sequence_length": 16},
                 "dataset": {"sample_ratio": 1., "train_ratio": .8, "val_ratio": .2},
                 "model": {"hidden_channels": [8, 16], "stream_embedding_dim": 8,
                           "pulse_embedding_dim": 8, "contrast_dim": 4}}
    model = PulseCLS()
    result = model.train(pipeline=source, **overrides)
    assert np.isfinite(result["macro_f1"])
    model.train(pipeline=source, **overrides, resume=True, resume_checkpoint=checkpoint)
    assert torch.load(checkpoint, map_location="cpu", weights_only=False)["epoch"] == 2
    predict_overrides = {key: value for key, value in overrides.items() if key != "data_cfg"}
    metrics = model.predict(pipeline=source, **predict_overrides, checkpoint=checkpoint, split="val",
                            data_cfg={"sequence_length": 32, "length_policy": {"short": "repeat"}})
    assert model.predictor.data_cfg["sequence_length"] == 32
    assert model.predictor.data_cfg["length_policy"]["short"] == "repeat"
    assert np.isfinite(metrics["macro_f1"])
    # Inference uses the trained stats, even when input length/padding changes.
    metrics = model.predict(pipeline=source, **predict_overrides, checkpoint=checkpoint, split="val",
                            data_cfg={"sequence_length": 24, "length_policy": {"short": "zero"}})
    assert np.isfinite(metrics["macro_f1"])


@pytest.mark.parametrize("overrides", [
    {"num_emitters": 0}, {"sequences_per_emitter": [3]},
    {"sequence": {"length_range": [20, 10]}}, {"noise": {"channel_scales": [1, 0, 1]}},
    {"noise": {"invalid_policy": "typo"}}, {"sequences_per_emitter": 0},
    {"transmitter": {"pri_modes": ["typo"]}}, {"noise": {"typo": 1}},
])
def test_invalid_settings_are_rejected_before_writing(overrides):
    with pytest.raises(ValueError):
        small_generator(**overrides)
