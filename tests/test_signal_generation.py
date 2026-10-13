from __future__ import annotations

import torch

from pulsecls.signals.generator import HypothesisSignalGenerator


MODULATIONS = (
    "lfm",
    "nlfm",
    "fsk",
    "bpsk",
    "lfm_fsk",
    "lfm_bpsk",
)


def _conditions() -> torch.Tensor:
    return torch.tensor(
        [
            [-1.0, -0.5],
            [0.0, 0.0],
            [1.0, 0.75],
        ],
        dtype=torch.float32,
    )


def test_each_modulation_has_expected_shape_and_finite_values() -> None:
    for modulation in MODULATIONS:
        generator = HypothesisSignalGenerator(
            signal_length=64,
            modulation_type=modulation,
            modulation_types=MODULATIONS,
            num_modes=len(MODULATIONS),
        )
        signals = generator(_conditions(), randomize=False)
        assert signals.shape == (3, 2, 64)
        assert torch.isfinite(signals).all()


def test_deterministic_generation_is_repeatable() -> None:
    generator = HypothesisSignalGenerator(
        signal_length=32,
        modulation_type="lfm",
        phase_jitter=False,
    )
    first = generator(_conditions(), randomize=False)
    second = generator(_conditions(), randomize=False)
    assert torch.equal(first, second)


def test_signal_modes_have_expected_iq_structure() -> None:
    conditions = _conditions()

    envelope = HypothesisSignalGenerator(
        signal_length=32,
        modulation_type="lfm",
        signal_mode="envelope",
    )(conditions)
    assert torch.allclose(envelope[:, 1], torch.zeros_like(envelope[:, 1]))
    assert torch.all(envelope[:, 0] >= 0)

    phase = HypothesisSignalGenerator(
        signal_length=32,
        modulation_type="lfm",
        signal_mode="phase",
    )(conditions)
    magnitude = phase.square().sum(dim=1)
    assert torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-5)


def test_randomized_generation_changes_at_least_one_view() -> None:
    torch.manual_seed(7)
    generator = HypothesisSignalGenerator(
        signal_length=32,
        modulation_type="random",
        modulation_types=MODULATIONS,
        num_modes=len(MODULATIONS),
    )
    deterministic = generator(_conditions(), randomize=False)
    randomized = generator(_conditions(), randomize=True)
    assert not torch.allclose(deterministic, randomized)
