from __future__ import annotations

import torch

from pulsecls.losses.infonce import (
    classwise_pair_infonce,
    cross_modal_infonce,
    paired_infonce,
    supervised_infonce,
)


def test_infonce_losses_are_finite_and_differentiable() -> None:
    first = torch.randn(6, 12, requires_grad=True)
    second = torch.randn(6, 12, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    loss = (
        paired_infonce(first, second)
        + cross_modal_infonce(first, second)
        + supervised_infonce(torch.cat([first, second]), torch.cat([labels, labels]))
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert first.grad is not None
    assert second.grad is not None


def test_supervised_infonce_handles_singletons() -> None:
    embeddings = torch.randn(3, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2])
    loss = supervised_infonce(embeddings, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None


def test_classwise_pair_infonce_uses_same_and_different_classes() -> None:
    embeddings = torch.randn(8, 12, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3])
    loss = classwise_pair_infonce(
        embeddings,
        labels,
        positives_per_anchor=2,
        negatives_per_anchor=2,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None


def test_classwise_pair_infonce_handles_batches_without_valid_pairs() -> None:
    embeddings = torch.randn(3, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2])
    loss = classwise_pair_infonce(embeddings, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None
