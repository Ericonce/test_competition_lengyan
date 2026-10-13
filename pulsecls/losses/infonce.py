from __future__ import annotations

import torch
from torch.nn import functional as F


def paired_infonce(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """SimCLR-style InfoNCE where matching rows across two views are positives."""
    if view_a.numel() == 0 or view_b.numel() == 0 or view_a.size(0) < 1:
        return (view_a.sum() + view_b.sum()) * 0.0
    if view_a.shape != view_b.shape:
        raise ValueError(f"paired_infonce expects equal shapes, got {view_a.shape} and {view_b.shape}")

    count = view_a.size(0)
    embeddings = F.normalize(torch.cat([view_a, view_b], dim=0), dim=-1)
    logits = embeddings @ embeddings.T
    logits = logits / temperature
    eye = torch.eye(count * 2, device=embeddings.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, -torch.finfo(logits.dtype).max)

    targets = torch.arange(count * 2, device=embeddings.device)
    targets = (targets + count) % (count * 2)
    return F.cross_entropy(logits, targets)


def cross_modal_infonce(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Bidirectional InfoNCE over paired cross-modal candidates only."""
    if view_a.numel() == 0 or view_b.numel() == 0 or view_a.size(0) < 1:
        return (view_a.sum() + view_b.sum()) * 0.0
    if view_a.shape != view_b.shape:
        raise ValueError(
            f"cross_modal_infonce expects equal shapes, got "
            f"{view_a.shape} and {view_b.shape}"
        )

    left = F.normalize(view_a, dim=-1)
    right = F.normalize(view_b, dim=-1)
    logits = (left @ right.T) / temperature
    targets = torch.arange(left.size(0), device=left.device)
    left_to_right = F.cross_entropy(logits, targets)
    right_to_left = F.cross_entropy(logits.T, targets)
    return 0.5 * (left_to_right + right_to_left)


def cross_modal_supervised_infonce(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Cross-modal supervised InfoNCE with all same-label pairs as positives."""
    if (
        view_a.numel() == 0
        or view_b.numel() == 0
        or view_a.size(0) < 1
        or view_b.size(0) < 1
    ):
        return (view_a.sum() + view_b.sum()) * 0.0
    if view_a.size(1) != view_b.size(1):
        raise ValueError(
            f"cross_modal_supervised_infonce expects equal embedding dimensions, "
            f"got {view_a.shape} and {view_b.shape}"
        )

    left = F.normalize(view_a, dim=-1)
    right = F.normalize(view_b, dim=-1)
    labels_a = labels_a.reshape(-1)
    labels_b = labels_b.reshape(-1)
    if labels_a.size(0) != left.size(0) or labels_b.size(0) != right.size(0):
        raise ValueError(
            "Cross-modal labels must match their respective embedding counts: "
            f"{left.size(0)}, {labels_a.size(0)} and {right.size(0)}, {labels_b.size(0)}"
        )

    logits = (left @ right.T) / temperature
    positive_mask = labels_a[:, None].eq(labels_b[None, :])

    def _directional_loss(scores: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        valid = positives.any(dim=1)
        if not valid.any():
            return scores.sum() * 0.0
        positive_scores = scores.masked_fill(~positives, -torch.finfo(scores.dtype).max)
        numerator = torch.logsumexp(positive_scores, dim=1)
        denominator = torch.logsumexp(scores, dim=1)
        return (denominator - numerator)[valid].mean()

    left_to_right = _directional_loss(logits, positive_mask)
    right_to_left = _directional_loss(logits.T, positive_mask.T)
    return 0.5 * (left_to_right + right_to_left)


def supervised_infonce(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Supervised InfoNCE with all same-label samples as positives."""
    if embeddings.numel() == 0 or embeddings.size(0) < 2:
        return embeddings.sum() * 0.0

    embeddings = F.normalize(embeddings, dim=-1)
    labels = labels.reshape(-1)
    logits = embeddings @ embeddings.T
    logits = logits / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    eye = torch.eye(labels.numel(), device=labels.device, dtype=torch.bool)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~eye
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0

    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    numerator = (exp_logits * positive_mask.float()).sum(dim=1)
    denominator = exp_logits.sum(dim=1).clamp_min(1e-12)
    loss = -torch.log((numerator / denominator).clamp_min(1e-12))
    return loss[valid].mean()


def classwise_pair_infonce(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
    positives_per_anchor: int = 2,
    negatives_per_anchor: int = 2,
) -> torch.Tensor:
    """Class-aware InfoNCE using explicit same/different-class candidates.

    Each anchor selects up to ``positives_per_anchor`` other streams with the
    same label and up to ``negatives_per_anchor`` streams with a different
    label.  The selected same-class streams are positives and the selected
    different-class streams are negatives.  This deliberately avoids treating
    two augmented views of the same stream as the stream-level positive pair.
    """
    if embeddings.numel() == 0 or embeddings.size(0) < 2:
        return embeddings.sum() * 0.0
    if labels.reshape(-1).size(0) != embeddings.size(0):
        raise ValueError(
            "classwise_pair_infonce expects one label per embedding, got "
            f"{embeddings.size(0)} embeddings and {labels.reshape(-1).size(0)} labels"
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    positive_limit = max(int(positives_per_anchor), 1)
    negative_limit = max(int(negatives_per_anchor), 1)
    embeddings = F.normalize(embeddings, dim=-1)
    labels = labels.reshape(-1)
    similarities = (embeddings @ embeddings.T) / float(temperature)

    losses: list[torch.Tensor] = []
    for anchor_index in range(embeddings.size(0)):
        same = torch.where(labels == labels[anchor_index])[0]
        same = same[same != anchor_index]
        different = torch.where(labels != labels[anchor_index])[0]
        if same.numel() == 0 or different.numel() == 0:
            continue

        if same.numel() > positive_limit:
            same = same[
                torch.randperm(same.numel(), device=embeddings.device)[
                    :positive_limit
                ]
            ]
        if different.numel() > negative_limit:
            different = different[
                torch.randperm(different.numel(), device=embeddings.device)[
                    :negative_limit
                ]
            ]

        candidates = torch.cat([same, different], dim=0)
        scores = similarities[anchor_index, candidates]
        positive_scores = scores[: same.numel()]
        losses.append(
            torch.logsumexp(scores, dim=0)
            - torch.logsumexp(positive_scores, dim=0)
        )

    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


def sample_pulse_embeddings(
    pulse_embeddings: torch.Tensor,
    labels: torch.Tensor,
    pulses_per_sequence: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, dim = pulse_embeddings.shape
    pulses_per_sequence = min(max(int(pulses_per_sequence), 1), sequence_length)
    indices = torch.randint(
        0,
        sequence_length,
        (batch_size, pulses_per_sequence),
        device=pulse_embeddings.device,
    )
    batch_indices = torch.arange(batch_size, device=pulse_embeddings.device)[:, None]
    sampled = pulse_embeddings[batch_indices, indices].reshape(-1, dim)
    sampled_labels = labels[:, None].expand(-1, pulses_per_sequence).reshape(-1)
    return sampled, sampled_labels
