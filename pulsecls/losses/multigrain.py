from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from pulsecls.losses.infonce import (
    classwise_pair_infonce,
    cross_modal_infonce,
    cross_modal_supervised_infonce,
    paired_infonce,
    sample_pulse_embeddings,
    supervised_infonce,
)


@dataclass
class LossBreakdown:
    total: torch.Tensor
    logits: torch.Tensor
    ce: torch.Tensor
    stream: torch.Tensor
    pulse: torch.Tensor
    adversarial: torch.Tensor

    def as_tensors(self) -> dict[str, torch.Tensor]:
        """Detached metrics remain on the device until the trainer reports them."""
        return {"ce_loss": self.ce.detach(), "stream_infonce_loss": self.stream.detach(),
                "pulse_infonce_loss": self.pulse.detach(), "adversarial_loss": self.adversarial.detach(),
                "total_loss": self.total.detach()}

    def as_scalars(self) -> dict[str, float]:
        return {
            "ce_loss": float(self.ce.detach().cpu()),
            "stream_infonce_loss": float(self.stream.detach().cpu()),
            "pulse_infonce_loss": float(self.pulse.detach().cpu()),
            "adversarial_loss": float(self.adversarial.detach().cpu()),
            "total_loss": float(self.total.detach().cpu()),
        }


class MultiGranularityObjective:
    """Compute CE + stream-level + pulse-level objectives.

    The model usually supplies two augmented forward outputs. In the
    ``classwise_pair`` stream mode, stream positives and negatives are
    selected from labels inside one batch instead: same-class streams are
    positives and different-class streams are negatives.
    """

    def __init__(
        self,
        loss_cfg: dict[str, Any],
        criterion: nn.Module,
    ) -> None:
        self.loss_cfg = loss_cfg
        self.criterion = criterion

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def _stream_loss(
        self,
        out_a: dict[str, Any],
        out_b: dict[str, Any],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.loss_cfg.get("stream_infonce", {})
        weight = float(cfg.get("weight", 0.0))
        reference = out_a["logits"]
        if weight <= 0.0:
            return self._zero(reference)

        temperature = float(cfg.get("temperature", 0.2))
        mode = str(cfg.get("mode", "supervised")).lower()
        if mode in {"classwise", "classwise_pair"}:
            return classwise_pair_infonce(
                out_a["stream_contrast"],
                labels,
                temperature=temperature,
                positives_per_anchor=int(cfg.get("positive_samples", 2)),
                negatives_per_anchor=int(cfg.get("negative_samples", 2)),
            )
        if mode == "instance":
            return paired_infonce(
                out_a["stream_contrast"],
                out_b["stream_contrast"],
                temperature=temperature,
            )
        if mode == "hybrid":
            instance = paired_infonce(
                out_a["stream_contrast"],
                out_b["stream_contrast"],
                temperature=temperature,
            )
            supervised = supervised_infonce(
                torch.cat(
                    [out_a["stream_contrast"], out_b["stream_contrast"]],
                    dim=0,
                ),
                torch.cat([labels, labels], dim=0),
                temperature=temperature,
            )
            ratio = float(cfg.get("instance_ratio", 0.5))
            return ratio * instance + (1.0 - ratio) * supervised
        return supervised_infonce(
            torch.cat(
                [out_a["stream_contrast"], out_b["stream_contrast"]],
                dim=0,
            ),
            torch.cat([labels, labels], dim=0),
            temperature=temperature,
        )

    def _pulse_loss(
        self,
        out_a: dict[str, Any],
        out_b: dict[str, Any],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.loss_cfg.get("pulse_infonce", {})
        weight = float(cfg.get("weight", 0.0))
        reference = out_a["logits"]
        if weight <= 0.0:
            return self._zero(reference)

        temperature = float(cfg.get("temperature", 0.2))
        pair_a = out_a.get("pulse_pair_contrast")
        pair_b = out_b.get("pulse_pair_contrast")
        if pair_a is not None and pair_b is not None:
            samples_per_sequence = max(pair_a[0].size(0) // labels.size(0), 1)
            pulse_labels = labels[:, None].expand(
                -1,
                samples_per_sequence,
            ).reshape(-1)
            mode = str(cfg.get("mode", "pulse_condition"))

            def pair_loss(
                raw: torch.Tensor,
                iq: torch.Tensor,
            ) -> torch.Tensor:
                if mode == "supervised":
                    return cross_modal_supervised_infonce(
                        raw,
                        iq,
                        pulse_labels,
                        pulse_labels,
                        temperature=temperature,
                    )
                if mode == "mixed":
                    exact_ratio = float(cfg.get("exact_ratio", 0.25))
                    exact = cross_modal_infonce(
                        raw,
                        iq,
                        temperature=temperature,
                    )
                    grouped = cross_modal_supervised_infonce(
                        raw,
                        iq,
                        pulse_labels,
                        pulse_labels,
                        temperature=temperature,
                    )
                    return exact_ratio * exact + (1.0 - exact_ratio) * grouped
                return cross_modal_infonce(
                    raw,
                    iq,
                    temperature=temperature,
                )

            return 0.5 * (
                pair_loss(pair_a[0], pair_a[1])
                + pair_loss(pair_b[0], pair_b[1])
            )

        pulse_a, labels_a = sample_pulse_embeddings(
            out_a["pulse_contrast"],
            labels,
            int(cfg.get("pulses_per_sequence", 8)),
        )
        pulse_b, labels_b = sample_pulse_embeddings(
            out_b["pulse_contrast"],
            labels,
            int(cfg.get("pulses_per_sequence", 8)),
        )
        return supervised_infonce(
            torch.cat([pulse_a, pulse_b], dim=0),
            torch.cat([labels_a, labels_b], dim=0),
            temperature=temperature,
        )

    def __call__(
        self,
        out_a: dict[str, Any],
        out_b: dict[str, Any],
        labels: torch.Tensor,
    ) -> LossBreakdown:
        logits = 0.5 * (out_a["logits"] + out_b["logits"])
        ce = 0.5 * (
            self.criterion(out_a["logits"], labels)
            + self.criterion(out_b["logits"], labels)
        )
        stream = self._stream_loss(out_a, out_b, labels)
        pulse = self._pulse_loss(out_a, out_b, labels)

        adv_cfg = self.loss_cfg.get("adversarial", {})
        adv_weight = float(adv_cfg.get("weight", 0.0))
        if (
            adv_weight > 0.0
            and "modality_domain_logits" in out_a
            and "modality_domain_logits" in out_b
        ):
            domain_criterion = nn.CrossEntropyLoss()
            adversarial = 0.5 * (
                domain_criterion(
                    out_a["modality_domain_logits"],
                    out_a["modality_domain_labels"],
                )
                + domain_criterion(
                    out_b["modality_domain_logits"],
                    out_b["modality_domain_labels"],
                )
            )
        else:
            adversarial = self._zero(logits)

        ce_weight = float(self.loss_cfg.get("ce", {}).get("weight", 1.0))
        stream_weight = float(
            self.loss_cfg.get("stream_infonce", {}).get("weight", 0.0)
        )
        pulse_weight = float(
            self.loss_cfg.get("pulse_infonce", {}).get("weight", 0.0)
        )
        total = (
            ce_weight * ce
            + stream_weight * stream
            + pulse_weight * pulse
            + adv_weight * adversarial
        )
        return LossBreakdown(
            total=total,
            logits=logits,
            ce=ce,
            stream=stream,
            pulse=pulse,
            adversarial=adversarial,
        )
