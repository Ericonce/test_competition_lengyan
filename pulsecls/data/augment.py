from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch


@dataclass
class MixupState:
    lam: float
    partner_index: torch.Tensor


class LegacyPulseAugment:
    """Sequence augmentations adapted from Ultralytics image policies.

    Kept as label-preserving geometric/value jitter in ``__call__``. MixUp is
    applied separately in ``apply`` because it also mixes labels.
    """

    def __init__(
        self,
        enabled: bool = True,
        noise_std: float = 0.02,
        scale_std: float = 0.03,
        pulse_mask_prob: float = 0.03,
        feature_mask_prob: float = 0.02,
        time_shift_ratio: float = 0.0,
        time_flip_prob: float = 0.0,
        mixup_prob: float = 0.0,
        mixup_alpha: float = 0.2,
    ) -> None:
        self.enabled = enabled
        self.noise_std = float(noise_std)
        self.scale_std = float(scale_std)
        self.pulse_mask_prob = float(pulse_mask_prob)
        self.feature_mask_prob = float(feature_mask_prob)
        self.time_shift_ratio = float(time_shift_ratio)
        self.time_flip_prob = float(time_flip_prob)
        self.mixup_prob = float(mixup_prob)
        self.mixup_alpha = float(mixup_alpha)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return inputs
        output = inputs
        output = self._scale(output)
        output = self._noise(output)
        output = self._time_shift(output)
        output = self._time_flip(output)
        output = self._pulse_mask(output)
        output = self._feature_mask(output)
        return output

    def apply(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MixupState | None]:
        output = self(inputs)
        if (
            not self.enabled
            or labels is None
            or self.mixup_prob <= 0.0
            or self.mixup_alpha <= 0.0
            or inputs.size(0) < 2
            or float(torch.rand((), device=inputs.device)) >= self.mixup_prob
        ):
            return output, None
        partner = torch.randperm(output.size(0), device=output.device)
        lam = float(
            torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
        )
        mixed = lam * output + (1.0 - lam) * output[partner]
        return mixed, MixupState(lam=lam, partner_index=partner)

    def _scale(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.scale_std <= 0:
            return inputs
        scale = 1.0 + torch.randn(
            inputs.size(0),
            inputs.size(1),
            1,
            device=inputs.device,
            dtype=inputs.dtype,
        ) * self.scale_std
        return inputs * scale

    def _noise(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.noise_std <= 0:
            return inputs
        noise = torch.randn_like(inputs) * self.noise_std
        return inputs + noise

    def _pulse_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.pulse_mask_prob <= 0:
            return inputs
        mask = torch.rand(inputs.size(0), 1, inputs.size(2), device=inputs.device)
        return inputs.masked_fill(mask < self.pulse_mask_prob, 0.0)

    def _feature_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.feature_mask_prob <= 0:
            return inputs
        mask = torch.rand(inputs.size(0), inputs.size(1), 1, device=inputs.device)
        return inputs.masked_fill(mask < self.feature_mask_prob, 0.0)

    def _time_shift(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ultralytics RandomPerspective/translation analog: cyclic pulse roll.
        if self.time_shift_ratio <= 0:
            return inputs
        length = inputs.size(2)
        max_shift = int(round(length * self.time_shift_ratio))
        if max_shift <= 0:
            return inputs
        shifts = torch.randint(
            -max_shift,
            max_shift + 1,
            (inputs.size(0),),
            device=inputs.device,
        )
        index = (
            torch.arange(length, device=inputs.device)[None, :] - shifts[:, None]
        ) % length
        gather_index = index[:, None, :].expand(-1, inputs.size(1), -1)
        return inputs.gather(2, gather_index)

    def _time_flip(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ultralytics RandomFlip analog: reverse the pulse order.
        if self.time_flip_prob <= 0:
            return inputs
        flip = torch.rand(inputs.size(0), device=inputs.device) < self.time_flip_prob
        if not bool(flip.any()):
            return inputs
        flipped = inputs.flip(-1)
        return torch.where(flip[:, None, None], flipped, inputs)



class PulseAugment(LegacyPulseAugment):
    """Switchable, ordered augmentation; old flat configurations remain supported."""

    def __init__(self, *, methods: dict | None = None, order: list[str] | None = None,
                 check_finite: bool = True, **legacy):
        super().__init__(**legacy)
        from pulsecls.data.augmentation_config import resolve_methods, LABEL_METHODS
        from pulsecls.data.augmentation_methods import MethodRunner
        self.methods, self.order = (None, []) if methods is None else resolve_methods(methods, order)
        self.runner = MethodRunner()
        self.label_methods = LABEL_METHODS
        self.check_finite = bool(check_finite)
        self._warned_mixup = False

    def __call__(self, inputs: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.methods is None:
            return super().__call__(inputs)
        return self._execute(inputs, labels, allow_mixup=False)[0]

    def apply(self, inputs: torch.Tensor, labels: torch.Tensor | None = None,
              *, allow_mixup: bool = True) -> tuple[torch.Tensor, MixupState | None]:
        if self.methods is None:
            if allow_mixup:
                return super().apply(inputs, labels)
            return super().__call__(inputs), None
        return self._execute(inputs, labels, allow_mixup)

    def _execute(self, inputs, labels, allow_mixup):
        if not self.enabled:
            return inputs, None
        if inputs.ndim != 3 or inputs.size(-1) < 2:
            raise ValueError("Augmentation expects [batch, channels, length>=2]")
        if labels is not None and (labels.ndim != 1 or len(labels) != inputs.size(0)):
            raise ValueError("One class label per input sequence is required")
        output, mix = inputs, None
        for name in self.order:
            parameters = self.methods[name]
            if not parameters["enabled"] or parameters["prob"] == 0:
                continue
            if name == "mixup":
                if not allow_mixup:
                    if not self._warned_mixup:
                        warnings.warn("MixUp is skipped for contrastive views; use same-class interpolation for label-preserving fusion.", stacklevel=2)
                        self._warned_mixup = True
                    continue
                if labels is None:
                    raise ValueError("MixUp needs class labels; call augment.apply(inputs, labels)")
                if output.size(0) < 2 or float(torch.rand(())) >= parameters["prob"]:
                    continue
                partner = torch.randperm(output.size(0), device=output.device)
                lam = float(torch.distributions.Beta(parameters["alpha"], parameters["alpha"]).sample())
                output = lam * output + (1 - lam) * output[partner]
                mix = MixupState(lam=lam, partner_index=partner)
            else:
                selected = torch.rand(output.size(0), device=output.device) < parameters["prob"]
                if not bool(selected.any()):
                    continue
                if name in self.label_methods:
                    if labels is None:
                        raise ValueError(f"{name} needs class labels; call augment.apply(inputs, labels)")
                    eligible = (labels[:, None] == labels[None, :]).sum(1) > 1
                    selected = selected & eligible.to(selected.device)
                    if not bool(selected.any()):
                        continue
                candidate = self.runner.run(name, output, parameters, labels)
                if candidate.shape != output.shape:
                    raise RuntimeError(f"{name} changed input shape from {output.shape} to {candidate.shape}")
                output = torch.where(selected[:, None, None], candidate, output)
            if self.check_finite and not bool(torch.isfinite(output).all()):
                raise FloatingPointError(f"Non-finite output after augmentation {name}")
        return output, mix


from pulsecls.data.augmentation_config import default_augment_config  # public convenience API
