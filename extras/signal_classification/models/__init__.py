from __future__ import annotations

from .resnet import *
from .cnn import CNN
from .efficientnet import EfficientNet, EfficientNetB0
from .nn import (
    ATENetClassifier,
    InvConvClassifier,
    MambaSLClassifier,
    ModernTCNClassifier,
    MPTSNetClassifier,
    MultiGrainCNN,
)


def _build_cnn(cfg: dict, num_classes: int):
    return CNN(num_classes=int(num_classes), in_channel=int(cfg.get("in_channels", 1)))


def _build_efficientnet(cfg: dict, num_classes: int):
    return EfficientNetB0(
        num_classes=int(num_classes),
        in_channels=int(cfg.get("in_channels", 1)),
    )


MODEL_REGISTRY = {
    "cnn": _build_cnn,
    "cnn1d": MultiGrainCNN,
    "multigrain_cnn": MultiGrainCNN,
    "efficientnet": _build_efficientnet,
    "efficientnetb0": _build_efficientnet,
    "efficientnet-b0": _build_efficientnet,
    "atenet": ATENetClassifier,
    "atenet1d": ATENetClassifier,
    "invconv1d": InvConvClassifier,
    "invconv": InvConvClassifier,
    "mambasl": MambaSLClassifier,
    "mamba_sl": MambaSLClassifier,
    "mamba": MambaSLClassifier,
    "modern_tcn": ModernTCNClassifier,
    "moderntcn": ModernTCNClassifier,
    "mptsnet1d": MPTSNetClassifier,
    "mptsnet": MPTSNetClassifier,
}


def build_model(model_cfg: dict, num_classes: int):
    cfg = dict(model_cfg)
    model_type = str(cfg.pop("type", "cnn1d")).lower()
    try:
        factory = MODEL_REGISTRY[model_type]
    except KeyError as exc:
        raise ValueError(f"Unknown model type {model_type!r}; available: {sorted(MODEL_REGISTRY)}") from exc
    return factory(cfg, num_classes=int(num_classes))


__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "CNN",
    "EfficientNet",
    "EfficientNetB0",
    "MultiGrainCNN",
    "ATENetClassifier",
    "InvConvClassifier",
    "MambaSLClassifier",
    "ModernTCNClassifier",
    "MPTSNetClassifier",
]