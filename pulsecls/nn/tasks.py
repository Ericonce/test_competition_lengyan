from __future__ import annotations

from collections.abc import Callable

from torch import nn

from pulsecls.nn.classifiers.atenet import ATENetClassifier
from pulsecls.nn.classifiers.baseline import ResNet1DClassifier
from pulsecls.nn.classifiers.invconv import InvConvClassifier
from pulsecls.nn.classifiers.mambasl import MambaSLClassifier
from pulsecls.nn.classifiers.modern_tcn import ModernTCNClassifier
from pulsecls.nn.classifiers.mptsnet import MPTSNetClassifier
from pulsecls.nn.classifiers.resnet1d_multigrain import ModularResNet1DClassifier
from pulsecls.nn.classifiers.resnet_softmax import ResNetSoftmaxClassifier
from pulsecls.nn.multigrain_cnn import MultiGrainCNN
from pulsecls.nn.pulse_hypothesis_cnn import PulseConditionedContrastiveCNN
from pulsecls.nn.resnet1d_hypothesis import ResNet1DHypothesisClassifier


ModelFactory = Callable[[dict, int], nn.Module]


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "cnn1d": MultiGrainCNN,
    "multigrain_cnn": MultiGrainCNN,
    "pulse_conditioned_contrastive_cnn": PulseConditionedContrastiveCNN,
    "resnet1d": ResNet1DClassifier,
    "resnet_softmax": ResNetSoftmaxClassifier,
    "resnet1d_softmax": ResNetSoftmaxClassifier,
    "mambasl": MambaSLClassifier,
    "mamba_sl": MambaSLClassifier,
    "mamba": MambaSLClassifier,
    "atenet": ATENetClassifier,
    "atenet1d": ATENetClassifier,
    "modern_tcn": ModernTCNClassifier,
    "moderntcn": ModernTCNClassifier,
    "invconv1d": InvConvClassifier,
    "invconv": InvConvClassifier,
    "mptsnet1d": MPTSNetClassifier,
    "mptsnet": MPTSNetClassifier,
    "modular_resnet1d_hypothesis": ModularResNet1DClassifier,
    # Keep historical checkpoint keys and old pipeline files loadable.
    "resnet1d_hypothesis": ResNet1DHypothesisClassifier,
}


def register_model(name: str, factory: ModelFactory) -> None:
    key = str(name).lower()
    if key in MODEL_REGISTRY:
        raise KeyError(f"Model {name!r} is already registered")
    MODEL_REGISTRY[key] = factory


def build_registered_model(model_cfg: dict, num_classes: int) -> nn.Module:
    model_type = str(model_cfg.get("type", "multigrain_cnn")).lower()
    try:
        factory = MODEL_REGISTRY[model_type]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model type {model_type!r}; available: {available}"
        ) from exc
    return factory(model_cfg, num_classes=num_classes)


def build_model(model_cfg: dict, data_cfg: dict) -> nn.Module:
    cfg = dict(model_cfg)
    if data_cfg.get("features"):
        channels = len(data_cfg["features"])
        cfg.setdefault("in_channels", channels)
        if int(cfg["in_channels"]) != channels:
            raise ValueError(f"Model in_channels={cfg['in_channels']} does not match {channels} data features")
    if "sequence_length" not in cfg and "sequence_length" in data_cfg:
        cfg["sequence_length"] = int(data_cfg["sequence_length"])
    return build_registered_model(
        cfg,
        num_classes=int(data_cfg["num_classes"]),
    )
