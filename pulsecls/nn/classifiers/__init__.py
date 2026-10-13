"""Classifier assemblies."""

from pulsecls.nn.classifiers.atenet import ATENetClassifier
from pulsecls.nn.classifiers.baseline import ResNet1DClassifier
from pulsecls.nn.classifiers.invconv import InvConvClassifier
from pulsecls.nn.classifiers.mambasl import MambaSLClassifier
from pulsecls.nn.classifiers.modern_tcn import ModernTCNClassifier
from pulsecls.nn.classifiers.mptsnet import MPTSNetClassifier
from pulsecls.nn.classifiers.resnet1d_multigrain import (
    ModularResNet1DClassifier,
)
from pulsecls.nn.classifiers.resnet_softmax import ResNetSoftmaxClassifier

__all__ = [
    "ATENetClassifier",
    "InvConvClassifier",
    "MambaSLClassifier",
    "ModernTCNClassifier",
    "MPTSNetClassifier",
    "ModularResNet1DClassifier",
    "ResNet1DClassifier",
    "ResNetSoftmaxClassifier",
]
