from __future__ import annotations

from typing import Any

from pulsecls.engine.model import Model
from pulsecls.models.classify.predict import ClassificationPredictor
from pulsecls.models.classify.train import ClassificationTrainer
from pulsecls.nn.tasks import build_model


class PulseCLS(Model):
    """Ultralytics-style facade for pulse-sequence classification.

    Examples:
        >>> from pulsecls import PulseCLS
        >>> model = PulseCLS("configs/pipeline/modular_multigrain_iq_synthetic.yaml")
        >>> model.train(epochs=20, device="npu:0")
        >>> model.predict(checkpoint="results/.../best_model.pt", split="val")
    """

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {
            "classify": {
                "model": build_model,
                "trainer": ClassificationTrainer,
                "predictor": ClassificationPredictor,
            }
        }
