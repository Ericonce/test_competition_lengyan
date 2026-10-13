from __future__ import annotations

from pathlib import Path
from typing import Any

from pulsecls.cfg import DEFAULT_CFG_DICT, get_cfg, load_yaml, normalize_task


class Model:
    """Unified facade for train / val / predict, following Ultralytics Model."""

    def __init__(
        self,
        model: str | Path | None = None,
        task: str | None = None,
    ) -> None:
        self.task = normalize_task(task or DEFAULT_CFG_DICT.get("task", "classify"))
        self.predictor = None
        self.trainer = None
        self.metrics = None
        self.pipeline: dict[str, Any] | None = None
        self.overrides: dict[str, Any] = {}
        self.model_name = None if model is None else str(model)
        if model is not None:
            self._load_source(model)

    def _load_source(self, model: str | Path) -> None:
        path = Path(model)
        if path.suffix.lower() in {".yaml", ".yml"}:
            loaded = load_yaml(path)
            if "data" in loaded and "model" in loaded:
                self.pipeline = get_cfg(path)
                self.task = normalize_task(self.pipeline.get("task", self.task))
                return
            self.overrides["model"] = str(path)
            return
        self.overrides["checkpoint"] = str(path)

    def __call__(self, *args: Any, **kwargs: Any):
        return self.predict(*args, **kwargs)

    def train(self, **kwargs: Any) -> dict[str, float]:
        pipeline = self._resolve_pipeline(**kwargs)
        trainer_cls = self._smart_load("trainer")
        self.trainer = trainer_cls(pipeline)
        self.metrics = self.trainer.fit()
        return self.metrics

    def predict(
        self,
        checkpoint: str | Path | None = None,
        split: str | None = None,
        save_dir: str | Path | None = None,
        source: str | Path | None = None,
        labels_file: str | Path | None = None,
        num_classes: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if num_classes is not None:
            kwargs["inference.num_classes"] = num_classes
        pipeline = self._resolve_pipeline(**kwargs)
        from pulsecls.data.selection import validation_enabled
        has_val = validation_enabled(pipeline["data_cfg"])
        inference_cfg = pipeline.get("inference", {})
        checkpoint = (
            checkpoint
            or inference_cfg.get("checkpoint")
            or self.overrides.get("checkpoint")
            or (Path(pipeline["output_dir"]) / ("best_model.pt" if has_val else "final_model.pt"))
        )
        save_dir = save_dir or inference_cfg.get("save_dir")
        split = split or inference_cfg.get("split") or ("val" if has_val else "train")
        predictor_cls = self._smart_load("predictor")
        source = source or inference_cfg.get("source")
        self.predictor = predictor_cls(pipeline, checkpoint, use_checkpoint_data=source is not None)
        labels_file = labels_file if labels_file is not None else inference_cfg.get("labels_file")
        if source is not None:
            self.metrics = self.predictor.predict_folder(source, labels_file=labels_file, save_dir=save_dir)
        else:
            if labels_file:
                raise ValueError("labels_file requires an inference source folder")
            self.metrics = self.predictor.predict_split(split, save_dir=save_dir)
        return self.metrics

    def infer(self, **kwargs: Any):
        """Alias for predict(), accepting folder paths and optional label TXT."""
        return self.predict(**kwargs)

    def infer_vote(self, config, **kwargs: Any):
        from pulsecls.engine.voter import VotePredictor
        self.predictor = VotePredictor(config, **kwargs)
        self.metrics = self.predictor.predict()
        return self.metrics

    def val(self, **kwargs: Any) -> dict[str, float]:
        kwargs.setdefault("split", "val")
        return self.predict(**kwargs)

    def _resolve_pipeline(self, **kwargs: Any) -> dict[str, Any]:
        source = kwargs.pop("pipeline", None)
        if source is None:
            if self.pipeline is not None:
                source = self.pipeline
            elif self.model_name and str(self.model_name).endswith((".yaml", ".yml")):
                source = self.model_name
            else:
                source = DEFAULT_CFG_DICT["pipeline"]
        overrides = {**self.overrides, **kwargs}
        return get_cfg(source, overrides=overrides)

    def _smart_load(self, key: str):
        try:
            return self.task_map[self.task][key]
        except Exception as exc:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support '{key}' for task '{self.task}'"
            ) from exc

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        raise NotImplementedError("Please provide task_map for your model")
