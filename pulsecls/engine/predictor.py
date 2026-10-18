from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pulsecls.cfg import normalize_dataset_path, remap_repo_path, validate_num_classes

from pulsecls.data.dataset import PulseArrayDataset, load_split_arrays, zero_padding_enabled
from pulsecls.data.selection import validation_enabled
from pulsecls.nn.tasks import build_model
from pulsecls.utils.device import amp_dtype_for, autocast_context, get_device, pin_memory_for, use_amp_for
from pulsecls.utils.metrics import compute_metrics, save_classification_outputs
from pulsecls.utils.normalization import FeatureNormalizer
from pulsecls.utils.progress import Progress


class BasePredictor:
    """Task-neutral inference loop for pulse-sequence checkpoints."""

    def __init__(self, pipeline: dict[str, Any], checkpoint_path: str | Path,
                 *, use_checkpoint_data: bool = False) -> None:
        self.pipeline = pipeline
        self.output_dir = Path(pipeline["output_dir"])
        self.checkpoint_path = Path(checkpoint_path)
        inference_cfg = pipeline.get("inference", {})
        configured_classes = validate_num_classes(inference_cfg.get("num_classes"), "inference.num_classes")
        self.show_progress = bool(inference_cfg.get("progress", True))
        device_spec = inference_cfg.get("device")
        if device_spec is None:
            device_spec = pipeline.get("train", {}).get("device")
        self.device = get_device(device_spec)
        self.use_amp = use_amp_for(self.device, inference_cfg.get("use_amp", False))
        self.amp_dtype = amp_dtype_for(self.device, inference_cfg.get("amp_dtype", pipeline.get("train", {}).get("amp_dtype", "float16")))
        print(f"Using inference device: {self.device}")
        with Progress(f"Load checkpoint {self.checkpoint_path.name}", unit="steps", enabled=self.show_progress):
            self.checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        self.data_cfg = dict(self.checkpoint.get("data_cfg", pipeline["data_cfg"]))
        self.checkpoint_num_classes = int(self.data_cfg["num_classes"])
        self.num_classes = configured_classes if configured_classes is not None else self.checkpoint_num_classes
        # Folder inputs have no training data.yaml; the checkpoint defines their class/feature semantics.
        requested = pipeline["data_cfg"]
        if not use_checkpoint_data:
            self.data_cfg["slicing"] = requested.get("slicing", {})
            if requested.get("features", self.data_cfg.get("features")) != self.data_cfg.get("features", requested.get("features")) or (configured_classes is None and int(requested.get("num_classes", self.checkpoint_num_classes)) != self.checkpoint_num_classes):
                raise ValueError(
                    f"Split inference configuration does not match checkpoint {self.checkpoint_path}: "
                    f"pipeline classes={requested.get('num_classes')}, checkpoint classes={self.data_cfg['num_classes']}; "
                    "use the training data configuration or specify source=<CSV folder> for checkpoint-based inference"
                )
        input_keys = (("sequence_length", "length_policy") if use_checkpoint_data else
                      ("root", "all", "selection", "train", "val", "test", "sequence_length", "length_policy", "cache"))
        for key in input_keys:
            if key in requested:
                self.data_cfg[key] = requested[key]
        self.data_cfg["num_classes"] = self.num_classes
        if isinstance(self.data_cfg.get("class_names"), list):
            names = self.data_cfg["class_names"][:self.num_classes]
            self.data_cfg["class_names"] = names + [str(i) for i in range(len(names), self.num_classes)]
        if self.data_cfg.get("root"):
            self.data_cfg["root"] = str(remap_repo_path(self.data_cfg["root"]))
        for split in ("train", "val", "test"):
            if self.data_cfg.get(split):
                self.data_cfg[split] = normalize_dataset_path(self.data_cfg[split])
        self.model_cfg = self.checkpoint.get("model_cfg", pipeline["model_cfg"])
        # Load the entire learned classifier strictly at its original width, then adapt its logits.
        self.model_data_cfg = {**self.data_cfg, "num_classes": self.checkpoint_num_classes}
        print(f"Inference classes: {self.num_classes}; checkpoint classes: {self.checkpoint_num_classes}", flush=True)
        if self.num_classes < self.checkpoint_num_classes:
            print(f"Using pipeline label range 0..{self.num_classes - 1}; higher checkpoint labels are excluded.", flush=True)
        elif self.num_classes > self.checkpoint_num_classes:
            print(f"Added {self.num_classes - self.checkpoint_num_classes} output classes without trained weights; their prediction probability is 0.", flush=True)
        with Progress("Initialize inference model", unit="steps", enabled=self.show_progress):
            self.model = self.get_model().to(self.device)
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
            self.model.eval()
        self.normalizer = FeatureNormalizer.from_checkpoint(self.checkpoint, len(self.data_cfg["features"]))
        self.mean, self.std = self.normalizer.mean, self.normalizer.std
        print(f"Checkpoint normalization: {self.normalizer.config['method']}", flush=True)
        class_counts = np.asarray(
            self.checkpoint.get(
                "train_class_counts",
                np.ones(self.checkpoint_num_classes, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        if class_counts.shape != (self.checkpoint_num_classes,):
            raise ValueError("Checkpoint train_class_counts must match its trained class count")
        class_counts = class_counts[:self.num_classes]
        if self.num_classes > self.checkpoint_num_classes:
            class_counts = np.pad(class_counts, (0, self.num_classes - self.checkpoint_num_classes), constant_values=1)
        self.log_priors = torch.log(
            torch.from_numpy(np.maximum(class_counts, 1.0))
        ).to(self.device)
        self.prior_alpha = float(
            inference_cfg.get(
                "prior_alpha",
                self.checkpoint.get("metrics_prior_alpha", 0.0),
            )
        )

    def get_model(self):
        return build_model(self.model_cfg, self.model_data_cfg)

    def _adapt_logits(self, logits):
        """Keep label IDs stable; new untrained classes receive zero probability."""
        if logits.ndim != 2 or logits.size(1) != self.checkpoint_num_classes:
            raise ValueError("Model logits do not match the checkpoint's trained class count")
        if self.num_classes <= self.checkpoint_num_classes:
            return logits[:, :self.num_classes]
        extra = logits.new_full((logits.size(0), self.num_classes - self.checkpoint_num_classes), float("-inf"))
        return torch.cat((logits, extra), dim=1)

    @torch.no_grad()
    def _predict_dataset(self, dataset):
        loader = DataLoader(
            dataset,
            batch_size=int(self.pipeline.get("inference", {}).get("batch_size", self.pipeline.get("train", {}).get("batch_size", 256))),
            shuffle=False, num_workers=0, pin_memory=pin_memory_for(self.device),
        )
        predictions, confidence = [], []
        label = self.pipeline.get("inference", {}).get("progress_label", "Inference")
        lazy = hasattr(dataset, "on_loaded")
        with Progress("Read input CSV", len(dataset), enabled=self.show_progress and lazy, position=0) as loading, \
             Progress(label, len(dataset), enabled=self.show_progress, position=1 if lazy else 0) as progress:
            if lazy:
                dataset.on_loaded = loading.update
            try:
                progress.stage("initialize data loader")
                iterator = iter(loader)
                for batch_number in range(1, len(loader) + 1):
                    progress.stage(f"batch {batch_number}/{len(loader)}: loading" if lazy else f"batch {batch_number}/{len(loader)}")
                    features, _, _ = next(iterator)
                    progress.stage(f"batch {batch_number}/{len(loader)}: transfer to {self.device}")
                    features = features.to(self.device, non_blocking=True)
                    progress.stage(f"batch {batch_number}/{len(loader)}: predicting")
                    with autocast_context(self.device, self.use_amp, self.amp_dtype):
                        logits = self._adapt_logits(self.model(features)["logits"])
                    logits = logits.float()
                    if self.prior_alpha != 0.0:
                        logits = logits - self.prior_alpha * self.log_priors.to(logits.device, logits.dtype)
                    probs = torch.softmax(logits, dim=1)
                    score, predicted = probs.max(dim=1)
                    progress.stage(f"batch {batch_number}/{len(loader)}: collect predictions")
                    predictions.append(predicted.cpu().numpy())
                    confidence.append(score.cpu().numpy())
                    progress.update(features.size(0))
                progress.stage("done")
            finally:
                if lazy:
                    dataset.on_loaded = None
        return np.concatenate(predictions), np.concatenate(confidence)

    def predict_paths(self, paths):
        from pulsecls.data.inference import InferenceCSVDataset
        dataset = InferenceCSVDataset(paths, self.data_cfg, self.mean, self.std, normalizer=self.normalizer)
        return self._predict_dataset(dataset)

    def predict_folder(self, source, labels_file=None, save_dir=None):
        from pulsecls.data.inference import find_inference_csvs, read_labels
        paths = find_inference_csvs(source, progress=self.show_progress)
        with Progress("Read inference labels", unit="steps", enabled=self.show_progress and bool(labels_file)):
            labels = read_labels(labels_file, paths, self.num_classes)
        predictions, confidence = self.predict_paths(paths)
        target = Path(save_dir) if save_dir is not None else self.output_dir / "infer_test"
        return save_prediction_outputs(target, paths, predictions, confidence, labels, self.num_classes,
                                       progress=self.show_progress)

    def predict_split(self, split_key="val", save_dir=None):
        if split_key == "val" and not validation_enabled(self.data_cfg):
            raise ValueError("Validation is disabled (val_ratio=0); choose split='train' or configure a validation split")
        data, labels, paths = load_split_arrays({**self.data_cfg, "progress": self.show_progress}, split_key)
        if not len(labels):
            raise ValueError(f"No sequences in requested split: {split_key}")
        dataset = PulseArrayDataset(data, labels, self.mean, self.std, zero_padding=zero_padding_enabled(self.data_cfg),
                                    normalizer=self.normalizer)
        predictions, confidence = self._predict_dataset(dataset)
        target = Path(save_dir) if save_dir is not None else self.output_dir / f"infer_{split_key}"
        return save_prediction_outputs(target, paths, predictions, confidence, labels, self.num_classes,
                                       prefix=split_key, active_labels=(self.checkpoint.get("active_labels") if self.num_classes == self.checkpoint_num_classes else None), progress=self.show_progress)


def save_prediction_outputs(target_dir, paths, predictions, confidence, labels, num_classes,
                            prefix="test", active_labels=None, extra_columns=None, progress=True):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    columns = {"filename": [Path(p).name for p in paths], "path": [str(p) for p in paths],
               "prediction": predictions, "confidence": confidence}
    if extra_columns:
        columns.update(extra_columns)
    if labels is not None:
        columns.update(label=labels, correct=labels == predictions)
    csv_path = target_dir / "predictions.csv"
    with Progress("Write predictions.csv", unit="steps", enabled=progress):
        pd.DataFrame(columns).to_csv(csv_path, index=False, encoding="utf-8-sig")
    result = {"num_sequences": len(paths), "num_classes": int(num_classes), "evaluated": labels is not None, "predictions_csv": str(csv_path)}
    if labels is not None:
        with Progress("Evaluate predictions", unit="steps", enabled=progress):
            metrics = compute_metrics(labels, predictions, num_classes, active_labels=active_labels)
            result.update(metrics)
            save_classification_outputs(target_dir, labels, predictions, num_classes, prefix=prefix)
        print(f"Inference {prefix}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    else:
        for name in ("metrics.json", f"{prefix}_confusion_matrix.csv", f"{prefix}_confusion_matrix_normalized.csv",
                     f"{prefix}_classification_report.csv", f"{prefix}_classification_report.json"):
            (target_dir / name).unlink(missing_ok=True)
    (target_dir / "inference_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if labels is not None:
        (target_dir / "metrics.json").write_text(json.dumps({k: v for k, v in result.items() if k not in {"predictions_csv", "evaluated", "num_sequences", "num_classes"}}, indent=2), encoding="utf-8")
    print(f"Saved predictions to: {csv_path}")
    return result


# Historical name used by scripts and tools.
Inferencer = BasePredictor
