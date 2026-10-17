from __future__ import annotations

import json
import random
import warnings
from copy import deepcopy
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from pulsecls.cfg import TRAIN_DEFAULTS, get_cfg, portable_data_cfg, remap_repo_path

from pulsecls.data.augment import PulseAugment
from pulsecls.data.selection import select_records, validation_enabled
from pulsecls.data.dataset import (
    PulseArrayDataset,
    MergeLastSingletonBatchSampler,
    load_split_arrays,
    make_balanced_sampler,
    zero_padding_enabled,
)
from pulsecls.losses.classification import build_classification_loss
from pulsecls.losses.multigrain import MultiGranularityObjective
from pulsecls.nn.tasks import build_model
from pulsecls.utils.device import (
    amp_dtype_for,
    autocast_context,
    build_grad_scaler,
    device_name,
    get_device,
    pin_memory_for,
    use_amp_for,
)
from pulsecls.utils.metrics import compute_metrics, save_classification_outputs
from pulsecls.utils.seed import seed_everything
from pulsecls.utils.normalization import FeatureNormalizer
from pulsecls.utils.progress import Progress
from pulsecls.utils.training_outputs import (
    clean_analysis_outputs, reporting_config, save_configuration, save_predictions,
    save_training_figures, save_tsne, select_tsne_indices, training_log,
)


def _cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return deepcopy(value)


def _save_checkpoint(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class BaseTrainer:
    """Task-neutral training loop. Task packages override hooks, not this loop."""

    DEFAULT_TRAIN_PARAMS = deepcopy(TRAIN_DEFAULTS)

    def __init__(self, pipeline: dict[str, Any] | str | Path, **overrides: Any) -> None:
        pipeline = get_cfg(pipeline, overrides=overrides)
        self.pipeline = pipeline
        self.data_cfg = pipeline["data_cfg"]
        self.model_cfg = pipeline["model_cfg"]
        self.train_cfg = pipeline.get("train", {})
        if self.train_cfg.get("num_classes") is not None:
            self.data_cfg["num_classes"] = self.train_cfg["num_classes"]
        self.loss_cfg = pipeline.get("loss", {})
        self.output_dir = Path(pipeline["output_dir"])
        self.num_classes = int(self.data_cfg["num_classes"])
        self.device = get_device(self.train_cfg.get("device"))
        self.amp_dtype = amp_dtype_for(self.device, self.train_cfg.get("amp_dtype", "float16"))
        channels = len(self.data_cfg.get("features", ["CF_MHz", "PW_us", "PRI_us"]))
        self.normalizer = FeatureNormalizer.from_checkpoint(
            {"mean": np.zeros(channels, np.float32), "std": np.ones(channels, np.float32)}, channels).to(self.device)
        self.normalization_mean, self.normalization_std = self.normalizer.offset, self.normalizer.scale

    def _build_loaders(self) -> tuple[DataLoader, DataLoader | None, np.ndarray, np.ndarray, np.ndarray, list[int]]:
        self.selection_info = None
        selection_kwargs = {}
        if "selection" in self.data_cfg:
            selection_plan = select_records(self.data_cfg)
            _, self.selection_info = selection_plan
            selection_kwargs["selection_plan"] = selection_plan
            info = self.selection_info
            print(f"Data selection: total={info['total_count']}, sampled={info['sampled_count']}, train={info['train_count']}, val={info['val_count']}")
        train_data, train_labels, self.train_paths = load_split_arrays(self.data_cfg, "train", **selection_kwargs)
        has_val = validation_enabled(self.data_cfg)
        if has_val:
            val_data, val_labels, self.val_paths = load_split_arrays(self.data_cfg, "val", **selection_kwargs)
            if not len(val_labels):
                raise ValueError("Validation split is empty; set dataset.val_ratio=0 or increase sample_ratio")
        else:
            val_labels = np.empty(0, dtype=np.int64)
            self.val_paths = []
        with Progress("Prepare raw train/val data and normalization statistics", unit="steps", enabled=self.data_cfg.get("progress", True)):
            zero_padding = zero_padding_enabled(self.data_cfg)
            resumed = getattr(self, "resume_checkpoint", None)
            if resumed is not None:
                self.normalizer = FeatureNormalizer.from_checkpoint(resumed, train_data.shape[-1]).to(self.device)
                print(f"Resume normalization restored: {self.normalizer.config['method']}", flush=True)
            else:
                self.normalizer = FeatureNormalizer.fit(train_data, self.train_cfg.get("normalization"),
                                                       ignore_zero_padding=zero_padding).to(self.device)
            self.train_cfg["normalization"] = deepcopy(self.normalizer.config)
            mean, std = self.normalizer.mean, self.normalizer.std
            self.normalization_mean = torch.as_tensor(mean, device=self.device).view(1, -1, 1)
            self.normalization_std = torch.as_tensor(std, device=self.device).view(1, -1, 1)
            train_dataset = PulseArrayDataset(train_data, train_labels, mean, std, zero_padding=zero_padding, normalize=False)
            val_dataset = PulseArrayDataset(val_data, val_labels, mean, std, zero_padding=zero_padding, normalize=False) if has_val else None

        sampler = None
        shuffle = True
        sampler_name = self.train_cfg.get("sampler", "random")
        if sampler_name in {"class_balanced", "power_balanced", "sqrt_balanced"}:
            sampler_power = float(
                self.train_cfg.get(
                    "sampler_power",
                    0.5 if sampler_name == "sqrt_balanced" else 1.0,
                )
            )
            sampler = make_balanced_sampler(
                train_labels,
                self.num_classes,
                power=sampler_power,
            )
            shuffle = False

        batch_size = int(self.train_cfg.get("batch_size", 256))
        if len(train_dataset) < 2:
            raise ValueError("Training requires at least two sampled sequences; increase sample_ratio")
        if batch_size > 1 and len(train_dataset) % batch_size == 1:
            train_loader = DataLoader(train_dataset,
                batch_sampler=MergeLastSingletonBatchSampler(sampler if sampler is not None else RandomSampler(train_dataset), batch_size, drop_last=False),
                num_workers=int(self.train_cfg.get("num_workers", 0)), pin_memory=pin_memory_for(self.device))
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                num_workers=int(self.train_cfg.get("num_workers", 0)), pin_memory=pin_memory_for(self.device))
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(self.train_cfg.get("batch_size", 256)),
            shuffle=False,
            num_workers=int(self.train_cfg.get("num_workers", 0)),
            pin_memory=pin_memory_for(self.device),
        ) if has_val else None
        active_labels = sorted(set(train_labels.tolist()) | set(val_labels.tolist()))
        return train_loader, val_loader, train_labels, val_labels, np.stack([mean, std]), active_labels

    def get_model(self):
        return build_model(self.model_cfg, self.data_cfg)

    def _criterion(self, labels: np.ndarray) -> nn.Module:
        return build_classification_loss(
            self.loss_cfg.get("ce", {}),
            labels,
            self.num_classes,
            self.device,
        )

    def _normalize_features(self, features):
        """Apply fixed training statistics only after the raw augmentation chain."""
        return self.normalizer(features, zero_padding=zero_padding_enabled(self.data_cfg))

    def _augment_features(self, augment, features, labels, *, allow_mixup=True):
        lengths = None
        if zero_padding_enabled(self.data_cfg):
            valid = ~features.eq(0).all(dim=1)
            positions = torch.arange(1, features.size(-1) + 1, device=features.device)
            lengths = (valid * positions).amax(dim=-1)
        return augment.apply(features, labels, allow_mixup=allow_mixup, valid_lengths=lengths)

    def _loss_batch(
        self,
        model: nn.Module,
        features: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
        augment: PulseAugment,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        stream_weight = float(
            self.loss_cfg.get("stream_infonce", {}).get("weight", 0.0)
        )
        pulse_weight = float(
            self.loss_cfg.get("pulse_infonce", {}).get("weight", 0.0)
        )
        adversarial_weight = float(
            self.loss_cfg.get("adversarial", {}).get("weight", 0.0)
        )
        stream_mode = str(
            self.loss_cfg.get("stream_infonce", {}).get("mode", "supervised")
        ).lower()
        # Classwise stream InfoNCE uses one augmented view per stream and
        # builds positives/negatives from labels in the same batch. It does
        # not need the old two-view instance construction.
        stream_requires_two_views = (
            stream_weight > 0.0
            and stream_mode not in {"classwise", "classwise_pair"}
        )
        use_two_views = training and (
            stream_requires_two_views
            or pulse_weight > 0.0
            or adversarial_weight > 0.0
        )

        if use_two_views:
            view_a, _ = self._augment_features(augment, features, labels, allow_mixup=False)
            view_b, _ = self._augment_features(augment, features, labels, allow_mixup=False)
            out_a = model(self._normalize_features(view_a))
            out_b = model(self._normalize_features(view_b))
            breakdown = self.objective(out_a, out_b, labels)
            return (
                breakdown.total,
                breakdown.logits,
                breakdown.as_tensors(),
            )

        mix = None
        use_contrast = training and (
            stream_weight > 0.0
            or pulse_weight > 0.0
            or adversarial_weight > 0.0
        )
        if training:
            if use_contrast:
                features, _ = self._augment_features(augment, features, labels, allow_mixup=False)
            else:
                features, mix = self._augment_features(augment, features, labels)
        outputs = model(self._normalize_features(features))
        if use_contrast:
            # A classwise stream objective needs only one forward pass. The
            # objective ignores out_b for this mode, while keeping one common
            # loss path for the trainer.
            breakdown = self.objective(outputs, outputs, labels)
            return (
                breakdown.total,
                breakdown.logits,
                breakdown.as_tensors(),
            )

        logits = outputs["logits"]
        ce_loss = criterion(logits, labels)
        if mix is not None:
            partner_loss = criterion(logits, labels[mix.partner_index])
            ce_loss = mix.lam * ce_loss + (1.0 - mix.lam) * partner_loss
        zero = ce_loss.detach().new_zeros(())
        parts = {
            "ce_loss": ce_loss.detach(),
            "stream_infonce_loss": zero,
            "pulse_infonce_loss": zero,
            "adversarial_loss": zero,
            "total_loss": ce_loss.detach(),
        }
        return ce_loss, logits, parts

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scaler: Any,
        augment: PulseAugment,
        log_priors: torch.Tensor | None = None,
        metric_prior_alpha: float = 0.0,
        progress_description: str | None = None,
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        training = optimizer is not None
        model.train(training)
        use_amp = use_amp_for(self.device, self.train_cfg.get("use_amp", True))
        loss_keys = ("ce_loss", "stream_infonce_loss", "pulse_infonce_loss", "adversarial_loss", "total_loss")
        n_items = 0
        all_labels: list[np.ndarray] = []
        all_preds: list[torch.Tensor] = []

        totals = torch.zeros(len(loss_keys), device=self.device)
        metric_interval = int(self.train_cfg.get("progress_metric_interval", 10))
        if metric_interval < 1:
            raise ValueError("progress_metric_interval must be positive")

        n_correct = torch.zeros((), device=self.device)
        description = progress_description or ("Train" if training else "Validation")
        batch_count = len(loader)
        with Progress(description, batch_count, unit="batch",
                      enabled=self.data_cfg.get("progress", True)) as progress:
            progress.stage("initialize data loader")
            iterator = iter(loader)
            for batch_number in range(1, batch_count + 1):
                batch = f"batch {batch_number}/{batch_count}"
                progress.stage(f"{batch}: loading")
                features, labels, _ = next(iterator)
                all_labels.append(labels.numpy())
                progress.stage(f"{batch}: transfer to {self.device}")
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    progress.stage(f"{batch}: forward / loss")
                    with autocast_context(self.device, use_amp, self.amp_dtype):
                        loss, logits, parts = self._loss_batch(
                            model,
                            features,
                            labels,
                            criterion,
                            augment,
                            training,
                        )
                    if training:
                        progress.stage(f"{batch}: backward")
                        if use_amp:
                            scaler.scale(loss).backward()
                            progress.stage(f"{batch}: optimizer step")
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            progress.stage(f"{batch}: optimizer step")
                            optimizer.step()
                progress.stage(f"{batch}: collect metrics")
                batch_size = labels.size(0)
                n_items += batch_size
                totals += torch.stack([parts[key].float() for key in loss_keys]) * batch_size
                metric_logits = logits
                if metric_prior_alpha != 0.0 and log_priors is not None:
                    metric_logits = logits - metric_prior_alpha * log_priors.to(
                        device=logits.device,
                        dtype=logits.dtype,
                    )
                prediction = metric_logits.detach().argmax(dim=1)
                all_preds.append(prediction)
                n_correct += (prediction == labels).sum()
                if batch_number % metric_interval == 0 or batch_number == batch_count:
                    loss_value, accuracy = (torch.stack((totals[-1], n_correct)) / n_items).cpu().tolist()
                    progress.stage(f"{batch}: completed, loss={loss_value:.4f}, acc={accuracy:.4f}")
                else:
                    progress.stage(f"{batch}: completed")
                progress.update()
            values = (totals / max(n_items, 1)).cpu().tolist()
            averaged = dict(zip(loss_keys, values))
            predictions = torch.cat(all_preds).cpu().numpy()
            progress.stage(f"done, loss={averaged['total_loss']:.4f}")

        return averaged, np.concatenate(all_labels), predictions

    def _collect_predictions(self, model, loader, indices, log_priors, metric_prior_alpha):
        """Sequential, unaugmented evaluation; retain features only for selected t-SNE samples."""
        model.eval()
        selected = set(indices.tolist())
        embeddings, embedded_indices, labels_out, predictions, confidence = [], [], [], [], []
        with Progress("Collect final evaluation predictions", len(loader.dataset),
                      enabled=self.data_cfg.get("progress", True)) as progress, torch.no_grad():
            for features, labels, sample_indices in loader:
                with autocast_context(self.device, use_amp_for(self.device, self.train_cfg.get("use_amp", True)), self.amp_dtype):
                    outputs = model(self._normalize_features(features.to(self.device, non_blocking=True)))
                logits = outputs["logits"].float()
                if metric_prior_alpha != 0:
                    logits = logits - metric_prior_alpha * log_priors.to(logits.device, dtype=logits.dtype)
                score, prediction = logits.softmax(dim=1).max(dim=1)
                labels_out.append(labels.numpy())
                predictions.append(prediction.cpu().numpy())
                confidence.append(score.cpu().numpy())
                batch_indices = sample_indices.numpy()
                keep = np.array([i in selected for i in batch_indices])
                if keep.any():
                    embedding = outputs["stream_embedding"]
                    if embedding.ndim != 2:
                        raise ValueError("t-SNE requires [batch, features] stream_embedding")
                    positions = torch.as_tensor(np.flatnonzero(keep), device=embedding.device, dtype=torch.long)
                    embeddings.append(embedding.index_select(0, positions).float().cpu().numpy())
                    embedded_indices.extend(batch_indices[keep].tolist())
                progress.update(len(labels))
        if len(indices) and embedded_indices != indices.tolist():
            raise ValueError("t-SNE samples are not aligned with evaluation records")
        return (np.concatenate(labels_out), np.concatenate(predictions), np.concatenate(confidence),
                np.concatenate(embeddings) if embeddings else None)

    def fit(self) -> dict[str, float]:
        self.reporting = reporting_config(self.train_cfg.get("reporting"))
        self.train_cfg["reporting"] = self.reporting
        with training_log(self.output_dir, resume=bool(self.train_cfg.get("resume", False))):
            print(f"Training classes: {self.num_classes}", flush=True)
            save_configuration(self.output_dir, self.pipeline)
            clean_analysis_outputs(self.output_dir)
            return self._fit_impl()

    def _fit_impl(self) -> dict[str, float]:
        additional_epochs = int(self.train_cfg.get("epochs", 20))
        if additional_epochs < 1:
            raise ValueError("epochs must be at least 1")
        resume = self.train_cfg.get("resume", False)
        if not isinstance(resume, bool):
            raise TypeError("resume must be true/false; specify the path in resume_checkpoint")
        self.resume_checkpoint = None
        self.resume_path = None
        if resume:
            source = self.train_cfg.get("resume_checkpoint")
            if not source:
                raise ValueError("resume=True requires resume_checkpoint")
            self.resume_path = remap_repo_path(source)
            self.resume_checkpoint = torch.load(self.resume_path, map_location="cpu", weights_only=False)
            saved_data = self.resume_checkpoint.get("data_cfg", {})
            for key in ("num_classes", "sequence_length", "features"):
                if key in saved_data and key in self.data_cfg and saved_data[key] != self.data_cfg[key]:
                    raise ValueError(f"Resume checkpoint data {key} does not match the current configuration")
            if "normalization" not in self.resume_checkpoint and ("mean" not in self.resume_checkpoint or "std" not in self.resume_checkpoint):
                raise ValueError("Resume checkpoint must contain normalization mean/std")
        seed_everything(int(self.train_cfg.get("seed", 2026)), device=self.device)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using device: {self.device}")
        name = device_name(self.device)
        if name:
            print(f"Accelerator: {name}")

        train_loader, val_loader, train_labels, _val_labels, stats, active_labels = self._build_loaders()
        has_val = val_loader is not None
        tsne_indices = (select_tsne_indices(_val_labels, self.reporting["tsne"]["max_samples"],
                                           self.reporting["tsne"]["seed"]) if has_val and self.train_cfg["visualize"] else np.empty(0, dtype=np.int64))
        if not has_val:
            print("Validation disabled: train all requested epochs; save final_model.pt; no early stopping or validation LR scheduling.")
            for filename in ("best_model.pt", "val_classification_report.csv", "val_classification_report.json", "val_confusion_matrix.csv"):
                (self.output_dir / filename).unlink(missing_ok=True)
        if self.selection_info is not None:
            (self.output_dir / "dataset_selection.json").write_text(json.dumps(self.selection_info, indent=2), encoding="utf-8")
        mean, std = stats[0], stats[1]
        print(f"Train feature mean: {mean.tolist()}")
        print(f"Train feature std: {std.tolist()}")
        print(f"Normalization: {self.normalizer.config}")
        class_counts = np.bincount(
            train_labels,
            minlength=self.num_classes,
        ).astype(np.float32)
        log_priors = torch.log(
            torch.from_numpy(np.maximum(class_counts, 1.0))
        ).to(self.device)
        metric_prior_alpha = float(self.train_cfg.get("metrics_prior_alpha", 0.0))

        model = self.get_model().to(self.device)
        criterion = self._criterion(train_labels)
        self.objective = MultiGranularityObjective(
            self.loss_cfg,
            criterion,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.train_cfg.get("lr", 1e-3)),
            weight_decay=float(self.train_cfg.get("weight_decay", 1e-4)),
            foreach=self.train_cfg.get("optimizer_foreach", False),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )
        scaler = build_grad_scaler(
            self.device,
            bool(self.train_cfg.get("use_amp", True)),
            self.amp_dtype,
        )
        augment = PulseAugment(**self.pipeline.get("augment", {}))

        normalization = self.normalizer.export_state()
        np.savez(self.output_dir / "normalization.npz", mean=mean, std=std,
                 method=self.normalizer.config["method"], offset=normalization["offset"], scale=normalization["scale"],
                 state_json=json.dumps(normalization))
        (self.output_dir / "normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")

        history: list[dict[str, float | int]] = []
        best_macro_f1 = -1.0
        best_epoch = 0
        stale_epochs = 0
        patience = int(self.train_cfg.get("early_stopping_patience", 6))
        start_epoch = 0
        best_checkpoint = None
        if self.resume_checkpoint is not None:
            saved = self.resume_checkpoint
            model.load_state_dict(saved["model_state_dict"], strict=True)
            start_epoch = int(saved.get("epoch", 0))
            history = deepcopy(saved.get("history", []))
            if "history" not in saved:
                history_file = self.resume_path.parent / "training_history.csv"
                if history_file.exists():
                    frame = pd.read_csv(history_file)
                    history = frame[frame["epoch"] <= start_epoch].to_dict("records")
            if has_val and saved.get("validation_enabled", True) and saved.get("data_cfg", {}).get("selection") == self.data_cfg.get("selection"):
                best_checkpoint = _cpu_snapshot(saved.get("best_checkpoint") or saved)
                best_epoch = int(best_checkpoint.get("epoch", start_epoch))
                best_macro_f1 = float(best_checkpoint.get("val_metrics", {}).get("macro_f1", -1.0))
            if "optimizer_state_dict" in saved:
                optimizer.load_state_dict(saved["optimizer_state_dict"])
                scheduler.load_state_dict(saved["scheduler_state_dict"])
                scaler.load_state_dict(saved.get("scaler_state_dict", {}))
            else:
                warnings.warn("Legacy weight-only checkpoint: model/normalization/epoch restored; optimizer, scheduler and AMP states start fresh.", stacklevel=2)
            rng = saved.get("rng_state")
            if rng:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"])
                if self.device.type == "cuda" and "cuda" in rng and len(rng["cuda"]) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(rng["cuda"])
                if self.device.type == "npu" and "npu" in rng and len(rng["npu"]) == torch.npu.device_count():
                    torch.npu.set_rng_state_all(rng["npu"])
            if best_checkpoint is not None:
                _save_checkpoint(best_checkpoint, self.output_dir / "best_model.pt")
            print(f"Resume from epoch {start_epoch}; train {additional_epochs} additional epoch(s).")

        runtime = {"device": str(self.device), "accelerator": name, "validation_enabled": has_val,
                   "use_amp": use_amp_for(self.device, self.train_cfg.get("use_amp", True)),
                   "amp_dtype": str(self.amp_dtype), "normalization": deepcopy(self.normalizer.config),
                   "initial_epoch": start_epoch, "additional_epochs": additional_epochs,
                   "optimizer": type(optimizer).__name__,
                   "optimizer_initial_lr": [group["lr"] for group in optimizer.param_groups],
                   "optimizer_weight_decay": [group["weight_decay"] for group in optimizer.param_groups]}
        save_configuration(self.output_dir, self.pipeline, runtime=runtime, print_config=False)
        print(f"Effective runtime settings: {runtime}", flush=True)

        for epoch in range(start_epoch + 1, start_epoch + additional_epochs + 1):
            started = time.perf_counter()
            train_losses, train_y, train_pred = self._run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                augment,
                log_priors=log_priors,
                metric_prior_alpha=metric_prior_alpha,
                progress_description=f"Epoch {epoch}/{start_epoch + additional_epochs} Train",
            )
            train_metrics = compute_metrics(train_y, train_pred, self.num_classes, active_labels)
            val_losses, val_metrics = {}, {}
            if has_val:
                val_losses, val_y, val_pred = self._run_epoch(
                    model, val_loader, criterion, None, scaler, augment,
                    log_priors=log_priors, metric_prior_alpha=metric_prior_alpha,
                    progress_description=f"Epoch {epoch}/{start_epoch + additional_epochs} Validation",
                )
                val_metrics = compute_metrics(val_y, val_pred, self.num_classes, active_labels)
                scheduler.step(val_metrics["macro_f1"])
            elapsed = time.perf_counter() - started

            record: dict[str, float | int] = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": elapsed,
                "validation_enabled": has_val,
            }
            record.update({f"train_{key}": value for key, value in train_losses.items()})
            record.update({f"val_{key}": value for key, value in val_losses.items()})
            record.update({f"train_{key}": value for key, value in train_metrics.items()})
            record.update({f"val_{key}": value for key, value in val_metrics.items()})
            record.setdefault("val_accuracy", float("nan"))
            record.setdefault("val_macro_f1", float("nan"))
            history.append(record)
            pd.DataFrame(history).to_csv(
                self.output_dir / "training_history.csv",
                index=False,
                encoding="utf-8-sig",
            )

            metric_message = (f"val_acc={val_metrics['accuracy']:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f} | val_weighted_f1={val_metrics['weighted_f1']:.4f}" if has_val else f"train_acc={train_metrics['accuracy']:.4f} | train_macro_f1={train_metrics['macro_f1']:.4f} | validation=disabled")
            print(f"Epoch {epoch:02d} | loss={train_losses['total_loss']:.4f} | {metric_message} | lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

            improved = has_val and val_metrics["macro_f1"] > best_macro_f1
            if improved:
                best_macro_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                stale_epochs = 0
            elif has_val:
                stale_epochs += 1
            rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
            if self.device.type == "cuda":
                rng["cuda"] = torch.cuda.get_rng_state_all()
            elif self.device.type == "npu":
                rng["npu"] = torch.npu.get_rng_state_all()
            payload = {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "model_cfg": self.model_cfg, "data_cfg": portable_data_cfg(self.data_cfg),
                "mean": mean, "std": std, "normalization": normalization, "active_labels": active_labels,
                "train_class_counts": class_counts, "metrics_prior_alpha": metric_prior_alpha,
                "val_metrics": val_metrics, "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                "rng_state": rng, "history": history, "best_epoch": best_epoch,
                "best_macro_f1": best_macro_f1, "stale_epochs": stale_epochs,
                "validation_enabled": has_val,
            }
            if improved:
                best_checkpoint = _cpu_snapshot(payload)
                _save_checkpoint(best_checkpoint, self.output_dir / "best_model.pt")
            payload["best_checkpoint"] = best_checkpoint
            _save_checkpoint(payload, self.output_dir / "last_model.pt")
            if has_val and stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

        if not has_val:
            _save_checkpoint(payload, self.output_dir / "final_model.pt")
        checkpoint = torch.load(
            self.output_dir / ("best_model.pt" if has_val else "final_model.pt"),
            map_location=self.device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        evaluation_loader = val_loader if has_val else DataLoader(
            train_loader.dataset, batch_size=int(self.train_cfg.get("batch_size", 256)),
            shuffle=False, num_workers=int(self.train_cfg.get("num_workers", 0)), pin_memory=pin_memory_for(self.device))
        final_y, final_pred, confidence, embeddings = self._collect_predictions(
            model, evaluation_loader, tsne_indices, log_priors, metric_prior_alpha)
        final_metrics = compute_metrics(final_y, final_pred, self.num_classes, active_labels)
        evaluation_split = "val" if has_val else "train"
        save_classification_outputs(self.output_dir, final_y, final_pred, self.num_classes, prefix=evaluation_split)
        paths = self.val_paths if has_val else self.train_paths
        save_predictions(self.output_dir, evaluation_split, final_y, final_pred, confidence, paths)
        metadata = {"best_epoch": int(checkpoint["epoch"])} if has_val else {"epoch": int(checkpoint["epoch"])}
        metadata.update(validation_enabled=has_val, evaluation_split=evaluation_split)
        with (self.output_dir / "final_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(
                {**metadata, **final_metrics},
                file,
                ensure_ascii=False,
                indent=2,
            )
        print(f"{'Best' if has_val else 'Final'} epoch: {checkpoint['epoch']}")
        print(f"{evaluation_split} macro-F1: {final_metrics['macro_f1']:.6f}")
        if self.train_cfg["visualize"]:
            save_training_figures(self.output_dir, history, self.reporting, has_val=has_val,
                                  show_progress=self.data_cfg.get("progress", True))
            if has_val:
                save_tsne(self.output_dir, embeddings, tsne_indices, final_y, final_pred, paths, self.reporting,
                          show_progress=self.data_cfg.get("progress", True))
        else:
            print("Visualization disabled: saved metrics and checkpoints; skipped figures and t-SNE.")
        print(f"Saved training outputs to: {self.output_dir}", flush=True)
        return {**metadata, **final_metrics}


# Historical name used by scripts and tools.
Trainer = BaseTrainer
