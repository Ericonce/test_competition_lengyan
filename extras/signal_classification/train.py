"""Train pulse-description classifiers on legacy arrays or cut_data CSV slices."""

from __future__ import annotations

import argparse
import glob
import os
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data_process.augment import RadarAugment, parse_augment_methods
from data_process.dataset import SliceDataset, fit_normalization, load_cut_splits
from tool import pyutils


MODEL_CHOICES = [
    "ResNet18", "CNN", "EfficientNetB0", "MultiGrainCNN", "ATENet",
    "InvConv", "MambaSL", "ModernTCN", "MPTSNet",
]
FUSION_CHOICES = ["none", "vote", "score", "probability"]


def extract_logits(output):
    """Accept legacy tuple outputs, logits tensors, and migrated dict outputs."""
    if isinstance(output, dict):
        return output["logits"]
    if isinstance(output, (tuple, list)):
        return output[-1]
    return output


def class_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels[:, 0] if labels.ndim > 1 else labels


def synchronize_device(device: torch.device) -> None:
    """Synchronize asynchronous accelerators before reading wall-clock time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def load_pretrained_weights(
    model,
    checkpoint_path: str | Path,
    expected_model: str | None = None,
) -> dict[str, object]:
    """Load shape-compatible encoder weights and skip incompatible task heads."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported pretrained checkpoint type: {type(checkpoint).__name__}")
    checkpoint_model = checkpoint.get("model")
    if (
        expected_model
        and checkpoint_model
        and str(checkpoint_model).lower() != str(expected_model).lower()
    ):
        raise ValueError(
            f"Pretrained model mismatch: checkpoint={checkpoint_model!r}, "
            f"requested={expected_model!r}"
        )

    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state_dict, dict):
        raise TypeError("Pretrained checkpoint does not contain a state_dict")

    current_state = model.state_dict()
    compatible = {}
    skipped = []
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key in current_state and hasattr(value, "shape") and value.shape == current_state[key].shape:
            compatible[key] = value
        else:
            skipped.append(key)

    if not compatible:
        raise ValueError(
            f"No compatible parameters found in pretrained checkpoint: {checkpoint_path}"
        )
    result = model.load_state_dict(compatible, strict=False)
    summary = {
        "path": str(checkpoint_path),
        "loaded": len(compatible),
        "skipped": len(skipped),
        "missing": len(result.missing_keys),
        "unexpected": len(result.unexpected_keys),
    }
    print(
        "Loaded pretrained weights: "
        f"path={summary['path']}, loaded={summary['loaded']}, "
        f"skipped={summary['skipped']}, missing={summary['missing']}, "
        f"unexpected={summary['unexpected']}"
    )
    return summary


def average_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> float:
    labels = np.arange(num_classes) if num_classes is not None else None
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    counts = matrix.sum(axis=1)
    valid = counts > 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(matrix.diagonal()[valid] / counts[valid]))


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    counts = matrix.sum(axis=1)
    return np.divide(matrix.diagonal(), counts, out=np.full(num_classes, np.nan), where=counts != 0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, object]:
    per_class_f1 = (
        f1_score(
            y_true,
            y_pred,
            labels=np.arange(num_classes),
            average=None,
            zero_division=0,
        )
        if len(y_true)
        else np.zeros(num_classes, dtype=float)
    )
    return {
        "acc": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "aa": average_accuracy(y_true, y_pred, num_classes),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if len(y_true) else 0.0,
        "macro_f1": float(per_class_f1.mean()) if len(per_class_f1) else 0.0,
        "per_class_f1": per_class_f1,
        "per_class": per_class_accuracy(y_true, y_pred, num_classes),
    }


def _natural_key(value: str) -> tuple:
    parts = re.split(r"(\d+)", str(value))
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts)


def plot_confusion_matrix(y_true, y_pred, labels, output_path: str | Path, title: str):
    """Save a row-normalized confusion matrix as a PNG image."""
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    denominator = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix, dtype=float),
        where=denominator != 0,
    )
    figure_size = max(8.0, min(16.0, len(labels) * 0.65))
    fig, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(normalized, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(labels)), [str(label) for label in labels], rotation=90)
    axis.set_yticks(np.arange(len(labels)), [str(label) for label in labels])
    axis.set_ylabel("True class")
    axis.set_xlabel("Predicted class")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to: {output_path}")


def mixup_batch(features: torch.Tensor, labels: torch.Tensor, alpha: float):
    """Mix a batch of latent features and return the paired labels.

    The caller is responsible for deciding where in the model the feature
    tensor is obtained.  Keeping this operation independent from the input
    pipeline makes it usable for manifold Mixup at the classifier boundary.
    """
    if alpha <= 0:
        return features, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(features.size(0), device=features.device)
    return (
        lam * features + (1.0 - lam) * features[permutation],
        labels,
        labels[permutation],
        lam,
    )


def _manifold_mixup_classifier(model: torch.nn.Module) -> torch.nn.Module:
    """Find the final classifier whose input is the model representation.

    The project contains both legacy models (``fc2``/``linear``) and newer
    classifier wrappers (``classifier`` or ``core.classifier``).  Registering
    a pre-hook on this module lets all of them share the same manifold Mixup
    implementation without changing their forward contracts.
    """
    for attribute in ("fc2", "linear", "classifier", "fc"):
        candidate = getattr(model, attribute, None)
        if isinstance(candidate, torch.nn.Module):
            return candidate
    core = getattr(model, "core", None)
    for attribute in ("classifier", "fc"):
        candidate = getattr(core, attribute, None)
        if isinstance(candidate, torch.nn.Module):
            return candidate
    raise TypeError(
        f"{type(model).__name__} does not expose a supported final classifier; "
        "cannot apply manifold Mixup"
    )


def _manifold_mixup_forward(model, inputs, labels, alpha: float):
    """Run one forward pass with Mixup applied to the latent features."""
    classifier = _manifold_mixup_classifier(model)
    lam = 1.0
    labels_a, labels_b = labels, labels

    def mixup_hook(_module, module_inputs):
        nonlocal lam, labels_a, labels_b
        if not module_inputs or not isinstance(module_inputs[0], torch.Tensor):
            raise TypeError("The final classifier must receive a tensor feature input")
        features = module_inputs[0]
        if alpha <= 0:
            return module_inputs
        mixed_features, labels_a, labels_b, lam = mixup_batch(features, labels, alpha)
        return (mixed_features, *module_inputs[1:])

    handle = classifier.register_forward_pre_hook(mixup_hook)
    try:
        logits = extract_logits(model(inputs))
    finally:
        handle.remove()
    return logits, labels_a, labels_b, lam


def _batch_groups(groups, offset: int, size: int) -> list[str]:
    if groups is None:
        return [str(offset + index) for index in range(size)]
    return [str(group) for group in groups]


def aggregate_predictions(
    logits: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    mode: str,
):
    """Aggregate slice predictions per source with hard vote, logits, or probabilities."""
    if mode not in FUSION_CHOICES:
        raise ValueError(f"Unsupported fusion mode {mode!r}; choose from {FUSION_CHOICES}")
    slice_pred = logits.argmax(axis=1) if len(logits) else np.empty(0, dtype=np.int64)
    if mode == "none":
        return labels, slice_pred, groups

    pooled: dict[str, dict[str, object]] = {}
    num_classes = logits.shape[1]
    probabilities = None
    if mode == "probability":
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    for index, (logit, label, group) in enumerate(zip(logits, labels, groups)):
        item = pooled.setdefault(
            group,
            {
                "value": np.zeros(num_classes, dtype=np.float64),
                "label": int(label),
                "count": 0,
            },
        )
        if int(item["label"]) != int(label):
            raise ValueError(f"Source group {group!r} contains inconsistent labels")
        if mode == "vote":
            item["value"][slice_pred[index]] += 1.0
        elif mode == "score":
            item["value"] += logit
        else:
            item["value"] += probabilities[index]
        item["count"] = int(item["count"]) + 1
    pooled = dict(sorted(pooled.items(), key=lambda pair: _natural_key(pair[0])))
    aggregated_true = np.asarray([int(item["label"]) for item in pooled.values()], dtype=np.int64)
    aggregated_pred = np.asarray(
        [
            int(np.argmax(item["value"] / max(int(item["count"]), 1)))
            for item in pooled.values()
        ],
        dtype=np.int64,
    )
    return aggregated_true, aggregated_pred, list(pooled)


def vote_predictions(probabilities: np.ndarray, labels: np.ndarray, groups: list[str]):
    """Backward-compatible probability fusion helper."""
    probabilities = np.clip(probabilities, np.finfo(np.float64).tiny, None)
    return aggregate_predictions(np.log(probabilities), labels, groups, "probability")


def evaluate(model, loader, criterion, device, num_classes: int, fusion: str = "probability"):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_groups: list[str] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            inputs, labels = batch[:2]
            groups = batch[2] if len(batch) > 2 else None
            inputs = inputs.float().to(device)
            labels = class_labels(labels.long()).to(device)
            logits = extract_logits(model(inputs))
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_count += labels.size(0)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_groups.extend(_batch_groups(groups, batch_index * loader.batch_size, labels.size(0)))

    logits = np.concatenate(all_logits) if all_logits else np.empty((0, num_classes))
    labels = np.concatenate(all_labels) if all_labels else np.empty(0, dtype=np.int64)
    slice_pred = logits.argmax(axis=1) if len(logits) else np.empty(0, dtype=np.int64)
    slice_metrics = metrics(labels, slice_pred, num_classes)
    if len(logits):
        selected_true, selected_pred, selected_groups = aggregate_predictions(
            logits, labels, all_groups, fusion
        )
    else:
        selected_true, selected_pred = labels, slice_pred
        selected_groups = all_groups
    selected_metrics = metrics(selected_true, selected_pred, num_classes)
    return {
        "loss": total_loss / max(total_count, 1),
        "slice": slice_metrics,
        "selected": selected_metrics,
        "voted": selected_metrics,
        "true": selected_true,
        "pred": selected_pred,
        "groups": selected_groups,
        "fusion": fusion,
    }


def save_class_accuracy(path: str | Path, epoch: int, class_names, train_metrics, val_metrics) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("epoch class_index class_name train_slice_accuracy val_selected_accuracy")
        stream.write("\n")
        for class_index, class_name in enumerate(class_names):
            train_acc = train_metrics["per_class"][class_index]
            val_acc = val_metrics["per_class"][class_index]
            train_text = "NA" if np.isnan(train_acc) else f"{train_acc:.6f}"
            val_text = "NA" if np.isnan(val_acc) else f"{val_acc:.6f}"
            stream.write(f"{epoch}\t{class_index}\t{class_name}\t{train_text}\t{val_text}\n")


def add_bool_switch(parser: argparse.ArgumentParser, name: str, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    group.add_argument(f"--no-{name.replace('_', '-')}", dest=name, action="store_false")
    parser.set_defaults(**{name: False})


def train_one_epoch(model, loader, optimizer, criterion, device, args, num_classes: int):
    model.train()
    total_loss = 0.0
    total_count = 0
    true_labels: list[np.ndarray] = []
    predicted_labels: list[np.ndarray] = []
    for batch in tqdm(loader, desc="train", leave=False):
        inputs, labels = batch[:2]
        inputs = inputs.float().to(device)
        labels = class_labels(labels.long()).to(device)

        optimizer.zero_grad(set_to_none=True)
        if args.mixup:
            # Mix the representation immediately before the final classifier;
            # the raw input remains unchanged (manifold/feature Mixup).
            logits, labels_a, labels_b, lam = _manifold_mixup_forward(
                model, inputs, labels, args.mixup_alpha
            )
            loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
        else:
            logits = extract_logits(model(inputs))
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_count += labels.size(0)
        true_labels.append(labels.detach().cpu().numpy())
        predicted_labels.append(logits.detach().argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(true_labels) if true_labels else np.empty(0, dtype=np.int64)
    y_pred = np.concatenate(predicted_labels) if predicted_labels else np.empty(0, dtype=np.int64)
    return total_loss / max(total_count, 1), metrics(y_true, y_pred, num_classes)


def build_datasets(args):
    if args.dataset == "cut_data":
        data_root = Path(args.data_root)
        required_roots = [data_root / "train", data_root / "val", data_root / "test"]
        missing_roots = [str(path) for path in required_roots if not path.is_dir()]
        if missing_roots:
            raise FileNotFoundError(
                "训练模式要求 data_root 下同时存在 train、val、test 目录，缺少: "
                + ", ".join(missing_roots)
            )
        bundle = load_cut_splits(
            args.data_root,
            sequence_length=args.sequence_length,
            train_ratio=args.train_ratio,
            seed=args.seed,
            normalization=args.normalization,
            data_format=args.data_format,
            no_header_order=args.no_header_order,
        )
        train_data, train_labels, train_groups = bundle["train"]
        val_data, val_labels, val_groups = bundle["val"]
        test_data, test_labels, test_groups = bundle["test"]
        class_names = bundle["class_names"]
        normalization_stats = bundle["stats"]
    else:
        X_train, X_val, X_test, y_train, y_val, y_test, class_names = utils.load_data(
            args.dataset, args.train_ratio, args.seed
        )
        train_data, val_data, test_data = X_train, X_val, X_test
        train_labels = class_labels(torch.as_tensor(y_train)).numpy()
        val_labels = class_labels(torch.as_tensor(y_val)).numpy()
        test_labels = class_labels(torch.as_tensor(y_test)).numpy()
        train_groups = np.arange(len(train_labels), dtype=object)
        val_groups = np.arange(len(val_labels), dtype=object)
        test_groups = np.arange(len(test_labels), dtype=object)

        stats = fit_normalization(np.asarray(train_data, dtype=np.float32), args.normalization)
        train_data = stats.transform(np.asarray(train_data, dtype=np.float32))
        val_data = stats.transform(np.asarray(val_data, dtype=np.float32))
        test_data = stats.transform(np.asarray(test_data, dtype=np.float32))
        normalization_stats = stats

    transform = None
    if args.augment:
        transform = RadarAugment(
            methods=parse_augment_methods(args.augment_methods),
            prob=args.augment_prob,
            shift_max=args.shift_max,
            noise_std=args.noise_std,
            feature_mask_prob=args.feature_mask_prob,
            pulse_mask_prob=args.pulse_mask_prob,
        )
    return (
        SliceDataset(train_data, train_labels, train_groups, transform=transform),
        SliceDataset(val_data, val_labels, val_groups),
        SliceDataset(test_data, test_labels, test_groups),
        class_names,
        normalization_stats,
    )


def train_test(args, experiment_dir):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = utils.resolve_device(args.device, args.device_type)
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.manual_seed(args.seed)
    print(f"Using device: {device}")

    train_dataset, val_dataset, test_dataset, class_names, normalization_stats = build_datasets(args)
    class_num = len(class_names)
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Train, validation, and test splits must all contain at least one slice")
    print(
        f"Train slices: {len(train_dataset)}, val slices: {len(val_dataset)}, "
        f"test slices: {len(test_dataset)}, classes: {class_num}"
    )

    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    train_metric_dataset = SliceDataset(
        train_dataset.data.numpy(), train_dataset.labels.numpy(), train_dataset.groups
    )
    train_metric_loader = DataLoader(train_metric_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = utils.load_model(
        args.model,
        class_num,
        train_dataset.data.shape[1],
        dropout=args.dropout,
        length=args.sequence_length,
        device=device,
    )
    pretrained_summary = None
    if args.pretrained:
        if not args.pretrained_path:
            raise ValueError("--pretrained requires --pretrained_path")
        pretrained_summary = load_pretrained_weights(
            model, args.pretrained_path, expected_model=args.model
        )
    smoothing = args.label_smoothing_value if args.label_smoothing else 0.0
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(os.path.join(experiment_dir, "weights"), exist_ok=True)
    best_val_acc = -float("inf")
    last_accuracy_snapshot = None
    best_accuracy_snapshot = None
    timing_path = Path(experiment_dir) / "epoch_times.txt"
    timing_path.write_text("epoch\tseconds\n", encoding="utf-8")

    for epoch in range(1, args.epoches + 1):
        synchronize_device(device)
        epoch_start = time.perf_counter()
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, criterion, device, args, class_num)
        train_result = evaluate(model, train_metric_loader, criterion, device, class_num, fusion="none")
        train_metrics = train_result["slice"]
        val_result = evaluate(model, val_loader, criterion, device, class_num, fusion=args.fusion)
        val_metrics = val_result["selected"]
        last_accuracy_snapshot = (epoch, train_metrics, val_metrics)
        synchronize_device(device)
        epoch_seconds = time.perf_counter() - epoch_start
        with timing_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{epoch}\t{epoch_seconds:.6f}\n")

        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, "
            f"train_slice_acc={train_metrics['acc']:.4f}, "
            f"train_slice_macro_f1={train_metrics['macro_f1']:.4f}, "
            f"val_{args.fusion}_acc={val_metrics['acc']:.4f}, "
            f"val_{args.fusion}_aa={val_metrics['aa']:.4f}, "
            f"val_{args.fusion}_macro_f1={val_metrics['macro_f1']:.4f}, "
            f"epoch_time={epoch_seconds:.2f}s"
        )

        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": max(best_val_acc, val_metrics["acc"]),
            "normalization": args.normalization,
            "normalization_center": normalization_stats.center,
            "normalization_scale": normalization_stats.scale,
            "no_header_order": args.no_header_order,
            "class_names": class_names,
            "fusion": args.fusion,
            "epoch_seconds": epoch_seconds,
            "pretrained": pretrained_summary,
        }
        final_acc = val_metrics["acc"]
        torch.save(state, os.path.join(experiment_dir, "weights", f"model_final.pth"))
        if val_metrics["acc"] > best_val_acc:
            best_accuracy_snapshot = (epoch, train_metrics, val_metrics)
            best_val_acc = val_metrics["acc"]
            state["best_acc"] = best_val_acc
            torch.save(state, os.path.join(experiment_dir, "weights", f"model_best.pth"))
            plot_confusion_matrix(
                val_result["true"],
                val_result["pred"],
                class_names,
                Path(experiment_dir) / "confusion_matrix_val_best.png",
                f"Best validation confusion matrix (epoch {epoch})",
            )

    if last_accuracy_snapshot is not None:
        save_class_accuracy(Path(experiment_dir) / "class_accuracy_last.txt", last_accuracy_snapshot[0], class_names, last_accuracy_snapshot[1], last_accuracy_snapshot[2])
    if best_accuracy_snapshot is not None:
        save_class_accuracy(Path(experiment_dir) / "class_accuracy_best.txt", best_accuracy_snapshot[0], class_names, best_accuracy_snapshot[1], best_accuracy_snapshot[2])

    checkpoint = torch.load(os.path.join(experiment_dir, "weights", f"model_best.pth"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_result = evaluate(model, test_loader, criterion, device, class_num, fusion=args.fusion)
    test_metrics = test_result["selected"]
    plot_confusion_matrix(
        test_result["true"],
        test_result["pred"],
        class_names,
        Path(experiment_dir) / "confusion_matrix_test.png",
        "Test confusion matrix",
    )
    print(
        f"Test {args.fusion} metrics: acc={test_metrics['acc']:.4f}, "
        f"aa={test_metrics['aa']:.4f}, f1={test_metrics['f1']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train radar pulse-description classifiers")
    parser.add_argument("--dataset", default="cut_data", choices=["cut_data"])
    parser.add_argument(
        "--data_root",
        default="./dataset/cut_data",
        help="包含train、val、test子目录的数据根目录；每个子目录可包含data.h5或类别CSV目录",
    )
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--normalization", choices=["none", "standard", "minmax"], default="standard")
    parser.add_argument("--data_format", choices=["auto", "h5", "csv"], default="csv", help="切片读取格式：auto优先H5，h5强制H5，csv强制CSV")
    parser.add_argument(
        "--no-header-order",
        default=None,
        help="无表头CSV的物理列顺序，例如 CF_MHz,PW_us,PRI_us；有表头CSV自动按列名识别",
    )
    parser.add_argument("--train_ratio", default=0.8, type=float)
    parser.add_argument("--outdir", default="./result/", type=str)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--session_name", default="radar", type=str)
    parser.add_argument("--model", default="ModernTCN", choices=MODEL_CHOICES)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epoches", default=50, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--device", default="cuda:0", help="设备编号或完整设备名，如0、npu:0、cuda:0、cpu")
    parser.add_argument("--device_type", choices=["auto", "npu", "cuda", "cpu"], default="auto", help="设备类型；auto优先NPU，其次CUDA，最后CPU")
    parser.add_argument("--num_workers", default=0, type=int)
    add_bool_switch(parser, "pretrained", "加载自监督预训练权重")
    parser.add_argument(
        "--pretrained_path",
        default=None,
        help="pretrain.py生成的pretrain_best.pth或pretrain_last.pth路径",
    )
    parser.add_argument(
        "--fusion",
        choices=FUSION_CHOICES,
        default="probability",
        help="验证/测试聚合方式: none=切片级, vote=硬投票, score=logits均值, probability=softmax概率均值",
    )

    add_bool_switch(parser, "augment", "启用雷达脉冲描述字数据增强")
    parser.add_argument("--augment_methods", default="shift,gaussian_noise,scale", help="逗号分隔: shift,gaussian_noise,scale,feature_mask,pulse_mask,reverse")
    parser.add_argument("--augment_prob", default=0.5, type=float)
    parser.add_argument("--shift_max", default=8, type=int)
    parser.add_argument("--noise_std", default=0.02, type=float)
    parser.add_argument("--feature_mask_prob", default=0.1, type=float)
    parser.add_argument("--pulse_mask_prob", default=0.05, type=float)
    add_bool_switch(parser, "mixup", "启用Manifold MixUp（分类头前的特征MixUp）")
    parser.add_argument("--mixup_alpha", default=0.2, type=float)
    add_bool_switch(parser, "label_smoothing", "启用Label Smoothing")
    parser.add_argument("--label_smoothing_value", default=0.1, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runs = sorted(glob.glob(os.path.join(args.outdir, args.dataset, args.model, args.session_name + "_*")))
    run_id = int(runs[-1].split("_")[-1]) + 1 if runs else 0
    experiment_dir = os.path.join(args.outdir, args.dataset, args.model, f"{args.session_name}_{run_id}")
    os.makedirs(experiment_dir, exist_ok=True)
    logger = pyutils.Logger(os.path.join(experiment_dir, f"{args.session_name}_{run_id}.log"))
    try:
        pyutils.save_config(experiment_dir, args)
        train_test(args, experiment_dir)
    finally:
        logger.close()
