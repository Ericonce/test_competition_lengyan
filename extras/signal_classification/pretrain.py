"""Self-supervised contrastive pretraining for the classifiers in models/."""

from __future__ import annotations

import argparse
import glob
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data_process.augment import RadarAugment, parse_augment_methods
from data_process.dataset import SliceDataset, load_cut_splits
from tool import pyutils
from train import MODEL_CHOICES, synchronize_device


class ProjectionHead(nn.Module):
    """Map model embeddings to the space used by the contrastive loss."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(inputs), dim=-1)


class NTXentLoss(nn.Module):
    """Dynamic-batch NT-Xent loss for two augmented views."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        batch_size = first.shape[0]
        if batch_size < 2:
            raise ValueError("Contrastive loss requires at least two samples per batch")
        representations = torch.cat((first, second), dim=0)
        logits = representations @ representations.transpose(0, 1)
        logits = logits / self.temperature
        diagonal = torch.eye(2 * batch_size, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
        targets = (torch.arange(2 * batch_size, device=logits.device) + batch_size) % (
            2 * batch_size
        )
        return F.cross_entropy(logits, targets)


class HardNegativeLoss(nn.Module):
    """Debiased hard-negative contrastive loss adapted from the reference code."""

    def __init__(
        self,
        temperature: float = 0.1,
        class_prior: float = 0.1,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= class_prior < 1:
            raise ValueError("class_prior must be in [0, 1)")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        self.temperature = float(temperature)
        self.class_prior = float(class_prior)
        self.beta = float(beta)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        batch_size = first.shape[0]
        if batch_size < 2:
            raise ValueError("Contrastive loss requires at least two samples per batch")
        representations = torch.cat((first, second), dim=0)
        similarity = representations @ representations.transpose(0, 1)
        positives = torch.cat(
            (
                torch.diagonal(similarity, offset=batch_size),
                torch.diagonal(similarity, offset=-batch_size),
            )
        )

        sample_count = 2 * batch_size
        negative_mask = ~torch.eye(
            sample_count, dtype=torch.bool, device=similarity.device
        )
        row_indices = torch.arange(sample_count, device=similarity.device)
        positive_indices = (row_indices + batch_size) % sample_count
        negative_mask[row_indices, positive_indices] = False

        negative_logits = (
            similarity[negative_mask].view(sample_count, -1) / self.temperature
        )
        positive_logits = positives / self.temperature
        positives = torch.exp(torch.clamp(positive_logits, min=-50.0, max=50.0))
        negative_count = sample_count - 2

        log_reweighted = (
            np.log(negative_count)
            + torch.logsumexp((self.beta + 1.0) * negative_logits, dim=-1)
            - torch.logsumexp(self.beta * negative_logits, dim=-1)
        )
        reweighted = torch.exp(
            torch.clamp(log_reweighted, min=-50.0, max=50.0)
        )
        negative_term = (
            -self.class_prior * negative_count * positives + reweighted
        ) / (1.0 - self.class_prior)
        lower_bound = negative_count * np.exp(max(-50.0, -1.0 / self.temperature))
        negative_term = torch.clamp(negative_term, min=lower_bound)
        return (-torch.log(positives / (positives + negative_term))).mean()


class ContrastiveViews:
    """Create two independent, label-preserving PDW augmentations."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.transform = RadarAugment(
            methods=parse_augment_methods(args.augment_methods),
            prob=args.augment_prob,
            shift_max=args.shift_max,
            noise_std=args.noise_std,
            feature_mask_prob=args.feature_mask_prob,
            pulse_mask_prob=args.pulse_mask_prob,
        )

    def _apply(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.transform(sample.clone()) for sample in batch])

    def __call__(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._apply(batch), self._apply(batch)


def extract_representation(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Return a two-dimensional encoder representation for every supported model."""
    feature_method = getattr(model, "feature", None)
    if callable(feature_method):
        representation = feature_method(inputs)
    else:
        output = model(inputs)
        if isinstance(output, dict):
            representation = output.get("stream_embedding")
            if representation is None:
                representation = output.get("stream_contrast", output["logits"])
        elif isinstance(output, (tuple, list)):
            representation = output[0]
        else:
            representation = output
    if representation.ndim > 2:
        representation = representation.flatten(start_dim=1)
    if representation.ndim != 2:
        raise ValueError(
            f"Expected a [batch, features] representation, got {representation.shape}"
        )
    return representation


def encoder_state_dict(model: nn.Module) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Keep parameters trained by the SSL loss plus all non-parameter buffers."""
    parameters = dict(model.named_parameters())
    trained_names = sorted(
        name for name, parameter in parameters.items() if parameter.grad is not None
    )
    trained_set = set(trained_names)
    state = {
        name: value
        for name, value in model.state_dict().items()
        if name not in parameters or name in trained_set
    }
    if not trained_names:
        raise RuntimeError("No model parameters received gradients during pretraining")
    return state, trained_names


def build_datasets(args: argparse.Namespace):
    bundle = load_cut_splits(
        args.data_root,
        sequence_length=args.sequence_length,
        train_ratio=1.0,
        seed=args.seed,
        normalization=args.normalization,
        data_format=args.data_format,
        no_header_order=args.no_header_order,
    )
    train_data, train_labels, train_groups = bundle["train"]
    val_data, val_labels, val_groups = bundle["val"]
    return (
        SliceDataset(train_data, train_labels, train_groups),
        SliceDataset(val_data, val_labels, val_groups),
        bundle["class_names"],
        bundle["stats"],
    )


def run_epoch(
    model: nn.Module,
    projector: nn.Module,
    loader: DataLoader,
    views: ContrastiveViews,
    loss_function: nn.Module,
    device: torch.device,
    optimizer=None,
) -> float:
    training = optimizer is not None
    model.train(training)
    projector.train(training)
    total_loss = 0.0
    total_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(
            loader,
            desc="pretrain" if training else "pretrain-val",
            leave=False,
        ):
            inputs = batch[0].float().to(device)
            if inputs.shape[0] < 2:
                continue
            first_view, second_view = views(inputs)
            first_projection = projector(extract_representation(model, first_view))
            second_projection = projector(extract_representation(model, second_view))
            loss = loss_function(first_projection, second_projection)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * inputs.shape[0]
            total_count += inputs.shape[0]

    if total_count == 0:
        raise ValueError(
            "No contrastive batch was produced; each split needs at least two samples"
        )
    return total_loss / total_count


def pretrain(args: argparse.Namespace, experiment_dir: Path) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = utils.resolve_device(args.device, args.device_type)
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.manual_seed(args.seed)
    print(f"Using device: {device}")

    train_dataset, val_dataset, class_names, normalization_stats = build_datasets(args)
    if len(train_dataset) < 2 or len(val_dataset) < 2:
        raise ValueError("Pretraining requires at least two train and validation slices")
    print(
        f"Pretrain slices: {len(train_dataset)}, val slices: {len(val_dataset)}, "
        f"classes: {len(class_names)}"
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = utils.load_model(
        args.model,
        len(class_names),
        train_dataset.data.shape[1],
        dropout=args.dropout,
        length=args.sequence_length,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        probe = train_dataset.data[: min(2, len(train_dataset))].float().to(device)
        representation_dim = extract_representation(model, probe).shape[1]
    projector = ProjectionHead(
        representation_dim,
        args.projection_hidden_dim,
        args.projection_dim,
    ).to(device)

    if args.loss == "nt_xent":
        loss_function = NTXentLoss(args.temperature)
    else:
        loss_function = HardNegativeLoss(
            args.temperature,
            args.class_prior,
            args.hard_negative_beta,
        )
    views = ContrastiveViews(args)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projector.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )

    weights_dir = experiment_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    timing_path = experiment_dir / "epoch_times.txt"
    timing_path.write_text("epoch\tseconds\n", encoding="utf-8")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        synchronize_device(device)
        epoch_start = time.perf_counter()
        train_loss = run_epoch(
            model, projector, train_loader, views, loss_function, device, optimizer
        )
        val_loss = run_epoch(
            model, projector, val_loader, views, loss_function, device
        )
        synchronize_device(device)
        epoch_seconds = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        with timing_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{epoch}\t{epoch_seconds:.6f}\n")
        print(
            f"Epoch {epoch:03d}: lr={current_lr:.6g}, "
            f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
            f"epoch_time={epoch_seconds:.2f}s"
        )

        pretrained_state, trained_parameter_names = encoder_state_dict(model)
        state = {
            "epoch": epoch,
            "state_dict": pretrained_state,
            "trained_parameter_names": trained_parameter_names,
            "projector_state_dict": projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": min(best_val_loss, val_loss),
            "model": args.model,
            "loss": args.loss,
            "normalization": args.normalization,
            "normalization_center": normalization_stats.center,
            "normalization_scale": normalization_stats.scale,
            "no_header_order": args.no_header_order,
            "class_names": class_names,
            "sequence_length": args.sequence_length,
            "epoch_seconds": epoch_seconds,
        }
        torch.save(state, weights_dir / "pretrain_last.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state["best_val_loss"] = best_val_loss
            torch.save(state, weights_dir / "pretrain_best.pth")

    print(f"Pretraining completed. Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {weights_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-supervised contrastive pretraining for radar PDW classifiers"
    )
    parser.add_argument(
        "--data_root",
        default="./dataset/cut_data",
        help="包含train、val、test子目录的数据根目录；每个子目录可包含data.h5或类别CSV目录",
    )
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument(
        "--normalization",
        choices=["none", "standard", "minmax"],
        default="standard",
    )
    parser.add_argument("--data_format", choices=["auto", "h5", "csv"], default="auto", help="切片读取格式：auto优先H5，h5强制H5，csv强制CSV")
    parser.add_argument(
        "--no-header-order",
        default=None,
        help="无表头CSV的物理列顺序，例如 CF_MHz,PW_us,PRI_us；有表头CSV自动按列名识别",
    )
    parser.add_argument("--outdir", default="./result/pretrain")
    parser.add_argument("--session_name", default="radar")
    parser.add_argument("--model", default="ResNet18", choices=MODEL_CHOICES)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", "--epoches", dest="epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="npu:0",
        help="设备编号或完整设备名，如0、npu:0、cuda:0、cpu",
    )
    parser.add_argument(
        "--device_type",
        choices=["auto", "npu", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument(
        "--loss",
        choices=["nt_xent", "hard_negative"],
        default="hard_negative",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--class_prior", type=float, default=0.1)
    parser.add_argument("--hard_negative_beta", type=float, default=1.0)
    parser.add_argument("--projection_hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=128)

    parser.add_argument(
        "--augment_methods",
        default="shift,gaussian_noise,scale,pulse_mask",
        help="逗号分隔: shift,gaussian_noise,scale,feature_mask,pulse_mask,reverse",
    )
    parser.add_argument("--augment_prob", type=float, default=0.5)
    parser.add_argument("--shift_max", type=int, default=8)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--feature_mask_prob", type=float, default=0.1)
    parser.add_argument("--pulse_mask_prob", type=float, default=0.05)
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("--batch_size must be at least 2 for contrastive pretraining")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    return args


def next_experiment_dir(args: argparse.Namespace) -> Path:
    base = Path(args.outdir) / args.model
    pattern = str(base / f"{args.session_name}_*")
    runs = sorted(glob.glob(pattern))
    run_ids = []
    for run in runs:
        suffix = Path(run).name.rsplit("_", 1)[-1]
        if suffix.isdigit():
            run_ids.append(int(suffix))
    run_id = max(run_ids, default=-1) + 1
    return base / f"{args.session_name}_{run_id}"


if __name__ == "__main__":
    arguments = parse_args()
    output_dir = next_experiment_dir(arguments)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = pyutils.Logger(output_dir / f"{output_dir.name}.log")
    try:
        pyutils.save_config(output_dir, arguments)
        pretrain(arguments, output_dir)
    finally:
        logger.close()
