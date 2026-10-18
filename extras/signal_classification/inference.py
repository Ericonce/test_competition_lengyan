"""Evaluate a trained classifier with slice-level and source-level voting."""

from __future__ import annotations

import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader

import utils
from train import FUSION_CHOICES, MODEL_CHOICES, evaluate
from data_process.dataset import NormalizationStats, SliceDataset, load_cut_test


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate radar pulse-description classifiers")
    parser.add_argument("--dataset", default="cut_data", choices=["cut_data", "rml2016a", "cemee"])
    parser.add_argument(
        "--data_root",
        "--test_root",
        dest="data_root",
        default="./dataset/cut_data/test",
        help="包含data.h5的测试目录、直接H5路径或旧版类别CSV目录",
    )
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--data_format", choices=["auto", "h5", "csv"], default="auto", help="切片读取格式：auto优先H5，h5强制H5，csv强制CSV")
    parser.add_argument(
        "--no-header-order",
        default=None,
        help="无表头CSV的物理列顺序，例如 CF_MHz,PW_us,PRI_us；有表头CSV自动按列名识别",
    )
    parser.add_argument("--normalization", choices=["none", "standard", "minmax"], default="standard")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--outdir", default="./result/")
    parser.add_argument("--session_name", default="radar_1")
    parser.add_argument("--model", default="ResNet18", choices=MODEL_CHOICES)
    parser.add_argument("--weight_name", default="model_best")
    parser.add_argument("--output_csv", default="predictions.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="0", help="设备编号或完整设备名，如0、npu:0、cuda:0、cpu")
    parser.add_argument("--device_type", choices=["auto", "npu", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--fusion",
        choices=FUSION_CHOICES,
        default="probability",
        help="输出聚合方式: none=每个切片, vote=硬投票, score=logits均值, probability=softmax概率均值",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = utils.resolve_device(args.device, args.device_type)
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.manual_seed(args.seed)
    print(f"Using device: {device}")

    experiment_dir = os.path.join(args.outdir, args.dataset, args.model, args.session_name)
    checkpoint_path = os.path.join(experiment_dir, "weights", f"{args.weight_name}.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    class_names = list(checkpoint.get("class_names", []))
    checkpoint_stats = None
    if "normalization_center" in checkpoint and "normalization_scale" in checkpoint:
        checkpoint_stats = NormalizationStats(
            checkpoint.get("normalization", args.normalization),
            checkpoint["normalization_center"],
            checkpoint["normalization_scale"],
        )
    no_header_order = args.no_header_order or checkpoint.get("no_header_order")
    bundle = load_cut_test(
        args.data_root,
        sequence_length=args.sequence_length,
        class_names=class_names or None,
        normalization_stats=checkpoint_stats,
        normalization=args.normalization,
        data_format=args.data_format,
        no_header_order=no_header_order,
    )
    test_data, test_labels, test_groups = bundle["test"]
    test_dataset = SliceDataset(test_data, test_labels, test_groups)
    class_names = bundle["class_names"]
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = utils.load_model(
        args.model,
        len(class_names),
        test_dataset.data.shape[1],
        dropout=args.dropout,
        length=args.sequence_length,
        device=device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    criterion = torch.nn.CrossEntropyLoss()
    result = evaluate(model, loader, criterion, device, len(class_names), fusion=args.fusion)
    slice_metrics = result["slice"]
    selected_metrics = result["selected"]
    print(
        f"Slice metrics: acc={slice_metrics['acc']:.4f}, "
        f"aa={slice_metrics['aa']:.4f}, f1={slice_metrics['f1']:.4f}, "
        f"macro_f1={slice_metrics['macro_f1']:.4f}"
    )
    print(
        f"{args.fusion.capitalize()} metrics: acc={selected_metrics['acc']:.4f}, "
        f"aa={selected_metrics['aa']:.4f}, f1={selected_metrics['f1']:.4f}, "
        f"macro_f1={selected_metrics['macro_f1']:.4f}"
    )

    output_path = args.output_csv
    if not os.path.isabs(output_path):
        output_path = os.path.join(experiment_dir, output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "predicted_label"])
        for index, prediction in enumerate(result["pred"]):
            writer.writerow([index, class_names[int(prediction)]])
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
