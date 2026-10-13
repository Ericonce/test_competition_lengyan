"""通用 CSV 文件夹推理入口；常用默认参数直接在 model.infer() 中修改。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsecls import PulseCLS
from pulsecls.cfg import get_cfg


def build_parser():
    parser = argparse.ArgumentParser(description="直接读取测试文件夹内的 CSV，每个 CSV 为一个序列")
    parser.add_argument("--pipeline", help="切换 pipeline 时，未提供的参数继承该 YAML")
    parser.add_argument("--source", help="测试 CSV 文件夹")
    parser.add_argument("--checkpoint", help="推理权重路径")
    parser.add_argument("--labels-file", help="可选的 labels.txt，指定后才评估")
    parser.add_argument("--output-dir", help="训练结果目录，用于自动选择权重")
    parser.add_argument("--save-dir", help="预测 CSV 和评估输出目录")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--augment-yaml", help="增强 YAML 引用；推理不执行随机增强")
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--short-policy", choices=("zero", "repeat"))
    parser.add_argument("--long-policy", choices=("head", "center"))
    parser.add_argument("--no-progress", action="store_true", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    model = PulseCLS()
    # CLI 切换模型时，继承指定 pipeline，避免带入 CNN 的类别数和权重。
    if args.pipeline is not None:
        overrides = {key: getattr(args, key) for key in (
            "source", "checkpoint", "labels_file", "output_dir", "save_dir",
            "device", "num_classes", "sequence_length",
        ) if getattr(args, key) is not None}
        if args.batch_size is not None:
            overrides["inference"] = {"batch_size": args.batch_size}
        if args.no_progress:
            overrides.setdefault("inference", {})["progress"] = False
        if args.augment_yaml is not None:
            overrides["augment"] = {"preset": args.augment_yaml}
        policy = {key: value for key, value in (
            ("short", args.short_policy), ("long", args.long_policy),
        ) if value is not None}
        if policy:
            overrides["length_policy"] = {
                **get_cfg(args.pipeline)["data_cfg"].get("length_policy", {}), **policy,
            }
        return model.infer(
            pipeline=args.pipeline,
            source=overrides.pop("source", PROJECT_ROOT / "datasets/synthetic_radar_test"),
            checkpoint=overrides.pop("checkpoint", None),  # 未指定时读取该 pipeline 的权重路径。
            **overrides,
        )

    # 常用参数直接修改这里；CLI 显式参数优先于这里的默认值。
    return model.infer(
        pipeline=PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
        source=args.source or PROJECT_ROOT / "datasets/synthetic_radar_test",
        checkpoint=args.checkpoint or (
            None if args.output_dir else
            PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap/best_model.pt"
        ),  # 无验证集训练时改为 final_model.pt。
        output_dir=args.output_dir,
        save_dir=args.save_dir or (
            None if args.output_dir else PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap/infer_test"
        ),
        labels_file=args.labels_file if args.labels_file is not None else "",  # 空字符串只预测。
        num_classes=args.num_classes if args.num_classes is not None else 100,
        device=args.device or "auto",
        sequence_length=args.sequence_length if args.sequence_length is not None else 256,
        length_policy={"short": args.short_policy or "zero", "long": args.long_policy or "head"},
        inference={
            "batch_size": args.batch_size if args.batch_size is not None else 256,
            "progress": not args.no_progress,
        },
        augment={"preset": args.augment_yaml or PROJECT_ROOT / "configs/augment/default.yaml"},
    )


if __name__ == "__main__":
    main()
