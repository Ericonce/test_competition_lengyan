"""投票推理入口：常用参数和各模型权重直接在 model.infer_vote() 中修改。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsecls import PulseCLS


def main():
    parser = argparse.ArgumentParser(description="多模型投票推理，标签 TXT 可选")
    parser.add_argument("--config", help="指定其他配置时，未提供的参数继承该 YAML")
    parser.add_argument("--source")
    parser.add_argument("--labels-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--augment-yaml")
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--no-progress", action="store_true", default=None)
    args = parser.parse_args()
    model = PulseCLS()
    if args.config is not None:
        overrides = {key: getattr(args, key) for key in (
            "source", "labels_file", "output_dir", "device", "num_classes",
            "batch_size", "sequence_length",
        ) if getattr(args, key) is not None}
        if args.augment_yaml is not None:
            overrides["augment"] = {"preset": args.augment_yaml}
        if args.no_progress:
            overrides["progress"] = False
        return model.infer_vote(config=args.config, **overrides)

    return model.infer_vote(
        config=PROJECT_ROOT / "configs/inference/vote.yaml",
        source=args.source or PROJECT_ROOT / "datasets/synthetic_radar_test",
        output_dir=args.output_dir or PROJECT_ROOT / "results/infer_vote",
        labels_file=args.labels_file if args.labels_file is not None else "",  # 填写 labels.txt 才评估。
        main_model="cnn_softmax",  # 最高票并列时优先采用此模型。
        num_classes=args.num_classes if args.num_classes is not None else 100,
        device=args.device or "auto",
        batch_size=args.batch_size if args.batch_size is not None else 256,
        progress=not args.no_progress,
        sequence_length=args.sequence_length if args.sequence_length is not None else 256,
        length_policy={"short": "zero", "long": "head"},
        augment={"preset": args.augment_yaml or PROJECT_ROOT / "configs/augment/default.yaml"},
        models=[
            {
                "name": "cnn_softmax",
                "enabled": True,
                "pipeline": PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
                "checkpoint": PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap/best_model.pt",
            },
            {
                "name": "modular_multigrain_iq",
                "enabled": False,  # 先训练同一套辐射源，再改为 True。
                "pipeline": PROJECT_ROOT / "configs/pipeline/modular_multigrain_iq_synthetic.yaml",
                "checkpoint": PROJECT_ROOT / "results/modular_multigrain_iq_synthetic/best_model.pt",
            },
            {
                "name": "pulse_hypothesis_fused",
                "enabled": False,
                "pipeline": PROJECT_ROOT / "configs/pipeline/pulse_hypothesis_fused_synthetic.yaml",
                "checkpoint": PROJECT_ROOT / "results/pulse_hypothesis_fused_synthetic/best_model.pt",
            },
            {
                "name": "resnet_softmax",
                "enabled": False,  # 训练同一套辐射源后可参与投票。
                "pipeline": PROJECT_ROOT / "configs/pipeline/resnet_softmax_synthetic.yaml",
                "checkpoint": PROJECT_ROOT / "results/resnet_softmax_synthetic/best_model.pt",
            },
        ],
    )


if __name__ == "__main__":
    main()
