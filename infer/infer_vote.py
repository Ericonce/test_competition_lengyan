"""投票推理入口：常用参数和各模型权重直接在 model.infer_vote() 中修改。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsecls import PulseCLS


def main():
    model = PulseCLS()
    # 显式参数优先于 YAML；只有 enabled=True 的模型参与投票。
    return model.infer_vote(
        config=PROJECT_ROOT / "configs/inference/vote.yaml",
        source=PROJECT_ROOT / "datasets/synthetic_radar_test",
        output_dir=PROJECT_ROOT / "results/infer_vote",  # predictions.csv 保存在此目录。
        labels_file="",  # 只预测；填写测试目录的 labels.txt 路径后评估。
        main_model="cnn_softmax",  # 主模型须开启；预测处于最高票并列候选时优先采用。
        num_classes=100,  # 统一各成员的输出类别数；None 继承 YAML/权重。
        device="auto",  # 优先 CUDA，其次昇腾 NPU，最后 CPU。
        batch_size=256,
        progress=True,  # 显示每个启用模型的数据读取与推理进度。
        sequence_length=256,
        length_policy={"short": "zero", "long": "head"},  # short：zero/repeat；long：head/center。
        augment={"preset": PROJECT_ROOT / "configs/augment/default.yaml"},  # 配置引用；推理不执行随机增强。
        models=[
            {
                "name": "cnn_softmax",
                "enabled": True,
                "pipeline": PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
                "data": PROJECT_ROOT / "datasets/synthetic_radar_overlap/data.yaml",
                "checkpoint": PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap/best_model.pt",
            },
            {
                "name": "modular_multigrain_iq",
                "enabled": True,  # 各成员须使用相同的辐射源类别与标签编号训练。
                "pipeline": PROJECT_ROOT / "configs/pipeline/modular_multigrain_iq_synthetic.yaml",
                "data": PROJECT_ROOT / "datasets/synthetic_radar_overlap/data.yaml",
                "checkpoint": PROJECT_ROOT / "results/modular_multigrain_iq_synthetic/best_model.pt",
            },
            {
                "name": "pulse_hypothesis_fused",
                "enabled": True,
                "pipeline": PROJECT_ROOT / "configs/pipeline/pulse_hypothesis_fused_synthetic.yaml",
                "data": PROJECT_ROOT / "datasets/synthetic_radar_overlap/data.yaml",
                "checkpoint": PROJECT_ROOT / "results/pulse_hypothesis_fused_synthetic/best_model.pt",
            },
            {
                "name": "resnet_softmax",
                "enabled": True,
                "pipeline": PROJECT_ROOT / "configs/pipeline/resnet_softmax_synthetic.yaml",
                "data": PROJECT_ROOT / "datasets/synthetic_radar_overlap/data.yaml",
                "checkpoint": PROJECT_ROOT / "results/resnet_softmax_synthetic/best_model.pt",
            },
            {
                "name": "invconv1d",
                "enabled": True,
                "pipeline": PROJECT_ROOT / "configs/pipeline/invconv1d_synthetic.yaml",
                "data": PROJECT_ROOT / "datasets/synthetic_radar_overlap/data.yaml",
                "checkpoint": PROJECT_ROOT / "results/invconv1d_synthetic/best_model.pt",
            },
        ],
    )


if __name__ == "__main__":
    main()
