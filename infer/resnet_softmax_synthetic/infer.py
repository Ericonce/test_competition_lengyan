"""ResNet1D-18 + Softmax 推理入口；输入平铺 CSV 文件夹，输出预测 CSV。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsecls import PulseCLS


def main():
    model = PulseCLS()
    return model.infer(
        pipeline=PROJECT_ROOT / "configs/pipeline/resnet_softmax_synthetic.yaml",
        source=PROJECT_ROOT / "datasets/synthetic_radar_test",
        checkpoint=PROJECT_ROOT / "results/resnet_softmax_synthetic/best_model.pt",  # 无验证训练时用 final_model.pt。
        save_dir=PROJECT_ROOT / "results/resnet_softmax_synthetic/infer_test",
        labels_file="",  # 只预测；填写测试目录的 labels.txt 路径后评估。
        num_classes=100,  # 与训练类别对应；None 继承 pipeline/权重。
        device="auto",  # 优先 CUDA，其次昇腾 NPU，最后 CPU。
        sequence_length=256,
        length_policy={"short": "zero", "long": "head"},
        inference={"batch_size": 256, "progress": True},
        augment={"preset": PROJECT_ROOT / "configs/augment/default.yaml"},  # 推理配置引用。
    )


if __name__ == "__main__":
    main()
