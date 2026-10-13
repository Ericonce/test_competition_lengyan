"""合成辐射源推理入口：常用参数直接在 model.infer() 中修改。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsecls import PulseCLS


def main():
    model = PulseCLS()
    # 显式参数优先于 YAML；输入是平铺 CSV 的文件夹。
    return model.infer(
        pipeline=PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
        source=PROJECT_ROOT / "datasets/synthetic_radar_test",
        checkpoint=PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap/best_model.pt",  # 无验证集训练时改为 final_model.pt。
        save_dir=PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap/infer_test",  # predictions.csv 直接保存在此目录。
        labels_file="",  # 空字符串只预测；填写 labels.txt 路径才评估。
        num_classes=100,  # 代码优先；None 继承 pipeline/权重类别数。
        device="auto",  # 优先 CUDA，其次昇腾 NPU，最后 CPU。
        sequence_length=256,
        length_policy={'short': 'zero', 'long': 'head'},  # short：zero/repeat；long：head/center。
        inference={"batch_size": 256, "progress": True},
        augment={"preset": PROJECT_ROOT / "configs/augment/default.yaml"},  # 配置引用；推理不执行随机增强。
    )


if __name__ == "__main__":
    main()
