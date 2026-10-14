"""ResNet1D-18 + Softmax 训练入口；常用参数直接在 model.train() 中修改。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsecls import PulseCLS


def main():
    model = PulseCLS()
    return model.train(
        pipeline=PROJECT_ROOT / "configs/pipeline/resnet_softmax_synthetic.yaml",
        data=PROJECT_ROOT / "datasets/synthetic_radar_overlap/data.yaml",
        output_dir=PROJECT_ROOT / "results/resnet_softmax_synthetic",
        num_classes=100,  # 小数据集同步改为 10；None 继承 pipeline/data.yaml。
        epochs=40,  # resume=True 时表示本次额外训练轮数。
        batch_size=64,
        lr=0.001,
        labelsmooth=0.0,  # 0 关闭标签平滑；例如 0.1 开启。
        visualize=True,  # True 保存结果图和 t-SNE；False 关闭可视化。
        optimizer="adamw",
        weight_decay=0.0001,
        patience=6,  # 连续多少轮验证 Macro-F1 未改善时早停。
        device="auto",  # 优先 CUDA，其次昇腾 NPU，最后 CPU。
        use_amp=True,
        num_workers=0,
        seed=2026,
        resume=False,
        resume_checkpoint=PROJECT_ROOT / "results/resnet_softmax_synthetic/last_model.pt",
        sequence_length=256,
        length_policy={"short": "zero", "long": "head"},  # short：zero/repeat；long：head/center。
        dataset={
            "sample_ratio": 0.01,
            "train_ratio": 0.8,
            "val_ratio": 0.2,  # 全量训练用 sample_ratio=1、train_ratio=1、val_ratio=0。
            "seed": 2026,
            "stratified": True,
        },
        data_cfg={"cache": True, "progress": True},
        augment={
            "preset": PROJECT_ROOT / "configs/augment/default.yaml",
            "enabled": False,  # True 开启增强；方法、强度和顺序修改对应 YAML。
        },
    )


if __name__ == "__main__":
    main()
