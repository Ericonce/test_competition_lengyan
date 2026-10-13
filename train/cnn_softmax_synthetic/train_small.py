"""合成辐射源训练入口：常用参数直接在 model.train() 中修改。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsecls import PulseCLS


def main():
    model = PulseCLS()
    # 显式参数优先于 YAML；删除对应参数即可继承 YAML。
    return model.train(
        pipeline=PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
        data=PROJECT_ROOT / "datasets/synthetic_radar_overlap_small/data.yaml",
        output_dir=PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap_small",
        num_classes=10,  # 辐射源类别数；None 继承 pipeline/data.yaml。
        epochs=50,  # resume=True 时表示本次额外训练轮数。
        batch_size=64,
        lr=0.001,
        optimizer='adamw',
        weight_decay=0.0001,
        early_stopping_patience=6,
        device='auto',  # auto：优先 CUDA，其次昇腾 NPU，最后 CPU。
        use_amp=True,
        num_workers=0,
        seed=2026,
        resume=False,
        resume_checkpoint=PROJECT_ROOT / "results/cnn_softmax_synthetic_overlap_small/last_model.pt",
        sequence_length=256,
        length_policy={'short': 'zero', 'long': 'head'},  # short：zero/repeat；long：head/center。
        dataset={
            "sample_ratio": 0.01,  # 先采样总池，再划分训练/验证。
            "train_ratio": 0.8,
            "val_ratio": 0.2,  # 全量训练用 train_ratio=1.0、val_ratio=0.0。
            "seed": 2026,
            "stratified": True,
        },
        data_cfg={"cache": True, "progress": True},
        augment={
            "preset": PROJECT_ROOT / "configs/augment/default.yaml",
            "enabled": False,  # True 开启增强；方法、阈值及顺序修改上面的 YAML。
        },
    )


if __name__ == "__main__":
    main()
