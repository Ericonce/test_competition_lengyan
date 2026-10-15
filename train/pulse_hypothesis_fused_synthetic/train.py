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
        pipeline=PROJECT_ROOT / "configs/pipeline/pulse_hypothesis_fused_synthetic.yaml",
        data=PROJECT_ROOT / "datasets/synthetic_radar_overlap/data.yaml",
        output_dir=PROJECT_ROOT / "results/pulse_hypothesis_fused_synthetic",
        num_classes=100,  # 辐射源类别数；None 继承 pipeline/data.yaml。
        epochs=40,  # resume=True 时表示本次额外训练轮数。
        batch_size=64,
        lr=0.001,
        labelsmooth=0.0,  # 0 关闭标签平滑；例如 0.1 开启。
        visualize=True,  # True 保存结果图和 t-SNE；False 关闭可视化。
        optimizer='adamw',
        weight_decay=0.0001,
        early_stopping_patience=6,
        device='auto',  # auto：优先 CUDA，其次昇腾 NPU，最后 CPU。
        use_amp=True,
        num_workers=0,
        seed=2026,
        resume=False,
        resume_checkpoint=PROJECT_ROOT / "results/pulse_hypothesis_fused_synthetic/last_model.pt",
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
            "enabled": False,  # 总开关：False 时下面及 YAML 中的增强都不执行。
            # 未填写的项继承 YAML，代码中填写的项优先。
            "methods": {
                "jittering": {
                    "enabled": True,
                    "prob": 1.0,
                    "snr_db": 40.0,
                },
            },
            # 多方法依次处理前一步结果；同类融合须放在 MixUp 之前。
            # "order": [
            #     "jittering",
            #     "scaling",
            #     "time_shift",
            #     "pulse_mask",
            #     "interpolation",
            #     "mixup",
            # ],
        },
    )


if __name__ == "__main__":
    main()
