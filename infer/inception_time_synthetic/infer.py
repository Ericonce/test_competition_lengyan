"""InceptionTime 推理入口；输入平铺 CSV 文件夹，输出预测 CSV。"""
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
        pipeline=PROJECT_ROOT / "configs/pipeline/inception_time_synthetic.yaml",
        source=PROJECT_ROOT / "datasets/synthetic_radar_test",
        checkpoint=PROJECT_ROOT / "results/inception_time_synthetic/best_model.pt",
        save_dir=PROJECT_ROOT / "results/inception_time_synthetic/infer_test",
        labels_file="",
        num_classes=100,
        device="auto",
        sequence_length=256,
        length_policy={
            "short": "zero",
            "long": "head",
        },
        inference={
            "batch_size": 256,
            "progress": True,
        },
        augment={
            "preset": PROJECT_ROOT / "configs/augment/default.yaml",
        },
    )


if __name__ == "__main__":
    main()
