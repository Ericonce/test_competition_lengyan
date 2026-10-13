"""Generate independent flat test CSVs using the saved training emitter identities."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsecls.data.synthetic import SyntheticRadarGenerator


# 与训练/验证生成脚本选择相同 YAML；也可填写训练池的 generation_config.yaml。
GENERATOR_YAML = PROJECT_ROOT / "configs/generator/synthetic_radar.yaml"


def main():
    parser = argparse.ArgumentParser(description="生成平铺的三通道独立测试集与 labels.txt")
    parser.add_argument("--config", default=str(GENERATOR_YAML), help="生成配置 YAML；默认使用文件开头的 GENERATOR_YAML")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-sequences", type=int, default=None)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    print(f"Generator YAML: {config_path}", flush=True)
    result = SyntheticRadarGenerator(config_path).generate_test(args.output_dir, args.num_sequences)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
