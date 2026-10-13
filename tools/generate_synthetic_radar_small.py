"""Generate a synthetic CF/PW/PRI dataset from YAML and optional CLI overrides."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pulsecls.data.synthetic import DEFAULT_CONFIG, SyntheticRadarGenerator
from pulsecls.cfg import deep_update, load_yaml


# 在这里显式选择生成配置；使用其他 YAML 时，直接修改这一行。
GENERATOR_YAML = PROJECT_ROOT / "configs/generator/synthetic_radar_small.yaml"


def main():
    parser = argparse.ArgumentParser(description="生成三通道辐射源脉冲流数据集")
    parser.add_argument("--config", default=str(GENERATOR_YAML), help="生成配置 YAML；默认使用文件开头的 GENERATOR_YAML")
    parser.add_argument("--output-dir", default=None, help="输出目录；同一路径会替换原数据，默认读取 YAML")
    parser.add_argument("--num-emitters", type=int)
    parser.add_argument("--sequences-per-emitter", type=int)
    parser.add_argument("--num-transmitter-types", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--length", type=int, help="基准长度；配合 --fixed-length 关闭随机长度")
    parser.add_argument("--length-min", type=int)
    parser.add_argument("--length-max", type=int)
    parser.add_argument("--fixed-length", action="store_true")
    parser.add_argument("--snr-min", type=float)
    parser.add_argument("--snr-max", type=float)
    parser.add_argument("--no-noise", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    print(f"Generator YAML: {config_path}", flush=True)
    overrides = {key: getattr(args, key) for key in ("output_dir", "num_emitters", "sequences_per_emitter", "num_transmitter_types", "seed") if getattr(args, key) is not None}
    # Resolve CLI ranges against the selected YAML, so specifying one end preserves the other.
    config = deep_update(load_yaml(DEFAULT_CONFIG), load_yaml(config_path))
    seq, noise = {}, {}
    if args.length is not None:
        seq["length"] = args.length
    if args.fixed_length:
        seq["random_length"] = False
    if args.length_min is not None or args.length_max is not None:
        seq["length_range"] = [args.length_min if args.length_min is not None else config["sequence"]["length_range"][0], args.length_max if args.length_max is not None else config["sequence"]["length_range"][1]]
    if args.snr_min is not None or args.snr_max is not None:
        noise["snr_db_range"] = [args.snr_min if args.snr_min is not None else config["noise"]["snr_db_range"][0], args.snr_max if args.snr_max is not None else config["noise"]["snr_db_range"][1]]
    if args.no_noise:
        noise["enabled"] = False
    overrides.update(sequence=seq, noise=noise)
    result = SyntheticRadarGenerator(config, **overrides).generate()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
