"""Repeat training seeds using pipeline sampling and train/validation splits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from pulsecls import PulseCLS
from pulsecls.cfg import get_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default="configs/pipeline/cnn_softmax_synthetic.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    args = parser.parse_args()
    cfg = get_cfg(args.pipeline)
    base = Path(cfg["output_dir"])
    rows = []
    report = PROJECT_ROOT / "results/synthetic_dataset_audit/retraining_summary.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    for index, seed in enumerate(args.seeds):
        output = base if index == 0 else base.with_name(f"{base.name}_seed{seed}")
        model = PulseCLS()
        print(f"TRAIN seed={seed} output={output}", flush=True)
        validation = model.train(pipeline=args.pipeline, seed=seed, output_dir=output, resume=False)
        rows.append({"seed": seed, "output_dir": str(output), "metrics": validation})
        metrics = [r["metrics"]["macro_f1"] for r in rows]
        summary = {"pipeline": args.pipeline, "runs": rows, "macro_f1_mean": float(np.mean(metrics)),
                   "macro_f1_std": float(np.std(metrics, ddof=1)) if len(metrics) > 1 else 0.0,
                   "evaluation_split": validation["evaluation_split"],
                   "selection": "best validation Macro-F1 when validation enabled; otherwise final epoch"}
        report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
