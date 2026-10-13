"""Local one-epoch synthetic smoke checks for each current train entry."""
from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
import runpy
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from pulsecls import PulseCLS
from pulsecls.data.dataset import load_sample, find_samples
from pulsecls.engine import trainer, predictor


def subset(cfg, split, *, selection_plan=None):
    if "selection" in cfg:
        from pulsecls.data.selection import select_records
        records = (selection_plan or select_records(cfg))[0][split]
    else:
        records = find_samples(Path(cfg["root"]) / cfg[split])
    selected, counts = [], {}
    for path, label in records:
        if label not in counts and len(counts) == 4:
            continue
        if counts.get(label, 0) < 2:
            selected.append((path, label))
            counts[label] = counts.get(label, 0) + 1
    return (np.stack([load_sample(p, cfg["sequence_length"], cfg["features"], cfg.get("length_policy")) for p, _ in selected]),
            np.array([label for _, label in selected], dtype=np.int64),
            [str(p) for p, _ in selected])


def main():
    output = ROOT / "results/three_feature_smoke"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    original_train = PulseCLS.train
    for entry in sorted((ROOT / "train").rglob("train.py")):
        start = time.time()
        name = entry.relative_to(ROOT / "train").as_posix().replace("/", "__").replace(".py", "")
        target = output / name
        captured = {}

        def train_subset(self, **kwargs):
            cfg = self._resolve_pipeline(**kwargs)
            assert cfg["model_cfg"]["in_channels"] == len(cfg["data_cfg"]["features"]) == 3
            cfg["train"].update(epochs=1, batch_size=4, num_workers=0, device="auto")
            cfg["output_dir"] = str(target)
            captured["cfg"] = cfg
            return original_train(self, pipeline=cfg)

        try:
            with patch.object(PulseCLS, "train", train_subset), \
                 patch.object(trainer, "load_split_arrays", subset), \
                 patch.object(predictor, "load_split_arrays", subset):
                ns = runpy.run_path(str(entry))
                with patch.object(sys, "argv", [str(entry)]):
                    ns["main"]()
                metrics = PulseCLS().predict(pipeline=captured["cfg"], checkpoint=target / "best_model.pt",
                                            save_dir=target / "inference")
            with (target / "training_history.csv").open() as f:
                for row in csv.DictReader(f):
                    assert all(np.isfinite(float(v)) for k, v in row.items() if "loss" in k and v)
            assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float)))
            result = {"entry": entry.relative_to(ROOT).as_posix(), "status": "PASS", "seconds": round(time.time() - start, 2)}
        except Exception:
            result = {"entry": entry.relative_to(ROOT).as_posix(), "status": "FAIL", "error": traceback.format_exc()}
            print(result["error"], flush=True)
        results.append(result)
        (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(result["status"], result["entry"], flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("COMPLETE", len(results), "PASS", sum(r["status"] == "PASS" for r in results), flush=True)
    if any(r["status"] != "PASS" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
