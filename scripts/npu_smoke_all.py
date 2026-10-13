"""Exercise current train entries on small synthetic subsets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(entry: Path, output: Path):
    import numpy as np
    from unittest.mock import patch
    from pulsecls import PulseCLS
    from pulsecls.data.dataset import load_split_arrays
    from pulsecls.engine import trainer, predictor
    from pulsecls.utils.device import get_device

    assert get_device("auto").type == "npu", "Smoke test requires an Ascend NPU"
    original_train = PulseCLS.train
    captured = {}

    from scripts.smoke_three_feature_inputs import subset

    def smoke_train(self, **kwargs):
        cfg = self._resolve_pipeline(**kwargs)
        cfg["train"].update(epochs=1, batch_size=4, num_workers=0,
                            device=os.environ.get("PULSECLS_SMOKE_DEVICE", "auto"))
        cfg["output_dir"] = str(output)
        captured["pipeline"] = cfg
        return original_train(self, pipeline=cfg)

    with patch.object(PulseCLS, "train", smoke_train), \
         patch.object(trainer, "load_split_arrays", subset), \
         patch.object(predictor, "load_split_arrays", subset):
        namespace = runpy.run_path(str(entry))
        if entry == ROOT / "train/train.py":
            with patch.object(sys, "argv", [str(entry)]):
                namespace["main"]()
        else:
            namespace["main"]()
        import csv
        with (output / "training_history.csv").open() as history:
            for row in csv.DictReader(history):
                for key, value in row.items():
                    if "loss" in key and value:
                        assert np.isfinite(float(value)), (key, value)
        import torch
        saved = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
        assert all(torch.isfinite(value).all() for value in saved["model_state_dict"].values()
                   if value.is_floating_point()), "Non-finite checkpoint parameters"
        metrics = PulseCLS().predict(pipeline=captured["pipeline"],
                                    checkpoint=output / "best_model.pt",
                                    save_dir=output / "inference")
        assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))), metrics
        print("SMOKE_PASS", entry.relative_to(ROOT), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--only", default="")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    entries = sorted((ROOT / "train").rglob("train.py"), key=lambda p: ("ablation" in p.parts, str(p)))
    if args.only:
        entries = [p for p in entries if args.only in p.as_posix()]
    summary_path = args.output / "summary.json"
    results = json.loads(summary_path.read_text()) if args.resume and summary_path.exists() else []
    done = {r["entry"] for r in results if r["status"] == "PASS"}
    entries = [p for p in entries if p.relative_to(ROOT).as_posix() not in done]
    results = [r for r in results if r["status"] == "PASS"]

    def run_entry(number, entry):
        label = entry.relative_to(ROOT / "train").as_posix().replace("/", "__").replace(".py", "")
        destination = args.output / label
        destination.mkdir(parents=True, exist_ok=True)
        start = time.time()
        print("START", entry.relative_to(ROOT), flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(entry), "--output", str(destination)]
        try:
            with (destination / "run.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=args.timeout, env={**os.environ, "PYTHONUNBUFFERED": "1",
                                            "PULSECLS_SMOKE_DEVICE": f"npu:{number % 2}" if args.jobs > 1 else "auto"})
            status = "PASS" if result.returncode == 0 else "FAIL"
            code = result.returncode
        except subprocess.TimeoutExpired:
            status, code = "TIMEOUT", None
        row = {"entry": entry.relative_to(ROOT).as_posix(), "status": status,
               "returncode": code, "seconds": round(time.time() - start, 2), "log": str(destination / "run.log")}
        print(status, row["entry"], row["seconds"], flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_entry, number, entry) for number, entry in enumerate(entries)]
        for future in as_completed(futures):
            results.append(future.result())
            summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE", len(results), "passed", sum(r["status"] == "PASS" for r in results), flush=True)


if __name__ == "__main__":
    main()
