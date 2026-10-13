"""Recover actual TOA from raw streams, preserving all existing split memberships."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    ("tsrd", "train_1950", "split_random256", 256, "脉冲到达时间"),
    ("real_radar", "raw", "split_random128", 128, "脉冲到达时间(ms)"),
]


def main():
    totals = {}
    for dataset, source, old_split, length, toa in DATASETS:
        base = ROOT / "datasets" / dataset
        destination = base / (old_split + "_3ch")
        destination.mkdir(exist_ok=True)
        files = sorted((base / old_split).rglob("sample_*.csv"))
        by_class = {}
        for path in files:
            by_class.setdefault(path.parent.name, []).append(path)
        count = 0
        for label, samples in by_class.items():
            raw = pd.read_csv(base / source / f"{label}.csv", encoding="utf-8-sig")
            columns = ["载频(MHz)", "脉宽(us)", toa]
            for path in samples:
                start = (int(path.stem.split("_")[-1]) - 1) * length
                segment = raw.iloc[start:start + length][columns]
                previous = pd.read_csv(path, encoding="utf-8-sig")
                if len(segment) != length or not np.allclose(
                    previous.iloc[:, :2].to_numpy(dtype=np.float64),
                    segment.iloc[:, :2].to_numpy(dtype=np.float64), rtol=0, atol=1e-6,
                ):
                    raise ValueError(f"Raw source does not match existing sample: {path}")
                if not np.isfinite(segment.to_numpy(dtype=np.float64)).all():
                    raise ValueError(f"Non-finite raw feature: {path}")
                target = destination / path.relative_to(base / old_split)
                target.parent.mkdir(parents=True, exist_ok=True)
                segment.to_csv(target, index=False, encoding="utf-8-sig")
                count += 1
            print(dataset, "class", label, "verified", len(samples), flush=True)
        totals[dataset] = {"samples": count, "sequence_length": length,
                           "features": ["CF", "PW", "TOA"],
                           "source_toa_column": toa, "split_membership_preserved": True}
        (destination / "three_feature_manifest.json").write_text(
            json.dumps(totals[dataset], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
