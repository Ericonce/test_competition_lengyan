"""Audit shortcuts, overlap and split independence without using neural weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from pulsecls.cfg import get_cfg
from pulsecls.data.dataset import load_split_arrays
from pulsecls.data.selection import validation_enabled
from pulsecls.utils.metrics import compute_metrics


def pulse_means(data):
    return np.stack([x[~np.all(x == 0, axis=1)].mean(axis=0) for x in data])


def audit(pipeline, output):
    cfg = get_cfg(pipeline)["data_cfg"]
    split_names = ("train", "val") if validation_enabled(cfg) else ("train",)
    loaded = {split: load_split_arrays(cfg, split) for split in split_names}
    features = {split: pulse_means(values[0]) for split, values in loaded.items()}
    train, labels = features["train"], loaded["train"][1]
    # Fit all statistics and centers on train alone.
    classes = sorted(set(labels.tolist()))
    scale = np.maximum(train.std(axis=0), 1e-6)
    centers = np.stack([train[labels == label].mean(axis=0) for label in classes])
    result = {"pipeline": str(pipeline), "num_classes": cfg["num_classes"], "splits": {}}
    paths = {split: set(values[2]) for split, values in loaded.items()}
    result["shared_paths_between_splits"] = len(paths.get("train", set()) & paths.get("val", set()))
    hashes = {split: {hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in values[2]} for split, values in loaded.items()}
    result["split_content_fingerprints"] = {split: hashlib.sha256("\n".join(sorted(digests)).encode()).hexdigest() for split, digests in hashes.items()}
    result["generation_config_sha256"] = hashlib.sha256((Path(cfg["root"]) / "generation_config.yaml").read_bytes()).hexdigest()
    result["identical_csv_contents_between_splits"] = len(hashes.get("train", set()) & hashes.get("val", set()))
    for split in ("val",) if "val" in loaded else ():
        d = np.sum(((features[split][:, None] - centers[None]) / scale) ** 2, axis=2)
        pred = np.array(classes)[d.argmin(axis=1)]
        result["splits"][split] = {"count": len(pred), "mean_only_nearest_centroid": compute_metrics(loaded[split][1], pred, cfg["num_classes"])}
    emitters = json.loads((Path(cfg["root"]) / "emitters.json").read_text())["emitters"]
    groups = sorted({e["group_id"] for e in emitters if e.get("group_id") is not None})
    overlaps = []
    stds = np.stack([train[labels == label].std(axis=0) for label in classes])
    for group in groups:
        members = [e["label"] for e in emitters if e.get("group_id") == group and e["label"] in classes]
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                li, ri = classes.index(left), classes.index(right)
                # Report pair-wise intersections of empirical mean +/- 2 std intervals.
                intersection = np.abs(centers[li] - centers[ri]) <= 2 * (stds[li] + stds[ri])
                overlaps.append(intersection)
    result["overlap_group_count"] = len(groups)
    result["within_group_pair_count"] = len(overlaps)
    result["within_group_mean_interval_overlap_fraction_cf_pw_pri"] = np.mean(overlaps, axis=0).tolist() if overlaps else None
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default="configs/pipeline/cnn_softmax_synthetic.yaml")
    parser.add_argument("--output", default="results/synthetic_dataset_audit/after_overlap_audit.json")
    args = parser.parse_args()
    print(json.dumps(audit(args.pipeline, args.output), indent=2))


if __name__ == "__main__":
    main()
