"""Bounded real-data checks for the Ascend training and inference paths."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import torch_npu
import yaml

from pulsecls import PulseCLS
from pulsecls.nn.classifiers.sequence_models import TransformerClassifier
from pulsecls.utils.device import autocast_context


def check_fusion(output):
    cfg = yaml.safe_load((PROJECT_ROOT / "configs/model/transformer.yaml").read_text())
    cfg.update(dropout=0.0, sequence_length=256, npu_attention_backend="math")
    reference = TransformerClassifier(cfg, 100).npu().train()
    fused = deepcopy(reference)
    fused.npu_attention_backend = "fusion"
    inputs = torch.randn(4, 3, 256, device="npu")
    fp32 = deepcopy(reference)
    outputs, gradients = [], []
    for model in (reference, fused, fp32):
        with autocast_context(torch.device("npu"), model is not fp32, "float16"):
            logits = model(inputs)["logits"]
            loss = logits.float().square().mean()
        loss.backward()
        outputs.append(logits.detach().float().cpu())
        gradients.append({name: p.grad.detach().float().cpu() for name, p in model.named_parameters()})
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.025, atol=0.006)
    max_error = 0.0
    gradient_errors = {}
    for name in gradients[0]:
        assert torch.isfinite(gradients[1][name]).all(), name
        max_error = max(max_error, (gradients[0][name] - gradients[1][name]).abs().max().item())
        scale = gradients[2][name].norm().clamp_min(1e-8)
        gradient_errors[name] = {"fusion_relative_l2": ((gradients[1][name] - gradients[2][name]).norm() / scale).item(),
                                "math_fp16_relative_l2": ((gradients[0][name] - gradients[2][name]).norm() / scale).item()}
    result = {"forward_max_abs_error": (outputs[0] - outputs[1]).abs().max().item(),
              "gradient_max_abs_error": max_error, "all_gradients_finite": True,
              "gradient_errors_vs_fp32": gradient_errors}
    (output / "fusion_correctness.json").write_text(json.dumps(result, indent=2))
    print("FUSION_CHECK", result, flush=True)
    for name, error in gradient_errors.items():
        assert error["fusion_relative_l2"] < 0.05, (name, error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--baseline-tcn", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    torch.npu.set_device(0)
    check_fusion(args.output)
    if args.check_only:
        return
    results = {}
    if args.baseline_tcn:
        source = args.output / "backup/pulsecls/engine/trainer.py"
        spec = importlib.util.spec_from_file_location("trainer_before", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from pulsecls.cfg import get_cfg
        pipeline = get_cfg(PROJECT_ROOT / "configs/pipeline/tcn_synthetic.yaml", overrides={
            "output_dir": args.output / "smoke_tcn_trainer_before", "epochs": args.epochs,
            "visualize": False, "dataset.sample_ratio": 1.0,
            "batch_size": 64, "use_amp": True, "device": "npu:0", "resume": False})
        module.BaseTrainer(pipeline).fit()
    for name in ("transformer", "tcn"):
        directory = args.output / ("smoke_" + name)
        pipeline = PROJECT_ROOT / f"configs/pipeline/{name}_synthetic.yaml"
        common = {"pipeline": pipeline, "data": PROJECT_ROOT / "datasets/synthetic_radar_overlap/data.yaml",
                  "output_dir": directory, "num_classes": 100, "device": "npu:0",
                  "dataset": {"sample_ratio": 1.0, "train_ratio": 0.8, "val_ratio": 0.2}}
        model = PulseCLS()
        start = time.perf_counter()
        metrics = model.train(**common, epochs=args.epochs, batch_size=64, visualize=False,
                              use_amp=True, amp_dtype="float16", resume=False)
        train_seconds = time.perf_counter() - start
        history = pd.read_csv(directory / "training_history.csv")
        assert len(history) == args.epochs
        assert history["train_total_loss"].map(lambda x: bool(torch.isfinite(torch.tensor(x)))).all()
        assert history["val_total_loss"].map(lambda x: bool(torch.isfinite(torch.tensor(x)))).all()
        inference_start = time.perf_counter()
        predictions = PulseCLS().infer(**common, checkpoint=directory / "best_model.pt", split="val",
                                      save_dir=directory / "infer_val",
                                      inference={"batch_size": 64, "use_amp": True, "amp_dtype": "float16"})
        inference_seconds = time.perf_counter() - inference_start
        assert predictions["num_sequences"] == 1600
        assert predictions["evaluated"]
        # Also exercise the user's flat CSV folder entry point with a small subset.
        folder = args.output / "folder_input"
        folder.mkdir(exist_ok=True)
        folder_paths = sorted((PROJECT_ROOT / "datasets/synthetic_radar_test").glob("*.csv"))[:128]
        assert folder_paths, "The folder inference check requires generated test CSVs"
        for path in folder_paths:
            link = folder / path.name
            if not link.exists():
                link.symlink_to(path.resolve())
        folder_predictions = PulseCLS().infer(**common, checkpoint=directory / "best_model.pt", source=folder,
                                             labels_file="", save_dir=directory / "infer_folder",
                                             inference={"batch_size": 64, "use_amp": True, "amp_dtype": "float16"})
        assert folder_predictions["num_sequences"] == len(folder_paths)
        results[name] = {"train_wall_seconds": train_seconds,
                         "epoch_seconds": history["epoch_seconds"].tolist(),
                         "validation_inference_wall_seconds": inference_seconds,
                         "validation_sequences": predictions["num_sequences"],
                         "folder_sequences": folder_predictions["num_sequences"],
                         "final_metrics": metrics}
        (args.output / "real_data_checks.json").write_text(json.dumps(results, indent=2))
        print("REAL_DATA_CHECK", name, results[name], flush=True)


if __name__ == "__main__":
    main()
