"""Short reproducible training/validation benchmarks on Ascend NPU."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch_npu
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer", "tcn"], required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--backend", default="native")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.source:
        spec = importlib.util.spec_from_file_location("sequence_baseline", args.source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        from pulsecls.nn.classifiers import sequence_models as module
    cfg = yaml.safe_load((PROJECT_ROOT / f"configs/model/{args.model}.yaml").read_text())
    cfg["sequence_length"] = 256
    if args.model == "transformer":
        cfg["npu_attention_backend"] = args.backend
        factory = module.TransformerClassifier
    else:
        cfg["npu_conv_backend"] = args.backend
        factory = module.TCNClassifier
    torch.manual_seed(2026)
    torch.npu.set_device(0)
    model = factory(cfg, 100).npu()
    inputs = torch.randn(args.batch_size, 3, 256, device="npu")
    labels = torch.randint(100, (args.batch_size,), device="npu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    scaler = torch.npu.amp.GradScaler(enabled=args.dtype == "fp16")

    def step(training):
        model.train(training)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.npu.amp.autocast(enabled=args.dtype != "fp32", dtype=dtype):
                logits = model(inputs)["logits"]
                loss = torch.nn.functional.cross_entropy(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        return loss

    result = {"model": args.model, "backend": args.backend, "dtype": args.dtype,
              "batch_size": args.batch_size, "sequence_length": 256,
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "device": torch.npu.get_device_name(0)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for training in (True, False):
        name = "train" if training else "validation"
        timings = []
        for number in range(args.warmup + args.steps):
            torch.npu.synchronize()
            start = time.perf_counter()
            loss = step(training)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - start
            print(f"{name} {number + 1}: {elapsed:.6f}s", flush=True)
            if number >= args.warmup:
                timings.append(elapsed)
        result[name] = {"median_batch_seconds": statistics.median(timings),
                        "mean_batch_seconds": statistics.mean(timings),
                        "sequences_per_second": args.batch_size / statistics.mean(timings),
                        "last_loss": float(loss.cpu()), "batch_seconds": timings}
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.profile:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output.with_suffix("")) + "_profile"),
        ) as profiler:
            step(True)
            profiler.step()
            step(False)
            profiler.step()
            torch.npu.synchronize()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
