"""Small, repeatable CPU train/resume/infer run for Windows and VS Code."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from pulsecls import PulseCLS
from pulsecls.data.synthetic import SyntheticRadarGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/windows_cpu_debug")
    args = parser.parse_args()
    # Each run uses a new folder, preserving earlier datasets and checkpoints.
    output = args.output.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    pool, test_pool, run = output / "train_pool", output / "test_pool", output / "training"
    generator = SyntheticRadarGenerator(
        output_dir=str(pool), test_output_dir=str(test_pool),
        num_emitters=3, sequences_per_emitter=8, test_num_sequences=9,
        noise={"enabled": False}, observation={"enabled": False}, overlap={"enabled": False},
        sequence={"random_length": False, "length": 32},
    )
    generator.generate()
    generator.generate_test()
    settings = dict(
        pipeline=PROJECT_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml",
        data=pool / "data.yaml", output_dir=run, num_classes=3,
        device="cpu", use_amp=False, num_workers=0, seed=2026,
        epochs=2, batch_size=4, sequence_length=32,
        dataset={"sample_ratio": 1.0, "train_ratio": 0.75, "val_ratio": 0.25,
                 "seed": 2026, "stratified": True},
        model_cfg={"hidden_channels": [8, 16], "stream_embedding_dim": 8,
                   "pulse_embedding_dim": 8, "contrast_dim": 4},
        data_cfg={"cache": False, "progress": False},
        augment={"enabled": False},
        reporting={"dpi": 75, "formats": ["png"],
                   "tsne": {"max_samples": 6, "max_iter": 300, "perplexity": 3}},
    )
    model = PulseCLS()
    # Set a breakpoint here, or inside pulsecls/engine/trainer.py.
    trained = model.train(**settings)
    initial_weight = torch.load(run / "last_model.pt", map_location="cpu", weights_only=False)
    assert initial_weight["epoch"] == 2
    resumed_settings = {**settings, "epochs": 1, "resume": True,
                        "resume_checkpoint": run / "last_model.pt"}
    resumed = model.train(**resumed_settings)
    weight = torch.load(run / "last_model.pt", map_location="cpu", weights_only=False)
    assert weight["epoch"] == 3
    assert all(torch.isfinite(v).all() for v in weight["model_state_dict"].values()
               if v.is_floating_point())
    predicted = model.infer(
        pipeline=model.trainer.pipeline, checkpoint=run / "last_model.pt", device="cpu",
        source=test_pool, labels_file=test_pool / "labels.txt", save_dir=output / "inference",
    )
    assert predicted["num_sequences"] == 9 and predicted["evaluated"]
    assert (run / "figures/training_curves.png").is_file()
    summary = {"python": sys.version, "torch": torch.__version__, "device": "cpu",
               "output": str(output), "train": trained, "resume": resumed, "infer": predicted}
    (output / "debug_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nCPU debug run passed. Results: {output}")


if __name__ == "__main__":
    main()
