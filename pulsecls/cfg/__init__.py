"""Configuration handling and CLI entrypoint."""

from __future__ import annotations

import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path
from numbers import Integral, Real
from types import SimpleNamespace
from typing import Any

import yaml

from pulsecls.utils.normalization import normalization_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG_PATH = Path(__file__).resolve().parent / "default.yaml"

TASKS = ("classify",)
MODES = ("train", "val", "predict")
TASK_ALIASES = {
    "classification": "classify",
    "cls": "classify",
}


def normalize_task(task: str | None) -> str:
    if task is None:
        return DEFAULT_CFG_DICT.get("task", "classify")
    key = str(task).strip().lower()
    return TASK_ALIASES.get(key, key)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


DEFAULT_CFG_DICT = load_yaml(DEFAULT_CFG_PATH)
DEFAULT_CFG = SimpleNamespace(**DEFAULT_CFG_DICT)


# Shared trainer defaults; experiment YAMLs and explicit train() overrides win.
TRAIN_DEFAULTS = {key: deepcopy(DEFAULT_CFG_DICT[key]) for key in (
    "device", "seed", "epochs", "resume", "resume_checkpoint", "batch_size",
    "num_workers", "sampler", "optimizer", "lr", "weight_decay",
    "early_stopping_patience", "use_amp", "amp_dtype", "normalization", "progress_metric_interval", "two_views", "metrics_prior_alpha", "num_classes", "reporting", "visualize",
)}
TRAIN_DEFAULTS["optimizer_foreach"] = False


def _normalize_train_patience(options: dict[str, Any]) -> dict[str, Any]:
    """Accept patience as the short name; explicit non-null aliases win locally."""
    result = dict(options)
    value = result.pop("patience", None)
    if value is not None:
        result["early_stopping_patience"] = value
    return result


def validate_num_classes(value: Any, name: str = "num_classes") -> int | None:
    """null means automatic; explicit class counts must be positive integers."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer or null (automatic), got {value!r}")
    return int(value)


def validate_label_smoothing(value: Any) -> float:
    """Zero disables smoothing; accept finite numeric values in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"labelsmooth / loss.ce.label_smoothing must be a finite number in [0, 1], got {value!r}")
    return float(value)


def resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = []
    if base_dir is not None:
        candidates.append(base_dir / path)
    candidates.append(REPO_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def normalize_dataset_path(path: str | Path) -> str:
    """Translate historical Chinese dataset directory components to ASCII."""
    aliases = {
        "TSRD数据集未划分": "tsrd",
        "真实雷达脉冲流数据集": "real_radar",
        "划分后数据_随机256": "split_random256",
        "划分后数据": "split_random128",
        "全部脉冲流数据": "raw",
        "训练集": "train", "验证集": "val", "测试集": "test",
    }
    return "/".join(aliases.get(part, part) for part in str(path).replace("\\", "/").split("/"))


def remap_repo_path(path: str | Path, base_dir: Path | None = None) -> Path:
    """Map Windows repo paths such as F:/competition/... onto this checkout."""
    normalized = normalize_dataset_path(path)
    prefix = "f:/competition"
    if normalized.lower().startswith(prefix):
        rest = normalized[len(prefix):].lstrip("/")
        return (REPO_ROOT / rest).resolve()
    return resolve_path(normalized, base_dir)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def portable_data_cfg(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Store project-local dataset roots relative to the checkout in checkpoints."""
    result = deepcopy(data_cfg)
    if result.get("root"):
        root = Path(result["root"])
        if not (root.is_absolute() and root.resolve().is_relative_to(REPO_ROOT.resolve())):
            root = remap_repo_path(root)
        else:
            root = root.resolve()
        try:
            result["root"] = root.relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            result["root"] = str(root)
    return result


def _apply_pipeline_input_settings(pipeline: dict[str, Any]) -> None:
    """Pipeline owns input length; keep the resolved values in data_cfg for loaders."""
    for key in ("sequence_length", "length_policy"):
        if key in pipeline:
            pipeline["data_cfg"][key] = deepcopy(pipeline[key])
    if "dataset" in pipeline:
        from pulsecls.data.selection import selection_config
        pipeline["data_cfg"]["selection"] = selection_config(pipeline["dataset"])


def load_pipeline(
    pipeline_path: str | Path,
    data_override: str | Path | None = None,
) -> dict[str, Any]:
    pipeline_path = Path(pipeline_path).resolve()
    pipeline = load_yaml(pipeline_path)
    if isinstance(pipeline.get("train"), dict):
        pipeline["train"] = _normalize_train_patience(pipeline["train"])
    base_dir = pipeline_path.parent

    data_ref = data_override if data_override is not None else pipeline["data"]
    model_ref = pipeline["model"]
    data_path = resolve_path(data_ref, base_dir)
    model_path = resolve_path(model_ref, base_dir)

    pipeline["pipeline_path"] = str(pipeline_path)
    pipeline["data_path"] = str(data_path)
    pipeline["model_path"] = str(model_path)
    pipeline["data_cfg"] = load_yaml(data_path)
    pipeline["model_cfg"] = load_yaml(model_path)
    _apply_pipeline_input_settings(pipeline)
    model_overrides = pipeline.get("model_overrides")
    if model_overrides is not None:
        if not isinstance(model_overrides, dict):
            raise TypeError("pipeline.model_overrides must be a mapping")
        pipeline["model_cfg"] = deep_update(
            pipeline["model_cfg"],
            model_overrides,
        )
    for section_name, override_key in (
        ("train", "train_overrides"),
        ("augment", "augment_overrides"),
        ("loss", "loss_overrides"),
        ("inference", "inference_overrides"),
    ):
        overrides = pipeline.get(override_key)
        if overrides is None:
            continue
        if not isinstance(overrides, dict):
            raise TypeError(f"pipeline.{override_key} must be a mapping")
        if section_name == "train":
            overrides = _normalize_train_patience(overrides)
        pipeline[section_name] = deep_update(
            pipeline.get(section_name, {}),
            overrides,
        )
    augment = dict(pipeline.get("augment", {}))
    preset = augment.pop("preset", None)
    if preset is not None:
        pipeline["augment"] = deep_update(load_yaml(resolve_path(preset, base_dir)), augment)
    pipeline["output_dir"] = str(remap_repo_path(pipeline["output_dir"], base_dir))
    data_root = pipeline["data_cfg"].get("root")
    if data_root:
        root_base = data_path.parent if pipeline["data_cfg"].get("root_relative_to_yaml", False) else None
        pipeline["data_cfg"]["root"] = str(remap_repo_path(data_root, root_base))
    pipeline["task"] = normalize_task(pipeline.get("task", DEFAULT_CFG_DICT.get("task", "classify")))
    pipeline.setdefault("mode", DEFAULT_CFG_DICT.get("mode", "train"))
    return pipeline


def get_cfg(
    cfg: str | Path | dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load an experiment pipeline and apply Ultralytics-style overrides."""
    if cfg is None:
        cfg = DEFAULT_CFG_DICT["pipeline"]
    if isinstance(cfg, dict):
        if "data_cfg" in cfg and "model_cfg" in cfg:
            pipeline = deepcopy(cfg)
        else:
            raise TypeError("cfg dict must be a resolved pipeline")
    else:
        data_override = (overrides or {}).get("data")
        pipeline = load_pipeline(cfg, data_override=data_override if isinstance(data_override, (str, Path)) else None)

    if isinstance(pipeline.get("train"), dict):
        pipeline["train"] = _normalize_train_patience(pipeline["train"])
    overrides = _normalize_train_patience({} if overrides is None else overrides)
    if isinstance(overrides.get("train"), dict):
        overrides["train"] = _normalize_train_patience(overrides["train"])
    dotted_patience = overrides.pop("train.patience", None)
    if dotted_patience is not None:
        overrides["train.early_stopping_patience"] = dotted_patience
    # Explicit code-level data class counts also outrank YAML train.num_classes.
    explicit_train_count = (overrides.get("num_classes") is not None or "train.num_classes" in overrides) or (
        isinstance(overrides.get("train"), dict) and "num_classes" in overrides["train"])
    data_count_marker = object()
    explicit_data_count = data_count_marker
    for key in ("data", "data_cfg"):
        value = overrides.get(key)
        if isinstance(value, dict) and "num_classes" in value:
            explicit_data_count = value["num_classes"]
    for key, value in overrides.items():
        if key in ("data.num_classes", "data_cfg.num_classes"):
            explicit_data_count = value
    if not explicit_train_count and explicit_data_count is not data_count_marker:
        train_override = overrides.get("train", {})
        if isinstance(train_override, dict):
            overrides["train"] = {**train_override, "num_classes": explicit_data_count}
    base_dir = Path(pipeline.get("pipeline_path", REPO_ROOT / "pipeline.yaml")).parent
    # Resolve replacement references without discarding other resolved sections.
    for alias, section in (("data", "data_cfg"), ("model", "model_cfg")):
        value = overrides.pop(alias, None)
        if value is None:
            continue
        if isinstance(value, dict):
            pipeline[section] = deep_update(pipeline.get(section, {}), value)
        else:
            path = resolve_path(value, base_dir)
            pipeline[alias] = str(path)
            pipeline[f"{alias}_path"] = str(path)
            pipeline[section] = deep_update(load_yaml(path), pipeline.get(f"{alias}_overrides", {}))
            if alias == "data":
                _apply_pipeline_input_settings(pipeline)
    dataset_override = overrides.pop("dataset", None)
    if dataset_override is not None:
        from pulsecls.data.selection import selection_config
        pipeline["data_cfg"]["selection"] = selection_config(deep_update(pipeline["data_cfg"].get("selection", {}), dataset_override))
    for section in ("train", "data_cfg", "model_cfg", "augment", "loss", "inference"):
        value = overrides.pop(section, None)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise TypeError(f"{section} override must be a mapping")
        if section == "augment" and "preset" in value:
            value = dict(value)
            preset = value.pop("preset")
            pipeline[section] = load_yaml(resolve_path(preset, base_dir))
        pipeline[section] = deep_update(pipeline.get(section, {}), value)
    normalization = pipeline.setdefault("train", {}).get("normalization")
    if normalization is None or isinstance(normalization, str):
        pipeline["train"]["normalization"] = {} if normalization is None else {"method": normalization}
    # Dotted keys allow overriding any leaf, including custom model parameters.
    aliases = {"data": "data_cfg", "model": "model_cfg"}
    for key in list(overrides):
        if "." not in key:
            continue
        parts = key.split(".")
        if parts[0] == "dataset":
            parts = ["data_cfg", "selection", *parts[1:]]
        parts[0] = aliases.get(parts[0], parts[0])
        if parts[0] not in {"train", "data_cfg", "model_cfg", "augment", "loss", "inference"} or any(not part for part in parts):
            raise ValueError(f"Invalid configuration override: {key}")
        node = pipeline
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise TypeError(f"Cannot override non-mapping parent: {key}")
        node[parts[-1]] = deepcopy(overrides.pop(key))
    train_keys = set(TRAIN_DEFAULTS) | set(pipeline.get("train", {})) | {"sampler_power", "optimizer_foreach"}
    for key, value in overrides.items():
        if value is None:
            continue  # Optional CLI values must not override YAML defaults.
        if key in {"sequence_length", "length_policy"}:
            pipeline["data_cfg"][key] = deepcopy(value)
        elif key == "data_input":
            from pulsecls.data.sources import input_config
            pipeline["data_cfg"]["input"] = input_config(value)
        elif key == "labelsmooth":
            pipeline.setdefault("loss", {}).setdefault("ce", {})["label_smoothing"] = validate_label_smoothing(value)
        elif key in train_keys:
            pipeline.setdefault("train", {})[key] = deepcopy(value)
            if key == "device":
                pipeline.setdefault("inference", {})[key] = deepcopy(value)
        elif key in {"split", "checkpoint", "save_dir", "source", "labels_file"}:
            pipeline.setdefault("inference", {})[key] = deepcopy(value)
        elif key in {"output_dir", "task", "mode", "name"}:
            pipeline[key] = deepcopy(value)
        else:
            raise ValueError(f"Unknown configuration override '{key}'; use train={{...}}, model={{...}}, data_cfg={{...}}, augment={{...}}, loss={{...}} or a dotted key")
    pipeline["train"] = deep_update(TRAIN_DEFAULTS, pipeline.get("train", {}))
    pipeline["train"]["normalization"] = normalization_config(
        pipeline["train"]["normalization"], defaults=TRAIN_DEFAULTS["normalization"])
    if not isinstance(pipeline["train"]["visualize"], bool):
        raise TypeError("visualize must be true/false")
    ce_cfg = pipeline.get("loss", {}).get("ce", {})
    if "label_smoothing" in ce_cfg:
        ce_cfg["label_smoothing"] = validate_label_smoothing(ce_cfg["label_smoothing"])
    pipeline["train"]["num_classes"] = validate_num_classes(pipeline["train"].get("num_classes"), "train.num_classes")
    if pipeline["train"]["num_classes"] is not None:
        pipeline["data_cfg"]["num_classes"] = pipeline["train"]["num_classes"]
    pipeline.setdefault("inference", {})["num_classes"] = validate_num_classes(
        pipeline.get("inference", {}).get("num_classes"), "inference.num_classes")
    pipeline["output_dir"] = str(remap_repo_path(pipeline["output_dir"]))
    if pipeline["data_cfg"].get("root"):
        root_base = Path(pipeline["data_path"]).parent if pipeline["data_cfg"].get("root_relative_to_yaml", False) and pipeline.get("data_path") else None
        pipeline["data_cfg"]["root"] = str(remap_repo_path(pipeline["data_cfg"]["root"], root_base))
    if "input" in pipeline["data_cfg"]:
        from pulsecls.data.sources import require_class_folders
        pipeline["data_cfg"]["input"] = require_class_folders(pipeline["data_cfg"])
    pipeline["task"] = normalize_task(pipeline.get("task"))
    return pipeline


CLI_HELP = """
PulseCLS commands:

    python -m pulsecls train --pipeline configs/pipeline/cnn_softmax_synthetic.yaml
    python -m pulsecls predict --pipeline configs/pipeline/cnn_softmax_synthetic.yaml --checkpoint results/.../best_model.pt
    python -m pulsecls val --pipeline configs/pipeline/cnn_softmax_synthetic.yaml --checkpoint results/.../best_model.pt
"""


def entrypoint(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PulseCLS CLI")
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--pipeline", default=DEFAULT_CFG_DICT["pipeline"])
        target.add_argument("--data", default=None)
        target.add_argument("--device", default=None)
        target.add_argument("--output-dir", default=None)
        target.add_argument("--num-classes", type=int, default=None)

    train_parser = sub.add_parser("train")
    add_common(train_parser)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--labelsmooth", type=float, default=None, help="Label smoothing in [0, 1]; 0 disables it")
    train_parser.add_argument("--normalization", choices=("zscore", "minmax", "maxabs", "robust", "log_zscore", "none"),
                              default=None, help="Feature normalization fitted only on training pulses")
    train_parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=None,
                              help="Save training figures and validation t-SNE (default: enabled)")
    train_parser.add_argument("--resume", action="store_true", default=None)
    train_parser.add_argument("--resume-checkpoint", default=None)

    predict_parser = sub.add_parser("predict")
    add_common(predict_parser)
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--split", default="val", choices=("train", "val"))
    predict_parser.add_argument("--save-dir", default=None)
    predict_parser.add_argument("--source", default=None, help="Flat CSV folder; takes precedence over split")
    predict_parser.add_argument("--labels-file", default=None, help="Optional filename/label TXT for evaluation")

    val_parser = sub.add_parser("val")
    add_common(val_parser)
    val_parser.add_argument("--checkpoint", required=True)
    val_parser.add_argument("--split", default="val", choices=("train", "val"))
    val_parser.add_argument("--save-dir", default=None)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    from pulsecls import PulseCLS

    model = PulseCLS(args.pipeline)
    if args.mode == "train":
        model.train(
            resume=args.resume,
            resume_checkpoint=args.resume_checkpoint,
            data=args.data,
            device=args.device,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            labelsmooth=args.labelsmooth,
            normalization=args.normalization,
            visualize=args.visualize,
            num_classes=args.num_classes,
        )
        return
    model.predict(
        checkpoint=args.checkpoint,
        split=args.split,
        save_dir=args.save_dir,
        data=args.data,
        device=args.device,
        output_dir=args.output_dir,
        source=getattr(args, "source", None),
        labels_file=getattr(args, "labels_file", None),
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    entrypoint()
