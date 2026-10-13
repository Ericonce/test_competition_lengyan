"""Shared defaults and validation for ordered PDW augmentation."""
from copy import deepcopy
from pathlib import Path
import math

from pulsecls.cfg import load_yaml, deep_update

PRESET_PATH = Path(__file__).resolve().parents[2] / "configs/augment/default.yaml"

# Parameters are defined explicitly to reject misspelled or meaningless thresholds.
METHOD_DEFAULTS = {
    "jittering": {"std": .01}, "scaling": {"std": .02},
    "rotation": {"max_angle_degrees": 5.},
    "magnitude_warping": {"std": .05, "knots": 4},
    "white_noise_windows": {"std": .02, "window_ratio": .2},
    "permutation": {"segments": 4},
    "slicing": {"keep_ratio": .9}, "window_slicing": {"keep_ratio": .9},
    "time_warping": {"std": .1, "knots": 4},
    "window_warping": {"window_ratio": .2, "scales": [.8, 1.2]},
    "time_shift": {"ratio": .1}, "time_flip": {},
    "frequency_warping": {"max_ratio": .05},
    "vtlp": {"max_ratio": .05, "pivot_ratio": .8},
    "frequency_slicing": {"band_ratio": .1},
    "random_noise_boxes": {"std": .02, "time_ratio": .2, "frequency_ratio": .1, "n_fft": 32},
    "stft_mask": {"time_ratio": .1, "frequency_ratio": .1, "n_fft": 32},
    "aaft": {}, "iaaft": {"iterations": 10},
    "sfcc": {"bands": 3, "blend": .3},
    "rim": {"mask_ratio": .1}, "interpolation": {"blend": .3},
    "spawner": {"std": .01, "window_ratio": .1, "max_points": 64},
    "random_guided_warp": {"window_ratio": .1, "max_points": 64},
    "random_guided_warp_shape": {"window_ratio": .1, "max_points": 64, "shape_radius": 2},
    "discriminative_guided_warp": {"window_ratio": .1, "max_points": 64, "candidates": 3},
    "discriminative_guided_warp_shape": {"window_ratio": .1, "max_points": 64, "candidates": 3, "shape_radius": 2},
    "dtw_mix": {"blend": .3, "window_ratio": .1, "max_points": 64},
    "dba": {"iterations": 2, "neighbors": 3, "window_ratio": .1, "max_points": 64},
    "emda": {"blend": .3, "gain_std": .05, "max_delay_ratio": .05},
    "statistical_style": {"blend": .2},
    "sample_combining": {"bands": 3, "blend": .3},
    "pulse_mask": {"ratio": .01}, "feature_mask": {"ratio": .01},
    "mixup": {"alpha": .2},
}

LABEL_METHODS = {"sfcc", "interpolation", "spawner", "random_guided_warp",
                 "random_guided_warp_shape", "discriminative_guided_warp",
                 "discriminative_guided_warp_shape", "dtw_mix", "dba", "emda",
                 "statistical_style", "sample_combining"}


def default_augment_config(enabled: bool | None = None) -> dict:
    """Read the shared editable preset; callers can override nested values locally."""
    result = deepcopy(load_yaml(PRESET_PATH.resolve()))
    result["methods"], result["order"] = resolve_methods(result.get("methods", {}), result.get("order"))
    if enabled is not None:
        result["enabled"] = enabled
    return result


def resolve_methods(methods: dict, order: list[str] | None) -> tuple[dict, list]:
    unknown = set(methods) - set(METHOD_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown augmentation methods: {sorted(unknown)}")
    resolved = {}
    for name, parameters in METHOD_DEFAULTS.items():
        override = methods.get(name, {})
        if not isinstance(override, dict):
            raise TypeError(f"{name} must have enabled/prob/parameter settings")
        allowed = set(parameters) | {"enabled", "prob"}
        if set(override) - allowed:
            raise ValueError(f"Unknown parameters for {name}: {sorted(set(override) - allowed)}")
        spec = deep_update({"enabled": False, "prob": 1., **parameters}, override)
        for key, value in spec.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{name}.{key} must be finite")
        if not isinstance(spec["enabled"], bool):
            raise TypeError(f"{name}.enabled must be true/false")
        if not 0 <= float(spec["prob"]) <= 1:
            raise ValueError(f"{name}.prob must be between 0 and 1")
        for key, value in spec.items():
            if key in ("ratio", "mask_ratio", "window_ratio", "time_ratio", "frequency_ratio", "band_ratio", "max_delay_ratio", "blend", "max_ratio"):
                if not 0 <= float(value) <= 1:
                    raise ValueError(f"{name}.{key} must be between 0 and 1")
            if key in ("keep_ratio", "pivot_ratio") and not 0 < float(value) <= 1:
                raise ValueError(f"{name}.{key} must be in (0, 1]")
            if key in ("std", "gain_std", "max_angle_degrees") and float(value) < 0:
                raise ValueError(f"{name}.{key} must be nonnegative")
            if key in ("knots", "segments", "bands", "iterations", "neighbors", "candidates", "shape_radius", "max_points", "n_fft"):
                minimum = 4 if key in ("max_points", "n_fft") else 1
                if isinstance(value, bool) or int(value) != value or int(value) < minimum:
                    raise ValueError(f"{name}.{key} must be an integer >= {minimum}")
        if "scales" in spec and (not spec["scales"] or any(not math.isfinite(float(s)) or float(s) <= 0 for s in spec["scales"])):
            raise ValueError(f"{name}.scales must be a nonempty list of positive scales")
        if "alpha" in spec and float(spec["alpha"]) <= 0:
            raise ValueError("mixup.alpha must be positive")
        resolved[name] = spec
    sequence = list(METHOD_DEFAULTS) if order is None else list(order)
    if len(sequence) != len(set(sequence)) or set(sequence) - set(METHOD_DEFAULTS):
        raise ValueError("order must contain unique known method names")
    missing = {name for name, spec in resolved.items() if spec["enabled"]} - set(sequence)
    if missing:
        raise ValueError(f"Enabled methods missing from order: {sorted(missing)}")
    if "mixup" in sequence and resolved["mixup"]["enabled"]:
        later = sequence[sequence.index("mixup") + 1:]
        if any(resolved[name]["enabled"] and name in LABEL_METHODS for name in later):
            raise ValueError("Put same-class sample fusion before mixup to keep label semantics correct")
    return resolved, sequence
