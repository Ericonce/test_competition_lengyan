"""Reproducible CF/PW/PRI pulse-descriptor synthesis (no intrapulse IQ)."""
from __future__ import annotations

import itertools
import json
import shutil
import tempfile
import warnings
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml

from pulsecls.cfg import REPO_ROOT, deep_update, load_yaml, remap_repo_path

DEFAULT_CONFIG = REPO_ROOT / "configs/generator/synthetic_radar.yaml"
FEATURES = ["CF_MHz", "PW_us", "PRI_us"]
PRI_MODES = ("fixed", "staggered", "jittered", "sinusoidal", "sliding")
CF_MODES = ("linear_fm", "nonlinear_fm", "frequency_coded", "phase_coded")
PW_MODES = ("fixed", "multilevel", "sinusoidal")


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _range(value, name, *, positive=False, integer=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must have [minimum, maximum]")
    if integer:
        low, high = (_integer(v, name, 0 if not positive else 1) for v in value)
    else:
        low, high = map(float, value)
    if not np.isfinite([low, high]).all() or low > high or (positive and low <= 0):
        raise ValueError(f"Invalid range for {name}: {value}")
    return low, high


def _sample(rng, bounds, integer=False):
    low, high = bounds
    return int(rng.integers(low, high + 1)) if integer else float(rng.uniform(low, high))


def add_gaussian_noise(clean, snr_db, channel_scales, rng, power_allocation="per_channel", correlation=0.0):
    """Compute full scaled-signal energy, then scale Gaussian draws to target SNR."""
    clean = np.asarray(clean, dtype=np.float64)
    scales = np.asarray(channel_scales, dtype=np.float64)
    if clean.ndim != 2 or clean.shape[1] != 3 or not len(clean) or not np.isfinite(clean).all():
        raise ValueError("clean must be a finite nonempty [length, 3] sequence")
    if scales.shape != (3,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("channel_scales must contain three positive finite numbers")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    signal_energy = float(np.sum(np.square(clean / scales)))
    if signal_energy <= 0:
        raise ValueError("Cannot define SNR for a zero-energy sequence")
    target_energy = signal_energy * 10.0 ** (-float(snr_db) / 10.0)
    if not np.isfinite(target_energy) or target_energy <= 0:
        raise ValueError("SNR is outside the numerically supported range")
    normalized_noise = rng.normal(size=clean.shape)
    if not np.isfinite(correlation) or not 0 <= correlation < 1:
        raise ValueError("noise correlation must be in [0, 1)")
    if correlation:
        innovation = np.sqrt(1 - correlation ** 2)
        for i in range(1, len(normalized_noise)):
            normalized_noise[i] = correlation * normalized_noise[i - 1] + innovation * normalized_noise[i]
    if power_allocation == "per_channel":
        channel_energy = np.sum(np.square(clean / scales), axis=0)
        target_per_channel = channel_energy * (target_energy / signal_energy)
        normalized_noise *= np.sqrt(target_per_channel / np.sum(normalized_noise ** 2, axis=0))
    elif power_allocation == "equal":
        normalized_noise *= np.sqrt(target_energy / np.sum(normalized_noise ** 2))
    else:
        raise ValueError("power_allocation must be per_channel or equal")
    observed = clean + normalized_noise * scales
    # Measure the actual written values, including floating-point rounding.
    noise_energy = float(np.sum(np.square((observed - clean) / scales)))
    if noise_energy <= 0:
        raise ValueError("SNR is too high to represent the noise at float64 precision")
    return observed, {
        "snr_db": float(snr_db), "achieved_snr_db": float(10 * np.log10(signal_energy / noise_energy)),
        "signal_energy": signal_energy, "noise_energy": noise_energy,
        "power_allocation": power_allocation,
        "noise_correlation": float(correlation),
    }


def _valid_observation(observed, duty_ratio):
    return np.isfinite(observed).all() and (observed > 0).all() and (observed[:, 1] < observed[:, 2] * duty_ratio).all()


def _repair_observation(clean, observed, scales, duty_ratio, minimum, details):
    repaired = np.maximum(observed, minimum)
    repaired[:, 1] = np.minimum(repaired[:, 1], repaired[:, 2] * duty_ratio * (1 - 1e-6))
    details["corrected_values"] = int(np.count_nonzero(repaired != observed))
    details["noise_energy"] = float(np.sum(((repaired - clean) / scales) ** 2))
    details["achieved_snr_db"] = float(10 * np.log10(details["signal_energy"] / details["noise_energy"]))
    return repaired


def _remove_temporary_directory(path, parent, prefix):
    """Verify the exact generated sibling path before any recursive removal."""
    path, parent = Path(path), Path(parent).resolve()
    resolved = path.resolve()
    if resolved.parent != parent or not path.name.startswith(prefix) or path.is_symlink():
        raise ValueError(f"Refusing to remove an unexpected temporary directory: {path}")
    if path.exists():
        shutil.rmtree(path)


class SyntheticRadarGenerator:
    """Template -> emitter identity -> variable-length noisy pulse sequences.

    YAML values can be overridden with constructor keywords (nested dictionaries
    are merged). New data is staged completely before replacing an old dataset.
    """

    def __init__(self, config: str | Path | dict | None = None, **overrides):
        source = load_yaml(config) if isinstance(config, (str, Path)) else (config or {})
        source = deepcopy(source)
        # Old saved generation configs remain readable; partitioning moved to pipelines.
        if "split" in source:
            source.pop("split")
            warnings.warn("Generator split settings are ignored; configure dataset sampling/splitting in pipeline YAML")
        self.config = deep_update(deep_update(load_yaml(DEFAULT_CONFIG), source), overrides)
        self._validate()

    def _validate(self):
        c = self.config
        defaults = load_yaml(DEFAULT_CONFIG)
        for key in c:
            if key not in defaults:
                raise ValueError(f"Unknown generator setting: {key}")
        for section in ("sequence", "transmitter", "parameters", "noise", "overlap", "observation"):
            if not isinstance(c[section], dict) or set(c[section]) - set(defaults[section]):
                raise ValueError(f"Unknown or invalid settings in {section}")
        for section, key in (("sequence", "random_length"), ("noise", "enabled"), ("overlap", "enabled"), ("observation", "enabled")):
            if not isinstance(c[section][key], bool):
                raise ValueError(f"{section}.{key} must be true/false")
        if not isinstance(c["save_clean"], bool):
            raise ValueError("save_clean must be true/false")
        _integer(c["seed"], "seed", 0)
        count = _integer(c["num_emitters"], "num_emitters")
        sizes = c["sequences_per_emitter"]
        self.counts = sizes if isinstance(sizes, list) else [sizes] * count
        if len(self.counts) != count:
            raise ValueError("sequences_per_emitter list must match num_emitters")
        for size in self.counts:
            _integer(size, "sequences_per_emitter")
        _integer(c["num_transmitter_types"], "num_transmitter_types")
        _integer(c["test_num_sequences"], "test_num_sequences")
        seq = c["sequence"]
        _integer(seq["length"], "sequence.length")
        _range(seq["length_range"], "sequence.length_range", positive=True, integer=True)
        _range(seq["start_index_range"], "sequence.start_index_range", integer=True)
        if not 0 <= float(seq["parameter_drift_ratio"]) < 1:
            raise ValueError("parameter_drift_ratio must be in [0, 1)")
        t = c["transmitter"]
        for key, allowed in (("pri_modes", PRI_MODES), ("cf_modes", CF_MODES), ("pw_modes", PW_MODES)):
            if not isinstance(t[key], list) or not t[key] or len(t[key]) != len(set(t[key])) or set(t[key]) - set(allowed):
                raise ValueError(f"Invalid {key}; allowed: {allowed}")
        templates = t["templates"]
        if templates is not None:
            if not isinstance(templates, list) or not templates:
                raise ValueError("templates must be a nonempty list or null")
            for template in templates:
                if not isinstance(template, dict) or set(template) != {"pri_mode", "cf_mode", "pw_mode"}:
                    raise ValueError("Each template requires pri_mode, cf_mode, pw_mode")
                if template["pri_mode"] not in PRI_MODES or template["cf_mode"] not in CF_MODES or template["pw_mode"] not in PW_MODES:
                    raise ValueError(f"Invalid transmitter template: {template}")
        elif c["num_transmitter_types"] > len(t["pri_modes"]) * len(t["cf_modes"]):
            raise ValueError("num_transmitter_types exceeds available PRI x CF combinations")
        p = c["parameters"]
        integer_keys = {"period_pulses", "stagger_levels", "frequency_levels", "code_length", "code_hold_pulses", "pw_levels"}
        for key, value in p.items():
            if key != "max_duty_ratio":
                _range(value, f"parameters.{key}", positive=True, integer=key in integer_keys)
        for key in ("pri_depth_ratio", "cf_depth_ratio", "pw_depth_ratio"):
            if p[key][1] >= 1:
                raise ValueError(f"{key} must be smaller than 1")
        if not 0 < float(p["max_duty_ratio"]) < 1:
            raise ValueError("max_duty_ratio must be in (0, 1)")
        noise = c["noise"]
        _range(noise["snr_db_range"], "noise.snr_db_range")
        _, max_correlation = _range(noise["correlation_range"], "noise.correlation_range")
        if noise["correlation_range"][0] < 0 or max_correlation >= 1:
            raise ValueError("noise.correlation_range must be in [0, 1)")
        scales = np.asarray(noise["channel_scales"], dtype=float)
        if scales.shape != (3,) or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("channel_scales must contain three positive finite numbers")
        _integer(noise["max_attempts"], "noise.max_attempts")
        if noise["power_allocation"] not in {"per_channel", "equal"}:
            raise ValueError("power_allocation must be per_channel or equal")
        if noise["invalid_policy"] not in {"clip", "resample", "error"}:
            raise ValueError("invalid_policy must be clip, resample or error")
        if not np.isfinite(noise["min_value"]) or noise["min_value"] <= 0:
            raise ValueError("noise.min_value must be positive and finite")
        overlap, observation = c["overlap"], c["observation"]
        _integer(overlap["emitters_per_group"], "overlap.emitters_per_group")
        for section, key in ((overlap, "base_jitter_ratio"), (observation, "session_bias_std_ratio")):
            ratios = np.asarray(section[key], dtype=float)
            if ratios.shape != (3,) or not np.isfinite(ratios).all() or (ratios < 0).any() or (ratios >= 1).any():
                raise ValueError(f"{key} must contain three ratios in [0, 1)")
        if not 0 <= float(overlap["modulation_jitter_ratio"]) < 1:
            raise ValueError("modulation_jitter_ratio must be in [0, 1)")
        if not 0 <= float(observation["session_bias_max_ratio"]) < 1:
            raise ValueError("session_bias_max_ratio must be in [0, 1)")
        lo, hi = _range(observation["missing_pulse_rate_range"], "missing_pulse_rate_range")
        if lo < 0 or hi >= 1:
            raise ValueError("missing_pulse_rate_range must be in [0, 1)")

    def _rng(self, *key):
        return np.random.default_rng(np.random.SeedSequence(self.config["seed"], spawn_key=key))

    def create_emitters(self):
        """Sample identity parameters once; reuse them for every same-class stream."""
        c, t, p = self.config, self.config["transmitter"], self.config["parameters"]
        rng = self._rng(0)
        if t["templates"] is not None:
            templates = deepcopy(t["templates"])
        else:
            pairs = list(itertools.product(t["pri_modes"], t["cf_modes"]))
            selected = rng.choice(len(pairs), c["num_transmitter_types"], replace=False)
            templates = [{"pri_mode": pairs[i][0], "cf_mode": pairs[i][1], "pw_mode": str(rng.choice(t["pw_modes"]))} for i in selected]
        template_ids = rng.permutation(np.arange(c["num_emitters"]) % len(templates))
        emitters = []
        for label, template_id in enumerate(template_ids):
            group = label // c["overlap"]["emitters_per_group"] if c["overlap"]["enabled"] else None
            if group is not None:
                template_id = group % len(templates)
            r = self._rng(1, label)
            emitter = {"label": label, "group_id": group, "template_id": int(template_id), **templates[int(template_id)]}
            for key in ("cf_mhz", "pw_us", "pri_us", "pri_depth_ratio", "cf_depth_ratio", "pw_depth_ratio"):
                emitter[key] = _sample(r, p[key])
            if group is not None:
                group_rng = self._rng(4, group)
                for key, jitter in zip(("cf_mhz", "pw_us", "pri_us"), c["overlap"]["base_jitter_ratio"]):
                    center = _sample(group_rng, p[key])
                    emitter[key] = float(np.clip(center * (1 + r.uniform(-jitter, jitter)), *p[key]))
                for key in ("pri_depth_ratio", "cf_depth_ratio", "pw_depth_ratio"):
                    center = _sample(group_rng, p[key])
                    jitter = c["overlap"]["modulation_jitter_ratio"]
                    emitter[key] = float(np.clip(center * (1 + r.uniform(-jitter, jitter)), *p[key]))
            for key in ("period_pulses", "stagger_levels", "frequency_levels", "code_length", "code_hold_pulses", "pw_levels"):
                emitter[key] = _sample(r, p[key], integer=True)
            emitter["phase"] = float(r.uniform(0, 2 * np.pi))
            emitter["sweep_direction"] = int(r.choice([-1, 1]))
            emitter["pri_levels"] = r.uniform(-1, 1, emitter["stagger_levels"]).tolist()
            emitter["frequency_code"] = r.integers(0, emitter["frequency_levels"], emitter["code_length"]).tolist()
            emitter["phase_code"] = r.choice([-1, 1], emitter["code_length"]).tolist()
            if emitter["cf_mode"] == "phase_coded":
                emitter["representation"] = "phase_code_driven_frequency_proxy"
            # Worst-case modulation and per-stream drift must preserve pulse separation.
            d = c["sequence"]["parameter_drift_ratio"]
            if emitter["pw_us"] * (1 + emitter["pw_depth_ratio"]) * (1 + d) >= emitter["pri_us"] * (1 - emitter["pri_depth_ratio"]) * (1 - d) * p["max_duty_ratio"]:
                raise ValueError("PW/PRI ranges allow overlapping pulses; lower PW or increase PRI")
            emitters.append(emitter)
        return templates, emitters

    def clean_sequence(self, emitter, length, start_index, rng):
        """All modulation is represented across pulse indices, not inside a pulse."""
        index = np.arange(length, dtype=np.int64) + start_index
        phase = index / emitter["period_pulses"]
        ramp = emitter["sweep_direction"] * (2 * (phase % 1) - 1)
        pri_mode = emitter["pri_mode"]
        if pri_mode == "fixed":
            pri_shape = np.zeros(length)
        elif pri_mode == "staggered":
            levels = np.asarray(emitter["pri_levels"])
            pri_shape = levels[index % len(levels)]
        elif pri_mode == "jittered":
            pri_shape = rng.uniform(-1, 1, length)
        elif pri_mode == "sinusoidal":
            pri_shape = np.sin(2 * np.pi * phase + emitter["phase"])
        else:
            pri_shape = ramp
        cf_mode = emitter["cf_mode"]
        if cf_mode == "linear_fm":
            cf_shape = ramp
        elif cf_mode == "nonlinear_fm":
            cf_shape = ramp ** 3
        else:
            code_index = (index // emitter["code_hold_pulses"]) % emitter["code_length"]
            if cf_mode == "frequency_coded":
                code = np.asarray(emitter["frequency_code"])
                cf_shape = 2 * code[code_index] / max(1, emitter["frequency_levels"] - 1) - 1
            else:
                cf_shape = np.asarray(emitter["phase_code"])[code_index]
        if emitter["pw_mode"] == "fixed":
            pw_shape = np.zeros(length)
        elif emitter["pw_mode"] == "multilevel":
            pw_shape = 2 * (index % emitter["pw_levels"]) / max(1, emitter["pw_levels"] - 1) - 1
        else:
            pw_shape = np.sin(2 * np.pi * phase + emitter["phase"])
        drift = rng.uniform(-1, 1, 3) * self.config["sequence"]["parameter_drift_ratio"]
        base = np.array([emitter["cf_mhz"], emitter["pw_us"], emitter["pri_us"]]) * (1 + drift)
        shapes = np.column_stack([cf_shape, pw_shape, pri_shape])
        depths = np.array([emitter["cf_depth_ratio"], emitter["pw_depth_ratio"], emitter["pri_depth_ratio"]])
        return base * (1 + shapes * depths)

    def observation_sequence(self, clean, rng):
        """Apply session bias and detection loss; recompute forward PRI from TOA."""
        cfg = self.config["observation"]
        metadata = {"generated_length": len(clean), "missing_pulses": 0,
                    "requested_missing_rate": 0.0, "session_bias_ratios": [0., 0., 0.]}
        if not cfg["enabled"]:
            return clean, metadata
        for _ in range(256):
            bias = np.clip(rng.normal(size=3) * cfg["session_bias_std_ratio"],
                           -cfg["session_bias_max_ratio"], cfg["session_bias_max_ratio"])
            observed = clean * (1 + bias)
            if _valid_observation(observed, self.config["parameters"]["max_duty_ratio"]):
                break
        else:
            raise ValueError("Session bias cannot satisfy PW/PRI constraints; reduce bias ratios")
        rate = _sample(rng, cfg["missing_pulse_rate_range"])
        keep = rng.random(len(observed)) >= rate
        keep[0] = keep[-1] = True  # Preserve interval endpoints; internal losses alter forward PRI.
        indices = np.flatnonzero(keep)
        arrivals = np.r_[0., np.cumsum(observed[:, 2])]
        result = observed[indices].copy()
        result[:, 2] = np.diff(np.r_[arrivals[indices], arrivals[-1]])
        metadata.update(missing_pulses=int(len(observed) - len(result)),
                        requested_missing_rate=rate, session_bias_ratios=bias.tolist())
        return result, metadata

    def generate(self, output_dir: str | Path | None = None):
        templates, emitters = self.create_emitters()
        root = remap_repo_path(output_dir or self.config["output_dir"])
        result = self._generate_output(root, templates, emitters)
        self.training_root = root
        return result

    def generate_test(self, output_dir=None, num_sequences=None):
        """Reuse training emitter identities and distribution with an independent RNG."""
        training_root = getattr(self, "training_root", remap_repo_path(self.config["output_dir"]))
        for name in ("generation_config.yaml", "emitters.json", "summary.json"):
            if not (training_root / name).is_file():
                raise FileNotFoundError(f"Generate training data first; missing {training_root / name}")
        saved = load_yaml(training_root / "generation_config.yaml")
        ignored = {"output_dir", "test_output_dir", "test_num_sequences", "save_clean", "sequences_per_emitter", "split"}
        for key in set(saved) - ignored:
            if saved[key] != self.config[key]:
                raise ValueError(f"Training snapshot differs from YAML at {key}; use its generation_config.yaml or regenerate training data first")
        identities = json.loads((training_root / "emitters.json").read_text(encoding="utf-8"))
        emitters = identities["emitters"]
        if [e["label"] for e in emitters] != list(range(self.config["num_emitters"])):
            raise ValueError("Saved emitter labels do not match num_emitters")
        total = _integer(num_sequences if num_sequences is not None else self.config["test_num_sequences"], "test_num_sequences")
        root = remap_repo_path(output_dir or self.config["test_output_dir"])
        if not root.name.isascii():
            raise ValueError("Use an ASCII test directory name, such as synthetic_radar_test")
        if root == training_root or root.is_relative_to(training_root) or training_root.is_relative_to(root):
            raise ValueError("Test output must be separate from the training pool")
        return self._generate_output(root, identities["templates"], emitters, test_count=total)

    def _generate_output(self, root, templates, emitters, test_count=None):
        # Validate the final absolute target before moving or deleting directories.
        if root == Path(root.anchor) or root == REPO_ROOT.resolve() or REPO_ROOT.resolve().is_relative_to(root):
            raise ValueError(f"Output must be a dataset subdirectory, not a project/drive root: {root}")
        if root.exists():
            if not root.is_dir():
                raise ValueError(f"Output is not a directory: {root}")
            if any(root.iterdir()) and not all((root / name).is_file() for name in ("generation_config.yaml", "emitters.json")):
                raise ValueError(f"Refusing to replace an unrelated directory without generator metadata: {root}")
        root.parent.mkdir(parents=True, exist_ok=True)
        prefix = f".{root.name}.generating-"
        stage = Path(tempfile.mkdtemp(prefix=prefix, dir=root.parent)).resolve()
        backup = root.with_name(f".{root.name}.previous-{uuid4().hex}")
        try:
            result = (self._write_test_dataset(stage, root, templates, emitters, test_count)
                      if test_count is not None else self._write_dataset(stage, root, templates, emitters))
            # Both moves stay inside the explicitly selected dataset's parent.
            if stage.parent != root.parent or backup.resolve().parent != root.parent:
                raise ValueError("Unexpected dataset replacement paths")
            if root.exists():
                root.rename(backup)
            try:
                stage.rename(root)
            except Exception:
                if backup.exists():
                    backup.rename(root)
                raise
            if backup.exists():
                try:
                    _remove_temporary_directory(backup, root.parent, f".{root.name}.previous-")
                except OSError as error:
                    warnings.warn(f"New data is ready, but old backup could not be removed: {backup}: {error}")
            return result
        finally:
            _remove_temporary_directory(stage, root.parent, prefix)

    def _make_sequence(self, emitter, rng):
        c, seq, noise = self.config, self.config["sequence"], self.config["noise"]
        length = _sample(rng, seq["length_range"], True) if seq["random_length"] else seq["length"]
        start = _sample(rng, seq["start_index_range"], True)
        clean = self.clean_sequence(emitter, length, start, rng)
        clean, observation = self.observation_sequence(clean, rng)
        length = len(clean)
        details = {"snr_db": None, "achieved_snr_db": None,
                   "signal_energy": float(np.sum((clean / noise["channel_scales"]) ** 2)), "noise_energy": 0.0}
        observed = clean
        if noise["enabled"]:
            snr = _sample(rng, noise["snr_db_range"])
            correlation = _sample(rng, noise["correlation_range"])
            attempts = noise["max_attempts"] if noise["invalid_policy"] == "resample" else 1
            for attempt in range(attempts):
                observed, details = add_gaussian_noise(clean, snr, noise["channel_scales"], rng, noise["power_allocation"], correlation)
                details.update(invalid_policy=noise["invalid_policy"], corrected_values=0)
                if noise["invalid_policy"] == "clip":
                    observed = _repair_observation(clean, observed, noise["channel_scales"], c["parameters"]["max_duty_ratio"], noise["min_value"], details)
                if _valid_observation(observed, c["parameters"]["max_duty_ratio"]):
                    break
            else:
                raise ValueError(f"Cannot generate valid noisy emitter {emitter['label']}; use noise.invalid_policy=clip, increase SNR or adjust PW/PRI ranges")
            details["noise_attempts"] = attempt + 1
        return clean, observed, {"length": length, "start_index": start, **observation, **details}

    def _write_dataset(self, root, final_root, templates, emitters):
        c, seq, noise = self.config, self.config["sequence"], self.config["noise"]
        snapshot = deepcopy(c)
        snapshot["output_dir"] = final_root.relative_to(REPO_ROOT).as_posix() if final_root.is_relative_to(REPO_ROOT) else str(final_root)
        (root / "generation_config.yaml").write_text(yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (root / "emitters.json").write_text(json.dumps({"templates": templates, "emitters": emitters}, ensure_ascii=False, indent=2), encoding="utf-8")
        lengths, snrs = [], []
        with (root / "sequences.jsonl").open("w", encoding="utf-8") as manifest:
            for emitter, count in zip(emitters, self.counts):
                label = emitter["label"]
                (root / str(label)).mkdir()
                for number in range(count):
                    rng = self._rng(2, label, number)
                    clean, observed, details = self._make_sequence(emitter, rng)
                    length = details["length"]
                    if details["snr_db"] is not None:
                        snrs.append(details["snr_db"])
                    relative = f"{label}/{number}.csv"
                    np.savetxt(root / relative, observed, delimiter=",", header=",".join(FEATURES), comments="", fmt="%.17g", encoding="utf-8")
                    if c["save_clean"]:
                        folder = root / "clean" / str(label)
                        folder.mkdir(parents=True, exist_ok=True)
                        np.save(folder / f"{number}.npy", clean)
                    manifest.write(json.dumps({"path": relative, "label": label, "length": length,
                                              "group_id": emitter["group_id"], **details}) + "\n")
                    lengths.append(length)
        data_cfg = {"name": "synthetic_radar", "root": ".", "root_relative_to_yaml": True, "num_classes": c["num_emitters"],
                    "features": FEATURES, "cache": True}
        data_cfg["all"] = "."
        data_text = "# 三通道输入：载频 CF(MHz)、脉宽 PW(us)、脉冲重复周期 PRI(us)。\n"
        data_text += "# 固定输入长度与补齐/裁剪策略统一设置在 configs/pipeline/*.yaml。\n"
        data_text += yaml.safe_dump(data_cfg, allow_unicode=True, sort_keys=False)
        (root / "data.yaml").write_text(data_text, encoding="utf-8")
        summary = {"num_emitters": len(emitters), "num_sequences": len(lengths),
                   "length_min": min(lengths), "length_max": max(lengths),
                   "snr_min_db": min(snrs) if snrs else None, "snr_max_db": max(snrs) if snrs else None,
                   "partitioning": "configured_in_pipeline"}
        (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (root / "dataset_index.json").write_text(json.dumps({
            "version": 1, "layout": "numbered_csv", "generation_id": uuid4().hex,
            "classes": [{"label": i, "count": n} for i, n in enumerate(self.counts)]
        }, indent=2), encoding="utf-8")
        return {"output_dir": str(final_root), **summary}

    def _write_test_dataset(self, root, final_root, templates, emitters, total):
        snapshot = deepcopy(self.config)
        training_root = getattr(self, "training_root", remap_repo_path(self.config["output_dir"]))
        for key, path in (("output_dir", training_root), ("test_output_dir", final_root)):
            snapshot[key] = path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)
        (root / "generation_config.yaml").write_text(yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (root / "emitters.json").write_text(json.dumps({"templates": templates, "emitters": emitters}, indent=2), encoding="utf-8")
        # Balanced labels are shuffled; names carry no class information.
        labels = np.arange(total) % len(emitters)
        self._rng(6).shuffle(labels)
        counters = np.zeros(len(emitters), dtype=int)
        lengths = []
        with (root / "labels.txt").open("w", encoding="utf-8") as label_file, (root / "sequences.jsonl").open("w", encoding="utf-8") as manifest:
            for number, label in enumerate(labels):
                emitter = emitters[int(label)]
                rng = self._rng(5, int(label), int(counters[label]))
                counters[label] += 1
                _, observed, details = self._make_sequence(emitter, rng)
                name = f"{number}.csv"
                np.savetxt(root / name, observed, delimiter=",", header=",".join(FEATURES), comments="", fmt="%.17g", encoding="utf-8")
                label_file.write(f"{name}\t{int(label)}\n")
                manifest.write(json.dumps({"path": name, "label": int(label), **details}) + "\n")
                lengths.append(details["length"])
        summary = {"num_emitters": len(emitters), "num_sequences": total,
                   "length_min": min(lengths), "length_max": max(lengths),
                   "partitioning": "independent_test", "rng_domain": 5}
        (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"output_dir": str(final_root), "labels_file": str(final_root / "labels.txt"), **summary}
