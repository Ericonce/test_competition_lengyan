"""Training configuration snapshots, logs and figures from saved numeric records."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import inspect
import json
from pathlib import Path
import re
import threading
import traceback

import numpy as np
import pandas as pd
import yaml

from pulsecls.cfg import TRAIN_DEFAULTS, deep_update
from pulsecls.utils.progress import Progress


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class _Tee:
    def __init__(self, terminal, log, lock):
        self.terminal, self.log, self.lock = terminal, log, lock

    def write(self, text):
        with self.lock:
            self.terminal.write(text)
            self.log.write(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).replace("\r", "\n"))
        return len(text)

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.log.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


@contextmanager
def training_log(output_dir, *, resume=False):
    import sys
    path = Path(output_dir) / "training.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if resume else "w", encoding="utf-8", buffering=1) as log:
        lock = threading.RLock()
        with redirect_stdout(_Tee(sys.stdout, log, lock)), redirect_stderr(_Tee(sys.stderr, log, lock)):
            print(f"Training started: {datetime.now().astimezone().isoformat()}", flush=True)
            try:
                yield
            except BaseException:
                traceback.print_exc(file=log)
                raise


def reporting_config(options):
    cfg = deep_update(TRAIN_DEFAULTS["reporting"], options or {})
    if not isinstance(cfg["dpi"], int) or cfg["dpi"] < 1:
        raise ValueError("reporting.dpi must be a positive integer")
    formats = cfg["formats"]
    if not isinstance(formats, (list, tuple)) or set(formats) - {"png", "svg", "pdf"}:
        raise ValueError("reporting.formats must be a list containing png/svg/pdf")
    cfg["formats"] = list(dict.fromkeys(["png", *formats]))
    tsne = cfg["tsne"]
    for key, minimum in (("max_samples", 0), ("seed", 0), ("max_iter", 250)):
        if not isinstance(tsne[key], int) or isinstance(tsne[key], bool) or tsne[key] < minimum:
            raise ValueError(f"reporting.tsne.{key} must be an integer >= {minimum}")
    if not np.isfinite(tsne["perplexity"]) or tsne["perplexity"] <= 0:
        raise ValueError("reporting.tsne.perplexity must be positive")
    return cfg


def save_configuration(output_dir, pipeline, *, runtime=None, print_config=True):
    root = Path(output_dir)
    folder = root / "configs"
    folder.mkdir(parents=True, exist_ok=True)
    effective = jsonable(pipeline)
    if runtime is not None:
        effective["runtime"] = jsonable(runtime)
    sections = {"pipeline": effective, "data": effective["data_cfg"],
                "augment": effective.get("augment", {}), "model": effective["model_cfg"]}
    for name, content in sections.items():
        text = yaml.safe_dump(content, allow_unicode=True, sort_keys=False)
        (folder / f"{name}.yaml").write_text(text, encoding="utf-8")
        if print_config and name != "pipeline":
            print(f"\nEffective {name}.yaml:\n{text}", flush=True)
    (root / "resolved_pipeline.json").write_text(json.dumps(effective, ensure_ascii=False, indent=2), encoding="utf-8")
    if print_config:
        summary = {key: value for key, value in effective.items()
                   if key not in {"data_cfg", "augment", "model_cfg"}}
        print("\nEffective pipeline/train settings:\n" + yaml.safe_dump(summary, allow_unicode=True, sort_keys=False), flush=True)


def clean_analysis_outputs(output_dir):
    """Remove only files owned by this reporter, including stale opposite-split data."""
    root = Path(output_dir)
    for split in ("val", "train"):
        for suffix in ("classification_report.csv", "classification_report.json", "confusion_matrix.csv",
                       "confusion_matrix_normalized.csv", "predictions.csv"):
            (root / f"{split}_{suffix}").unlink(missing_ok=True)
        for figure in ("per_class_metrics", "confusion_matrix", "tsne"):
            for extension in ("png", "svg", "pdf"):
                (root / "figures" / f"{split}_{figure}.{extension}").unlink(missing_ok=True)
        for name in (f"{split}_embeddings.npz", f"{split}_tsne.csv"):
            (root / "tsne" / name).unlink(missing_ok=True)
    (root / "tsne/settings.json").unlink(missing_ok=True)
    for figure in ("accuracy_curve", "macro_f1_curve", "training_curves"):
        for extension in ("png", "svg", "pdf"):
            (root / "figures" / f"{figure}.{extension}").unlink(missing_ok=True)


def select_tsne_indices(labels, max_samples, seed):
    """Balanced deterministic subset, without replacement, before feature extraction."""
    labels = np.asarray(labels)
    if max_samples == 0 or max_samples >= len(labels):
        return np.arange(len(labels))
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    if max_samples < len(groups):
        raise ValueError("reporting.tsne.max_samples must cover all represented classes, or be 0")
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(group) for group in groups]
    selected, depth = [], 0
    while len(selected) < max_samples:
        for group in groups:
            if depth < len(group):
                selected.append(group[depth])
                if len(selected) == max_samples:
                    break
        depth += 1
    return np.sort(np.asarray(selected, dtype=np.int64))


def save_predictions(output_dir, split, labels, predictions, confidence, paths):
    frame = pd.DataFrame({"sample_index": np.arange(len(labels)),
                          "filename": [Path(path).name for path in paths], "path": [str(p) for p in paths],
                          "label": labels, "prediction": predictions, "confidence": confidence,
                          "correct": np.asarray(labels) == predictions})
    frame.to_csv(Path(output_dir) / f"{split}_predictions.csv", index=False, encoding="utf-8-sig")


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_figure(fig, stem, cfg):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        for extension in cfg["formats"]:
            fig.savefig(stem.with_suffix("." + extension), dpi=cfg["dpi"], bbox_inches="tight", facecolor="white")
    finally:
        _plotting().close(fig)


def _curve(ax, frame, metric, title):
    colors = {"train": "#36759A", "val": "#D19A60"}
    for split, label in (("train", "Training"), ("val", "Validation")):
        key = f"{split}_{metric}"
        if key in frame:
            valid = frame[key].notna()
            if valid.any():
                ax.plot(frame.loc[valid, "epoch"], frame.loc[valid, key], color=colors[split],
                        linewidth=1.8, marker="o" if len(frame) < 12 else None, markersize=3, label=label)
    ax.set(xlabel="Epoch", ylabel=title, ylim=(0, 1.02), title=title + " by epoch")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=.2)
    ax.legend(frameon=False)


def save_training_figures(output_dir, history, cfg, *, has_val, show_progress=True):
    """Figures describe recorded epochs and evaluation, with no inferred uncertainty."""
    plt = _plotting()
    style = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
             "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42}
    root = Path(output_dir)
    with plt.rc_context(style), Progress("Save training curves", unit="steps", enabled=show_progress):
        frame = pd.DataFrame(history)
        for metric, title, name in (("accuracy", "Accuracy", "accuracy_curve"), ("macro_f1", "Macro F1", "macro_f1_curve")):
            fig, ax = plt.subplots(figsize=(7, 4))
            _curve(ax, frame, metric, title)
            _save_figure(fig, root / "figures" / name, cfg)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        _curve(axes[0], frame, "accuracy", "Accuracy")
        _curve(axes[1], frame, "macro_f1", "Macro F1")
        _save_figure(fig, root / "figures/training_curves", cfg)
    if not has_val:
        print("Validation disabled: saved training curves; skipped validation analysis figures and t-SNE.", flush=True)
        return
    with plt.rc_context(style), Progress("Save validation analysis figures", unit="steps", enabled=show_progress):
        report = json.loads((root / "val_classification_report.json").read_text(encoding="utf-8"))
        classes = sorted((int(k) for k in report if k.isdecimal()))
        fig, axes = plt.subplots(2, 1, figsize=(max(8, min(20, len(classes) * .17)), 6), layout="constrained", sharex=True)
        for ax, key, title in zip(axes, ("class_accuracy", "f1-score"), ("Per-class accuracy (recall)", "Per-class F1")):
            values = [np.nan if report[str(k)].get(key) is None else report[str(k)][key] for k in classes]
            ax.bar(np.arange(len(classes)), values, color="#729CBC", width=.75)
            ax.set(ylabel=title, ylim=(0, 1.03))
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        axes[-1].set_xticks(np.arange(len(classes)), [str(k) for k in classes], rotation=90 if len(classes) > 20 else 0,
                           fontsize=6 if len(classes) > 50 else 9)
        axes[-1].set_xlabel("True class")
        fig.suptitle("Validation per-class performance")
        _save_figure(fig, root / "figures/val_per_class_metrics", cfg)
        matrix = pd.read_csv(root / "val_confusion_matrix_normalized.csv", index_col=0).to_numpy()
        fig, ax = plt.subplots(figsize=(max(6, min(11, len(classes) * .28)), max(5, min(10, len(classes) * .25))))
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        stride = max(1, int(np.ceil(len(classes) / 25)))
        ticks = np.arange(0, len(classes), stride)
        ax.set_xticks(ticks, [str(classes[i]) for i in ticks], rotation=90)
        ax.set_yticks(ticks, [str(classes[i]) for i in ticks])
        ax.set(xlabel="Predicted class", ylabel="True class", title="Validation confusion matrix (row-normalized)")
        if len(classes) <= 20:
            for i, j in np.ndindex(matrix.shape):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if matrix[i, j] > .5 else "#333333")
        fig.colorbar(image, ax=ax, label="Fraction within true class", shrink=.82)
        _save_figure(fig, root / "figures/val_confusion_matrix", cfg)


def save_tsne(output_dir, embedding, indices, labels, predictions, paths, cfg, *, show_progress=True):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits
    root, options = Path(output_dir), cfg["tsne"]
    folder = root / "tsne"
    folder.mkdir(parents=True, exist_ok=True)
    total_samples = len(labels)
    labels, predictions = np.asarray(labels)[indices], np.asarray(predictions)[indices]
    paths = np.asarray([str(path) for path in paths])[indices]
    np.savez_compressed(folder / "val_embeddings.npz", embeddings=embedding, sample_indices=indices,
                        labels=labels, predictions=predictions, paths=paths)
    settings = {**options, "num_samples": len(indices), "total_validation_samples": total_samples,
                "sampling": "all validation sequences" if len(indices) == total_samples else "class-balanced without replacement",
                "feature": "stream_embedding before classification", "source_split": "val",
                "sample_indices": indices.tolist(), "embedding_dim": embedding.shape[1]}
    if len(indices) < 3:
        settings.update(status="skipped", reason="t-SNE requires at least three validation samples")
        (folder / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        print(settings["reason"], flush=True)
        return
    if not np.isfinite(embedding).all():
        raise ValueError("Non-finite validation embeddings")
    with Progress("Compute validation t-SNE", unit="steps", enabled=show_progress), threadpool_limits(limits=1):
        components = min(50, embedding.shape[1], len(embedding) - 1)
        reduced = PCA(n_components=components, svd_solver="full").fit_transform(embedding) if embedding.shape[1] > components else embedding
        perplexity = min(float(options["perplexity"]), float(len(indices) - 1))
        init = "pca" if reduced.shape[1] >= 2 and float(np.var(reduced)) > 1e-12 else "random"
        iteration_key = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
        coordinates = TSNE(n_components=2, perplexity=perplexity, random_state=options["seed"], init=init,
                           learning_rate="auto", n_jobs=1, **{iteration_key: options["max_iter"]}).fit_transform(reduced)
        settings.update(status="completed", effective_perplexity=perplexity, pca_components=reduced.shape[1], init=init)
        pd.DataFrame({"sample_index": indices, "path": paths, "label": labels, "prediction": predictions,
                      "correct": labels == predictions, "tsne_x": coordinates[:, 0], "tsne_y": coordinates[:, 1]}).to_csv(
                          folder / "val_tsne.csv", index=False, encoding="utf-8-sig")
        (folder / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    plt = _plotting()
    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42}):
        classes = np.unique(labels)
        colors = plt.get_cmap("tab20")
        markers = ("o", "s", "^", "v", "D", "P", "X")
        fig, ax = plt.subplots(figsize=(14, 8) if len(classes) > 30 else (9, 6))
        for i, label in enumerate(classes):
            mask = labels == label
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=16, color=colors(i % 20),
                       marker=markers[(i // 20) % len(markers)], alpha=.8, linewidths=0, label=str(label))
        ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title=f"Validation feature t-SNE | {len(indices):,} sequences")
        ax.legend(title="True class", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,
                  ncol=max(1, int(np.ceil(len(classes) / 20))), fontsize=7, columnspacing=.8, handletextpad=.3)
        fig.text(.12, -.015, "Colors/markers identify true classes. A t-SNE view is descriptive, not a classification metric.", fontsize=9)
        _save_figure(fig, root / "figures/val_tsne", cfg)
