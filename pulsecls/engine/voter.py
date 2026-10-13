"""Plurality voting for checkpoints with the same emitter label semantics."""
from pathlib import Path

import numpy as np

from pulsecls.cfg import deep_update, get_cfg, load_yaml, remap_repo_path
from pulsecls.data.inference import find_inference_csvs, read_labels
from pulsecls.engine.predictor import BasePredictor, save_prediction_outputs
from pulsecls.utils.progress import Progress


def vote_predictions(predictions, main_index, num_classes):
    predictions = np.asarray(predictions, dtype=np.int64)
    if predictions.ndim != 2 or not predictions.shape[0] or not predictions.shape[1]:
        raise ValueError("Voting requires [models, sequences] predictions")
    if not 0 <= main_index < predictions.shape[0] or num_classes < 1:
        raise ValueError("Invalid main model index or num_classes")
    if (predictions < 0).any() or (predictions >= num_classes).any():
        raise ValueError("Predicted label outside num_classes")
    counts = np.zeros((predictions.shape[1], num_classes), dtype=np.int64)
    for row in predictions:
        np.add.at(counts, (np.arange(predictions.shape[1]), row), 1)
    maximum = counts.max(axis=1)
    winners = counts == maximum[:, None]
    tied = winners.sum(axis=1) > 1
    chosen = counts.argmax(axis=1)  # Deterministic smallest label if main isn't tied for first.
    main = predictions[main_index]
    prefer_main = tied & winners[np.arange(len(main)), main]
    chosen[prefer_main] = main[prefer_main]
    return chosen, maximum, tied


class VotePredictor:
    def __init__(self, config, **overrides):
        self.config = deep_update(load_yaml(config) if isinstance(config, (str, Path)) else config, overrides)

    def predict(self):
        cfg = self.config
        source = cfg.get("source")
        if not source:
            raise ValueError("Voting needs a source CSV folder")
        show_progress = bool(cfg.get("progress", True))
        paths = find_inference_csvs(source, progress=show_progress)
        members = [m for m in cfg["models"] if m.get("enabled", True)]
        names = [m["name"] for m in members]
        if not names or len(names) != len(set(names)):
            raise ValueError("Enable at least one model, with unique names")
        if cfg["main_model"] not in names:
            raise ValueError("main_model must name an enabled model")
        main_index = names.index(cfg["main_model"])
        predicted, confidences, semantics = [], [], None
        labels = None
        for number, member in enumerate(members, 1):
            print(f"Voting model {number}/{len(members)}: {member['name']}", flush=True)
            overrides = dict(member.get("overrides", {}))
            for key in ("sequence_length", "length_policy", "device"):
                if cfg.get(key) is not None:
                    overrides[key] = cfg[key]
            if member.get("data") is not None:
                overrides["data"] = member["data"]
            if cfg.get("batch_size") is not None:
                overrides["inference"] = deep_update(overrides.get("inference", {}), {"batch_size": cfg["batch_size"]})
            if cfg.get("augment") is not None:
                overrides["augment"] = deep_update(overrides.get("augment", {}), cfg["augment"])
            pipeline = get_cfg(remap_repo_path(member["pipeline"]), overrides=overrides)
            pipeline.setdefault("inference", {})["progress"] = show_progress
            if cfg.get("num_classes") is not None:
                pipeline["inference"]["num_classes"] = cfg["num_classes"]
            pipeline["inference"]["progress_label"] = f"Infer {number}/{len(members)} {member['name']}"
            predictor = BasePredictor(pipeline, remap_repo_path(member["checkpoint"]), use_checkpoint_data=True)
            current = (predictor.num_classes, tuple(predictor.data_cfg["features"]), predictor.data_cfg.get("class_names"))
            if semantics is None:
                semantics = current
                with Progress("Read voting labels", unit="steps", enabled=show_progress and bool(cfg.get("labels_file"))):
                    labels = read_labels(cfg.get("labels_file"), paths, predictor.num_classes)
            elif current != semantics:
                raise ValueError(f"Model {member['name']} has different features/classes; all voting models must use the same emitter labels")
            prediction, confidence = predictor.predict_paths(paths)
            predicted.append(prediction)
            confidences.append(confidence)
            del predictor  # Run models sequentially to bound accelerator memory.
        matrix = np.stack(predicted)
        with Progress("Combine model votes", unit="steps", enabled=show_progress):
            chosen, counts, tied = vote_predictions(matrix, main_index, semantics[0])
        columns = {f"{name}_prediction": row for name, row in zip(names, matrix)}
        columns.update(vote_count=counts, vote_fraction=counts / len(members), tied=tied,
                       main_prediction=matrix[main_index], main_confidence=np.stack(confidences)[main_index])
        # Vote fraction is agreement, not a calibrated probability.
        return save_prediction_outputs(remap_repo_path(cfg["output_dir"]), paths, chosen,
                                       counts / len(members), labels, semantics[0],
                                       prefix="vote", extra_columns=columns, progress=show_progress)
