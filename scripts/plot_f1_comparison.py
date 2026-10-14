"""复用现有结果绘制 Macro F1 对比；路径在底部 main() 调用中修改。"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHOD_LABELS = {
    "cnn_softmax": "CNN + Softmax", "atenet": "ATENet", "invconv1d": "InvConv1D",
    "mambasl": "MambaSL", "modern_tcn": "ModernTCN", "mptsnet1d": "MPTSNet1D",
    "multigrain_infonce": "Multi-grain CNN", "pulse_hypothesis_fused": "Hypothesis-fused CNN",
    "modular_multigrain_iq": "Modular IQ ResNet1D", "resnet_softmax": "ResNet1D + Softmax",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def matrix_metrics(matrix):
    """每个类别等权；无真值且无预测的类别 F1 记为 0。"""
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or (matrix < 0).any() or matrix.sum() == 0:
        raise ValueError("Invalid confusion matrix")
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2.0 * matrix.diagonal(), denominator, out=np.zeros(len(matrix)), where=denominator > 0)
    return float(f1.mean()), float(matrix.trace() / matrix.sum())


def validation_results(results_dir):
    rows, audit, reference = [], [], None
    for source in sorted(Path(results_dir).glob("*/final_metrics.json")):
        metrics = read_json(source)
        if not metrics.get("validation_enabled", True) or metrics.get("evaluation_split", "val") != "val":
            continue
        cfg = read_json(source.with_name("resolved_pipeline.json"))
        selection = read_json(source.with_name("dataset_selection.json"))
        report = read_json(source.with_name("val_classification_report.json"))
        with source.with_name("val_confusion_matrix.csv").open(encoding="utf-8-sig", newline="") as f:
            table = list(csv.reader(f))
        matrix = np.array([row[1:] for row in table[1:]], dtype=np.int64)
        labels = list(range(len(matrix)))
        if [int(v) for v in table[0][1:]] != labels or [int(row[0]) for row in table[1:]] != labels:
            raise ValueError(f"Unexpected class numbering: {source}")
        macro_f1, accuracy = matrix_metrics(matrix)
        if any(not np.isclose(macro_f1, value, rtol=0, atol=1e-10)
               for value in (metrics["macro_f1"], report["macro avg"]["f1-score"])):
            raise ValueError(f"Macro F1 disagrees with saved confusion matrix: {source}")
        if not np.isclose(accuracy, metrics["accuracy"], rtol=0, atol=1e-10) or matrix.sum() != selection["val_count"]:
            raise ValueError(f"Accuracy/sample count mismatch: {source}")
        signature = {
            "source_fingerprint": selection["source_fingerprint"],
            "train_records_hash": digest(selection["train"]), "val_records_hash": digest(selection["val"]),
            "features": cfg["data_cfg"]["features"], "sequence_length": cfg["sequence_length"],
            "length_policy": cfg["length_policy"], "num_classes": len(matrix),
        }
        if reference is not None and signature != reference:
            raise ValueError(f"Different dataset/split/features; compare this run separately: {source.parent}")
        reference = signature
        method = source.parent.name.split("_synthetic", 1)[0]
        train, augment = cfg["train"], cfg.get("augment", {})
        rows.append({
            "split": "validation", "method": METHOD_LABELS.get(method, method), "method_key": method,
            "macro_f1": macro_f1, "macro_f1_percent": macro_f1 * 100, "accuracy": accuracy,
            "num_sequences": int(matrix.sum()), "num_classes": len(matrix), "best_epoch": metrics.get("best_epoch"),
            "train_sequences": selection["train_count"], "seed": train.get("seed"),
            "batch_size": train.get("batch_size"), "lr": train.get("lr"), "sampler": train.get("sampler"),
            "early_stopping_patience": train.get("early_stopping_patience"),
            "augmentation_enabled": bool(augment.get("enabled")),
            "enabled_augmentation_methods": ";".join(k for k, v in augment.get("methods", {}).items() if v.get("enabled")),
            "source": relative(source), "labels_source": "saved validation confusion matrix",
        })
        audit.append({"experiment": source.parent.name, **signature})
    if not rows:
        raise FileNotFoundError(f"No validation results in {results_dir}")
    return rows, {"identical_splits_and_inputs": True, "common": reference, "runs": audit}


def test_results(predictions_csv, labels_file, num_classes):
    """只读取预测记录及标签，不重新读取脉冲数据或执行模型。"""
    predictions_csv, labels_file = Path(predictions_csv), Path(labels_file)
    with predictions_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        records = list(reader)
        columns = [k for k in reader.fieldnames or [] if k.endswith("_prediction") and k != "main_prediction"]
    if not records or not columns or "prediction" not in records[0]:
        raise ValueError("Voting CSV needs member and final predictions")
    names, labels = [r["filename"] for r in records], {}
    if len(set(names)) != len(names):
        raise ValueError("Duplicate prediction filenames")
    for line in labels_file.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, value = line.rsplit(None, 1)
        if name in labels or Path(name).name != name:
            raise ValueError(f"Duplicate/invalid label filename: {name}")
        labels[name] = int(value)
    if set(labels) != set(names):
        raise ValueError("Labels must match prediction filenames exactly")
    actual = np.array([labels[name] for name in names])
    if (actual < 0).any() or (actual >= num_classes).any() or len(set(actual)) != num_classes:
        raise ValueError("Test labels must cover all training classes and remain within range")
    if any(Path(r["path"]).resolve().parent != labels_file.resolve().parent for r in records):
        raise ValueError("Label folder differs from saved prediction input folder")
    rows = []
    for column in [*columns, "prediction"]:
        predicted = np.array([int(r[column]) for r in records])
        if (predicted < 0).any() or (predicted >= num_classes).any():
            raise ValueError(f"Out-of-range prediction: {column}")
        matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        np.add.at(matrix, (actual, predicted), 1)
        macro_f1, accuracy = matrix_metrics(matrix)
        method = column.removesuffix("_prediction") if column != "prediction" else "vote"
        rows.append({
            "split": "test", "method": f"{len(columns)} 模型投票" if method == "vote" else METHOD_LABELS.get(method, method),
            "method_key": method, "macro_f1": macro_f1, "macro_f1_percent": macro_f1 * 100,
            "accuracy": accuracy, "num_sequences": len(records), "num_classes": num_classes,
            "source": relative(predictions_csv), "labels_source": relative(labels_file),
        })
    return rows, {
        "predictions_sha256": hashlib.sha256(predictions_csv.read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(labels_file.read_bytes()).hexdigest(),
        "filenames_match_exactly": True, "all_classes_present": True, "member_columns": columns,
        "num_sequences": len(records),
        "note": "Metrics use current labels.txt; keep predictions and labels from the same dataset generation.",
    }


def draw_panel(ax, rows, title, subtitle):
    rows = sorted(rows, key=lambda r: r["macro_f1"], reverse=True)
    colors = ["#D4A16C" if r["method_key"] == "vote" else "#36759A" if i == 0 else "#A2B8CD" for i, r in enumerate(rows)]
    bars = ax.barh(np.arange(len(rows)), [r["macro_f1_percent"] for r in rows], height=0.61, color=colors)
    ax.bar_label(bars, labels=[f"{r['macro_f1_percent']:.2f}%" for r in rows], padding=6, fontsize=11)
    ax.set_yticks(np.arange(len(rows)), [r["method"] for r in rows], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Macro F1（%）", labelpad=10)
    ax.set_title(title + "\n" + subtitle, loc="left", fontsize=13, pad=17, linespacing=1.6)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E6EAF0", linewidth=0.7)
    ax.spines["left"].set_visible(False)


def main(*, results_dir, predictions_csv, labels_file, output_dir):
    # 证据：相同验证划分的全部方法，以及相同测试记录中的成员与投票。
    # 每条柱形是一项已保存实验，不添加没有重复实验支持的误差条。
    val, val_audit = validation_results(results_dir)
    test, test_audit = test_results(predictions_csv, labels_file, val_audit["common"]["num_classes"])
    output_dir, rows = Path(output_dir), val + test
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "f1_source_data.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "comparison_audit.json").write_text(json.dumps({"validation": val_audit, "test": test_audit}, ensure_ascii=False, indent=2), encoding="utf-8")
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((n for n in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if n in available), None)
    if font is None:
        raise RuntimeError("Install a Chinese font: Microsoft YaHei, SimHei, or Noto Sans CJK SC")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [font, "DejaVu Sans"], "font.size": 11,
        "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 8.1))
    classes = val_audit["common"]["num_classes"]
    draw_panel(axes[0], val, "a  验证集：全部已完成方法", f"{len(val)} 种方法 · {classes} 类 · {val[0]['num_sequences']:,} 条序列")
    draw_panel(axes[1], test, "b  测试集：单模型与投票", f"{len(test)-1} 个成员 + 投票 · {test[0]['num_sequences']:,} 条序列")
    fig.suptitle("合成雷达实验结果对比", fontsize=19, y=0.975)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.80, bottom=0.27, wspace=0.74)
    fig.text(0.055, 0.045,
             f"验证集已核对相同的采样与划分：训练 {val[0]['train_sequences']:,} 条；输入 CF / PW / PRI，长度 {val_audit['common']['sequence_length']}。\n"
             "测试集指标由保存的各成员预测、最终投票及 labels.txt 重新计算；两个面板分别排序，横轴均从 0 开始。\n"
             "每种方法为一次实验记录；采样器、损失与早停等训练设置有差异，无重复实验误差条。",
             fontsize=10, color="#5B6572", linespacing=1.7)
    stem = output_dir / "macro_f1_comparison"
    for extension in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix("." + extension), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    best_val, best_test = max(val, key=lambda r: r["macro_f1"]), max(test, key=lambda r: r["macro_f1"])
    vote = next(r for r in test if r["method_key"] == "vote")
    notes = f"""# 实验结果图说明

本图描述现有单次实验的 Macro F1，左侧为验证集，右侧为测试集，分别排序。

- 验证集最高：{best_val['method']}，{best_val['macro_f1_percent']:.2f}%。
- 测试集最高：{best_test['method']}，{best_test['macro_f1_percent']:.2f}%；投票为 {vote['macro_f1_percent']:.2f}%。
- 验证集指标从混淆矩阵重算，并与 final_metrics.json 和分类报告交叉核对。
- 所有验证实验的原数据指纹、训练/验证文件及标签哈希、输入特征、长度、补长和裁剪策略均一致。
- 测试集来源：{relative(predictions_csv)}；标签：{relative(labels_file)}。逐文件名匹配，全部类别均存在，没有重新训练或推理。
- 预测记录未保存原生成批次指纹，使用当前 labels.txt 计算。请保留同一生成批次的预测和标签；来源文件哈希见 comparison_audit.json。
- 对 {classes} 个类别的 F1 等权平均，未定义的类别 F1 记为 0。每条柱形是一次实验记录，没有多种子误差条或显著性结论。
- 验证集用于选择权重，测试集用于本次已保存预测的评估，不跨面板比较排名。
- 采样器、损失、视图及早停等训练设置有差异，不将模型间差异归因于单个网络模块。
- 输出 300 dpi 白底 PNG、保留文本的 SVG/PDF、f1_source_data.csv 和 comparison_audit.json。

重新画图：`python scripts/plot_f1_comparison.py`；在脚本底部 main() 调用中修改路径。
"""
    (output_dir / "figure_notes.md").write_text(notes, encoding="utf-8")
    print(f"Validation: {len(val)} methods; identical splits verified")
    print(f"Best validation Macro F1: {best_val['method']} = {best_val['macro_f1_percent']:.2f}%")
    print(f"Best test Macro F1: {best_test['method']} = {best_test['macro_f1_percent']:.2f}%")
    print(f"Voting test Macro F1: {vote['macro_f1_percent']:.2f}%")
    print(f"Saved figure: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main(
        results_dir=ROOT / "results",
        predictions_csv=ROOT / "results/infer_vote/predictions.csv",
        labels_file=ROOT / "datasets/synthetic_radar_test/labels.txt",
        output_dir=ROOT / "results/figures/f1_comparison_synthetic",
    )
