"""通过 data.yaml 抽样分析原始 CF/PW/PRI 序列，输出图片、统计表和 HTML 汇总。"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import defaultdict
from numbers import Integral
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np

from pulsecls.cfg import load_yaml, remap_repo_path, validate_num_classes
from pulsecls.data.dataset import DEFAULT_FEATURES, read_sequence_csv, validate_label_range
from pulsecls.data.file_index import load_inventory, record_at
from pulsecls.utils.progress import Progress

FEATURE_LABELS = {"CF_MHz": "CF (MHz)", "PW_us": "PW (us)", "PRI_us": "PRI (us)"}
COLORS = ("#4477AA", "#EE9944", "#228833", "#CC6677", "#AA4499", "#66AABB", "#888888", "#BB9955")


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _sample_records(inventory, class_ids, per_class, maximum, rng):
    """Choose file identifiers before reading any sequence content."""
    groups = defaultdict(list)
    for group in inventory:
        if group["count"]:
            groups[group["label"]].append(group)
    if class_ids is None:
        labels = sorted(groups)
    else:
        requested = list(class_ids)
        if not requested or any(isinstance(x, bool) or not isinstance(x, Integral) for x in requested):
            raise ValueError("class_ids must be a nonempty list of integer labels, or None")
        labels = sorted(set(int(x) for x in requested))
        missing = set(labels) - set(groups)
        if missing:
            raise ValueError(f"Classes have no CSV sequences: {sorted(missing)}")
    if maximum < len(labels):
        raise ValueError("max_sequences must cover every selected class; increase it or select fewer class_ids")
    targets = np.array([min(per_class, sum(g["count"] for g in groups[label])) for label in labels])
    budget = min(maximum, int(targets.sum()))
    quotas = np.ones(len(labels), dtype=np.int64)
    remaining = budget - len(labels)
    while remaining:
        for i in rng.permutation(np.flatnonzero(quotas < targets)):
            quotas[i] += 1
            remaining -= 1
            if not remaining:
                break
    records = []
    for label, quota in zip(labels, quotas):
        capacity = sum(g["count"] for g in groups[label])
        for index in rng.choice(capacity, size=int(quota), replace=False):
            records.append(record_at(groups[label], int(index)))
    return records, labels


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig, relative_path, output, formats, dpi, show, plt):
    path = output / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(path.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight", facecolor="white")
    if not show:
        plt.close(fig)
    return path.with_suffix(".png").relative_to(output).as_posix()


def _plots(output, rows, counts, points, features, examples, plotted_labels, bins, dpi, formats, show, progress):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
             "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
             "axes.linewidth": .8, "legend.frameon": False, "svg.fonttype": "none",
             "pdf.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white"}
    gallery = []
    colors = {label: COLORS[i % len(COLORS)] for i, label in enumerate(plotted_labels)}
    with plt.rc_context(style), Progress("Save dataset figures", 4 + len(plotted_labels), unit="fig", enabled=progress) as bar:
        def save(fig, name, caption):
            image = _save_figure(fig, name, output, formats, dpi, show, plt)
            gallery.append({"image": image, "caption": caption})
            bar.update()

        fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), layout="constrained")
        labels = [r["label"] for r in counts]
        x = np.arange(len(labels))
        axes[0].bar(x, [r["total_sequences"] for r in counts], color=COLORS[0], width=.75)
        ticks = np.unique(np.linspace(0, len(labels) - 1, min(20, len(labels)), dtype=int))
        axes[0].set_xticks(ticks, [str(labels[i]) for i in ticks])
        axes[0].set(xlabel="Emitter class ID", ylabel="Sequences in total pool",
                    title=f"Class counts from dataset inventory (total n={sum(r['total_sequences'] for r in counts):,})")
        axes[0].grid(axis="y", alpha=.2)
        axes[0].set_axisbelow(True)
        axes[1].hist([r["num_pulses"] for r in rows], bins=bins, color=COLORS[0], edgecolor="white")
        axes[1].set(xlabel="Original pulses per sequence", ylabel="Sampled sequences",
                    title=f"Original sequence lengths (sampled n={len(rows):,})")
        save(fig, "dataset_overview", "上图：完整数据池的类别数量。下图：已抽取序列的原始长度分布，未经补长或裁剪。")

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for i, (feature, ax) in enumerate(zip(features, axes)):
            ax.hist(points[:, i + 3], bins=bins, color=COLORS[0], edgecolor="white")
            ax.set(xlabel=FEATURE_LABELS[feature], ylabel="Sampled pulses", title=feature)
        fig.suptitle(f"Raw pulse feature distributions (sampled pulses n={len(points):,})", fontsize=13)
        save(fig, "feature_distributions", "抽样脉冲的三通道原始数值分布，保留物理单位；序列内和全局点数上限会影响采样权重。")

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout="constrained")
        for feature, ax in zip(features, axes):
            values = [[r[f"{feature}_mean"] for r in rows if r["label"] == label] for label in plotted_labels]
            boxes = ax.boxplot(values, positions=np.arange(len(plotted_labels)), widths=.55,
                               showfliers=False, patch_artist=True)
            for label, box in zip(plotted_labels, boxes["boxes"]):
                box.set_facecolor(colors[label])
                box.set_alpha(.55)
            for position, (label, value) in enumerate(zip(plotted_labels, values)):
                offsets = np.linspace(-.13, .13, len(value)) if len(value) > 1 else [0.]
                ax.scatter(position + np.asarray(offsets), value, s=12, color=colors[label], alpha=.75, zorder=3)
            ax.set_xticks(np.arange(len(plotted_labels)), [str(label) for label in plotted_labels])
            ax.set(xlabel="Emitter class ID", ylabel=f"Sequence mean: {FEATURE_LABELS[feature]}")
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("Class comparison: full-sequence feature means", fontsize=13)
        save(fig, "class_feature_means", "展示指定范围内的类别。每个点是一条完整原始序列的均值；箱体为中位数与四分位范围，须为 1.5 IQR。")

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
        for ax, (left, right) in zip(axes, ((features[0], features[1]), (features[0], features[2]), (features[1], features[2]))):
            for label in plotted_labels:
                selected = [r for r in rows if r["label"] == label]
                ax.scatter([r[f"{left}_mean"] for r in selected], [r[f"{right}_mean"] for r in selected],
                           s=18, alpha=.75, color=colors[label], label=f"Class {label}")
            ax.set(xlabel=f"Mean {FEATURE_LABELS[left]}", ylabel=f"Mean {FEATURE_LABELS[right]}")
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="outside lower center", ncol=min(4, len(plotted_labels)), fontsize=9)
        fig.suptitle("Between-class overlap in sequence means", fontsize=13)
        save(fig, "feature_relationships", "每个点对应一条序列的两个通道均值，用于观察类别参数重叠；均值分离不代表模型分类结论。")

        for label in plotted_labels:
            fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True, layout="constrained")
            for number, (path, raw) in enumerate(examples[label]):
                positions = np.arange(1, len(raw) + 1)
                name = f"{path} (n={len(raw)})"
                for i, ax in enumerate(axes):
                    ax.plot(positions, raw[:, i], linewidth=.85, alpha=.85,
                            color=COLORS[number % len(COLORS)], label=name)
                    ax.set_ylabel(FEATURE_LABELS[features[i]])
                    ax.grid(alpha=.18)
            handles, names = axes[0].get_legend_handles_labels()
            fig.legend(handles, names, loc="outside lower center",
                       ncol=min(2, len(names)), fontsize=8)
            axes[-1].set_xlabel("Pulse index (original order)")
            fig.suptitle(f"Class {label}: raw CF / PW / PRI sequences", fontsize=13)
            save(fig, f"sequences/class_{label:03d}", "同一辐射源的多条原始序列，横轴为脉冲序号，纵轴为物理单位；每条曲线保持原长度。")
        if show:
            plt.show()
    return gallery


def _write_report(output, summary, gallery):
    cards = "\n".join(f'<section><a href="{item["image"]}"><img src="{item["image"]}"></a>'
                        f'<p>{html.escape(item["caption"])}</p></section>' for item in gallery)
    pulse_link = ' · <a href="sampled_pulses.csv">绘图脉冲数据</a>' if summary["saved_sampled_pulses"] else ""
    output.joinpath("index.html").write_text(f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Radar dataset visualization</title>
<style>body{{font-family:Arial,sans-serif;max-width:1280px;margin:32px auto;padding:0 20px;color:#243044;background:#fafbfd}}
section{{background:white;padding:16px;margin:24px 0;border:1px solid #dde3ec;border-radius:8px}}
img{{width:100%;height:auto}}p{{line-height:1.7}}a{{color:#315f96}}code{{overflow-wrap:anywhere}}</style>
<h1>数据集可视化：{html.escape(summary['dataset_name'])}</h1>
<p>数据描述：<code>{html.escape(summary['data_yaml'])}</code><br>
完整总池：{summary['total_sequences']:,} 条，实际分析：{summary['sampled_sequences']:,} 条；
原始脉冲：{summary['sampled_sequence_pulses']:,} 个，分布图抽样点：{summary['plotted_pulses']:,} 个。</p>
<p>抽样类别：{html.escape(str(summary['sampled_class_ids']))}。类别对比和序列图展示：{html.escape(str(summary['plotted_class_ids']))}。</p>
<p>类别内均匀随机选择序列，按每类上限和总数上限分配。
统计使用完整原始序列，未归一化、补零、裁剪或增强。脉冲分布先按每条序列抽点，再限制总点数，不能当作全量脉冲分布。
直方图反映样本分布；散点图仅比较序列均值，不是分类性能评估。</p>
<p><a href="class_counts.csv">类别数量</a> · <a href="sequence_stats.csv">逐序列统计</a> ·
<a href="analysis_summary.json">分析设置与来源</a>{pulse_link}</p>
{cards}</html>''', encoding="utf-8")


def visualize_dataset(data, output_dir=None, *, class_ids=None, sequences_per_class=20,
                      max_sequences=2000, plot_classes=8, examples_per_class=2,
                      pulses_per_sequence=256, max_plot_points=100000, seed=2026,
                      bins=50, dpi=300, formats=("png",), show=False, progress=True,
                      save_sampled_pulses=True):
    """Analyze selected raw CSVs; sample identifiers before loading their content."""
    for name, value in (("sequences_per_class", sequences_per_class), ("max_sequences", max_sequences),
                        ("plot_classes", plot_classes), ("examples_per_class", examples_per_class),
                        ("pulses_per_sequence", pulses_per_sequence), ("max_plot_points", max_plot_points),
                        ("bins", bins), ("dpi", dpi)):
        _positive_integer(value, name)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    formats = tuple(dict.fromkeys(("png", *formats)))  # 汇总页面始终有 PNG 预览。
    if set(formats) - {"png", "svg", "pdf"}:
        raise ValueError("formats supports png, svg and pdf")
    data_path = remap_repo_path(data)
    cfg = load_yaml(data_path)
    features = cfg.get("features", list(DEFAULT_FEATURES))
    if not isinstance(features, list) or len(features) != 3 or set(features) != set(DEFAULT_FEATURES):
        raise ValueError("Visualization expects features CF_MHz, PW_us and PRI_us in data.yaml")
    base = data_path.parent if cfg.get("root_relative_to_yaml", False) else None
    root = remap_repo_path(cfg.get("root", "."), base)
    output = remap_repo_path(output_dir) if output_dir is not None else PROJECT_ROOT / "results/dataset_visualization" / data_path.parent.name
    output.mkdir(parents=True, exist_ok=True)
    print(f"Data YAML: {data_path}\nData root: {root}\nOutput: {output}", flush=True)
    with Progress("Read dataset inventory", unit="steps", enabled=progress):
        inventory, fingerprint, inventory_source = load_inventory(root, cfg)
    all_labels = sorted({g["label"] for g in inventory if g["count"]})
    declared_classes = validate_num_classes(cfg.get("num_classes"), "data.num_classes")
    validate_label_range(np.asarray(all_labels), declared_classes, str(data_path))
    rng = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(0,)))
    pulse_rng = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(1,)))
    records, selected_labels = _sample_records(inventory, class_ids, sequences_per_class, max_sequences, rng)
    plotted_labels = selected_labels[:plot_classes]
    print(f"Visualization sampling: pool={sum(g['count'] for g in inventory):,}, selected={len(records):,}, classes={len(selected_labels)}", flush=True)
    rows, chunks, examples = [], [], defaultdict(list)
    with Progress("Load visualization CSVs", len(records), enabled=progress) as bar:
        for sequence_id, (path, label) in enumerate(records):
            raw = read_sequence_csv(path, features)
            relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
            row = {"sequence_id": sequence_id, "label": label, "path": relative, "num_pulses": len(raw)}
            for i, feature in enumerate(features):
                column = raw[:, i]
                row.update({f"{feature}_mean": float(column.mean()), f"{feature}_std": float(column.std(ddof=0)),
                            f"{feature}_min": float(column.min()), f"{feature}_max": float(column.max()),
                            f"{feature}_nonpositive": int((column <= 0).sum())})
            rows.append(row)
            if label in plotted_labels and len(examples[label]) < examples_per_class:
                examples[label].append((relative, raw))
            indices = np.sort(pulse_rng.choice(len(raw), size=min(len(raw), pulses_per_sequence), replace=False))
            chunks.append(np.column_stack((np.full(len(indices), sequence_id), indices + 1,
                                           np.full(len(indices), label), raw[indices])))
            bar.update()
    points = np.concatenate(chunks)
    if len(points) > max_plot_points:
        points = points[np.sort(pulse_rng.choice(len(points), size=max_plot_points, replace=False))]
    sampled_counts = defaultdict(int)
    for row in rows:
        sampled_counts[row["label"]] += 1
    totals = defaultdict(int)
    for group in inventory:
        totals[group["label"]] += group["count"]
    counts = [{"label": label, "total_sequences": totals[label], "sampled_sequences": sampled_counts[label],
               "selected_for_analysis": label in selected_labels} for label in all_labels]
    with Progress("Write visualization statistics", unit="steps", enabled=progress):
        _write_csv(output / "class_counts.csv", counts)
        _write_csv(output / "sequence_stats.csv", rows)
        if save_sampled_pulses:
            with (output / "sampled_pulses.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sequence_id", "pulse_index", "label", *features])
                writer.writerows([int(row[0]), int(row[1]), int(row[2]), *row[3:].tolist()] for row in points)
    summary = {
        "dataset_name": str(cfg.get("name", data_path.parent.name)), "data_yaml": str(data_path),
        "data_root": str(root), "features": features, "declared_num_classes": declared_classes,
        "available_class_ids": all_labels, "sampled_class_ids": selected_labels, "plotted_class_ids": plotted_labels,
        "missing_declared_class_ids": sorted(set(range(declared_classes)) - set(all_labels)) if declared_classes is not None else [],
        "total_sequences": int(sum(totals.values())), "sampled_sequences": len(rows),
        "sampled_sequence_pulses": int(sum(r["num_pulses"] for r in rows)), "plotted_pulses": len(points),
        "seed": int(seed), "sequences_per_class": sequences_per_class, "max_sequences": max_sequences,
        "examples_per_class": examples_per_class, "pulses_per_sequence": pulses_per_sequence,
        "max_plot_points": max_plot_points, "bins": bins, "dpi": dpi, "formats": list(formats),
        "inventory_source": inventory_source, "inventory_fingerprint": fingerprint,
        "statistics": "Full raw selected sequences; std uses ddof=0; no resizing, normalization or augmentation",
        "pulse_sampling": "Uniform within each selected sequence up to pulses_per_sequence, then global random cap",
        "saved_sampled_pulses": bool(save_sampled_pulses),
    }
    gallery = _plots(output, rows, counts, points, features, examples, plotted_labels,
                     bins, dpi, formats, show, progress)
    summary["figures"] = gallery
    (output / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(output, summary, gallery)
    print(f"Visualization finished: {output / 'index.html'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="指定 data.yaml，抽样可视化原始 CF/PW/PRI 数据")
    parser.add_argument("--data", help="训练总池的 data.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--classes", type=int, nargs="+", help="只分析指定类别，例如 --classes 0 1 2")
    parser.add_argument("--sequences-per-class", type=int)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--plot-classes", type=int)
    parser.add_argument("--examples-per-class", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--show", action="store_true", help="本机显示图窗；服务器默认只保存文件")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    # 常用设置直接改这里；只需要替换 data.yaml 即可分析另一个同格式数据集。
    return visualize_dataset(
        data=args.data or PROJECT_ROOT / "datasets/synthetic_radar_overlap/data.yaml",
        output_dir=args.output_dir or None,  # None：results/dataset_visualization/<data.yaml所在目录名>；也可填路径。
        class_ids=args.classes or None,  # None 分析所有类别；例如 [0, 1, 2]。
        sequences_per_class=args.sequences_per_class if args.sequences_per_class is not None else 20,
        max_sequences=args.max_sequences if args.max_sequences is not None else 2000,
        plot_classes=args.plot_classes if args.plot_classes is not None else 8,  # 类别对比和序列图最多显示几类。
        examples_per_class=args.examples_per_class if args.examples_per_class is not None else 2,
        pulses_per_sequence=256,  # 每条序列随机抽取的分布图脉冲上限；统计与序列图仍使用完整原始序列。
        max_plot_points=100000,  # 分布图及 sampled_pulses.csv 的总点数上限。
        seed=args.seed if args.seed is not None else 2026,
        bins=50,
        dpi=args.dpi if args.dpi is not None else 300,
        formats=("png",),  # 可改为 ("png", "svg", "pdf")。
        show=args.show,  # False 可直接在无图形界面的服务器运行。
        progress=not args.no_progress,
        save_sampled_pulses=True,
    )


if __name__ == "__main__":
    main()
