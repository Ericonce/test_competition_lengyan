"""核验平铺测试 CSV 与训练序列的相同或连续包含关系，并按类别复制唯一匹配。"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import csv
import hashlib
import json
from numbers import Integral
from pathlib import Path
import shutil
import sqlite3  # Python 标准库；临时索引沿用 Python 自带 SQLite，无需新增 pip 依赖。
import sys
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from pulsecls.cfg import load_yaml, remap_repo_path, validate_num_classes
from pulsecls.data.dataset import DEFAULT_FEATURES, read_sequence_csv, validate_label_range
from pulsecls.data.file_index import load_inventory, record_at
from pulsecls.data.inference import find_inference_csvs
from pulsecls.utils.progress import Progress

# 仅决定报表中 strongest_rule 的展示顺序；跨类别冲突仍汇总所有匹配证据。
RULES = ("sha256_equal", "sequence_equal", "test_in_train", "train_in_test")
ASSIGNMENT_FIELDS = ("filename", "status", "prediction", "candidate_labels", "match_count",
                     "strongest_rule", "num_pulses", "file_sha256", "feature_sha256", "output_path")
MATCH_FIELDS = ("filename", "label", "rule", "train_path", "test_num_pulses", "train_num_pulses",
                "test_start_row", "train_start_row", "overlap_rows", "test_file_sha256",
                "train_file_sha256", "test_feature_sha256", "train_feature_sha256")


def _sha256_file(path):
    """分块计算原始文件 SHA-256，与特征序列的数值 SHA 区分。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_raw(path, features):
    """复用训练原始 CSV 读取器，保留全长、配置列顺序及 float64 精度。"""
    before = path.stat()
    values = read_sequence_csv(path, features)
    digest = _sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"CSV changed while reading: {path}; rerun with stable input files")
    # 固定小端编码并统一正负零，避免 CSV 格式差异及机器字节序影响数值匹配。
    values = np.array(values, dtype="<f8", order="C", copy=True)
    values[values == 0] = 0.0
    raw = values.tobytes()
    row_size = values.shape[1] * 8
    rows = np.unique(np.frombuffer(raw, dtype=f"V{row_size}"))
    return len(values), raw, [row.tobytes() for row in rows], digest, hashlib.sha256(raw).hexdigest()


def _find_aligned(haystack, needle, row_size):
    """只接受从完整脉冲行边界开始的连续匹配，返回从 0 开始的脉冲行号。"""
    start = haystack.find(needle)
    while start >= 0 and start % row_size:
        start = haystack.find(needle, start + 1)
    return start // row_size if start >= 0 else -1


def _relation(train, test, row_size, train_sha, test_sha, min_overlap_rows):
    """返回 (匹配规则, 训练起始行, 测试起始行)；每对文件保留首个匹配位置。"""
    if len(train) == len(test):
        if train != test:
            return None
        return ("sha256_equal" if train_sha == test_sha else "sequence_equal", 0, 0)
    if min(len(train), len(test)) // row_size < min_overlap_rows:
        return None
    if len(test) < len(train):
        offset = _find_aligned(train, test, row_size)
        return ("test_in_train", offset, 0) if offset >= 0 else None
    offset = _find_aligned(test, train, row_size)
    return ("train_in_test", 0, offset) if offset >= 0 else None


def _reference_records(root, cfg, selection_file):
    """默认复用仓库总池清单；指定 selection_file 时仅核验该次训练的 train 项。"""
    if selection_file is None:
        inventory, fingerprint, origin = load_inventory(root, cfg)
        records = [record_at([group], index) for group in inventory for index in range(group["count"])]
        scope = {"reference_scope": "data_yaml_inventory", "inventory_source": origin,
                 "inventory_fingerprint": fingerprint}
    else:
        selection_file = remap_repo_path(selection_file).resolve()
        manifest = json.loads(selection_file.read_text(encoding="utf-8"))
        entries = manifest.get("train") if isinstance(manifest, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ValueError("selection_file must be a dataset_selection.json with a nonempty train list")
        records = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or "label" not in entry:
                raise ValueError(f"Invalid training record in {selection_file}: {entry}")
            relative = Path(entry["path"])
            label = entry["label"]
            if (relative.is_absolute() or ".." in relative.parts
                    or relative.suffix.lower() != ".csv" or relative.parent.name != str(label)
                    or isinstance(label, bool) or not isinstance(label, int) or label < 0):
                raise ValueError(f"Invalid training record in {selection_file}: {entry}")
            records.append((root / relative, label))
        scope = {"reference_scope": "selected_training_only", "selection_file": str(selection_file),
                 "selection_file_sha256": _sha256_file(selection_file)}
    unique = {}
    for path, label in records:
        path = path.resolve()
        if not path.is_file() or path.suffix.lower() != ".csv":
            raise FileNotFoundError(f"Reference CSV does not exist: {path}")
        if selection_file is not None and not path.is_relative_to(root):
            raise ValueError(f"Training record escapes data root: {path}")
        if path in unique and unique[path] != label:
            raise ValueError(f"Reference CSV has conflicting labels: {path}")
        unique[path] = label
    return sorted(unique.items(), key=lambda row: (row[1], str(row[0]))), scope


def _create_index(connection):
    # 索引随本次任务清理；测试数组存磁盘，SQLite 页缓存上限约 32 MiB。
    connection.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA cache_size = -32768;
        CREATE TABLE tests (
            id INTEGER PRIMARY KEY, filename TEXT NOT NULL, num_rows INTEGER NOT NULL,
            file_sha TEXT NOT NULL, feature_sha TEXT NOT NULL,
            first_row BLOB NOT NULL, sequence BLOB NOT NULL
        );
        CREATE INDEX test_prefix ON tests(first_row);
        CREATE TABLE anchors (
            row_bytes BLOB NOT NULL, test_id INTEGER NOT NULL,
            PRIMARY KEY (row_bytes, test_id)
        ) WITHOUT ROWID;
        CREATE TABLE evidence (
            test_id INTEGER NOT NULL, train_path TEXT NOT NULL, label INTEGER NOT NULL,
            rule TEXT NOT NULL, train_rows INTEGER NOT NULL, train_start INTEGER NOT NULL,
            test_start INTEGER NOT NULL, overlap_rows INTEGER NOT NULL,
            train_sha TEXT NOT NULL, train_feature_sha TEXT NOT NULL,
            PRIMARY KEY (test_id, train_path)
        ) WITHOUT ROWID;
    """)


def _index_tests(connection, paths, features, progress):
    row_size = len(features) * 8
    with Progress("Index raw test CSVs", len(paths), unit="file", enabled=progress) as bar:
        for index, path in enumerate(paths):
            count, raw, row_keys, digest, feature_digest = _read_raw(path, features)
            connection.execute("INSERT INTO tests VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (index, path.name, count, digest, feature_digest, raw[:row_size], raw))
            connection.executemany("INSERT INTO anchors VALUES (?, ?)", ((key, index) for key in row_keys))
            if index % 100 == 0:
                connection.commit()
            bar.update()
    connection.commit()


def _candidate_ids(connection, row_keys, first_row, num_rows):
    candidates = set()
    # 测试被训练包含时，测试首行必须出现在训练中；这一步只筛候选，随后验证完整片段。
    # 每批最多 401 个 SQL 参数，兼容默认参数数量上限为 999 的 SQLite 版本。
    for start in range(0, len(row_keys), 400):
        batch = row_keys[start:start + 400]
        placeholders = ",".join("?" for _ in batch)
        query = f"SELECT id FROM tests WHERE num_rows <= ? AND first_row IN ({placeholders})"
        candidates.update(row[0] for row in connection.execute(query, (num_rows, *batch)))
    # 训练被测试包含时，训练首行必须出现在测试中；按相反方向补齐候选。
    query = """SELECT t.id FROM anchors a JOIN tests t ON t.id = a.test_id
               WHERE a.row_bytes = ? AND t.num_rows > ?"""
    candidates.update(row[0] for row in connection.execute(query, (first_row, num_rows)))
    return sorted(candidates)


def _scan_references(connection, records, features, output, min_overlap_rows, progress):
    row_size = len(features) * 8
    columns = ("train_path", "label", "num_pulses", "file_sha256", "feature_sha256")
    with (output / "reference_files.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        with Progress("Check training CSVs", len(records), unit="file", enabled=progress) as bar:
            for index, (path, label) in enumerate(records):
                count, raw, row_keys, digest, feature_digest = _read_raw(path, features)
                writer.writerow((str(path), label, count, digest, feature_digest))
                for test_id in _candidate_ids(connection, row_keys, raw[:row_size], count):
                    test_count, test_sha, test_raw = connection.execute(
                        "SELECT num_rows, file_sha, sequence FROM tests WHERE id = ?", (test_id,)).fetchone()
                    match = _relation(raw, test_raw, row_size, digest, test_sha, min_overlap_rows)
                    if match is not None:
                        rule, train_start, test_start = match
                        connection.execute("INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                           (test_id, str(path), label, rule, count, train_start, test_start,
                                            min(count, test_count), digest, feature_digest))
                if index % 100 == 0:
                    connection.commit()
                bar.update()
    connection.commit()


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _export(connection, stage, source, output, cfg, num_classes, train_counts, progress):
    """按仓库数字类别目录导出原 CSV，同时保留冲突与未匹配的完整清单。"""
    classified = stage / "classified"
    classified.mkdir()
    for label in range(num_classes):
        (classified / str(label)).mkdir()
    # 输出是新的总池；继承特征等设置，重建相对路径，清除旧训练/验证清单引用。
    data_cfg = {key: value for key, value in cfg.items()
                if key not in {"root", "root_relative_to_yaml", "train", "val", "test", "all", "selection"}}
    data_cfg.update({"name": f"{cfg.get('name', 'radar')}_overlap_matched", "root": ".",
                     "root_relative_to_yaml": True, "all": ".", "num_classes": num_classes})
    (classified / "data.yaml").write_text(yaml.safe_dump(data_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    assignments, matched_counts, conflict_counts = [], Counter(), Counter()
    total = connection.execute("SELECT COUNT(*) FROM tests").fetchone()[0]
    with (stage / "matched_labels.txt").open("w", encoding="utf-8") as labels_file:
        with Progress("Copy uniquely matched CSVs", total, unit="file", enabled=progress) as bar:
            for test_id, filename, count, digest, feature_digest in connection.execute(
                    "SELECT id, filename, num_rows, file_sha, feature_sha FROM tests ORDER BY id"):
                matches = connection.execute(
                    "SELECT label, rule, COUNT(*) FROM evidence WHERE test_id = ? GROUP BY label, rule",
                    (test_id,)).fetchall()
                labels = sorted({row[0] for row in matches})
                rules = {row[1] for row in matches}
                strongest = next((rule for rule in RULES if rule in rules), "")
                status = "matched" if len(labels) == 1 else "conflict" if labels else "unmatched"
                relative, prediction = "", ""
                if status == "matched":
                    prediction = labels[0]
                    relative = f"classified/{prediction}/{filename}"
                    destination = stage / relative
                    shutil.copy2(source / filename, destination)
                    if _sha256_file(destination) != digest:
                        raise RuntimeError(f"Test CSV changed since indexing: {source / filename}")
                    labels_file.write(f"{filename}\t{prediction}\n")
                    matched_counts[prediction] += 1
                elif status == "conflict":
                    # 相同文件 SHA 也不能消除其他类别的冲突证据。
                    conflict_counts.update(labels)
                assignments.append(dict(zip(ASSIGNMENT_FIELDS, (
                    filename, status, prediction, json.dumps(labels), sum(row[2] for row in matches),
                    strongest, count, digest, feature_digest, relative))))
                bar.update()
    _write_csv(stage / "assignments.csv", ASSIGNMENT_FIELDS, assignments)
    for status, filename in (("conflict", "conflicts.csv"), ("unmatched", "unmatched.csv")):
        _write_csv(stage / filename, ASSIGNMENT_FIELDS,
                   (row for row in assignments if row["status"] == status))
    query = """SELECT t.filename, e.label, e.rule, e.train_path, t.num_rows, e.train_rows,
                      e.test_start, e.train_start, e.overlap_rows, t.file_sha, e.train_sha,
                      t.feature_sha, e.train_feature_sha
               FROM evidence e JOIN tests t ON t.id = e.test_id
               ORDER BY e.test_id, e.label, e.train_path"""
    _write_csv(stage / "matches.csv", MATCH_FIELDS,
               (dict(zip(MATCH_FIELDS, row)) for row in connection.execute(query)))
    columns = ("label", "reference_files", "matched_test_files", "conflicting_test_files")
    _write_csv(stage / "class_counts.csv", columns, (
        dict(zip(columns, (label, train_counts[label], matched_counts[label], conflict_counts[label])))
        for label in range(num_classes)))
    status_counts = Counter(row["status"] for row in assignments)
    return {"test_files": total, "matched_files": status_counts["matched"],
            "conflict_files": status_counts["conflict"], "unmatched_files": status_counts["unmatched"],
            "matching_pairs_by_rule": dict(connection.execute("SELECT rule, COUNT(*) FROM evidence GROUP BY rule")),
            "assigned_files_by_strongest_rule": dict(Counter(row["strongest_rule"] for row in assignments if row["status"] == "matched")),
            "matched_class_counts": {str(label): matched_counts[label] for label in range(num_classes)},
            "classified_dir": str(output / "classified")}


def postcheck_overlap(data: str | Path, source: str | Path, save_dir: str | Path, *,
                      selection_file: str | Path | None = None,
                      min_overlap_rows: int = 1, progress: bool = True) -> dict:
    """核验原始序列的相同及连续包含关系，返回与 summary.json 一致的汇总。

    data: 沿用训练读取规范的 data.yaml；默认核验其完整已知类别数据池。
    source: 类别混放的测试目录，沿用 find_inference_csvs 的直接下级 CSV 扫描规则。
    save_dir: 新结果目录，唯一类别匹配复制到 classified/<label>/<filename>.csv。
    selection_file: 可选 dataset_selection.json，限制参考范围为该次训练的 train 清单。
    min_overlap_rows: 包含匹配的最短脉冲数；整条相同匹配始终保留。
    progress: 沿用仓库 Progress 进度显示。

    全部匹配参考文件的标签唯一时赋值；冲突和未匹配项保留空 prediction。
    输入 CSV 内容保持原样；训练文件索引缓存由 load_inventory 按仓库惯例维护。
    """
    if isinstance(min_overlap_rows, bool) or not isinstance(min_overlap_rows, Integral) or min_overlap_rows < 1:
        raise ValueError("min_overlap_rows must be a positive integer")
    data_path = remap_repo_path(data).resolve()
    cfg = load_yaml(data_path)
    features = cfg.get("features", list(DEFAULT_FEATURES))
    if (not isinstance(features, (list, tuple)) or not features
            or any(not isinstance(feature, str) or not feature for feature in features)
            or len(set(features)) != len(features)):
        raise ValueError("data.features must be a nonempty list of unique column names")
    cfg["features"] = list(features)
    base = data_path.parent if cfg.get("root_relative_to_yaml", False) else None
    root = remap_repo_path(cfg.get("root", "."), base).resolve()
    source, output = remap_repo_path(source).resolve(), remap_repo_path(save_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new save_dir")
    if output.is_relative_to(root) or output.is_relative_to(source):
        raise ValueError("save_dir must be outside the reference data root and test source folder")
    paths = find_inference_csvs(source, progress=progress)
    print(f"Data YAML: {data_path}\nReference root: {root}\nTest source: {source}\nOutput: {output}", flush=True)
    with Progress("Read reference inventory", unit="steps", enabled=progress):
        records, scope = _reference_records(root, cfg, selection_file)
    # 样本 JSON 的 root 可以指向 data.yaml.root 之外，继续检查实际 CSV 类别目录。
    if any(output.is_relative_to(path.parent) for path, _ in records):
        raise ValueError("save_dir must be outside every reference CSV class folder")
    labels = np.array([label for _, label in records], dtype=np.int64)
    declared = validate_num_classes(cfg.get("num_classes"), "data.num_classes")
    validate_label_range(labels, declared, str(data_path))
    num_classes = declared if declared is not None else int(labels.max()) + 1
    train_counts = Counter(label for _, label in records)
    print(f"References: {len(records):,}; test CSVs: {len(paths):,}; classes: {num_classes}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    # 同一父目录暂存并发布完整结果，保证失败时不留下半成品；Windows 清理前先关闭 SQLite。
    with TemporaryDirectory(prefix=".postcheck-", dir=output.parent) as temporary:
        scratch = Path(temporary)
        stage = scratch / "export"
        stage.mkdir()
        with closing(sqlite3.connect(scratch / "index.sqlite3")) as connection:
            _create_index(connection)
            _index_tests(connection, paths, features, progress)
            _scan_references(connection, records, features, stage, int(min_overlap_rows), progress)
            summary = _export(connection, stage, source, output, cfg, num_classes, train_counts, progress)
        summary.update({"data_yaml": str(data_path), "data_yaml_sha256": _sha256_file(data_path),
                        "reference_root": str(root), "reference_files": len(records), "source": str(source),
                        "num_classes": num_classes, "features": list(features),
                        "min_overlap_rows": int(min_overlap_rows), "save_dir": str(output), **scope,
                        "comparison": "Complete raw float64 feature rows in original order; exact contiguous containment; no resizing or normalization",
                        "row_offsets": "Zero-based data rows, excluding the CSV header; first occurrence per training/test pair",
                        "conflict_policy": "Assign only if all matching reference files agree on one class",
                        "matched_labels_scope": "matched_labels.txt covers uniquely matched files only; not a complete label file for source"})
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copy2(data_path, stage / "reference_data.yaml")
        if output.exists():
            raise FileExistsError(f"Output appeared during processing: {output}; choose a new save_dir")
        stage.rename(output)
    print(f"Postcheck finished: matched={summary['matched_files']}, conflicts={summary['conflict_files']}, "
          f"unmatched={summary['unmatched_files']}\nResults: {output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="训练总池的 data.yaml；沿用 root/all/features 和数字类别目录")
    parser.add_argument("--source", help="各类别 CSV 混放的测试文件夹；读取其直接下级 CSV")
    parser.add_argument("--save-dir", help="新的输出目录；匹配文件复制到 classified/<类别编号>/")
    parser.add_argument("--selection-file", help="可选：已有训练结果的 dataset_selection.json，仅核验 train 清单")
    parser.add_argument("--min-overlap-rows", type=int, help="包含匹配所需的最少连续脉冲数；相同序列不受此限制")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    # 常用路径和参数直接改这里；命令行显式参数优先。
    return postcheck_overlap(
        data=args.data or PROJECT_ROOT / "datasets/synthetic_radar_overlap/data.yaml",  # 与现有训练入口的默认总池一致。
        source=args.source or PROJECT_ROOT / "datasets/synthetic_radar_test",  # 测试 CSV 平铺混放，保留原始文件名。
        save_dir=args.save_dir or PROJECT_ROOT / "results/overlap_postcheck",  # 每次填写新目录，避免混入上次分类结果。
        selection_file=args.selection_file or None,  # None 核验完整总池；可填写某次训练的 dataset_selection.json。
        min_overlap_rows=args.min_overlap_rows if args.min_overlap_rows is not None else 1,  # 提高到 8/16 等可过滤极短包含片段。
        progress=not args.no_progress,  # 默认显示索引、核验和复制进度；批处理可用 --no-progress 关闭。
    )


if __name__ == "__main__":
    main()
