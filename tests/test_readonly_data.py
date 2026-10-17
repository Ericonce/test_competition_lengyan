"""Read-only source data must support sampling, loading, and cache reuse."""
import errno
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pulsecls.data.dataset import load_split_arrays
from pulsecls.data.selection import select_records


def make_data(root, offset=3000):
    for label in range(2):
        (root / str(label)).mkdir(parents=True)
        for sample in range(4):
            (root / str(label) / f"{sample}.csv").write_text(
                "CF_MHz,PW_us,PRI_us\n" +
                "".join(f"{offset + label * 100 + i},10,500\n" for i in range(16)),
                encoding="utf-8")
    return {"root": str(root), "all": ".", "num_classes": 2,
            "features": ["CF_MHz", "PW_us", "PRI_us"], "sequence_length": 16,
            "cache": True, "progress": False,
            "selection": {"sample_ratio": 1., "train_ratio": .75, "val_ratio": .25,
                          "seed": 2026, "stratified": True}}


@pytest.mark.parametrize("enabled", [False, True])
def test_readonly_source_loads_without_writes_and_reuses_enabled_cache(tmp_path, monkeypatch, enabled):
    root = tmp_path / "readonly"
    cfg = make_data(root)
    cfg["cache"] = enabled
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.delenv("PULSECLS_CACHE_DIR", raising=False)
    monkeypatch.setattr("pulsecls.data.cache.tempfile.gettempdir", lambda: str(temporary))
    original_mkdir = Path.mkdir

    def mkdir(path, *args, **kwargs):
        if path.is_relative_to(root):
            raise OSError(errno.EROFS, "Read-only file system", str(path))
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    plan = select_records(cfg)
    train, labels, paths = load_split_arrays(cfg, "train", selection_plan=plan)
    val, _, _ = load_split_arrays(cfg, "val", selection_plan=plan)
    assert (len(train), len(val)) == (6, 2)
    assert not (root / ".pulsecls_cache").exists()
    assert all(Path(p).is_relative_to(root) for p in paths)
    if enabled:
        with patch("pulsecls.data.dataset.load_sample", side_effect=AssertionError("CSV reread")):
            cached, cached_labels, cached_paths = load_split_arrays(cfg, "train")
        np.testing.assert_array_equal(cached, train)
        np.testing.assert_array_equal(cached_labels, labels)
        assert cached_paths == paths
        assert len(list(temporary.rglob("*.npz"))) == 2
        assert len(list(temporary.rglob("file_index_*.json"))) == 1
    else:
        assert list(temporary.iterdir()) == []


def test_external_cache_isolates_datasets_and_cache_false_disables_index(tmp_path, monkeypatch):
    external = tmp_path / "external"
    monkeypatch.setenv("PULSECLS_CACHE_DIR", str(external))
    configurations = [make_data(tmp_path / "first", 3000), make_data(tmp_path / "second", 5000)]
    for cfg, expected in zip(configurations, [3000, 5000]):
        data, _, paths = load_split_arrays(cfg, "train")
        assert data[:, 0, 0].min() >= expected
        assert data[:, 0, 0].max() < expected + 200
        assert all(Path(p).is_relative_to(Path(cfg["root"])) for p in paths)
        assert not (Path(cfg["root"]) / ".pulsecls_cache").exists()
    assert len(list(external.iterdir())) == 2
    assert len(list(external.rglob("*.npz"))) == 2
    disabled = make_data(tmp_path / "disabled")
    disabled.update(cache=False, cache_dir=str(tmp_path / "unused"))
    load_split_arrays(disabled, "train")
    assert not (tmp_path / "unused").exists()
    assert not (Path(disabled["root"]) / ".pulsecls_cache").exists()
