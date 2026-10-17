"""Writable dataset caches, including read-only competition data mounts."""
from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import tempfile

from pulsecls.cfg import remap_repo_path


def _writable_directory(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    # Existing directories may themselves be mounted read-only.
    with tempfile.TemporaryFile(dir=directory):
        pass
    return directory


def cache_directory(root: Path, data_cfg: dict) -> Path | None:
    """Keep writable datasets' existing caches; isolate external caches by root."""
    if not data_cfg.get("cache", True):
        return None
    root = root.resolve()
    key = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    configured = data_cfg.get("cache_dir") or os.environ.get("PULSECLS_CACHE_DIR")
    if configured:
        return _writable_directory(remap_repo_path(configured) / key)
    try:
        return _writable_directory(root / ".pulsecls_cache")
    except OSError as error:
        if error.errno not in {errno.EACCES, errno.EPERM, errno.EROFS}:
            raise
    return _writable_directory(Path(tempfile.gettempdir()) / "pulsecls-cache" / key)
