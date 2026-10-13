"""Compatibility shim. Prefer pulsecls.cfg."""

from pulsecls.cfg import (
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_PATH,
    REPO_ROOT,
    deep_update,
    get_cfg,
    load_pipeline,
    load_yaml,
    normalize_task,
    remap_repo_path,
    resolve_path,
)

__all__ = [
    "DEFAULT_CFG",
    "DEFAULT_CFG_DICT",
    "DEFAULT_CFG_PATH",
    "REPO_ROOT",
    "deep_update",
    "get_cfg",
    "load_pipeline",
    "load_yaml",
    "normalize_task",
    "remap_repo_path",
    "resolve_path",
]
