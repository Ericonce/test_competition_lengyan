"""Shared CSV parsing for radar PDW feature columns.

CSV files in the project may use different feature names and column orders.
This module resolves RF/CF, PW and PRI by aliases when a header is present.
Files without a recognizable header are intentionally not guessed: callers
must provide the physical column order explicitly with ``no_header_order``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import numpy as np


FEATURE_COLS = ("CF_MHz", "PW_us", "PRI_us")

# Keep aliases broad enough for common radar-data exports, but do not use
# value-range heuristics: an unlabeled numeric column is inherently ambiguous.
FEATURE_ALIASES = {
    "CF_MHz": (
        "CF_MHz",
        "cfmhz",
        "cf",
        "RF",
        "rfmhz",
        "rf",
        "carrier",
        "carrierfrequency",
        "carrier_frequency",
        "carrierfrequencymhz",
        "centerfrequency",
        "frequency",
        "载频",
        "载频MHz",
        "载频兆赫",
    ),
    "PW_us": (
        "PW_us",
        "pwus",
        "pw",
        "pulsewidth",
        "pulse_width",
        "脉宽",
        "脉宽us",
        "脉宽微秒",
    ),
    "PRI_us": (
        "PRI_us",
        "prius",
        "pri",
        "pulseinterval",
        "pulse_interval",
        "pulserepetitioninterval",
        "pulse_repetition_interval",
        "脉冲重复间隔",
        "脉冲重复周期",
        "脉冲间隔",
    ),
}


class FeatureSchemaError(ValueError):
    """Raised when a CSV cannot be mapped to the three PDW features safely."""


def normalize_column_name(value: object) -> str:
    """Normalize an English or Chinese column label for alias matching."""
    text = unicodedata.normalize("NFKC", str(value)).strip().lstrip("\ufeff").lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", text)


_NORMALIZED_ALIASES = {
    feature: {normalize_column_name(alias) for alias in aliases}
    for feature, aliases in FEATURE_ALIASES.items()
}


def _feature_from_token(token: object) -> str:
    normalized = normalize_column_name(token)
    matches = [
        feature for feature, aliases in _NORMALIZED_ALIASES.items() if normalized in aliases
    ]
    if len(matches) != 1:
        raise FeatureSchemaError(
            f"unknown or ambiguous feature token {token!r}; use CF_MHz,PW_us,PRI_us"
        )
    return matches[0]


def parse_no_header_order(value: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """Parse ``CF_MHz,PW_us,PRI_us`` for a headerless CSV.

    ``None`` or an empty string means that no fallback order was supplied.
    """
    if value is None:
        return None
    tokens = [item.strip() for item in value.split(",")] if isinstance(value, str) else list(value)
    if not tokens or not any(tokens):
        return None
    order = tuple(_feature_from_token(token) for token in tokens if str(token).strip())
    if len(order) != len(FEATURE_COLS) or set(order) != set(FEATURE_COLS):
        raise FeatureSchemaError(
            "no_header_order must contain each feature exactly once: CF_MHz,PW_us,PRI_us"
        )
    return order


def resolve_feature_columns(columns: Iterable[object]) -> dict[str, object]:
    """Map canonical feature names to source DataFrame columns.

    Extra columns are ignored. Duplicate aliases for one feature are rejected
    instead of silently choosing the first one.
    """
    matches: dict[str, list[object]] = {feature: [] for feature in FEATURE_COLS}
    for column in columns:
        normalized = normalize_column_name(column)
        for feature, aliases in _NORMALIZED_ALIASES.items():
            if normalized in aliases:
                matches[feature].append(column)
    missing = [feature for feature, columns_for_feature in matches.items() if not columns_for_feature]
    duplicate = {
        feature: columns_for_feature
        for feature, columns_for_feature in matches.items()
        if len(columns_for_feature) > 1
    }
    if missing or duplicate:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if duplicate:
            details.append(f"duplicate={duplicate}")
        raise FeatureSchemaError(
            "CSV header cannot identify exactly one RF/CF, PW and PRI column ("
            + "; ".join(details)
            + ")"
        )
    return {feature: matches[feature][0] for feature in FEATURE_COLS}


def _read_with_encoding(
    path: Path,
    *,
    header: int | None,
    nrows: int | None = None,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, header=header, encoding=encoding, nrows=nrows)
        except (UnicodeDecodeError, LookupError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise OSError(f"unable to read CSV: {path}")


def _looks_like_header(path: Path) -> bool:
    """Detect a header only when all three feature labels are recognizable."""
    sample = _read_with_encoding(path, header=None, nrows=1)
    if sample.empty:
        return False
    labels = [value for value in sample.iloc[0].tolist() if pd.notna(value)]
    if not labels:
        return False
    try:
        resolve_feature_columns(labels)
    except FeatureSchemaError:
        return False
    return True


@dataclass(frozen=True)
class FeatureTable:
    """A parsed source table and its canonical three-feature view."""

    frame: pd.DataFrame
    features: pd.DataFrame
    has_header: bool
    source_columns: tuple[object, ...]
    column_map: dict[str, object]


def read_feature_csv(
    path: str | Path,
    *,
    no_header_order: str | Sequence[str] | None = None,
    drop_invalid: bool = True,
) -> FeatureTable:
    """Read a CSV and return canonical columns in ``CF, PW, PRI`` order.

    Header behavior:
      * If the first row contains recognizable aliases, it is treated as a
        header and columns are selected by name regardless of their order.
      * Otherwise the file is treated as headerless. Since numeric values do
        not encode their semantic feature reliably, ``no_header_order`` is
        required and defines the physical order of the first three columns.
    """
    path = Path(path)
    has_header = _looks_like_header(path)
    order = None if has_header else parse_no_header_order(no_header_order)
    if not has_header and order is None:
        raise FeatureSchemaError(
            f"{path} has no recognizable RF/CF, PW or PRI header. "
            "Pass no_header_order='CF_MHz,PW_us,PRI_us' (or the actual physical order)."
        )
    frame = _read_with_encoding(path, header=0 if has_header else None)
    source_columns = tuple(frame.columns)
    if has_header:
        column_map = resolve_feature_columns(frame.columns)
        features = frame[[column_map[feature] for feature in FEATURE_COLS]].copy()
    else:
        if frame.shape[1] < len(FEATURE_COLS):
            raise FeatureSchemaError(
                f"{path} has {frame.shape[1]} columns; at least 3 are required for RF/CF, PW and PRI"
            )
        physical_columns = list(frame.columns[: len(FEATURE_COLS)])
        physical_to_feature = dict(zip(physical_columns, order))
        column_map = {
            feature: next(column for column, mapped in physical_to_feature.items() if mapped == feature)
            for feature in FEATURE_COLS
        }
        features = frame[[column_map[feature] for feature in FEATURE_COLS]].copy()
    features.columns = list(FEATURE_COLS)
    for feature in FEATURE_COLS:
        features[feature] = pd.to_numeric(features[feature], errors="coerce")
    if drop_invalid:
        numeric = features.to_numpy(dtype=np.float64, copy=False)
        valid = np.isfinite(numeric).all(axis=1)
        features = features.loc[valid]
    if features.empty:
        raise FeatureSchemaError(f"{path} contains no valid numeric RF/CF, PW and PRI rows")
    return FeatureTable(
        frame=frame,
        features=features.reset_index(drop=True),
        has_header=has_header,
        source_columns=source_columns,
        column_map=column_map,
    )


def feature_cli_help() -> str:
    return (
        "Headerless CSV physical feature order, e.g. "
        "CF_MHz,PW_us,PRI_us. Headered CSV files are detected automatically."
    )
