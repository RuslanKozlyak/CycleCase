"""Normalisation helpers for scalar and aggregated OSM tags."""

from __future__ import annotations

from ast import literal_eval
from collections.abc import Iterable
import math
import re
from typing import Any

import pandas as pd


def _missing(value: Any) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def tag_values(value: Any) -> set[str]:
    """Lower-case values of one tag: OSMnx lists, ``a;b`` strings and empty table cells alike."""

    if _missing(value):
        return set()
    if isinstance(value, (list, tuple, set)):
        result: set[str] = set()
        for item in value:
            result.update(tag_values(item))
        return result
    text = str(value).strip().lower()
    if text.startswith("["):
        try:
            return tag_values(literal_eval(text))
        except (ValueError, SyntaxError):
            pass
    return {part.strip() for part in re.split(r"[;,|]", text) if part.strip()}


def join_values(values: Iterable[str]) -> str | None:
    """Table cell for a set of tag values: sorted and joined with ``;``, ``None`` when empty."""

    return ";".join(sorted(values)) or None


def first_number(value: Any) -> float | None:
    """First number in a tag (``"50 mph"`` -> 50.0); numbers from the table pass through."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return None if math.isnan(value) else float(value)
    match = re.search(r"\d+(?:[.,]\d+)?", str(value))
    return float(match.group().replace(",", ".")) if match else None


def has_any(column: pd.Series, values: Iterable[str] | str) -> pd.Series:
    """True where at least one of the ``;``-separated values of a cell is in ``values``.

    ``has_any(edges["highway"], {"steps", "footway"})``; empty cells give False.
    """

    wanted = {values} if isinstance(values, str) else set(values)
    lookup = {text: not wanted.isdisjoint(text.split(";")) for text in column.dropna().unique()}
    return column.map(lookup).fillna(False).astype(bool)


def edge_osmid_set(data: dict[str, Any]) -> set[int]:
    values = data.get("osmid")
    if values is None:
        return set()
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    result = set()
    for value in values:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return result
