"""Normalisation helpers for scalar and aggregated OSM tags."""

from __future__ import annotations

from ast import literal_eval
import re
from typing import Any


def tag_values(value: Any) -> set[str]:
    if value is None:
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


def first_number(value: Any) -> float | None:
    match = re.search(r"\d+(?:[.,]\d+)?", str(value))
    return float(match.group().replace(",", ".")) if match else None


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
