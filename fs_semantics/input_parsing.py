from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import re

from .logic import split_top_level


def _split_prompt_items(raw: str) -> list[str]:
    if ";" in raw:
        parts = split_top_level(raw, separator=";")
    else:
        parts = [line.strip() for line in raw.splitlines()]
    return [part.strip() for part in parts if part.strip()]


def parse_belief_set_text(text: str) -> list[str]:
    raw = text.strip()
    if not raw:
        return []

    # Accept "B={...}" or "beliefs={...}".
    raw = re.sub(r"^\s*(B|beliefs?)\s*=\s*", "", raw, flags=re.IGNORECASE)
    if raw.startswith("{") and raw.endswith("}"):
        raw = raw[1:-1].strip()
    if not raw:
        return []

    return _split_prompt_items(raw)


def parse_gradient_values_text(text: str) -> list[Fraction]:
    raw = text.strip()
    if not raw:
        return []

    raw = re.sub(r"^\s*(G|gradients?)\s*=\s*", "", raw, flags=re.IGNORECASE)
    if raw.startswith("{") and raw.endswith("}"):
        raw = raw[1:-1].strip()
    if not raw:
        return []

    values: list[Fraction] = []
    for idx, token in enumerate(_split_prompt_items(raw), start=1):
        cleaned = token.replace(",", ".")
        try:
            value = Fraction(cleaned)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid gradient value at position {idx}: {token!r}") from exc
        if value < 0 or value > 1:
            raise ValueError(
                f"Gradient value out of range [0,1] at position {idx}: {token!r}"
            )
        values.append(value)
    return values


def load_belief_strings(
    cli_beliefs: list[str],
    belief_file: str | None,
    belief_set: str | None,
) -> list[str]:
    items = list(cli_beliefs)
    if belief_set:
        items.extend(parse_belief_set_text(belief_set))
    if belief_file:
        path = Path(belief_file)
        for line in path.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue
            items.append(clean)
    return items
