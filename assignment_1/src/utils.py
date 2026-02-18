from __future__ import annotations

import math
import re
from typing import Iterable


TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9\.\-]{0,14}$")


def clean_ticker(t: str) -> str:
    return t.strip().upper()


def is_plausible_ticker(t: str) -> bool:
    t = clean_ticker(t)
    return bool(TICKER_RE.match(t))


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(x, default: float | None = None) -> float | None:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

