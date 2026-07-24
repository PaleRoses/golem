"""Coordinate rounding + deterministic (byte-freezable) JSON serialization."""

from __future__ import annotations

import json

import numpy as np

from golem.kernel.body.types import _ROUND


def _r(x) -> float:
    return round(float(x), _ROUND)


def _rvec(v) -> list[float]:
    return [round(float(c), _ROUND) for c in np.asarray(v, dtype=np.float64)]


def _canonical_json(obj: dict) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False)
