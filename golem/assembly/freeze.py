"""Immutability helpers: read-only arrays and frozen report views."""

from __future__ import annotations

import numpy as np

from types import MappingProxyType
from typing import Mapping

def _read_only_array(value: object) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    array.setflags(write=False)
    return array


def _freeze_value(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(map(_freeze_value, value))
    if isinstance(value, set):
        return frozenset(map(_freeze_value, value))
    if isinstance(value, np.ndarray):
        return _read_only_array(value)
    return value


def _frozen_report(report: dict[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {key: _freeze_value(item) for key, item in report.items()}
    )
