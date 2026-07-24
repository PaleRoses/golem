"""Numeric floor for the balance algebra: dense scatter, interface reduction,
canonical cell ordering, and the scalar finiteness predicates interpreters share."""

from __future__ import annotations

from math import isfinite

import numpy as np
from numpy.typing import NDArray

from golem.kernel.sheaf.vocabulary import CellId


def _positive_finite(value: float) -> bool:
    return isfinite(value) and value > 0.0


def _nonnegative_finite(value: float) -> bool:
    return isfinite(value) and value >= 0.0


def _scatter(
    indices: NDArray[np.int64],
    values: NDArray[np.float64],
    cell_count: int,
) -> NDArray[np.float64]:
    return np.bincount(indices, weights=values, minlength=cell_count)


def _cell_totals(
    cell_count: int,
    cells: tuple[CellId, ...],
    values: tuple[float, ...],
) -> NDArray[np.float64]:
    return _scatter(
        np.asarray(tuple(map(int, cells)), dtype=np.int64),
        np.asarray(values, dtype=np.float64),
        cell_count,
    )


def _interface_balance(
    cell_count: int,
    left_cells: tuple[CellId, ...],
    right_cells: tuple[CellId, ...],
    fluxes: tuple[float, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return (
        _cell_totals(cell_count, left_cells, fluxes)
        + _cell_totals(cell_count, right_cells, tuple(-flux for flux in fluxes)),
        _cell_totals(cell_count, left_cells, tuple(map(abs, fluxes)))
        + _cell_totals(cell_count, right_cells, tuple(map(abs, fluxes))),
    )


def _ordered_cells(left: CellId, right: CellId) -> tuple[CellId, CellId]:
    return (left, right) if left < right else (right, left)
