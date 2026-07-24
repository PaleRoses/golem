"""Immutable spatial cell identity and its row-major indexing algebra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

CellId = NewType("CellId", int)


@dataclass(frozen=True)
class GridDims:
    nx: int
    ny: int
    nz: int

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny * self.nz

    def contains(self, x: int, y: int, z: int) -> bool:
        return 0 <= x < self.nx and 0 <= y < self.ny and 0 <= z < self.nz


def flatten(x: int, y: int, z: int, dims: GridDims) -> CellId:
    return CellId((x * dims.ny + y) * dims.nz + z)


def unflatten(cell: CellId, dims: GridDims) -> tuple[int, int, int]:
    plane = dims.ny * dims.nz
    x, remainder = divmod(int(cell), plane)
    y, z = divmod(remainder, dims.nz)
    return (x, y, z)
