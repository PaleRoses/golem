"""Immutable input carriers for the balance algebra: the cellular complex, the
five local term laws, and the sealed solver configuration."""

from __future__ import annotations

from dataclasses import dataclass

from golem.kernel.sheaf.arrays import _ordered_cells
from golem.kernel.sheaf.vocabulary import (
    AmbientExchangeId,
    BalanceInterfaceId,
    BalanceProblemId,
    BalanceSourceId,
    BoundaryId,
    CellId,
)


@dataclass(frozen=True)
class CellAdjacency:
    left: CellId
    right: CellId
    seam_measure: float

    @property
    def pair(self) -> tuple[CellId, CellId]:
        return _ordered_cells(self.left, self.right)


@dataclass(frozen=True)
class CellularComplex:
    complex_id: str
    cell_centers: tuple[tuple[float, float, float], ...]
    adjacencies: tuple[CellAdjacency, ...]

    @property
    def cell_count(self) -> int:
        return len(self.cell_centers)


@dataclass(frozen=True)
class SymmetricConductance:
    """Pair law with canonical flux ``g * (left - right)``."""

    interface_id: BalanceInterfaceId
    left_cell: CellId
    right_cell: CellId
    conductance: float


@dataclass(frozen=True)
class DirectedTransport:
    """Upwind law with flux ``coefficient * upstream``."""

    interface_id: BalanceInterfaceId
    upstream_cell: CellId
    downstream_cell: CellId
    transport_coefficient: float


@dataclass(frozen=True)
class CellSource:
    """Positive injection satisfying ``net_outflow = source`` at a free cell."""

    source_id: BalanceSourceId
    cell_id: CellId
    value: float


@dataclass(frozen=True)
class FixedBoundary:
    """Dirichlet value whose derived supply closes the global balance."""

    boundary_id: BoundaryId
    cell_id: CellId
    value: float


@dataclass(frozen=True)
class AmbientExchange:
    """Robin law with outward flux ``g * (cell - ambient)``."""

    exchange_id: AmbientExchangeId
    cell_id: CellId
    conductance: float
    ambient_value: float


@dataclass(frozen=True)
class BalanceProblem:
    problem_id: BalanceProblemId
    domain: CellularComplex
    symmetric_conductances: tuple[SymmetricConductance, ...] = ()
    directed_transports: tuple[DirectedTransport, ...] = ()
    sources: tuple[CellSource, ...] = ()
    fixed_boundaries: tuple[FixedBoundary, ...] = ()
    ambient_exchanges: tuple[AmbientExchange, ...] = ()


@dataclass(frozen=True)
class SealedSolverConfig:
    relative_tolerance: float = 1.0e-13
    residual_tolerance: float = 1.0e-10
    imbalance_tolerance: float = 1.0e-10
    normalization_floor: float = 1.0e-15
    maximum_iteration_factor: int = 10
