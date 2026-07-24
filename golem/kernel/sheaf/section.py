"""Immutable output carriers: the solved 0-cochain section, the per-interface and
per-cell flux receipts, and the assembled balance solution."""

from __future__ import annotations

from dataclasses import dataclass

from golem.kernel.sheaf.problem import BalanceProblem, SealedSolverConfig
from golem.kernel.sheaf.vocabulary import (
    AmbientExchangeId,
    BalanceInterfaceId,
    BoundaryId,
    CellId,
)


@dataclass(frozen=True)
class SectionValue:
    field_id: str
    cell_id: CellId
    value: float
    normalized_residual: float


@dataclass(frozen=True)
class Section:
    section_id: str
    field_id: str
    values: tuple[SectionValue, ...]

    def value_at(self, cell_id: CellId) -> float:
        return self.values[int(cell_id)].value

    @property
    def max_normalized_residual(self) -> float:
        return max(
            map(lambda value: value.normalized_residual, self.values),
            default=0.0,
        )


@dataclass(frozen=True)
class SymmetricConductanceFlux:
    interface_id: BalanceInterfaceId
    left_cell: CellId
    right_cell: CellId
    conductance: float
    left_to_right_flux: float


@dataclass(frozen=True)
class DirectedTransportFlux:
    interface_id: BalanceInterfaceId
    upstream_cell: CellId
    downstream_cell: CellId
    transport_coefficient: float
    upstream_to_downstream_flux: float


type BalanceInterfaceFlux = SymmetricConductanceFlux | DirectedTransportFlux


@dataclass(frozen=True)
class AmbientExchangeFlux:
    exchange_id: AmbientExchangeId
    cell_id: CellId
    conductance: float
    ambient_value: float
    cell_to_ambient_flux: float


@dataclass(frozen=True)
class FixedBoundarySupply:
    boundary_id: BoundaryId
    cell_id: CellId
    boundary_to_domain_flux: float


@dataclass(frozen=True)
class BalanceCell:
    cell_id: CellId
    is_fixed: bool
    authored_source: float
    incident_flux: float
    net_outflow: float
    equation_residual: float
    normalized_residual: float


@dataclass(frozen=True)
class BalanceSolution:
    problem: BalanceProblem
    field: Section
    interface_fluxes: tuple[BalanceInterfaceFlux, ...]
    ambient_fluxes: tuple[AmbientExchangeFlux, ...]
    fixed_boundary_supplies: tuple[FixedBoundarySupply, ...]
    cell_balances: tuple[BalanceCell, ...]
    relative_imbalance: float
    solver_config: SealedSolverConfig

    @property
    def maximum_normalized_residual(self) -> float:
        return self.field.max_normalized_residual
