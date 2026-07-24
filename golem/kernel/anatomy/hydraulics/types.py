"""Physical-hydraulics carriers, obstruction ADTs, and result types."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from golem.kernel.sheaf import (
        BalanceSolution,
        Obstruction as SheafObstruction,
    )

    from golem.kernel.anatomy.graph import VascularEdge


@dataclass(frozen=True)
class PumpPressureBoundary:
    """Gauge or absolute pump pins; only their positive difference drives flow."""

    outlet_pressure_pascal: float
    inlet_pressure_pascal: float

    @property
    def pressure_drop_pascal(self) -> float:
        return self.outlet_pressure_pascal - self.inlet_pressure_pascal


@dataclass(frozen=True)
class PhysicalHydraulicReynoldsValidity:
    reynolds_number: float
    maximum_laminar_reynolds: float
    maximum_is_exclusive: bool

    @property
    def is_poiseuille_valid(self) -> bool:
        return (
            math.isfinite(self.reynolds_number)
            and self.reynolds_number >= 0.0
            and (
                self.reynolds_number < self.maximum_laminar_reynolds
                if self.maximum_is_exclusive
                else self.reynolds_number <= self.maximum_laminar_reynolds
            )
        )


@dataclass(frozen=True)
class PhysicalHydraulicEdgeFlow:
    edge_id: str
    volumetric_flow_cubic_metres_per_second: float
    poiseuille_conductance_cubic_metres_per_pascal_second: float
    reynolds_validity: PhysicalHydraulicReynoldsValidity


@dataclass(frozen=True)
class PhysicalHydraulicNodePressure:
    node_id: str
    pressure_pascal: float


@dataclass(frozen=True)
class PhysicalHydraulicsReceipt:
    maximum_free_node_normalized_residual: float
    maximum_free_node_absolute_residual_cubic_metres_per_second: float
    boundary_flow_imbalance_cubic_metres_per_second: float
    relative_boundary_flow_imbalance: float
    pump_volumetric_flow_cubic_metres_per_second: float
    pump_pressure_drop_pascal: float
    hydraulic_pump_power_watts: float


@dataclass(frozen=True)
class InvalidMetresPerWorldUnitObstruction:
    metres_per_world_unit: float


@dataclass(frozen=True)
class InvalidFluidDensityObstruction:
    density_kilograms_per_cubic_metre: float


@dataclass(frozen=True)
class InvalidDynamicViscosityObstruction:
    dynamic_viscosity_pascal_second: float


@dataclass(frozen=True)
class InvalidPumpPressureBoundaryObstruction:
    outlet_pressure_pascal: float
    inlet_pressure_pascal: float


@dataclass(frozen=True)
class EmptyPhysicalHydraulicGraphObstruction:
    node_count: int
    edge_count: int


@dataclass(frozen=True)
class DuplicatePhysicalHydraulicNodeObstruction:
    node_id: str


@dataclass(frozen=True)
class DuplicatePhysicalHydraulicEdgeObstruction:
    edge_id: str


@dataclass(frozen=True)
class MissingPhysicalHydraulicNodeObstruction:
    address: str
    node_id: str


@dataclass(frozen=True)
class InvalidPhysicalHydraulicNodePositionObstruction:
    node_id: str
    position: tuple[float, float, float]
    metres_per_world_unit: float


@dataclass(frozen=True)
class InvalidPhysicalHydraulicEdgeGeometryObstruction:
    edge_id: str
    radius_world_units: float
    length_world_units: float


@dataclass(frozen=True)
class InvalidPhysicalHydraulicConductanceObstruction:
    edge_id: str
    conductance_cubic_metres_per_pascal_second: float


@dataclass(frozen=True)
class PhysicalHydraulicBalanceObstruction:
    obstruction: "SheafObstruction"


class PhysicalHydraulicResultKind(StrEnum):
    NODE_PRESSURE = "node_pressure"
    EDGE_FLOW = "edge_flow"
    REYNOLDS_NUMBER = "reynolds_number"
    BOUNDARY_FLOW = "boundary_flow"
    PUMP_POWER = "pump_power"
    CONSERVATION_RESIDUAL = "conservation_residual"


@dataclass(frozen=True)
class NonFinitePhysicalHydraulicResultObstruction:
    result_kind: PhysicalHydraulicResultKind
    identifier: str
    value: float


@dataclass(frozen=True)
class NonPositivePhysicalHydraulicFlowObstruction:
    edge_id: str
    volumetric_flow_cubic_metres_per_second: float


@dataclass(frozen=True)
class NonPositivePhysicalPumpFlowObstruction:
    outlet_flow_cubic_metres_per_second: float
    inlet_flow_cubic_metres_per_second: float


@dataclass(frozen=True)
class NonLaminarPhysicalHydraulicFlowObstruction:
    edge_id: str
    reynolds_number: float
    maximum_laminar_reynolds: float


type PhysicalHydraulicsObstruction = (
    InvalidMetresPerWorldUnitObstruction
    | InvalidFluidDensityObstruction
    | InvalidDynamicViscosityObstruction
    | InvalidPumpPressureBoundaryObstruction
    | EmptyPhysicalHydraulicGraphObstruction
    | DuplicatePhysicalHydraulicNodeObstruction
    | DuplicatePhysicalHydraulicEdgeObstruction
    | MissingPhysicalHydraulicNodeObstruction
    | InvalidPhysicalHydraulicNodePositionObstruction
    | InvalidPhysicalHydraulicEdgeGeometryObstruction
    | InvalidPhysicalHydraulicConductanceObstruction
    | PhysicalHydraulicBalanceObstruction
    | NonFinitePhysicalHydraulicResultObstruction
    | NonPositivePhysicalHydraulicFlowObstruction
    | NonPositivePhysicalPumpFlowObstruction
    | NonLaminarPhysicalHydraulicFlowObstruction
)


@dataclass(frozen=True)
class AcceptedPhysicalHydraulics:
    edge_flows: tuple[PhysicalHydraulicEdgeFlow, ...]
    node_pressures: tuple[PhysicalHydraulicNodePressure, ...]
    balance_solution: "BalanceSolution"
    receipt: PhysicalHydraulicsReceipt


@dataclass(frozen=True)
class RejectedPhysicalHydraulics:
    obstructions: tuple[PhysicalHydraulicsObstruction, ...]


type PhysicalHydraulicsResult = (
    AcceptedPhysicalHydraulics | RejectedPhysicalHydraulics
)


@dataclass(frozen=True)
class _PhysicalHydraulicEdgeLaw:
    edge: VascularEdge
    radius_metres: float
    length_metres: float
    conductance_cubic_metres_per_pascal_second: float
