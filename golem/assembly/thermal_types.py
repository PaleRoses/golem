"""Typed assembly carriers for lowering into the kernel thermal sheaf."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from golem.assembly.carriers import ElementThermalReceipt
from golem.assembly.service import AmbientCondition, HeatSource
from golem.kernel.mechanics import BoundaryFace, CellIndex
from golem.kernel.sheaf import (CellId, CellularComplex, CoolantAdvectionLink, SolidConductionLink, ThermalSolution, ThermalSource, WallHeatExchange)
from golem.materials import IncompressibleFluid


type _ResolvedThermalSource = tuple[HeatSource, tuple[int, ...]]
type _SurfaceFace = tuple[CellIndex, BoundaryFace, int, float]


@dataclass(frozen=True)
class _SolvedElementThermal:
    solution: ThermalSolution
    receipt: ElementThermalReceipt
    cell_temperatures: tuple[tuple[CellIndex, float], ...]
    coupling_iterations: int = 1
    maximum_state_delta: float = 0.0


@dataclass(frozen=True)
class _AssembledThermalProblem:
    thermal_domain: CellularComplex
    conduction_links: tuple[SolidConductionLink, ...]
    coolant_links: tuple[CoolantAdvectionLink, ...]
    wall_links: tuple[WallHeatExchange, ...]
    sources: tuple[ThermalSource, ...]
    surface_faces: tuple[_SurfaceFace, ...]
    surface_cell_ids: tuple[int, ...]
    cell_id_by_index: Mapping[CellIndex, int]
    coolant_cell_id: Callable[[str], CellId]
    reference_temperature_kelvin: float
    ambient: AmbientCondition | None
    gravity_magnitude: float
    solid_count: int
    solid_cells: tuple[CellIndex, ...]
    fluid: IncompressibleFluid
