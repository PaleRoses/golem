"""Physical hydraulics sections and per-element hydraulic solve."""

from __future__ import annotations

import math

from dataclasses import dataclass
from golem.assembly.service import (ChannelResolutionPolicy, PowerPumpBudget, PressurePumpBudget, ServiceIntent)
from golem.kernel.anatomy import (AcceptedPhysicalHydraulics, AcceptedVascularMaterial, ClosedVascularGraph, PhysicalHydraulicsResult, PumpPressureBoundary, UnresolvedLumenObstruction, UnresolvedVascularWallObstruction, VascularMaterialResult, solve_physical_hydraulics)

from golem.assembly.carriers import (RejectedAssembly, _CompiledElement)
from golem.assembly.obstructions import (MissingPhysicalVasculatureObstruction, PumpPowerPressureLimitObstruction)

@dataclass(frozen=True)
class _ResolvedVascularSection:
    element: _CompiledElement
    material: AcceptedVascularMaterial


@dataclass(frozen=True)
class _EmbeddedVascularSection:
    element: _CompiledElement


type _VascularPhysicalSection = (
    _ResolvedVascularSection | _EmbeddedVascularSection
)


def _vascular_physical_section(
    element: _CompiledElement,
    result: VascularMaterialResult,
    policy: ChannelResolutionPolicy,
) -> _VascularPhysicalSection | RejectedAssembly:
    if isinstance(result, AcceptedVascularMaterial):
        return _ResolvedVascularSection(element, result)
    unresolved_only = bool(result.obstructions) and all(
        isinstance(
            obstruction,
            (UnresolvedLumenObstruction, UnresolvedVascularWallObstruction),
        )
        for obstruction in result.obstructions
    )
    return (
        _EmbeddedVascularSection(element)
        if policy is ChannelResolutionPolicy.RESOLVED_OR_EMBEDDED_SLENDER
        and unresolved_only
        else RejectedAssembly(result.obstructions)
    )


def _solve_element_physical_hydraulics(
    graph: ClosedVascularGraph, service: ServiceIntent
) -> PhysicalHydraulicsResult | RejectedAssembly:

    coolant = service.coolant
    if coolant is None:
        return RejectedAssembly(
            (MissingPhysicalVasculatureObstruction("<coolant>"),)
        )
    fluid = coolant.material

    def solve_at_pressure(pressure_pascal: float) -> PhysicalHydraulicsResult:
        return solve_physical_hydraulics(
            graph,
            metres_per_world_unit=service.scale.metres_per_world_unit,
            fluid_density_kilograms_per_cubic_metre=(
                fluid.density_kilograms_per_cubic_metre
            ),
            dynamic_viscosity_pascal_second=(
                fluid.dynamic_viscosity_pascal_second
            ),
            pump_pressure_boundary=PumpPressureBoundary(
                outlet_pressure_pascal=pressure_pascal,
                inlet_pressure_pascal=0.0,
            ),
        )

    match coolant.pump_budget:
        case PressurePumpBudget(maximum_pressure_pascal=pressure_pascal):
            return solve_at_pressure(pressure_pascal)
        case PowerPumpBudget(maximum_power_watts=power_watts):
            unit_result = solve_at_pressure(1.0)
            if not isinstance(unit_result, AcceptedPhysicalHydraulics):
                return unit_result
            unit_flow = (
                unit_result.receipt.pump_volumetric_flow_cubic_metres_per_second
            )
            required_pressure = math.sqrt(power_watts / unit_flow)
            maximum_pressure = fluid.allowable_pressure.maximum_pascal
            return (
                RejectedAssembly(
                    (
                        PumpPowerPressureLimitObstruction(
                            required_pressure, maximum_pressure
                        ),
                    )
                )
                if required_pressure > maximum_pressure
                else solve_at_pressure(required_pressure)
            )


def _assembly_hydraulics_result(
    result: PhysicalHydraulicsResult | RejectedAssembly,
) -> AcceptedPhysicalHydraulics | RejectedAssembly:
    return (
        result
        if isinstance(result, (AcceptedPhysicalHydraulics, RejectedAssembly))
        else RejectedAssembly(result.obstructions)
    )
