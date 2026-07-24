"""Ambient-coupling fixed-point algebra for lowered thermal problems."""

from __future__ import annotations

from math import fsum

from golem.assembly.carriers import RejectedAssembly, _CompiledElement
from golem.assembly.obstructions import ThermalCouplingNonConvergenceObstruction
from golem.assembly.service import ServiceCase
from golem.assembly.thermal_types import _AssembledThermalProblem
from golem.kernel.anatomy import ClosedVascularGraph
from golem.kernel.sheaf import (Accepted, AmbientExchangeId, AmbientHeatExchange, BalanceProblemId, BoundaryId, CellId, FixedTemperature, NaturalConvectionHeatTransferCoefficient, Rejected, Result, ThermalBalanceProblem, ThermalSolution, evaluate_natural_convection_correlation, solve_thermal_balance)


type _AmbientCoupledSolution = tuple[
    ThermalSolution,
    NaturalConvectionHeatTransferCoefficient | None,
    int,
    float,
]


def _solve_ambient_coupled(
    problem: _AssembledThermalProblem,
    element: _CompiledElement,
    service_case: ServiceCase,
    graph: ClosedVascularGraph,
) -> _AmbientCoupledSolution | RejectedAssembly:
    ambient = problem.ambient

    def evaluate_ambient(
        surface_temperature_kelvin: float,
    ) -> Result[NaturalConvectionHeatTransferCoefficient] | None:
        return (
            evaluate_natural_convection_correlation(
                ambient.natural_convection_correlation,
                ambient.gas,
                ambient.temperature_kelvin,
                surface_temperature_kelvin,
                ambient.pressure_pascal,
                problem.gravity_magnitude,
            )
            if ambient is not None
            else None
        )

    def solve_ambient_fixed_point(
        correlation_result: Result[NaturalConvectionHeatTransferCoefficient] | None,
        previous_surface_temperature: float,
        iteration: int,
    ) -> _AmbientCoupledSolution | RejectedAssembly:
        coefficient = (
            correlation_result.value
            if isinstance(correlation_result, Accepted)
            else None
        )
        ambient_exchanges = (
            tuple(
                AmbientHeatExchange(
                    AmbientExchangeId(f"ambient:{cell}:{boundary_face.value}"),
                    CellId(problem.cell_id_by_index[cell]),
                    coefficient.watt_per_square_metre_kelvin,
                    area,
                    ambient.temperature_kelvin
                    - problem.reference_temperature_kelvin,
                )
                for cell, boundary_face, _axis, area in problem.surface_faces
            )
            if coefficient is not None and ambient is not None
            else ()
        )
        thermal_result = solve_thermal_balance(
            ThermalBalanceProblem(
                problem_id=BalanceProblemId(
                    f"thermal:{element.element_id}:{service_case.case_id}"
                ),
                domain=problem.thermal_domain,
                solid_conduction_links=problem.conduction_links,
                coolant_advection_links=problem.coolant_links,
                wall_heat_exchanges=problem.wall_links,
                sources=problem.sources,
                fixed_temperatures=(
                    FixedTemperature(
                        BoundaryId("coolant-inlet"),
                        problem.coolant_cell_id(graph.pump_outlet_node_id),
                        0.0,
                    ),
                ),
                ambient_heat_exchanges=ambient_exchanges,
            )
        )
        if isinstance(thermal_result, Rejected):
            return RejectedAssembly(thermal_result.obstructions)
        solution = thermal_result.value
        surface_temperature = (
            fsum(
                solution.balance.field.value_at(CellId(cell_id))
                + problem.reference_temperature_kelvin
                for cell_id in problem.surface_cell_ids
            )
            / len(problem.surface_cell_ids)
            if problem.surface_cell_ids
            else previous_surface_temperature
        )
        state_delta = abs(
            surface_temperature - previous_surface_temperature
        ) / max(abs(surface_temperature), 1.0)
        if ambient is None or state_delta <= 1.0e-8:
            return solution, coefficient, iteration, state_delta
        if iteration >= 25:
            return RejectedAssembly(
                (
                    ThermalCouplingNonConvergenceObstruction(
                        element.element_id,
                        service_case.case_id,
                        iteration,
                        25,
                        state_delta,
                        1.0e-8,
                    ),
                )
            )
        next_correlation = evaluate_ambient(surface_temperature)
        return (
            RejectedAssembly(next_correlation.obstructions)
            if isinstance(next_correlation, Rejected)
            else solve_ambient_fixed_point(
                next_correlation,
                surface_temperature,
                iteration + 1,
            )
        )

    initial_surface_temperature = (
        ambient.temperature_kelvin + 1.0 if ambient is not None else 0.0
    )
    initial_correlation = evaluate_ambient(initial_surface_temperature)
    return (
        RejectedAssembly(initial_correlation.obstructions)
        if isinstance(initial_correlation, Rejected)
        else solve_ambient_fixed_point(
            initial_correlation,
            initial_surface_temperature,
            1,
        )
    )
