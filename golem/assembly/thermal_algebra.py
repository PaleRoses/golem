"""Thermal verdict, receipt, and result-gluing algebra."""

from __future__ import annotations

import numpy as np

from dataclasses import replace
from math import fsum

from golem.assembly.address import _semantic_address_for_cell
from golem.assembly.carriers import ElementThermalReceipt, RejectedAssembly, _CompiledElement
from golem.assembly.obstructions import (CoolantTemperatureValidityObstruction, ThermalGradientLimitObstruction, ThermalLimitObstruction)
from golem.assembly.service import ServiceCase, ServiceIntent
from golem.assembly.thermal_types import _AssembledThermalProblem, _SolvedElementThermal
from golem.assembly.voxel import _ResolvedMaterialDomain
from golem.kernel.anatomy import ClosedVascularGraph
from golem.kernel.sheaf import CellId, NaturalConvectionHeatTransferCoefficient, ThermalSolution


def _derive_thermal_receipt(
    problem: _AssembledThermalProblem,
    solution: ThermalSolution,
    natural_convection: NaturalConvectionHeatTransferCoefficient | None,
    coupling_iterations: int,
    maximum_state_delta: float,
    element: _CompiledElement,
    service: ServiceIntent,
    service_case: ServiceCase,
    domain: _ResolvedMaterialDomain,
    graph: ClosedVascularGraph,
) -> _SolvedElementThermal | RejectedAssembly:
    solid_temperatures = tuple(
        solution.balance.field.value_at(CellId(cell_index))
        + problem.reference_temperature_kelvin
        for cell_index in range(problem.solid_count)
    )
    maximum_temperature = max(solid_temperatures)
    maximum_gradient = max(
        (
            abs(
                solid_temperatures[problem.cell_id_by_index[adjacency.left_cell]]
                - solid_temperatures[problem.cell_id_by_index[adjacency.right_cell]]
            )
            / adjacency.path_length_metres
            for adjacency in domain.adjacencies
        ),
        default=0.0,
    )
    coolant_temperatures = tuple(
        (
            node.node_id,
            solution.balance.field.value_at(problem.coolant_cell_id(node.node_id))
            + problem.reference_temperature_kelvin,
        )
        for node in graph.nodes
    )
    thermal_obstructions = (
        (
            ThermalLimitObstruction(
                element.element_id,
                maximum_temperature,
                service.limits.maximum_temperature_kelvin,
            ),
        )
        if maximum_temperature > service.limits.maximum_temperature_kelvin
        else ()
    ) + (
        (
            ThermalGradientLimitObstruction(
                element.element_id,
                maximum_gradient,
                service.limits.maximum_temperature_gradient_kelvin_per_metre,
            ),
        )
        if maximum_gradient
        > service.limits.maximum_temperature_gradient_kelvin_per_metre
        else ()
    ) + tuple(
        CoolantTemperatureValidityObstruction(
            node_id,
            temperature,
            problem.fluid.operating_temperature.minimum_kelvin,
            problem.fluid.operating_temperature.maximum_kelvin,
        )
        for node_id, temperature in coolant_temperatures
        if not problem.fluid.operating_temperature.contains(temperature)
    )
    if thermal_obstructions:
        return RejectedAssembly(thermal_obstructions)
    outlet_temperature = (
        solution.balance.field.value_at(
            problem.coolant_cell_id(graph.pump_inlet_node_id)
        )
        + problem.reference_temperature_kelvin
    )
    hottest_cell = problem.solid_cells[int(np.argmax(solid_temperatures))]
    return _SolvedElementThermal(
        solution=_shift_thermal_solution_reference(
            solution,
            problem.reference_temperature_kelvin,
        ),
        receipt=ElementThermalReceipt(
            element_id=element.element_id,
            case_id=service_case.case_id,
            maximum_temperature_kelvin=maximum_temperature,
            mean_temperature_kelvin=(
                fsum(solid_temperatures) / problem.solid_count
            ),
            maximum_temperature_gradient_kelvin_per_metre=maximum_gradient,
            coolant_outlet_temperature_kelvin=outlet_temperature,
            hottest_semantic_address=_semantic_address_for_cell(
                element,
                domain,
                hottest_cell,
            ),
            heat_transfer_correlations=tuple(
                dict.fromkeys(
                    (
                        *(
                            flux.correlation_kind
                            for flux in solution.wall_heat_fluxes
                        ),
                        *(
                            (natural_convection.correlation_kind,)
                            if natural_convection is not None
                            else ()
                        ),
                    )
                )
            ),
            natural_convection=natural_convection,
            energy=solution.energy,
        ),
        cell_temperatures=tuple(
            zip(problem.solid_cells, solid_temperatures, strict=True)
        ),
        coupling_iterations=coupling_iterations,
        maximum_state_delta=maximum_state_delta,
    )


def _shift_thermal_solution_reference(
    solution: ThermalSolution,
    reference_temperature_kelvin: float,
) -> ThermalSolution:
    balance = solution.balance
    problem = balance.problem
    return replace(
        solution,
        balance=replace(
            balance,
            problem=replace(
                problem,
                fixed_boundaries=tuple(
                    replace(
                        boundary,
                        value=boundary.value + reference_temperature_kelvin,
                    )
                    for boundary in problem.fixed_boundaries
                ),
                ambient_exchanges=tuple(
                    replace(
                        exchange,
                        ambient_value=(
                            exchange.ambient_value
                            + reference_temperature_kelvin
                        ),
                    )
                    for exchange in problem.ambient_exchanges
                ),
            ),
            field=replace(
                balance.field,
                values=tuple(
                    replace(
                        value,
                        value=value.value + reference_temperature_kelvin,
                    )
                    for value in balance.field.values
                ),
            ),
            ambient_fluxes=tuple(
                replace(
                    flux,
                    ambient_value=(
                        flux.ambient_value + reference_temperature_kelvin
                    ),
                )
                for flux in balance.ambient_fluxes
            ),
        ),
    )
