"""Physical-hydraulics sheaf-balance solver and edge-law obstruction folds."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from golem.kernel.anatomy.balance import _directed_edge_flow
from golem.kernel.anatomy.geometry import _point_distance
from golem.kernel.anatomy.hydraulics.types import (
    AcceptedPhysicalHydraulics,
    DuplicatePhysicalHydraulicEdgeObstruction,
    DuplicatePhysicalHydraulicNodeObstruction,
    EmptyPhysicalHydraulicGraphObstruction,
    InvalidDynamicViscosityObstruction,
    InvalidFluidDensityObstruction,
    InvalidMetresPerWorldUnitObstruction,
    InvalidPhysicalHydraulicConductanceObstruction,
    InvalidPhysicalHydraulicEdgeGeometryObstruction,
    InvalidPhysicalHydraulicNodePositionObstruction,
    InvalidPumpPressureBoundaryObstruction,
    MissingPhysicalHydraulicNodeObstruction,
    NonFinitePhysicalHydraulicResultObstruction,
    NonLaminarPhysicalHydraulicFlowObstruction,
    NonPositivePhysicalHydraulicFlowObstruction,
    NonPositivePhysicalPumpFlowObstruction,
    PhysicalHydraulicBalanceObstruction,
    PhysicalHydraulicEdgeFlow,
    PhysicalHydraulicNodePressure,
    PhysicalHydraulicResultKind,
    PhysicalHydraulicReynoldsValidity,
    PhysicalHydraulicsReceipt,
    RejectedPhysicalHydraulics,
    _PhysicalHydraulicEdgeLaw,
)
from golem.kernel.anatomy.lineage import _duplicate_values
from golem.kernel.anatomy.vocabulary import _MAXIMUM_POISEUILLE_REYNOLDS

if TYPE_CHECKING:
    from golem.kernel.sheaf import (
        BalanceSolution,
        Lowering,
        Result,
        SymmetricConductanceFlux,
    )

    from golem.kernel.anatomy.graph import (
        ClosedVascularGraph,
        VascularEdge,
        VascularNode,
    )
    from golem.kernel.anatomy.hydraulics.types import (
        PhysicalHydraulicsObstruction,
        PhysicalHydraulicsResult,
        PumpPressureBoundary,
    )


@dataclass(frozen=True)
class _PhysicalHydraulicsCarrier:
    graph: ClosedVascularGraph
    node_index: dict[str, int]
    edge_laws: tuple[_PhysicalHydraulicEdgeLaw, ...]


@dataclass(frozen=True)
class _PhysicalHydraulicsInterpreter:
    """Lower a vascular graph into one dimensional SI hydraulic balance."""

    metres_per_world_unit: float
    fluid_density_kilograms_per_cubic_metre: float
    dynamic_viscosity_pascal_second: float
    pump_pressure_boundary: PumpPressureBoundary

    def lower(
        self, graph: ClosedVascularGraph
    ) -> "Lowering[_PhysicalHydraulicsCarrier, PhysicalHydraulicsResult]":
        from golem.kernel import sheaf

        input_obstructions = _physical_hydraulic_input_obstructions(
            self.metres_per_world_unit,
            self.fluid_density_kilograms_per_cubic_metre,
            self.dynamic_viscosity_pascal_second,
            self.pump_pressure_boundary,
        )
        if input_obstructions:
            return sheaf.LoweringRejected(
                RejectedPhysicalHydraulics(input_obstructions)
            )
        graph_obstructions = _physical_hydraulic_graph_obstructions(
            graph, self.metres_per_world_unit
        )
        if graph_obstructions:
            return sheaf.LoweringRejected(
                RejectedPhysicalHydraulics(graph_obstructions)
            )

        node_by_id = graph.node_by_id
        node_index = {
            node.node_id: index for index, node in enumerate(graph.nodes)
        }
        edge_laws = tuple(
            _physical_hydraulic_edge_law(
                edge,
                node_by_id,
                self.metres_per_world_unit,
                self.dynamic_viscosity_pascal_second,
            )
            for edge in graph.edges
        )
        conductance_obstructions: tuple[PhysicalHydraulicsObstruction, ...] = tuple(
            InvalidPhysicalHydraulicConductanceObstruction(
                law.edge.edge_id,
                law.conductance_cubic_metres_per_pascal_second,
            )
            for law in edge_laws
            if not math.isfinite(
                law.conductance_cubic_metres_per_pascal_second
            )
            or law.conductance_cubic_metres_per_pascal_second <= 0.0
        )
        nonfinite_flow_obstructions: tuple[PhysicalHydraulicsObstruction, ...] = tuple(
            NonFinitePhysicalHydraulicResultObstruction(
                PhysicalHydraulicResultKind.EDGE_FLOW,
                law.edge.edge_id,
                law.conductance_cubic_metres_per_pascal_second
                * self.pump_pressure_boundary.pressure_drop_pascal,
            )
            for law in edge_laws
            if math.isfinite(law.conductance_cubic_metres_per_pascal_second)
            and law.conductance_cubic_metres_per_pascal_second > 0.0
            and not math.isfinite(
                law.conductance_cubic_metres_per_pascal_second
                * self.pump_pressure_boundary.pressure_drop_pascal
            )
        )
        if conductance_obstructions or nonfinite_flow_obstructions:
            return sheaf.LoweringRejected(
                RejectedPhysicalHydraulics(
                    (*conductance_obstructions, *nonfinite_flow_obstructions)
                )
            )

        domain = sheaf.CellularComplex(
            complex_id="physical-vascular-graph-si",
            cell_centers=tuple(
                tuple(
                    coordinate * self.metres_per_world_unit
                    for coordinate in node.position
                )
                for node in graph.nodes
            ),
            adjacencies=tuple(
                sheaf.CellAdjacency(
                    sheaf.CellId(node_index[law.edge.source_node_id]),
                    sheaf.CellId(node_index[law.edge.target_node_id]),
                    law.conductance_cubic_metres_per_pascal_second,
                )
                for law in edge_laws
            ),
        )
        return sheaf.LoweredProblem(
            sheaf.BalanceProblem(
                problem_id=sheaf.BalanceProblemId(
                    "physical-vascular-hydraulics-si"
                ),
                domain=domain,
                symmetric_conductances=tuple(
                    sheaf.SymmetricConductance(
                        sheaf.BalanceInterfaceId(law.edge.edge_id),
                        sheaf.CellId(node_index[law.edge.source_node_id]),
                        sheaf.CellId(node_index[law.edge.target_node_id]),
                        law.conductance_cubic_metres_per_pascal_second,
                    )
                    for law in edge_laws
                ),
                fixed_boundaries=(
                    sheaf.FixedBoundary(
                        sheaf.BoundaryId("physical-pump-outlet"),
                        sheaf.CellId(node_index[graph.pump_outlet_node_id]),
                        self.pump_pressure_boundary.pressure_drop_pascal,
                    ),
                    sheaf.FixedBoundary(
                        sheaf.BoundaryId("physical-pump-inlet"),
                        sheaf.CellId(node_index[graph.pump_inlet_node_id]),
                        0.0,
                    ),
                ),
            ),
            replace(
                sheaf.SealedSolverConfig(),
                relative_tolerance=1.0e-14,
            ),
            _PhysicalHydraulicsCarrier(graph, node_index, edge_laws),
        )

    def lift(
        self,
        carrier: _PhysicalHydraulicsCarrier,
        balance: "Result[BalanceSolution]",
    ) -> PhysicalHydraulicsResult:
        from golem.kernel import sheaf

        if isinstance(balance, sheaf.Rejected):
            return RejectedPhysicalHydraulics(
                tuple(
                    PhysicalHydraulicBalanceObstruction(obstruction)
                    for obstruction in balance.obstructions
                )
            )

        gauge_solution = balance.value
        solution = _translate_hydraulic_balance_pressure(
            gauge_solution, self.pump_pressure_boundary.inlet_pressure_pascal
        )
        flow_by_edge_id = {
            str(flow.interface_id): flow
            for flow in solution.interface_fluxes
            if isinstance(flow, sheaf.SymmetricConductanceFlux)
        }
        edge_flows = tuple(
            _physical_hydraulic_edge_flow(
                law,
                carrier.node_index,
                flow_by_edge_id,
                self.fluid_density_kilograms_per_cubic_metre,
                self.dynamic_viscosity_pascal_second,
            )
            for law in carrier.edge_laws
        )
        node_pressures = tuple(
            PhysicalHydraulicNodePressure(
                node.node_id,
                solution.field.value_at(sheaf.CellId(index)),
            )
            for index, node in enumerate(carrier.graph.nodes)
        )
        boundary_supply_by_id = {
            str(supply.boundary_id): supply.boundary_to_domain_flux
            for supply in solution.fixed_boundary_supplies
        }
        outlet_flow = boundary_supply_by_id["physical-pump-outlet"]
        inlet_flow = -boundary_supply_by_id["physical-pump-inlet"]
        free_node_balances = tuple(
            balance for balance in solution.cell_balances if not balance.is_fixed
        )
        boundary_flow_imbalance = abs(outlet_flow - inlet_flow)
        pump_power = (
            self.pump_pressure_boundary.pressure_drop_pascal * outlet_flow
        )
        receipt = PhysicalHydraulicsReceipt(
            maximum_free_node_normalized_residual=max(
                map(lambda balance: balance.normalized_residual, free_node_balances),
                default=0.0,
            ),
            maximum_free_node_absolute_residual_cubic_metres_per_second=max(
                map(lambda balance: abs(balance.equation_residual), free_node_balances),
                default=0.0,
            ),
            boundary_flow_imbalance_cubic_metres_per_second=boundary_flow_imbalance,
            relative_boundary_flow_imbalance=solution.relative_imbalance,
            pump_volumetric_flow_cubic_metres_per_second=outlet_flow,
            pump_pressure_drop_pascal=(
                self.pump_pressure_boundary.pressure_drop_pascal
            ),
            hydraulic_pump_power_watts=pump_power,
        )
        result_obstructions = _physical_hydraulic_result_obstructions(
            edge_flows,
            node_pressures,
            outlet_flow,
            inlet_flow,
            receipt,
        )
        return (
            RejectedPhysicalHydraulics(result_obstructions)
            if result_obstructions
            else AcceptedPhysicalHydraulics(
                edge_flows,
                node_pressures,
                solution,
                receipt,
            )
        )


def solve_physical_hydraulics(
    graph: ClosedVascularGraph,
    *,
    metres_per_world_unit: float,
    fluid_density_kilograms_per_cubic_metre: float,
    dynamic_viscosity_pascal_second: float,
    pump_pressure_boundary: PumpPressureBoundary,
) -> PhysicalHydraulicsResult:
    """Glue the existing vascular graph into one dimensional SI flow section.

    ``VascularEdge.solved_flow`` is D24's normalized allocation field.  This
    specialization deliberately ignores it, recomputes every conductance from
    metre-scaled geometry and SI viscosity, and returns a disjoint physical
    result instead of laundering normalized values into cubic metres per second.
    """
    from golem.kernel import sheaf

    return sheaf.interpret_balance(
        _PhysicalHydraulicsInterpreter(
            metres_per_world_unit,
            fluid_density_kilograms_per_cubic_metre,
            dynamic_viscosity_pascal_second,
            pump_pressure_boundary,
        ),
        graph,
    )


def _translate_hydraulic_balance_pressure(
    gauge_solution: "BalanceSolution",
    inlet_pressure_pascal: float,
) -> "BalanceSolution":
    """Transport the gauge solve to the requested absolute pressure origin.

    A constant pressure shift leaves every symmetric Poiseuille flux, residual,
    and balance invariant.  Solving at a zero inlet avoids cancellation between
    large absolute pins; this affine transport then makes the retained shared
    section agree exactly with the caller's pressure boundary.
    """
    return replace(
        gauge_solution,
        problem=replace(
            gauge_solution.problem,
            fixed_boundaries=tuple(
                replace(
                    boundary,
                    value=boundary.value + inlet_pressure_pascal,
                )
                for boundary in gauge_solution.problem.fixed_boundaries
            ),
        ),
        field=replace(
            gauge_solution.field,
            values=tuple(
                replace(
                    value,
                    value=value.value + inlet_pressure_pascal,
                )
                for value in gauge_solution.field.values
            ),
        ),
    )


def _physical_hydraulic_input_obstructions(
    metres_per_world_unit: float,
    fluid_density_kilograms_per_cubic_metre: float,
    dynamic_viscosity_pascal_second: float,
    pump_pressure_boundary: PumpPressureBoundary,
) -> tuple[PhysicalHydraulicsObstruction, ...]:
    return (
        (
            (InvalidMetresPerWorldUnitObstruction(metres_per_world_unit),)
            if not math.isfinite(metres_per_world_unit)
            or metres_per_world_unit <= 0.0
            else ()
        )
        + (
            (
                InvalidFluidDensityObstruction(
                    fluid_density_kilograms_per_cubic_metre
                ),
            )
            if not math.isfinite(fluid_density_kilograms_per_cubic_metre)
            or fluid_density_kilograms_per_cubic_metre <= 0.0
            else ()
        )
        + (
            (
                InvalidDynamicViscosityObstruction(
                    dynamic_viscosity_pascal_second
                ),
            )
            if not math.isfinite(dynamic_viscosity_pascal_second)
            or dynamic_viscosity_pascal_second <= 0.0
            else ()
        )
        + (
            (
                InvalidPumpPressureBoundaryObstruction(
                    pump_pressure_boundary.outlet_pressure_pascal,
                    pump_pressure_boundary.inlet_pressure_pascal,
                ),
            )
            if not math.isfinite(
                pump_pressure_boundary.outlet_pressure_pascal
            )
            or not math.isfinite(
                pump_pressure_boundary.inlet_pressure_pascal
            )
            or not math.isfinite(pump_pressure_boundary.pressure_drop_pascal)
            or pump_pressure_boundary.pressure_drop_pascal <= 0.0
            else ()
        )
    )


def _physical_hydraulic_graph_obstructions(
    graph: ClosedVascularGraph,
    metres_per_world_unit: float,
) -> tuple[PhysicalHydraulicsObstruction, ...]:
    node_ids = tuple(node.node_id for node in graph.nodes)
    edge_ids = tuple(edge.edge_id for edge in graph.edges)
    known_nodes = frozenset(node_ids)
    node_by_id = graph.node_by_id
    missing_references = (
        (
            ("pump/outlet", graph.pump_outlet_node_id),
            ("pump/inlet", graph.pump_inlet_node_id),
        )
        + tuple(
            reference
            for edge in graph.edges
            for reference in (
                (f"edge/{edge.edge_id}/source", edge.source_node_id),
                (f"edge/{edge.edge_id}/target", edge.target_node_id),
            )
        )
    )
    invalid_positions = tuple(
        node
        for node in graph.nodes
        if not all(map(math.isfinite, node.position))
        or not all(
            math.isfinite(coordinate * metres_per_world_unit)
            for coordinate in node.position
        )
    )
    edge_lengths = tuple(
        (
            edge,
            _point_distance(
                node_by_id[edge.source_node_id].position,
                node_by_id[edge.target_node_id].position,
            ),
        )
        for edge in graph.edges
        if edge.source_node_id in known_nodes
        and edge.target_node_id in known_nodes
    )
    geometry_obstructions = tuple(
        InvalidPhysicalHydraulicEdgeGeometryObstruction(
            edge.edge_id,
            edge.radius,
            length_world_units,
        )
        for edge, length_world_units in edge_lengths
        if (
            not math.isfinite(edge.radius)
            or edge.radius <= 0.0
            or not math.isfinite(length_world_units)
            or length_world_units <= 0.0
        )
    )
    return (
        (
            (EmptyPhysicalHydraulicGraphObstruction(len(node_ids), len(edge_ids)),)
            if not node_ids or not edge_ids
            else ()
        )
        + tuple(
            map(
                DuplicatePhysicalHydraulicNodeObstruction,
                _duplicate_values(node_ids),
            )
        )
        + tuple(
            map(
                DuplicatePhysicalHydraulicEdgeObstruction,
                _duplicate_values(edge_ids),
            )
        )
        + tuple(
            MissingPhysicalHydraulicNodeObstruction(address, node_id)
            for address, node_id in missing_references
            if node_id not in known_nodes
        )
        + tuple(
            InvalidPhysicalHydraulicNodePositionObstruction(
                node.node_id,
                node.position,
                metres_per_world_unit,
            )
            for node in invalid_positions
        )
        + geometry_obstructions
    )


def _physical_hydraulic_edge_law(
    edge: VascularEdge,
    node_by_id: dict[str, VascularNode],
    metres_per_world_unit: float,
    dynamic_viscosity_pascal_second: float,
) -> _PhysicalHydraulicEdgeLaw:
    radius_metres = edge.radius * metres_per_world_unit
    length_metres = (
        _point_distance(
            node_by_id[edge.source_node_id].position,
            node_by_id[edge.target_node_id].position,
        )
        * metres_per_world_unit
    )
    radius_squared = radius_metres * radius_metres
    numerator = math.pi * radius_squared * radius_squared
    denominator = 8.0 * dynamic_viscosity_pascal_second * length_metres
    conductance = numerator / denominator if denominator > 0.0 else math.nan
    return _PhysicalHydraulicEdgeLaw(
        edge,
        radius_metres,
        length_metres,
        conductance,
    )


def _physical_hydraulic_edge_flow(
    law: _PhysicalHydraulicEdgeLaw,
    node_index: dict[str, int],
    flow_by_edge_id: dict[str, "SymmetricConductanceFlux"],
    fluid_density_kilograms_per_cubic_metre: float,
    dynamic_viscosity_pascal_second: float,
) -> PhysicalHydraulicEdgeFlow:
    flow = _directed_edge_flow(law.edge, node_index, flow_by_edge_id)
    reynolds_denominator = (
        math.pi * law.radius_metres * dynamic_viscosity_pascal_second
    )
    reynolds_number = (
        2.0
        * fluid_density_kilograms_per_cubic_metre
        * abs(flow)
        / reynolds_denominator
        if reynolds_denominator > 0.0
        else math.inf
    )
    return PhysicalHydraulicEdgeFlow(
        law.edge.edge_id,
        flow,
        law.conductance_cubic_metres_per_pascal_second,
        PhysicalHydraulicReynoldsValidity(
            reynolds_number,
            _MAXIMUM_POISEUILLE_REYNOLDS,
            True,
        ),
    )


def _physical_hydraulic_result_obstructions(
    edge_flows: tuple[PhysicalHydraulicEdgeFlow, ...],
    node_pressures: tuple[PhysicalHydraulicNodePressure, ...],
    outlet_flow: float,
    inlet_flow: float,
    receipt: PhysicalHydraulicsReceipt,
) -> tuple[PhysicalHydraulicsObstruction, ...]:
    nonfinite: tuple[PhysicalHydraulicsObstruction, ...] = (
        tuple(
            NonFinitePhysicalHydraulicResultObstruction(
                PhysicalHydraulicResultKind.NODE_PRESSURE,
                pressure.node_id,
                pressure.pressure_pascal,
            )
            for pressure in node_pressures
            if not math.isfinite(pressure.pressure_pascal)
        )
        + tuple(
            NonFinitePhysicalHydraulicResultObstruction(
                PhysicalHydraulicResultKind.EDGE_FLOW,
                edge_flow.edge_id,
                edge_flow.volumetric_flow_cubic_metres_per_second,
            )
            for edge_flow in edge_flows
            if not math.isfinite(
                edge_flow.volumetric_flow_cubic_metres_per_second
            )
        )
        + tuple(
            NonFinitePhysicalHydraulicResultObstruction(
                PhysicalHydraulicResultKind.REYNOLDS_NUMBER,
                edge_flow.edge_id,
                edge_flow.reynolds_validity.reynolds_number,
            )
            for edge_flow in edge_flows
            if not math.isfinite(edge_flow.reynolds_validity.reynolds_number)
        )
        + tuple(
            NonFinitePhysicalHydraulicResultObstruction(
                PhysicalHydraulicResultKind.BOUNDARY_FLOW,
                identifier,
                value,
            )
            for identifier, value in (
                ("pump-outlet", outlet_flow),
                ("pump-inlet", inlet_flow),
            )
            if not math.isfinite(value)
        )
        + tuple(
            NonFinitePhysicalHydraulicResultObstruction(
                result_kind,
                "physical-vascular-hydraulics-si",
                value,
            )
            for result_kind, value in (
                (
                    PhysicalHydraulicResultKind.CONSERVATION_RESIDUAL,
                    receipt.maximum_free_node_absolute_residual_cubic_metres_per_second,
                ),
                (
                    PhysicalHydraulicResultKind.CONSERVATION_RESIDUAL,
                    receipt.boundary_flow_imbalance_cubic_metres_per_second,
                ),
                (
                    PhysicalHydraulicResultKind.PUMP_POWER,
                    receipt.hydraulic_pump_power_watts,
                ),
            )
            if not math.isfinite(value)
        )
    )
    nonpositive = tuple(
        NonPositivePhysicalHydraulicFlowObstruction(
            edge_flow.edge_id,
            edge_flow.volumetric_flow_cubic_metres_per_second,
        )
        for edge_flow in edge_flows
        if math.isfinite(edge_flow.volumetric_flow_cubic_metres_per_second)
        and edge_flow.volumetric_flow_cubic_metres_per_second <= 0.0
    ) + (
        (
            NonPositivePhysicalPumpFlowObstruction(outlet_flow, inlet_flow),
        )
        if math.isfinite(outlet_flow)
        and math.isfinite(inlet_flow)
        and (outlet_flow <= 0.0 or inlet_flow <= 0.0)
        else ()
    )
    nonlaminar = tuple(
        NonLaminarPhysicalHydraulicFlowObstruction(
            edge_flow.edge_id,
            edge_flow.reynolds_validity.reynolds_number,
            edge_flow.reynolds_validity.maximum_laminar_reynolds,
        )
        for edge_flow in edge_flows
        if math.isfinite(edge_flow.reynolds_validity.reynolds_number)
        and not edge_flow.reynolds_validity.is_poiseuille_valid
    )
    return (*nonfinite, *nonpositive, *nonlaminar)
