"""Closed-vascular sheaf-balance lowering, solve, and Poiseuille conductance."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from golem.kernel.anatomy.geometry import _point_distance
from golem.kernel.anatomy.graph import (
    RejectedVasculature,
    VascularBridgeWitness,
    VascularBudgetHypothesis,
    VascularConditioningHypothesis,
    VascularConditionSectionWitness,
    VascularDemandHypothesis,
    VascularGluingInterrogation,
    VascularGluingObstruction,
    VascularResidualSectionWitness,
    VascularStratum,
    VascularTopologyHypothesis,
    vascular_edge_geometry,
)
from golem.kernel.anatomy.vocabulary import CapillaryBed

if TYPE_CHECKING:
    import numpy as np

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
    from golem.kernel.anatomy.vocabulary import AcceptedAnatomy, SealedVascularConfig


def _calibrate_and_solve(
    graph: ClosedVascularGraph,
    accepted: AcceptedAnatomy,
    _allocation: tuple[tuple[str, int], ...],
    config: SealedVascularConfig,
):
    import numpy as np

    target_demand = {
        circuit.capillary_bed.region.region_id: circuit.effective_demand
        for circuit in accepted.circuits
    }
    region_ids = tuple(target_demand)
    free_region_ids = region_ids[:-1]
    total_demand = math.fsum(target_demand.values())

    def evaluate(log_conductance_scales):
        scale_by_region = {
            region_id: math.exp(float(log_scale) / 4.0)
            for region_id, log_scale in zip(
                free_region_ids, log_conductance_scales
            )
        }
        return _calibration_section(
            graph, config, scale_by_region, target_demand, total_demand
        )

    def residual(section) -> np.ndarray:
        delivered_by_region = dict(section[2])
        return np.asarray(
            tuple(
                math.log(
                    delivered_by_region[region_id]
                    / target_demand[region_id]
                )
                for region_id in free_region_ids
            ),
            dtype=np.float64,
        )

    def maximum_error(section) -> float:
        return max(map(lambda item: item[1], section[3]), default=0.0)

    def descend(log_scales: np.ndarray, iteration: int):
        section = evaluate(log_scales)
        if isinstance(section, RejectedVasculature):
            return section
        if (
            maximum_error(section) <= config.delivery_tolerance
            or iteration <= 0
            or not free_region_ids
        ):
            return section
        base_residual = residual(section)
        finite_difference_step = 1.0e-4
        perturbations = tuple(
            log_scales
            + finite_difference_step
            * np.eye(len(free_region_ids), dtype=np.float64)[index]
            for index in range(len(free_region_ids))
        )
        perturbed_sections = tuple(map(evaluate, perturbations))
        rejected = next(
            (
                result
                for result in perturbed_sections
                if isinstance(result, RejectedVasculature)
            ),
            None,
        )
        if isinstance(rejected, RejectedVasculature):
            return rejected
        jacobian = np.column_stack(
            tuple(
                (residual(result) - base_residual) / finite_difference_step
                for result in perturbed_sections
                if not isinstance(result, RejectedVasculature)
            )
        )
        try:
            newton_step = np.linalg.lstsq(
                jacobian, -base_residual, rcond=None
            )[0]
        except np.linalg.LinAlgError:
            return RejectedVasculature(
                (
                    VascularGluingObstruction(
                        "closed-vascular-field/calibration",
                        "finite-difference Jacobian could not be solved",
                    ),
                )
            )
        return descend(
            np.clip(log_scales + np.clip(newton_step, -2.0, 2.0), -20.0, 20.0),
            iteration - 1,
        )

    return descend(
        np.zeros(len(free_region_ids), dtype=np.float64),
        config.calibration_iterations,
    )


def _scaled_exchange_graph(
    graph: ClosedVascularGraph,
    scale_by_region: dict[str, float],
) -> ClosedVascularGraph:
    return replace(
        graph,
        edges=tuple(
            replace(
                edge,
                radius=edge.radius
                * scale_by_region.get(edge.stage.region.region_id, 1.0),
                solved_flow=0.0,
            )
            if edge.stratum is VascularStratum.EXCHANGE
            and isinstance(edge.stage, CapillaryBed)
            else replace(edge, solved_flow=0.0)
            for edge in graph.edges
        ),
    )


def _calibration_section(
    graph: ClosedVascularGraph,
    config: SealedVascularConfig,
    scale_by_region: dict[str, float],
    target_demand: dict[str, float],
    total_demand: float,
):
    adjusted = _scaled_exchange_graph(graph, scale_by_region)
    unit_solution = _solve_vascular_graph(adjusted, config)
    if isinstance(unit_solution, RejectedVasculature):
        return unit_solution
    unit_graph, unit_balance = unit_solution
    boundary_supply_by_id = {
        str(supply.boundary_id): supply.boundary_to_domain_flux
        for supply in unit_balance.fixed_boundary_supplies
    }
    unit_outflow = abs(boundary_supply_by_id["pump-outlet"])
    if not math.isfinite(unit_outflow) or unit_outflow <= 0.0:
        return RejectedVasculature(
            (
                VascularGluingObstruction(
                    "closed-vascular-field/pump-outlet",
                    "unit-pressure solve produced no finite outlet flow",
                ),
            )
        )
    pump_pressure_drop = total_demand / unit_outflow
    solved_graph = _scale_vascular_graph(unit_graph, pump_pressure_drop)
    delivery = _delivered_demand(solved_graph, target_demand)
    invalid_delivery = next(
        (
            (region_id, delivered)
            for region_id, delivered in delivery
            if not math.isfinite(delivered) or delivered <= 0.0
        ),
        None,
    )
    if invalid_delivery is not None:
        region_id, delivered = invalid_delivery
        return RejectedVasculature(
            (
                VascularGluingObstruction(
                    f"closed-vascular-field/{region_id}",
                    f"calibration produced invalid delivery {delivered!r}",
                ),
            )
        )
    delivery_error = tuple(
        (
            region_id,
            abs(delivered - target_demand[region_id])
            / target_demand[region_id],
        )
        for region_id, delivered in delivery
    )
    return (
        solved_graph,
        unit_balance,
        delivery,
        delivery_error,
        pump_pressure_drop,
    )


def _scale_vascular_graph(
    graph: ClosedVascularGraph,
    pressure_scale: float,
) -> ClosedVascularGraph:
    """Apply the linear pump-pressure scale to the unit-potential solution.

    ``solve_balance`` establishes the normalized potential shape.  The authored
    total tissue demand then determines the unique pressure drop that scales
    that section to absolute delivery.  D24's solved flow remains normalized;
    the separate physical solve below never consumes it as an SI quantity.
    """
    return replace(
        graph,
        edges=tuple(
            replace(
                edge,
                solved_flow=edge.solved_flow * pressure_scale,
            )
            for edge in graph.edges
        ),
    )


def _incident_bottleneck_edge(
    node_id: str,
    graph: "ClosedVascularGraph",
    node_by_id: dict[str, "VascularNode"],
    viscosity: float,
) -> "VascularEdge | None":
    incident = tuple(
        edge
        for edge in graph.edges
        if edge.source_node_id == node_id or edge.target_node_id == node_id
    )
    return min(
        incident,
        key=lambda edge: _poiseuille_conductance(
            edge, node_by_id, viscosity
        ),
        default=None,
    )


def _gluing_obstruction(
    config: "SealedVascularConfig",
    carrier: "_ClosedVascularCarrier",
    obstruction: "sheaf.Obstruction",
) -> VascularGluingObstruction:
    """Project one sheaf obstruction into the gluing channel.  Every
    variant keeps the historical two-field shape; a non-convergent balance
    additionally carries R13's interrogation, lifted from solver cells to
    vascular addresses (nodes, regions, edges) while the carrier that owns
    the mapping is still in scope."""
    from golem.kernel import sheaf

    reason = sheaf.render_obstruction(obstruction)
    if not isinstance(obstruction, sheaf.NonConvergentBalanceObstruction):
        return VascularGluingObstruction("closed-vascular-field", reason)
    interrogation = obstruction.interrogation
    if interrogation is None:
        return VascularGluingObstruction("closed-vascular-field", reason)
    nodes = carrier.graph.nodes
    node_by_id = carrier.graph.node_by_id

    def region_of(cell_id: object) -> str | None:
        return nodes[int(cell_id)].region_id

    def node_id_of(cell_id: object) -> str:
        return nodes[int(cell_id)].node_id

    suspect_node_ids = tuple(
        dict.fromkeys(
            (
                *(witness.cell_id for witness in interrogation.conditioning.worst_cells),
                *(witness.cell_id for witness in interrogation.demand.worst_cells),
            )
        )
    )
    suspect_edge_ids = tuple(
        dict.fromkeys(
            (
                *(str(bridge.interface_id) for bridge in interrogation.topology.weakest_bridges),
                *(
                    bottleneck.edge_id
                    for cell_id in suspect_node_ids
                    for bottleneck in (
                        _incident_bottleneck_edge(
                            node_id_of(cell_id),
                            carrier.graph,
                            node_by_id,
                            config.viscosity,
                        ),
                    )
                    if bottleneck is not None
                ),
            )
        )
    )
    edges_by_id = {edge.edge_id: edge for edge in carrier.graph.edges}
    failing_segments = tuple(
        vascular_edge_geometry(edges_by_id[edge_id], node_by_id)
        for edge_id in suspect_edge_ids
        if edge_id in edges_by_id
    )
    return VascularGluingObstruction(
        "closed-vascular-field",
        reason,
        failing_segments,
        VascularGluingInterrogation(
            topology=VascularTopologyHypothesis(
                free_component_count=interrogation.topology.free_component_count,
                weakest_bridges=tuple(
                    VascularBridgeWitness(str(bridge.interface_id), bridge.conductance)
                    for bridge in interrogation.topology.weakest_bridges
                ),
            ),
            conditioning=VascularConditioningHypothesis(
                gershgorin_upper_bound=interrogation.conditioning.gershgorin_upper_bound,
                diagonal_spread=interrogation.conditioning.diagonal_spread,
                worst_sections=tuple(
                    VascularConditionSectionWitness(
                        node_id_of(witness.cell_id),
                        region_of(witness.cell_id),
                        witness.diagonal,
                        witness.conductance_ratio,
                    )
                    for witness in interrogation.conditioning.worst_cells
                ),
            ),
            demand=VascularDemandHypothesis(
                maximum_stop_residual=interrogation.demand.maximum_stop_residual,
                residual_tolerance=interrogation.demand.residual_tolerance,
                relative_residual=interrogation.demand.relative_residual,
                relative_tolerance=interrogation.demand.relative_tolerance,
                iteration_cap=interrogation.demand.iteration_cap,
                iterations_used=interrogation.demand.iterations_used,
                exhausted=interrogation.demand.exhausted,
                worst_sections=tuple(
                    VascularResidualSectionWitness(
                        node_id_of(witness.cell_id),
                        region_of(witness.cell_id),
                        witness.residual,
                    )
                    for witness in interrogation.demand.worst_cells
                ),
            ),
            budget=VascularBudgetHypothesis(
                iteration_cap=interrogation.budget.iteration_cap,
                iterations_used=interrogation.budget.iterations_used,
                relative_tolerance=interrogation.budget.relative_tolerance,
                exhausted=interrogation.budget.exhausted,
            ),
        ),
    )


type _ClosedVascularSolveResult = (
    tuple[ClosedVascularGraph, "BalanceSolution"] | RejectedVasculature
)


@dataclass(frozen=True)
class _ClosedVascularCarrier:
    graph: ClosedVascularGraph
    node_index: dict[str, int]


@dataclass(frozen=True)
class _ClosedVascularInterpreter:
    """Lower a closed vascular graph into the unit-potential balance algebra."""

    config: SealedVascularConfig

    def lower(
        self, graph: ClosedVascularGraph
    ) -> "Lowering[_ClosedVascularCarrier, _ClosedVascularSolveResult]":
        from golem.kernel import sheaf

        node_by_id = graph.node_by_id
        node_index = {
            node.node_id: index for index, node in enumerate(graph.nodes)
        }
        complex_id = "closed-vascular-graph"
        conductance_by_edge_id = {
            edge.edge_id: _poiseuille_conductance(
                edge, node_by_id, self.config.viscosity
            )
            for edge in graph.edges
        }
        complex_value = sheaf.CellularComplex(
            complex_id=complex_id,
            cell_centers=tuple(node.position for node in graph.nodes),
            adjacencies=tuple(
                sheaf.CellAdjacency(
                    sheaf.CellId(node_index[edge.source_node_id]),
                    sheaf.CellId(node_index[edge.target_node_id]),
                    conductance_by_edge_id[edge.edge_id],
                )
                for edge in graph.edges
            ),
        )
        return sheaf.LoweredProblem(
            sheaf.BalanceProblem(
                problem_id=sheaf.BalanceProblemId("closed-vascular-balance"),
                domain=complex_value,
                symmetric_conductances=tuple(
                    sheaf.SymmetricConductance(
                        sheaf.BalanceInterfaceId(edge.edge_id),
                        sheaf.CellId(node_index[edge.source_node_id]),
                        sheaf.CellId(node_index[edge.target_node_id]),
                        conductance_by_edge_id[edge.edge_id],
                    )
                    for edge in graph.edges
                ),
                fixed_boundaries=(
                    sheaf.FixedBoundary(
                        sheaf.BoundaryId("pump-outlet"),
                        sheaf.CellId(node_index[graph.pump_outlet_node_id]),
                        1.0,
                    ),
                    sheaf.FixedBoundary(
                        sheaf.BoundaryId("pump-inlet"),
                        sheaf.CellId(node_index[graph.pump_inlet_node_id]),
                        0.0,
                    ),
                ),
            ),
            sheaf.SealedSolverConfig(
                residual_tolerance=self.config.solver_tolerance,
                imbalance_tolerance=self.config.solver_tolerance,
            ),
            _ClosedVascularCarrier(graph, node_index),
        )

    def lift(
        self,
        carrier: _ClosedVascularCarrier,
        balance: "Result[BalanceSolution]",
    ) -> _ClosedVascularSolveResult:
        from golem.kernel import sheaf

        if isinstance(balance, sheaf.Rejected):
            return RejectedVasculature(
                tuple(
                    _gluing_obstruction(self.config, carrier, obstruction)
                    for obstruction in balance.obstructions
                )
            )
        solution = balance.value
        flow_by_edge_id = {
            str(flow.interface_id): flow
            for flow in solution.interface_fluxes
            if isinstance(flow, sheaf.SymmetricConductanceFlux)
        }
        solved_edges = tuple(
            replace(
                edge,
                solved_flow=_directed_edge_flow(
                    edge, carrier.node_index, flow_by_edge_id
                ),
            )
            for edge in carrier.graph.edges
        )
        return replace(carrier.graph, edges=solved_edges), solution


def _solve_vascular_graph(
    graph: ClosedVascularGraph, config: SealedVascularConfig
):
    from golem.kernel import sheaf

    return sheaf.interpret_balance(_ClosedVascularInterpreter(config), graph)


def _directed_edge_flow(
    edge: VascularEdge,
    node_index: dict[str, int],
    flow_by_edge_id: dict[str, "SymmetricConductanceFlux"],
) -> float:
    source = node_index[edge.source_node_id]
    flow = flow_by_edge_id[edge.edge_id]
    return (
        flow.left_to_right_flux
        if source == int(flow.left_cell)
        else -flow.left_to_right_flux
    )


def _poiseuille_conductance(
    edge: VascularEdge,
    node_by_id: dict[str, VascularNode],
    viscosity: float,
) -> float:
    length = _point_distance(
        node_by_id[edge.source_node_id].position,
        node_by_id[edge.target_node_id].position,
    )
    return math.pi * edge.radius**4 / (8.0 * viscosity * length)


def _delivered_demand(
    graph: ClosedVascularGraph, target_demand: dict[str, float]
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (
            region_id,
            math.fsum(
                edge.solved_flow
                for edge in graph.edges
                if edge.stratum is VascularStratum.EXCHANGE
                and isinstance(edge.stage, CapillaryBed)
                and edge.stage.region.region_id == region_id
            ),
        )
        for region_id in target_demand
    )
