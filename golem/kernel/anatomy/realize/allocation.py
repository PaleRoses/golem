"""Terminal-pair allocation and cached vasculature realization."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from functools import lru_cache
from itertools import groupby
from typing import TYPE_CHECKING, assert_never

from golem.kernel.anatomy.balance import _calibrate_and_solve
from golem.kernel.anatomy.graph import (
    AcceptedVasculature,
    CapsuleEscapeObstruction,
    CollectingVenule,
    DemandDeliveryMismatchObstruction,
    DistributingArtery,
    InvalidStructuralGuidanceObstruction,
    InfeasibleBifurcationObstruction,
    RejectedVasculature,
    ResistanceArteriole,
    ReturningVein,
    ReversedVascularFlowObstruction,
    VascularIntersectionObstruction,
    VascularFlowReceipt,
    VascularGluingObstruction,
    VascularSearchBudgetObstruction,
    VascularSolverResidualObstruction,
    vascular_edge_geometry,
)
from golem.kernel.anatomy.geometry import _localized_segment_geometry
from golem.kernel.anatomy.realize.carriers import (
    _CircuitGeometry,
    _StructuralCostField,
)
from golem.kernel.anatomy.realize.clearance import _validate_vascular_geometry
from golem.kernel.anatomy.realize.constructor import DEFAULT_CIRCUIT_CONSTRUCTOR
from golem.kernel.anatomy.realize.corridor import _shared_pump_interface_points
from golem.kernel.anatomy.realize.lineage import (
    _load_vasculature_result,
    _realize_circuits,
    _store_bypassed,
    _store_vasculature_result,
    _vasculature_result_key,
)
from golem.kernel.anatomy.realize.materialize import _glue_circuit_sections
from golem.kernel.anatomy.realize.parity import (
    _circuit_is_mirrored,
    _repair_bilateral_parity,
)
from golem.kernel.anatomy.vocabulary import CapillaryBed, SealedVascularConfig

if TYPE_CHECKING:
    from golem.kernel.mechanics import CellMechanics

    from golem.kernel.anatomy.graph import (
        VascularEdge,
        VasculatureObstruction,
        VasculatureResult,
    )
    from golem.kernel.anatomy.realize.constructor import CircuitConstructor
    from golem.kernel.anatomy.vocabulary import AcceptedAnatomy


def allocate_terminal_pairs(
    accepted: AcceptedAnatomy,
    config: SealedVascularConfig = SealedVascularConfig(),
) -> tuple[tuple[str, int], ...]:
    """Largest-remainder allocation with a sealed minimum and bilateral parity."""
    circuits = accepted.circuits
    bed_count = len(circuits)
    minimum_total = config.minimum_terminal_pairs_per_bed * bed_count
    if bed_count == 0 or minimum_total > config.terminal_pair_budget:
        return ()
    distributable = config.terminal_pair_budget - minimum_total
    total_demand = math.fsum(map(lambda circuit: circuit.effective_demand, circuits))
    quotas = tuple(
        distributable * circuit.effective_demand / total_demand
        for circuit in circuits
    )
    floors = tuple(map(math.floor, quotas))
    preliminary = tuple(
        config.minimum_terminal_pairs_per_bed + floor
        for floor in floors
    )
    remainder_count = config.terminal_pair_budget - sum(preliminary)
    ranked = tuple(
        index
        for index, _fraction in sorted(
            enumerate(tuple(quota - floor for quota, floor in zip(quotas, floors))),
            key=lambda item: (-item[1], circuits[item[0]].capillary_bed.region.region_id),
        )
    )
    rounded = tuple(
        value + int(index in frozenset(ranked[:remainder_count]))
        for index, value in enumerate(preliminary)
    )
    mirrored = tuple(_circuit_is_mirrored(circuit, accepted) for circuit in circuits)
    parity_repaired = _repair_bilateral_parity(rounded, quotas, mirrored)
    return tuple(
        (circuit.capillary_bed.region.region_id, allocation)
        for circuit, allocation in zip(circuits, parity_repaired)
    )


def _derive_structural_cost_field(
    cells: tuple["CellMechanics", ...],
    metres_per_world_unit: float,
    cost_weight: float,
) -> _StructuralCostField | tuple[InvalidStructuralGuidanceObstruction, ...] | None:
    if not cells:
        return None
    obstructions = (
        (
            InvalidStructuralGuidanceObstruction(
                "structural_guidance/metres_per_world_unit",
                "expected a finite positive scale",
            ),
        )
        if not math.isfinite(metres_per_world_unit)
        or metres_per_world_unit <= 0.0
        else ()
    ) + (
        (
            InvalidStructuralGuidanceObstruction(
                "structural_guidance/cost_weight",
                "expected a finite nonnegative weight",
            ),
        )
        if not math.isfinite(cost_weight) or cost_weight < 0.0
        else ()
    ) + tuple(
        InvalidStructuralGuidanceObstruction(
            f"structural_guidance/cells/{index}",
            "expected a finite center and finite nonnegative von Mises stress",
        )
        for index, cell in enumerate(cells)
        if not all(map(math.isfinite, cell.center))
        or not math.isfinite(cell.von_mises_stress)
        or cell.von_mises_stress < 0.0
    )
    if obstructions:
        return obstructions
    grouped_samples = tuple(
        (
            center,
            max(stress for _sample_center, stress in samples),
        )
        for center, samples in groupby(
            sorted(
                (
                    tuple(map(float, cell.center)),
                    float(cell.von_mises_stress),
                )
                for cell in cells
            ),
            key=lambda sample: sample[0],
        )
    )
    maximum_stress = max(
        (stress for _center, stress in grouped_samples), default=0.0
    )
    return _StructuralCostField(
        centers_metres=tuple(center for center, _stress in grouped_samples),
        normalized_von_mises_stress=tuple(
            stress / maximum_stress if maximum_stress > 0.0 else 0.0
            for _center, stress in grouped_samples
        ),
        metres_per_world_unit=metres_per_world_unit,
        cost_weight=cost_weight,
    )


def realize_vasculature(
    accepted: AcceptedAnatomy,
    body_graph: dict,
    config: SealedVascularConfig = SealedVascularConfig(),
    *,
    structural_cells: tuple["CellMechanics", ...] = (),
    structural_metres_per_world_unit: float = 1.0,
    constructor: "CircuitConstructor" = DEFAULT_CIRCUIT_CONSTRUCTOR,
) -> VasculatureResult:
    """Generate, glue, solve, and validate the closed detailed vasculature.

    Terminal allocation, Sobol sites, CCO-style bifurcation insertion, exact
    bilateral reflection, Poiseuille conductance, and flow are solver-owned.
    Optional provisional mechanics cells modify only the local CCO candidate
    cost.  They neither author nor replace the skeleton-derived macro corridor.
    """
    intent = body_graph.get("intent", {})
    canonical_body = json.dumps(
        {
            "parts": body_graph.get("parts", ()),
            "intent": {
                "provenance": intent.get("provenance", {}),
                "landmarks": intent.get("landmarks", {}),
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    structural_field = _derive_structural_cost_field(
        structural_cells,
        structural_metres_per_world_unit,
        config.structural_cost_weight,
    )
    return (
        RejectedVasculature(structural_field)
        if isinstance(structural_field, tuple)
        else (
            _realize_vasculature_uncached(
                accepted,
                canonical_body,
                config,
                structural_field,
                constructor,
            )
            if (
                _store_bypassed()
                or constructor is not DEFAULT_CIRCUIT_CONSTRUCTOR
            )
            else _realize_vasculature_cached(
                accepted,
                canonical_body,
                config,
                structural_field,
                constructor,
            )
        )
    )


@lru_cache(maxsize=8)
def _realize_vasculature_cached(
    accepted: AcceptedAnatomy,
    canonical_body: str,
    config: SealedVascularConfig,
    structural_field: _StructuralCostField | None,
    constructor: "CircuitConstructor" = DEFAULT_CIRCUIT_CONSTRUCTOR,
) -> VasculatureResult:
    key = _vasculature_result_key(
        accepted, canonical_body, config, structural_field
    )
    cached = _load_vasculature_result(key)
    if isinstance(cached, (AcceptedVasculature, RejectedVasculature)):
        return cached
    result = _realize_vasculature_uncached(
        accepted, canonical_body, config, structural_field, constructor
    )
    _store_vasculature_result(key, result)
    return result


def _realize_vasculature_uncached(
    accepted: AcceptedAnatomy,
    canonical_body: str,
    config: SealedVascularConfig,
    structural_field: _StructuralCostField | None,
    constructor: "CircuitConstructor" = DEFAULT_CIRCUIT_CONSTRUCTOR,
) -> VasculatureResult:
    """Content-addressed pure realization shared by repeated assembly compiles."""
    body_graph = json.loads(canonical_body)
    allocation = allocate_terminal_pairs(accepted, config)
    if len(allocation) != len(accepted.circuits):
        return RejectedVasculature(
            (
                VascularGluingObstruction(
                    "anatomy/overall/circulation/exchange_beds",
                    "terminal budget cannot satisfy the sealed minimum",
                ),
            )
        )
    allocation_by_region = dict(allocation)
    provenance = body_graph.get("intent", {}).get("provenance", {})
    landmarks = body_graph.get("intent", {}).get("landmarks", {})
    part_by_id = {part["id"]: part for part in body_graph.get("parts", ())}
    bone_use_count = {
        bone_id: sum(
            map(lambda circuit: int(bone_id in circuit.bones), accepted.circuits)
        )
        for bone_id in frozenset(
            bone for circuit in accepted.circuits for bone in circuit.bones
        )
    }
    circuit_results = _realize_circuits(
        accepted.circuits,
        allocation_by_region,
        accepted,
        part_by_id,
        provenance,
        landmarks,
        bone_use_count,
        _shared_pump_interface_points(accepted, landmarks),
        config,
        structural_field,
        constructor=constructor,
    )
    circuit_obstructions = tuple(
        obstruction
        for result in circuit_results
        if isinstance(result, RejectedVasculature)
        for obstruction in result.obstructions
    )
    if circuit_obstructions:
        return _localized_rejection(
            RejectedVasculature(circuit_obstructions),
            tuple(body_graph.get("parts", ())),
            provenance,
        )
    sections = tuple(
        result
        for result in circuit_results
        if isinstance(result, _CircuitGeometry)
    )
    glued = _glue_circuit_sections(sections)
    if isinstance(glued, RejectedVasculature):
        return glued
    whole_parts = tuple(body_graph.get("parts", ()))
    initial_geometry_validation = _validate_vascular_geometry(
        glued, whole_parts, config
    )
    if initial_geometry_validation.obstructions:
        return _localized_rejection(
            RejectedVasculature(initial_geometry_validation.obstructions),
            whole_parts,
            provenance,
        )
    solved = _calibrate_and_solve(glued, accepted, allocation, config)
    if isinstance(solved, RejectedVasculature):
        return _localized_rejection(solved, whole_parts, provenance)
    (
        solved_graph,
        balance_solution,
        delivery,
        delivery_error,
        pump_pressure_drop,
    ) = solved
    final_geometry_validation = _validate_vascular_geometry(
        solved_graph, whole_parts, config
    )
    if final_geometry_validation.obstructions:
        return _localized_rejection(
            RejectedVasculature(final_geometry_validation.obstructions),
            whole_parts,
            provenance,
        )
    maximum_residual = balance_solution.maximum_normalized_residual
    boundary_error = balance_solution.relative_imbalance
    solver_obstructions: tuple[VasculatureObstruction, ...] = (
        (
            VascularSolverResidualObstruction(
                maximum_residual, config.solver_tolerance
            ),
        )
        if maximum_residual > config.solver_tolerance
        or boundary_error > config.solver_tolerance
        else ()
    ) + tuple(
        DemandDeliveryMismatchObstruction(
            region_id, relative_error, config.delivery_tolerance
        )
        for region_id, relative_error in delivery_error
        if relative_error > config.delivery_tolerance
    ) + tuple(
        ReversedVascularFlowObstruction(
            edge.edge_id,
            edge.solved_flow,
            (vascular_edge_geometry(edge, solved_graph.node_by_id),),
        )
        for edge in solved_graph.edges
        if edge.solved_flow <= 0.0
    )
    if solver_obstructions:
        return _localized_rejection(
            RejectedVasculature(solver_obstructions),
            whole_parts,
            provenance,
        )
    stage_counts = _vascular_stage_counts(solved_graph.edges)
    conservation_error = abs(
        math.fsum(
            balance.net_outflow * pump_pressure_drop
            for balance in balance_solution.cell_balances
        )
    )
    receipt = VascularFlowReceipt(
        terminal_pair_budget=config.terminal_pair_budget,
        terminal_pairs_by_region=allocation,
        stage_counts=stage_counts,
        minimum_capsule_margin=min(
            initial_geometry_validation.minimum_capsule_margin,
            final_geometry_validation.minimum_capsule_margin,
        ),
        maximum_free_cell_residual=maximum_residual,
        boundary_balance_error=boundary_error,
        maximum_delivery_relative_error=max(
            map(lambda item: item[1], delivery_error), default=0.0
        ),
        delivered_demand_by_region=delivery,
        pump_pressure_drop=pump_pressure_drop,
        conservation_error=conservation_error,
        maximum_symmetry_error=(
            final_geometry_validation.maximum_symmetry_error
        ),
        checked_nonincident_pair_count=(
            final_geometry_validation.checked_nonincident_pair_count
        ),
        structural_guidance_sample_count=(
            len(structural_field.centers_metres)
            if structural_field is not None
            else 0
        ),
        maximum_structural_cost_multiplier=(
            structural_field.maximum_cost_multiplier
            if structural_field is not None
            else 1.0
        ),
        search_effort_by_region=tuple(
            (
                circuit.capillary_bed.region.region_id,
                section.search_effort,
            )
            for circuit, section in zip(
                accepted.circuits, sections, strict=True
            )
            if section.search_effort is not None
        ),
    )
    return AcceptedVasculature(
        graph=solved_graph,
        receipt=receipt,
    )


def _localized_rejection(
    rejection: RejectedVasculature,
    parts: tuple[dict, ...],
    provenance: dict[str, str],
) -> RejectedVasculature:
    def localize(obstruction: VasculatureObstruction) -> VasculatureObstruction:
        match obstruction:
            case (
                InfeasibleBifurcationObstruction()
                | VascularSearchBudgetObstruction()
                | CapsuleEscapeObstruction()
                | VascularIntersectionObstruction()
                | ReversedVascularFlowObstruction()
                | VascularGluingObstruction()
            ) as geometric:
                return replace(
                    geometric,
                    failing_segments=tuple(
                        _localized_segment_geometry(
                            segment,
                            parts,
                            provenance,
                        )
                        for segment in geometric.failing_segments
                    ),
                )
            case _:
                return obstruction

    return RejectedVasculature(tuple(map(localize, rejection.obstructions)))


def _vascular_stage_counts(
    edges: tuple[VascularEdge, ...]
) -> tuple[tuple[str, int], ...]:
    stage_names = tuple(map(_vascular_edge_stage_name, edges))
    return tuple(
        (name, stage_names.count(name)) for name in tuple(dict.fromkeys(stage_names))
    )


def _vascular_edge_stage_name(edge: VascularEdge) -> str:
    match edge.stage:
        case DistributingArtery():
            return "artery"
        case ResistanceArteriole():
            return "arteriole"
        case CapillaryBed():
            return "exchange"
        case CollectingVenule():
            return "venule"
        case ReturningVein():
            return "vein"
        case _ as unreachable:
            assert_never(unreachable)
