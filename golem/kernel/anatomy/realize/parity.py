"""Bilateral parity repair and per-circuit realization."""

from __future__ import annotations

from dataclasses import replace
from functools import reduce
from typing import TYPE_CHECKING

from golem.kernel.anatomy.graph import (
    EmptyPerfusionTerritoryObstruction,
    InfeasibleBifurcationObstruction,
    InsufficientTerminalSitesObstruction,
    MissingProvenanceObstruction,
    RejectedVasculature,
    VascularGluingObstruction,
    VascularInfeasibilityPredicate,
    VascularRelaxationStage,
    VascularSearchEffort,
)
from golem.kernel.anatomy.realize.carriers import (
    AcceptedSearch,
    BudgetExhaustedSearch,
    ExhaustedSearch,
    SearchResult,
    _CapsuleClearanceObligation,
    _CircuitConstructionDomain,
    _PairedCorridorSection,
    _SegmentCapsule,
    _TerminalPortPair,
)
from golem.kernel.anatomy.realize.cco import (
    _build_local_tree_pair,
    _build_local_tree_pair_wider,
    _final_infeasible_obstruction,
    _vascular_search_budget_obstruction,
)
from golem.kernel.anatomy.realize.corridor import (
    _admissible_adjusted_corridors,
    _circuit_corridor,
    _macro_corridor_node_count,
    _paired_corridor_section,
    _sobol_terminal_sites,
)
from golem.kernel.anatomy.realize.materialize import (
    _materialize_circuit_geometry,
    _merge_circuit_geometry,
    _reflect_circuit_geometry,
)

if TYPE_CHECKING:
    from golem.kernel.anatomy.realize.carriers import (
        _CircuitGeometry,
        _LocalVascularTree,
        _StructuralCostIndex,
    )
    from golem.kernel.anatomy.vocabulary import (
        AcceptedAnatomy,
        CirculationCircuit,
        SealedVascularConfig,
    )


def _repair_bilateral_parity(
    allocations: tuple[int, ...],
    quotas: tuple[float, ...],
    mirrored: tuple[bool, ...],
) -> tuple[int, ...]:
    odd_mirrored = tuple(
        index
        for index, (allocation, is_mirrored) in enumerate(
            zip(allocations, mirrored)
        )
        if is_mirrored and allocation % 2 == 1
    )
    if not odd_mirrored:
        return allocations
    paired = tuple(zip(odd_mirrored[::2], odd_mirrored[1::2]))
    repaired_pairs = reduce(
        lambda current, pair: _transfer_parity(current, quotas, pair[0], pair[1]),
        paired,
        allocations,
    )
    if len(odd_mirrored) % 2 == 0:
        return repaired_pairs
    orphan = odd_mirrored[-1]
    unmirrored = tuple(
        index for index, is_mirrored in enumerate(mirrored) if not is_mirrored
    )
    if not unmirrored:
        return repaired_pairs
    partner = min(
        unmirrored,
        key=lambda index: abs(repaired_pairs[index] - quotas[index]),
    )
    return _transfer_parity(repaired_pairs, quotas, orphan, partner)


def _transfer_parity(
    allocations: tuple[int, ...],
    quotas: tuple[float, ...],
    left: int,
    right: int,
) -> tuple[int, ...]:
    left_up_cost = abs((allocations[left] + 1) - quotas[left]) + abs(
        (allocations[right] - 1) - quotas[right]
    )
    right_up_cost = abs((allocations[left] - 1) - quotas[left]) + abs(
        (allocations[right] + 1) - quotas[right]
    )
    delta_left, delta_right = (
        (1, -1) if left_up_cost <= right_up_cost else (-1, 1)
    )
    return tuple(
        value
        + (delta_left if index == left else delta_right if index == right else 0)
        for index, value in enumerate(allocations)
    )


def _circuit_is_mirrored(
    circuit: CirculationCircuit, accepted: AcceptedAnatomy
) -> bool:
    return circuit.effective_demand > circuit.capillary_bed.demand


def _circuit_construction_domain(
    circuit: CirculationCircuit,
    terminal_pair_count: int,
    accepted: AcceptedAnatomy,
    part_by_id: dict[str, dict],
    provenance: dict[str, str],
    landmarks: dict[str, object],
    bone_use_count: dict[str, int],
    shared_pump_interfaces: tuple[tuple[float, float, float], ...],
    config: SealedVascularConfig,
) -> _CircuitConstructionDomain | RejectedVasculature:
    """Derive the constructor-neutral admissible domain of one circuit."""
    region_id = circuit.capillary_bed.region.region_id
    mirrored = _circuit_is_mirrored(circuit, accepted)
    path_parts_by_bone = {
        bone_id: tuple(
            part_by_id[part_id]
            for part_id, address in provenance.items()
            if part_id in part_by_id
            and isinstance(address, str)
            and address.startswith(f"skeleton/{bone_id}/")
        )
        for bone_id in circuit.bones
    }
    missing = tuple(
        bone_id
        for bone_id, parts in path_parts_by_bone.items()
        if not parts
    )
    if missing:
        return RejectedVasculature(
            tuple(map(lambda bone: MissingProvenanceObstruction(region_id, bone), missing))
        )
    pump_bone = accepted.overall.circulation.pump.region.host_bone_id
    local_bones = tuple(
        bone_id
        for bone_id in circuit.bones
        if bone_id != pump_bone
        and bone_use_count.get(bone_id, 0) == 1
        and (
            not mirrored
            or any(part.get("mirror", False) for part in path_parts_by_bone[bone_id])
        )
    )
    territory_parts = tuple(
        _representative_part(part)
        for bone_id in local_bones
        for part in path_parts_by_bone[bone_id]
        if not mirrored or part.get("mirror", False)
    )
    if not territory_parts:
        return RejectedVasculature((EmptyPerfusionTerritoryObstruction(region_id),))
    containment_parts = tuple(
        _representative_part(part)
        for bone_id in circuit.bones
        for part in path_parts_by_bone[bone_id]
        if not part.get("mirror", False) or mirrored
    )
    representative_count = terminal_pair_count // 2 if mirrored else terminal_pair_count
    if representative_count <= 0 or mirrored and terminal_pair_count % 2:
        return RejectedVasculature(
            (
                VascularGluingObstruction(
                    f"anatomy/regions/{region_id}",
                    "bilateral territory requires an even terminal allocation",
                ),
            )
        )
    corridor = _circuit_corridor(circuit, landmarks, shared_pump_interfaces)
    if corridor is None:
        return RejectedVasculature(
            (
                VascularGluingObstruction(
                    f"anatomy/regions/{region_id}",
                    "a skeleton corridor landmark is missing",
                ),
            )
        )
    maximum_tree_radius = config.terminal_radius * representative_count ** (
        1.0 / config.murray_exponent
    )
    paired_corridor = _paired_corridor_section(
        corridor.points, config.terminal_port_half_separation
    )
    if paired_corridor is None:
        return RejectedVasculature(
            (
                VascularGluingObstruction(
                    f"anatomy/regions/{region_id}",
                    "macro corridor cannot derive a transverse transport frame",
                ),
            )
        )
    return _CircuitConstructionDomain(
        region_id=region_id,
        mirrored=mirrored,
        representative_count=representative_count,
        terminal_pair_count=terminal_pair_count,
        territory_parts=territory_parts,
        containment_parts=containment_parts,
        corridor=corridor,
        paired_corridor=paired_corridor,
        per_terminal_flow=circuit.effective_demand / terminal_pair_count,
        macro_node_count=_macro_corridor_node_count(circuit, corridor, landmarks),
        maximum_tree_radius=maximum_tree_radius,
    )


def _greedy_circuit_geometry(
    circuit: CirculationCircuit,
    domain: _CircuitConstructionDomain,
    config: SealedVascularConfig,
    structural_cost_index: _StructuralCostIndex | None,
) -> _CircuitGeometry | RejectedVasculature:
    """Greedy CCO terminal insertion with staged corridor relaxation."""
    region_id = domain.region_id
    mirrored = domain.mirrored
    paired_corridor = domain.paired_corridor
    territory_parts = domain.territory_parts
    representative_count = domain.representative_count
    per_terminal_flow = domain.per_terminal_flow
    containment_parts = domain.containment_parts
    macro_node_count = domain.macro_node_count
    corridor = domain.corridor
    maximum_tree_radius = domain.maximum_tree_radius
    fixed_result = _greedy_corridor_tree_pair(
        paired_corridor,
        territory_parts,
        representative_count,
        region_id,
        config,
        per_terminal_flow,
        containment_parts,
        macro_node_count,
        structural_cost_index,
    )
    if isinstance(fixed_result, tuple):
        return _materialize_relaxed_circuit(
            circuit,
            mirrored,
            fixed_result,
            config,
            VascularSearchEffort(VascularRelaxationStage.FIXED_CORRIDOR, 1),
        )
    stage_one_corridors = _admissible_adjusted_corridors(
        corridor,
        containment_parts,
        maximum_tree_radius,
        config,
        include_combined=False,
    )
    stage_one_result = _descend_greedy_corridors(
        stage_one_corridors,
        territory_parts,
        representative_count,
        region_id,
        config,
        per_terminal_flow,
        containment_parts,
        macro_node_count,
        structural_cost_index,
    )
    if isinstance(stage_one_result, AcceptedSearch):
        return _materialize_relaxed_circuit(
            circuit,
            mirrored,
            stage_one_result.value,
            config,
            VascularSearchEffort(
                VascularRelaxationStage.WAYPOINT_ADJUSTMENT,
                1 + stage_one_result.evaluated_state_count,
            ),
        )
    stage_two_corridors = (
        paired_corridor,
        *_admissible_adjusted_corridors(
            corridor,
            containment_parts,
            maximum_tree_radius,
            config,
            include_combined=True,
        ),
    )
    stage_two_result = _descend_wider_corridors(
        stage_two_corridors,
        territory_parts,
        representative_count,
        region_id,
        config,
        per_terminal_flow,
        containment_parts,
        macro_node_count,
        structural_cost_index,
        config.search_state_budget,
        obstructions=(
            *(
                (fixed_result,)
                if isinstance(
                    fixed_result, InfeasibleBifurcationObstruction
                )
                else ()
            ),
            *stage_one_result.obstructions,
        ),
        attempted_candidate_count=stage_one_result.attempted_candidate_count,
    )
    if isinstance(stage_two_result, AcceptedSearch):
        return _materialize_relaxed_circuit(
            circuit,
            mirrored,
            stage_two_result.value,
            config,
            VascularSearchEffort(
                VascularRelaxationStage.WIDER_SEARCH,
                (
                    1
                    + stage_one_result.evaluated_state_count
                    + stage_two_result.evaluated_state_count
                ),
            ),
        )
    if isinstance(stage_two_result, BudgetExhaustedSearch):
        return RejectedVasculature(
            (
                _vascular_search_budget_obstruction(
                    region_id,
                    stage_two_result,
                    config.search_state_budget,
                ),
            )
        )
    final_obstructions = stage_two_result.obstructions
    if final_obstructions:
        return RejectedVasculature(
            (
                _final_infeasible_obstruction(
                    final_obstructions,
                    region_id,
                    stage_two_result.attempted_candidate_count,
                ),
            )
        )
    terminal_obstruction = next(
        (
            obstruction
            for obstruction in (fixed_result,)
            if isinstance(
                obstruction, InsufficientTerminalSitesObstruction
            )
        ),
        InsufficientTerminalSitesObstruction(
            region_id, representative_count, 0
        ),
    )
    return RejectedVasculature((terminal_obstruction,))


def _greedy_corridor_tree_pair(
    paired_corridor: _PairedCorridorSection,
    territory_parts: tuple[dict, ...],
    representative_count: int,
    region_id: str,
    config: SealedVascularConfig,
    per_terminal_flow: float,
    containment_parts: tuple[dict, ...],
    macro_node_count: int,
    structural_cost_index: _StructuralCostIndex | None,
) -> (
    tuple[_LocalVascularTree, _LocalVascularTree]
    | InfeasibleBifurcationObstruction
    | InsufficientTerminalSitesObstruction
):
    terminal_data = _corridor_terminal_data(
        paired_corridor,
        territory_parts,
        representative_count,
        region_id,
        config,
    )
    if isinstance(terminal_data, InsufficientTerminalSitesObstruction):
        return terminal_data
    terminal_pairs, exchange_clearance_obligations = terminal_data
    return _build_local_tree_pair(
        paired_corridor.supply,
        paired_corridor.returning,
        tuple(pair.supply for pair in terminal_pairs),
        tuple(pair.returning for pair in terminal_pairs),
        per_terminal_flow,
        containment_parts,
        region_id,
        config,
        protected_corridor_node_count=macro_node_count,
        exchange_clearance_obligations=exchange_clearance_obligations,
        structural_cost_index=structural_cost_index,
    )


def _descend_greedy_corridors(
    corridors: tuple[_PairedCorridorSection, ...],
    territory_parts: tuple[dict, ...],
    representative_count: int,
    region_id: str,
    config: SealedVascularConfig,
    per_terminal_flow: float,
    containment_parts: tuple[dict, ...],
    macro_node_count: int,
    structural_cost_index: _StructuralCostIndex | None,
    *,
    obstructions: tuple[InfeasibleBifurcationObstruction, ...] = (),
    evaluated_state_count: int = 0,
) -> SearchResult[tuple[_LocalVascularTree, _LocalVascularTree]]:
    if not corridors:
        return ExhaustedSearch(obstructions, evaluated_state_count, 0)
    result = _greedy_corridor_tree_pair(
        corridors[0],
        territory_parts,
        representative_count,
        region_id,
        config,
        per_terminal_flow,
        containment_parts,
        macro_node_count,
        structural_cost_index,
    )
    if isinstance(result, tuple):
        return AcceptedSearch(result, evaluated_state_count + 1, 0)
    return _descend_greedy_corridors(
        corridors[1:],
        territory_parts,
        representative_count,
        region_id,
        config,
        per_terminal_flow,
        containment_parts,
        macro_node_count,
        structural_cost_index,
        obstructions=(
            *obstructions,
            *((result,) if isinstance(result, InfeasibleBifurcationObstruction) else ()),
        ),
        evaluated_state_count=evaluated_state_count + 1,
    )


def _descend_wider_corridors(
    corridors: tuple[_PairedCorridorSection, ...],
    territory_parts: tuple[dict, ...],
    representative_count: int,
    region_id: str,
    config: SealedVascularConfig,
    per_terminal_flow: float,
    containment_parts: tuple[dict, ...],
    macro_node_count: int,
    structural_cost_index: _StructuralCostIndex | None,
    state_budget: int,
    *,
    obstructions: tuple[InfeasibleBifurcationObstruction, ...] = (),
    evaluated_state_count: int = 0,
    attempted_candidate_count: int = 0,
) -> SearchResult[tuple[_LocalVascularTree, _LocalVascularTree]]:
    if not corridors:
        return ExhaustedSearch(
            obstructions,
            evaluated_state_count,
            attempted_candidate_count,
        )
    if evaluated_state_count >= state_budget:
        return BudgetExhaustedSearch(
            obstructions,
            evaluated_state_count,
            attempted_candidate_count,
            len(corridors),
        )
    terminal_data = _corridor_terminal_data(
        corridors[0],
        territory_parts,
        representative_count,
        region_id,
        config,
    )
    if isinstance(terminal_data, InsufficientTerminalSitesObstruction):
        return _descend_wider_corridors(
            corridors[1:],
            territory_parts,
            representative_count,
            region_id,
            config,
            per_terminal_flow,
            containment_parts,
            macro_node_count,
            structural_cost_index,
            state_budget,
            obstructions=obstructions,
            evaluated_state_count=evaluated_state_count + 1,
            attempted_candidate_count=attempted_candidate_count,
        )
    terminal_pairs, exchange_clearance_obligations = terminal_data
    result = _build_local_tree_pair_wider(
        corridors[0].supply,
        corridors[0].returning,
        tuple(pair.supply for pair in terminal_pairs),
        tuple(pair.returning for pair in terminal_pairs),
        per_terminal_flow,
        containment_parts,
        region_id,
        config,
        macro_node_count,
        exchange_clearance_obligations,
        structural_cost_index,
        state_budget - evaluated_state_count - 1,
    )
    combined_evaluated = evaluated_state_count + 1 + result.evaluated_state_count
    combined_attempted = attempted_candidate_count + result.attempted_candidate_count
    if isinstance(result, AcceptedSearch):
        return AcceptedSearch(
            result.value, combined_evaluated, combined_attempted
        )
    combined_obstructions = (*obstructions, *result.obstructions)
    if isinstance(result, BudgetExhaustedSearch):
        return BudgetExhaustedSearch(
            combined_obstructions,
            combined_evaluated,
            combined_attempted,
            result.remaining_queue_size + len(corridors[1:]),
        )
    return _descend_wider_corridors(
        corridors[1:],
        territory_parts,
        representative_count,
        region_id,
        config,
        per_terminal_flow,
        containment_parts,
        macro_node_count,
        structural_cost_index,
        state_budget,
        obstructions=combined_obstructions,
        evaluated_state_count=combined_evaluated,
        attempted_candidate_count=combined_attempted,
    )


def _corridor_terminal_data(
    paired_corridor: _PairedCorridorSection,
    territory_parts: tuple[dict, ...],
    representative_count: int,
    region_id: str,
    config: SealedVascularConfig,
) -> (
    tuple[
        tuple[_TerminalPortPair, ...],
        tuple[_CapsuleClearanceObligation, ...],
    ]
    | InsufficientTerminalSitesObstruction
):
    maximum_tree_radius = config.terminal_radius * representative_count ** (
        1.0 / config.murray_exponent
    )
    corridor_capsules = tuple(
        _SegmentCapsule(left, right, maximum_tree_radius)
        for shifted_corridor in (
            paired_corridor.supply,
            paired_corridor.returning,
        )
        for left, right in zip(shifted_corridor, shifted_corridor[1:])
    )
    terminal_pairs = _sobol_terminal_sites(
        territory_parts,
        representative_count,
        region_id,
        config,
        anchor=paired_corridor.medial[-1],
        corridor=paired_corridor.medial,
        forbidden_capsules=corridor_capsules,
    )
    if isinstance(terminal_pairs, InsufficientTerminalSitesObstruction):
        return terminal_pairs
    exchange_segments = tuple(
        (pair.supply, pair.returning) for pair in terminal_pairs
    )
    return (
        terminal_pairs,
        (
            _CapsuleClearanceObligation(
                tuple(
                    _SegmentCapsule(left, right, config.terminal_radius)
                    for left, right in exchange_segments
                ),
                tuple(range(len(exchange_segments))),
                config.vessel_clearance,
                VascularInfeasibilityPredicate.EXCHANGE_CLEARANCE,
                None,
            ),
        ),
    )


def _materialize_relaxed_circuit(
    circuit: CirculationCircuit,
    mirrored: bool,
    tree_pair: tuple[_LocalVascularTree, _LocalVascularTree],
    config: SealedVascularConfig,
    effort: VascularSearchEffort,
) -> _CircuitGeometry:
    supply_tree, return_tree = tree_pair
    positive = _materialize_circuit_geometry(
        circuit, "positive" if mirrored else "center", supply_tree, return_tree, config
    )
    geometry = (
        positive
        if not mirrored
        else _merge_circuit_geometry(
            positive, _reflect_circuit_geometry(positive)
        )
    )
    return replace(geometry, search_effort=effort)


def _representative_part(part: dict) -> dict:
    return {key: value for key, value in part.items() if key != "mirror"}
