"""Constrained constructive optimization vascular tree growth."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import accumulate
from typing import TYPE_CHECKING

from golem.kernel.anatomy.geometry import (
    _certified_segment_capsule_margins,
    _point_distance,
    _point_segment_distance,
)
from golem.kernel.anatomy.graph import (
    InfeasibleBifurcationObstruction,
    VascularGrowthPhase,
    VascularInfeasibilityPredicate,
    VascularLane,
    VascularNodeKind,
    VascularSearchBudgetObstruction,
    vascular_segment_geometry,
)
from golem.kernel.anatomy.realize.bifurcation import (
    _evaluate_bifurcation_candidates,
)
from golem.kernel.anatomy.realize.carriers import (
    AcceptedSearch,
    BudgetExhaustedSearch,
    ExhaustedSearch,
    FeasibleBifurcation,
    RejectedBifurcationCandidate,
    SearchResult,
    _CapsuleClearanceObligation,
    _LocalVascularTree,
    _SegmentCapsule,
)
from golem.kernel.anatomy.realize.clearance import (
    _capsule_sets_clearance_violation,
)
from golem.kernel.anatomy.realize.tree import (
    _ancestor_edge_child_indices,
    _local_tree_indexed_capsules,
    _local_tree_subtree_flows,
)

if TYPE_CHECKING:
    from golem.kernel.anatomy.realize.carriers import _StructuralCostIndex
    from golem.kernel.anatomy.vocabulary import SealedVascularConfig


@dataclass(frozen=True)
class _SeedSearch:
    candidates: tuple[tuple[int, _LocalVascularTree], ...]
    obstructions: tuple[InfeasibleBifurcationObstruction, ...]
    attempted_candidate_count: int


@dataclass(frozen=True)
class _SeedCandidateStage:
    initial: _LocalVascularTree
    indexed_capsules: tuple[tuple[int, _SegmentCapsule], ...]
    capsules: tuple[_SegmentCapsule, ...]
    direct_length: float
    clearance_violation: tuple | None
    terminal_ordinal: int


@dataclass(frozen=True)
class _InsertionBatch:
    candidates: tuple[FeasibleBifurcation, ...]
    obstructions: tuple[InfeasibleBifurcationObstruction, ...]
    checked_ancestor_child_indices: frozenset[int]
    escaped_ancestor_margins: tuple[tuple[int, float], ...]
    attempted_candidate_count: int


@dataclass(frozen=True)
class _InsertionContinuation:
    tree: _LocalVascularTree
    remaining_sites: tuple[tuple[int, tuple[float, float, float]], ...]
    lower_bound_cost: float


def _search_exhausted_obstruction(
    region_id: str,
    phase: VascularGrowthPhase,
    lane: VascularLane,
    terminal_index: int | None,
    attempted_candidate_count: int,
) -> InfeasibleBifurcationObstruction:
    return InfeasibleBifurcationObstruction(
        region_id=region_id,
        phase=phase,
        lane=lane,
        predicate=VascularInfeasibilityPredicate.SEARCH_EXHAUSTED,
        required=1.0,
        observed=0.0,
        terminal_index=terminal_index,
        attempted_candidate_count=attempted_candidate_count,
    )


def _point3(value: object) -> tuple[float, float, float] | None:
    return (
        tuple(map(float, value))
        if isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(component, (int, float)) for component in value)
        else None
    )


def _site_total_travel(
    sites: tuple[tuple[float, float, float], ...],
    ordinal: int,
) -> float:
    return math.fsum(_point_distance(sites[ordinal], site) for site in sites)


def _candidates_by_minimum_total_travel(
    feasible: tuple[tuple[int, _LocalVascularTree], ...],
    sites: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[int, _LocalVascularTree], ...]:
    return tuple(
        sorted(
            feasible,
            key=lambda candidate: (
                _site_total_travel(sites, candidate[0]),
                candidate[0],
            ),
        )
    )


def _initial_local_tree_candidates(
    corridor: tuple[tuple[float, float, float], ...],
    sites: tuple[tuple[float, float, float], ...],
    terminal_flow: float,
    terminal_kind: VascularNodeKind,
    containment_parts: tuple[dict, ...],
    config: SealedVascularConfig,
    protected_corridor_node_count: int,
    clearance_obligations: tuple[_CapsuleClearanceObligation, ...],
    region_id: str,
    lane: VascularLane,
) -> _SeedSearch:
    staged = tuple(
        _stage_initial_local_tree_candidate(
            corridor,
            site,
            terminal_flow,
            terminal_ordinal,
            terminal_kind,
            config,
            protected_corridor_node_count,
            clearance_obligations,
        )
        for terminal_ordinal, site in enumerate(sites)
    )
    batched_margins = _certified_segment_capsule_margins(
        tuple(capsule for stage in staged for capsule in stage.capsules),
        containment_parts,
        min(3, config.capsule_samples),
        config.wall_clearance,
    )
    offsets = (0, *accumulate(len(stage.capsules) for stage in staged))
    results = tuple(
        _finish_initial_local_tree_candidate(
            stage,
            batched_margins[offsets[order]:offsets[order + 1]],
            config,
            region_id,
            lane,
        )
        for order, stage in enumerate(staged)
    )
    feasible = _candidates_by_minimum_total_travel(
        tuple(
            candidate
            for candidate in results
            if isinstance(candidate, tuple)
        ),
        sites,
    )
    obstructions = tuple(
        result
        for result in results
        if isinstance(result, InfeasibleBifurcationObstruction)
    )
    return _SeedSearch(
        feasible,
        obstructions
        or (
            _search_exhausted_obstruction(
                region_id,
                VascularGrowthPhase.SEED,
                lane,
                None,
                len(results),
            ),
        ),
        len(results),
    )


def _complete_local_tree_candidate(
    candidate: tuple[int, _LocalVascularTree],
    sites: tuple[tuple[float, float, float], ...],
    terminal_flow: float,
    terminal_kind: VascularNodeKind,
    containment_parts: tuple[dict, ...],
    region_id: str,
    config: SealedVascularConfig,
    clearance_obligations: tuple[_CapsuleClearanceObligation, ...],
    structural_cost_index: _StructuralCostIndex | None,
    lane: VascularLane,
) -> _LocalVascularTree | InfeasibleBifurcationObstruction:
    initial_ordinal, initial = candidate
    return _insert_feasible_terminal_sequence(
        initial,
        tuple(
            indexed_site
            for indexed_site in enumerate(sites)
            if indexed_site[0] != initial_ordinal
        ),
        terminal_flow,
        terminal_kind,
        containment_parts,
        region_id,
        config,
        clearance_obligations,
        structural_cost_index,
        lane,
    )


def _build_local_tree(
    corridor: tuple[tuple[float, float, float], ...],
    sites: tuple[tuple[float, float, float], ...],
    terminal_flow: float,
    terminal_kind: VascularNodeKind,
    containment_parts: tuple[dict, ...],
    region_id: str,
    config: SealedVascularConfig,
    lane: VascularLane,
    protected_corridor_node_count: int = 0,
    clearance_obligations: tuple[_CapsuleClearanceObligation, ...] = (),
    insertion_clearance_obligations: tuple[
        _CapsuleClearanceObligation, ...
    ] = (),
    structural_cost_index: _StructuralCostIndex | None = None,
) -> _LocalVascularTree | InfeasibleBifurcationObstruction:
    if len(corridor) < 2 or not sites:
        return _search_exhausted_obstruction(
            region_id,
            VascularGrowthPhase.SEED,
            lane,
            None,
            0,
        )
    all_clearance_obligations = (
        *clearance_obligations,
        *insertion_clearance_obligations,
    )
    initial_search = _initial_local_tree_candidates(
        corridor,
        sites,
        terminal_flow,
        terminal_kind,
        containment_parts,
        config,
        protected_corridor_node_count,
        all_clearance_obligations,
        region_id,
        lane,
    )

    def descend(
        candidates: tuple[tuple[int, _LocalVascularTree], ...],
        last_obstruction: InfeasibleBifurcationObstruction,
    ) -> _LocalVascularTree | InfeasibleBifurcationObstruction:
        if not candidates:
            return last_obstruction
        completed = _complete_local_tree_candidate(
            candidates[0],
            sites,
            terminal_flow,
            terminal_kind,
            containment_parts,
            region_id,
            config,
            all_clearance_obligations,
            structural_cost_index,
            lane,
        )
        return (
            completed
            if isinstance(completed, _LocalVascularTree)
            else descend(candidates[1:], completed)
        )

    return descend(
        initial_search.candidates,
        initial_search.obstructions[0],
    )


def _build_local_tree_pair(
    supply_corridor: tuple[tuple[float, float, float], ...],
    return_corridor: tuple[tuple[float, float, float], ...],
    supply_sites: tuple[tuple[float, float, float], ...],
    return_sites: tuple[tuple[float, float, float], ...],
    terminal_flow: float,
    containment_parts: tuple[dict, ...],
    region_id: str,
    config: SealedVascularConfig,
    protected_corridor_node_count: int,
    exchange_clearance_obligations: tuple[
        _CapsuleClearanceObligation, ...
    ],
    structural_cost_index: _StructuralCostIndex | None,
) -> (
    tuple[_LocalVascularTree, _LocalVascularTree]
    | InfeasibleBifurcationObstruction
):
    """Reserve a local return section, then descend and glue both trees."""
    if (
        len(supply_corridor) < 2
        or len(return_corridor) < 2
        or not supply_sites
        or not return_sites
    ):
        lane = (
            VascularLane.SUPPLY
            if len(supply_corridor) < 2 or not supply_sites
            else VascularLane.RETURN
        )
        return _search_exhausted_obstruction(
            region_id,
            VascularGrowthPhase.SEED,
            lane,
            None,
            0,
        )
    return_reservation_search = _initial_local_tree_candidates(
        return_corridor,
        return_sites,
        terminal_flow,
        VascularNodeKind.RETURN_TERMINAL,
        containment_parts,
        config,
        protected_corridor_node_count,
        exchange_clearance_obligations,
        region_id,
        VascularLane.RETURN,
    )

    def descend(
        candidates: tuple[tuple[int, _LocalVascularTree], ...],
        last_obstruction: InfeasibleBifurcationObstruction,
    ) -> (
        tuple[_LocalVascularTree, _LocalVascularTree]
        | InfeasibleBifurcationObstruction
    ):
        if not candidates:
            return last_obstruction
        reserved_return_tree = candidates[0][1]
        reservation_obligation = _tree_clearance_obligation(
            reserved_return_tree, config, VascularLane.RETURN
        )
        supply_obligations = (
            reservation_obligation,
            *exchange_clearance_obligations,
        )
        supply_search = _initial_local_tree_candidates(
            supply_corridor,
            supply_sites,
            terminal_flow,
            VascularNodeKind.SUPPLY_TERMINAL,
            containment_parts,
            config,
            protected_corridor_node_count,
            supply_obligations,
            region_id,
            VascularLane.SUPPLY,
        )

        def descend_supply(
            local_candidates: tuple[tuple[int, _LocalVascularTree], ...],
            local_obstruction: InfeasibleBifurcationObstruction,
        ) -> (
            tuple[_LocalVascularTree, _LocalVascularTree]
            | InfeasibleBifurcationObstruction
        ):
            if not local_candidates:
                return local_obstruction
            supply_tree = _complete_local_tree_candidate(
                local_candidates[0],
                supply_sites,
                terminal_flow,
                VascularNodeKind.SUPPLY_TERMINAL,
                containment_parts,
                region_id,
                config,
                supply_obligations,
                structural_cost_index,
                VascularLane.SUPPLY,
            )
            if isinstance(supply_tree, InfeasibleBifurcationObstruction):
                return descend_supply(local_candidates[1:], supply_tree)
            return_tree = _build_local_tree(
                return_corridor,
                return_sites,
                terminal_flow,
                VascularNodeKind.RETURN_TERMINAL,
                containment_parts,
                region_id,
                config,
                VascularLane.RETURN,
                protected_corridor_node_count=protected_corridor_node_count,
                clearance_obligations=(
                    _tree_clearance_obligation(
                        supply_tree, config, VascularLane.SUPPLY
                    ),
                ),
                insertion_clearance_obligations=(
                    exchange_clearance_obligations
                ),
                structural_cost_index=structural_cost_index,
            )
            return (
                (supply_tree, return_tree)
                if isinstance(return_tree, _LocalVascularTree)
                else descend_supply(local_candidates[1:], return_tree)
            )

        compatible = descend_supply(
            supply_search.candidates,
            supply_search.obstructions[0],
        )
        return (
            compatible
            if isinstance(compatible, tuple)
            else descend(candidates[1:], compatible)
        )

    return descend(
        return_reservation_search.candidates,
        return_reservation_search.obstructions[0],
    )


def _insert_feasible_terminal_sequence(
    tree: _LocalVascularTree,
    remaining_sites: tuple[tuple[int, tuple[float, float, float]], ...],
    terminal_flow: float,
    terminal_kind: VascularNodeKind,
    containment_parts: tuple[dict, ...],
    region_id: str,
    config: SealedVascularConfig,
    clearance_obligations: tuple[_CapsuleClearanceObligation, ...],
    structural_cost_index: _StructuralCostIndex | None,
    lane: VascularLane,
) -> _LocalVascularTree | InfeasibleBifurcationObstruction:
    if not remaining_sites:
        return tree

    def first_feasible(
        sites: tuple[tuple[int, tuple[float, float, float]], ...],
        first_obstruction: InfeasibleBifurcationObstruction | None = None,
    ) -> (
        tuple[tuple[int, tuple[float, float, float]], _LocalVascularTree]
        | InfeasibleBifurcationObstruction
    ):
        if not sites:
            assert first_obstruction is not None
            return first_obstruction
        indexed_site = sites[0]
        inserted = _insert_terminal_cco(
            tree,
            indexed_site[1],
            terminal_flow,
            indexed_site[0],
            terminal_kind,
            containment_parts,
            region_id,
            config,
            lane,
            clearance_obligations,
            structural_cost_index,
        )
        return (
            (indexed_site, inserted)
            if isinstance(inserted, _LocalVascularTree)
            else first_feasible(
                sites[1:],
                first_obstruction or inserted,
            )
        )

    selected = first_feasible(remaining_sites)
    if isinstance(selected, InfeasibleBifurcationObstruction):
        return selected
    selected_site, selected_tree = selected
    return _insert_feasible_terminal_sequence(
        selected_tree,
        tuple(
            indexed_site
            for indexed_site in remaining_sites
            if indexed_site[0] != selected_site[0]
        ),
        terminal_flow,
        terminal_kind,
        containment_parts,
        region_id,
        config,
        clearance_obligations,
        structural_cost_index,
        lane,
    )


def _stage_initial_local_tree_candidate(
    corridor: tuple[tuple[float, float, float], ...],
    site: tuple[float, float, float],
    terminal_flow: float,
    terminal_ordinal: int,
    terminal_kind: VascularNodeKind,
    config: SealedVascularConfig,
    protected_corridor_node_count: int,
    clearance_obligations: tuple[_CapsuleClearanceObligation, ...],
) -> _SeedCandidateStage:
    initial = _LocalVascularTree(
        positions=(*corridor, site),
        parents=(-1, *tuple(range(len(corridor) - 1)), len(corridor) - 1),
        kinds=(
            *tuple(
                VascularNodeKind.MACRO_CORRIDOR
                if index < protected_corridor_node_count
                else VascularNodeKind.CORRIDOR
                for index, _point in enumerate(corridor)
            ),
            terminal_kind,
        ),
        terminal_flows=(*tuple(0.0 for _ in corridor), terminal_flow),
        terminal_ordinals=(*tuple(None for _ in corridor), terminal_ordinal),
    )
    indexed_capsules = _local_tree_indexed_capsules(initial, config)
    capsules = tuple(capsule for _child_index, capsule in indexed_capsules)
    clearance_violation = next(
        (
            (obligation, violation)
            for obligation in clearance_obligations
            for violation in (
                _capsule_sets_clearance_violation(
                    capsules,
                    obligation.capsules,
                    obligation.minimum_surface_clearance,
                ),
            )
            if violation is not None
        ),
        None,
    )
    return _SeedCandidateStage(
        initial=initial,
        indexed_capsules=indexed_capsules,
        capsules=capsules,
        direct_length=_point_distance(corridor[-1], site),
        clearance_violation=clearance_violation,
        terminal_ordinal=terminal_ordinal,
    )


def _finish_initial_local_tree_candidate(
    stage: _SeedCandidateStage,
    margins: tuple[float, ...],
    config: SealedVascularConfig,
    region_id: str,
    lane: VascularLane,
) -> tuple[int, _LocalVascularTree] | InfeasibleBifurcationObstruction:
    initial = stage.initial
    indexed_capsules = stage.indexed_capsules
    direct_length = stage.direct_length
    clearance_violation = stage.clearance_violation
    terminal_ordinal = stage.terminal_ordinal
    wall_violation = next(
        (
            (child_index, capsule, margin)
            for (child_index, capsule), margin in zip(
                indexed_capsules, margins, strict=True
            )
            if margin < config.wall_clearance
        ),
        None,
    )
    if direct_length < config.minimum_segment_length:
        return InfeasibleBifurcationObstruction(
            region_id=region_id,
            phase=VascularGrowthPhase.SEED,
            lane=lane,
            predicate=VascularInfeasibilityPredicate.SEGMENT_LENGTH,
            required=config.minimum_segment_length,
            observed=direct_length,
            terminal_index=None,
            candidate_point_index=terminal_ordinal,
            failing_segments=(
                vascular_segment_geometry(
                    initial.positions[-2],
                    initial.positions[-1],
                    config.terminal_radius,
                ),
            ),
        )
    if wall_violation is not None:
        child_index, capsule, observed = wall_violation
        return InfeasibleBifurcationObstruction(
            region_id=region_id,
            phase=VascularGrowthPhase.SEED,
            lane=lane,
            predicate=VascularInfeasibilityPredicate.WALL_CONTAINMENT,
            required=config.wall_clearance,
            observed=observed,
            terminal_index=None,
            split_edge=(initial.parents[child_index], child_index),
            failing_segments=(
                vascular_segment_geometry(
                    capsule.left,
                    capsule.right,
                    capsule.radius,
                ),
            ),
        )
    if clearance_violation is not None:
        obligation, violation = clearance_violation
        candidate_capsule = indexed_capsules[violation.left_index][0]
        obligation_capsule = obligation.capsule_indices[
            violation.right_index
        ]
        left_capsule = stage.capsules[violation.left_index]
        right_capsule = obligation.capsules[violation.right_index]
        reserved_lane = (
            obligation.predicate
            is VascularInfeasibilityPredicate.RESERVED_LANE_CLEARANCE
        )
        return InfeasibleBifurcationObstruction(
            region_id=region_id,
            phase=VascularGrowthPhase.SEED,
            lane=lane,
            predicate=obligation.predicate,
            required=obligation.minimum_surface_clearance,
            observed=violation.observed,
            terminal_index=None,
            supply_capsule=(
                candidate_capsule
                if reserved_lane and lane is VascularLane.SUPPLY
                else obligation_capsule
                if reserved_lane
                and obligation.lane is VascularLane.SUPPLY
                else None
            ),
            return_capsule=(
                candidate_capsule
                if reserved_lane and lane is VascularLane.RETURN
                else obligation_capsule
                if reserved_lane
                and obligation.lane is VascularLane.RETURN
                else None
            ),
            candidate_point_index=(
                None if reserved_lane else terminal_ordinal
            ),
            failing_segments=(
                vascular_segment_geometry(
                    left_capsule.left,
                    left_capsule.right,
                    left_capsule.radius,
                ),
                vascular_segment_geometry(
                    right_capsule.left,
                    right_capsule.right,
                    right_capsule.radius,
                ),
            ),
        )
    return terminal_ordinal, initial


def _insert_terminal_cco(
    tree: _LocalVascularTree,
    site: tuple[float, float, float],
    terminal_flow: float,
    terminal_ordinal: int,
    terminal_kind: VascularNodeKind,
    containment_parts: tuple[dict, ...],
    region_id: str,
    config: SealedVascularConfig,
    lane: VascularLane,
    clearance_obligations: tuple[_CapsuleClearanceObligation, ...] = (),
    structural_cost_index: _StructuralCostIndex | None = None,
) -> _LocalVascularTree | InfeasibleBifurcationObstruction:
    batch = _insert_terminal_candidates(
        tree,
        site,
        terminal_flow,
        terminal_ordinal,
        terminal_kind,
        containment_parts,
        region_id,
        config,
        lane,
        clearance_obligations,
        structural_cost_index,
        search_all_edges=False,
    )
    if batch.candidates:
        return min(batch.candidates, key=lambda candidate: candidate.cost).tree
    return next(
        iter(batch.obstructions),
        _search_exhausted_obstruction(
            region_id,
            VascularGrowthPhase.INSERTION,
            lane,
            terminal_ordinal,
            batch.attempted_candidate_count,
        ),
    )


def _insert_terminal_candidates(
    tree: _LocalVascularTree,
    site: tuple[float, float, float],
    terminal_flow: float,
    terminal_ordinal: int,
    terminal_kind: VascularNodeKind,
    containment_parts: tuple[dict, ...],
    region_id: str,
    config: SealedVascularConfig,
    lane: VascularLane,
    clearance_obligations: tuple[_CapsuleClearanceObligation, ...] = (),
    structural_cost_index: _StructuralCostIndex | None = None,
    *,
    search_all_edges: bool,
) -> _InsertionBatch:
    ranked_candidates = _ranked_insertion_edges(tree, site)
    subtree_flows = _local_tree_subtree_flows(tree)
    minimum_terminal_flow = min(
        filter(lambda value: value > 0.0, tree.terminal_flows)
    )

    def evaluate(
        candidate_edges: tuple[tuple[int, int], ...],
        checked_ancestor_child_indices: frozenset[int] = frozenset(),
        escaped_ancestor_margins: tuple[tuple[int, float], ...] = (),
    ) -> _InsertionBatch:
        required_ancestor_child_indices = frozenset(
            ancestor_child_index
            for parent_index, _child_index in candidate_edges
            for ancestor_child_index in _ancestor_edge_child_indices(
                tree.parents, parent_index
            )
        )
        unchecked_ancestor_child_indices = (
            required_ancestor_child_indices
            - checked_ancestor_child_indices
        )
        unchecked_ancestor_capsules = _unchecked_ancestor_capsules(
            tree,
            unchecked_ancestor_child_indices,
            subtree_flows,
            terminal_flow,
            minimum_terminal_flow,
            config,
        )
        ancestor_margins = tuple(
            (ancestor_child_index, margin)
            for (ancestor_child_index, _capsule), margin in zip(
                unchecked_ancestor_capsules,
                _certified_segment_capsule_margins(
                    tuple(
                        capsule
                        for _child_index, capsule in unchecked_ancestor_capsules
                    ),
                    containment_parts,
                    min(3, config.capsule_samples),
                    config.wall_clearance,
                ),
                strict=True,
            )
        )
        newly_escaped_ancestor_margins = tuple(
            (ancestor_child_index, margin)
            for ancestor_child_index, margin in ancestor_margins
            if margin < config.wall_clearance
        )
        escaped = (
            *escaped_ancestor_margins,
            *newly_escaped_ancestor_margins,
        )
        results = _evaluate_bifurcation_candidates(
            tree,
            candidate_edges,
            site,
            terminal_flow,
            terminal_ordinal,
            terminal_kind,
            region_id,
            lane,
            containment_parts,
            config,
            clearance_obligations,
            structural_cost_index,
            subtree_flows=subtree_flows,
            minimum_terminal_flow=minimum_terminal_flow,
            escaped_ancestor_margins=escaped,
        )
        return _InsertionBatch(
            tuple(
                candidate
                for candidate in results
                if isinstance(candidate, FeasibleBifurcation)
            ),
            tuple(
                rejected.obstruction
                for rejected in results
                if isinstance(rejected, RejectedBifurcationCandidate)
            ),
            checked_ancestor_child_indices
            | unchecked_ancestor_child_indices,
            escaped,
            len(results),
        )

    if search_all_edges:
        return evaluate(ranked_candidates)
    nearest = evaluate(ranked_candidates[:4])
    if nearest.candidates:
        return nearest
    extended = evaluate(
        ranked_candidates[4:12],
        nearest.checked_ancestor_child_indices,
        nearest.escaped_ancestor_margins,
    )
    return _InsertionBatch(
        extended.candidates,
        (*nearest.obstructions, *extended.obstructions),
        extended.checked_ancestor_child_indices,
        extended.escaped_ancestor_margins,
        (
            nearest.attempted_candidate_count
            + extended.attempted_candidate_count
        ),
    )


def _build_local_tree_wider(
    corridor: tuple[tuple[float, float, float], ...],
    sites: tuple[tuple[float, float, float], ...],
    terminal_flow: float,
    terminal_kind: VascularNodeKind,
    containment_parts: tuple[dict, ...],
    region_id: str,
    config: SealedVascularConfig,
    lane: VascularLane,
    protected_corridor_node_count: int,
    clearance_obligations: tuple[_CapsuleClearanceObligation, ...],
    structural_cost_index: _StructuralCostIndex | None,
    state_budget: int,
    excluded_complete_trees: tuple[_LocalVascularTree, ...] = (),
) -> SearchResult[_LocalVascularTree]:
    if len(corridor) < 2 or not sites:
        obstruction = _search_exhausted_obstruction(
            region_id, VascularGrowthPhase.SEED, lane, None, 0
        )
        return ExhaustedSearch((obstruction,), 0, 0)
    seed_search = _initial_local_tree_candidates(
        corridor,
        sites,
        terminal_flow,
        terminal_kind,
        containment_parts,
        config,
        protected_corridor_node_count,
        clearance_obligations,
        region_id,
        lane,
    )
    queue = tuple(
        _InsertionContinuation(
            tree,
            tuple(
                indexed_site
                for indexed_site in enumerate(sites)
                if indexed_site[0] != ordinal
            ),
            _point_distance(corridor[-1], sites[ordinal]),
        )
        for ordinal, tree in sorted(
            seed_search.candidates,
            key=lambda candidate: (
                _point_distance(corridor[-1], sites[candidate[0]]),
                candidate[0],
            ),
        )
    )
    return _descend_insertion_queue(
        queue,
        terminal_flow,
        terminal_kind,
        containment_parts,
        region_id,
        config,
        lane,
        clearance_obligations,
        structural_cost_index,
        state_budget,
        excluded_complete_trees,
        obstructions=seed_search.obstructions,
        attempted_candidate_count=seed_search.attempted_candidate_count,
    )


def _descend_insertion_queue(
    queue: tuple[_InsertionContinuation, ...],
    terminal_flow: float,
    terminal_kind: VascularNodeKind,
    containment_parts: tuple[dict, ...],
    region_id: str,
    config: SealedVascularConfig,
    lane: VascularLane,
    clearance_obligations: tuple[_CapsuleClearanceObligation, ...],
    structural_cost_index: _StructuralCostIndex | None,
    state_budget: int,
    excluded_complete_trees: tuple[_LocalVascularTree, ...],
    *,
    obstructions: tuple[InfeasibleBifurcationObstruction, ...],
    evaluated_state_count: int = 0,
    attempted_candidate_count: int = 0,
) -> SearchResult[_LocalVascularTree]:
    if not queue:
        return ExhaustedSearch(
            obstructions,
            evaluated_state_count,
            attempted_candidate_count,
        )
    current = queue[0]
    if not current.remaining_sites and current.tree not in excluded_complete_trees:
        return AcceptedSearch(
            current.tree,
            evaluated_state_count,
            attempted_candidate_count,
        )
    if evaluated_state_count >= state_budget:
        return BudgetExhaustedSearch(
            obstructions,
            evaluated_state_count,
            attempted_candidate_count,
            len(queue),
        )
    if not current.remaining_sites:
        return _descend_insertion_queue(
            queue[1:],
            terminal_flow,
            terminal_kind,
            containment_parts,
            region_id,
            config,
            lane,
            clearance_obligations,
            structural_cost_index,
            state_budget,
            excluded_complete_trees,
            obstructions=obstructions,
            evaluated_state_count=evaluated_state_count + 1,
            attempted_candidate_count=attempted_candidate_count,
        )
    batches = tuple(
        (
            indexed_site,
            _insert_terminal_candidates(
                current.tree,
                indexed_site[1],
                terminal_flow,
                indexed_site[0],
                terminal_kind,
                containment_parts,
                region_id,
                config,
                lane,
                clearance_obligations,
                structural_cost_index,
                search_all_edges=True,
            ),
        )
        for indexed_site in current.remaining_sites
    )
    continuations = tuple(
        sorted(
            (
                _InsertionContinuation(
                    candidate.tree,
                    tuple(
                        remaining
                        for remaining in current.remaining_sites
                        if remaining[0] != indexed_site[0]
                    ),
                    current.lower_bound_cost + candidate.cost,
                )
                for indexed_site, batch in batches
                for candidate in batch.candidates
            ),
            key=lambda continuation: (
                continuation.lower_bound_cost,
                tuple(
                    ordinal for ordinal, _site in continuation.remaining_sites
                ),
                continuation.tree.parents,
            ),
        )
    )
    new_obstructions = (
        *obstructions,
        *tuple(
            obstruction
            for _indexed_site, batch in batches
            for obstruction in batch.obstructions
        ),
    )
    new_attempted_candidate_count = attempted_candidate_count + sum(
        batch.attempted_candidate_count for _indexed_site, batch in batches
    )
    return _descend_insertion_queue(
        (*continuations, *queue[1:]),
        terminal_flow,
        terminal_kind,
        containment_parts,
        region_id,
        config,
        lane,
        clearance_obligations,
        structural_cost_index,
        state_budget,
        excluded_complete_trees,
        obstructions=new_obstructions,
        evaluated_state_count=evaluated_state_count + 1,
        attempted_candidate_count=new_attempted_candidate_count,
    )


def _build_local_tree_pair_wider(
    supply_corridor: tuple[tuple[float, float, float], ...],
    return_corridor: tuple[tuple[float, float, float], ...],
    supply_sites: tuple[tuple[float, float, float], ...],
    return_sites: tuple[tuple[float, float, float], ...],
    terminal_flow: float,
    containment_parts: tuple[dict, ...],
    region_id: str,
    config: SealedVascularConfig,
    protected_corridor_node_count: int,
    exchange_clearance_obligations: tuple[
        _CapsuleClearanceObligation, ...
    ],
    structural_cost_index: _StructuralCostIndex | None,
    state_budget: int,
) -> SearchResult[tuple[_LocalVascularTree, _LocalVascularTree]]:
    reservation_search = _initial_local_tree_candidates(
        return_corridor,
        return_sites,
        terminal_flow,
        VascularNodeKind.RETURN_TERMINAL,
        containment_parts,
        config,
        protected_corridor_node_count,
        exchange_clearance_obligations,
        region_id,
        VascularLane.RETURN,
    )

    def descend_reservations(
        reservations: tuple[tuple[int, _LocalVascularTree], ...],
        accumulated_obstructions: tuple[
            InfeasibleBifurcationObstruction, ...
        ],
        evaluated_state_count: int,
        attempted_candidate_count: int,
    ) -> SearchResult[tuple[_LocalVascularTree, _LocalVascularTree]]:
        if not reservations:
            return ExhaustedSearch(
                accumulated_obstructions,
                evaluated_state_count,
                attempted_candidate_count,
            )
        if evaluated_state_count >= state_budget:
            return BudgetExhaustedSearch(
                accumulated_obstructions,
                evaluated_state_count,
                attempted_candidate_count,
                len(reservations),
            )
        reservation = reservations[0][1]
        supply_obligations = (
            _tree_clearance_obligation(
                reservation, config, VascularLane.RETURN
            ),
            *exchange_clearance_obligations,
        )

        def descend_supply_completions(
            excluded_supply_trees: tuple[_LocalVascularTree, ...],
            local_obstructions: tuple[
                InfeasibleBifurcationObstruction, ...
            ],
            local_evaluated_state_count: int,
            local_attempted_candidate_count: int,
        ) -> SearchResult[tuple[_LocalVascularTree, _LocalVascularTree]]:
            remaining_budget = state_budget - local_evaluated_state_count
            supply_result = _build_local_tree_wider(
                supply_corridor,
                supply_sites,
                terminal_flow,
                VascularNodeKind.SUPPLY_TERMINAL,
                containment_parts,
                region_id,
                config,
                VascularLane.SUPPLY,
                protected_corridor_node_count,
                supply_obligations,
                structural_cost_index,
                remaining_budget,
                excluded_supply_trees,
            )
            supply_evaluated = (
                local_evaluated_state_count
                + supply_result.evaluated_state_count
            )
            supply_attempted = (
                local_attempted_candidate_count
                + supply_result.attempted_candidate_count
            )
            supply_obstructions = (
                *local_obstructions,
                *(
                    ()
                    if isinstance(supply_result, AcceptedSearch)
                    else supply_result.obstructions
                ),
            )
            if isinstance(supply_result, BudgetExhaustedSearch):
                return BudgetExhaustedSearch(
                    supply_obstructions,
                    supply_evaluated,
                    supply_attempted,
                    supply_result.remaining_queue_size
                    + len(reservations[1:]),
                )
            if isinstance(supply_result, ExhaustedSearch):
                return descend_reservations(
                    reservations[1:],
                    supply_obstructions,
                    supply_evaluated,
                    supply_attempted,
                )
            return_result = _build_local_tree_wider(
                return_corridor,
                return_sites,
                terminal_flow,
                VascularNodeKind.RETURN_TERMINAL,
                containment_parts,
                region_id,
                config,
                VascularLane.RETURN,
                protected_corridor_node_count,
                (
                    _tree_clearance_obligation(
                        supply_result.value, config, VascularLane.SUPPLY
                    ),
                    *exchange_clearance_obligations,
                ),
                structural_cost_index,
                state_budget - supply_evaluated,
            )
            combined_evaluated = (
                supply_evaluated + return_result.evaluated_state_count
            )
            combined_attempted = (
                supply_attempted + return_result.attempted_candidate_count
            )
            if isinstance(return_result, AcceptedSearch):
                return AcceptedSearch(
                    (supply_result.value, return_result.value),
                    combined_evaluated,
                    combined_attempted,
                )
            combined_obstructions = (
                *supply_obstructions,
                *return_result.obstructions,
            )
            if isinstance(return_result, BudgetExhaustedSearch):
                return BudgetExhaustedSearch(
                    combined_obstructions,
                    combined_evaluated,
                    combined_attempted,
                    return_result.remaining_queue_size
                    + 1
                    + len(reservations[1:]),
                )
            return descend_supply_completions(
                (*excluded_supply_trees, supply_result.value),
                combined_obstructions,
                combined_evaluated,
                combined_attempted,
            )

        return descend_supply_completions(
            (),
            accumulated_obstructions,
            evaluated_state_count + 1,
            attempted_candidate_count,
        )

    return descend_reservations(
        tuple(
            sorted(
                reservation_search.candidates,
                key=lambda candidate: (
                    _point_distance(
                        return_corridor[-1], return_sites[candidate[0]]
                    ),
                    candidate[0],
                ),
            )
        ),
        reservation_search.obstructions,
        0,
        reservation_search.attempted_candidate_count,
    )


def _final_infeasible_obstruction(
    obstructions: tuple[InfeasibleBifurcationObstruction, ...],
    region_id: str,
    attempted_candidate_count: int,
) -> InfeasibleBifurcationObstruction:
    limiting = _limiting_bifurcation_obstruction(obstructions)
    selected = (
        limiting
        if limiting is not None
        else next(
            iter(obstructions),
            _search_exhausted_obstruction(
                region_id,
                VascularGrowthPhase.SEED,
                VascularLane.SUPPLY,
                None,
                attempted_candidate_count,
            ),
        )
    )
    return replace(
        selected, attempted_candidate_count=attempted_candidate_count
    )


def _vascular_search_budget_obstruction(
    region_id: str,
    search: BudgetExhaustedSearch,
    required_state_budget: int,
) -> VascularSearchBudgetObstruction:
    limiting = _limiting_bifurcation_obstruction(search.obstructions)
    selected = limiting or _search_exhausted_obstruction(
        region_id,
        VascularGrowthPhase.SEED,
        VascularLane.SUPPLY,
        None,
        search.attempted_candidate_count,
    )
    return VascularSearchBudgetObstruction(
        region_id=region_id,
        phase=selected.phase,
        lane=selected.lane,
        required_state_budget=required_state_budget,
        observed_evaluated_states=search.evaluated_state_count,
        remaining_queue_size=search.remaining_queue_size,
        limiting_predicate=selected.predicate,
        limiting_required=selected.required,
        limiting_observed=selected.observed,
        failing_segments=selected.failing_segments,
    )


def _limiting_bifurcation_obstruction(
    obstructions: tuple[InfeasibleBifurcationObstruction, ...],
) -> InfeasibleBifurcationObstruction | None:
    geometric = tuple(
        obstruction
        for obstruction in obstructions
        if obstruction.predicate
        is not VascularInfeasibilityPredicate.SEARCH_EXHAUSTED
    )
    return (
        None
        if not geometric
        else min(geometric, key=_limiting_obstruction_order)
    )


def _limiting_obstruction_order(
    obstruction: InfeasibleBifurcationObstruction,
) -> tuple[float, int, int, int, tuple[int, int]]:
    normalized_deficit = (
        obstruction.required - obstruction.observed
    ) / max(abs(obstruction.required), 1.0e-300)
    phase_order = (
        VascularGrowthPhase.SEED,
        VascularGrowthPhase.INSERTION,
        VascularGrowthPhase.ANCESTOR_GROWTH,
    )
    lane_order = (VascularLane.SUPPLY, VascularLane.RETURN)
    return (
        -normalized_deficit,
        phase_order.index(obstruction.phase),
        lane_order.index(obstruction.lane),
        -1 if obstruction.terminal_index is None else obstruction.terminal_index,
        obstruction.split_edge or (-1, -1),
    )


def _ranked_insertion_edges(
    tree: _LocalVascularTree,
    site: tuple[float, float, float],
) -> tuple[tuple[int, int], ...]:
    edges = tuple(
        (parent, child)
        for child, parent in enumerate(tree.parents)
        if parent >= 0
        and tree.kinds[child] is not VascularNodeKind.MACRO_CORRIDOR
    )
    return tuple(
        sorted(
            edges,
            key=lambda edge: _point_segment_distance(
                site, tree.positions[edge[0]], tree.positions[edge[1]]
            ),
        )
    )


def _unchecked_ancestor_capsules(
    tree: _LocalVascularTree,
    unchecked_ancestor_child_indices: frozenset[int],
    subtree_flows: tuple[float, ...],
    terminal_flow: float,
    minimum_terminal_flow: float,
    config: SealedVascularConfig,
) -> tuple[tuple[int, _SegmentCapsule], ...]:
    return tuple(
        (
            ancestor_child_index,
            _SegmentCapsule(
                tree.positions[tree.parents[ancestor_child_index]],
                tree.positions[ancestor_child_index],
                config.terminal_radius
                * (
                    (
                        subtree_flows[ancestor_child_index]
                        + terminal_flow
                    )
                    / minimum_terminal_flow
                )
                ** (1.0 / config.murray_exponent),
            ),
        )
        for ancestor_child_index in sorted(
            unchecked_ancestor_child_indices
        )
    )


def _tree_clearance_obligation(
    tree: _LocalVascularTree,
    config: SealedVascularConfig,
    lane: VascularLane,
) -> _CapsuleClearanceObligation:
    indexed_capsules = _local_tree_indexed_capsules(tree, config)
    return _CapsuleClearanceObligation(
        tuple(capsule for _child, capsule in indexed_capsules),
        tuple(child for child, _capsule in indexed_capsules),
        config.vessel_clearance,
        VascularInfeasibilityPredicate.RESERVED_LANE_CLEARANCE,
        lane,
    )
