"""Bifurcation candidate evaluation for CCO insertion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING

from golem.kernel.anatomy.geometry import (
    _certified_segment_capsule_margins,
    _point_distance,
)
from golem.kernel.anatomy.graph import (
    InfeasibleBifurcationObstruction,
    VascularGrowthPhase,
    VascularInfeasibilityPredicate,
    VascularLane,
    VascularNodeKind,
    vascular_segment_geometry,
)
from golem.kernel.anatomy.realize.carriers import (
    BifurcationCandidateResult,
    FeasibleBifurcation,
    _LocalVascularTree,
    RejectedBifurcationCandidate,
    _segment_structural_cost_multiplier,
)
from golem.kernel.anatomy.realize.clearance import (
    _capsule_sets_clearance_violation,
    _segment_sets_clearance_violation,
)
from golem.kernel.anatomy.realize.tree import (
    _ancestor_edge_child_indices,
    _local_tree_indexed_capsules,
    _weighted_geometric_median,
)

if TYPE_CHECKING:
    import numpy as np

    from golem.kernel.anatomy.realize.carriers import (
        _CapsuleClearanceObligation,
        _StructuralCostIndex,
    )
    from golem.kernel.anatomy.vocabulary import SealedVascularConfig


@dataclass(frozen=True)
class _BifurcationInvariants:
    tree: _LocalVascularTree
    terminal_flow: float
    terminal_ordinal: int
    terminal_kind: VascularNodeKind
    region_id: str
    lane: VascularLane
    containment_parts: tuple[dict, ...]
    config: SealedVascularConfig
    clearance_obligations: tuple[_CapsuleClearanceObligation, ...]
    structural_cost_index: _StructuralCostIndex | None
    minimum_terminal_flow: float


@dataclass(frozen=True)
class _BifurcationContext:
    split_edge: tuple[int, int]
    parent_index: int
    child_index: int
    child_flow: float
    parent: object
    child: object
    terminal: object
    weights: object
    bifurcation: object
    existing: tuple
    ancestor_child_indices: frozenset[int]


@dataclass(frozen=True)
class _PointPending:
    candidate: _LocalVascularTree
    indexed_local_changed_capsules: tuple
    local_changed_capsules: tuple
    weights: object
    lengths: tuple[float, ...]
    segments: tuple
    split_edge: tuple[int, int]
    candidate_point_index: int


def _bifurcation_context(
    tree: _LocalVascularTree,
    split_edge: tuple[int, int],
    site: tuple[float, float, float],
    terminal_flow: float,
    config: SealedVascularConfig,
    subtree_flows: tuple[float, ...],
) -> _BifurcationContext:
    import numpy as np

    parent_index, child_index = split_edge
    parent = np.asarray(tree.positions[parent_index], dtype=np.float64)
    child = np.asarray(tree.positions[child_index], dtype=np.float64)
    terminal = np.asarray(site, dtype=np.float64)
    child_flow = subtree_flows[child_index]
    exponent = 2.0 / config.murray_exponent
    weights = np.asarray(
        (
            (child_flow + terminal_flow) ** exponent,
            child_flow ** exponent,
            terminal_flow ** exponent,
        ),
        dtype=np.float64,
    )
    anchors = np.stack((parent, child, terminal))
    centroid = np.average(anchors, axis=0, weights=weights)
    raw = _weighted_geometric_median(anchors, weights, centroid, 10)
    bifurcation = 0.88 * raw + 0.12 * centroid
    existing = tuple(
        (tree.positions[parent_value], tree.positions[child_index_value])
        for child_index_value, parent_value in enumerate(tree.parents)
        if parent_value >= 0
        and (parent_value, child_index_value) != split_edge
    )
    ancestor_child_indices = _ancestor_edge_child_indices(
        tree.parents, parent_index
    )
    return _BifurcationContext(
        split_edge,
        parent_index,
        child_index,
        child_flow,
        parent,
        child,
        terminal,
        weights,
        bifurcation,
        existing,
        ancestor_child_indices,
    )


def _escaped_rejection(
    inv: _BifurcationInvariants,
    context: _BifurcationContext,
    escaped_ancestor_margins: tuple[tuple[int, float], ...],
) -> RejectedBifurcationCandidate | None:
    escaped_ancestor = next(
        (
            (child_index_value, margin)
            for child_index_value, margin in escaped_ancestor_margins
            if child_index_value in context.ancestor_child_indices
        ),
        None,
    )
    if escaped_ancestor is None:
        return None
    escaped_child_index, observed = escaped_ancestor
    escaped_geometry = tuple(
        vascular_segment_geometry(capsule.left, capsule.right, capsule.radius)
        for child_index, capsule in _local_tree_indexed_capsules(
            inv.tree,
            inv.config,
        )
        if child_index == escaped_child_index
    )
    return RejectedBifurcationCandidate(
        InfeasibleBifurcationObstruction(
            region_id=inv.region_id,
            phase=VascularGrowthPhase.ANCESTOR_GROWTH,
            lane=inv.lane,
            predicate=VascularInfeasibilityPredicate.WALL_CONTAINMENT,
            required=inv.config.wall_clearance,
            observed=observed,
            terminal_index=inv.terminal_ordinal,
            split_edge=(
                inv.tree.parents[escaped_child_index],
                escaped_child_index,
            ),
            failing_segments=escaped_geometry,
        ),
    )


def _fallback_points(context: _BifurcationContext) -> tuple:
    return tuple(
        context.bifurcation + fraction * (anchor - context.bifurcation)
        for fraction in (0.10, 0.20, 0.35, 0.50)
        for anchor in (context.parent, context.child, context.terminal)
    )


def _stage_bifurcation_point(
    inv: _BifurcationInvariants,
    context: _BifurcationContext,
    indexed_point: tuple[int, object],
) -> RejectedBifurcationCandidate | _PointPending:
    candidate_point_index, point = indexed_point
    parent = context.parent
    child = context.child
    terminal = context.terminal
    segments = (
        (tuple(parent), tuple(point)),
        (tuple(point), tuple(child)),
        (tuple(point), tuple(terminal)),
    )
    segment_radii = tuple(
        inv.config.terminal_radius
        * (flow / inv.minimum_terminal_flow)
        ** (1.0 / inv.config.murray_exponent)
        for flow in (
            context.child_flow + inv.terminal_flow,
            context.child_flow,
            inv.terminal_flow,
        )
    )
    lengths = tuple(
        _point_distance(left, right) for left, right in segments
    )
    minimum_length = min(lengths)
    if minimum_length < inv.config.minimum_segment_length:
        minimum_index = lengths.index(minimum_length)
        return RejectedBifurcationCandidate(
            InfeasibleBifurcationObstruction(
                region_id=inv.region_id,
                phase=VascularGrowthPhase.INSERTION,
                lane=inv.lane,
                predicate=VascularInfeasibilityPredicate.SEGMENT_LENGTH,
                required=inv.config.minimum_segment_length,
                observed=minimum_length,
                terminal_index=inv.terminal_ordinal,
                split_edge=context.split_edge,
                candidate_point_index=candidate_point_index,
                failing_segments=(
                    vascular_segment_geometry(
                        *segments[minimum_index],
                        segment_radii[minimum_index],
                    ),
                ),
            )
        )
    centerline_violation = _segment_sets_clearance_violation(
        segments,
        context.existing,
        inv.config.nonincident_centerline_separation,
    )
    if centerline_violation is not None:
        return RejectedBifurcationCandidate(
            InfeasibleBifurcationObstruction(
                region_id=inv.region_id,
                phase=VascularGrowthPhase.INSERTION,
                lane=inv.lane,
                predicate=(
                    VascularInfeasibilityPredicate.CENTERLINE_SEPARATION
                ),
                required=inv.config.nonincident_centerline_separation,
                observed=centerline_violation.observed,
                terminal_index=inv.terminal_ordinal,
                split_edge=context.split_edge,
                candidate_point_index=candidate_point_index,
                failing_segments=(
                    vascular_segment_geometry(
                        *segments[centerline_violation.left_index],
                        segment_radii[centerline_violation.left_index],
                    ),
                    vascular_segment_geometry(
                        *context.existing[centerline_violation.right_index],
                    ),
                ),
            )
        )
    bifurcation_index = len(inv.tree.positions)
    candidate = _candidate_bifurcation_tree(
        inv.tree,
        context.parent_index,
        context.child_index,
        bifurcation_index,
        point,
        terminal,
        inv.terminal_kind,
        inv.terminal_ordinal,
        inv.terminal_flow,
    )
    precheck = _candidate_capsules_precheck(
        candidate,
        context.ancestor_child_indices,
        context.child_index,
        bifurcation_index,
        inv.config,
        inv.clearance_obligations,
        inv.region_id,
        inv.lane,
        inv.terminal_ordinal,
        context.split_edge,
        candidate_point_index,
    )
    if isinstance(precheck, InfeasibleBifurcationObstruction):
        return RejectedBifurcationCandidate(precheck)
    indexed_local_changed_capsules, local_changed_capsules = precheck
    return _PointPending(
        candidate,
        indexed_local_changed_capsules,
        local_changed_capsules,
        context.weights,
        lengths,
        segments,
        context.split_edge,
        candidate_point_index,
    )


def _finish_bifurcation_point(
    inv: _BifurcationInvariants,
    pending: _PointPending,
    margins: tuple[float, ...],
) -> FeasibleBifurcation | RejectedBifurcationCandidate:
    wall_obstruction = _wall_obstruction(
        pending.indexed_local_changed_capsules,
        margins,
        inv.config,
        inv.region_id,
        inv.lane,
        inv.terminal_ordinal,
        pending.split_edge,
        pending.candidate_point_index,
    )
    if wall_obstruction is not None:
        return RejectedBifurcationCandidate(wall_obstruction)
    return FeasibleBifurcation(
        _bifurcation_cost(
            pending.weights,
            pending.lengths,
            pending.segments,
            inv.structural_cost_index,
        ),
        pending.candidate,
    )


def _resolve_points(
    inv: _BifurcationInvariants,
    staged: tuple,
) -> tuple:
    pendings = tuple(
        (position, point)
        for position, point in enumerate(staged)
        if isinstance(point, _PointPending)
    )
    margins = _certified_segment_capsule_margins(
        tuple(
            capsule
            for _position, point in pendings
            for capsule in point.local_changed_capsules
        ),
        inv.containment_parts,
        min(3, inv.config.capsule_samples),
        inv.config.wall_clearance,
    )
    offsets = (
        0,
        *accumulate(
            len(point.local_changed_capsules)
            for _position, point in pendings
        ),
    )
    finished = {
        position: _finish_bifurcation_point(
            inv, point, margins[offsets[order]:offsets[order + 1]]
        )
        for order, (position, point) in enumerate(pendings)
    }
    return tuple(
        finished.get(position, point)
        for position, point in enumerate(staged)
    )


def _selected_bifurcation(
    fallback_results: tuple,
    primary: BifurcationCandidateResult,
) -> BifurcationCandidateResult:
    feasible_fallbacks = tuple(
        result
        for result in fallback_results
        if isinstance(result, FeasibleBifurcation)
    )
    return (
        min(feasible_fallbacks, key=lambda candidate: candidate.cost)
        if feasible_fallbacks
        else primary
    )


def _evaluate_bifurcation_candidates(
    tree: _LocalVascularTree,
    candidate_edges: tuple[tuple[int, int], ...],
    site: tuple[float, float, float],
    terminal_flow: float,
    terminal_ordinal: int,
    terminal_kind: VascularNodeKind,
    region_id: str,
    lane: VascularLane,
    containment_parts: tuple[dict, ...],
    config: SealedVascularConfig,
    clearance_obligations: tuple[_CapsuleClearanceObligation, ...] = (),
    structural_cost_index: _StructuralCostIndex | None = None,
    *,
    subtree_flows: tuple[float, ...],
    minimum_terminal_flow: float,
    escaped_ancestor_margins: tuple[tuple[int, float], ...],
) -> tuple[BifurcationCandidateResult, ...]:
    inv = _BifurcationInvariants(
        tree,
        terminal_flow,
        terminal_ordinal,
        terminal_kind,
        region_id,
        lane,
        containment_parts,
        config,
        clearance_obligations,
        structural_cost_index,
        minimum_terminal_flow,
    )
    contexts = tuple(
        _bifurcation_context(
            tree, edge, site, terminal_flow, config, subtree_flows
        )
        for edge in candidate_edges
    )
    escaped = tuple(
        _escaped_rejection(inv, context, escaped_ancestor_margins)
        for context in contexts
    )
    primary_staged = tuple(
        None
        if rejection is not None
        else _stage_bifurcation_point(inv, context, (0, context.bifurcation))
        for context, rejection in zip(contexts, escaped, strict=True)
    )
    primary_results = tuple(
        rejection if rejection is not None else resolved
        for rejection, resolved in zip(
            escaped, _resolve_points(inv, primary_staged), strict=True
        )
    )
    fallback_points = tuple(
        _fallback_points(context)
        if rejection is None
        and not isinstance(primary, FeasibleBifurcation)
        else ()
        for context, rejection, primary in zip(
            contexts, escaped, primary_results, strict=True
        )
    )
    fallback_resolved = _resolve_points(
        inv,
        tuple(
            _stage_bifurcation_point(inv, context, indexed_point)
            for context, points in zip(
                contexts, fallback_points, strict=True
            )
            for indexed_point in enumerate(points, start=1)
        ),
    )
    fallback_offsets = (
        0,
        *accumulate(len(points) for points in fallback_points),
    )
    return tuple(
        primary
        if not points
        else _selected_bifurcation(
            fallback_resolved[
                fallback_offsets[position]:fallback_offsets[position + 1]
            ],
            primary,
        )
        for position, (points, primary) in enumerate(
            zip(fallback_points, primary_results, strict=True)
        )
    )


def _candidate_bifurcation_tree(
    tree: _LocalVascularTree,
    parent_index: int,
    child_index: int,
    bifurcation_index: int,
    point,
    terminal,
    terminal_kind: VascularNodeKind,
    terminal_ordinal: int,
    terminal_flow: float,
) -> _LocalVascularTree:
    parents = tuple(
        bifurcation_index if index == child_index else parent_value
        for index, parent_value in enumerate(tree.parents)
    )
    return _LocalVascularTree(
        positions=(
            *tree.positions,
            tuple(map(float, point)),
            tuple(map(float, terminal)),
        ),
        parents=(*parents, parent_index, bifurcation_index),
        kinds=(*tree.kinds, VascularNodeKind.BIFURCATION, terminal_kind),
        terminal_flows=(*tree.terminal_flows, 0.0, terminal_flow),
        terminal_ordinals=(
            *tree.terminal_ordinals,
            None,
            terminal_ordinal,
        ),
    )


def _candidate_capsules_precheck(
    candidate: _LocalVascularTree,
    ancestor_child_indices: frozenset[int],
    child_index: int,
    bifurcation_index: int,
    config: SealedVascularConfig,
    clearance_obligations: tuple[_CapsuleClearanceObligation, ...],
    region_id: str,
    lane: VascularLane,
    terminal_ordinal: int,
    split_edge: tuple[int, int],
    candidate_point_index: int,
) -> InfeasibleBifurcationObstruction | tuple:
    indexed_candidate_capsules = _local_tree_indexed_capsules(
        candidate, config
    )
    all_candidate_capsules = tuple(
        capsule for _child, capsule in indexed_candidate_capsules
    )
    local_changed_child_indices = frozenset(
        (
            child_index,
            bifurcation_index,
            len(candidate.positions) - 1,
        )
    )
    changed_child_indices = frozenset(
        (
            *ancestor_child_indices,
            *local_changed_child_indices,
        )
    )
    indexed_changed_capsules = tuple(
        (child, capsule)
        for child, capsule in indexed_candidate_capsules
        if child in changed_child_indices
    )
    changed_capsules = tuple(
        capsule for _child, capsule in indexed_changed_capsules
    )
    indexed_local_changed_capsules = tuple(
        (child, capsule)
        for child, capsule in indexed_candidate_capsules
        if child in local_changed_child_indices
    )
    local_changed_capsules = tuple(
        capsule for _child, capsule in indexed_local_changed_capsules
    )
    self_violation = _capsule_sets_clearance_violation(
        changed_capsules,
        all_candidate_capsules,
        config.vessel_clearance,
    )
    if self_violation is not None:
        changed_child = indexed_changed_capsules[
            self_violation.left_index
        ][0]
        left_capsule = changed_capsules[self_violation.left_index]
        right_capsule = all_candidate_capsules[self_violation.right_index]
        ancestor_growth = changed_child in ancestor_child_indices
        return InfeasibleBifurcationObstruction(
            region_id=region_id,
            phase=(
                VascularGrowthPhase.ANCESTOR_GROWTH
                if ancestor_growth
                else VascularGrowthPhase.INSERTION
            ),
            lane=lane,
            predicate=VascularInfeasibilityPredicate.SELF_CLEARANCE,
            required=config.vessel_clearance,
            observed=self_violation.observed,
            terminal_index=terminal_ordinal,
            split_edge=(
                (candidate.parents[changed_child], changed_child)
                if ancestor_growth
                else split_edge
            ),
            candidate_point_index=(
                None if ancestor_growth else candidate_point_index
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
    obligation_violation = next(
        (
            (obligation, violation)
            for obligation in clearance_obligations
            for violation in (
                _capsule_sets_clearance_violation(
                    changed_capsules,
                    obligation.capsules,
                    obligation.minimum_surface_clearance,
                ),
            )
            if violation is not None
        ),
        None,
    )
    if obligation_violation is not None:
        obligation, violation = obligation_violation
        changed_child = indexed_changed_capsules[violation.left_index][0]
        obligation_child = obligation.capsule_indices[
            violation.right_index
        ]
        left_capsule = changed_capsules[violation.left_index]
        right_capsule = obligation.capsules[violation.right_index]
        ancestor_growth = changed_child in ancestor_child_indices
        reserved_lane = (
            obligation.predicate
            is VascularInfeasibilityPredicate.RESERVED_LANE_CLEARANCE
        )
        return InfeasibleBifurcationObstruction(
            region_id=region_id,
            phase=(
                VascularGrowthPhase.ANCESTOR_GROWTH
                if ancestor_growth
                else VascularGrowthPhase.INSERTION
            ),
            lane=lane,
            predicate=obligation.predicate,
            required=obligation.minimum_surface_clearance,
            observed=violation.observed,
            terminal_index=terminal_ordinal,
            supply_capsule=(
                changed_child
                if reserved_lane and lane is VascularLane.SUPPLY
                else obligation_child
                if reserved_lane
                and obligation.lane is VascularLane.SUPPLY
                else None
            ),
            return_capsule=(
                changed_child
                if reserved_lane and lane is VascularLane.RETURN
                else obligation_child
                if reserved_lane
                and obligation.lane is VascularLane.RETURN
                else None
            ),
            split_edge=(
                (candidate.parents[changed_child], changed_child)
                if ancestor_growth
                else None
                if reserved_lane
                else split_edge
            ),
            candidate_point_index=(
                None
                if reserved_lane or ancestor_growth
                else candidate_point_index
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
    return indexed_local_changed_capsules, local_changed_capsules


def _wall_obstruction(
    indexed_local_changed_capsules: tuple,
    local_margins: tuple[float, ...],
    config: SealedVascularConfig,
    region_id: str,
    lane: VascularLane,
    terminal_ordinal: int,
    split_edge: tuple[int, int],
    candidate_point_index: int,
) -> InfeasibleBifurcationObstruction | None:
    wall_violation = next(
        (
            (child, capsule, margin)
            for (child, capsule), margin in zip(
                indexed_local_changed_capsules,
                local_margins,
                strict=True,
            )
            if margin < config.wall_clearance
        ),
        None,
    )
    if wall_violation is None:
        return None
    _child, capsule, observed = wall_violation
    return InfeasibleBifurcationObstruction(
        region_id=region_id,
        phase=VascularGrowthPhase.INSERTION,
        lane=lane,
        predicate=VascularInfeasibilityPredicate.WALL_CONTAINMENT,
        required=config.wall_clearance,
        observed=observed,
        terminal_index=terminal_ordinal,
        split_edge=split_edge,
        candidate_point_index=candidate_point_index,
        failing_segments=(
            vascular_segment_geometry(
                capsule.left,
                capsule.right,
                capsule.radius,
            ),
        ),
    )


def _bifurcation_cost(
    weights,
    lengths: tuple[float, ...],
    segments: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], ...
    ],
    structural_cost_index: _StructuralCostIndex | None,
) -> float:
    return (
        math.fsum(
            weight * length
            for weight, length in zip(weights, lengths)
        )
        if structural_cost_index is None
        else math.fsum(
            weight
            * length
            * _segment_structural_cost_multiplier(
                left, right, structural_cost_index
            )
            for weight, length, (left, right) in zip(
                weights, lengths, segments
            )
        )
    )
