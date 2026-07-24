"""Vascular geometry clearance validation and edge-pair checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from golem.kernel.anatomy.geometry import (
    _bilateral_symmetry_validation,
    _certified_segment_capsule_margins,
    _segment_distance,
    _segment_distances_batch,
    _segments_share_endpoint_batch,
)
from golem.kernel.anatomy.graph import (
    CapsuleEscapeObstruction,
    VascularIntersectionObstruction,
    vascular_edge_geometry,
)
from golem.kernel.anatomy.realize.carriers import (
    _IndexedClearanceViolation,
    _SegmentCapsule,
)

if TYPE_CHECKING:
    import numpy as np

    from golem.kernel.anatomy.graph import (
        ClosedVascularGraph,
        VascularEdge,
        VascularNode,
        VasculatureObstruction,
    )
    from golem.kernel.anatomy.vocabulary import SealedVascularConfig


_packed_segments_memo: dict[int, tuple] = {}
_packed_capsules_memo: dict[int, tuple] = {}


@dataclass(frozen=True)
class _VascularGeometryValidation:
    obstructions: tuple[VasculatureObstruction, ...]
    minimum_capsule_margin: float
    checked_nonincident_pair_count: int
    maximum_symmetry_error: float


def _packed_segments(segments: tuple):
    import numpy as np

    return (
        (cached[1], cached[2])
        if (cached := _packed_segments_memo.get(id(segments))) is not None
        else _packed_segments_memo.setdefault(
            id(segments),
            (
                segments,
                np.asarray(tuple(segment[0] for segment in segments)),
                np.asarray(tuple(segment[1] for segment in segments)),
            ),
        )[1:]
    )


def _packed_capsules(capsules: tuple):
    import numpy as np

    return (
        (cached[1], cached[2], cached[3])
        if (cached := _packed_capsules_memo.get(id(capsules))) is not None
        else _packed_capsules_memo.setdefault(
            id(capsules),
            (
                capsules,
                np.asarray(tuple(capsule.left for capsule in capsules)),
                np.asarray(tuple(capsule.right for capsule in capsules)),
                np.asarray(tuple(capsule.radius for capsule in capsules)),
            ),
        )[1:]
    )


def _validate_vascular_geometry(
    graph: ClosedVascularGraph,
    body_parts: tuple[dict, ...],
    config: SealedVascularConfig,
) -> _VascularGeometryValidation:
    by_id = graph.node_by_id
    edge_capsules = tuple(
        _SegmentCapsule(
            by_id[edge.source_node_id].position,
            by_id[edge.target_node_id].position,
            edge.radius,
        )
        for edge in graph.edges
    )
    margins = tuple(
        (
            edge,
            margin,
        )
        for edge, margin in zip(
            graph.edges,
            _certified_segment_capsule_margins(
                edge_capsules,
                body_parts,
                config.capsule_samples,
                config.wall_clearance,
            ),
            strict=True,
        )
    )
    escapes: tuple[VasculatureObstruction, ...] = tuple(
        CapsuleEscapeObstruction(
            edge.edge_id,
            margin,
            (vascular_edge_geometry(edge, by_id),),
        )
        for edge, margin in margins
        if margin < config.wall_clearance
    )
    possible_pairs = _possible_vascular_edge_pairs(graph.edges, by_id, config)
    nonincident_pair_clearances = tuple(
        (left, right, _edge_clearance(left, right, by_id))
        for left, right in possible_pairs
        if not _vascular_edges_are_incident(left, right)
    )
    intersections: tuple[VasculatureObstruction, ...] = tuple(
        VascularIntersectionObstruction(
            left.edge_id,
            right.edge_id,
            clearance,
            (
                vascular_edge_geometry(left, by_id),
                vascular_edge_geometry(right, by_id),
            ),
        )
        for left, right, clearance in nonincident_pair_clearances
        if clearance < config.vessel_clearance
    )
    symmetry, maximum_symmetry_error = _bilateral_symmetry_validation(
        graph, 1.0e-12
    )
    return _VascularGeometryValidation(
        obstructions=(*escapes, *intersections, *symmetry),
        minimum_capsule_margin=min(
            map(lambda item: item[1], margins), default=math.inf
        ),
        checked_nonincident_pair_count=len(nonincident_pair_clearances),
        maximum_symmetry_error=maximum_symmetry_error,
    )


def _edge_clearance(
    left: VascularEdge,
    right: VascularEdge,
    node_by_id: dict[str, VascularNode],
) -> float:
    return _segment_distance(
        node_by_id[left.source_node_id].position,
        node_by_id[left.target_node_id].position,
        node_by_id[right.source_node_id].position,
        node_by_id[right.target_node_id].position,
    ) - left.radius - right.radius


def _vascular_edges_are_incident(left: VascularEdge, right: VascularEdge) -> bool:
    return bool(
        frozenset((left.source_node_id, left.target_node_id))
        & frozenset((right.source_node_id, right.target_node_id))
    )


def _possible_vascular_edge_pairs(
    edges: tuple[VascularEdge, ...],
    node_by_id: dict[str, VascularNode],
    config: SealedVascularConfig,
) -> tuple[tuple[VascularEdge, VascularEdge], ...]:
    import numpy as np

    starts = np.asarray(
        tuple(node_by_id[edge.source_node_id].position for edge in edges),
        dtype=np.float64,
    )
    ends = np.asarray(
        tuple(node_by_id[edge.target_node_id].position for edge in edges),
        dtype=np.float64,
    )
    midpoints = (starts + ends) / 2.0
    half_lengths = np.linalg.norm(ends - starts, axis=1) / 2.0
    radii = np.asarray(tuple(edge.radius for edge in edges), dtype=np.float64)
    midpoint_delta = midpoints[:, None, :] - midpoints[None, :, :]
    midpoint_distance_squared = np.einsum(
        "ijk,ijk->ij", midpoint_delta, midpoint_delta
    )
    reach = (
        half_lengths[:, None]
        + half_lengths[None, :]
        + radii[:, None]
        + radii[None, :]
        + config.vessel_clearance
    )
    row, column = np.nonzero(
        np.triu(midpoint_distance_squared <= reach * reach, k=1)
    )
    return tuple((edges[int(left)], edges[int(right)]) for left, right in zip(row, column))


def _segment_sets_violate_clearance(
    left_segments: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], ...
    ],
    right_segments: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], ...
    ],
    clearance: float,
) -> bool:
    return (
        _segment_sets_clearance_violation(
            left_segments, right_segments, clearance
        )
        is not None
    )


def _segment_sets_clearance_violation(
    left_segments: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], ...
    ],
    right_segments: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], ...
    ],
    clearance: float,
) -> _IndexedClearanceViolation | None:
    import numpy as np

    if not left_segments or not right_segments:
        return None
    left_start, left_end = _packed_segments(left_segments)
    right_start, right_end = _packed_segments(right_segments)
    left_mid = (left_start + left_end) / 2.0
    right_mid = (right_start + right_end) / 2.0
    left_half = np.linalg.norm(left_end - left_start, axis=1) / 2.0
    right_half = np.linalg.norm(right_end - right_start, axis=1) / 2.0
    delta = left_mid[:, None, :] - right_mid[None, :, :]
    distance_squared = np.einsum("ijk,ijk->ij", delta, delta)
    reach = left_half[:, None] + right_half[None, :] + clearance
    left_index, right_index = np.nonzero(distance_squared <= reach * reach)
    candidate_left_start = left_start[left_index]
    candidate_left_end = left_end[left_index]
    candidate_right_start = right_start[right_index]
    candidate_right_end = right_end[right_index]
    observed = _segment_distances_batch(
        candidate_left_start,
        candidate_left_end,
        candidate_right_start,
        candidate_right_end,
    )
    shares = _segments_share_endpoint_batch(
        candidate_left_start,
        candidate_left_end,
        candidate_right_start,
        candidate_right_end,
    )
    violating = np.nonzero(~shares & (observed < clearance))[0]
    return (
        _IndexedClearanceViolation(
            int(left_index[violating[0]]),
            int(right_index[violating[0]]),
            float(observed[violating[0]]),
        )
        if violating.size
        else None
    )


def _capsule_sets_violate_clearance(
    left_capsules: tuple[_SegmentCapsule, ...],
    right_capsules: tuple[_SegmentCapsule, ...],
    minimum_surface_clearance: float,
) -> bool:
    return (
        _capsule_sets_clearance_violation(
            left_capsules,
            right_capsules,
            minimum_surface_clearance,
        )
        is not None
    )


def _capsule_sets_clearance_violation(
    left_capsules: tuple[_SegmentCapsule, ...],
    right_capsules: tuple[_SegmentCapsule, ...],
    minimum_surface_clearance: float,
) -> _IndexedClearanceViolation | None:
    """Measure the first exact variable-radius capsule violation."""
    import numpy as np

    if not left_capsules or not right_capsules:
        return None
    left_start, left_end, left_radius = _packed_capsules(left_capsules)
    right_start, right_end, right_radius = _packed_capsules(right_capsules)
    left_mid = (left_start + left_end) / 2.0
    right_mid = (right_start + right_end) / 2.0
    left_half = np.linalg.norm(left_end - left_start, axis=1) / 2.0
    right_half = np.linalg.norm(right_end - right_start, axis=1) / 2.0
    delta = left_mid[:, None, :] - right_mid[None, :, :]
    distance_squared = np.einsum("ijk,ijk->ij", delta, delta)
    reach = (
        left_half[:, None]
        + right_half[None, :]
        + left_radius[:, None]
        + right_radius[None, :]
        + minimum_surface_clearance
    )
    left_index, right_index = np.nonzero(distance_squared <= reach * reach)
    candidate_left_start = left_start[left_index]
    candidate_left_end = left_end[left_index]
    candidate_right_start = right_start[right_index]
    candidate_right_end = right_end[right_index]
    observed = (
        _segment_distances_batch(
            candidate_left_start,
            candidate_left_end,
            candidate_right_start,
            candidate_right_end,
        )
        - left_radius[left_index]
        - right_radius[right_index]
    )
    shares = _segments_share_endpoint_batch(
        candidate_left_start,
        candidate_left_end,
        candidate_right_start,
        candidate_right_end,
    )
    violating = np.nonzero(~shares & (observed < minimum_surface_clearance))[0]
    return (
        _IndexedClearanceViolation(
            int(left_index[violating[0]]),
            int(right_index[violating[0]]),
            float(observed[violating[0]]),
        )
        if violating.size
        else None
    )
