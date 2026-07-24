"""Anatomy-local capsule and segment distance primitives."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import accumulate
from typing import TYPE_CHECKING

from golem.kernel.anatomy.graph import (
    SymmetryMismatchObstruction,
    VascularSegmentGeometry,
    VascularSymmetryPredicate,
    VascularSymmetryWitness,
)
from golem.kernel.anatomy.realize.carriers import _SegmentCapsule

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from golem.kernel.anatomy.graph import ClosedVascularGraph
    from golem.kernel.engine.types import Part


_decoded_part_memo: dict[int, tuple[dict, "Part"]] = {}
_prepared_part_memo: dict[tuple[int, bool], tuple[dict, "Callable"]] = {}
_sample_grid_memo: dict[int, "np.ndarray"] = {}


def _sample_grid(sample_count: int):
    import numpy as np

    return (
        cached
        if (cached := _sample_grid_memo.get(sample_count)) is not None
        else _sample_grid_memo.setdefault(
            sample_count, np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
        )
    )


def _decoded_part(part: dict) -> "Part":
    from golem.kernel.engine.algebra import decode_part, require_accepted

    return (
        cached[1]
        if (cached := _decoded_part_memo.get(id(part))) is not None
        else _decoded_part_memo.setdefault(
            id(part), (part, require_accepted(decode_part(part)))
        )[1]
    )


def _prepared_part(part: dict, mirrored: bool) -> "Callable":
    from golem.kernel.engine.algebra import prepared_part_sdf

    return (
        cached[1]
        if (cached := _prepared_part_memo.get((id(part), mirrored))) is not None
        else _prepared_part_memo.setdefault(
            (id(part), mirrored),
            (part, prepared_part_sdf(_decoded_part(part), mirrored)),
        )[1]
    )


def _mirror_node_id(node_id: str) -> str | None:
    return (
        node_id.replace(":positive:", ":negative:")
        if ":positive:" in node_id
        else node_id.replace(":negative:", ":positive:")
        if ":negative:" in node_id
        else None
    )


def _bilateral_symmetry_validation(
    graph: ClosedVascularGraph,
    tolerance: float,
) -> tuple[tuple[SymmetryMismatchObstruction, ...], float]:
    by_id = graph.node_by_id
    mirrored_nodes = tuple(
        (node, mirror_node_id)
        for node in graph.nodes
        if (mirror_node_id := _mirror_node_id(node.node_id)) is not None
    )
    missing_pair_obstructions = tuple(
        SymmetryMismatchObstruction(
            region_id=node.region_id or "bilateral",
            predicate=VascularSymmetryPredicate.MIRROR_PAIRING,
            required=1.0,
            observed=0.0,
            witness=VascularSymmetryWitness(node.node_id, mirror_node_id),
        )
        for node, mirror_node_id in mirrored_nodes
        if mirror_node_id not in by_id
    )
    paired_nodes = tuple(
        (node, by_id[mirror_node_id])
        for node, mirror_node_id in mirrored_nodes
        if ":positive:" in node.node_id and mirror_node_id in by_id
    )
    measurements = tuple(
        (
            node,
            mirror,
            _point_distance(
                (-node.position[0], node.position[1], node.position[2]),
                mirror.position,
            ),
        )
        for node, mirror in paired_nodes
    )
    reflection_obstructions = tuple(
        SymmetryMismatchObstruction(
            region_id=node.region_id or "bilateral",
            predicate=VascularSymmetryPredicate.REFLECTION_DISTANCE,
            required=tolerance,
            observed=error,
            witness=VascularSymmetryWitness(node.node_id, mirror.node_id),
        )
        for node, mirror, error in measurements
        if error > tolerance
    )
    return (
        (*missing_pair_obstructions, *reflection_obstructions),
        max(map(lambda measurement: measurement[2], measurements), default=0.0),
    )


def _union_sdf(points, parts: tuple[dict, ...]):
    import numpy as np

    evaluation_points = np.asarray(points)
    instances = tuple(
        (part, mirrored)
        for part in parts
        for mirrored in ((False, True) if part.get("mirror", False) else (False,))
    )
    fields = tuple(
        _prepared_part(part, mirrored)(evaluation_points)
        for part, mirrored in instances
    )
    return np.minimum.reduce(fields) if fields else np.full(len(points), np.inf)


def _bone_address(part_address: str) -> str | None:
    components = tuple(filter(None, part_address.split("/")))
    return (
        "/".join(components[:2])
        if len(components) >= 2 and components[0] == "skeleton"
        else None
    )


def _endpoint_geometry(
    position: tuple[float, float, float],
    parts: tuple[dict, ...],
    provenance: dict[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    import numpy as np

    point = np.asarray((position,), dtype=np.float64)
    instances = tuple(
        (part, mirrored)
        for part in parts
        for mirrored in ((False, True) if part.get("mirror", False) else (False,))
    )
    distances = tuple(
        (float(_prepared_part(part, mirrored)(point)[0]), part)
        for part, mirrored in instances
    )
    minimum = min((distance for distance, _part in distances), default=math.inf)
    nearest_parts = tuple(
        part
        for distance, part in distances
        if math.isclose(distance, minimum, rel_tol=1.0e-12, abs_tol=1.0e-12)
    )
    part_addresses = tuple(
        dict.fromkeys(
            address
            for part in nearest_parts
            for address in (provenance.get(str(part.get("id"))),)
            if isinstance(address, str)
        )
    )
    return (
        tuple(
            dict.fromkeys(
                bone_address
                for address in part_addresses
                for bone_address in (_bone_address(address),)
                if bone_address is not None
            )
        ),
        part_addresses,
    )


def _localized_segment_geometry(
    segment: VascularSegmentGeometry,
    parts: tuple[dict, ...],
    provenance: dict[str, str],
) -> VascularSegmentGeometry:
    source_bones, source_parts = _endpoint_geometry(
        segment.source_position,
        parts,
        provenance,
    )
    target_bones, target_parts = _endpoint_geometry(
        segment.target_position,
        parts,
        provenance,
    )
    return replace(
        segment,
        source_host_bone_addresses=source_bones,
        source_host_part_addresses=source_parts,
        target_host_bone_addresses=target_bones,
        target_host_part_addresses=target_parts,
    )


def _segment_capsule_margin(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    radius: float,
    parts: tuple[dict, ...],
    sample_count: int,
) -> float:
    import numpy as np

    start = np.asarray(left, dtype=np.float64)
    end = np.asarray(right, dtype=np.float64)
    t = _sample_grid(sample_count)
    samples = (1.0 - t[:, None]) * start + t[:, None] * end
    return -float(np.max(_union_sdf(samples, parts))) - radius


def _certified_segment_capsule_margin(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    radius: float,
    parts: tuple[dict, ...],
    minimum_samples: int,
    required_margin: float,
    maximum_samples: int = 8193,
) -> float:
    return _certified_segment_capsule_margins(
        (_SegmentCapsule(left, right, radius),),
        parts,
        minimum_samples,
        required_margin,
        maximum_samples,
    )[0]


def _certified_segment_capsule_margins(
    capsules: tuple[_SegmentCapsule, ...],
    parts: tuple[dict, ...],
    minimum_samples: int,
    required_margin: float,
    maximum_samples: int = 8193,
) -> tuple[float, ...]:
    """Conservative whole-segment capsule margin from a Lipschitz bound.

    The analytic union SDF is treated as a 1-Lipschitz field.  Between two
    evenly spaced centerline samples its clearance can fall by at most half a
    sample interval, so ``sampled_margin - spacing/2`` bounds every point on
    the segment rather than blessing nine isolated points and hoping.
    """
    if not capsules:
        return ()

    def descend(
        pending: tuple[tuple[int, _SegmentCapsule, int], ...],
        certified: tuple[tuple[int, float], ...],
    ) -> tuple[float, ...]:
        if not pending:
            return tuple(
                margin for _index, margin in sorted(certified)
            )
        sampled_fields = _capsule_sample_fields(pending, parts)
        sections = _capsule_margin_sections(pending, sampled_fields)
        local_certificates = tuple(
            (
                index,
                sampled_margin
                - 0.5 * length / max(sample_count - 1, 1),
            )
            for (
                index,
                _capsule,
                sample_count,
                sampled_margin,
                length,
            ) in sections
        )
        finished_indices = frozenset(
            index
            for (
                index,
                _capsule,
                sample_count,
                sampled_margin,
                _length,
            ), (_certificate_index, certificate) in zip(
                sections, local_certificates, strict=True
            )
            if certificate >= required_margin
            or sampled_margin <= required_margin
            or sample_count >= maximum_samples
        )
        retries = tuple(
            (
                index,
                capsule,
                _refined_sample_count(
                    sample_count,
                    sampled_margin,
                    length,
                    required_margin,
                    maximum_samples,
                ),
            )
            for (
                index,
                capsule,
                sample_count,
                sampled_margin,
                length,
            ) in sections
            if index not in finished_indices
        )
        return descend(
            retries,
            (
                *certified,
                *tuple(
                    certificate
                    for certificate in local_certificates
                    if certificate[0] in finished_indices
                ),
            ),
        )

    return descend(
        tuple(
            (index, capsule, max(2, minimum_samples))
            for index, capsule in enumerate(capsules)
        ),
        (),
    )


def _capsule_sample_fields(
    pending: tuple[tuple[int, _SegmentCapsule, int], ...],
    parts: tuple[dict, ...],
) -> tuple:
    import numpy as np

    sample_blocks = tuple(
        (
            (1.0 - parameter[:, None])
            * np.asarray(capsule.left, dtype=np.float64)
            + parameter[:, None]
            * np.asarray(capsule.right, dtype=np.float64)
        )
        for _index, capsule, sample_count in pending
        for parameter in (_sample_grid(sample_count),)
    )
    split_offsets = tuple(
        accumulate(block.shape[0] for block in sample_blocks)
    )
    return tuple(
        np.split(
            _union_sdf(np.concatenate(sample_blocks), parts),
            split_offsets[:-1],
        )
    )


def _capsule_margin_sections(
    pending: tuple[tuple[int, _SegmentCapsule, int], ...],
    sampled_fields: tuple,
) -> tuple[tuple[int, _SegmentCapsule, int, float, float], ...]:
    import numpy as np

    return tuple(
        (
            index,
            capsule,
            sample_count,
            -float(np.max(field)) - capsule.radius,
            _point_distance(capsule.left, capsule.right),
        )
        for (index, capsule, sample_count), field in zip(
            pending, sampled_fields
        )
    )


def _refined_sample_count(
    sample_count: int,
    sampled_margin: float,
    length: float,
    required_margin: float,
    maximum_samples: int,
) -> int:
    return min(
        maximum_samples,
        max(
            2 * sample_count - 1,
            int(
                math.ceil(
                    length / (2.0 * (sampled_margin - required_margin))
                )
            )
            + 1,
        ),
    )


def _point_distance(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right)))


def _point_segment_distance(point, left, right) -> float:
    px, py, pz = point
    ax, ay, az = left
    bx, by, bz = right
    dx, dy, dz = bx - ax, by - ay, bz - az
    denominator = dx * dx + dy * dy + dz * dz
    t = (
        0.0
        if denominator <= 1.0e-30
        else min(
            max(
                ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / denominator,
                0.0,
            ),
            1.0,
        )
    )
    ex, ey, ez = px - (ax + t * dx), py - (ay + t * dy), pz - (az + t * dz)
    return math.sqrt(ex * ex + ey * ey + ez * ez)


def _segment_distance(a0, a1, b0, b1) -> float:
    """Shortest distance between two closed 3D segments."""
    ux, uy, uz = a1[0] - a0[0], a1[1] - a0[1], a1[2] - a0[2]
    vx, vy, vz = b1[0] - b0[0], b1[1] - b0[1], b1[2] - b0[2]
    wx, wy, wz = a0[0] - b0[0], a0[1] - b0[1], a0[2] - b0[2]
    a = ux * ux + uy * uy + uz * uz
    b = ux * vx + uy * vy + uz * vz
    c = vx * vx + vy * vy + vz * vz
    d = ux * wx + uy * wy + uz * wz
    e = vx * wx + vy * wy + vz * wz
    denominator = a * c - b * b
    s_numerator, s_denominator = denominator, denominator
    t_numerator, t_denominator = denominator, denominator
    if denominator < 1.0e-30:
        s_numerator, s_denominator = 0.0, 1.0
        t_numerator, t_denominator = e, c
    else:
        s_numerator, t_numerator = b * e - c * d, a * e - b * d
        if s_numerator < 0.0:
            s_numerator, t_numerator, t_denominator = 0.0, e, c
        elif s_numerator > s_denominator:
            s_numerator, t_numerator, t_denominator = s_denominator, e + b, c
    if t_numerator < 0.0:
        t_numerator = 0.0
        s_numerator = min(max(-d, 0.0), a)
        s_denominator = a
    elif t_numerator > t_denominator:
        t_numerator = t_denominator
        s_numerator = min(max(b - d, 0.0), a)
        s_denominator = a
    sc = 0.0 if abs(s_numerator) < 1.0e-30 else s_numerator / s_denominator
    tc = 0.0 if abs(t_numerator) < 1.0e-30 else t_numerator / t_denominator
    dx = wx + sc * ux - tc * vx
    dy = wy + sc * uy - tc * vy
    dz = wz + sc * uz - tc * vz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _segments_share_endpoint(a0, a1, b0, b1) -> bool:
    tolerance_squared = 1.0e-24
    return any(
        math.fsum((
            (left[0] - right[0]) ** 2,
            (left[1] - right[1]) ** 2,
            (left[2] - right[2]) ** 2,
        ))
        <= tolerance_squared
        for left in (a0, a1)
        for right in (b0, b1)
    )


def _segment_distances_batch(a0, a1, b0, b1):
    import numpy as np

    a0 = np.asarray(a0, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    b0 = np.asarray(b0, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0

    def dot(left, right):
        return (
            left[:, 0] * right[:, 0]
            + left[:, 1] * right[:, 1]
            + left[:, 2] * right[:, 2]
        )

    a = dot(u, u)
    b = dot(u, v)
    c = dot(v, v)
    d = dot(u, w)
    e = dot(v, w)
    denominator = a * c - b * b
    parallel = denominator < 1.0e-30
    free_s_numerator = b * e - c * d
    free_t_numerator = a * e - b * d
    below = free_s_numerator < 0.0
    above = free_s_numerator > denominator
    general_s_numerator = np.where(
        below, 0.0, np.where(above, denominator, free_s_numerator)
    )
    general_t_numerator = np.where(
        below, e, np.where(above, e + b, free_t_numerator)
    )
    general_t_denominator = np.where(below | above, c, denominator)
    s_numerator = np.where(parallel, 0.0, general_s_numerator)
    s_denominator = np.where(parallel, 1.0, denominator)
    t_numerator = np.where(parallel, e, general_t_numerator)
    t_denominator = np.where(parallel, c, general_t_denominator)
    t_below = t_numerator < 0.0
    t_above = t_numerator > t_denominator
    s_numerator = np.where(
        t_below,
        np.clip(-d, 0.0, a),
        np.where(t_above, np.clip(b - d, 0.0, a), s_numerator),
    )
    s_denominator = np.where(t_below | t_above, a, s_denominator)
    t_numerator = np.where(
        t_below, 0.0, np.where(t_above, t_denominator, t_numerator)
    )
    sc = np.where(
        np.abs(s_numerator) < 1.0e-30, 0.0, s_numerator / s_denominator
    )
    tc = np.where(
        np.abs(t_numerator) < 1.0e-30, 0.0, t_numerator / t_denominator
    )
    difference = w + sc[:, None] * u - tc[:, None] * v
    return np.sqrt(
        difference[:, 0] * difference[:, 0]
        + difference[:, 1] * difference[:, 1]
        + difference[:, 2] * difference[:, 2]
    )


def _segments_share_endpoint_batch(a0, a1, b0, b1):
    import numpy as np

    a0 = np.asarray(a0, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    b0 = np.asarray(b0, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    tolerance_squared = 1.0e-24
    return np.logical_or.reduce(
        tuple(
            np.sum((left - right) ** 2, axis=1) <= tolerance_squared
            for left in (a0, a1)
            for right in (b0, b1)
        )
    )
