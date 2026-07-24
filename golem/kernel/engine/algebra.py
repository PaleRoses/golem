"""Geometry algebra over checked engine parts.

Loft approximation bound: surface-distance error is at most the Hausdorff distance between the authored ideal sweep and its piecewise-linear station interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from itertools import chain, pairwise
from typing import Callable, TypeVar, assert_never

import numpy as np

from .types import (
    AcceptedComposition,
    AbsoluteProfileDepths,
    BlobPart,
    BlendSupportEscaped,
    BoxPart,
    CanalEnvelopeObstruction,
    CertifiedLocalCompositionProblem,
    CompositionResult,
    CompositionOperator,
    CompositionSectionId,
    CompositionSupportRegion,
    CompositionSupportWitness,
    FieldScaleUncalibrated,
    GencylPart,
    IntegralRadiusUnsatisfied,
    LegacyCompositionProblem,
    GradientDegeneracy,
    GeometryGraph,
    LocalCompositionEvidence,
    MuscleFormationEvidence,
    MuscleFormationFrame,
    MuscleFormationInterval,
    MuscleFormationJunction,
    MuscleFormationKind,
    MuscleFormationObstruction,
    NoCompatibleOverlap,
    Part,
    Profile,
    RejectedSurfaceFormation,
    RelativeProfileDepths,
    Rotation,
    SpanningWeb,
    TopologyChanged,
    UnsupportedMuscleFormation,
    WebAnchorCurve,
    decode_part,
    decode_web,
    quaternion_for_instance,
    require_accepted,
)


_MIRROR_3 = np.asarray((-1.0, 1.0, 1.0))
_MIRROR_2 = np.asarray((-1.0, 1.0))
_MINIMUM_LOCAL_GRADIENT = 1.0e-8
_MAXIMUM_LOCAL_GRADIENT_CONDITION = 32.0
_MUSCLE_FORMATION_SUPPORT_SCALE = 1.25
_MUSCLE_FORMATION_JUNCTION_SIDES = 8
_MUSCLE_FORMATION_LEGAL_FAMILY = (
    "finite strictly ordered collinear centerline",
    "branch-free degree at most two",
    "circular quadratic profile sections",
    "linear variable-radius canal intervals",
    "bounded regular internal junction polygons",
)


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def sdf_gencyl(pts, spine, radii):
    sampled_points = np.asarray(pts)
    decoded_spine = np.asarray(spine, dtype=np.float64)
    decoded_radii = np.asarray(radii, dtype=np.float64)

    def segment_field(index: int) -> np.ndarray:
        start, end = decoded_spine[index], decoded_spine[index + 1]
        start_radius, end_radius = decoded_radii[index], decoded_radii[index + 1]
        axis = end - start
        squared_length = float(axis @ axis)
        parameter = (
            np.zeros(sampled_points.shape[0])
            if squared_length < 1e-12
            else np.clip(((sampled_points - start) @ axis) / squared_length, 0.0, 1.0)
        )
        closest = start + parameter[:, None] * axis
        radius = start_radius + (end_radius - start_radius) * parameter
        return np.linalg.norm(sampled_points - closest, axis=1) - radius

    return reduce(
        np.minimum,
        map(segment_field, range(len(decoded_spine) - 1)),
        np.full(sampled_points.shape[0], np.inf),
    )


def sdf_blob(pts, center, size):
    decoded_center = np.asarray(center, dtype=np.float64)
    decoded_size = np.asarray(size, dtype=np.float64)
    normalized = (pts - decoded_center) / decoded_size
    distance = np.linalg.norm(normalized, axis=1)
    return (distance - 1.0) * float(np.min(decoded_size))


def sdf_box(pts, center, size, round=0.0):
    decoded_center = np.asarray(center, dtype=np.float64)
    half_extents = np.asarray(size, dtype=np.float64)
    radius = float(round)
    offset = np.abs(pts - decoded_center) - half_extents
    outside = np.linalg.norm(np.maximum(offset, 0.0), axis=1)
    inside = np.minimum(np.max(offset, axis=1), 0.0)
    return outside + inside - radius


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    interpolation = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return (
        b * (1 - interpolation)
        + a * interpolation
        - k * interpolation * (1 - interpolation)
    )


def _legacy_composition(
    problem: LegacyCompositionProblem,
) -> AcceptedComposition:
    match problem.operator:
        case CompositionOperator.BLEND:
            return AcceptedComposition(
                smin(
                    problem.accumulated,
                    problem.incoming,
                    problem.radius,
                ),
                None,
            )
        case CompositionOperator.CHAMFER:
            return AcceptedComposition(
                (
                    np.minimum(problem.accumulated, problem.incoming)
                    if problem.radius <= 0.0
                    else np.minimum(
                        np.minimum(problem.accumulated, problem.incoming),
                        (
                            problem.accumulated
                            + problem.incoming
                            - problem.radius
                        )
                        * np.sqrt(0.5),
                    )
                ),
                None,
            )
        case CompositionOperator.CREASE:
            return AcceptedComposition(
                np.minimum(problem.accumulated, problem.incoming),
                None,
            )
        case _ as unreachable:
            assert_never(unreachable)


def _field_scale_obstruction(
    section: CompositionSectionId,
    bounds: tuple[float, float],
) -> FieldScaleUncalibrated | None:
    minimum, maximum = bounds
    finite = np.isfinite(minimum) and np.isfinite(maximum)
    condition = maximum / minimum if finite and minimum > 0.0 else 1.0
    return (
        None
        if finite
        and maximum > 0.0
        and condition <= _MAXIMUM_LOCAL_GRADIENT_CONDITION
        else FieldScaleUncalibrated(
            section,
            minimum,
            maximum,
            _MAXIMUM_LOCAL_GRADIENT_CONDITION,
        )
    )


def _gradient_degeneracy(
    section: CompositionSectionId,
    bounds: tuple[float, float],
    active_sample_count: int,
) -> GradientDegeneracy | None:
    minimum, _maximum = bounds
    return (
        None
        if minimum >= _MINIMUM_LOCAL_GRADIENT
        else GradientDegeneracy(
            section,
            active_sample_count,
            minimum,
            _MINIMUM_LOCAL_GRADIENT,
        )
    )


def _support_bounds(
    problem: CertifiedLocalCompositionProblem,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.asarray(
        problem.sample_points,
        dtype=np.float64,
    ).reshape((-1, 3))
    active_points = points[active_mask]
    return (
        (np.min(active_points, axis=0), np.max(active_points, axis=0))
        if active_points.size
        else None
    )


def _support_escape(
    problem: CertifiedLocalCompositionProblem,
    incoming: CompositionSectionId,
    support: tuple[np.ndarray, np.ndarray] | None,
) -> BlendSupportEscaped | None:
    domain_lower = np.asarray(problem.domain_lower, dtype=np.float64)
    domain_upper = np.asarray(problem.domain_upper, dtype=np.float64)
    escaped_region = next(
        (
            region
            for overlap in problem.overlaps
            for region in (overlap.support,)
            if not bool(
                np.all(np.asarray(region.lower) >= domain_lower)
                and np.all(np.asarray(region.upper) <= domain_upper)
            )
        ),
        None,
    )
    if escaped_region is not None:
        return BlendSupportEscaped(
            incoming,
            escaped_region.lower,
            escaped_region.upper,
            tuple(map(float, domain_lower)),
            tuple(map(float, domain_upper)),
        )
    if support is None:
        return None
    support_lower, support_upper = support
    return (
        None
        if bool(
            np.all(support_lower >= domain_lower)
            and np.all(support_upper <= domain_upper)
        )
        else BlendSupportEscaped(
            incoming,
            tuple(map(float, support_lower)),
            tuple(map(float, support_upper)),
            tuple(map(float, domain_lower)),
            tuple(map(float, domain_upper)),
        )
    )


def _support_region_mask(
    problem: CertifiedLocalCompositionProblem,
    sample_count: int,
) -> np.ndarray:
    points = np.asarray(
        problem.sample_points,
        dtype=np.float64,
    ).reshape((-1, 3))
    regions = tuple(map(lambda overlap: overlap.support, problem.overlaps))

    def contains(region: CompositionSupportRegion) -> np.ndarray:
        lower = np.asarray(region.lower, dtype=np.float64)
        upper = np.asarray(region.upper, dtype=np.float64)
        return np.all((points >= lower) & (points <= upper), axis=1)

    return reduce(
        np.logical_or,
        map(contains, regions),
        np.zeros(sample_count, dtype=bool),
    )


def _component_count(
    field: np.ndarray,
    shape: tuple[int, int, int] | None,
) -> int | None:
    if shape is None or int(np.prod(shape)) != field.size:
        return None
    from scipy.ndimage import label

    _labels, count = label(
        np.asarray(field).reshape(shape) <= 0.0,
        structure=np.ones((3, 3, 3), dtype=np.int8),
    )
    return int(count)


def local_topology_obstruction(
    incoming: CompositionSectionId,
    clean: np.ndarray,
    composed: np.ndarray,
    shape: tuple[int, int, int] | None,
) -> TopologyChanged | None:
    clean_components, composed_components = local_topology_counts(
        clean,
        composed,
        shape,
    )
    return (
        TopologyChanged(
            incoming,
            clean_components,
            composed_components,
        )
        if clean_components is not None
        and composed_components is not None
        and clean_components != composed_components
        else None
    )


def local_topology_counts(
    clean: np.ndarray,
    composed: np.ndarray,
    shape: tuple[int, int, int] | None,
) -> tuple[int | None, int | None]:
    return _component_count(clean, shape), _component_count(composed, shape)


def _local_composition(
    problem: CertifiedLocalCompositionProblem,
) -> CompositionResult:
    incoming = problem.incoming_section
    accumulated_section = problem.accumulated_section
    if not problem.overlaps:
        return RejectedSurfaceFormation(
            (
                NoCompatibleOverlap(
                    incoming,
                    (accumulated_section,),
                    0,
                    float("inf"),
                ),
            )
        )
    accumulated = np.asarray(problem.accumulated, dtype=np.float64)
    arriving = np.asarray(problem.incoming, dtype=np.float64)
    radius = max(float(problem.radius), 0.0)
    calibration_sample_count = sum(
        overlap.probe_count for overlap in problem.overlaps
    )
    accumulated_bounds = (
        problem.accumulated_gradient_certificate.minimum,
        problem.accumulated_gradient_certificate.maximum,
    )
    incoming_bounds = (
        problem.incoming_gradient_certificate.minimum,
        problem.incoming_gradient_certificate.maximum,
    )
    scale_obstructions = tuple(
        obstruction
        for obstruction in (
            _field_scale_obstruction(
                accumulated_section,
                accumulated_bounds,
            ),
            _field_scale_obstruction(incoming, incoming_bounds),
        )
        if obstruction is not None
    )
    if scale_obstructions:
        return RejectedSurfaceFormation(scale_obstructions)
    degeneracies = tuple(
        obstruction
        for obstruction in (
            _gradient_degeneracy(
                accumulated_section,
                accumulated_bounds,
                calibration_sample_count,
            ),
            _gradient_degeneracy(
                incoming,
                incoming_bounds,
                calibration_sample_count,
            ),
        )
        if obstruction is not None
    )
    if degeneracies:
        return RejectedSurfaceFormation(degeneracies)
    accumulated_scale = float(np.median(accumulated_bounds))
    incoming_scale = float(np.median(incoming_bounds))
    normalized_accumulated = accumulated / accumulated_scale
    normalized_incoming = arriving / incoming_scale
    normalized_radius = radius
    proximity_gate = np.clip(
        1.0
        - np.maximum(normalized_accumulated, normalized_incoming)
        / max(normalized_radius, 1.0e-12),
        0.0,
        1.0,
    )
    accumulated_gradient = np.asarray(
        problem.accumulated_gradient,
        dtype=np.float64,
    ).reshape((-1, 3))
    incoming_gradient = np.asarray(
        problem.incoming_gradient,
        dtype=np.float64,
    ).reshape((-1, 3))
    accumulated_norm = np.linalg.norm(accumulated_gradient, axis=1)
    incoming_norm = np.linalg.norm(incoming_gradient, axis=1)
    gradient_denominator = accumulated_norm * incoming_norm
    gradient_cosine = np.divide(
        np.einsum(
            "ij,ij->i",
            accumulated_gradient,
            incoming_gradient,
        ),
        gradient_denominator,
        out=np.ones_like(gradient_denominator),
        where=gradient_denominator > _MINIMUM_LOCAL_GRADIENT,
    )
    opening = np.clip((1.0 - gradient_cosine) * 0.5, 0.0, 1.0)
    effective_radius = 0.5 * normalized_radius * opening
    difference_gate = np.where(
        effective_radius > 0.0,
        np.clip(
            1.0
            - np.abs(
                normalized_accumulated.reshape(-1)
                - normalized_incoming.reshape(-1)
            )
            / np.where(effective_radius > 0.0, effective_radius, 1.0),
            0.0,
            1.0,
        ),
        0.0,
    )
    raw_weight = (
        difference_gate.reshape(-1)
        * proximity_gate.reshape(-1)
        * opening
        * _support_region_mask(problem, accumulated.size)
    )
    support_weight = raw_weight * raw_weight * (3.0 - 2.0 * raw_weight)
    active_mask = support_weight > 0.0
    support = _support_bounds(problem, active_mask)
    escaped = _support_escape(problem, incoming, support)
    if escaped is not None:
        return RejectedSurfaceFormation((escaped,))
    safe_radius = np.where(
        effective_radius > 0.0,
        effective_radius,
        1.0,
    )
    interpolation = np.clip(
        0.5
        + 0.5
        * (
            normalized_incoming.reshape(-1)
            - normalized_accumulated.reshape(-1)
        )
        / safe_radius,
        0.0,
        1.0,
    )
    normalized_smooth = (
        normalized_incoming.reshape(-1) * (1.0 - interpolation)
        + normalized_accumulated.reshape(-1) * interpolation
        - effective_radius * interpolation * (1.0 - interpolation)
    )
    normalized_hard = np.minimum(
        normalized_accumulated,
        normalized_incoming,
    ).reshape(-1)
    hard = np.minimum(accumulated, arriving).reshape(-1)
    correction = (
        support_weight
        * min(accumulated_scale, incoming_scale)
        * (normalized_smooth - normalized_hard)
    )
    composed = (hard + correction).reshape(accumulated.shape)
    outside_delta = float(
        np.max(
            np.abs(composed.reshape(-1)[~active_mask] - hard[~active_mask]),
            initial=0.0,
        )
    )
    if outside_delta != 0.0:
        region = problem.overlaps[0].support
        return RejectedSurfaceFormation(
            (
                BlendSupportEscaped(
                    incoming,
                    region.lower,
                    region.upper,
                    tuple(map(float, problem.domain_lower)),
                    tuple(map(float, problem.domain_upper)),
                ),
            )
        )
    topology_obstruction = local_topology_obstruction(
        incoming,
        np.minimum(accumulated, arriving),
        composed,
        problem.topology_shape,
    )
    if topology_obstruction is not None:
        return RejectedSurfaceFormation((topology_obstruction,))
    clean_components, composed_components = local_topology_counts(
        np.minimum(accumulated, arriving),
        composed,
        problem.topology_shape,
    )
    evidence = LocalCompositionEvidence(
        incoming=incoming,
        overlapping=(accumulated_section,),
        support=CompositionSupportWitness(
            (
                tuple(map(float, support[0]))
                if support is not None
                else None
            ),
            (
                tuple(map(float, support[1]))
                if support is not None
                else None
            ),
            int(np.count_nonzero(active_mask)),
            tuple(map(lambda overlap: overlap.support, problem.overlaps)),
        ),
        accumulated_gradient_bounds=accumulated_bounds,
        incoming_gradient_bounds=incoming_bounds,
        maximum_outside_support_delta=outside_delta,
        maximum_blend_displacement=float(
            np.max(np.abs(correction), initial=0.0)
        ),
        clean_component_count=clean_components,
        composed_component_count=composed_components,
    )
    return AcceptedComposition(composed, evidence)


def compose_union(
    problem: LegacyCompositionProblem | CertifiedLocalCompositionProblem,
) -> CompositionResult:
    match problem:
        case CertifiedLocalCompositionProblem():
            return _local_composition(problem)
        case LegacyCompositionProblem():
            return _legacy_composition(problem)
        case _ as unreachable:
            assert_never(unreachable)


def sdf_difference(host: np.ndarray, carve: np.ndarray) -> np.ndarray:
    return np.maximum(host, -carve)


_InstancePart = TypeVar("_InstancePart")


def _instances(
    parts,
    is_mirrored: Callable[[_InstancePart], bool],
):
    return chain.from_iterable(
        map(
            lambda part: ((part, False), (part, True))
            if is_mirrored(part)
            else ((part, False),),
            parts,
        )
    )


def instances(graph):
    return _instances(graph["parts"], lambda part: bool(part.get("mirror")))


def part_instances(graph: GeometryGraph):
    return _instances(graph.parts, lambda part: part.mirror)


def carve_part_instances(graph: GeometryGraph):
    return _instances(graph.carves, lambda part: part.mirror)


def web_instances(graph: GeometryGraph):
    return _instances(graph.webs, lambda web: web.mirror)


type _PreparedWebTriangle = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]


def _anchor_pair_triangles(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
) -> tuple[_PreparedWebTriangle, ...]:
    left_spine, left_radii = left
    right_spine, right_radii = right
    return tuple(
        chain.from_iterable(
            map(
                lambda index: (
                    (
                        left_spine[index],
                        left_spine[index + 1],
                        right_spine[index],
                        float(left_radii[index]),
                        float(left_radii[index + 1]),
                        float(right_radii[index]),
                    ),
                    (
                        left_spine[index + 1],
                        right_spine[index + 1],
                        right_spine[index],
                        float(left_radii[index + 1]),
                        float(right_radii[index + 1]),
                        float(right_radii[index]),
                    ),
                ),
                range(len(left_spine) - 1),
            )
        )
    )


def _prepared_web_triangles(
    anchors: tuple[WebAnchorCurve, ...],
    mirrored: bool,
) -> tuple[_PreparedWebTriangle, ...]:
    reflection = _MIRROR_3 if mirrored else np.ones(3)
    prepared_anchors = tuple(
        map(
            lambda anchor: (
                np.asarray(anchor.spine, dtype=np.float64) * reflection,
                np.asarray(anchor.radii, dtype=np.float64),
            ),
            anchors,
        )
    )
    return tuple(
        chain.from_iterable(
            map(
                lambda anchor_pair: _anchor_pair_triangles(*anchor_pair),
                pairwise(prepared_anchors),
            )
        )
    )


def _segment_candidate(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    start_half_extent: float,
    end_half_extent: float,
) -> tuple[np.ndarray, np.ndarray]:
    axis = end - start
    squared_length = float(axis @ axis)
    parameter = (
        np.zeros(points.shape[0])
        if squared_length < 1e-12
        else np.clip(((points - start) @ axis) / squared_length, 0.0, 1.0)
    )
    closest = start + parameter[:, None] * axis
    squared_distance = np.einsum(
        "ij,ij->i", points - closest, points - closest
    )
    half_extent = (
        start_half_extent
        + (end_half_extent - start_half_extent) * parameter
    )
    return squared_distance, half_extent


def _triangle_web_field(
    points: np.ndarray,
    triangle: _PreparedWebTriangle,
) -> np.ndarray:
    a, b, c, radius_a, radius_b, radius_c = triangle
    edge_ab = b - a
    edge_ac = c - a
    offset = points - a
    dot_ab_ab = float(edge_ab @ edge_ab)
    dot_ab_ac = float(edge_ab @ edge_ac)
    dot_ac_ac = float(edge_ac @ edge_ac)
    dot_offset_ab = offset @ edge_ab
    dot_offset_ac = offset @ edge_ac
    barycentric_denominator = dot_ab_ab * dot_ac_ac - dot_ab_ac**2
    safe_barycentric_denominator = max(barycentric_denominator, 1e-24)
    weight_b = (
        dot_ac_ac * dot_offset_ab - dot_ab_ac * dot_offset_ac
    ) / safe_barycentric_denominator
    weight_c = (
        dot_ab_ab * dot_offset_ac - dot_ab_ac * dot_offset_ab
    ) / safe_barycentric_denominator
    weight_a = 1.0 - weight_b - weight_c
    normal = _cross3(edge_ab, edge_ac)
    squared_normal = float(normal @ normal)
    face_squared_distance = (
        (offset @ normal) ** 2 / max(squared_normal, 1e-24)
    )
    face_inside = (
        (barycentric_denominator > 1e-24)
        & (weight_a >= 0.0)
        & (weight_b >= 0.0)
        & (weight_c >= 0.0)
    )
    face_half_extent = (
        weight_a * radius_a + weight_b * radius_b + weight_c * radius_c
    )
    def select_nearer(
        selected: tuple[np.ndarray, np.ndarray],
        candidate: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        selected_distance, selected_half_extent = selected
        candidate_distance, candidate_half_extent = candidate
        choose_candidate = candidate_distance < selected_distance
        return (
            np.minimum(selected_distance, candidate_distance),
            np.where(
                choose_candidate,
                candidate_half_extent,
                selected_half_extent,
            ),
        )

    squared_distance, half_extent = reduce(
        select_nearer,
        map(
            lambda edge: _segment_candidate(points, *edge),
            (
                (a, b, radius_a, radius_b),
                (b, c, radius_b, radius_c),
                (c, a, radius_c, radius_a),
            ),
        ),
        (
            np.where(face_inside, face_squared_distance, np.inf),
            face_half_extent,
        ),
    )
    return np.sqrt(np.maximum(squared_distance, 0.0)) - half_extent


def prepared_web_sdf(
    web: SpanningWeb,
    mirrored: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    triangles = _prepared_web_triangles(web.anchors, mirrored)

    def evaluate(points: np.ndarray) -> np.ndarray:
        sampled_points = np.asarray(points, dtype=np.float64)
        return reduce(
            np.minimum,
            map(
                lambda triangle: _triangle_web_field(
                    sampled_points, triangle
                ),
                triangles,
            ),
            np.full(sampled_points.shape[0], np.inf),
        )

    return evaluate


def web_sdf_checked(
    points: np.ndarray,
    web: SpanningWeb,
    mirrored: bool = False,
) -> np.ndarray:
    return prepared_web_sdf(web, mirrored)(points)


def sdf_web(points, web, mirrored=False):
    decoded = require_accepted(decode_web(web))
    return web_sdf_checked(
        np.asarray(points, dtype=np.float64), decoded, bool(mirrored)
    )


def _rotation_for_instance(rotation: Rotation, mirrored: bool) -> np.ndarray:
    return quaternion_for_instance(rotation.quaternion, mirrored)


def _cross3(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.array(
        [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]
    )


def _norm3(vector: np.ndarray) -> float:
    return float(np.sqrt(vector @ vector))


def _profile_axes(
    tangent: np.ndarray,
    up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    epsilon = 1.0e-12
    reference = (
        up
        if abs(float(up @ tangent)) < 0.999
        else np.array([0.0, 1.0, 0.0])
    )
    fallback = (
        np.array([1.0, 0.0, 0.0])
        if abs(float(reference @ tangent)) >= 0.999
        else reference
    )
    width_axis = _cross3(fallback, tangent)
    normalized_width_axis = width_axis / max(_norm3(width_axis), epsilon)
    return normalized_width_axis, _cross3(tangent, normalized_width_axis)


def _profile_cross_section_distance(
    pts: np.ndarray,
    origin: np.ndarray,
    tangent: np.ndarray,
    width: np.ndarray | float,
    depth: np.ndarray | float,
    exponent: np.ndarray | float,
    offset_width: np.ndarray | float,
    offset_depth: np.ndarray | float,
    roll: np.ndarray | float,
    up: np.ndarray,
) -> np.ndarray:
    epsilon = 1.0e-12
    width_axis, depth_axis = _profile_axes(tangent, up)
    checked_width = np.maximum(width, epsilon)
    checked_depth = np.maximum(depth, epsilon)
    width_coordinate = pts @ width_axis - float(origin @ width_axis)
    depth_coordinate = pts @ depth_axis - float(origin @ depth_axis)
    decoded_roll = np.asarray(roll, dtype=np.float64)
    unrolled = bool(np.all(decoded_roll == 0.0))
    angle = np.deg2rad(decoded_roll)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    section_width_coordinate = (
        width_coordinate
        if unrolled
        else width_coordinate * cosine + depth_coordinate * sine
    )
    section_depth_coordinate = (
        depth_coordinate
        if unrolled
        else -width_coordinate * sine + depth_coordinate * cosine
    )
    normalized_width = (
        np.abs(section_width_coordinate - offset_width) / checked_width
    )
    normalized_depth = (
        np.abs(section_depth_coordinate - offset_depth) / checked_depth
    )
    maximum = np.maximum(normalized_width, normalized_depth)
    minimum = np.minimum(normalized_width, normalized_depth)
    ratio = minimum / np.maximum(maximum, epsilon)
    radial_ratio = maximum * (1.0 + ratio**exponent) ** (1.0 / exponent)
    return (radial_ratio - 1.0) * np.minimum(checked_width, checked_depth)


@dataclass(frozen=True)
class _ProfileSweep:
    points: np.ndarray
    spine: np.ndarray
    widths: np.ndarray
    exponents: np.ndarray
    depths: RelativeProfileDepths | AbsoluteProfileDepths
    offsets: np.ndarray
    rolls: np.ndarray
    up: np.ndarray

    @property
    def segment_count(self) -> int:
        return len(self.spine) - 1


@dataclass(frozen=True)
class _CollinearCover:
    tangent: np.ndarray
    length: float
    anchor_coordinates: np.ndarray


def _collinear_cover(sweep: _ProfileSweep) -> _CollinearCover | None:
    whole_axis = sweep.spine[-1] - sweep.spine[0]
    whole_length = _norm3(whole_axis)
    whole_tangent = (
        whole_axis / whole_length if whole_length > 1.0e-12 else whole_axis
    )
    anchor_coordinates = (
        (sweep.spine - sweep.spine[0]) @ whole_tangent
        if whole_length > 1.0e-12
        else np.zeros(len(sweep.spine))
    )
    reconstructed_spine = (
        sweep.spine[0] + anchor_coordinates[:, None] * whole_tangent
        if whole_length > 1.0e-12
        else sweep.spine
    )
    is_cover = (
        sweep.segment_count > 1
        and whole_length > 1.0e-12
        and bool(np.all(np.diff(anchor_coordinates) > 1.0e-12))
        and float(np.max(np.linalg.norm(sweep.spine - reconstructed_spine, axis=1)))
        <= 1.0e-6
    )
    return (
        _CollinearCover(whole_tangent, whole_length, anchor_coordinates)
        if is_cover
        else None
    )


def _station_values(
    coordinate: np.ndarray,
    anchors: np.ndarray,
    values: np.ndarray,
) -> np.ndarray | float:
    return (
        float(values[0])
        if bool(np.all(values == values[0]))
        else np.interp(coordinate, anchors, values)
    )


def _station_dimensions(
    sweep: _ProfileSweep,
    coordinate: np.ndarray,
    anchors: np.ndarray,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    width = _station_values(coordinate, anchors, sweep.widths)
    match sweep.depths:
        case RelativeProfileDepths(ratios):
            depth = width * _station_values(
                coordinate,
                anchors,
                np.asarray(ratios, dtype=np.float64),
            )
        case AbsoluteProfileDepths(values):
            depth = _station_values(
                coordinate,
                anchors,
                np.asarray(values, dtype=np.float64),
            )
        case _ as unreachable:
            assert_never(unreachable)
    return width, depth


def _segment_dimensions(
    sweep: _ProfileSweep,
    index: int,
    parameter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    width = sweep.widths[index] + (
        sweep.widths[index + 1] - sweep.widths[index]
    ) * parameter
    match sweep.depths:
        case RelativeProfileDepths(ratios):
            decoded = np.asarray(ratios, dtype=np.float64)
            depth = width * (
                decoded[index]
                + (decoded[index + 1] - decoded[index]) * parameter
            )
        case AbsoluteProfileDepths(values):
            decoded = np.asarray(values, dtype=np.float64)
            depth = decoded[index] + (
                decoded[index + 1] - decoded[index]
            ) * parameter
        case _ as unreachable:
            assert_never(unreachable)
    return width, depth


def _anchor_depths(sweep: _ProfileSweep) -> np.ndarray:
    match sweep.depths:
        case RelativeProfileDepths(ratios):
            return sweep.widths * np.asarray(ratios, dtype=np.float64)
        case AbsoluteProfileDepths(values):
            return np.asarray(values, dtype=np.float64)
        case _ as unreachable:
            assert_never(unreachable)


def _collinear_profile_field(
    sweep: _ProfileSweep,
    cover: _CollinearCover,
) -> np.ndarray:
    axial_coordinate = (sweep.points - sweep.spine[0]) @ cover.tangent
    clipped_coordinate = np.clip(
        axial_coordinate, 0.0, cover.anchor_coordinates[-1]
    )
    width, depth = _station_dimensions(
        sweep, clipped_coordinate, cover.anchor_coordinates
    )
    exponent = _station_values(
        clipped_coordinate, cover.anchor_coordinates, sweep.exponents
    )
    roll = _station_values(
        clipped_coordinate, cover.anchor_coordinates, sweep.rolls
    )
    offset_width = _station_values(
        clipped_coordinate, cover.anchor_coordinates, sweep.offsets[:, 0]
    )
    offset_depth = _station_values(
        clipped_coordinate, cover.anchor_coordinates, sweep.offsets[:, 1]
    )
    radial_distance = _profile_cross_section_distance(
        sweep.points,
        sweep.spine[0],
        cover.tangent,
        width,
        depth,
        exponent,
        offset_width,
        offset_depth,
        roll,
        sweep.up,
    )
    axial_distance = (
        np.abs(axial_coordinate - cover.length / 2.0) - cover.length / 2.0
    )
    return np.minimum(np.maximum(radial_distance, axial_distance), 0.0) + np.hypot(
        np.maximum(radial_distance, 0.0), np.maximum(axial_distance, 0.0)
    )


def _segment_profile_field(sweep: _ProfileSweep, index: int) -> np.ndarray:
    start, end = sweep.spine[index], sweep.spine[index + 1]
    axis = end - start
    squared_length = float(axis @ axis)
    raw_parameter = (
        np.zeros(sweep.points.shape[0])
        if squared_length < 1.0e-12
        else ((sweep.points - start) @ axis) / squared_length
    )
    parameter = (
        np.zeros(sweep.points.shape[0])
        if squared_length < 1.0e-12
        else np.clip(raw_parameter, 0.0, 1.0)
    )
    segment_length = 0.0 if squared_length < 1.0e-12 else float(np.sqrt(squared_length))
    tangent = (
        np.array([0.0, 1.0, 0.0])
        if squared_length < 1.0e-12
        else axis / segment_length
    )
    closest = start + parameter[:, None] * axis
    width, depth = _segment_dimensions(sweep, index, parameter)
    exponent = sweep.exponents[index] + (
        sweep.exponents[index + 1] - sweep.exponents[index]
    ) * parameter
    roll = sweep.rolls[index] + (
        sweep.rolls[index + 1] - sweep.rolls[index]
    ) * parameter
    offset = sweep.offsets[index] + (
        sweep.offsets[index + 1] - sweep.offsets[index]
    ) * parameter[:, None]
    cross_section_distance = _profile_cross_section_distance(
        sweep.points,
        start,
        tangent,
        width,
        depth,
        exponent,
        offset[:, 0],
        offset[:, 1],
        roll,
        sweep.up,
    )
    axial_coordinate = (sweep.points - start) @ tangent
    axial_distance = (
        np.abs(axial_coordinate - segment_length / 2.0) - segment_length / 2.0
        if sweep.segment_count == 1
        else -axial_coordinate
        if index == 0
        else axial_coordinate - segment_length
        if index == sweep.segment_count - 1
        else np.full(sweep.points.shape[0], -np.inf)
    )
    owns_axial_slab = (
        np.ones(sweep.points.shape[0], dtype=bool)
        if sweep.segment_count == 1
        else raw_parameter <= 1.0
        if index == 0
        else raw_parameter >= 0.0
        if index == sweep.segment_count - 1
        else (raw_parameter >= 0.0) & (raw_parameter <= 1.0)
    )
    segment = np.minimum(
        np.maximum(cross_section_distance, axial_distance), 0.0
    ) + np.hypot(
        np.maximum(cross_section_distance, 0.0), np.maximum(axial_distance, 0.0)
    )
    endpoint_distance = np.linalg.norm(sweep.points - closest, axis=1)
    return np.where(owns_axial_slab, segment, endpoint_distance)


def _anchor_profile_field(
    sweep: _ProfileSweep,
    record: tuple[
        np.ndarray,
        float,
        float,
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
    ],
) -> np.ndarray:
    anchor, width, depth, offset, roll, previous, following = record
    tangent_vector = following - previous
    tangent = tangent_vector / max(_norm3(tangent_vector), 1.0e-12)
    width_axis, depth_axis = _profile_axes(tangent, sweep.up)
    angle = np.deg2rad(roll)
    rolled_width_axis = np.cos(angle) * width_axis + np.sin(angle) * depth_axis
    rolled_depth_axis = -np.sin(angle) * width_axis + np.cos(angle) * depth_axis
    center = (
        anchor
        + offset[0] * rolled_width_axis
        + offset[1] * rolled_depth_axis
    )
    return np.linalg.norm(sweep.points - center, axis=1) - min(width, depth)


def _profile_sweep_field(sweep: _ProfileSweep) -> np.ndarray:
    cover = _collinear_cover(sweep)
    if cover is not None:
        return _collinear_profile_field(sweep, cover)
    return reduce(
        np.minimum,
        chain(
            map(lambda index: _segment_profile_field(sweep, index), range(sweep.segment_count)),
            map(
                lambda record: _anchor_profile_field(sweep, record),
                zip(
                    sweep.spine[1:-1],
                    sweep.widths[1:-1],
                    _anchor_depths(sweep)[1:-1],
                    sweep.offsets[1:-1],
                    sweep.rolls[1:-1],
                    sweep.spine[:-2],
                    sweep.spine[2:],
                ),
            ),
        ),
    )


def _profile_for_instance(profile: Profile, mirrored: bool) -> Profile:
    if not mirrored:
        return profile
    return Profile(
        profile.exponents,
        profile.depths,
        tuple(
            (float(x), float(y))
            for x, y in np.asarray(profile.offsets, dtype=np.float64) * _MIRROR_2
        ),
        tuple(-roll for roll in profile.rolls),
        tuple(
            map(
                float,
                np.asarray(profile.up, dtype=np.float64) * _MIRROR_3,
            )
        ),
    )


@dataclass(frozen=True)
class _SkeletonIntegralPreparation:
    spine: np.ndarray
    radii: np.ndarray
    tangent: np.ndarray
    stations: np.ndarray
    radius_derivatives: np.ndarray
    evidence: MuscleFormationEvidence


def _unsupported_muscle_formation(
    part: GencylPart,
    mirrored: bool,
    reason: str,
) -> MuscleFormationObstruction:
    return MuscleFormationObstruction(
        part.part_id,
        mirrored,
        UnsupportedMuscleFormation(
            reason,
            _MUSCLE_FORMATION_LEGAL_FAMILY,
        ),
    )


def _formation_profile_depths(
    profile: Profile | None,
    radii: np.ndarray,
) -> np.ndarray:
    if profile is None:
        return radii
    match profile.depths:
        case RelativeProfileDepths(ratios):
            return radii * np.asarray(ratios, dtype=np.float64)
        case AbsoluteProfileDepths(values):
            return np.asarray(values, dtype=np.float64)
        case _ as unreachable:
            assert_never(unreachable)


def _formation_frames(
    spine: np.ndarray,
    stations: np.ndarray,
    tangent: np.ndarray,
    profile: Profile | None,
) -> tuple[MuscleFormationFrame, ...]:
    up = np.asarray(
        (0.0, 0.0, 1.0) if profile is None else profile.up,
        dtype=np.float64,
    )
    width_axis, depth_axis = _profile_axes(tangent, up)
    rolls = (
        np.zeros(len(spine), dtype=np.float64)
        if profile is None
        else np.asarray(profile.rolls, dtype=np.float64)
    )

    def frame(record: tuple[int, np.ndarray, float, float]) -> MuscleFormationFrame:
        station_index, origin, station, roll = record
        angle = np.deg2rad(roll)
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        rolled_width = cosine * width_axis + sine * depth_axis
        rolled_depth = -sine * width_axis + cosine * depth_axis
        determinant = float(
            np.linalg.det(
                np.stack((rolled_width, rolled_depth, tangent), axis=1)
            )
        )
        return MuscleFormationFrame(
            station_index,
            float(station),
            tuple(map(float, origin)),
            tuple(map(float, rolled_width)),
            tuple(map(float, rolled_depth)),
            tuple(map(float, tangent)),
            determinant,
        )

    return tuple(
        map(
            frame,
            zip(
                range(len(spine)),
                spine,
                stations,
                rolls,
                strict=True,
            ),
        )
    )


def _formation_junctions(
    frames: tuple[MuscleFormationFrame, ...],
    radii: np.ndarray,
) -> tuple[MuscleFormationJunction, ...]:
    angles = tuple(
        2.0 * np.pi * index / _MUSCLE_FORMATION_JUNCTION_SIDES
        for index in range(_MUSCLE_FORMATION_JUNCTION_SIDES)
    )

    def junction(
        record: tuple[MuscleFormationFrame, float],
    ) -> MuscleFormationJunction:
        frame, radius = record
        center = np.asarray(frame.origin, dtype=np.float64)
        width_axis = np.asarray(frame.width_axis, dtype=np.float64)
        depth_axis = np.asarray(frame.depth_axis, dtype=np.float64)
        vertices = tuple(
            tuple(
                map(
                    float,
                    center
                    + radius
                    * (
                        np.cos(angle) * width_axis
                        + np.sin(angle) * depth_axis
                    ),
                )
            )
            for angle in angles
        )
        return MuscleFormationJunction(
            frame.station_index,
            frame.station,
            frame.origin,
            float(radius),
            vertices,
        )

    return tuple(
        map(
            junction,
            zip(frames[1:-1], radii[1:-1], strict=True),
        )
    )


def _formation_volume_centroid(
    spine: np.ndarray,
    stations: np.ndarray,
    radii: np.ndarray,
    tangent: np.ndarray,
) -> tuple[float, tuple[float, float, float]]:
    lengths = np.diff(stations)
    lower_radii = radii[:-1]
    upper_radii = radii[1:]
    frustum_denominators = (
        lower_radii * lower_radii
        + lower_radii * upper_radii
        + upper_radii * upper_radii
    )
    frustum_volumes = (
        np.pi * lengths * frustum_denominators / 3.0
    )
    frustum_centroids = (
        stations[:-1]
        + lengths
        * (
            lower_radii * lower_radii
            + 2.0 * lower_radii * upper_radii
            + 3.0 * upper_radii * upper_radii
        )
        / (4.0 * frustum_denominators)
    )
    lower_cap_volume = 2.0 * np.pi * radii[0] ** 3 / 3.0
    upper_cap_volume = 2.0 * np.pi * radii[-1] ** 3 / 3.0
    lower_cap_centroid = stations[0] - 3.0 * radii[0] / 8.0
    upper_cap_centroid = stations[-1] + 3.0 * radii[-1] / 8.0
    volume = float(
        np.sum(frustum_volumes)
        + lower_cap_volume
        + upper_cap_volume
    )
    centroid_station = float(
        (
            np.sum(frustum_volumes * frustum_centroids)
            + lower_cap_volume * lower_cap_centroid
            + upper_cap_volume * upper_cap_centroid
        )
        / volume
    )
    centroid = spine[0] + centroid_station * tangent
    return volume, tuple(map(float, centroid))


def _prepare_muscle_formation(
    part: GencylPart,
    mirrored: bool,
) -> _SkeletonIntegralPreparation | MuscleFormationObstruction:
    if part.formation is not MuscleFormationKind.SKELETON_INTEGRAL:
        return _unsupported_muscle_formation(
            part,
            mirrored,
            f"formation is {part.formation!r}",
        )
    decoded_spine = np.asarray(part.spine, dtype=np.float64)
    spine = decoded_spine * _MIRROR_3 if mirrored else decoded_spine
    radii = np.asarray(part.radii, dtype=np.float64)
    whole_axis = spine[-1] - spine[0]
    whole_length = _norm3(whole_axis)
    tolerance = max(whole_length, float(np.max(radii))) * 1.0e-9
    if whole_length <= tolerance:
        return _unsupported_muscle_formation(
            part,
            mirrored,
            "centerline endpoints are degenerate",
        )
    tangent = whole_axis / whole_length
    stations = (spine - spine[0]) @ tangent
    reconstructed = spine[0] + stations[:, None] * tangent
    maximum_collinearity_residual = float(
        np.max(np.linalg.norm(spine - reconstructed, axis=1))
    )
    if maximum_collinearity_residual > tolerance:
        return _unsupported_muscle_formation(
            part,
            mirrored,
            "centerline is non-collinear with residual "
            f"{maximum_collinearity_residual:.17g}",
        )
    spine = reconstructed
    interval_lengths = np.diff(stations)
    invalid_interval = next(
        (
            index
            for index, length in enumerate(interval_lengths)
            if float(length) <= tolerance
        ),
        None,
    )
    if invalid_interval is not None:
        return _unsupported_muscle_formation(
            part,
            mirrored,
            f"centerline interval {invalid_interval} is not strictly ordered",
        )
    instance_profile = (
        None
        if part.profile is None
        else _profile_for_instance(part.profile, mirrored)
    )
    if instance_profile is not None:
        nonquadratic = next(
            (
                index
                for index, exponent in enumerate(instance_profile.exponents)
                if abs(exponent - 2.0) > 1.0e-9
            ),
            None,
        )
        if nonquadratic is not None:
            return _unsupported_muscle_formation(
                part,
                mirrored,
                f"profile section {nonquadratic} is not quadratic",
            )
        offset = next(
            (
                index
                for index, value in enumerate(instance_profile.offsets)
                if _norm3(
                    np.asarray((value[0], value[1], 0.0), dtype=np.float64)
                )
                > tolerance
            ),
            None,
        )
        if offset is not None:
            return _unsupported_muscle_formation(
                part,
                mirrored,
                f"profile section {offset} has a nonzero offset",
            )
    depths = _formation_profile_depths(instance_profile, radii)
    radius_residuals = np.abs(depths - radii)
    maximum_radius_residual = float(np.max(radius_residuals))
    if maximum_radius_residual > tolerance:
        section_index = int(np.argmax(radius_residuals))
        return MuscleFormationObstruction(
            part.part_id,
            mirrored,
            IntegralRadiusUnsatisfied(
                section_index,
                float(radii[section_index]),
                float(depths[section_index]),
                float(radius_residuals[section_index]),
                tolerance,
            ),
        )
    radius_derivatives = np.diff(radii) / interval_lengths
    envelope_ratios = (
        _MUSCLE_FORMATION_SUPPORT_SCALE * np.abs(radius_derivatives)
    )
    invalid_envelope = next(
        (
            index
            for index, ratio in enumerate(envelope_ratios)
            if float(ratio) >= 1.0
        ),
        None,
    )
    if invalid_envelope is not None:
        return MuscleFormationObstruction(
            part.part_id,
            mirrored,
            CanalEnvelopeObstruction(
                invalid_envelope,
                float(stations[invalid_envelope]),
                float(stations[invalid_envelope + 1]),
                float(radius_derivatives[invalid_envelope]),
                _MUSCLE_FORMATION_SUPPORT_SCALE,
                float(envelope_ratios[invalid_envelope]),
                1.0,
            ),
        )
    intervals = tuple(
        MuscleFormationInterval(
            index,
            float(stations[index]),
            float(stations[index + 1]),
            float(radii[index]),
            float(radii[index + 1]),
            float(radius_derivatives[index]),
            float(envelope_ratios[index]),
        )
        for index in range(len(interval_lengths))
    )
    frames = _formation_frames(
        spine,
        stations,
        tangent,
        instance_profile,
    )
    if min(map(lambda frame: frame.determinant, frames)) <= 0.0:
        return _unsupported_muscle_formation(
            part,
            mirrored,
            "local frame orientation is not right-handed",
        )
    junctions = _formation_junctions(frames, radii)
    solved_volume, centroid = _formation_volume_centroid(
        spine,
        stations,
        radii,
        tangent,
    )
    support_radius = _MUSCLE_FORMATION_SUPPORT_SCALE * float(np.max(radii))
    evidence = MuscleFormationEvidence(
        part.part_id,
        MuscleFormationKind.SKELETON_INTEGRAL,
        mirrored,
        intervals,
        junctions,
        maximum_radius_residual,
        solved_volume,
        solved_volume,
        0.0,
        centroid,
        0.0,
        float(np.max(envelope_ratios)),
        _MUSCLE_FORMATION_SUPPORT_SCALE,
        frames,
        tuple(map(float, np.min(spine, axis=0) - support_radius)),
        tuple(map(float, np.max(spine, axis=0) + support_radius)),
        True,
    )
    return _SkeletonIntegralPreparation(
        spine,
        radii,
        tangent,
        stations,
        radius_derivatives,
        evidence,
    )


def solve_muscle_formation(
    part: GencylPart,
    mirrored: bool = False,
) -> MuscleFormationEvidence | MuscleFormationObstruction:
    prepared = _prepare_muscle_formation(part, mirrored)
    return (
        prepared
        if isinstance(prepared, MuscleFormationObstruction)
        else prepared.evidence
    )


def _skeleton_integral_canal(
    prepared: _SkeletonIntegralPreparation,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sampled = np.asarray(points, dtype=np.float64)
    axial = (sampled - prepared.spine[0]) @ prepared.tangent
    clipped = np.clip(axial, prepared.stations[0], prepared.stations[-1])
    interval_indices = np.clip(
        np.searchsorted(prepared.stations, clipped, side="right") - 1,
        0,
        len(prepared.radius_derivatives) - 1,
    )
    local_station = clipped - prepared.stations[interval_indices]
    radius = (
        prepared.radii[interval_indices]
        + prepared.radius_derivatives[interval_indices] * local_station
    )
    closest = prepared.spine[0] + clipped[:, None] * prepared.tangent
    side_distance = np.linalg.norm(sampled - closest, axis=1) - radius
    lower_distance = (
        np.linalg.norm(sampled - prepared.spine[0], axis=1)
        - prepared.radii[0]
    )
    upper_distance = (
        np.linalg.norm(sampled - prepared.spine[-1], axis=1)
        - prepared.radii[-1]
    )
    canal = np.where(
        axial < prepared.stations[0],
        lower_distance,
        np.where(
            axial > prepared.stations[-1],
            upper_distance,
            side_distance,
        ),
    )
    return canal, radius


def _quartic_line_factor(
    canal: np.ndarray,
    radius: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    support_scale = _MUSCLE_FORMATION_SUPPORT_SCALE
    coordinate = np.maximum(radius + canal, 0.0) / (
        support_scale * radius
    )
    base = np.maximum(1.0 - coordinate * coordinate, 0.0)
    integral = np.where(coordinate < 1.0, base ** 2.5, 0.0)
    surface_integral = (1.0 - support_scale ** -2.0) ** 2.5
    factor = (1.0 + integral) / (1.0 + surface_integral)
    derivative = np.where(
        coordinate < 1.0,
        -5.0 * coordinate * base ** 1.5 / (1.0 + surface_integral),
        0.0,
    )
    return factor, derivative


def _skeleton_integral_field(
    prepared: _SkeletonIntegralPreparation,
    points: np.ndarray,
) -> np.ndarray:
    canal, radius = _skeleton_integral_canal(prepared, points)
    factor, _derivative = _quartic_line_factor(canal, radius)
    return canal * factor


def _skeleton_integral_gradient(
    prepared: _SkeletonIntegralPreparation,
    points: np.ndarray,
) -> np.ndarray:
    sampled = np.asarray(points, dtype=np.float64)
    canal, radius = _skeleton_integral_canal(prepared, sampled)
    axial = (sampled - prepared.spine[0]) @ prepared.tangent
    clipped = np.clip(axial, prepared.stations[0], prepared.stations[-1])
    interval_indices = np.clip(
        np.searchsorted(prepared.stations, clipped, side="right") - 1,
        0,
        len(prepared.radius_derivatives) - 1,
    )
    radius_derivative = prepared.radius_derivatives[interval_indices]
    closest = prepared.spine[0] + clipped[:, None] * prepared.tangent
    side_delta = sampled - closest
    side_norm = np.linalg.norm(side_delta, axis=1)
    side_radial = np.divide(
        side_delta,
        side_norm[:, None],
        out=np.zeros_like(side_delta),
        where=side_norm[:, None] > 0.0,
    )
    lower_delta = sampled - prepared.spine[0]
    lower_norm = np.linalg.norm(lower_delta, axis=1)
    lower_gradient = np.divide(
        lower_delta,
        lower_norm[:, None],
        out=np.zeros_like(lower_delta),
        where=lower_norm[:, None] > 0.0,
    )
    upper_delta = sampled - prepared.spine[-1]
    upper_norm = np.linalg.norm(upper_delta, axis=1)
    upper_gradient = np.divide(
        upper_delta,
        upper_norm[:, None],
        out=np.zeros_like(upper_delta),
        where=upper_norm[:, None] > 0.0,
    )
    side_gradient = (
        side_radial - radius_derivative[:, None] * prepared.tangent
    )
    canal_gradient = np.where(
        (axial < prepared.stations[0])[:, None],
        lower_gradient,
        np.where(
            (axial > prepared.stations[-1])[:, None],
            upper_gradient,
            side_gradient,
        ),
    )
    radius_gradient = np.where(
        (
            (axial >= prepared.stations[0])
            & (axial <= prepared.stations[-1])
        )[:, None],
        radius_derivative[:, None] * prepared.tangent,
        0.0,
    )
    factor, factor_derivative = _quartic_line_factor(canal, radius)
    coordinate_gradient = (
        canal_gradient
        / (_MUSCLE_FORMATION_SUPPORT_SCALE * radius[:, None])
        - canal[:, None]
        * radius_gradient
        / (
            _MUSCLE_FORMATION_SUPPORT_SCALE
            * radius[:, None]
            * radius[:, None]
        )
    )
    return (
        factor[:, None] * canal_gradient
        + canal[:, None]
        * factor_derivative[:, None]
        * coordinate_gradient
    )


def _prepared_skeleton_integral(
    part: GencylPart,
    mirrored: bool,
) -> Callable[[np.ndarray], np.ndarray] | MuscleFormationObstruction:
    prepared = _prepare_muscle_formation(part, mirrored)
    return (
        prepared
        if isinstance(prepared, MuscleFormationObstruction)
        else lambda points: _skeleton_integral_field(prepared, points)
    )


def _sdf_gencyl_profile(
    points: np.ndarray,
    spine: tuple[tuple[float, float, float], ...],
    radii: tuple[float, ...],
    profile: Profile,
) -> np.ndarray:
    sweep = _ProfileSweep(
        points,
        np.asarray(spine, dtype=np.float64),
        np.asarray(radii, dtype=np.float64),
        np.asarray(profile.exponents, dtype=np.float64),
        profile.depths,
        np.asarray(profile.offsets, dtype=np.float64),
        np.asarray(profile.rolls, dtype=np.float64),
        np.asarray(profile.up, dtype=np.float64),
    )
    return _profile_sweep_field(sweep)


def sdf_gencyl_profile(pts, spine, radii, profile):
    decoded = require_accepted(
        decode_part(
            {
                "id": "<unnamed>",
                "type": "gencyl",
                "spine": spine,
                "radii": radii,
                "profile": profile,
            }
        )
    )
    return part_sdf_checked(np.asarray(pts), decoded)


def part_sdf_checked(
    pts: np.ndarray,
    part: Part,
    mirrored: bool = False,
) -> np.ndarray | MuscleFormationObstruction:
    match part:
        case GencylPart(
            formation=MuscleFormationKind.SKELETON_INTEGRAL
        ) as formed_part:
            prepared = _prepare_muscle_formation(formed_part, mirrored)
            return (
                prepared
                if isinstance(prepared, MuscleFormationObstruction)
                else _skeleton_integral_field(prepared, pts)
            )
        case GencylPart(spine=spine, radii=radii, profile=None):
            instance_spine = np.asarray(spine, dtype=np.float64)
            reflected_spine = instance_spine * _MIRROR_3 if mirrored else instance_spine
            return sdf_gencyl(pts, reflected_spine, radii)
        case GencylPart(spine=spine, radii=radii, profile=Profile() as profile):
            instance_spine = np.asarray(spine, dtype=np.float64)
            reflected_spine = instance_spine * _MIRROR_3 if mirrored else instance_spine
            return _sdf_gencyl_profile(
                pts,
                tuple(tuple(map(float, row)) for row in reflected_spine),
                radii,
                _profile_for_instance(profile, mirrored),
            )
        case BlobPart(center=center, size=size, rotation=rotation):
            instance_center = np.asarray(center, dtype=np.float64)
            reflected_center = instance_center * _MIRROR_3 if mirrored else instance_center
            if rotation is None:
                return sdf_blob(pts, reflected_center, size)
            rotation_matrix = quat_to_matrix(_rotation_for_instance(rotation, mirrored))
            local_points = (pts - reflected_center) @ rotation_matrix
            return sdf_blob(local_points, np.zeros(3), size)
        case BoxPart(
            center=center,
            size=size,
            round_radius=round_radius,
            rotation=rotation,
        ):
            instance_center = np.asarray(center, dtype=np.float64)
            reflected_center = instance_center * _MIRROR_3 if mirrored else instance_center
            if rotation is None:
                return sdf_box(pts, reflected_center, size, round_radius)
            rotation_matrix = quat_to_matrix(_rotation_for_instance(rotation, mirrored))
            local_points = (pts - reflected_center) @ rotation_matrix
            return sdf_box(local_points, np.zeros(3), size, round_radius)
        case _ as unreachable:
            assert_never(unreachable)


def _prepared_gencyl_plain(
    spine: np.ndarray, radii, mirrored: bool
) -> Callable[[np.ndarray], np.ndarray]:
    reflected_spine = spine * _MIRROR_3 if mirrored else spine
    decoded_spine = np.asarray(reflected_spine, dtype=np.float64)
    decoded_radii = np.asarray(radii, dtype=np.float64)
    segments = tuple(
        (
            decoded_spine[index],
            axis,
            float(axis @ axis),
            decoded_radii[index],
            decoded_radii[index + 1],
        )
        for index in range(len(decoded_spine) - 1)
        for axis in (decoded_spine[index + 1] - decoded_spine[index],)
    )

    def evaluate(pts: np.ndarray) -> np.ndarray:
        sampled_points = np.asarray(pts)

        def segment_field(prepared) -> np.ndarray:
            start, axis, squared_length, start_radius, end_radius = prepared
            parameter = (
                np.zeros(sampled_points.shape[0])
                if squared_length < 1e-12
                else np.clip(
                    ((sampled_points - start) @ axis) / squared_length, 0.0, 1.0
                )
            )
            closest = start + parameter[:, None] * axis
            radius = start_radius + (end_radius - start_radius) * parameter
            return np.linalg.norm(sampled_points - closest, axis=1) - radius

        return reduce(
            np.minimum,
            map(segment_field, segments),
            np.full(sampled_points.shape[0], np.inf),
        )

    return evaluate


def _prepared_collinear_profile(
    sweep: _ProfileSweep, cover: _CollinearCover
) -> Callable[[np.ndarray], np.ndarray]:
    epsilon = 1.0e-12
    tangent = cover.tangent
    origin = sweep.spine[0]
    width_axis, depth_axis = _profile_axes(tangent, sweep.up)
    origin_width = float(origin @ width_axis)
    origin_depth = float(origin @ depth_axis)
    anchor = cover.anchor_coordinates
    anchor_last = anchor[-1]
    widths = sweep.widths
    exponents = sweep.exponents
    depth_values = (
        np.asarray(sweep.depths.ratios, dtype=np.float64)
        if isinstance(sweep.depths, RelativeProfileDepths)
        else np.asarray(sweep.depths.values, dtype=np.float64)
    )
    offsets_width = sweep.offsets[:, 0]
    offsets_depth = sweep.offsets[:, 1]
    rolls = sweep.rolls
    half_length = cover.length / 2.0

    def evaluate(pts: np.ndarray) -> np.ndarray:
        axial_coordinate = (pts - origin) @ tangent
        clipped_coordinate = np.clip(axial_coordinate, 0.0, anchor_last)
        width = _station_values(clipped_coordinate, anchor, widths)
        exponent = _station_values(clipped_coordinate, anchor, exponents)
        interpolated_depth = _station_values(
            clipped_coordinate, anchor, depth_values
        )
        depth = (
            width * interpolated_depth
            if isinstance(sweep.depths, RelativeProfileDepths)
            else interpolated_depth
        )
        offset_width = _station_values(
            clipped_coordinate, anchor, offsets_width
        )
        offset_depth = _station_values(
            clipped_coordinate, anchor, offsets_depth
        )
        roll = _station_values(clipped_coordinate, anchor, rolls)
        checked_width = np.maximum(width, epsilon)
        checked_depth = np.maximum(depth, epsilon)
        width_coordinate = pts @ width_axis - origin_width
        depth_coordinate = pts @ depth_axis - origin_depth
        unrolled = bool(np.all(roll == 0.0))
        angle = np.deg2rad(roll)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        section_width_coordinate = (
            width_coordinate
            if unrolled
            else width_coordinate * cosine + depth_coordinate * sine
        )
        section_depth_coordinate = (
            depth_coordinate
            if unrolled
            else -width_coordinate * sine + depth_coordinate * cosine
        )
        normalized_width = (
            np.abs(section_width_coordinate - offset_width) / checked_width
        )
        normalized_depth = (
            np.abs(section_depth_coordinate - offset_depth) / checked_depth
        )
        maximum = np.maximum(normalized_width, normalized_depth)
        minimum = np.minimum(normalized_width, normalized_depth)
        ratio = minimum / np.maximum(maximum, epsilon)
        radial_ratio = maximum * (1.0 + ratio**exponent) ** (1.0 / exponent)
        radial_distance = (radial_ratio - 1.0) * np.minimum(
            checked_width, checked_depth
        )
        axial_distance = np.abs(axial_coordinate - half_length) - half_length
        return np.minimum(
            np.maximum(radial_distance, axial_distance), 0.0
        ) + np.hypot(
            np.maximum(radial_distance, 0.0), np.maximum(axial_distance, 0.0)
        )

    return evaluate


def _prepared_gencyl_profile(
    spine: np.ndarray, radii, profile: Profile, mirrored: bool
) -> Callable[[np.ndarray], np.ndarray]:
    reflected_spine = spine * _MIRROR_3 if mirrored else spine
    instance_profile = _profile_for_instance(profile, mirrored)
    sweep = _ProfileSweep(
        np.empty((0, 3), dtype=np.float64),
        np.asarray(
            tuple(tuple(map(float, row)) for row in reflected_spine),
            dtype=np.float64,
        ),
        np.asarray(radii, dtype=np.float64),
        np.asarray(instance_profile.exponents, dtype=np.float64),
        instance_profile.depths,
        np.asarray(instance_profile.offsets, dtype=np.float64),
        np.asarray(instance_profile.rolls, dtype=np.float64),
        np.asarray(instance_profile.up, dtype=np.float64),
    )
    cover = _collinear_cover(sweep)
    if cover is not None:
        return _prepared_collinear_profile(sweep, cover)
    return lambda pts: _profile_sweep_field(
        _ProfileSweep(
            pts,
            sweep.spine,
            sweep.widths,
            sweep.exponents,
            sweep.depths,
            sweep.offsets,
            sweep.rolls,
            sweep.up,
        )
    )


def _prepared_blob(
    center, size, rotation: Rotation | None, mirrored: bool
) -> Callable[[np.ndarray], np.ndarray]:
    instance_center = np.asarray(center, dtype=np.float64)
    reflected_center = instance_center * _MIRROR_3 if mirrored else instance_center
    decoded_size = np.asarray(size, dtype=np.float64)
    minimum_size = float(np.min(decoded_size))
    if rotation is None:
        return lambda pts: (
            np.linalg.norm((pts - reflected_center) / decoded_size, axis=1) - 1.0
        ) * minimum_size
    rotation_matrix = quat_to_matrix(_rotation_for_instance(rotation, mirrored))
    center = np.zeros(3)
    return lambda pts: (
        np.linalg.norm(
            ((pts - reflected_center) @ rotation_matrix - center) / decoded_size,
            axis=1,
        )
        - 1.0
    ) * minimum_size


def _prepared_box(
    center, size, round_radius: float, rotation: Rotation | None, mirrored: bool
) -> Callable[[np.ndarray], np.ndarray]:
    instance_center = np.asarray(center, dtype=np.float64)
    reflected_center = instance_center * _MIRROR_3 if mirrored else instance_center
    half_extents = np.asarray(size, dtype=np.float64)
    radius = float(round_radius)

    def field(offset: np.ndarray) -> np.ndarray:
        return (
            np.linalg.norm(np.maximum(offset, 0.0), axis=1)
            + np.minimum(np.max(offset, axis=1), 0.0)
            - radius
        )

    if rotation is None:
        return lambda pts: field(np.abs(pts - reflected_center) - half_extents)
    rotation_matrix = quat_to_matrix(_rotation_for_instance(rotation, mirrored))
    center = np.zeros(3)
    return lambda pts: field(
        np.abs((pts - reflected_center) @ rotation_matrix - center) - half_extents
    )


def prepared_part_sdf(
    part: Part, mirrored: bool = False
) -> Callable[[np.ndarray], np.ndarray] | MuscleFormationObstruction:
    match part:
        case GencylPart(
            formation=MuscleFormationKind.SKELETON_INTEGRAL
        ) as formed_part:
            return _prepared_skeleton_integral(formed_part, mirrored)
        case GencylPart(spine=spine, radii=radii, profile=None):
            return _prepared_gencyl_plain(
                np.asarray(spine, dtype=np.float64), radii, mirrored
            )
        case GencylPart(spine=spine, radii=radii, profile=Profile() as profile):
            return _prepared_gencyl_profile(
                np.asarray(spine, dtype=np.float64), radii, profile, mirrored
            )
        case BlobPart(center=center, size=size, rotation=rotation):
            return _prepared_blob(center, size, rotation, mirrored)
        case BoxPart(
            center=center,
            size=size,
            round_radius=round_radius,
            rotation=rotation,
        ):
            return _prepared_box(center, size, round_radius, rotation, mirrored)
        case _ as unreachable:
            assert_never(unreachable)


@dataclass(frozen=True)
class PreparedGradient:
    evaluate: Callable[[np.ndarray], np.ndarray]
    magnitude_bounds: tuple[float, float]


def _prepared_blob_gradient(
    center,
    size,
    rotation: Rotation | None,
    mirrored: bool,
) -> PreparedGradient:
    instance_center = np.asarray(center, dtype=np.float64)
    reflected_center = (
        instance_center * _MIRROR_3 if mirrored else instance_center
    )
    decoded_size = np.asarray(size, dtype=np.float64)
    minimum_size = float(np.min(decoded_size))
    maximum_size = float(np.max(decoded_size))
    rotation_matrix = (
        None
        if rotation is None
        else quat_to_matrix(_rotation_for_instance(rotation, mirrored))
    )

    def evaluate(points: np.ndarray) -> np.ndarray:
        world_offset = np.asarray(points, dtype=np.float64) - reflected_center
        local_offset = (
            world_offset
            if rotation_matrix is None
            else world_offset @ rotation_matrix
        )
        normalized = local_offset / decoded_size
        normalized_norm = np.linalg.norm(normalized, axis=1)
        local_gradient = np.divide(
            minimum_size * local_offset,
            decoded_size * decoded_size * normalized_norm[:, None],
            out=np.zeros_like(local_offset),
            where=normalized_norm[:, None] > 0.0,
        )
        return (
            local_gradient
            if rotation_matrix is None
            else local_gradient @ rotation_matrix.T
        )

    return PreparedGradient(
        evaluate,
        (minimum_size / maximum_size, 1.0),
    )


def _prepared_box_gradient(
    center,
    size,
    rotation: Rotation | None,
    mirrored: bool,
) -> PreparedGradient:
    instance_center = np.asarray(center, dtype=np.float64)
    reflected_center = (
        instance_center * _MIRROR_3 if mirrored else instance_center
    )
    half_extents = np.asarray(size, dtype=np.float64)
    rotation_matrix = (
        None
        if rotation is None
        else quat_to_matrix(_rotation_for_instance(rotation, mirrored))
    )

    def evaluate(points: np.ndarray) -> np.ndarray:
        world_offset = np.asarray(points, dtype=np.float64) - reflected_center
        local_offset = (
            world_offset
            if rotation_matrix is None
            else world_offset @ rotation_matrix
        )
        offset = np.abs(local_offset) - half_extents
        positive = np.maximum(offset, 0.0)
        outside_norm = np.linalg.norm(positive, axis=1)
        outside_gradient = np.divide(
            positive * np.sign(local_offset),
            outside_norm[:, None],
            out=np.zeros_like(local_offset),
            where=outside_norm[:, None] > 0.0,
        )
        maximum = np.max(offset, axis=1)
        owners = offset == maximum[:, None]
        inside_gradient = (
            owners
            * np.sign(local_offset)
            * (np.count_nonzero(owners, axis=1) == 1)[:, None]
        )
        local_gradient = np.where(
            (outside_norm > 0.0)[:, None],
            outside_gradient,
            inside_gradient,
        )
        return (
            local_gradient
            if rotation_matrix is None
            else local_gradient @ rotation_matrix.T
        )

    return PreparedGradient(evaluate, (1.0, 1.0))


def _prepared_gencyl_gradient(
    spine,
    radii,
    mirrored: bool,
) -> PreparedGradient:
    decoded_spine = np.asarray(spine, dtype=np.float64)
    reflected_spine = (
        decoded_spine * _MIRROR_3 if mirrored else decoded_spine
    )
    decoded_radii = np.asarray(radii, dtype=np.float64)
    segments = tuple(
        (
            reflected_spine[index],
            reflected_spine[index + 1] - reflected_spine[index],
            float(
                (reflected_spine[index + 1] - reflected_spine[index])
                @ (reflected_spine[index + 1] - reflected_spine[index])
            ),
            decoded_radii[index],
            decoded_radii[index + 1],
        )
        for index in range(len(reflected_spine) - 1)
    )

    def evaluate(points: np.ndarray) -> np.ndarray:
        sampled = np.asarray(points, dtype=np.float64)

        def segment_record(
            segment: tuple[np.ndarray, np.ndarray, float, float, float],
        ) -> tuple[np.ndarray, np.ndarray]:
            start, axis, squared_length, start_radius, end_radius = segment
            unclipped = (
                np.zeros(sampled.shape[0], dtype=np.float64)
                if squared_length < 1.0e-12
                else ((sampled - start) @ axis) / squared_length
            )
            parameter = np.clip(unclipped, 0.0, 1.0)
            delta = sampled - (start + parameter[:, None] * axis)
            distance = np.linalg.norm(delta, axis=1)
            radial = np.divide(
                delta,
                distance[:, None],
                out=np.zeros_like(delta),
                where=distance[:, None] > 0.0,
            )
            radius_delta = end_radius - start_radius
            interior = (unclipped > 0.0) & (unclipped < 1.0)
            gradient = radial - (
                interior[:, None]
                * radius_delta
                * axis
                / max(squared_length, 1.0e-12)
            )
            field = (
                distance
                - start_radius
                - radius_delta * parameter
            )
            return field, gradient

        records = tuple(map(segment_record, segments))
        fields = np.stack(tuple(map(lambda record: record[0], records)), axis=1)
        gradients = np.stack(
            tuple(map(lambda record: record[1], records)),
            axis=1,
        )
        owners = np.argmin(fields, axis=1)
        selected = np.take_along_axis(
            gradients,
            owners[:, None, None],
            axis=1,
        )[:, 0, :]
        tied = (
            np.zeros(sampled.shape[0], dtype=bool)
            if fields.shape[1] == 1
            else (
                np.partition(fields, 1, axis=1)[:, 1]
                - np.min(fields, axis=1)
                <= 1.0e-12
            )
        )
        return np.where(tied[:, None], 0.0, selected)

    upper = max(
        map(
            lambda segment: (
                1.0
                if segment[2] < 1.0e-12
                else np.sqrt(
                    1.0
                    + (
                        (segment[4] - segment[3])
                        / np.sqrt(segment[2])
                    )
                    ** 2
                )
            ),
            segments,
        )
    )
    return PreparedGradient(evaluate, (1.0, float(upper)))


def prepared_part_gradient(
    part: Part,
    mirrored: bool = False,
) -> PreparedGradient | MuscleFormationObstruction | None:
    match part:
        case GencylPart(
            formation=MuscleFormationKind.SKELETON_INTEGRAL
        ) as formed_part:
            prepared = _prepare_muscle_formation(formed_part, mirrored)
            if isinstance(prepared, MuscleFormationObstruction):
                return prepared
            surface_integral = (
                1.0 - _MUSCLE_FORMATION_SUPPORT_SCALE ** -2.0
            ) ** 2.5
            canal_upper = np.sqrt(
                1.0 + _MUSCLE_FORMATION_SUPPORT_SCALE ** -2.0
            )
            factor_upper = 2.0 / (1.0 + surface_integral)
            derivative_upper = (
                15.0
                * np.sqrt(3.0)
                / (16.0 * (1.0 + surface_integral))
            )
            gradient_upper = (
                factor_upper * canal_upper
                + derivative_upper
                / _MUSCLE_FORMATION_SUPPORT_SCALE
                * (
                    canal_upper
                    + 1.0 / _MUSCLE_FORMATION_SUPPORT_SCALE
                )
            )
            return PreparedGradient(
                lambda points: _skeleton_integral_gradient(
                    prepared,
                    points,
                ),
                (1.0, float(gradient_upper)),
            )
        case GencylPart(spine=spine, radii=radii, profile=None) if len(spine) == 2:
            return _prepared_gencyl_gradient(spine, radii, mirrored)
        case GencylPart(profile=None):
            return None
        case GencylPart(profile=Profile()):
            return None
        case BlobPart(center=center, size=size, rotation=rotation):
            return _prepared_blob_gradient(
                center,
                size,
                rotation,
                mirrored,
            )
        case BoxPart(center=center, size=size, rotation=rotation):
            return _prepared_box_gradient(
                center,
                size,
                rotation,
                mirrored,
            )
        case _ as unreachable:
            assert_never(unreachable)


def part_sdf(pts, part, mirrored=False):
    decoded = require_accepted(decode_part(part))
    return part_sdf_checked(np.asarray(pts), decoded, bool(mirrored))


def rotation_matrix_for_instance(
    rotation: Rotation | None,
    mirrored: bool,
) -> np.ndarray | None:
    return (
        None
        if rotation is None
        else quat_to_matrix(_rotation_for_instance(rotation, mirrored))
    )
