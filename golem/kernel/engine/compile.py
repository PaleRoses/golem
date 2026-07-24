"""Checked graph descent, field gluing, and mesh lowering."""

from __future__ import annotations

import hashlib
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import cache, reduce
from itertools import accumulate, chain
from numbers import Integral
from pathlib import Path
from typing import Callable, Literal, assert_never

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse import diags, eye
from scipy.sparse.linalg import expm_multiply
from scipy.spatial import cKDTree

from .algebra import (
    PreparedGradient,
    carve_part_instances,
    compose_union,
    local_topology_obstruction,
    local_topology_counts,
    part_instances,
    prepared_part_gradient,
    part_sdf_checked,
    prepared_part_sdf,
    prepared_web_sdf,
    rotation_matrix_for_instance,
    sdf_difference,
    solve_muscle_formation,
    web_instances,
    web_sdf_checked,
)
from .types import (
    AcceptedComposition,
    AbsoluteProfileDepths,
    BlobPart,
    BoxPart,
    CertifiedLocalCompositionProblem,
    CertifiedOverlapWitness,
    CompositionOperator,
    CompositionSectionId,
    CompositionSupportRegion,
    CurvatureDisplacement,
    CurvatureSurfaceDetail,
    EvaluatedMorphology,
    GencylPart,
    GeometryDecodeFailure,
    GeometryGraph,
    GeometryObstruction,
    GeometryRule,
    GradientMagnitudeCertificate,
    FieldScaleUncalibrated,
    GradientDegeneracy,
    LegacyCompositionProblem,
    LocalCompositionEvidence,
    LocalSectionsIncompatible,
    MuscleFormationEvidence,
    MuscleFormationKind,
    MuscleFormationObstruction,
    NoCompatibleOverlap,
    Part,
    Profile,
    RejectedSurfaceFormation,
    RelativeProfileDepths,
    AcceptedSkinLayer,
    SkinContainmentUnsatisfied,
    SkinConvergenceUnsatisfied,
    SkinCorrespondence,
    SkinDomainEscape,
    SkinFoldover,
    SkinGradientDegeneracy,
    SkinLayerProblem,
    SkinOffsetUnsatisfied,
    SkinProjectionNonlocal,
    SkinProjectionUnsatisfied,
    SkinRelaxationEvidence,
    SkinRelaxationObstruction,
    SkinRelaxationPolicy,
    SkinRootAbsent,
    SkinRootMultiplicity,
    SkinStretchUnsatisfied,
    SpanningWeb,
    decode_graph,
    read_only_array,
    require_accepted,
)


_MIRROR = np.asarray((-1.0, 1.0, 1.0))
_FIELD_CHUNK_SIZE = 1_000_000
_WEB_FIELD_CHUNK_SIZE = 100_000
_FIELD_CACHE_COLD_ENV = "GOLEM_CIRCUIT_CACHE_COLD"
_FIELD_CACHE_SUBDIR = "field"
_FIELD_CACHE_MAX_BYTES = 1 << 30
_COMPOSED_CACHE_SUBDIR = "composed"
_COMPOSED_CACHE_MAX_BYTES = 4 << 30
_SECTION_CACHE_SUBDIR = "sections"
_SECTION_CACHE_MAX_BYTES = int(
    os.environ.get("GOLEM_SECTION_CACHE_MAX_BYTES", 1 << 35)
)
_SKIN_MEMO_MAX_ENTRIES = 2
_LOCAL_TOPOLOGY_CERTIFICATION_RESOLUTION = 17
_SKIN_CUBIC_INTERPOLATION_NODES = np.asarray(
    (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
    dtype=np.float64,
)
_SKIN_CUBIC_INVERSE_VANDERMONDE = np.asarray(
    (
        (-4.5, 13.5, -13.5, 4.5),
        (9.0, -22.5, 18.0, -4.5),
        (-5.5, 9.0, -4.5, 1.0),
        (1.0, 0.0, 0.0, 0.0),
    ),
    dtype=np.float64,
)

SEALED_SKIN_POLICY = SkinRelaxationPolicy(
    projection_descent_budget=64,
    descent_step_pitch=0.0625,
    projection_sample_count=17,
    iteration_budget=12,
    projection_iteration_budget=4,
    minimum_gradient=1.0e-8,
    minimum_containment_margin=0.0,
    offset_tolerance_pitch=0.05,
    stretch_lower_ratio=0.75,
    stretch_upper_ratio=1.25,
    minimum_stretch_reference_pitch=1.7320508075688772,
    minimum_face_area_reference_pitch_squared=1.0,
    minimum_face_orientation=0.0,
    minimum_face_area_ratio=0.5,
    convergence_tolerance_pitch=0.05,
    projection_tolerance_pitch=0.05,
    locality_radius_thickness=8.0,
    locality_radius_pitch=2.0,
    neighborhood_radius_pitch=2.0,
    tangential_step=0.5,
    neighbor_weight=0.3,
    maximum_tangential_step_pitch=0.25,
    maximum_projection_step_pitch=0.5,
)


@dataclass(frozen=True)
class _Bounds:
    lower: np.ndarray
    upper: np.ndarray

@dataclass(frozen=True)
class _SkinMemoEntry:
    vertices: np.ndarray
    normals: np.ndarray
    correspondence: SkinCorrespondence
    evidence: SkinRelaxationEvidence


_SKIN_MEMO: dict[str, _SkinMemoEntry] = {}

@dataclass(frozen=True)
class _PreparedCompositionSection:
    identifier: CompositionSectionId
    evaluate: Callable[[np.ndarray], np.ndarray]
    bounds: _Bounds
    gradient: PreparedGradient | None


@dataclass(frozen=True)
class _SampledCompositionSection:
    prepared: _PreparedCompositionSection
    field: np.ndarray
    gradient: np.ndarray | None


@dataclass(frozen=True)
class _SectionSample:
    field: np.ndarray
    gradient: np.ndarray | None


@dataclass(frozen=True)
class _OverlapAssessment:
    section: _PreparedCompositionSection
    witnesses: tuple[CertifiedOverlapWitness, ...]
    obstructions: tuple[GradientDegeneracy, ...]
    probe_count: int
    minimum_residual: float
    closest_point: tuple[float, float, float] | None


@dataclass(frozen=True)
class _OverlapDescent:
    compatible: tuple[_OverlapAssessment, ...]
    obstructions: tuple[GradientDegeneracy, ...]
    probe_count: int
    minimum_residual: float
    closest_point: tuple[float, float, float] | None


@dataclass(frozen=True)
class _ComposedField:
    field: np.ndarray
    sections: tuple[_SampledCompositionSection, ...]
    evidence: tuple[LocalCompositionEvidence, ...]


@dataclass(frozen=True)
class _SectionGather:
    result: _ComposedField | RejectedSurfaceFormation | np.ndarray | None
    pending: tuple[tuple[str, _SectionSample], ...]


@dataclass(frozen=True)
class _TrilinearSamples:
    values: np.ndarray
    gradients: np.ndarray
    inside: np.ndarray


@dataclass(frozen=True)
class _SkinRootState:
    lower: np.ndarray
    upper: np.ndarray
    lower_values: np.ndarray
    upper_values: np.ndarray


@dataclass(frozen=True)
class _SkinIterationState:
    vertices: np.ndarray
    minimum_gradient: float
    maximum_projection_residual: float
    domain_escape_indices: tuple[int, ...]
    degenerate_indices: tuple[int, ...]


def _profile_radial_bound(
    widths: tuple[float, ...],
    profile: Profile,
) -> float:
    decoded_widths = np.asarray(widths, dtype=np.float64)
    offset_bound = float(
        np.max(
            np.linalg.norm(
                np.asarray(profile.offsets, dtype=np.float64), axis=1
            )
        )
    )
    match profile.depths:
        case RelativeProfileDepths(ratios):
            return (
                float(np.max(decoded_widths))
                * max(1.0, float(np.max(ratios)))
                + offset_bound
            )
        case AbsoluteProfileDepths(values):
            return (
                float(
                    np.max(
                        np.hypot(
                            decoded_widths,
                            np.asarray(values, dtype=np.float64),
                        )
                    )
                )
                + offset_bound
            )
        case _ as unreachable:
            assert_never(unreachable)


def _profile_padding_scale(
    widths: tuple[float, ...],
    profile: Profile,
) -> float:
    match profile.depths:
        case RelativeProfileDepths(ratios):
            return 1.0 / max(float(np.min(ratios)), 1.0e-6)
        case AbsoluteProfileDepths(values):
            decoded_widths = np.asarray(widths, dtype=np.float64)
            decoded_depths = np.asarray(values, dtype=np.float64)
            minimum = np.maximum(
                np.minimum(decoded_widths, decoded_depths), 1.0e-6
            )
            maximum = np.maximum(decoded_widths, decoded_depths)
            return float(np.max(maximum / minimum))
        case _ as unreachable:
            assert_never(unreachable)


def _instance_bounds(
    geometry: Part | SpanningWeb,
    mirrored: bool,
) -> _Bounds:
    match geometry:
        case GencylPart(spine=spine, radii=radii, profile=profile):
            decoded_spine = np.asarray(spine, dtype=np.float64)
            instance_spine = decoded_spine * _MIRROR if mirrored else decoded_spine
            profile_radius = (
                float(np.max(radii))
                if profile is None
                else _profile_radial_bound(radii, profile)
            )
            return _Bounds(
                instance_spine.min(axis=0) - profile_radius,
                instance_spine.max(axis=0) + profile_radius,
            )
        case BlobPart(center=center, size=size, rotation=rotation):
            decoded_center = np.asarray(center, dtype=np.float64)
            instance_center = decoded_center * _MIRROR if mirrored else decoded_center
            decoded_size = np.asarray(size, dtype=np.float64)
            rotation_matrix = rotation_matrix_for_instance(rotation, mirrored)
            extent = (
                decoded_size
                if rotation_matrix is None
                else np.abs(rotation_matrix) @ decoded_size
            )
            return _Bounds(instance_center - extent, instance_center + extent)
        case BoxPart(
            center=center,
            size=size,
            round_radius=round_radius,
            rotation=rotation,
        ):
            decoded_center = np.asarray(center, dtype=np.float64)
            instance_center = decoded_center * _MIRROR if mirrored else decoded_center
            decoded_size = np.asarray(size, dtype=np.float64)
            rotation_matrix = rotation_matrix_for_instance(rotation, mirrored)
            extent = (
                decoded_size + round_radius
                if rotation_matrix is None
                else np.abs(rotation_matrix) @ decoded_size + round_radius
            )
            return _Bounds(instance_center - extent, instance_center + extent)
        case SpanningWeb(anchors=anchors):
            reflection = _MIRROR if mirrored else np.ones(3)
            points = np.concatenate(
                tuple(
                    map(
                        lambda anchor: np.asarray(
                            anchor.spine, dtype=np.float64
                        )
                        * reflection,
                        anchors,
                    )
                )
            )
            half_extents = np.concatenate(
                tuple(
                    map(
                        lambda anchor: np.asarray(
                            anchor.radii, dtype=np.float64
                        ),
                        anchors,
                    )
                )
            )[:, None]
            return _Bounds(
                np.min(points - half_extents, axis=0),
                np.max(points + half_extents, axis=0),
            )
        case _ as unreachable:
            assert_never(unreachable)


def graph_bounds_checked(graph: GeometryGraph, pad: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    def merge(
        bounds: _Bounds,
        instance: tuple[Part | SpanningWeb, bool],
    ) -> _Bounds:
        local = _instance_bounds(*instance)
        return _Bounds(
            np.minimum(bounds.lower, local.lower),
            np.maximum(bounds.upper, local.upper),
        )

    bounds = reduce(
        merge,
        chain(part_instances(graph), web_instances(graph)),
        _Bounds(np.full(3, np.inf), np.full(3, -np.inf)),
    )
    span = bounds.upper - bounds.lower
    return bounds.lower - pad * span, bounds.upper + pad * span


def graph_bounds(graph, pad=0.15):
    decoded = require_accepted(decode_graph(graph))
    return graph_bounds_checked(decoded, float(pad))


def _positive_instances(
    graph: GeometryGraph,
) -> tuple[tuple[Part | SpanningWeb, bool], ...]:
    return tuple(chain(part_instances(graph), web_instances(graph)))


def _uses_local_composition(graph: GeometryGraph) -> bool:
    return any(
        geometry.operator is CompositionOperator.LOCAL_BLEND
        for geometry, _mirrored in _positive_instances(graph)
    )


def _formation_instances(
    graph: GeometryGraph,
) -> tuple[tuple[GencylPart, bool], ...]:
    return tuple(
        (part, mirrored)
        for part, mirrored in chain(
            part_instances(graph),
            carve_part_instances(graph),
        )
        if isinstance(part, GencylPart)
        and part.formation is MuscleFormationKind.SKELETON_INTEGRAL
    )


def _uses_muscle_formation(graph: GeometryGraph) -> bool:
    return any(
        isinstance(part, GencylPart) and part.formation is not None
        for part in chain(graph.parts, graph.carves)
    )


def _assess_muscle_formations(
    graph: GeometryGraph,
) -> tuple[MuscleFormationEvidence, ...] | RejectedSurfaceFormation:
    assessments = tuple(
        solve_muscle_formation(part, mirrored)
        for part, mirrored in _formation_instances(graph)
    )
    obstructions = tuple(
        assessment
        for assessment in assessments
        if isinstance(assessment, MuscleFormationObstruction)
    )
    return (
        RejectedSurfaceFormation(obstructions)
        if obstructions
        else tuple(
            assessment
            for assessment in assessments
            if isinstance(assessment, MuscleFormationEvidence)
        )
    )


def _prepared_composition_section(
    instance: tuple[Part | SpanningWeb, bool],
) -> _PreparedCompositionSection | RejectedSurfaceFormation:
    geometry, mirrored = instance
    if isinstance(geometry, SpanningWeb):
        return _PreparedCompositionSection(
            CompositionSectionId(geometry.part_id, mirrored),
            prepared_web_sdf(geometry, mirrored),
            _instance_bounds(geometry, mirrored),
            None,
        )
    evaluate = prepared_part_sdf(geometry, mirrored)
    if isinstance(evaluate, MuscleFormationObstruction):
        return RejectedSurfaceFormation((evaluate,))
    gradient = prepared_part_gradient(geometry, mirrored)
    if isinstance(gradient, MuscleFormationObstruction):
        return RejectedSurfaceFormation((gradient,))
    return _PreparedCompositionSection(
        CompositionSectionId(geometry.part_id, mirrored),
        evaluate,
        _instance_bounds(geometry, mirrored),
        gradient,
    )


def _overlap_probe_points(
    left: _PreparedCompositionSection,
    right: _PreparedCompositionSection,
) -> tuple[
    np.ndarray,
    tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    _Bounds | None,
]:
    lower = np.maximum(left.bounds.lower, right.bounds.lower)
    upper = np.minimum(left.bounds.upper, right.bounds.upper)
    if bool(np.any(lower > upper)):
        return np.empty((0, 3), dtype=np.float64), None, None
    axes = tuple(
        np.linspace(lower[index], upper[index], 9)
        for index in range(3)
    )
    points = np.stack(
        np.meshgrid(*axes, indexing="ij", copy=False),
        axis=-1,
    ).reshape((-1, 3))
    return points, axes, _Bounds(lower, upper)


def _cell_zero_crossings(field: np.ndarray) -> np.ndarray:
    corners = np.stack(
        tuple(
            field[
                x : x + field.shape[0] - 1,
                y : y + field.shape[1] - 1,
                z : z + field.shape[2] - 1,
            ]
            for x in (0, 1)
            for y in (0, 1)
            for z in (0, 1)
        ),
        axis=0,
    )
    return (np.min(corners, axis=0) <= 0.0) & (
        np.max(corners, axis=0) >= 0.0
    )


def _gradient_scale(
    section: _PreparedCompositionSection,
) -> float:
    return (
        1.0
        if section.gradient is None
        else float(np.mean(section.gradient.magnitude_bounds))
    )


def _refined_overlap_witness(
    left: _PreparedCompositionSection,
    right: _PreparedCompositionSection,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    component: np.ndarray,
    radius: float,
    probe_count: int,
) -> CertifiedOverlapWitness | GradientDegeneracy | None:
    from scipy.optimize import least_squares

    component_indices = np.argwhere(component)
    lower_index = np.min(component_indices, axis=0)
    upper_index = np.max(component_indices, axis=0) + 1
    component_lower = np.asarray(
        tuple(axes[axis][lower_index[axis]] for axis in range(3)),
        dtype=np.float64,
    )
    component_upper = np.asarray(
        tuple(axes[axis][upper_index[axis]] for axis in range(3)),
        dtype=np.float64,
    )
    left_scale = _gradient_scale(left)
    right_scale = _gradient_scale(right)
    candidate_axes = tuple(
        axes[axis][
            (axes[axis] >= component_lower[axis])
            & (axes[axis] <= component_upper[axis])
        ]
        for axis in range(3)
    )
    candidates = np.stack(
        np.meshgrid(*candidate_axes, indexing="ij", copy=False),
        axis=-1,
    ).reshape((-1, 3))
    candidate_residuals = np.maximum(
        np.abs(left.evaluate(candidates)) / left_scale,
        np.abs(right.evaluate(candidates)) / right_scale,
    )
    seed = candidates[int(np.argmin(candidate_residuals))]

    def residual(point: np.ndarray) -> np.ndarray:
        sampled = np.asarray(point, dtype=np.float64).reshape((1, 3))
        return np.asarray(
            (
                float(left.evaluate(sampled)[0]) / left_scale,
                float(right.evaluate(sampled)[0]) / right_scale,
            ),
            dtype=np.float64,
        )

    point = (
        seed
        if bool(np.any(component_lower >= component_upper))
        else np.asarray(
            least_squares(
                residual,
                seed,
                bounds=(component_lower, component_upper),
                max_nfev=96,
                ftol=1.0e-12,
                xtol=1.0e-12,
                gtol=1.0e-12,
            ).x,
            dtype=np.float64,
        )
    )
    normalized_residual = float(np.max(np.abs(residual(point))))
    tolerance = max(
        float(np.linalg.norm(component_upper - component_lower))
        * 1.0e-7,
        1.0e-9,
    )
    if normalized_residual > tolerance:
        return None
    gradients = tuple(
        (
            None
            if section.gradient is None
            else section.gradient.evaluate(point.reshape((1, 3)))[0]
        )
        for section in (left, right)
    )
    gradient_norms = tuple(
        (
            None
            if gradient is None
            else float(np.linalg.norm(gradient))
        )
        for gradient in gradients
    )
    degeneracy = next(
        (
            GradientDegeneracy(
                section.identifier,
                probe_count,
                float(norm),
                1.0e-8,
            )
            for section, norm in zip(
                (left, right),
                gradient_norms,
                strict=True,
            )
            if norm is not None and norm < 1.0e-8
        ),
        None,
    )
    if degeneracy is not None:
        return degeneracy
    if gradients[0] is not None and gradients[1] is not None:
        transversality = float(
            np.linalg.norm(np.cross(gradients[0], gradients[1]))
            / max(
                float(gradient_norms[0]) * float(gradient_norms[1]),
                1.0e-12,
            )
        )
        if transversality < 1.0e-4:
            return None
    support_padding = max(float(radius), 0.0)
    support = CompositionSupportRegion(
        left.identifier,
        tuple(map(float, component_lower - support_padding)),
        tuple(map(float, component_upper + support_padding)),
    )
    return CertifiedOverlapWitness(
        tuple(map(float, point)),
        normalized_residual,
        probe_count,
        support,
    )


def _compatible_overlap(
    left: _PreparedCompositionSection,
    right: _PreparedCompositionSection,
    radius: float,
) -> _OverlapAssessment:
    points, axes, bounds = _overlap_probe_points(left, right)
    if axes is None or bounds is None:
        return _OverlapAssessment(
            left,
            (),
            (),
            0,
            float("inf"),
            None,
        )
    left_field = left.evaluate(points)
    right_field = right.evaluate(points)
    left_grid = left_field.reshape((9, 9, 9))
    right_grid = right_field.reshape((9, 9, 9))
    joint_cells = _cell_zero_crossings(left_grid) & _cell_zero_crossings(
        right_grid
    )
    from scipy.ndimage import label

    labels, component_count = label(
        joint_cells,
        structure=np.ones((3, 3, 3), dtype=np.int8),
    )
    refined = tuple(
        map(
            lambda component_id: _refined_overlap_witness(
                left,
                right,
                axes,
                labels == component_id,
                radius,
                len(points),
            ),
            range(1, int(component_count) + 1),
        )
    )
    witnesses = tuple(
        result
        for result in refined
        if isinstance(result, CertifiedOverlapWitness)
    )
    degeneracies = tuple(
        result
        for result in refined
        if isinstance(result, GradientDegeneracy)
    )
    scaled_residual = np.maximum(
        np.abs(left_field) / _gradient_scale(left),
        np.abs(right_field) / _gradient_scale(right),
    )
    closest_index = int(np.argmin(scaled_residual))
    return _OverlapAssessment(
        left,
        witnesses,
        degeneracies,
        len(points),
        float(np.min(scaled_residual)),
        tuple(map(float, points[closest_index])),
    )


def _compatible_overlaps(
    prior: tuple[_PreparedCompositionSection, ...],
    incoming: _PreparedCompositionSection,
    radius: float,
) -> _OverlapDescent:
    assessments = tuple(
        map(
            lambda section: _compatible_overlap(
                section,
                incoming,
                radius,
            ),
            prior,
        )
    )
    compatible = tuple(
        assessment for assessment in assessments if assessment.witnesses
    )
    degeneracies = tuple(
        obstruction
        for assessment in assessments
        for obstruction in assessment.obstructions
    )
    closest = min(
        assessments,
        key=lambda assessment: assessment.minimum_residual,
        default=None,
    )
    return _OverlapDescent(
        compatible,
        degeneracies,
        sum(map(lambda assessment: assessment.probe_count, assessments)),
        (
            float(closest.minimum_residual)
            if closest is not None
            else float("inf")
        ),
        closest.closest_point if closest is not None else None,
    )


def _overlapping_support_conflict(
    incoming: CompositionSectionId,
    assessments: tuple[_OverlapAssessment, ...],
) -> LocalSectionsIncompatible | None:
    regions = tuple(
        (assessment.section.identifier, witness.support)
        for assessment in assessments
        for witness in assessment.witnesses
    )

    def conflict(
        pair: tuple[
            tuple[CompositionSectionId, CompositionSupportRegion],
            tuple[CompositionSectionId, CompositionSupportRegion],
        ],
    ) -> LocalSectionsIncompatible | None:
        (left_id, left), (right_id, right) = pair
        lower = np.maximum(left.lower, right.lower)
        upper = np.minimum(left.upper, right.upper)
        return (
            LocalSectionsIncompatible(
                incoming,
                left_id,
                right_id,
                tuple(map(float, lower)),
                tuple(map(float, upper)),
            )
            if left_id != right_id and bool(np.all(lower < upper))
            else None
        )

    return next(
        (
            obstruction
            for left_index, left in enumerate(regions)
            for right in regions[left_index + 1 :]
            for obstruction in (conflict((left, right)),)
            if obstruction is not None
        ),
        None,
    )


def _compose_certified_field(
    state: _ComposedField | None,
    incoming: _SampledCompositionSection,
    operator: CompositionOperator,
    radius: float,
    points: np.ndarray,
    domain: _Bounds,
    topology_shape: tuple[int, int, int] | None,
) -> _ComposedField | RejectedSurfaceFormation:
    if state is None:
        return _ComposedField(
            incoming.field,
            (incoming,),
            (),
        )
    if operator is not CompositionOperator.LOCAL_BLEND:
        legacy = compose_union(
            LegacyCompositionProblem(
                state.field,
                incoming.field,
                operator,
                radius,
            )
        )
        if isinstance(legacy, RejectedSurfaceFormation):
            return legacy
        return _ComposedField(
            legacy.field,
            (*state.sections, incoming),
            state.evidence,
        )
    overlap = _compatible_overlaps(
        tuple(map(lambda section: section.prepared, state.sections)),
        incoming.prepared,
        radius,
    )
    if overlap.obstructions:
        return RejectedSurfaceFormation(overlap.obstructions)
    if not overlap.compatible:
        return RejectedSurfaceFormation(
            (
                NoCompatibleOverlap(
                    incoming.prepared.identifier,
                    tuple(
                        map(
                            lambda section: section.prepared.identifier,
                            state.sections,
                        )
                    ),
                    overlap.probe_count,
                    overlap.minimum_residual,
                    overlap.closest_point,
                ),
            )
        )
    support_conflict = _overlapping_support_conflict(
        incoming.prepared.identifier,
        overlap.compatible,
    )
    if support_conflict is not None:
        return RejectedSurfaceFormation((support_conflict,))
    sampled_by_identifier = dict(
        map(
            lambda section: (section.prepared.identifier, section),
            state.sections,
        )
    )
    missing_scale = tuple(
        FieldScaleUncalibrated(
            section.prepared.identifier,
            float("nan"),
            float("nan"),
            32.0,
        )
        for section in (
            *tuple(
                sampled_by_identifier[assessment.section.identifier]
                for assessment in overlap.compatible
            ),
            incoming,
        )
        if section.prepared.gradient is None
    )
    if missing_scale:
        return RejectedSurfaceFormation(missing_scale)

    def compose_pair(
        assessment: _OverlapAssessment,
    ) -> AcceptedComposition | RejectedSurfaceFormation:
        prior = sampled_by_identifier[assessment.section.identifier]
        prior_gradient = prior.prepared.gradient
        incoming_gradient = incoming.prepared.gradient
        if (
            prior.gradient is None
            or incoming.gradient is None
            or prior_gradient is None
            or incoming_gradient is None
        ):
            return RejectedSurfaceFormation(
                (
                    FieldScaleUncalibrated(
                        (
                            prior.prepared.identifier
                            if prior_gradient is None
                            else incoming.prepared.identifier
                        ),
                        float("nan"),
                        float("nan"),
                        32.0,
                    ),
                )
            )
        return compose_union(
            CertifiedLocalCompositionProblem(
                accumulated=prior.field,
                incoming=incoming.field,
                radius=radius,
                accumulated_gradient=prior.gradient,
                incoming_gradient=incoming.gradient,
                accumulated_gradient_certificate=(
                    GradientMagnitudeCertificate(
                        *prior_gradient.magnitude_bounds
                    )
                ),
                incoming_gradient_certificate=(
                    GradientMagnitudeCertificate(
                        *incoming_gradient.magnitude_bounds
                    )
                ),
                accumulated_section=prior.prepared.identifier,
                incoming_section=incoming.prepared.identifier,
                overlaps=assessment.witnesses,
                sample_points=points,
                domain_lower=tuple(map(float, domain.lower)),
                domain_upper=tuple(map(float, domain.upper)),
            )
        )

    pair_results = tuple(map(compose_pair, overlap.compatible))
    pair_obstructions = tuple(
        obstruction
        for result in pair_results
        if isinstance(result, RejectedSurfaceFormation)
        for obstruction in result.obstructions
    )
    if pair_obstructions:
        return RejectedSurfaceFormation(pair_obstructions)
    accepted_pairs = tuple(
        result
        for result in pair_results
        if isinstance(result, AcceptedComposition)
    )
    clean = np.minimum(state.field, incoming.field)
    composed = reduce(
        np.minimum,
        map(lambda result: result.field, accepted_pairs),
        clean,
    )
    topology_obstruction = local_topology_obstruction(
        incoming.prepared.identifier,
        clean,
        composed,
        topology_shape,
    )
    if topology_obstruction is not None:
        return RejectedSurfaceFormation((topology_obstruction,))
    clean_components, composed_components = local_topology_counts(
        clean,
        composed,
        topology_shape,
    )
    local_evidence = tuple(
        replace(
            result.evidence,
            clean_component_count=clean_components,
            composed_component_count=composed_components,
        )
        for result in accepted_pairs
        if result.evidence is not None
    )
    return _ComposedField(
        composed,
        (*state.sections, incoming),
        (*state.evidence, *local_evidence),
    )


@cache
def _field_source_salt() -> bytes:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.digest()


@cache
def _field_cache_root() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / ".golem-cache"
        / _FIELD_CACHE_SUBDIR
    )


def _field_cache_key(
    part: Part,
    mirrored: bool,
    blend_radius: float,
    localized_morphology: bool,
    grid_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> str:
    payload = (
        part,
        mirrored,
        blend_radius,
        localized_morphology,
        tuple(
            (float(axis[0]), float(axis[-1]), len(axis))
            for axis in grid_axes
        ),
        _field_source_salt(),
    )
    return hashlib.sha256(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _field_cache_bypassed() -> bool:
    return os.environ.get(
        _FIELD_CACHE_COLD_ENV, ""
    ).strip().lower() in ("1", "true", "yes")


def _prune_field_cache(
    root: Path,
    max_bytes: int = _FIELD_CACHE_MAX_BYTES,
    pattern: str = "*.pkl",
) -> None:
    records = tuple(
        sorted(
            (
                (path, path.stat())
                for path in root.glob(pattern)
            ),
            key=lambda record: (
                record[1].st_mtime_ns,
                record[0].name,
            ),
        )
    )
    excess = (
        sum(stat.st_size for _path, stat in records)
        - max_bytes
    )
    victim_count = (
        next(
            (
                index
                for index, removed in enumerate(
                    accumulate(stat.st_size for _path, stat in records),
                    start=1,
                )
                if removed >= excess
            ),
            0,
        )
        if excess > 0
        else 0
    )
    tuple(
        path.unlink(missing_ok=True)
        for path, _stat in records[:victim_count]
    )


@cache
def _composed_snapshot_root() -> Path:
    return _field_cache_root() / _COMPOSED_CACHE_SUBDIR


def _section_sample_root() -> Path:
    return _field_cache_root() / _SECTION_CACHE_SUBDIR


def _points_digest(points: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(points.tobytes())
    digest.update(repr(points.shape).encode())
    digest.update(str(points.dtype).encode())
    return digest.hexdigest()


def _composed_snapshot_key(
    graph: GeometryGraph,
    points: np.ndarray,
    points_digest: str | None = None,
) -> str:
    lower, upper = graph_bounds_checked(graph)
    payload = (
        tuple(
            (geometry, mirrored, geometry.operator, geometry.blend)
            for geometry, mirrored in _positive_instances(graph)
        ),
        tuple(carve_part_instances(graph)),
        graph.blend,
        (tuple(map(float, lower)), tuple(map(float, upper))),
        _LOCAL_TOPOLOGY_CERTIFICATION_RESOLUTION,
        points_digest if points_digest is not None else _points_digest(points),
        _field_source_salt(),
    )
    return hashlib.sha256(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _load_composed_snapshot(key: str) -> _ComposedField | None:
    path = _composed_snapshot_root() / f"{key}.pkl"
    if not path.exists():
        return None
    payload = path.read_bytes()
    os.utime(path, None)
    field, evidence = pickle.loads(payload)
    return _ComposedField(field, (), evidence)


def _store_composed_snapshot(key: str, result: _ComposedField) -> None:
    root = _composed_snapshot_root()
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f"{key}.{os.getpid()}.tmp"
    temporary.write_bytes(
        pickle.dumps(
            (result.field, result.evidence),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )
    os.replace(temporary, root / f"{key}.pkl")
    _prune_field_cache(root, _COMPOSED_CACHE_MAX_BYTES)


def _section_shape_signature(
    geometry: Part | SpanningWeb,
) -> tuple:
    match geometry:
        case GencylPart(
            spine=spine,
            radii=radii,
            profile=profile,
            formation=formation,
        ):
            return ("gencyl", formation, spine, radii, profile)
        case BlobPart(center=center, size=size, rotation=rotation):
            return ("blob", center, size, rotation)
        case BoxPart(
            center=center,
            size=size,
            round_radius=round_radius,
            rotation=rotation,
        ):
            return ("box", center, size, round_radius, rotation)
        case SpanningWeb(anchors=anchors):
            return ("web", anchors)
        case _ as unreachable:
            assert_never(unreachable)


def _section_sample_key(
    geometry: Part | SpanningWeb,
    mirrored: bool,
    points_digest: str,
    *,
    role: Literal["positive", "carve"],
) -> str:
    # Positive sections sample prepared evaluators with gradients; carves
    # sample part_sdf_checked without them. Sharing a key across roles
    # would assume a bitwise equivalence nobody proves.
    payload = (
        role,
        _section_shape_signature(geometry),
        mirrored,
        points_digest,
        _field_source_salt(),
    )
    return hashlib.sha256(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _load_section_sample(key: str) -> _SectionSample | None:
    path = _section_sample_root() / f"{key}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        field = archive["field"]
        gradient = archive["gradient"] if "gradient" in archive.files else None
    os.utime(path, None)
    return _SectionSample(field, gradient)


def _store_section_sample(key: str, sample: _SectionSample) -> None:
    root = _section_sample_root()
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f"{key}.{os.getpid()}.npz.tmp"
    arrays = (
        {"field": sample.field}
        if sample.gradient is None
        else {"field": sample.field, "gradient": sample.gradient}
    )
    with open(temporary, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, root / f"{key}.npz")
    _prune_field_cache(root, _SECTION_CACHE_MAX_BYTES, "*.npz")


def _flush_section_samples(
    pending: tuple[tuple[str, _SectionSample], ...],
) -> None:
    tuple(
        _store_section_sample(key, sample)
        for key, sample in pending
    )


def _grid_points(
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    grids = np.meshgrid(*axes, indexing="ij", copy=False)
    return np.stack(grids, axis=-1).reshape((-1, 3))


def _sample_sdf(
    points: np.ndarray,
    field: Callable[[np.ndarray], np.ndarray],
    chunk_size: int,
    executor: ThreadPoolExecutor,
) -> np.ndarray:
    chunks = tuple(
        map(
            lambda start: points[start : start + chunk_size],
            range(0, len(points), chunk_size),
        )
    )
    return (
        field(points)
        if len(chunks) <= 1
        else np.concatenate(
            tuple(executor.map(field, chunks))
        )
    )


def _sample_part_sdf(
    points: np.ndarray,
    part: Part,
    mirrored: bool,
    executor: ThreadPoolExecutor,
) -> np.ndarray | MuscleFormationObstruction:
    if (
        isinstance(part, GencylPart)
        and part.formation is MuscleFormationKind.SKELETON_INTEGRAL
    ):
        prepared = prepared_part_sdf(part, mirrored)
        return (
            prepared
            if isinstance(prepared, MuscleFormationObstruction)
            else _sample_sdf(
                points,
                prepared,
                _FIELD_CHUNK_SIZE,
                executor,
            )
        )
    return _sample_sdf(
        points,
        lambda sampled_points: part_sdf_checked(
            sampled_points, part, mirrored
        ),
        _FIELD_CHUNK_SIZE,
        executor,
    )


def _sample_web_sdf(
    points: np.ndarray,
    web: SpanningWeb,
    mirrored: bool,
    executor: ThreadPoolExecutor,
) -> np.ndarray:
    return _sample_sdf(
        points,
        prepared_web_sdf(web, mirrored),
        _WEB_FIELD_CHUNK_SIZE,
        executor,
    )


def _sampled_part_field_uncached(
    part: Part,
    mirrored: bool,
    blend_radius: float,
    localized_morphology: bool,
    grid_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    executor: ThreadPoolExecutor,
) -> (
    tuple[tuple[slice, slice, slice] | None, np.ndarray]
    | MuscleFormationObstruction
):
    match part:
        case GencylPart(
            spine=decoded_spine,
            radii=decoded_radii,
            profile=profile,
        ) if profile is not None and localized_morphology:
            spine = np.asarray(decoded_spine, dtype=np.float64) * (
                _MIRROR if mirrored else 1.0
            )
            radial_bound = _profile_radial_bound(decoded_radii, profile)
            field_padding = 2.0 * blend_radius * _profile_padding_scale(
                decoded_radii, profile
            )
            lower = np.min(spine, axis=0) - radial_bound - field_padding
            upper = np.max(spine, axis=0) + radial_bound + field_padding
        case BlobPart(
            center=center,
            size=decoded_size,
            rotation=rotation,
        ) if localized_morphology:
            instance_center = np.asarray(center, dtype=np.float64) * (
                _MIRROR if mirrored else 1.0
            )
            size = np.asarray(decoded_size, dtype=np.float64)
            support_scale = 1.0 + 2.0 * blend_radius / float(np.min(size))
            rotation_matrix = rotation_matrix_for_instance(rotation, mirrored)
            extent = (
                size * support_scale
                if rotation_matrix is None
                else np.abs(rotation_matrix) @ (size * support_scale)
            )
            lower = instance_center - extent
            upper = instance_center + extent
        case _:
            shape = tuple(map(len, grid_axes))
            sampled = _sample_part_sdf(
                _grid_points(grid_axes), part, mirrored, executor
            )
            return (
                sampled
                if isinstance(sampled, MuscleFormationObstruction)
                else (None, sampled.reshape(shape))
            )
    region = tuple(
        slice(
            int(np.searchsorted(axis_coordinates, lower[axis_index], side="left")),
            int(np.searchsorted(axis_coordinates, upper[axis_index], side="right")),
        )
        for axis_index, axis_coordinates in enumerate(grid_axes)
    )
    local_axes = tuple(
        axis_coordinates[axis_slice]
        for axis_coordinates, axis_slice in zip(grid_axes, region, strict=True)
    )
    shape = tuple(map(len, local_axes))
    sampled = _sample_part_sdf(
        _grid_points(local_axes), part, mirrored, executor
    )
    return (
        sampled
        if isinstance(sampled, MuscleFormationObstruction)
        else (region, sampled.reshape(shape))
    )


def _sampled_part_field(
    part: Part,
    mirrored: bool,
    blend_radius: float,
    localized_morphology: bool,
    grid_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    executor: ThreadPoolExecutor,
) -> (
    tuple[tuple[slice, slice, slice] | None, np.ndarray]
    | MuscleFormationObstruction
):
    if _field_cache_bypassed() or not localized_morphology:
        return _sampled_part_field_uncached(
            part,
            mirrored,
            blend_radius,
            localized_morphology,
            grid_axes,
            executor,
        )
    key = _field_cache_key(
        part, mirrored, blend_radius, localized_morphology, grid_axes
    )
    root = _field_cache_root()
    path = root / f"{key}.pkl"
    if path.exists():
        payload = path.read_bytes()
        os.utime(path, None)
        return pickle.loads(payload)
    sampled = _sampled_part_field_uncached(
        part,
        mirrored,
        blend_radius,
        localized_morphology,
        grid_axes,
        executor,
    )
    if isinstance(sampled, MuscleFormationObstruction):
        return sampled
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f"{key}.{os.getpid()}.tmp"
    temporary.write_bytes(
        pickle.dumps(sampled, protocol=pickle.HIGHEST_PROTOCOL)
    )
    os.replace(temporary, path)
    _prune_field_cache(root)
    return sampled


def _sample_graph_field_legacy_checked(
    graph: GeometryGraph,
    points: np.ndarray,
) -> np.ndarray | RejectedSurfaceFormation:
    lower, upper = graph_bounds_checked(graph)
    diagonal = float(np.linalg.norm(upper - lower))
    global_blend_radius = graph.blend * diagonal

    def merge_instance_field(
        field: np.ndarray | RejectedSurfaceFormation | None,
        instance: tuple[Part | SpanningWeb, bool],
    ) -> np.ndarray | RejectedSurfaceFormation:
        if isinstance(field, RejectedSurfaceFormation):
            return field
        geometry, mirrored = instance
        blend_radius = (
            global_blend_radius
            if geometry.blend is None
            else geometry.blend * diagonal
        )
        local_field = (
            web_sdf_checked(points, geometry, mirrored)
            if isinstance(geometry, SpanningWeb)
            else part_sdf_checked(points, geometry, mirrored)
        )
        if isinstance(local_field, MuscleFormationObstruction):
            return RejectedSurfaceFormation((local_field,))
        return (
            local_field
            if field is None
            else compose_union(
                LegacyCompositionProblem(
                    field,
                    local_field,
                    geometry.operator,
                    blend_radius,
                )
            ).field
        )

    sampled_positive = reduce(
        merge_instance_field,
        chain(part_instances(graph), web_instances(graph)),
        None,
    )
    sampled_field = (
        sampled_positive
        if sampled_positive is not None
        else np.full(points.shape[0], np.inf, dtype=np.float64)
    )
    if isinstance(sampled_field, RejectedSurfaceFormation):
        return sampled_field

    def subtract_carve_field(
        field: np.ndarray | RejectedSurfaceFormation,
        instance: tuple[Part, bool],
    ) -> np.ndarray | RejectedSurfaceFormation:
        if isinstance(field, RejectedSurfaceFormation):
            return field
        part, mirrored = instance
        carve = part_sdf_checked(points, part, mirrored)
        if isinstance(carve, MuscleFormationObstruction):
            return RejectedSurfaceFormation((carve,))
        return sdf_difference(
            field,
            carve,
        )

    return reduce(
        subtract_carve_field,
        carve_part_instances(graph),
        sampled_field
    )


def _sample_section_arrays(
    section: _PreparedCompositionSection,
    points: np.ndarray,
    chunk_size: int,
    executor: ThreadPoolExecutor | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    field = (
        section.evaluate(points)
        if executor is None
        else _sample_sdf(points, section.evaluate, chunk_size, executor)
    )
    gradient = (
        None
        if section.gradient is None
        else (
            section.gradient.evaluate(points)
            if executor is None
            else _sample_sdf(
                points, section.gradient.evaluate, chunk_size, executor
            )
        )
    )
    return field, gradient


def _sample_carve_array(
    part: Part,
    mirrored: bool,
    points: np.ndarray,
    executor: ThreadPoolExecutor | None,
) -> np.ndarray | MuscleFormationObstruction:
    return (
        part_sdf_checked(points, part, mirrored)
        if executor is None
        else _sample_part_sdf(points, part, mirrored, executor)
    )


def _sample_graph_field_certified_fold(
    graph: GeometryGraph,
    points: np.ndarray,
    topology_shape: tuple[int, int, int] | None = None,
    executor: ThreadPoolExecutor | None = None,
) -> _ComposedField | RejectedSurfaceFormation:
    lower, upper = graph_bounds_checked(graph)
    domain = _Bounds(lower, upper)
    diagonal = float(np.linalg.norm(upper - lower))
    global_blend_radius = graph.blend * diagonal
    instances = _positive_instances(graph)
    prepared_sections = tuple(map(_prepared_composition_section, instances))
    preparation_failure = next(
        (
            section
            for section in prepared_sections
            if isinstance(section, RejectedSurfaceFormation)
        ),
        None,
    )
    if preparation_failure is not None:
        return preparation_failure
    sections = tuple(
        section
        for section in prepared_sections
        if isinstance(section, _PreparedCompositionSection)
    )

    memo_active = topology_shape is None and not _field_cache_bypassed()
    points_digest = _points_digest(points) if memo_active else None

    def gather_section(
        geometry: Part | SpanningWeb,
        mirrored: bool,
        section: _PreparedCompositionSection,
        chunk_size: int,
    ) -> tuple[np.ndarray, np.ndarray | None, tuple[str, _SectionSample] | None]:
        if not memo_active:
            field, gradient = _sample_section_arrays(
                section, points, chunk_size, executor
            )
            return field, gradient, None
        key = _section_sample_key(
            geometry, mirrored, points_digest, role="positive"
        )
        cached = _load_section_sample(key)
        if cached is not None:
            return cached.field, cached.gradient, None
        field, gradient = _sample_section_arrays(
            section, points, chunk_size, executor
        )
        return field, gradient, (key, _SectionSample(field, gradient))

    def merge(
        state: _SectionGather,
        record: tuple[
            tuple[Part | SpanningWeb, bool],
            _PreparedCompositionSection,
        ],
    ) -> _SectionGather:
        composed, pending = state.result, state.pending
        if isinstance(composed, RejectedSurfaceFormation):
            return state
        (geometry, mirrored), section = record
        chunk_size = (
            _WEB_FIELD_CHUNK_SIZE
            if isinstance(geometry, SpanningWeb)
            else _FIELD_CHUNK_SIZE
        )
        field, gradient, write = gather_section(
            geometry, mirrored, section, chunk_size
        )
        radius = (
            global_blend_radius
            if geometry.blend is None
            else geometry.blend * diagonal
        )
        return _SectionGather(
            _compose_certified_field(
                composed,
                _SampledCompositionSection(section, field, gradient),
                geometry.operator,
                radius,
                points,
                domain,
                topology_shape,
            ),
            pending if write is None else (*pending, write),
        )

    positive_gather = reduce(
        merge,
        zip(instances, sections, strict=True),
        _SectionGather(None, ()),
    )
    positive = positive_gather.result
    if isinstance(positive, RejectedSurfaceFormation):
        return positive
    if not isinstance(positive, _ComposedField):
        obstruction = GeometryObstruction(
            "<graph>",
            "graph",
            GeometryRule.MALFORMED_GRAPH,
            "parts must be a non-empty sequence",
            True,
        )
        raise GeometryDecodeFailure((obstruction,))

    def gather_carve(
        part: Part,
        mirrored: bool,
    ) -> tuple[
        np.ndarray | MuscleFormationObstruction,
        tuple[str, _SectionSample] | None,
    ]:
        if not memo_active:
            return _sample_carve_array(part, mirrored, points, executor), None
        key = _section_sample_key(part, mirrored, points_digest, role="carve")
        cached = _load_section_sample(key)
        if cached is not None:
            return cached.field, None
        carve = _sample_carve_array(part, mirrored, points, executor)
        if isinstance(carve, MuscleFormationObstruction):
            return carve, None
        return carve, (key, _SectionSample(carve, None))

    def subtract_carve_field(
        state: _SectionGather,
        instance: tuple[Part, bool],
    ) -> _SectionGather:
        field, pending = state.result, state.pending
        if isinstance(field, RejectedSurfaceFormation):
            return state
        part, mirrored = instance
        carve, write = gather_carve(part, mirrored)
        if isinstance(carve, MuscleFormationObstruction):
            return _SectionGather(RejectedSurfaceFormation((carve,)), pending)
        return _SectionGather(
            sdf_difference(field, carve),
            pending if write is None else (*pending, write),
        )

    carve_gather = reduce(
        subtract_carve_field,
        carve_part_instances(graph),
        _SectionGather(positive.field, positive_gather.pending),
    )
    carved = carve_gather.result
    if isinstance(carved, RejectedSurfaceFormation):
        return carved
    if memo_active:
        _flush_section_samples(carve_gather.pending)
    return replace(positive, field=carved)


def _sample_graph_field_certified_checked(
    graph: GeometryGraph,
    points: np.ndarray,
    executor: ThreadPoolExecutor | None = None,
) -> _ComposedField | RejectedSurfaceFormation:
    if _field_cache_bypassed():
        return _sample_graph_field_certified_uncached(graph, points, executor)
    key = _composed_snapshot_key(graph, points)
    cached = _load_composed_snapshot(key)
    if cached is not None:
        return cached
    result = _sample_graph_field_certified_uncached(graph, points, executor)
    if isinstance(result, _ComposedField):
        _store_composed_snapshot(key, result)
    return result


def _sample_graph_field_certified_uncached(
    graph: GeometryGraph,
    points: np.ndarray,
    executor: ThreadPoolExecutor | None = None,
) -> _ComposedField | RejectedSurfaceFormation:
    lower, upper = graph_bounds_checked(graph)
    certification_axes = tuple(
        np.linspace(
            lower[index],
            upper[index],
            _LOCAL_TOPOLOGY_CERTIFICATION_RESOLUTION,
        )
        for index in range(3)
    )
    certification_points = _grid_points(certification_axes)
    certification_shape = (
        _LOCAL_TOPOLOGY_CERTIFICATION_RESOLUTION,
    ) * 3
    certified = _sample_graph_field_certified_fold(
        graph,
        certification_points,
        certification_shape,
        executor,
    )
    if isinstance(certified, RejectedSurfaceFormation):
        return certified
    if (
        points.shape == certification_points.shape
        and np.array_equal(points, certification_points)
    ):
        return certified
    sampled = _sample_graph_field_certified_fold(
        graph,
        points,
        None,
        executor,
    )
    if isinstance(sampled, RejectedSurfaceFormation):
        return sampled
    topology_evidence = dict(
        map(
            lambda evidence: (evidence.incoming, evidence),
            certified.evidence,
        )
    )
    return replace(
        sampled,
        evidence=tuple(
            replace(
                evidence,
                clean_component_count=topology_evidence[
                    evidence.incoming
                ].clean_component_count,
                composed_component_count=topology_evidence[
                    evidence.incoming
                ].composed_component_count,
            )
            for evidence in sampled.evidence
        ),
    )


def _trilinear_samples(
    field: np.ndarray,
    lower: np.ndarray,
    pitch: np.ndarray,
    points: np.ndarray,
) -> _TrilinearSamples:
    sampled = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    coordinates = (sampled - lower) / pitch
    maximum = np.asarray(field.shape, dtype=np.float64) - 1.0
    inside = np.all(
        (coordinates >= 0.0) & (coordinates <= maximum),
        axis=1,
    )
    clipped = np.clip(coordinates, 0.0, maximum)
    base = np.minimum(
        np.floor(clipped).astype(np.int64),
        np.asarray(field.shape, dtype=np.int64) - 2,
    )
    fractions = clipped - base
    ix, iy, iz = base.T
    fx, fy, fz = fractions.T
    c000 = field[ix, iy, iz]
    c100 = field[ix + 1, iy, iz]
    c010 = field[ix, iy + 1, iz]
    c110 = field[ix + 1, iy + 1, iz]
    c001 = field[ix, iy, iz + 1]
    c101 = field[ix + 1, iy, iz + 1]
    c011 = field[ix, iy + 1, iz + 1]
    c111 = field[ix + 1, iy + 1, iz + 1]
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    values = c0 * (1.0 - fz) + c1 * fz
    dx0 = (
        (c100 - c000) * (1.0 - fy)
        + (c110 - c010) * fy
    )
    dx1 = (
        (c101 - c001) * (1.0 - fy)
        + (c111 - c011) * fy
    )
    dx = (dx0 * (1.0 - fz) + dx1 * fz) / pitch[0]
    dy0 = c10 - c00
    dy1 = c11 - c01
    dy = (dy0 * (1.0 - fz) + dy1 * fz) / pitch[1]
    dz = (c1 - c0) / pitch[2]
    return _TrilinearSamples(
        values,
        np.stack((dx, dy, dz), axis=1),
        inside,
    )


def _symmetric_trilinear_gradients(
    field: np.ndarray,
    lower: np.ndarray,
    pitch: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    half_axis_offsets = 0.5 * np.diag(pitch)
    offsets = np.concatenate(
        (half_axis_offsets, -half_axis_offsets),
        axis=0,
    )
    sampled_points = (
        np.asarray(points, dtype=np.float64)[:, None, :]
        + offsets[None, :, :]
    )
    values = _trilinear_samples(
        field,
        lower,
        pitch,
        sampled_points.reshape((-1, 3)),
    ).values.reshape((-1, 6))
    return (values[:, :3] - values[:, 3:]) / pitch


def _skin_domain_boundary(
    field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = np.ogrid[
        : field.shape[0],
        : field.shape[1],
        : field.shape[2],
    ]
    boundary = (
        (x == 0)
        | (x == field.shape[0] - 1)
        | (y == 0)
        | (y == field.shape[1] - 1)
        | (z == 0)
        | (z == field.shape[2] - 1)
    )
    return boundary, np.argwhere(boundary & (field <= 0.0))


def _skin_formation_sections(
    graph: GeometryGraph,
    morphology: EvaluatedMorphology,
) -> (
    tuple[tuple[CompositionSectionId, ...], ...]
    | RejectedSurfaceFormation
):
    vertices = np.asarray(morphology.vertices, dtype=np.float64)
    lower, upper = graph_bounds_checked(graph)
    diagonal = float(np.linalg.norm(upper - lower))
    instances = _positive_instances(graph)
    prepared = tuple(map(_prepared_composition_section, instances))
    preparation_failures = tuple(
        obstruction
        for result in prepared
        if isinstance(result, RejectedSurfaceFormation)
        for obstruction in result.obstructions
    )
    if preparation_failures:
        return RejectedSurfaceFormation(preparation_failures)
    sections = tuple(
        section
        for section in prepared
        if isinstance(section, _PreparedCompositionSection)
    )

    def sample_section(
        state: tuple[
            np.ndarray,
            tuple[tuple[CompositionSectionId, np.ndarray], ...],
        ],
        record: tuple[
            tuple[Part | SpanningWeb, bool],
            _PreparedCompositionSection,
        ],
    ) -> tuple[
        np.ndarray,
        tuple[tuple[CompositionSectionId, np.ndarray], ...],
    ]:
        minimum, formed_sections = state
        (geometry, _mirrored), section = record
        values = section.evaluate(vertices) / _gradient_scale(section)
        formed = (
            isinstance(geometry, GencylPart)
            and geometry.formation
            is MuscleFormationKind.SKELETON_INTEGRAL
        )
        return (
            np.minimum(minimum, values),
            (
                (*formed_sections, (section.identifier, values))
                if formed
                else formed_sections
            ),
        )

    minimum_field, sampled_sections = reduce(
        sample_section,
        zip(instances, sections, strict=True),
        (
            np.full(len(vertices), np.inf, dtype=np.float64),
            (),
        ),
    )
    tie_tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * max(diagonal, np.finfo(np.float64).tiny)
    )

    def sections_at(vertex_index: int) -> tuple[CompositionSectionId, ...]:
        return tuple(
            identifier
            for identifier, values in sampled_sections
            if abs(values[vertex_index] - minimum_field[vertex_index])
            <= tie_tolerance
        )

    return tuple(map(sections_at, range(len(vertices))))


def _skin_correspondence(
    morphology: EvaluatedMorphology,
    formation_sections: tuple[
        tuple[CompositionSectionId, ...],
        ...,
    ],
) -> SkinCorrespondence:
    vertices = np.asarray(morphology.vertices, dtype=np.float64)
    return SkinCorrespondence(
        vertices,
        formation_sections,
    )


def _skin_memo_key(
    morphology: EvaluatedMorphology,
    problem: SkinLayerProblem,
    policy: SkinRelaxationPolicy,
    formation_sections: tuple[
        tuple[CompositionSectionId, ...],
        ...,
    ],
) -> str:
    digest = hashlib.sha256(
        pickle.dumps(
            (
                problem,
                policy,
                formation_sections,
                morphology.lower,
                morphology.upper,
                morphology.resolution,
                morphology.world_pitch,
            ),
            protocol=5,
        )
    )

    def update(array: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(array)
        digest.update(
            pickle.dumps(
                (contiguous.shape, contiguous.dtype.str),
                protocol=5,
            )
        )
        digest.update(memoryview(contiguous).cast("B"))

    update(np.asarray(morphology.field))
    update(np.asarray(morphology.vertices))
    update(np.asarray(morphology.faces))
    return digest.hexdigest()


def _skin_memo_lookup(
    key: str,
    skin_field: np.ndarray,
) -> AcceptedSkinLayer | None:
    entry = _SKIN_MEMO.get(key)
    return (
        None
        if entry is None
        else AcceptedSkinLayer(
            entry.vertices,
            entry.normals,
            read_only_array(skin_field),
            entry.correspondence,
            entry.evidence,
        )
    )


def _skin_memo_store(
    key: str,
    accepted: AcceptedSkinLayer,
) -> None:
    if key not in _SKIN_MEMO and len(_SKIN_MEMO) >= _SKIN_MEMO_MAX_ENTRIES:
        _SKIN_MEMO.pop(next(iter(_SKIN_MEMO)))
    _SKIN_MEMO[key] = _SkinMemoEntry(
        accepted.vertices,
        accepted.normals,
        accepted.correspondence,
        accepted.evidence,
    )


def _skin_projection_obstruction(
    problem: SkinLayerProblem,
    failure: (
        SkinDomainEscape
        | SkinGradientDegeneracy
        | SkinRootAbsent
        | SkinRootMultiplicity
        | SkinProjectionNonlocal
    ),
) -> SkinRelaxationObstruction:
    return SkinRelaxationObstruction(
        problem.region_ids,
        SkinProjectionUnsatisfied(failure),
    )


def _deduplicated_skin_root_parameters(
    ordered_parameters: np.ndarray,
    location_tolerances: np.ndarray,
) -> np.ndarray:
    finite_parameters = np.isfinite(ordered_parameters)
    previous_finite_parameters = np.pad(
        finite_parameters[:, :-1],
        ((0, 0), (1, 0)),
        constant_values=False,
    )
    previous_parameters = np.pad(
        ordered_parameters[:, :-1],
        ((0, 0), (1, 0)),
        constant_values=0.0,
    )
    parameter_gaps = np.subtract(
        ordered_parameters,
        previous_parameters,
        out=np.full_like(ordered_parameters, np.inf),
        where=finite_parameters & previous_finite_parameters,
    )
    distinct_parameters = finite_parameters & (
        ~previous_finite_parameters
        | (parameter_gaps > location_tolerances[:, None])
    )
    return np.sort(
        np.where(distinct_parameters, ordered_parameters, np.inf),
        axis=1,
    )


def _skin_segment_root_parameters(
    flesh_vertices: np.ndarray,
    skin_vertices: np.ndarray,
    field: np.ndarray,
    lower: np.ndarray,
    pitch: np.ndarray,
    minimum_gradient: float,
    root_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flesh = np.asarray(flesh_vertices, dtype=np.float64)
    skin = np.asarray(skin_vertices, dtype=np.float64)
    grid_lower = np.asarray(lower, dtype=np.float64)
    grid_pitch = np.asarray(pitch, dtype=np.float64)
    displacements = skin - flesh
    segment_lengths = np.linalg.norm(displacements, axis=1)
    nonzero_segments = segment_lengths > 0.0
    location_tolerances = np.divide(
        root_tolerance / minimum_gradient,
        segment_lengths,
        out=np.full_like(segment_lengths, np.inf),
        where=nonzero_segments,
    )

    grid_starts = (flesh - grid_lower) / grid_pitch
    grid_ends = (skin - grid_lower) / grid_pitch
    minimum_grid_coordinates = np.minimum(grid_starts, grid_ends)
    maximum_grid_coordinates = np.maximum(grid_starts, grid_ends)
    first_plane_indices = (
        np.floor(minimum_grid_coordinates).astype(np.int64) + 1
    )
    past_last_plane_indices = np.ceil(
        maximum_grid_coordinates
    ).astype(np.int64)
    plane_counts = np.maximum(
        past_last_plane_indices - first_plane_indices,
        0,
    )
    maximum_plane_count = int(np.max(plane_counts, initial=0))
    plane_offsets = np.arange(maximum_plane_count, dtype=np.int64)
    plane_indices = first_plane_indices[..., None] + plane_offsets
    grid_displacements = grid_ends - grid_starts
    valid_planes = (
        plane_offsets < plane_counts[..., None]
    ) & (grid_displacements[..., None] != 0.0)
    plane_parameters = np.divide(
        plane_indices - grid_starts[..., None],
        grid_displacements[..., None],
        out=np.ones_like(plane_indices, dtype=np.float64),
        where=valid_planes,
    )
    interior_plane_parameters = np.where(
        valid_planes
        & (plane_parameters > 0.0)
        & (plane_parameters < 1.0),
        plane_parameters,
        1.0,
    )
    segment_count = len(flesh)
    boundaries = np.sort(
        np.concatenate(
            (
                np.zeros((segment_count, 1), dtype=np.float64),
                interior_plane_parameters.reshape((segment_count, -1)),
                np.ones((segment_count, 1), dtype=np.float64),
            ),
            axis=1,
        ),
        axis=1,
    )
    interval_starts = boundaries[:, :-1]
    interval_ends = boundaries[:, 1:]
    interval_widths = interval_ends - interval_starts
    parameter_epsilon = 64.0 * np.finfo(np.float64).eps
    active_intervals = (
        interval_widths > parameter_epsilon
    ) & nonzero_segments[:, None]
    local_sample_parameters = (
        interval_starts[..., None]
        + interval_widths[..., None] * _SKIN_CUBIC_INTERPOLATION_NODES
    )
    sample_points = (
        flesh[:, None, None, :]
        + local_sample_parameters[..., None]
        * displacements[:, None, None, :]
    )
    sample_values = _trilinear_samples(
        field,
        grid_lower,
        grid_pitch,
        sample_points.reshape((-1, 3)),
    ).values.reshape(local_sample_parameters.shape)
    coefficients = np.einsum(
        "...j,kj->...k",
        sample_values,
        _SKIN_CUBIC_INVERSE_VANDERMONDE,
    )
    cubic_coefficients = coefficients[..., 0]
    quadratic_coefficients = coefficients[..., 1]
    linear_coefficients = coefficients[..., 2]
    constant_coefficients = coefficients[..., 3]
    coefficient_scales = np.max(np.abs(coefficients), axis=-1)
    coefficient_tolerances = (
        128.0 * np.finfo(np.float64).eps * coefficient_scales
    )
    riding_intervals = active_intervals & np.all(
        np.abs(coefficients) <= root_tolerance,
        axis=-1,
    )
    cubic_intervals = (
        active_intervals
        & ~riding_intervals
        & (np.abs(cubic_coefficients) > coefficient_tolerances)
    )
    quadratic_intervals = (
        active_intervals
        & ~riding_intervals
        & ~cubic_intervals
        & (np.abs(quadratic_coefficients) > coefficient_tolerances)
    )
    linear_intervals = (
        active_intervals
        & ~riding_intervals
        & ~cubic_intervals
        & ~quadratic_intervals
        & (np.abs(linear_coefficients) > coefficient_tolerances)
    )

    normalized_quadratic = np.divide(
        quadratic_coefficients,
        cubic_coefficients,
        out=np.zeros_like(cubic_coefficients),
        where=cubic_intervals,
    )
    normalized_linear = np.divide(
        linear_coefficients,
        cubic_coefficients,
        out=np.zeros_like(cubic_coefficients),
        where=cubic_intervals,
    )
    normalized_constant = np.divide(
        constant_coefficients,
        cubic_coefficients,
        out=np.zeros_like(cubic_coefficients),
        where=cubic_intervals,
    )
    depressed_linear = (
        normalized_linear - normalized_quadratic**2 / 3.0
    )
    depressed_constant = (
        2.0 * normalized_quadratic**3 / 27.0
        - normalized_quadratic * normalized_linear / 3.0
        + normalized_constant
    )
    half_depressed_constant = depressed_constant / 2.0
    third_depressed_linear = depressed_linear / 3.0
    cubic_discriminants = (
        half_depressed_constant**2 + third_depressed_linear**3
    )
    cubic_discriminant_tolerances = (
        128.0
        * np.finfo(np.float64).eps
        * (
            half_depressed_constant**2
            + np.abs(third_depressed_linear**3)
        )
    )
    one_root_cubics = cubic_intervals & (
        cubic_discriminants > cubic_discriminant_tolerances
    )
    three_root_cubics = cubic_intervals & (
        cubic_discriminants < -cubic_discriminant_tolerances
    )
    repeated_root_cubics = (
        cubic_intervals & ~one_root_cubics & ~three_root_cubics
    )
    cubic_discriminant_roots = np.sqrt(
        np.maximum(cubic_discriminants, 0.0)
    )
    dominant_cardano_terms = np.cbrt(
        -half_depressed_constant
        - np.copysign(
            cubic_discriminant_roots,
            depressed_constant,
        )
    )
    companion_cardano_terms = np.divide(
        -third_depressed_linear,
        dominant_cardano_terms,
        out=np.zeros_like(dominant_cardano_terms),
        where=one_root_cubics & (dominant_cardano_terms != 0.0),
    )
    cardano_roots = (
        dominant_cardano_terms
        + companion_cardano_terms
        - normalized_quadratic / 3.0
    )
    trigonometric_scales = np.sqrt(
        np.maximum(-(third_depressed_linear**3), 0.0)
    )
    trigonometric_arguments = np.divide(
        -half_depressed_constant,
        trigonometric_scales,
        out=np.zeros_like(half_depressed_constant),
        where=three_root_cubics & (trigonometric_scales > 0.0),
    )
    trigonometric_angles = np.arccos(
        np.clip(trigonometric_arguments, -1.0, 1.0)
    ) / 3.0
    trigonometric_radii = 2.0 * np.sqrt(
        np.maximum(-third_depressed_linear, 0.0)
    )
    trigonometric_roots = np.stack(
        (
            trigonometric_radii * np.cos(trigonometric_angles),
            trigonometric_radii
            * np.cos(trigonometric_angles - 2.0 * np.pi / 3.0),
            trigonometric_radii
            * np.cos(trigonometric_angles - 4.0 * np.pi / 3.0),
        ),
        axis=-1,
    ) - normalized_quadratic[..., None] / 3.0
    repeated_cardano_terms = np.cbrt(-half_depressed_constant)
    repeated_cubic_roots = np.stack(
        (
            2.0 * repeated_cardano_terms
            - normalized_quadratic / 3.0,
            -repeated_cardano_terms - normalized_quadratic / 3.0,
        ),
        axis=-1,
    )
    cubic_root_candidates = np.stack(
        (
            np.where(
                one_root_cubics,
                cardano_roots,
                np.where(
                    three_root_cubics,
                    trigonometric_roots[..., 0],
                    np.where(
                        repeated_root_cubics,
                        repeated_cubic_roots[..., 0],
                        np.inf,
                    ),
                ),
            ),
            np.where(
                three_root_cubics,
                trigonometric_roots[..., 1],
                np.where(
                    repeated_root_cubics,
                    repeated_cubic_roots[..., 1],
                    np.inf,
                ),
            ),
            np.where(
                three_root_cubics,
                trigonometric_roots[..., 2],
                np.inf,
            ),
        ),
        axis=-1,
    )

    quadratic_discriminants = (
        linear_coefficients**2
        - 4.0 * quadratic_coefficients * constant_coefficients
    )
    quadratic_discriminant_tolerances = (
        128.0
        * np.finfo(np.float64).eps
        * (
            linear_coefficients**2
            + np.abs(
                4.0
                * quadratic_coefficients
                * constant_coefficients
            )
        )
    )
    two_root_quadratics = quadratic_intervals & (
        quadratic_discriminants > quadratic_discriminant_tolerances
    )
    repeated_root_quadratics = quadratic_intervals & (
        np.abs(quadratic_discriminants)
        <= quadratic_discriminant_tolerances
    )
    quadratic_discriminant_roots = np.sqrt(
        np.maximum(quadratic_discriminants, 0.0)
    )
    stable_quadratic_terms = -0.5 * (
        linear_coefficients
        + np.copysign(
            quadratic_discriminant_roots,
            linear_coefficients,
        )
    )
    first_quadratic_roots = np.divide(
        stable_quadratic_terms,
        quadratic_coefficients,
        out=np.zeros_like(stable_quadratic_terms),
        where=two_root_quadratics,
    )
    second_quadratic_roots = np.divide(
        constant_coefficients,
        stable_quadratic_terms,
        out=np.zeros_like(stable_quadratic_terms),
        where=two_root_quadratics & (stable_quadratic_terms != 0.0),
    )
    repeated_quadratic_roots = np.divide(
        -linear_coefficients,
        2.0 * quadratic_coefficients,
        out=np.zeros_like(linear_coefficients),
        where=repeated_root_quadratics,
    )
    quadratic_root_candidates = np.stack(
        (
            np.where(
                two_root_quadratics,
                first_quadratic_roots,
                np.where(
                    repeated_root_quadratics,
                    repeated_quadratic_roots,
                    np.inf,
                ),
            ),
            np.where(
                two_root_quadratics,
                second_quadratic_roots,
                np.inf,
            ),
        ),
        axis=-1,
    )
    linear_roots = np.divide(
        -constant_coefficients,
        linear_coefficients,
        out=np.zeros_like(constant_coefficients),
        where=linear_intervals,
    )
    linear_root_candidates = np.where(
        linear_intervals,
        linear_roots,
        np.inf,
    )[..., None]
    analytic_root_candidates = np.concatenate(
        (
            cubic_root_candidates,
            quadratic_root_candidates,
            linear_root_candidates,
        ),
        axis=-1,
    )
    finite_analytic_candidates = np.isfinite(
        analytic_root_candidates
    )

    def analytic_values(parameters: np.ndarray) -> np.ndarray:
        return (
            (
                (
                    cubic_coefficients[..., None] * parameters
                    + quadratic_coefficients[..., None]
                )
                * parameters
                + linear_coefficients[..., None]
            )
            * parameters
            + constant_coefficients[..., None]
        )

    def polish_analytic_candidates(
        candidates: np.ndarray,
        _iteration: int,
    ) -> np.ndarray:
        safe_candidates = np.where(
            finite_analytic_candidates,
            candidates,
            0.0,
        )
        values = analytic_values(safe_candidates)
        derivatives = (
            (
                3.0
                * cubic_coefficients[..., None]
                * safe_candidates
                + 2.0 * quadratic_coefficients[..., None]
            )
            * safe_candidates
            + linear_coefficients[..., None]
        )
        derivative_uncertainties = (
            coefficient_tolerances[..., None]
            * (
                3.0 * np.abs(safe_candidates) ** 2
                + 2.0 * np.abs(safe_candidates)
                + 1.0
            )
        )
        corrections = np.divide(
            values,
            derivatives,
            out=np.zeros_like(values),
            where=finite_analytic_candidates
            & (np.abs(derivatives) > derivative_uncertainties),
        )
        proposals = safe_candidates - corrections
        proposal_values = analytic_values(proposals)
        accepted_proposals = (
            finite_analytic_candidates
            & np.isfinite(proposals)
            & (proposals >= 0.0)
            & (proposals <= 1.0)
            & (np.abs(proposal_values) < np.abs(values))
        )
        return np.where(
            accepted_proposals,
            proposals,
            candidates,
        )

    analytic_root_candidates = reduce(
        polish_analytic_candidates,
        range(2),
        analytic_root_candidates,
    )
    safe_analytic_candidates = np.where(
        finite_analytic_candidates,
        analytic_root_candidates,
        0.0,
    )
    analytic_residuals = analytic_values(safe_analytic_candidates)
    absolute_analytic_candidates = np.abs(safe_analytic_candidates)
    analytic_backward_errors = (
        coefficient_tolerances[..., None]
        * (
            absolute_analytic_candidates**3
            + absolute_analytic_candidates**2
            + absolute_analytic_candidates
            + 1.0
        )
        + 8.0
        * np.finfo(np.float64).eps
        * (
            (
                (
                    np.abs(cubic_coefficients)[..., None]
                    * absolute_analytic_candidates
                    + np.abs(quadratic_coefficients)[..., None]
                )
                * absolute_analytic_candidates
                + np.abs(linear_coefficients)[..., None]
            )
            * absolute_analytic_candidates
            + np.abs(constant_coefficients)[..., None]
        )
    )
    residual_valid_analytic_candidates = (
        finite_analytic_candidates
        & (
            np.abs(analytic_residuals)
            <= root_tolerance + analytic_backward_errors
        )
    )
    coincident_repeated_cubic_roots = repeated_root_cubics & (
        np.abs(
            safe_analytic_candidates[..., 0]
            - safe_analytic_candidates[..., 1]
        )
        * interval_widths
        <= parameter_epsilon
    )
    valid_cubic_root_candidates = np.stack(
        (
            residual_valid_analytic_candidates[..., 0],
            residual_valid_analytic_candidates[..., 1]
            & ~coincident_repeated_cubic_roots,
            residual_valid_analytic_candidates[..., 2],
        ),
        axis=-1,
    )
    valid_analytic_candidates = np.concatenate(
        (
            valid_cubic_root_candidates,
            residual_valid_analytic_candidates[..., 3:],
        ),
        axis=-1,
    )

    cubic_stationary_discriminants = (
        quadratic_coefficients**2
        - 3.0 * cubic_coefficients * linear_coefficients
    )
    cubic_stationary_tolerances = (
        128.0
        * np.finfo(np.float64).eps
        * (
            quadratic_coefficients**2
            + np.abs(
                3.0 * cubic_coefficients * linear_coefficients
            )
        )
    )
    distinct_cubic_stationary_points = cubic_intervals & (
        cubic_stationary_discriminants
        > cubic_stationary_tolerances
    )
    cubic_stationary_roots = np.sqrt(
        np.maximum(cubic_stationary_discriminants, 0.0)
    )
    stable_cubic_stationary_terms = (
        -quadratic_coefficients
        - np.copysign(
            cubic_stationary_roots,
            quadratic_coefficients,
        )
    )
    first_cubic_stationary_points = np.divide(
        stable_cubic_stationary_terms,
        3.0 * cubic_coefficients,
        out=np.zeros_like(stable_cubic_stationary_terms),
        where=distinct_cubic_stationary_points,
    )
    second_cubic_stationary_points = np.divide(
        linear_coefficients,
        stable_cubic_stationary_terms,
        out=np.zeros_like(stable_cubic_stationary_terms),
        where=distinct_cubic_stationary_points
        & (stable_cubic_stationary_terms != 0.0),
    )
    quadratic_stationary_points = np.divide(
        -linear_coefficients,
        2.0 * quadratic_coefficients,
        out=np.zeros_like(linear_coefficients),
        where=quadratic_intervals,
    )
    stationary_candidates = np.stack(
        (
            np.where(
                distinct_cubic_stationary_points,
                first_cubic_stationary_points,
                np.inf,
            ),
            np.where(
                distinct_cubic_stationary_points,
                second_cubic_stationary_points,
                np.inf,
            ),
            np.where(
                quadratic_intervals,
                quadratic_stationary_points,
                np.inf,
            ),
        ),
        axis=-1,
    )
    finite_stationary_candidates = np.isfinite(
        stationary_candidates
    )
    safe_stationary_candidates = np.where(
        finite_stationary_candidates,
        stationary_candidates,
        0.0,
    )
    stationary_values = (
        (
            (
                cubic_coefficients[..., None]
                * safe_stationary_candidates
                + quadratic_coefficients[..., None]
            )
            * safe_stationary_candidates
            + linear_coefficients[..., None]
        )
        * safe_stationary_candidates
        + constant_coefficients[..., None]
    )
    final_intervals = active_intervals & (interval_ends == 1.0)
    valid_analytic_candidates &= (
        analytic_root_candidates >= 0.0
    ) & (
        (analytic_root_candidates < 1.0)
        | (
            final_intervals[..., None]
            & (analytic_root_candidates <= 1.0)
        )
    )
    boundary_location_tolerances = np.minimum(
        location_tolerances,
        np.cbrt(np.finfo(np.float64).eps),
    )
    boundary_start_locations = (
        active_intervals
        & (np.abs(sample_values[..., 0]) <= root_tolerance)
    )
    boundary_end_locations = (
        active_intervals
        & (np.abs(sample_values[..., 3]) <= root_tolerance)
    )
    safe_boundary_analytic_candidates = np.where(
        valid_analytic_candidates,
        analytic_root_candidates,
        0.0,
    )
    analytic_start_distances = (
        np.abs(safe_boundary_analytic_candidates)
        * interval_widths[..., None]
    )
    analytic_end_distances = (
        np.abs(1.0 - safe_boundary_analytic_candidates)
        * interval_widths[..., None]
    )
    nearest_analytic_start_distances = np.min(
        np.where(
            valid_analytic_candidates,
            analytic_start_distances,
            np.inf,
        ),
        axis=-1,
    )
    nearest_analytic_end_distances = np.min(
        np.where(
            valid_analytic_candidates,
            analytic_end_distances,
            np.inf,
        ),
        axis=-1,
    )
    analytic_boundary_representations = (
        boundary_start_locations[..., None]
        & (
            nearest_analytic_start_distances[..., None]
            <= location_tolerances[:, None, None]
        )
        & (
            analytic_start_distances
            <= (
                nearest_analytic_start_distances[..., None]
                + boundary_location_tolerances[:, None, None]
            )
        )
    ) | (
        boundary_end_locations[..., None]
        & (
            nearest_analytic_end_distances[..., None]
            <= location_tolerances[:, None, None]
        )
        & (
            analytic_end_distances
            <= (
                nearest_analytic_end_distances[..., None]
                + boundary_location_tolerances[:, None, None]
            )
        )
    )
    valid_analytic_candidates &= ~analytic_boundary_representations
    safe_analytic_candidates = np.where(
        valid_analytic_candidates,
        analytic_root_candidates,
        0.0,
    )
    candidate_stationary_locations = (
        finite_stationary_candidates
        & (np.abs(stationary_values) <= root_tolerance)
        & (stationary_candidates >= 0.0)
        & (
            (stationary_candidates < 1.0)
            | (
                final_intervals[..., None]
                & (stationary_candidates <= 1.0)
            )
        )
    )
    stationary_analytic_offsets = (
        safe_analytic_candidates[..., None, :]
        - safe_stationary_candidates[..., :, None]
    ) * interval_widths[..., None, None]
    stationary_analytic_pairs = (
        candidate_stationary_locations[..., None]
        & valid_analytic_candidates[..., None, :]
    )
    stationary_analytic_distances = np.abs(
        stationary_analytic_offsets
    )
    stationary_left_pairs = stationary_analytic_pairs & (
        stationary_analytic_offsets <= 0.0
    )
    stationary_right_pairs = stationary_analytic_pairs & (
        stationary_analytic_offsets >= 0.0
    )
    nearest_left_distances = np.min(
        np.where(
            stationary_left_pairs,
            stationary_analytic_distances,
            np.inf,
        ),
        axis=-1,
    )
    nearest_right_distances = np.min(
        np.where(
            stationary_right_pairs,
            stationary_analytic_distances,
            np.inf,
        ),
        axis=-1,
    )
    bracketing_distance_minima = np.minimum(
        nearest_left_distances,
        nearest_right_distances,
    )
    bracketing_distance_maxima = np.maximum(
        nearest_left_distances,
        nearest_right_distances,
    )
    balanced_stationary_brackets = (
        np.isfinite(nearest_left_distances)
        & np.isfinite(nearest_right_distances)
        & (
            nearest_left_distances + nearest_right_distances
            <= location_tolerances[:, None, None]
        )
        & (
            bracketing_distance_maxima
            <= (
                2.0 * bracketing_distance_minima
                + boundary_location_tolerances[:, None, None]
            )
        )
    )
    coincident_stationary_pairs = stationary_analytic_pairs & (
        stationary_analytic_distances
        <= boundary_location_tolerances[:, None, None, None]
    )
    nearest_bracketing_pairs = (
        stationary_left_pairs
        & (
            stationary_analytic_distances
            <= (
                nearest_left_distances[..., None]
                + boundary_location_tolerances[:, None, None, None]
            )
        )
    ) | (
        stationary_right_pairs
        & (
            stationary_analytic_distances
            <= (
                nearest_right_distances[..., None]
                + boundary_location_tolerances[:, None, None, None]
            )
        )
    )
    canonical_stationary_locations = (
        candidate_stationary_locations
        & (
            np.any(coincident_stationary_pairs, axis=-1)
            | balanced_stationary_brackets
        )
    )
    analytic_tangency_representations = np.any(
        canonical_stationary_locations[..., None]
        & (
            coincident_stationary_pairs
            | (
                balanced_stationary_brackets[..., None]
                & nearest_bracketing_pairs
            )
        ),
        axis=-2,
    )
    valid_analytic_candidates &= ~analytic_tangency_representations
    stationary_locations_near_analytic = np.any(
        stationary_analytic_pairs
        & (
            stationary_analytic_distances
            <= location_tolerances[:, None, None, None]
        ),
        axis=-1,
    )
    valid_stationary_candidates = candidate_stationary_locations & (
        ~stationary_locations_near_analytic
        | canonical_stationary_locations
    )
    safe_analytic_candidates = np.where(
        valid_analytic_candidates,
        analytic_root_candidates,
        0.0,
    )
    safe_stationary_candidates = np.where(
        valid_stationary_candidates,
        stationary_candidates,
        0.0,
    )
    node_candidates = np.broadcast_to(
        _SKIN_CUBIC_INTERPOLATION_NODES,
        sample_values.shape,
    )
    valid_node_candidates = (
        active_intervals[..., None]
        & (np.abs(sample_values) <= root_tolerance)
        & (
            (node_candidates < 1.0)
            | final_intervals[..., None]
        )
    )
    canonical_boundary_nodes = (
        boundary_start_locations[..., None]
        & (node_candidates == 0.0)
    ) | (
        final_intervals[..., None]
        & boundary_end_locations[..., None]
        & (node_candidates == 1.0)
    )
    node_analytic_distances = (
        np.abs(
            node_candidates[..., :, None]
            - safe_analytic_candidates[..., None, :]
        )
        * interval_widths[..., None, None]
    )
    node_stationary_distances = (
        np.abs(
            node_candidates[..., :, None]
            - safe_stationary_candidates[..., None, :]
        )
        * interval_widths[..., None, None]
    )
    node_locations_shadowed = (
        np.any(
            valid_analytic_candidates[..., None, :]
            & (
                node_analytic_distances
                <= location_tolerances[:, None, None, None]
            ),
            axis=-1,
        )
        | np.any(
            valid_stationary_candidates[..., None, :]
            & (
                node_stationary_distances
                <= location_tolerances[:, None, None, None]
            ),
            axis=-1,
        )
    )
    valid_node_candidates &= (
        canonical_boundary_nodes | ~node_locations_shadowed
    )
    ordinary_candidates = np.concatenate(
        (
            analytic_root_candidates,
            stationary_candidates,
            node_candidates,
        ),
        axis=-1,
    )
    valid_ordinary_candidates = np.concatenate(
        (
            valid_analytic_candidates,
            valid_stationary_candidates,
            valid_node_candidates,
        ),
        axis=-1,
    ) & ~riding_intervals[..., None]

    interval_indices = np.broadcast_to(
        np.arange(active_intervals.shape[1], dtype=np.int64),
        active_intervals.shape,
    )
    latest_active_indices = np.maximum.accumulate(
        np.where(active_intervals, interval_indices, -1),
        axis=1,
    )
    previous_active_indices = np.pad(
        latest_active_indices[:, :-1],
        ((0, 0), (1, 0)),
        constant_values=-1,
    )
    previous_riding_intervals = (
        previous_active_indices >= 0
    ) & np.take_along_axis(
        riding_intervals,
        np.maximum(previous_active_indices, 0),
        axis=1,
    )
    riding_interval_starts = (
        riding_intervals & ~previous_riding_intervals
    )
    local_candidates = np.concatenate(
        (
            ordinary_candidates,
            np.full(
                riding_intervals.shape + (1,),
                0.5,
                dtype=np.float64,
            ),
        ),
        axis=-1,
    )
    valid_candidates = np.concatenate(
        (
            valid_ordinary_candidates,
            riding_interval_starts[..., None],
        ),
        axis=-1,
    )
    analytic_candidate_flags = np.broadcast_to(
        np.arange(local_candidates.shape[-1])
        < analytic_root_candidates.shape[-1],
        local_candidates.shape,
    )
    valid_candidates &= (
        local_candidates >= 0.0
    ) & (
        (local_candidates < 1.0)
        | (
            final_intervals[..., None]
            & (local_candidates <= 1.0)
        )
    )
    safe_local_candidates = np.where(
        valid_candidates,
        local_candidates,
        0.0,
    )
    global_candidates = (
        interval_starts[..., None]
        + interval_widths[..., None] * safe_local_candidates
    )
    valid_candidates &= ~(
        previous_riding_intervals[..., None]
        & (
            np.abs(
                global_candidates - interval_starts[..., None]
            )
            <= parameter_epsilon
        )
    )
    ordered_analytic_candidates = np.sort(
        np.where(
            valid_candidates & analytic_candidate_flags,
            global_candidates,
            np.inf,
        ).reshape(
            (segment_count, -1)
        ),
        axis=1,
    )
    ordered_auxiliary_candidates = np.sort(
        np.where(
            valid_candidates & ~analytic_candidate_flags,
            global_candidates,
            np.inf,
        ).reshape(
            (segment_count, -1)
        ),
        axis=1,
    )
    analytic_parameters = _deduplicated_skin_root_parameters(
        ordered_analytic_candidates,
        boundary_location_tolerances,
    )
    auxiliary_parameters = _deduplicated_skin_root_parameters(
        ordered_auxiliary_candidates,
        location_tolerances,
    )
    root_parameters = _deduplicated_skin_root_parameters(
        np.sort(
            np.concatenate(
                (analytic_parameters, auxiliary_parameters),
                axis=1,
            ),
            axis=1,
        ),
        boundary_location_tolerances,
    )
    root_counts = np.count_nonzero(np.isfinite(root_parameters), axis=1)
    return root_parameters, root_counts, segment_lengths


def _skin_segment_crossings(
    flesh_vertices: np.ndarray,
    skin_vertices: np.ndarray,
    field: np.ndarray,
    lower: np.ndarray,
    pitch: np.ndarray,
    minimum_gradient: float,
    root_tolerance: float,
) -> np.ndarray:
    return _skin_segment_root_parameters(
        flesh_vertices,
        skin_vertices,
        field,
        lower,
        pitch,
        minimum_gradient,
        root_tolerance,
    )[1]


def _skin_segment_crossing_distances(
    flesh_vertices: np.ndarray,
    skin_vertices: np.ndarray,
    field: np.ndarray,
    lower: np.ndarray,
    pitch: np.ndarray,
    minimum_gradient: float,
    root_tolerance: float,
) -> tuple[tuple[float, ...], ...]:
    root_parameters, root_counts, segment_lengths = (
        _skin_segment_root_parameters(
            flesh_vertices,
            skin_vertices,
            field,
            lower,
            pitch,
            minimum_gradient,
            root_tolerance,
        )
    )
    world_distances = np.multiply(
        root_parameters,
        segment_lengths[:, None],
        out=np.full_like(root_parameters, np.inf),
        where=np.isfinite(root_parameters),
    )
    return tuple(
        map(
            lambda row_and_count: tuple(
                map(
                    float,
                    row_and_count[0][: row_and_count[1]],
                )
            ),
            zip(world_distances, root_counts, strict=True),
        )
    )


def _skin_field_boundary_minimum(field: np.ndarray) -> float:
    return float(
        min(
            map(
                np.min,
                (
                    field[0, :, :],
                    field[-1, :, :],
                    field[:, 0, :],
                    field[:, -1, :],
                    field[:, :, 0],
                    field[:, :, -1],
                ),
            )
        )
    )


@dataclass(frozen=True)
class _SkinDescentState:
    vertices: np.ndarray
    domain_escape: np.ndarray
    degenerate: np.ndarray
    minimum_gradient: float


def _skin_local_scaffold(
    morphology: EvaluatedMorphology,
    field: np.ndarray,
    problem: SkinLayerProblem,
    policy: SkinRelaxationPolicy,
) -> (
    tuple[np.ndarray, float, np.ndarray]
    | SkinRelaxationObstruction
):
    vertices = np.asarray(morphology.vertices, dtype=np.float64)
    lower = np.asarray(morphology.lower, dtype=np.float64)
    upper = np.asarray(morphology.upper, dtype=np.float64)
    pitch = np.asarray(morphology.world_pitch, dtype=np.float64)
    maximum_pitch = morphology.maximum_pitch
    source_samples = _trilinear_samples(field, lower, pitch, vertices)
    source_gradients = _symmetric_trilinear_gradients(
        field, lower, pitch, vertices
    )
    source_gradient_magnitudes = np.linalg.norm(source_gradients, axis=1)
    boundary_minimum = _skin_field_boundary_minimum(field)
    source_domain_escape = tuple(
        map(int, np.flatnonzero(~source_samples.inside))
    )
    if source_domain_escape:
        return _skin_projection_obstruction(
            problem,
            SkinDomainEscape(
                (),
                source_domain_escape,
                tuple(map(float, lower)),
                tuple(map(float, upper)),
                boundary_minimum,
                0.0,
            ),
        )
    source_gradient_degeneracy = np.flatnonzero(
        source_gradient_magnitudes < policy.minimum_gradient
    )
    if source_gradient_degeneracy.size:
        return _skin_projection_obstruction(
            problem,
            SkinGradientDegeneracy(
                tuple(map(int, source_gradient_degeneracy)),
                float(np.min(source_gradient_magnitudes)),
                policy.minimum_gradient,
            ),
        )
    if policy.projection_descent_budget < 1:
        return _skin_projection_obstruction(
            problem,
            SkinRootAbsent(
                tuple(map(int, range(len(vertices)))),
                policy.projection_sample_count,
                float(np.min(source_samples.values)),
                float(np.max(source_samples.values)),
            ),
        )
    step_cap = policy.descent_step_pitch * maximum_pitch

    def descend(
        state: _SkinDescentState,
        _iteration: int,
    ) -> _SkinDescentState:
        samples = _trilinear_samples(field, lower, pitch, state.vertices)
        gradients = _symmetric_trilinear_gradients(
            field, lower, pitch, state.vertices
        )
        magnitudes = np.linalg.norm(gradients, axis=1)
        squared = magnitudes * magnitudes
        corrections = _bounded_vectors(
            np.divide(
                samples.values[:, None] * gradients,
                squared[:, None],
                out=np.zeros_like(gradients),
                where=squared[:, None] > 0.0,
            ),
            step_cap,
        )
        return _SkinDescentState(
            state.vertices - corrections,
            state.domain_escape | ~samples.inside,
            state.degenerate | (magnitudes < policy.minimum_gradient),
            min(state.minimum_gradient, float(np.min(magnitudes))),
        )

    descended = reduce(
        descend,
        range(policy.projection_descent_budget),
        _SkinDescentState(
            vertices.copy(),
            np.zeros(len(vertices), dtype=bool),
            np.zeros(len(vertices), dtype=bool),
            float(np.min(source_gradient_magnitudes)),
        ),
    )
    escape_indices = np.flatnonzero(descended.domain_escape)
    if escape_indices.size:
        return _skin_projection_obstruction(
            problem,
            SkinDomainEscape(
                (),
                tuple(map(int, escape_indices)),
                tuple(map(float, lower)),
                tuple(map(float, upper)),
                boundary_minimum,
                0.0,
            ),
        )
    if descended.degenerate.any():
        return _skin_projection_obstruction(
            problem,
            SkinGradientDegeneracy(
                tuple(map(int, np.flatnonzero(descended.degenerate))),
                descended.minimum_gradient,
                policy.minimum_gradient,
            ),
        )
    scaffold = descended.vertices
    root_gradients = _symmetric_trilinear_gradients(
        field, lower, pitch, scaffold
    )
    gradient_magnitudes = np.linalg.norm(root_gradients, axis=1)
    degenerate = np.flatnonzero(
        gradient_magnitudes < policy.minimum_gradient
    )
    if degenerate.size:
        return _skin_projection_obstruction(
            problem,
            SkinGradientDegeneracy(
                tuple(map(int, degenerate)),
                float(np.min(gradient_magnitudes)),
                policy.minimum_gradient,
            ),
        )
    return (
        scaffold,
        min(
            descended.minimum_gradient,
            float(np.min(gradient_magnitudes)),
        ),
        root_gradients / gradient_magnitudes[:, None],
    )

def _skin_adjacency(
    faces: np.ndarray,
    scaffold: np.ndarray,
    normals: np.ndarray,
    maximum_pitch: float,
    policy: SkinRelaxationPolicy,
) -> object:
    vertex_count = len(scaffold)
    topology_edges = _mesh_edges(faces)
    directed_topology_edges = np.concatenate(
        (topology_edges, topology_edges[:, ::-1]),
        axis=0,
    )
    topology_adjacency = coo_matrix(
        (
            np.ones(len(directed_topology_edges), dtype=np.float64),
            (
                directed_topology_edges[:, 0],
                directed_topology_edges[:, 1],
            ),
        ),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    _, component_labels = connected_components(
        topology_adjacency,
        directed=False,
        return_labels=True,
    )
    radius = policy.neighborhood_radius_pitch * maximum_pitch
    candidate_pairs = cKDTree(scaffold).query_pairs(
        radius,
        output_type="ndarray",
    )
    pair_displacements = (
        scaffold[candidate_pairs[:, 1]]
        - scaffold[candidate_pairs[:, 0]]
    )
    pair_distances = np.linalg.norm(pair_displacements, axis=1)
    normalized_distances = np.divide(
        pair_distances,
        radius,
        out=np.ones_like(pair_distances),
        where=radius > 0.0,
    )
    compact_weights = (
        (1.0 - normalized_distances) ** 4
        * (1.0 + 4.0 * normalized_distances)
    )
    normal_alignment = np.maximum(
        np.einsum(
            "ij,ij->i",
            normals[candidate_pairs[:, 0]],
            normals[candidate_pairs[:, 1]],
        ),
        0.0,
    )
    pair_weights = compact_weights * normal_alignment * normal_alignment
    retained_pairs = (
        (
            component_labels[candidate_pairs[:, 0]]
            == component_labels[candidate_pairs[:, 1]]
        )
        & (pair_weights > 0.0)
    )
    edges = candidate_pairs[retained_pairs]
    weights = pair_weights[retained_pairs]
    directed_edges = np.concatenate((edges, edges[:, ::-1]), axis=0)
    directed_weights = np.concatenate((weights, weights), axis=0)
    adjacency = coo_matrix(
        (
            directed_weights,
            (directed_edges[:, 0], directed_edges[:, 1]),
        ),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    isolated = degree <= 0.0
    return (
        diags(
            np.divide(
                1.0,
                degree,
                out=np.zeros_like(degree),
                where=degree > 0.0,
            )
        )
        @ adjacency
        + diags(isolated.astype(np.float64))
    )


def _bounded_vectors(
    vectors: np.ndarray,
    maximum_norm: float,
) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    factors = np.minimum(
        1.0,
        np.divide(
            maximum_norm,
            norms,
            out=np.ones_like(norms),
            where=norms > 0.0,
        ),
    )
    return vectors * factors[:, None]


def _skin_iteration(
    state: _SkinIterationState,
    scaffold: np.ndarray,
    adjacency: object,
    field: np.ndarray,
    lower: np.ndarray,
    pitch: np.ndarray,
    maximum_pitch: float,
    policy: SkinRelaxationPolicy,
) -> _SkinIterationState:
    samples = _trilinear_samples(
        field,
        lower,
        pitch,
        state.vertices,
    )
    gradients = _symmetric_trilinear_gradients(
        field,
        lower,
        pitch,
        state.vertices,
    )
    gradient_magnitudes = np.linalg.norm(gradients, axis=1)
    normals = np.divide(
        gradients,
        gradient_magnitudes[:, None],
        out=np.zeros_like(gradients),
        where=gradient_magnitudes[:, None] > 0.0,
    )
    neighbor_mean = np.asarray(adjacency @ state.vertices)
    force = (
        (1.0 - policy.neighbor_weight)
        * (scaffold - state.vertices)
        + policy.neighbor_weight
        * (neighbor_mean - state.vertices)
    )
    tangential = force - (
        np.einsum("ij,ij->i", force, normals)[:, None] * normals
    )
    displacement = _bounded_vectors(
        policy.tangential_step * tangential,
        policy.maximum_tangential_step_pitch * maximum_pitch,
    )
    moved = state.vertices + displacement
    domain_escape_indices = tuple(
        map(
            int,
            np.union1d(
                np.asarray(state.domain_escape_indices, dtype=np.int64),
                np.flatnonzero(~samples.inside),
            ),
        )
    )
    degenerate_indices = tuple(
        map(
            int,
            np.union1d(
                np.asarray(state.degenerate_indices, dtype=np.int64),
                np.flatnonzero(
                    gradient_magnitudes < policy.minimum_gradient
                ),
            ),
        )
    )
    projected_state = _SkinIterationState(
        moved,
        min(
            state.minimum_gradient,
            float(np.min(gradient_magnitudes)),
        ),
        state.maximum_projection_residual,
        domain_escape_indices,
        degenerate_indices,
    )

    def project(
        projection_state: _SkinIterationState,
        _iteration: int,
    ) -> _SkinIterationState:
        projection_samples = _trilinear_samples(
            field,
            lower,
            pitch,
            projection_state.vertices,
        )
        projection_gradients = _symmetric_trilinear_gradients(
            field,
            lower,
            pitch,
            projection_state.vertices,
        )
        magnitudes = np.linalg.norm(projection_gradients, axis=1)
        squared = magnitudes * magnitudes
        corrections = _bounded_vectors(
            np.divide(
                projection_samples.values[:, None]
                * projection_gradients,
                squared[:, None],
                out=np.zeros_like(projection_gradients),
                where=squared[:, None] > 0.0,
            ),
            policy.maximum_projection_step_pitch * maximum_pitch,
        )
        return _SkinIterationState(
            projection_state.vertices - corrections,
            min(
                projection_state.minimum_gradient,
                float(np.min(magnitudes)),
            ),
            max(
                projection_state.maximum_projection_residual,
                float(np.max(np.abs(projection_samples.values))),
            ),
            tuple(
                map(
                    int,
                    np.union1d(
                        np.asarray(
                            projection_state.domain_escape_indices,
                            dtype=np.int64,
                        ),
                        np.flatnonzero(~projection_samples.inside),
                    ),
                )
            ),
            tuple(
                map(
                    int,
                    np.union1d(
                        np.asarray(
                            projection_state.degenerate_indices,
                            dtype=np.int64,
                        ),
                        np.flatnonzero(
                            magnitudes < policy.minimum_gradient
                        ),
                    ),
                )
            ),
        )

    return reduce(
        project,
        range(policy.projection_iteration_budget),
        projected_state,
    )


def _solve_skin_layer_with_policy(
    morphology: EvaluatedMorphology,
    problem: SkinLayerProblem,
    policy: SkinRelaxationPolicy,
    formation_sections: tuple[
        tuple[CompositionSectionId, ...],
        ...,
    ],
) -> AcceptedSkinLayer | SkinRelaxationObstruction:
    flesh_field = np.asarray(morphology.field, dtype=np.float64)
    skin_field = flesh_field - problem.thickness
    memo_key = (
        None
        if _field_cache_bypassed()
        else _skin_memo_key(
            morphology, problem, policy, formation_sections
        )
    )
    cached = (
        None
        if memo_key is None
        else _skin_memo_lookup(memo_key, skin_field)
    )
    if cached is not None:
        return cached
    boundary, escaped_indices = _skin_domain_boundary(skin_field)
    if escaped_indices.size:
        return _skin_projection_obstruction(
            problem,
            SkinDomainEscape(
                tuple(
                    tuple(map(int, index))
                    for index in escaped_indices
                ),
                (),
                morphology.lower,
                morphology.upper,
                float(np.min(skin_field[boundary])),
                0.0,
            ),
        )
    scaffold_result = _skin_local_scaffold(
        morphology,
        skin_field,
        problem,
        policy,
    )
    if isinstance(scaffold_result, SkinRelaxationObstruction):
        return scaffold_result
    (
        scaffold,
        scaffold_minimum_gradient,
        scaffold_normals,
    ) = scaffold_result
    faces = np.asarray(morphology.faces, dtype=np.int64)
    lower = np.asarray(morphology.lower, dtype=np.float64)
    pitch = np.asarray(morphology.world_pitch, dtype=np.float64)
    maximum_pitch = morphology.maximum_pitch
    adjacency = _skin_adjacency(
        faces,
        scaffold,
        scaffold_normals,
        maximum_pitch,
        policy,
    )
    initial = _SkinIterationState(
        scaffold,
        scaffold_minimum_gradient,
        0.0,
        (),
        (),
    )

    def relax(
        state: _SkinIterationState,
        _iteration: int,
    ) -> _SkinIterationState:
        return _skin_iteration(
            state,
            scaffold,
            adjacency,
            skin_field,
            lower,
            pitch,
            maximum_pitch,
            policy,
        )

    relaxed = reduce(
        relax,
        range(policy.iteration_budget),
        initial,
    )
    probe = relax(relaxed, policy.iteration_budget)
    combined_domain_escape = tuple(
        map(
            int,
            np.union1d(
                np.asarray(relaxed.domain_escape_indices, dtype=np.int64),
                np.asarray(probe.domain_escape_indices, dtype=np.int64),
            ),
        )
    )
    if combined_domain_escape:
        return _skin_projection_obstruction(
            problem,
            SkinDomainEscape(
                (),
                combined_domain_escape,
                morphology.lower,
                morphology.upper,
                float(np.min(skin_field[boundary])),
                0.0,
            ),
        )
    combined_degenerate = tuple(
        map(
            int,
            np.union1d(
                np.asarray(relaxed.degenerate_indices, dtype=np.int64),
                np.asarray(probe.degenerate_indices, dtype=np.int64),
            ),
        )
    )
    if combined_degenerate:
        return _skin_projection_obstruction(
            problem,
            SkinGradientDegeneracy(
                combined_degenerate,
                min(relaxed.minimum_gradient, probe.minimum_gradient),
                policy.minimum_gradient,
            ),
        )
    final_samples = _trilinear_samples(
        skin_field,
        lower,
        pitch,
        relaxed.vertices,
    )
    flesh_samples = _trilinear_samples(
        flesh_field,
        lower,
        pitch,
        relaxed.vertices,
    )
    containment_margin = flesh_samples.values
    containment_failures = np.flatnonzero(
        containment_margin < policy.minimum_containment_margin
    )
    if containment_failures.size:
        return SkinRelaxationObstruction(
            problem.region_ids,
            SkinContainmentUnsatisfied(
                tuple(map(int, containment_failures)),
                float(np.min(containment_margin)),
                policy.minimum_containment_margin,
            ),
        )
    projection_residuals = np.abs(final_samples.values)
    offset_tolerance = policy.offset_tolerance_pitch * maximum_pitch
    offset_failures = np.flatnonzero(
        projection_residuals > offset_tolerance
    )
    if offset_failures.size:
        return SkinRelaxationObstruction(
            problem.region_ids,
            SkinOffsetUnsatisfied(
                tuple(map(int, offset_failures)),
                float(np.max(projection_residuals)),
                offset_tolerance,
            ),
        )
    edges = _mesh_edges(faces)
    flesh_vertices = np.asarray(morphology.vertices, dtype=np.float64)
    projection_displacements = np.linalg.norm(
        relaxed.vertices - flesh_vertices, axis=1
    )
    locality_radius = max(
        policy.locality_radius_thickness * problem.thickness,
        policy.locality_radius_pitch * maximum_pitch,
    )
    maximum_projection_displacement = float(
        np.max(projection_displacements)
    )
    nonlocal_failures = np.flatnonzero(
        projection_displacements > locality_radius
    )
    if nonlocal_failures.size:
        return _skin_projection_obstruction(
            problem,
            SkinProjectionNonlocal(
                tuple(map(int, nonlocal_failures)),
                maximum_projection_displacement,
                locality_radius,
            ),
        )
    correspondence_tolerance = max(
        1.0e-12, policy.minimum_gradient * maximum_pitch
    )
    crossing_counts = _skin_segment_crossings(
        flesh_vertices,
        scaffold,
        skin_field,
        lower,
        pitch,
        policy.minimum_gradient,
        correspondence_tolerance,
    )
    multiple = np.flatnonzero(crossing_counts > 1)
    if multiple.size:
        return _skin_projection_obstruction(
            problem,
            SkinRootMultiplicity(
                tuple(map(int, multiple)),
                policy.projection_sample_count,
                int(np.min(crossing_counts[multiple])),
                int(np.max(crossing_counts[multiple])),
                root_distances_world=_skin_segment_crossing_distances(
                    flesh_vertices[multiple],
                    scaffold[multiple],
                    skin_field,
                    lower,
                    pitch,
                    policy.minimum_gradient,
                    correspondence_tolerance,
                ),
            ),
        )
    absent = np.flatnonzero(crossing_counts == 0)
    if absent.size:
        absent_parameters = np.linspace(
            0.0, 1.0, policy.projection_sample_count
        )
        absent_points = (
            flesh_vertices[absent][:, None, :]
            + absent_parameters[None, :, None]
            * (scaffold[absent] - flesh_vertices[absent])[:, None, :]
        )
        absent_values = _trilinear_samples(
            skin_field,
            lower,
            pitch,
            absent_points.reshape((-1, 3)),
        ).values
        return _skin_projection_obstruction(
            problem,
            SkinRootAbsent(
                tuple(map(int, absent)),
                policy.projection_sample_count,
                float(np.min(absent_values)),
                float(np.max(absent_values)),
            ),
        )
    root_counts = crossing_counts
    reference_lengths = np.linalg.norm(
        flesh_vertices[edges[:, 1]] - flesh_vertices[edges[:, 0]],
        axis=1,
    )
    final_lengths = np.linalg.norm(
        relaxed.vertices[edges[:, 1]] - relaxed.vertices[edges[:, 0]],
        axis=1,
    )
    stretch_reference = np.maximum(
        reference_lengths,
        policy.minimum_stretch_reference_pitch * maximum_pitch,
    )
    stretch = 1.0 + np.divide(
        final_lengths - reference_lengths,
        stretch_reference,
        out=np.zeros_like(final_lengths),
        where=stretch_reference > 0.0,
    )
    stretch_failures = np.flatnonzero(
        (stretch < policy.stretch_lower_ratio)
        | (stretch > policy.stretch_upper_ratio)
    )
    if stretch_failures.size:
        return SkinRelaxationObstruction(
            problem.region_ids,
            SkinStretchUnsatisfied(
                tuple(
                    tuple(map(int, edge))
                    for edge in edges[stretch_failures]
                ),
                float(np.min(stretch)),
                float(np.max(stretch)),
                policy.stretch_lower_ratio,
                policy.stretch_upper_ratio,
            ),
        )
    reference_cross = np.cross(
        flesh_vertices[faces[:, 1]] - flesh_vertices[faces[:, 0]],
        flesh_vertices[faces[:, 2]] - flesh_vertices[faces[:, 0]],
    )
    final_cross = np.cross(
        relaxed.vertices[faces[:, 1]] - relaxed.vertices[faces[:, 0]],
        relaxed.vertices[faces[:, 2]] - relaxed.vertices[faces[:, 0]],
    )
    reference_area = np.linalg.norm(reference_cross, axis=1)
    final_area = np.linalg.norm(final_cross, axis=1)
    orientation = np.divide(
        np.einsum("ij,ij->i", reference_cross, final_cross),
        reference_area * final_area,
        out=np.full_like(reference_area, -1.0),
        where=(reference_area * final_area) > 0.0,
    )
    area_reference = np.maximum(
        reference_area,
        policy.minimum_face_area_reference_pitch_squared
        * maximum_pitch
        * maximum_pitch,
    )
    area_ratio = 1.0 + np.divide(
        final_area - reference_area,
        area_reference,
        out=np.zeros_like(final_area),
        where=area_reference > 0.0,
    )
    foldover_failures = np.flatnonzero(
        (orientation <= policy.minimum_face_orientation)
        | (area_ratio < policy.minimum_face_area_ratio)
    )
    if foldover_failures.size:
        return SkinRelaxationObstruction(
            problem.region_ids,
            SkinFoldover(
                tuple(map(int, foldover_failures)),
                float(np.min(orientation)),
                float(np.min(area_ratio)),
                policy.minimum_face_orientation,
                policy.minimum_face_area_ratio,
            ),
        )
    fixed_point_residual = float(
        np.max(
            np.linalg.norm(
                probe.vertices - relaxed.vertices,
                axis=1,
            )
        )
    )
    projection_residual = float(np.max(projection_residuals))
    convergence_tolerance = (
        policy.convergence_tolerance_pitch * maximum_pitch
    )
    projection_tolerance = (
        policy.projection_tolerance_pitch * maximum_pitch
    )
    if (
        fixed_point_residual > convergence_tolerance
        or projection_residual > projection_tolerance
    ):
        return SkinRelaxationObstruction(
            problem.region_ids,
            SkinConvergenceUnsatisfied(
                policy.iteration_budget,
                fixed_point_residual,
                convergence_tolerance,
                projection_residual,
                projection_tolerance,
            ),
        )
    correspondence = _skin_correspondence(
        morphology,
        formation_sections,
    )
    owner_multiplicities = tuple(
        map(len, correspondence.formation_sections)
    )
    final_gradients = _symmetric_trilinear_gradients(
        skin_field,
        lower,
        pitch,
        relaxed.vertices,
    )
    gradient_magnitudes = np.linalg.norm(final_gradients, axis=1)
    normals = np.divide(
        final_gradients,
        gradient_magnitudes[:, None],
        out=np.zeros_like(final_gradients),
        where=gradient_magnitudes[:, None] > 0.0,
    )
    scaffold_displacement = relaxed.vertices - scaffold
    tangential_displacement = scaffold_displacement - (
        np.einsum(
            "ij,ij->i",
            scaffold_displacement,
            normals,
        )[:, None]
        * normals
    )
    maximum_tangential_displacement = float(
        np.max(np.linalg.norm(tangential_displacement, axis=1))
    )
    evidence = SkinRelaxationEvidence(
        problem.formation,
        problem.region_ids,
        problem.thickness,
        len(morphology.vertices),
        len(morphology.faces),
        int(np.count_nonzero(np.asarray(owner_multiplicities) > 0)),
        int(np.count_nonzero(np.asarray(owner_multiplicities) == 0)),
        max(owner_multiplicities, default=0),
        policy.projection_descent_budget,
        policy.projection_sample_count,
        int(np.min(root_counts)),
        int(np.max(root_counts)),
        min(
            relaxed.minimum_gradient,
            float(np.min(gradient_magnitudes)),
        ),
        float(np.min(containment_margin)),
        projection_residual,
        float(np.min(stretch)),
        float(np.max(stretch)),
        float(np.min(orientation)),
        float(np.min(area_ratio)),
        policy.iteration_budget,
        policy.projection_iteration_budget,
        relaxed.maximum_projection_residual,
        maximum_tangential_displacement,
        fixed_point_residual,
        maximum_projection_displacement,
    )
    accepted = AcceptedSkinLayer(
        read_only_array(np.array(relaxed.vertices, copy=True)),
        read_only_array(np.array(normals, copy=True)),
        read_only_array(np.array(skin_field, copy=True)),
        correspondence,
        evidence,
    )
    if memo_key is not None:
        _skin_memo_store(memo_key, accepted)
    return accepted


def sample_graph_field_checked(
    graph: GeometryGraph,
    points: np.ndarray,
) -> np.ndarray | RejectedSurfaceFormation:
    if _uses_muscle_formation(graph):
        formation = _assess_muscle_formations(graph)
        if isinstance(formation, RejectedSurfaceFormation):
            return formation
    sampled = (
        (
            sampled.field
            if isinstance(
                sampled := _sample_graph_field_certified_checked(
                    graph,
                    points,
                ),
                _ComposedField,
            )
            else sampled
        )
        if _uses_local_composition(graph)
        else _sample_graph_field_legacy_checked(graph, points)
    )
    return (
        sampled
        if isinstance(sampled, RejectedSurfaceFormation)
        or graph.skin is None
        else sampled - graph.skin.thickness
    )


def sample_graph_field(
    graph: dict,
    points: np.ndarray,
) -> np.ndarray | RejectedSurfaceFormation:
    decoded = require_accepted(decode_graph(graph))
    sampled_points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    return sample_graph_field_checked(decoded, sampled_points)


def _resolution(value: object) -> int:
    if isinstance(value, Integral) and int(value) >= 2:
        return int(value)
    obstruction = GeometryObstruction(
        "<graph>",
        "graph",
        GeometryRule.MALFORMED_GRAPH,
        f"resolution must be an integer >= 2, got {value!r}",
        True,
    )
    raise GeometryDecodeFailure((obstruction,))


def _regional_padding(
    region: tuple[slice, slice, slice],
    resolution: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            int(axis_slice.start or 0),
            resolution - int(axis_slice.stop or resolution),
        )
        for axis_slice in region
    )


def _compose_sampled_region(
    field: np.ndarray | None,
    sampled: np.ndarray,
    region: tuple[slice, slice, slice] | None,
    operator: CompositionOperator,
    blend_radius: float,
    resolution: int,
    diagonal: float,
) -> np.ndarray:
    if region is None:
        return (
            sampled
            if field is None
            else compose_union(
                LegacyCompositionProblem(
                    field,
                    sampled,
                    operator,
                    blend_radius,
                )
            ).field
        )
    padding = _regional_padding(region, resolution)
    if field is None:
        return np.pad(
            sampled,
            padding,
            mode="constant",
            constant_values=2.0 * diagonal,
        )
    padded = np.pad(sampled, padding, mode="constant", constant_values=0.0)
    mask = np.pad(
        np.ones(sampled.shape, dtype=bool),
        padding,
        mode="constant",
        constant_values=False,
    )
    return np.where(
        mask,
        compose_union(
            LegacyCompositionProblem(
                field,
                padded,
                operator,
                blend_radius,
            )
        ).field,
        field,
    )


def _lower_sampled_field(
    field: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    resolution: int,
    composition_evidence: tuple[LocalCompositionEvidence, ...] = (),
    formation_evidence: tuple[MuscleFormationEvidence, ...] = (),
) -> EvaluatedMorphology:
    spacing = tuple(
        (upper[index] - lower[index]) / (resolution - 1)
        for index in range(3)
    )
    from skimage import measure

    vertices, faces, normals, _ = measure.marching_cubes(
        field,
        level=0.0,
        spacing=spacing,
    )
    vertices = vertices + lower
    return EvaluatedMorphology(
        vertices=read_only_array(vertices),
        faces=read_only_array(faces),
        normals=read_only_array(normals),
        field=read_only_array(field),
        lower=tuple(map(float, lower)),
        upper=tuple(map(float, upper)),
        resolution=resolution,
        world_pitch=tuple(map(float, spacing)),
        composition_evidence=composition_evidence,
        formation_evidence=formation_evidence,
    )


def _evaluate_legacy_checked(
    graph: GeometryGraph,
    resolution: int,
    formation_evidence: tuple[MuscleFormationEvidence, ...] = (),
) -> EvaluatedMorphology | RejectedSurfaceFormation:
    lower, upper = graph_bounds_checked(graph)
    diagonal = float(np.linalg.norm(upper - lower))
    global_blend_radius = graph.blend * diagonal
    axes = tuple(
        np.linspace(lower[index], upper[index], resolution)
        for index in range(3)
    )
    localized_morphology = any(
        isinstance(part, GencylPart) and part.profile is not None
        for part in graph.parts
    )
    auxiliary_points = (
        _grid_points(axes)
        if graph.webs or graph.carves
        else np.empty((0, 3), dtype=np.float64)
    )

    def merge_part_field(
        field: np.ndarray | RejectedSurfaceFormation | None,
        instance: tuple[Part, bool],
    ) -> np.ndarray | RejectedSurfaceFormation:
        if isinstance(field, RejectedSurfaceFormation):
            return field
        part, mirrored = instance
        blend_radius = (
            global_blend_radius
            if part.blend is None
            else part.blend * diagonal
        )
        sampled_part = _sampled_part_field(
            part,
            mirrored,
            blend_radius,
            localized_morphology,
            axes,
            executor,
        )
        if isinstance(sampled_part, MuscleFormationObstruction):
            return RejectedSurfaceFormation((sampled_part,))
        region, sampled = sampled_part
        return _compose_sampled_region(
            field,
            sampled,
            region,
            part.operator,
            blend_radius,
            resolution,
            diagonal,
        )

    def merge_web_field(
        field: np.ndarray | RejectedSurfaceFormation | None,
        instance: tuple[SpanningWeb, bool],
    ) -> np.ndarray | RejectedSurfaceFormation:
        if isinstance(field, RejectedSurfaceFormation):
            return field
        web, mirrored = instance
        blend_radius = (
            global_blend_radius
            if web.blend is None
            else web.blend * diagonal
        )
        sampled = _sample_web_sdf(
            auxiliary_points,
            web,
            mirrored,
            executor,
        ).reshape((resolution,) * 3)
        return _compose_sampled_region(
            field,
            sampled,
            None,
            web.operator,
            blend_radius,
            resolution,
            diagonal,
        )

    def subtract_carve_field(
        field: np.ndarray | RejectedSurfaceFormation,
        instance: tuple[Part, bool],
    ) -> np.ndarray | RejectedSurfaceFormation:
        if isinstance(field, RejectedSurfaceFormation):
            return field
        part, mirrored = instance
        sampled = _sample_part_sdf(
            auxiliary_points,
            part,
            mirrored,
            executor,
        )
        if isinstance(sampled, MuscleFormationObstruction):
            return RejectedSurfaceFormation((sampled,))
        sampled = sampled.reshape((resolution,) * 3)
        return sdf_difference(field, sampled)

    with ThreadPoolExecutor(max_workers=4) as executor:
        sampled_union = reduce(
            merge_part_field, part_instances(graph), None
        )
        sampled_positive = reduce(
            merge_web_field,
            web_instances(graph),
            sampled_union,
        )
        sampled_field = (
            None
            if sampled_positive is None
            else reduce(
                subtract_carve_field,
                carve_part_instances(graph),
                sampled_positive,
            )
        )
    if isinstance(sampled_field, RejectedSurfaceFormation):
        return sampled_field
    if sampled_field is None:
        obstruction = GeometryObstruction(
            "<graph>",
            "graph",
            GeometryRule.MALFORMED_GRAPH,
            "parts must be a non-empty sequence",
            True,
        )
        raise GeometryDecodeFailure((obstruction,))
    return _lower_sampled_field(
        sampled_field,
        lower,
        upper,
        resolution,
        formation_evidence=formation_evidence,
    )


def _evaluate_certified_checked(
    graph: GeometryGraph,
    resolution: int,
    formation_evidence: tuple[MuscleFormationEvidence, ...] = (),
) -> EvaluatedMorphology | RejectedSurfaceFormation:
    lower, upper = graph_bounds_checked(graph)
    axes = tuple(
        np.linspace(lower[index], upper[index], resolution)
        for index in range(3)
    )
    points = _grid_points(axes)
    shape = (resolution,) * 3
    with ThreadPoolExecutor(max_workers=4) as executor:
        composed = _sample_graph_field_certified_checked(
            graph,
            points,
            executor,
        )
    if isinstance(composed, RejectedSurfaceFormation):
        return composed
    field = composed.field.reshape(shape)
    return _lower_sampled_field(
        field,
        lower,
        upper,
        resolution,
        composed.evidence,
        formation_evidence,
    )


def evaluate_checked(
    graph: GeometryGraph,
    resolution: int,
) -> EvaluatedMorphology | RejectedSurfaceFormation:
    formation_evidence: tuple[MuscleFormationEvidence, ...] = ()
    if _uses_muscle_formation(graph):
        formation = _assess_muscle_formations(graph)
        if isinstance(formation, RejectedSurfaceFormation):
            return formation
        formation_evidence = formation
    flesh = (
        _evaluate_certified_checked(
            graph,
            resolution,
            formation_evidence,
        )
        if _uses_local_composition(graph)
        else _evaluate_legacy_checked(
            graph,
            resolution,
            formation_evidence,
        )
    )
    if isinstance(flesh, RejectedSurfaceFormation) or graph.skin is None:
        return flesh
    formation_sections = _skin_formation_sections(graph, flesh)
    if isinstance(formation_sections, RejectedSurfaceFormation):
        return formation_sections
    skin = _solve_skin_layer_with_policy(
        flesh,
        graph.skin,
        SEALED_SKIN_POLICY,
        formation_sections,
    )
    return (
        RejectedSurfaceFormation((skin,))
        if isinstance(skin, SkinRelaxationObstruction)
        else replace(
            flesh,
            vertices=skin.vertices,
            normals=skin.normals,
            field=skin.field,
            skin_evidence=skin.evidence,
            skin_correspondence=skin.correspondence,
        )
    )


def evaluate(
    graph,
    res=170,
) -> EvaluatedMorphology | RejectedSurfaceFormation:
    decoded = require_accepted(decode_graph(graph))
    return evaluate_checked(decoded, _resolution(res))


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    import trimesh

    normals = np.asarray(
        trimesh.Trimesh(vertices, faces, process=False).vertex_normals,
        dtype=np.float64,
    )
    normals.setflags(write=False)
    return normals


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64)
    return np.unique(
        np.sort(
            np.concatenate(
                (
                    triangles[:, (0, 1)],
                    triangles[:, (1, 2)],
                    triangles[:, (2, 0)],
                )
            ),
            axis=1,
        ),
        axis=0,
    )


def _outward_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    normals = vertex_normals(points, triangles)
    signed_six_volume = float(
        np.sum(
            np.einsum(
                "ij,ij->i",
                points[triangles[:, 0]],
                np.cross(
                    points[triangles[:, 1]],
                    points[triangles[:, 2]],
                ),
            )
        )
    )
    return normals if signed_six_volume >= 0.0 else -normals


def _diffused_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    scale: float,
) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    edges = _mesh_edges(faces)
    edge_lengths = np.linalg.norm(
        points[edges[:, 1]] - points[edges[:, 0]],
        axis=1,
    )
    characteristic_edge_length = float(np.median(edge_lengths))
    directed_edges = np.concatenate((edges, edges[:, ::-1]), axis=0)
    adjacency = coo_matrix(
        (
            np.ones(directed_edges.shape[0], dtype=np.float64),
            (directed_edges[:, 0], directed_edges[:, 1]),
        ),
        shape=(points.shape[0], points.shape[0]),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    laplacian = eye(points.shape[0], format="csr") - diags(1.0 / degree) @ adjacency
    diffusion_time = 0.5 * (scale / characteristic_edge_length) ** 2
    return np.asarray(
        expm_multiply(
            -diffusion_time * laplacian,
            points,
            traceA=-diffusion_time * float(points.shape[0]),
        ),
        dtype=np.float64,
    )


def apply_curvature_surface_detail(
    vertices: np.ndarray,
    faces: np.ndarray,
    detail: CurvatureSurfaceDetail,
) -> CurvatureDisplacement:
    points = np.asarray(vertices, dtype=np.float64)
    if detail.amplitude == 0.0:
        return CurvatureDisplacement(
            points,
            np.zeros(points.shape[0], dtype=np.float64),
        )
    normals = _outward_normals(points, faces)
    diffused = _diffused_vertices(points, faces, detail.scale)
    signed_relief = np.einsum("ij,ij->i", points - diffused, normals)
    signed_displacements = detail.amplitude * np.tanh(
        4.0 * signed_relief / detail.scale
    )
    return CurvatureDisplacement(
        points + normals * signed_displacements[:, None],
        signed_displacements,
    )


def coherence_report(verts, faces):
    vertices = np.asarray(verts, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    vertices, vertex_inverse = np.unique(
        vertices, axis=0, return_inverse=True
    )
    triangles = vertex_inverse[triangles]
    edge_left = np.concatenate(
        (triangles[:, 0], triangles[:, 1], triangles[:, 2])
    )
    edge_right = np.concatenate(
        (triangles[:, 1], triangles[:, 2], triangles[:, 0])
    )
    edge_lower = np.minimum(edge_left, edge_right)
    edge_upper = np.maximum(edge_left, edge_right)
    edge_keys = edge_lower * len(vertices) + edge_upper
    edge_faces = np.tile(
        np.arange(len(triangles), dtype=np.int64), 3
    )
    edge_order = np.argsort(edge_keys)
    sorted_edge_keys = edge_keys[edge_order]
    _unique_edges, first_edges, edge_counts = np.unique(
        sorted_edge_keys, return_index=True, return_counts=True
    )
    manifold = edge_counts == 2
    paired_edges = first_edges[manifold]
    adjacency = coo_matrix(
        (
            np.ones(len(paired_edges), dtype=np.bool_),
            (
                edge_faces[edge_order[paired_edges]],
                edge_faces[edge_order[paired_edges + 1]],
            ),
        ),
        shape=(len(triangles), len(triangles)),
    ).tocsr()
    component_count, face_labels = connected_components(
        adjacency, directed=False
    )
    face_counts = np.bincount(
        face_labels, minlength=component_count
    )
    substantial = np.flatnonzero(face_counts >= 10)
    edge_face_labels = np.concatenate(
        (face_labels, face_labels, face_labels)
    )
    signed_face_volumes = (
        np.einsum(
            "ij,ij->i",
            vertices[triangles[:, 0]],
            np.cross(
                vertices[triangles[:, 1]],
                vertices[triangles[:, 2]],
            ),
        )
        / 6.0
    )
    component_volumes = np.abs(
        np.bincount(
            face_labels,
            weights=signed_face_volumes,
            minlength=component_count,
        )
    )
    volumes = np.sort(component_volumes[substantial])[::-1]
    main = (
        int(substantial[np.argmax(face_counts[substantial])])
        if len(substantial)
        else None
    )
    main_edge_counts = (
        np.unique(
            edge_keys[edge_face_labels == main], return_counts=True
        )[1]
        if main is not None
        else np.empty(0, dtype=np.int64)
    )
    return {
        "components": len(substantial),
        "grid_dust_slivers": component_count - len(substantial),
        "watertight_main": (
            bool(np.all(main_edge_counts == 2))
            if main is not None
            else False
        ),
        "largest_component_volume_share": (
            volumes[0] / np.sum(volumes) if len(volumes) else 0.0
        ),
        "faces": len(triangles),
    }
