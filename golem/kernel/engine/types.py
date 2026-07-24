"""Immutable geometry values and the checked raw-graph boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import chain
from math import isfinite
from typing import Literal, assert_never, cast

import numpy as np


FRAME_DOC = """\
Axes: +x = creature's LEFT, +y = up, +z = forward (facing +z); world units.

graph:
  schema  {"parts":[...], "webs"?:[...], "carves"?:[...],
           "blend"?: finite number, "skin"?:{...}}
  parts   non-empty positive primitives, smoothly unioned in authored order.
  webs    optional spanning sheets, unioned after parts in authored order.
  carves  optional primitives mirrored by the same law, then subtracted by
          exact hard difference max(host, -carve) after the positive union.
  bounds  derived from positive parts and webs; carves never enlarge the domain.

operator (ordered union edge carried by the incoming part):
  blend       polynomial smooth-min using the selected blend radius (default).
  chamfer     linear bevel union using the selected blend radius.
  crease      exact hard union min(a,b), with no seam overshoot.
  local_blend certified overlap-local gradient blend with bounded support.

gencyl formation:
  schema  "formation"?: "skeleton_integral"
  absent  preserves the legacy gencyl field exactly.
  present admits only a collinear branch-free circular radius family.

skin solved layer:
  schema  {"formation":"static_implicit_relaxation",
           "thickness":positive finite number,
           "region_ids":[unique non-empty strings]}
  absent  preserves the flesh field and mesh exactly.
  present solves Psi(p) = F_flesh(p) - thickness after flesh formation.

box (rounded box / prism):
  schema  {"id","type":"box","center":[x,y,z],"size":[hx,hy,hz],
           "round": r>=0, "rot"?: [w,x,y,z], "mirror"?, "blend"?}
  size    CORE half-extents along the part's LOCAL axes (local == world unless
          rot present). NOT full widths.
  round   additive OUTWARD fillet radius; total half-extent per axis is
          size + round (round applies equally to every face/edge/corner).
  sdf     q = |R^T (p - center)| - size;
          d = ||max(q,0)|| + min(max(q_x,q_y,q_z),0) - round.   (R = I if no rot)

rot (per-part orientation, blob + box ONLY):
  schema  "rot":[w,x,y,z], unit quaternion, canonical w >= 0 (double cover).
          Normalized and sign-canonicalized on read; |q| far from 1 is a
          reported violation but still used (after normalization).
  sdf     isometry conjugation d(p) = d_local(R^T (p - center)); R = the
          rotation matrix of the (canonicalized) quaternion. Exact for box;
          for blob the frozen scaled-space approximation is preserved inside
          the rotated frame.
  gencyl  rot on a gencyl is FORBIDDEN (spines already carry orientation). It
          is a reported violation and is NEVER applied -- the gencyl is
          evaluated exactly as in v0.2.

mirror (across x = 0):
  center.x -> -center.x, and the quaternion (w,x,y,z) -> (w,x,-y,-z).
  (For box/blob this reflection is realized as a proper rotation R' = M R M,
  M = diag(-1,1,1); the improper factor is absorbed by the per-axis reflection
  symmetry of the box/ellipsoid local SDF, so the mirrored instance is the
  exact geometric reflection of the original.)

web (spanning sheet):
  schema  {"id", "anchors":[{"spine":[...],"radii":[...]}, ...],
           "mirror"?, "blend"?, "operator"?}
  anchors ordered curves, at least two, with identical station counts.
  radii   positive local sheet half-extents at each station, interpolated
          across the loft. They are not full thicknesses.
  mirror  reflects the complete ordered anchor tuple as one instance.
"""

FEATURE_MIN_CELLS = 2.0
QUAT_UNIT_TOL = 1e-6

type Vector2 = tuple[float, float]
type Vector3 = tuple[float, float, float]
type Quaternion = tuple[float, float, float, float]


def quaternion_for_instance(
    quaternion: Quaternion | np.ndarray,
    mirrored: bool,
) -> np.ndarray:
    decoded = np.asarray(quaternion, dtype=np.float64)
    if not mirrored:
        return decoded
    w, x, y, z = decoded
    reflected = np.array([w, x, -y, -z])
    return -reflected if reflected[0] < 0.0 else reflected


def read_quat(rot, mirrored: bool = False) -> np.ndarray:
    decoded = np.asarray(rot, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(decoded))
    normalized = (
        np.array([1.0, 0.0, 0.0, 0.0]) if norm < 1e-12 else decoded / norm
    )
    canonical = -normalized if normalized[0] < 0.0 else normalized
    return quaternion_for_instance(tuple(map(float, canonical)), mirrored)


class GeometryRule(StrEnum):
    FENCE_ROT_ON_GENCYL = "FENCE-rot-on-gencyl"
    FENCE_PROFILE_ON_NONGENCYL = "FENCE-profile-on-nongencyl"
    PROFILE_ANCHOR_ARITY = "R-profile-anchor-arity"
    PROFILE_OFFSET = "R-profile-offset"
    PROFILE_ASPECT = "R-profile-aspect"
    PROFILE_DEPTH = "R-profile-depth"
    PROFILE_DEPTH_CONFLICT = "R-profile-depth-conflict"
    PROFILE_EXPONENT = "R-profile-exponent"
    PROFILE_ROLL = "R-profile-roll"
    COMPOSITION_OPERATOR = "R-composition-operator"
    QUAT_NONUNIT = "R-quat-nonunit"
    BOX_NEGATIVE_ROUND = "R-box-negative-round"
    WEB_ANCHOR_ARITY = "R-web-anchor-arity"
    WEB_STATION_ARITY = "R-web-station-arity"
    WEB_HALF_EXTENT = "R-web-half-extent"
    FEATURE_SIZE = "R-8-feature-size"
    SHARP_EDGE = "R-8-sharp-edge"
    UNKNOWN_PART_KIND = "R-unknown-part-kind"
    MALFORMED_PART = "R-malformed-part"
    MALFORMED_GRAPH = "R-malformed-graph"


@dataclass(frozen=True)
class GeometryObstruction:
    part_id: str
    kind: str
    rule: GeometryRule
    detail: str
    fatal: bool


@dataclass(frozen=True)
class Accepted[value]:
    value: value
    obstructions: tuple[GeometryObstruction, ...] = ()


@dataclass(frozen=True)
class Rejected:
    obstructions: tuple[GeometryObstruction, ...]


type DecodeResult[value] = Accepted[value] | Rejected


@dataclass(frozen=True)
class Rotation:
    quaternion: Quaternion
    authored_norm: float


class CompositionOperator(StrEnum):
    BLEND = "blend"
    CHAMFER = "chamfer"
    CREASE = "crease"
    LOCAL_BLEND = "local_blend"


class MuscleFormationKind(StrEnum):
    SKELETON_INTEGRAL = "skeleton_integral"


class SkinFormationKind(StrEnum):
    STATIC_IMPLICIT_RELAXATION = "static_implicit_relaxation"


type LegacyCompositionOperator = Literal[
    CompositionOperator.BLEND,
    CompositionOperator.CHAMFER,
    CompositionOperator.CREASE,
]


@dataclass(frozen=True)
class CompositionSectionId:
    part_id: str
    mirrored: bool


@dataclass(frozen=True)
class CompositionSupportRegion:
    overlapping: CompositionSectionId
    lower: Vector3
    upper: Vector3


@dataclass(frozen=True)
class CompositionSupportWitness:
    lower: Vector3 | None
    upper: Vector3 | None
    active_sample_count: int
    regions: tuple[CompositionSupportRegion, ...] = ()


@dataclass(frozen=True)
class LocalCompositionEvidence:
    incoming: CompositionSectionId
    overlapping: tuple[CompositionSectionId, ...]
    support: CompositionSupportWitness
    accumulated_gradient_bounds: tuple[float, float]
    incoming_gradient_bounds: tuple[float, float]
    maximum_outside_support_delta: float
    maximum_blend_displacement: float
    clean_component_count: int | None
    composed_component_count: int | None


@dataclass(frozen=True)
class FieldScaleUncalibrated:
    section: CompositionSectionId
    minimum_gradient: float
    maximum_gradient: float
    maximum_condition: float


@dataclass(frozen=True)
class NoCompatibleOverlap:
    incoming: CompositionSectionId
    candidates: tuple[CompositionSectionId, ...]
    probe_count: int
    minimum_joint_residual: float
    closest_point: Vector3 | None = None


@dataclass(frozen=True)
class BlendSupportEscaped:
    incoming: CompositionSectionId
    support_lower: Vector3
    support_upper: Vector3
    domain_lower: Vector3
    domain_upper: Vector3


@dataclass(frozen=True)
class GradientDegeneracy:
    section: CompositionSectionId
    active_sample_count: int
    minimum_gradient: float
    required_minimum_gradient: float


@dataclass(frozen=True)
class TopologyChanged:
    incoming: CompositionSectionId
    clean_component_count: int
    composed_component_count: int


@dataclass(frozen=True)
class LocalSectionsIncompatible:
    incoming: CompositionSectionId
    left: CompositionSectionId
    right: CompositionSectionId
    overlap_lower: Vector3
    overlap_upper: Vector3


@dataclass(frozen=True)
class MuscleFormationInterval:
    index: int
    lower: float
    upper: float
    lower_radius: float
    upper_radius: float
    radius_derivative: float
    envelope_ratio: float


@dataclass(frozen=True)
class MuscleFormationFrame:
    station_index: int
    station: float
    origin: Vector3
    width_axis: Vector3
    depth_axis: Vector3
    tangent: Vector3
    determinant: float


@dataclass(frozen=True)
class MuscleFormationJunction:
    station_index: int
    station: float
    center: Vector3
    radius: float
    vertices: tuple[Vector3, ...]


@dataclass(frozen=True)
class IntegralRadiusUnsatisfied:
    section_index: int
    prescribed_radius: float
    solved_radius: float
    residual: float
    tolerance: float


@dataclass(frozen=True)
class CanalEnvelopeObstruction:
    interval_index: int
    lower: float
    upper: float
    radius_derivative: float
    support_scale: float
    envelope_ratio: float
    required_upper_bound: float


@dataclass(frozen=True)
class UnsupportedMuscleFormation:
    reason: str
    legal_family: tuple[str, ...]


type MuscleFormationFailure = (
    IntegralRadiusUnsatisfied
    | CanalEnvelopeObstruction
    | UnsupportedMuscleFormation
)


@dataclass(frozen=True)
class MuscleFormationObstruction:
    part_id: str
    mirrored: bool
    failure: MuscleFormationFailure


@dataclass(frozen=True)
class MuscleFormationEvidence:
    part_id: str
    kind: MuscleFormationKind
    mirrored: bool
    intervals: tuple[MuscleFormationInterval, ...]
    junctions: tuple[MuscleFormationJunction, ...]
    maximum_radius_residual: float
    target_volume: float
    solved_volume: float
    volume_residual: float
    centroid: Vector3
    maximum_curvature: float
    maximum_envelope_ratio: float
    support_scale: float
    frames: tuple[MuscleFormationFrame, ...]
    contact_lower: Vector3
    contact_upper: Vector3
    branch_compatible: bool


@dataclass(frozen=True)
class SkinLayerProblem:
    formation: SkinFormationKind
    thickness: float
    region_ids: tuple[str, ...]


@dataclass(frozen=True)
class SkinRelaxationPolicy:
    projection_descent_budget: int
    descent_step_pitch: float
    projection_sample_count: int
    iteration_budget: int
    projection_iteration_budget: int
    minimum_gradient: float
    minimum_containment_margin: float
    offset_tolerance_pitch: float
    stretch_lower_ratio: float
    stretch_upper_ratio: float
    minimum_stretch_reference_pitch: float
    minimum_face_area_reference_pitch_squared: float
    minimum_face_orientation: float
    minimum_face_area_ratio: float
    convergence_tolerance_pitch: float
    projection_tolerance_pitch: float
    locality_radius_thickness: float
    locality_radius_pitch: float
    neighborhood_radius_pitch: float
    tangential_step: float
    neighbor_weight: float
    maximum_tangential_step_pitch: float
    maximum_projection_step_pitch: float


@dataclass(frozen=True)
class SkinDomainEscape:
    boundary_grid_indices: tuple[tuple[int, int, int], ...]
    vertex_indices: tuple[int, ...]
    domain_lower: Vector3
    domain_upper: Vector3
    minimum_boundary_value: float
    required_minimum_value: float


@dataclass(frozen=True)
class SkinGradientDegeneracy:
    vertex_indices: tuple[int, ...]
    minimum_gradient: float
    required_minimum_gradient: float


@dataclass(frozen=True)
class SkinRootAbsent:
    vertex_indices: tuple[int, ...]
    sample_count: int
    minimum_sample_value: float
    maximum_sample_value: float


@dataclass(frozen=True)
class SkinRootMultiplicity:
    vertex_indices: tuple[int, ...]
    sample_count: int
    minimum_root_count: int
    maximum_root_count: int
    root_distances_world: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class SkinProjectionNonlocal:
    vertex_indices: tuple[int, ...]
    maximum_displacement: float
    locality_radius: float


type SkinProjectionFailure = (
    SkinDomainEscape
    | SkinGradientDegeneracy
    | SkinRootAbsent
    | SkinRootMultiplicity
    | SkinProjectionNonlocal
)


@dataclass(frozen=True)
class SkinProjectionUnsatisfied:
    failure: SkinProjectionFailure


@dataclass(frozen=True)
class SkinContainmentUnsatisfied:
    vertex_indices: tuple[int, ...]
    minimum_margin: float
    required_minimum_margin: float


@dataclass(frozen=True)
class SkinOffsetUnsatisfied:
    vertex_indices: tuple[int, ...]
    maximum_residual: float
    tolerance: float


@dataclass(frozen=True)
class SkinStretchUnsatisfied:
    edges: tuple[tuple[int, int], ...]
    minimum_ratio: float
    maximum_ratio: float
    required_lower_ratio: float
    required_upper_ratio: float


@dataclass(frozen=True)
class SkinFoldover:
    face_indices: tuple[int, ...]
    minimum_orientation: float
    minimum_area_ratio: float
    required_minimum_orientation: float
    required_minimum_area_ratio: float


@dataclass(frozen=True)
class SkinConvergenceUnsatisfied:
    iteration_budget: int
    fixed_point_residual: float
    tolerance: float
    maximum_projection_residual: float
    projection_tolerance: float


type SkinRelaxationFailure = (
    SkinProjectionUnsatisfied
    | SkinContainmentUnsatisfied
    | SkinOffsetUnsatisfied
    | SkinStretchUnsatisfied
    | SkinFoldover
    | SkinConvergenceUnsatisfied
)


@dataclass(frozen=True)
class SkinRelaxationObstruction:
    region_ids: tuple[str, ...]
    failure: SkinRelaxationFailure


@dataclass(frozen=True)
class SkinCorrespondence:
    flesh_vertices: np.ndarray
    formation_sections: tuple[tuple[CompositionSectionId, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "flesh_vertices",
            read_only_array(
                np.array(self.flesh_vertices, dtype=np.float64, copy=True)
            ),
        )


@dataclass(frozen=True)
class SkinRelaxationEvidence:
    formation: SkinFormationKind
    region_ids: tuple[str, ...]
    thickness: float
    source_vertex_count: int
    source_face_count: int
    formed_correspondence_count: int
    legacy_correspondence_count: int
    maximum_owner_multiplicity: int
    projection_descent_budget: int
    projection_sample_count: int
    minimum_root_multiplicity: int
    maximum_root_multiplicity: int
    minimum_projection_gradient: float
    minimum_containment_margin: float
    maximum_offset_residual: float
    minimum_edge_stretch: float
    maximum_edge_stretch: float
    minimum_face_orientation: float
    minimum_face_area_ratio: float
    iteration_budget: int
    projection_iteration_budget: int
    maximum_projection_residual: float
    maximum_tangential_displacement: float
    fixed_point_residual: float
    maximum_projection_displacement: float


@dataclass(frozen=True)
class AcceptedSkinLayer:
    vertices: np.ndarray
    normals: np.ndarray
    field: np.ndarray
    correspondence: SkinCorrespondence
    evidence: SkinRelaxationEvidence


type LocalCompositionObstruction = (
    FieldScaleUncalibrated
    | NoCompatibleOverlap
    | BlendSupportEscaped
    | GradientDegeneracy
    | TopologyChanged
    | LocalSectionsIncompatible
)


type SurfaceFormationObstruction = (
    LocalCompositionObstruction
    | MuscleFormationObstruction
    | SkinRelaxationObstruction
)


@dataclass(frozen=True)
class LegacyCompositionProblem:
    accumulated: np.ndarray
    incoming: np.ndarray
    operator: LegacyCompositionOperator
    radius: float


@dataclass(frozen=True)
class GradientMagnitudeCertificate:
    minimum: float
    maximum: float


@dataclass(frozen=True)
class CertifiedOverlapWitness:
    point: Vector3
    residual: float
    probe_count: int
    support: CompositionSupportRegion


@dataclass(frozen=True)
class CertifiedLocalCompositionProblem:
    accumulated: np.ndarray
    incoming: np.ndarray
    radius: float
    accumulated_gradient: np.ndarray
    incoming_gradient: np.ndarray
    accumulated_gradient_certificate: GradientMagnitudeCertificate
    incoming_gradient_certificate: GradientMagnitudeCertificate
    accumulated_section: CompositionSectionId
    incoming_section: CompositionSectionId
    overlaps: tuple[CertifiedOverlapWitness, ...]
    sample_points: np.ndarray
    domain_lower: Vector3
    domain_upper: Vector3
    topology_shape: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class AcceptedComposition:
    field: np.ndarray
    evidence: LocalCompositionEvidence | None


@dataclass(frozen=True)
class RejectedSurfaceFormation:
    obstructions: tuple[SurfaceFormationObstruction, ...]


type CompositionResult = AcceptedComposition | RejectedSurfaceFormation


@dataclass(frozen=True)
class RelativeProfileDepths:
    ratios: tuple[float, ...]


@dataclass(frozen=True)
class AbsoluteProfileDepths:
    values: tuple[float, ...]


type ProfileDepths = RelativeProfileDepths | AbsoluteProfileDepths


@dataclass(frozen=True)
class Profile:
    exponents: tuple[float, ...]
    depths: ProfileDepths
    offsets: tuple[Vector2, ...]
    rolls: tuple[float, ...]
    up: Vector3


def profile_anchor_depths(
    profile: Profile,
    widths: tuple[float, ...],
) -> tuple[float, ...]:
    match profile.depths:
        case RelativeProfileDepths(ratios):
            return tuple(
                width * ratio
                for width, ratio in zip(widths, ratios, strict=True)
            )
        case AbsoluteProfileDepths(values):
            return values
        case _ as unreachable:
            assert_never(unreachable)


@dataclass(frozen=True)
class GencylPart:
    part_id: str
    spine: tuple[Vector3, ...]
    radii: tuple[float, ...]
    mirror: bool
    blend: float | None
    operator: CompositionOperator
    profile: Profile | None
    obstructions: tuple[GeometryObstruction, ...]
    formation: MuscleFormationKind | None = None


@dataclass(frozen=True)
class BlobPart:
    part_id: str
    center: Vector3
    size: Vector3
    mirror: bool
    blend: float | None
    operator: CompositionOperator
    rotation: Rotation | None
    obstructions: tuple[GeometryObstruction, ...]


@dataclass(frozen=True)
class BoxPart:
    part_id: str
    center: Vector3
    size: Vector3
    round_radius: float
    mirror: bool
    blend: float | None
    operator: CompositionOperator
    rotation: Rotation | None
    obstructions: tuple[GeometryObstruction, ...]


type Part = GencylPart | BlobPart | BoxPart


@dataclass(frozen=True)
class WebAnchorCurve:
    spine: tuple[Vector3, ...]
    radii: tuple[float, ...]


@dataclass(frozen=True)
class SpanningWeb:
    """A thickened ruled loft over an ordered tuple of anchor curves.

    Consecutive anchors are joined at corresponding stations by two linear
    triangles per station interval. Anchor radii are interpolated over those
    triangles as local sheet half-extents; no tension or elastic solve is
    implied.
    """

    part_id: str
    anchors: tuple[WebAnchorCurve, ...]
    mirror: bool
    blend: float | None
    operator: CompositionOperator
    obstructions: tuple[GeometryObstruction, ...]


@dataclass(frozen=True)
class GeometryGraph:
    blend: float
    parts: tuple[Part, ...]
    webs: tuple[SpanningWeb, ...]
    carves: tuple[Part, ...]
    skin: SkinLayerProblem | None = None


class SurfaceDetailKind(StrEnum):
    CURVATURE = "curvature"


class SurfaceDetailRule(StrEnum):
    DETAIL_OBJECT = "detail_object"
    KIND = "kind"
    FINITE_NON_NEGATIVE_AMPLITUDE = "finite_non_negative_amplitude"
    FINITE_POSITIVE_SCALE = "finite_positive_scale"
    CREATURE_ROLE = "creature_role"
    AMPLITUDE_PITCH_LIMIT = "amplitude_pitch_limit"
    SCALE_MESH_DIAGONAL = "scale_mesh_diagonal"


@dataclass(frozen=True)
class CurvatureSurfaceDetail:
    kind: SurfaceDetailKind
    amplitude: float
    scale: float


@dataclass(frozen=True)
class SurfaceDetailObstruction:
    address: str
    rule: SurfaceDetailRule
    authored: object
    required: object


@dataclass(frozen=True)
class AcceptedSurfaceDetail:
    detail: CurvatureSurfaceDetail


@dataclass(frozen=True)
class RejectedSurfaceDetail:
    obstructions: tuple[SurfaceDetailObstruction, ...]


type SurfaceDetailResult = AcceptedSurfaceDetail | RejectedSurfaceDetail


@dataclass(frozen=True)
class CurvatureDisplacement:
    vertices: np.ndarray
    signed_displacements: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vertices",
            read_only_array(np.array(self.vertices, dtype=np.float64, copy=True)),
        )
        object.__setattr__(
            self,
            "signed_displacements",
            read_only_array(
                np.array(self.signed_displacements, dtype=np.float64, copy=True)
            ),
        )


def decode_surface_detail(payload: object) -> SurfaceDetailResult:
    if not isinstance(payload, Mapping):
        return RejectedSurfaceDetail(
            (
                SurfaceDetailObstruction(
                    "/surface_detail",
                    SurfaceDetailRule.DETAIL_OBJECT,
                    payload,
                    "object",
                ),
            )
        )
    kind = payload.get("kind")
    amplitude = payload.get("amplitude")
    scale = payload.get("scale")
    obstructions = (
        *(
            ()
            if kind == SurfaceDetailKind.CURVATURE.value
            else (
                SurfaceDetailObstruction(
                    "/surface_detail/kind",
                    SurfaceDetailRule.KIND,
                    kind,
                    tuple(item.value for item in SurfaceDetailKind),
                ),
            )
        ),
        *(
            ()
            if isinstance(amplitude, (int, float))
            and not isinstance(amplitude, bool)
            and isfinite(float(amplitude))
            and float(amplitude) >= 0.0
            else (
                SurfaceDetailObstruction(
                    "/surface_detail/amplitude",
                    SurfaceDetailRule.FINITE_NON_NEGATIVE_AMPLITUDE,
                    amplitude,
                    {"finite": True, "minimum": 0.0},
                ),
            )
        ),
        *(
            ()
            if isinstance(scale, (int, float))
            and not isinstance(scale, bool)
            and isfinite(float(scale))
            and float(scale) > 0.0
            else (
                SurfaceDetailObstruction(
                    "/surface_detail/scale",
                    SurfaceDetailRule.FINITE_POSITIVE_SCALE,
                    scale,
                    {"finite": True, "exclusive_minimum": 0.0},
                ),
            )
        ),
    )
    return (
        RejectedSurfaceDetail(obstructions)
        if obstructions
        else AcceptedSurfaceDetail(
            CurvatureSurfaceDetail(
                SurfaceDetailKind(str(kind)),
                float(cast(float, amplitude)),
                float(cast(float, scale)),
            )
        )
    )


@dataclass(frozen=True)
class EvaluatedMorphology:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    field: np.ndarray
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    resolution: int
    world_pitch: tuple[float, float, float]
    composition_evidence: tuple[LocalCompositionEvidence, ...] = ()
    formation_evidence: tuple[MuscleFormationEvidence, ...] = ()
    skin_evidence: SkinRelaxationEvidence | None = None
    skin_correspondence: SkinCorrespondence | None = None

    @property
    def maximum_pitch(self) -> float:
        return max(self.world_pitch)


class GeometryDecodeFailure(ValueError):
    def __init__(self, obstructions: tuple[GeometryObstruction, ...]) -> None:
        self.obstructions = obstructions
        super().__init__("; ".join(map(render_obstruction, obstructions)))


def render_obstruction(obstruction: GeometryObstruction) -> str:
    return (
        f"[{obstruction.rule.value}] {obstruction.part_id} "
        f"({obstruction.kind}): {obstruction.detail}"
    )


def _reject(
    part_id: str,
    kind: str,
    rule: GeometryRule,
    detail: str,
) -> Rejected:
    return Rejected((GeometryObstruction(part_id, kind, rule, detail, True),))


def _array(
    value: object,
    *,
    part_id: str,
    kind: str,
    address: str,
) -> DecodeResult[np.ndarray]:
    try:
        decoded = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as failure:
        return _reject(
            part_id,
            kind,
            GeometryRule.MALFORMED_PART,
            f"{address} is not numeric: {failure}",
        )
    if not np.isfinite(decoded).all():
        return _reject(
            part_id,
            kind,
            GeometryRule.MALFORMED_PART,
            f"{address} must contain only finite numbers",
        )
    return Accepted(decoded)


def _vector3(
    value: object,
    *,
    part_id: str,
    kind: str,
    address: str,
) -> DecodeResult[Vector3]:
    decoded = _array(value, part_id=part_id, kind=kind, address=address)
    if isinstance(decoded, Rejected):
        return decoded
    if decoded.value.shape != (3,):
        return _reject(
            part_id,
            kind,
            GeometryRule.MALFORMED_PART,
            f"{address} has shape {decoded.value.shape}; expected (3,)",
        )
    x, y, z = map(float, decoded.value)
    return Accepted((x, y, z))


def _optional_blend(
    raw_part: Mapping[str, object],
    *,
    part_id: str,
    kind: str,
) -> DecodeResult[float | None]:
    value = raw_part.get("blend")
    if value is None:
        return Accepted(None)
    try:
        blend = float(value)
    except (TypeError, ValueError) as failure:
        return _reject(
            part_id,
            kind,
            GeometryRule.MALFORMED_PART,
            f"blend is not numeric: {failure}",
        )
    if not isfinite(blend):
        return _reject(
            part_id,
            kind,
            GeometryRule.MALFORMED_PART,
            "blend must be finite",
        )
    return Accepted(blend)


def _composition_operator(
    raw_part: Mapping[str, object],
    *,
    part_id: str,
    kind: str,
) -> DecodeResult[CompositionOperator]:
    value = raw_part.get("operator", CompositionOperator.BLEND.value)
    if not isinstance(value, str):
        return _reject(
            part_id,
            kind,
            GeometryRule.COMPOSITION_OPERATOR,
            f"operator must be one of {[item.value for item in CompositionOperator]}",
        )
    try:
        return Accepted(CompositionOperator(value))
    except ValueError:
        return _reject(
            part_id,
            kind,
            GeometryRule.COMPOSITION_OPERATOR,
            f"unknown composition operator {value!r}; expected one of "
            f"{[item.value for item in CompositionOperator]}",
        )


def _rotation(
    value: object,
    *,
    part_id: str,
    kind: str,
) -> DecodeResult[Rotation]:
    decoded = _array(value, part_id=part_id, kind=kind, address="rot")
    if isinstance(decoded, Rejected):
        return decoded
    if decoded.value.shape != (4,):
        return _reject(
            part_id,
            kind,
            GeometryRule.MALFORMED_PART,
            f"rot has shape {decoded.value.shape}; expected (4,)",
        )
    norm = float(np.linalg.norm(decoded.value))
    canonical = read_quat(decoded.value)
    quaternion = tuple(map(float, canonical))
    obstruction = GeometryObstruction(
        part_id,
        kind,
        GeometryRule.QUAT_NONUNIT,
        f"|rot| = {norm:.9f} differs from 1 by {abs(norm - 1.0):.2e}; "
        "normalized on read.",
        False,
    )
    return Accepted(
        Rotation(quaternion, norm),
        () if abs(norm - 1.0) <= QUAT_UNIT_TOL else (obstruction,),
    )


def _anchor_values(
    value: object,
    count: int,
    *,
    part_id: str,
    parameter: str,
    default: float,
    rule: GeometryRule,
) -> DecodeResult[tuple[float, ...]]:
    selected = default if value is None else value
    decoded = _array(
        selected,
        part_id=part_id,
        kind="gencyl",
        address=f"profile.{parameter}",
    )
    if isinstance(decoded, Rejected):
        return _reject(
            part_id,
            "gencyl",
            rule,
            decoded.obstructions[0].detail,
        )
    flattened = decoded.value.reshape(-1)
    if decoded.value.ndim == 0 or flattened.shape[0] == 1:
        return Accepted(tuple(float(flattened[0]) for _ in range(count)))
    if flattened.shape[0] != count:
        return _reject(
            part_id,
            "gencyl",
            GeometryRule.PROFILE_ANCHOR_ARITY,
            f"profile per-anchor list has {flattened.shape[0]} entries; "
            f"spine has {count}",
        )
    return Accepted(tuple(map(float, flattened)))


def _profile_offsets(
    value: object,
    count: int,
    *,
    part_id: str,
) -> DecodeResult[tuple[Vector2, ...]]:
    if value is None:
        return Accepted(tuple((0.0, 0.0) for _ in range(count)))
    try:
        offsets = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as failure:
        return _reject(
            part_id,
            "gencyl",
            GeometryRule.PROFILE_ANCHOR_ARITY,
            f"profile offset is not numeric: {failure}",
        )
    if offsets.shape == (2,):
        broadcast = np.broadcast_to(offsets, (count, 2))
    elif offsets.shape == (count, 2):
        broadcast = offsets
    else:
        return _reject(
            part_id,
            "gencyl",
            GeometryRule.PROFILE_ANCHOR_ARITY,
            f"profile offset has shape {offsets.shape}; expected (2,) or "
            f"({count}, 2)",
        )
    if not np.isfinite(broadcast).all():
        return _reject(
            part_id,
            "gencyl",
            GeometryRule.PROFILE_OFFSET,
            "profile offsets must all be finite",
        )
    return Accepted(tuple((float(x), float(y)) for x, y in broadcast))


def decode_profile(
    value: object,
    anchor_count: int,
    *,
    part_id: str,
) -> DecodeResult[Profile]:
    if not isinstance(value, Mapping):
        return _reject(
            part_id,
            "gencyl",
            GeometryRule.PROFILE_ANCHOR_ARITY,
            "profile must be an object",
        )
    depth_conflict = "depth" in value and "aspect" in value
    if depth_conflict:
        return _reject(
            part_id,
            "gencyl",
            GeometryRule.PROFILE_DEPTH_CONFLICT,
            "profile must author exactly one of depth or aspect",
        )
    exponents = _anchor_values(
        value.get("n"),
        anchor_count,
        part_id=part_id,
        parameter="n",
        default=4.0,
        rule=GeometryRule.PROFILE_EXPONENT,
    )
    depth_values = _anchor_values(
        value.get("depth") if "depth" in value else value.get("aspect"),
        anchor_count,
        part_id=part_id,
        parameter="depth" if "depth" in value else "aspect",
        default=1.0,
        rule=(
            GeometryRule.PROFILE_DEPTH
            if "depth" in value
            else GeometryRule.PROFILE_ASPECT
        ),
    )
    offsets = _profile_offsets(value.get("offset"), anchor_count, part_id=part_id)
    rolls = _anchor_values(
        value.get("roll"),
        anchor_count,
        part_id=part_id,
        parameter="roll",
        default=0.0,
        rule=GeometryRule.PROFILE_ROLL,
    )
    up = _vector3(
        value.get("up", (0.0, 0.0, 1.0)),
        part_id=part_id,
        kind="gencyl",
        address="profile.up",
    )
    structural = tuple(
        chain.from_iterable(
            result.obstructions
            for result in (exponents, depth_values, offsets, rolls, up)
            if isinstance(result, Rejected)
        )
    )
    if structural:
        return Rejected(structural)
    accepted_exponents = cast(Accepted[tuple[float, ...]], exponents)
    accepted_depth_values = cast(Accepted[tuple[float, ...]], depth_values)
    accepted_offsets = cast(Accepted[tuple[Vector2, ...]], offsets)
    accepted_rolls = cast(Accepted[tuple[float, ...]], rolls)
    accepted_up = cast(Accepted[Vector3], up)
    depth_min = min(accepted_depth_values.value)
    depth_max = max(accepted_depth_values.value)
    exponent_min = min(accepted_exponents.value)
    exponent_max = max(accepted_exponents.value)
    depth_obstruction = (
        GeometryObstruction(
            part_id,
            "gencyl",
            GeometryRule.PROFILE_DEPTH,
            f"depth range [{depth_min:.3f}, {depth_max:.3f}] must be positive.",
            True,
        )
        if "depth" in value
        else GeometryObstruction(
            part_id,
            "gencyl",
            GeometryRule.PROFILE_ASPECT,
            f"aspect range [{depth_min:.3f}, {depth_max:.3f}] outside (0, 1]; "
            "the width axis is the major axis -- swap roles via 'up'.",
            depth_min <= 0.0,
        )
    )
    exponent_obstruction = GeometryObstruction(
        part_id,
        "gencyl",
        GeometryRule.PROFILE_EXPONENT,
        f"n range [{exponent_min:.2f}, {exponent_max:.2f}] outside [2, 12].",
        exponent_min <= 0.0,
    )
    semantic = (
        (
            ()
            if depth_min > 0.0 and ("depth" in value or depth_max <= 1.0)
            else (depth_obstruction,)
        )
        + (() if 2.0 <= exponent_min and exponent_max <= 12.0 else (exponent_obstruction,))
    )
    profile = Profile(
        accepted_exponents.value,
        (
            AbsoluteProfileDepths(accepted_depth_values.value)
            if "depth" in value
            else RelativeProfileDepths(accepted_depth_values.value)
        ),
        accepted_offsets.value,
        accepted_rolls.value,
        accepted_up.value,
    )
    return (
        Rejected(semantic)
        if any(map(lambda obstruction: obstruction.fatal, semantic))
        else Accepted(profile, semantic)
    )


def _decode_gencyl(
    raw_part: Mapping[str, object],
    *,
    part_id: str,
    mirror: bool,
    blend: float | None,
    operator: CompositionOperator,
) -> DecodeResult[GencylPart]:
    spine = _array(raw_part.get("spine"), part_id=part_id, kind="gencyl", address="spine")
    radii = _array(raw_part.get("radii"), part_id=part_id, kind="gencyl", address="radii")
    structural = tuple(
        chain.from_iterable(
            result.obstructions
            for result in (spine, radii)
            if isinstance(result, Rejected)
        )
    )
    if structural:
        return Rejected(structural)
    accepted_spine = cast(Accepted[np.ndarray], spine)
    accepted_radii = cast(Accepted[np.ndarray], radii)
    if (
        accepted_spine.value.ndim != 2
        or accepted_spine.value.shape[1:] != (3,)
        or accepted_spine.value.shape[0] < 2
    ):
        return _reject(
            part_id,
            "gencyl",
            GeometryRule.MALFORMED_PART,
            f"spine has shape {accepted_spine.value.shape}; expected (n, 3) with n >= 2",
        )
    if accepted_radii.value.shape != (accepted_spine.value.shape[0],):
        return _reject(
            part_id,
            "gencyl",
            GeometryRule.MALFORMED_PART,
            f"radii has shape {accepted_radii.value.shape}; expected "
            f"({accepted_spine.value.shape[0]},)",
        )
    if float(np.min(accepted_radii.value)) <= 0.0:
        return _reject(
            part_id,
            "gencyl",
            GeometryRule.MALFORMED_PART,
            "radii must all be positive",
        )
    fence = (
        (
            GeometryObstruction(
                part_id,
                "gencyl",
                GeometryRule.FENCE_ROT_ON_GENCYL,
                "rot is forbidden on gencyl (spines carry orientation); it is "
                "ignored, never applied.",
                False,
            ),
        )
        if raw_part.get("rot") is not None
        else ()
    )
    profile_value = raw_part.get("profile")
    profile = (
        Accepted(None)
        if profile_value is None
        else decode_profile(
            profile_value, accepted_radii.value.shape[0], part_id=part_id
        )
    )
    if isinstance(profile, Rejected):
        return Rejected(fence + profile.obstructions)
    formation_value = raw_part.get("formation")
    try:
        formation = (
            None
            if formation_value is None
            else MuscleFormationKind(formation_value)
        )
    except (TypeError, ValueError):
        return Rejected(
            fence
            + profile.obstructions
            + (
                GeometryObstruction(
                    part_id,
                    "gencyl",
                    GeometryRule.MALFORMED_PART,
                    "formation must be one of "
                    f"{tuple(item.value for item in MuscleFormationKind)}, "
                    f"got {formation_value!r}",
                    True,
                ),
            )
        )
    decoded_spine = tuple(tuple(map(float, row)) for row in accepted_spine.value)
    decoded_radii = tuple(map(float, accepted_radii.value))
    obstructions = fence + profile.obstructions
    return Accepted(
        GencylPart(
            part_id,
            decoded_spine,
            decoded_radii,
            mirror,
            blend,
            operator,
            profile.value,
            obstructions,
            formation,
        ),
        obstructions,
    )


def _decode_centered_part(
    raw_part: Mapping[str, object],
    *,
    part_id: str,
    kind: str,
    mirror: bool,
    blend: float | None,
    operator: CompositionOperator,
) -> DecodeResult[BlobPart | BoxPart]:
    center = _vector3(
        raw_part.get("center"), part_id=part_id, kind=kind, address="center"
    )
    size = _vector3(raw_part.get("size"), part_id=part_id, kind=kind, address="size")
    structural = tuple(
        chain.from_iterable(
            result.obstructions
            for result in (center, size)
            if isinstance(result, Rejected)
        )
    )
    if structural:
        return Rejected(structural)
    accepted_center = cast(Accepted[Vector3], center)
    accepted_size = cast(Accepted[Vector3], size)
    if min(accepted_size.value) <= 0.0:
        return _reject(
            part_id,
            kind,
            GeometryRule.MALFORMED_PART,
            "size components must all be positive",
        )
    profile_fence = (
        (
            GeometryObstruction(
                part_id,
                kind,
                GeometryRule.FENCE_PROFILE_ON_NONGENCYL,
                "profile is the gencyl cross-section language; on any other kind "
                "it is ignored, never applied.",
                False,
            ),
        )
        if raw_part.get("profile") is not None
        else ()
    )
    rotation = (
        Accepted(None)
        if raw_part.get("rot") is None
        else _rotation(raw_part.get("rot"), part_id=part_id, kind=kind)
    )
    if isinstance(rotation, Rejected):
        return Rejected(profile_fence + rotation.obstructions)
    obstructions = profile_fence + rotation.obstructions
    if kind == "blob":
        part = BlobPart(
            part_id,
            accepted_center.value,
            accepted_size.value,
            mirror,
            blend,
            operator,
            rotation.value,
            obstructions,
        )
        return Accepted(part, obstructions)
    try:
        round_radius = float(raw_part.get("round", 0.0))
    except (TypeError, ValueError) as failure:
        return Rejected(
            obstructions
            + (
                GeometryObstruction(
                    part_id,
                    "box",
                    GeometryRule.MALFORMED_PART,
                    f"round is not numeric: {failure}",
                    True,
                ),
            )
        )
    if not isfinite(round_radius):
        return Rejected(
            obstructions
            + (
                GeometryObstruction(
                    part_id,
                    "box",
                    GeometryRule.MALFORMED_PART,
                    "round must be finite",
                    True,
                ),
            )
        )
    negative_round = (
        (
            GeometryObstruction(
                part_id,
                "box",
                GeometryRule.BOX_NEGATIVE_ROUND,
                f"round = {round_radius} < 0 is ill-formed.",
                True,
            ),
        )
        if round_radius < 0.0
        else ()
    )
    if negative_round:
        return Rejected(obstructions + negative_round)
    part = BoxPart(
        part_id,
        accepted_center.value,
        accepted_size.value,
        round_radius,
        mirror,
        blend,
        operator,
        rotation.value,
        obstructions,
    )
    return Accepted(part, obstructions)


def decode_part(raw_part: object) -> DecodeResult[Part]:
    if not isinstance(raw_part, Mapping):
        return _reject(
            "<unnamed>",
            "<unknown>",
            GeometryRule.MALFORMED_PART,
            "part must be an object",
        )
    kind_value = raw_part.get("type")
    kind = kind_value if isinstance(kind_value, str) else repr(kind_value)
    part_id_value = raw_part.get("id", "<unnamed>")
    part_id = part_id_value if isinstance(part_id_value, str) else str(part_id_value)
    if kind not in ("gencyl", "blob", "box"):
        return _reject(
            part_id,
            kind,
            GeometryRule.UNKNOWN_PART_KIND,
            f"unknown part type {kind!r}",
        )
    blend = _optional_blend(raw_part, part_id=part_id, kind=kind)
    if isinstance(blend, Rejected):
        return blend
    operator = _composition_operator(raw_part, part_id=part_id, kind=kind)
    if isinstance(operator, Rejected):
        return operator
    mirror = bool(raw_part.get("mirror"))
    return (
        _decode_gencyl(
            raw_part,
            part_id=part_id,
            mirror=mirror,
            blend=blend.value,
            operator=operator.value,
        )
        if kind == "gencyl"
        else _decode_centered_part(
            raw_part,
            part_id=part_id,
            kind=kind,
            mirror=mirror,
            blend=blend.value,
            operator=operator.value,
        )
    )


def _decode_web_anchor(
    raw_anchor: object,
    index: int,
    *,
    part_id: str,
) -> DecodeResult[WebAnchorCurve]:
    if not isinstance(raw_anchor, Mapping):
        return _reject(
            part_id,
            "web",
            GeometryRule.MALFORMED_PART,
            f"anchors[{index}] must be an object",
        )
    spine = _array(
        raw_anchor.get("spine"),
        part_id=part_id,
        kind="web",
        address=f"anchors[{index}].spine",
    )
    radii = _array(
        raw_anchor.get("radii"),
        part_id=part_id,
        kind="web",
        address=f"anchors[{index}].radii",
    )
    structural = tuple(
        chain.from_iterable(
            map(
                lambda result: result.obstructions,
                filter(
                    lambda result: isinstance(result, Rejected),
                    (spine, radii),
                ),
            )
        )
    )
    if structural:
        return Rejected(structural)
    accepted_spine = cast(Accepted[np.ndarray], spine)
    accepted_radii = cast(Accepted[np.ndarray], radii)
    if (
        accepted_spine.value.ndim != 2
        or accepted_spine.value.shape[1:] != (3,)
        or accepted_spine.value.shape[0] < 2
    ):
        return _reject(
            part_id,
            "web",
            GeometryRule.WEB_STATION_ARITY,
            f"anchors[{index}].spine has shape {accepted_spine.value.shape}; "
            "expected (n, 3) with n >= 2",
        )
    if accepted_radii.value.shape != (accepted_spine.value.shape[0],):
        return _reject(
            part_id,
            "web",
            GeometryRule.WEB_STATION_ARITY,
            f"anchors[{index}].radii has shape {accepted_radii.value.shape}; "
            f"expected ({accepted_spine.value.shape[0]},)",
        )
    if float(np.min(accepted_radii.value)) <= 0.0:
        return _reject(
            part_id,
            "web",
            GeometryRule.WEB_HALF_EXTENT,
            f"anchors[{index}].radii must all be positive half-extents",
        )
    return Accepted(
        WebAnchorCurve(
            tuple(
                map(
                    lambda row: tuple(map(float, row)),
                    accepted_spine.value,
                )
            ),
            tuple(map(float, accepted_radii.value)),
        )
    )


def decode_web(raw_web: object) -> DecodeResult[SpanningWeb]:
    if not isinstance(raw_web, Mapping):
        return _reject(
            "<unnamed>",
            "web",
            GeometryRule.MALFORMED_PART,
            "web must be an object",
        )
    part_id_value = raw_web.get("id", "<unnamed>")
    part_id = (
        part_id_value
        if isinstance(part_id_value, str)
        else str(part_id_value)
    )
    raw_anchors = raw_web.get("anchors")
    if not isinstance(raw_anchors, (list, tuple)) or len(raw_anchors) < 2:
        return _reject(
            part_id,
            "web",
            GeometryRule.WEB_ANCHOR_ARITY,
            "anchors must be an ordered sequence with at least two curves",
        )
    blend = _optional_blend(raw_web, part_id=part_id, kind="web")
    operator = _composition_operator(raw_web, part_id=part_id, kind="web")
    if isinstance(blend, Rejected):
        return blend
    if isinstance(operator, Rejected):
        return operator
    decoded_anchors = tuple(
        map(
            lambda indexed: _decode_web_anchor(
                indexed[1], indexed[0], part_id=part_id
            ),
            enumerate(raw_anchors),
        )
    )
    obstructions = tuple(
        chain.from_iterable(
            map(lambda result: result.obstructions, decoded_anchors)
        )
    )
    if any(map(lambda result: isinstance(result, Rejected), decoded_anchors)):
        return Rejected(obstructions)
    anchors = tuple(
        map(
            lambda result: result.value,
            filter(
                lambda result: isinstance(result, Accepted),
                decoded_anchors,
            ),
        )
    )
    station_counts = tuple(map(lambda anchor: len(anchor.spine), anchors))
    if len(frozenset(station_counts)) != 1:
        return _reject(
            part_id,
            "web",
            GeometryRule.WEB_STATION_ARITY,
            "all anchor curves must have identical station counts; got "
            f"{station_counts}",
        )
    return Accepted(
        SpanningWeb(
            part_id,
            anchors,
            bool(raw_web.get("mirror")),
            blend.value,
            operator.value,
            obstructions,
        ),
        obstructions,
    )


def decode_skin_layer_problem(
    payload: object,
) -> DecodeResult[SkinLayerProblem]:
    if not isinstance(payload, Mapping):
        return _reject(
            "<graph>",
            "skin",
            GeometryRule.MALFORMED_GRAPH,
            "skin must be an object",
        )
    required_fields = frozenset(("formation", "thickness", "region_ids"))
    observed_fields = frozenset(map(str, payload.keys()))
    if observed_fields != required_fields:
        return _reject(
            "<graph>",
            "skin",
            GeometryRule.MALFORMED_GRAPH,
            "skin fields must be exactly "
            f"{tuple(sorted(required_fields))}; "
            f"missing={tuple(sorted(required_fields - observed_fields))}, "
            f"extra={tuple(sorted(observed_fields - required_fields))}",
        )
    formation = payload.get("formation")
    thickness = payload.get("thickness")
    region_ids = payload.get("region_ids")
    obstructions = (
        (
            GeometryObstruction(
                "<graph>",
                "skin",
                GeometryRule.MALFORMED_GRAPH,
                "skin formation must be one of "
                f"{tuple(item.value for item in SkinFormationKind)}, "
                f"got {formation!r}",
                True,
            ),
        )
        if formation not in tuple(item.value for item in SkinFormationKind)
        else ()
    ) + (
        (
            GeometryObstruction(
                "<graph>",
                "skin",
                GeometryRule.MALFORMED_GRAPH,
                "skin thickness must be a finite positive number",
                True,
            ),
        )
        if not isinstance(thickness, (int, float))
        or isinstance(thickness, bool)
        or not isfinite(float(thickness))
        or float(thickness) <= 0.0
        else ()
    ) + (
        (
            GeometryObstruction(
                "<graph>",
                "skin",
                GeometryRule.MALFORMED_GRAPH,
                "skin region_ids must be a non-empty sequence of unique "
                "non-empty strings",
                True,
            ),
        )
        if not isinstance(region_ids, (list, tuple))
        or not region_ids
        or any(
            not isinstance(region_id, str) or not region_id
            for region_id in region_ids
        )
        or len(frozenset(region_ids)) != len(region_ids)
        else ()
    )
    return (
        Rejected(obstructions)
        if obstructions
        else Accepted(
            SkinLayerProblem(
                SkinFormationKind(str(formation)),
                float(cast(float, thickness)),
                tuple(cast(list[str] | tuple[str, ...], region_ids)),
            )
        )
    )


def decode_graph(raw_graph: object) -> DecodeResult[GeometryGraph]:
    if not isinstance(raw_graph, Mapping):
        return _reject(
            "<graph>",
            "graph",
            GeometryRule.MALFORMED_GRAPH,
            "graph must be an object",
        )
    raw_parts = raw_graph.get("parts")
    if not isinstance(raw_parts, (list, tuple)) or not raw_parts:
        return _reject(
            "<graph>",
            "graph",
            GeometryRule.MALFORMED_GRAPH,
            "parts must be a non-empty sequence",
        )
    raw_webs = raw_graph.get("webs", ())
    if not isinstance(raw_webs, (list, tuple)):
        return _reject(
            "<graph>",
            "graph",
            GeometryRule.MALFORMED_GRAPH,
            "webs must be a sequence when present",
        )
    raw_carves = raw_graph.get("carves", ())
    if not isinstance(raw_carves, (list, tuple)):
        return _reject(
            "<graph>",
            "graph",
            GeometryRule.MALFORMED_GRAPH,
            "carves must be a sequence when present",
        )
    try:
        blend = float(raw_graph.get("blend", 0.05))
    except (TypeError, ValueError) as failure:
        return _reject(
            "<graph>",
            "graph",
            GeometryRule.MALFORMED_GRAPH,
            f"blend is not numeric: {failure}",
        )
    if not isfinite(blend):
        return _reject(
            "<graph>",
            "graph",
            GeometryRule.MALFORMED_GRAPH,
            "blend must be finite",
        )
    skin = (
        Accepted(None)
        if "skin" not in raw_graph
        else decode_skin_layer_problem(raw_graph["skin"])
    )
    if isinstance(skin, Rejected):
        return skin
    decoded_parts = tuple(map(decode_part, raw_parts))
    decoded_webs = tuple(map(decode_web, raw_webs))
    decoded_carves = tuple(map(decode_part, raw_carves))
    decoded_primitives = decoded_parts + decoded_webs + decoded_carves
    obstructions = tuple(
        chain.from_iterable(result.obstructions for result in decoded_primitives)
    )
    if any(map(lambda result: isinstance(result, Rejected), decoded_primitives)):
        return Rejected(obstructions)
    parts = tuple(
        result.value for result in decoded_parts if isinstance(result, Accepted)
    )
    webs = tuple(
        map(
            lambda result: result.value,
            filter(
                lambda result: isinstance(result, Accepted),
                decoded_webs,
            ),
        )
    )
    positive_section_ids = tuple(
        section.part_id for section in chain(parts, webs)
    )
    duplicate_section_ids = tuple(
        sorted(
            identifier
            for identifier in frozenset(positive_section_ids)
            if positive_section_ids.count(identifier) > 1
        )
    )
    if skin.value is not None and duplicate_section_ids:
        return Rejected(
            (
                *obstructions,
                GeometryObstruction(
                    "<graph>",
                    "graph",
                    GeometryRule.MALFORMED_GRAPH,
                    "positive composition section ids must be unique: "
                    + ", ".join(duplicate_section_ids),
                    True,
                ),
            )
        )
    carves = tuple(
        result.value for result in decoded_carves if isinstance(result, Accepted)
    )
    return Accepted(
        GeometryGraph(blend, parts, webs, carves, skin.value),
        obstructions,
    )


def require_accepted[value](result: DecodeResult[value]) -> value:
    if isinstance(result, Rejected):
        raise GeometryDecodeFailure(result.obstructions)
    return result.value


def read_only_array(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value
