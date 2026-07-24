"""Algebra: analytic surface sampling, per-instance bbox/radii/volume, gaps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol, assert_never

import numpy as np

from golem.kernel import engine as ev  # v0.3 wrapper: part_sdf/bounds support box+rot
from golem.senses.model import immutable_array


_MIRROR_X = np.array([-1.0, 1.0, 1.0])


def _mirror_x(arr: np.ndarray) -> np.ndarray:
    """Reflect world coordinates across the x=0 plane (authored +x -> -x)."""
    return arr * _MIRROR_X


# --------------------------------------------------------------------------- #
# Analytic surface sampling (fable-b sec.1.2).                                 #
#   gencyl : 8 stations x 12 angles per segment + 2 end caps                   #
#   blob   : 96-point Fibonacci sphere scaled by semi-axes (rot via M1 quat)   #
#   box    : rounded-box surface -- 6 face grids + 12 rounded edges + 8        #
#            rounded corners (documented below), rot via M1 quat               #
# All samples lie (to taper error h^2/8R) on the true surface, so a gap read   #
# off ``sdf_other(sample)`` is an upper bound on the true gap (fable-b).       #
# --------------------------------------------------------------------------- #
_GENCYL_STATIONS = 8
_GENCYL_ANGLES = 12
_BLOB_FIB = 96
_CAP_FIB = 14  # hemisphere points per gencyl end cap


def _fib_sphere(n: int) -> np.ndarray:
    """Deterministic ~uniform unit points on S^2 (Fibonacci spiral)."""
    i = np.arange(n, dtype=np.float64)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, None))
    theta = phi * i
    return np.stack([r * np.cos(theta), y, r * np.sin(theta)], axis=1)


def _perp_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to ``axis``
    (unit). Deterministic tie-break identical in spirit to the frame doc."""
    a = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, axis)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def _sample_gencyl(spine: np.ndarray, radii: np.ndarray) -> np.ndarray:
    thetas = 2.0 * math.pi * np.arange(_GENCYL_ANGLES) / _GENCYL_ANGLES
    ring = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    segments = tuple(
        (spine[index], spine[index + 1], radii[index], radii[index + 1])
        for index in range(len(spine) - 1)
        if np.linalg.norm(spine[index + 1] - spine[index]) >= 1.0e-9
    )
    sides = tuple(
        _sample_gencyl_segment(left, right, left_radius, right_radius, ring)
        for left, right, left_radius, right_radius in segments
    )
    caps = tuple(
        _sample_gencyl_cap(endpoint, radius, direction)
        for endpoint, radius, direction in (
            (spine[0], radii[0], -(spine[1] - spine[0])),
            (spine[-1], radii[-1], spine[-1] - spine[-2]),
        )
    )
    return np.concatenate((*sides, *caps), axis=0)


def _sample_gencyl_segment(
    left: np.ndarray,
    right: np.ndarray,
    left_radius: float,
    right_radius: float,
    ring: np.ndarray,
) -> np.ndarray:
    segment = right - left
    axis = segment / np.linalg.norm(segment)
    u_axis, v_axis = _perp_frame(axis)
    return np.concatenate(
        tuple(
            left
            + parameter * segment
            + (left_radius + (right_radius - left_radius) * parameter)
            * (ring[:, 0:1] * u_axis + ring[:, 1:2] * v_axis)
            for parameter in np.linspace(0.0, 1.0, _GENCYL_STATIONS)
        ),
        axis=0,
    )


def _sample_gencyl_cap(
    endpoint: np.ndarray, radius: float, outward: np.ndarray
) -> np.ndarray:
    normal = outward / np.linalg.norm(outward)
    hemisphere = _fib_sphere(_CAP_FIB)
    return np.concatenate(
        (
            endpoint + radius * hemisphere[hemisphere @ normal >= 0.0],
            (endpoint + radius * normal)[None, :],
        ),
        axis=0,
    )


def _sample_formed_interval(
    interval: ev.MuscleFormationInterval,
    left: ev.MuscleFormationFrame,
    right: ev.MuscleFormationFrame,
    ring: np.ndarray,
) -> np.ndarray:
    parameters = np.linspace(0.0, 1.0, _GENCYL_STATIONS)
    left_origin = np.asarray(left.origin, dtype=np.float64)
    right_origin = np.asarray(right.origin, dtype=np.float64)
    width_axis = np.asarray(left.width_axis, dtype=np.float64)
    depth_axis = np.asarray(left.depth_axis, dtype=np.float64)
    radial_directions = (
        ring[:, 0:1] * width_axis + ring[:, 1:2] * depth_axis
    )

    def station(parameter: float) -> np.ndarray:
        origin = (
            (1.0 - parameter) * left_origin + parameter * right_origin
        )
        radius = (
            (1.0 - parameter) * interval.lower_radius
            + parameter * interval.upper_radius
        )
        return origin + radius * radial_directions

    return np.concatenate(tuple(map(station, parameters)), axis=0)


def _sample_formed_cap(
    frame: ev.MuscleFormationFrame,
    radius: float,
    start: bool,
) -> np.ndarray:
    origin = np.asarray(frame.origin, dtype=np.float64)
    tangent = np.asarray(frame.tangent, dtype=np.float64)
    directions = _fib_sphere(2 * _CAP_FIB)
    axial = directions @ tangent
    exposed = directions[axial <= 0.0 if start else axial >= 0.0]
    apex = -tangent if start else tangent
    return origin + radius * np.concatenate((exposed, apex[None, :]), axis=0)


def _sample_formed_muscle(
    evidence: ev.MuscleFormationEvidence,
) -> np.ndarray:
    thetas = 2.0 * math.pi * np.arange(_GENCYL_ANGLES) / _GENCYL_ANGLES
    ring = np.stack((np.cos(thetas), np.sin(thetas)), axis=1)
    framed_intervals = tuple(
        zip(evidence.intervals, pairwise(evidence.frames), strict=True)
    )
    sides = tuple(
        _sample_formed_interval(interval, left, right, ring)
        for interval, (left, right) in framed_intervals
    )
    first_interval = evidence.intervals[0]
    last_interval = evidence.intervals[-1]
    caps = (
        _sample_formed_cap(
            evidence.frames[0],
            first_interval.lower_radius,
            True,
        ),
        _sample_formed_cap(
            evidence.frames[-1],
            last_interval.upper_radius,
            False,
        ),
    )
    junctions = tuple(
        np.asarray(junction.vertices, dtype=np.float64)
        for junction in evidence.junctions
    )
    return np.concatenate((*sides, *caps, *junctions), axis=0)


def _sample_blob(center: np.ndarray, size: np.ndarray, R: np.ndarray | None) -> np.ndarray:
    d = _fib_sphere(_BLOB_FIB) * size  # ellipsoid in local frame
    if R is not None:
        d = d @ R.T
    return center + d


def _sample_box(
    center: np.ndarray, size: np.ndarray, round_r: float, R: np.ndarray | None
) -> np.ndarray:
    """Rounded-box surface sampling (documented scheme, ~140 points):

      * 6 FACES: for axis i, both signs; a 4x4 grid over the CORE extent
        [-h_j,h_j]x[-h_k,h_k], pushed out to the total half-extent h_i+round
        along +/- e_i (the flat region of each rounded face).
      * 12 EDGES: 3 points along each core edge, bulged outward by round along
        the 45-degree diagonal of the two incident faces (the quarter-cylinder).
      * 8 CORNERS: the core corner pushed out by round along the (+-1,+-1,+-1)/
        sqrt(3) diagonal (the sphere-octant).
    ``size`` = CORE half-extents; total half-extent per axis = size+round.
    ``rot`` (if any) is applied to every local point.
    """
    half_extents = np.asarray(size, dtype=np.float64)
    radius = float(round_r)
    total_extents = half_extents + radius
    grid = np.linspace(-1.0, 1.0, 4)
    basis = np.eye(3, dtype=np.float64)
    faces = tuple(
        sign * total_extents[axis] * basis[axis]
        + first_grid * half_extents[first_axis] * basis[first_axis]
        + second_grid * half_extents[second_axis] * basis[second_axis]
        for axis in range(3)
        for first_axis, second_axis in (((axis + 1) % 3, (axis + 2) % 3),)
        for sign in (-1.0, 1.0)
        for first_grid in grid
        for second_grid in grid
    )
    edges = tuple(
        axial * half_extents[axis] * basis[axis]
        + first_sign
        * (half_extents[first_axis] + radius / math.sqrt(2.0))
        * basis[first_axis]
        + second_sign
        * (half_extents[second_axis] + radius / math.sqrt(2.0))
        * basis[second_axis]
        for axis in range(3)
        for first_axis, second_axis in (((axis + 1) % 3, (axis + 2) % 3),)
        for first_sign in (-1.0, 1.0)
        for second_sign in (-1.0, 1.0)
        for axial in np.linspace(-1.0, 1.0, 3)
    )
    corners = tuple(
        np.asarray((x_sign, y_sign, z_sign))
        * (half_extents + radius / math.sqrt(3.0))
        for x_sign in (-1.0, 1.0)
        for y_sign in (-1.0, 1.0)
        for z_sign in (-1.0, 1.0)
    )
    points = np.asarray(
        (*faces, *((*edges, *corners) if radius > 0.0 else ())),
        dtype=np.float64,
    )
    if R is not None:
        points = points @ R.T
    return center + points


def _part_rot_matrix(part, mirrored: bool) -> np.ndarray | None:
    if part.get("rot") is None:
        return None
    return ev.quat_to_matrix(ev.read_quat(part["rot"], mirrored=mirrored))


@dataclass(frozen=True)
class GeneralizedCylinder:
    spine: np.ndarray
    radii: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "spine", immutable_array(self.spine))
        object.__setattr__(self, "radii", immutable_array(self.radii))


def _freeze_oriented(shape: Blob | Box) -> None:
    """Freeze the shared center/size/rotation arrays of an oriented primitive."""
    object.__setattr__(shape, "center", immutable_array(shape.center))
    object.__setattr__(shape, "size", immutable_array(shape.size))
    object.__setattr__(
        shape,
        "rotation",
        immutable_array(shape.rotation) if shape.rotation is not None else None,
    )


@dataclass(frozen=True)
class Blob:
    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray | None

    def __post_init__(self) -> None:
        _freeze_oriented(self)


@dataclass(frozen=True)
class Box:
    center: np.ndarray
    size: np.ndarray
    round_radius: float
    rotation: np.ndarray | None

    def __post_init__(self) -> None:
        _freeze_oriented(self)


@dataclass(frozen=True)
class Web:
    """A spanning membrane: ordered anchor curves joined as a ruled loft."""

    spines: tuple[np.ndarray, ...]
    radii: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "spines", tuple(immutable_array(s) for s in self.spines)
        )
        object.__setattr__(
            self, "radii", tuple(immutable_array(r) for r in self.radii)
        )


type PartShape = (
    GeneralizedCylinder | Blob | Box | Web | ev.MuscleFormationEvidence
)


class PartShapeAlgebra[result](Protocol):
    def generalized_cylinder(self, shape: GeneralizedCylinder) -> result: ...
    def blob(self, shape: Blob) -> result: ...
    def box(self, shape: Box) -> result: ...
    def web(self, shape: Web) -> result: ...
    def formed_muscle(self, shape: ev.MuscleFormationEvidence) -> result: ...


def _part_shape(part, mirrored: bool) -> PartShape:
    if "anchors" in part:
        spines = tuple(
            np.asarray(anchor["spine"], dtype=np.float64)
            for anchor in part["anchors"]
        )
        return Web(
            spines=(
                tuple(_mirror_x(spine) for spine in spines)
                if mirrored
                else spines
            ),
            radii=tuple(
                np.asarray(anchor["radii"], dtype=np.float64)
                for anchor in part["anchors"]
            ),
        )
    match part["type"]:
        case "gencyl":
            spine = np.asarray(part["spine"], dtype=np.float64)
            return GeneralizedCylinder(
                spine=_mirror_x(spine) if mirrored else spine,
                radii=np.asarray(part["radii"], dtype=np.float64),
            )
        case "blob":
            center = np.asarray(part["center"], dtype=np.float64)
            return Blob(
                center=_mirror_x(center) if mirrored else center,
                size=np.asarray(part["size"], dtype=np.float64),
                rotation=_part_rot_matrix(part, mirrored),
            )
        case "box":
            center = np.asarray(part["center"], dtype=np.float64)
            return Box(
                center=_mirror_x(center) if mirrored else center,
                size=np.asarray(part["size"], dtype=np.float64),
                round_radius=float(part.get("round", 0.0)),
                rotation=_part_rot_matrix(part, mirrored),
            )
        case unknown:
            raise ValueError(f"unknown part type {unknown!r}")


def _fold_part_shape[result](
    shape: PartShape, algebra: PartShapeAlgebra[result]
) -> result:
    match shape:
        case GeneralizedCylinder():
            return algebra.generalized_cylinder(shape)
        case Blob():
            return algebra.blob(shape)
        case Box():
            return algebra.box(shape)
        case Web():
            return algebra.web(shape)
        case ev.MuscleFormationEvidence():
            return algebra.formed_muscle(shape)
        case _ as unreachable:
            assert_never(unreachable)


# Barycentric weights sampling one ruled triangle: vertices, edge midpoints,
# barycenter. Each sample is pushed out along the triangle normal by the
# barycentric-interpolated half-extent, landing on both sheet faces.
_WEB_BARYCENTRIC = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
)


def _web_triangles(
    shape: Web,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float], ...]:
    """The ruled triangulation the engine lofts: consecutive anchors joined at
    corresponding stations by two triangles per station interval."""
    return tuple(
        triangle
        for (left, right), (left_radii, right_radii) in zip(
            zip(shape.spines, shape.spines[1:]),
            zip(shape.radii, shape.radii[1:]),
        )
        for index in range(len(left) - 1)
        for triangle in (
            (
                left[index],
                left[index + 1],
                right[index],
                float(left_radii[index]),
                float(left_radii[index + 1]),
                float(right_radii[index]),
            ),
            (
                left[index + 1],
                right[index + 1],
                right[index],
                float(left_radii[index + 1]),
                float(right_radii[index + 1]),
                float(right_radii[index]),
            ),
        )
    )


def _sample_web(shape: Web) -> np.ndarray:
    samples = []
    for a, b, c, radius_a, radius_b, radius_c in _web_triangles(shape):
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        unit_normal = normal / norm if norm >= 1.0e-12 else np.zeros(3)
        for weight_a, weight_b, weight_c in _WEB_BARYCENTRIC:
            point = weight_a * a + weight_b * b + weight_c * c
            half_extent = (
                weight_a * radius_a + weight_b * radius_b + weight_c * radius_c
            )
            samples.append(point + half_extent * unit_normal)
            samples.append(point - half_extent * unit_normal)
    return (
        np.asarray(samples, dtype=np.float64)
        if samples
        else np.zeros((0, 3), dtype=np.float64)
    )


class _SampleAlgebra:
    def generalized_cylinder(self, shape: GeneralizedCylinder) -> np.ndarray:
        return _sample_gencyl(shape.spine, shape.radii)

    def blob(self, shape: Blob) -> np.ndarray:
        return _sample_blob(shape.center, shape.size, shape.rotation)

    def box(self, shape: Box) -> np.ndarray:
        return _sample_box(
            shape.center, shape.size, shape.round_radius, shape.rotation
        )

    def web(self, shape: Web) -> np.ndarray:
        return _sample_web(shape)

    def formed_muscle(
        self, shape: ev.MuscleFormationEvidence
    ) -> np.ndarray:
        return _sample_formed_muscle(shape)


class _BoundsAlgebra:
    def generalized_cylinder(
        self, shape: GeneralizedCylinder
    ) -> tuple[np.ndarray, np.ndarray]:
        radius = shape.radii[:, None]
        return (
            (shape.spine - radius).min(axis=0),
            (shape.spine + radius).max(axis=0),
        )

    def blob(self, shape: Blob) -> tuple[np.ndarray, np.ndarray]:
        extent = _oriented_extent(shape.size, shape.rotation)
        return shape.center - extent, shape.center + extent

    def box(self, shape: Box) -> tuple[np.ndarray, np.ndarray]:
        extent = _oriented_extent(
            shape.size + shape.round_radius, shape.rotation
        )
        return shape.center - extent, shape.center + extent

    def web(self, shape: Web) -> tuple[np.ndarray, np.ndarray]:
        points = np.concatenate(shape.spines, axis=0)
        radii = np.concatenate(shape.radii, axis=0)[:, None]
        return (
            (points - radii).min(axis=0),
            (points + radii).max(axis=0),
        )

    def formed_muscle(
        self, shape: ev.MuscleFormationEvidence
    ) -> tuple[np.ndarray, np.ndarray]:
        origins = np.asarray(
            tuple(frame.origin for frame in shape.frames),
            dtype=np.float64,
        )
        station_radii = np.asarray(
            (
                shape.intervals[0].lower_radius,
                *(interval.upper_radius for interval in shape.intervals),
            ),
            dtype=np.float64,
        )[:, None]
        return (
            np.min(origins - station_radii, axis=0),
            np.max(origins + station_radii, axis=0),
        )


class _MinimumRadiusAlgebra:
    def generalized_cylinder(self, shape: GeneralizedCylinder) -> float:
        return float(np.min(shape.radii))

    def blob(self, shape: Blob) -> float:
        return float(np.min(shape.size))

    def box(self, shape: Box) -> float:
        return float(np.min(shape.size)) + shape.round_radius

    def web(self, shape: Web) -> float:
        return float(min(np.min(radii) for radii in shape.radii))

    def formed_muscle(self, shape: ev.MuscleFormationEvidence) -> float:
        return min(
            min(interval.lower_radius, interval.upper_radius)
            for interval in shape.intervals
        )


class _VolumeCentroidAlgebra:
    def generalized_cylinder(
        self, shape: GeneralizedCylinder
    ) -> tuple[float, np.ndarray]:
        starts = shape.spine[:-1]
        vectors = shape.spine[1:] - starts
        left_radius = shape.radii[:-1]
        right_radius = shape.radii[1:]
        denominator = (
            left_radius**2 + left_radius * right_radius + right_radius**2
        )
        volumes = (
            math.pi
            * np.linalg.norm(vectors, axis=1)
            * denominator
            / 3.0
        )
        fractions = np.where(
            denominator > 0.0,
            (
                left_radius**2
                + 2.0 * left_radius * right_radius
                + 3.0 * right_radius**2
            )
            / (4.0 * np.maximum(denominator, 1.0e-30)),
            0.5,
        )
        centroids = starts + fractions[:, None] * vectors
        total_volume = float(np.sum(volumes))
        centroid = (
            np.sum(volumes[:, None] * centroids, axis=0) / total_volume
            if total_volume > 0.0
            else shape.spine.mean(axis=0)
        )
        return total_volume, centroid

    def blob(self, shape: Blob) -> tuple[float, np.ndarray]:
        return 4.0 / 3.0 * math.pi * float(np.prod(shape.size)), shape.center

    def box(self, shape: Box) -> tuple[float, np.ndarray]:
        return (
            8.0 * float(np.prod(shape.size + shape.round_radius)),
            shape.center,
        )

    def web(self, shape: Web) -> tuple[float, np.ndarray]:
        sections = tuple(
            (
                0.5 * float(np.linalg.norm(np.cross(b - a, c - a))),
                (radius_a + radius_b + radius_c) / 3.0,
                (a + b + c) / 3.0,
            )
            for a, b, c, radius_a, radius_b, radius_c in _web_triangles(shape)
        )
        volumes = np.asarray(
            tuple(2.0 * area * mean_radius for area, mean_radius, _c in sections),
            dtype=np.float64,
        )
        centroids = np.asarray(
            tuple(centroid for _a, _r, centroid in sections), dtype=np.float64
        )
        total_volume = float(np.sum(volumes))
        centroid = (
            np.sum(volumes[:, None] * centroids, axis=0) / total_volume
            if total_volume > 0.0
            else np.concatenate(shape.spines, axis=0).mean(axis=0)
        )
        return total_volume, centroid

    def formed_muscle(
        self, shape: ev.MuscleFormationEvidence
    ) -> tuple[float, np.ndarray]:
        return (
            shape.solved_volume,
            np.asarray(shape.centroid, dtype=np.float64),
        )


_SAMPLE_ALGEBRA = _SampleAlgebra()
_BOUNDS_ALGEBRA = _BoundsAlgebra()
_MINIMUM_RADIUS_ALGEBRA = _MinimumRadiusAlgebra()
_VOLUME_CENTROID_ALGEBRA = _VolumeCentroidAlgebra()


def sample_instance(shape: PartShape) -> np.ndarray:
    return immutable_array(_fold_part_shape(shape, _SAMPLE_ALGEBRA))


def instance_bbox(shape: PartShape) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = _fold_part_shape(shape, _BOUNDS_ALGEBRA)
    return immutable_array(lower), immutable_array(upper)


def part_min_radius(shape: PartShape) -> float:
    return _fold_part_shape(shape, _MINIMUM_RADIUS_ALGEBRA)


def _part_volume_centroid(shape: PartShape) -> tuple[float, np.ndarray]:
    volume, centroid = _fold_part_shape(shape, _VOLUME_CENTROID_ALGEBRA)
    return volume, immutable_array(centroid)


def _oriented_extent(size: np.ndarray, rotation: np.ndarray | None) -> np.ndarray:
    return np.abs(rotation) @ size if rotation is not None else size


def _aabb_gap(loA, hiA, loB, hiB) -> float:
    """Distance between two AABBs (0 if overlapping)."""
    d = np.maximum(np.maximum(loA - hiB, loB - hiA), 0.0)
    return float(np.linalg.norm(d))


def _instance_sdf(points: np.ndarray, part, mirrored: bool) -> np.ndarray:
    return (
        ev.sdf_web(points, part, mirrored)
        if "anchors" in part
        else ev.part_sdf(points, part, mirrored)
    )


def symmetric_gap(da, db) -> float:
    """Symmetric SDF-min gap between two sampled instances, rounded to 3dp:
    ``min`` over each instance's samples read against the other's ``part_sdf``.
    The single value the fusion prefilter and the anomaly quick-gap agree on."""
    gA = _instance_sdf(da.samples, db.part, db.mirrored).min()
    gB = _instance_sdf(db.samples, da.part, da.mirrored).min()
    return round(float(min(gA, gB)), 3)


def field_at(graph, points: np.ndarray) -> np.ndarray:
    return ev.sample_graph_field(graph, points)
