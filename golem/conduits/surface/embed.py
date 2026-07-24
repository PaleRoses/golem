"""Analytic embeddings from authored part stations onto compiled surfaces."""

from __future__ import annotations

import numpy as np

from golem.kernel import engine


def part_station(part: dict, t: float):
    if part["type"] != "gencyl":
        center = np.asarray(part["center"], dtype=np.float64)
        return center, np.array([0.0, 1.0, 0.0]), float(np.max(part["size"]))
    spine = np.asarray(part["spine"], dtype=np.float64)
    radii = np.asarray(part["radii"], dtype=np.float64)
    segments = spine[1:] - spine[:-1]
    segment_lengths = np.linalg.norm(segments, axis=1)
    total_length = float(segment_lengths.sum())
    if total_length < 1.0e-12:
        return spine[0], np.array([0.0, 1.0, 0.0]), float(radii[0])
    target = float(np.clip(t, 0.0, 1.0)) * total_length
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    segment_index = int(
        np.clip(
            np.searchsorted(cumulative_lengths, target, side="right") - 1,
            0,
            len(segments) - 1,
        )
    )
    segment_fraction = (
        target - cumulative_lengths[segment_index]
    ) / max(segment_lengths[segment_index], 1.0e-12)
    point = spine[segment_index] + segments[segment_index] * segment_fraction
    tangent = segments[segment_index] / max(
        segment_lengths[segment_index], 1.0e-12
    )
    radius = radii[segment_index] + (
        radii[segment_index + 1] - radii[segment_index]
    ) * segment_fraction
    return point, tangent, float(radius)


def _near_part(part: dict, vertices: np.ndarray, tolerance: float) -> np.ndarray:
    return np.abs(engine.part_sdf(vertices, part)) < tolerance


def _part_anchor(part: dict) -> np.ndarray:
    return (
        np.asarray(part["spine"], dtype=np.float64).mean(axis=0)
        if part["type"] == "gencyl"
        else np.asarray(part["center"], dtype=np.float64)
    )


def _normal_projected_part_mask(
    part: dict,
    vertices: np.ndarray,
    face_normal: object,
    tolerance: float,
) -> np.ndarray:
    authored_normal = np.asarray(face_normal, dtype=np.float64)
    magnitude = float(np.linalg.norm(authored_normal))
    if not np.isfinite(magnitude) or magnitude < 1.0e-12:
        return np.zeros(vertices.shape[0], dtype=bool)
    normal = authored_normal / magnitude
    anchor = _part_anchor(part)
    normal_offsets = (vertices - anchor) @ normal
    projected_vertices = vertices - normal_offsets[:, None] * normal
    return (
        (engine.part_sdf(projected_vertices, part) <= tolerance)
        & (normal_offsets >= -tolerance)
    )


def axial_loop(
    part: dict, vertices: np.ndarray, station: float, sdf_tolerance: float
):
    point, tangent, _radius = part_station(part, station)
    signed_distance = (vertices - point) @ tangent
    return _near_part(part, vertices, sdf_tolerance), signed_distance


def face_line(
    part: dict,
    vertices: np.ndarray,
    axis: int,
    sdf_tolerance: float,
    face_normal: object,
):
    midpoint = (
        float(np.mean(np.asarray(part["spine"], dtype=np.float64)[:, axis]))
        if part["type"] == "gencyl"
        else float(part["center"][axis])
    )
    signed_distance = vertices[:, axis] - midpoint
    host_surface = _near_part(part, vertices, sdf_tolerance)
    near = (
        host_surface
        if host_surface.any() or face_normal is None
        else _normal_projected_part_mask(
            part,
            vertices,
            face_normal,
            sdf_tolerance,
        )
    )
    return near, signed_distance


__all__ = ["axial_loop", "face_line", "part_station"]
