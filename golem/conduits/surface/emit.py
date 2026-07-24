"""Bounded groove displacement and surface-band extraction."""

from __future__ import annotations

import numpy as np

from golem.kernel.engine.compile import vertex_normals as _vertex_normals


DEFAULT_BAND_LIFT = 0.004


def groove(
    vertices: np.ndarray,
    faces: np.ndarray,
    mask: np.ndarray,
    signed_distance: np.ndarray,
    width: float,
    depth: float,
) -> np.ndarray:
    if not mask.any():
        return np.array(vertices, copy=True)
    normals = _vertex_normals(vertices, faces)
    normalized_distance = np.clip(
        np.abs(signed_distance) / (width / 2.0), 0.0, 1.0
    )
    profile = 0.5 * (1.0 + np.cos(np.pi * normalized_distance))
    displacement = (
        normals * (depth * profile * mask.astype(np.float64))[:, None]
    )
    return vertices - displacement


def band_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    signed_distance: np.ndarray,
    width: float,
    near_mask: np.ndarray,
    *,
    normals: np.ndarray | None = None,
    face_normal=None,
    normal_min_dot: float = 0.3,
    lift: float = DEFAULT_BAND_LIFT,
):
    face_distance = np.abs(signed_distance[faces].mean(axis=1))
    face_mask = (face_distance < width / 2.0) & (
        near_mask[faces].sum(axis=1) >= 2
    )
    if face_normal is not None:
        triangles = vertices[faces]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        face_normals = face_normals / np.maximum(
            np.linalg.norm(face_normals, axis=1, keepdims=True), 1.0e-12
        )
        face_mask = face_mask & (
            (face_normals @ np.asarray(face_normal, dtype=np.float64))
            > normal_min_dot
        )
    if not face_mask.any():
        return None, None
    shared_normals = (
        _vertex_normals(vertices, faces) if normals is None else normals
    )
    selected_faces = faces[face_mask]
    used_vertices = np.unique(selected_faces)
    band_vertices = (
        vertices[used_vertices] + shared_normals[used_vertices] * lift
    )
    return band_vertices, np.searchsorted(used_vertices, selected_faces)


__all__ = ["DEFAULT_BAND_LIFT", "band_faces", "groove"]
