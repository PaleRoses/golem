"""Plate-surface descent from policy and authoritative morphology."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from golem.kernel.anatomy import ClosedVascularGraph
from golem.plates.core import (
    AcceptedPlateSurface,
    PlatePolicy,
    PlateSurface,
    PlateSurfaceObstruction,
    PlateSurfaceRule,
    PlateSurfaceResult,
    RejectedPlateSurface,
)
from golem.plates.geometry import (
    active_seam_faces,
    farthest_point_seed_indices,
    plate_adjacency,
    triangle_areas,
    unique_edges,
    vertex_cells,
)


def derive_plate_surface(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    normals: NDArray[np.float64],
    policy: PlatePolicy,
    *,
    morphology_graph: dict | None = None,
    circulation: ClosedVascularGraph | None = None,
) -> PlateSurfaceResult:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    vertex_normals = np.asarray(normals, dtype=np.float64)
    shape_obstructions = tuple(
        obstruction
        for invalid, obstruction in (
            (
                points.ndim != 2 or points.shape[1:] != (3,),
                PlateSurfaceObstruction(
                    "/",
                    PlateSurfaceRule.VERTEX_ARRAY_SHAPE,
                    tuple(points.shape),
                    {"rank": 2, "column_count": 3},
                ),
            ),
            (
                triangles.ndim != 2 or triangles.shape[1:] != (3,),
                PlateSurfaceObstruction(
                    "/",
                    PlateSurfaceRule.FACE_ARRAY_SHAPE,
                    tuple(triangles.shape),
                    {"rank": 2, "column_count": 3},
                ),
            ),
            (
                vertex_normals.shape != points.shape,
                PlateSurfaceObstruction(
                    "/",
                    PlateSurfaceRule.NORMAL_ARRAY_SHAPE,
                    tuple(vertex_normals.shape),
                    tuple(points.shape),
                ),
            ),
            (
                policy.cell_count > points.shape[0],
                PlateSurfaceObstruction(
                    "/cell_count",
                    PlateSurfaceRule.AVAILABLE_VERTEX_COUNT,
                    policy.cell_count,
                    {"maximum": int(points.shape[0])},
                ),
            ),
        )
        if invalid
    )
    if shape_obstructions:
        return RejectedPlateSurface(shape_obstructions)
    seed_indices = farthest_point_seed_indices(
        points, policy.cell_count, policy.random_seed
    )
    seed_points = points[seed_indices]
    vertex_cells_result = vertex_cells(
        points, seed_points, policy, morphology_graph
    )
    if isinstance(vertex_cells_result, PlateSurfaceObstruction):
        return RejectedPlateSurface((vertex_cells_result,))
    cell_ids = vertex_cells_result
    edges = unique_edges(triangles)
    seam_edges = edges[cell_ids[edges[:, 0]] != cell_ids[edges[:, 1]]]
    seam_vertex_indices = np.unique(seam_edges.reshape(-1))
    seam_vertices = np.isin(
        np.arange(points.shape[0], dtype=np.int64), seam_vertex_indices
    )
    seam_faces = np.any(seam_vertices[triangles], axis=1)
    candidate_seam_faces = np.count_nonzero(seam_vertices[triangles], axis=1) >= 2
    active_faces_result = active_seam_faces(
        points,
        triangles,
        candidate_seam_faces,
        policy,
        circulation,
    )
    if isinstance(active_faces_result, PlateSurfaceObstruction):
        return RejectedPlateSurface((active_faces_result,))
    displacement = np.where(seam_vertices, -policy.groove, policy.relief)
    displaced = points + vertex_normals * displacement[:, None]
    face_cells = cell_ids[triangles[:, 0]]
    cell_areas = np.bincount(
        face_cells,
        weights=triangle_areas(points, triangles),
        minlength=policy.cell_count,
    )
    return AcceptedPlateSurface(
        PlateSurface(
            policy=policy,
            source_vertices=points,
            vertices=displaced,
            faces=triangles,
            seed_indices=seed_indices,
            seed_points=seed_points,
            vertex_cells=cell_ids,
            seam_edges=seam_edges,
            seam_vertices=seam_vertices,
            seam_faces=seam_faces,
            active_seam_faces=active_faces_result,
            cell_areas=cell_areas,
            adjacency=plate_adjacency(cell_ids, seam_edges),
        )
    )
