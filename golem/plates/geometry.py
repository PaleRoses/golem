"""Vectorized geometry algebra for plate derivation and measurement."""

from __future__ import annotations

from functools import reduce

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from golem.kernel.anatomy import ClosedVascularGraph, VascularStratum
from golem.plates.core import (
    PlateAdjacency,
    PlateLayout,
    PlatePolicy,
    PlateSurfaceObstruction,
    PlateSurfaceRule,
    immutable_array,
)


def farthest_point_seed_indices(
    points: NDArray[np.float64], cell_count: int, random_seed: int
) -> NDArray[np.int64]:
    first = int(np.random.default_rng(random_seed).integers(points.shape[0]))
    initial_distance = np.linalg.norm(points - points[first], axis=1)

    def extend(
        state: tuple[tuple[int, ...], NDArray[np.float64]], _step: int
    ) -> tuple[tuple[int, ...], NDArray[np.float64]]:
        indices, minimum_distance = state
        candidate = int(np.argmax(minimum_distance))
        candidate_distance = np.linalg.norm(points - points[candidate], axis=1)
        return (*indices, candidate), np.minimum(
            minimum_distance, candidate_distance
        )

    indices, _ = reduce(
        extend,
        range(1, cell_count),
        ((first,), initial_distance),
    )
    return np.asarray(indices, dtype=np.int64)


def unique_edges(faces: NDArray[np.int64]) -> NDArray[np.int64]:
    directed = np.concatenate(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0
    )
    return np.unique(np.sort(directed, axis=1), axis=0)


def vertex_cells(
    points: NDArray[np.float64],
    seed_points: NDArray[np.float64],
    policy: PlatePolicy,
    morphology_graph: dict | None,
) -> NDArray[np.int64] | PlateSurfaceObstruction:
    if policy.layout is PlateLayout.SURFACE_VORONOI:
        return np.asarray(
            cKDTree(seed_points).query(points, k=1)[1], dtype=np.int64
        )
    if morphology_graph is None:
        return PlateSurfaceObstruction(
            "/layout",
            PlateSurfaceRule.MORPHOLOGY_GRAPH,
            None,
            {"morphology_graph": True},
        )
    axes = nearest_morphology_axes(seed_points, morphology_graph)
    if axes is None:
        return PlateSurfaceObstruction(
            "/layout",
            PlateSurfaceRule.GENERALIZED_CYLINDER_SECTIONS,
            0,
            {"minimum": 1},
        )
    candidate_count = min(12, seed_points.shape[0])
    candidate_indices = np.asarray(
        cKDTree(seed_points).query(points, k=candidate_count)[1],
        dtype=np.int64,
    ).reshape(points.shape[0], candidate_count)
    deltas = points[:, None, :] - seed_points[candidate_indices]
    candidate_axes = axes[candidate_indices]
    axial = np.einsum("nki,nki->nk", deltas, candidate_axes)
    squared = np.einsum("nki,nki->nk", deltas, deltas)
    anisotropic_squared = squared + (policy.axis_weight - 1.0) * axial**2
    choices = np.argmin(anisotropic_squared, axis=1)
    return np.take_along_axis(candidate_indices, choices[:, None], axis=1)[:, 0]


def nearest_morphology_axes(
    points: NDArray[np.float64], graph: dict
) -> NDArray[np.float64] | None:
    segments = tuple(
        (
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
        )
        for part in graph.get("parts", ())
        if part.get("type") == "gencyl"
        for left, right in zip(part["spine"], part["spine"][1:])
    )
    if not segments:
        return None
    starts = np.asarray(tuple(segment[0] for segment in segments))
    vectors = np.asarray(tuple(segment[1] - segment[0] for segment in segments))
    squared_lengths = np.maximum(
        np.einsum("ij,ij->i", vectors, vectors), 1.0e-15
    )
    relative = points[:, None, :] - starts[None, :, :]
    parameters = np.clip(
        np.einsum("nsi,si->ns", relative, vectors)
        / squared_lengths[None, :],
        0.0,
        1.0,
    )
    closest = starts[None, :, :] + parameters[:, :, None] * vectors[None, :, :]
    nearest = np.argmin(
        np.linalg.norm(points[:, None, :] - closest, axis=2), axis=1
    )
    return vectors[nearest] / np.sqrt(squared_lengths[nearest])[:, None]


def active_seam_faces(
    points: NDArray[np.float64],
    faces: NDArray[np.int64],
    candidate_mask: NDArray[np.bool_],
    policy: PlatePolicy,
    circulation: ClosedVascularGraph | None,
) -> NDArray[np.bool_] | PlateSurfaceObstruction:
    if not policy.circulation_emission:
        return candidate_mask
    if circulation is None:
        return PlateSurfaceObstruction(
            "/circulation_emission",
            PlateSurfaceRule.ACCEPTED_CIRCULATION,
            None,
            {"accepted_closed_circulation": True},
        )
    major_edges = tuple(
        sorted(
            (
                edge
                for edge in circulation.edges
                if edge.stratum is VascularStratum.SUPPLY
                and edge.solved_flow > 0.0
            ),
            key=lambda edge: (edge.solved_flow, edge.radius, edge.edge_id),
            reverse=True,
        )[:32]
    )
    candidate_indices = np.flatnonzero(candidate_mask)
    if not major_edges or candidate_indices.size == 0:
        return np.zeros(faces.shape[0], dtype=np.bool_)
    nodes = circulation.node_by_id
    starts = np.asarray(
        tuple(nodes[edge.source_node_id].position for edge in major_edges),
        dtype=np.float64,
    )
    ends = np.asarray(
        tuple(nodes[edge.target_node_id].position for edge in major_edges),
        dtype=np.float64,
    )
    centroids = np.mean(points[faces[candidate_indices]], axis=1)
    minimum_distances = minimum_point_segment_distances(centroids, starts, ends)
    focus_distance = 0.075 * float(np.max(np.ptp(points, axis=0)))
    active_indices = candidate_indices[minimum_distances <= focus_distance]
    return np.isin(np.arange(faces.shape[0], dtype=np.int64), active_indices)


def minimum_point_segment_distances(
    points: NDArray[np.float64],
    starts: NDArray[np.float64],
    ends: NDArray[np.float64],
) -> NDArray[np.float64]:
    vectors = ends - starts
    squared_lengths = np.maximum(
        np.einsum("ij,ij->i", vectors, vectors), 1.0e-15
    )
    relative = points[:, None, :] - starts[None, :, :]
    parameters = np.clip(
        np.einsum("nsi,si->ns", relative, vectors)
        / squared_lengths[None, :],
        0.0,
        1.0,
    )
    closest = starts[None, :, :] + parameters[:, :, None] * vectors[None, :, :]
    return np.min(np.linalg.norm(points[:, None, :] - closest, axis=2), axis=1)


def plate_adjacency(
    vertex_cell_ids: NDArray[np.int64], seam_edges: NDArray[np.int64]
) -> tuple[PlateAdjacency, ...]:
    cell_pairs = np.sort(vertex_cell_ids[seam_edges], axis=1)
    pairs, counts = np.unique(cell_pairs, axis=0, return_counts=True)
    return tuple(
        PlateAdjacency(int(pair[0]), int(pair[1]), int(count))
        for pair, count in zip(pairs, counts)
    )


def triangle_areas(
    vertices: NDArray[np.float64], faces: NDArray[np.int64]
) -> NDArray[np.float64]:
    triangles = vertices[faces]
    return 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
