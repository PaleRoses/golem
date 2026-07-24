"""Diagnostic tube geometry derived from accepted vascular graphs."""

from __future__ import annotations

import numpy as np

from golem.kernel.anatomy import ClosedVascularGraph, VascularStratum


def vascular_stratum_mesh(
    graph: ClosedVascularGraph,
    stratum: VascularStratum,
    *,
    sides: int = 6,
    visual_radius_floor: float = 1.2e-3,
) -> tuple[np.ndarray, np.ndarray]:
    node_by_id = graph.node_by_id
    pieces = tuple(
        _vascular_edge_tube(
            np.asarray(node_by_id[edge.source_node_id].position),
            np.asarray(node_by_id[edge.target_node_id].position),
            max(edge.radius, visual_radius_floor),
            sides,
        )
        for edge in graph.edges
        if edge.stratum is stratum
    )
    if not pieces:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
        )
    offsets = np.cumsum((0, *tuple(piece[0].shape[0] for piece in pieces[:-1])))
    return (
        np.concatenate(tuple(piece[0] for piece in pieces), axis=0),
        np.concatenate(
            tuple(
                piece[1] + offset
                for piece, offset in zip(pieces, offsets)
            ),
            axis=0,
        ),
    )


def _vascular_edge_tube(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    sides: int,
) -> tuple[np.ndarray, np.ndarray]:
    tangent = end - start
    tangent = tangent / max(float(np.linalg.norm(tangent)), 1.0e-15)
    z_axis = np.array([0.0, 0.0, 1.0])
    reference = (
        z_axis
        if abs(float(tangent @ z_axis)) < 0.99
        else np.array([1.0, 0.0, 0.0])
    )
    normal = np.cross(reference, tangent)
    normal = normal / float(np.linalg.norm(normal))
    binormal = np.cross(tangent, normal)
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    radial = radius * (
        np.cos(angles)[:, None] * normal
        + np.sin(angles)[:, None] * binormal
    )
    vertices = np.concatenate(
        (start + radial, end + radial, start[None, :], end[None, :]), axis=0
    )
    faces = np.asarray(
        tuple(
            face
            for index in range(sides)
            for successor in ((index + 1) % sides,)
            for face in (
                (index, successor, sides + index),
                (successor, sides + successor, sides + index),
                (successor, index, 2 * sides),
                (sides + index, sides + successor, 2 * sides + 1),
            )
        ),
        dtype=np.int64,
    )
    return vertices, faces


__all__ = ["vascular_stratum_mesh"]
