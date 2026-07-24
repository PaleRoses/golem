"""Pure metric folds over one lowered triangle mesh."""

from __future__ import annotations

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import cKDTree

import judge


PITCH: float = judge.PITCH


def legacy_symmetry_iou(mesh: trimesh.Trimesh) -> float:
    _, voxel_grid = judge.occupancy(mesh)
    filled = ndimage.binary_fill_holes(np.asarray(voxel_grid.matrix, dtype=bool))
    voxel_indices = tuple(
        np.round((voxel_grid.points - voxel_grid.translation) / PITCH).astype(int).T
    )
    points = voxel_grid.points[filled[voxel_indices]]
    occupied = np.round(points / PITCH).astype(np.int64)
    return _reflection_iou(occupied, x_offset=0)


def _aligned_occupancy(
    mesh: trimesh.Trimesh, pitch: float
) -> NDArray[np.int64]:
    shifted = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices)
        + np.asarray((pitch / 2.0, 0.0, 0.0)),
        faces=np.asarray(mesh.faces),
        process=False,
    )
    voxel_grid = shifted.voxelized(pitch=pitch)
    filled = ndimage.binary_fill_holes(np.asarray(voxel_grid.matrix, dtype=bool))
    indices = np.argwhere(filled)
    centers = voxel_grid.indices_to_points(indices.astype(float)) - np.asarray(
        (pitch / 2.0, 0.0, 0.0)
    )
    return np.round(
        np.column_stack((centers[:, 0] / pitch - 0.5, centers[:, 1:] / pitch))
    ).astype(np.int64)


def aligned_symmetry_iou(
    mesh: trimesh.Trimesh, pitch: float = PITCH
) -> float:
    occupied = _aligned_occupancy(mesh, pitch)
    return _reflection_iou(occupied, x_offset=-1)


def _reflection_iou(occupied: NDArray[np.int64], *, x_offset: int) -> float:
    if not occupied.size:
        return 0.0
    mirrored = (
        occupied * np.asarray((-1, 1, 1), dtype=np.int64)
        + np.asarray((x_offset, 0, 0), dtype=np.int64)
    )
    combined = np.concatenate((occupied, mirrored))
    normalized = combined - np.min(combined, axis=0)
    dimensions = tuple((np.max(normalized, axis=0) + 1).tolist())
    lattice_keys = np.ravel_multi_index(normalized.T, dimensions)
    occupied_count = int(np.unique(lattice_keys[: len(occupied)]).size)
    union_count = int(np.unique(lattice_keys).size)
    return (2 * occupied_count - union_count) / union_count


def mesh_symmetry_distance(
    mesh: trimesh.Trimesh,
    samples: int = 40000,
    tol_frac: float = 0.0075,
    seed: int = 0,
) -> dict[str, float]:
    diagonal = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    points, _ = trimesh.sample.sample_surface(mesh, samples, seed=seed)
    reflected = points * np.asarray((-1.0, 1.0, 1.0))
    distance, _ = cKDTree(points).query(reflected)
    fraction = distance / diagonal
    return {
        "within_frac": float((fraction < tol_frac).mean()),
        "rms": float(np.sqrt(np.mean(fraction**2))),
        "max": float(fraction.max()),
    }
