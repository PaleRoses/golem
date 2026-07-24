"""Ground, centroid, support, and silhouette-band algebra."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from operator import itemgetter

import numpy as np

from golem.senses.model import Balance, Band, NoSupport, Supported
from golem.senses.proprio.geometry import PartShape, _part_volume_centroid

GROUND_Y = 0.02


def compute_ground(parts, inst_data, ground_decl):
    finest_feature = min(datum.min_r for datum in inst_data.values())
    declared = frozenset(ground_decl)
    clearances = {
        part_id: round(float(part.bbox_lo[1]) - GROUND_Y, 3)
        for part_id, part in parts.items()
    }
    return (
        {part_id: value for part_id, value in clearances.items() if part_id in declared},
        {
            part_id: value
            for part_id, value in clearances.items()
            if part_id not in declared and value <= 3.0 * finest_feature
        },
    )


def compute_centroid(
    instances: tuple[tuple[str, dict, bool, PartShape], ...],
) -> np.ndarray:
    volume_centroids = tuple(
        _part_volume_centroid(shape)
        for _address, _part, _mirrored, shape in instances
    )
    if not volume_centroids:
        return np.zeros(3)
    volumes = np.fromiter(map(itemgetter(0), volume_centroids), dtype=np.float64)
    centroids = np.stack(tuple(map(itemgetter(1), volume_centroids)))
    total_volume = float(volumes.sum())
    return volumes @ centroids / total_volume if total_volume > 0.0 else np.zeros(3)


def _balance(inst_data, parts, ground_decl, ground_tol, centroid) -> Balance:
    support = tuple(
        floor[:, (0, 2)]
        for part_id in ground_decl
        if part_id in parts
        for address in parts[part_id].instances
        for samples in (inst_data[address].samples,)
        for floor in (samples[samples[:, 1] <= GROUND_Y + ground_tol + 1.0e-9],)
        if len(floor)
    )
    if not support:
        return NoSupport()
    points = np.concatenate(support)
    lower, upper = points.min(axis=0), points.max(axis=0)
    centroid_x, centroid_z = float(centroid[0]), float(centroid[2])
    return Supported(
        hull_x=(float(lower[0]), float(upper[0])),
        hull_z=(float(lower[1]), float(upper[1])),
        centroid_xz=(centroid_x, centroid_z),
        margin_x=round(min(centroid_x - lower[0], upper[0] - centroid_x), 3),
        margin_z=round(min(centroid_z - lower[1], upper[1] - centroid_z), 3),
        margin_z_fwd=round(upper[1] - centroid_z, 3),
        margin_z_aft=round(centroid_z - lower[1], 3),
        inside=_point_in_hull(points, np.asarray((centroid_x, centroid_z))),
    )


def _point_in_hull(points_xz: np.ndarray, point: np.ndarray) -> bool:
    try:
        from scipy.spatial import ConvexHull

        equations = ConvexHull(points_xz).equations
        return bool(np.all(equations[:, :2] @ point + equations[:, 2] <= 1.0e-9))
    except Exception:
        lower, upper = points_xz.min(axis=0), points_xz.max(axis=0)
        return bool(
            np.all(point >= lower - 1.0e-9) and np.all(point <= upper + 1.0e-9)
        )


def _bands(parts, raw_lo, raw_hi, n_bands: int = 10) -> list[Band]:
    lower = np.asarray(tuple(part.bbox_lo for part in parts.values())).reshape((-1, 3))
    upper = np.asarray(tuple(part.bbox_hi for part in parts.values())).reshape((-1, 3))
    edges = np.linspace(float(raw_lo[1]), float(raw_hi[1]), n_bands + 1)
    return [
        _band(lower, upper, float(y_lower), float(y_upper))
        for y_lower, y_upper in pairwise(edges)
    ]


def _band(
    lower: np.ndarray,
    upper: np.ndarray,
    y_lower: float,
    y_upper: float,
) -> Band:
    overlaps = (upper[:, 1] >= y_lower) & (lower[:, 1] <= y_upper)
    x_intervals = np.column_stack((lower[overlaps, 0], upper[overlaps, 0]))
    z_intervals = np.column_stack((lower[overlaps, 2], upper[overlaps, 2]))
    return Band(
        ylo=y_lower,
        yhi=y_upper,
        width=_union_len(x_intervals),
        depth=_union_len(z_intervals),
        zmid=_union_mid(z_intervals),
    )


def _union_len(intervals: Sequence[tuple[float, float]] | np.ndarray) -> float:
    values = np.asarray(intervals, dtype=np.float64).reshape((-1, 2))
    if not values.size:
        return 0.0
    ordered = values[np.lexsort((values[:, 1], values[:, 0]))]
    running_upper = np.maximum.accumulate(ordered[:, 1])
    starts = np.concatenate(((True,), ordered[1:, 0] > running_upper[:-1]))
    start_indices = np.flatnonzero(starts)
    end_indices = np.concatenate((start_indices[1:] - 1, (len(ordered) - 1,)))
    return float(np.sum(running_upper[end_indices] - ordered[start_indices, 0]))


def _union_mid(intervals: Sequence[tuple[float, float]] | np.ndarray) -> float:
    values = np.asarray(intervals, dtype=np.float64).reshape((-1, 2))
    return (
        float((values[:, 0].min() + values[:, 1].max()) / 2.0)
        if values.size
        else 0.0
    )
