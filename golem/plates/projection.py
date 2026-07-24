"""Derived geometry and receipt projections of an accepted plate surface."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from golem.kernel.engine.compile import vertex_normals
from golem.plates.core import PlateSurface, immutable_array


def seam_geometry(
    surface: PlateSurface,
    final_vertices: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    points = np.asarray(final_vertices, dtype=np.float64)
    normals = vertex_normals(points, surface.faces)
    return (
        immutable_array(points + normals * surface.policy.seam_lift),
        immutable_array(surface.faces[surface.active_seam_faces]),
    )


def plate_surface_report(surface: PlateSurface) -> dict[str, object]:
    nonzero_areas = surface.cell_areas[surface.cell_areas > 0.0]
    area_mean = float(np.mean(nonzero_areas)) if nonzero_areas.size else 0.0
    area_coefficient_of_variation = (
        float(np.std(nonzero_areas) / area_mean) if area_mean > 0.0 else 0.0
    )
    return {
        "layout": surface.policy.layout.value,
        "cells": surface.policy.cell_count,
        "adjacencies": len(surface.adjacency),
        "seam_edges": int(surface.seam_edges.shape[0]),
        "seam_faces": int(np.count_nonzero(surface.seam_faces)),
        "active_seam_faces": int(np.count_nonzero(surface.active_seam_faces)),
        "cell_area_coefficient_of_variation": area_coefficient_of_variation,
    }
