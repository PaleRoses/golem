"""Immutable carriers for the derived plate surface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class PlateLayout(StrEnum):
    SURFACE_VORONOI = "surface_voronoi"
    MORPHOLOGY_ANISOTROPIC = "morphology_anisotropic"


class PlatePolicyRule(StrEnum):
    POLICY_OBJECT = "policy_object"
    LAYOUT = "layout"
    CELL_COUNT = "cell_count"
    RANDOM_SEED = "random_seed"
    FINITE_NON_NEGATIVE = "finite_non_negative"
    APPEARANCE_MATERIAL = "appearance_material"
    FINITE_POSITIVE = "finite_positive"
    BOOLEAN = "boolean"


class PlateSurfaceRule(StrEnum):
    VERTEX_ARRAY_SHAPE = "vertex_array_shape"
    FACE_ARRAY_SHAPE = "face_array_shape"
    NORMAL_ARRAY_SHAPE = "normal_array_shape"
    AVAILABLE_VERTEX_COUNT = "available_vertex_count"
    MORPHOLOGY_GRAPH = "morphology_graph"
    GENERALIZED_CYLINDER_SECTIONS = "generalized_cylinder_sections"
    ACCEPTED_CIRCULATION = "accepted_circulation"


@dataclass(frozen=True)
class PlatePolicy:
    layout: PlateLayout
    cell_count: int
    random_seed: int
    relief: float
    groove: float
    seam_lift: float
    seam_material: str
    axis_weight: float
    circulation_emission: bool


@dataclass(frozen=True)
class PlatePolicyObstruction:
    address: str
    rule: PlatePolicyRule
    authored: object
    required: object


@dataclass(frozen=True)
class AcceptedPlatePolicy:
    policy: PlatePolicy


@dataclass(frozen=True)
class RejectedPlatePolicy:
    obstructions: tuple[PlatePolicyObstruction, ...]


type PlatePolicyResult = AcceptedPlatePolicy | RejectedPlatePolicy


@dataclass(frozen=True)
class PlateAdjacency:
    left_cell: int
    right_cell: int
    seam_edge_count: int


@dataclass(frozen=True)
class PlateSurface:
    policy: PlatePolicy
    source_vertices: NDArray[np.float64]
    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]
    seed_indices: NDArray[np.int64]
    seed_points: NDArray[np.float64]
    vertex_cells: NDArray[np.int64]
    seam_edges: NDArray[np.int64]
    seam_vertices: NDArray[np.bool_]
    seam_faces: NDArray[np.bool_]
    active_seam_faces: NDArray[np.bool_]
    cell_areas: NDArray[np.float64]
    adjacency: tuple[PlateAdjacency, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_vertices", immutable_array(self.source_vertices))
        object.__setattr__(self, "vertices", immutable_array(self.vertices))
        object.__setattr__(self, "faces", immutable_array(self.faces))
        object.__setattr__(self, "seed_indices", immutable_array(self.seed_indices))
        object.__setattr__(self, "seed_points", immutable_array(self.seed_points))
        object.__setattr__(self, "vertex_cells", immutable_array(self.vertex_cells))
        object.__setattr__(self, "seam_edges", immutable_array(self.seam_edges))
        object.__setattr__(self, "seam_vertices", immutable_array(self.seam_vertices))
        object.__setattr__(self, "seam_faces", immutable_array(self.seam_faces))
        object.__setattr__(
            self, "active_seam_faces", immutable_array(self.active_seam_faces)
        )
        object.__setattr__(self, "cell_areas", immutable_array(self.cell_areas))
        object.__setattr__(self, "adjacency", tuple(self.adjacency))


@dataclass(frozen=True)
class PlateSurfaceObstruction:
    address: str
    rule: PlateSurfaceRule
    authored: object
    required: object


@dataclass(frozen=True)
class AcceptedPlateSurface:
    surface: PlateSurface


@dataclass(frozen=True)
class RejectedPlateSurface:
    obstructions: tuple[PlateSurfaceObstruction, ...]


type PlateSurfaceResult = AcceptedPlateSurface | RejectedPlateSurface


def immutable_array[value: np.generic](
    value: NDArray[value],
) -> NDArray[value]:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result
