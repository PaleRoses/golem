"""Voxel material-domain geometry: resolved domains, cell/axis indexing, adjacency."""

from __future__ import annotations

import math
import numpy as np

from collections.abc import Callable
from dataclasses import dataclass
from golem.kernel import engine as E
from golem.kernel.mechanics import (BoundaryFace, CellIndex, DomainSource, NodeIndex, StructuredHexDomain)
from itertools import product

from golem.assembly.carriers import _CompiledElement
from golem.assembly.obstructions import EmptyPhysicalSolidDomainObstruction

@dataclass(frozen=True)
class _VoxelAdjacency:
    left_cell: CellIndex
    right_cell: CellIndex
    axis: int
    exchange_area_square_metres: float
    path_length_metres: float


@dataclass(frozen=True)
class _ResolvedMaterialDomain:
    element_id: str
    mechanics_domain: StructuredHexDomain
    cell_centers_world: tuple[tuple[float, float, float], ...]
    wall_cells: frozenset[CellIndex]
    lumen_cells: frozenset[CellIndex]
    adjacencies: tuple[_VoxelAdjacency, ...]

    @property
    def cell_id_by_index(self) -> dict[CellIndex, int]:
        return {
            cell: index
            for index, cell in enumerate(self.mechanics_domain.solid_cells)
        }


@dataclass(frozen=True)
class _WallBinding:
    edge_id: str
    solid_cell: CellIndex
    coolant_node_id: str


def _voxel_axis_edges(
    lower: float,
    upper: float,
    resolution: int,
    metres_per_world_unit: float,
) -> tuple[float, ...]:
    centers = np.linspace(lower, upper, resolution, dtype=np.float64)
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    edges = np.concatenate(
        (
            np.asarray((centers[0] - 0.5 * (centers[1] - centers[0]),)),
            midpoints,
            np.asarray((centers[-1] + 0.5 * (centers[-1] - centers[-2]),)),
        )
    )
    return tuple(map(float, metres_per_world_unit * edges))


def _mask_cell_indices(mask: np.ndarray) -> tuple[CellIndex, ...]:
    return tuple(
        CellIndex(*tuple(map(int, row))) for row in np.argwhere(mask)
    )


def _neighbor_cell(cell: CellIndex, axis: int, step: int = 1) -> CellIndex:
    coordinates = (cell.x_index, cell.y_index, cell.z_index)
    shifted = tuple(
        coordinate + (step if index == axis else 0)
        for index, coordinate in enumerate(coordinates)
    )
    return CellIndex(*shifted)


def _domain_axes(
    domain: StructuredHexDomain,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    return (
        domain.x_coordinates,
        domain.y_coordinates,
        domain.z_coordinates,
    )


def _cell_ijk(cell: CellIndex | NodeIndex) -> tuple[int, int, int]:
    return (cell.x_index, cell.y_index, cell.z_index)


def _cell_center_metres(
    domain: StructuredHexDomain, cell: CellIndex
) -> tuple[float, float, float]:
    axes = _domain_axes(domain)
    indices = _cell_ijk(cell)
    return tuple(
        0.5 * (axis[index] + axis[index + 1])
        for axis, index in zip(axes, indices)
    )


def _voxel_adjacency(
    domain: StructuredHexDomain, left_cell: CellIndex, axis: int
) -> _VoxelAdjacency:
    right_cell = _neighbor_cell(left_cell, axis)
    axes = _domain_axes(domain)
    indices = _cell_ijk(left_cell)
    lengths = tuple(
        coordinate_axis[index + 1] - coordinate_axis[index]
        for coordinate_axis, index in zip(axes, indices)
    )
    left_center = _cell_center_metres(domain, left_cell)
    right_center = _cell_center_metres(domain, right_cell)
    return _VoxelAdjacency(
        left_cell=left_cell,
        right_cell=right_cell,
        axis=axis,
        exchange_area_square_metres=math.prod(
            length for index, length in enumerate(lengths) if index != axis
        ),
        path_length_metres=abs(right_center[axis] - left_center[axis]),
    )


def _derive_resolved_material_domain(
    element: _CompiledElement,
    load_bearing_solid: np.ndarray,
    lumen: np.ndarray,
    wall: np.ndarray,
    metres_per_world_unit: float,
) -> _ResolvedMaterialDomain | EmptyPhysicalSolidDomainObstruction:
    evaluated = element.evaluated
    solid_cells = _mask_cell_indices(load_bearing_solid)
    if not solid_cells:
        return EmptyPhysicalSolidDomainObstruction(element.element_id)
    mechanics_domain = StructuredHexDomain(
        x_coordinates=_voxel_axis_edges(
            evaluated.lower[0],
            evaluated.upper[0],
            evaluated.resolution,
            metres_per_world_unit,
        ),
        y_coordinates=_voxel_axis_edges(
            evaluated.lower[1],
            evaluated.upper[1],
            evaluated.resolution,
            metres_per_world_unit,
        ),
        z_coordinates=_voxel_axis_edges(
            evaluated.lower[2],
            evaluated.upper[2],
            evaluated.resolution,
            metres_per_world_unit,
        ),
        solid_cells=solid_cells,
        source=DomainSource.SDF_DERIVED,
    )
    solid_set = frozenset(solid_cells)
    adjacencies = tuple(
        _voxel_adjacency(mechanics_domain, cell, axis)
        for cell in solid_cells
        for axis in range(3)
        if _neighbor_cell(cell, axis) in solid_set
    )
    scale = metres_per_world_unit
    return _ResolvedMaterialDomain(
        element_id=element.element_id,
        mechanics_domain=mechanics_domain,
        cell_centers_world=tuple(
            tuple(coordinate / scale for coordinate in _cell_center_metres(mechanics_domain, cell))
            for cell in solid_cells
        ),
        wall_cells=frozenset(_mask_cell_indices(wall)),
        lumen_cells=frozenset(_mask_cell_indices(lumen)),
        adjacencies=adjacencies,
    )


def _unperforated_material_masks(
    evaluated: E.EvaluatedMorphology,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    solid = np.asarray(evaluated.field <= 0.0, dtype=np.bool_)
    empty = np.zeros_like(solid, dtype=np.bool_)
    return solid, empty, empty


def _boundary_face_area_square_metres(
    domain: StructuredHexDomain, cell: CellIndex, axis: int
) -> float:
    axes = _domain_axes(domain)
    indices = _cell_ijk(cell)
    lengths = tuple(
        coordinate_axis[index + 1] - coordinate_axis[index]
        for coordinate_axis, index in zip(axes, indices)
    )
    return math.prod(
        length for dimension, length in enumerate(lengths) if dimension != axis
    )


def _faces_where(
    domain: _ResolvedMaterialDomain,
    predicate: Callable[[CellIndex, int, int], bool],
) -> tuple[tuple[CellIndex, BoundaryFace, int, int], ...]:
    return tuple(
        (cell, boundary_face, axis, direction)
        for cell in domain.mechanics_domain.solid_cells
        for boundary_face, axis, direction in _pressure_face_specs()
        if predicate(cell, axis, direction)
    )


def _surface_face_specs(
    domain: _ResolvedMaterialDomain,
) -> tuple[tuple[CellIndex, BoundaryFace, int, float], ...]:
    solid_set = frozenset(domain.mechanics_domain.solid_cells)
    return tuple(
        (
            cell,
            boundary_face,
            axis,
            _boundary_face_area_square_metres(
                domain.mechanics_domain, cell, axis
            ),
        )
        for cell, boundary_face, axis, _direction in _faces_where(
            domain,
            lambda cell, axis, direction: (
                _neighbor_cell(cell, axis, direction) not in solid_set
            ),
        )
    )


def _nearest_domain_cells(
    domain: _ResolvedMaterialDomain,
    points_world: tuple[tuple[float, float, float], ...],
) -> tuple[CellIndex, ...]:
    centers = np.asarray(domain.cell_centers_world, dtype=np.float64)
    return tuple(
        domain.mechanics_domain.solid_cells[
            int(
                np.argmin(
                    np.sum(
                        np.square(centers - np.asarray(point, dtype=np.float64)),
                        axis=1,
                    )
                )
            )
        ]
        for point in points_world
    )


def _cell_nodes(cell: CellIndex) -> tuple[NodeIndex, ...]:
    return tuple(
        NodeIndex(
            cell.x_index + x_offset,
            cell.y_index + y_offset,
            cell.z_index + z_offset,
        )
        for x_offset, y_offset, z_offset in product((0, 1), repeat=3)
    )


def _active_domain_nodes(domain: _ResolvedMaterialDomain) -> tuple[NodeIndex, ...]:
    return tuple(
        dict.fromkeys(
            node
            for cell in domain.mechanics_domain.solid_cells
            for node in _cell_nodes(cell)
        )
    )


def _node_position_metres(
    domain: _ResolvedMaterialDomain, node: NodeIndex
) -> np.ndarray:
    mechanics_domain = domain.mechanics_domain
    axes = _domain_axes(mechanics_domain)
    indices = _cell_ijk(node)
    return np.asarray(
        tuple(axis[index] for axis, index in zip(axes, indices)),
        dtype=np.float64,
    )


def _nearest_domain_node(
    domain: _ResolvedMaterialDomain, point_world: tuple[float, float, float], scale: float
) -> NodeIndex:
    nodes = _active_domain_nodes(domain)
    point_metres = scale * np.asarray(point_world, dtype=np.float64)
    positions = np.asarray(
        tuple(_node_position_metres(domain, node) for node in nodes)
    )
    return nodes[
        int(np.argmin(np.sum(np.square(positions - point_metres), axis=1)))
    ]


def _evaluated_solid_centers_world(
    evaluated: E.EvaluatedMorphology,
) -> np.ndarray:
    indices = np.argwhere(evaluated.field <= 0.0)
    axes = tuple(
        np.linspace(lower, upper, evaluated.resolution, dtype=np.float64)
        for lower, upper in zip(evaluated.lower, evaluated.upper)
    )
    return np.column_stack(
        tuple(axis[indices[:, dimension]] for dimension, axis in enumerate(axes))
    )


def _pressure_face_specs() -> tuple[tuple[BoundaryFace, int, int], ...]:
    return (
        (BoundaryFace.X_MIN, 0, -1),
        (BoundaryFace.X_MAX, 0, 1),
        (BoundaryFace.Y_MIN, 1, -1),
        (BoundaryFace.Y_MAX, 1, 1),
        (BoundaryFace.Z_MIN, 2, -1),
        (BoundaryFace.Z_MAX, 2, 1),
    )
