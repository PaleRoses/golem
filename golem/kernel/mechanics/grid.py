"""Structured-grid geometry: node activation, neighbours, corners, flat ids.

``CellIndex`` bounds are decided solely by ``StructuredHexDomain.cell_dims``
(a :class:`GridDims`) through :func:`_cell_is_in_bounds`.  There is no flat-id
projection of the geometric triple; the string-id/DOF boundary stays elsewhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .element import (
    _CORNER_OFFSET_TUPLES,
    _CORNER_OFFSETS,
    _FACE_CORNER_INDICES,
    _FACE_NEIGHBOR_OFFSETS,
)
from .model import (
    BoundaryFace,
    CellIndex,
    InternalPressureFace,
    NodeIndex,
    StructuredHexDomain,
    Vector3,
)

if TYPE_CHECKING:
    from .discretize import _Discretization


def _active_nodes(cells: tuple[CellIndex, ...]) -> frozenset[NodeIndex]:
    return frozenset(
        NodeIndex(
            cell.x_index + int(offset[0]),
            cell.y_index + int(offset[1]),
            cell.z_index + int(offset[2]),
        )
        for cell in cells
        for offset in _CORNER_OFFSET_TUPLES
    )


def _cell_is_in_bounds(domain: StructuredHexDomain, cell: CellIndex) -> bool:
    return domain.cell_dims.contains(cell.x_index, cell.y_index, cell.z_index)


def _neighbor_cell(cell: CellIndex, face: BoundaryFace) -> CellIndex:
    dx, dy, dz = _FACE_NEIGHBOR_OFFSETS[face]
    return CellIndex(cell.x_index + dx, cell.y_index + dy, cell.z_index + dz)


def _active_face_nodes(
    domain: StructuredHexDomain,
    discretization: _Discretization,
    faces: tuple[InternalPressureFace, ...],
) -> NDArray[np.int64]:
    face_nodes = np.asarray(
        tuple(
            tuple(
                _cell_corner_node(pressure.cell, corner_index)
                for corner_index in _FACE_CORNER_INDICES[pressure.face]
            )
            for pressure in faces
        ),
        dtype=np.int64,
    )
    return np.searchsorted(
        discretization.active_node_flat_ids,
        _flat_node_ids(domain, face_nodes).reshape(-1),
    ).reshape((-1, 4))


def _flat_node_ids(
    domain: StructuredHexDomain, nodes: NDArray[np.int64]
) -> NDArray[np.int64]:
    return np.ravel_multi_index(
        np.moveaxis(nodes, -1, 0),
        (
            len(domain.x_coordinates),
            len(domain.y_coordinates),
            len(domain.z_coordinates),
        ),
    )


def _cell_corner_node(cell: CellIndex, corner_index: int) -> tuple[int, int, int]:
    offset = _CORNER_OFFSETS[corner_index]
    return (
        cell.x_index + int(offset[0]),
        cell.y_index + int(offset[1]),
        cell.z_index + int(offset[2]),
    )


def _cell_size(domain: StructuredHexDomain, cell: CellIndex) -> Vector3:
    return tuple(
        float(coordinates[index + 1] - coordinates[index])
        for coordinates, index in zip(
            (domain.x_coordinates, domain.y_coordinates, domain.z_coordinates),
            (cell.x_index, cell.y_index, cell.z_index),
        )
    )
