"""Coalgebra: unfold a structured domain into the active-node/cell arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .element import _CORNER_OFFSETS
from .grid import _flat_node_ids
from .model import NodeIndex, StructuredHexDomain


@dataclass(frozen=True)
class _Discretization:
    cells: NDArray[np.int64]
    cell_nodes: NDArray[np.int64]
    active_node_flat_ids: NDArray[np.int64]
    node_indices: NDArray[np.int64]
    node_positions: NDArray[np.float64]
    cell_sizes: NDArray[np.float64]
    cell_centers: NDArray[np.float64]

    @property
    def degree_count(self) -> int:
        return len(self.node_indices) * 3


def _discretize(domain: StructuredHexDomain) -> _Discretization:
    cells = np.asarray(
        tuple(
            (cell.x_index, cell.y_index, cell.z_index)
            for cell in domain.solid_cells
        ),
        dtype=np.int64,
    )
    lattice_nodes = cells[:, None, :] + _CORNER_OFFSETS[None, :, :]
    shape = (
        len(domain.x_coordinates),
        len(domain.y_coordinates),
        len(domain.z_coordinates),
    )
    flat_lattice_nodes = np.ravel_multi_index(
        np.moveaxis(lattice_nodes, -1, 0), shape
    )
    active_node_flat_ids, inverse = np.unique(
        flat_lattice_nodes.reshape(-1), return_inverse=True
    )
    node_indices = np.column_stack(
        np.unravel_index(active_node_flat_ids, shape)
    ).astype(np.int64)
    axes = tuple(
        np.asarray(coordinates, dtype=np.float64)
        for coordinates in (
            domain.x_coordinates,
            domain.y_coordinates,
            domain.z_coordinates,
        )
    )
    node_positions = np.column_stack(
        tuple(axis[node_indices[:, component]] for component, axis in enumerate(axes))
    )
    cell_sizes = np.column_stack(
        tuple(
            axis[cells[:, component] + 1] - axis[cells[:, component]]
            for component, axis in enumerate(axes)
        )
    )
    cell_centers = np.column_stack(
        tuple(
            0.5 * (axis[cells[:, component] + 1] + axis[cells[:, component]])
            for component, axis in enumerate(axes)
        )
    )
    return _Discretization(
        cells=cells,
        cell_nodes=inverse.reshape((-1, 8)),
        active_node_flat_ids=active_node_flat_ids,
        node_indices=node_indices,
        node_positions=node_positions,
        cell_sizes=cell_sizes,
        cell_centers=cell_centers,
    )


def _active_node_position(
    domain: StructuredHexDomain,
    discretization: _Discretization,
    node: NodeIndex,
) -> int:
    return int(
        np.searchsorted(
            discretization.active_node_flat_ids,
            _flat_node_ids(
                domain,
                np.asarray(((node.x_index, node.y_index, node.z_index),)),
            )[0],
        )
    )
