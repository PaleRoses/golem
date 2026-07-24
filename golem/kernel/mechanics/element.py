"""Reference trilinear-hex algebra: shape functions, quadrature, constitutive.

The lowest structured-element layer.  Pure numpy over reference-cell geometry
with no dependency on the result vocabulary; the symmetric-tensor packers that
would pull in ``results`` live in ``verdict`` instead.
"""

from __future__ import annotations

from itertools import product
from math import sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .model import (
    BoundaryFace,
    IsotropicConstitutiveProperties,
    Vector3,
)


_CORNER_OFFSETS = np.asarray(
    (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ),
    dtype=np.int64,
)
_CORNER_OFFSET_TUPLES = tuple(map(tuple, _CORNER_OFFSETS.tolist()))
_REFERENCE_NODE_SIGNS = 2.0 * _CORNER_OFFSETS.astype(np.float64) - 1.0
_GAUSS_ABSCISSA = 1.0 / sqrt(3.0)
_GAUSS_POINTS = np.asarray(
    tuple(product((-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA), repeat=3)),
    dtype=np.float64,
)
_FACE_GAUSS_POINTS = np.asarray(
    tuple(product((-_GAUSS_ABSCISSA, _GAUSS_ABSCISSA), repeat=2)),
    dtype=np.float64,
)
_FACE_REFERENCE_NODE_SIGNS = np.asarray(
    ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),
    dtype=np.float64,
)
_THERMAL_STRAIN_DIRECTION = np.asarray((1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
_LAMBDA_CONSTITUTIVE_BASIS = np.pad(np.ones((3, 3)), ((0, 3), (0, 3)))
_MU_CONSTITUTIVE_BASIS = np.diag((2.0, 2.0, 2.0, 1.0, 1.0, 1.0))

_FACE_CORNER_INDICES: dict[BoundaryFace, tuple[int, int, int, int]] = {
    # Each ordering follows (-,-),(+,-),(+,+),(-,+) in face coordinates and
    # gives x_,s cross x_,t = the solid outward normal.
    BoundaryFace.X_MIN: (0, 4, 7, 3),
    BoundaryFace.X_MAX: (1, 2, 6, 5),
    BoundaryFace.Y_MIN: (0, 1, 5, 4),
    BoundaryFace.Y_MAX: (3, 7, 6, 2),
    BoundaryFace.Z_MIN: (0, 3, 2, 1),
    BoundaryFace.Z_MAX: (4, 5, 6, 7),
}
_FACE_NORMALS: dict[BoundaryFace, Vector3] = {
    BoundaryFace.X_MIN: (-1.0, 0.0, 0.0),
    BoundaryFace.X_MAX: (1.0, 0.0, 0.0),
    BoundaryFace.Y_MIN: (0.0, -1.0, 0.0),
    BoundaryFace.Y_MAX: (0.0, 1.0, 0.0),
    BoundaryFace.Z_MIN: (0.0, 0.0, -1.0),
    BoundaryFace.Z_MAX: (0.0, 0.0, 1.0),
}
_FACE_NEIGHBOR_OFFSETS: dict[BoundaryFace, tuple[int, int, int]] = {
    BoundaryFace.X_MIN: (-1, 0, 0),
    BoundaryFace.X_MAX: (1, 0, 0),
    BoundaryFace.Y_MIN: (0, -1, 0),
    BoundaryFace.Y_MAX: (0, 1, 0),
    BoundaryFace.Z_MIN: (0, 0, -1),
    BoundaryFace.Z_MAX: (0, 0, 1),
}
_FACE_AREA_AXES: dict[BoundaryFace, tuple[int, int]] = {
    BoundaryFace.X_MIN: (1, 2),
    BoundaryFace.X_MAX: (1, 2),
    BoundaryFace.Y_MIN: (0, 2),
    BoundaryFace.Y_MAX: (0, 2),
    BoundaryFace.Z_MIN: (0, 1),
    BoundaryFace.Z_MAX: (0, 1),
}


def _shape_values(points: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.125 * np.prod(
        1.0
        + points[:, None, :] * _REFERENCE_NODE_SIGNS[None, :, :],
        axis=2,
    )


def _face_shape_values(points: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.25 * np.prod(
        1.0
        + points[:, None, :]
        * _FACE_REFERENCE_NODE_SIGNS[None, :, :],
        axis=2,
    )


def _face_shape_derivatives(
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    first_coordinate = points[:, None, 0]
    second_coordinate = points[:, None, 1]
    first_sign = _FACE_REFERENCE_NODE_SIGNS[None, :, 0]
    second_sign = _FACE_REFERENCE_NODE_SIGNS[None, :, 1]
    return 0.25 * np.stack(
        (
            first_sign * (1.0 + second_coordinate * second_sign),
            second_sign * (1.0 + first_coordinate * first_sign),
        ),
        axis=2,
    )


def _reference_shape_derivatives(
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.stack(
        tuple(
            0.125
            * _REFERENCE_NODE_SIGNS[None, :, derivative_axis]
            * np.prod(
                1.0
                + points[:, None, other_axes]
                * _REFERENCE_NODE_SIGNS[None, :, other_axes],
                axis=2,
            )
            for derivative_axis in range(3)
            for other_axes in (
                tuple(axis for axis in range(3) if axis != derivative_axis),
            )
        ),
        axis=2,
    )


def _strain_displacement_matrices(
    cell_sizes: NDArray[np.float64], points: NDArray[np.float64]
) -> NDArray[np.float64]:
    section_cell_sizes = (
        cell_sizes[:1] if _cell_sizes_are_uniform(cell_sizes) else cell_sizes
    )
    derivatives = (
        _reference_shape_derivatives(points)[None, :, :, :]
        * (2.0 / section_cell_sizes)[:, None, None, :]
    )
    dx, dy, dz = tuple(derivatives[:, :, :, axis] for axis in range(3))
    zeros = np.zeros_like(dx)
    node_blocks = np.stack(
        (
            np.stack((dx, zeros, zeros), axis=-1),
            np.stack((zeros, dy, zeros), axis=-1),
            np.stack((zeros, zeros, dz), axis=-1),
            np.stack((dy, dx, zeros), axis=-1),
            np.stack((zeros, dz, dy), axis=-1),
            np.stack((dz, zeros, dx), axis=-1),
        ),
        axis=3,
    )
    section_matrices = np.transpose(node_blocks, (0, 1, 3, 2, 4)).reshape(
        (len(section_cell_sizes), len(points), 6, 24)
    )
    return np.broadcast_to(
        section_matrices,
        (len(cell_sizes), *section_matrices.shape[1:]),
    )


def _cell_sizes_are_uniform(cell_sizes: NDArray[np.float64]) -> bool:
    return bool(
        False
        if len(cell_sizes) == 0
        else np.all(
            np.abs(cell_sizes - cell_sizes[:1])
            <= 8.0
            * np.finfo(np.float64).eps
            * max(float(np.max(np.abs(cell_sizes))), 1.0)
        )
    )


def _constitutive_matrices(
    properties: tuple[IsotropicConstitutiveProperties, ...],
) -> NDArray[np.float64]:
    young = np.asarray(tuple(value.young_modulus for value in properties))
    poisson = np.asarray(tuple(value.poisson_ratio for value in properties))
    lame_lambda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    lame_mu = young / (2.0 * (1.0 + poisson))
    return (
        lame_lambda[:, None, None] * _LAMBDA_CONSTITUTIVE_BASIS[None, :, :]
        + lame_mu[:, None, None] * _MU_CONSTITUTIVE_BASIS[None, :, :]
    )


def _cell_volumes(cell_sizes: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.prod(cell_sizes, axis=1)


def _quadrature_volumes(volumes: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.broadcast_to(
        (volumes / 8.0)[:, None], (len(volumes), len(_GAUSS_POINTS))
    )


def _gauss_temperature_changes(
    element_temperature_changes: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.einsum(
        "qn,en->eq",
        _shape_values(_GAUSS_POINTS),
        element_temperature_changes,
    )


def _gauss_thermal_strain(
    thermal_expansion: NDArray[np.float64],
    gauss_temperature_changes: NDArray[np.float64],
) -> NDArray[np.float64]:
    return (
        thermal_expansion[:, None, None]
        * gauss_temperature_changes[:, :, None]
        * _THERMAL_STRAIN_DIRECTION[None, None, :]
    )


def _assemble_sparse_stiffness(
    element_dofs: NDArray[np.int64],
    element_stiffness: NDArray[np.float64],
    degree_count: int | None = None,
) -> csr_matrix:
    local_degree_count = element_dofs.shape[1]
    assembled_degree_count = (
        int(np.max(element_dofs)) + 1
        if degree_count is None
        else degree_count
    )
    index_dtype = (
        np.int32
        if assembled_degree_count <= np.iinfo(np.int32).max
        else np.int64
    )
    indexed_element_dofs = element_dofs.astype(index_dtype, copy=False)
    return csr_matrix(
        (
            element_stiffness.reshape(-1),
            (
                np.repeat(
                    indexed_element_dofs, local_degree_count, axis=1
                ).reshape(-1),
                np.tile(
                    indexed_element_dofs, (1, local_degree_count)
                ).reshape(-1),
            ),
        ),
        shape=(assembled_degree_count, assembled_degree_count),
    )


def _node_dofs(nodes: NDArray[np.int64]) -> NDArray[np.int64]:
    return nodes[..., None] * 3 + np.arange(3, dtype=np.int64)


def _voigt_stress_tensors(
    stress: NDArray[np.float64],
) -> NDArray[np.float64]:
    xx, yy, zz, xy, yz, xz = tuple(
        stress[..., component] for component in range(6)
    )
    return np.stack(
        (
            np.stack((xx, xy, xz), axis=-1),
            np.stack((xy, yy, yz), axis=-1),
            np.stack((xz, yz, zz), axis=-1),
        ),
        axis=-2,
    )


def _cross_product_matrices(
    vectors: NDArray[np.float64],
) -> NDArray[np.float64]:
    first, second, third = tuple(
        vectors[..., component] for component in range(3)
    )
    zeros = np.zeros_like(first)
    return np.stack(
        (
            np.stack((zeros, -third, second), axis=-1),
            np.stack((third, zeros, -first), axis=-1),
            np.stack((-second, first, zeros), axis=-1),
        ),
        axis=-2,
    )


def _von_mises(stress: NDArray[np.float64]) -> NDArray[np.float64]:
    xx, yy, zz, xy, yz, xz = tuple(stress[:, component] for component in range(6))
    return np.sqrt(
        0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2)
        + 3.0 * (xy**2 + yz**2 + xz**2)
    )


def _sum_nodal_vector(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sum(vector.reshape((-1, 3)), axis=0)


def _vector3(value: NDArray[np.float64]) -> Vector3:
    x, y, z = map(float, value)
    return x, y, z
