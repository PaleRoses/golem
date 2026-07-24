"""Constrained linear solve of the assembled thermoelastic system."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .assemble import _AssembledSystem
from .discretize import _Discretization, _active_node_position
from .model import MechanicsProblem
from .obstructions import (
    LinearSolverFailureObstruction,
    RigidBodyModeObstruction,
)


@dataclass(frozen=True)
class _SolvedDisplacements:
    values: NDArray[np.float64]
    free_dofs: NDArray[np.int64]
    constrained_dofs: NDArray[np.int64]


def _solve_displacements(
    problem: MechanicsProblem,
    discretization: _Discretization,
    assembled: _AssembledSystem,
) -> _SolvedDisplacements | RigidBodyModeObstruction | LinearSolverFailureObstruction:
    support_dof_values = tuple(
        (
            3 * _active_node_position(problem.domain, discretization, support.node)
            + component,
            float(value),
        )
        for support in problem.supports
        for component, value in enumerate(support.prescribed_displacement)
        if value is not None
    )
    constrained_dofs = np.asarray(
        tuple(dof for dof, _ in support_dof_values), dtype=np.int64
    )
    prescribed_values = np.asarray(
        tuple(value for _, value in support_dof_values), dtype=np.float64
    )
    degree_count = assembled.stiffness.shape[0]
    rigid_body_obstruction = _rigid_body_mode_obstruction(
        discretization, constrained_dofs
    )
    if rigid_body_obstruction is not None:
        return rigid_body_obstruction
    free_dofs = np.setdiff1d(
        np.arange(degree_count, dtype=np.int64), constrained_dofs
    )
    if free_dofs.size == 0:
        return _SolvedDisplacements(
            prescribed_values[np.argsort(constrained_dofs)],
            free_dofs,
            constrained_dofs,
        )
    free_stiffness = assembled.stiffness[free_dofs][:, free_dofs]
    if np.any(np.abs(free_stiffness.diagonal()) <= problem.criteria.normalization_floor):
        return RigidBodyModeObstruction(
            "a free degree of freedom has zero stiffness; supports or solid connectivity "
            "leave a rigid-body mode"
        )
    rhs = assembled.physical_load + assembled.thermal_load
    free_rhs = rhs[free_dofs] - (
        assembled.stiffness[free_dofs][:, constrained_dofs] @ prescribed_values
        if constrained_dofs.size
        else 0.0
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            free_values = np.atleast_1d(spsolve(free_stiffness, free_rhs))
    except MatrixRankWarning:
        return RigidBodyModeObstruction(
            "the constrained stiffness is singular; supports do not remove every "
            "rigid-body mode or a solid component is unsupported"
        )
    except (RuntimeError, ValueError) as failure:
        return LinearSolverFailureObstruction(str(failure))
    all_dofs = np.concatenate((free_dofs, constrained_dofs))
    all_values = np.concatenate((free_values, prescribed_values))
    return _SolvedDisplacements(
        values=all_values[np.argsort(all_dofs)],
        free_dofs=free_dofs,
        constrained_dofs=constrained_dofs,
    )


def _rigid_body_mode_obstruction(
    discretization: _Discretization,
    constrained_dofs: NDArray[np.int64],
) -> RigidBodyModeObstruction | None:
    node_count = len(discretization.node_indices)
    element_nodes = discretization.cell_nodes
    node_adjacency = csr_matrix(
        (
            np.ones(element_nodes.shape[0] * 64, dtype=np.int8),
            (
                np.repeat(element_nodes, 8, axis=1).reshape(-1),
                np.tile(element_nodes, (1, 8)).reshape(-1),
            ),
        ),
        shape=(node_count, node_count),
    )
    component_count, component_labels = connected_components(
        node_adjacency, directed=False
    )
    centered_positions = (
        discretization.node_positions
        - np.mean(discretization.node_positions, axis=0, keepdims=True)
    )
    x, y, z = tuple(centered_positions[:, axis] for axis in range(3))
    zeros = np.zeros(node_count)
    ones = np.ones(node_count)
    rigid_modes = np.stack(
        (
            np.stack((ones, zeros, zeros, zeros, z, -y), axis=1),
            np.stack((zeros, ones, zeros, -z, zeros, x), axis=1),
            np.stack((zeros, zeros, ones, y, -x, zeros), axis=1),
        ),
        axis=1,
    ).reshape((-1, 6))
    component_ranks = tuple(
        (
            component_id,
            int(
                np.linalg.matrix_rank(
                    rigid_modes[
                        constrained_dofs[
                            component_labels[constrained_dofs // 3] == component_id
                        ]
                    ]
                )
            ),
        )
        for component_id in range(component_count)
    )
    deficient = next(
        (
            (component_id, rank)
            for component_id, rank in component_ranks
            if rank < 6
        ),
        None,
    )
    return (
        RigidBodyModeObstruction(
            f"solid component {deficient[0]} constrains only rank "
            f"{deficient[1]} of 6 rigid-body modes"
        )
        if deficient is not None
        else None
    )
