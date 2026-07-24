"""Algebra: assemble the stiffness, load, geometric, and follower-pressure systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .discretize import _Discretization, _active_node_position
from .element import (
    _FACE_AREA_AXES,
    _FACE_GAUSS_POINTS,
    _FACE_NORMALS,
    _GAUSS_POINTS,
    _assemble_sparse_stiffness,
    _cell_sizes_are_uniform,
    _cell_volumes,
    _constitutive_matrices,
    _cross_product_matrices,
    _face_shape_derivatives,
    _face_shape_values,
    _gauss_temperature_changes,
    _gauss_thermal_strain,
    _node_dofs,
    _quadrature_volumes,
    _reference_shape_derivatives,
    _strain_displacement_matrices,
    _voigt_stress_tensors,
)
from .grid import _active_face_nodes, _cell_size
from .model import (
    IsotropicConstitutiveProperties,
    MechanicsProblem,
    NodeIndex,
)
from .obstructions import UnresolvedBucklingSpectrumObstruction


@dataclass(frozen=True)
class _AssembledSystem:
    stiffness: csr_matrix
    physical_load: NDArray[np.float64]
    body_load: NDArray[np.float64]
    nodal_load: NDArray[np.float64]
    pressure_load: NDArray[np.float64]
    thermal_load: NDArray[np.float64]
    element_dofs: NDArray[np.int64]
    constitutive_matrices: NDArray[np.float64]
    element_temperature_changes: NDArray[np.float64]


def _ordered_cell_properties(
    problem: MechanicsProblem,
) -> tuple[IsotropicConstitutiveProperties, ...]:
    properties_by_cell = {
        assignment.cell: assignment.properties for assignment in problem.cell_properties
    }
    return tuple(
        properties_by_cell[cell] for cell in problem.domain.solid_cells
    )


def _assemble_system(
    problem: MechanicsProblem, discretization: _Discretization
) -> _AssembledSystem:
    (
        properties,
        constitutive,
        gauss_b,
        cell_sizes_are_uniform,
        volumes,
        quadrature_volumes,
    ) = _element_material_and_volumes(problem, discretization)
    stiffness, element_dofs = _assemble_stiffness_matrix(
        discretization,
        constitutive,
        gauss_b,
        volumes,
        quadrature_volumes,
        cell_sizes_are_uniform,
    )
    element_body_load = _element_body_load(problem, properties, volumes)
    (
        element_thermal_load,
        element_temperature_changes,
    ) = _thermal_load_contributions(
        problem,
        discretization,
        properties,
        constitutive,
        gauss_b,
        quadrature_volumes,
    )
    body_load = np.bincount(
        element_dofs.reshape(-1),
        weights=element_body_load.reshape(-1),
        minlength=discretization.degree_count,
    )
    thermal_load = np.bincount(
        element_dofs.reshape(-1),
        weights=element_thermal_load.reshape(-1),
        minlength=discretization.degree_count,
    )
    nodal_load = _nodal_force_vector(problem, discretization)
    pressure_load = _pressure_force_vector(problem, discretization)
    physical_load = body_load + nodal_load + pressure_load
    return _AssembledSystem(
        stiffness=stiffness,
        physical_load=physical_load,
        body_load=body_load,
        nodal_load=nodal_load,
        pressure_load=pressure_load,
        thermal_load=thermal_load,
        element_dofs=element_dofs,
        constitutive_matrices=constitutive,
        element_temperature_changes=element_temperature_changes,
    )


def _element_material_and_volumes(
    problem: MechanicsProblem, discretization: _Discretization
) -> tuple[
    tuple[IsotropicConstitutiveProperties, ...],
    NDArray[np.float64],
    NDArray[np.float64],
    bool,
    NDArray[np.float64],
    NDArray[np.float64],
]:
    properties = _ordered_cell_properties(problem)
    constitutive = _constitutive_matrices(properties)
    gauss_b = _strain_displacement_matrices(
        discretization.cell_sizes, _GAUSS_POINTS
    )
    cell_sizes_are_uniform = _cell_sizes_are_uniform(
        discretization.cell_sizes
    )
    raw_volumes = _cell_volumes(discretization.cell_sizes)
    volumes = (
        np.full_like(raw_volumes, raw_volumes[0])
        if cell_sizes_are_uniform
        else raw_volumes
    )
    quadrature_volumes = _quadrature_volumes(volumes)
    return (
        properties,
        constitutive,
        gauss_b,
        cell_sizes_are_uniform,
        volumes,
        quadrature_volumes,
    )


def _assemble_stiffness_matrix(
    discretization: _Discretization,
    constitutive: NDArray[np.float64],
    gauss_b: NDArray[np.float64],
    volumes: NDArray[np.float64],
    quadrature_volumes: NDArray[np.float64],
    cell_sizes_are_uniform: bool,
) -> tuple[csr_matrix, NDArray[np.int64]]:
    constitutive_is_uniform = bool(
        len(constitutive) > 0
        and np.array_equal(
            constitutive,
            np.broadcast_to(constitutive[:1], constitutive.shape),
        )
    )
    stiffness_section_count = (
        1
        if cell_sizes_are_uniform and constitutive_is_uniform
        else len(volumes)
    )
    section_stiffness = np.einsum(
        "eqai,eab,eqbj,eq->eij",
        gauss_b[:stiffness_section_count],
        constitutive[:stiffness_section_count],
        gauss_b[:stiffness_section_count],
        quadrature_volumes[:stiffness_section_count],
        optimize=True,
    )
    element_stiffness = np.broadcast_to(
        section_stiffness,
        (len(volumes), *section_stiffness.shape[1:]),
    )
    element_dofs = _node_dofs(discretization.cell_nodes).reshape((-1, 24))
    stiffness = _assemble_sparse_stiffness(element_dofs, element_stiffness)
    return stiffness, element_dofs


def _element_body_load(
    problem: MechanicsProblem,
    properties: tuple[IsotropicConstitutiveProperties, ...],
    volumes: NDArray[np.float64],
) -> NDArray[np.float64]:
    density = np.asarray(tuple(value.density for value in properties))
    acceleration = np.asarray(problem.body_acceleration, dtype=np.float64)
    return np.broadcast_to(
        density[:, None, None] * volumes[:, None, None] * acceleration / 8.0,
        (len(volumes), 8, 3),
    ).reshape((-1, 24))


def _thermal_load_contributions(
    problem: MechanicsProblem,
    discretization: _Discretization,
    properties: tuple[IsotropicConstitutiveProperties, ...],
    constitutive: NDArray[np.float64],
    gauss_b: NDArray[np.float64],
    quadrature_volumes: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    temperature_by_node = {
        temperature.node: temperature.delta_temperature
        for temperature in problem.nodal_temperature_changes
    }
    nodal_temperature_changes = np.asarray(
        tuple(
            temperature_by_node.get(NodeIndex(*map(int, index)), 0.0)
            for index in discretization.node_indices
        ),
        dtype=np.float64,
    )
    element_temperature_changes = nodal_temperature_changes[
        discretization.cell_nodes
    ]
    gauss_temperature_changes = _gauss_temperature_changes(
        element_temperature_changes
    )
    thermal_expansion = np.asarray(
        tuple(value.thermal_expansion_coefficient for value in properties)
    )
    gauss_thermal_strain = _gauss_thermal_strain(
        thermal_expansion, gauss_temperature_changes
    )
    gauss_thermal_stress = np.einsum(
        "eab,eqb->eqa", constitutive, gauss_thermal_strain
    )
    element_thermal_load = np.einsum(
        "eqai,eqa,eq->ei",
        gauss_b,
        gauss_thermal_stress,
        quadrature_volumes,
        optimize=True,
    )
    return element_thermal_load, element_temperature_changes


def _assemble_geometric_stiffness(
    problem: MechanicsProblem,
    discretization: _Discretization,
    assembled: _AssembledSystem,
    displacement: NDArray[np.float64],
) -> tuple[csr_matrix, NDArray[np.float64]]:
    gauss_stress = _gauss_stress(
        problem, discretization, assembled, displacement
    )
    gauss_stress_tensors = _voigt_stress_tensors(gauss_stress)
    shape_gradients = (
        _reference_shape_derivatives(_GAUSS_POINTS)[None, :, :, :]
        * (2.0 / discretization.cell_sizes)[:, None, None, :]
    )
    volumes = _cell_volumes(discretization.cell_sizes)
    quadrature_volumes = _quadrature_volumes(volumes)
    scalar_geometric_stiffness = np.einsum(
        "eqai,eqij,eqbj,eq->eab",
        shape_gradients,
        gauss_stress_tensors,
        shape_gradients,
        quadrature_volumes,
        optimize=True,
    )
    element_geometric_stiffness = np.einsum(
        "eab,ij->eaibj",
        scalar_geometric_stiffness,
        np.eye(3),
    ).reshape((-1, 24, 24))
    return (
        _assemble_sparse_stiffness(
            assembled.element_dofs, element_geometric_stiffness
        ),
        gauss_stress,
    )


def _gauss_stress(
    problem: MechanicsProblem,
    discretization: _Discretization,
    assembled: _AssembledSystem,
    displacement: NDArray[np.float64],
) -> NDArray[np.float64]:
    gauss_b = _strain_displacement_matrices(
        discretization.cell_sizes, _GAUSS_POINTS
    )
    gauss_total_strain = np.einsum(
        "eqai,ei->eqa",
        gauss_b,
        displacement[assembled.element_dofs],
    )
    gauss_temperature_changes = _gauss_temperature_changes(
        assembled.element_temperature_changes
    )
    properties = _ordered_cell_properties(problem)
    thermal_expansion = np.asarray(
        tuple(value.thermal_expansion_coefficient for value in properties)
    )
    gauss_thermal_strain = _gauss_thermal_strain(
        thermal_expansion, gauss_temperature_changes
    )
    return np.einsum(
        "eab,eqb->eqa",
        assembled.constitutive_matrices,
        gauss_total_strain - gauss_thermal_strain,
    )


def _nodal_force_vector(
    problem: MechanicsProblem, discretization: _Discretization
) -> NDArray[np.float64]:
    node_positions = np.asarray(
        tuple(
            _active_node_position(problem.domain, discretization, force.node)
            for force in problem.nodal_forces
        ),
        dtype=np.int64,
    )
    return (
        np.zeros(discretization.degree_count)
        if not problem.nodal_forces
        else np.bincount(
            _node_dofs(node_positions).reshape(-1),
            weights=np.asarray(
                tuple(force.force for force in problem.nodal_forces),
                dtype=np.float64,
            ).reshape(-1),
            minlength=discretization.degree_count,
        )
    )


def _pressure_force_vector(
    problem: MechanicsProblem, discretization: _Discretization
) -> NDArray[np.float64]:
    if not problem.internal_pressure_faces:
        return np.zeros(discretization.degree_count)
    node_positions = _active_face_nodes(
        problem.domain, discretization, problem.internal_pressure_faces
    )
    cell_sizes = np.asarray(
        tuple(_cell_size(problem.domain, pressure.cell) for pressure in problem.internal_pressure_faces)
    )
    face_areas = np.asarray(
        tuple(
            float(np.prod(cell_size[list(_FACE_AREA_AXES[pressure.face])]))
            for pressure, cell_size in zip(problem.internal_pressure_faces, cell_sizes)
        )
    )
    face_forces_per_node = np.asarray(
        tuple(
            -pressure.pressure
            * face_area
            * np.asarray(_FACE_NORMALS[pressure.face])
            / 4.0
            for pressure, face_area in zip(problem.internal_pressure_faces, face_areas)
        )
    )
    dofs = _node_dofs(node_positions)
    return np.bincount(
        dofs.reshape(-1),
        weights=np.broadcast_to(
            face_forces_per_node[:, None, :], dofs.shape
        ).reshape(-1),
        minlength=discretization.degree_count,
    )


def _assemble_pressure_load_stiffness(
    problem: MechanicsProblem,
    discretization: _Discretization,
    displacement: NDArray[np.float64],
) -> csr_matrix | UnresolvedBucklingSpectrumObstruction:
    """Differentiate physical pressure traction on the displaced Q1 faces."""

    nonzero_pressures = tuple(
        pressure
        for pressure in problem.internal_pressure_faces
        if pressure.pressure > 0.0
    )
    if not nonzero_pressures:
        return csr_matrix(
            (discretization.degree_count, discretization.degree_count),
            dtype=np.float64,
        )
    active_face_nodes = _active_face_nodes(
        problem.domain, discretization, nonzero_pressures
    )
    reference_face_positions = discretization.node_positions[active_face_nodes]
    displaced_node_positions = (
        discretization.node_positions + displacement.reshape((-1, 3))
    )
    displaced_face_positions = displaced_node_positions[active_face_nodes]
    face_derivatives = _face_shape_derivatives(_FACE_GAUSS_POINTS)
    reference_tangents = np.einsum(
        "qac,fan->fqcn",
        face_derivatives,
        reference_face_positions,
        optimize=True,
    )
    displaced_tangents = np.einsum(
        "qac,fan->fqcn",
        face_derivatives,
        displaced_face_positions,
        optimize=True,
    )
    reference_area_vectors = np.cross(
        reference_tangents[:, :, 0, :],
        reference_tangents[:, :, 1, :],
    )
    displaced_area_vectors = np.cross(
        displaced_tangents[:, :, 0, :],
        displaced_tangents[:, :, 1, :],
    )
    reference_area_norms = np.linalg.norm(reference_area_vectors, axis=2)
    displaced_area_norms = np.linalg.norm(displaced_area_vectors, axis=2)
    orientation_products = np.einsum(
        "fqi,fqi->fq",
        reference_area_vectors,
        displaced_area_vectors,
    )
    regular_orientation = (
        np.all(np.isfinite(displaced_area_vectors))
        and np.all(
            displaced_area_norms
            > problem.criteria.normalization_floor * reference_area_norms
        )
        and np.all(orientation_products > 0.0)
    )
    if not regular_orientation:
        return UnresolvedBucklingSpectrumObstruction(
            "a displaced follower-pressure face has a singular Jacobian or "
            "reverses its reference orientation"
        )
    first_tangent_cross = _cross_product_matrices(
        displaced_tangents[:, :, 0, :]
    )
    second_tangent_cross = _cross_product_matrices(
        displaced_tangents[:, :, 1, :]
    )
    pressure_area_derivative = (
        face_derivatives[None, :, :, 0, None, None]
        * second_tangent_cross[:, :, None, :, :]
        - face_derivatives[None, :, :, 1, None, None]
        * first_tangent_cross[:, :, None, :, :]
    )
    element_load_stiffness = np.einsum(
        "f,qa,fqbik->faibk",
        np.asarray(tuple(value.pressure for value in nonzero_pressures)),
        _face_shape_values(_FACE_GAUSS_POINTS),
        pressure_area_derivative,
        optimize=True,
    ).reshape((-1, 12, 12))
    face_dofs = _node_dofs(active_face_nodes).reshape((-1, 12))
    return _assemble_sparse_stiffness(
        face_dofs,
        element_load_stiffness,
        discretization.degree_count,
    )
