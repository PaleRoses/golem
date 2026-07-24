"""Lower solved fields into typed mechanics results and acceptance evidence."""

from __future__ import annotations

from math import inf, isfinite

import numpy as np
from numpy.typing import NDArray

from .assemble import _AssembledSystem, _ordered_cell_properties
from .buckling import (
    _linearized_buckling_is_required,
    _solve_linearized_buckling,
)
from .discretize import _Discretization
from .element import (
    _THERMAL_STRAIN_DIRECTION,
    _cell_volumes,
    _strain_displacement_matrices,
    _sum_nodal_vector,
    _vector3,
    _von_mises,
)
from .linear_solve import _SolvedDisplacements
from .model import (
    DISCRETIZATION_NAME,
    MODEL_NAME,
    MechanicsCriteria,
    MechanicsProblem,
    NodeIndex,
    UnevaluatedPhysics,
)
from .obstructions import (
    AbsentBucklingSpectrumObstruction,
    ExcessiveDisplacementObstruction,
    ForceBalanceObstruction,
    InsufficientBucklingMarginObstruction,
    InsufficientYieldMarginObstruction,
    InvalidBucklingEigenpairObstruction,
    MechanicsObstruction,
    NonFiniteMechanicsResultObstruction,
    SingularBucklingSpectrumObstruction,
    SmallStrainLimitObstruction,
    SolverResidualObstruction,
    UnresolvedBucklingSpectrumObstruction,
)
from .results import (
    AcceptedLinearizedBuckling,
    AcceptedMechanics,
    BucklingNotRequested,
    BucklingResult,
    CellMechanics,
    MechanicsReceipt,
    MechanicsResult,
    NodeMechanics,
    RejectedMechanics,
    SymmetricTensor3,
)


def _derive_verdict(
    problem: MechanicsProblem,
    discretization: _Discretization,
    assembled: _AssembledSystem,
    solved: _SolvedDisplacements,
) -> MechanicsResult:
    (
        nodal_displacement,
        nodal_physical_load,
        nodal_thermal_load,
        nodal_reaction,
        reaction,
        cell_values,
        normalized_residual,
    ) = _derive_nodal_and_cell_values(
        problem,
        discretization,
        assembled,
        solved,
    )
    (
        maximum_displacement,
        maximum_principal_strain,
        maximum_von_mises,
        minimum_yield_margin,
        minimum_yield_factor,
    ) = _derive_acceptance_extrema(nodal_displacement, cell_values)
    (
        total_body_force,
        total_nodal_force,
        total_pressure_force,
        total_external_force,
        total_reaction_force,
        force_balance,
        relative_force_imbalance,
        moment_balance,
        relative_moment_imbalance,
    ) = _derive_equilibrium_values(
        problem.criteria,
        discretization,
        assembled,
        nodal_physical_load,
        nodal_thermal_load,
        nodal_reaction,
        reaction,
    )
    concrete_obstructions = _collect_static_obstructions(
        problem.criteria,
        maximum_displacement,
        maximum_principal_strain,
        maximum_von_mises,
        minimum_yield_margin,
        minimum_yield_factor,
        normalized_residual,
        relative_force_imbalance,
        relative_moment_imbalance,
    )
    if concrete_obstructions:
        return RejectedMechanics(concrete_obstructions)
    buckling_evaluation = _derive_buckling_evaluation(
        problem,
        discretization,
        assembled,
        solved,
    )
    if isinstance(
        buckling_evaluation,
        (
            AbsentBucklingSpectrumObstruction,
            SingularBucklingSpectrumObstruction,
            UnresolvedBucklingSpectrumObstruction,
            InvalidBucklingEigenpairObstruction,
        ),
    ):
        return RejectedMechanics((buckling_evaluation,))
    buckling_margin_obstruction = _buckling_margin_obstruction(
        problem.criteria,
        buckling_evaluation,
    )
    if buckling_margin_obstruction is not None:
        return RejectedMechanics((buckling_margin_obstruction,))
    nodes = _derive_node_values(
        discretization,
        nodal_displacement,
        nodal_physical_load,
        nodal_thermal_load,
        nodal_reaction,
    )
    receipt = _build_mechanics_receipt(
        problem,
        solved,
        nodes,
        cell_values,
        maximum_displacement,
        maximum_principal_strain,
        maximum_von_mises,
        minimum_yield_margin,
        minimum_yield_factor,
        normalized_residual,
        total_body_force,
        total_nodal_force,
        total_pressure_force,
        total_external_force,
        total_reaction_force,
        force_balance,
        relative_force_imbalance,
        moment_balance,
        relative_moment_imbalance,
        buckling_evaluation,
    )
    return AcceptedMechanics(
        domain=problem.domain,
        nodes=nodes,
        cells=cell_values,
        buckling=buckling_evaluation,
        receipt=receipt,
    )


def _derive_nodal_and_cell_values(
    problem: MechanicsProblem,
    discretization: _Discretization,
    assembled: _AssembledSystem,
    solved: _SolvedDisplacements,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    tuple[CellMechanics, ...],
    float,
]:
    displacement = solved.values
    rhs = assembled.physical_load + assembled.thermal_load
    residual = assembled.stiffness @ displacement - rhs
    residual_scale = max(
        float(np.linalg.norm(rhs[solved.free_dofs])),
        problem.criteria.normalization_floor,
    )
    normalized_residual = float(
        np.linalg.norm(residual[solved.free_dofs]) / residual_scale
    )
    constrained_mask = np.isin(
        np.arange(len(displacement), dtype=np.int64), solved.constrained_dofs
    )
    reaction = np.where(constrained_mask, residual, 0.0)
    nodal_displacement = displacement.reshape((-1, 3))
    nodal_physical_load = assembled.physical_load.reshape((-1, 3))
    nodal_thermal_load = assembled.thermal_load.reshape((-1, 3))
    nodal_reaction = reaction.reshape((-1, 3))
    cell_values = _derive_cell_values(
        problem, discretization, assembled, displacement
    )
    return (
        nodal_displacement,
        nodal_physical_load,
        nodal_thermal_load,
        nodal_reaction,
        reaction,
        cell_values,
        normalized_residual,
    )


def _derive_acceptance_extrema(
    nodal_displacement: NDArray[np.float64],
    cell_values: tuple[CellMechanics, ...],
) -> tuple[float, float, float, float, float | None]:
    displacement_magnitudes = np.linalg.norm(nodal_displacement, axis=1)
    maximum_displacement = float(np.max(displacement_magnitudes, initial=0.0))
    maximum_principal_strain = max(
        map(_maximum_absolute_principal_strain, cell_values), default=0.0
    )
    maximum_von_mises = max(
        map(lambda cell: cell.von_mises_stress, cell_values), default=0.0
    )
    minimum_yield_margin = min(
        map(lambda cell: cell.yield_margin, cell_values), default=inf
    )
    finite_yield_factors = tuple(
        cell.yield_safety_factor
        for cell in cell_values
        if cell.yield_safety_factor is not None
    )
    minimum_yield_factor = (
        min(finite_yield_factors) if finite_yield_factors else None
    )
    return (
        maximum_displacement,
        maximum_principal_strain,
        maximum_von_mises,
        minimum_yield_margin,
        minimum_yield_factor,
    )


def _derive_equilibrium_values(
    criteria: MechanicsCriteria,
    discretization: _Discretization,
    assembled: _AssembledSystem,
    nodal_physical_load: NDArray[np.float64],
    nodal_thermal_load: NDArray[np.float64],
    nodal_reaction: NDArray[np.float64],
    reaction: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    NDArray[np.float64],
    float,
]:
    total_body_force = _sum_nodal_vector(assembled.body_load)
    total_nodal_force = _sum_nodal_vector(assembled.nodal_load)
    total_pressure_force = _sum_nodal_vector(assembled.pressure_load)
    total_external_force = _sum_nodal_vector(assembled.physical_load)
    total_reaction_force = _sum_nodal_vector(reaction)
    force_balance = total_external_force + total_reaction_force
    relative_force_imbalance = _relative_imbalance(
        force_balance,
        (nodal_physical_load, nodal_thermal_load, nodal_reaction),
        criteria.normalization_floor,
    )
    external_moment = np.sum(
        np.cross(discretization.node_positions, nodal_physical_load), axis=0
    )
    reaction_moment = np.sum(
        np.cross(discretization.node_positions, nodal_reaction), axis=0
    )
    moment_balance = external_moment + reaction_moment
    relative_moment_imbalance = _relative_imbalance(
        moment_balance,
        (
            np.cross(discretization.node_positions, nodal_physical_load),
            np.cross(discretization.node_positions, nodal_thermal_load),
            np.cross(discretization.node_positions, nodal_reaction),
        ),
        criteria.normalization_floor,
    )
    return (
        total_body_force,
        total_nodal_force,
        total_pressure_force,
        total_external_force,
        total_reaction_force,
        force_balance,
        relative_force_imbalance,
        moment_balance,
        relative_moment_imbalance,
    )


def _collect_static_obstructions(
    criteria: MechanicsCriteria,
    maximum_displacement: float,
    maximum_principal_strain: float,
    maximum_von_mises: float,
    minimum_yield_margin: float,
    minimum_yield_factor: float | None,
    normalized_residual: float,
    relative_force_imbalance: float,
    relative_moment_imbalance: float,
) -> tuple[MechanicsObstruction, ...]:
    result_obstructions: tuple[MechanicsObstruction | None, ...] = (
        _non_finite_result_obstruction(
            maximum_displacement,
            maximum_principal_strain,
            maximum_von_mises,
            minimum_yield_margin,
            normalized_residual,
            relative_force_imbalance,
            relative_moment_imbalance,
        ),
        _solver_residual_obstruction(criteria, normalized_residual),
        _force_balance_obstruction(
            criteria,
            relative_force_imbalance,
            relative_moment_imbalance,
        ),
        _small_strain_limit_obstruction(criteria, maximum_principal_strain),
        _displacement_limit_obstruction(criteria, maximum_displacement),
        _yield_margin_obstruction(criteria, minimum_yield_factor),
    )
    concrete_obstructions = tuple(
        obstruction for obstruction in result_obstructions if obstruction is not None
    )
    return concrete_obstructions


def _non_finite_result_obstruction(
    maximum_displacement: float,
    maximum_principal_strain: float,
    maximum_von_mises: float,
    minimum_yield_margin: float,
    normalized_residual: float,
    relative_force_imbalance: float,
    relative_moment_imbalance: float,
) -> NonFiniteMechanicsResultObstruction | None:
    return (
        NonFiniteMechanicsResultObstruction("displacement/stress/equilibrium")
        if not all(
            map(
                isfinite,
                (
                    maximum_displacement,
                    maximum_principal_strain,
                    maximum_von_mises,
                    minimum_yield_margin,
                    normalized_residual,
                    relative_force_imbalance,
                    relative_moment_imbalance,
                ),
            )
        )
        else None
    )


def _solver_residual_obstruction(
    criteria: MechanicsCriteria,
    normalized_residual: float,
) -> SolverResidualObstruction | None:
    return (
        SolverResidualObstruction(
            normalized_residual,
            criteria.linear_solver_relative_tolerance,
        )
        if normalized_residual > criteria.linear_solver_relative_tolerance
        else None
    )


def _force_balance_obstruction(
    criteria: MechanicsCriteria,
    relative_force_imbalance: float,
    relative_moment_imbalance: float,
) -> ForceBalanceObstruction | None:
    return (
        ForceBalanceObstruction(
            relative_force_imbalance,
            relative_moment_imbalance,
            criteria.equilibrium_relative_tolerance,
        )
        if max(relative_force_imbalance, relative_moment_imbalance)
        > criteria.equilibrium_relative_tolerance
        else None
    )


def _small_strain_limit_obstruction(
    criteria: MechanicsCriteria,
    maximum_principal_strain: float,
) -> SmallStrainLimitObstruction | None:
    return (
        SmallStrainLimitObstruction(
            maximum_principal_strain,
            criteria.maximum_principal_strain,
        )
        if maximum_principal_strain > criteria.maximum_principal_strain
        else None
    )


def _displacement_limit_obstruction(
    criteria: MechanicsCriteria,
    maximum_displacement: float,
) -> ExcessiveDisplacementObstruction | None:
    return (
        ExcessiveDisplacementObstruction(
            maximum_displacement,
            criteria.maximum_displacement,
        )
        if criteria.maximum_displacement is not None
        and maximum_displacement > criteria.maximum_displacement
        else None
    )


def _yield_margin_obstruction(
    criteria: MechanicsCriteria,
    minimum_yield_factor: float | None,
) -> InsufficientYieldMarginObstruction | None:
    return (
        InsufficientYieldMarginObstruction(
            minimum_yield_factor,
            criteria.minimum_yield_safety_factor,
        )
        if criteria.minimum_yield_safety_factor is not None
        and minimum_yield_factor is not None
        and minimum_yield_factor < criteria.minimum_yield_safety_factor
        else None
    )


def _derive_buckling_evaluation(
    problem: MechanicsProblem,
    discretization: _Discretization,
    assembled: _AssembledSystem,
    solved: _SolvedDisplacements,
) -> (
    BucklingResult
    | AbsentBucklingSpectrumObstruction
    | SingularBucklingSpectrumObstruction
    | UnresolvedBucklingSpectrumObstruction
    | InvalidBucklingEigenpairObstruction
):
    return (
        _solve_linearized_buckling(
            problem,
            discretization,
            assembled,
            solved,
        )
        if _linearized_buckling_is_required(problem.criteria)
        else BucklingNotRequested()
    )


def _buckling_margin_obstruction(
    criteria: MechanicsCriteria,
    buckling_evaluation: BucklingResult,
) -> InsufficientBucklingMarginObstruction | None:
    return (
        InsufficientBucklingMarginObstruction(
            buckling_evaluation.receipt.lowest_positive_load_factor,
            criteria.minimum_linearized_buckling_load_factor,
        )
        if isinstance(buckling_evaluation, AcceptedLinearizedBuckling)
        and criteria.minimum_linearized_buckling_load_factor is not None
        and buckling_evaluation.receipt.lowest_positive_load_factor
        < criteria.minimum_linearized_buckling_load_factor
        else None
    )


def _derive_node_values(
    discretization: _Discretization,
    nodal_displacement: NDArray[np.float64],
    nodal_physical_load: NDArray[np.float64],
    nodal_thermal_load: NDArray[np.float64],
    nodal_reaction: NDArray[np.float64],
) -> tuple[NodeMechanics, ...]:
    return tuple(
        NodeMechanics(
            node=NodeIndex(*map(int, index)),
            position=_vector3(position),
            displacement=_vector3(node_displacement),
            applied_force=_vector3(applied_force),
            thermal_equivalent_force=_vector3(thermal_force),
            reaction=_vector3(node_reaction),
        )
        for index, position, node_displacement, applied_force, thermal_force, node_reaction
        in zip(
            discretization.node_indices,
            discretization.node_positions,
            nodal_displacement,
            nodal_physical_load,
            nodal_thermal_load,
            nodal_reaction,
        )
    )


def _build_mechanics_receipt(
    problem: MechanicsProblem,
    solved: _SolvedDisplacements,
    nodes: tuple[NodeMechanics, ...],
    cell_values: tuple[CellMechanics, ...],
    maximum_displacement: float,
    maximum_principal_strain: float,
    maximum_von_mises: float,
    minimum_yield_margin: float,
    minimum_yield_factor: float | None,
    normalized_residual: float,
    total_body_force: NDArray[np.float64],
    total_nodal_force: NDArray[np.float64],
    total_pressure_force: NDArray[np.float64],
    total_external_force: NDArray[np.float64],
    total_reaction_force: NDArray[np.float64],
    force_balance: NDArray[np.float64],
    relative_force_imbalance: float,
    moment_balance: NDArray[np.float64],
    relative_moment_imbalance: float,
    buckling_evaluation: BucklingResult,
) -> MechanicsReceipt:
    return MechanicsReceipt(
        constitutive_model=MODEL_NAME,
        discretization=DISCRETIZATION_NAME,
        domain_source=problem.domain.source,
        node_count=len(nodes),
        cell_count=len(cell_values),
        minimum_solid_fraction=min(
            assignment.solid_fraction for assignment in problem.cell_properties
        ),
        fractional_cell_count=sum(
            assignment.solid_fraction < 1.0
            for assignment in problem.cell_properties
        ),
        material_fraction_model=(
            "explicit full-solid active mask or caller-supplied homogenized "
            "effective properties; no inferred mixture law"
        ),
        unconstrained_degree_count=len(solved.free_dofs),
        constrained_degree_count=len(solved.constrained_dofs),
        maximum_displacement=maximum_displacement,
        maximum_absolute_principal_strain=maximum_principal_strain,
        maximum_von_mises_stress=maximum_von_mises,
        minimum_yield_margin=minimum_yield_margin,
        minimum_yield_safety_factor=minimum_yield_factor,
        normalized_linear_solver_residual=normalized_residual,
        total_body_force=_vector3(total_body_force),
        total_nodal_force=_vector3(total_nodal_force),
        total_internal_pressure_force=_vector3(total_pressure_force),
        total_external_force=_vector3(total_external_force),
        total_reaction_force=_vector3(total_reaction_force),
        force_balance=_vector3(force_balance),
        relative_force_imbalance=relative_force_imbalance,
        moment_balance_about_origin=_vector3(moment_balance),
        relative_moment_imbalance=relative_moment_imbalance,
        lowest_positive_buckling_load_factor=(
            buckling_evaluation.receipt.lowest_positive_load_factor
            if isinstance(buckling_evaluation, AcceptedLinearizedBuckling)
            else None
        ),
        normalized_buckling_eigen_residual=(
            buckling_evaluation.receipt.normalized_eigen_residual
            if isinstance(buckling_evaluation, AcceptedLinearizedBuckling)
            else None
        ),
        not_evaluated=(
            (
                (UnevaluatedPhysics.LINEARIZED_BUCKLING,)
                if isinstance(buckling_evaluation, BucklingNotRequested)
                else ()
            )
            + (
            UnevaluatedPhysics.FATIGUE,
            UnevaluatedPhysics.FRACTURE,
            UnevaluatedPhysics.LARGE_DEFORMATION,
            UnevaluatedPhysics.CONTACT_PLASTICITY,
            UnevaluatedPhysics.ANISOTROPIC_COMPOSITE_FAILURE,
            UnevaluatedPhysics.SUBGRID_CHANNEL_STRESS_CONCENTRATION,
            )
        ),
    )


def _derive_cell_values(
    problem: MechanicsProblem,
    discretization: _Discretization,
    assembled: _AssembledSystem,
    displacement: NDArray[np.float64],
) -> tuple[CellMechanics, ...]:
    center_b = _strain_displacement_matrices(
        discretization.cell_sizes, np.zeros((1, 3), dtype=np.float64)
    )[:, 0, :, :]
    element_displacement = displacement[assembled.element_dofs]
    total_strain = np.einsum("eai,ei->ea", center_b, element_displacement)
    assignments_by_cell = {
        assignment.cell: assignment for assignment in problem.cell_properties
    }
    assignments = tuple(
        assignments_by_cell[cell] for cell in problem.domain.solid_cells
    )
    properties = _ordered_cell_properties(problem)
    center_temperature = np.mean(assembled.element_temperature_changes, axis=1)
    thermal_strain = (
        np.asarray(
            tuple(value.thermal_expansion_coefficient for value in properties)
        )[:, None]
        * center_temperature[:, None]
        * _THERMAL_STRAIN_DIRECTION[None, :]
    )
    elastic_strain = total_strain - thermal_strain
    stress = np.einsum(
        "eab,eb->ea", assembled.constitutive_matrices, elastic_strain
    )
    von_mises = _von_mises(stress)
    yield_strength = np.asarray(tuple(value.yield_strength for value in properties))
    yield_margin = yield_strength - von_mises
    yield_safety_factor = np.divide(
        yield_strength,
        von_mises,
        out=np.full_like(yield_strength, np.inf),
        where=von_mises > problem.criteria.normalization_floor,
    )
    volumes = _cell_volumes(discretization.cell_sizes)
    return tuple(
        CellMechanics(
            cell=cell,
            center=_vector3(center),
            volume=float(volume),
            solid_fraction=assignment.solid_fraction,
            fraction_resolution=assignment.fraction_resolution,
            total_strain=_strain_tensor(total),
            thermal_strain=_strain_tensor(thermal),
            elastic_strain=_strain_tensor(elastic),
            stress=_stress_tensor(stress_value),
            von_mises_stress=float(equivalent),
            yield_margin=float(margin),
            yield_safety_factor=(
                None if not isfinite(safety_factor) else float(safety_factor)
            ),
        )
        for cell, assignment, center, volume, total, thermal, elastic, stress_value, equivalent, margin, safety_factor
        in zip(
            problem.domain.solid_cells,
            assignments,
            discretization.cell_centers,
            volumes,
            total_strain,
            thermal_strain,
            elastic_strain,
            stress,
            von_mises,
            yield_margin,
            yield_safety_factor,
        )
    )


def _maximum_absolute_principal_strain(cell: CellMechanics) -> float:
    strain = cell.total_strain
    tensor = np.asarray(
        (
            (strain.xx, strain.xy, strain.xz),
            (strain.xy, strain.yy, strain.yz),
            (strain.xz, strain.yz, strain.zz),
        )
    )
    return float(np.max(np.abs(np.linalg.eigvalsh(tensor))))


def _strain_tensor(value: NDArray[np.float64]) -> SymmetricTensor3:
    return SymmetricTensor3(
        xx=float(value[0]),
        yy=float(value[1]),
        zz=float(value[2]),
        xy=float(value[3] / 2.0),
        yz=float(value[4] / 2.0),
        xz=float(value[5] / 2.0),
    )


def _stress_tensor(value: NDArray[np.float64]) -> SymmetricTensor3:
    return SymmetricTensor3(*map(float, value))


def _relative_imbalance(
    imbalance: NDArray[np.float64],
    nodal_contributions: tuple[NDArray[np.float64], ...],
    normalization_floor: float,
) -> float:
    scale = max(
        tuple(
            float(np.sum(np.linalg.norm(contribution, axis=1)))
            for contribution in nodal_contributions
        )
        + (normalization_floor,)
    )
    return float(np.linalg.norm(imbalance) / scale)
