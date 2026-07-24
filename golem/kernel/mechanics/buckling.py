"""Linearized buckling: initial-stress + follower-pressure generalized pencil."""

from __future__ import annotations

from math import isfinite, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, eig, eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigs, eigsh

from .assemble import (
    _AssembledSystem,
    _assemble_geometric_stiffness,
    _assemble_pressure_load_stiffness,
)
from .discretize import _Discretization
from .element import _vector3
from .linear_solve import _SolvedDisplacements
from .model import MechanicsCriteria, MechanicsProblem
from .obstructions import (
    AbsentBucklingSpectrumObstruction,
    InvalidBucklingEigenpairObstruction,
    SingularBucklingSpectrumObstruction,
    UnresolvedBucklingSpectrumObstruction,
)
from .results import (
    AcceptedLinearizedBuckling,
    LinearizedBucklingReceipt,
)


_BUCKLING_EIGEN_SOLVER_TOLERANCE = 1.0e-10
_BUCKLING_EIGEN_RESIDUAL_TOLERANCE = 1.0e-8
_BUCKLING_RECIPROCAL_EIGENVALUE_TOLERANCE = 1.0e-12
_BUCKLING_DENSE_DEGREE_LIMIT = 48
_BUCKLING_REQUESTED_MODE_COUNT = 8
_BUCKLING_MATRIX_SYMMETRY_TOLERANCE = 1.0e-10
_BUCKLING_COMPLEX_SPECTRUM_TOLERANCE = 1.0e-8


def _linearized_buckling_is_required(criteria: MechanicsCriteria) -> bool:
    return (
        criteria.require_linearized_buckling
        or criteria.minimum_linearized_buckling_load_factor is not None
    )


def _solve_linearized_buckling(
    problem: MechanicsProblem,
    discretization: _Discretization,
    assembled: _AssembledSystem,
    solved: _SolvedDisplacements,
) -> (
    AcceptedLinearizedBuckling
    | AbsentBucklingSpectrumObstruction
    | SingularBucklingSpectrumObstruction
    | UnresolvedBucklingSpectrumObstruction
    | InvalidBucklingEigenpairObstruction
):
    stiffnesses = _assemble_linearized_buckling_stiffnesses(
        problem,
        discretization,
        assembled,
        solved,
    )
    if isinstance(
        stiffnesses,
        (AbsentBucklingSpectrumObstruction, UnresolvedBucklingSpectrumObstruction),
    ):
        return stiffnesses
    geometric_stiffness, pressure_load_stiffness = stiffnesses
    eigenproblem = _solve_reciprocal_buckling_eigenproblem(
        problem,
        assembled,
        solved,
        geometric_stiffness,
        pressure_load_stiffness,
    )
    if isinstance(
        eigenproblem,
        (SingularBucklingSpectrumObstruction, UnresolvedBucklingSpectrumObstruction),
    ):
        return eigenproblem
    (
        buckling_load_matrix,
        symmetric_stiffness,
        free_degree_count,
        stability_matrix_relative_asymmetry,
        symmetric_pencil,
        reciprocal_eigenvalues,
        reciprocal_modes,
    ) = eigenproblem
    critical_eigenpair = _extract_critical_buckling_eigenpair(
        problem,
        buckling_load_matrix,
        symmetric_stiffness,
        reciprocal_eigenvalues,
        reciprocal_modes,
    )
    if isinstance(
        critical_eigenpair,
        (
            AbsentBucklingSpectrumObstruction,
            UnresolvedBucklingSpectrumObstruction,
            InvalidBucklingEigenpairObstruction,
        ),
    ):
        return critical_eigenpair
    (
        reciprocal_eigenvalue,
        load_factor,
        free_mode,
        normalized_eigen_residual,
    ) = critical_eigenpair
    return _build_linearized_buckling_result(
        problem,
        solved,
        geometric_stiffness,
        pressure_load_stiffness,
        free_degree_count,
        stability_matrix_relative_asymmetry,
        symmetric_pencil,
        reciprocal_eigenvalue,
        load_factor,
        free_mode,
        normalized_eigen_residual,
    )


def _assemble_linearized_buckling_stiffnesses(
    problem: MechanicsProblem,
    discretization: _Discretization,
    assembled: _AssembledSystem,
    solved: _SolvedDisplacements,
) -> (
    tuple[csr_matrix, csr_matrix]
    | AbsentBucklingSpectrumObstruction
    | UnresolvedBucklingSpectrumObstruction
):
    geometric_stiffness, gauss_stress = _assemble_geometric_stiffness(
        problem, discretization, assembled, solved.values
    )
    pressure_load_stiffness = _assemble_pressure_load_stiffness(
        problem,
        discretization,
        solved.values,
    )
    if isinstance(
        pressure_load_stiffness,
        UnresolvedBucklingSpectrumObstruction,
    ):
        return pressure_load_stiffness
    maximum_young_modulus = max(
        assignment.properties.young_modulus
        for assignment in problem.cell_properties
    )
    prestress_floor = max(
        problem.criteria.normalization_floor,
        maximum_young_modulus * np.finfo(np.float64).eps * 1.0e3,
    )
    maximum_absolute_prestress = float(
        np.max(np.abs(gauss_stress), initial=0.0)
    )
    if maximum_absolute_prestress <= prestress_floor:
        return AbsentBucklingSpectrumObstruction(
            "accepted static state has no prestress above the roundoff-scaled "
            f"threshold {prestress_floor:.6e} Pa"
        )
    return geometric_stiffness, pressure_load_stiffness


def _solve_reciprocal_buckling_eigenproblem(
    problem: MechanicsProblem,
    assembled: _AssembledSystem,
    solved: _SolvedDisplacements,
    geometric_stiffness: csr_matrix,
    pressure_load_stiffness: csr_matrix,
) -> (
    tuple[
        csr_matrix,
        csr_matrix,
        int,
        float,
        bool,
        NDArray[np.complex128],
        NDArray[np.complex128],
    ]
    | SingularBucklingSpectrumObstruction
    | UnresolvedBucklingSpectrumObstruction
):
    free_stiffness = assembled.stiffness[solved.free_dofs][
        :, solved.free_dofs
    ]
    buckling_load_matrix = (
        -geometric_stiffness + pressure_load_stiffness
    )[solved.free_dofs][
        :, solved.free_dofs
    ]
    free_degree_count = free_stiffness.shape[0]
    if free_degree_count == 0:
        return SingularBucklingSpectrumObstruction(
            "no free degree of freedom remains for a buckling mode"
        )
    symmetric_stiffness = 0.5 * (free_stiffness + free_stiffness.T)
    buckling_load_norm = float(np.linalg.norm(buckling_load_matrix.data))
    stability_matrix_relative_asymmetry = float(
        np.linalg.norm(
            (buckling_load_matrix - buckling_load_matrix.T).data
        )
        / max(
            buckling_load_norm,
            problem.criteria.normalization_floor,
        )
    )
    symmetric_pencil = (
        stability_matrix_relative_asymmetry
        <= _BUCKLING_MATRIX_SYMMETRY_TOLERANCE
    )
    spectral_load_matrix = (
        0.5 * (buckling_load_matrix + buckling_load_matrix.T)
        if symmetric_pencil
        else buckling_load_matrix
    )
    spectrum = _reciprocal_buckling_spectrum(
        spectral_load_matrix,
        symmetric_stiffness,
        symmetric_pencil,
    )
    if isinstance(
        spectrum,
        (SingularBucklingSpectrumObstruction, UnresolvedBucklingSpectrumObstruction),
    ):
        return spectrum
    reciprocal_eigenvalues, reciprocal_modes = spectrum
    return (
        buckling_load_matrix,
        symmetric_stiffness,
        free_degree_count,
        stability_matrix_relative_asymmetry,
        symmetric_pencil,
        reciprocal_eigenvalues,
        reciprocal_modes,
    )


def _extract_critical_buckling_eigenpair(
    problem: MechanicsProblem,
    buckling_load_matrix: csr_matrix,
    symmetric_stiffness: csr_matrix,
    reciprocal_eigenvalues: NDArray[np.complex128],
    reciprocal_modes: NDArray[np.complex128],
) -> (
    tuple[float, float, NDArray[np.float64], float]
    | AbsentBucklingSpectrumObstruction
    | UnresolvedBucklingSpectrumObstruction
    | InvalidBucklingEigenpairObstruction
):
    finite_indices = (
        np.isfinite(reciprocal_eigenvalues.real)
        & np.isfinite(reciprocal_eigenvalues.imag)
    )
    positive_real_indices = finite_indices & (
        reciprocal_eigenvalues.real
        > _BUCKLING_RECIPROCAL_EIGENVALUE_TOLERANCE
    )
    real_positive_indices = np.flatnonzero(
        positive_real_indices
        & (
            np.abs(reciprocal_eigenvalues.imag)
            <= _BUCKLING_COMPLEX_SPECTRUM_TOLERANCE
            * np.maximum(1.0, np.abs(reciprocal_eigenvalues.real))
        )
    )
    if real_positive_indices.size == 0 and np.any(positive_real_indices):
        return UnresolvedBucklingSpectrumObstruction(
            "the generalized follower-load pencil exposes only complex "
            "positive-reciprocal modes; a real divergence factor is unresolved"
        )
    if real_positive_indices.size == 0:
        return AbsentBucklingSpectrumObstruction(
            "the accepted prestress and follower-pressure tangent have no "
            "positive real reciprocal buckling eigenvalue"
        )
    critical_index = int(
        real_positive_indices[
            np.argmax(
                reciprocal_eigenvalues.real[real_positive_indices]
            )
        ]
    )
    reciprocal_eigenvalue = float(
        reciprocal_eigenvalues.real[critical_index]
    )
    load_factor = 1.0 / reciprocal_eigenvalue
    free_mode = _real_buckling_mode(
        reciprocal_modes[:, critical_index],
        problem.criteria.normalization_floor,
    )
    if free_mode is None:
        return UnresolvedBucklingSpectrumObstruction(
            "the critical generalized eigenvector has no stable real phase"
        )
    stiffness_action = symmetric_stiffness @ free_mode
    stability_action = buckling_load_matrix @ free_mode
    eigen_residual = stiffness_action - load_factor * stability_action
    normalized_eigen_residual = float(
        np.linalg.norm(eigen_residual)
        / max(
            float(np.linalg.norm(stiffness_action)),
            float(abs(load_factor) * np.linalg.norm(stability_action)),
            problem.criteria.normalization_floor,
        )
    )
    if (
        not isfinite(load_factor)
        or load_factor <= 0.0
        or not isfinite(normalized_eigen_residual)
        or normalized_eigen_residual > _BUCKLING_EIGEN_RESIDUAL_TOLERANCE
    ):
        return InvalidBucklingEigenpairObstruction(
            load_factor,
            normalized_eigen_residual,
            _BUCKLING_EIGEN_RESIDUAL_TOLERANCE,
        )
    return (
        reciprocal_eigenvalue,
        load_factor,
        free_mode,
        normalized_eigen_residual,
    )


def _build_linearized_buckling_result(
    problem: MechanicsProblem,
    solved: _SolvedDisplacements,
    geometric_stiffness: csr_matrix,
    pressure_load_stiffness: csr_matrix,
    free_degree_count: int,
    stability_matrix_relative_asymmetry: float,
    symmetric_pencil: bool,
    reciprocal_eigenvalue: float,
    load_factor: float,
    free_mode: NDArray[np.float64],
    normalized_eigen_residual: float,
) -> AcceptedLinearizedBuckling | UnresolvedBucklingSpectrumObstruction:
    normalized_mode = _normalized_full_buckling_mode(
        free_mode,
        solved.free_dofs,
        solved.constrained_dofs,
        problem.criteria.normalization_floor,
    )
    if normalized_mode is None:
        return UnresolvedBucklingSpectrumObstruction(
            "generalized eigenvector has no finite nonzero nodal amplitude"
        )
    return AcceptedLinearizedBuckling(
        mode_displacements=tuple(map(_vector3, normalized_mode)),
        receipt=LinearizedBucklingReceipt(
            eigenproblem=(
                "K phi = lambda (-K_G(sigma_0) + K_pressure(u_0)) phi"
            ),
            prestress_model=(
                "accepted static Q1 Gauss-point Cauchy stress and consistent "
                "physical follower-pressure surface tangent; all reference "
                "loads scale proportionally with lambda"
            ),
            lowest_positive_load_factor=load_factor,
            reciprocal_eigenvalue=reciprocal_eigenvalue,
            normalized_eigen_residual=normalized_eigen_residual,
            geometric_stiffness_frobenius_norm=float(
                np.linalg.norm(geometric_stiffness.data)
            ),
            follower_pressure_stiffness_frobenius_norm=float(
                np.linalg.norm(pressure_load_stiffness.data)
            ),
            stability_matrix_relative_asymmetry=(
                stability_matrix_relative_asymmetry
            ),
            eigensolver_kind=(
                f"{'symmetric' if symmetric_pencil else 'nonsymmetric'} "
                f"{'dense' if free_degree_count <= _BUCKLING_DENSE_DEGREE_LIMIT else 'sparse'} "
                "generalized reciprocal"
            ),
            free_degree_count=free_degree_count,
        ),
    )


def _reciprocal_buckling_spectrum(
    geometric_load: csr_matrix,
    elastic_stiffness: csr_matrix,
    symmetric_pencil: bool,
) -> (
    tuple[NDArray[np.complex128], NDArray[np.complex128]]
    | SingularBucklingSpectrumObstruction
    | UnresolvedBucklingSpectrumObstruction
):
    degree_count = elastic_stiffness.shape[0]
    try:
        if degree_count <= _BUCKLING_DENSE_DEGREE_LIMIT:
            eigenvalues, eigenvectors = (
                eigh(
                    geometric_load.toarray(),
                    elastic_stiffness.toarray(),
                    check_finite=True,
                    driver="gvd",
                )
                if symmetric_pencil
                else eig(
                    geometric_load.toarray(),
                    elastic_stiffness.toarray(),
                    check_finite=True,
                )
            )
        else:
            requested_count = min(
                _BUCKLING_REQUESTED_MODE_COUNT,
                degree_count - 1,
            )
            deterministic_start = np.sin(
                np.arange(1, degree_count + 1, dtype=np.float64)
            ) + np.cos(
                np.arange(1, degree_count + 1, dtype=np.float64) * sqrt(2.0)
            )
            eigenvalues, eigenvectors = (
                eigsh(
                    geometric_load,
                    k=requested_count,
                    M=elastic_stiffness,
                    which="LA",
                    v0=deterministic_start,
                    tol=_BUCKLING_EIGEN_SOLVER_TOLERANCE,
                    maxiter=max(100, degree_count * 20),
                )
                if symmetric_pencil
                else eigs(
                    geometric_load,
                    k=requested_count,
                    M=elastic_stiffness,
                    which="LR",
                    v0=deterministic_start,
                    tol=_BUCKLING_EIGEN_SOLVER_TOLERANCE,
                    maxiter=max(100, degree_count * 20),
                )
            )
    except LinAlgError as failure:
        return SingularBucklingSpectrumObstruction(str(failure))
    except ArpackNoConvergence as failure:
        converged_count = (
            0 if failure.eigenvalues is None else len(failure.eigenvalues)
        )
        return UnresolvedBucklingSpectrumObstruction(
            f"ARPACK converged {converged_count} reciprocal eigenpairs "
            "before exhausting its iteration budget"
        )
    except RuntimeError as failure:
        return SingularBucklingSpectrumObstruction(str(failure))
    except ValueError as failure:
        return UnresolvedBucklingSpectrumObstruction(str(failure))
    return (
        np.asarray(eigenvalues, dtype=np.complex128),
        np.asarray(eigenvectors, dtype=np.complex128),
    )


def _real_buckling_mode(
    mode: NDArray[np.complex128],
    normalization_floor: float,
) -> NDArray[np.float64] | None:
    if not np.all(np.isfinite(mode)):
        return None
    maximum_component = int(np.argmax(np.abs(mode)))
    maximum_magnitude = float(np.abs(mode[maximum_component]))
    if maximum_magnitude <= normalization_floor:
        return None
    phased_mode = mode * np.exp(-1j * np.angle(mode[maximum_component]))
    real_mode = phased_mode.real
    imaginary_norm = float(np.linalg.norm(phased_mode.imag))
    real_norm = float(np.linalg.norm(real_mode))
    return (
        np.asarray(real_mode, dtype=np.float64)
        if imaginary_norm
        <= _BUCKLING_COMPLEX_SPECTRUM_TOLERANCE
        * max(real_norm, normalization_floor)
        else None
    )


def _normalized_full_buckling_mode(
    free_mode: NDArray[np.float64],
    free_dofs: NDArray[np.int64],
    constrained_dofs: NDArray[np.int64],
    normalization_floor: float,
) -> NDArray[np.float64] | None:
    all_dofs = np.concatenate((free_dofs, constrained_dofs))
    full_mode = np.concatenate(
        (free_mode, np.zeros(len(constrained_dofs), dtype=np.float64))
    )[np.argsort(all_dofs)]
    nodal_mode = full_mode.reshape((-1, 3))
    maximum_amplitude = float(
        np.max(np.linalg.norm(nodal_mode, axis=1), initial=0.0)
    )
    if not isfinite(maximum_amplitude) or maximum_amplitude <= normalization_floor:
        return None
    maximum_component = int(np.argmax(np.abs(full_mode)))
    canonical_sign = -1.0 if full_mode[maximum_component] < 0.0 else 1.0
    return canonical_sign * nodal_mode / maximum_amplitude
