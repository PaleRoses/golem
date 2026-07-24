"""The balance solver core: assemble the sparse operator, solve the free-cell
system, derive fluxes and residuals, and glue the global section."""

from __future__ import annotations

from math import fsum

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg, splu

from golem.kernel.sheaf.arrays import _cell_totals, _interface_balance, _scatter
from golem.kernel.sheaf.interrogate import interrogate_balance_failure
from golem.kernel.sheaf.problem import BalanceProblem, SealedSolverConfig
from golem.kernel.sheaf.result import (
    Accepted,
    BalanceImbalanceObstruction,
    BalanceMatrixObstruction,
    BalanceResidualObstruction,
    NonConvergentBalanceObstruction,
    Obstruction,
    Rejected,
    Result,
)
from golem.kernel.sheaf.section import (
    AmbientExchangeFlux,
    BalanceCell,
    BalanceSolution,
    DirectedTransportFlux,
    FixedBoundarySupply,
    Section,
    SectionValue,
    SymmetricConductanceFlux,
)
from golem.kernel.sheaf.validate import (
    _balance_solver_config_obstructions,
    _canonical_balance_problem,
    _validate_balance_problem,
)
from golem.kernel.sheaf.vocabulary import BalanceMatrixFailure, CellId


def solve_balance(
    problem: BalanceProblem,
    config: SealedSolverConfig = SealedSolverConfig(),
) -> Result[BalanceSolution]:
    """Descend immutable local balance terms and glue one global section."""
    config_obstructions = _balance_solver_config_obstructions(
        problem.problem_id, config
    )
    if config_obstructions:
        return Rejected(config_obstructions)
    canonical_problem = _canonical_balance_problem(problem)
    validation = _validate_balance_problem(canonical_problem)
    if isinstance(validation, Rejected):
        return validation
    operator, source_vector = _assemble_balance_system(canonical_problem)
    pinned_cells, pinned_values, free_cells = _partition_boundary_cells(
        canonical_problem
    )
    solved_values = _solve_balance_values(
        canonical_problem,
        operator,
        source_vector,
        free_cells,
        pinned_cells,
        pinned_values,
        config,
    )
    if isinstance(solved_values, Rejected):
        return solved_values
    solution = _derive_balance_solution(
        canonical_problem,
        operator,
        source_vector,
        solved_values.value,
        free_cells,
        config,
    )
    obstructions = _solution_obstructions(canonical_problem, solution, config)
    return Rejected(obstructions) if obstructions else Accepted(solution)


def _partition_boundary_cells(
    problem: BalanceProblem,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
    pinned_cells = np.asarray(
        tuple(int(boundary.cell_id) for boundary in problem.fixed_boundaries),
        dtype=np.int64,
    )
    pinned_values = np.asarray(
        tuple(boundary.value for boundary in problem.fixed_boundaries),
        dtype=np.float64,
    )
    pinned_set = frozenset(map(int, pinned_cells))
    free_cells = np.asarray(
        tuple(
            cell_index
            for cell_index in range(problem.domain.cell_count)
            if cell_index not in pinned_set
        ),
        dtype=np.int64,
    )
    return pinned_cells, pinned_values, free_cells


def _solution_obstructions(
    problem: BalanceProblem,
    solution: BalanceSolution,
    config: SealedSolverConfig,
) -> tuple[Obstruction, ...]:
    residual_obstructions: tuple[Obstruction, ...] = tuple(
        BalanceResidualObstruction(
            problem.problem_id,
            balance.cell_id,
            balance.normalized_residual,
            config.residual_tolerance,
        )
        for balance in solution.cell_balances
        if not balance.is_fixed
        and balance.normalized_residual > config.residual_tolerance
    )
    imbalance_obstructions: tuple[Obstruction, ...] = (
        (
            BalanceImbalanceObstruction(
                problem.problem_id,
                solution.relative_imbalance,
                config.imbalance_tolerance,
            ),
        )
        if solution.relative_imbalance > config.imbalance_tolerance
        else ()
    )
    return (*residual_obstructions, *imbalance_obstructions)


def _operator_triplet_entries(
    problem: BalanceProblem,
) -> tuple[tuple[int, int, float], ...]:
    return (
        *((int(term.left_cell), int(term.left_cell), term.conductance)
          for term in problem.symmetric_conductances),
        *((int(term.right_cell), int(term.right_cell), term.conductance)
          for term in problem.symmetric_conductances),
        *((int(term.left_cell), int(term.right_cell), -term.conductance)
          for term in problem.symmetric_conductances),
        *((int(term.right_cell), int(term.left_cell), -term.conductance)
          for term in problem.symmetric_conductances),
        *((int(term.upstream_cell), int(term.upstream_cell),
           term.transport_coefficient)
          for term in problem.directed_transports),
        *((int(term.downstream_cell), int(term.upstream_cell),
           -term.transport_coefficient)
          for term in problem.directed_transports),
        *((int(term.cell_id), int(term.cell_id), term.conductance)
          for term in problem.ambient_exchanges),
    )


def _balance_source_vector(problem: BalanceProblem) -> NDArray[np.float64]:
    return np.asarray(
        _cell_totals(
            problem.domain.cell_count,
            tuple(term.cell_id for term in problem.sources),
            tuple(term.value for term in problem.sources),
        )
        + _cell_totals(
            problem.domain.cell_count,
            tuple(term.cell_id for term in problem.ambient_exchanges),
            tuple(
                term.conductance * term.ambient_value
                for term in problem.ambient_exchanges
            ),
        ),
        dtype=np.float64,
    )


def _assemble_balance_system(
    problem: BalanceProblem,
) -> tuple[csr_matrix, NDArray[np.float64]]:
    matrix_entries = _operator_triplet_entries(problem)
    rows = np.asarray(tuple(row for row, _column, _value in matrix_entries), dtype=np.int64)
    columns = np.asarray(tuple(column for _row, column, _value in matrix_entries), dtype=np.int64)
    matrix_values = np.asarray(
        tuple(value for _row, _column, value in matrix_entries), dtype=np.float64
    )
    return (
        csr_matrix(
            (matrix_values, (rows, columns)),
            shape=(problem.domain.cell_count, problem.domain.cell_count),
        ),
        _balance_source_vector(problem),
    )


def _reduced_free_system(
    operator: csr_matrix,
    source_vector: NDArray[np.float64],
    free_cells: NDArray[np.int64],
    pinned_cells: NDArray[np.int64],
    pinned_values: NDArray[np.float64],
) -> tuple[csr_matrix, NDArray[np.float64]]:
    free_operator = operator[free_cells][:, free_cells]
    free_source = source_vector[free_cells] - (
        operator[free_cells][:, pinned_cells] @ pinned_values
    )
    return free_operator, free_source


def _iteratively_refined_solution(
    factorization: object,
    free_operator: csr_matrix,
    free_source: NDArray[np.float64],
    values: NDArray[np.float64],
    remaining_steps: int,
    config: SealedSolverConfig,
) -> NDArray[np.float64]:
    residual = free_source - free_operator @ values
    relative_residual = float(np.linalg.norm(residual)) / max(
        float(np.linalg.norm(free_source)),
        config.normalization_floor,
    )
    return (
        values
        if remaining_steps == 0
        or relative_residual <= config.relative_tolerance
        else _iteratively_refined_solution(
            factorization,
            free_operator,
            free_source,
            values + factorization.solve(residual),
            remaining_steps - 1,
            config,
        )
    )


def _solve_free_operator(
    problem: BalanceProblem,
    free_operator: csr_matrix,
    free_source: NDArray[np.float64],
    free_cells: NDArray[np.int64],
    iteration_cap: int,
    config: SealedSolverConfig,
) -> Result[NDArray[np.float64]]:
    try:
        if problem.directed_transports:
            factorization = splu(free_operator.tocsc())
            free_values = _iteratively_refined_solution(
                factorization,
                free_operator,
                free_source,
                factorization.solve(free_source),
                3,
                config,
            )
            solver_info = 0
        else:
            free_values, solver_info = cg(
                free_operator,
                free_source,
                rtol=config.relative_tolerance,
                atol=0.0,
                maxiter=iteration_cap,
            )
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return Rejected(
            (
                BalanceMatrixObstruction(
                    problem.problem_id,
                    BalanceMatrixFailure.LINEAR_SOLVE_FAILED,
                ),
            )
        )
    if int(solver_info) != 0:
        return Rejected(
            (
                NonConvergentBalanceObstruction(
                    problem.problem_id,
                    int(solver_info),
                    interrogate_balance_failure(
                        problem,
                        free_operator,
                        free_source,
                        free_cells,
                        free_values,
                        int(solver_info),
                        iteration_cap,
                        config,
                    ),
                ),
            )
        )
    finite_free_values = np.asarray(free_values, dtype=np.float64)
    if not np.all(np.isfinite(finite_free_values)):
        return Rejected(
            (
                BalanceMatrixObstruction(
                    problem.problem_id,
                    BalanceMatrixFailure.NONFINITE_SOLUTION,
                ),
            )
        )
    return Accepted(finite_free_values)


def _solve_balance_values(
    problem: BalanceProblem,
    operator: csr_matrix,
    source_vector: NDArray[np.float64],
    free_cells: NDArray[np.int64],
    pinned_cells: NDArray[np.int64],
    pinned_values: NDArray[np.float64],
    config: SealedSolverConfig,
) -> Result[NDArray[np.float64]]:
    if free_cells.size == 0:
        return Accepted(
            _scatter(pinned_cells, pinned_values, problem.domain.cell_count)
        )
    free_operator, free_source = _reduced_free_system(
        operator, source_vector, free_cells, pinned_cells, pinned_values
    )
    free_solution = _solve_free_operator(
        problem,
        free_operator,
        free_source,
        free_cells,
        operator.shape[0] * config.maximum_iteration_factor,
        config,
    )
    if isinstance(free_solution, Rejected):
        return free_solution
    return Accepted(
        _scatter(pinned_cells, pinned_values, problem.domain.cell_count)
        + _scatter(free_cells, free_solution.value, problem.domain.cell_count)
    )


def _symmetric_conductance_fluxes(
    problem: BalanceProblem, values: NDArray[np.float64]
) -> tuple[SymmetricConductanceFlux, ...]:
    return tuple(
        SymmetricConductanceFlux(
            term.interface_id,
            term.left_cell,
            term.right_cell,
            term.conductance,
            term.conductance
            * (values[int(term.left_cell)] - values[int(term.right_cell)]),
        )
        for term in problem.symmetric_conductances
    )


def _directed_transport_fluxes(
    problem: BalanceProblem, values: NDArray[np.float64]
) -> tuple[DirectedTransportFlux, ...]:
    return tuple(
        DirectedTransportFlux(
            term.interface_id,
            term.upstream_cell,
            term.downstream_cell,
            term.transport_coefficient,
            term.transport_coefficient * values[int(term.upstream_cell)],
        )
        for term in problem.directed_transports
    )


def _ambient_exchange_fluxes(
    problem: BalanceProblem, values: NDArray[np.float64]
) -> tuple[AmbientExchangeFlux, ...]:
    return tuple(
        AmbientExchangeFlux(
            term.exchange_id,
            term.cell_id,
            term.conductance,
            term.ambient_value,
            term.conductance * (values[int(term.cell_id)] - term.ambient_value),
        )
        for term in problem.ambient_exchanges
    )


def _cell_aggregates(
    problem: BalanceProblem,
    symmetric_fluxes: tuple[SymmetricConductanceFlux, ...],
    directed_fluxes: tuple[DirectedTransportFlux, ...],
    ambient_fluxes: tuple[AmbientExchangeFlux, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    cell_count = problem.domain.cell_count
    symmetric_balance, symmetric_incident = _interface_balance(
        cell_count,
        tuple(flux.left_cell for flux in symmetric_fluxes),
        tuple(flux.right_cell for flux in symmetric_fluxes),
        tuple(flux.left_to_right_flux for flux in symmetric_fluxes),
    )
    directed_balance, directed_incident = _interface_balance(
        cell_count,
        tuple(flux.upstream_cell for flux in directed_fluxes),
        tuple(flux.downstream_cell for flux in directed_fluxes),
        tuple(flux.upstream_to_downstream_flux for flux in directed_fluxes),
    )
    source_by_cell = _cell_totals(
        cell_count,
        tuple(term.cell_id for term in problem.sources),
        tuple(term.value for term in problem.sources),
    )
    ambient_cells = tuple(flux.cell_id for flux in ambient_fluxes)
    ambient_values = tuple(flux.cell_to_ambient_flux for flux in ambient_fluxes)
    ambient_by_cell = _cell_totals(cell_count, ambient_cells, ambient_values)
    incident_flux = (
        symmetric_incident
        + directed_incident
        + _cell_totals(
            cell_count, ambient_cells, tuple(map(abs, ambient_values))
        )
    )
    net_outflow = symmetric_balance + directed_balance + ambient_by_cell
    return source_by_cell, incident_flux, net_outflow


def _balance_residuals(
    operator: csr_matrix,
    values: NDArray[np.float64],
    source_vector: NDArray[np.float64],
    free_cells: NDArray[np.int64],
    config: SealedSolverConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    equation_residual = np.asarray(operator @ values).reshape(-1) - source_vector
    residual_scale = np.maximum(
        np.asarray(abs(operator) @ np.abs(values)).reshape(-1)
        + np.abs(source_vector),
        config.normalization_floor,
    )
    free_mask = np.zeros(len(values), dtype=np.bool_)
    free_mask[free_cells] = True
    normalized_residual = np.where(
        free_mask,
        np.abs(equation_residual) / residual_scale,
        0.0,
    )
    return equation_residual, normalized_residual


def _balance_section(
    problem: BalanceProblem,
    values: NDArray[np.float64],
    normalized_residual: NDArray[np.float64],
) -> Section:
    return Section(
        f"{problem.problem_id}/section",
        str(problem.problem_id),
        tuple(
            SectionValue(
                str(problem.problem_id),
                CellId(cell_index),
                float(values[cell_index]),
                float(normalized_residual[cell_index]),
            )
            for cell_index in range(len(values))
        ),
    )


def _fixed_boundary_supplies(
    problem: BalanceProblem, equation_residual: NDArray[np.float64]
) -> tuple[FixedBoundarySupply, ...]:
    return tuple(
        FixedBoundarySupply(
            boundary.boundary_id,
            boundary.cell_id,
            float(equation_residual[int(boundary.cell_id)]),
        )
        for boundary in problem.fixed_boundaries
    )


def _cell_balances(
    problem: BalanceProblem,
    source_by_cell: NDArray[np.float64],
    incident_flux: NDArray[np.float64],
    net_outflow: NDArray[np.float64],
    equation_residual: NDArray[np.float64],
    normalized_residual: NDArray[np.float64],
) -> tuple[BalanceCell, ...]:
    fixed_cells = frozenset(
        int(boundary.cell_id) for boundary in problem.fixed_boundaries
    )
    return tuple(
        BalanceCell(
            CellId(cell_index),
            cell_index in fixed_cells,
            float(source_by_cell[cell_index]),
            float(incident_flux[cell_index]),
            float(net_outflow[cell_index]),
            float(equation_residual[cell_index]),
            float(normalized_residual[cell_index]),
        )
        for cell_index in range(problem.domain.cell_count)
    )


def _relative_imbalance(
    problem: BalanceProblem,
    fixed_boundary_supplies: tuple[FixedBoundarySupply, ...],
    ambient_fluxes: tuple[AmbientExchangeFlux, ...],
    config: SealedSolverConfig,
) -> float:
    source_total = fsum(term.value for term in problem.sources)
    boundary_supply_total = fsum(
        supply.boundary_to_domain_flux for supply in fixed_boundary_supplies
    )
    ambient_outflow_total = fsum(
        flux.cell_to_ambient_flux for flux in ambient_fluxes
    )
    balance_scale = max(
        fsum(
            (
                *(abs(term.value) for term in problem.sources),
                *(
                    abs(supply.boundary_to_domain_flux)
                    for supply in fixed_boundary_supplies
                ),
                *(abs(flux.cell_to_ambient_flux) for flux in ambient_fluxes),
            )
        ),
        config.normalization_floor,
    )
    return (
        abs(source_total + boundary_supply_total - ambient_outflow_total)
        / balance_scale
    )


def _derive_balance_solution(
    problem: BalanceProblem,
    operator: csr_matrix,
    source_vector: NDArray[np.float64],
    values: NDArray[np.float64],
    free_cells: NDArray[np.int64],
    config: SealedSolverConfig,
) -> BalanceSolution:
    symmetric_fluxes = _symmetric_conductance_fluxes(problem, values)
    directed_fluxes = _directed_transport_fluxes(problem, values)
    ambient_fluxes = _ambient_exchange_fluxes(problem, values)
    source_by_cell, incident_flux, net_outflow = _cell_aggregates(
        problem, symmetric_fluxes, directed_fluxes, ambient_fluxes
    )
    equation_residual, normalized_residual = _balance_residuals(
        operator, values, source_vector, free_cells, config
    )
    fixed_boundary_supplies = _fixed_boundary_supplies(
        problem, equation_residual
    )
    return BalanceSolution(
        problem,
        _balance_section(problem, values, normalized_residual),
        (*symmetric_fluxes, *directed_fluxes),
        ambient_fluxes,
        fixed_boundary_supplies,
        _cell_balances(
            problem,
            source_by_cell,
            incident_flux,
            net_outflow,
            equation_residual,
            normalized_residual,
        ),
        _relative_imbalance(
            problem, fixed_boundary_supplies, ambient_fluxes, config
        ),
        config,
    )
