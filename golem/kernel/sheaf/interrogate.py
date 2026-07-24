"""R13's interrogation of a non-convergent balance solve.

A solver code is not an address.  When the iterative solver stops without
converging, the operator, the stop iterate, and the sealed budget are all
still in scope — discarding them and reporting ``solver_info`` alone withholds
evidence the kernel could have produced.  This module decomposes the failure
into four typed hypotheses, each with its witness:

- topology — what the structural guards proved, and the weakest interfaces
  still holding the glue;
- conditioning — a Gershgorin spectral bound, the diagonal spread, and the
  worst local sections;
- demand — the balance equations the stop iterate least satisfies;
- budget — the iteration and tolerance state at stop.

Everything here is a pure function of the failed problem and the solver
state; the accepted path never pays for it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from golem.kernel.sheaf.problem import BalanceProblem, SealedSolverConfig
from golem.kernel.sheaf.result import (
    BalanceBridgeWitness,
    BalanceBudgetWitness,
    BalanceConditionCellWitness,
    BalanceConditioningWitness,
    BalanceDemandWitness,
    BalanceInterrogation,
    BalanceResidualCellWitness,
    BalanceTopologyWitness,
)
from golem.kernel.sheaf.vocabulary import CellId

_WITNESS_COUNT = 3


def _topology_witness(
    problem: BalanceProblem, free_operator: csr_matrix
) -> BalanceTopologyWitness:
    component_count = int(
        connected_components(
            free_operator, directed=False, return_labels=False
        )
    )
    weakest = tuple(
        sorted(
            problem.symmetric_conductances,
            key=lambda term: (term.conductance, str(term.interface_id)),
        )[:_WITNESS_COUNT]
    )
    return BalanceTopologyWitness(
        free_component_count=component_count,
        weakest_bridges=tuple(
            BalanceBridgeWitness(
                term.interface_id, term.left_cell, term.right_cell, term.conductance
            )
            for term in weakest
        ),
    )


def _conditioning_witness(
    free_operator: csr_matrix, free_cells: NDArray[np.int64]
) -> BalanceConditioningWitness:
    cell_count = free_operator.shape[0]
    diagonal = np.asarray(np.abs(free_operator.diagonal()), dtype=np.float64)
    coo = free_operator.tocoo()
    off_diagonal = coo.row != coo.col
    rows = coo.row[off_diagonal]
    magnitudes = np.abs(coo.data[off_diagonal])
    row_max = np.full(cell_count, -np.inf)
    row_min = np.full(cell_count, np.inf)
    np.maximum.at(row_max, rows, magnitudes)
    np.minimum.at(row_min, rows, magnitudes)
    off_diagonal_count = np.bincount(rows, minlength=cell_count)
    ratios = np.divide(
        row_max, row_min, out=np.ones(cell_count), where=off_diagonal_count >= 2
    )
    row_absolute_sums = np.asarray(
        abs(free_operator).sum(axis=1), dtype=np.float64
    ).reshape(-1)
    order = np.argsort(-ratios, kind="stable")
    worst = tuple(
        BalanceConditionCellWitness(
            CellId(int(free_cells[local])),
            float(diagonal[local]),
            float(ratios[local]),
        )
        for local in order[:_WITNESS_COUNT]
        if ratios[local] > 1.0
    )
    return BalanceConditioningWitness(
        gershgorin_upper_bound=float(row_absolute_sums.max(initial=0.0)),
        diagonal_spread=float(diagonal.max() / diagonal.min()),
        worst_cells=worst,
    )


def _demand_witness(
    free_operator: csr_matrix,
    free_source: NDArray[np.float64],
    free_cells: NDArray[np.int64],
    stop_values: NDArray[np.float64],
    solver_info: int,
    iteration_cap: int,
    config: SealedSolverConfig,
) -> BalanceDemandWitness:
    stop_residual = np.asarray(
        free_source - free_operator @ stop_values, dtype=np.float64
    ).reshape(-1)
    source_norm = float(np.linalg.norm(free_source))
    residual_norm = float(np.linalg.norm(stop_residual))
    magnitudes = np.abs(stop_residual)
    order = np.argsort(-magnitudes, kind="stable")
    worst = tuple(
        BalanceResidualCellWitness(
            CellId(int(free_cells[local])), float(stop_residual[local])
        )
        for local in order[:_WITNESS_COUNT]
        if magnitudes[local] > 0.0
    )
    return BalanceDemandWitness(
        maximum_stop_residual=float(magnitudes.max(initial=0.0)),
        residual_tolerance=config.residual_tolerance,
        relative_residual=(
            residual_norm / source_norm if source_norm > 0.0 else 0.0
        ),
        relative_tolerance=config.relative_tolerance,
        iteration_cap=iteration_cap,
        iterations_used=solver_info,
        exhausted=solver_info >= iteration_cap,
        worst_cells=worst,
    )


def _budget_witness(
    solver_info: int, iteration_cap: int, config: SealedSolverConfig
) -> BalanceBudgetWitness:
    return BalanceBudgetWitness(
        iteration_cap=iteration_cap,
        iterations_used=solver_info,
        relative_tolerance=config.relative_tolerance,
        exhausted=solver_info >= iteration_cap,
    )


def interrogate_balance_failure(
    problem: BalanceProblem,
    free_operator: csr_matrix,
    free_source: NDArray[np.float64],
    free_cells: NDArray[np.int64],
    stop_values: NDArray[np.float64],
    solver_info: int,
    iteration_cap: int,
    config: SealedSolverConfig,
) -> BalanceInterrogation:
    """Decompose a non-convergent solve into typed hypotheses with witnesses.

    ``problem`` is the canonical (validated) problem; ``free_operator`` and
    ``free_source`` are the reduced system the solver refused; ``stop_values``
    is the iterate the solver returned at stop; ``free_cells`` maps the
    reduced rows back to global cell ids so every witness names the same
    addresses the section would have.
    """
    return BalanceInterrogation(
        topology=_topology_witness(problem, free_operator),
        conditioning=_conditioning_witness(free_operator, free_cells),
        demand=_demand_witness(
            free_operator,
            free_source,
            free_cells,
            stop_values,
            solver_info,
            iteration_cap,
            config,
        ),
        budget=_budget_witness(solver_info, iteration_cap, config),
    )
