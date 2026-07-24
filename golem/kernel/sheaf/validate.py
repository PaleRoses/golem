"""Structural and connectivity guards for a balance problem: solver-config
validation, canonical ordering, per-term metadata, and component anchoring."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import assert_never, cast

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from golem.kernel.sheaf.arrays import (
    _nonnegative_finite,
    _ordered_cells,
    _positive_finite,
)
from golem.kernel.sheaf.problem import (
    AmbientExchange,
    BalanceProblem,
    CellSource,
    DirectedTransport,
    FixedBoundary,
    SealedSolverConfig,
    SymmetricConductance,
)
from golem.kernel.sheaf.result import (
    Accepted,
    ConflictingFixedBalanceBoundaryObstruction,
    DegenerateBalanceInterfaceObstruction,
    DuplicateBalanceTermIdObstruction,
    EmptyBalanceDomainObstruction,
    InvalidBalanceCellObstruction,
    InvalidBalanceCoefficientObstruction,
    InvalidBalanceSolverConfigObstruction,
    InvalidBalanceTermIdObstruction,
    NonFiniteBalanceValueObstruction,
    Obstruction,
    Rejected,
    Result,
    UnanchoredBalanceComponentObstruction,
)
from golem.kernel.sheaf.vocabulary import (
    BalanceInterfaceId,
    BalanceProblemId,
    BalanceSolverParameter,
    BalanceTermKind,
    CellId,
)


def _balance_solver_config_obstructions(
    problem_id: BalanceProblemId, config: SealedSolverConfig
) -> tuple[Obstruction, ...]:
    checks = (
        (
            BalanceSolverParameter.RELATIVE_TOLERANCE,
            config.relative_tolerance,
            _positive_finite(config.relative_tolerance),
        ),
        (
            BalanceSolverParameter.RESIDUAL_TOLERANCE,
            config.residual_tolerance,
            _nonnegative_finite(config.residual_tolerance),
        ),
        (
            BalanceSolverParameter.IMBALANCE_TOLERANCE,
            config.imbalance_tolerance,
            _nonnegative_finite(config.imbalance_tolerance),
        ),
        (
            BalanceSolverParameter.NORMALIZATION_FLOOR,
            config.normalization_floor,
            _positive_finite(config.normalization_floor),
        ),
        (
            BalanceSolverParameter.MAXIMUM_ITERATION_FACTOR,
            float(config.maximum_iteration_factor),
            isinstance(config.maximum_iteration_factor, int)
            and config.maximum_iteration_factor > 0,
        ),
    )
    return tuple(
        InvalidBalanceSolverConfigObstruction(problem_id, parameter, value)
        for parameter, value, is_valid in checks
        if not is_valid
    )


def _canonical_balance_problem(problem: BalanceProblem) -> BalanceProblem:
    symmetric_conductances = tuple(
        sorted(
            (
                SymmetricConductance(
                    term.interface_id,
                    *_ordered_cells(term.left_cell, term.right_cell),
                    term.conductance,
                )
                for term in problem.symmetric_conductances
            ),
            key=lambda term: (
                str(term.interface_id),
                int(term.left_cell),
                int(term.right_cell),
            ),
        )
    )
    return BalanceProblem(
        problem.problem_id,
        problem.domain,
        symmetric_conductances,
        tuple(
            sorted(
                problem.directed_transports,
                key=lambda term: (
                    str(term.interface_id),
                    int(term.upstream_cell),
                    int(term.downstream_cell),
                ),
            )
        ),
        tuple(
            sorted(
                problem.sources,
                key=lambda term: (str(term.source_id), int(term.cell_id)),
            )
        ),
        tuple(
            sorted(
                problem.fixed_boundaries,
                key=lambda term: (int(term.cell_id), str(term.boundary_id)),
            )
        ),
        tuple(
            sorted(
                problem.ambient_exchanges,
                key=lambda term: (str(term.exchange_id), int(term.cell_id)),
            )
        ),
    )


@dataclass(frozen=True)
class _BalanceTermMetadata:
    kind: BalanceTermKind
    term_id: str
    cells: tuple[CellId, ...]
    coefficient: float | None
    authored_values: tuple[float, ...]


type _BalanceTerm = (
    SymmetricConductance
    | DirectedTransport
    | CellSource
    | FixedBoundary
    | AmbientExchange
)


def _balance_term_metadata(term: _BalanceTerm) -> _BalanceTermMetadata:
    match term:
        case SymmetricConductance(interface_id, left, right, conductance):
            return _BalanceTermMetadata(
                BalanceTermKind.SYMMETRIC_CONDUCTANCE,
                str(interface_id),
                (left, right),
                conductance,
                (),
            )
        case DirectedTransport(interface_id, upstream, downstream, coefficient):
            return _BalanceTermMetadata(
                BalanceTermKind.DIRECTED_TRANSPORT,
                str(interface_id),
                (upstream, downstream),
                coefficient,
                (),
            )
        case CellSource(source_id, cell_id, value):
            return _BalanceTermMetadata(
                BalanceTermKind.CELL_SOURCE,
                str(source_id),
                (cell_id,),
                None,
                (value,),
            )
        case FixedBoundary(boundary_id, cell_id, value):
            return _BalanceTermMetadata(
                BalanceTermKind.FIXED_BOUNDARY,
                str(boundary_id),
                (cell_id,),
                None,
                (value,),
            )
        case AmbientExchange(exchange_id, cell_id, conductance, ambient_value):
            return _BalanceTermMetadata(
                BalanceTermKind.AMBIENT_EXCHANGE,
                str(exchange_id),
                (cell_id,),
                conductance,
                (ambient_value,),
            )
        case _ as unreachable:
            assert_never(unreachable)


def _balance_terms(problem: BalanceProblem) -> tuple[_BalanceTerm, ...]:
    return (
        *problem.symmetric_conductances,
        *problem.directed_transports,
        *problem.sources,
        *problem.fixed_boundaries,
        *problem.ambient_exchanges,
    )


def _empty_domain_obstructions(
    problem: BalanceProblem,
) -> tuple[Obstruction, ...]:
    return (
        (
            EmptyBalanceDomainObstruction(
                problem.problem_id, problem.domain.complex_id
            ),
        )
        if problem.domain.cell_count == 0
        else ()
    )


def _invalid_term_id_obstructions(
    problem: BalanceProblem, metadata: tuple[_BalanceTermMetadata, ...]
) -> tuple[Obstruction, ...]:
    return tuple(
        InvalidBalanceTermIdObstruction(problem.problem_id, term.kind)
        for term in metadata
        if not term.term_id
    )


def _duplicate_term_id_obstructions(
    problem: BalanceProblem, metadata: tuple[_BalanceTermMetadata, ...]
) -> tuple[Obstruction, ...]:
    identifier_counts = Counter((term.kind, term.term_id) for term in metadata)
    return tuple(
        DuplicateBalanceTermIdObstruction(problem.problem_id, term_kind, term_id)
        for (term_kind, term_id), count in identifier_counts.items()
        if count > 1
    )


def _invalid_cell_obstructions(
    problem: BalanceProblem, metadata: tuple[_BalanceTermMetadata, ...]
) -> tuple[Obstruction, ...]:
    return tuple(
        InvalidBalanceCellObstruction(
            problem.problem_id,
            term.kind,
            term.term_id,
            cell_id,
            problem.domain.cell_count,
        )
        for term in metadata
        for cell_id in term.cells
        if not 0 <= int(cell_id) < problem.domain.cell_count
    )


def _invalid_coefficient_obstructions(
    problem: BalanceProblem, metadata: tuple[_BalanceTermMetadata, ...]
) -> tuple[Obstruction, ...]:
    return tuple(
        InvalidBalanceCoefficientObstruction(
            problem.problem_id,
            term.kind,
            term.term_id,
            cast(float, term.coefficient),
        )
        for term in metadata
        if term.coefficient is not None
        and (not isfinite(term.coefficient) or term.coefficient <= 0.0)
    )


def _nonfinite_value_obstructions(
    problem: BalanceProblem, metadata: tuple[_BalanceTermMetadata, ...]
) -> tuple[Obstruction, ...]:
    return tuple(
        NonFiniteBalanceValueObstruction(
            problem.problem_id, term.kind, term.term_id, value
        )
        for term in metadata
        for value in term.authored_values
        if not isfinite(value)
    )


def _degenerate_interface_obstructions(
    problem: BalanceProblem, metadata: tuple[_BalanceTermMetadata, ...]
) -> tuple[Obstruction, ...]:
    return tuple(
        DegenerateBalanceInterfaceObstruction(
            problem.problem_id,
            cast(BalanceInterfaceId, term.term_id),
            term.cells[0],
        )
        for term in metadata
        if len(term.cells) == 2 and term.cells[0] == term.cells[1]
    )


def _boundary_conflict_obstructions(
    problem: BalanceProblem,
) -> tuple[Obstruction, ...]:
    return tuple(
        ConflictingFixedBalanceBoundaryObstruction(
            problem.problem_id,
            left.cell_id,
            left.boundary_id,
            right.boundary_id,
        )
        for left, right in combinations(problem.fixed_boundaries, 2)
        if left.cell_id == right.cell_id
    )


def _unanchored_component_obstructions(
    problem: BalanceProblem,
) -> tuple[Obstruction, ...]:
    component_count, labels = connected_components(
        _balance_connectivity(problem), directed=False, return_labels=True
    )
    anchored_components = frozenset(
        (
            *(int(labels[int(term.cell_id)]) for term in problem.fixed_boundaries),
            *(int(labels[int(term.cell_id)]) for term in problem.ambient_exchanges),
        )
    )
    return tuple(
        UnanchoredBalanceComponentObstruction(problem.problem_id, component_id)
        for component_id in range(component_count)
        if component_id not in anchored_components
    )


def _validate_balance_problem(problem: BalanceProblem) -> Result[BalanceProblem]:
    metadata = tuple(map(_balance_term_metadata, _balance_terms(problem)))
    structural = (
        *_empty_domain_obstructions(problem),
        *_invalid_term_id_obstructions(problem, metadata),
        *_duplicate_term_id_obstructions(problem, metadata),
        *_invalid_cell_obstructions(problem, metadata),
        *_invalid_coefficient_obstructions(problem, metadata),
        *_nonfinite_value_obstructions(problem, metadata),
        *_degenerate_interface_obstructions(problem, metadata),
        *_boundary_conflict_obstructions(problem),
    )
    if structural:
        return Rejected(structural)
    unanchored = _unanchored_component_obstructions(problem)
    return Rejected(unanchored) if unanchored else Accepted(problem)


def _balance_connectivity(problem: BalanceProblem) -> csr_matrix:
    edges = np.asarray(
        (
            *(
                (int(term.left_cell), int(term.right_cell))
                for term in problem.symmetric_conductances
            ),
            *(
                (int(term.upstream_cell), int(term.downstream_cell))
                for term in problem.directed_transports
            ),
        ),
        dtype=np.int64,
    ).reshape((-1, 2))
    return csr_matrix(
        (
            np.ones(len(edges) * 2, dtype=np.int8),
            (
                np.concatenate((edges[:, 0], edges[:, 1])),
                np.concatenate((edges[:, 1], edges[:, 0])),
            ),
        ),
        shape=(problem.domain.cell_count, problem.domain.cell_count),
    )
