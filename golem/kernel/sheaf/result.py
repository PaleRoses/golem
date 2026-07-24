"""Typed obstruction vocabulary, the ``Result`` monad the balance algebra returns,
and the total renderer naming every obstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from golem.kernel.sheaf.vocabulary import (
    BalanceInterfaceId,
    BalanceMatrixFailure,
    BalanceProblemId,
    BalanceSolverParameter,
    BalanceTermKind,
    BoundaryId,
    CellId,
    HeatTransferCorrelationId,
    HeatTransferCorrelationKind,
    HeatTransferValidityAxis,
    ThermalParameter,
)


@dataclass(frozen=True)
class EmptyBalanceDomainObstruction:
    problem_id: BalanceProblemId
    complex_id: str


@dataclass(frozen=True)
class InvalidBalanceTermIdObstruction:
    problem_id: BalanceProblemId
    term_kind: BalanceTermKind


@dataclass(frozen=True)
class DuplicateBalanceTermIdObstruction:
    problem_id: BalanceProblemId
    term_kind: BalanceTermKind
    term_id: str


@dataclass(frozen=True)
class InvalidBalanceCellObstruction:
    problem_id: BalanceProblemId
    term_kind: BalanceTermKind
    term_id: str
    cell_id: CellId
    cell_count: int


@dataclass(frozen=True)
class InvalidBalanceCoefficientObstruction:
    problem_id: BalanceProblemId
    term_kind: BalanceTermKind
    term_id: str
    coefficient: float


@dataclass(frozen=True)
class NonFiniteBalanceValueObstruction:
    problem_id: BalanceProblemId
    term_kind: BalanceTermKind
    term_id: str
    value: float


@dataclass(frozen=True)
class DegenerateBalanceInterfaceObstruction:
    problem_id: BalanceProblemId
    interface_id: BalanceInterfaceId
    cell_id: CellId


@dataclass(frozen=True)
class ConflictingFixedBalanceBoundaryObstruction:
    problem_id: BalanceProblemId
    cell_id: CellId
    left_boundary_id: BoundaryId
    right_boundary_id: BoundaryId


@dataclass(frozen=True)
class UnanchoredBalanceComponentObstruction:
    problem_id: BalanceProblemId
    component_id: int


@dataclass(frozen=True)
class BalanceMatrixObstruction:
    problem_id: BalanceProblemId
    failure: BalanceMatrixFailure


@dataclass(frozen=True)
class InvalidBalanceSolverConfigObstruction:
    problem_id: BalanceProblemId
    parameter: BalanceSolverParameter
    value: float


@dataclass(frozen=True)
class BalanceBridgeWitness:
    """One weak conductance interface: a near-separation of the free graph
    that no validation guard rejected but the spectrum feels."""

    interface_id: BalanceInterfaceId
    left_cell: CellId
    right_cell: CellId
    conductance: float


@dataclass(frozen=True)
class BalanceTopologyWitness:
    """What is proven about the glued free graph at solver stop: validation
    anchored every component pre-solve; what remains suspect are the weakest
    interfaces holding the glue."""

    free_component_count: int
    weakest_bridges: tuple[BalanceBridgeWitness, ...]


@dataclass(frozen=True)
class BalanceConditionCellWitness:
    """One local section whose incident conductance ratio poisons the
    global spectrum."""

    cell_id: CellId
    diagonal: float
    conductance_ratio: float


@dataclass(frozen=True)
class BalanceConditioningWitness:
    """Quantitative bounds on the operator the iterative solver refused:
    the Gershgorin row-sum spectral bound, the diagonal spread, and the
    worst local sections."""

    gershgorin_upper_bound: float
    diagonal_spread: float
    worst_cells: tuple[BalanceConditionCellWitness, ...]


@dataclass(frozen=True)
class BalanceResidualCellWitness:
    """One cell whose balance equation the stop iterate least satisfies."""

    cell_id: CellId
    residual: float


@dataclass(frozen=True)
class BalanceDemandWitness:
    """The demand the solve could not meet at the returned stop iterate."""

    maximum_stop_residual: float
    residual_tolerance: float
    relative_residual: float
    relative_tolerance: float
    iteration_cap: int
    iterations_used: int
    exhausted: bool
    worst_cells: tuple[BalanceResidualCellWitness, ...]


@dataclass(frozen=True)
class BalanceBudgetWitness:
    """The iteration and tolerance state at stop: how much budget the law
    granted, how much the solver spent, and whether exhaustion forced the
    stop."""

    iteration_cap: int
    iterations_used: int
    relative_tolerance: float
    exhausted: bool


@dataclass(frozen=True)
class BalanceInterrogation:
    """R13's decomposition of a non-convergent balance solve into typed
    hypotheses, each with its witness: topology, conditioning, demand,
    budget.  Computed once, at the failure site, where the operator and the
    stop iterate still exist."""

    topology: BalanceTopologyWitness
    conditioning: BalanceConditioningWitness
    demand: BalanceDemandWitness
    budget: BalanceBudgetWitness


@dataclass(frozen=True)
class NonConvergentBalanceObstruction:
    problem_id: BalanceProblemId
    solver_info: int
    interrogation: BalanceInterrogation | None = None


@dataclass(frozen=True)
class BalanceResidualObstruction:
    problem_id: BalanceProblemId
    cell_id: CellId
    magnitude: float
    tolerance: float


@dataclass(frozen=True)
class BalanceImbalanceObstruction:
    problem_id: BalanceProblemId
    magnitude: float
    tolerance: float


@dataclass(frozen=True)
class UnsupportedHeatTransferRegimeObstruction:
    correlation_id: HeatTransferCorrelationId
    correlation_kind: HeatTransferCorrelationKind
    axis: HeatTransferValidityAxis
    actual: float | str
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class InvalidThermalParameterObstruction:
    problem_id: BalanceProblemId
    term_id: str
    parameter: ThermalParameter
    value: float


type Obstruction = (
    EmptyBalanceDomainObstruction
    | InvalidBalanceTermIdObstruction
    | DuplicateBalanceTermIdObstruction
    | InvalidBalanceCellObstruction
    | InvalidBalanceCoefficientObstruction
    | NonFiniteBalanceValueObstruction
    | DegenerateBalanceInterfaceObstruction
    | ConflictingFixedBalanceBoundaryObstruction
    | UnanchoredBalanceComponentObstruction
    | BalanceMatrixObstruction
    | InvalidBalanceSolverConfigObstruction
    | NonConvergentBalanceObstruction
    | BalanceResidualObstruction
    | BalanceImbalanceObstruction
    | UnsupportedHeatTransferRegimeObstruction
    | InvalidThermalParameterObstruction
)


@dataclass(frozen=True)
class Accepted[value]:
    value: value


@dataclass(frozen=True)
class Rejected:
    obstructions: tuple[Obstruction, ...]


type Result[value] = Accepted[value] | Rejected


def _render_regime_bounds(minimum: float | None, maximum: float | None) -> str:
    if minimum is None and maximum is None:
        return "the closed vocabulary"
    return f"{minimum}..{maximum}" if maximum is not None else f"> {minimum}"


def render_obstruction(obstruction: Obstruction) -> str:
    match obstruction:
        case EmptyBalanceDomainObstruction(problem_id, complex_id):
            return f"EmptyBalanceDomain @{problem_id}/complex:{complex_id}"
        case InvalidBalanceTermIdObstruction(problem_id, term_kind):
            return f"InvalidBalanceTermId @{problem_id}/{term_kind.value}"
        case DuplicateBalanceTermIdObstruction(problem_id, term_kind, term_id):
            return f"DuplicateBalanceTermId @{problem_id}/{term_kind.value}:{term_id}"
        case InvalidBalanceCellObstruction(
            problem_id, term_kind, term_id, cell_id, cell_count
        ):
            return (
                f"InvalidBalanceCell @{problem_id}/{term_kind.value}:{term_id}/"
                f"cell:{int(cell_id)}: valid range is 0..{cell_count - 1}"
            )
        case InvalidBalanceCoefficientObstruction(
            problem_id, term_kind, term_id, coefficient
        ):
            return (
                f"InvalidBalanceCoefficient @{problem_id}/{term_kind.value}:"
                f"{term_id}: {coefficient}"
            )
        case NonFiniteBalanceValueObstruction(
            problem_id, term_kind, term_id, value
        ):
            return (
                f"NonFiniteBalanceValue @{problem_id}/{term_kind.value}:"
                f"{term_id}: {value}"
            )
        case DegenerateBalanceInterfaceObstruction(
            problem_id, interface_id, cell_id
        ):
            return (
                f"DegenerateBalanceInterface @{problem_id}/interface:"
                f"{interface_id}/cell:{int(cell_id)}"
            )
        case ConflictingFixedBalanceBoundaryObstruction(
            problem_id, cell_id, left, right
        ):
            return (
                f"ConflictingFixedBalanceBoundary @{problem_id}/cell:"
                f"{int(cell_id)}: {left} overlaps {right}"
            )
        case UnanchoredBalanceComponentObstruction(problem_id, component_id):
            return f"UnanchoredBalanceComponent @{problem_id}/component:{component_id}"
        case BalanceMatrixObstruction(problem_id, failure):
            return f"BalanceMatrix @{problem_id}: {failure.value}"
        case InvalidBalanceSolverConfigObstruction(problem_id, parameter, value):
            return (
                f"InvalidBalanceSolverConfig @{problem_id}/{parameter.value}: "
                f"{value}"
            )
        case NonConvergentBalanceObstruction(problem_id, solver_info):
            return f"NonConvergentBalance @{problem_id}: solver_info={solver_info}"
        case BalanceResidualObstruction(
            problem_id, cell_id, magnitude, tolerance
        ):
            return (
                f"BalanceResidual @{problem_id}/cell:{int(cell_id)}: "
                f"{magnitude:.3e} > {tolerance:.3e}"
            )
        case BalanceImbalanceObstruction(problem_id, magnitude, tolerance):
            return (
                f"BalanceImbalance @{problem_id}: "
                f"{magnitude:.3e} > {tolerance:.3e}"
            )
        case UnsupportedHeatTransferRegimeObstruction(
            correlation_id, correlation_kind, axis, actual, minimum, maximum
        ):
            bounds = _render_regime_bounds(minimum, maximum)
            return (
                f"UnsupportedHeatTransferRegime @{correlation_id}/"
                f"{correlation_kind.value}/{axis.value}: {actual} not in {bounds}"
            )
        case InvalidThermalParameterObstruction(
            problem_id, term_id, parameter, value
        ):
            return (
                f"InvalidThermalParameter @{problem_id}/{term_id}/"
                f"{parameter.value}: {value}"
            )
        case _ as unreachable:
            assert_never(unreachable)
