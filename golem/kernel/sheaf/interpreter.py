"""The extension seam: one balance algebra, many interpreters. ``lower`` descends
a domain problem into the shared algebra; ``lift`` reads the solved section back."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, assert_never

from golem.kernel.sheaf.problem import BalanceProblem, SealedSolverConfig
from golem.kernel.sheaf.result import Result
from golem.kernel.sheaf.section import BalanceSolution
from golem.kernel.sheaf.solve import solve_balance


@dataclass(frozen=True)
class LoweredProblem[carrier]:
    balance_problem: BalanceProblem
    config: SealedSolverConfig
    carrier: carrier


@dataclass(frozen=True)
class LoweringRejected[outcome]:
    outcome: outcome


type Lowering[carrier, outcome] = (
    LoweredProblem[carrier] | LoweringRejected[outcome]
)


class BalanceInterpreter[problem, carrier, outcome](Protocol):
    """One algebra, many interpreters: ``lower`` descends a domain problem into
    the shared balance algebra; ``lift`` reads the solved section back."""

    def lower(self, problem: problem) -> Lowering[carrier, outcome]: ...

    def lift(
        self, carrier: carrier, balance: Result[BalanceSolution]
    ) -> outcome: ...


def interpret_balance[problem, carrier, outcome](
    interpreter: BalanceInterpreter[problem, carrier, outcome],
    problem: problem,
) -> outcome:
    """Drive one interpreter through the shared ``solve_balance`` core."""
    lowering = interpreter.lower(problem)
    match lowering:
        case LoweringRejected(outcome):
            return outcome
        case LoweredProblem(balance_problem, config, carrier):
            return interpreter.lift(
                carrier, solve_balance(balance_problem, config)
            )
        case _ as unreachable:
            assert_never(unreachable)
