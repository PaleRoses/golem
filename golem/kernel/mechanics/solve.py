"""Harness the mechanics validation, discretization, solve, and verdict stages."""

from __future__ import annotations

from .assemble import _assemble_system
from .discretize import _discretize
from .linear_solve import _solve_displacements
from .model import MechanicsProblem
from .obstructions import (
    LinearSolverFailureObstruction,
    RigidBodyModeObstruction,
)
from .results import MechanicsResult, RejectedMechanics
from .validate import _validate_problem
from .verdict import _derive_verdict


def solve_mechanics(problem: MechanicsProblem) -> MechanicsResult:
    """Evaluate one immutable mechanics problem and return its typed verdict."""

    input_obstructions = _validate_problem(problem)
    if input_obstructions:
        return RejectedMechanics(input_obstructions)

    discretization = _discretize(problem.domain)
    assembled = _assemble_system(problem, discretization)
    solved = _solve_displacements(problem, discretization, assembled)
    if isinstance(solved, (RigidBodyModeObstruction, LinearSolverFailureObstruction)):
        return RejectedMechanics((solved,))
    return _derive_verdict(problem, discretization, assembled, solved)
