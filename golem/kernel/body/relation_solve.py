"""Deferred hierarchical relational solve: residual algebra, deepening, scipy boundary."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from golem.addressing.scope import Bone, Scope
from golem.kernel.body.relations import (
    BranchCycleObstruction,
    BranchSearchBudgetObstruction,
    ConventionId,
    FixedPlacementConflictObstruction,
    RELATION_CONVENTIONS,
    RelationDeclaration,
    RelationKind,
    RelationSolveObstruction,
    RelationalSolveExhaustedObstruction,
    RelationalSolverBudgetObstruction,
    UnderconstrainedRelationObstruction,
)


@dataclass(frozen=True)
class RelationalSolveConfig:
    position_tolerance: float = 1.0e-4
    angle_tolerance_deg: float = 0.25
    singular_value_tolerance: float = 1.0e-8
    maximum_evaluations: int = 2000
    maximum_active_set_resolves: int = 8


class GeometryView(Protocol):
    def point(self, scope: Scope) -> np.ndarray: ...

    def axis(self, scope: Scope) -> np.ndarray: ...

    def frame(self, scope: Scope) -> np.ndarray: ...

    def extent_sites_y(self, scope: Scope) -> tuple[tuple[str, float], ...]: ...

    def poses(self) -> Mapping[str, tuple[np.ndarray, np.ndarray]]: ...


type Evaluator = Callable[[tuple[float, ...]], GeometryView]


class VariableKind(StrEnum):
    TRANSLATION = "translation"
    SWING = "swing"
    TWIST = "twist"


class DeepeningStage(StrEnum):
    LOCAL = "local"
    TRANSLATION = "translation"
    PARAMETER = "parameter"


@dataclass(frozen=True)
class VariableSpec:
    address: str
    bone_id: str
    kind: VariableKind
    initial: float


@dataclass(frozen=True)
class Frontier:
    stage: DeepeningStage
    depth: int
    added_variables: tuple[VariableSpec, ...]


@dataclass(frozen=True)
class RelationMeasurement:
    relation_id: str
    convention: ConventionId | None
    required: float
    observed: float
    normalized_residual: float
    satisfied: bool


@dataclass(frozen=True)
class ConventionChoice:
    relation_id: str
    candidates: tuple[ConventionId, ...]
    initial_residuals: tuple[float, ...]
    selected: ConventionId
    tie_broken: bool


@dataclass(frozen=True)
class LocalSolveReceipt:
    owner_address: str
    measurements: tuple[RelationMeasurement, ...]
    conventions: tuple[ConventionChoice, ...]
    stage: DeepeningStage
    depth: int
    variable_count: int
    rank: int
    active_site_history: tuple[tuple[str, ...], ...]
    evaluations: int
    maximum_normalized_residual: float


@dataclass(frozen=True)
class AcceptedLocalSolve:
    solution: tuple[float, ...]
    variables: tuple[VariableSpec, ...]
    receipt: LocalSolveReceipt


@dataclass(frozen=True)
class LocalRelationProblem:
    owner_address: str
    declarations: tuple[RelationDeclaration, ...]
    evaluator: Evaluator
    base_variables: tuple[VariableSpec, ...]
    frontiers: tuple[Frontier, ...]


type LocalSolveResult = AcceptedLocalSolve | RelationSolveObstruction


# -- residual algebra ----------------------------------------------------- #


def _angle_tolerance_rad(config: RelationalSolveConfig) -> float:
    return math.radians(config.angle_tolerance_deg)


def _mirror_reflect_point(point: np.ndarray) -> np.ndarray:
    return point * np.array([-1.0, 1.0, 1.0])


def _mirror_reflect_frame(frame: np.ndarray) -> np.ndarray:
    reflection = np.diag([-1.0, 1.0, 1.0])
    improper = reflection @ frame
    u, _, vt = np.linalg.svd(improper)
    proper = u @ np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))]) @ vt
    return proper


def _subject_endpoints(
    view: GeometryView, scope: Scope
) -> tuple[np.ndarray, np.ndarray]:
    assert isinstance(scope, Bone)
    head, tail = view.poses()[scope.bone_id]
    return np.asarray(head, dtype=np.float64), np.asarray(tail, dtype=np.float64)


def _between_references(
    declaration: RelationDeclaration, convention: ConventionId | None
) -> tuple[Scope, Scope]:
    assert declaration.reference_b is not None
    return (
        (declaration.reference_b, declaration.reference)
        if convention is ConventionId.BETWEEN_REVERSED
        else (declaration.reference, declaration.reference_b)
    )


def _extent_y(
    view: GeometryView, scope: Scope, site: str | None, bound: str
) -> float:
    sites = view.extent_sites_y(scope)
    if site is not None:
        by_id = dict(sites)
        if site in by_id:
            return by_id[site]
    values = tuple(value for _, value in sites)
    return min(values) if bound == "min" else max(values)


def _active_site(
    view: GeometryView, scope: Scope, bound: str
) -> str:
    sites = view.extent_sites_y(scope)
    chooser = min if bound == "min" else max
    return chooser(sorted(sites), key=lambda pair: pair[1])[0]


def _relation_components(
    declaration: RelationDeclaration,
    convention: ConventionId | None,
    active: tuple[str | None, str | None],
    view: GeometryView,
    config: RelationalSolveConfig,
) -> np.ndarray:
    position_scale = 1.0 / config.position_tolerance
    angle_scale = 1.0 / _angle_tolerance_rad(config)
    subject_site, reference_site = active
    match declaration.kind:
        case RelationKind.ATTACH_AT_NAMED_SITE | RelationKind.COINCIDENT:
            difference = view.point(declaration.subject) - view.point(
                declaration.reference
            )
            return difference * position_scale
        case RelationKind.ABOVE:
            gap = (
                _extent_y(view, declaration.reference, reference_site, "max")
                + declaration.distance
                - _extent_y(view, declaration.subject, subject_site, "min")
            )
            return np.array([max(0.0, gap)]) * position_scale
        case RelationKind.BELOW:
            gap = (
                _extent_y(view, declaration.subject, subject_site, "max")
                + declaration.distance
                - _extent_y(view, declaration.reference, reference_site, "min")
            )
            return np.array([max(0.0, gap)]) * position_scale
        case RelationKind.ALIGNED:
            sign = (
                1.0
                if convention is ConventionId.ALIGNED_PARALLEL
                else -1.0
            )
            difference = view.axis(declaration.subject) - sign * view.axis(
                declaration.reference
            )
            return difference * angle_scale
        case RelationKind.PERPENDICULAR_TO:
            dot = float(
                view.axis(declaration.subject)
                @ view.axis(declaration.reference)
            )
            return np.array([dot]) * angle_scale
        case RelationKind.MIRROR_OF:
            positional = (
                view.point(declaration.subject)
                - _mirror_reflect_point(view.point(declaration.reference))
            ) * position_scale
            angular = (
                view.frame(declaration.subject)
                - _mirror_reflect_frame(view.frame(declaration.reference))
            ).ravel() * angle_scale
            return np.concatenate((positional, angular))
        case RelationKind.BETWEEN:
            head, tail = _subject_endpoints(view, declaration.subject)
            first, second = _between_references(declaration, convention)
            return np.concatenate(
                (
                    (head - view.point(first)) * position_scale,
                    (tail - view.point(second)) * position_scale,
                )
            )


def measure_relation(
    declaration: RelationDeclaration,
    convention: ConventionId | None,
    active: tuple[str | None, str | None],
    view: GeometryView,
    config: RelationalSolveConfig,
) -> RelationMeasurement:
    components = _relation_components(
        declaration, convention, active, view, config
    )
    normalized = float(np.linalg.norm(components))
    match declaration.kind:
        case RelationKind.ATTACH_AT_NAMED_SITE | RelationKind.COINCIDENT:
            observed = float(
                np.linalg.norm(
                    view.point(declaration.subject)
                    - view.point(declaration.reference)
                )
            )
            required = 0.0
            satisfied = observed <= config.position_tolerance
        case RelationKind.ABOVE:
            observed = _extent_y(
                view, declaration.subject, None, "min"
            ) - _extent_y(view, declaration.reference, None, "max")
            required = declaration.distance
            satisfied = observed >= required - config.position_tolerance
        case RelationKind.BELOW:
            observed = _extent_y(
                view, declaration.reference, None, "min"
            ) - _extent_y(view, declaration.subject, None, "max")
            required = declaration.distance
            satisfied = observed >= required - config.position_tolerance
        case RelationKind.ALIGNED:
            dot = float(
                np.clip(
                    view.axis(declaration.subject)
                    @ view.axis(declaration.reference),
                    -1.0,
                    1.0,
                )
            )
            angle = math.degrees(math.acos(dot))
            observed = (
                angle
                if convention is ConventionId.ALIGNED_PARALLEL
                else 180.0 - angle
            )
            required = 0.0
            satisfied = observed <= config.angle_tolerance_deg
        case RelationKind.PERPENDICULAR_TO:
            dot = float(
                np.clip(
                    view.axis(declaration.subject)
                    @ view.axis(declaration.reference),
                    -1.0,
                    1.0,
                )
            )
            observed = math.degrees(math.acos(abs(dot)))
            required = 90.0
            satisfied = (
                abs(90.0 - observed) <= config.angle_tolerance_deg
            )
        case RelationKind.MIRROR_OF:
            positional = float(
                np.linalg.norm(
                    view.point(declaration.subject)
                    - _mirror_reflect_point(
                        view.point(declaration.reference)
                    )
                )
            )
            observed = positional
            required = 0.0
            satisfied = normalized <= 1.0
        case RelationKind.BETWEEN:
            head, tail = _subject_endpoints(view, declaration.subject)
            first, second = _between_references(declaration, convention)
            gap_head = float(np.linalg.norm(head - view.point(first)))
            gap_tail = float(np.linalg.norm(tail - view.point(second)))
            observed = max(gap_head, gap_tail)
            required = 0.0
            satisfied = observed <= config.position_tolerance
    return RelationMeasurement(
        relation_id=declaration.relation_id,
        convention=convention,
        required=required,
        observed=observed,
        normalized_residual=normalized,
        satisfied=satisfied,
    )


# -- convention selection (frozen before any solve) ----------------------- #


def select_conventions(
    declarations: tuple[RelationDeclaration, ...],
    initial_view: GeometryView,
    config: RelationalSolveConfig,
) -> tuple[ConventionChoice, ...]:
    def choose(declaration: RelationDeclaration) -> ConventionChoice | None:
        candidates = RELATION_CONVENTIONS.get(declaration.kind, ())
        if not candidates:
            return None
        residuals = tuple(
            float(
                np.linalg.norm(
                    _relation_components(
                        declaration,
                        candidate,
                        (None, None),
                        initial_view,
                        config,
                    )
                )
            )
            for candidate in candidates
        )
        least = min(residuals)
        ties = tuple(
            candidate
            for candidate, residual in zip(candidates, residuals)
            if abs(residual - least) <= 1.0
        )
        return ConventionChoice(
            relation_id=declaration.relation_id,
            candidates=candidates,
            initial_residuals=residuals,
            selected=ties[0],
            tie_broken=len(ties) > 1,
        )

    return tuple(
        choice
        for choice in map(choose, declarations)
        if choice is not None
    )


def _conventions_by_id(
    choices: tuple[ConventionChoice, ...],
) -> Mapping[str, ConventionId]:
    return {choice.relation_id: choice.selected for choice in choices}


# -- active sites --------------------------------------------------------- #


def _active_sites(
    declarations: tuple[RelationDeclaration, ...], view: GeometryView
) -> tuple[tuple[str | None, str | None], ...]:
    def sites(
        declaration: RelationDeclaration,
    ) -> tuple[str | None, str | None]:
        match declaration.kind:
            case RelationKind.ABOVE:
                return (
                    _active_site(view, declaration.subject, "min"),
                    _active_site(view, declaration.reference, "max"),
                )
            case RelationKind.BELOW:
                return (
                    _active_site(view, declaration.subject, "max"),
                    _active_site(view, declaration.reference, "min"),
                )
            case _:
                return (None, None)

    return tuple(map(sites, declarations))


def _active_signature(
    active: tuple[tuple[str | None, str | None], ...],
) -> tuple[str, ...]:
    return tuple(
        f"{subject or ''}~{reference or ''}" for subject, reference in active
    )


# -- scipy boundary ------------------------------------------------------- #


@dataclass(frozen=True)
class SolverAttempt:
    solution: tuple[float, ...]
    residual_norm: float
    evaluations: int
    jacobian: tuple[tuple[float, ...], ...]


def _run_least_squares(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    budget: int,
) -> SolverAttempt:
    from scipy.optimize import least_squares

    result = least_squares(
        residual_fn,
        initial,
        method="trf",
        loss="linear",
        max_nfev=budget,
    )
    return SolverAttempt(
        solution=tuple(float(value) for value in result.x),
        residual_norm=float(np.linalg.norm(result.fun)),
        evaluations=int(result.nfev),
        jacobian=tuple(
            tuple(float(value) for value in row) for row in result.jac
        ),
    )


# -- deepening orchestration ---------------------------------------------- #


def _cumulative_frontiers(
    base: tuple[VariableSpec, ...], frontiers: tuple[Frontier, ...]
) -> tuple[tuple[DeepeningStage, int, tuple[VariableSpec, ...]], ...]:
    def fold(
        accumulated: tuple[
            tuple[DeepeningStage, int, tuple[VariableSpec, ...]], ...
        ],
        frontier: Frontier,
    ) -> tuple[tuple[DeepeningStage, int, tuple[VariableSpec, ...]], ...]:
        previous_stage = accumulated[-1][0] if accumulated else None
        previous_variables = (
            accumulated[-1][2]
            if accumulated and previous_stage == frontier.stage
            else base
        )
        merged = previous_variables + tuple(
            variable
            for variable in frontier.added_variables
            if variable not in previous_variables
        )
        return accumulated + ((frontier.stage, frontier.depth, merged),)

    local = ((DeepeningStage.LOCAL, 0, base),)
    return functools.reduce(fold, frontiers, local)


def _measurements(
    declarations: tuple[RelationDeclaration, ...],
    conventions: Mapping[str, ConventionId],
    active: tuple[tuple[str | None, str | None], ...],
    view: GeometryView,
    config: RelationalSolveConfig,
) -> tuple[RelationMeasurement, ...]:
    return tuple(
        measure_relation(
            declaration,
            conventions.get(declaration.relation_id),
            sites,
            view,
            config,
        )
        for declaration, sites in zip(declarations, active)
    )


def _fixed_conflicts(
    problem: LocalRelationProblem,
    conventions: Mapping[str, ConventionId],
    config: RelationalSolveConfig,
) -> tuple[FixedPlacementConflictObstruction, ...]:
    every_variable = _cumulative_frontiers(
        problem.base_variables, problem.frontiers
    )[-1][2]
    initial = tuple(variable.initial for variable in every_variable)
    view0 = problem.evaluator(initial)
    active0 = _active_sites(problem.declarations, view0)

    def depends(index: int) -> np.ndarray:
        declaration = problem.declarations[index]
        reference = _relation_components(
            declaration,
            conventions.get(declaration.relation_id),
            active0[index],
            view0,
            config,
        )
        step = 1.0e-5

        def perturbed(position: int) -> np.ndarray:
            trial = tuple(
                value + (step if column == position else 0.0)
                for column, value in enumerate(initial)
            )
            return _relation_components(
                declaration,
                conventions.get(declaration.relation_id),
                active0[index],
                problem.evaluator(trial),
                config,
            )

        return np.array(
            [
                float(np.linalg.norm(perturbed(position) - reference))
                for position in range(len(initial))
            ]
        )

    measurements = _measurements(
        problem.declarations, conventions, active0, view0, config
    )
    return tuple(
        FixedPlacementConflictObstruction(
            relation_id=declaration.relation_id,
            subject_address=declaration.subject_selector,
            reference_address=declaration.reference_selector,
            required=measurement.required,
            observed=measurement.observed,
        )
        for index, (declaration, measurement) in enumerate(
            zip(problem.declarations, measurements)
        )
        if not measurement.satisfied
        and (len(initial) == 0 or bool(np.all(depends(index) < 1.0e-12)))
    )


def _solve_active_sets(
    problem: LocalRelationProblem,
    conventions: Mapping[str, ConventionId],
    variables: tuple[VariableSpec, ...],
    config: RelationalSolveConfig,
    spent: int,
) -> (
    tuple[
        SolverAttempt,
        tuple[tuple[str, ...], ...],
        tuple[RelationMeasurement, ...],
        int,
    ]
    | RelationSolveObstruction
):
    initial = np.array([variable.initial for variable in variables])
    view = problem.evaluator(tuple(initial))
    active = _active_sites(problem.declarations, view)
    seen: tuple[tuple[str, ...], ...] = ()
    current = initial

    for _ in range(config.maximum_active_set_resolves):
        signature = _active_signature(active)
        if signature in seen:
            worst = max(
                _measurements(
                    problem.declarations,
                    conventions,
                    active,
                    problem.evaluator(tuple(current)),
                    config,
                ),
                key=lambda measurement: measurement.normalized_residual,
            )
            return BranchCycleObstruction(
                relation_id=worst.relation_id,
                active_sets=seen + (signature,),
                required=worst.required,
                observed=worst.observed,
            )
        seen = seen + (signature,)
        frozen_active = active

        def residual_fn(x: np.ndarray) -> np.ndarray:
            trial_view = problem.evaluator(tuple(float(v) for v in x))
            parts = tuple(
                _relation_components(
                    declaration,
                    conventions.get(declaration.relation_id),
                    sites,
                    trial_view,
                    config,
                )
                for declaration, sites in zip(
                    problem.declarations, frozen_active
                )
            )
            return (
                np.concatenate(parts) if parts else np.zeros(0)
            )

        remaining = config.maximum_evaluations - spent
        if remaining <= 0:
            return RelationalSolverBudgetObstruction(
                owner_address=problem.owner_address,
                required_evaluations=config.maximum_evaluations,
                observed_evaluations=spent,
                best_residual=float(
                    np.linalg.norm(residual_fn(current))
                ),
            )
        attempt = _run_least_squares(residual_fn, current, remaining)
        spent = spent + attempt.evaluations
        solved_view = problem.evaluator(attempt.solution)
        recomputed = _active_sites(problem.declarations, solved_view)
        measurements = _measurements(
            problem.declarations, conventions, recomputed, solved_view, config
        )
        if _active_signature(recomputed) == _active_signature(active):
            return attempt, seen, measurements, spent
        active = recomputed
        current = np.array(attempt.solution)

    final_active = _active_sites(
        problem.declarations, problem.evaluator(tuple(current))
    )
    final_residuals = tuple(
        float(
            np.linalg.norm(
                _relation_components(
                    declaration,
                    conventions.get(declaration.relation_id),
                    sites,
                    problem.evaluator(tuple(current)),
                    config,
                )
            )
        )
        for declaration, sites in zip(problem.declarations, final_active)
    )
    return BranchSearchBudgetObstruction(
        relation_id=problem.declarations[0].relation_id
        if problem.declarations
        else "",
        required_active_sets=config.maximum_active_set_resolves,
        observed_active_sets=len(seen),
        best_residual=float(np.linalg.norm(np.array(final_residuals))),
    )


def _rank_diagnosis(
    problem: LocalRelationProblem,
    attempt: SolverAttempt,
    variables: tuple[VariableSpec, ...],
    config: RelationalSolveConfig,
) -> UnderconstrainedRelationObstruction | int:
    jacobian = np.array(attempt.jacobian)
    if jacobian.size == 0 or len(variables) == 0:
        return 0
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(
        np.sum(singular_values > config.singular_value_tolerance)
    )
    if rank >= len(variables):
        return rank
    _, _, vt = np.linalg.svd(jacobian)
    null_vectors = vt[rank:]
    base_poses = problem.evaluator(attempt.solution).poses()

    def moves_pose(direction: np.ndarray) -> bool:
        trial = tuple(
            value + float(delta)
            for value, delta in zip(attempt.solution, direction)
        )
        trial_poses = problem.evaluator(trial).poses()
        return any(
            float(np.linalg.norm(np.asarray(head) - np.asarray(base_poses[bone][0])))
            > config.position_tolerance
            for bone, (head, _) in trial_poses.items()
        )

    if any(moves_pose(vector) for vector in null_vectors):
        return UnderconstrainedRelationObstruction(
            owner_address=problem.owner_address,
            variable_count=len(variables),
            rank=rank,
            free_dimension_count=len(variables) - rank,
            parameter_addresses=tuple(
                variable.address for variable in variables
            ),
        )
    return rank


def solve_local(
    problem: LocalRelationProblem, config: RelationalSolveConfig
) -> LocalSolveResult:
    if not problem.declarations:
        return AcceptedLocalSolve(
            solution=tuple(
                variable.initial for variable in problem.base_variables
            ),
            variables=problem.base_variables,
            receipt=LocalSolveReceipt(
                owner_address=problem.owner_address,
                measurements=(),
                conventions=(),
                stage=DeepeningStage.LOCAL,
                depth=0,
                variable_count=len(problem.base_variables),
                rank=0,
                active_site_history=(),
                evaluations=0,
                maximum_normalized_residual=0.0,
            ),
        )

    initial_full = tuple(
        variable.initial
        for variable in _cumulative_frontiers(
            problem.base_variables, problem.frontiers
        )[-1][2]
    )
    initial_view = problem.evaluator(initial_full)
    choices = select_conventions(
        problem.declarations, initial_view, config
    )
    conventions = _conventions_by_id(choices)

    conflicts = _fixed_conflicts(problem, conventions, config)
    if conflicts:
        return conflicts[0]

    spent = 0
    last_measurements: tuple[RelationMeasurement, ...] = ()
    last_stage, last_depth = DeepeningStage.LOCAL, 0

    for stage, depth, variables in _cumulative_frontiers(
        problem.base_variables, problem.frontiers
    ):
        narrowed = LocalRelationProblem(
            owner_address=problem.owner_address,
            declarations=problem.declarations,
            evaluator=_project_evaluator(
                problem, variables
            ),
            base_variables=variables,
            frontiers=(),
        )
        outcome = _solve_active_sets(
            narrowed, conventions, variables, config, spent
        )
        last_stage, last_depth = stage, depth
        if isinstance(
            outcome,
            (
                RelationalSolverBudgetObstruction,
                BranchCycleObstruction,
                BranchSearchBudgetObstruction,
            ),
        ):
            return outcome
        attempt, history, measurements, spent = outcome
        last_measurements = measurements
        if all(measurement.satisfied for measurement in measurements):
            rank = _rank_diagnosis(narrowed, attempt, variables, config)
            if isinstance(rank, UnderconstrainedRelationObstruction):
                return rank
            return AcceptedLocalSolve(
                solution=attempt.solution,
                variables=variables,
                receipt=LocalSolveReceipt(
                    owner_address=problem.owner_address,
                    measurements=measurements,
                    conventions=choices,
                    stage=stage,
                    depth=depth,
                    variable_count=len(variables),
                    rank=rank,
                    active_site_history=history,
                    evaluations=spent,
                    maximum_normalized_residual=max(
                        (
                            measurement.normalized_residual
                            for measurement in measurements
                        ),
                        default=0.0,
                    ),
                ),
            )

    worst = max(
        last_measurements,
        key=lambda measurement: measurement.normalized_residual,
        default=None,
    )
    return RelationalSolveExhaustedObstruction(
        owner_address=problem.owner_address,
        stage=last_stage.value,
        depth=last_depth,
        relation_id=worst.relation_id if worst else "",
        required=worst.required if worst else 0.0,
        observed=worst.observed if worst else 0.0,
        maximum_normalized_residual=(
            worst.normalized_residual if worst else 0.0
        ),
        attempted_evaluations=spent,
    )


def _project_evaluator(
    problem: LocalRelationProblem, variables: tuple[VariableSpec, ...]
) -> Evaluator:
    full = _cumulative_frontiers(
        problem.base_variables, problem.frontiers
    )[-1][2]
    positions = {
        variable.address: index for index, variable in enumerate(full)
    }
    defaults = tuple(variable.initial for variable in full)

    def evaluate(values: tuple[float, ...]) -> GeometryView:
        assignment = dict(
            zip((variable.address for variable in variables), values)
        )
        complete = tuple(
            assignment.get(variable.address, defaults[positions[variable.address]])
            for variable in full
        )
        return problem.evaluator(complete)

    return evaluate


__all__ = [
    "AcceptedLocalSolve",
    "ConventionChoice",
    "DeepeningStage",
    "Evaluator",
    "Frontier",
    "GeometryView",
    "LocalRelationProblem",
    "LocalSolveReceipt",
    "LocalSolveResult",
    "RelationMeasurement",
    "RelationalSolveConfig",
    "SolverAttempt",
    "VariableKind",
    "VariableSpec",
    "measure_relation",
    "select_conventions",
    "solve_local",
]
