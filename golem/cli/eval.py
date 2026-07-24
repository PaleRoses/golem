from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from golem.cli.model import (
    CommandDescriptor,
    CommandResult,
    CommandRunner,
    PositionalArgument,
)
from golem.cli.spec import (
    LoadedSpec,
    SpecSourceObstruction,
    is_body_document,
    load_spec,
    render_source_obstruction,
)
from golem.paths import SPECS

if TYPE_CHECKING:
    from golem.evals.harness import (
        AcceptedEvaluation,
        EvaluationObstruction,
        RenderedEvaluation,
    )


type EvaluationOutcome = AcceptedEvaluation | EvaluationObstruction


@dataclass(frozen=True)
class EvaluationDescriptor:
    applies: Callable[[LoadedSpec], bool]
    evaluate: Callable[[LoadedSpec], EvaluationOutcome]


@dataclass(frozen=True)
class EvalArguments:
    spec_path: Path


def _eval_arguments(namespace: Namespace) -> EvalArguments:
    return EvalArguments(spec_path=namespace.spec)


def _is_canonical_path(source: LoadedSpec, path: Path) -> bool:
    return source.path.resolve() == path.resolve()


def _is_anatomy_case(source: LoadedSpec) -> bool:
    return (
        _is_canonical_path(source, SPECS / "knight_body.json")
        and is_body_document(source.payload)
        and isinstance(source.payload.get("anatomy"), dict)
    )


def _is_elemental_flow_case(source: LoadedSpec) -> bool:
    return (
        _is_canonical_path(source, SPECS / "elemental_field.json")
        and isinstance(source.payload.get("field_id"), str)
        and isinstance(source.payload.get("boundary_exprs"), list)
    )


def _is_plate_policy(source: LoadedSpec) -> bool:
    return (
        isinstance(source.payload.get("layout"), str)
        and isinstance(source.payload.get("cell_count"), int)
    )


def _evaluate_anatomy_document(_source: LoadedSpec) -> EvaluationOutcome:
    # Lazy: evaluation cases pull the render stack (trimesh/PIL), which must
    # stay optional for `--help`.
    from golem.evals.eval_anatomy import evaluate_anatomy

    return evaluate_anatomy()


def _evaluate_elemental_flow_document(
    _source: LoadedSpec,
) -> EvaluationOutcome:
    from golem.evals.eval_elemental_flow import evaluate_elemental_flow

    return evaluate_elemental_flow()


def _evaluate_plate_policy(source: LoadedSpec) -> EvaluationOutcome:
    from golem.evals.eval_plate_authoring import evaluate_plate_authoring

    return evaluate_plate_authoring(
        SPECS / "knight_body.json",
        source.path,
    )


EVALUATIONS: tuple[EvaluationDescriptor, ...] = (
    EvaluationDescriptor(_is_anatomy_case, _evaluate_anatomy_document),
    EvaluationDescriptor(
        _is_elemental_flow_case,
        _evaluate_elemental_flow_document,
    ),
    EvaluationDescriptor(_is_plate_policy, _evaluate_plate_policy),
)


def _rendered_command_result(
    rendered: tuple[RenderedEvaluation, ...],
) -> CommandResult:
    return CommandResult(
        exit_code=max(map(lambda report: report.exit_code, rendered)),
        stdout="".join(report.stdout for report in rendered),
        stderr="".join(report.stderr for report in rendered),
    )


def _evaluate_loaded(source: LoadedSpec) -> CommandResult:
    from golem.evals.harness import EvaluationObstruction, render_outcome

    applicable = tuple(
        descriptor for descriptor in EVALUATIONS if descriptor.applies(source)
    )
    rendered = (
        tuple(
            render_outcome(descriptor.evaluate(source))
            for descriptor in applicable
        )
        if applicable
        else (
            render_outcome(
                EvaluationObstruction(
                    str(source.path),
                    "no registered evaluation applies to this document kind",
                )
            ),
        )
    )
    return _rendered_command_result(rendered)


def run(namespace: Namespace) -> CommandResult:
    arguments = _eval_arguments(namespace)
    source = load_spec(arguments.spec_path)
    return (
        CommandResult(1, stderr=render_source_obstruction(source))
        if isinstance(source, SpecSourceObstruction)
        else _evaluate_loaded(source)
    )


COMMAND = CommandDescriptor(
    name="eval",
    help_line="run stable evaluations applicable to a spec",
    runner=CommandRunner(
        arguments=(
            PositionalArgument(
                "spec",
                "document selecting its registered evaluation cases",
                Path,
            ),
        ),
        evaluate=run,
    ),
)
