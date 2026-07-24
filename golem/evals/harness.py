"""Shared typed outcome, renderer, exit law, and subprocess boundary."""

from __future__ import annotations

import json
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import assert_never


class AcceptedEvaluation(ABC):
    @property
    def guard_passes(self) -> bool:
        return True

    @abstractmethod
    def evaluation_payload(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class EvaluationObstruction:
    address: str
    reason: str


type EvaluationOutcome[evaluation: AcceptedEvaluation] = (
    evaluation | EvaluationObstruction
)


@dataclass(frozen=True)
class RenderedEvaluation:
    stdout: str
    stderr: str
    exit_code: int


@dataclass(frozen=True)
class ProcessReceipt:
    command: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str


def _stable_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"


def render_outcome(
    outcome: EvaluationOutcome[AcceptedEvaluation],
    *,
    enforce_guard: bool = False,
) -> RenderedEvaluation:
    match outcome:
        case EvaluationObstruction(address, reason):
            payload = {
                "address": address,
                "reason": reason,
                "status": "obstructed",
            }
            return RenderedEvaluation(
                stdout="",
                stderr=_stable_json(payload),
                exit_code=2,
            )
        case AcceptedEvaluation() as evaluation:
            payload = {"status": "accepted", **evaluation.evaluation_payload()}
            return RenderedEvaluation(
                stdout=_stable_json(payload),
                stderr="",
                exit_code=(
                    0
                    if not enforce_guard or evaluation.guard_passes
                    else 1
                ),
            )
        case _ as unreachable:
            assert_never(unreachable)


def render_outcome_text(
    outcome: EvaluationOutcome[AcceptedEvaluation],
) -> str:
    rendered = render_outcome(outcome)
    return (rendered.stdout or rendered.stderr).rstrip("\n")


def emit_rendered_evaluation(rendered: RenderedEvaluation) -> int:
    sys.stdout.write(rendered.stdout)
    sys.stderr.write(rendered.stderr)
    return rendered.exit_code


def run_process(command: Sequence[str]) -> ProcessReceipt:
    completed = subprocess.run(
        tuple(command),
        check=False,
        capture_output=True,
        text=True,
    )
    return ProcessReceipt(
        command=tuple(command),
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
