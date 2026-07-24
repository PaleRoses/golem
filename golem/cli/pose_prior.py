from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from golem.cli.model import (
    CommandDescriptor,
    CommandResult,
    CommandRunner,
    PositionalArgument,
)
from golem.cli.spec import (
    BODY_DOCUMENT,
    SpecSourceObstruction,
    classify_document,
    load_spec,
    render_source_obstruction,
    render_unrecognized_document,
)
from golem.kernel.anatomy.quadruped import (
    RejectedQuadrupedPrior,
    diagnose_quadruped_prior,
    quadruped_prior_to_dict,
)


@dataclass(frozen=True)
class PosePriorArguments:
    spec_path: Path


def _pose_prior_arguments(namespace: Namespace) -> PosePriorArguments:
    return PosePriorArguments(spec_path=namespace.spec)


def run(namespace: Namespace) -> CommandResult:
    arguments = _pose_prior_arguments(namespace)
    source = load_spec(arguments.spec_path)
    if isinstance(source, SpecSourceObstruction):
        return CommandResult(1, stderr=render_source_obstruction(source))
    unrecognized = classify_document(source, (BODY_DOCUMENT,))
    if unrecognized is not None:
        return CommandResult(1, stderr=render_unrecognized_document(unrecognized))
    report = diagnose_quadruped_prior(source.payload)
    return CommandResult(
        exit_code=1 if isinstance(report, RejectedQuadrupedPrior) else 0,
        stdout=json.dumps(
            quadruped_prior_to_dict(report),
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


COMMAND = CommandDescriptor(
    name="pose-prior",
    help_line="diagnose quadruped proportions and propose rest-direction repairs",
    runner=CommandRunner(
        arguments=(
            PositionalArgument(
                "spec",
                "body/0.3 quadruped skeleton document",
                Path,
            ),
        ),
        evaluate=run,
    ),
)
