from __future__ import annotations

import json
from dataclasses import asdict
from typing import assert_never

from golem.kernel.engine.types import (
    GeometryObstruction,
    MuscleFormationObstruction,
)
from golem.senses.model import (
    RejectedSenses,
    SensesObstruction,
)


def project_senses_obstruction(
    obstruction: SensesObstruction,
) -> dict[str, object]:
    match obstruction:
        case GeometryObstruction():
            return {
                "obstruction": type(obstruction).__name__,
                **asdict(obstruction),
            }
        case MuscleFormationObstruction():
            return {
                "obstruction": type(obstruction).__name__,
                **asdict(obstruction),
            }
        case _ as unreachable:
            assert_never(unreachable)


def project_rejected_senses(
    rejected: RejectedSenses,
) -> dict[str, object]:
    return {
        "status": "senses_obstructed",
        "obstructions": tuple(
            map(project_senses_obstruction, rejected.obstructions)
        ),
    }


def rejected_senses_text(rejected: RejectedSenses) -> str:
    return (
        json.dumps(
            project_rejected_senses(rejected),
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


__all__ = (
    "project_rejected_senses",
    "project_senses_obstruction",
    "rejected_senses_text",
)
