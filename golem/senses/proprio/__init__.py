"""Immutable analytic proprioception with a demand-loaded CLI boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from golem.senses.proprio.anomaly import detect_anomalies
from golem.senses.proprio.build import build_senses
from golem.senses.proprio.format import est_tokens, f2, f3, s2, s3
from golem.senses.proprio.geometry import (
    field_at,
    instance_bbox,
    part_min_radius,
    sample_instance,
)
from golem.senses.proprio.posture import GROUND_Y
from golem.senses.proprio.render import (
    SketchSurfaceFormationObstruction,
    render_receipt,
    render_section,
    render_sketch,
)
from golem.senses.model import RejectedSenses, SensesResult

from golem.senses.proprio.anomaly import _ANOMALY_CAP as _ANOMALY_CAP
from golem.senses.proprio.anomaly import _apply_suppression as _apply_suppression
from golem.senses.proprio.geometry import _part_volume_centroid as _part_volume_centroid
from golem.senses.proprio.posture import _union_len as _union_len
from golem.senses.proprio.render import _SKETCH_ROWS as _SKETCH_ROWS

if TYPE_CHECKING:
    from golem.senses.proprio.cli import main as main, run


def __getattr__(name: str) -> object:
    match name:
        case "run" | "main":
            from golem.senses.proprio.cli import main, run

            return run if name == "run" else main
        case _:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted((*globals(), "run", "main"))


__all__ = [
    "build_senses",
    "run",
    "detect_anomalies",
    "render_receipt",
    "render_section",
    "render_sketch",
    "RejectedSenses",
    "SensesResult",
    "SketchSurfaceFormationObstruction",
    "est_tokens",
    "f2",
    "f3",
    "s2",
    "s3",
    "sample_instance",
    "instance_bbox",
    "part_min_radius",
    "field_at",
    "GROUND_Y",
]
