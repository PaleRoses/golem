"""Typed carriers for contract construction, measurement, and verdicts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, assert_never

from golem.addressing.scope import Scope, ScopeObstruction

if TYPE_CHECKING:
    from golem.senses.model import Senses


@dataclass(frozen=True)
class MalformedClauseObstruction:
    field: str
    reason: str


@dataclass(frozen=True)
class UnknownMetricObstruction:
    metric: str


@dataclass(frozen=True)
class UnsupportedOperatorObstruction:
    operator: str


@dataclass(frozen=True)
class MissingSenseObstruction:
    address: str


@dataclass(frozen=True)
class InvalidScopeShapeObstruction:
    metric: str
    expected: str


@dataclass(frozen=True)
class InvalidMetricDescriptorObstruction:
    metric: str
    reason: str


type ContractObstruction = (
    ScopeObstruction
    | MalformedClauseObstruction
    | UnknownMetricObstruction
    | UnsupportedOperatorObstruction
    | MissingSenseObstruction
    | InvalidScopeShapeObstruction
    | InvalidMetricDescriptorObstruction
)


@dataclass(frozen=True)
class Accepted[value]:
    value: value


@dataclass(frozen=True)
class Rejected:
    obstructions: tuple[ContractObstruction, ...]


type Result[value] = Accepted[value] | Rejected


def result_obstructions[value](
    result: Result[value],
) -> tuple[ContractObstruction, ...]:
    match result:
        case Accepted():
            return ()
        case Rejected(obstructions):
            return obstructions
        case _ as unreachable:
            assert_never(unreachable)


@dataclass(frozen=True)
class ClauseIdentity:
    clause_id: str
    authored_metric: str


@dataclass(frozen=True)
class Measurement:
    measured: float | int
    unit: str
    offenders: tuple[str, ...]
    part_scopes: tuple[str, ...]
    detail: str


class MetricEvaluator(Protocol):
    def __call__(
        self, senses: Senses, clause: CheckedClause
    ) -> Result[Measurement]: ...


class MetricKind(StrEnum):
    WIDTH = "width"
    HEIGHT = "height"
    DEPTH = "depth"
    SPAN = "span"
    CONTACT_ERROR = "contact_error"
    CLEARANCE = "clearance"
    LANDMARK_DISTANCE = "landmark_distance"
    FORWARD_OFFSET = "forward_offset"
    VERTICAL_OFFSET = "vertical_offset"
    SCHEMATIC_OFFSET = "schematic_offset"
    AXIS_FROM_VERTICAL = "axis_from_vertical"
    HEAD_EXPOSURE = "head_exposure"
    HEADS_TALL = "heads_tall"
    CENTROID_SUPPORT_MARGIN = "centroid_support_margin"
    POST_DUST_COMPONENTS = "post_dust_components"


@dataclass(frozen=True)
class MetricDescriptor:
    kind: MetricKind
    evaluator: MetricEvaluator
    axis: int | None = None
    aliases: tuple[str, ...] = ()


class OperatorKind(StrEnum):
    MAXIMUM = "maximum"
    MAX_DISTANCE = "max_distance"
    MAX_ANGLE = "max_angle"
    MINIMUM = "minimum"
    EQUAL = "equal"
    WITHIN = "within"


@dataclass(frozen=True)
class UpperBoundOperator:
    kind: OperatorKind
    value: float
    tolerance: float


@dataclass(frozen=True)
class MinimumOperator:
    value: float
    tolerance: float


@dataclass(frozen=True)
class EqualOperator:
    value: float
    tolerance: float


@dataclass(frozen=True)
class WithinOperator:
    lower: float
    upper: float
    tolerance: float


type Operator = UpperBoundOperator | MinimumOperator | EqualOperator | WithinOperator


@dataclass(frozen=True)
class CheckedClause:
    identity: ClauseIdentity
    scopes: tuple[Scope, ...]
    metric: MetricDescriptor
    operator: Operator
    unit: str | None
    knob: str | None
    direction: str | None
    view: str | None


@dataclass(frozen=True)
class RejectedClause:
    identity: ClauseIdentity
    obstructions: tuple[ContractObstruction, ...]


type ClauseResult = CheckedClause | RejectedClause


@dataclass(frozen=True)
class Satisfied:
    pass


@dataclass(frozen=True)
class Violated:
    direction: str
    estimate: float
    magnitude: float


type Comparison = Satisfied | Violated


@dataclass(frozen=True)
class Gradient:
    knob: str
    direction: str
    estimate: float


@dataclass(frozen=True)
class Passed:
    clause: CheckedClause
    measurement: Measurement


@dataclass(frozen=True)
class Failed:
    clause: CheckedClause
    measurement: Measurement
    violation: Violated
    gradient: Gradient
    skeleton_offenders: tuple[str, ...]


@dataclass(frozen=True)
class Unmeasurable:
    identity: ClauseIdentity
    obstructions: tuple[ContractObstruction, ...]


type Verdict = Passed | Failed | Unmeasurable
