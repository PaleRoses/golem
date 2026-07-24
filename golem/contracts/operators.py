"""Checked construction and comparison algebra for contract operators."""

from __future__ import annotations

import math
from itertools import chain
from typing import assert_never

from golem.contracts.model import (
    Accepted,
    Comparison,
    EqualOperator,
    MalformedClauseObstruction,
    MinimumOperator,
    Operator,
    OperatorKind,
    Rejected,
    Result,
    Satisfied,
    UnsupportedOperatorObstruction,
    UpperBoundOperator,
    Violated,
    WithinOperator,
    result_obstructions,
)
from golem.senses.proprio.format import f3


def lower_operator(
    raw_name: object, raw_value: object, raw_tolerance: object
) -> Result[Operator]:
    kind_result = _operator_kind(raw_name)
    tolerance_result = _nonnegative_number(
        0 if raw_tolerance is None else raw_tolerance,
        "tolerance",
    )
    target_result = (
        _within_target(raw_value)
        if raw_name == OperatorKind.WITHIN.value
        else _number(raw_value, "value")
    )
    obstructions = tuple(
        chain.from_iterable(
            map(
                result_obstructions,
                (kind_result, tolerance_result, target_result),
            )
        )
    )
    match kind_result, tolerance_result, target_result:
        case Accepted(kind), Accepted(tolerance), Accepted((lower, upper)):
            if kind is OperatorKind.WITHIN:
                return Accepted(WithinOperator(lower, upper, tolerance))
        case Accepted(kind), Accepted(tolerance), Accepted(value):
            match kind:
                case OperatorKind.MAXIMUM | OperatorKind.MAX_DISTANCE | OperatorKind.MAX_ANGLE:
                    return Accepted(UpperBoundOperator(kind, value, tolerance))
                case OperatorKind.MINIMUM:
                    return Accepted(MinimumOperator(value, tolerance))
                case OperatorKind.EQUAL:
                    return Accepted(EqualOperator(value, tolerance))
                case OperatorKind.WITHIN:
                    pass
                case _ as unreachable:
                    assert_never(unreachable)
        case _:
            pass
    return Rejected(obstructions)


def _operator_kind(raw_name: object) -> Result[OperatorKind]:
    if not isinstance(raw_name, str):
        return Rejected(
            (MalformedClauseObstruction("operator", "expected an operator name"),)
        )
    match raw_name:
        case "maximum":
            return Accepted(OperatorKind.MAXIMUM)
        case "max_distance":
            return Accepted(OperatorKind.MAX_DISTANCE)
        case "max_angle":
            return Accepted(OperatorKind.MAX_ANGLE)
        case "minimum":
            return Accepted(OperatorKind.MINIMUM)
        case "equal":
            return Accepted(OperatorKind.EQUAL)
        case "within":
            return Accepted(OperatorKind.WITHIN)
        case _:
            return Rejected((UnsupportedOperatorObstruction(raw_name),))


def _within_target(raw_value: object) -> Result[tuple[float, float]]:
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 2:
        return Rejected(
            (
                MalformedClauseObstruction(
                    "value", "within requires exactly two numeric bounds"
                ),
            )
        )
    lower_result = _number(raw_value[0], "value[0]")
    upper_result = _number(raw_value[1], "value[1]")
    obstructions = tuple(
        chain.from_iterable(
            map(result_obstructions, (lower_result, upper_result))
        )
    )
    match lower_result, upper_result:
        case Accepted(lower), Accepted(upper) if lower <= upper:
            return Accepted((lower, upper))
        case Accepted(), Accepted():
            return Rejected(
                (MalformedClauseObstruction("value", "lower bound exceeds upper bound"),)
            )
        case _:
            return Rejected(obstructions)


def _number(raw_value: object, field: str) -> Result[float]:
    if isinstance(raw_value, bool):
        return Rejected(
            (MalformedClauseObstruction(field, "expected a finite number"),)
        )
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return Rejected(
            (MalformedClauseObstruction(field, "expected a finite number"),)
        )
    return (
        Accepted(value)
        if math.isfinite(value)
        else Rejected(
            (MalformedClauseObstruction(field, "expected a finite number"),)
        )
    )


def _nonnegative_number(raw_value: object, field: str) -> Result[float]:
    result = _number(raw_value, field)
    match result:
        case Accepted(value) if value >= 0:
            return result
        case Accepted():
            return Rejected(
                (MalformedClauseObstruction(field, "expected a nonnegative number"),)
            )
        case Rejected():
            return result


def apply_operator(operator: Operator, measured: float | int) -> Comparison:
    value = float(measured)
    match operator:
        case UpperBoundOperator(_, maximum, tolerance):
            return (
                Satisfied()
                if value <= maximum + tolerance
                else Violated(
                    "-",
                    value - maximum,
                    (value - maximum) / abs(maximum) if maximum else value,
                )
            )
        case MinimumOperator(minimum, tolerance):
            return (
                Satisfied()
                if value >= minimum - tolerance
                else Violated(
                    "+",
                    minimum - value,
                    (minimum - value) / abs(minimum) if minimum else minimum - value,
                )
            )
        case EqualOperator(target, tolerance):
            delta = value - target
            return (
                Satisfied()
                if abs(delta) <= tolerance
                else Violated(
                    "-" if delta > 0 else "+",
                    abs(delta),
                    abs(delta) / abs(target) if target else abs(delta),
                )
            )
        case WithinOperator(lower, upper, tolerance):
            if lower - tolerance <= value <= upper + tolerance:
                return Satisfied()
            return (
                Violated(
                    "+",
                    lower - value,
                    (lower - value) / abs(lower) if lower else lower - value,
                )
                if value < lower
                else Violated(
                    "-",
                    value - upper,
                    (value - upper) / abs(upper) if upper else value - upper,
                )
            )
        case _ as unreachable:
            assert_never(unreachable)


def operator_name(operator: Operator) -> str:
    match operator:
        case UpperBoundOperator(kind, _, _):
            return kind.value
        case MinimumOperator():
            return OperatorKind.MINIMUM.value
        case EqualOperator():
            return OperatorKind.EQUAL.value
        case WithinOperator():
            return OperatorKind.WITHIN.value
        case _ as unreachable:
            assert_never(unreachable)


def operator_value(operator: Operator) -> float | list[float]:
    match operator:
        case UpperBoundOperator(_, value, _) | MinimumOperator(value, _) | EqualOperator(value, _):
            return value
        case WithinOperator(lower, upper, _):
            return [lower, upper]
        case _ as unreachable:
            assert_never(unreachable)


def target_text(operator: Operator) -> str:
    match operator:
        case UpperBoundOperator(_, value, _):
            return f"<= {f3(value)}"
        case MinimumOperator(value, _):
            return f">= {f3(value)}"
        case EqualOperator(value, _):
            return f"== {f3(value)}"
        case WithinOperator(lower, upper, _):
            return f"[{f3(lower)},{f3(upper)}]"
        case _ as unreachable:
            assert_never(unreachable)
