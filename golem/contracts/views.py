"""Human and JSON projections of authoritative contract verdicts."""

from __future__ import annotations

from typing import assert_never

from golem.addressing.scope import ScopeObstruction
from golem.contracts.model import (
    ContractObstruction,
    Failed,
    InvalidMetricDescriptorObstruction,
    InvalidScopeShapeObstruction,
    MalformedClauseObstruction,
    MissingSenseObstruction,
    Passed,
    UnknownMetricObstruction,
    Unmeasurable,
    UnsupportedOperatorObstruction,
    Verdict,
)
from golem.contracts.operators import operator_name, operator_value, target_text
from golem.senses.proprio.format import f3


def verdict_record(verdict: Verdict) -> dict[str, object]:
    match verdict:
        case Passed(clause, measurement):
            status = "pass"
            magnitude = 0.0
            gradient_record: list[dict[str, object]] = []
            skeleton_offenders: tuple[str, ...] = ()
        case Failed(
            clause,
            measurement,
            violation,
            gradient,
            skeleton_offenders,
        ):
            status = "fail"
            magnitude = round(float(violation.magnitude), 3)
            gradient_record = [
                {
                    "knob": gradient.knob,
                    "direction": gradient.direction,
                    "estimate": round(gradient.estimate, 3),
                }
            ]
        case Unmeasurable(identity, (first_obstruction, *_) as obstructions):
            return {
                "id": identity.clause_id,
                "clause": identity.clause_id,
                "metric": identity.authored_metric,
                "status": "unmeasurable",
                "measured": None,
                "missing": _obstruction_address(first_obstruction),
                "part_scopes": [],
                "obstructions": [
                    _obstruction_record(obstruction)
                    for obstruction in obstructions
                ],
                "human": verdict_human(verdict),
            }
        case _ as unreachable:
            assert_never(unreachable)
    return {
        "id": clause.identity.clause_id,
        "clause": clause.identity.clause_id,
        "metric": clause.identity.authored_metric,
        "status": status,
        "measured": round(float(measurement.measured), 3),
        "target": {
            "operator": operator_name(clause.operator),
            "value": operator_value(clause.operator),
        },
        "unit": measurement.unit,
        "magnitude": magnitude,
        "offenders": list(measurement.offenders),
        "detail": measurement.detail,
        "gradient": gradient_record,
        "part_scopes": list(measurement.part_scopes),
        "human": verdict_human(verdict),
        **(
            {"skeleton_offenders": list(skeleton_offenders)}
            if skeleton_offenders
            else {}
        ),
    }


def verdict_human(verdict: Verdict) -> str:
    match verdict:
        case Passed(clause, measurement):
            return _measured_human(
                clause.identity.clause_id,
                clause.identity.authored_metric,
                measurement,
                target_text(clause.operator),
            ) + "  [pass]"
        case Failed(clause, measurement, violation, gradient, skeleton_offenders):
            base = _measured_human(
                clause.identity.clause_id,
                clause.identity.authored_metric,
                measurement,
                target_text(clause.operator),
            )
            correction = (
                f" -> {gradient.knob} {gradient.direction} "
                f"(est {f3(gradient.estimate)}, x{f3(violation.magnitude)})"
            )
            provenance = (
                "  [@ " + ", ".join(skeleton_offenders) + "]"
                if skeleton_offenders
                else ""
            )
            return base + correction + provenance
        case Unmeasurable(identity, (first_obstruction, *_)):
            return (
                f"{identity.clause_id} {identity.authored_metric}: "
                f"UNMEASURABLE ({_obstruction_human(first_obstruction)})"
            )
        case _ as unreachable:
            assert_never(unreachable)


def _measured_human(clause_id, metric, measurement, target) -> str:
    return (
        f"{clause_id} {metric} @ {_address_label(measurement)}: "
        f"{f3(measurement.measured)} {measurement.unit}, target {target}"
    )


def _address_label(measurement) -> str:
    offenders = measurement.offenders
    short = tuple(offender.split(":", 1)[-1] for offender in offenders)
    if len(measurement.part_scopes) == 2:
        return "~".join(measurement.part_scopes)
    match short:
        case (left, right):
            return f"{left}<->{right}"
        case (first, *_):
            return first
        case ():
            return "?"


def _obstruction_address(obstruction: ContractObstruction) -> str:
    match obstruction:
        case ScopeObstruction(source, _):
            return source
        case MalformedClauseObstruction(field, _):
            return f"clause:{field}"
        case UnknownMetricObstruction(metric):
            return f"metric:{metric}"
        case UnsupportedOperatorObstruction(operator):
            return f"operator:{operator}"
        case MissingSenseObstruction(address):
            return address
        case InvalidScopeShapeObstruction(_, expected):
            return expected
        case InvalidMetricDescriptorObstruction(metric, _):
            return f"metric:{metric}"
        case _ as unreachable:
            assert_never(unreachable)


def _obstruction_human(obstruction: ContractObstruction) -> str:
    match obstruction:
        case UnknownMetricObstruction(metric):
            return f"unknown metric {metric}"
        case ScopeObstruction(source, reason):
            return f"invalid scope {source}: {reason}"
        case MalformedClauseObstruction(field, reason):
            return f"invalid {field}: {reason}"
        case UnsupportedOperatorObstruction(operator):
            return f"unsupported operator {operator}"
        case MissingSenseObstruction(address):
            return f"missing {address}"
        case InvalidScopeShapeObstruction(_, expected):
            return f"missing {expected}"
        case InvalidMetricDescriptorObstruction(metric, reason):
            return f"invalid metric descriptor {metric}: {reason}"
        case _ as unreachable:
            assert_never(unreachable)


def _obstruction_record(obstruction: ContractObstruction) -> dict[str, str]:
    match obstruction:
        case ScopeObstruction(source, reason):
            return {"kind": "scope", "source": source, "reason": reason}
        case MalformedClauseObstruction(field, reason):
            return {"kind": "clause", "source": field, "reason": reason}
        case UnknownMetricObstruction(metric):
            return {
                "kind": "metric",
                "source": metric,
                "reason": "unknown metric",
            }
        case UnsupportedOperatorObstruction(operator):
            return {
                "kind": "operator",
                "source": operator,
                "reason": "unsupported operator",
            }
        case MissingSenseObstruction(address):
            return {
                "kind": "sense",
                "source": address,
                "reason": "missing measurement",
            }
        case InvalidScopeShapeObstruction(metric, expected):
            return {
                "kind": "metric_scope",
                "source": metric,
                "reason": f"expected {expected}",
            }
        case InvalidMetricDescriptorObstruction(metric, reason):
            return {
                "kind": "metric_descriptor",
                "source": metric,
                "reason": reason,
            }
        case _ as unreachable:
            assert_never(unreachable)
