"""Authoritative contract evaluation into a closed verdict algebra."""

from __future__ import annotations

from typing import assert_never

from golem.contracts.clauses import lower_clause
from golem.contracts.model import (
    Accepted,
    CheckedClause,
    Failed,
    Gradient,
    Passed,
    Rejected,
    RejectedClause,
    Satisfied,
    Unmeasurable,
    Verdict,
    Violated,
)
from golem.contracts.operators import apply_operator
from golem.senses.model import Senses


def evaluate_verdict(source: dict[str, object], senses: Senses) -> Verdict:
    match lower_clause(source):
        case RejectedClause(identity, obstructions):
            return Unmeasurable(identity, obstructions)
        case CheckedClause() as clause:
            measurement_result = clause.metric.evaluator(senses, clause)
        case _ as unreachable:
            assert_never(unreachable)
    match measurement_result:
        case Rejected(obstructions):
            return Unmeasurable(clause.identity, obstructions)
        case Accepted(measurement):
            comparison = apply_operator(clause.operator, measurement.measured)
        case _ as unreachable:
            assert_never(unreachable)
    match comparison:
        case Satisfied():
            return Passed(clause, measurement)
        case Violated(direction, estimate, _) as violation:
            match measurement.offenders:
                case (first_offender, *_):
                    default_knob = first_offender
                case ():
                    default_knob = clause.identity.clause_id
            knob = clause.knob or default_knob
            gradient = Gradient(
                knob,
                clause.direction or direction,
                estimate,
            )
            skeleton_offenders = (
                tuple(
                    _skeleton_offender(offender, senses.provenance)
                    for offender in measurement.offenders
                )
                if senses.provenance
                else ()
            )
            return Failed(
                clause,
                measurement,
                violation,
                gradient,
                skeleton_offenders,
            )
        case _ as unreachable:
            assert_never(unreachable)


def evaluate_verdict_pack(
    clauses: list[dict[str, object]], senses: Senses
) -> tuple[Verdict, ...]:
    return tuple(evaluate_verdict(clause, senses) for clause in clauses)


def _skeleton_offender(offender: str, provenance: dict[str, str]) -> str:
    kind, _, value = offender.partition(":")
    return provenance.get(value, offender) if kind == "part" else offender
