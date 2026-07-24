"""Applicative lowering from untyped clause payloads into checked clauses."""

from __future__ import annotations

from itertools import chain
from typing import assert_never

from golem.addressing.scope import (
    Bone,
    Chain,
    Contact,
    Element,
    Landmark,
    Mount,
    Part,
    Port,
    Region,
    RejectedScope,
    Scope,
    ScopeObstruction,
    Whole,
    World,
)
from golem.addressing.scope_grammar import parse_scope
from golem.contracts.metrics import resolve_metric
from golem.contracts.model import (
    Accepted,
    CheckedClause,
    ClauseIdentity,
    ClauseResult,
    MalformedClauseObstruction,
    Rejected,
    RejectedClause,
    Result,
    UnknownMetricObstruction,
    result_obstructions,
)
from golem.contracts.operators import lower_operator


def lower_clause(source: dict[str, object]) -> ClauseResult:
    identity = _clause_identity(source)
    metric_result = _metric(source.get("metric"))
    operator_result = lower_operator(
        source.get("operator"),
        source.get("value"),
        source.get("tolerance", 0),
    )
    scopes_result = _scopes(source.get("scope"))
    unit_result = _optional_string(source.get("unit"), "unit")
    knob_result = _optional_string(source.get("knob"), "knob")
    direction_result = _optional_string(source.get("direction"), "direction")
    view_result = _optional_string(source.get("view"), "view")
    results = (
        metric_result,
        operator_result,
        scopes_result,
        unit_result,
        knob_result,
        direction_result,
        view_result,
    )
    obstructions = tuple(chain.from_iterable(map(result_obstructions, results)))
    match results:
        case (
            Accepted(metric),
            Accepted(operator),
            Accepted(scopes),
            Accepted(unit),
            Accepted(knob),
            Accepted(direction),
            Accepted(view),
        ):
            return CheckedClause(
                identity,
                scopes,
                metric,
                operator,
                unit,
                knob,
                direction,
                view,
            )
        case _:
            return RejectedClause(identity, obstructions)


def _clause_identity(source: dict[str, object]) -> ClauseIdentity:
    raw_metric = source.get("metric")
    metric = raw_metric if isinstance(raw_metric, str) else "?"
    raw_id = source.get("id", metric)
    clause_id = raw_id if isinstance(raw_id, str) else str(raw_id)
    return ClauseIdentity(clause_id, metric)


def _metric(raw_metric: object):
    if not isinstance(raw_metric, str) or not raw_metric:
        return Rejected(
            (MalformedClauseObstruction("metric", "expected a metric name"),)
        )
    descriptor = resolve_metric(raw_metric)
    return (
        Accepted(descriptor)
        if descriptor is not None
        else Rejected((UnknownMetricObstruction(raw_metric),))
    )


def _scopes(raw_scopes: object) -> Result[tuple[Scope, ...]]:
    if not isinstance(raw_scopes, (list, tuple)):
        return Rejected(
            (MalformedClauseObstruction("scope", "expected a scope sequence"),)
        )
    results = tuple(
        _scope(raw_scope, index) for index, raw_scope in enumerate(raw_scopes)
    )
    obstructions = tuple(chain.from_iterable(map(result_obstructions, results)))
    return (
        Rejected(obstructions)
        if obstructions
        else Accepted(
            tuple(
                result.value
                for result in results
                if isinstance(result, Accepted)
            )
        )
    )


def _scope(raw_scope: object, index: int) -> Result[Scope]:
    if not isinstance(raw_scope, str):
        return Rejected(
            (
                MalformedClauseObstruction(
                    f"scope[{index}]", "expected a scope token"
                ),
            )
        )
    match parse_scope(raw_scope):
        case RejectedScope(obstructions):
            return Rejected(obstructions)
        case Bone(_, str()) | Contact(_, str()) | Element() | Port() | Mount() | Region():
            return Rejected(
                (
                    ScopeObstruction(
                        raw_scope,
                        "assembly-qualified address is not an assertion scope",
                    ),
                )
            )
        case (Whole() | World() | Part() | Bone(_, None) | Landmark() | Chain() | Contact(_, None)) as scope:
            return Accepted(scope)
        case _ as unreachable:
            assert_never(unreachable)


def _optional_string(raw_value: object, field: str) -> Result[str | None]:
    return (
        Accepted(raw_value)
        if raw_value is None or isinstance(raw_value, str)
        else Rejected(
            (MalformedClauseObstruction(field, "expected a string when present"),)
        )
    )
