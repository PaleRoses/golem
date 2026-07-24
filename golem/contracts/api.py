"""Compatibility API derived from the typed contract verdict algebra."""

from __future__ import annotations

from golem.contracts.verdicts import evaluate_verdict, evaluate_verdict_pack
from golem.contracts.views import verdict_record
from golem.senses.model import Senses


def evaluate_clause(clause: dict[str, object], senses: Senses) -> dict[str, object]:
    return verdict_record(evaluate_verdict(clause, senses))


def evaluate_pack(
    clauses: list[dict[str, object]], senses: Senses
) -> list[dict[str, object]]:
    return [verdict_record(verdict) for verdict in evaluate_verdict_pack(clauses, senses)]
