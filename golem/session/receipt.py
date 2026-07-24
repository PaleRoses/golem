"""Deterministic transaction-receipt projection."""

from __future__ import annotations

from itertools import chain

from golem.senses.proprio.format import est_tokens, f2, f3


_ANOMALY_DELTA_CAP = 8


def _clause_key(record: dict, index: int) -> str:
    return record.get("id") or record.get("clause", {}).get("id") or f"clause[{index}]"


def _index(records) -> dict[str, dict]:
    return {_clause_key(record, index): record for index, record in enumerate(records)}


def _anomaly_key(anomaly: dict) -> tuple[str, str]:
    return str(anomaly.get("det")), str(anomaly.get("addr"))


def _senses_digest_delta(previous, current) -> tuple[str, ...]:
    if previous is None or previous.senses is None or current.senses is None:
        return ()
    previous_global = previous.senses.global_dims
    current_global = current.senses.global_dims
    dimensions = tuple(
        f"{label} {f2(old)} -> {f2(new)}"
        for label, old, new in (
            ("W", previous_global.W, current_global.W),
            ("H", previous_global.H, current_global.H),
            ("D", previous_global.D, current_global.D),
            ("HW", previous_global.HW, current_global.HW),
        )
        if f2(old) != f2(new)
    )
    components = (
        (
            f"components {previous.senses.n_components} -> {current.senses.n_components}",
        )
        if previous.senses.n_components != current.senses.n_components
        else ()
    )
    curvature = (
        (f"k {f3(previous_global.k)} -> {f3(current_global.k)}",)
        if f3(previous_global.k) != f3(current_global.k)
        else ()
    )
    return (*dimensions, *components, *curvature)


def _solved_delta(previous, current) -> tuple[str, ...]:
    if previous is None or previous.receipt is None or current.receipt is None:
        return ()
    previous_solved = previous.receipt["solved"]
    current_solved = current.receipt["solved"]
    ground = (
        (
            f"ground_lift {previous_solved['ground_lift']} -> {current_solved['ground_lift']}",
        )
        if previous_solved["ground_lift"] != current_solved["ground_lift"]
        else ()
    )
    previous_goals = {
        goal.get("id"): goal for goal in previous_solved.get("goals", [])
    }

    def changed_goal(goal: dict) -> str | None:
        old = previous_goals.get(goal.get("id"))
        old_residual = old.get("residual") if old else None
        new_residual = goal.get("residual")
        return (
            f"goal {goal.get('id')} residual "
            f"{f3(old_residual) if old_residual is not None else '?'} -> "
            f"{f3(new_residual or 0.0)}"
            if old is None or f3(old_residual or 0.0) != f3(new_residual or 0.0)
            else None
        )

    goals = tuple(
        rendered
        for goal in current_solved.get("goals", [])
        if (rendered := changed_goal(goal)) is not None
    )
    return (*ground, *goals)


def compute(previous, current, entries: list[dict]) -> str:
    entry_lines = tuple(
        f"txn {entry['txn']}  {entry['label']}" for entry in entries
    )
    if current.error is not None:
        lines = (
            *entry_lines,
            f"COMPILE ERROR  {current.error}",
            "(state kept; undo or fix -- the session survives)",
        )
        text = "\n".join(lines)
        return text + f"\nest_tokens: {est_tokens(text)}"
    previous_records = (
        _index(previous.records)
        if previous is not None and previous.error is None
        else {}
    )
    current_records = _index(current.records)
    bad = frozenset({"fail", "unmeasurable"})
    newly_bad = tuple(
        key
        for key, record in current_records.items()
        if record["status"] in bad
        and (
            key not in previous_records
            or previous_records[key]["status"] not in bad
        )
    )
    cleared = tuple(
        key
        for key, record in previous_records.items()
        if record["status"] in bad
        and (
            key not in current_records
            or current_records[key]["status"] == "pass"
        )
    )
    cleared_lines = (
        (f"cleared {len(cleared)}: {', '.join(cleared)}",) if cleared else ()
    )
    newly_bad_lines = (
        (
            f"new {len(newly_bad)}:",
            *(
                "  "
                + current_records[key].get(
                    "human", f"{key}: {current_records[key]['status']}"
                )
                for key in newly_bad
            ),
        )
        if newly_bad
        else ()
    )
    failures = sum(record["status"] == "fail" for record in current.records)
    passes = sum(record["status"] == "pass" for record in current.records)
    unmeasurable = sum(
        record["status"] == "unmeasurable" for record in current.records
    )
    contract = f"contract {failures} FAIL / {passes} pass" + (
        f" / {unmeasurable} unmeasurable" if unmeasurable else ""
    )
    previous_anomalies = {
        _anomaly_key(anomaly): anomaly
        for anomaly in (previous.anomalies if previous is not None else ())
    }
    current_anomalies = {
        _anomaly_key(anomaly): anomaly for anomaly in current.anomalies
    }
    anomaly_lines = tuple(
        chain(
            (
                f"anomaly + {current_anomalies[key].get('det')} @ "
                f"{current_anomalies[key].get('addr')}: "
                f"{current_anomalies[key].get('detail')}"
                for key in sorted(current_anomalies.keys() - previous_anomalies.keys())
            ),
            (
                f"anomaly - {previous_anomalies[key].get('det')} @ "
                f"{previous_anomalies[key].get('addr')}"
                for key in sorted(previous_anomalies.keys() - current_anomalies.keys())
            ),
        )
    )
    bounded_anomalies = (
        (
            *anomaly_lines[:_ANOMALY_DELTA_CAP],
            f"anomaly ... +{len(anomaly_lines) - _ANOMALY_DELTA_CAP} more deltas"
            ' -- outline "" 1 constraint',
        )
        if len(anomaly_lines) > _ANOMALY_DELTA_CAP
        else anomaly_lines
    )
    digest = (*_senses_digest_delta(previous, current), *_solved_delta(previous, current))
    digest_lines = (("senses  " + " | ".join(digest)),) if digest else ()
    lines = (
        *entry_lines,
        *cleared_lines,
        *newly_bad_lines,
        contract,
        *bounded_anomalies,
        *digest_lines,
    )
    text = "\n".join(lines)
    return text + f"\nest_tokens: {est_tokens(text)}"
