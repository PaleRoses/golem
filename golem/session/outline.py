"""Deterministic bounded projection of session state."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from itertools import chain

from golem.addressing.session_grammar import encode_address_segment
from golem.senses.proprio.format import est_tokens, f2


_BUDGET = {1: 350, 2: 900, 3: 2400}


def _bone_children(spec: dict) -> dict[str, tuple[dict, ...]]:
    bones = tuple(spec.get("skeleton", {}).get("bones", []))
    parents = tuple(
        dict.fromkeys(bone.get("parent", "root") for bone in bones)
    )
    return {
        parent: tuple(bone for bone in bones if bone.get("parent", "root") == parent)
        for parent in parents
    }


def _bone_line(bone: dict, indent: int) -> str:
    flesh = ",".join(item.get("name", "?") for item in bone.get("flesh", []))
    degree_of_freedom = bone.get("joint", {}).get("dof", "?")
    mirror = " mirror" if bone.get("mirror") else ""
    padding = "  " * indent
    return (
        f"{padding}skeleton/{encode_address_segment(str(bone['id']))}  "
        f"len {f2(float(bone.get('length', 0.0)))} "
        f"dof {degree_of_freedom}{mirror}"
        + (f"  flesh[{flesh}]" if flesh else "")
    )


def _flesh_lines(address: str, bone: dict, indent: int) -> tuple[str, ...]:
    padding = "  " * indent

    def render(flesh: dict) -> str:
        kind = flesh.get("kind", "?")
        geometry = (
            f" span {flesh.get('span')} radii {flesh.get('radii')}"
            if kind == "gencyl"
            else f" sections {len(flesh.get('sections', ()))}"
            if kind == "loft"
            else f" t {flesh.get('t')} size {flesh.get('size')}"
            if kind in ("blob", "box")
            else ""
        )
        operator = (
            f" operator {flesh['operator']}" if "operator" in flesh else ""
        )
        return (
            f"{padding}{address}/"
            f"{encode_address_segment(str(flesh.get('name', '?')))}  "
            f"{kind}{geometry}{operator}"
        )

    return tuple(render(flesh) for flesh in bone.get("flesh", []))


def _structure(state, scope: str, depth: int) -> tuple[str, ...]:
    spec = state.spec
    skeleton = spec.get("skeleton", {})
    bones = tuple(skeleton.get("bones", []))
    flesh_count = sum(len(bone.get("flesh", [])) for bone in bones) + len(
        skeleton.get("root", {}).get("flesh", [])
    )
    root_id = skeleton.get("root", {}).get("id", "root")
    children = _bone_children(spec)

    def in_scope(address: str) -> bool:
        return not scope or address.startswith(scope) or scope.startswith(address)

    def walk(parent_id: str, level: int) -> tuple[str, ...]:
        def visit(bone: dict) -> tuple[str, ...]:
            address = f"skeleton/{encode_address_segment(str(bone['id']))}"
            visible = not scope or (
                address.startswith(scope)
                or scope.startswith(address)
                or scope == "skeleton"
            )
            line = (_bone_line(bone, level),) if visible and level <= depth else ()
            flesh = (
                _flesh_lines(address, bone, level + 1)
                if visible and level <= depth and depth >= 3
                else ()
            )
            return (*line, *flesh, *walk(bone["id"], level + 1))

        return tuple(
            chain.from_iterable(visit(bone) for bone in children.get(parent_id, ()))
        )

    meta = (
        (
            f"meta  {spec.get('name', '?')}  blend {spec.get('blend')} "
            f"ground_y {spec.get('ground_y')} @ meta",
        )
        if in_scope("meta")
        else ()
    )
    skeleton_lines = (
        (
            f"skeleton  {len(bones)} bones, {flesh_count} flesh, "
            f"root {skeleton.get('root', {}).get('world')} @ skeleton/root",
            *walk(root_id, 1),
        )
        if in_scope("skeleton")
        else ()
    )
    props = (
        tuple(
            f"props/{encode_address_segment(str(prop['id']))}  "
            f"anchor {prop.get('anchor', {}).get('kind')}  "
            f"parts[{','.join(part.get('id', '?') for part in prop.get('parts', []))}]"
            for prop in spec.get("props", [])
        )
        if not scope or scope.startswith("props")
        else ()
    )
    mounts = (
        tuple(
            f"mounts/mounts[{index}]  {mount.get('subgrammar')} @ port "
            f"{mount.get('port')} scale {mount.get('scale')}"
            for index, mount in enumerate(spec.get("mounts", []))
        )
        if not scope or scope.startswith("mounts")
        else ()
    )
    contacts = (
        tuple(
            f"contacts/{encode_address_segment(str(contact['group']))}  "
            f"parts {contact.get('parts')} "
            f"solve {contact.get('solve')}"
            for contact in spec.get("contacts", [])
        )
        if not scope or scope.startswith("contacts")
        else ()
    )
    pose = spec.get("pose", {})
    joints = pose.get("joints", {})
    posed = (
        (f"pose  joints[{','.join(sorted(joints))}] @ pose",) if joints else ()
    )
    goals = tuple(
        f"pose/goals/{encode_address_segment(str(goal['id']))}  "
        f"chain {goal.get('chain')} -> {goal.get('target')}"
        for goal in pose.get("goals", [])
    )
    pose_lines = (*posed, *goals) if not scope or scope.startswith("pose") else ()
    return (*meta, *skeleton_lines, *props, *mounts, *contacts, *pose_lines)


def _constraint(state, scope: str, depth: int) -> tuple[str, ...]:
    cache = state.cache
    if cache.error is not None:
        return (f"COMPILE ERROR  {cache.error} @ meta",)
    failures = sum(record["status"] == "fail" for record in cache.records)
    passes = sum(record["status"] == "pass" for record in cache.records)
    unmeasurable = sum(
        record["status"] == "unmeasurable" for record in cache.records
    )
    summary = (
        f"contract  {failures} FAIL / {passes} pass"
        + (f" / {unmeasurable} unmeasurable" if unmeasurable else "")
        + " @ meta"
    )
    record_lines = tuple(
        "  " + record.get("human", f"{record.get('id')}: {record['status']}")
        if record["status"] in ("fail", "unmeasurable")
        else f"  pass {record.get('id')} (measured {record.get('measured')})"
        for record in cache.records
        if record["status"] in ("fail", "unmeasurable") or depth >= 2
    )
    anomaly_lines = tuple(
        f"anomaly N{index} {anomaly.get('det')} @ {anomaly.get('addr')}: "
        f"{anomaly.get('detail')}"
        for index, anomaly in enumerate(cache.anomalies, 1)
    )
    violation_lines = tuple(
        f"compile {violation.get('rule')} @ {violation.get('address')}: "
        f"{violation.get('detail')}"
        for violation in (cache.receipt or {}).get("violations", [])
    )
    return (summary, *record_lines, *anomaly_lines, *violation_lines)


def _recent(journal, count: int) -> tuple[str, ...]:
    return (
        tuple(
            f"txn {entry['txn']}  {entry['label']}"
            for entry in journal.entries[-count:]
        )
        if journal is not None and journal.entries
        else ("journal empty @ meta",)
    )


@dataclass(frozen=True)
class _Budgeted:
    kept: tuple[str, ...]
    used: int
    stopped: bool


def _bounded(lines: tuple[str, ...], budget: int, scope: str) -> tuple[str, ...]:
    def include(accumulator: _Budgeted, indexed: tuple[int, str]) -> _Budgeted:
        if accumulator.stopped:
            return accumulator
        index, line = indexed
        cost = est_tokens(line) + 1
        if accumulator.used + cost > budget and accumulator.kept:
            return _Budgeted(
                (
                    *accumulator.kept,
                    f"... +{len(lines) - index} more @ {scope or 'meta'} "
                    "(raise depth or narrow scope)",
                ),
                accumulator.used,
                True,
            )
        return _Budgeted(
            (*accumulator.kept, line),
            accumulator.used + cost,
            False,
        )

    return reduce(include, enumerate(lines), _Budgeted((), 0, False)).kept


def render(
    state,
    scope: str = "",
    depth: int = 1,
    aspect: str = "structure",
    journal=None,
) -> str:
    bounded_depth = max(1, min(3, int(depth)))
    lines = (
        _structure(state, scope, bounded_depth)
        if aspect == "structure"
        else _constraint(state, scope, bounded_depth)
        if aspect == "constraint"
        else _recent(journal, 5 * bounded_depth)
        if aspect == "recent"
        else (
            f"unknown aspect {aspect!r} (structure|constraint|recent) @ meta",
        )
    )
    return "\n".join(_bounded(lines, _BUDGET[bounded_depth], scope))
