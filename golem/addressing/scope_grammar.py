"""Total, distinct grammars for assertion scopes and qualified address tokens."""

from __future__ import annotations

from typing import assert_never

from golem.addressing.scope import (
    AssertionScope,
    Bone,
    Chain,
    Contact,
    Element,
    Landmark,
    Mount,
    Part,
    Port,
    QualifiedScope,
    Region,
    RejectedScope,
    Scope,
    ScopeKind,
    ScopeObstruction,
    ScopeResult,
    Whole,
    World,
)


def parse_scope(source: str) -> ScopeResult:
    prefix, separator, payload = source.partition(":")
    if source in ("whole", "world") or prefix in (
        "part",
        "landmark",
        "chain",
    ):
        return _parse_assertion_scope(source)
    if prefix in ("element", "port", "mount", "region"):
        return _parse_qualified_scope(source)
    if prefix in ("bone", "contact"):
        return (
            _parse_qualified_scope(source)
            if separator and "/" in payload
            else _parse_assertion_scope(source)
        )
    return _rejected(
        source,
        (
            f"unknown scope kind {prefix!r}"
            if separator
            else "expected '<kind>:<...>', 'whole', or 'world'"
        ),
    )


def _parse_assertion_scope(source: str) -> AssertionScope | RejectedScope:
    if source == "whole":
        return Whole()
    if source == "world":
        return World()
    prefix, separator, payload = source.partition(":")
    if not separator:
        return _rejected(source, "expected '<kind>:<...>', 'whole', or 'world'")
    match prefix:
        case "part":
            return Part(payload) if payload else _missing_id(source, "part")
        case "bone":
            return Bone(payload) if payload and "/" not in payload else _missing_id(source, "bone")
        case "landmark":
            return _parse_landmark(source, payload)
        case "chain":
            return _parse_chain(source, payload)
        case "contact":
            return (
                Contact(payload)
                if payload and "/" not in payload
                else _missing_id(source, "contact")
            )
        case _:
            return _rejected(source, f"unknown assertion scope kind {prefix!r}")


def _parse_qualified_scope(source: str) -> QualifiedScope | RejectedScope:
    prefix, separator, payload = source.partition(":")
    if not separator:
        return _rejected(source, "expected a qualified '<kind>:<...>' token")
    match prefix:
        case "bone":
            return _parse_qualified_local(source, payload, Bone, "bone")
        case "contact":
            return _parse_qualified_local(source, payload, Contact, "contact")
        case "element":
            return _parse_element(source, payload, Element, "element")
        case "mount":
            return _parse_element(source, payload, Mount, "mount")
        case "region":
            return _parse_qualified_pair(source, payload, Region, "region")
        case "port":
            return _parse_qualified_pair(source, payload, Port, "port")
        case _:
            return _rejected(source, f"unknown qualified scope kind {prefix!r}")


def _missing_id(source: str, kind: str) -> RejectedScope:
    return _rejected(source, f"{kind}:<id> requires a non-empty identifier")


def _parse_landmark(source: str, payload: str) -> Landmark | RejectedScope:
    if not payload:
        return _rejected(source, "landmark:<name> requires a non-empty name")
    if "@" not in payload:
        return Landmark(payload)
    name, _, view = payload.rpartition("@")
    return (
        Landmark(name, view)
        if name and view
        else _rejected(source, "landmark:<name>@<view> requires both parts")
    )


def _parse_chain(source: str, payload: str) -> Chain | RejectedScope:
    start, separator, end = payload.partition("..")
    if not separator:
        return _rejected(source, "chain:<start>..<end> requires '..'")
    return (
        Chain(start, end)
        if start and end
        else _rejected(source, "chain:<start>..<end> requires both endpoints")
    )


def _parse_qualified_local(source, payload, make, kind) -> QualifiedScope | RejectedScope:
    element_id, separator, local_id = payload.partition("/")
    return (
        make(local_id, element_id)
        if separator and element_id and local_id
        else _rejected(source, f"{kind}:<element>/<id> requires both parts")
    )


def _parse_qualified_pair(source, payload, make, kind) -> QualifiedScope | RejectedScope:
    element_id, separator, local_id = payload.partition("/")
    return (
        make(element_id, local_id)
        if separator and element_id and local_id
        else _rejected(source, f"{kind}:<element>/<id> requires both parts")
    )


def _parse_element(source, payload, make, kind) -> QualifiedScope | RejectedScope:
    return (
        make(payload)
        if payload and "/" not in payload
        else _rejected(
            source,
            f"{kind}:<id> requires a non-empty identifier without '/'",
        )
    )


def _rejected(source: str, reason: str) -> RejectedScope:
    return RejectedScope((ScopeObstruction(source, reason),))


def render_scope(scope: Scope) -> str:
    match scope:
        case Bone(_, element_id) | Contact(_, element_id) if element_id is not None:
            return _render_qualified_scope(scope)
        case Element() | Port() | Mount() | Region():
            return _render_qualified_scope(scope)
        case _:
            return _render_assertion_scope(scope)


def _render_assertion_scope(scope: AssertionScope) -> str:
    match scope:
        case Whole():
            return "whole"
        case World():
            return "world"
        case Part(part_id):
            return f"part:{part_id}"
        case Bone(bone_id, None):
            return f"bone:{bone_id}"
        case Landmark(name, view):
            return f"landmark:{name}@{view}" if view is not None else f"landmark:{name}"
        case Chain(start, end):
            return f"chain:{start}..{end}"
        case Contact(contact_id, None):
            return f"contact:{contact_id}"
        case _ as unreachable:
            assert_never(unreachable)


def _render_qualified_scope(scope: QualifiedScope) -> str:
    match scope:
        case Bone(bone_id, str(element_id)):
            return f"bone:{element_id}/{bone_id}"
        case Contact(contact_id, str(element_id)):
            return f"contact:{element_id}/{contact_id}"
        case Element(element_id):
            return f"element:{element_id}"
        case Port(element_id, port_id):
            return f"port:{element_id}/{port_id}"
        case Mount(element_id):
            return f"mount:{element_id}"
        case Region(element_id, region_id):
            return f"region:{element_id}/{region_id}"
        case _ as unreachable:
            assert_never(unreachable)


def scope_kind(scope: Scope) -> ScopeKind:
    match scope:
        case Whole():
            return ScopeKind.WHOLE
        case World():
            return ScopeKind.WORLD
        case Part():
            return ScopeKind.PART
        case Bone():
            return ScopeKind.BONE
        case Landmark():
            return ScopeKind.LANDMARK
        case Chain():
            return ScopeKind.CHAIN
        case Contact():
            return ScopeKind.CONTACT
        case Element():
            return ScopeKind.ELEMENT
        case Port():
            return ScopeKind.PORT
        case Mount():
            return ScopeKind.MOUNT
        case Region():
            return ScopeKind.REGION
        case _ as unreachable:
            assert_never(unreachable)
