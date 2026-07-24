"""Total parser and renderer for parametric bone anchors."""

from __future__ import annotations

from golem.addressing.anchor import (
    Anchor,
    AnchorObstruction,
    AnchorResult,
    RejectedAnchor,
)

_ANCHOR_MINIMUM = -0.5
_ANCHOR_MAXIMUM = 1.5


def parse_anchor(source: str) -> AnchorResult:
    bone, _, raw_t = source.partition(":")
    if not bone:
        return _rejected(source, "anchor must be <bone_id>:<t>")
    parameter = 1.0 if not raw_t else _parse_float(raw_t)
    if parameter is None:
        return _rejected(source, f"anchor t {raw_t!r} is not a number")
    if not (_ANCHOR_MINIMUM <= parameter <= _ANCHOR_MAXIMUM):
        return _rejected(
            source,
            f"anchor t {parameter} outside [{_ANCHOR_MINIMUM}, {_ANCHOR_MAXIMUM}]",
        )
    return Anchor(bone, parameter)


def _parse_float(raw: str) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value == value else None


def _rejected(source: str, reason: str) -> RejectedAnchor:
    return RejectedAnchor((AnchorObstruction(source, reason),))


def render_anchor(anchor: Anchor) -> str:
    return f"{anchor.bone}:{anchor.t}"
