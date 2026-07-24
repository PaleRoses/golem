"""Immutable carriers for parametric bone anchors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Anchor:
    bone: str
    t: float


@dataclass(frozen=True)
class AnchorObstruction:
    source: str
    reason: str


@dataclass(frozen=True)
class RejectedAnchor:
    obstructions: tuple[AnchorObstruction, ...]


type AnchorResult = Anchor | RejectedAnchor
