"""Immutable carriers for seeded appendage growth."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import TypedDict

import numpy as np


class GrowthTermination(StrEnum):
    ATTRACTORS_EXHAUSTED = "attractors_exhausted"
    MAXIMUM_SECTIONS = "maximum_sections"
    NO_INFLUENCED_ATTRACTORS = "no_influenced_attractors"
    NO_INTERIOR_BRANCH = "no_interior_branch"


@dataclass(frozen=True)
class GrowthParameters:
    parts: tuple[dict, ...]
    influence: float
    kill: float
    step: float
    maximum_sections: int
    interior_margin: float = 0.012


@dataclass(frozen=True)
class GrowthState:
    nodes: np.ndarray
    parents: np.ndarray
    attractors: np.ndarray
    alive: np.ndarray
    section: int = 0


@dataclass(frozen=True)
class ContinueGrowth:
    state: GrowthState


@dataclass(frozen=True)
class TerminatedGrowth:
    state: GrowthState
    termination: GrowthTermination


type GrowthSection = ContinueGrowth | TerminatedGrowth


@dataclass(frozen=True)
class GrowthNetwork:
    nodes: np.ndarray
    parents: np.ndarray
    radii: np.ndarray
    termination: GrowthTermination

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter((self.nodes, self.parents, self.radii))


class AppendageChain(TypedDict):
    spine: np.ndarray
    radii: np.ndarray


__all__ = [
    "AppendageChain",
    "ContinueGrowth",
    "GrowthNetwork",
    "GrowthParameters",
    "GrowthSection",
    "GrowthState",
    "GrowthTermination",
    "TerminatedGrowth",
]
