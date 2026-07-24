"""Typed results for surface-conduit emission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np


@dataclass(frozen=True)
class EmissionPitchObstruction:
    declaration_id: str
    emission: str
    displacement: float
    maximum_displacement: float
    pitch: float

    @property
    def kind(self) -> str:
        return "EmissionPitchBoundViolation"

    @property
    def address(self) -> str:
        return f"/conduits/{self.declaration_id}/{self.emission}"

    @property
    def reason(self) -> str:
        return (
            f"{self.emission} displacement {self.displacement:.12g} exceeds "
            f"the hard 0.9*pitch bound {self.maximum_displacement:.12g} "
            f"for pitch {self.pitch:.12g}"
        )


class ConduitBand(TypedDict):
    id: str
    verts: np.ndarray | None
    faces: np.ndarray | None
    material: str
    empty: bool


type AppliedConduits = tuple[np.ndarray, tuple[ConduitBand, ...]]
type ConduitApplicationResult = AppliedConduits | EmissionPitchObstruction


__all__ = [
    "AppliedConduits",
    "ConduitApplicationResult",
    "ConduitBand",
    "EmissionPitchObstruction",
]
