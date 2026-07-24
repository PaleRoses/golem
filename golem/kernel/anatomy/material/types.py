"""Vascular-material mask carriers and obstruction ADTs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class VascularMaterialMasks:
    """Resolved lumen and wall sections over an evaluated morphology."""

    lumen: "NDArray[np.bool_]"
    wall: "NDArray[np.bool_]"
    load_bearing_solid: "NDArray[np.bool_]"
    minimum_radius_to_pitch: float
    minimum_wall_to_pitch: float


@dataclass(frozen=True)
class UnresolvedLumenObstruction:
    edge_id: str
    radius: float
    pitch: float
    required_cells: float
    unresolved_edge_count: int


@dataclass(frozen=True)
class UnresolvedVascularWallObstruction:
    wall_thickness: float
    pitch: float
    required_cells: float


@dataclass(frozen=True)
class VascularMaterialEscapeObstruction:
    leaking_cell_count: int


@dataclass(frozen=True)
class EmptyVascularMaterialObstruction:
    stratum: str


@dataclass(frozen=True)
class ChannelInducedDisconnectionObstruction:
    component_count: int


type VascularMaterialObstruction = (
    UnresolvedLumenObstruction
    | UnresolvedVascularWallObstruction
    | VascularMaterialEscapeObstruction
    | EmptyVascularMaterialObstruction
    | ChannelInducedDisconnectionObstruction
)


@dataclass(frozen=True)
class AcceptedVascularMaterial:
    masks: VascularMaterialMasks


@dataclass(frozen=True)
class RejectedVascularMaterial:
    obstructions: tuple[VascularMaterialObstruction, ...]


type VascularMaterialResult = AcceptedVascularMaterial | RejectedVascularMaterial
