"""Immutable embedded-channel evidence, results, and typed obstructions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import fsum, pi, sqrt
from typing import TYPE_CHECKING

from ..model import (
    IsotropicConstitutiveProperties,
    MaterialFractionResolution,
    UnevaluatedPhysics,
)

if TYPE_CHECKING:
    from golem.kernel.anatomy import VascularEdge, VascularNode


@dataclass(frozen=True)
class EmbeddedChannelCriteria:
    """Validity and acceptance gates for a channel family below the Q1 grid."""

    maximum_outer_diameter_to_grid_pitch: float = 1.0e-1
    maximum_lumen_volume_fraction: float = 1.0e-3
    maximum_channel_envelope_volume_fraction: float = 2.0e-3
    minimum_burst_safety_factor: float | None = None
    normalization_floor: float = 1.0e-14


@dataclass(frozen=True)
class EmbeddedChannelSectionMechanics:
    edge_id: str
    radius_metres: float
    outer_radius_metres: float
    length_metres: float
    maximum_transmural_pressure_pascal: float
    lumen_capsule_volume_upper_bound_cubic_metres: float
    wall_annulus_volume_upper_bound_cubic_metres: float
    thin_wall_hoop_stress_pascal: float
    lame_inner_wall_hoop_stress_pascal: float
    burst_safety_factor: float


@dataclass(frozen=True)
class FullSolidEmbeddedChannelResponse:
    properties: IsotropicConstitutiveProperties
    material_fraction_resolution: MaterialFractionResolution = (
        MaterialFractionResolution.FULL_SOLID
    )


@dataclass(frozen=True)
class HomogenizedEmbeddedChannelResponse:
    properties: IsotropicConstitutiveProperties
    validated_lumen_volume_fraction: float
    material_fraction_resolution: MaterialFractionResolution = (
        MaterialFractionResolution.HOMOGENIZED_EFFECTIVE_PROPERTIES
    )


@dataclass(frozen=True)
class UnresolvedEmbeddedChannelConstitutiveObstruction:
    reason: str
    lumen_volume_fraction_upper_bound: float
    required_resolution: MaterialFractionResolution = (
        MaterialFractionResolution.HOMOGENIZED_EFFECTIVE_PROPERTIES
    )


type EmbeddedChannelConstitutiveResponse = (
    FullSolidEmbeddedChannelResponse
    | HomogenizedEmbeddedChannelResponse
    | UnresolvedEmbeddedChannelConstitutiveObstruction
)


@dataclass(frozen=True)
class EmbeddedChannelMechanicsReceipt:
    geometry_model: str
    pressure_containment_model: str
    background_solid_model: str
    edge_count: int
    positive_radius_edge_count: int
    grid_pitch_metres: float
    host_volume_cubic_metres: float
    wall_thickness_metres: float
    wall_allowable_stress_pascal: float
    external_pressure_pascal: float
    maximum_outer_diameter_to_grid_pitch: float
    lumen_volume_upper_bound_cubic_metres: float
    wall_volume_upper_bound_cubic_metres: float
    lumen_volume_fraction_upper_bound: float
    channel_envelope_volume_fraction_upper_bound: float
    conservative_solid_fraction_lower_bound: float
    maximum_thin_wall_hoop_stress_pascal: float
    maximum_lame_inner_wall_hoop_stress_pascal: float
    minimum_burst_safety_factor: float
    constitutive_response: EmbeddedChannelConstitutiveResponse
    not_evaluated: tuple[UnevaluatedPhysics, ...]


@dataclass(frozen=True)
class AcceptedEmbeddedChannelMechanics:
    sections: tuple[EmbeddedChannelSectionMechanics, ...]
    receipt: EmbeddedChannelMechanicsReceipt


@dataclass(frozen=True)
class InvalidEmbeddedChannelInputObstruction:
    address: str
    reason: str


@dataclass(frozen=True)
class EmbeddedChannelScaleSeparationObstruction:
    maximum_outer_diameter_to_grid_pitch: float
    permitted_maximum: float


class EmbeddedChannelVolumeKind(StrEnum):
    LUMEN = "lumen"
    OUTER_ENVELOPE = "outer_envelope"


@dataclass(frozen=True)
class EmbeddedChannelVolumeFractionObstruction:
    kind: EmbeddedChannelVolumeKind
    volume_fraction_upper_bound: float
    permitted_maximum: float


@dataclass(frozen=True)
class UnsupportedEmbeddedChannelExternalPressureObstruction:
    edge_id: str
    minimum_transmural_pressure_pascal: float
    normalization_floor: float


@dataclass(frozen=True)
class InsufficientEmbeddedChannelBurstMarginObstruction:
    minimum_burst_safety_factor: float
    required_safety_factor: float


type EmbeddedChannelObstruction = (
    InvalidEmbeddedChannelInputObstruction
    | EmbeddedChannelScaleSeparationObstruction
    | EmbeddedChannelVolumeFractionObstruction
    | UnsupportedEmbeddedChannelExternalPressureObstruction
    | InsufficientEmbeddedChannelBurstMarginObstruction
)


@dataclass(frozen=True)
class RejectedEmbeddedChannelMechanics:
    obstructions: tuple[EmbeddedChannelObstruction, ...]


type EmbeddedChannelMechanicsResult = (
    AcceptedEmbeddedChannelMechanics | RejectedEmbeddedChannelMechanics
)


def _vascular_edge_length_world(
    edge: "VascularEdge", node_by_id: dict[str, "VascularNode"]
) -> float:
    source = node_by_id[edge.source_node_id]
    target = node_by_id[edge.target_node_id]
    return sqrt(
        fsum(
            (right - left) ** 2
            for left, right in zip(source.position, target.position)
        )
    )


def _capsule_volume(radius: float, length: float) -> float:
    return pi * radius**2 * length + (4.0 / 3.0) * pi * radius**3
