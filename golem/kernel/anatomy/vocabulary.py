"""Anatomy vocabulary: kinds, carriers, obstruction ADTs, and sealed products."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from golem.kernel.engine.types import SkinFormationKind

if TYPE_CHECKING:
    from golem.kernel.anatomy.graph import (
        CollectingVenule,
        DistributingArtery,
        ResistanceArteriole,
        ReturningVein,
    )


DIALECT = "anatomy/0.1"
_CLOSED_VASCULAR_KIND = "closed_vascular"
_MURRAY_BRANCH_EXPONENT = 3.0
_MINIMUM_CARRIER_RADIUS = 0.04
_MAXIMUM_POISEUILLE_REYNOLDS = 2300.0


class BodyRegionKind(StrEnum):
    TORSO = "torso"
    HEAD = "head"
    LIMB = "limb"
    TAIL = "tail"
    WING = "wing"
    FIN = "fin"
    PSEUDOPOD = "pseudopod"
    SEGMENT = "segment"


class PerfusedTissueKind(StrEnum):
    NEURAL_TISSUE = "neural_tissue"
    SKELETAL_MUSCLE = "skeletal_muscle"
    VISCERAL_TISSUE = "visceral_tissue"
    INTEGUMENT = "integument"


class CarrierConstraintKind(StrEnum):
    CIRCULATION_CLEARANCE = "circulation_clearance"
    TISSUE_ENVELOPE = "tissue_envelope"


_REGION_KIND_VALUES = tuple(kind.value for kind in BodyRegionKind)
_TISSUE_KIND_VALUES = tuple(kind.value for kind in PerfusedTissueKind)


@dataclass(frozen=True)
class AnatomyRegion:
    region_id: str
    kind: BodyRegionKind
    host_bone_id: str


@dataclass(frozen=True)
class PumpOrgan:
    region: AnatomyRegion


@dataclass(frozen=True)
class CapillaryBed:
    region: AnatomyRegion
    tissue: PerfusedTissueKind
    demand: float
    tissue_minimum_radius: float | None


@dataclass(frozen=True)
class ClosedVascularSystem:
    pump: PumpOrgan
    exchange_beds: tuple[CapillaryBed, ...]
    carrier_radius_scale: float
    distance_decay: float


@dataclass(frozen=True)
class TissueAnchor:
    bone_id: str
    parameter: float
    offset: tuple[float, float]


@dataclass(frozen=True)
class MyotendinousUnit:
    muscle_id: str
    region_id: str
    origin: TissueAnchor
    joint: TissueAnchor
    insertion: TissueAnchor
    tendon_radius: float
    belly_radius: float


@dataclass(frozen=True)
class IntegumentLayer:
    region_id: str
    thickness: float
    formation: SkinFormationKind | None = None


@dataclass(frozen=True)
class OverallAnatomy:
    regions: tuple[AnatomyRegion, ...]
    circulation: ClosedVascularSystem
    muscles: tuple[MyotendinousUnit, ...] = ()
    integument: tuple[IntegumentLayer, ...] = ()


@dataclass(frozen=True)
class MalformedAnatomyObstruction:
    address: str
    reason: str


@dataclass(frozen=True)
class InvalidSkeletonAddressObstruction:
    address: str
    value: object


@dataclass(frozen=True)
class DuplicateAnatomyRegionObstruction:
    region_id: str


@dataclass(frozen=True)
class DuplicateRegionHostObstruction:
    host_bone_id: str


@dataclass(frozen=True)
class DuplicateExchangeBedObstruction:
    region_id: str


@dataclass(frozen=True)
class UnknownAnatomyRegionObstruction:
    address: str
    region_id: object


@dataclass(frozen=True)
class UnreferencedAnatomyRegionObstruction:
    region_id: str


@dataclass(frozen=True)
class MissingTerminalRegionObstruction:
    host_bone_id: str


@dataclass(frozen=True)
class NonterminalExchangeRegionObstruction:
    region_id: str
    host_bone_id: str


@dataclass(frozen=True)
class InsufficientVascularStagesObstruction:
    region_id: str
    interface_count: int


@dataclass(frozen=True)
class DisconnectedAnatomyRegionObstruction:
    region_id: str
    pump_host_bone_id: str
    exchange_host_bone_id: str


@dataclass(frozen=True)
class DuplicateMyotendinousUnitObstruction:
    muscle_id: str


@dataclass(frozen=True)
class InvalidMyotendinousPathObstruction:
    muscle_id: str
    origin_bone_id: str
    insertion_bone_id: str


@dataclass(frozen=True)
class UnperfusedMyotendinousUnitObstruction:
    muscle_id: str
    region_id: str


@dataclass(frozen=True)
class UnsupportedTissueCarrierObstruction:
    muscle_id: str
    bone_id: str


@dataclass(frozen=True)
class MissingSkeletalCrossSectionObstruction:
    muscle_id: str
    bone_id: str


@dataclass(frozen=True)
class DuplicateIntegumentLayerObstruction:
    region_id: str


@dataclass(frozen=True)
class MissingIntegumentLayerObstruction:
    muscle_id: str
    region_id: str


@dataclass(frozen=True)
class IncompatibleIntegumentFormationObstruction:
    formed_region_ids: tuple[str, ...]
    legacy_region_ids: tuple[str, ...]
    uncovered_region_ids: tuple[str, ...]
    formed_thicknesses: tuple[float, ...]


type AnatomyInputObstruction = (
    MalformedAnatomyObstruction
    | InvalidSkeletonAddressObstruction
    | DuplicateAnatomyRegionObstruction
    | DuplicateRegionHostObstruction
    | DuplicateExchangeBedObstruction
    | UnknownAnatomyRegionObstruction
    | UnreferencedAnatomyRegionObstruction
    | MissingTerminalRegionObstruction
    | NonterminalExchangeRegionObstruction
    | DuplicateMyotendinousUnitObstruction
    | InvalidMyotendinousPathObstruction
    | UnperfusedMyotendinousUnitObstruction
    | UnsupportedTissueCarrierObstruction
    | MissingSkeletalCrossSectionObstruction
    | DuplicateIntegumentLayerObstruction
    | MissingIntegumentLayerObstruction
    | IncompatibleIntegumentFormationObstruction
)
type AnatomyObstruction = (
    AnatomyInputObstruction
    | InsufficientVascularStagesObstruction
    | DisconnectedAnatomyRegionObstruction
)


@dataclass(frozen=True)
class SealedVascularConfig:
    terminal_pair_budget: int = 256
    minimum_terminal_pairs_per_bed: int = 4
    sobol_candidate_factor: int = 128
    wall_clearance: float = 5.0e-4
    vessel_clearance: float = 1.0e-8
    terminal_port_half_separation: float = 4.0e-3
    minimum_segment_length: float = 2.0e-4
    terminal_radius: float = 3.0e-5
    exchange_radius_ratio: float = 0.8
    nonincident_centerline_separation: float = 5.0e-4
    murray_exponent: float = 3.0
    viscosity: float = 1.0
    capsule_samples: int = 9
    delivery_tolerance: float = 1.0e-3
    solver_tolerance: float = 1.0e-10
    calibration_iterations: int = 50
    structural_cost_weight: float = 0.05
    search_state_budget: int = 512
    corridor_candidate_budget: int = 128
    waypoint_stencil_step: float = 1.0e-3
    maximum_waypoint_adjustment: float = 4.0e-3


@dataclass(frozen=True)
class CirculationCircuit:
    capillary_bed: CapillaryBed
    effective_demand: float
    bones: tuple[str, ...]
    distributing_arteries: tuple[DistributingArtery, ...]
    resistance_arteriole: ResistanceArteriole
    collecting_venule: CollectingVenule
    returning_veins: tuple[ReturningVein, ...]


@dataclass(frozen=True)
class CarrierFleshSample:
    flesh_index: int
    minimum_radius: float


@dataclass(frozen=True)
class AnatomyCarrierRow:
    bone_id: str
    interface_address: str
    shape_address: str | None
    normalized_flow: float
    distance_from_pump: int
    circulation_minimum_radius: float
    tissue_minimum_radius: float | None
    measured_minimum_radius: float | None

    @property
    def controlling_constraint(self) -> CarrierConstraintKind:
        return (
            CarrierConstraintKind.TISSUE_ENVELOPE
            if (self.tissue_minimum_radius or 0.0)
            > self.circulation_minimum_radius
            else CarrierConstraintKind.CIRCULATION_CLEARANCE
        )

    @property
    def target_minimum_radius(self) -> float:
        return max(
            self.circulation_minimum_radius, self.tissue_minimum_radius or 0.0
        )

    @property
    def deficit(self) -> float:
        measured = self.measured_minimum_radius
        return (
            max(0.0, self.target_minimum_radius - measured)
            if measured is not None
            else 0.0
        )

    @property
    def suggested_scale(self) -> float:
        measured = self.measured_minimum_radius
        return (
            self.target_minimum_radius / measured
            if measured is not None and self.deficit > 0.0
            else 1.0
        )


@dataclass(frozen=True)
class TissueEnvelopeStation:
    parameter: float
    offset: tuple[float, float]
    half_size: tuple[float, float]


@dataclass(frozen=True)
class TissueEnvelopeSection:
    bone_id: str
    shape_address: str
    muscle_ids: tuple[str, ...]
    stations: tuple[TissueEnvelopeStation, ...]


@dataclass(frozen=True)
class AcceptedAnatomy:
    overall: OverallAnatomy
    circuits: tuple[CirculationCircuit, ...]
    carrier_rows: tuple[AnatomyCarrierRow, ...]
    tissue_envelopes: tuple[TissueEnvelopeSection, ...]

    @property
    def authoring_counts(self) -> tuple[int, int]:
        authored = (
            2
            + len(self.overall.circulation.exchange_beds)
            + sum(
                exchange_bed.tissue_minimum_radius is not None
                for exchange_bed in self.overall.circulation.exchange_beds
            )
            + 10 * len(self.overall.muscles)
            + len(self.overall.integument)
        )
        derived = sum(
            row.measured_minimum_radius is not None for row in self.carrier_rows
        ) + 4 * sum(
            len(envelope.stations) for envelope in self.tissue_envelopes
        )
        return authored, derived

    @property
    def authoring_leverage(self) -> float:
        authored, derived = self.authoring_counts
        return derived / authored

    @property
    def conservation(self) -> tuple[float, float]:
        """Maximum free-cell imbalance and boundary-relative error."""
        from golem.kernel.anatomy.descent import _conservation

        return _conservation(self.circuits, self.carrier_rows)


@dataclass(frozen=True)
class RejectedAnatomy:
    obstructions: tuple[AnatomyObstruction, ...]


type AnatomyResult = AcceptedAnatomy | RejectedAnatomy
