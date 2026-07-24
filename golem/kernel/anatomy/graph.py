"""Closed vascular graph carriers and vasculature obstruction ADTs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from golem.kernel.anatomy.vocabulary import CapillaryBed


@dataclass(frozen=True)
class DistributingArtery:
    interface_address: str


@dataclass(frozen=True)
class ResistanceArteriole:
    interface_address: str


@dataclass(frozen=True)
class CollectingVenule:
    interface_address: str


@dataclass(frozen=True)
class ReturningVein:
    interface_address: str


type VascularSegment = (
    DistributingArtery | ResistanceArteriole | CollectingVenule | ReturningVein
)


class VascularNodeKind(StrEnum):
    PUMP_OUTLET = "pump_outlet"
    PUMP_INLET = "pump_inlet"
    MACRO_CORRIDOR = "macro_corridor"
    CORRIDOR = "corridor"
    BIFURCATION = "bifurcation"
    SUPPLY_TERMINAL = "supply_terminal"
    RETURN_TERMINAL = "return_terminal"


class VascularStratum(StrEnum):
    SUPPLY = "supply"
    RETURN = "return"
    EXCHANGE = "exchange"


class VascularGrowthPhase(StrEnum):
    SEED = "seed"
    INSERTION = "insertion"
    ANCESTOR_GROWTH = "ancestor_growth"


class VascularLane(StrEnum):
    SUPPLY = "supply"
    RETURN = "return"


class VascularRelaxationStage(StrEnum):
    FIXED_CORRIDOR = "fixed_corridor"
    WAYPOINT_ADJUSTMENT = "waypoint_adjustment"
    WIDER_SEARCH = "wider_search"


class VascularInfeasibilityPredicate(StrEnum):
    SEGMENT_LENGTH = "SegmentLength"
    WALL_CONTAINMENT = "WallContainment"
    CENTERLINE_SEPARATION = "CenterlineSeparation"
    RESERVED_LANE_CLEARANCE = "ReservedLaneClearance"
    EXCHANGE_CLEARANCE = "ExchangeClearance"
    SELF_CLEARANCE = "SelfClearance"
    SEARCH_EXHAUSTED = "SearchExhausted"


class VascularSymmetryPredicate(StrEnum):
    MIRROR_PAIRING = "MirrorPairing"
    REFLECTION_DISTANCE = "ReflectionDistance"


@dataclass(frozen=True)
class VascularNode:
    node_id: str
    position: tuple[float, float, float]
    kind: VascularNodeKind
    region_id: str | None


@dataclass(frozen=True)
class VascularEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    stage: VascularSegment | CapillaryBed
    stratum: VascularStratum
    radius: float
    target_flow: float
    solved_flow: float


@dataclass(frozen=True)
class ClosedVascularGraph:
    nodes: tuple[VascularNode, ...]
    edges: tuple[VascularEdge, ...]
    pump_outlet_node_id: str
    pump_inlet_node_id: str

    @property
    def node_by_id(self) -> dict[str, VascularNode]:
        return {node.node_id: node for node in self.nodes}


@dataclass(frozen=True)
class VascularFlowReceipt:
    terminal_pair_budget: int
    terminal_pairs_by_region: tuple[tuple[str, int], ...]
    stage_counts: tuple[tuple[str, int], ...]
    minimum_capsule_margin: float
    maximum_free_cell_residual: float
    boundary_balance_error: float
    maximum_delivery_relative_error: float
    delivered_demand_by_region: tuple[tuple[str, float], ...]
    pump_pressure_drop: float
    conservation_error: float
    maximum_symmetry_error: float
    checked_nonincident_pair_count: int
    structural_guidance_sample_count: int = 0
    maximum_structural_cost_multiplier: float = 1.0
    search_effort_by_region: tuple[
        tuple[str, "VascularSearchEffort"], ...
    ] = ()


@dataclass(frozen=True)
class VascularSearchEffort:
    deepest_stage: VascularRelaxationStage
    evaluated_state_count: int


@dataclass(frozen=True, slots=True)
class VascularSegmentGeometry:
    source_position: tuple[float, float, float]
    target_position: tuple[float, float, float]
    radius: float | None = None
    edge_id: str | None = None
    source_host_bone_addresses: tuple[str, ...] = ()
    source_host_part_addresses: tuple[str, ...] = ()
    target_host_bone_addresses: tuple[str, ...] = ()
    target_host_part_addresses: tuple[str, ...] = ()


def vascular_segment_geometry(
    source: tuple[float, float, float],
    target: tuple[float, float, float],
    radius: float | None = None,
    edge_id: str | None = None,
) -> VascularSegmentGeometry:
    return VascularSegmentGeometry(
        source,
        target,
        radius,
        edge_id,
    )


def vascular_edge_geometry(
    edge: VascularEdge,
    node_by_id: dict[str, VascularNode],
) -> VascularSegmentGeometry:
    return vascular_segment_geometry(
        node_by_id[edge.source_node_id].position,
        node_by_id[edge.target_node_id].position,
        edge.radius,
        edge.edge_id,
    )


@dataclass(frozen=True)
class MissingProvenanceObstruction:
    region_id: str
    bone_id: str


@dataclass(frozen=True)
class EmptyPerfusionTerritoryObstruction:
    region_id: str


@dataclass(frozen=True)
class InsufficientTerminalSitesObstruction:
    region_id: str
    requested: int
    available: int


@dataclass(frozen=True)
class InfeasibleBifurcationObstruction:
    region_id: str
    phase: VascularGrowthPhase
    lane: VascularLane
    predicate: VascularInfeasibilityPredicate
    required: float
    observed: float
    terminal_index: int | None
    supply_capsule: int | None = None
    return_capsule: int | None = None
    split_edge: tuple[int, int] | None = None
    candidate_point_index: int | None = None
    attempted_candidate_count: int | None = None
    failing_segments: tuple[VascularSegmentGeometry, ...] = ()


@dataclass(frozen=True)
class VascularSearchBudgetObstruction:
    region_id: str
    phase: VascularGrowthPhase
    lane: VascularLane
    required_state_budget: int
    observed_evaluated_states: int
    remaining_queue_size: int
    limiting_predicate: VascularInfeasibilityPredicate
    limiting_required: float
    limiting_observed: float
    failing_segments: tuple[VascularSegmentGeometry, ...] = ()


@dataclass(frozen=True)
class CapsuleEscapeObstruction:
    edge_id: str
    margin: float
    failing_segments: tuple[VascularSegmentGeometry, ...] = ()


@dataclass(frozen=True)
class VascularIntersectionObstruction:
    left_edge_id: str
    right_edge_id: str
    clearance: float
    failing_segments: tuple[VascularSegmentGeometry, ...] = ()


@dataclass(frozen=True)
class VascularSymmetryWitness:
    node_id: str
    expected_mirror_node_id: str


@dataclass(frozen=True)
class SymmetryMismatchObstruction:
    region_id: str
    predicate: VascularSymmetryPredicate
    required: float
    observed: float
    witness: VascularSymmetryWitness


@dataclass(frozen=True)
class VascularBridgeWitness:
    """One weak conductance interface in the glued vascular network."""

    edge_id: str
    conductance: float


@dataclass(frozen=True)
class VascularTopologyHypothesis:
    free_component_count: int
    weakest_bridges: tuple[VascularBridgeWitness, ...]


@dataclass(frozen=True)
class VascularConditionSectionWitness:
    """One vascular node whose incident conductance ratio poisons the
    global spectrum."""

    node_id: str
    region_id: str | None
    diagonal: float
    conductance_ratio: float


@dataclass(frozen=True)
class VascularConditioningHypothesis:
    gershgorin_upper_bound: float
    diagonal_spread: float
    worst_sections: tuple[VascularConditionSectionWitness, ...]


@dataclass(frozen=True)
class VascularResidualSectionWitness:
    """One vascular node whose balance equation the stop iterate least
    satisfies."""

    node_id: str
    region_id: str | None
    residual: float


@dataclass(frozen=True)
class VascularDemandHypothesis:
    maximum_stop_residual: float
    residual_tolerance: float
    relative_residual: float
    relative_tolerance: float
    iteration_cap: int
    iterations_used: int
    exhausted: bool
    worst_sections: tuple[VascularResidualSectionWitness, ...]


@dataclass(frozen=True)
class VascularBudgetHypothesis:
    iteration_cap: int
    iterations_used: int
    relative_tolerance: float
    exhausted: bool


@dataclass(frozen=True)
class VascularGluingInterrogation:
    """R13's decomposition of a failed closed-vascular balance solve into
    typed hypotheses, each witnessed in vascular addresses (nodes, regions,
    edges) rather than solver codes."""

    topology: VascularTopologyHypothesis
    conditioning: VascularConditioningHypothesis
    demand: VascularDemandHypothesis
    budget: VascularBudgetHypothesis


@dataclass(frozen=True)
class VascularGluingObstruction:
    interface_address: str
    reason: str
    failing_segments: tuple[VascularSegmentGeometry, ...] = ()
    gluing_interrogation: VascularGluingInterrogation | None = None


@dataclass(frozen=True)
class DisconnectedVascularGraphObstruction:
    node_id: str


@dataclass(frozen=True)
class VascularSolverResidualObstruction:
    maximum_residual: float
    tolerance: float


@dataclass(frozen=True)
class DemandDeliveryMismatchObstruction:
    region_id: str
    relative_error: float
    tolerance: float


@dataclass(frozen=True)
class ReversedVascularFlowObstruction:
    edge_id: str
    solved_flow: float
    failing_segments: tuple[VascularSegmentGeometry, ...] = ()


@dataclass(frozen=True)
class InvalidStructuralGuidanceObstruction:
    address: str
    reason: str


type VasculatureObstruction = (
    MissingProvenanceObstruction
    | EmptyPerfusionTerritoryObstruction
    | InsufficientTerminalSitesObstruction
    | InfeasibleBifurcationObstruction
    | VascularSearchBudgetObstruction
    | CapsuleEscapeObstruction
    | VascularIntersectionObstruction
    | SymmetryMismatchObstruction
    | VascularGluingObstruction
    | DisconnectedVascularGraphObstruction
    | VascularSolverResidualObstruction
    | DemandDeliveryMismatchObstruction
    | ReversedVascularFlowObstruction
    | InvalidStructuralGuidanceObstruction
)


@dataclass(frozen=True)
class AcceptedVasculature:
    graph: ClosedVascularGraph
    receipt: VascularFlowReceipt


@dataclass(frozen=True)
class RejectedVasculature:
    obstructions: tuple[VasculatureObstruction, ...]


type VasculatureResult = AcceptedVasculature | RejectedVasculature
