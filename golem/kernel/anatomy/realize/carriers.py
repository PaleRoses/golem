"""Vasculature realization carriers and structural-cost field indexing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from scipy.spatial import cKDTree

    from golem.kernel.anatomy.graph import (
        InfeasibleBifurcationObstruction,
        VascularEdge,
        VascularInfeasibilityPredicate,
        VascularLane,
        VascularNode,
        VascularNodeKind,
        VascularSearchEffort,
    )


@dataclass(frozen=True)
class _LocalVascularTree:
    positions: tuple[tuple[float, float, float], ...]
    parents: tuple[int, ...]
    kinds: tuple[VascularNodeKind, ...]
    terminal_flows: tuple[float, ...]
    terminal_ordinals: tuple[int | None, ...]


@dataclass(frozen=True)
class FeasibleBifurcation:
    cost: float
    tree: _LocalVascularTree


@dataclass(frozen=True)
class RejectedBifurcationCandidate:
    obstruction: InfeasibleBifurcationObstruction


type BifurcationCandidateResult = (
    FeasibleBifurcation | RejectedBifurcationCandidate
)


@dataclass(frozen=True)
class AcceptedSearch[value]:
    value: value
    evaluated_state_count: int
    attempted_candidate_count: int


@dataclass(frozen=True)
class ExhaustedSearch:
    obstructions: tuple[InfeasibleBifurcationObstruction, ...]
    evaluated_state_count: int
    attempted_candidate_count: int


@dataclass(frozen=True)
class BudgetExhaustedSearch:
    obstructions: tuple[InfeasibleBifurcationObstruction, ...]
    evaluated_state_count: int
    attempted_candidate_count: int
    remaining_queue_size: int


type SearchResult[value] = (
    AcceptedSearch[value] | ExhaustedSearch | BudgetExhaustedSearch
)


@dataclass(frozen=True)
class _MacroCorridorSection:
    points: tuple[tuple[float, float, float], ...]
    movable_waypoint_indices: tuple[int, ...]


@dataclass(frozen=True)
class _StructuralCostField:
    centers_metres: tuple[tuple[float, float, float], ...]
    normalized_von_mises_stress: tuple[float, ...]
    metres_per_world_unit: float
    cost_weight: float

    @property
    def maximum_cost_multiplier(self) -> float:
        return 1.0 + self.cost_weight * max(
            self.normalized_von_mises_stress, default=0.0
        )


@dataclass(frozen=True, eq=False)
class _StructuralCostIndex:
    spatial_index: "cKDTree"
    normalized_von_mises_stress: "NDArray[np.float64]"
    metres_per_world_unit: float
    cost_weight: float


@dataclass(frozen=True)
class _SegmentCapsule:
    left: tuple[float, float, float]
    right: tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class _CapsuleClearanceObligation:
    capsules: tuple[_SegmentCapsule, ...]
    capsule_indices: tuple[int, ...]
    minimum_surface_clearance: float
    predicate: VascularInfeasibilityPredicate
    lane: VascularLane | None


@dataclass(frozen=True)
class _IndexedClearanceViolation:
    left_index: int
    right_index: int
    observed: float


@dataclass(frozen=True)
class _CircuitGeometry:
    nodes: tuple[VascularNode, ...]
    edges: tuple[VascularEdge, ...]
    supply_terminal_by_ordinal: tuple[tuple[int, str], ...]
    return_terminal_by_ordinal: tuple[tuple[int, str], ...]
    search_effort: VascularSearchEffort | None = None


@dataclass(frozen=True)
class _TerminalPortPair:
    supply: tuple[float, float, float]
    returning: tuple[float, float, float]

    @property
    def midpoint(self) -> tuple[float, float, float]:
        return tuple(
            (supply + returning) / 2.0
            for supply, returning in zip(self.supply, self.returning)
        )


@dataclass(frozen=True)
class _PairedCorridorSection:
    medial: tuple[tuple[float, float, float], ...]
    transverse_normals: tuple[tuple[float, float, float], ...]
    supply: tuple[tuple[float, float, float], ...]
    returning: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class _CircuitConstructionDomain:
    """Constructor-neutral problem statement for one circulation circuit."""

    region_id: str
    mirrored: bool
    representative_count: int
    terminal_pair_count: int
    territory_parts: tuple[dict, ...]
    containment_parts: tuple[dict, ...]
    corridor: "_MacroCorridorSection"
    paired_corridor: _PairedCorridorSection
    per_terminal_flow: float
    macro_node_count: int
    maximum_tree_radius: float


def _index_structural_cost_field(
    field: _StructuralCostField,
) -> _StructuralCostIndex:
    """Derive the deterministic nearest-section index from the frozen field."""
    import numpy as np
    from scipy.spatial import cKDTree

    normalized_stress = np.asarray(
        field.normalized_von_mises_stress, dtype=np.float64
    )
    normalized_stress.setflags(write=False)
    return _StructuralCostIndex(
        spatial_index=cKDTree(
            np.asarray(field.centers_metres, dtype=np.float64)
        ),
        normalized_von_mises_stress=normalized_stress,
        metres_per_world_unit=field.metres_per_world_unit,
        cost_weight=field.cost_weight,
    )


def _segment_structural_cost_multiplier(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    structural_cost_index: _StructuralCostIndex | None,
) -> float:
    if structural_cost_index is None:
        return 1.0
    import numpy as np

    midpoint = tuple(
        0.5 * (left_value + right_value)
        for left_value, right_value in zip(left, right)
    )
    sample_points = np.asarray((left, midpoint, right), dtype=np.float64)
    reflected_sample_points = sample_points * np.asarray(
        (-1.0, 1.0, 1.0), dtype=np.float64
    )
    sample_points_metres = (
        structural_cost_index.metres_per_world_unit
        * np.vstack((sample_points, reflected_sample_points))
    )
    _distance, indices = structural_cost_index.spatial_index.query(
        sample_points_metres, workers=1
    )
    return 1.0 + structural_cost_index.cost_weight * float(
        np.max(
            structural_cost_index.normalized_von_mises_stress[indices]
        )
    )
