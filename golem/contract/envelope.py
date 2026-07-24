from __future__ import annotations

from golem.contract.model import ContractSection
from golem.kernel.anatomy.graph import (
    InfeasibleBifurcationObstruction,
    VascularInfeasibilityPredicate,
    VascularRelaxationStage,
    VascularSearchBudgetObstruction,
)
from golem.kernel.anatomy.vocabulary import SealedVascularConfig


def render() -> str:
    config = SealedVascularConfig()
    stages = " then ".join(stage.value for stage in VascularRelaxationStage)
    geometric_predicates = ", ".join(
        predicate.value
        for predicate in VascularInfeasibilityPredicate
        if predicate is not VascularInfeasibilityPredicate.SEARCH_EXHAUSTED
    )
    return f"""FEASIBLE ENVELOPE

Anatomy intent that passes `check` can still be rejected by `compile`. The vascular solver realizes every limb circuit as physical corridors buried in flesh, and those corridors must fit. This section states the envelope the solver enforces. Every threshold below is either a sealed `SealedVascularConfig` value, projected live, or a fixed solver constant stated as such.

CORRIDOR
A limb circuit must admit a whole pump-to-territory corridor whose two offset lanes stay embedded in flesh. The lanes run at the sealed half-separation {config.terminal_port_half_separation} world units on either side of the centerline. They must not cross each other at bends, must hold capsule clearance {config.vessel_clearance} from each other and from every exchange segment, and must remain buried in flesh along the whole route. Sharp direction changes near the pump are the dominant failure: the pump carries a fixed detour of 0.015 world Y (a solver constant), and when the two offset polylines cross through that bend the circuit is fatal. Crossed lanes are the most common rejection cause.

TERRITORY
Terminal territory must offer sites at least ~0.00053 world units inside the body surface (a solver constant) and at least {config.wall_clearance} world units apart; site spacing equals the sealed wall clearance. Terminal vessels are realized at the sealed terminal radius {config.terminal_radius}. A bed whose flesh is too thin, or whose demanded sites collide, has no admissible territory.

AERIAL COVER
An exchange-bed path follows the flesh topology of its own region and pump-to-bed route; it is not constrained to the host bone centerline. The solver may route a capsule over a bone, through a poll/neck notch, or around a joint, so every point on that "aerial" excursion still needs authored flesh above, below, and around it. Cover is provenance-scoped: only flesh owned by the bed's own region/route bones participates. Nearby flesh from another bone — even thorax or pump flesh occupying the same world space — cannot cover that bed. Carrier-floor scale rows are suggestions, not applied geometry, and cannot absorb an escape. A `CapsuleEscapeObstruction` therefore asks for route-local cover at the reported failing segment, not arbitrary inflation of the named host bone.

BIFURCATION
Every bifurcation must find a valid split edge. A split is valid only when all three branches reach at least the sealed minimum segment length {config.minimum_segment_length}, centerline separation reaches at least {config.nonincident_centerline_separation}, capsule clearance reaches at least {config.vessel_clearance}, and wall margin reaches at least {config.wall_clearance}. If no eligible edge satisfies all four at once after the staged search below, the bifurcation is infeasible and the whole vascular realization is rejected.

STAGED RELAXATION
Realization is failure-driven across the ordered stages {stages}. FIXED_CORRIDOR reuses the skeleton-derived medial corridor and fixed interface points. Only on its failure does WAYPOINT_ADJUSTMENT run: it unlocks solver-created corridor points within a deterministic local stencil, stepped by the sealed {config.waypoint_stencil_step} world units and bounded by the sealed maximum adjustment {config.maximum_waypoint_adjustment} world units, with at most {config.corridor_candidate_budget} candidate corridors examined. Only on its failure does WIDER_SEARCH run: cost-ordered depth-first backtracking over all eligible insertion edges and seed continuations, bounded by the sealed state budget of {config.search_state_budget} evaluated states. Each stage changes the route, never anatomy ownership, terminal demand, or vessel radii.

The staged search ends in a strict trichotomy of typed outcomes; none is confused with another. Accepted: a complete supply/return pair, with the deepest stage reached recorded in the receipt. Proven infeasible: `{InfeasibleBifurcationObstruction.__name__}` carrying one limiting geometric predicate ({geometric_predicates}) with its required and observed values and the total attempted-candidate count, emitted only when the finite configured search space is fully enumerated. Exhausted: the same obstruction with the `{VascularInfeasibilityPredicate.SEARCH_EXHAUSTED.value}` predicate when enumeration completed with no single dominating geometric floor. Budget cutoff: `{VascularSearchBudgetObstruction.__name__}` when the {config.search_state_budget}-state budget is reached while candidates remain; it carries evaluated-state count, remaining queue size, and the limiting predicate, and it never counterfeits infeasibility.

GLOBAL COUPLING
The envelope is globally conditional. One limb's feasibility depends on every other limb present: terminal allocation, shared-bone territory ownership, and shared pump-interface waypoints all shift when any limb is added, removed, or moved. `The same terminal` is not a stable identity across edits. A corridor that fit before an edit may not survive it, and a rejection may clear when an unrelated limb changes.

ABSOLUTE UNITS
Every threshold here is an absolute world-unit floor; the solver is not scale-invariant. Uniformly scaling a body does not scale these floors, so a small body can starve corridors that a larger one admits. Geometric defects survive uniform scaling exactly: crossed centerlines stay crossed at any size.

See READING RECEIPTS: a green `CIRCULATION` line proves topological closure only. Require `VASCULAR feasibility: ACCEPTED` from `check`, or a successful `compile`, before trusting that these constraints are met.
"""


SECTION = ContractSection("envelope", render)
