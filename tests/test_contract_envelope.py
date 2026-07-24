from __future__ import annotations

from golem.contract import SECTIONS, render_section, section_names
from golem.kernel.anatomy.graph import (
    InfeasibleBifurcationObstruction,
    VascularInfeasibilityPredicate,
    VascularRelaxationStage,
    VascularSearchBudgetObstruction,
)
from golem.kernel.anatomy.vocabulary import SealedVascularConfig


def test_envelope_section_is_registered_after_receipts() -> None:
    names = section_names()
    assert "envelope" in names
    assert names.index("envelope") == names.index("receipts") + 1


def test_envelope_section_is_nonempty_and_titled() -> None:
    rendered = render_section("envelope")
    assert "FEASIBLE ENVELOPE" in rendered
    assert rendered.endswith("\n")


def test_envelope_projects_live_sealed_thresholds() -> None:
    config = SealedVascularConfig()
    rendered = render_section("envelope")
    for value in (
        config.terminal_port_half_separation,
        config.vessel_clearance,
        config.wall_clearance,
        config.terminal_radius,
        config.minimum_segment_length,
        config.nonincident_centerline_separation,
    ):
        assert str(value) in rendered


def test_envelope_states_the_global_and_scale_caveats() -> None:
    rendered = render_section("envelope")
    assert "globally conditional" in rendered
    assert "not scale-invariant" in rendered


def test_receipts_cross_references_the_envelope() -> None:
    assert "FEASIBLE ENVELOPE" in render_section("receipts")
    assert any(section.name == "envelope" for section in SECTIONS)


def test_envelope_drops_the_stale_twelve_split_edge_claim() -> None:
    assert "twelve" not in render_section("envelope")


def test_envelope_projects_live_staged_relaxation_budgets() -> None:
    config = SealedVascularConfig()
    rendered = render_section("envelope")
    for value in (
        config.search_state_budget,
        config.corridor_candidate_budget,
        config.waypoint_stencil_step,
        config.maximum_waypoint_adjustment,
    ):
        assert str(value) in rendered
    for stage in VascularRelaxationStage:
        assert stage.value in rendered


def test_envelope_states_the_search_trichotomy() -> None:
    rendered = render_section("envelope")
    assert InfeasibleBifurcationObstruction.__name__ in rendered
    assert VascularSearchBudgetObstruction.__name__ in rendered
    assert VascularInfeasibilityPredicate.SEARCH_EXHAUSTED.value in rendered
