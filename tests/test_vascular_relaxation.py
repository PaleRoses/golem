"""Deterministic staged vascular relaxation laws."""

from __future__ import annotations

import json

from functools import cache
from pathlib import Path
from unittest.mock import Mock

import pytest

from golem.kernel import sheaf
from golem.kernel.anatomy.balance import (
    _ClosedVascularCarrier,
    _ClosedVascularInterpreter,
)
from golem.kernel.anatomy.graph import (
    AcceptedVasculature,
    ClosedVascularGraph,
    DistributingArtery,
    InfeasibleBifurcationObstruction,
    RejectedVasculature,
    VascularEdge,
    VascularGluingObstruction,
    VascularGrowthPhase,
    VascularInfeasibilityPredicate,
    VascularIntersectionObstruction,
    VascularLane,
    VascularNodeKind,
    VascularNode,
    VascularRelaxationStage,
    VascularSearchEffort,
    VascularStratum,
    VascularSearchBudgetObstruction,
)
from golem.kernel.anatomy.project import vasculature_to_dict
from golem.kernel.anatomy.realize import allocation
from golem.kernel.anatomy.realize.bifurcation import (
    _evaluate_bifurcation_candidates,
)
from golem.kernel.anatomy.realize.carriers import (
    AcceptedSearch,
    BudgetExhaustedSearch,
    ExhaustedSearch,
    FeasibleBifurcation,
    RejectedBifurcationCandidate,
    _CircuitGeometry,
    _LocalVascularTree,
    _MacroCorridorSection,
)
from golem.kernel.anatomy.realize import cco, parity
from golem.kernel.anatomy.realize.constructor import (
    DEFAULT_CIRCUIT_CONSTRUCTOR,
    realize_circuit,
)
from golem.kernel.anatomy.realize.cco import (
    _InsertionBatch,
    _InsertionContinuation,
    _descend_insertion_queue,
    _final_infeasible_obstruction,
    _insert_terminal_candidates,
    _vascular_search_budget_obstruction,
)
from golem.kernel.anatomy.realize.corridor import (
    _adjusted_corridor_candidates,
    _paired_corridor_section,
    _shared_pump_interface_points,
)
from golem.kernel.anatomy.vocabulary import (
    AcceptedAnatomy,
    SealedVascularConfig,
)
from golem.kernel.body import compile_file


_QUADRUPED_PATH = (
    Path(__file__).parent / "fixtures" / "quadruped_trial_body.json"
)
_KNIGHT_PATH = (
    Path(__file__).parents[1] / "specs" / "knight_body.json"
)


def _obstruction(
    observed: float = 0.0,
    *,
    phase: VascularGrowthPhase = VascularGrowthPhase.INSERTION,
    lane: VascularLane = VascularLane.SUPPLY,
    terminal_index: int | None = 0,
    split_edge: tuple[int, int] | None = (0, 1),
) -> InfeasibleBifurcationObstruction:
    return InfeasibleBifurcationObstruction(
        region_id="probe",
        phase=phase,
        lane=lane,
        predicate=VascularInfeasibilityPredicate.WALL_CONTAINMENT,
        required=1.0,
        observed=observed,
        terminal_index=terminal_index,
        split_edge=split_edge,
    )


def _tree(tag: float) -> _LocalVascularTree:
    return _LocalVascularTree(
        positions=((tag, 0.0, 0.0),),
        parents=(-1,),
        kinds=(VascularNodeKind.CORRIDOR,),
        terminal_flows=(1.0,),
        terminal_ordinals=(None,),
    )


def test_bifurcation_candidate_result_algebra_is_closed() -> None:
    tree = _LocalVascularTree(
        positions=((0.0, 0.0, 0.0), (0.0, 0.1, 0.0)),
        parents=(-1, 0),
        kinds=(VascularNodeKind.CORRIDOR, VascularNodeKind.CORRIDOR),
        terminal_flows=(0.0, 1.0),
        terminal_ordinals=(None, 0),
    )
    results = _evaluate_bifurcation_candidates(
        tree,
        ((0, 1),),
        (0.05, 0.05, 0.0),
        1.0,
        1,
        VascularNodeKind.SUPPLY_TERMINAL,
        "probe",
        VascularLane.SUPPLY,
        (
            {
                "id": "territory",
                "type": "blob",
                "center": [0.0, 0.05, 0.0],
                "size": [1.0, 1.0, 1.0],
            },
        ),
        SealedVascularConfig(
            wall_clearance=0.0,
            vessel_clearance=0.0,
            nonincident_centerline_separation=0.0,
            minimum_segment_length=1.0e-8,
        ),
        subtree_flows=(1.0, 1.0),
        minimum_terminal_flow=1.0,
        escaped_ancestor_margins=(),
    )
    assert len(results) == 1
    assert isinstance(
        results[0], FeasibleBifurcation | RejectedBifurcationCandidate
    )


def test_first_feasible_terminal_dead_end_backtracks_to_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = _tree(0.0)
    dead_end = _tree(10.0)
    viable = _tree(20.0)
    complete = _tree(30.0)

    def fake_insertions(*args: object, **_kwargs: object) -> _InsertionBatch:
        tree = args[0]
        terminal_ordinal = args[3]
        assert isinstance(tree, _LocalVascularTree)
        assert isinstance(terminal_ordinal, int)
        candidates = (
            (FeasibleBifurcation(0.0, dead_end),)
            if tree == start and terminal_ordinal == 0
            else (FeasibleBifurcation(1.0, viable),)
            if tree == start and terminal_ordinal == 1
            else (FeasibleBifurcation(0.0, complete),)
            if tree == viable and terminal_ordinal == 0
            else ()
        )
        return _InsertionBatch(
            candidates,
            () if candidates else (_obstruction(),),
            frozenset(),
            (),
            1,
        )

    monkeypatch.setattr(cco, "_insert_terminal_candidates", fake_insertions)
    result = _descend_insertion_queue(
        (
            _InsertionContinuation(
                start,
                ((0, (1.0, 0.0, 0.0)), (1, (2.0, 0.0, 0.0))),
                0.0,
            ),
        ),
        1.0,
        VascularNodeKind.SUPPLY_TERMINAL,
        (),
        "probe",
        SealedVascularConfig(),
        VascularLane.SUPPLY,
        (),
        None,
        16,
        (),
        obstructions=(),
    )
    assert isinstance(result, AcceptedSearch)
    assert result.value == complete
    assert result.evaluated_state_count == 3


def test_fixed_corridor_failure_succeeds_after_waypoint_adjustment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corridor = _MacroCorridorSection(
        ((0.0, 0.0, 0.0), (0.0, 0.015, 0.0), (0.0, 0.1, 0.0)),
        (1,),
    )
    adjusted = _adjusted_corridor_candidates(
        corridor, SealedVascularConfig(), include_combined=False
    )
    assert adjusted
    fixed_section = _paired_corridor_section(corridor.points, 0.004)
    adjusted_section = _paired_corridor_section(adjusted[0], 0.004)
    assert fixed_section is not None
    assert adjusted_section is not None
    tree_pair = (_tree(1.0), _tree(2.0))

    def fake_greedy(section: object, *_args: object) -> object:
        return _obstruction() if section == fixed_section else tree_pair

    monkeypatch.setattr(parity, "_greedy_corridor_tree_pair", fake_greedy)
    assert isinstance(
        parity._greedy_corridor_tree_pair(
            fixed_section, (), 1, "probe", SealedVascularConfig(), 1.0, (), 0, None
        ),
        InfeasibleBifurcationObstruction,
    )
    result = parity._descend_greedy_corridors(
        (adjusted_section,),
        (),
        1,
        "probe",
        SealedVascularConfig(),
        1.0,
        (),
        0,
        None,
    )
    assert isinstance(result, AcceptedSearch)
    assert result.value == tree_pair


def test_wider_search_considers_edges_beyond_twelve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _LocalVascularTree(
        positions=tuple((float(index), 0.0, 0.0) for index in range(14)),
        parents=(-1, *tuple(0 for _index in range(13))),
        kinds=tuple(VascularNodeKind.CORRIDOR for _index in range(14)),
        terminal_flows=(0.0, *tuple(1.0 for _index in range(13))),
        terminal_ordinals=tuple(None for _index in range(14)),
    )
    ranked_edges = tuple((0, child) for child in range(1, 14))

    def fake_evaluate(*args: object, **_kwargs: object) -> object:
        candidate_edges = args[1]
        assert isinstance(candidate_edges, tuple)
        assert all(isinstance(split_edge, tuple) for split_edge in candidate_edges)
        return tuple(
            FeasibleBifurcation(0.0, tree)
            if split_edge == ranked_edges[-1]
            else RejectedBifurcationCandidate(
                _obstruction(split_edge=split_edge)
            )
            for split_edge in candidate_edges
        )

    monkeypatch.setattr(cco, "_ranked_insertion_edges", lambda *_args: ranked_edges)
    monkeypatch.setattr(cco, "_evaluate_bifurcation_candidates", fake_evaluate)
    fixed = _insert_terminal_candidates(
        tree,
        (20.0, 0.0, 0.0),
        1.0,
        13,
        VascularNodeKind.SUPPLY_TERMINAL,
        (),
        "probe",
        SealedVascularConfig(),
        VascularLane.SUPPLY,
        search_all_edges=False,
    )
    wider = _insert_terminal_candidates(
        tree,
        (20.0, 0.0, 0.0),
        1.0,
        13,
        VascularNodeKind.SUPPLY_TERMINAL,
        (),
        "probe",
        SealedVascularConfig(),
        VascularLane.SUPPLY,
        search_all_edges=True,
    )
    assert fixed.candidates == ()
    assert fixed.attempted_candidate_count == 12
    assert len(wider.candidates) == 1
    assert wider.attempted_candidate_count == 13


def test_exhaustive_failure_selects_enriched_limiting_evidence() -> None:
    obstructions = (
        _obstruction(
            0.5,
            phase=VascularGrowthPhase.SEED,
            lane=VascularLane.RETURN,
            terminal_index=None,
            split_edge=None,
        ),
        _obstruction(
            -1.0,
            phase=VascularGrowthPhase.INSERTION,
            lane=VascularLane.SUPPLY,
            terminal_index=4,
            split_edge=(2, 7),
        ),
    )
    exhausted = _descend_insertion_queue(
        (),
        1.0,
        VascularNodeKind.SUPPLY_TERMINAL,
        (),
        "probe",
        SealedVascularConfig(),
        VascularLane.SUPPLY,
        (),
        None,
        9,
        (),
        obstructions=obstructions,
        evaluated_state_count=9,
        attempted_candidate_count=37,
    )
    assert isinstance(exhausted, ExhaustedSearch)
    final = _final_infeasible_obstruction(
        exhausted.obstructions, "probe", exhausted.attempted_candidate_count
    )
    assert isinstance(final, InfeasibleBifurcationObstruction)
    assert final.phase is VascularGrowthPhase.INSERTION
    assert final.lane is VascularLane.SUPPLY
    assert final.required == 1.0
    assert final.observed == -1.0
    assert final.attempted_candidate_count == 37


def test_budget_exhaustion_never_claims_infeasibility() -> None:
    search = _descend_insertion_queue(
        (
            _InsertionContinuation(
                _tree(0.0), ((0, (1.0, 0.0, 0.0)),), 0.0
            ),
        ),
        1.0,
        VascularNodeKind.SUPPLY_TERMINAL,
        (),
        "probe",
        SealedVascularConfig(),
        VascularLane.SUPPLY,
        (),
        None,
        0,
        (),
        obstructions=(_obstruction(-0.25),),
        evaluated_state_count=3,
        attempted_candidate_count=19,
    )
    assert isinstance(search, BudgetExhaustedSearch)
    obstruction = _vascular_search_budget_obstruction("probe", search, 3)
    assert isinstance(obstruction, VascularSearchBudgetObstruction)
    assert not isinstance(obstruction, InfeasibleBifurcationObstruction)
    assert obstruction.required_state_budget == 3
    assert obstruction.observed_evaluated_states == 3
    assert obstruction.remaining_queue_size == 1
    view = vasculature_to_dict(RejectedVasculature((obstruction,)))
    assert view["obstructions"][0]["kind"] == "VascularSearchBudget"


@cache
def _compiled_knight():
    return compile_file(str(_KNIGHT_PATH))


def test_fixed_corridor_receipt_records_regional_search_effort() -> None:
    compiled = _compiled_knight()
    accepted = compiled.anatomy
    assert isinstance(accepted, AcceptedAnatomy)
    result = allocation.realize_vasculature(accepted, compiled.graph)
    assert isinstance(result, AcceptedVasculature)
    assert result.receipt.search_effort_by_region == tuple(
        (
            circuit.capillary_bed.region.region_id,
            VascularSearchEffort(VascularRelaxationStage.FIXED_CORRIDOR, 1),
        )
        for circuit in accepted.circuits
    )


def test_vascular_realization_cache_tracks_only_consumed_body_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = _compiled_knight()
    accepted = compiled.anatomy
    assert isinstance(accepted, AcceptedAnatomy)
    first_realization = object()
    changed_realization = object()
    uncached = Mock(side_effect=(first_realization, changed_realization))
    monkeypatch.setattr(allocation, "_realize_vasculature_uncached", uncached)
    monkeypatch.setattr(allocation, "_store_bypassed", lambda: False)
    monkeypatch.setattr(
        allocation, "_load_vasculature_result", lambda _key: None
    )
    monkeypatch.setattr(
        allocation, "_store_vasculature_result", lambda _key, _result: None
    )
    allocation._realize_vasculature_cached.cache_clear()

    first = allocation.realize_vasculature(accepted, compiled.graph)
    second = allocation.realize_vasculature(
        accepted,
        {
            **compiled.graph,
            "conduits": ({"id": "irrelevant-surface-view"},),
            "appendages": ({"id": "irrelevant-derived-view"},),
        },
    )
    changed = allocation.realize_vasculature(
        accepted,
        {
            **compiled.graph,
            "parts": (
                {
                    **compiled.graph["parts"][0],
                    "id": "relevant-vascular-geometry",
                },
                *compiled.graph["parts"][1:],
            ),
        },
    )

    assert first is second is first_realization
    assert changed is changed_realization
    assert uncached.call_count == 2


def test_cold_vascular_realization_bypasses_memory_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = _compiled_knight()
    accepted = compiled.anatomy
    assert isinstance(accepted, AcceptedAnatomy)
    first_realization = object()
    second_realization = object()
    uncached = Mock(side_effect=(first_realization, second_realization))
    monkeypatch.setenv("GOLEM_CIRCUIT_CACHE_COLD", "1")
    monkeypatch.setattr(allocation, "_realize_vasculature_uncached", uncached)
    allocation._realize_vasculature_cached.cache_clear()

    first = allocation.realize_vasculature(accepted, compiled.graph)
    second = allocation.realize_vasculature(accepted, compiled.graph)

    assert first is first_realization
    assert second is second_realization
    assert uncached.call_count == 2
    assert uncached.call_args_list[0] == uncached.call_args_list[1]


@cache
def _compiled_quadruped():
    return compile_file(str(_QUADRUPED_PATH))


def _quadruped_circuit_results() -> tuple[_CircuitGeometry | RejectedVasculature, ...]:
    compiled = _compiled_quadruped()
    accepted = compiled.anatomy
    assert isinstance(accepted, AcceptedAnatomy)
    config = SealedVascularConfig()
    allocation_by_region = dict(allocation.allocate_terminal_pairs(accepted, config))
    provenance = compiled.graph.get("intent", {}).get("provenance", {})
    landmarks = compiled.graph.get("intent", {}).get("landmarks", {})
    part_by_id = {part["id"]: part for part in compiled.graph.get("parts", ())}
    bone_use_count = {
        bone_id: sum(
            int(bone_id in circuit.bones) for circuit in accepted.circuits
        )
        for bone_id in frozenset(
            bone for circuit in accepted.circuits for bone in circuit.bones
        )
    }
    shared_interfaces = _shared_pump_interface_points(accepted, landmarks)
    return tuple(
        realize_circuit(
            DEFAULT_CIRCUIT_CONSTRUCTOR,
            circuit,
            allocation_by_region[circuit.capillary_bed.region.region_id],
            accepted,
            part_by_id,
            provenance,
            landmarks,
            bone_use_count,
            shared_interfaces,
            config,
            None,
        )
        for circuit in accepted.circuits
    )


def test_quadruped_exhaustive_relaxation_is_deterministic() -> None:
    first_sections = _quadruped_circuit_results()
    second_sections = _quadruped_circuit_results()
    assert first_sections == second_sections
    assert all(
        isinstance(section, _CircuitGeometry) for section in first_sections
    )
    assert all(
        section.search_effort
        == VascularSearchEffort(VascularRelaxationStage.FIXED_CORRIDOR, 1)
        for section in first_sections
        if isinstance(section, _CircuitGeometry)
    )
    allocation._realize_vasculature_cached.cache_clear()
    first = allocation.realize_vasculature(
        _compiled_quadruped().anatomy, _compiled_quadruped().graph
    )
    allocation._realize_vasculature_cached.cache_clear()
    second = allocation.realize_vasculature(
        _compiled_quadruped().anatomy, _compiled_quadruped().graph
    )
    assert first == second
    assert isinstance(first, RejectedVasculature)
    assert first.obstructions
    assert all(
        isinstance(obstruction, VascularIntersectionObstruction)
        for obstruction in first.obstructions
    )


def test_solver_rejection_passes_through_host_address_localization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = _compiled_knight()
    accepted = compiled.anatomy
    assert isinstance(accepted, AcceptedAnatomy)
    allocations = tuple(
        (
            circuit.capillary_bed.region.region_id,
            4,
        )
        for circuit in accepted.circuits
    )
    rejection = RejectedVasculature(
        (
            VascularGluingObstruction(
                "closed-vascular-field",
                "NonConvergentBalance",
            ),
        )
    )
    localized = object()
    localize = Mock(return_value=localized)
    monkeypatch.setattr(
        allocation,
        "allocate_terminal_pairs",
        lambda _accepted, _config: allocations,
    )
    monkeypatch.setattr(allocation, "_realize_circuits", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(allocation, "_glue_circuit_sections", lambda _sections: object())
    monkeypatch.setattr(
        allocation,
        "_validate_vascular_geometry",
        lambda *_args: Mock(obstructions=()),
    )
    monkeypatch.setattr(
        allocation,
        "_calibrate_and_solve",
        lambda *_args: rejection,
    )
    monkeypatch.setattr(allocation, "_localized_rejection", localize)

    canonical_body = json.dumps(compiled.graph)
    result = allocation._realize_vasculature_uncached(
        accepted,
        canonical_body,
        SealedVascularConfig(),
        None,
    )

    assert result is localized
    canonical_graph = json.loads(canonical_body)
    localize.assert_called_once_with(
        rejection,
        tuple(canonical_graph.get("parts", ())),
        canonical_graph.get("intent", {}).get("provenance", {}),
    )


def test_accepted_relaxation_reaches_post_acceptance_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = _compiled_quadruped()
    accepted = compiled.anatomy
    assert isinstance(accepted, AcceptedAnatomy)
    validator = Mock(wraps=allocation._validate_vascular_geometry)
    monkeypatch.setattr(allocation, "_validate_vascular_geometry", validator)
    monkeypatch.setattr(allocation, "_store_bypassed", lambda: True)
    allocation._realize_vasculature_cached.cache_clear()
    result = allocation.realize_vasculature(accepted, compiled.graph)
    assert isinstance(result, RejectedVasculature)
    assert all(
        isinstance(obstruction, VascularIntersectionObstruction)
        for obstruction in result.obstructions
    )
    assert validator.call_count == 1


def test_nonconvergent_balance_lifts_cells_and_interfaces_to_vascular_addresses() -> None:
    graph = ClosedVascularGraph(
        nodes=(
            VascularNode(
                "pump-outlet",
                (0.0, 0.0, 0.0),
                VascularNodeKind.PUMP_OUTLET,
                "axial_core",
            ),
            VascularNode(
                "wing-node",
                (1.0, 0.0, 0.0),
                VascularNodeKind.CORRIDOR,
                "wing_tips",
            ),
            VascularNode(
                "pump-inlet",
                (2.0, 0.0, 0.0),
                VascularNodeKind.PUMP_INLET,
                "axial_core",
            ),
        ),
        edges=(
            VascularEdge(
                "edge-0",
                "pump-outlet",
                "wing-node",
                DistributingArtery("skeleton/wing"),
                VascularStratum.SUPPLY,
                0.001,
                0.0,
                0.0,
            ),
            VascularEdge(
                "edge-1",
                "wing-node",
                "pump-inlet",
                DistributingArtery("skeleton/wing"),
                VascularStratum.SUPPLY,
                0.01,
                0.0,
                0.0,
            ),
        ),
        pump_outlet_node_id="pump-outlet",
        pump_inlet_node_id="pump-inlet",
    )
    interrogation = sheaf.BalanceInterrogation(
        sheaf.BalanceTopologyWitness(
            1,
            (
                sheaf.BalanceBridgeWitness(
                    sheaf.BalanceInterfaceId("edge-0"),
                    sheaf.CellId(0),
                    sheaf.CellId(1),
                    1.0e-6,
                ),
            ),
        ),
        sheaf.BalanceConditioningWitness(
            2.0e6,
            1.0e12,
            (
                sheaf.BalanceConditionCellWitness(
                    sheaf.CellId(1), 1.0e6, 1.0e12
                ),
            ),
        ),
        sheaf.BalanceDemandWitness(
            maximum_stop_residual=3.0e-10,
            residual_tolerance=1.0e-10,
            relative_residual=3.0e-13,
            relative_tolerance=1.0e-13,
            iteration_cap=30,
            iterations_used=30,
            exhausted=True,
            worst_cells=(
                sheaf.BalanceResidualCellWitness(
                    sheaf.CellId(1), -3.0e-10
                ),
            ),
        ),
        sheaf.BalanceBudgetWitness(30, 30, 1.0e-13, True),
    )
    carrier = _ClosedVascularCarrier(
        graph,
        {node.node_id: index for index, node in enumerate(graph.nodes)},
    )
    result = _ClosedVascularInterpreter(SealedVascularConfig()).lift(
        carrier,
        sheaf.Rejected(
            (
                sheaf.NonConvergentBalanceObstruction(
                    sheaf.BalanceProblemId("closed-vascular-balance"),
                    30,
                    interrogation,
                ),
            )
        ),
    )

    assert isinstance(result, RejectedVasculature)
    assert len(result.obstructions) == 1
    obstruction = result.obstructions[0]
    assert isinstance(obstruction, VascularGluingObstruction)
    assert obstruction.reason == (
        "NonConvergentBalance @closed-vascular-balance: solver_info=30"
    )
    assert tuple(
        segment.edge_id for segment in obstruction.failing_segments
    ) == ("edge-0",)
    gluing = obstruction.gluing_interrogation
    assert gluing is not None
    assert gluing.conditioning.worst_sections[0].node_id == "wing-node"
    assert gluing.conditioning.worst_sections[0].region_id == "wing_tips"
    assert gluing.demand.worst_sections[0].node_id == "wing-node"
