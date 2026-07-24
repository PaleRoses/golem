"""Hierarchical vascular co-design: sparse intent descends to closed flow."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from functools import cache

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from golem import paths
from golem.conduits import apply_conduits
from golem.conduits.emit import vascular_stratum_mesh
from golem.conduits.harness import (
    AcceptedCommand,
    HarnessState,
    RejectedCommand,
    run_line,
)
from golem.kernel import engine
from golem.kernel.anatomy import (
    AcceptedAnatomy,
    AcceptedVasculature,
    CapsuleEscapeObstruction,
    ClosedVascularGraph,
    DemandDeliveryMismatchObstruction,
    DisconnectedVascularGraphObstruction,
    EmptyPerfusionTerritoryObstruction,
    InfeasibleBifurcationObstruction,
    InsufficientTerminalSitesObstruction,
    InvalidStructuralGuidanceObstruction,
    MissingProvenanceObstruction,
    RejectedVasculature,
    ReversedVascularFlowObstruction,
    SealedVascularConfig,
    SymmetryMismatchObstruction,
    VascularGluingObstruction,
    VascularGrowthPhase,
    VascularInfeasibilityPredicate,
    VascularIntersectionObstruction,
    VascularLane,
    VascularNode,
    VascularNodeKind,
    VascularSolverResidualObstruction,
    VascularStratum,
    VascularSymmetryPredicate,
    VascularSymmetryWitness,
    allocate_terminal_pairs,
    realize_vasculature,
    vasculature_to_dict,
    _LocalVascularTree,
    _candidates_by_minimum_total_travel,
    _certified_segment_capsule_margin,
    _edge_clearance,
    _extend_corridor,
    _paired_corridor_section,
    _possible_vascular_edge_pairs,
    _segment_capsule_margin,
    _StructuralCostField,
    _index_structural_cost_field,
    _segment_structural_cost_multiplier,
    _vascular_edges_are_incident,
)
from golem.kernel.anatomy.realize.clearance import _validate_vascular_geometry
from golem.kernel.body import CompiledBody, Compiler, compile_file


_BODY_PATH = paths.SPECS / "knight_body.json"


@cache
def _compiled_body() -> CompiledBody:
    return compile_file(str(_BODY_PATH))


@cache
def _accepted_vasculature() -> AcceptedVasculature:
    compiled = _compiled_body()
    assert isinstance(compiled.anatomy, AcceptedAnatomy)
    result = realize_vasculature(compiled.anatomy, compiled.graph)
    assert isinstance(result, AcceptedVasculature), vasculature_to_dict(result)
    return result


@pytest.fixture(params=(":positive:", ":negative:"))
def vascular_graph_with_unmatched_mirrored_node(
    request: pytest.FixtureRequest,
) -> tuple[ClosedVascularGraph, VascularNode, str]:
    graph = _accepted_vasculature().graph
    node = next(
        candidate
        for candidate in graph.nodes
        if request.param in candidate.node_id
        and candidate.node_id.replace(
            request.param,
            ":negative:" if request.param == ":positive:" else ":positive:",
        )
        in graph.node_by_id
    )
    missing_node_id = node.node_id.replace(
        request.param,
        ":negative:" if request.param == ":positive:" else ":positive:",
    )
    return (
        replace(
            graph,
            nodes=tuple(
                candidate
                for candidate in graph.nodes
                if candidate.node_id != missing_node_id
            ),
            edges=tuple(
                edge
                for edge in graph.edges
                if missing_node_id
                not in (edge.source_node_id, edge.target_node_id)
            ),
        ),
        node,
        missing_node_id,
    )


def _incident_edges(graph, node_id: str, stratum: VascularStratum):
    incoming = tuple(
        edge
        for edge in graph.edges
        if edge.stratum is stratum and edge.target_node_id == node_id
    )
    outgoing = tuple(
        edge
        for edge in graph.edges
        if edge.stratum is stratum and edge.source_node_id == node_id
    )
    return incoming, outgoing


def _all_mapping_keys(value: object) -> frozenset[str]:
    if isinstance(value, dict):
        return frozenset(value).union(
            *tuple(map(_all_mapping_keys, value.values()))
        )
    if isinstance(value, (list, tuple)):
        return frozenset().union(*tuple(map(_all_mapping_keys, value)))
    return frozenset()


def test_sparse_anatomy_owns_no_vessel_geometry_or_seed() -> None:
    payload = json.loads(_BODY_PATH.read_text())["anatomy"]
    assert _all_mapping_keys(payload).isdisjoint(
        {
            "nodes",
            "edges",
            "coordinates",
            "seed",
            "root_part",
            "root_t",
            "count",
        }
    )
    compiled = _compiled_body()
    assert isinstance(compiled.anatomy, AcceptedAnatomy)
    assert compiled.receipt["anatomy"]["status"] == "accepted"


def test_largest_remainder_allocation_is_seedless_fixed_and_bilateral() -> None:
    accepted = _compiled_body().anatomy
    assert isinstance(accepted, AcceptedAnatomy)
    allocation = allocate_terminal_pairs(accepted)
    assert allocation == (
        ("cranial", 28),
        ("forelimb", 14),
        ("hindlimb", 214),
    )
    assert math.fsum(count for _region, count in allocation) == 256
    assert all(count >= 4 for _region, count in allocation)
    assert all(count % 2 == 0 for _region, count in allocation)


def test_mid_bone_interface_glues_without_endpoint_backtracking() -> None:
    assert _extend_corridor(
        ((0.0, 0.2, 0.0),),
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.6, 0.0),
    ) == ((0.0, 0.2, 0.0), (0.0, 0.6, 0.0))


def test_axial_macro_corridor_derives_transverse_supply_return_sections() -> None:
    section = _paired_corridor_section(
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), 0.004
    )
    assert section is not None
    assert section.transverse_normals == (
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    )
    assert section.supply == ((-0.004, 0.0, 0.0), (-0.004, 0.0, 1.0))
    assert section.returning == ((0.004, 0.0, 0.0), (0.004, 0.0, 1.0))


def test_initial_tree_descent_prefers_minimum_total_travel_section() -> None:
    empty_tree = _LocalVascularTree((), (), (), (), ())
    sites = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (4.0, 0.0, 0.0))
    feasible = tuple((ordinal, empty_tree) for ordinal in range(len(sites)))
    ordered = _candidates_by_minimum_total_travel(feasible, sites)
    assert tuple(ordinal for ordinal, _tree in ordered) == (2, 0, 1)


def test_structural_cost_uses_one_prebuilt_index_and_bilateral_maximum() -> None:
    field = _StructuralCostField(
        centers_metres=(
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
        ),
        normalized_von_mises_stress=(1.0, 0.0, 0.0),
        metres_per_world_unit=1.0,
        cost_weight=3.0,
    )
    # One index, built once, then handed to every multiplier call: the prebuilt
    # index is a required parameter that the multiplier can only query, so it is
    # structurally incapable of rebuilding the index it never receives the field for.
    structural_cost_index = _index_structural_cost_field(field)
    positive_cost = _segment_structural_cost_multiplier(
        (0.8, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        structural_cost_index,
    )
    reflected_cost = _segment_structural_cost_multiplier(
        (-0.8, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        structural_cost_index,
    )
    unstressed_cost = _segment_structural_cost_multiplier(
        (0.0, 1.9, 0.0),
        (0.0, 2.1, 0.0),
        structural_cost_index,
    )
    assert positive_cost == reflected_cost == 4.0
    assert unstressed_cost == 1.0
    assert not structural_cost_index.normalized_von_mises_stress.flags.writeable


def test_closed_graph_has_binary_trees_murray_radii_and_exact_reflection() -> None:
    graph = _accepted_vasculature().graph
    bifurcations = tuple(
        node
        for node in graph.nodes
        if node.kind is VascularNodeKind.BIFURCATION
    )
    supply_laws = tuple(
        _incident_edges(graph, node.node_id, VascularStratum.SUPPLY)
        for node in bifurcations
    )
    return_laws = tuple(
        _incident_edges(graph, node.node_id, VascularStratum.RETURN)
        for node in bifurcations
    )
    assert all(
        len(incoming) == 1
        and len(outgoing) == 2
        and incoming[0].radius**3
        == pytest.approx(
            math.fsum(edge.radius**3 for edge in outgoing), rel=1.0e-12
        )
        for incoming, outgoing in supply_laws
        if incoming or outgoing
    )
    assert all(
        len(incoming) == 2
        and len(outgoing) == 1
        and outgoing[0].radius**3
        == pytest.approx(
            math.fsum(edge.radius**3 for edge in incoming), rel=1.0e-12
        )
        for incoming, outgoing in return_laws
        if incoming or outgoing
    )
    by_id = graph.node_by_id
    positive = tuple(
        node for node in graph.nodes if ":positive:" in node.node_id
    )
    assert positive
    assert all(
        by_id[node.node_id.replace(":positive:", ":negative:")].position
        == (-node.position[0], node.position[1], node.position[2])
        for node in positive
    )
    assert _accepted_vasculature().receipt.maximum_symmetry_error == 0.0
    assert all(edge.solved_flow > 0.0 for edge in graph.edges)


def test_missing_mirror_node_produces_typed_symmetry_obstruction(
    vascular_graph_with_unmatched_mirrored_node: tuple[
        ClosedVascularGraph, VascularNode, str
    ],
) -> None:
    graph, node, missing_node_id = vascular_graph_with_unmatched_mirrored_node
    validation = _validate_vascular_geometry(
        graph,
        tuple(_compiled_body().graph.get("parts", ())),
        SealedVascularConfig(),
    )
    symmetry_obstructions = tuple(
        obstruction
        for obstruction in validation.obstructions
        if isinstance(obstruction, SymmetryMismatchObstruction)
    )
    assert symmetry_obstructions == (
        SymmetryMismatchObstruction(
            region_id=node.region_id or "bilateral",
            predicate=VascularSymmetryPredicate.MIRROR_PAIRING,
            required=1.0,
            observed=0.0,
            witness=VascularSymmetryWitness(node.node_id, missing_node_id),
        ),
    )
    assert vasculature_to_dict(
        RejectedVasculature(symmetry_obstructions)
    )["obstructions"] == (
        {
            "kind": "SymmetryMismatch",
            "address": node.node_id,
            "region_id": node.region_id or "bilateral",
            "predicate": "MirrorPairing",
            "required": 1.0,
            "observed": 0.0,
            "witness": {
                "node_id": node.node_id,
                "expected_mirror_node_id": missing_node_id,
            },
        },
    )


def test_every_exchange_bed_lies_on_a_pump_supply_exchange_return_path() -> None:
    graph = _accepted_vasculature().graph
    node_index = {
        node.node_id: index for index, node in enumerate(graph.nodes)
    }
    directed_pairs = tuple(
        (node_index[edge.source_node_id], node_index[edge.target_node_id])
        for edge in graph.edges
    )
    adjacency = csr_matrix(
        (
            np.ones(len(directed_pairs), dtype=np.int8),
            tuple(zip(*directed_pairs)),
        ),
        shape=(len(graph.nodes), len(graph.nodes)),
    )
    from_outlet = frozenset(
        map(
            int,
            breadth_first_order(
                adjacency,
                i_start=node_index[graph.pump_outlet_node_id],
                directed=True,
                return_predecessors=False,
            ),
        )
    )
    to_inlet = frozenset(
        map(
            int,
            breadth_first_order(
                adjacency.transpose(),
                i_start=node_index[graph.pump_inlet_node_id],
                directed=True,
                return_predecessors=False,
            ),
        )
    )
    exchange_edges = tuple(
        edge
        for edge in graph.edges
        if edge.stratum is VascularStratum.EXCHANGE
    )
    assert len(exchange_edges) == 256
    assert all(
        node_index[edge.source_node_id] in from_outlet
        and node_index[edge.target_node_id] in to_inlet
        for edge in exchange_edges
    )


def test_geometry_flow_and_demand_receipts_clear_acceptance_thresholds() -> None:
    accepted = _accepted_vasculature()
    receipt = accepted.receipt
    assert len(accepted.graph.nodes) == 1060
    assert len(accepted.graph.edges) == 1314
    assert receipt.minimum_capsule_margin >= 5.0e-4
    assert receipt.maximum_free_cell_residual <= 1.0e-10
    assert receipt.boundary_balance_error <= 1.0e-10
    assert receipt.maximum_delivery_relative_error <= 1.0e-3
    assert receipt.pump_pressure_drop > 0.0
    delivered = dict(receipt.delivered_demand_by_region)
    assert delivered == pytest.approx(
        {"cranial": 0.1, "forelimb": 0.04, "hindlimb": 0.88},
        rel=1.0e-3,
    )
    exchange_flow = math.fsum(
        abs(edge.solved_flow)
        for edge in accepted.graph.edges
        if edge.stratum is VascularStratum.EXCHANGE
    )
    assert exchange_flow == pytest.approx(1.02, rel=1.0e-10)
    assert all(math.isfinite(edge.solved_flow) for edge in accepted.graph.edges)


def test_whole_capsule_certificate_catches_between_sample_escape() -> None:
    parts = tuple(
        {
            "id": f"sample-island-{index}",
            "type": "blob",
            "center": [index / 8.0, 0.0, 0.0],
            "size": [0.05, 0.05, 0.05],
        }
        for index in range(9)
    )
    sampled = _segment_capsule_margin(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        0.001,
        parts,
        9,
    )
    certified = _certified_segment_capsule_margin(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        0.001,
        parts,
        9,
        0.001,
    )
    assert sampled > 0.001
    assert certified < 0.001


def test_every_nonincident_vessel_respects_clearance() -> None:
    graph = _accepted_vasculature().graph
    nonincident_candidates = tuple(
        (left, right)
        for left, right in _possible_vascular_edge_pairs(
            graph.edges,
            graph.node_by_id,
            SealedVascularConfig(),
        )
        if not _vascular_edges_are_incident(left, right)
    )
    assert nonincident_candidates
    assert (
        _accepted_vasculature().receipt.checked_nonincident_pair_count
        == len(nonincident_candidates)
    )
    assert min(
        _edge_clearance(left, right, graph.node_by_id)
        for left, right in nonincident_candidates
    ) >= SealedVascularConfig().vessel_clearance


def test_derived_strata_are_complete_finite_and_deterministic() -> None:
    accepted = _accepted_vasculature()
    compiled = _compiled_body()
    assert isinstance(compiled.anatomy, AcceptedAnatomy)
    assert realize_vasculature(compiled.anatomy, compiled.graph) == accepted
    graph = accepted.graph
    meshes = tuple(
        (stratum, vascular_stratum_mesh(graph, stratum))
        for stratum in VascularStratum
    )
    assert all(
        vertices.shape[1] == 3
        and faces.shape[1] == 3
        and np.isfinite(vertices).all()
        and faces.shape[0]
        == 24 * sum(edge.stratum is stratum for edge in graph.edges)
        for stratum, (vertices, faces) in meshes
    )
    assert all(
        np.array_equal(vertices, vascular_stratum_mesh(graph, stratum)[0])
        and np.array_equal(faces, vascular_stratum_mesh(graph, stratum)[1])
        for stratum, (vertices, faces) in meshes
    )


def test_hindlimb_demand_only_changes_allocation_flow_and_derived_radii() -> None:
    original_spec = json.loads(_BODY_PATH.read_text())
    circulation = original_spec["anatomy"]["overall"]["circulation"]
    revised_beds = tuple(
        ({**bed, "demand": 0.30} if bed["region_id"] == "hindlimb" else bed)
        for bed in circulation["exchange_beds"]
    )
    revised_spec = {
        **original_spec,
        "anatomy": {
            **original_spec["anatomy"],
            "overall": {
                **original_spec["anatomy"]["overall"],
                "circulation": {
                    **circulation,
                    "exchange_beds": list(revised_beds),
                },
            },
        },
    }
    revised = Compiler(revised_spec, spec_dir=paths.SPECS).compile()
    assert revised_spec["skeleton"] == original_spec["skeleton"]
    assert isinstance(revised.anatomy, AcceptedAnatomy)
    revised_vasculature = realize_vasculature(revised.anatomy, revised.graph)
    assert isinstance(revised_vasculature, AcceptedVasculature)
    original = _accepted_vasculature()
    assert (
        revised_vasculature.receipt.terminal_pairs_by_region
        != original.receipt.terminal_pairs_by_region
    )
    assert dict(revised_vasculature.receipt.delivered_demand_by_region)[
        "hindlimb"
    ] == pytest.approx(0.60, rel=1.0e-3)
    assert sorted(edge.radius for edge in revised_vasculature.graph.edges) != sorted(
        edge.radius for edge in original.graph.edges
    )


def test_every_vascular_failure_has_a_typed_serialized_obstruction() -> None:
    obstructions = (
        MissingProvenanceObstruction("limb", "bone"),
        EmptyPerfusionTerritoryObstruction("limb"),
        InsufficientTerminalSitesObstruction("limb", 4, 2),
        InfeasibleBifurcationObstruction(
            region_id="limb",
            phase=VascularGrowthPhase.INSERTION,
            lane=VascularLane.SUPPLY,
            predicate=VascularInfeasibilityPredicate.SEGMENT_LENGTH,
            required=1.0e-4,
            observed=5.0e-5,
            terminal_index=3,
            split_edge=(1, 2),
            candidate_point_index=0,
        ),
        CapsuleEscapeObstruction("edge", -0.1),
        VascularIntersectionObstruction("left", "right", -0.01),
        SymmetryMismatchObstruction(
            region_id="limb",
            predicate=VascularSymmetryPredicate.REFLECTION_DISTANCE,
            required=1.0e-12,
            observed=0.2,
            witness=VascularSymmetryWitness(
                "limb:positive:node", "limb:negative:node"
            ),
        ),
        VascularGluingObstruction("interface", "incompatible sections"),
        DisconnectedVascularGraphObstruction("node"),
        VascularSolverResidualObstruction(1.0e-4, 1.0e-10),
        DemandDeliveryMismatchObstruction("limb", 0.2, 1.0e-3),
        ReversedVascularFlowObstruction("exchange:edge", -0.1),
        InvalidStructuralGuidanceObstruction(
            "structural_guidance/cells/0", "nonfinite stress"
        ),
    )
    view = vasculature_to_dict(RejectedVasculature(obstructions))
    assert tuple(item["kind"] for item in view["obstructions"]) == (
        "MissingProvenance",
        "EmptyPerfusionTerritory",
        "InsufficientTerminalSites",
        "InfeasibleBifurcation",
        "CapsuleEscape",
        "Intersection",
        "SymmetryMismatch",
        "FailedGluing",
        "Disconnection",
        "SolverResidual",
        "DemandDeliveryMismatch",
        "ReversedFlow",
        "InvalidStructuralGuidance",
    )
    bifurcation = view["obstructions"][3]
    assert bifurcation == {
        "kind": "InfeasibleBifurcation",
        "address": "anatomy/regions/limb/terminals/3",
        "region_id": "limb",
        "phase": "insertion",
        "lane": "supply",
        "predicate": "SegmentLength",
        "required": 1.0e-4,
        "observed": 5.0e-5,
        "terminal_index": 3,
        "supply_capsule": None,
        "return_capsule": None,
        "split_edge": (1, 2),
        "candidate_point_index": 0,
        "attempted_candidate_count": None,
        "failing_segments": (),
    }


def test_deleted_raw_vascular_conduit_surface_is_rejected() -> None:
    graph = {
        "name": "surface-only",
        "parts": [
            {
                "id": "host",
                "type": "gencyl",
                "spine": [[0, 0, 0], [0, 1, 0]],
                "radii": [0.2, 0.2],
            }
        ],
    }
    evaluated = engine.evaluate(graph, res=60)
    with pytest.raises(ValueError, match="unknown kind 'vascular'"):
        apply_conduits(
            graph,
            evaluated.vertices,
            evaluated.faces,
            [
                {
                    "id": "forbidden",
                    "kind": "vascular",
                    "part": "host",
                    "width": 0.1,
                    "emit": ["band"],
                }
            ],
        )


def test_harness_exposes_solve_receipt_render_contract_not_raw_add() -> None:
    state = HarnessState(
        spec=json.loads(_BODY_PATH.read_text()),
        spec_dir=paths.SPECS,
        resolution=60,
    )
    raw_add = run_line(
        state, "conduit add vascular old root torso:0.5 count 260 seed 7"
    )
    assert isinstance(raw_add, RejectedCommand)
    assert raw_add.obstruction.kind == "UnknownConduitKind"
    solved = run_line(state, "vascular solve")
    assert isinstance(solved, AcceptedCommand)
    assert "VASCULAR ACCEPTED" in (solved.output or "")
    receipt = run_line(solved.state, "vascular receipt")
    assert isinstance(receipt, AcceptedCommand)
    view = json.loads(receipt.output or "{}")
    assert view["vasculature"]["status"] == "accepted"
    assert "reinforcement" not in view
