"""Tests for typed whole-body anatomy and closed vascular descent."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from golem import paths as _paths
from golem.evals import eval_anatomy as anatomy_evaluation
from golem.kernel import anatomy
from golem.kernel import body
from golem.kernel import engine
from golem.kernel import sheaf
from golem.senses import proprio
from golem.senses.proprio.render import _anatomy_receipt_lines


_BODY_PATH = _paths.SPECS / "knight_body.json"


def _body_spec() -> dict:
    return json.loads(_BODY_PATH.read_text(encoding="utf-8"))


def _anatomy_payload() -> dict:
    return _body_spec()["anatomy"]


def _compile_with_anatomy(anatomy_payload: object) -> body.CompiledBody:
    result = body.Compiler(
        {**_body_spec(), "anatomy": anatomy_payload}, spec_dir=_paths.SPECS
    ).compile()
    assert isinstance(result, body.CompiledBody)
    return result


def test_overall_derives_real_closed_vascular_circuits_and_clearances() -> None:
    compiled = _compile_with_anatomy(_anatomy_payload())
    anatomy_view = compiled.receipt["anatomy"]
    assert anatomy_view["status"] == "accepted"
    assert anatomy_view["circulation_kind"] == "closed_vascular"
    assert anatomy_view["overall"]["pump"] == {
        "kind": "pump_organ",
        "region_id": "axial_core",
        "host_bone_id": "spine_upper",
    }
    assert tuple(
        circuit["exchange_region_id"] for circuit in anatomy_view["circuits"]
    ) == ("cranial", "forelimb", "hindlimb")
    assert all(_is_closed_circuit(circuit) for circuit in anatomy_view["circuits"])
    assert anatomy_view["terminal_coverage"] == 1.0
    assert anatomy_view["maximum_normalized_free_cell_imbalance"] <= 1.0e-10
    assert anatomy_view["boundary_relative_error"] <= 1.0e-10
    assert math.fsum(
        circuit["flow_fraction"] for circuit in anatomy_view["circuits"]
    ) == 1.0
    assert {
        circuit["exchange_region_id"]: circuit["effective_demand"]
        for circuit in anatomy_view["circuits"]
    } == {"cranial": 0.1, "forelimb": 0.04, "hindlimb": 0.88}
    assert anatomy_view["authored_numeric_values"] == 48
    assert anatomy_view["derived_radius_values"] == 141
    assert anatomy_view["authoring_leverage"] == 141 / 48


def test_vascular_stage_position_is_enforced_by_distinct_types() -> None:
    compiler = body.Compiler(_body_spec(), spec_dir=_paths.SPECS)
    compiler.compile()
    result = anatomy.derive_anatomy(
        compiler.bones, compiler.order, _anatomy_payload()
    )
    assert isinstance(result, anatomy.AcceptedAnatomy)
    assert all(
        isinstance(circuit.capillary_bed, anatomy.CapillaryBed)
        and all(
            isinstance(segment, anatomy.DistributingArtery)
            for segment in circuit.distributing_arteries
        )
        and isinstance(circuit.resistance_arteriole, anatomy.ResistanceArteriole)
        and isinstance(circuit.collecting_venule, anatomy.CollectingVenule)
        and all(
            isinstance(segment, anatomy.ReturningVein)
            for segment in circuit.returning_veins
        )
        for circuit in result.circuits
    )


def test_bone_muscle_skin_descent_owns_the_emitted_limb_profiles() -> None:
    compiled = _compile_with_anatomy(_anatomy_payload())
    envelopes = compiled.receipt["anatomy"]["tissue_envelopes"]
    assert tuple(envelope["bone_id"] for envelope in envelopes) == (
        "upper_arm",
        "forearm",
        "thigh",
        "shin",
    )
    profiled_parts = {
        compiled.receipt["provenance"][part["id"]]: part
        for part in compiled.graph["parts"]
        if "profile" in part
    }
    assert all(
        envelope["shape_address"] in profiled_parts
        and len(profiled_parts[envelope["shape_address"]]["spine"])
        == len(envelope["stations"])
        and set(profiled_parts[envelope["shape_address"]]["profile"]["n"])
        == {2.0}
        and "offset"
        in profiled_parts[envelope["shape_address"]]["profile"]
        for envelope in envelopes
    )
    assert all(
        envelope["stations"][0]["parameter"] == 0.0
        and envelope["stations"][-1]["parameter"] == 1.0
        for envelope in envelopes
    )


def test_anatomy_evaluation_is_a_controlled_tissue_ablation() -> None:
    result = anatomy_evaluation.evaluate_anatomy()
    assert isinstance(result, anatomy_evaluation.AnatomyEvaluation), result
    assert result.anatomy_iou > result.control_iou
    assert result.reference_iou_gain == result.anatomy_iou - result.control_iou
    assert result.control_mesh_integrity == 1.0
    assert result.anatomy_mesh_integrity == 1.0
    assert result.rootedness == 1.0
    assert result.interface_addressability == 1.0
    assert result.authoring_leverage == 1.0
    assert result.tissue_coverage == 1.0
    assert result.anatomy_closure == 1.0
    assert result.anatomy_guided_construction_score == result.anatomy_iou


def test_invalid_myotendinous_path_is_a_typed_obstruction() -> None:
    payload = _anatomy_payload()
    overall = payload["overall"]
    first, *remaining = overall["myotendinous_units"]
    candidate = {
        **payload,
        "overall": {
            **overall,
            "myotendinous_units": [
                {
                    **first,
                    "insertion": {
                        **first["insertion"],
                        "bone_id": "shin",
                    },
                },
                *remaining,
            ],
        },
    }
    compiled = _compile_with_anatomy(candidate)
    assert (
        "InvalidMyotendinousPath",
        "anatomy/myotendinous_units/forelimb_flexor",
    ) in {
        (obstruction["kind"], obstruction["address"])
        for obstruction in compiled.receipt["anatomy"]["obstructions"]
    }


def test_missing_bone_section_and_skin_are_distinct_typed_obstructions() -> None:
    spec = _body_spec()
    overall = spec["anatomy"]["overall"]
    candidate = {
        **spec,
        "anatomy": {
            **spec["anatomy"],
            "overall": {
                **overall,
                "integument_layers": [
                    layer
                    for layer in overall["integument_layers"]
                    if layer["region_id"] != "forelimb"
                ],
            },
        },
        "skeleton": {
            **spec["skeleton"],
            "bones": tuple(
                {
                    key: value
                    for key, value in bone.items()
                    if key != "bone_radii"
                }
                if bone["id"] == "upper_arm"
                else bone
                for bone in spec["skeleton"]["bones"]
            ),
        },
    }
    compiled = body.Compiler(candidate, spec_dir=_paths.SPECS).compile()
    assert isinstance(compiled, body.CompiledBody)
    kinds = tuple(
        obstruction["kind"]
        for obstruction in compiled.receipt["anatomy"]["obstructions"]
    )
    assert kinds.count("MissingSkeletalCrossSection") == 2
    assert kinds.count("MissingIntegumentLayer") == 2


def test_tissue_and_circulation_constraints_glue_at_the_host() -> None:
    compiled = _compile_with_anatomy(_anatomy_payload())
    rows = compiled.receipt["anatomy"]["carrier_rows"]
    helm_row = next(row for row in rows if row["bone_id"] == "helm")
    upper_arm_row = next(row for row in rows if row["bone_id"] == "upper_arm")
    assert helm_row["circulation_minimum_radius"] < 0.24
    assert helm_row["tissue_minimum_radius"] == 0.07
    assert helm_row["controlling_constraint"] == "tissue_envelope"
    assert helm_row["target_minimum_radius"] == 0.07
    assert upper_arm_row["controlling_constraint"] == "circulation_clearance"
    assert all(
        row["interface_address"].startswith("skeleton/")
        and (
            row["shape_address"] is None
            or row["shape_address"].startswith("skeleton/")
        )
        for row in rows
    )


def test_missing_terminal_exchange_is_a_typed_whole_body_obstruction() -> None:
    payload = _anatomy_payload()
    overall = payload["overall"]
    circulation = overall["circulation"]
    candidate = {
        **payload,
        "overall": {
            **overall,
            "circulation": {
                **circulation,
                "exchange_beds": circulation["exchange_beds"][:-1],
            },
        },
    }
    compiled = _compile_with_anatomy(candidate)
    obstructions = compiled.receipt["anatomy"]["obstructions"]
    assert {
        (obstruction["kind"], obstruction["address"])
        for obstruction in obstructions
    } == {
        ("MissingTerminalRegion", "skeleton/foot"),
        ("UnreferencedAnatomyRegion", "anatomy/regions/hindlimb"),
    }


def test_malformed_exchange_fields_accumulate_typed_obstructions() -> None:
    payload = _anatomy_payload()
    overall = payload["overall"]
    circulation = overall["circulation"]
    first_exchange = circulation["exchange_beds"][0]
    candidate = {
        **payload,
        "overall": {
            **overall,
            "circulation": {
                **circulation,
                "exchange_beds": [
                    {
                        **first_exchange,
                        "tissue": "decorative-string-soup",
                        "demand": 0.0,
                        "tissue_envelope": {"minimum_radius": 0.0},
                    },
                    *circulation["exchange_beds"][1:],
                ],
            },
        },
    }
    compiled = _compile_with_anatomy(candidate)
    assert {
        (obstruction["kind"], obstruction["address"])
        for obstruction in compiled.receipt["anatomy"]["obstructions"]
    } == {
        ("MalformedAnatomy", "/anatomy/overall/circulation/exchange_beds/0/tissue"),
        ("MalformedAnatomy", "/anatomy/overall/circulation/exchange_beds/0/demand"),
        (
            "MalformedAnatomy",
            "/anatomy/overall/circulation/exchange_beds/0/tissue_envelope/minimum_radius",
        ),
    }


def test_clearance_suggestion_names_the_measured_gencyl_record() -> None:
    anatomy_payload = {
        "dialect": "anatomy/0.1",
        "overall": {
            "regions": [
                {"region_id": "core", "kind": "torso", "host_bone_id": "root"},
                {"region_id": "distal", "kind": "limb", "host_bone_id": "tip"},
            ],
            "circulation": {
                "kind": "closed_vascular",
                "pump_region_id": "core",
                "exchange_beds": [
                    {
                        "region_id": "distal",
                        "tissue": "skeletal_muscle",
                        "demand": 1.0,
                    }
                ],
                "carrier_radius_scale": 0.2,
                "distance_decay": 1.0,
            },
        },
    }
    spec = {
        "name": "address-probe",
        "dialect": "body/0.3",
        "emit_target": "v03",
        "anatomy": anatomy_payload,
        "skeleton": {
            "root": {"id": "root", "world": [0, 0, 0]},
            "bones": [
                {
                    "id": "trunk",
                    "parent": "root",
                    "attach": {"t": 0},
                    "length": 1.0,
                    "joint": {"dof": "fixed"},
                },
                {
                    "id": "tip",
                    "parent": "trunk",
                    "attach": {"t": 1},
                    "length": 1.0,
                    "joint": {"dof": "fixed"},
                    "flesh": [
                        {
                            "kind": "box",
                            "name": "shell",
                            "t": 0.5,
                            "size": [0.1, 0.1, 0.1],
                        },
                        {
                            "kind": "gencyl",
                            "name": "conduit",
                            "span": [0, 1],
                            "radii": [0.05, 0.05],
                        },
                    ],
                },
            ],
        },
    }
    compiled = body.Compiler(spec, spec_dir=_paths.SPECS).compile()
    assert isinstance(compiled, body.CompiledBody)
    tip_row = next(
        row
        for row in compiled.receipt["anatomy"]["carrier_rows"]
        if row["bone_id"] == "tip"
    )
    assert tip_row["shape_address"] == "skeleton/tip/flesh[1]"


def test_proprio_receipt_exposes_real_circuit_without_clearance_edits() -> None:
    graph = _compile_with_anatomy(_anatomy_payload()).graph
    senses, records = body.run_asserts(graph, graph["intent"])
    report = proprio.render_receipt(
        senses,
        records,
        proprio.detect_anomalies(senses),
        pack="knight.intent",
    )
    assert (
        "CIRCULATION pump spine_upper | 3/3 closed | "
        "0 suggested carrier scales"
        in report
    )
    assert (
        "NEURAL_TISSUE 0.098 | PUMP > ARTERY > ARTERIOLE > CAPILLARY > VENULE > VEIN"
        in report
    )
    assert (
        "SKELETAL_MUSCLE 0.863 | PUMP > ARTERY > ARTERIOLE > CAPILLARY > VENULE > VEIN"
        in report
    )
    assert "CIRCULATION_CLEARANCE @" not in report


def test_proprio_receipt_names_unapplied_carrier_scale_suggestions() -> None:
    report = "\n".join(
        _anatomy_receipt_lines(
            {
                "status": "accepted",
                "circuits": (),
                "carrier_rows": (
                    {
                        "suggested_scale": 2.0,
                        "controlling_constraint": "circulation_clearance",
                        "interface_address": "skeleton/wing/joint",
                        "measured_minimum_radius": 0.02,
                        "target_minimum_radius": 0.04,
                        "shape_address": "skeleton/wing/flesh[0]",
                    },
                ),
                "overall": {"pump": {"host_bone_id": "root"}},
                "terminal_coverage": 1.0,
                "authoring_leverage": 1.0,
            }
        )
    )
    assert "0/0 closed | 1 suggested carrier scales" in report
    assert "suggest scale skeleton/wing/flesh[0] x2.000" in report
    assert " edits" not in report


def test_absent_anatomy_is_a_strict_compiler_no_op() -> None:
    spec = {key: value for key, value in _body_spec().items() if key != "anatomy"}
    compiled = body.Compiler(spec, spec_dir=_paths.SPECS).compile()
    assert isinstance(compiled, body.CompiledBody)
    assert "anatomy" not in compiled.graph["intent"]
    assert "anatomy" not in compiled.receipt
    assert "body_guide" not in compiled.graph["intent"]
    assert "body_guide" not in compiled.receipt


def test_body_import_does_not_eagerly_load_the_anatomy_solver() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from golem.kernel import body; "
                "raise SystemExit(bool({'golem.kernel.anatomy', 'golem.kernel.sheaf'} & sys.modules.keys()))"
            ),
        ],
        cwd=_paths.KERNEL_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_disconnected_overall_is_a_typed_anatomy_obstruction() -> None:
    bones = {
        "root_a": {"parent": None, "mirrored": False, "flesh": ()},
        "pump": {"parent": "root_a", "mirrored": False, "flesh": ()},
        "leaf_a": {"parent": "pump", "mirrored": False, "flesh": ()},
        "root_b": {"parent": None, "mirrored": False, "flesh": ()},
        "leaf_b": {"parent": "root_b", "mirrored": False, "flesh": ()},
    }
    payload = {
        "dialect": "anatomy/0.1",
        "overall": {
            "regions": [
                {"region_id": "core", "kind": "torso", "host_bone_id": "pump"},
                {"region_id": "left", "kind": "limb", "host_bone_id": "leaf_a"},
                {"region_id": "right", "kind": "limb", "host_bone_id": "leaf_b"},
            ],
            "circulation": {
                "kind": "closed_vascular",
                "pump_region_id": "core",
                "carrier_radius_scale": 0.3,
                "distance_decay": 0.9,
                "exchange_beds": [
                    {"region_id": "left", "tissue": "skeletal_muscle", "demand": 1.0},
                    {"region_id": "right", "tissue": "skeletal_muscle", "demand": 1.0},
                ],
            },
        },
    }
    receipt = anatomy.anatomy_to_dict(
        anatomy.derive_anatomy(bones, tuple(bones), payload)
    )
    assert receipt == {
        "status": "rejected",
        "obstructions": (
            {
                "kind": "DisconnectedAnatomyRegion",
                "address": "anatomy/regions/right",
                "reason": (
                    "pump host 'pump' and exchange host 'leaf_b' "
                    "do not share a skeleton root"
                ),
            },
        ),
    }


def _is_closed_circuit(circuit: dict) -> bool:
    supply = tuple(circuit["supply"])
    returning = tuple(circuit["return"])
    return (
        len(supply) >= 2
        and all(
            segment["kind"] == "distributing_artery"
            for segment in supply[:-1]
        )
        and supply[-1]["kind"] == "resistance_arteriole"
        and circuit["exchange"]["kind"] == "capillary_bed"
        and returning[0]["kind"] == "collecting_venule"
        and all(segment["kind"] == "returning_vein" for segment in returning[1:])
        and tuple(segment["interface_address"] for segment in returning)
        == tuple(reversed(tuple(segment["interface_address"] for segment in supply)))
    )


def _resolved_channel_graph(radius: float) -> anatomy.ClosedVascularGraph:
    return anatomy.ClosedVascularGraph(
        nodes=(
            anatomy.VascularNode(
                "outlet",
                (0.0, -0.2, 0.0),
                anatomy.VascularNodeKind.PUMP_OUTLET,
                "core",
            ),
            anatomy.VascularNode(
                "inlet",
                (0.0, 0.2, 0.0),
                anatomy.VascularNodeKind.PUMP_INLET,
                "core",
            ),
        ),
        edges=(
            anatomy.VascularEdge(
                "channel",
                "outlet",
                "inlet",
                anatomy.DistributingArtery("skeleton/core/joint"),
                anatomy.VascularStratum.SUPPLY,
                radius,
                1.0,
                1.0,
            ),
        ),
        pump_outlet_node_id="outlet",
        pump_inlet_node_id="inlet",
    )


def test_resolved_vascular_material_is_carved_from_authoritative_field() -> None:
    evaluated = engine.evaluate(
        {
            "name": "resolved-channel-host",
            "blend": 0.0,
            "parts": [
                {
                    "id": "host",
                    "type": "box",
                    "center": [0.0, 0.0, 0.0],
                    "size": [0.4, 0.4, 0.4],
                    "round": 0.02,
                }
            ],
        },
        res=48,
    )
    result = anatomy.derive_vascular_material_masks(
        evaluated,
        _resolved_channel_graph(0.08),
        wall_thickness=0.06,
    )
    assert isinstance(result, anatomy.AcceptedVascularMaterial)
    masks = result.masks
    assert np.any(masks.lumen)
    assert np.any(masks.wall)
    assert not np.any(masks.lumen & masks.load_bearing_solid)
    assert not masks.lumen.flags.writeable
    assert masks.minimum_radius_to_pitch >= 2.0

def test_authored_cavity_gutting_carrier_is_a_typed_rejection() -> None:
    evaluated = engine.evaluate(
        {
            "name": "carved-channel-host",
            "blend": 0.0,
            "parts": [
                {
                    "id": "host",
                    "type": "box",
                    "center": [0.0, 0.0, 0.0],
                    "size": [0.4, 0.4, 0.4],
                    "round": 0.02,
                }
            ],
            "carves": [
                {
                    "id": "carrier-gut",
                    "type": "box",
                    "center": [0.1, 0.0, 0.0],
                    "size": [0.02, 0.5, 0.5],
                    "round": 0.0,
                }
            ],
        },
        res=48,
    )
    result = anatomy.derive_vascular_material_masks(
        evaluated,
        _resolved_channel_graph(0.08),
        wall_thickness=0.06,
    )
    assert isinstance(result, anatomy.RejectedVascularMaterial)
    assert len(result.obstructions) == 1
    obstruction = result.obstructions[0]
    assert isinstance(
        obstruction, anatomy.ChannelInducedDisconnectionObstruction
    )
    assert obstruction.component_count > 1


def test_subgrid_lumen_is_an_addressed_obstruction_not_an_inflated_mask() -> None:
    evaluated = engine.evaluate(
        {
            "name": "unresolved-channel-host",
            "blend": 0.0,
            "parts": [
                {
                    "id": "host",
                    "type": "box",
                    "center": [0.0, 0.0, 0.0],
                    "size": [0.4, 0.4, 0.4],
                    "round": 0.02,
                }
            ],
        },
        res=32,
    )
    graph = _resolved_channel_graph(1.0e-4)
    result = anatomy.derive_vascular_material_masks(
        evaluated, graph, wall_thickness=0.08
    )
    assert result == anatomy.RejectedVascularMaterial(
        (
            anatomy.UnresolvedLumenObstruction(
                edge_id="channel",
                radius=1.0e-4,
                pitch=evaluated.maximum_pitch,
                required_cells=2.0,
                unresolved_edge_count=1,
            ),
        )
    )


def _physical_hydraulic_node(
    node_id: str,
    position: tuple[float, float, float],
    kind: anatomy.VascularNodeKind = anatomy.VascularNodeKind.CORRIDOR,
) -> anatomy.VascularNode:
    return anatomy.VascularNode(node_id, position, kind, "analytic")


def _physical_hydraulic_edge(
    edge_id: str,
    source_node_id: str,
    target_node_id: str,
    radius_world_units: float,
    *,
    normalized_solved_flow: float = 0.0,
) -> anatomy.VascularEdge:
    return anatomy.VascularEdge(
        edge_id,
        source_node_id,
        target_node_id,
        anatomy.DistributingArtery(f"analytic/{edge_id}"),
        anatomy.VascularStratum.SUPPLY,
        radius_world_units,
        1.0,
        normalized_solved_flow,
    )


def test_physical_hydraulics_matches_single_circular_tube_solution() -> None:
    graph = anatomy.ClosedVascularGraph(
        nodes=(
            _physical_hydraulic_node(
                "outlet",
                (0.0, 0.0, 0.0),
                anatomy.VascularNodeKind.PUMP_OUTLET,
            ),
            _physical_hydraulic_node(
                "inlet",
                (2.0, 0.0, 0.0),
                anatomy.VascularNodeKind.PUMP_INLET,
            ),
        ),
        edges=(
            _physical_hydraulic_edge(
                "tube",
                "outlet",
                "inlet",
                0.02,
                normalized_solved_flow=123.0,
            ),
        ),
        pump_outlet_node_id="outlet",
        pump_inlet_node_id="inlet",
    )
    result = anatomy.solve_physical_hydraulics(
        graph,
        metres_per_world_unit=0.5,
        fluid_density_kilograms_per_cubic_metre=1000.0,
        dynamic_viscosity_pascal_second=1.0e-3,
        pump_pressure_boundary=anatomy.PumpPressureBoundary(101326.0, 101325.0),
    )
    assert isinstance(result, anatomy.AcceptedPhysicalHydraulics), result
    expected_conductance = math.pi * 0.01**4 / (8.0 * 1.0e-3 * 1.0)
    expected_reynolds = (
        2.0 * 1000.0 * expected_conductance / (math.pi * 0.01 * 1.0e-3)
    )
    assert math.isclose(
        result.edge_flows[0].volumetric_flow_cubic_metres_per_second,
        expected_conductance,
        rel_tol=1.0e-12,
    )
    assert math.isclose(
        result.edge_flows[0].reynolds_validity.reynolds_number,
        expected_reynolds,
        rel_tol=1.0e-12,
    )
    assert result.edge_flows[0].reynolds_validity.is_poiseuille_valid
    assert {row.node_id: row.pressure_pascal for row in result.node_pressures} == {
        "outlet": 101326.0,
        "inlet": 101325.0,
    }
    assert tuple(
        value.value for value in result.balance_solution.field.values
    ) == (101326.0, 101325.0)
    assert {
        str(boundary.boundary_id): boundary.value
        for boundary in result.balance_solution.problem.fixed_boundaries
    } == {
        "physical-pump-outlet": 101326.0,
        "physical-pump-inlet": 101325.0,
    }
    assert math.isclose(
        result.receipt.hydraulic_pump_power_watts,
        expected_conductance,
        rel_tol=1.0e-12,
    )
    assert result.receipt.boundary_flow_imbalance_cubic_metres_per_second == 0.0
    assert graph.edges[0].solved_flow == 123.0


def test_physical_hydraulic_network_glues_branch_flow_and_conservation() -> None:
    graph = anatomy.ClosedVascularGraph(
        nodes=(
            _physical_hydraulic_node(
                "outlet",
                (0.0, 0.0, 0.0),
                anatomy.VascularNodeKind.PUMP_OUTLET,
            ),
            _physical_hydraulic_node("split", (1.0, 0.0, 0.0)),
            _physical_hydraulic_node("upper", (2.0, 1.0, 0.0)),
            _physical_hydraulic_node("lower", (2.0, -1.0, 0.0)),
            _physical_hydraulic_node("merge", (3.0, 0.0, 0.0)),
            _physical_hydraulic_node(
                "inlet",
                (4.0, 0.0, 0.0),
                anatomy.VascularNodeKind.PUMP_INLET,
            ),
        ),
        edges=tuple(
            _physical_hydraulic_edge(edge_id, source, target, 1.0e-3)
            for edge_id, source, target in (
                ("outlet-split", "outlet", "split"),
                ("split-upper", "split", "upper"),
                ("split-lower", "split", "lower"),
                ("upper-merge", "upper", "merge"),
                ("lower-merge", "lower", "merge"),
                ("merge-inlet", "merge", "inlet"),
            )
        ),
        pump_outlet_node_id="outlet",
        pump_inlet_node_id="inlet",
    )
    result = anatomy.solve_physical_hydraulics(
        graph,
        metres_per_world_unit=1.0,
        fluid_density_kilograms_per_cubic_metre=1000.0,
        dynamic_viscosity_pascal_second=1.0e-3,
        pump_pressure_boundary=anatomy.PumpPressureBoundary(101000.0, 100000.0),
    )
    assert isinstance(result, anatomy.AcceptedPhysicalHydraulics), result
    unit_length_conductance = math.pi * (1.0e-3) ** 4 / (8.0 * 1.0e-3)
    expected_total_flow = (
        unit_length_conductance * 1000.0 / (2.0 + math.sqrt(2.0))
    )
    flow_by_edge = {
        row.edge_id: row.volumetric_flow_cubic_metres_per_second
        for row in result.edge_flows
    }
    assert math.isclose(
        flow_by_edge["outlet-split"], expected_total_flow, rel_tol=1.0e-10
    )
    assert math.isclose(
        flow_by_edge["merge-inlet"], expected_total_flow, rel_tol=1.0e-10
    )
    assert all(
        math.isclose(flow_by_edge[edge_id], expected_total_flow / 2.0, rel_tol=1.0e-10)
        for edge_id in (
            "split-upper",
            "split-lower",
            "upper-merge",
            "lower-merge",
        )
    )
    pressure_by_node = {
        row.node_id: row.pressure_pascal for row in result.node_pressures
    }
    assert math.isclose(pressure_by_node["upper"], 100500.0, rel_tol=1.0e-12)
    assert math.isclose(pressure_by_node["lower"], 100500.0, rel_tol=1.0e-12)
    assert result.receipt.maximum_free_node_normalized_residual <= 1.0e-10
    assert result.receipt.relative_boundary_flow_imbalance <= 1.0e-10
    assert math.isclose(
        result.receipt.pump_volumetric_flow_cubic_metres_per_second,
        expected_total_flow,
        rel_tol=1.0e-10,
    )
    assert all(
        row.reynolds_validity.is_poiseuille_valid for row in result.edge_flows
    )


def test_physical_hydraulics_accumulates_invalid_si_input_obstructions() -> None:
    result = anatomy.solve_physical_hydraulics(
        _resolved_channel_graph(0.08),
        metres_per_world_unit=0.0,
        fluid_density_kilograms_per_cubic_metre=math.nan,
        dynamic_viscosity_pascal_second=-1.0,
        pump_pressure_boundary=anatomy.PumpPressureBoundary(5.0, 5.0),
    )
    assert result == anatomy.RejectedPhysicalHydraulics(
        (
            anatomy.InvalidMetresPerWorldUnitObstruction(0.0),
            anatomy.InvalidFluidDensityObstruction(math.nan),
            anatomy.InvalidDynamicViscosityObstruction(-1.0),
            anatomy.InvalidPumpPressureBoundaryObstruction(5.0, 5.0),
        )
    )


def test_physical_hydraulics_rejects_nonlaminar_poiseuille_extrapolation() -> None:
    result = anatomy.solve_physical_hydraulics(
        _resolved_channel_graph(0.01),
        metres_per_world_unit=1.0,
        fluid_density_kilograms_per_cubic_metre=1000.0,
        dynamic_viscosity_pascal_second=1.0e-3,
        pump_pressure_boundary=anatomy.PumpPressureBoundary(1000.0, 0.0),
    )
    assert isinstance(result, anatomy.RejectedPhysicalHydraulics)
    assert len(result.obstructions) == 1
    obstruction = result.obstructions[0]
    assert isinstance(
        obstruction, anatomy.NonLaminarPhysicalHydraulicFlowObstruction
    )
    assert obstruction.edge_id == "channel"
    assert obstruction.reynolds_number >= obstruction.maximum_laminar_reynolds


def test_physical_hydraulics_preserves_typed_balance_singularity() -> None:
    graph = anatomy.ClosedVascularGraph(
        nodes=(
            _physical_hydraulic_node(
                "outlet",
                (0.0, 0.0, 0.0),
                anatomy.VascularNodeKind.PUMP_OUTLET,
            ),
            _physical_hydraulic_node(
                "inlet",
                (1.0, 0.0, 0.0),
                anatomy.VascularNodeKind.PUMP_INLET,
            ),
            _physical_hydraulic_node("unglued-a", (0.0, 1.0, 0.0)),
            _physical_hydraulic_node("unglued-b", (1.0, 1.0, 0.0)),
        ),
        edges=(
            _physical_hydraulic_edge("pump-path", "outlet", "inlet", 1.0e-3),
            _physical_hydraulic_edge(
                "unglued-path", "unglued-a", "unglued-b", 1.0e-3
            ),
        ),
        pump_outlet_node_id="outlet",
        pump_inlet_node_id="inlet",
    )
    result = anatomy.solve_physical_hydraulics(
        graph,
        metres_per_world_unit=1.0,
        fluid_density_kilograms_per_cubic_metre=1000.0,
        dynamic_viscosity_pascal_second=1.0e-3,
        pump_pressure_boundary=anatomy.PumpPressureBoundary(1.0, 0.0),
    )
    assert isinstance(result, anatomy.RejectedPhysicalHydraulics)
    assert any(
        isinstance(obstruction, anatomy.PhysicalHydraulicBalanceObstruction)
        and isinstance(
            obstruction.obstruction,
            sheaf.UnanchoredBalanceComponentObstruction,
        )
        for obstruction in result.obstructions
    )


# --------------------------------------------------------------------------- #
# Dialect ontology roles (wave-3, R3): `handle` bones and `non_carrier`       #
# flesh are authorable, opt-in vocabulary. Absence of a role keeps the        #
# default perfused-organ/carrier law exactly.                                 #
# --------------------------------------------------------------------------- #
def _handle_probe_spec(*, roles: bool) -> dict:
    """Verdigris shape (FRICTION-006): a decorative horn child on the skull
    and a mid-chain routing waypoint -- both pure transform handles."""
    horn: dict = {
        "id": "horn",
        "parent": "skull",
        "attach": {"t": 1},
        "length": 0.3,
        "rest_dir": [0, 0.5, 1],
        "joint": {"dof": "fixed"},
        "flesh": [
            {
                "kind": "gencyl",
                "name": "horn_blade",
                "span": [0, 1],
                "radii": [0.03, 0.01],
            }
        ],
    }
    waypoint: dict = {
        "id": "waypoint",
        "parent": "root",
        "attach": {"t": 1},
        "length": 0.4,
        "rest_dir": [1, 0, 0],
        "joint": {"dof": "fixed"},
        "flesh": [
            {
                "kind": "gencyl",
                "name": "wp_carrier",
                "span": [0, 1],
                "radii": [0.06, 0.06],
            }
        ],
    }
    if roles:
        horn["role"] = "handle"
        waypoint["role"] = "handle"
    return {
        "name": "handle-probe",
        "dialect": "body/0.3",
        "emit_target": "v03",
        "anatomy": {
            "dialect": "anatomy/0.1",
            "overall": {
                "regions": [
                    {"region_id": "core", "kind": "torso", "host_bone_id": "root"},
                    {
                        "region_id": "cranial",
                        "kind": "head",
                        "host_bone_id": "skull",
                    },
                    {"region_id": "distal", "kind": "limb", "host_bone_id": "tip"},
                ],
                "circulation": {
                    "kind": "closed_vascular",
                    "pump_region_id": "core",
                    "exchange_beds": [
                        {
                            "region_id": "cranial",
                            "tissue": "neural_tissue",
                            "demand": 0.4,
                        },
                        {
                            "region_id": "distal",
                            "tissue": "skeletal_muscle",
                            "demand": 0.6,
                        },
                    ],
                    "carrier_radius_scale": 0.2,
                    "distance_decay": 1.0,
                },
            },
        },
        "skeleton": {
            "root": {
                "id": "root",
                "world": [0, 0, 0],
                "flesh": [
                    {
                        "kind": "gencyl",
                        "name": "core",
                        "span": [0, 1],
                        "radii": [0.2, 0.2],
                    }
                ],
            },
            "bones": [
                {
                    "id": "neck",
                    "parent": "root",
                    "attach": {"t": 1},
                    "length": 0.4,
                    "rest_dir": [0, 0, 1],
                    "joint": {"dof": "fixed"},
                    "flesh": [
                        {
                            "kind": "gencyl",
                            "name": "neck_carrier",
                            "span": [0, 1],
                            "radii": [0.08, 0.08],
                        }
                    ],
                },
                {
                    "id": "skull",
                    "parent": "neck",
                    "attach": {"t": 1},
                    "length": 0.5,
                    "rest_dir": [0, 0, 1],
                    "joint": {"dof": "fixed"},
                    "flesh": [
                        {
                            "kind": "gencyl",
                            "name": "braincase",
                            "span": [0, 1],
                            "radii": [0.15, 0.15],
                        }
                    ],
                },
                horn,
                waypoint,
                {
                    "id": "tip",
                    "parent": "waypoint",
                    "attach": {"t": 1},
                    "length": 0.5,
                    "rest_dir": [0, -1, 0],
                    "joint": {"dof": "fixed"},
                    "flesh": [
                        {
                            "kind": "gencyl",
                            "name": "carrier",
                            "span": [0, 1],
                            "radii": [0.05, 0.05],
                        }
                    ],
                },
            ],
        },
    }


def test_handle_bones_are_exempt_from_terminal_region_law() -> None:
    compiled = body.Compiler(
        _handle_probe_spec(roles=True), spec_dir=_paths.SPECS
    ).compile()
    assert isinstance(compiled, body.CompiledBody)
    anatomy_view = compiled.receipt["anatomy"]
    assert anatomy_view["status"] == "accepted"
    # The declaration is visible contract vocabulary (R3), threaded through
    # intent and receipt, and reaches the senses expectations channel.
    expected_roles = {"handle": ("horn", "waypoint")}
    assert compiled.graph["intent"]["roles"] == {
        role: list(declared) for role, declared in expected_roles.items()
    }
    assert compiled.receipt["roles"] == {
        role: list(declared) for role, declared in expected_roles.items()
    }
    senses = proprio.build_senses(compiled.graph, compiled.graph["intent"])
    assert senses.roles == expected_roles
    # A handle is not perfused: no carrier row, no exchange-bed demand; the
    # mid-chain handle stays on its circuit as an ordinary routed bone.
    carrier_bones = {row["bone_id"] for row in anatomy_view["carrier_rows"]}
    assert "horn" not in carrier_bones
    distal = next(
        circuit
        for circuit in anatomy_view["circuits"]
        if circuit["exchange_region_id"] == "distal"
    )
    assert "waypoint" in distal["bones"]


def test_handle_free_bones_obey_the_default_terminal_law() -> None:
    # Honest negative: the same frame composition without the role is the
    # Verdigris-006 wall -- the leaf demands a bed and the host flips
    # nonterminal. The role is the only difference.
    compiled = body.Compiler(
        _handle_probe_spec(roles=False), spec_dir=_paths.SPECS
    ).compile()
    assert isinstance(compiled, body.CompiledBody)
    anatomy_view = compiled.receipt["anatomy"]
    assert {
        (obstruction["kind"], obstruction["address"])
        for obstruction in anatomy_view["obstructions"]
    } == {
        ("MissingTerminalRegion", "skeleton/horn"),
        ("NonterminalExchangeRegion", "anatomy/regions/cranial"),
    }
    assert "roles" not in compiled.graph["intent"]
    assert "roles" not in compiled.receipt


def _membrane_probe_spec(*, role: bool, carrier: bool = True) -> dict:
    flesh: list[dict] = []
    if carrier:
        flesh.append(
            {
                "kind": "gencyl",
                "name": "carrier",
                "span": [0, 1],
                "radii": [0.05, 0.05],
            }
        )
    membrane: dict = {
        "kind": "gencyl",
        "name": "membrane",
        "span": [0, 1],
        "radii": [0.018, 0.012],
    }
    if role:
        membrane["role"] = "non_carrier"
    flesh.append(membrane)
    return {
        "name": "membrane-probe",
        "dialect": "body/0.3",
        "emit_target": "v03",
        "anatomy": {
            "dialect": "anatomy/0.1",
            "overall": {
                "regions": [
                    {"region_id": "core", "kind": "torso", "host_bone_id": "root"},
                    {"region_id": "distal", "kind": "limb", "host_bone_id": "tip"},
                ],
                "circulation": {
                    "kind": "closed_vascular",
                    "pump_region_id": "core",
                    "exchange_beds": [
                        {
                            "region_id": "distal",
                            "tissue": "skeletal_muscle",
                            "demand": 1.0,
                        }
                    ],
                    "carrier_radius_scale": 0.2,
                    "distance_decay": 1.0,
                },
            },
        },
        "skeleton": {
            "root": {
                "id": "root",
                "world": [0, 0, 0],
                "flesh": [
                    {
                        "kind": "gencyl",
                        "name": "core",
                        "span": [0, 1],
                        "radii": [0.2, 0.2],
                    }
                ],
            },
            "bones": [
                {
                    "id": "mid",
                    "parent": "root",
                    "attach": {"t": 1},
                    "length": 0.6,
                    "rest_dir": [1, 0, 0],
                    "joint": {"dof": "fixed"},
                    "flesh": [
                        {
                            "kind": "gencyl",
                            "name": "mid_carrier",
                            "span": [0, 1],
                            "radii": [0.08, 0.08],
                        }
                    ],
                },
                {
                    "id": "tip",
                    "parent": "mid",
                    "attach": {"t": 1},
                    "length": 0.5,
                    "rest_dir": [0, -1, 0],
                    "joint": {"dof": "fixed"},
                    "flesh": flesh,
                },
            ],
        },
    }


def test_non_carrier_flesh_keeps_authored_dimensions_out_of_carrier_law() -> None:
    compiled = body.Compiler(
        _membrane_probe_spec(role=True), spec_dir=_paths.SPECS
    ).compile()
    assert isinstance(compiled, body.CompiledBody)
    anatomy_view = compiled.receipt["anatomy"]
    assert anatomy_view["status"] == "accepted"
    tip_row = next(
        row for row in anatomy_view["carrier_rows"] if row["bone_id"] == "tip"
    )
    # The membrane neither deflates the measured carrier cross-section nor
    # attracts a floor-inflation suggestion: only the declared carrier is
    # measured (Verdigris-007).
    assert tip_row["measured_minimum_radius"] == 0.05
    assert tip_row["shape_address"] == "skeleton/tip/flesh[0]"
    # Emitted geometry keeps the authored thin dimensions exactly -- no
    # silent inflation to the 0.04 carrier floor.
    membrane = next(
        part for part in compiled.graph["parts"] if part["id"] == "membrane"
    )
    assert membrane["radii"] == [0.018, 0.012]
    assert membrane["role"] == "non_carrier"
    # The membrane carries no carrier provenance; the declaration is visible.
    assert "membrane" not in compiled.receipt["provenance"]
    assert compiled.graph["intent"]["roles"] == {"non_carrier": ["membrane"]}
    assert compiled.receipt["roles"] == {"non_carrier": ["membrane"]}


def test_carrier_sampling_still_measures_membrane_without_the_role() -> None:
    # Honest negative: undeclared thin flesh on a route bone is sampled like
    # any flesh -- the role is the only exemption (R3: never a silent
    # reclassification).
    compiled = body.Compiler(
        _membrane_probe_spec(role=False), spec_dir=_paths.SPECS
    ).compile()
    assert isinstance(compiled, body.CompiledBody)
    tip_row = next(
        row
        for row in compiled.receipt["anatomy"]["carrier_rows"]
        if row["bone_id"] == "tip"
    )
    assert tip_row["measured_minimum_radius"] == 0.012
    assert tip_row["shape_address"] == "skeleton/tip/flesh[1]"
    assert "membrane" in compiled.receipt["provenance"]
    assert "roles" not in compiled.graph["intent"]


def test_route_bone_with_only_non_carrier_flesh_is_honestly_rejected() -> None:
    # Maiden-011/Verdigris-007 amendment: a route left without a lawful
    # carrier is rejected by name -- never patched by inflating the membrane.
    compiled = body.Compiler(
        _membrane_probe_spec(role=True, carrier=False), spec_dir=_paths.SPECS
    ).compile()
    assert isinstance(compiled, body.CompiledBody)
    anatomy_view = compiled.receipt["anatomy"]
    assert anatomy_view["status"] == "accepted"
    tip_row = next(
        row for row in anatomy_view["carrier_rows"] if row["bone_id"] == "tip"
    )
    # No carrier is measured and none is manufactured: the floor does not
    # invent a carrier cross-section for the route.
    assert tip_row["measured_minimum_radius"] is None
    assert tip_row["shape_address"] is None
    assert tip_row["suggested_scale"] == 1.0
    vascular = anatomy.realize_vasculature(compiled.anatomy, compiled.graph)
    assert isinstance(vascular, anatomy.RejectedVasculature)
    assert any(
        isinstance(obstruction, anatomy.MissingProvenanceObstruction)
        and obstruction.region_id == "distal"
        and obstruction.bone_id == "tip"
        for obstruction in vascular.obstructions
    )
