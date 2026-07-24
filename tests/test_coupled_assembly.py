"""End-to-end proofs for M7 assembly descent and its honest scale gate."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from golem import paths as _paths
from golem.kernel import body as gbody
from golem.assembly.core import (
    AcceptedAssembly,
    AssemblyRecord,
    CoupledStages,
    RejectedAssembly,
    _CompiledElement,
    _attach_resolution_refinement_evidence,
    _compile_coupled_assembly,
    _domain_cell_indices_owned_by_provenance,
    _evaluate_compiled_for_physics,
    _thermally_size_vasculature,
    compile_assembly,
)
from golem.assembly.__main__ import (
    _coupled_receipt_lines,
    _solid_receipt_lines,
    main as assembly_main,
)
from golem.assembly.service import (
    AcceptedServiceIntent,
    ElementAddress,
    MountAddress,
    PortAddress,
    decode_service_intent,
)
from golem.assembly.obstructions import ElementSurfaceFormationObstruction
from golem.assembly.vascular import _replace_element_vasculature
from golem.kernel import engine
from golem.kernel.anatomy import (
    AcceptedAnatomy,
    AcceptedVasculature,
    ClosedVascularGraph,
    DistributingArtery,
    SealedVascularConfig,
    VascularEdge,
    VascularFlowReceipt,
    VascularNode,
    VascularNodeKind,
    VascularStratum,
    realize_vasculature,
)
from golem.kernel.mechanics import (
    CellIndex,
    CellMechanics,
    MaterialFractionResolution,
    SymmetricTensor3,
)
from golem.materials.surface import (
    AcceptedAppearancePalette,
    decode_appearance_palette,
    seed_appearance_palette,
)


def test_provenance_ownership_descends_over_mirrored_instances() -> None:
    parts = (
        {
            "id": "foot",
            "type": "blob",
            "center": [2.0, 0.0, 0.0],
            "size": [0.5, 0.5, 0.5],
            "mirror": True,
        },
        {
            "id": "torso",
            "type": "blob",
            "center": [0.0, 0.0, 0.0],
            "size": [0.5, 0.5, 0.5],
        },
    )
    element = SimpleNamespace(
        graph={
            "parts": parts,
            "intent": {
                "provenance": {
                    "foot": "skeleton/foot/flesh[0]",
                    "torso": "skeleton/torso/flesh[0]",
                }
            },
        }
    )
    domain = SimpleNamespace(cell_centers_world=((-2.0, 0.0, 0.0),))

    assert _domain_cell_indices_owned_by_provenance(
        element, domain, "skeleton/foot/"
    ) == (0,)


def _resolved_service():
    payload = {
        "metres_per_world_unit": 1.0,
        "constitutive_model": "isotropic_linear_thermoelastic",
        "solid_materials": [
            {
                "role": "structure",
                "domain": "element:body",
                "material": "stainless_steel_316l_room_temperature",
            },
            {
                "role": "lumen_wall",
                "domain": "element:body",
                "material": "stainless_steel_316l_room_temperature",
            },
        ],
        "cases": [
            {
                "id": "resolved_coupled_probe",
                "gravity_m_s2": [0.0, -0.1, 0.0],
                "inertial_acceleration_m_s2": [0.0, 0.0, 0.0],
                "loads": [
                    {
                        "kind": "force",
                        "address": "port:body/pump_inlet",
                        "vector_newtons": [0.0, -1.0, 0.0],
                    }
                ],
                "supports": [
                    {"address": "mount:body", "law": "fixed"}
                ],
                "joints": [],
                "heat_sources": [
                    {"address": "element:body", "watts": 0.1}
                ],
            }
        ],
        "ambient": None,
        "coolant": {
            "material": "water_0_1_mpa_293k",
            "inlet_temperature_k": 293.15,
            "pump": {
                "kind": "pressure_budget",
                    "maximum_pressure_pa": 5.0e-7,
            },
        },
        "manufacturing": {
            "minimum_lumen_diameter_m": 0.3,
            "minimum_wall_thickness_m": 0.2,
            "minimum_remaining_ligament_m": 0.04,
        },
        "limits": {
            "maximum_temperature_k": 350.0,
            "maximum_temperature_gradient_k_per_m": 1.0e5,
            "maximum_displacement_m": 0.05,
            "minimum_yield_safety_factor": 1.01,
            "minimum_buckling_safety_factor": 1.0e-8,
            "minimum_burst_safety_factor": 1.01,
        },
        "evidence": {
            "channel_resolution_policy": "resolved_only",
            "physics_resolution": 16,
            "minimum_cells_across_lumen": 4,
            "resolution_refinement_factor": 2,
            "maximum_refinement_change_fraction": 0.05,
        },
    }
    result = decode_service_intent(
        payload,
        (
            ElementAddress("body"),
            MountAddress("body"),
            PortAddress("body", "pump_outlet"),
            PortAddress("body", "pump_inlet"),
        ),
    )
    assert isinstance(result, AcceptedServiceIntent)
    return result.intent


def _single_vessel_graph() -> ClosedVascularGraph:
    return ClosedVascularGraph(
        nodes=(
            VascularNode(
                "outlet",
                (0.25, 0.5, 0.5),
                VascularNodeKind.PUMP_OUTLET,
                None,
            ),
            VascularNode(
                "inlet",
                (0.75, 0.5, 0.5),
                VascularNodeKind.PUMP_INLET,
                None,
            ),
        ),
        edges=(
            VascularEdge(
                "resolved-vessel",
                "outlet",
                "inlet",
                DistributingArtery("probe"),
                VascularStratum.SUPPLY,
                0.2,
                1.0,
                1.0,
            ),
        ),
        pump_outlet_node_id="outlet",
        pump_inlet_node_id="inlet",
    )


def _resolved_compiled_element() -> _CompiledElement:
    resolution = 9
    field = np.full((resolution,) * 3, -1.0, dtype=np.float64)
    evaluated = engine.EvaluatedMorphology(
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=np.float64,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        normals=np.asarray(((0.0, 0.0, 1.0),) * 3, dtype=np.float64),
        field=field,
        lower=(0.0, 0.0, 0.0),
        upper=(1.0, 1.0, 1.0),
        resolution=resolution,
        world_pitch=(0.125, 0.125, 0.125),
    )
    graph = {
        "name": "resolved-body",
        "blend": 0.0,
        "parts": [
            {
                "id": "body",
                "type": "box",
                "center": [0.5, 0.5, 0.5],
                "size": [0.5, 0.5, 0.5],
                "round": 0.0,
            }
        ],
    }
    record = AssemblyRecord.from_derived_geometry(
        record_id="body",
        element_id="body",
        role="creature",
        stratum="solid",
        appearance_material="neutral_gray",
        vertices=evaluated.vertices,
        faces=evaluated.faces,
        resolution=resolution,
        report={"components": 1, "watertight_main": True, "faces": 1},
    )
    vascular_graph = _single_vessel_graph()
    accepted_vasculature = AcceptedVasculature(
        vascular_graph,
        VascularFlowReceipt(
            1,
            (("probe", 1),),
            (("artery", 1),),
            0.1,
            0.0,
            0.0,
            0.0,
            (("probe", 1.0),),
            1.0,
            0.0,
            0.0,
            0,
        ),
    )
    return _CompiledElement(
        element_id="body",
        entry={"id": "body", "role": "creature", "graph": graph},
        source_graph=graph,
        graph=graph,
        records=(record,),
        evaluated=evaluated,
        anatomy=None,
        vasculature=accepted_vasculature,
        mounted_vasculature=vascular_graph,
    )


def _structural_guidance_cell() -> CellMechanics:
    zero_tensor = SymmetricTensor3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return CellMechanics(
        cell=CellIndex(0, 0, 0),
        center=(0.5, 0.5, 0.5),
        volume=1.0,
        solid_fraction=1.0,
        fraction_resolution=MaterialFractionResolution.FULL_SOLID,
        total_strain=zero_tensor,
        thermal_strain=zero_tensor,
        elastic_strain=zero_tensor,
        stress=zero_tensor,
        von_mises_stress=1.0,
        yield_margin=1.0,
        yield_safety_factor=1.0,
    )


def test_vascular_regeneration_preserves_the_authored_appearance_palette() -> None:
    decoded = decode_appearance_palette(
        {
            "vascular_supply_gold": {
                "tint_linear_rgb": [0.90, 0.25, 0.08]
            }
        }
    )
    assert isinstance(decoded, AcceptedAppearancePalette)
    element = replace(
        _resolved_compiled_element(),
        appearance_palette=seed_appearance_palette(
            decoded.palette,
            "00" * 32,
            "body",
        ),
    )
    regenerated = _replace_element_vasculature(element, element.vasculature)
    supply = next(
        record
        for record in regenerated.records
        if record.appearance_material == "vascular_supply_gold"
    )
    assert supply.surface_color is not None
    assert supply.surface_color.recipe.tint_linear_rgb == (0.90, 0.25, 0.08)


def test_structural_guidance_is_not_discarded_without_thermal_resize() -> None:
    element = _resolved_compiled_element()
    guided_element = replace(element, element_id="structurally-guided-body")
    service = _resolved_service()
    service_without_heat = replace(
        service,
        cases=tuple(
            replace(service_case, heat_sources=())
            for service_case in service.cases
        ),
    )
    structural_cell = _structural_guidance_cell()

    def return_structurally_guided_element(
        _element,
        _service,
        config,
        structural_cells,
    ):
        assert config.terminal_radius == SealedVascularConfig().terminal_radius
        assert structural_cells == (structural_cell,)
        return guided_element

    result = _thermally_size_vasculature(
        (element,),
        service_without_heat,
        (structural_cell,),
        stages=CoupledStages(realize=return_structurally_guided_element),
    )
    assert isinstance(result, tuple)
    assert result == ((guided_element,), ())


def test_resolved_local_sections_glue_hydraulics_heat_and_mechanics() -> None:
    compiled = (_resolved_compiled_element(),)
    service = _resolved_service()
    result = _compile_coupled_assembly(compiled, service)
    assert isinstance(result, AcceptedAssembly), getattr(result, "obstructions", ())
    assert len(result.hydraulics) == 1
    assert len(result.thermal_solutions) == 1
    assert len(result.mechanics_solutions) == 1
    assert result.receipt.physical_evidence.value == "requested"
    assert result.receipt.hydraulic_receipts[0].relative_boundary_flow_imbalance <= 1.0e-10
    assert result.receipt.thermal_receipts[0].energy.relative_energy_imbalance <= 1.0e-10
    mechanics_receipt = result.receipt.mechanics_receipts[0]
    thermal_receipt = result.receipt.thermal_receipts[0]
    assert mechanics_receipt.mechanics.relative_force_imbalance <= 1.0e-8
    assert mechanics_receipt.mechanics.lowest_positive_buckling_load_factor is not None
    assert mechanics_receipt.minimum_burst_safety_factor is not None
    assert mechanics_receipt.critical_semantic_address == "element:body"
    assert thermal_receipt.hottest_semantic_address == "element:body"
    assert thermal_receipt.heat_transfer_correlations
    receipt_lines = _coupled_receipt_lines(result)
    assert any(line.startswith("hydraulics ") for line in receipt_lines)
    assert any(line.startswith("thermal ") for line in receipt_lines)
    assert any(line.startswith("mechanics ") for line in receipt_lines)
    repeated = _compile_coupled_assembly(compiled, service)
    assert isinstance(repeated, AcceptedAssembly)
    assert repeated.receipt == result.receipt
    assert repeated.hydraulics == result.hydraulics
    assert repeated.thermal_solutions == result.thermal_solutions
    assert repeated.mechanics_solutions == result.mechanics_solutions
    def reuse_accepted_physics(
        _compiled,
        _analysis_compiled,
        _service,
        sizing_receipts,
        *,
        include_linearized_buckling_evidence,
    ):
        assert sizing_receipts == result.receipt.vascular_sizing_receipts
        assert include_linearized_buckling_evidence is False
        return result

    fixed_topology_refinement = _attach_resolution_refinement_evidence(
        compiled,
        service,
        result,
        stages=CoupledStages(
            evaluate=lambda local_compiled, _resolution: local_compiled,
            solve=reuse_accepted_physics,
        ),
    )
    assert isinstance(fixed_topology_refinement, AcceptedAssembly)
    assert (
        fixed_topology_refinement.receipt.refinement_receipts[
            0
        ].maximum_displacement_utilization_change
        == 0.0
    )


def test_physics_reevaluation_preserves_surface_formation_rejection() -> None:
    element = _resolved_compiled_element()
    graph = {
        "name": "contained_local_pair",
        "blend": 0.02,
        "parts": [
            {
                "id": "outer",
                "type": "blob",
                "center": [0.0, 0.0, 0.0],
                "size": [1.0, 1.0, 1.0],
            },
            {
                "id": "inner",
                "type": "blob",
                "center": [0.0, 0.0, 0.0],
                "size": [0.3, 0.3, 0.3],
                "operator": "local_blend",
            },
        ],
    }
    result = _evaluate_compiled_for_physics(
        (replace(element, graph=graph),),
        24,
    )

    assert isinstance(result, RejectedAssembly)
    assert len(result.obstructions) == 1
    assert isinstance(
        result.obstructions[0],
        ElementSurfaceFormationObstruction,
    )
    assert isinstance(
        result.obstructions[0].obstructions[0],
        engine.NoCompatibleOverlap,
    )


def test_canonical_knight_accepts_mixed_scale_coupled_evidence() -> None:
    assembly_path = _paths.REHEARSAL / "reforge" / "knight_separated.json"
    assembly_spec = json.loads(assembly_path.read_text())
    result = compile_assembly(assembly_spec, assembly_path.parent)
    assert isinstance(result, AcceptedAssembly), getattr(result, "obstructions", ())
    assert tuple(
        record.record_id
        for record in result.records
        if record.stratum == "solid"
    ) == ("body", "sword")
    assert all(
        assignment["role"] != "armor"
        for assignment in assembly_spec["service"]["solid_materials"]
    )
    assert result.receipt.vascular_sizing_receipts[0].terminal_radius_scale > 1.0
    assert result.vasculature[0].receipt.structural_guidance_sample_count > 0
    assert (
        result.vasculature[0].receipt.maximum_structural_cost_multiplier
        == pytest.approx(1.05)
    )
    assert result.receipt.thermal_receipts[0].maximum_temperature_kelvin <= 313.15
    assert max(
        flow.reynolds_validity.reynolds_number
        for hydraulics in result.hydraulics
        for flow in hydraulics.edge_flows
    ) < 2300.0
    assert len(result.receipt.interface_receipts) == 1
    assert result.receipt.not_evaluated
    refinement = result.receipt.refinement_receipts[0]
    assert refinement.base_resolution == 32
    assert refinement.refined_resolution == 64
    assert refinement.maximum_temperature_utilization_change <= 0.05
    assert refinement.maximum_displacement_utilization_change <= 0.05
    assert refinement.minimum_yield_utilization_change is not None
    assert refinement.minimum_yield_utilization_change <= 0.05
    embedded = result.receipt.mechanics_receipts[0].embedded_channel_mechanics
    assert embedded is not None
    assert embedded.receipt.maximum_outer_diameter_to_grid_pitch < 0.05
    assert embedded.receipt.lumen_volume_fraction_upper_bound < 1.0e-3
    assert embedded.receipt.not_evaluated
    receipt_lines = _coupled_receipt_lines(result)
    assert any(line.startswith("vascular sizing ") for line in receipt_lines)
    assert any(line.startswith("ambient convection ") for line in receipt_lines)
    assert any(
        "solver_residual=" in line
        for line in receipt_lines
        if line.startswith("thermal ")
    )
    assert any(
        "prandtl=" in line and "valid_rayleigh=" in line
        for line in receipt_lines
        if line.startswith("ambient convection ")
    )
    assert any(
        line.startswith("tube heat-transfer validity ")
        for line in receipt_lines
    )
    assert any(line.startswith("coolant validity ") for line in receipt_lines)
    assert any(line.startswith("embedded channels ") for line in receipt_lines)
    assert sum(line.startswith("rigid payload ") for line in receipt_lines) == 1
    assert any(line.startswith("resolution refinement ") for line in receipt_lines)
    assert any(line.startswith("coupled not_evaluated=") for line in receipt_lines)
    assert any(
        line.startswith("  structural guidance ")
        for record in result.records
        for line in _solid_receipt_lines(record)
    )
    body_reference = next(
        element["body"]
        for element in assembly_spec["elements"]
        if "body" in element
    )
    compiled_body = gbody.compile_file(
        str(assembly_path.parent / body_reference)
    )
    assert isinstance(compiled_body.anatomy, AcceptedAnatomy)
    terminal_radius_scale = (
        result.receipt.vascular_sizing_receipts[0].terminal_radius_scale
    )
    unguided = realize_vasculature(
        compiled_body.anatomy,
        compiled_body.graph,
        replace(
            SealedVascularConfig(),
            terminal_radius=(
                SealedVascularConfig().terminal_radius
                * terminal_radius_scale
            ),
        ),
    )
    assert isinstance(unguided, AcceptedVasculature)
    guided = result.vasculature[0]
    assert tuple(node.node_id for node in unguided.graph.nodes) == tuple(
        node.node_id for node in guided.graph.nodes
    )
    assert len(unguided.graph.edges) == len(guided.graph.edges)
    assert unguided.receipt.structural_guidance_sample_count == 0
    assert unguided.receipt.maximum_structural_cost_multiplier == 1.0
    assert guided.receipt.structural_guidance_sample_count > 0
    assert guided.receipt.maximum_structural_cost_multiplier == pytest.approx(1.05)
    repeated = compile_assembly(
        assembly_spec, assembly_path.parent
    )
    assert isinstance(repeated, AcceptedAssembly)
    assert repeated.receipt == result.receipt
    assert repeated.vasculature == result.vasculature
    assert repeated.thermal_solutions == result.thermal_solutions
    assert repeated.mechanics_solutions == result.mechanics_solutions


def test_rejected_coupled_cli_does_not_emit_a_glb(tmp_path: Path) -> None:
    assembly_path = _paths.REHEARSAL / "reforge" / "knight_separated.json"
    rejected_spec = json.loads(assembly_path.read_text())
    rejected_spec["service"]["evidence"]["channel_resolution_policy"] = (
        "resolved_only"
    )
    rejected_spec["elements"] = [
        {
            **element,
            **(
                {"body": str((assembly_path.parent / element["body"]).resolve())}
                if isinstance(element.get("body"), str)
                else {}
            ),
            **(
                {"graph": str((assembly_path.parent / element["graph"]).resolve())}
                if isinstance(element.get("graph"), str)
                else {}
            ),
        }
        for element in rejected_spec["elements"]
    ]
    rejected_path = tmp_path / "rejected.json"
    rejected_path.write_text(json.dumps(rejected_spec))
    output = tmp_path / "forbidden.glb"
    exit_code = assembly_main(
        (str(rejected_path), "--glb", str(output))
    )
    assert exit_code == 1
    assert not output.exists()
