"""CLI: ``python -m golem.assembly <spec.json> [--glb out.glb]``.

Solid integrity, surface strata, and the anatomy-owned vascular descent are
reported independently.  A rejected anatomy or vasculature is printed as its
typed obstruction and makes the assembly verdict fail; derived vascular meshes
never counterfeit solid integrity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections.abc import Iterable, Mapping

from golem.assembly.carriers import AssemblyObstruction
from golem.assembly.core import (
    AcceptedAssembly,
    AssemblyRecord,
    CoupledPerformanceReceipt,
    RejectedAssembly,
    compile_assembly,
)
from golem.assembly.exchange import export_scene
from golem.assembly.project import project_json_value
from golem.assembly.service import ServiceIntent
from golem.kernel.sheaf import (
    HeatTransferCorrelationKind,
    HeatTransferValidityInterval,
)
from golem.kernel.anatomy.graph import InfeasibleBifurcationObstruction


def _record_line(record: AssemblyRecord) -> str:
    report = record.report
    stratum = record.stratum
    if stratum == "conduit":
        state = (
            "EMPTY (embedding found no surface)"
            if record.empty
            else f"faces={report['faces']}"
        )
        return (
            f"conduit {record.record_id:22s} "
            f"appearance={record.appearance_material:24s} "
            f"{state}"
        )
    if stratum == "plate_seam":
        return (
            f"plate seam {record.record_id:19s} "
            f"appearance={record.appearance_material:24s} "
            f"cells={report['cells']:4d} faces={report['faces']:6d}"
        )
    if stratum == "eye":
        return (
            f"eye {record.record_id:28s} "
            f"appearance={record.appearance_material:24s} "
            f"faces={report['faces']:6d}"
        )
    if stratum.startswith("vascular_"):
        return (
            f"{stratum:18s} {record.record_id:28s} "
            f"appearance={record.appearance_material:24s} "
            f"edges={report['edges']:4d} "
            f"faces={report['faces']:6d}"
        )
    return (
        f"element {record.record_id:10s} role={record.role:9s} "
        f"res={record.resolution:3d} components={report['components']} "
        f"watertight={report['watertight_main']} "
        f"dust={report['grid_dust_slivers']} faces={report['faces']} "
        f"violations={len(record.violations)}"
    )


def _typed_obstruction_lines(label: str, view: object) -> tuple[str, ...]:
    if not isinstance(view, Mapping) or view.get("status") != "rejected":
        return ()
    return tuple(
        f"  REJECTED {label} [{obstruction.get('kind', 'Obstruction')}] "
        f"{obstruction.get('address', '<unknown>')}: "
        f"{obstruction.get('reason', json.dumps(obstruction, sort_keys=True))}"
        for obstruction in view.get("obstructions", ())
        if isinstance(obstruction, dict)
    )


def _solid_receipt_lines(record: AssemblyRecord) -> tuple[str, ...]:
    if record.stratum != "solid":
        return ()
    report = record.report
    vasculature = report.get("vasculature")
    vascular_lines = (
        (
            "  vascular "
            f"nodes={vasculature['topology']['nodes']} "
            f"edges={vasculature['topology']['edges']} "
            f"terminal_pairs={vasculature['topology']['terminal_pairs']} "
            f"residual={vasculature['maximum_free_cell_residual']:.3e} "
            f"balance={vasculature['boundary_balance_error']:.3e}"
        ),
        (
            "  allocation "
            + json.dumps(
                project_json_value(vasculature["terminal_pairs_by_region"]),
                sort_keys=True,
            )
        ),
        (
            "  stages     "
            + json.dumps(
                project_json_value(vasculature["stage_counts"]), sort_keys=True
            )
        ),
        (
            "  geometry   "
            f"capsule_margin={vasculature['minimum_capsule_margin']:.6e} "
            f"symmetry_error={vasculature['maximum_symmetry_error']:.3e} "
            "intersections=0 "
            f"({vasculature['checked_nonincident_pair_count']} pairs checked)"
        ),
        (
            "  delivery   "
            + json.dumps(
                project_json_value(vasculature["delivered_demand_by_region"]),
                sort_keys=True,
            )
            + (
                " max_relative_error="
                f"{vasculature['maximum_delivery_relative_error']:.3e}"
            )
        ),
        (
            "  solve      "
            f"normalized_pressure_drop={vasculature['pump_pressure_drop']:.6e} "
            f"conservation={vasculature['conservation_error']:.3e}"
        ),
        (
            "  structural guidance "
            f"samples={vasculature.get('structural_guidance_sample_count', 0)} "
            "maximum_cost_multiplier="
            f"{vasculature.get('maximum_structural_cost_multiplier', 1.0):.6e}"
        ),
    ) if isinstance(vasculature, Mapping) and vasculature.get("status") == "accepted" else ()
    return (
        *vascular_lines,
        *_typed_obstruction_lines("anatomy", report.get("anatomy")),
        *_typed_obstruction_lines("vasculature", vasculature),
    )


def _output_lines(records: Iterable[AssemblyRecord]) -> tuple[str, ...]:
    return tuple(
        line
        for record in records
        for line in (_record_line(record), *_solid_receipt_lines(record))
    )


def _assembly_obstruction_lines(result: RejectedAssembly) -> tuple[str, ...]:
    return tuple(map(_assembly_obstruction_line, result.obstructions))


def _assembly_obstruction_line(obstruction: AssemblyObstruction) -> str:
    if isinstance(obstruction, InfeasibleBifurcationObstruction):
        witness = (
            (
                f" supply_capsule={obstruction.supply_capsule}"
                f" return_capsule={obstruction.return_capsule}"
            )
            if obstruction.supply_capsule is not None
            and obstruction.return_capsule is not None
            else (
                " split_edge="
                f"{obstruction.split_edge[0]},{obstruction.split_edge[1]}"
                if obstruction.split_edge is not None
                else ""
            )
            + (
                " candidate_point_index="
                f"{obstruction.candidate_point_index}"
                if obstruction.candidate_point_index is not None
                else ""
            )
        )
        attempted = (
            " attempted_candidate_count="
            f"{obstruction.attempted_candidate_count}"
            if obstruction.attempted_candidate_count is not None
            else ""
        )
        return (
            "REJECTED [InfeasibleBifurcationObstruction] "
            f"region_id={obstruction.region_id} "
            f"phase={obstruction.phase.value} "
            f"lane={obstruction.lane.value} "
            f"predicate={obstruction.predicate.value} "
            f"required={obstruction.required:g} "
            f"observed={obstruction.observed:g} "
            f"terminal_index={obstruction.terminal_index}"
            f"{witness}{attempted}"
        )
    return (
        "REJECTED "
        f"[{type(obstruction).__name__}] "
        f"{json.dumps(project_json_value(obstruction), sort_keys=True, default=str)}"
    )


def _optional_metric(value: float | None) -> str:
    return "not_evaluated" if value is None else f"{value:.6e}"


def _fit_receipt_lines(result: AcceptedAssembly) -> tuple[str, ...]:
    fit_receipts = result.receipt.fit_receipts
    declared = tuple(
        receipt
        for receipt in fit_receipts
        if receipt.declaring_element_id is not None
    )
    return (
        (
            "solid fit "
            f"pairs={len(fit_receipts)} declared={len(declared)} "
            "maximum_penetration_world="
            f"{max((receipt.maximum_penetration_world for receipt in fit_receipts), default=0.0):.6e}"
        ),
        *tuple(
            "fit relation "
            f"element={receipt.declaring_element_id} "
            "with="
            f"{receipt.right_element_id if receipt.declaring_element_id == receipt.left_element_id else receipt.left_element_id} "
            f"law={receipt.fit_law.value} "
            f"minimum_clearance_world={_optional_metric(receipt.minimum_clearance_world)} "
            f"permitted_clearance_world={_optional_metric(receipt.permitted_clearance_world)} "
            f"maximum_penetration_world={receipt.maximum_penetration_world:.6e} "
            f"tolerance_world={receipt.tolerance_world:.6e} "
            f"fitted_surface_fraction={_optional_metric(receipt.fitted_surface_fraction)}"
            for receipt in declared
        ),
    )


def _volume_lines(receipt: CoupledPerformanceReceipt) -> tuple[str, ...]:
    return tuple(
        "physical volume "
        f"element={volume.element_id} solid={volume.solid_fraction:.6e} "
        f"lumen={volume.lumen_fraction:.6e} wall={volume.wall_fraction:.6e} "
        f"minimum_ligament_m={_optional_metric(volume.minimum_ligament_metres)}"
        for volume in receipt.material_volumes
    )


def _sizing_lines(receipt: CoupledPerformanceReceipt) -> tuple[str, ...]:
    return tuple(
        "vascular sizing "
        f"element={sizing.element_id} heat_w={sizing.heat_load_watts:.6e} "
        f"total_rise_k={sizing.maximum_total_temperature_rise_kelvin:.6e} "
        f"approach_reserve_k={sizing.reserved_thermal_approach_kelvin:.6e} "
        f"bulk_rise_k={sizing.allowable_bulk_coolant_rise_kelvin:.6e} "
        f"required_flow_m3_s={sizing.required_volumetric_flow_cubic_metres_per_second:.6e} "
        f"baseline_flow_m3_s={sizing.baseline_volumetric_flow_cubic_metres_per_second:.6e} "
        f"radius_scale={sizing.terminal_radius_scale:.6e} "
        f"achieved_flow_m3_s={sizing.achieved_volumetric_flow_cubic_metres_per_second:.6e}"
        for sizing in receipt.vascular_sizing_receipts
    )


def _hydraulic_lines(result: AcceptedAssembly) -> tuple[str, ...]:
    return tuple(
        "hydraulics "
        f"circuit={circuit_index} "
        f"flow_m3_s={hydraulics.receipt.pump_volumetric_flow_cubic_metres_per_second:.6e} "
        f"pressure_drop_pa={hydraulics.receipt.pump_pressure_drop_pascal:.6e} "
        f"power_w={hydraulics.receipt.hydraulic_pump_power_watts:.6e} "
        f"max_reynolds={max(map(lambda edge: edge.reynolds_validity.reynolds_number, hydraulics.edge_flows), default=0.0):.6e} "
        f"residual={hydraulics.receipt.maximum_free_node_normalized_residual:.3e} "
        f"balance={hydraulics.receipt.relative_boundary_flow_imbalance:.3e}"
        for circuit_index, hydraulics in enumerate(result.hydraulics)
    )


def _thermal_lines(receipt: CoupledPerformanceReceipt) -> tuple[str, ...]:
    return tuple(
        "thermal "
        f"element={thermal.element_id} case={thermal.case_id} "
        f"maximum_k={thermal.maximum_temperature_kelvin:.6e} "
        f"mean_k={thermal.mean_temperature_kelvin:.6e} "
        f"gradient_k_m={thermal.maximum_temperature_gradient_kelvin_per_metre:.6e} "
        f"coolant_outlet_k={_optional_metric(thermal.coolant_outlet_temperature_kelvin)} "
        f"hottest={thermal.hottest_semantic_address} "
        "correlations="
        + json.dumps(tuple(item.value for item in thermal.heat_transfer_correlations))
        + " "
        f"source_w={thermal.energy.authored_source_watts:.6e} "
        f"solver_residual={thermal.energy.maximum_normalized_residual:.3e} "
        f"energy_balance={thermal.energy.relative_energy_imbalance:.3e}"
        for thermal in receipt.thermal_receipts
    )


def _ambient_lines(receipt: CoupledPerformanceReceipt) -> tuple[str, ...]:
    return tuple(
        "ambient convection "
        f"element={thermal.element_id} case={thermal.case_id} "
        f"gas={convection.gas_material_id.value} "
        f"film_k={convection.film_temperature_kelvin:.6e} "
        f"prandtl={convection.prandtl_number:.6e} "
        f"rayleigh={convection.rayleigh_number:.6e} "
        f"nusselt={convection.nusselt_number:.6e} "
        f"h_w_m2_k={convection.watt_per_square_metre_kelvin:.6e} "
        f"valid_prandtl_gt={convection.validity.minimum_prandtl:.6e} "
        f"valid_rayleigh=[{convection.validity.minimum_rayleigh:.6e},"
        f"{convection.validity.maximum_rayleigh:.6e})"
        for thermal in receipt.thermal_receipts
        for convection in (thermal.natural_convection,)
        if convection is not None
    )


def _tube_validity_lines(
    result: AcceptedAssembly, coolant: object
) -> tuple[str, ...]:
    if coolant is None:
        return ()
    wall_heat_fluxes = tuple(
        flux
        for solution in result.thermal_solutions
        for flux in solution.wall_heat_fluxes
    )
    tube_validities = tuple(
        dict.fromkeys(
            (flux.correlation_kind, flux.validity)
            for flux in wall_heat_fluxes
        )
    )

    def tube_validity_line(
        kind: HeatTransferCorrelationKind,
        validity: HeatTransferValidityInterval,
    ) -> str:
        matching = tuple(
            flux
            for flux in wall_heat_fluxes
            if flux.correlation_kind is kind and flux.validity == validity
        )
        entrance_diameters = tuple(
            flux.upstream_development_length_metres
            / flux.hydraulic_diameter_metres
            for flux in matching
        )
        return (
            "tube heat-transfer validity "
            f"correlation={kind.value} "
            f"coolant={coolant.material.material_id.value} "
            f"reynolds=[{min((flux.reynolds_number for flux in matching), default=0.0):.6e},"
            f"{max((flux.reynolds_number for flux in matching), default=0.0):.6e}] "
            f"prandtl=[{min((flux.prandtl_number for flux in matching), default=0.0):.6e},"
            f"{max((flux.prandtl_number for flux in matching), default=0.0):.6e}] "
            f"entrance_diameters=[{min(entrance_diameters, default=0.0):.6e},"
            f"{max(entrance_diameters, default=0.0):.6e}] "
            f"hydraulic_diameter_m=[{min((flux.hydraulic_diameter_metres for flux in matching), default=0.0):.6e},"
            f"{max((flux.hydraulic_diameter_metres for flux in matching), default=0.0):.6e}] "
            f"valid_reynolds=[{validity.minimum_reynolds:.6e},"
            f"{validity.maximum_reynolds:.6e}{')' if validity.maximum_reynolds_is_exclusive else ']'} "
            f"valid_prandtl=[{validity.minimum_prandtl:.6e},"
            f"{validity.maximum_prandtl:.6e}] "
            f"minimum_entrance_diameters={validity.minimum_entrance_length_diameters:.6e}"
        )

    return tuple(
        tube_validity_line(kind, validity)
        for kind, validity in tube_validities
    )


def _coolant_validity_lines(
    result: AcceptedAssembly,
    receipt: CoupledPerformanceReceipt,
    coolant: object,
) -> tuple[str, ...]:
    if coolant is None:
        return ()
    return (
        "coolant validity "
        f"material={coolant.material.material_id.value} "
        f"inlet_k={coolant.inlet_temperature_kelvin:.6e} "
        f"outlet_k={max((thermal.coolant_outlet_temperature_kelvin for thermal in receipt.thermal_receipts if thermal.coolant_outlet_temperature_kelvin is not None), default=coolant.inlet_temperature_kelvin):.6e} "
        f"catalogue_temperature=[{coolant.material.operating_temperature.minimum_kelvin:.6e},"
        f"{coolant.material.operating_temperature.maximum_kelvin:.6e}] "
        f"pressure=[{min((pressure.pressure_pascal for hydraulics in result.hydraulics for pressure in hydraulics.node_pressures), default=0.0):.6e},"
        f"{max((pressure.pressure_pascal for hydraulics in result.hydraulics for pressure in hydraulics.node_pressures), default=0.0):.6e}] "
        f"catalogue_pressure=[{coolant.material.allowable_pressure.minimum_pascal:.6e},"
        f"{coolant.material.allowable_pressure.maximum_pascal:.6e}]",
    )


def _mechanics_lines(receipt: CoupledPerformanceReceipt) -> tuple[str, ...]:
    return tuple(
        "mechanics "
        f"element={element.element_id} case={element.case_id} "
        f"displacement_m={element.mechanics.maximum_displacement:.6e} "
        f"von_mises_pa={element.mechanics.maximum_von_mises_stress:.6e} "
        f"yield_safety={_optional_metric(element.mechanics.minimum_yield_safety_factor)} "
        f"buckling_factor={_optional_metric(element.mechanics.lowest_positive_buckling_load_factor)} "
        f"burst_safety={_optional_metric(element.minimum_burst_safety_factor)} "
        f"critical={element.critical_semantic_address} "
        f"solver_residual={element.mechanics.normalized_linear_solver_residual:.3e} "
        f"force_balance={element.mechanics.relative_force_imbalance:.3e} "
        f"moment_balance={element.mechanics.relative_moment_imbalance:.3e} "
        "not_evaluated="
        + json.dumps(tuple(item.value for item in element.mechanics.not_evaluated))
        for element in receipt.mechanics_receipts
    )


def _embedded_lines(receipt: CoupledPerformanceReceipt) -> tuple[str, ...]:
    return tuple(
        "embedded channels "
        f"element={element.element_id} case={element.case_id} "
        f"edges={embedded.receipt.positive_radius_edge_count} "
        f"diameter_pitch={embedded.receipt.maximum_outer_diameter_to_grid_pitch:.6e} "
        f"lumen_fraction={embedded.receipt.lumen_volume_fraction_upper_bound:.6e} "
        f"envelope_fraction={embedded.receipt.channel_envelope_volume_fraction_upper_bound:.6e} "
        f"lame_hoop_pa={embedded.receipt.maximum_lame_inner_wall_hoop_stress_pascal:.6e} "
        f"burst_safety={embedded.receipt.minimum_burst_safety_factor:.6e} "
        "not_evaluated="
        + json.dumps(tuple(item.value for item in embedded.receipt.not_evaluated))
        for element in receipt.mechanics_receipts
        for embedded in (element.embedded_channel_mechanics,)
        if embedded is not None
    )


def _interface_lines(receipt: CoupledPerformanceReceipt) -> tuple[str, ...]:
    return tuple(
        "rigid payload "
        f"element={interface.element_id} case={interface.case_id} "
        f"mass_kg={interface.mass_kilograms:.6e} "
        f"gap_m={interface.interface_gap_metres:.6e} "
        f"force_residual_n={interface.force_balance_residual_newtons:.3e} "
        f"moment_residual_nm={interface.moment_balance_residual_newton_metres:.3e}"
        for interface in receipt.interface_receipts
    )


def _refinement_lines(receipt: CoupledPerformanceReceipt) -> tuple[str, ...]:
    return tuple(
        "resolution refinement "
        f"element={refinement.element_id} case={refinement.case_id} "
        f"base={refinement.base_resolution} refined={refinement.refined_resolution} "
        f"temperature_utilization_change={_optional_metric(refinement.maximum_temperature_utilization_change)} "
        f"displacement_utilization_change={refinement.maximum_displacement_utilization_change:.6e} "
        f"yield_utilization_change={_optional_metric(refinement.minimum_yield_utilization_change)}"
        for refinement in receipt.refinement_receipts
    )


def _coupled_receipt_lines(result: AcceptedAssembly) -> tuple[str, ...]:
    receipt = result.receipt
    if not isinstance(receipt, CoupledPerformanceReceipt):
        return ()
    service = result.service
    coolant = service.coolant if isinstance(service, ServiceIntent) else None
    return (
        "coupled evidence "
        f"metres_per_world_unit={receipt.metres_per_world_unit:.6e} "
        f"materials={json.dumps(receipt.material_ids)} "
        f"cases={json.dumps(receipt.service_case_ids)}",
        *_volume_lines(receipt),
        *_sizing_lines(receipt),
        *_hydraulic_lines(result),
        *_thermal_lines(receipt),
        *_ambient_lines(receipt),
        *_tube_validity_lines(result, coolant),
        *_coolant_validity_lines(result, receipt, coolant),
        *_mechanics_lines(receipt),
        *_embedded_lines(receipt),
        *_interface_lines(receipt),
        *_refinement_lines(receipt),
        "coupled not_evaluated="
        + json.dumps(tuple(item.value for item in receipt.not_evaluated)),
        "coupling "
        f"iterations={receipt.fixed_point_iterations} "
        f"maximum_state_delta={receipt.maximum_state_delta:.3e}",
    )


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        import golem.assembly as package

        print(package.__doc__)
        return 0
    spec_path = Path(argv[0])
    spec = json.loads(spec_path.read_text())
    result = compile_assembly(spec, spec_path.parent)
    output_path = (
        argv[argv.index("--glb") + 1] if "--glb" in argv else None
    )
    if isinstance(result, RejectedAssembly):
        lines = (
            *_assembly_obstruction_lines(result),
            *(
                (f"scene refused -> {output_path} (typed rejection above)",)
                if output_path is not None
                else ()
            ),
            "assembly verdict: RejectedAssembly",
        )
        print("\n".join(lines))
        return 1
    if not isinstance(result, AcceptedAssembly):
        print("REJECTED [MalformedAssemblyResult] internal result mismatch")
        return 1
    if output_path is not None:
        export_scene(result, output_path)
    lines = (
        *_output_lines(result.records),
        f"physical_evidence: {result.receipt.physical_evidence.value}",
        *_fit_receipt_lines(result),
        *_coupled_receipt_lines(result),
        *(
            (f"scene -> {output_path}",) if output_path is not None else ()
        ),
        "assembly verdict: AcceptedAssembly",
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
