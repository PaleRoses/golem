"""Canonical assembly compiler hub and pinned coupled-stage seams."""

from __future__ import annotations

import json
import math

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import chain as _chain
from math import fsum
from pathlib import Path

from golem.assembly.address import (_domain_cell_indices_owned_by_provenance, _owner_provenance_at_points, _semantic_address_for_cell)
from golem.assembly.carriers import (AcceptedAssembly, AssemblyObstruction, AssemblyReceipt, AssemblyRecord, AssemblyResolutionPolicy, AssemblyResult, AssemblyStratum, CoupledPerformanceReceipt, CoupledUnevaluatedPhysics, ElementFitReceipt, ElementMaterialVolumeReceipt, ElementMechanicsReceipt, ElementRole, ElementThermalReceipt, PinnedResolution, RES_MAX, RES_MIN, ROLES, RejectedAssembly, ResolutionRefinementReceipt, RigidPayloadInterfaceReceipt, TargetPitch, VascularThermalSizingReceipt, VisualAssemblyReceipt, _CompiledElement, _LoadedElement, _SEALED_THERMAL_APPROACH_RESERVE_FRACTION, _SEALED_VASCULAR_SIZING_REFINEMENTS)
from golem.assembly.compile import (_compile_element, _compiled_integrity_obstructions, derive_vascular_records, pitch_res)
from golem.assembly.coupled import (_accepted_vascular_sections, _coupled_case_mechanics, _coupled_mechanics_receipts, _coupled_not_evaluated_kinds, _deformable_creature_element, _ligament_violation_obstructions, _physical_hydraulic_results, _resolved_material_masks, _rigid_payload_joint_obstructions, _solve_resolved_coupled_assembly_impl)
from golem.assembly.descent import (descend_local_results, traverse_results)
from golem.assembly.fit import (_attach_fit_receipts, _compiled_fit_descent)
from golem.assembly.freeze import _frozen_report
from golem.assembly.hydraulics import (_EmbeddedVascularSection, _ResolvedVascularSection, _VascularPhysicalSection, _assembly_hydraulics_result, _solve_element_physical_hydraulics, _vascular_physical_section)
from golem.assembly.ingest import (_AssemblySourceSnapshot, _assembly_input_obstructions, _assembly_source_digest, _capture_assembly_snapshot, _known_semantic_addresses, _load_element, _load_element_snapshot, load_element_graph)
from golem.assembly.materials_binding import (_material_binding_obstructions, _section_material_volume_receipt, _section_minimum_ligament_metres)
from golem.assembly.mechanics.algebra import (_burst_safety_factor, _critical_mechanics_address)
from golem.assembly.mechanics.coalgebra import _payload_interface_transfers_by_case
from golem.assembly.mechanics.compile import (_embedded_channel_mechanics, _solve_element_mechanics)
from golem.assembly.mounts import mount_vascular_graph
from golem.assembly.obstructions import (AssemblyInputObstruction, CoolantTemperatureValidityObstruction, DuplicateElementIdObstruction, ElementClearanceObstruction, ElementFitLaw, ElementFitObstruction, ElementSourceObstruction, ElementSurfaceFormationObstruction, EmptyPhysicalSolidDomainObstruction, EmptySurfaceConduitObstruction, EmptyThermalRegionObstruction, EquipmentBodyDialectObstruction, InsufficientBurstMarginObstruction, InsufficientVascularThermalCapacityObstruction, InvalidResolutionPolicyObstruction, MalformedAssemblyObstruction, MissingMechanicalSupportObstruction, MissingPhysicalVasculatureObstruction, MissingRigidPayloadInterfaceObstruction, MissingSolidMaterialAssignmentObstruction, MissingVascularWallMaterialObstruction, PlateElementObstruction, PumpPowerPressureLimitObstruction, RemainingLigamentViolationObstruction, ResolutionRefinementObstruction, RigidPayloadBalanceObstruction, RigidPayloadInterfaceGapObstruction, SolidIntegrityObstruction, ThermalCouplingNonConvergenceObstruction, ThermalGradientLimitObstruction, ThermalLimitObstruction, UnderspecifiedAmbientHeatTransferObstruction, UnknownElementRoleObstruction, UnsupportedAssemblyInterfaceObstruction, UnsupportedMechanicalAddressObstruction, UnsupportedStructuralRoutingMountObstruction, UnsupportedThermalAddressObstruction)
from golem.assembly.refinement import (_attach_resolution_refinement_evidence_impl, _local_refinement_receipt, _optional_yield_utilization_change, _receipts_by_case_key)
from golem.assembly.service import (JointLaw, PhysicalEvidenceStatus, PressurePumpBudget, RejectedServiceIntent, ServiceIntent, VisualOnlyServiceIntent, decode_service_intent)
from golem.assembly.thermal import (_SolvedElementThermal, _solve_element_thermal)
from golem.assembly.vascular import (_CoolantSizingContext, _replace_element_vasculature, _structurally_guided_elements, _thermally_sized_element)
from golem.assembly.voxel import (_derive_resolved_material_domain, _unperforated_material_masks)
from golem.kernel import engine as E
from golem.kernel.anatomy import (AcceptedAnatomy, AcceptedPhysicalHydraulics, AcceptedVasculature, RejectedAnatomy, RejectedVasculature, SealedVascularConfig, derive_vascular_material_masks, realize_vasculature, vasculature_to_dict)
from golem.kernel.mechanics import (AcceptedMechanics, CellMechanics, MechanicsCriteria)


def _resolution_policy_obstruction(
    policy: AssemblyResolutionPolicy,
) -> InvalidResolutionPolicyObstruction | None:
    match policy:
        case PinnedResolution(resolution) if (
            isinstance(resolution, bool)
            or not isinstance(resolution, int)
            or not RES_MIN <= resolution <= RES_MAX
        ):
            return InvalidResolutionPolicyObstruction(
                policy,
                f"pinned resolution must be an integer in [{RES_MIN}, {RES_MAX}]",
            )
        case TargetPitch(pitch) if (
            isinstance(pitch, bool)
            or not isinstance(pitch, (int, float))
            or not math.isfinite(float(pitch))
            or float(pitch) <= 0.0
        ):
            return InvalidResolutionPolicyObstruction(
                policy,
                "target pitch must be finite and positive",
            )
        case PinnedResolution() | TargetPitch():
            return None
        case _:
            return InvalidResolutionPolicyObstruction(
                policy,
                "expected PinnedResolution or TargetPitch",
            )


def compile_assembly(
    spec: object,
    base_dir: Path,
    resolution_policy: AssemblyResolutionPolicy | None = None,
) -> AssemblyResult:
    input_obstructions = _assembly_input_obstructions(spec)
    if input_obstructions:
        return RejectedAssembly(input_obstructions)
    if not isinstance(spec, dict):
        return RejectedAssembly(
            (MalformedAssemblyObstruction("/", "expected an object"),)
        )
    raw_pitch = spec.get("pitch", 0.017)
    if resolution_policy is None and (
        isinstance(raw_pitch, bool)
        or not isinstance(raw_pitch, (int, float))
    ):
        return RejectedAssembly(
            (
                InvalidResolutionPolicyObstruction(
                    raw_pitch,
                    "assembly pitch must be a finite positive number",
                ),
            )
        )
    policy = (
        resolution_policy
        if resolution_policy is not None
        else TargetPitch(float(raw_pitch))
    )
    policy_obstruction = _resolution_policy_obstruction(policy)
    if policy_obstruction is not None:
        return RejectedAssembly((policy_obstruction,))
    snapshot = _capture_assembly_snapshot(spec, base_dir)
    if isinstance(snapshot, ElementSourceObstruction):
        return RejectedAssembly((snapshot,))
    canonical_spec = json.dumps(
        spec,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _compile_assembly_cached(canonical_spec, snapshot, policy)


@lru_cache(maxsize=4)
def _compile_assembly_cached(
    canonical_spec: str,
    snapshot: _AssemblySourceSnapshot,
    resolution_policy: AssemblyResolutionPolicy,
) -> AssemblyResult:
    spec = json.loads(canonical_spec)
    loaded_descent = descend_local_results(
        tuple(
            _load_element_snapshot(entry, source)
            for entry, source in zip(
                spec["elements"],
                snapshot.elements,
                strict=True,
            )
        ),
        _LoadedElement,
    )
    if isinstance(loaded_descent, RejectedAssembly):
        return loaded_descent
    loaded = loaded_descent
    service_result = decode_service_intent(
        spec.get("service"),
        _known_semantic_addresses(loaded),
    )
    if isinstance(service_result, RejectedServiceIntent):
        return RejectedAssembly(service_result.obstructions)
    compiled_descent = descend_local_results(
        tuple(
            _compile_element(element, resolution_policy, snapshot.digest)
            for element in loaded
        ),
        _CompiledElement,
    )
    if isinstance(compiled_descent, RejectedAssembly):
        return compiled_descent
    compiled = compiled_descent
    embedded_obstructions = tuple(
        _chain.from_iterable(
            result.obstructions
            for result in _chain(
                (
                    element.anatomy
                    for element in compiled
                    if isinstance(element.anatomy, RejectedAnatomy)
                ),
                (
                    element.vasculature
                    for element in compiled
                    if isinstance(element.vasculature, RejectedVasculature)
                ),
            )
        )
    )
    if embedded_obstructions:
        return RejectedAssembly(embedded_obstructions)
    integrity_obstructions = _compiled_integrity_obstructions(
        compiled,
        resolution_policy,
    )
    if integrity_obstructions:
        return RejectedAssembly(integrity_obstructions)
    fit_descent = descend_local_results(
        _compiled_fit_descent(compiled),
        ElementFitReceipt,
    )
    if isinstance(fit_descent, RejectedAssembly):
        return fit_descent
    service = service_result.intent
    accepted = (
        _accepted_visual_assembly(compiled, service)
        if isinstance(service, VisualOnlyServiceIntent)
        else _compile_coupled_assembly(compiled, service)
    )
    return _attach_fit_receipts(accepted, fit_descent)


def _accepted_visual_assembly(
    compiled: tuple[_CompiledElement, ...],
    service: VisualOnlyServiceIntent,
) -> AcceptedAssembly:
    return AcceptedAssembly(
        records=tuple(
            _chain.from_iterable(element.records for element in compiled)
        ),
        service=service,
        receipt=VisualAssemblyReceipt(),
        vasculature=tuple(
            element.vasculature
            for element in compiled
            if isinstance(element.vasculature, AcceptedVasculature)
        ),
    )


def _element_peak_heat_load_watts(
    element_id: str,
    service: ServiceIntent,
) -> float:
    return max(
        (
            fsum(
                source.watts
                for source in service_case.heat_sources
                if getattr(source.address, "element_id", None) == element_id
            )
            for service_case in service.cases
        ),
        default=0.0,
    )


def _realize_element_vasculature(
    element: _CompiledElement,
    service: ServiceIntent,
    config: SealedVascularConfig,
    structural_cells: tuple[CellMechanics, ...],
) -> _CompiledElement | RejectedAssembly:
    if not isinstance(element.anatomy, AcceptedAnatomy) or not isinstance(
        element.vasculature,
        AcceptedVasculature,
    ):
        return element
    result = realize_vasculature(
        element.anatomy,
        element.source_graph,
        config,
        structural_cells=structural_cells,
        structural_metres_per_world_unit=(
            service.scale.metres_per_world_unit
        ),
    )
    return (
        _replace_element_vasculature(element, result)
        if isinstance(result, AcceptedVasculature)
        else RejectedAssembly(result.obstructions)
    )


def _descend_terminal_radius_sizing(
    terminal_radius_scale: float,
    remaining_refinements: int,
    baseline_element: _CompiledElement,
    baseline_hydraulics: AcceptedPhysicalHydraulics,
    required_flow: float,
    service: ServiceIntent,
    structural_cells: tuple[CellMechanics, ...],
    radius_scaling_exponent: float,
    stages: "CoupledStages",
) -> tuple[_CompiledElement, float, float] | RejectedAssembly:
    sized_element_result = (
        baseline_element
        if terminal_radius_scale == 1.0
        else stages.realize(
            baseline_element,
            service,
            replace(
                SealedVascularConfig(),
                terminal_radius=(
                    SealedVascularConfig().terminal_radius
                    * terminal_radius_scale
                ),
            ),
            structural_cells,
        )
    )
    if isinstance(sized_element_result, RejectedAssembly):
        return sized_element_result
    sized_element = sized_element_result
    achieved_hydraulics = _assembly_hydraulics_result(
        baseline_hydraulics
        if terminal_radius_scale == 1.0
        else _solve_element_physical_hydraulics(
            sized_element.mounted_vasculature,
            service,
        )
    )
    if isinstance(achieved_hydraulics, RejectedAssembly):
        return achieved_hydraulics
    achieved_flow = (
        achieved_hydraulics.receipt.pump_volumetric_flow_cubic_metres_per_second
    )
    if achieved_flow >= required_flow * (1.0 - 1.0e-10):
        return sized_element, terminal_radius_scale, achieved_flow
    if remaining_refinements == 0:
        return RejectedAssembly(
            (
                InsufficientVascularThermalCapacityObstruction(
                    baseline_element.element_id,
                    required_flow,
                    achieved_flow,
                ),
            )
        )
    return _descend_terminal_radius_sizing(
        terminal_radius_scale
        * (required_flow / achieved_flow)
        ** radius_scaling_exponent,
        remaining_refinements - 1,
        baseline_element,
        baseline_hydraulics,
        required_flow,
        service,
        structural_cells,
        radius_scaling_exponent,
        stages,
    )


def _thermally_size_vasculature(
    compiled: tuple[_CompiledElement, ...],
    service: ServiceIntent,
    structural_cells: tuple[CellMechanics, ...] = (),
    *,
    stages: "CoupledStages | None" = None,
) -> (
    tuple[tuple[_CompiledElement, ...], tuple[VascularThermalSizingReceipt, ...]]
    | RejectedAssembly
):
    stages = stages if stages is not None else PRODUCTION_COUPLED_STAGES
    if service.coolant is None:
        return _structurally_guided_elements(
            compiled,
            service,
            structural_cells,
            stages,
        )
    fluid = service.coolant.material
    allowable_coolant_temperature = min(
        service.limits.maximum_temperature_kelvin,
        fluid.operating_temperature.maximum_kelvin,
    )
    maximum_total_rise = (
        allowable_coolant_temperature
        - service.coolant.inlet_temperature_kelvin
    )
    if maximum_total_rise <= 0.0:
        return RejectedAssembly(
            (
                CoolantTemperatureValidityObstruction(
                    "pump:outlet",
                    service.coolant.inlet_temperature_kelvin,
                    fluid.operating_temperature.minimum_kelvin,
                    allowable_coolant_temperature,
                ),
            )
        )
    reserved_thermal_approach = (
        _SEALED_THERMAL_APPROACH_RESERVE_FRACTION * maximum_total_rise
    )
    context = _CoolantSizingContext(
        fluid=fluid,
        maximum_total_rise=maximum_total_rise,
        reserved_thermal_approach=reserved_thermal_approach,
        allowable_bulk_rise=maximum_total_rise - reserved_thermal_approach,
        radius_scaling_exponent=(
            0.25
            if isinstance(service.coolant.pump_budget, PressurePumpBudget)
            else 0.5
        ),
    )
    local_results = tuple(
        _thermally_sized_element(
            element,
            service,
            structural_cells,
            context,
            stages,
            _element_peak_heat_load_watts,
            _descend_terminal_radius_sizing,
        )
        for element in compiled
    )
    return traverse_results(
        local_results,
        lambda sized: (
            tuple(result[0] for result in sized),
            tuple(
                receipt
                for _element, receipt in sized
                if receipt is not None
            ),
        ),
    )


def _provisional_unperforated_structural_cells(
    compiled: tuple[_CompiledElement, ...],
    analysis_compiled: tuple[_CompiledElement, ...],
    service: ServiceIntent,
) -> tuple[CellMechanics, ...] | RejectedAssembly:
    element = _deformable_creature_element(analysis_compiled)
    if isinstance(element, RejectedAssembly):
        return element
    if element.entry.get("mount"):
        return RejectedAssembly(
            (
                UnsupportedStructuralRoutingMountObstruction(
                    element.element_id,
                    (
                        "mechanics-aware vascular routing currently requires "
                        "the creature authoring frame to equal the assembly frame"
                    ),
                ),
            )
        )
    solid, lumen, wall = _unperforated_material_masks(element.evaluated)
    domain = _derive_resolved_material_domain(
        element,
        solid,
        lumen,
        wall,
        service.scale.metres_per_world_unit,
    )
    if isinstance(domain, EmptyPhysicalSolidDomainObstruction):
        return RejectedAssembly((domain,))
    payloads = tuple(
        candidate
        for candidate in compiled
        if candidate.role is ElementRole.EQUIPMENT
    )
    payload_transfers = _payload_interface_transfers_by_case(
        element,
        payloads,
        domain,
        service,
    )
    if isinstance(payload_transfers, RejectedAssembly):
        return payload_transfers
    mechanics_results = tuple(
        _solve_element_mechanics(
            element,
            domain,
            service,
            service_case,
            case_index,
            None,
            element.mounted_vasculature,
            None,
            MechanicsCriteria(),
            tuple(
                force
                for case_id, transfers in payload_transfers
                if case_id == service_case.case_id
                for transfer in transfers
                for force in transfer.nodal_forces
            ),
        )
        for case_index, service_case in enumerate(service.cases)
    )
    return traverse_results(
        mechanics_results,
        lambda accepted: tuple(
            cell
            for result in accepted
            if isinstance(result, AcceptedMechanics)
            for cell in result.cells
        ),
    )


def _evaluate_compiled_for_physics(
    compiled: tuple[_CompiledElement, ...],
    resolution: int,
) -> tuple[_CompiledElement, ...] | RejectedAssembly:
    def evaluate_element(
        element: _CompiledElement,
    ) -> _CompiledElement | RejectedAssembly:
        evaluated = E.evaluate(element.graph, res=resolution)
        return (
            RejectedAssembly(
                (
                    ElementSurfaceFormationObstruction(
                        element.element_id,
                        evaluated.obstructions,
                    ),
                )
            )
            if isinstance(evaluated, E.RejectedSurfaceFormation)
            else replace(
                element,
                evaluated=evaluated,
            )
        )

    return traverse_results(
        tuple(
            map(
                evaluate_element,
                compiled,
            )
        ),
        lambda accepted: accepted,
    )


def _solve_sized_coupled_assembly(
    compiled: tuple[_CompiledElement, ...],
    analysis_compiled: tuple[_CompiledElement, ...],
    service: ServiceIntent,
    vascular_sizing_receipts: tuple[VascularThermalSizingReceipt, ...],
    *,
    include_linearized_buckling_evidence: bool = True,
) -> AssemblyResult:
    vascular_elements = tuple(
        element
        for element in analysis_compiled
        if isinstance(element.vasculature, AcceptedVasculature)
        and element.mounted_vasculature is not None
    )
    sections_result = _accepted_vascular_sections(vascular_elements, service)
    if isinstance(sections_result, RejectedAssembly):
        return sections_result
    ligament_obstructions = _ligament_violation_obstructions(
        sections_result,
        service,
    )
    if ligament_obstructions:
        return RejectedAssembly(ligament_obstructions)
    hydraulic_results = _physical_hydraulic_results(vascular_elements, service)
    return traverse_results(
        tuple(map(_assembly_hydraulics_result, hydraulic_results)),
        lambda accepted_hydraulics: _solve_resolved_coupled_assembly(
            compiled,
            analysis_compiled,
            service,
            sections_result,
            accepted_hydraulics,
            vascular_sizing_receipts,
            include_linearized_buckling_evidence=(
                include_linearized_buckling_evidence
            ),
        ),
    )


def _compile_coupled_assembly(
    compiled: tuple[_CompiledElement, ...],
    service: ServiceIntent,
    *,
    stages: "CoupledStages | None" = None,
) -> AssemblyResult:
    stages = stages if stages is not None else PRODUCTION_COUPLED_STAGES
    provisional_analysis = stages.evaluate(
        compiled,
        service.evidence_policy.physics_resolution,
    )
    if isinstance(provisional_analysis, RejectedAssembly):
        return provisional_analysis
    provisional_vascular_elements = tuple(
        element
        for element in provisional_analysis
        if isinstance(element.vasculature, AcceptedVasculature)
        and element.mounted_vasculature is not None
    )
    material_binding_obstructions = _material_binding_obstructions(
        provisional_analysis,
        provisional_vascular_elements,
        service,
    )
    if material_binding_obstructions:
        return RejectedAssembly(material_binding_obstructions)
    if service.coolant is not None and not provisional_vascular_elements:
        return RejectedAssembly(
            tuple(
                MissingPhysicalVasculatureObstruction(element.element_id)
                for element in provisional_analysis
                if element.role is ElementRole.CREATURE
            )
            or (MissingPhysicalVasculatureObstruction("<assembly>"),)
        )
    routing_element = _deformable_creature_element(provisional_analysis)
    if isinstance(routing_element, RejectedAssembly):
        return routing_element
    structural_cells_result = (
        _provisional_unperforated_structural_cells(
            compiled,
            provisional_analysis,
            service,
        )
        if isinstance(routing_element.anatomy, AcceptedAnatomy)
        and isinstance(routing_element.vasculature, AcceptedVasculature)
        else ()
    )
    if isinstance(structural_cells_result, RejectedAssembly):
        return structural_cells_result
    sizing_result = _thermally_size_vasculature(
        compiled,
        service,
        structural_cells_result,
        stages=stages,
    )
    if isinstance(sizing_result, RejectedAssembly):
        return sizing_result
    sized_compiled, vascular_sizing_receipts = sizing_result
    analysis_compiled = tuple(
        replace(element, evaluated=provisional.evaluated)
        for element, provisional in zip(
            sized_compiled,
            provisional_analysis,
            strict=True,
        )
    )
    base_result = stages.solve(
        sized_compiled,
        analysis_compiled,
        service,
        vascular_sizing_receipts,
    )
    return (
        _attach_resolution_refinement_evidence(
            sized_compiled,
            service,
            base_result,
            stages=stages,
        )
        if isinstance(base_result, AcceptedAssembly)
        and service.evidence_policy.resolution_refinement_factor > 1
        else base_result
    )


@dataclass(frozen=True)
class CoupledStages:
    realize: Callable[..., _CompiledElement | RejectedAssembly] = (
        _realize_element_vasculature
    )
    evaluate: Callable[
        ...,
        tuple[_CompiledElement, ...] | RejectedAssembly,
    ] = (
        _evaluate_compiled_for_physics
    )
    solve: Callable[..., AssemblyResult] = _solve_sized_coupled_assembly


PRODUCTION_COUPLED_STAGES = CoupledStages()


def _normalized_refinement_change(
    base: float,
    refined: float,
    normalization: float,
) -> float:
    return abs(refined - base) / max(abs(normalization), 1.0e-15)


def _attach_resolution_refinement_evidence(
    compiled: tuple[_CompiledElement, ...],
    service: ServiceIntent,
    base: AcceptedAssembly,
    *,
    stages: CoupledStages | None = None,
) -> AssemblyResult:
    return _attach_resolution_refinement_evidence_impl(
        compiled,
        service,
        base,
        stages=stages if stages is not None else PRODUCTION_COUPLED_STAGES,
        normalized_refinement_change=_normalized_refinement_change,
    )


def _solve_resolved_coupled_assembly(
    compiled: tuple[_CompiledElement, ...],
    analysis_compiled: tuple[_CompiledElement, ...],
    service: ServiceIntent,
    physical_sections: tuple[_VascularPhysicalSection, ...],
    hydraulic_results: tuple[AcceptedPhysicalHydraulics, ...],
    vascular_sizing_receipts: tuple[VascularThermalSizingReceipt, ...] = (),
    *,
    include_linearized_buckling_evidence: bool = True,
) -> AssemblyResult:
    return _solve_resolved_coupled_assembly_impl(
        compiled,
        analysis_compiled,
        service,
        physical_sections,
        hydraulic_results,
        vascular_sizing_receipts,
        include_linearized_buckling_evidence=(
            include_linearized_buckling_evidence
        ),
    )
