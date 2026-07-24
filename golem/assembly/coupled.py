"""Coupled assembly lowering and global result-gluing algebra."""

from __future__ import annotations

from dataclasses import replace
from itertools import chain as _chain
import numpy as np
from numpy.typing import NDArray

from golem.assembly.carriers import (AcceptedAssembly, AssemblyResult, CoupledPerformanceReceipt, CoupledUnevaluatedPhysics, ElementMechanicsReceipt, ElementThermalReceipt, ElementRole, RejectedAssembly, VascularThermalSizingReceipt, _CompiledElement)
from golem.assembly.descent import collect_results
from golem.assembly.hydraulics import (_EmbeddedVascularSection, _ResolvedVascularSection, _VascularPhysicalSection, _solve_element_physical_hydraulics, _vascular_physical_section)
from golem.assembly.materials_binding import (_section_material_volume_receipt, _section_minimum_ligament_metres)
from golem.assembly.mechanics.algebra import (_burst_safety_factor, _critical_mechanics_address)
from golem.assembly.mechanics.coalgebra import _payload_interface_transfers_by_case
from golem.assembly.mechanics.compile import (_embedded_channel_mechanics, _solve_element_mechanics)
from golem.assembly.mechanics.types import _RigidPayloadTransfer
from golem.assembly.obstructions import (EmptyPhysicalSolidDomainObstruction, RemainingLigamentViolationObstruction, UnsupportedAssemblyInterfaceObstruction)
from golem.assembly.service import JointLaw, PhysicalEvidenceStatus, ServiceIntent
from golem.assembly.thermal import _solve_element_thermal
from golem.assembly.thermal_types import _SolvedElementThermal
from golem.assembly.voxel import (_ResolvedMaterialDomain, _derive_resolved_material_domain, _unperforated_material_masks)
from golem.kernel.anatomy import (AcceptedPhysicalHydraulics, AcceptedVasculature, ClosedVascularGraph, PhysicalHydraulicsResult, derive_vascular_material_masks)
from golem.kernel.mechanics import (AcceptedEmbeddedChannelMechanics, AcceptedMechanics, MechanicsCriteria, NodalForce)

def _accepted_vascular_sections(
    vascular_elements: tuple[_CompiledElement, ...],
    service: ServiceIntent,
) -> (
    tuple[_ResolvedVascularSection | _EmbeddedVascularSection, ...]
    | RejectedAssembly
):
    """Derive material masks and resolve each element's physical channel section."""
    wall_thickness_world = (
        service.manufacturing.minimum_wall_thickness_metres
        / service.scale.metres_per_world_unit
    )
    minimum_cells_across_radius = (
        service.evidence_policy.minimum_cells_across_lumen / 2.0
    )
    material_results = tuple(
        (
            element,
            derive_vascular_material_masks(
                element.evaluated,
                element.mounted_vasculature,
                wall_thickness=wall_thickness_world,
                minimum_cells_across_radius=minimum_cells_across_radius,
            ),
        )
        for element in vascular_elements
        if element.mounted_vasculature is not None
    )
    physical_sections = tuple(
        _vascular_physical_section(
            element,
            result,
            service.evidence_policy.channel_resolution_policy,
        )
        for element, result in material_results
    )
    return collect_results(physical_sections)


def _ligament_violation_obstructions(
    accepted_sections: tuple[
        _ResolvedVascularSection | _EmbeddedVascularSection, ...
    ],
    service: ServiceIntent,
) -> tuple[RemainingLigamentViolationObstruction, ...]:
    return tuple(
        RemainingLigamentViolationObstruction(
            section.element.element_id,
            ligament,
            service.manufacturing.minimum_remaining_ligament_metres,
        )
        for section in accepted_sections
        for ligament in (
            _section_minimum_ligament_metres(section, service),
        )
        if ligament < service.manufacturing.minimum_remaining_ligament_metres
    )


def _physical_hydraulic_results(
    vascular_elements: tuple[_CompiledElement, ...],
    service: ServiceIntent,
) -> tuple[PhysicalHydraulicsResult | RejectedAssembly, ...]:
    return (
        tuple(
            _solve_element_physical_hydraulics(
                element.mounted_vasculature, service
            )
            for element in vascular_elements
            if element.mounted_vasculature is not None
        )
        if service.coolant is not None
        else ()
    )


def _deformable_creature_element(
    compiled: tuple[_CompiledElement, ...],
) -> _CompiledElement | RejectedAssembly:
    creature_elements = tuple(
        element
        for element in compiled
        if element.role is ElementRole.CREATURE
    )
    return (
        creature_elements[0]
        if len(creature_elements) == 1
        else RejectedAssembly(
            (
                UnsupportedAssemblyInterfaceObstruction(
                    tuple(element.element_id for element in compiled),
                    (
                        "this physical cut requires exactly one deformable "
                        "creature body; equipment descends only through "
                        "explicit rigid_payload interfaces"
                    ),
                ),
            )
        )
    )


def _rigid_payload_joint_obstructions(
    service: ServiceIntent,
) -> tuple[UnsupportedAssemblyInterfaceObstruction, ...]:
    return tuple(
        UnsupportedAssemblyInterfaceObstruction(
            (getattr(joint.address, "element_id", "<unknown>"),),
            f"joint law {joint.law.value!r} is not a rigid payload transfer",
        )
        for service_case in service.cases
        for joint in service_case.joints
        if joint.law is not JointLaw.RIGID_PAYLOAD
    )


def _resolved_material_masks(
    section: _VascularPhysicalSection | None,
    element: _CompiledElement,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    return (
        (
            section.material.masks.load_bearing_solid,
            section.material.masks.lumen,
            section.material.masks.wall,
        )
        if isinstance(section, _ResolvedVascularSection)
        else _unperforated_material_masks(element.evaluated)
    )


def _coupled_case_mechanics(
    element: _CompiledElement,
    domain_result: _ResolvedMaterialDomain,
    service: ServiceIntent,
    thermal_by_case: dict[str, _SolvedElementThermal],
    graph: ClosedVascularGraph | None,
    hydraulics: AcceptedPhysicalHydraulics | None,
    payload_transfers_by_case: tuple[
        tuple[str, tuple[_RigidPayloadTransfer, ...]], ...
    ],
    embedded_result: AcceptedEmbeddedChannelMechanics | None,
    include_linearized_buckling_evidence: bool,
) -> tuple[AcceptedMechanics | RejectedAssembly, ...]:
    return tuple(
        _solve_element_mechanics(
            element,
            domain_result,
            service,
            service_case,
            case_index,
            thermal_by_case.get(service_case.case_id),
            graph,
            hydraulics,
            MechanicsCriteria(
                maximum_displacement=(
                    service.limits.maximum_displacement_metres
                ),
                minimum_yield_safety_factor=(
                    service.limits.minimum_yield_safety_factor
                ),
                require_linearized_buckling=(
                    include_linearized_buckling_evidence
                ),
                minimum_linearized_buckling_load_factor=(
                    service.limits.minimum_buckling_safety_factor
                    if include_linearized_buckling_evidence
                    else None
                ),
            ),
            tuple(
                force
                for case_id, transfers in payload_transfers_by_case
                if case_id == service_case.case_id
                for transfer in transfers
                for force in transfer.nodal_forces
            ),
            embedded_result,
        )
        for case_index, service_case in enumerate(service.cases)
    )


def _coupled_mechanics_receipts(
    element: _CompiledElement,
    domain_result: _ResolvedMaterialDomain,
    service: ServiceIntent,
    accepted_mechanics: tuple[AcceptedMechanics, ...],
    embedded_result: AcceptedEmbeddedChannelMechanics | None,
) -> tuple[ElementMechanicsReceipt, ...]:
    return tuple(
        ElementMechanicsReceipt(
            element_id=element.element_id,
            case_id=service_case.case_id,
            mechanics=result.receipt,
            minimum_burst_safety_factor=_burst_safety_factor(
                element,
                domain_result,
                service,
                result,
                embedded_result,
            ),
            critical_semantic_address=_critical_mechanics_address(
                element, domain_result, result
            ),
            embedded_channel_mechanics=embedded_result,
        )
        for service_case, result in zip(
            service.cases, accepted_mechanics, strict=True
        )
    )


def _coupled_not_evaluated_kinds(
    equipment_elements: tuple[_CompiledElement, ...],
    section: _VascularPhysicalSection | None,
) -> tuple[CoupledUnevaluatedPhysics, ...]:
    return tuple(
        dict.fromkeys(
            (
                *(
                    (
                        CoupledUnevaluatedPhysics.EQUIPMENT_LOCAL_THERMAL_RESPONSE,
                        CoupledUnevaluatedPhysics.EQUIPMENT_LOCAL_DEFORMATION,
                        CoupledUnevaluatedPhysics.EQUIPMENT_LOCAL_STRESS,
                        CoupledUnevaluatedPhysics.EQUIPMENT_CONTACT_STRESS,
                    )
                    if equipment_elements
                    else ()
                ),
                *(
                    (
                        CoupledUnevaluatedPhysics.SUBGRID_CHANNEL_GLOBAL_CONSTITUTIVE_RESPONSE,
                    )
                    if isinstance(section, _EmbeddedVascularSection)
                    else ()
                ),
            )
        )
    )


def _solve_resolved_coupled_assembly_impl(
    compiled: tuple[_CompiledElement, ...],
    analysis_compiled: tuple[_CompiledElement, ...],
    service: ServiceIntent,
    physical_sections: tuple[_VascularPhysicalSection, ...],
    hydraulic_results: tuple[AcceptedPhysicalHydraulics, ...],
    vascular_sizing_receipts: tuple[VascularThermalSizingReceipt, ...] = (),
    *,
    include_linearized_buckling_evidence: bool = True,
) -> AssemblyResult:
    """Glue one deformable body plus explicitly declared rigid payloads."""
    element = _deformable_creature_element(analysis_compiled)
    if isinstance(element, RejectedAssembly):
        return element
    equipment_elements = tuple(
        visual_element
        for visual_element in compiled
        if visual_element.role is ElementRole.EQUIPMENT
    )
    unsupported_joints = _rigid_payload_joint_obstructions(service)
    if unsupported_joints:
        return RejectedAssembly(unsupported_joints)
    section = next(
        (
            candidate
            for candidate in physical_sections
            if candidate.element.element_id == element.element_id
        ),
        None,
    )
    load_bearing, lumen, wall = _resolved_material_masks(section, element)
    domain_result = _derive_resolved_material_domain(
        element,
        load_bearing,
        lumen,
        wall,
        service.scale.metres_per_world_unit,
    )
    if isinstance(domain_result, EmptyPhysicalSolidDomainObstruction):
        return RejectedAssembly((domain_result,))
    graph = element.mounted_vasculature
    hydraulics_by_element = {
        local_section.element.element_id: hydraulics
        for local_section, hydraulics in zip(
            physical_sections, hydraulic_results, strict=True
        )
    }
    hydraulics = hydraulics_by_element.get(element.element_id)
    embedded_result = _embedded_channel_mechanics(
        section, domain_result, service, hydraulics
    )
    if isinstance(embedded_result, RejectedAssembly):
        return embedded_result
    payload_transfers_by_case = _payload_interface_transfers_by_case(
        element,
        equipment_elements,
        domain_result,
        service,
    )
    if isinstance(payload_transfers_by_case, RejectedAssembly):
        return payload_transfers_by_case
    thermal_results = tuple(
        _solve_element_thermal(
            element,
            domain_result,
            service,
            service_case,
            graph,
            hydraulics,
        )
        for service_case in service.cases
    )
    thermal_descent = collect_results(thermal_results)
    if isinstance(thermal_descent, RejectedAssembly):
        return thermal_descent
    solved_thermal = thermal_descent
    thermal_by_case = {
        result.receipt.case_id: result for result in solved_thermal
    }
    mechanics_results = _coupled_case_mechanics(
        element,
        domain_result,
        service,
        thermal_by_case,
        graph,
        hydraulics,
        payload_transfers_by_case,
        embedded_result,
        include_linearized_buckling_evidence,
    )
    mechanics_descent = collect_results(mechanics_results)
    if isinstance(mechanics_descent, RejectedAssembly):
        return mechanics_descent
    accepted_mechanics = mechanics_descent
    volume_receipt = _section_material_volume_receipt(
        element,
        section,
        domain_result,
        embedded_result,
        load_bearing,
        lumen,
        wall,
        service,
    )
    interface_receipts = tuple(
        transfer.receipt
        for transfer in _chain.from_iterable(
            transfers for _case_id, transfers in payload_transfers_by_case
        )
    )
    return AcceptedAssembly(
        records=tuple(
            _chain.from_iterable(
                compiled_element.records for compiled_element in compiled
            )
        ),
        service=service,
        receipt=CoupledPerformanceReceipt(
            physical_evidence=PhysicalEvidenceStatus.REQUESTED,
            metres_per_world_unit=service.scale.metres_per_world_unit,
            service_case_ids=tuple(case.case_id for case in service.cases),
            material_ids=tuple(
                dict.fromkeys(
                    assignment.material.material_id.value
                    for assignment in service.solid_materials
                )
            ),
            material_volumes=(volume_receipt,),
            vascular_sizing_receipts=vascular_sizing_receipts,
            hydraulic_receipts=tuple(
                result.receipt for result in hydraulic_results
            ),
            thermal_receipts=tuple(
                result.receipt for result in solved_thermal
            ),
            mechanics_receipts=_coupled_mechanics_receipts(
                element,
                domain_result,
                service,
                accepted_mechanics,
                embedded_result,
            ),
            interface_receipts=interface_receipts,
            refinement_receipts=(),
            not_evaluated=_coupled_not_evaluated_kinds(
                equipment_elements, section
            ),
            fixed_point_iterations=max(
                (result.coupling_iterations for result in solved_thermal),
                default=1,
            ),
            maximum_state_delta=max(
                (result.maximum_state_delta for result in solved_thermal),
                default=0.0,
            ),
        ),
        vasculature=tuple(
            compiled_element.vasculature
            for compiled_element in compiled
            if isinstance(compiled_element.vasculature, AcceptedVasculature)
        ),
        hydraulics=hydraulic_results,
        thermal_solutions=tuple(
            result.solution for result in solved_thermal
        ),
        mechanics_solutions=accepted_mechanics,
    )
