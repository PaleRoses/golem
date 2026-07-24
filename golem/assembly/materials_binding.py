"""Solid and vascular material binding, ligament checks, and volume receipts."""

from __future__ import annotations

import math

import numpy as np

from golem.assembly.carriers import (
    ElementMaterialVolumeReceipt,
    ElementRole,
    _CompiledElement,
)
from golem.assembly.hydraulics import (
    _ResolvedVascularSection,
    _VascularPhysicalSection,
)
from golem.assembly.obstructions import (
    MissingSolidMaterialAssignmentObstruction,
    MissingVascularWallMaterialObstruction,
)
from golem.assembly.service import (
    ElementAddress,
    ServiceIntent,
    SolidMaterialAssignment,
    SolidMaterialRole,
)
from golem.assembly.voxel import _ResolvedMaterialDomain
from golem.kernel.anatomy import AcceptedVasculature
from golem.kernel.mechanics import AcceptedEmbeddedChannelMechanics, CellIndex
from golem.materials.core import IsotropicSolid


_CREATURE_SOLID_ROLES = (SolidMaterialRole.STRUCTURE,)
_EQUIPMENT_SOLID_ROLES = (
    SolidMaterialRole.ARMOR,
    SolidMaterialRole.EQUIPMENT,
)
_BULK_SOLID_ROLES = (*_CREATURE_SOLID_ROLES, *_EQUIPMENT_SOLID_ROLES)
_LUMEN_WALL_ROLES = (SolidMaterialRole.LUMEN_WALL,)


def _section_minimum_ligament_metres(
    section: _VascularPhysicalSection, service: ServiceIntent
) -> float:
    if isinstance(section, _ResolvedVascularSection):
        masks = section.material.masks
        return _minimum_remaining_ligament_metres(
            masks.load_bearing_solid,
            masks.lumen,
            section.element.evaluated.world_pitch,
            service.scale.metres_per_world_unit,
        )
    vasculature = section.element.vasculature
    return (
        vasculature.receipt.minimum_capsule_margin
        * service.scale.metres_per_world_unit
        if isinstance(vasculature, AcceptedVasculature)
        else 0.0
    )


def _minimum_remaining_ligament_metres(
    load_bearing_solid: np.ndarray,
    lumen: np.ndarray,
    world_pitch: tuple[float, float, float],
    metres_per_world_unit: float,
) -> float:
    from scipy import ndimage

    body = load_bearing_solid | lumen
    sampling = tuple(pitch * metres_per_world_unit for pitch in world_pitch)
    body_distance = ndimage.distance_transform_edt(body, sampling=sampling)
    lumen_distance = ndimage.distance_transform_edt(lumen, sampling=sampling)
    ligament = body_distance[lumen] - lumen_distance[lumen]
    return max(float(np.min(ligament)), 0.0) if ligament.size else math.inf


def _material_binding_obstructions(
    compiled: tuple[_CompiledElement, ...],
    vascular_elements: tuple[_CompiledElement, ...],
    service: ServiceIntent,
) -> tuple[
    MissingSolidMaterialAssignmentObstruction
    | MissingVascularWallMaterialObstruction,
    ...,
]:
    expected_roles_by_element = tuple(
        (
            element.element_id,
            _CREATURE_SOLID_ROLES
            if element.role is ElementRole.CREATURE
            else _EQUIPMENT_SOLID_ROLES,
        )
        for element in compiled
        if element.records
    )
    missing_solids = tuple(
        MissingSolidMaterialAssignmentObstruction(element_id, expected_roles)
        for element_id, expected_roles in expected_roles_by_element
        if _element_material_assignment(service, element_id, expected_roles) is None
    )
    missing_walls = tuple(
        MissingVascularWallMaterialObstruction(element.element_id)
        for element in vascular_elements
        if _element_material_assignment(
            service,
            element.element_id,
            _LUMEN_WALL_ROLES,
        ) is None
    )
    return (*missing_solids, *missing_walls)


def _element_material_assignment(
    service: ServiceIntent,
    element_id: str,
    roles: tuple[SolidMaterialRole, ...],
) -> SolidMaterialAssignment | None:
    return next(
        (
            assignment
            for assignment in service.solid_materials
            if isinstance(assignment.domain, ElementAddress)
            and assignment.domain.element_id == element_id
            and assignment.role in roles
        ),
        None,
    )


def _solid_material_for_cell(
    service: ServiceIntent,
    element_id: str,
    cell: CellIndex,
    wall_cells: frozenset[CellIndex],
) -> IsotropicSolid | None:
    wall_assignment = _element_material_assignment(
        service, element_id, _LUMEN_WALL_ROLES
    )
    bulk_assignment = _element_material_assignment(
        service,
        element_id,
        _BULK_SOLID_ROLES,
    )
    assignment = (
        wall_assignment
        if cell in wall_cells and wall_assignment is not None
        else bulk_assignment
    )
    return assignment.material if assignment is not None else None


def _section_material_volume_receipt(
    element: _CompiledElement,
    section: _VascularPhysicalSection | None,
    domain: _ResolvedMaterialDomain,
    embedded_result: AcceptedEmbeddedChannelMechanics | None,
    load_bearing: np.ndarray,
    lumen: np.ndarray,
    wall: np.ndarray,
    service: ServiceIntent,
) -> ElementMaterialVolumeReceipt:
    if isinstance(embedded_result, AcceptedEmbeddedChannelMechanics):
        receipt = embedded_result.receipt
        return ElementMaterialVolumeReceipt(
            element_id=element.element_id,
            solid_fraction=receipt.conservative_solid_fraction_lower_bound,
            lumen_fraction=receipt.lumen_volume_fraction_upper_bound,
            wall_fraction=(
                receipt.wall_volume_upper_bound_cubic_metres
                / receipt.host_volume_cubic_metres
            ),
            minimum_ligament_metres=(
                _section_minimum_ligament_metres(section, service)
                if section is not None
                else None
            ),
        )
    return _material_volume_receipt(
        element.element_id,
        load_bearing,
        lumen,
        wall,
        (
            _minimum_remaining_ligament_metres(
                load_bearing,
                lumen,
                element.evaluated.world_pitch,
                service.scale.metres_per_world_unit,
            )
            if bool(np.any(lumen))
            else None
        ),
    )


def _material_volume_receipt(
    element_id: str,
    load_bearing: np.ndarray,
    lumen: np.ndarray,
    wall: np.ndarray,
    minimum_ligament_metres: float | None,
) -> ElementMaterialVolumeReceipt:
    body_count = int(np.count_nonzero(load_bearing | lumen))
    denominator = max(body_count, 1)
    return ElementMaterialVolumeReceipt(
        element_id=element_id,
        solid_fraction=float(np.count_nonzero(load_bearing) / denominator),
        lumen_fraction=float(np.count_nonzero(lumen) / denominator),
        wall_fraction=float(np.count_nonzero(wall) / denominator),
        minimum_ligament_metres=minimum_ligament_metres,
    )
