"""Mechanics assembly receipt and acceptance algebra."""

from __future__ import annotations

import math

from dataclasses import replace

from golem.assembly.address import _semantic_address_for_cell
from golem.assembly.carriers import RejectedAssembly, _CompiledElement
from golem.assembly.materials_binding import _element_material_assignment
from golem.assembly.obstructions import InsufficientBurstMarginObstruction
from golem.assembly.service import ServiceIntent, SolidMaterialRole
from golem.assembly.voxel import _ResolvedMaterialDomain
from golem.kernel.mechanics import AcceptedEmbeddedChannelMechanics, AcceptedMechanics

def _finalize_element_mechanics(
    element: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    service: ServiceIntent,
    mechanics_result: AcceptedMechanics,
    embedded_channel_mechanics: AcceptedEmbeddedChannelMechanics | None,
) -> AcceptedMechanics | RejectedAssembly:
    mechanics_result = (
        replace(
            mechanics_result,
            receipt=replace(
                mechanics_result.receipt,
                not_evaluated=tuple(
                    dict.fromkeys(
                        (
                            *mechanics_result.receipt.not_evaluated,
                            *embedded_channel_mechanics.receipt.not_evaluated,
                        )
                    )
                ),
            ),
        )
        if isinstance(
            embedded_channel_mechanics, AcceptedEmbeddedChannelMechanics
        )
        else mechanics_result
    )
    burst_obstruction = _burst_margin_obstruction(
        element,
        domain,
        service,
        mechanics_result,
        embedded_channel_mechanics,
    )
    return (
        RejectedAssembly((burst_obstruction,))
        if burst_obstruction is not None
        else mechanics_result
    )


def _burst_margin_obstruction(
    element: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    service: ServiceIntent,
    mechanics: AcceptedMechanics,
    embedded_channel_mechanics: AcceptedEmbeddedChannelMechanics | None = None,
) -> InsufficientBurstMarginObstruction | None:
    actual = _burst_safety_factor(
        element,
        domain,
        service,
        mechanics,
        embedded_channel_mechanics,
    )
    return (
        InsufficientBurstMarginObstruction(
            element.element_id,
            0.0 if actual is None else actual,
            service.limits.minimum_burst_safety_factor,
        )
        if (domain.wall_cells or embedded_channel_mechanics is not None)
        and (actual is None or actual < service.limits.minimum_burst_safety_factor)
        else None
    )


def _burst_safety_factor(
    element: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    service: ServiceIntent,
    mechanics: AcceptedMechanics,
    embedded_channel_mechanics: AcceptedEmbeddedChannelMechanics | None = None,
) -> float | None:
    if isinstance(
        embedded_channel_mechanics, AcceptedEmbeddedChannelMechanics
    ):
        return embedded_channel_mechanics.receipt.minimum_burst_safety_factor
    if not domain.wall_cells:
        return None
    wall_stresses = tuple(
        cell.von_mises_stress
        for cell in mechanics.cells
        if cell.cell in domain.wall_cells
    )
    wall_assignment = _element_material_assignment(
        service, element.element_id, (SolidMaterialRole.LUMEN_WALL,)
    )
    if not wall_stresses or wall_assignment is None:
        return None
    maximum_stress = max(wall_stresses)
    return (
        wall_assignment.material.allowable_stress_pascal / maximum_stress
        if maximum_stress > 0.0
        else math.inf
    )


def _critical_mechanics_address(
    element: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    mechanics: AcceptedMechanics,
) -> str:
    critical = max(
        mechanics.cells,
        key=lambda cell: (cell.von_mises_stress, cell.cell),
    )
    return _semantic_address_for_cell(element, domain, critical.cell)
