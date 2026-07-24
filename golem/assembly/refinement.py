"""Resolution-refinement receipt comparison and gluing algebra."""

from __future__ import annotations

import math

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from golem.assembly.carriers import (AcceptedAssembly, AssemblyResult, CoupledPerformanceReceipt, ElementMechanicsReceipt, ElementThermalReceipt, RejectedAssembly, ResolutionRefinementReceipt, _CompiledElement)
from golem.assembly.obstructions import ResolutionRefinementObstruction
from golem.assembly.service import PhysicsEvidencePolicy, ServiceIntent

if TYPE_CHECKING:
    from golem.assembly.core import CoupledStages

def _optional_yield_utilization_change(
    base_safety_factor: float | None,
    refined_safety_factor: float | None,
    required_safety_factor: float,
) -> float | None:
    return (
        abs(
            required_safety_factor / base_safety_factor
            - required_safety_factor / refined_safety_factor
        )
        if base_safety_factor is not None
        and refined_safety_factor is not None
        and math.isfinite(base_safety_factor)
        and math.isfinite(refined_safety_factor)
        and base_safety_factor > 0.0
        and refined_safety_factor > 0.0
        else None
        if base_safety_factor is None and refined_safety_factor is None
        else 0.0
        if base_safety_factor == refined_safety_factor == math.inf
        else math.inf
    )


def _receipts_by_case_key[R](
    receipts: tuple[R, ...],
) -> dict[tuple[str, str], R]:
    """Index element/case receipts by their ``(element_id, case_id)`` key."""
    return {
        (receipt.element_id, receipt.case_id): receipt for receipt in receipts
    }


def _local_refinement_receipt(
    key: tuple[str, str],
    base_mechanics: dict[tuple[str, str], ElementMechanicsReceipt],
    refined_mechanics: dict[tuple[str, str], ElementMechanicsReceipt],
    base_thermal: dict[tuple[str, str], ElementThermalReceipt],
    refined_thermal: dict[tuple[str, str], ElementThermalReceipt],
    policy: PhysicsEvidencePolicy,
    refined_resolution: int,
    temperature_normalization: float,
    service: ServiceIntent,
    normalized_refinement_change: Callable[[float, float, float], float],
) -> ResolutionRefinementReceipt | ResolutionRefinementObstruction:
    element_id, case_id = key
    local_base_mechanics = base_mechanics.get(key)
    local_refined_mechanics = refined_mechanics.get(key)
    local_base_thermal = base_thermal.get(key)
    local_refined_thermal = refined_thermal.get(key)
    if local_base_mechanics is None or local_refined_mechanics is None:
        return ResolutionRefinementObstruction(
            element_id,
            case_id,
            "missing_mechanics_receipt",
            math.inf,
            policy.maximum_refinement_change_fraction,
        )
    temperature_change = (
        normalized_refinement_change(
            local_base_thermal.maximum_temperature_kelvin,
            local_refined_thermal.maximum_temperature_kelvin,
            temperature_normalization,
        )
        if local_base_thermal is not None
        and local_refined_thermal is not None
        else None
        if local_base_thermal is None and local_refined_thermal is None
        else math.inf
    )
    displacement_change = normalized_refinement_change(
        local_base_mechanics.mechanics.maximum_displacement,
        local_refined_mechanics.mechanics.maximum_displacement,
        service.limits.maximum_displacement_metres,
    )
    yield_change = _optional_yield_utilization_change(
        local_base_mechanics.mechanics.minimum_yield_safety_factor,
        local_refined_mechanics.mechanics.minimum_yield_safety_factor,
        service.limits.minimum_yield_safety_factor,
    )
    changes = tuple(
        (quantity, change)
        for quantity, change in (
            ("maximum_temperature", temperature_change),
            ("maximum_displacement", displacement_change),
            ("minimum_yield_safety_factor", yield_change),
        )
        if change is not None
    )
    exceeded = next(
        (
            (quantity, change)
            for quantity, change in changes
            if change > policy.maximum_refinement_change_fraction
        ),
        None,
    )
    return (
        ResolutionRefinementObstruction(
            element_id,
            case_id,
            exceeded[0],
            exceeded[1],
            policy.maximum_refinement_change_fraction,
        )
        if exceeded is not None
        else ResolutionRefinementReceipt(
            element_id=element_id,
            case_id=case_id,
            base_resolution=policy.physics_resolution,
            refined_resolution=refined_resolution,
            maximum_temperature_utilization_change=temperature_change,
            maximum_displacement_utilization_change=displacement_change,
            minimum_yield_utilization_change=yield_change,
        )
    )


def _attach_resolution_refinement_evidence_impl(
    compiled: tuple[_CompiledElement, ...],
    service: ServiceIntent,
    base: AcceptedAssembly,
    *,
    stages: CoupledStages,
    normalized_refinement_change: Callable[[float, float, float], float],
) -> AssemblyResult:
    policy = service.evidence_policy
    refined_resolution = (
        policy.physics_resolution * policy.resolution_refinement_factor
    )
    refined_service = replace(
        service,
        evidence_policy=replace(
            policy,
            physics_resolution=refined_resolution,
            resolution_refinement_factor=1,
        ),
    )
    if not isinstance(base.receipt, CoupledPerformanceReceipt):
        return RejectedAssembly(
            (
                ResolutionRefinementObstruction(
                    "<assembly>",
                    "<all>",
                    "missing_coupled_receipt",
                    math.inf,
                    policy.maximum_refinement_change_fraction,
                ),
            )
        )
    refined_analysis = stages.evaluate(compiled, refined_resolution)
    if isinstance(refined_analysis, RejectedAssembly):
        return refined_analysis
    refined = stages.solve(
        compiled,
        refined_analysis,
        refined_service,
        base.receipt.vascular_sizing_receipts,
        include_linearized_buckling_evidence=False,
    )
    if not isinstance(refined, AcceptedAssembly):
        return refined
    if not isinstance(refined.receipt, CoupledPerformanceReceipt):
        return RejectedAssembly(
            (
                ResolutionRefinementObstruction(
                    "<assembly>",
                    "<all>",
                    "missing_coupled_receipt",
                    math.inf,
                    policy.maximum_refinement_change_fraction,
                ),
            )
        )
    base_thermal = _receipts_by_case_key(base.receipt.thermal_receipts)
    refined_thermal = _receipts_by_case_key(refined.receipt.thermal_receipts)
    base_mechanics = _receipts_by_case_key(base.receipt.mechanics_receipts)
    refined_mechanics = _receipts_by_case_key(refined.receipt.mechanics_receipts)
    temperature_reference = min(
        (
            *(
                (service.coolant.inlet_temperature_kelvin,)
                if service.coolant is not None
                else ()
            ),
            *(
                (service.ambient.temperature_kelvin,)
                if service.ambient is not None
                else ()
            ),
        ),
        default=0.0,
    )
    temperature_normalization = (
        service.limits.maximum_temperature_kelvin - temperature_reference
    )
    keys = tuple(sorted(frozenset(base_mechanics) | frozenset(refined_mechanics)))

    local_results = tuple(
        _local_refinement_receipt(
            key,
            base_mechanics=base_mechanics,
            refined_mechanics=refined_mechanics,
            base_thermal=base_thermal,
            refined_thermal=refined_thermal,
            policy=policy,
            refined_resolution=refined_resolution,
            temperature_normalization=temperature_normalization,
            service=service,
            normalized_refinement_change=normalized_refinement_change,
        )
        for key in keys
    )
    local_obstructions = tuple(
        result
        for result in local_results
        if isinstance(result, ResolutionRefinementObstruction)
    )
    limitation_obstructions = (
        ()
        if base.receipt.not_evaluated == refined.receipt.not_evaluated
        else (
            ResolutionRefinementObstruction(
                "<assembly>",
                "<all>",
                "not_evaluated_set_changed",
                math.inf,
                policy.maximum_refinement_change_fraction,
            ),
        )
    )
    return (
        RejectedAssembly((*local_obstructions, *limitation_obstructions))
        if local_obstructions or limitation_obstructions
        else replace(
            base,
            receipt=replace(
                base.receipt,
                refinement_receipts=tuple(
                    result
                    for result in local_results
                    if isinstance(result, ResolutionRefinementReceipt)
                ),
            ),
        )
    )

