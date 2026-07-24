"""Root service-intent gate -- section fan-out and cross-section glue."""

from __future__ import annotations

from collections.abc import Collection

from golem.assembly.service.address import SemanticAddress
from golem.assembly.service.model import ServiceIntent, VisualOnlyServiceIntent
from golem.assembly.service.outcome import (
    AcceptedServiceIntent,
    MalformedServiceIntentObstruction,
    MaterialValidityObstruction,
    MissingHeatSinkObstruction,
    MissingUnitScaleObstruction,
    RejectedServiceIntent,
    ServiceIntentObstruction,
    ServiceIntentResult,
)
from golem.assembly.service.environment import _decode_ambient, _decode_coolant
from golem.assembly.service.limits import (
    _decode_evidence_policy,
    _decode_limits,
    _decode_manufacturing,
)
from golem.assembly.service.material import (
    _decode_constitutive_model,
    _decode_si_scale,
    _decode_solid_materials,
)
from golem.assembly.service.mechanical import _decode_cases
from golem.assembly.service.primitives import _Decoded, _result_obstructions


def decode_service_intent(
    payload: object,
    known_addresses: Collection[SemanticAddress] = frozenset(),
) -> ServiceIntentResult:
    """Decode ``spec.get('service')`` against assembly-owned addresses.

    ``None`` is an accepted visual-only artifact.  Any physical block fails
    closed when the address cover is absent or a physical material is unknown.
    """
    if payload is None:
        return AcceptedServiceIntent(VisualOnlyServiceIntent())
    if not isinstance(payload, dict):
        return RejectedServiceIntent(
            (MalformedServiceIntentObstruction("/service", "expected an object"),)
        )
    required_keys = frozenset(
        (
            "constitutive_model",
            "solid_materials",
            "cases",
            "ambient",
            "coolant",
            "manufacturing",
            "limits",
            "evidence",
        )
    )
    structural_obstructions: tuple[ServiceIntentObstruction, ...] = (
        (() if "metres_per_world_unit" in payload else (MissingUnitScaleObstruction(),))
        + tuple(
            MalformedServiceIntentObstruction(
                f"/service/{key}", "missing required field"
            )
            for key in sorted(required_keys - frozenset(payload))
        )
        + tuple(
            MalformedServiceIntentObstruction(
                f"/service/{key}", "unexpected field"
            )
            for key in sorted(
                (
                    frozenset(payload)
                    - required_keys
                    - frozenset(("metres_per_world_unit",))
                ),
                key=str,
            )
        )
    )
    if structural_obstructions:
        return RejectedServiceIntent(structural_obstructions)
    address_cover = frozenset(known_addresses)
    scale = _decode_si_scale(payload["metres_per_world_unit"])
    constitutive_model = _decode_constitutive_model(payload["constitutive_model"])
    solid_materials = _decode_solid_materials(
        payload["solid_materials"], address_cover
    )
    cases = _decode_cases(payload["cases"], address_cover)
    ambient = _decode_ambient(payload["ambient"])
    coolant = _decode_coolant(payload["coolant"])
    manufacturing = _decode_manufacturing(payload["manufacturing"])
    limits = _decode_limits(payload["limits"])
    evidence_policy = _decode_evidence_policy(payload["evidence"])
    results = (
        scale,
        constitutive_model,
        solid_materials,
        cases,
        ambient,
        coolant,
        manufacturing,
        limits,
        evidence_policy,
    )
    local_obstructions = _result_obstructions(results)
    if local_obstructions:
        return RejectedServiceIntent(local_obstructions)
    if not all(map(lambda result: isinstance(result, _Decoded), results)):
        return RejectedServiceIntent(
            (
                MalformedServiceIntentObstruction(
                    "/service", "internal decoder failed to produce a section"
                ),
            )
        )
    accepted_scale = scale.value
    accepted_model = constitutive_model.value
    accepted_materials = solid_materials.value
    accepted_cases = cases.value
    accepted_ambient = ambient.value
    accepted_coolant = coolant.value
    accepted_manufacturing = manufacturing.value
    accepted_limits = limits.value
    accepted_evidence = evidence_policy.value
    cross_obstructions: tuple[ServiceIntentObstruction, ...] = (
        *tuple(
            MissingHeatSinkObstruction(case.case_id)
            for case in accepted_cases
            if case.heat_sources
            and accepted_ambient is None
            and accepted_coolant is None
        ),
        *tuple(
            MaterialValidityObstruction(
                "/service/limits/maximum_temperature_k",
                assignment.material.material_id.value,
                accepted_limits.maximum_temperature_kelvin,
                assignment.material.operating_temperature.minimum_kelvin,
                assignment.material.operating_temperature.maximum_kelvin,
            )
            for assignment in accepted_materials
            if not assignment.material.operating_temperature.contains(
                accepted_limits.maximum_temperature_kelvin
            )
        ),
    )
    return (
        RejectedServiceIntent(cross_obstructions)
        if cross_obstructions
        else AcceptedServiceIntent(
            ServiceIntent(
                scale=accepted_scale,
                constitutive_model=accepted_model,
                solid_materials=accepted_materials,
                cases=accepted_cases,
                ambient=accepted_ambient,
                coolant=accepted_coolant,
                manufacturing=accepted_manufacturing,
                limits=accepted_limits,
                evidence_policy=accepted_evidence,
            )
        )
    )
