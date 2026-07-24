"""Solid-material and constitutive-model section decoders."""

from __future__ import annotations

from golem.materials import decode_isotropic_solid

from golem.assembly.service.address import SemanticAddress
from golem.assembly.service.model import (
    SIScale,
    SolidConstitutiveModel,
    SolidMaterialAssignment,
    SolidMaterialRole,
)
from golem.assembly.service.outcome import (
    DuplicateMaterialAssignmentObstruction,
    MalformedServiceIntentObstruction,
    UnsupportedConstitutiveModelObstruction,
)
from golem.assembly.service.primitives import (
    _DOMAIN_ADDRESS_KINDS,
    _DOMAIN_ADDRESS_TYPES,
    _Decoded,
    _DecodeResult,
    _closed_value,
    _decode_bound_address,
    _decode_material,
    _finite_number,
    _record,
    _result_obstructions,
    _traverse_array,
)


def _decode_si_scale(value: object) -> _DecodeResult[SIScale]:
    number = _finite_number(value)
    return (
        _Decoded(SIScale(number))
        if number is not None and number > 0.0
        else (
            MalformedServiceIntentObstruction(
                "/service/metres_per_world_unit",
                "expected a finite positive SI scale",
            ),
        )
    )


def _decode_constitutive_model(
    value: object,
) -> _DecodeResult[SolidConstitutiveModel]:
    return (
        _Decoded(SolidConstitutiveModel.ISOTROPIC_LINEAR_THERMOELASTIC)
        if value == SolidConstitutiveModel.ISOTROPIC_LINEAR_THERMOELASTIC.value
        else (UnsupportedConstitutiveModelObstruction(value),)
    )


def _decode_solid_materials(
    value: object, known_addresses: frozenset[SemanticAddress]
) -> _DecodeResult[tuple[SolidMaterialAssignment, ...]]:
    decoded = _traverse_array(
        value,
        "/service/solid_materials",
        lambda item, index: _decode_solid_material(
            item, index, known_addresses
        ),
        require_nonempty=True,
    )
    if not isinstance(decoded, _Decoded):
        return decoded
    assignments = decoded.value
    duplicates = tuple(
        DuplicateMaterialAssignmentObstruction(left.role, left.domain)
        for left_index, left in enumerate(assignments)
        for right in assignments[left_index + 1 :]
        if (left.role, left.domain) == (right.role, right.domain)
    )
    return duplicates if duplicates else _Decoded(assignments)


def _decode_solid_material(
    value: object,
    index: int,
    known_addresses: frozenset[SemanticAddress],
) -> _DecodeResult[SolidMaterialAssignment]:
    path = f"/service/solid_materials/{index}"
    obj = _record(value, path, frozenset(("role", "domain", "material")))
    if not isinstance(obj, _Decoded):
        return obj
    role = _closed_value(obj.value["role"], SolidMaterialRole, f"{path}/role")
    domain = _decode_bound_address(
        obj.value["domain"],
        f"{path}/domain",
        known_addresses,
        _DOMAIN_ADDRESS_TYPES,
        _DOMAIN_ADDRESS_KINDS,
    )
    material = _decode_material(
        decode_isotropic_solid, obj.value["material"], f"{path}/material"
    )
    obstructions = _result_obstructions((role, domain, material))
    return (
        obstructions
        if obstructions
        else _Decoded(
            SolidMaterialAssignment(role.value, domain.value, material.value)
        )
    )
