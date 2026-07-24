"""Pure lookup algebra over the authoritative material catalogue."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from golem.materials.core import (
    MATERIAL_CATALOGUE,
    AcceptedMaterial,
    AppearanceMaterial,
    CompressibleGas,
    IncompressibleFluid,
    IsotropicSolid,
    MaterialDecodeResult,
    MaterialKind,
    RejectedMaterial,
    UnknownMaterialObstruction,
    _NEUTRAL_GRAY,
)


class _CataloguedMaterial(Protocol):
    material_id: StrEnum


def _decode_material[material: _CataloguedMaterial](
    identifier: object,
    material_kind: MaterialKind,
    candidates: tuple[material, ...],
) -> MaterialDecodeResult[material]:
    accepted = next(
        (
            candidate
            for candidate in candidates
            if candidate.material_id.value == identifier
        ),
        None,
    )
    return (
        AcceptedMaterial(accepted)
        if accepted is not None
        else RejectedMaterial(
            (UnknownMaterialObstruction(material_kind, identifier),)
        )
    )


def decode_appearance_material(
    identifier: object,
) -> MaterialDecodeResult[AppearanceMaterial]:
    return _decode_material(
        identifier,
        MaterialKind.APPEARANCE,
        MATERIAL_CATALOGUE.appearance_materials,
    )


def resolve_appearance_material(identifier: object) -> AppearanceMaterial:
    result = decode_appearance_material(identifier)
    return result.material if isinstance(result, AcceptedMaterial) else _NEUTRAL_GRAY


def decode_isotropic_solid(
    identifier: object,
) -> MaterialDecodeResult[IsotropicSolid]:
    return _decode_material(
        identifier,
        MaterialKind.ISOTROPIC_SOLID,
        MATERIAL_CATALOGUE.isotropic_solids,
    )


def decode_incompressible_fluid(
    identifier: object,
) -> MaterialDecodeResult[IncompressibleFluid]:
    return _decode_material(
        identifier,
        MaterialKind.INCOMPRESSIBLE_FLUID,
        MATERIAL_CATALOGUE.incompressible_fluids,
    )


def decode_compressible_gas(
    identifier: object,
) -> MaterialDecodeResult[CompressibleGas]:
    return _decode_material(
        identifier,
        MaterialKind.COMPRESSIBLE_GAS,
        MATERIAL_CATALOGUE.compressible_gases,
    )
