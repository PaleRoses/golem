"""Focused laws for the single typed material catalogue."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from golem import materials as M
from golem.assembly.exchange import write_sidecar
from golem.materials.projection import appearance_materials_payload
from golem.materials.surface import (
    AcceptedAppearancePalette,
    RejectedSurfaceColor,
    decode_appearance_palette,
    decode_surface_color,
)


@pytest.mark.parametrize("material_id", tuple(M.AppearanceMaterialId))
def test_appearance_vocabulary_decodes_totally(material_id):
    result = M.decode_appearance_material(material_id.value)
    assert isinstance(result, M.AcceptedMaterial)
    assert result.material.material_id is material_id
    palette = decode_appearance_palette(
        {material_id.value: {"tint_linear_rgb": [0.1, 0.2, 0.3]}}
    )
    assert isinstance(palette, AcceptedAppearancePalette)
    assert tuple(
        map(lambda entry: entry.material_id, palette.palette.entries)
    ) == (material_id,)


def test_catalogue_is_total_and_identifiers_are_disjoint():
    catalogue = M.MATERIAL_CATALOGUE
    appearance_ids = tuple(
        material.material_id for material in catalogue.appearance_materials
    )
    solid_ids = tuple(material.material_id for material in catalogue.isotropic_solids)
    fluid_ids = tuple(
        material.material_id for material in catalogue.incompressible_fluids
    )
    gas_ids = tuple(
        material.material_id for material in catalogue.compressible_gases
    )
    assert appearance_ids == tuple(M.AppearanceMaterialId)
    assert solid_ids == tuple(M.IsotropicSolidMaterialId)
    assert fluid_ids == tuple(M.IncompressibleFluidMaterialId)
    assert gas_ids == tuple(M.CompressibleGasMaterialId)
    assert len(
        frozenset(
            (
                *tuple(map(lambda value: value.value, appearance_ids)),
                *tuple(map(lambda value: value.value, solid_ids)),
                *tuple(map(lambda value: value.value, fluid_ids)),
                *tuple(map(lambda value: value.value, gas_ids)),
            )
        )
    ) == len(appearance_ids) + len(solid_ids) + len(fluid_ids) + len(gas_ids)


def test_appearance_records_are_frozen_and_bounded():
    appearances = M.MATERIAL_CATALOGUE.appearance_materials
    assert all(0.0 <= material.roughness <= 1.0 for material in appearances)
    assert all(0.0 <= material.metallic <= 1.0 for material in appearances)
    assert all(
        0.0 <= channel <= 1.0
        for material in appearances
        for channel in (*material.base_color, *material.emissive_color)
    )
    with pytest.raises(FrozenInstanceError):
        appearances[0].roughness = 999.0


def test_unknown_visual_tag_has_explicit_visual_only_fallback():
    decoded = M.decode_appearance_material("no_such_appearance")
    fallback = M.resolve_appearance_material("no_such_appearance")
    assert isinstance(decoded, M.RejectedMaterial)
    assert decoded.obstructions == (
        M.UnknownMaterialObstruction(
            M.MaterialKind.APPEARANCE, "no_such_appearance"
        ),
    )
    assert fallback.material_id is M.DEFAULT_APPEARANCE_MATERIAL_ID


def test_surface_color_decoder_accumulates_typed_obstructions():
    result = decode_surface_color(
        {
            "tint_linear_rgb": [0.2, True, 1.2],
            "variation": {
                "kind": "random",
                "wavelength_world": 0.0,
                "amplitude": float("inf"),
            },
        }
    )
    assert isinstance(result, RejectedSurfaceColor)
    assert tuple(map(lambda obstruction: obstruction.address, result.obstructions)) == (
        "/surface_color/tint_linear_rgb/1",
        "/surface_color/tint_linear_rgb/2",
        "/surface_color/variation/kind",
        "/surface_color/variation/wavelength_world",
        "/surface_color/variation/amplitude",
    )


@pytest.mark.parametrize(
    ("decoder", "identifier", "kind"),
    (
        (
            M.decode_isotropic_solid,
            "neutral_gray",
            M.MaterialKind.ISOTROPIC_SOLID,
        ),
        (
            M.decode_incompressible_fluid,
            "neutral_gray",
            M.MaterialKind.INCOMPRESSIBLE_FLUID,
        ),
        (
            M.decode_compressible_gas,
            "neutral_gray",
            M.MaterialKind.COMPRESSIBLE_GAS,
        ),
    ),
)
def test_physics_lookup_fails_closed(decoder, identifier, kind):
    result = decoder(identifier)
    assert isinstance(result, M.RejectedMaterial)
    assert result.obstructions == (M.UnknownMaterialObstruction(kind, identifier),)


def test_physical_records_carry_constant_property_validity_and_evidence():
    solid_result = M.decode_isotropic_solid(
        M.IsotropicSolidMaterialId.STAINLESS_STEEL_316L_ROOM_TEMPERATURE.value
    )
    fluid_result = M.decode_incompressible_fluid(
        M.IncompressibleFluidMaterialId.WATER_0_1_MPA_293K.value
    )
    gas_result = M.decode_compressible_gas(
        M.CompressibleGasMaterialId.DRY_AIR_101325_PA_298K.value
    )
    assert isinstance(solid_result, M.AcceptedMaterial)
    assert isinstance(fluid_result, M.AcceptedMaterial)
    assert isinstance(gas_result, M.AcceptedMaterial)
    solid = solid_result.material
    fluid = fluid_result.material
    gas = gas_result.material
    assert solid.operating_temperature.contains(350.0)
    assert not solid.operating_temperature.contains(400.0)
    assert fluid.operating_temperature.contains(293.15)
    assert fluid.operating_temperature.contains(288.15)
    assert not fluid.operating_temperature.contains(288.14)
    assert fluid.allowable_pressure.contains(250_000.0)
    assert solid.allowable_stress_pascal < solid.yield_stress_pascal
    assert solid.evidence and fluid.evidence
    assert gas.equation_of_state is M.GasEquationOfState.IDEAL_GAS
    assert gas.operating_temperature.contains(298.15)
    assert gas.allowable_pressure.contains(101_325.0)
    assert gas.evidence


def test_sidecar_is_a_derived_visual_view(tmp_path):
    mapping = {
        "body": M.AppearanceMaterialId.OBSIDIAN_WARDEN,
        "unknown": "unregistered_visual_tag",
    }
    neutral_payload = json.loads(
        json.dumps(
            M.appearance_material_to_dict(
                M.resolve_appearance_material(M.DEFAULT_APPEARANCE_MATERIAL_ID.value)
            )
        )
    )

    # Pure projection preserves the element->tag mapping verbatim, including the
    # unregistered tag, and derives its record from the neutral default material.
    payload = appearance_materials_payload(mapping)
    assert payload["elements"] == {
        "body": M.AppearanceMaterialId.OBSIDIAN_WARDEN.value,
        "unknown": "unregistered_visual_tag",
    }

    # The effectful exchange writer emits exactly that payload as JSON to disk.
    path = tmp_path / "scene.glb.materials.json"
    write_sidecar(mapping, path)
    written = json.loads(path.read_text())
    assert written == json.loads(json.dumps(payload))
    assert written["records"]["unregistered_visual_tag"] == neutral_payload
