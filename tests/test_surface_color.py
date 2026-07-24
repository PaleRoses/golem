from __future__ import annotations

import json
import struct
from hashlib import sha256
from pathlib import Path

import numpy as np
import trimesh
from jsonschema import Draft202012Validator

from golem import assembly
from golem.assembly.obstructions import (
    AppearancePaletteElementObstruction,
    SurfaceColorElementObstruction,
)
from golem.cli.compile import compile_spec
from golem.contract import body_schema
from golem.materials.surface import (
    AppearancePaletteRule,
    RejectedAppearancePalette,
    RejectedSurfaceColor,
    SurfaceColorRule,
    decode_appearance_palette,
    decode_surface_color,
)


_KNIGHT_PATH = Path(__file__).parents[1] / "specs" / "knight_body.json"
_APPEARANCE_TINTS = {
    "obsidian_warden": (0.035, 0.030, 0.050),
    "vascular_supply_gold": (0.90, 0.25, 0.08),
    "vascular_return_violet": (0.22, 0.08, 0.65),
    "vascular_exchange_cyan": (0.10, 0.75, 0.82),
}
_DEFAULT_KNIGHT_GLB_SHA256 = (
    "61a82d28d97400d072f38fe0e208958cb88acf4ee0bf6cdce1b1d30c37d2ef91"
)
_DEFAULT_KNIGHT_SIDECAR_SHA256 = (
    "1d5a36c49e7c888c2897eb1d911de603db9594a2a8cdb54247c1e1df3b821559"
)


def _surface_spec() -> dict[str, object]:
    return {
        "name": "surface-color-probe",
        "pitch": 0.03,
        "elements": [
            {
                "id": "body",
                "role": "creature",
                "graph": {
                    "name": "surface-color-probe",
                    "blend": 0.04,
                    "parts": [
                        {
                            "id": "torso",
                            "type": "blob",
                            "center": [0.0, 0.5, 0.0],
                            "size": [0.3, 0.4, 0.25],
                        }
                    ],
                },
                "appearance_material": "neutral_gray",
                "surface_color": {
                    "tint_linear_rgb": [0.55, 0.2, 0.07],
                    "variation": {
                        "kind": "triplanar_value_noise",
                        "wavelength_world": 0.7,
                        "amplitude": 0.2,
                    },
                },
            }
        ],
    }


def _glb_json(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    chunk_length, chunk_kind = struct.unpack_from("<II", content, 12)
    assert chunk_kind == 0x4E4F534A
    return json.loads(content[20 : 20 + chunk_length])


def _palette_knight_path(tmp_path: Path) -> Path:
    payload = {
        **json.loads(_KNIGHT_PATH.read_text(encoding="utf-8")),
        "appearance_material": "obsidian_warden",
        "appearance_palette": {
            material_id: {"tint_linear_rgb": tint}
            for material_id, tint in _APPEARANCE_TINTS.items()
        },
    }
    path = tmp_path / "palette-knight.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_surface_color_rejection_is_typed_and_accumulates() -> None:
    decoded = decode_surface_color(
        {
            "tint_linear_rgb": [1.2, "red", -0.1],
            "variation": {
                "kind": "sparkles",
                "wavelength_world": 0.0,
                "amplitude": 2.0,
            },
            "seed": 7,
        }
    )
    assert isinstance(decoded, RejectedSurfaceColor)
    assert tuple(item.address for item in decoded.obstructions) == (
        "/surface_color/seed",
        "/surface_color/tint_linear_rgb/0",
        "/surface_color/tint_linear_rgb/1",
        "/surface_color/tint_linear_rgb/2",
        "/surface_color/variation/kind",
        "/surface_color/variation/wavelength_world",
        "/surface_color/variation/amplitude",
    )
    assert tuple(item.rule for item in decoded.obstructions) == (
        SurfaceColorRule.UNKNOWN_FIELD,
        SurfaceColorRule.BOUNDED_NUMBER,
        SurfaceColorRule.FINITE_NUMBER,
        SurfaceColorRule.BOUNDED_NUMBER,
        SurfaceColorRule.VARIATION_KIND,
        SurfaceColorRule.BOUNDED_NUMBER,
        SurfaceColorRule.BOUNDED_NUMBER,
    )
    assert decoded.obstructions[1].authored == 1.2
    assert decoded.obstructions[1].required == {
        "minimum": 0.0,
        "maximum": 1.0,
        "exclusive_minimum": False,
    }
    compiled = assembly.compile_assembly(
        {
            **_surface_spec(),
            "elements": [
                {
                    **_surface_spec()["elements"][0],
                    "surface_color": {"tint_linear_rgb": "clay"},
                }
            ],
        },
        Path("."),
    )
    assert isinstance(compiled, assembly.RejectedAssembly)
    assert isinstance(compiled.obstructions[0], SurfaceColorElementObstruction)


def test_appearance_palette_rejection_is_typed_and_accumulates() -> None:
    decoded = decode_appearance_palette(
        {
            "unknown_material": {"tint_linear_rgb": [0.2, 0.3, 0.4]},
            "obsidian_warden": {"tint_linear_rgb": [1.2, "black", -0.1]},
            "vascular_supply_gold": {"tint_linear_rgb": "gold"},
        }
    )
    assert isinstance(decoded, RejectedAppearancePalette)
    assert tuple(map(lambda item: item.address, decoded.obstructions)) == (
        "/appearance_palette/unknown_material",
        "/appearance_palette/obsidian_warden/tint_linear_rgb/0",
        "/appearance_palette/obsidian_warden/tint_linear_rgb/1",
        "/appearance_palette/obsidian_warden/tint_linear_rgb/2",
        "/appearance_palette/vascular_supply_gold/tint_linear_rgb",
    )
    assert decoded.obstructions[0].rule is AppearancePaletteRule.KNOWN_MATERIAL
    assert decoded.obstructions[0].authored == "unknown_material"
    compiled = assembly.compile_assembly(
        {
            **_surface_spec(),
            "elements": [
                {
                    **_surface_spec()["elements"][0],
                    "appearance_palette": {
                        "vascular_supply_gold": {"tint_linear_rgb": "gold"}
                    },
                }
            ],
        },
        Path("."),
    )
    assert isinstance(compiled, assembly.RejectedAssembly)
    assert isinstance(
        compiled.obstructions[0], AppearancePaletteElementObstruction
    )
    overlap = assembly.compile_assembly(
        {
            **_surface_spec(),
            "elements": [
                {
                    **_surface_spec()["elements"][0],
                    "appearance_palette": {
                        "neutral_gray": {"tint_linear_rgb": [0.1, 0.2, 0.3]}
                    },
                }
            ],
        },
        Path("."),
    )
    assert isinstance(overlap, assembly.RejectedAssembly)
    assert isinstance(
        overlap.obstructions[0], AppearancePaletteElementObstruction
    )
    overlap_obstruction = overlap.obstructions[0].obstructions[0]
    assert overlap_obstruction.rule is (
        AppearancePaletteRule.DISTINCT_LOCAL_COLOR_OWNER
    )
    assert overlap_obstruction.authored == {
        "appearance_palette_material": "neutral_gray",
        "surface_color": True,
    }
    assert overlap_obstruction.required == {"maximum_color_owner_count": 1}


def test_tinted_demo_uses_the_live_body_authoring_schema() -> None:
    schema = body_schema()
    payload = json.loads(
        (Path(__file__).parents[1] / "specs" / "knight_tinted_demo.json").read_text()
    )
    Draft202012Validator(schema).validate(payload)
    assert payload["surface_color"]["variation"]["kind"] == schema["$defs"][
        "surfaceColor"
    ]["properties"]["variation"]["properties"]["kind"]["const"]


def test_tinted_demo_colors_the_solid_exterior_not_diagnostic_strata() -> None:
    result = compile_spec(
        Path(__file__).parents[1] / "specs" / "knight_tinted_demo.json"
    )
    assert isinstance(result, assembly.AcceptedAssembly)
    solid_records = tuple(
        record
        for record in result.records
        if record.stratum is assembly.AssemblyStratum.SOLID
    )
    diagnostic_records = tuple(
        record
        for record in result.records
        if record.stratum is not assembly.AssemblyStratum.SOLID
    )
    assert len(solid_records) == 1
    assert solid_records[0].surface_color is not None
    assert diagnostic_records
    assert all(record.surface_color is None for record in diagnostic_records)


def test_tinted_glb_contains_color_attribute_and_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    first = assembly.compile_assembly(_surface_spec(), tmp_path)
    second = assembly.compile_assembly(_surface_spec(), tmp_path)
    assert isinstance(first, assembly.AcceptedAssembly)
    assert isinstance(second, assembly.AcceptedAssembly)
    first_path = tmp_path / "first.glb"
    second_path = tmp_path / "second.glb"
    assembly.export_scene(first, str(first_path))
    assembly.export_scene(second, str(second_path))
    assert first_path.read_bytes() == second_path.read_bytes()
    assert Path(f"{first_path}.materials.json").read_bytes() == Path(
        f"{second_path}.materials.json"
    ).read_bytes()
    primitive = _glb_json(first_path)["meshes"][0]["primitives"][0]
    assert "COLOR_0" in primitive["attributes"]
    scene = trimesh.load(first_path)
    colors = np.asarray(scene.geometry["body"].visual.vertex_attributes["color"])
    assert colors.shape == (len(scene.geometry["body"].vertices), 4)
    assert len(np.unique(colors, axis=0)) > 1
    sidecar = json.loads(Path(f"{first_path}.materials.json").read_text())
    assert sidecar["surface_colors"]["body"]["variation"]["kind"] == (
        "triplanar_value_noise"
    )
    assert len(sidecar["surface_colors"]["body"]["derived_seed"]) == 16


def test_surface_color_is_opt_in_at_the_glb_and_sidecar_boundary(
    tmp_path: Path,
) -> None:
    tinted = _surface_spec()
    plain_element = {
        key: value
        for key, value in tinted["elements"][0].items()
        if key != "surface_color"
    }
    plain = assembly.compile_assembly(
        {**tinted, "elements": [plain_element]},
        tmp_path,
    )
    assert isinstance(plain, assembly.AcceptedAssembly)
    output = tmp_path / "plain.glb"
    assembly.export_scene(plain, str(output))
    primitive = _glb_json(output)["meshes"][0]["primitives"][0]
    assert "COLOR_0" not in primitive["attributes"]
    sidecar = json.loads(Path(f"{output}.materials.json").read_text())
    assert "surface_colors" not in sidecar


def test_appearance_palette_bakes_body_and_vascular_colors_into_glb(
    tmp_path: Path,
) -> None:
    result = compile_spec(
        _palette_knight_path(tmp_path),
        assembly.PinnedResolution(60),
    )
    assert isinstance(result, assembly.AcceptedAssembly)
    material_by_record = {
        record.record_id: record.appearance_material for record in result.records
    }
    assert material_by_record == {
        "obsidian-knight-underbody": "obsidian_warden",
        "obsidian-knight-underbody__vascular_supply": "vascular_supply_gold",
        "obsidian-knight-underbody__vascular_return": "vascular_return_violet",
        "obsidian-knight-underbody__vascular_exchange": "vascular_exchange_cyan",
    }
    assert all(
        record.surface_color is not None
        and record.surface_color.recipe.tint_linear_rgb
        == _APPEARANCE_TINTS[record.appearance_material]
        for record in result.records
    )

    output = tmp_path / "palette-knight.glb"
    assembly.export_scene(result, str(output))
    tree = _glb_json(output)
    mesh_by_name = {mesh["name"]: mesh for mesh in tree["meshes"]}
    assert set(mesh_by_name) == set(material_by_record)
    assert all(
        "COLOR_0" in mesh["primitives"][0]["attributes"]
        for mesh in mesh_by_name.values()
    )
    assert all(
        tree["materials"][mesh["primitives"][0]["material"]][
            "pbrMetallicRoughness"
        ]["baseColorFactor"]
        == [1.0, 1.0, 1.0, 1.0]
        for mesh in mesh_by_name.values()
    )

    scene = trimesh.load(output)
    expected_rgba_by_record = {
        record_id: np.asarray(
            (
                *tuple(
                    np.floor(
                        np.asarray(_APPEARANCE_TINTS[material_id]) * 255.0 + 0.5
                    ).astype(np.uint8)
                ),
                255,
            ),
            dtype=np.uint8,
        )
        for record_id, material_id in material_by_record.items()
    }
    assert all(
        np.array_equal(
            geometry.visual.vertex_attributes["color"],
            np.broadcast_to(
                expected_rgba_by_record[record_id],
                (len(geometry.vertices), 4),
            ),
        )
        for record_id, geometry in scene.geometry.items()
    )
    assert len(
        frozenset(map(lambda color: tuple(color), expected_rgba_by_record.values()))
    ) == 4
    sidecar = json.loads(Path(f"{output}.materials.json").read_text())
    assert set(sidecar["surface_colors"]) == set(material_by_record)


def test_default_knight_glb_remains_byte_identical_without_palette(
    tmp_path: Path,
) -> None:
    result = compile_spec(_KNIGHT_PATH)
    assert isinstance(result, assembly.AcceptedAssembly)
    output = tmp_path / "knight-default.glb"
    assembly.export_scene(result, str(output))
    sidecar = Path(f"{output}.materials.json")
    assert sha256(output.read_bytes()).hexdigest() == _DEFAULT_KNIGHT_GLB_SHA256
    assert (
        sha256(sidecar.read_bytes()).hexdigest()
        == _DEFAULT_KNIGHT_SIDECAR_SHA256
    )
    assert all(
        "COLOR_0" not in mesh["primitives"][0]["attributes"]
        for mesh in _glb_json(output)["meshes"]
    )
    assert "surface_colors" not in json.loads(sidecar.read_text())
