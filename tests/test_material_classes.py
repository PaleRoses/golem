"""Material classes (wave-3, R8): opaque/emissive/translucent vocabulary.

Decode is strict and names the legal class set; look renders emissive
shading-exempt and composites translucent at authored opacity; export maps
emissive to glTF emissive factors and translucent to alpha blend. Specs
declaring no classes stay byte-identical (PNG and GLB hashes pinned
against the pre-class renderer).
"""

from __future__ import annotations

import json
import struct
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from golem import assembly
from golem.assembly import AssemblyRecord, AssemblyStratum, ElementRole
from golem.assembly.obstructions import AppearancePaletteElementObstruction
from golem.cli import look, main
from golem.cli.compile import compile_spec
from golem.materials import AppearanceMaterialId
from golem.materials.surface import (
    AcceptedAppearancePalette,
    EmissiveSurfaceClass,
    OpaqueSurfaceClass,
    RejectedAppearancePalette,
    RejectedSurfaceColor,
    SeededSurfaceColor,
    SurfaceColorOwner,
    SurfaceColorRecipe,
    SurfaceColorRule,
    TranslucentSurfaceClass,
    decode_appearance_palette,
    decode_surface_color,
)
from golem.senses import orthographic

_SPECS = Path(__file__).parents[1] / "specs"

_ABSENT_CLASS_LOOK_SHA256 = {
    ("knight_body", "flat", "front"): "e84e05432cc383daa1019bb59ed6a988046e0c1953d1042b3a6ebfbf4af74a80",
    ("knight_body", "flat", "side"): "c73d35ce3e6581a13a82224a4dc4d3e53701a33db3666a763f2e73e77990e5d6",
    ("knight_body", "flat", "top"): "c3399a5e0c82fe19747b732b905bdb81dbc0251176aaec936547d809467c9fd4",
    ("knight_body", "shaded", "front"): "bb267f466657b435cf0fc04146eaff478c4330dd0f779a28bc0cbce519e632fa",
    ("knight_body", "shaded", "side"): "13ef398cddcdbdb0e109a8f22fb4b50f30e50b3ee04b79c61d4a2533d73c993f",
    ("knight_body", "shaded", "top"): "d463ad555526fec08effc5dc92f466837568c26fa271a7afd25552ce2f19dafe",
    ("scree_maiden", "flat", "front"): "fbfbf197508af96d06c5b0ac2b7b9f9b9735f691c6ba0972d992f3d0c5df2ec0",
    ("scree_maiden", "flat", "side"): "d7917627a1035ee2778f0190fae98886bae5ef489b10e03f74c608e38fb527f5",
    ("scree_maiden", "flat", "top"): "6612a1b31d10d696ff2284c6e8d01e8849b20df57a94c740237af44275e43111",
    ("scree_maiden", "shaded", "front"): "60468209dc0f8d3fba622ef2a1efd9256ba27b17a9b4c8a1c860f667c7721c0e",
    ("scree_maiden", "shaded", "side"): "81353e1e38452328ff4aa52cd6b8e083db9f3484893a72b1e714134af5813ea6",
    ("scree_maiden", "shaded", "top"): "e8a432104a7aa269021bdf17f2e0f0119572d1a541e888d6b7d335c38637af30",
}
_ABSENT_CLASS_GLB_SHA256 = {
    "scree_maiden": (
        "98e6be1a322b248f4c2a68204c46344f1248b43cf15d21a15c4eb4bf53bb0d1b",
        "2dc1b14d85ae8b415ab2ea817074b568ef6553a3a4512c5f86a3bc8115371678",
    ),
}


def _srgb_bytes(linear_rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    linear = np.clip(np.asarray(linear_rgb, dtype=np.float64), 0.0, 1.0)
    encoded = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return tuple(
        int(channel)
        for channel in np.floor(encoded * 255.0 + 0.5).astype(np.uint8)
    )


def _triangle_record(
    record_id: str,
    surface_color: SeededSurfaceColor | None,
    *,
    x_offset: float = 0.0,
    z: float = 0.0,
    scale: float = 1.0,
    appearance_material: str = "neutral_gray",
) -> AssemblyRecord:
    return AssemblyRecord.from_derived_geometry(
        record_id=record_id,
        element_id="element",
        role=ElementRole.CREATURE,
        stratum=AssemblyStratum.SOLID,
        appearance_material=appearance_material,
        vertices=np.asarray(
            (
                (x_offset, 0.0, z),
                (x_offset + 0.6 * scale, 0.0, z),
                (x_offset, 0.7 * scale, z),
            ),
            dtype=np.float64,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        resolution=60,
        report={},
        surface_color=surface_color,
    )


def _palette_color(
    tint: tuple[float, float, float],
    surface_class=None,
    seed: int = 11,
) -> SeededSurfaceColor:
    return SeededSurfaceColor(
        SurfaceColorRecipe(tint),
        seed,
        SurfaceColorOwner.APPEARANCE_PALETTE,
        surface_class if surface_class is not None else OpaqueSurfaceClass(),
    )


def _render_records(
    records: tuple[AssemblyRecord, ...],
    *,
    shaded: bool,
    image_size: int = 64,
    fit_subject: bool = True,
) -> tuple[np.ndarray, ...]:
    rendered = look._render_orthographic_views(
        records,
        (orthographic.OrthographicView.FRONT,),
        image_size,
        fit_subject=fit_subject,
        shaded=shaded,
    )
    assert isinstance(rendered, orthographic.AcceptedOrthographicRender)
    return tuple(image.pixels for image in rendered.images)


def _glb_json(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    chunk_length, chunk_kind = struct.unpack_from("<II", content, 12)
    assert chunk_kind == 0x4E4F534A
    return json.loads(content[20 : 20 + chunk_length])


def _class_probe_spec(*, opaque_keyword: bool = False) -> dict[str, object]:
    def element(
        element_id: str,
        material: str,
        palette: dict[str, object],
        part: dict[str, object],
    ) -> dict[str, object]:
        return {
            "id": element_id,
            "role": "creature",
            "graph": {"name": element_id, "blend": 0.04, "parts": [part]},
            "appearance_material": material,
            "appearance_palette": palette,
        }

    hide_palette: dict[str, object] = {"tint_linear_rgb": [0.03, 0.03, 0.045]}
    if opaque_keyword:
        hide_palette = {"class": "opaque", **hide_palette}
    return {
        "name": "class-probe",
        "pitch": 0.03,
        "elements": [
            element(
                "hide",
                "obsidian_warden",
                {"obsidian_warden": hide_palette},
                {
                    "id": "body",
                    "type": "blob",
                    "center": [0.0, 0.5, 0.0],
                    "size": [0.3, 0.25, 0.2],
                },
            ),
            element(
                "maw",
                "emissive_seam",
                {
                    "emissive_seam": {
                        "class": "emissive",
                        "tint_linear_rgb": [1.0, 0.25, 0.05],
                        "intensity": 2.5,
                    }
                },
                {
                    "id": "glow",
                    "type": "blob",
                    "center": [0.0, 0.42, 0.55],
                    "size": [0.14, 0.1, 0.08],
                },
            ),
            element(
                "wing",
                "eye_gloss",
                {
                    "eye_gloss": {
                        "class": "translucent",
                        "tint_linear_rgb": [0.15, 0.45, 0.9],
                        "opacity": 0.45,
                    }
                },
                {
                    "id": "membrane",
                    "type": "blob",
                    "center": [0.0, 1.1, 0.0],
                    "size": [0.35, 0.15, 0.05],
                },
            ),
        ],
    }


def test_emissive_class_decodes_with_typed_parameters() -> None:
    decoded = decode_appearance_palette(
        {
            "emissive_seam": {
                "class": "emissive",
                "tint_linear_rgb": [1.0, 0.25, 0.05],
                "intensity": 2.5,
            }
        }
    )
    assert isinstance(decoded, AcceptedAppearancePalette)
    surface_class = decoded.palette.entries[0].color.surface_class
    assert isinstance(surface_class, EmissiveSurfaceClass)
    assert surface_class.intensity == 2.5
    defaulted = decode_appearance_palette(
        {
            "emissive_seam": {
                "class": "emissive",
                "tint_linear_rgb": [1.0, 0.25, 0.05],
            }
        }
    )
    assert isinstance(defaulted, AcceptedAppearancePalette)
    assert (
        defaulted.palette.entries[0].color.surface_class.intensity == 1.0
    )


def test_translucent_class_decodes_with_opacity_and_variation() -> None:
    decoded = decode_appearance_palette(
        {
            "eye_gloss": {
                "class": "translucent",
                "tint_linear_rgb": [0.15, 0.45, 0.9],
                "opacity": 0.45,
                "variation": {
                    "kind": "triplanar_value_noise",
                    "wavelength_world": 0.5,
                    "amplitude": 0.1,
                },
            }
        }
    )
    assert isinstance(decoded, AcceptedAppearancePalette)
    entry = decoded.palette.entries[0].color
    assert isinstance(entry.surface_class, TranslucentSurfaceClass)
    assert entry.surface_class.opacity == 0.45
    assert entry.recipe.variation is not None


def test_absent_and_explicit_class_decode_as_opaque() -> None:
    absent = decode_appearance_palette(
        {"neutral_gray": {"tint_linear_rgb": [0.5, 0.5, 0.5]}}
    )
    explicit = decode_appearance_palette(
        {
            "neutral_gray": {
                "class": "opaque",
                "tint_linear_rgb": [0.5, 0.5, 0.5],
            }
        }
    )
    assert isinstance(absent, AcceptedAppearancePalette)
    assert isinstance(explicit, AcceptedAppearancePalette)
    assert absent.palette == explicit.palette
    assert isinstance(
        absent.palette.entries[0].color.surface_class, OpaqueSurfaceClass
    )


def test_unknown_class_names_the_legal_set() -> None:
    decoded = decode_appearance_palette(
        {
            "obsidian_warden": {
                "class": "sparkly",
                "tint_linear_rgb": [0.1, 0.2, 0.3],
            }
        }
    )
    assert isinstance(decoded, RejectedAppearancePalette)
    obstruction = decoded.obstructions[0]
    assert obstruction.rule is SurfaceColorRule.MATERIAL_CLASS
    assert obstruction.address == "/appearance_palette/obsidian_warden/class"
    assert obstruction.authored == "sparkly"
    assert obstruction.required == ("emissive", "opaque", "translucent")


def test_class_parameter_rejections_are_typed_and_accumulate() -> None:
    decoded = decode_appearance_palette(
        {
            "emissive_seam": {
                "class": "emissive",
                "tint_linear_rgb": [1.0, 1.0, 1.0],
                "intensity": 0.0,
                "variation": {
                    "kind": "triplanar_value_noise",
                    "wavelength_world": 1.0,
                    "amplitude": 0.5,
                },
            },
            "eye_gloss": {
                "class": "translucent",
                "tint_linear_rgb": [0.2, 0.2, 0.2],
                "opacity": 1.5,
            },
            "neutral_gray": {
                "class": "translucent",
                "tint_linear_rgb": [0.5, 0.5, 0.5],
            },
            "gambeson_dark": {
                "class": "opaque",
                "tint_linear_rgb": [0.4, 0.4, 0.4],
                "opacity": 0.5,
            },
        }
    )
    assert isinstance(decoded, RejectedAppearancePalette)
    by_address = {
        obstruction.address: obstruction for obstruction in decoded.obstructions
    }
    emissive_intensity = by_address[
        "/appearance_palette/emissive_seam/intensity"
    ]
    assert emissive_intensity.rule is SurfaceColorRule.BOUNDED_NUMBER
    assert emissive_intensity.required == {
        "minimum": 0.0,
        "maximum": None,
        "exclusive_minimum": True,
    }
    emissive_variation = by_address[
        "/appearance_palette/emissive_seam/variation"
    ]
    assert emissive_variation.rule is SurfaceColorRule.UNKNOWN_FIELD
    wing_opacity = by_address["/appearance_palette/eye_gloss/opacity"]
    assert wing_opacity.rule is SurfaceColorRule.BOUNDED_NUMBER
    assert wing_opacity.required == {
        "minimum": 0.0,
        "maximum": 1.0,
        "exclusive_minimum": False,
    }
    missing_opacity = by_address["/appearance_palette/neutral_gray/opacity"]
    assert missing_opacity.rule is SurfaceColorRule.REQUIRED_FIELD
    opaque_opacity = by_address["/appearance_palette/gambeson_dark/opacity"]
    assert opaque_opacity.rule is SurfaceColorRule.UNKNOWN_FIELD
    non_finite = decode_appearance_palette(
        {
            "emissive_seam": {
                "class": "emissive",
                "tint_linear_rgb": [1.0, 1.0, 1.0],
                "intensity": "hot",
            }
        }
    )
    assert isinstance(non_finite, RejectedAppearancePalette)
    assert (
        non_finite.obstructions[0].rule is SurfaceColorRule.FINITE_NUMBER
    )


def test_local_surface_color_rejects_the_class_vocabulary() -> None:
    decoded = decode_surface_color(
        {"class": "emissive", "tint_linear_rgb": [1.0, 0.0, 0.0]}
    )
    assert isinstance(decoded, RejectedSurfaceColor)
    assert decoded.obstructions[0].rule is SurfaceColorRule.UNKNOWN_FIELD
    assert decoded.obstructions[0].required == (
        "tint_linear_rgb",
        "variation",
    )


def test_compile_threads_classes_onto_records() -> None:
    compiled = assembly.compile_assembly(_class_probe_spec(), Path("."))
    assert isinstance(compiled, assembly.AcceptedAssembly)
    classes = {
        record.element_id: type(record.surface_color.surface_class)
        for record in compiled.records
        if record.surface_color is not None
    }
    assert classes["hide"] is OpaqueSurfaceClass
    assert classes["maw"] is EmissiveSurfaceClass
    assert classes["wing"] is TranslucentSurfaceClass
    rejected = assembly.compile_assembly(
        {
            **_class_probe_spec(),
            "elements": [
                {
                    **_class_probe_spec()["elements"][1],
                    "appearance_palette": {
                        "emissive_seam": {
                            "class": "glow",
                            "tint_linear_rgb": [1.0, 0.25, 0.05],
                        }
                    },
                },
                *_class_probe_spec()["elements"][::2],
            ],
        },
        Path("."),
    )
    assert isinstance(rejected, assembly.RejectedAssembly)
    assert isinstance(
        rejected.obstructions[0], AppearancePaletteElementObstruction
    )


def test_emissive_renders_brighter_than_identically_tinted_opaque() -> None:
    tint = (0.5, 0.05, 0.05)
    emissive = _triangle_record(
        "emissive",
        _palette_color(tint, EmissiveSurfaceClass(1.0)),
        x_offset=0.0,
    )
    opaque = _triangle_record(
        "opaque",
        _palette_color(tint),
        x_offset=1.0,
    )
    (pixels,) = _render_records((emissive, opaque), shaded=True)

    expected = _srgb_bytes(tint)
    emissive_hits = np.all(pixels == np.asarray(expected), axis=2)
    red_surface = (pixels[..., 0] > 100) & (
        pixels[..., 0].astype(np.int64) > pixels[..., 1].astype(np.int64) + 30
    )
    opaque_hits = red_surface & ~emissive_hits
    luminance = pixels.astype(np.float64) @ np.asarray(
        (0.2126, 0.7152, 0.0722)
    )

    assert int(emissive_hits.sum()) >= 20
    assert int(opaque_hits.sum()) >= 20
    assert float(luminance[emissive_hits].mean()) > float(
        luminance[opaque_hits].mean()
    )


def test_emissive_matches_identically_tinted_opaque_when_unshaded() -> None:
    tint = (0.5, 0.05, 0.05)
    emissive = _triangle_record(
        "emissive",
        _palette_color(tint, EmissiveSurfaceClass(1.0)),
        x_offset=0.0,
    )
    opaque = _triangle_record(
        "opaque",
        _palette_color(tint),
        x_offset=1.0,
    )
    (pixels,) = _render_records((emissive, opaque), shaded=False)

    expected = np.asarray(_srgb_bytes(tint))
    authored_hits = np.all(pixels == expected, axis=2)
    left = authored_hits[:, : pixels.shape[1] // 2]
    right = authored_hits[:, pixels.shape[1] // 2 :]
    assert int(left.sum()) >= 20
    assert int(right.sum()) >= 20


def test_emissive_renders_at_full_authored_luminance_when_shaded() -> None:
    tint = (0.5, 0.05, 0.05)
    intensity = 2.0
    emissive = _triangle_record(
        "emissive",
        _palette_color(tint, EmissiveSurfaceClass(intensity)),
    )
    (pixels,) = _render_records((emissive,), shaded=True)

    expected = _srgb_bytes(
        tuple(min(component * intensity, 1.0) for component in tint)
    )
    emissive_hits = np.all(pixels == np.asarray(expected), axis=2)
    assert int(emissive_hits.sum()) >= 20


def test_emissive_behind_opaque_stays_occluded() -> None:
    tint = (0.5, 0.05, 0.05)
    emissive = _triangle_record(
        "emissive",
        _palette_color(tint, EmissiveSurfaceClass(2.0)),
        z=0.0,
    )
    occluder = _triangle_record(
        "occluder",
        _palette_color((0.05, 0.05, 0.4)),
        z=0.2,
        scale=1.4,
    )
    (pixels,) = _render_records((emissive, occluder), shaded=True)

    emissive_color = np.asarray(_srgb_bytes(tint))
    assert int(np.all(pixels == emissive_color, axis=2).sum()) == 0


@pytest.mark.parametrize("shaded", (False, True))
def test_translucent_composites_over_what_lies_behind(shaded: bool) -> None:
    backdrop_tint = (0.8, 0.1, 0.1)
    membrane_tint = (0.1, 0.1, 0.8)
    opacity = 0.5
    backdrop = _triangle_record(
        "backdrop",
        _palette_color(backdrop_tint),
        z=0.0,
    )
    membrane = _triangle_record(
        "membrane",
        _palette_color(membrane_tint, TranslucentSurfaceClass(opacity)),
        z=0.1,
        scale=0.5,
    )
    (pixels,) = _render_records((backdrop, membrane), shaded=shaded)

    body = np.any(pixels != 244, axis=2)
    red = pixels[..., 0].astype(np.int64)
    green = pixels[..., 1].astype(np.int64)
    blue = pixels[..., 2].astype(np.int64)
    membrane_hits = (
        body
        & (blue > 110)
        & (red > 100)
        & (red < 210)
        & (green < red - 20)
    )
    backdrop_hits = body & (red > blue + 30) & (red > green + 30)
    assert int(membrane_hits.sum()) >= 20
    assert int(backdrop_hits.sum()) >= 20
    membrane_flat = _srgb_bytes(membrane_tint)
    assert (
        int(np.all(pixels[body] == membrane_flat, axis=1).sum()) == 0
    )
    blended_red = float(red[membrane_hits].mean())
    blended_blue = float(blue[membrane_hits].mean())
    assert blended_red > membrane_flat[0] + 20
    assert blended_blue < membrane_flat[2] - 20


def test_translucent_composite_is_exact_over_flat_backdrop() -> None:
    backdrop_tint = (0.8, 0.1, 0.1)
    membrane_tint = (0.1, 0.1, 0.8)
    opacity = 0.5
    backdrop = _triangle_record(
        "backdrop",
        _palette_color(backdrop_tint),
        z=0.0,
    )
    membrane = _triangle_record(
        "membrane",
        _palette_color(membrane_tint, TranslucentSurfaceClass(opacity)),
        z=0.1,
        scale=0.5,
    )
    (pixels,) = _render_records(
        (backdrop, membrane),
        shaded=False,
        image_size=48,
        fit_subject=False,
    )

    expected = np.rint(
        0.5 * np.asarray(_srgb_bytes(membrane_tint), dtype=np.float64)
        + 0.5 * np.asarray(_srgb_bytes(backdrop_tint), dtype=np.float64)
    ).astype(np.uint8)
    assert int(np.all(pixels == expected, axis=2).sum()) > 0


def test_all_translucent_assembly_composites_over_background() -> None:
    membrane_tint = (0.1, 0.1, 0.8)
    opacity = 0.5
    membrane = _triangle_record(
        "membrane",
        _palette_color(membrane_tint, TranslucentSurfaceClass(opacity)),
        z=0.0,
    )
    (pixels,) = _render_records(
        (membrane,),
        shaded=False,
        image_size=48,
        fit_subject=False,
    )

    expected = np.rint(
        0.5 * np.asarray(_srgb_bytes(membrane_tint), dtype=np.float64)
        + 0.5 * np.asarray((244.0, 244.0, 244.0))
    ).astype(np.uint8)
    assert int(np.all(pixels == expected, axis=2).sum()) > 0


def test_class_render_names_its_compositing_in_figure_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "class-probe.json"
    output_directory = tmp_path / "classed"
    spec_path.write_text(
        json.dumps(_class_probe_spec()),
        encoding="utf-8",
    )
    exit_code = main(
        (
            "look",
            str(spec_path),
            "--view",
            "front",
            "--res",
            "60",
            "--size",
            "48",
            "--shaded",
            "--out",
            str(output_directory),
        )
    )
    assert exit_code == 0
    capsys.readouterr()
    with Image.open(output_directory / "class-probe_front.png") as image:
        note = image.text.get("golem.material_classes", "")
    assert "translucent" in note
    assert "emissive" in note
    assert "collapse to the nearest" in note

    plain_spec = _class_probe_spec()
    plain_spec["elements"] = [
        {
            **plain_spec["elements"][0],
            "appearance_palette": {
                "obsidian_warden": {"tint_linear_rgb": [0.03, 0.03, 0.045]}
            },
        }
    ]
    plain_path = tmp_path / "plain-probe.json"
    plain_directory = tmp_path / "plain"
    plain_path.write_text(json.dumps(plain_spec), encoding="utf-8")
    assert (
        main(
            (
                "look",
                str(plain_path),
                "--view",
                "front",
                "--res",
                "60",
                "--size",
                "48",
                "--out",
                str(plain_directory),
            )
        )
        == 0
    )
    capsys.readouterr()
    with Image.open(plain_directory / "plain-probe_front.png") as image:
        assert "golem.material_classes" not in image.text


def test_export_carries_emissive_factors_and_alpha_blend(
    tmp_path: Path,
) -> None:
    compiled = assembly.compile_assembly(_class_probe_spec(), tmp_path)
    assert isinstance(compiled, assembly.AcceptedAssembly)
    output = tmp_path / "class-probe.glb"
    assembly.export_scene(compiled, str(output))
    tree = _glb_json(output)
    material_by_mesh = {
        mesh["name"]: tree["materials"][mesh["primitives"][0]["material"]]
        for mesh in tree["meshes"]
    }
    maw = material_by_mesh["maw"]
    assert maw["emissiveFactor"] == [1.0, 0.25, 0.05]
    assert maw["pbrMetallicRoughness"]["baseColorFactor"] == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    wing = material_by_mesh["wing"]
    assert wing["alphaMode"] == "BLEND"
    wing_factor = wing["pbrMetallicRoughness"]["baseColorFactor"]
    assert wing_factor[:3] == [1.0, 1.0, 1.0]
    assert wing_factor[3] == pytest.approx(115 / 255)
    assert "alphaMode" not in material_by_mesh["hide"]

    scene = trimesh.load(output)
    wing_colors = np.asarray(
        scene.geometry["wing"].visual.vertex_attributes["color"]
    )
    assert wing_colors.shape == (len(scene.geometry["wing"].vertices), 4)
    assert frozenset(wing_colors[:, 3].tolist()) == {115}

    sidecar = json.loads(Path(f"{output}.materials.json").read_text())
    assert sidecar["surface_colors"]["maw"]["class"] == "emissive"
    assert sidecar["surface_colors"]["maw"]["intensity"] == 2.5
    assert sidecar["surface_colors"]["wing"]["class"] == "translucent"
    assert sidecar["surface_colors"]["wing"]["opacity"] == 0.45
    assert "class" not in sidecar["surface_colors"]["hide"]


def test_explicit_opaque_class_exports_byte_identical_glb(
    tmp_path: Path,
) -> None:
    plain = assembly.compile_assembly(_class_probe_spec(), tmp_path)
    keyworded = assembly.compile_assembly(
        _class_probe_spec(opaque_keyword=True),
        tmp_path,
    )
    assert isinstance(plain, assembly.AcceptedAssembly)
    assert isinstance(keyworded, assembly.AcceptedAssembly)
    plain_path = tmp_path / "plain.glb"
    keyworded_path = tmp_path / "keyworded.glb"
    assembly.export_scene(plain, str(plain_path))
    assembly.export_scene(keyworded, str(keyworded_path))
    assert plain_path.read_bytes() == keyworded_path.read_bytes()


@pytest.mark.parametrize("spec_stem", ("knight_body", "scree_maiden"))
@pytest.mark.parametrize("mode", ("flat", "shaded"))
def test_absent_classes_render_byte_identical_pngs(
    tmp_path: Path,
    spec_stem: str,
    mode: str,
) -> None:
    exit_code = main(
        (
            "look",
            str(_SPECS / f"{spec_stem}.json"),
            "--view",
            "all",
            "--out",
            str(tmp_path),
            *(("--shaded",) if mode == "shaded" else ()),
        )
    )
    assert exit_code == 0
    for view in ("front", "side", "top"):
        digest = sha256(
            (tmp_path / f"{spec_stem}_{view}.png").read_bytes()
        ).hexdigest()
        assert digest == _ABSENT_CLASS_LOOK_SHA256[(spec_stem, mode, view)]


def test_absent_classes_export_byte_identical_glb(tmp_path: Path) -> None:
    glb_sha, sidecar_sha = _ABSENT_CLASS_GLB_SHA256["scree_maiden"]
    compiled = compile_spec(_SPECS / "scree_maiden.json")
    assert isinstance(compiled, assembly.AcceptedAssembly)
    output = tmp_path / "maiden.glb"
    assembly.export_scene(compiled, str(output))
    assert sha256(output.read_bytes()).hexdigest() == glb_sha
    assert (
        sha256(Path(f"{output}.materials.json").read_bytes()).hexdigest()
        == sidecar_sha
    )
