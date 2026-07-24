from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from golem.assembly import AssemblyRecord, AssemblyStratum, ElementRole
from golem.cli import look, main
from golem.materials import AppearanceMaterialId
from golem.materials.projection import project_surface_color_to_face_srgb_bytes
from golem.materials.surface import (
    SeededSurfaceColor,
    SurfaceColorOwner,
    SurfaceColorRecipe,
    SurfaceColorVariation,
    SurfaceColorVariationKind,
    derive_vertex_colors,
)
from golem.senses import orthographic
from golem.senses.raster import RasterFrame


def _tetrahedron_record(
    record_id: str,
    appearance_material: str,
    surface_color: SeededSurfaceColor | None = None,
    x_offset: float = 0.0,
) -> AssemblyRecord:
    return AssemblyRecord.from_derived_geometry(
        record_id=record_id,
        element_id="element",
        role=ElementRole.CREATURE,
        stratum=AssemblyStratum.SOLID,
        appearance_material=appearance_material,
        vertices=np.asarray(
            (
                (x_offset, 0.0, 0.0),
                (x_offset + 0.6, 0.0, 0.0),
                (x_offset, 0.7, 0.0),
                (x_offset, 0.0, 0.8),
            ),
            dtype=np.float64,
        ),
        faces=np.asarray(
            ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
            dtype=np.int64,
        ),
        resolution=60,
        report={},
        surface_color=surface_color,
    )


def _palette_probe_payload() -> dict[str, object]:
    return {
        "name": "palette-probe",
        "pitch": 0.03,
        "elements": [
            {
                "id": "body",
                "role": "creature",
                "graph": {
                    "name": "palette-probe",
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
                "appearance_palette": {
                    "neutral_gray": {
                        "tint_linear_rgb": [0.08, 0.78, 0.16]
                    }
                },
            }
        ],
    }


def _read_rgb_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def test_smooth_face_normals_glue_incident_surface_sections() -> None:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 1, 2), (0, 3, 1)), dtype=np.int64)
    smoothed = orthographic._smooth_face_normals(vertices, faces)
    raw = np.asarray(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))

    assert smoothed.shape == (2, 3)
    assert np.allclose(np.linalg.norm(smoothed, axis=1), 1.0)
    assert float(smoothed[0] @ smoothed[1]) > float(raw[0] @ raw[1])


def test_enhanced_render_supersamples_before_fitted_downsampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_size = 32

    def render_supersampled_frame(
        vertices: np.ndarray,
        faces: np.ndarray,
        azimuth_degrees: float,
        *,
        image_size: int,
        elevation_degrees: float,
        face_colors: np.ndarray | None,
        shade_faces: bool,
    ) -> RasterFrame:
        assert vertices.shape == (3, 3)
        assert faces.shape == (1, 3)
        assert azimuth_degrees == 0.0
        assert elevation_degrees == 0.0
        assert face_colors is not None
        assert image_size == 3 * output_size
        assert shade_faces is False
        rows, columns = np.indices((image_size, image_size))
        body = (
            (np.abs(rows - image_size / 2) < image_size / 4)
            & (np.abs(columns - image_size / 2) < image_size / 8)
        )
        pixels = np.where(
            body[..., np.newaxis],
            np.asarray((96, 112, 128), dtype=np.uint8),
            np.asarray((244, 244, 244), dtype=np.uint8),
        )
        return RasterFrame(pixels, body)

    monkeypatch.setattr(
        orthographic,
        "render_raster_view",
        render_supersampled_frame,
    )
    record = AssemblyRecord.from_derived_geometry(
        record_id="record",
        element_id="element",
        role=ElementRole.CREATURE,
        stratum=AssemblyStratum.SOLID,
        appearance_material="neutral_gray",
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        ),
        faces=np.asarray(((0, 1, 2),)),
        resolution=60,
        report={},
    )

    rendered = orthographic.render_orthographic_views(
        (record,),
        (orthographic.OrthographicView.FRONT,),
        output_size,
        fit_subject=True,
        shaded=True,
    )

    assert isinstance(rendered, orthographic.AcceptedOrthographicRender)
    assert rendered.images[0].pixels.shape == (output_size, output_size, 3)


def test_face_projection_piecewise_encodes_linear_export_colors_to_srgb() -> None:
    surface_color = SeededSurfaceColor(
        SurfaceColorRecipe((0.002, 0.18, 0.50)),
        73,
        SurfaceColorOwner.APPEARANCE_PALETTE,
    )
    record = _tetrahedron_record(
        "variant",
        AppearanceMaterialId.GAMBESON_DARK.value,
        surface_color,
    )
    mesh = trimesh.Trimesh(record.vertices, record.faces, process=False)
    vertex_colors = derive_vertex_colors(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.vertex_normals, dtype=np.float64),
        surface_color,
    )
    projected = project_surface_color_to_face_srgb_bytes(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
        np.asarray(mesh.vertex_normals, dtype=np.float64),
        surface_color,
    )

    assert np.array_equal(
        vertex_colors,
        np.broadcast_to(
            np.asarray((1, 46, 128, 255), dtype=np.uint8),
            vertex_colors.shape,
        ),
    )
    assert np.array_equal(
        projected,
        np.broadcast_to(
            np.asarray((7, 118, 188), dtype=np.uint8),
            projected.shape,
        ),
    )


def test_surface_variation_descends_to_linear_export_and_srgb_projection() -> None:
    surface_color = SeededSurfaceColor(
        SurfaceColorRecipe(
            (0.16, 0.30, 0.10),
            SurfaceColorVariation(
                SurfaceColorVariationKind.TRIPLANAR_VALUE_NOISE,
                0.4,
                0.2,
            ),
        ),
        73,
        SurfaceColorOwner.APPEARANCE_PALETTE,
    )
    record = _tetrahedron_record(
        "variant",
        AppearanceMaterialId.GAMBESON_DARK.value,
        surface_color,
    )
    mesh = trimesh.Trimesh(record.vertices, record.faces, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    assert np.array_equal(
        derive_vertex_colors(vertices, vertex_normals, surface_color),
        np.asarray(
            (
                (41, 77, 26, 255),
                (46, 87, 29, 255),
                (38, 72, 24, 255),
                (37, 69, 23, 255),
            ),
            dtype=np.uint8,
        ),
    )
    assert np.array_equal(
        project_surface_color_to_face_srgb_bytes(
            vertices,
            faces,
            vertex_normals,
            surface_color,
        ),
        np.asarray(
            (
                (113, 151, 90),
                (112, 150, 89),
                (108, 145, 87),
                (111, 148, 89),
            ),
            dtype=np.uint8,
        ),
    )


def test_palette_projection_is_total_over_the_closed_material_vocabulary() -> None:
    material_ids = tuple(AppearanceMaterialId)
    palette_records = tuple(
        _tetrahedron_record(
            material_id.value,
            material_id.value,
            SeededSurfaceColor(
                SurfaceColorRecipe(
                    (
                        (index + 1) / (len(material_ids) + 1),
                        0.25,
                        0.50,
                    )
                ),
                index,
                SurfaceColorOwner.APPEARANCE_PALETTE,
            ),
            1.2 * index,
        )
        for index, material_id in enumerate(material_ids)
    )
    fallback_record = _tetrahedron_record(
        "fallback",
        AppearanceMaterialId.NEUTRAL_GRAY.value,
        x_offset=1.2 * len(material_ids),
    )
    local_record = _tetrahedron_record(
        "local",
        AppearanceMaterialId.GAMBESON_DARK.value,
        SeededSurfaceColor(
            SurfaceColorRecipe((0.95, 0.02, 0.02)),
            11,
            SurfaceColorOwner.LOCAL,
        ),
        1.2 * (len(material_ids) + 1),
    )
    geometry = look._surface_colored_geometry(
        (*palette_records, fallback_record, local_record)
    )

    assert isinstance(geometry, orthographic._CombinedGeometry)
    face_sections = np.split(
        geometry.face_colors,
        np.arange(4, 4 * (len(material_ids) + 2), 4),
    )
    expected_palette_rgb = np.asarray(
        (
            (85, 137, 188),
            (118, 137, 188),
            (143, 137, 188),
            (162, 137, 188),
            (180, 137, 188),
            (195, 137, 188),
            (209, 137, 188),
            (222, 137, 188),
            (233, 137, 188),
            (245, 137, 188),
        ),
        dtype=np.uint8,
    )
    assert all(
        np.array_equal(section, np.broadcast_to(expected, section.shape))
        for section, expected in zip(
            face_sections[: len(material_ids)],
            expected_palette_rgb,
            strict=True,
        )
    )
    assert np.array_equal(
        face_sections[-2],
        np.broadcast_to(
            orthographic._material_rgb(fallback_record),
            face_sections[-2].shape,
        ),
    )
    assert np.array_equal(
        face_sections[-1],
        np.broadcast_to(
            orthographic._material_rgb(local_record),
            face_sections[-1].shape,
        ),
    )


@pytest.mark.parametrize("shaded", (False, True))
@pytest.mark.parametrize("fit_subject", (False, True))
@pytest.mark.parametrize(
    "surface_color",
    (
        None,
        SeededSurfaceColor(
            SurfaceColorRecipe((0.95, 0.02, 0.02)),
            11,
            SurfaceColorOwner.LOCAL,
        ),
    ),
)
def test_palette_absence_is_pixel_byte_identical_to_the_catalogue_renderer(
    shaded: bool,
    fit_subject: bool,
    surface_color: SeededSurfaceColor | None,
) -> None:
    record = _tetrahedron_record(
        "plain",
        AppearanceMaterialId.NEUTRAL_GRAY.value,
        surface_color,
    )
    views = tuple(orthographic.OrthographicView)
    expected = orthographic.render_orthographic_views(
        (record,),
        views,
        48,
        fit_subject=fit_subject,
        shaded=shaded,
    )
    actual = look._render_orthographic_views(
        (record,),
        views,
        48,
        fit_subject=fit_subject,
        shaded=shaded,
    )

    assert isinstance(expected, orthographic.AcceptedOrthographicRender)
    assert isinstance(actual, orthographic.AcceptedOrthographicRender)
    assert tuple(image.pixels.tobytes() for image in actual.images) == tuple(
        image.pixels.tobytes() for image in expected.images
    )


@pytest.mark.parametrize("shaded", (False, True))
def test_palette_render_reaches_every_orthographic_view(shaded: bool) -> None:
    palette_record = _tetrahedron_record(
        "palette",
        AppearanceMaterialId.GAMBESON_DARK.value,
        SeededSurfaceColor(
            SurfaceColorRecipe((0.16, 0.80, 0.10)),
            17,
            SurfaceColorOwner.APPEARANCE_PALETTE,
        ),
    )
    fallback_record = _tetrahedron_record(
        "fallback",
        AppearanceMaterialId.NEUTRAL_GRAY.value,
        x_offset=0.9,
    )
    records = (palette_record, fallback_record)
    views = tuple(orthographic.OrthographicView)
    rendered = look._render_orthographic_views(
        records,
        views,
        64,
        fit_subject=True,
        shaded=shaded,
    )
    catalogue = orthographic.render_orthographic_views(
        records,
        views,
        64,
        fit_subject=True,
        shaded=shaded,
    )

    assert isinstance(rendered, orthographic.AcceptedOrthographicRender)
    assert isinstance(catalogue, orthographic.AcceptedOrthographicRender)
    assert tuple(image.view for image in rendered.images) == views
    assert all(
        not np.array_equal(authored.pixels, fixed.pixels)
        for authored, fixed in zip(
            rendered.images,
            catalogue.images,
            strict=True,
        )
    )


@pytest.mark.parametrize("shaded", (False, True))
def test_maiden_slate_palette_renders_in_a_mid_luminance_band(
    shaded: bool,
) -> None:
    maiden_slate = _tetrahedron_record(
        "maiden-slate",
        AppearanceMaterialId.NEUTRAL_GRAY.value,
        SeededSurfaceColor(
            SurfaceColorRecipe((0.145, 0.185, 0.24)),
            23,
            SurfaceColorOwner.APPEARANCE_PALETTE,
        ),
    )
    rendered = look._render_orthographic_views(
        (maiden_slate,),
        tuple(orthographic.OrthographicView),
        64,
        fit_subject=True,
        shaded=shaded,
    )

    assert isinstance(rendered, orthographic.AcceptedOrthographicRender)
    mean_body_rgb = tuple(
        np.mean(
            image.pixels[np.any(image.pixels != 244, axis=2)],
            axis=0,
        )
        for image in rendered.images
    )
    mean_body_luminance = tuple(
        float(color @ np.asarray((0.2126, 0.7152, 0.0722)))
        for color in mean_body_rgb
    )
    assert all(color[2] > color[1] > color[0] for color in mean_body_rgb)
    assert all(90.0 <= luminance <= 180.0 for luminance in mean_body_luminance)


@pytest.mark.parametrize(
    ("view", "expected_names"),
    (
        (look.LookView.ALL, ("front", "side", "top")),
        (look.LookView.FRONT, ("front",)),
        (look.LookView.SIDE, ("side",)),
        (look.LookView.TOP, ("top",)),
    ),
)
@pytest.mark.parametrize("shaded", (False, True))
def test_look_command_exposes_palette_in_every_view_and_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    view: look.LookView,
    expected_names: tuple[str, ...],
    shaded: bool,
) -> None:
    spec_path = tmp_path / "palette-probe.json"
    output_directory = tmp_path / f"{view.value}-{'shaded' if shaded else 'flat'}"
    spec_path.write_text(json.dumps(_palette_probe_payload()), encoding="utf-8")
    arguments = (
        "look",
        str(spec_path),
        "--view",
        view.value,
        "--res",
        "60",
        "--size",
        "48",
        "--out",
        str(output_directory),
        *(("--shaded",) if shaded else ()),
    )

    exit_code = main(arguments)
    captured = capsys.readouterr()
    paths = tuple(map(Path, filter(None, captured.out.splitlines())))
    body_rgb = tuple(map(_read_rgb_pixels, paths))
    masks = tuple(np.any(pixels != 244, axis=2) for pixels in body_rgb)
    mean_body_rgb = tuple(
        np.mean(pixels[mask], axis=0)
        for pixels, mask in zip(body_rgb, masks, strict=True)
    )

    assert exit_code == 0
    assert captured.err == ""
    assert tuple(path.stem.rsplit("_", 1)[-1] for path in paths) == expected_names
    assert all(color[1] > color[0] and color[1] > color[2] for color in mean_body_rgb)
