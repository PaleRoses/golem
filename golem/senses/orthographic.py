from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import accumulate
from pathlib import Path
from typing import assert_never

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

from golem.assembly import AssemblyRecord
from golem.materials import resolve_appearance_material
from golem.senses.raster import render_raster_view


DEFAULT_ORTHOGRAPHIC_IMAGE_SIZE = 256
_RIM_LIGHT_RGB = np.asarray((196.0, 218.0, 255.0), dtype=np.float64)
_QUALITY_KEY_LIGHT = np.asarray((0.35, 0.75, 0.56), dtype=np.float64)
_SUBJECT_FRAME_FRACTION = 0.72
_SUPERSAMPLING_FACTOR = 3


class OrthographicView(StrEnum):
    FRONT = "front"
    SIDE = "side"
    TOP = "top"


@dataclass(frozen=True)
class OrthographicImage:
    view: OrthographicView
    pixels: NDArray[np.uint8]


@dataclass(frozen=True)
class AcceptedOrthographicRender:
    images: tuple[OrthographicImage, ...]


@dataclass(frozen=True)
class OrthographicRenderObstruction:
    address: str
    reason: str


type OrthographicRenderResult = (
    AcceptedOrthographicRender | OrthographicRenderObstruction
)


@dataclass(frozen=True)
class AcceptedOrthographicWrite:
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class OrthographicWriteObstruction:
    path: Path
    reason: str


type OrthographicWriteResult = (
    AcceptedOrthographicWrite | OrthographicWriteObstruction
)


@dataclass(frozen=True)
class _CombinedGeometry:
    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]
    face_colors: NDArray[np.uint8]


@dataclass(frozen=True)
class _RenderableRecord:
    record: AssemblyRecord
    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]


def _renderable_record(record: AssemblyRecord) -> _RenderableRecord | None:
    return (
        _RenderableRecord(
            record,
            np.asarray(record.vertices, dtype=np.float64),
            np.asarray(record.faces, dtype=np.int64),
        )
        if not record.empty
        and record.vertices is not None
        and record.faces is not None
        and len(record.vertices) > 0
        and len(record.faces) > 0
        else None
    )


def _material_rgb(record: AssemblyRecord) -> NDArray[np.uint8]:
    material = resolve_appearance_material(record.appearance_material)
    linear_rgb = np.asarray(material.base_color, dtype=np.float64) + (
        np.asarray(material.emissive_color, dtype=np.float64)
        * min(material.emissive_strength, 1.0)
    )
    return np.rint(
        255.0 * np.power(np.clip(linear_rgb, 0.0, 1.0), 1.0 / 2.2)
    ).astype(np.uint8)


def _combine_records(
    records: tuple[AssemblyRecord, ...],
) -> _CombinedGeometry | OrthographicRenderObstruction:
    visible = tuple(
        candidate
        for candidate in map(_renderable_record, records)
        if candidate is not None
    )
    if not visible:
        return OrthographicRenderObstruction(
            "assembly/records",
            "accepted assembly contains no renderable solid or stratum records",
        )
    vertex_counts = tuple(len(record.vertices) for record in visible)
    offsets = tuple(accumulate(vertex_counts, initial=0))[:-1]
    return _CombinedGeometry(
        vertices=np.concatenate(
            tuple(record.vertices for record in visible)
        ),
        faces=np.concatenate(
            tuple(
                record.faces + offset
                for record, offset in zip(visible, offsets, strict=True)
            )
        ),
        face_colors=np.concatenate(
            tuple(
                np.broadcast_to(
                    _material_rgb(record.record),
                    (len(record.faces), 3),
                )
                for record in visible
            )
        ),
    )


def _view_angles(view: OrthographicView) -> tuple[float, float]:
    match view:
        case OrthographicView.FRONT:
            return 0.0, 0.0
        case OrthographicView.SIDE:
            return 90.0, 0.0
        case OrthographicView.TOP:
            return 0.0, 90.0
        case _ as unreachable:
            assert_never(unreachable)


def _render_view(
    geometry: _CombinedGeometry,
    view: OrthographicView,
    image_size: int,
    shaded: bool,
    fit_subject: bool,
) -> OrthographicImage:
    azimuth_degrees, elevation_degrees = _view_angles(view)
    face_colors = (
        _quality_face_colors(
            geometry,
            azimuth_degrees,
            elevation_degrees,
        )
        if shaded
        else geometry.face_colors
    )
    enhanced = shaded or fit_subject
    render_size = (
        image_size * _SUPERSAMPLING_FACTOR if enhanced else image_size
    )
    frame = render_raster_view(
        geometry.vertices,
        geometry.faces,
        azimuth_degrees,
        image_size=render_size,
        elevation_degrees=elevation_degrees,
        face_colors=face_colors,
        shade_faces=not shaded,
    )
    shaded_pixels = (
        _enhance_shading(
            _smooth_surface_shading(frame.pixels, frame.body, render_size),
            frame.body,
            render_size,
        )
        if shaded
        else frame.pixels
    )
    framed_pixels = (
        _fit_subject(shaded_pixels, frame.body) if fit_subject else shaded_pixels
    )
    return OrthographicImage(
        view,
        _downsample(framed_pixels, image_size) if enhanced else framed_pixels,
    )


def _unit_vectors(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    return vectors / np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True),
        1.0e-12,
    )


def _smooth_face_normals(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
) -> NDArray[np.float64]:
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    vertex_indices = faces.reshape(-1)
    incident_normals = np.repeat(face_normals, 3, axis=0)
    vertex_normals = np.column_stack(
        tuple(
            np.bincount(
                vertex_indices,
                weights=incident_normals[:, component],
                minlength=len(vertices),
            )
            for component in range(3)
        )
    )
    return _unit_vectors(
        np.mean(_unit_vectors(vertex_normals)[faces], axis=1)
    )


def _quality_face_colors(
    geometry: _CombinedGeometry,
    azimuth_degrees: float,
    elevation_degrees: float,
) -> NDArray[np.uint8]:
    azimuth = np.radians(azimuth_degrees)
    elevation = np.radians(elevation_degrees)
    yaw = np.asarray(
        (
            (np.cos(azimuth), 0.0, np.sin(azimuth)),
            (0.0, 1.0, 0.0),
            (-np.sin(azimuth), 0.0, np.cos(azimuth)),
        ),
        dtype=np.float64,
    )
    pitch = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, np.cos(elevation), -np.sin(elevation)),
            (0.0, np.sin(elevation), np.cos(elevation)),
        ),
        dtype=np.float64,
    )
    view_vertices = geometry.vertices @ (pitch @ yaw).T
    unit_normals = _smooth_face_normals(view_vertices, geometry.faces)
    key_light = _QUALITY_KEY_LIGHT / np.linalg.norm(_QUALITY_KEY_LIGHT)
    diffuse_lift = 0.38 + 0.50 * np.abs(unit_normals @ key_light)
    rim_lift = 0.16 * np.square(1.0 - np.abs(unit_normals[:, 2]))
    return np.rint(
        np.clip(
            np.asarray(geometry.face_colors, dtype=np.float64)
            * (diffuse_lift + rim_lift)[:, np.newaxis],
            0.0,
            255.0,
        )
    ).astype(np.uint8)


def _smooth_surface_shading(
    pixels: NDArray[np.uint8],
    body: NDArray[np.bool_],
    image_size: int,
) -> NDArray[np.uint8]:
    sigma = max(image_size * 0.003, 1.0)
    weights = gaussian_filter(
        np.asarray(body, dtype=np.float64),
        sigma=sigma,
    )
    smoothed = gaussian_filter(
        np.asarray(pixels, dtype=np.float64) * body[..., np.newaxis],
        sigma=(sigma, sigma, 0.0),
    ) / np.maximum(weights[..., np.newaxis], 1.0e-12)
    source = np.asarray(pixels, dtype=np.float64)
    color_delta = np.linalg.norm(smoothed - source, axis=2)
    smoothing_weight = np.exp(-np.square(color_delta / 48.0))[..., np.newaxis]
    blended = source * (1.0 - smoothing_weight) + smoothed * smoothing_weight
    return np.where(
        body[..., np.newaxis],
        np.rint(blended).astype(np.uint8),
        pixels,
    )


def _fit_subject(
    pixels: NDArray[np.uint8],
    body: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    occupied_rows, occupied_columns = np.nonzero(body)
    if occupied_rows.size == 0:
        return pixels
    subject_width = int(np.ptp(occupied_columns)) + 1
    subject_height = int(np.ptp(occupied_rows)) + 1
    crop_size = min(
        pixels.shape[0],
        int(
            np.ceil(
                max(subject_width, subject_height) / _SUBJECT_FRAME_FRACTION
            )
        ),
    )
    center_x = 0.5 * (
        int(np.min(occupied_columns)) + int(np.max(occupied_columns)) + 1
    )
    center_y = 0.5 * (
        int(np.min(occupied_rows)) + int(np.max(occupied_rows)) + 1
    )
    left = int(
        np.clip(
            round(center_x - crop_size / 2),
            0,
            pixels.shape[1] - crop_size,
        )
    )
    top = int(
        np.clip(
            round(center_y - crop_size / 2),
            0,
            pixels.shape[0] - crop_size,
        )
    )
    return pixels[top : top + crop_size, left : left + crop_size]


def _downsample(
    pixels: NDArray[np.uint8],
    image_size: int,
) -> NDArray[np.uint8]:
    return np.asarray(
        Image.fromarray(pixels).resize(
            (image_size, image_size),
            resample=Image.Resampling.LANCZOS,
        ),
        dtype=np.uint8,
    )


def _enhance_shading(
    pixels: NDArray[np.uint8],
    body: NDArray[np.bool_],
    image_size: int,
) -> NDArray[np.uint8]:
    normalized = np.asarray(pixels, dtype=np.float64) / 255.0
    contrasted = np.clip((normalized - 0.5) * 1.12 + 0.5, 0.0, 1.0)
    rim_width = max(image_size * 0.018, 1.0)
    rim_weight = (
        0.42
        * np.square(
            np.clip(
                1.0 - (distance_transform_edt(body) - 1.0) / rim_width,
                0.0,
                1.0,
            )
        )
    )[..., np.newaxis]
    lit = contrasted * (1.0 - rim_weight) + (
        (_RIM_LIGHT_RGB / 255.0) * rim_weight
    )
    return np.where(
        body[..., np.newaxis],
        np.rint(255.0 * lit).astype(np.uint8),
        pixels,
    )


def render_orthographic_views(
    records: tuple[AssemblyRecord, ...],
    views: tuple[OrthographicView, ...],
    image_size: int = DEFAULT_ORTHOGRAPHIC_IMAGE_SIZE,
    *,
    fit_subject: bool = False,
    shaded: bool = False,
) -> OrthographicRenderResult:
    if image_size < 1:
        return OrthographicRenderObstruction(
            "render/image_size",
            f"image size must be positive, received {image_size}",
        )
    geometry = _combine_records(records)
    return (
        geometry
        if isinstance(geometry, OrthographicRenderObstruction)
        else AcceptedOrthographicRender(
            tuple(
                _render_view(
                    geometry,
                    view,
                    image_size,
                    shaded,
                    fit_subject,
                )
                for view in views
            )
        )
    )


def _write_image(
    image: OrthographicImage,
    output_directory: Path,
    output_stem: str,
) -> Path:
    path = output_directory / f"{output_stem}_{image.view.value}.png"
    Image.fromarray(image.pixels).save(path)
    return path


def write_orthographic_views(
    rendered: AcceptedOrthographicRender,
    output_directory: Path,
    output_stem: str,
) -> OrthographicWriteResult:
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        return AcceptedOrthographicWrite(
            tuple(
                _write_image(image, output_directory, output_stem)
                for image in rendered.images
            )
        )
    except (OSError, ValueError) as failure:
        return OrthographicWriteObstruction(
            output_directory,
            f"{type(failure).__name__}: {failure}",
        )
