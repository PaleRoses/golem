"""Shared raster algebra for canonical GOLEM measurements."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from render import render_view as _render_view


_BACKGROUND_RGB = (244, 244, 244)
_FLAT_TINT = (168, 172, 182)


class CanonicalView(StrEnum):
    FRONT = "front"
    THREE_QUARTER = "three-quarter"
    SIDE = "side"
    BACK = "back"

    @property
    def azimuth_degrees(self) -> int:
        match self:
            case CanonicalView.FRONT:
                return 0
            case CanonicalView.THREE_QUARTER:
                return 40
            case CanonicalView.SIDE:
                return 90
            case CanonicalView.BACK:
                return 180


@dataclass(frozen=True)
class RasterFrame:
    pixels: NDArray[np.uint8]
    body: NDArray[np.bool_]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pixels", _read_only_view(self.pixels))
        object.__setattr__(self, "body", _read_only_view(self.body))


def render_raster_view(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    azimuth_degrees: float,
    *,
    image_size: int,
    elevation_degrees: float = 12.0,
    face_colors: NDArray[np.uint8] | None = None,
    shade_faces: bool = True,
) -> RasterFrame:
    pixels = np.asarray(
        _render_view(
            vertices,
            faces,
            azimuth_degrees,
            elev_deg=elevation_degrees,
            size=image_size,
            face_colors=face_colors,
        )
        if shade_faces
        else _render_flat_view(
            vertices,
            faces,
            azimuth_degrees,
            elevation_degrees=elevation_degrees,
            image_size=image_size,
            face_colors=face_colors,
        ),
        dtype=np.uint8,
    )
    return RasterFrame(
        pixels=pixels,
        body=body_mask(pixels),
    )


def _render_flat_view(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    azimuth_degrees: float,
    *,
    elevation_degrees: float,
    image_size: int,
    face_colors: NDArray[np.uint8] | None,
) -> Image.Image:
    # The pilot oracle is hash-frozen, so the unshaded variant of its
    # projection is owned here rather than threaded through its signature.
    azimuth = np.radians(azimuth_degrees)
    elevation = np.radians(elevation_degrees)
    cos_a, sin_a = np.cos(azimuth), np.sin(azimuth)
    cos_e, sin_e = np.cos(elevation), np.sin(elevation)
    rotate_y = np.array([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]])
    rotate_x = np.array([[1.0, 0.0, 0.0], [0.0, cos_e, -sin_e], [0.0, sin_e, cos_e]])
    view = vertices @ (rotate_x @ rotate_y).T
    view = view - (view.max(axis=0) + view.min(axis=0)) / 2.0
    scale = 0.44 * image_size / max(np.ptp(view[:, 0]), np.ptp(view[:, 1]))
    columns = view[:, 0] * scale + image_size / 2
    rows = image_size / 2 - view[:, 1] * scale
    projected = np.stack([columns[faces], rows[faces]], axis=2)
    order = np.argsort(view[:, 2][faces].mean(axis=1))
    colors = (
        np.tile(np.asarray(_FLAT_TINT, dtype=np.int64), (len(faces), 1))
        if face_colors is None
        else np.asarray(face_colors, dtype=np.int64)[:, :3]
    )
    image = Image.fromarray(
        np.full((image_size, image_size, 3), _BACKGROUND_RGB, dtype=np.uint8)
    )
    draw = ImageDraw.Draw(image)
    tuple(
        draw.polygon(
            tuple(map(tuple, projected[index])),
            fill=tuple(int(channel) for channel in colors[index]),
        )
        for index in order
    )
    return image


def body_mask(pixels: NDArray[np.uint8]) -> NDArray[np.bool_]:
    return np.any(
        pixels != np.asarray(_BACKGROUND_RGB, dtype=np.uint8),
        axis=2,
    )


def intersection_over_union(
    left: NDArray[np.bool_], right: NDArray[np.bool_]
) -> float:
    union_count = int(np.count_nonzero(left | right))
    return (
        float(np.count_nonzero(left & right) / union_count)
        if union_count > 0
        else 0.0
    )


def safe_mean(values: NDArray[np.generic]) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _read_only_view[value: np.generic](
    value: NDArray[value],
) -> NDArray[value]:
    result = np.asarray(value)
    result.setflags(write=False)
    return result
