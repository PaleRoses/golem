"""Pure exchange projections over the authoritative material catalogue."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from golem.materials.core import (
    DEFAULT_APPEARANCE_MATERIAL_ID,
    AppearanceMaterial,
    AppearanceMaterialId,
)
from golem.materials.decode import resolve_appearance_material
from golem.materials.surface import (
    SeededSurfaceColor,
    derive_linear_vertex_rgb,
    surface_color_to_dict,
)


def appearance_material_to_dict(material: AppearanceMaterial) -> dict[str, object]:
    return {
        "base_color": material.base_color,
        "roughness": material.roughness,
        "metallic": material.metallic,
        "emissive": material.is_emissive,
        "emissive_color": material.emissive_color,
        "emissive_strength": material.emissive_strength,
        "texture_set": material.texture_set,
    }


def project_surface_color_to_face_srgb_bytes(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    vertex_normals: NDArray[np.float64],
    surface_color: SeededSurfaceColor,
) -> NDArray[np.uint8]:
    linear_face_rgb = np.clip(
        np.mean(
            derive_linear_vertex_rgb(
                vertices,
                vertex_normals,
                surface_color,
            )[faces],
            axis=1,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )
    srgb = np.where(
        linear_face_rgb <= 0.0031308,
        12.92 * linear_face_rgb,
        1.055 * np.power(linear_face_rgb, 1.0 / 2.4) - 0.055,
    )
    return np.floor(srgb * 255.0 + 0.5).astype(np.uint8)


def appearance_materials_payload(
    element_materials: Mapping[str, str | AppearanceMaterialId | None],
    surface_colors: Mapping[str, SeededSurfaceColor] | None = None,
) -> dict[str, object]:
    elements = {
        element_id: (
            material_id.value
            if isinstance(material_id, AppearanceMaterialId)
            else material_id or DEFAULT_APPEARANCE_MATERIAL_ID.value
        )
        for element_id, material_id in element_materials.items()
    }
    return {
        "elements": elements,
        "records": {
            material_id: appearance_material_to_dict(
                resolve_appearance_material(material_id)
            )
            for material_id in frozenset(elements.values())
        },
        **(
            {
                "surface_colors": {
                    element_id: surface_color_to_dict(surface_color)
                    for element_id, surface_color in surface_colors.items()
                }
            }
            if surface_colors
            else {}
        ),
    }
