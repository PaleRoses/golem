"""Scene export for the assembly subsystem.

One GLB scene, one named geometry per element, each carrying a REAL glTF PBR
material resolved from the element's tag (base color, roughness, metallic,
emissive factors -- factor-only materials, no textures yet; the materials
subsystem's ``texture_set`` hook is where generated map sets will attach).
A JSON sidecar (``<out>.materials.json``) carries element->tag and the full
records so presentation backends and target engines can shade without
importing this package.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from golem.assembly.core import AcceptedAssembly, AssemblyRecord
from golem.materials.core import AppearanceMaterialId
from golem.materials.decode import resolve_appearance_material
from golem.materials.projection import appearance_materials_payload
from golem.materials.surface import (
    EmissiveSurfaceClass,
    OpaqueSurfaceClass,
    SeededSurfaceColor,
    TranslucentSurfaceClass,
    derive_vertex_colors,
)


def _pbr_material(
    tag: str,
    vertex_colored: bool = False,
    surface_color: SeededSurfaceColor | None = None,
):
    from trimesh.visual.material import PBRMaterial

    material = resolve_appearance_material(tag)
    surface_class = (
        surface_color.surface_class
        if surface_color is not None
        else OpaqueSurfaceClass()
    )
    opacity = (
        surface_class.opacity
        if isinstance(surface_class, TranslucentSurfaceClass)
        else 1.0
    )
    base = (
        [1.0, 1.0, 1.0, opacity]
        if vertex_colored
        else list(material.base_color) + [opacity]
    )
    emissive = (
        [
            component * min(surface_class.intensity, 1.0)
            for component in surface_color.recipe.tint_linear_rgb
        ]
        if isinstance(surface_class, EmissiveSurfaceClass)
        else (
            [
                component * min(material.emissive_strength, 1.0)
                for component in material.emissive_color
            ]
            if material.is_emissive
            else [0.0, 0.0, 0.0]
        )
    )
    return PBRMaterial(
        name=tag,
        baseColorFactor=base,
        roughnessFactor=material.roughness,
        metallicFactor=material.metallic,
        emissiveFactor=emissive,
        **(
            {"alphaMode": "BLEND"}
            if isinstance(surface_class, TranslucentSurfaceClass)
            else {}
        ),
    )


def export_scene(assembly: AcceptedAssembly, path: str) -> None:
    """Export derived views from an already accepted authoritative assembly."""
    import trimesh

    emitted = tuple(
        map(
            _exchange_geometry,
            filter(lambda record: not record.empty, assembly.records),
        )
    )
    scene = trimesh.Scene(
        geometry={
            element_id: mesh
            for element_id, _tag, _surface_color, mesh in emitted
        }
    )
    tags = {
        element_id: tag
        for element_id, tag, _surface_color, _mesh in emitted
    }
    surface_colors = {
        element_id: surface_color
        for element_id, _tag, surface_color, _mesh in emitted
        if surface_color is not None
    }
    scene.export(path, include_normals=True)
    write_sidecar(tags, f"{path}.materials.json", surface_colors)


def write_sidecar(
    element_materials: Mapping[str, str | AppearanceMaterialId | None],
    path: str | Path,
    surface_colors: Mapping[str, SeededSurfaceColor] | None = None,
) -> None:
    Path(path).write_text(
        json.dumps(
            appearance_materials_payload(element_materials, surface_colors),
            indent=1,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _exchange_geometry(record: AssemblyRecord):
    """Isolate trimesh's mutable object protocol at the exchange boundary."""
    import trimesh
    from trimesh.visual import TextureVisuals

    tag = record.appearance_material
    mesh = trimesh.Trimesh(record.vertices, record.faces, process=False)
    surface_color = record.surface_color
    visual = TextureVisuals(
        material=_pbr_material(
            tag,
            vertex_colored=surface_color is not None,
            surface_color=surface_color,
        )
    )
    if surface_color is not None:
        visual.vertex_attributes["color"] = derive_vertex_colors(
            mesh.vertices,
            mesh.vertex_normals,
            surface_color,
        )
    mesh.visual = visual
    mesh.metadata["golem_element"] = {
        "id": record.record_id,
        "element_id": record.element_id,
        "role": record.role,
        "appearance_material": tag,
    }
    return record.record_id, tag, surface_color, mesh
