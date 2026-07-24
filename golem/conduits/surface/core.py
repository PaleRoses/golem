"""Pure application algebra for authored surface conduits."""

from __future__ import annotations

from functools import reduce
import math

import numpy as np

from golem.conduits.surface import embed
from golem.conduits.surface import emit
from golem.conduits.surface.types import (
    ConduitBand,
    ConduitApplicationResult,
    EmissionPitchObstruction,
)
from golem.kernel.engine.compile import vertex_normals


_AXIS = {"x": 0, "y": 1, "z": 2}


def _find_part(graph: dict, part_id: str) -> dict:
    part = next(
        (
            candidate
            for candidate in graph["parts"]
            if candidate["id"] == part_id
        ),
        None,
    )
    if part is None:
        raise ValueError(f"conduit references unknown part {part_id!r}")
    return part


def _embed_one(
    declaration: dict,
    graph: dict,
    vertices: np.ndarray,
    sdf_tolerance: float,
):
    part = _find_part(graph, declaration["part"])
    match declaration["kind"]:
        case "axial_loop":
            return embed.axial_loop(
                part, vertices, float(declaration["t"]), sdf_tolerance
            )
        case "face_line":
            return embed.face_line(
                part,
                vertices,
                _AXIS[declaration.get("axis", "x")],
                sdf_tolerance,
                declaration.get("face_normal"),
            )
        case unknown_kind:
            raise ValueError(
                f"conduit {declaration.get('id')!r}: unknown kind {unknown_kind!r}"
            )


def _pitch_obstruction(
    declarations: tuple[dict, ...], pitch: float
) -> EmissionPitchObstruction | None:
    maximum_displacement = 0.9 * pitch
    invalid_pitch = not math.isfinite(pitch) or pitch <= 0.0
    if invalid_pitch:
        return EmissionPitchObstruction(
            "<pitch>", "pitch", pitch, maximum_displacement, pitch
        )
    excessive_groove = next(
        (
            declaration
            for declaration in declarations
            if "groove" in declaration.get("emit", ())
            and float(declaration.get("depth", 0.01)) > maximum_displacement
        ),
        None,
    )
    if excessive_groove is not None:
        return EmissionPitchObstruction(
            str(excessive_groove.get("id", "<unknown>")),
            "groove",
            float(excessive_groove.get("depth", 0.01)),
            maximum_displacement,
            pitch,
        )
    excessive_band = next(
        (
            declaration
            for declaration in declarations
            if "band" in declaration.get("emit", ())
            and emit.DEFAULT_BAND_LIFT > maximum_displacement
        ),
        None,
    )
    return (
        EmissionPitchObstruction(
            str(excessive_band.get("id", "<unknown>")),
            "band",
            emit.DEFAULT_BAND_LIFT,
            maximum_displacement,
            pitch,
        )
        if excessive_band is not None
        else None
    )


def apply_conduits(
    graph: dict,
    vertices: np.ndarray,
    faces: np.ndarray,
    declarations: list[dict] | tuple[dict, ...],
    sdf_tol: float = 0.02,
    pitch: float = 0.017,
    flesh_host_vertices: np.ndarray | None = None,
) -> ConduitApplicationResult:
    authored_declarations = tuple(declarations)
    pitch_obstruction = _pitch_obstruction(authored_declarations, pitch)
    if pitch_obstruction is not None:
        return pitch_obstruction
    triangles = np.asarray(faces)
    original_vertices = np.asarray(vertices, dtype=np.float64)
    fixed_flesh_vertices = (
        None
        if flesh_host_vertices is None
        else np.asarray(flesh_host_vertices, dtype=np.float64)
    )

    def apply_groove(
        current_vertices: np.ndarray, declaration: dict
    ) -> np.ndarray:
        if "groove" not in declaration.get("emit", ()):
            return current_vertices
        near, signed_distance = _embed_one(
            declaration,
            graph,
            (
                current_vertices
                if fixed_flesh_vertices is None
                else fixed_flesh_vertices
            ),
            sdf_tol,
        )
        width = float(declaration["width"])
        return emit.groove(
            current_vertices,
            triangles,
            near & (np.abs(signed_distance) < width / 2.0),
            signed_distance,
            width,
            float(declaration.get("depth", 0.01)),
        )

    displaced_vertices = reduce(
        apply_groove,
        authored_declarations,
        np.array(original_vertices, copy=True),
    )
    maximum_displacement = 0.9 * pitch
    realized_displacement = (
        float(
            np.max(
                np.linalg.norm(
                    displaced_vertices - original_vertices,
                    axis=1,
                )
            )
        )
        if original_vertices.shape[0]
        else 0.0
    )
    if realized_displacement > maximum_displacement + 1.0e-12:
        return EmissionPitchObstruction(
            "<combined>",
            "groove_fold",
            realized_displacement,
            maximum_displacement,
            pitch,
        )
    band_declarations = tuple(
        declaration
        for declaration in authored_declarations
        if "band" in declaration.get("emit", ())
    )
    final_normals = (
        vertex_normals(displaced_vertices, triangles)
        if band_declarations
        else np.empty_like(displaced_vertices)
    )

    def emit_band(declaration: dict) -> ConduitBand:
        near, signed_distance = _embed_one(
            declaration,
            graph,
            (
                displaced_vertices
                if fixed_flesh_vertices is None
                else fixed_flesh_vertices
            ),
            sdf_tol,
        )
        band_vertices, band_triangles = emit.band_faces(
            displaced_vertices,
            triangles,
            signed_distance,
            float(declaration["width"]),
            near,
            normals=final_normals,
            face_normal=declaration.get("face_normal"),
        )
        empty = band_vertices is None
        return {
            "id": declaration["id"],
            "verts": None if empty else band_vertices,
            "faces": None if empty else band_triangles,
            "material": declaration.get("material", "emissive_seam"),
            "empty": empty,
        }

    return displaced_vertices, tuple(map(emit_band, band_declarations))


__all__ = ["apply_conduits"]
