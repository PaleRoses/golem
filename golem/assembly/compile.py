"""Element compilation coalgebra: meshing, plate policy, appendages, vascular records."""

from __future__ import annotations

import difflib
from dataclasses import asdict, replace
from itertools import chain as _chain
import numpy as np
from typing import TYPE_CHECKING

from golem.assembly.mounts import (concretize_mirrors, mount_graph, mount_vascular_graph)
from golem.kernel import engine as E
from golem.kernel.anatomy import (AcceptedAnatomy, AcceptedVasculature, ClosedVascularGraph, VascularStratum, VasculatureResult, anatomy_to_dict, realize_vasculature, vasculature_to_dict)

from golem.assembly.carriers import (AssemblyRecord, AssemblyResolutionPolicy, AssemblyStratum, ElementRole, PinnedResolution, RES_MAX, RES_MIN, TargetPitch, _CompiledElement, _LoadedElement)
from golem.assembly.obstructions import (AppearancePaletteElementObstruction, ElementSurfaceDetailObstruction, ElementSurfaceFormationObstruction, EmptySurfaceConduitObstruction, MaskedIntegrityEvidence, PlateElementObstruction, SolidIntegrityObstruction, SurfaceColorElementObstruction, UnknownAppendageIntentObstruction)
from golem.conduits.surface.types import ConduitApplicationResult, EmissionPitchObstruction
from golem.kernel.body.eyes import eye_globe_graph
from golem.kernel.body.types import EyeGlobe
from golem.materials.surface import (AcceptedAppearancePalette, AcceptedSurfaceColor, AppearancePaletteObstruction, AppearancePaletteRule, RejectedAppearancePalette, RejectedSurfaceColor, SeededAppearancePalette, decode_appearance_palette, decode_surface_color, seed_appearance_palette, seed_surface_color)

if TYPE_CHECKING:
    from golem.plates.core import PlateSurface


def _record_with_appearance_palette(
    record: AssemblyRecord,
    palette: SeededAppearancePalette,
) -> AssemblyRecord:
    palette_color = palette.color_for(record.appearance_material)
    return (
        record
        if record.surface_color is not None or palette_color is None
        else replace(record, surface_color=palette_color)
    )


def apply_appearance_palette(
    records: tuple[AssemblyRecord, ...],
    palette: SeededAppearancePalette | None,
) -> tuple[AssemblyRecord, ...]:
    return (
        records
        if palette is None
        else tuple(
            map(
                lambda record: _record_with_appearance_palette(record, palette),
                records,
            )
        )
    )


def pitch_res(graph: dict, pitch: float) -> int:
    """Meshing res targeting a uniform world ``pitch`` on the element's own
    (padded) bounds -- D12's uniform-pitch rule."""
    lo, hi = E.graph_bounds(graph)
    span = float(np.max(hi - lo))
    return int(np.clip(int(np.ceil(span / pitch)) + 1, RES_MIN, RES_MAX))


def _resolution_for(
    graph: dict,
    policy: AssemblyResolutionPolicy,
) -> int:
    return (
        policy.resolution
        if isinstance(policy, PinnedResolution)
        else pitch_res(graph, policy.pitch)
    )


def _compiled_integrity_obstructions(
    compiled: tuple[_CompiledElement, ...],
    resolution_policy: AssemblyResolutionPolicy,
) -> tuple[SolidIntegrityObstruction | EmptySurfaceConduitObstruction, ...]:
    records = tuple(
        _chain.from_iterable(element.records for element in compiled)
    )
    return (
        *tuple(
            SolidIntegrityObstruction(
                element_id=record.element_id,
                component_count=int(record.report["components"]),
                watertight=bool(record.report["watertight_main"]),
                vocabulary_violation_count=len(record.violations),
                violations=record.violations,
            )
            for record in records
            if record.stratum is AssemblyStratum.SOLID
            and (
                int(record.report["components"]) != 1
                or record.report["watertight_main"] is not True
                or (
                    isinstance(resolution_policy, TargetPitch)
                    and bool(record.violations)
                )
            )
        ),
        *tuple(
            EmptySurfaceConduitObstruction(record.record_id)
            for record in records
            if record.stratum is AssemblyStratum.CONDUIT and record.empty
        ),
    )


def _compile_element(
    loaded: _LoadedElement,
    resolution_policy: AssemblyResolutionPolicy,
    source_digest: str,
) -> _CompiledElement | ElementSurfaceDetailObstruction | ElementSurfaceFormationObstruction | PlateElementObstruction | SurfaceColorElementObstruction | AppearancePaletteElementObstruction | UnknownAppendageIntentObstruction | EmissionPitchObstruction:
    entry = loaded.entry
    surface_color_result = (
        decode_surface_color(
            entry["surface_color"],
            f"/elements/{entry['id']}/surface_color",
        )
        if "surface_color" in entry
        else None
    )
    if isinstance(surface_color_result, RejectedSurfaceColor):
        return SurfaceColorElementObstruction(
            str(entry["id"]),
            surface_color_result.obstructions,
        )
    surface_color = (
        seed_surface_color(
            surface_color_result.recipe,
            source_digest,
            str(entry["id"]),
        )
        if isinstance(surface_color_result, AcceptedSurfaceColor)
        else None
    )
    appearance_material = str(
        entry.get("appearance_material", "neutral_gray")
    )
    appearance_palette_result = (
        decode_appearance_palette(
            entry["appearance_palette"],
            f"/elements/{entry['id']}/appearance_palette",
        )
        if "appearance_palette" in entry
        else None
    )
    if isinstance(appearance_palette_result, RejectedAppearancePalette):
        return AppearancePaletteElementObstruction(
            str(entry["id"]),
            appearance_palette_result.obstructions,
        )
    appearance_palette = (
        seed_appearance_palette(
            appearance_palette_result.palette,
            source_digest,
            str(entry["id"]),
        )
        if isinstance(appearance_palette_result, AcceptedAppearancePalette)
        else None
    )
    if (
        surface_color is not None
        and appearance_palette is not None
        and appearance_palette.color_for(appearance_material) is not None
    ):
        return AppearancePaletteElementObstruction(
            str(entry["id"]),
            (
                AppearancePaletteObstruction(
                    (
                        f"/elements/{entry['id']}/appearance_palette/"
                        f"{appearance_material}"
                    ),
                    AppearancePaletteRule.DISTINCT_LOCAL_COLOR_OWNER,
                    {
                        "appearance_palette_material": appearance_material,
                        "surface_color": True,
                    },
                    {"maximum_color_owner_count": 1},
                ),
            ),
        )
    role = ElementRole(
        str(entry.get("role", ElementRole.EQUIPMENT.value))
    )
    source_graph_result = _resolve_appendage_intents(
        entry["id"], loaded.source_graph
    )
    if isinstance(source_graph_result, UnknownAppendageIntentObstruction):
        return source_graph_result
    source_graph = source_graph_result
    anatomy = loaded.anatomy
    vascular_result: VasculatureResult | None = (
        realize_vasculature(anatomy, source_graph)
        if isinstance(anatomy, AcceptedAnatomy)
        else None
    )
    expanded_graph = _expand_appendages(source_graph)
    graph = mount_graph(
        concretize_mirrors(expanded_graph), entry.get("mount")
    )
    mounted_vasculature = (
        mount_vascular_graph(vascular_result.graph, entry.get("mount"))
        if isinstance(vascular_result, AcceptedVasculature)
        else None
    )
    res = _resolution_for(graph, resolution_policy)
    evaluated = E.evaluate(graph, res=res)
    if isinstance(evaluated, E.RejectedSurfaceFormation):
        # Wall 005: the rejection must not silently mask downstream
        # evidence. Vocabulary violations are a pure function of graph +
        # resolution and are carried as visible evidence; the mesh-derived
        # integrity checks are declared masked (no surface, no mesh),
        # never faked.
        return ElementSurfaceFormationObstruction(
            str(entry["id"]),
            evaluated.obstructions,
            vocabulary_violations=tuple(E.vocab_violations(graph, res=res)),
            masked_integrity=MaskedIntegrityEvidence(
                ("component_count", "watertight"),
                "surface formation rejected before a mesh exists; "
                "mesh-derived integrity evidence is not computable",
            ),
        )
    provisional_vertices = evaluated.vertices
    faces = evaluated.faces
    surface_detail_result = _apply_surface_detail(
        entry,
        role,
        provisional_vertices,
        faces,
        evaluated.normals,
        evaluated.maximum_pitch,
    )
    if isinstance(surface_detail_result, ElementSurfaceDetailObstruction):
        return surface_detail_result
    detailed_vertices, detailed_normals, surface_detail_report = surface_detail_result
    plate_result = _apply_plate_policy(
        entry,
        graph,
        detailed_vertices,
        faces,
        detailed_normals,
        mounted_vasculature,
    )
    if isinstance(plate_result, PlateElementObstruction):
        return plate_result
    plated_vertices, plate_surface, plate_report = plate_result
    surface_result = _apply_surface_conduits(
        graph,
        plated_vertices,
        faces,
        (
            resolution_policy.pitch
            if isinstance(resolution_policy, TargetPitch)
            else evaluated.maximum_pitch
        ),
        (
            evaluated.skin_correspondence.flesh_vertices
            if evaluated.skin_correspondence is not None
            else None
        ),
    )
    if isinstance(surface_result, EmissionPitchObstruction):
        return surface_result
    vertices, bands = surface_result
    vascular_view = (
        vasculature_to_dict(vascular_result)
        if vascular_result is not None
        else None
    )
    anatomy_view = anatomy_to_dict(anatomy) if anatomy is not None else None
    report = {
        **E.coherence_report(vertices, faces),
        "maximum_pitch": evaluated.maximum_pitch,
        "surface_conduit_displaced_vertices": int(
            np.count_nonzero(
                np.linalg.norm(vertices - plated_vertices, axis=1) > 1.0e-12
            )
        ),
        **({"anatomy": anatomy_view} if anatomy_view is not None else {}),
        **({"vasculature": vascular_view} if vascular_view is not None else {}),
        **(
            {"skin_formation": asdict(evaluated.skin_evidence)}
            if evaluated.skin_evidence is not None
            else {}
        ),
        **(
            {"plates": plate_report}
            if plate_report is not None
            else {}
        ),
        **(
            {"surface_detail": surface_detail_report}
            if surface_detail_report is not None
            else {}
        ),
    }
    solid = AssemblyRecord.from_derived_geometry(
        record_id=entry["id"],
        element_id=entry["id"],
        role=role,
        stratum=AssemblyStratum.SOLID,
        appearance_material=appearance_material,
        vertices=vertices,
        faces=faces,
        resolution=res,
        report=report,
        violations=E.vocab_violations(graph, res=res),
        surface_color=surface_color,
    )
    band_records = tuple(
        _surface_band_record(entry["id"], role, res, band) for band in bands
    )
    plate_records = (
        (_plate_seam_record(entry["id"], role, res, plate_surface, vertices),)
        if plate_surface is not None
        else ()
    )
    eye_records = derive_eye_records(
        entry["id"],
        role,
        loaded.eye_globes,
        entry.get("mount"),
    )
    vascular_records = (
        derive_vascular_records(
            entry["id"],
            role,
            res,
            mounted_vasculature,
            (
                resolution_policy.pitch
                if isinstance(resolution_policy, TargetPitch)
                else evaluated.maximum_pitch
            ),
        )
        if mounted_vasculature is not None
        else ()
    )
    records = apply_appearance_palette(
        (solid, *eye_records, *band_records, *plate_records, *vascular_records),
        appearance_palette,
    )
    return _CompiledElement(
        element_id=entry["id"],
        role=role,
        entry=entry,
        source_graph=source_graph,
        graph=graph,
        records=records,
        evaluated=evaluated,
        anatomy=anatomy,
        vasculature=vascular_result,
        mounted_vasculature=mounted_vasculature,
        appearance_palette=appearance_palette,
    )


def _apply_surface_detail(
    entry: dict,
    role: ElementRole,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    pitch: float,
) -> (
    tuple[np.ndarray, np.ndarray, dict[str, object] | None]
    | ElementSurfaceDetailObstruction
):
    payload = entry.get("surface_detail")
    if payload is None:
        return vertices, normals, None
    decoded = E.decode_surface_detail(payload)
    if isinstance(decoded, E.RejectedSurfaceDetail):
        return ElementSurfaceDetailObstruction(
            str(entry["id"]), decoded.obstructions
        )
    if role is not ElementRole.CREATURE:
        return ElementSurfaceDetailObstruction(
            str(entry["id"]),
            (
                E.SurfaceDetailObstruction(
                    "/surface_detail",
                    E.SurfaceDetailRule.CREATURE_ROLE,
                    role.value,
                    ElementRole.CREATURE.value,
                ),
            ),
        )
    detail = decoded.detail
    maximum_displacement = 0.9 * pitch
    maximum_scale = float(
        np.linalg.norm(np.max(vertices, axis=0) - np.min(vertices, axis=0))
    )
    if detail.amplitude > maximum_displacement:
        return ElementSurfaceDetailObstruction(
            str(entry["id"]),
            (
                E.SurfaceDetailObstruction(
                    "/surface_detail/amplitude",
                    E.SurfaceDetailRule.AMPLITUDE_PITCH_LIMIT,
                    detail.amplitude,
                    {
                        "maximum": maximum_displacement,
                        "pitch": pitch,
                        "pitch_factor": 0.9,
                    },
                ),
            ),
        )
    if detail.scale > maximum_scale:
        return ElementSurfaceDetailObstruction(
            str(entry["id"]),
            (
                E.SurfaceDetailObstruction(
                    "/surface_detail/scale",
                    E.SurfaceDetailRule.SCALE_MESH_DIAGONAL,
                    detail.scale,
                    {"maximum": maximum_scale},
                ),
            ),
        )
    if detail.amplitude == 0.0:
        return vertices, normals, None
    displaced = E.apply_curvature_surface_detail(vertices, faces, detail)
    signed = displaced.signed_displacements
    active_tolerance = 1.0e-12
    return (
        displaced.vertices,
        E.vertex_normals(displaced.vertices, faces),
        {
            "kind": detail.kind.value,
            "amplitude": detail.amplitude,
            "scale": detail.scale,
            "maximum_displacement": float(np.max(np.abs(signed))),
            "displaced_vertices": int(
                np.count_nonzero(np.abs(signed) > active_tolerance)
            ),
            "convex_vertices": int(np.count_nonzero(signed > active_tolerance)),
            "concave_vertices": int(np.count_nonzero(signed < -active_tolerance)),
        },
    )


def _apply_plate_policy(
    entry: dict,
    graph: dict,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    circulation: ClosedVascularGraph | None,
) -> (
    tuple[np.ndarray, PlateSurface | None, dict[str, object] | None]
    | PlateElementObstruction
):
    payload = entry.get("plate_policy")
    if payload is None:
        return vertices, None, None
    from golem.plates.core import AcceptedPlatePolicy, AcceptedPlateSurface
    from golem.plates.decode import decode_plate_policy
    from golem.plates.derive import derive_plate_surface
    from golem.plates.projection import plate_surface_report

    decoded = decode_plate_policy(payload)
    if not isinstance(decoded, AcceptedPlatePolicy):
        return PlateElementObstruction(entry["id"], decoded.obstructions)
    derived = derive_plate_surface(
        vertices,
        faces,
        normals,
        decoded.policy,
        morphology_graph=graph,
        circulation=circulation,
    )
    if not isinstance(derived, AcceptedPlateSurface):
        return PlateElementObstruction(entry["id"], derived.obstructions)
    return (
        derived.surface.vertices,
        derived.surface,
        plate_surface_report(derived.surface),
    )


def _plate_seam_record(
    element_id: str,
    role: ElementRole,
    resolution: int,
    surface: PlateSurface | None,
    vertices: np.ndarray,
) -> AssemblyRecord:
    from golem.plates.core import PlateSurface
    from golem.plates.projection import seam_geometry

    if not isinstance(surface, PlateSurface):
        return AssemblyRecord.from_derived_geometry(
            record_id=f"{element_id}__plate_seams",
            element_id=element_id,
            role=role,
            stratum=AssemblyStratum.PLATE_SEAM,
            appearance_material="emissive_seam",
            vertices=None,
            faces=None,
            resolution=resolution,
            report={"faces": 0},
            empty=True,
        )
    seam_vertices, seam_faces = seam_geometry(surface, vertices)
    return AssemblyRecord.from_derived_geometry(
        record_id=f"{element_id}__plate_seams",
        element_id=element_id,
        role=role,
        stratum=AssemblyStratum.PLATE_SEAM,
        appearance_material=surface.policy.seam_material,
        vertices=seam_vertices,
        faces=seam_faces,
        resolution=resolution,
        report={
            "faces": int(seam_faces.shape[0]),
            "cells": surface.policy.cell_count,
        },
        empty=seam_faces.shape[0] == 0,
    )


def _closest_intent_candidates(
    reference: str, known_ids: tuple[str, ...]
) -> tuple[str, ...]:
    close = tuple(difflib.get_close_matches(reference, known_ids, n=3))
    if close:
        return close
    return known_ids if len(known_ids) <= 6 else ()


def _resolve_appendage_intents(
    element_id: str, graph: dict
) -> dict | UnknownAppendageIntentObstruction:
    intents = {
        declaration["id"]: declaration
        for declaration in graph.get("_intents", ())
    }
    appendages = tuple(graph.get("appendages", ()))
    missing = next(
        (
            (str(declaration.get("id", "<unknown>")), str(reference))
            for declaration in appendages
            for reference in declaration.get("intent_refs", ())
            if reference not in intents
        ),
        None,
    )
    if missing is not None:
        appendage_id, intent_id = missing
        known_ids = tuple(sorted(str(identifier) for identifier in intents))
        return UnknownAppendageIntentObstruction(
            element_id,
            appendage_id,
            intent_id,
            _closest_intent_candidates(intent_id, known_ids),
        )
    resolved_appendages = tuple(
        {
            **{
                key: value
                for key, value in declaration.items()
                if key != "intent_refs"
            },
            **(
                {
                    "intent": tuple(
                        intents[reference]
                        for reference in declaration["intent_refs"]
                    )
                }
                if declaration.get("intent_refs")
                else {}
            ),
        }
        for declaration in appendages
    )
    return {
        **{key: value for key, value in graph.items() if key != "_intents"},
        **({"appendages": resolved_appendages} if appendages else {}),
    }


def _expand_appendages(graph: dict) -> dict:
    if not graph.get("appendages"):
        return graph
    from golem.conduits.appendage import expand_appendages

    return expand_appendages(graph)


def _apply_surface_conduits(
    graph: dict,
    vertices: np.ndarray,
    faces: np.ndarray,
    pitch: float,
    flesh_host_vertices: np.ndarray | None = None,
) -> ConduitApplicationResult:
    declarations = graph.get("conduits")
    if not declarations:
        return vertices, ()
    from golem.conduits import apply_conduits

    applied = apply_conduits(
        graph,
        vertices,
        faces,
        declarations,
        pitch=pitch,
        flesh_host_vertices=flesh_host_vertices,
    )
    if isinstance(applied, EmissionPitchObstruction):
        return applied
    displaced, bands = applied
    return displaced, tuple(bands)


def _surface_band_record(
    element_id: str, role: ElementRole, res: int, band: dict
) -> AssemblyRecord:
    empty = bool(band.get("empty"))
    return AssemblyRecord.from_derived_geometry(
        record_id=f"{element_id}__{band['id']}",
        element_id=element_id,
        role=role,
        stratum=AssemblyStratum.CONDUIT,
        appearance_material=band["material"],
        vertices=None if empty else band["verts"],
        faces=None if empty else band["faces"],
        resolution=res,
        report={"faces": 0 if empty else int(band["faces"].shape[0])},
        empty=empty,
    )


def derive_vascular_records(
    element_id: str,
    role: ElementRole | str,
    res: int,
    graph: ClosedVascularGraph,
    pitch: float,
) -> tuple[AssemblyRecord, ...]:
    from golem.conduits.emit import vascular_stratum_mesh

    typed_role = ElementRole(role)
    material_by_stratum = {
        VascularStratum.SUPPLY: "vascular_supply_gold",
        VascularStratum.RETURN: "vascular_return_violet",
        VascularStratum.EXCHANGE: "vascular_exchange_cyan",
    }
    visual_floor = 0.075 * pitch
    return tuple(
        AssemblyRecord.from_derived_geometry(
            record_id=f"{element_id}__vascular_{stratum.value}",
            element_id=element_id,
            role=typed_role,
            stratum=AssemblyStratum(f"vascular_{stratum.value}"),
            appearance_material=material_by_stratum[stratum],
            vertices=mesh[0],
            faces=mesh[1],
            resolution=res,
            report={
                "faces": int(mesh[1].shape[0]),
                "edges": sum(
                    map(lambda edge: int(edge.stratum is stratum), graph.edges)
                ),
                "visual_radius_floor": visual_floor,
            },
        )
        for stratum in (
            VascularStratum.SUPPLY,
            VascularStratum.RETURN,
            VascularStratum.EXCHANGE,
        )
        for mesh in (
            vascular_stratum_mesh(
                graph, stratum, visual_radius_floor=visual_floor
            ),
        )
    )


def _eye_record(
    element_id: str,
    role: ElementRole,
    globe: EyeGlobe,
    part: dict,
) -> AssemblyRecord:
    graph = {
        "name": f"{element_id}__eye_{part['id']}",
        "blend": 0.0,
        "parts": [part],
    }
    evaluated = E.evaluate(graph, res=RES_MIN)
    return AssemblyRecord.from_derived_geometry(
        record_id=f"{element_id}__eye_{part['id']}",
        element_id=element_id,
        role=role,
        stratum=AssemblyStratum.EYE,
        appearance_material=globe.appearance_material.value,
        vertices=evaluated.vertices,
        faces=evaluated.faces,
        resolution=RES_MIN,
        report={
            **E.coherence_report(evaluated.vertices, evaluated.faces),
            "eye_id": globe.eye_id,
            "mirrored": str(part["id"]).endswith("_m"),
        },
        violations=E.vocab_violations(graph, res=RES_MIN),
    )


def derive_eye_records(
    element_id: str,
    role: ElementRole | str,
    globes: tuple[EyeGlobe, ...],
    mount: dict | None,
) -> tuple[AssemblyRecord, ...]:
    typed_role = ElementRole(role)
    return tuple(
        _eye_record(element_id, typed_role, globe, part)
        for globe in globes
        for mounted in (
            mount_graph(
                concretize_mirrors(eye_globe_graph(globe)),
                mount,
            ),
        )
        for part in mounted["parts"]
    )
