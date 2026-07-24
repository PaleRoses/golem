"""Sealed mechanical compass for GOLEM's derived plate surface.

The score is intentionally narrower than beauty.  It measures whether a plate
cover is even, visibly legible, aligned with morphology-local axes, and focused
near the major closed-circulation routes while preserving the accepted body
silhouette and solid integrity.  Human blind preference remains the authority
for appeal; this evaluator merely prevents random Voronoi noise from winning by
adding ever more seams.

The final stdout line is stable JSON for ``codex-autoresearch``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import assert_never

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from golem.assembly.mounts import concretize_mirrors
from golem.kernel import body, engine
from golem.kernel.anatomy import (
    AcceptedVasculature,
    ClosedVascularGraph,
    VascularStratum,
    realize_vasculature,
)
from golem.materials import resolve_appearance_material
from golem.evals.harness import (
    AcceptedEvaluation,
    EvaluationObstruction,
    RenderedEvaluation,
    emit_rendered_evaluation,
    render_outcome,
)
from golem.kernel.engine.types import SurfaceFormationObstruction
from golem.plates import (
    AcceptedPlatePolicy,
    AcceptedPlateSurface,
    PlateSurface,
    decode_plate_policy,
    derive_plate_surface,
    minimum_point_segment_distances,
)
from golem.senses.raster import (
    CanonicalView,
    intersection_over_union,
    render_raster_view,
    safe_mean,
)


_DEFAULT_RESOLUTION = 110
_DEFAULT_IMAGE_SIZE = 192
_MAJOR_FLOW_EDGE_COUNT = 32
_FLOW_SURFACE_DISTANCE = 0.16
_MINIMUM_SILHOUETTE_IOU = 0.97


@dataclass(frozen=True)
class PlateAuthoringFrame:
    view: CanonicalView
    silhouette_iou: float
    seam_pixel_fraction: float
    seam_legibility: float


@dataclass(frozen=True)
class PlateAuthoringEvaluation(AcceptedEvaluation):
    frames: tuple[PlateAuthoringFrame, ...]
    structural_alignment: float
    cell_area_balance: float
    circulation_focus: float
    component_count: int
    watertight: bool
    surface: PlateSurface

    @property
    def seam_legibility(self) -> float:
        return fmean(frame.seam_legibility for frame in self.frames)

    @property
    def silhouette_iou_minimum(self) -> float:
        return min(frame.silhouette_iou for frame in self.frames)

    @property
    def plate_authoring_score(self) -> float:
        local_sections = (
            0.30 * self.structural_alignment
            + 0.25 * self.cell_area_balance
            + 0.25 * self.circulation_focus
            + 0.20 * self.seam_legibility
        )
        return local_sections * self.silhouette_iou_minimum

    @property
    def guard_passes(self) -> bool:
        return (
            self.component_count == 1
            and self.watertight
            and self.silhouette_iou_minimum >= _MINIMUM_SILHOUETTE_IOU
            and bool(np.any(self.surface.active_seam_faces))
        )

    def evaluation_payload(self) -> dict[str, object]:
        return {
            "cell_area_balance": self.cell_area_balance,
            "circulation_focus": self.circulation_focus,
            "component_count": self.component_count,
            "guard_passes": self.guard_passes,
            "plate_authoring_score": self.plate_authoring_score,
            "seam_legibility": self.seam_legibility,
            "seam_pixel_fraction_mean": fmean(
                frame.seam_pixel_fraction for frame in self.frames
            ),
            "silhouette_iou_minimum": self.silhouette_iou_minimum,
            "structural_alignment": self.structural_alignment,
            "watertight": self.watertight,
        }


@dataclass(frozen=True)
class PlateAuthoringSurfaceFormationObstruction:
    body_path: Path
    obstructions: tuple[SurfaceFormationObstruction, ...]


PlateAuthoringObstruction = EvaluationObstruction


type PlateAuthoringResult = (
    PlateAuthoringEvaluation
    | PlateAuthoringObstruction
    | PlateAuthoringSurfaceFormationObstruction
)


def evaluate_plate_authoring(
    body_path: Path,
    policy_path: Path,
    *,
    resolution: int = _DEFAULT_RESOLUTION,
    image_size: int = _DEFAULT_IMAGE_SIZE,
) -> PlateAuthoringResult:
    """Compile one body and descend the complete canonical evaluation cover."""
    try:
        compile_result = body.compile_file(str(body_path))
        policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as failure:
        return PlateAuthoringObstruction(str(body_path), str(failure))
    match compile_result:
        case body.CompiledBody() as compiled:
            pass
        case body.RejectedBody() as rejected:
            return PlateAuthoringObstruction(
                str(body_path),
                json.dumps(
                    body.project_rejected_body(rejected),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
        case _ as unreachable:
            assert_never(unreachable)
    decoded = decode_plate_policy(policy_payload)
    if not isinstance(decoded, AcceptedPlatePolicy):
        return PlateAuthoringObstruction(
            str(policy_path),
            json.dumps(
                tuple(
                    {
                        "address": obstruction.address,
                        "rule": obstruction.rule.value,
                        "authored": obstruction.authored,
                        "required": obstruction.required,
                    }
                    for obstruction in decoded.obstructions
                ),
                sort_keys=True,
            ),
        )
    graph = concretize_mirrors(compiled.graph)
    evaluated = engine.evaluate(graph, res=resolution)
    if isinstance(evaluated, engine.RejectedSurfaceFormation):
        return PlateAuthoringSurfaceFormationObstruction(
            body_path,
            evaluated.obstructions,
        )
    vasculature = (
        realize_vasculature(compiled.anatomy, compiled.graph)
        if compiled.anatomy is not None
        else None
    )
    if not isinstance(vasculature, AcceptedVasculature):
        return PlateAuthoringObstruction(
            str(body_path), "body did not produce accepted closed vasculature"
        )
    surface_result = derive_plate_surface(
        evaluated.vertices,
        evaluated.faces,
        evaluated.normals,
        decoded.policy,
        morphology_graph=graph,
        circulation=vasculature.graph,
    )
    if not isinstance(surface_result, AcceptedPlateSurface):
        return PlateAuthoringObstruction(
            str(policy_path),
            json.dumps(
                tuple(
                    {
                        "address": obstruction.address,
                        "rule": obstruction.rule.value,
                        "authored": obstruction.authored,
                        "required": obstruction.required,
                    }
                    for obstruction in surface_result.obstructions
                ),
                sort_keys=True,
            ),
        )
    surface = surface_result.surface
    coherence = engine.coherence_report(
        surface.vertices, surface.faces
    )
    return PlateAuthoringEvaluation(
        frames=_frame_cover(evaluated.vertices, surface, image_size),
        structural_alignment=_structural_alignment(graph, surface),
        cell_area_balance=_cell_area_balance(surface),
        circulation_focus=_circulation_focus(surface, vasculature.graph),
        component_count=int(coherence["components"]),
        watertight=bool(coherence["watertight_main"]),
        surface=surface,
    )


def _frame_cover(
    source_vertices: NDArray[np.float64],
    surface: PlateSurface,
    image_size: int,
) -> tuple[PlateAuthoringFrame, ...]:
    base_material = resolve_appearance_material("obsidian_deep")
    seam_material = resolve_appearance_material(surface.policy.seam_material)
    base_rgb = np.rint(255.0 * np.asarray(base_material.base_color)).astype(
        np.uint8
    )
    seam_rgb = np.rint(
        255.0 * np.asarray(seam_material.emissive_color)
    ).astype(np.uint8)
    base_colors = np.broadcast_to(base_rgb, (surface.faces.shape[0], 3))
    candidate_colors = np.where(
        surface.active_seam_faces[:, None], seam_rgb, base_rgb
    ).astype(np.uint8)
    return tuple(
        _measure_frame(
            view,
            source_vertices,
            surface,
            base_colors,
            candidate_colors,
            image_size,
        )
        for view in CanonicalView
    )


def _measure_frame(
    view: CanonicalView,
    source_vertices: NDArray[np.float64],
    surface: PlateSurface,
    base_colors: NDArray[np.uint8],
    candidate_colors: NDArray[np.uint8],
    image_size: int,
) -> PlateAuthoringFrame:
    source = render_raster_view(
        source_vertices,
        surface.faces,
        view.azimuth_degrees,
        image_size=image_size,
        face_colors=base_colors,
    )
    base = render_raster_view(
        surface.vertices,
        surface.faces,
        view.azimuth_degrees,
        image_size=image_size,
        face_colors=base_colors,
    )
    candidate = render_raster_view(
        surface.vertices,
        surface.faces,
        view.azimuth_degrees,
        image_size=image_size,
        face_colors=candidate_colors,
    )
    delta = np.linalg.norm(
        candidate.pixels.astype(np.float64)
        - base.pixels.astype(np.float64),
        axis=2,
    )
    seam_fraction = safe_mean(
        (delta[base.body] >= 32.0).astype(np.float64)
    )
    return PlateAuthoringFrame(
        view=view,
        silhouette_iou=intersection_over_union(source.body, base.body),
        seam_pixel_fraction=seam_fraction,
        seam_legibility=_coverage_legibility(seam_fraction),
    )


def _structural_alignment(graph: dict, surface: PlateSurface) -> float:
    segments = tuple(
        (np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
        for part in graph["parts"]
        if part.get("type") == "gencyl"
        for left, right in zip(part["spine"], part["spine"][1:])
    )
    if not segments or surface.seam_edges.size == 0:
        return 0.0
    starts = np.asarray(tuple(segment[0] for segment in segments))
    vectors = np.asarray(tuple(segment[1] - segment[0] for segment in segments))
    squared_lengths = np.maximum(
        np.einsum("ij,ij->i", vectors, vectors), 1.0e-15
    )
    axes = vectors / np.sqrt(squared_lengths)[:, None]
    edge_vectors = (
        surface.source_vertices[surface.seam_edges[:, 1]]
        - surface.source_vertices[surface.seam_edges[:, 0]]
    )
    edge_directions = edge_vectors / np.maximum(
        np.linalg.norm(edge_vectors, axis=1, keepdims=True), 1.0e-15
    )
    midpoints = 0.5 * (
        surface.source_vertices[surface.seam_edges[:, 0]]
        + surface.source_vertices[surface.seam_edges[:, 1]]
    )
    relative = midpoints[:, None, :] - starts[None, :, :]
    parameters = np.clip(
        np.einsum("nsi,si->ns", relative, vectors)
        / squared_lengths[None, :],
        0.0,
        1.0,
    )
    closest = starts[None, :, :] + parameters[:, :, None] * vectors[None, :, :]
    nearest = np.argmin(
        np.linalg.norm(midpoints[:, None, :] - closest, axis=2), axis=1
    )
    cosine = np.abs(np.einsum("ni,ni->n", edge_directions, axes[nearest]))
    grid_alignment = np.maximum(cosine, np.sqrt(np.maximum(1.0 - cosine**2, 0.0)))
    floor = 1.0 / math.sqrt(2.0)
    return float(np.clip((np.mean(grid_alignment) - floor) / (1.0 - floor), 0.0, 1.0))


def _cell_area_balance(surface: PlateSurface) -> float:
    nonzero = surface.cell_areas[surface.cell_areas > 0.0]
    mean_area = float(np.mean(nonzero)) if nonzero.size else 0.0
    coefficient_of_variation = (
        float(np.std(nonzero) / mean_area) if mean_area > 0.0 else math.inf
    )
    return float(1.0 / (1.0 + coefficient_of_variation))


def _circulation_focus(
    surface: PlateSurface, graph: ClosedVascularGraph
) -> float:
    major_edges = tuple(
        sorted(
            (
                edge
                for edge in graph.edges
                if edge.stratum is VascularStratum.SUPPLY
                and edge.solved_flow > 0.0
            ),
            key=lambda edge: (edge.solved_flow, edge.radius, edge.edge_id),
            reverse=True,
        )[:_MAJOR_FLOW_EDGE_COUNT]
    )
    active_faces = surface.faces[surface.active_seam_faces]
    if not major_edges or active_faces.size == 0:
        return 0.0
    nodes = graph.node_by_id
    starts = np.asarray(
        tuple(nodes[edge.source_node_id].position for edge in major_edges),
        dtype=np.float64,
    )
    ends = np.asarray(
        tuple(nodes[edge.target_node_id].position for edge in major_edges),
        dtype=np.float64,
    )
    seam_centroids = np.mean(surface.source_vertices[active_faces], axis=1)
    minimum_distances = minimum_point_segment_distances(
        seam_centroids, starts, ends
    )
    precision = safe_mean(
        (minimum_distances <= _FLOW_SURFACE_DISTANCE).astype(np.float64)
    )
    vascular_midpoints = 0.5 * (starts + ends)
    coverage = safe_mean(
        (
            cKDTree(seam_centroids).query(vascular_midpoints, k=1)[0]
            <= _FLOW_SURFACE_DISTANCE
        ).astype(np.float64)
    )
    return (
        2.0 * precision * coverage / (precision + coverage)
        if precision + coverage > 0.0
        else 0.0
    )


def _coverage_legibility(fraction: float) -> float:
    visibility = min(fraction / 0.06, 1.0)
    restraint = float(np.clip((0.38 - fraction) / 0.14, 0.0, 1.0))
    return visibility * restraint


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("body", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--res", type=int, default=_DEFAULT_RESOLUTION)
    parser.add_argument("--size", type=int, default=_DEFAULT_IMAGE_SIZE)
    parser.add_argument("--guard", action="store_true")
    return parser


def _render_surface_formation(
    obstruction: PlateAuthoringSurfaceFormationObstruction,
) -> RenderedEvaluation:
    return RenderedEvaluation(
        stdout="",
        stderr=json.dumps(
            {
                "address": str(obstruction.body_path),
                "obstructions": tuple(
                    {
                        "obstruction": type(member).__name__,
                        **asdict(member),
                    }
                    for member in obstruction.obstructions
                ),
                "status": "surface_formation_obstructed",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        exit_code=2,
    )


def _render_plate_authoring_result(
    result: PlateAuthoringResult,
    *,
    enforce_guard: bool = False,
) -> RenderedEvaluation:
    return (
        _render_surface_formation(result)
        if isinstance(result, PlateAuthoringSurfaceFormationObstruction)
        else render_outcome(result, enforce_guard=enforce_guard)
    )


def render_evaluation_json(result: PlateAuthoringResult) -> str:
    rendered = _render_plate_authoring_result(result)
    return (rendered.stdout or rendered.stderr).rstrip("\n")


def main(argv: list[str]) -> int:
    args = _argument_parser().parse_args(argv[1:])
    result = evaluate_plate_authoring(
        args.body,
        args.policy,
        resolution=args.res,
        image_size=args.size,
    )
    return emit_rendered_evaluation(
        _render_plate_authoring_result(result, enforce_guard=args.guard)
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
