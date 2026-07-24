"""Deterministic cross-artifact silhouette calibration for GOLEM v0.3.

The GOLEM specification already names ``SilhouetteScore`` as an L3
perception-plane relation.  The pilots supplied the canonical renderer but no
implementation that compared a candidate against a reference.  This module
fills that concrete hole without claiming that silhouette similarity is an
aesthetic oracle: it is a calibration signal for global proportion, negative
space, and outline preservation.

The pure scoring core consumes two immutable geometry values.  Filesystem,
JSON, body compilation, mesh loading, and process exit live at the CLI
boundary.  The final stdout line is the mean IoU scalar so the same command is
usable by codex-autoresearch without scraping prose.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import assert_never

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, distance_transform_edt

from golem.kernel import body
from golem.kernel.body.types import BodyObstruction
from golem.kernel import engine
from golem.kernel.engine.types import SurfaceFormationObstruction
from golem.senses.model import immutable_array
from golem.senses.raster import (
    CanonicalView,
    intersection_over_union,
    render_raster_view,
)


_DEFAULT_IMAGE_SIZE = 256
_DEFAULT_MESH_RESOLUTION = 120
_BOUNDARY_TOLERANCE_PIXELS = 2.0


@dataclass(frozen=True)
class Geometry:
    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertices", immutable_array(self.vertices))
        object.__setattr__(self, "faces", immutable_array(self.faces))


@dataclass(frozen=True)
class GeometryLoadFailure:
    path: Path
    reason: str


@dataclass(frozen=True)
class BodyGeometryLoadObstruction:
    path: Path
    obstructions: tuple[BodyObstruction, ...]


@dataclass(frozen=True)
class SurfaceFormationGeometryLoadObstruction:
    path: Path
    obstructions: tuple[SurfaceFormationObstruction, ...]


@dataclass(frozen=True)
class ViewScore:
    view: CanonicalView
    intersection_over_union: float
    boundary_f1: float


@dataclass(frozen=True)
class SilhouetteScore:
    brief_id: str
    views: tuple[ViewScore, ...]

    @property
    def mean_intersection_over_union(self) -> float:
        return fmean(score.intersection_over_union for score in self.views)

    @property
    def mean_boundary_f1(self) -> float:
        return fmean(score.boundary_f1 for score in self.views)


type GeometryLoadResult = (
    Geometry
    | GeometryLoadFailure
    | BodyGeometryLoadObstruction
    | SurfaceFormationGeometryLoadObstruction
)
type SilhouetteEvaluationResult = SilhouetteScore | (
    GeometryLoadFailure
    | BodyGeometryLoadObstruction
    | SurfaceFormationGeometryLoadObstruction
)


def silhouette_mask(
    geometry: Geometry,
    view: CanonicalView,
    image_size: int = _DEFAULT_IMAGE_SIZE,
) -> NDArray[np.bool_]:
    """Project one neutral canonical view and retain only occupied pixels."""
    return render_raster_view(
        geometry.vertices,
        geometry.faces,
        view.azimuth_degrees,
        image_size=image_size,
    ).body


def score_geometry(
    candidate: Geometry,
    reference: Geometry,
    brief_id: str,
    image_size: int = _DEFAULT_IMAGE_SIZE,
) -> SilhouetteScore:
    """Score the complete canonical cover; no single view owns the verdict."""
    return SilhouetteScore(
        brief_id=brief_id,
        views=tuple(
            _score_view(candidate, reference, view, image_size)
            for view in CanonicalView
        ),
    )


def evaluate_sources(
    brief_id: str,
    candidate_path: Path,
    reference_path: Path,
    mesh_resolution: int = _DEFAULT_MESH_RESOLUTION,
    image_size: int = _DEFAULT_IMAGE_SIZE,
) -> SilhouetteEvaluationResult:
    candidate_result = load_geometry(candidate_path, mesh_resolution)
    match candidate_result:
        case Geometry() as candidate:
            pass
        case (
            GeometryLoadFailure()
            | BodyGeometryLoadObstruction()
            | SurfaceFormationGeometryLoadObstruction()
        ):
            return candidate_result
        case _ as unreachable:
            assert_never(unreachable)
    reference_result = load_geometry(reference_path, mesh_resolution)
    match reference_result:
        case Geometry() as reference:
            return score_geometry(
                candidate,
                reference,
                brief_id,
                image_size,
            )
        case (
            GeometryLoadFailure()
            | BodyGeometryLoadObstruction()
            | SurfaceFormationGeometryLoadObstruction()
        ):
            return reference_result
        case _ as unreachable:
            assert_never(unreachable)


def load_geometry(path: Path, mesh_resolution: int) -> GeometryLoadResult:
    """Decode one source or return an addressed obstruction."""
    try:
        return _load_geometry(path, mesh_resolution)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return GeometryLoadFailure(path=path, reason=str(exc))


def render_report(report: SilhouetteScore) -> str:
    per_view = "\n".join(
        f"{score.view.value:13s} iou={score.intersection_over_union:.6f} "
        f"boundary_f1={score.boundary_f1:.6f}"
        for score in report.views
    )
    return (
        f"brief_id={report.brief_id}\n"
        f"{per_view}\n"
        f"mean_iou={report.mean_intersection_over_union:.6f}\n"
        f"mean_boundary_f1={report.mean_boundary_f1:.6f}"
    )


def _score_view(
    candidate: Geometry,
    reference: Geometry,
    view: CanonicalView,
    image_size: int,
) -> ViewScore:
    candidate_mask = silhouette_mask(candidate, view, image_size)
    reference_mask = silhouette_mask(reference, view, image_size)
    return ViewScore(
        view=view,
        intersection_over_union=intersection_over_union(candidate_mask, reference_mask),
        boundary_f1=_boundary_f1(candidate_mask, reference_mask),
    )


def _boundary_f1(
    candidate: NDArray[np.bool_], reference: NDArray[np.bool_]
) -> float:
    candidate_boundary = _boundary(candidate)
    reference_boundary = _boundary(reference)
    candidate_distance = distance_transform_edt(~candidate_boundary)
    reference_distance = distance_transform_edt(~reference_boundary)
    precision = _matched_fraction(
        candidate_distance[reference_boundary] <= _BOUNDARY_TOLERANCE_PIXELS
    )
    recall = _matched_fraction(
        reference_distance[candidate_boundary] <= _BOUNDARY_TOLERANCE_PIXELS
    )
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 1.0
    )


def _boundary(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    return mask & ~binary_erosion(mask)


def _matched_fraction(matches: NDArray[np.bool_]) -> float:
    return float(np.mean(matches)) if matches.size > 0 else 1.0


def _load_geometry(path: Path, mesh_resolution: int) -> GeometryLoadResult:
    if not path.is_file():
        raise ValueError("source does not exist")
    if path.suffix.lower() == ".json":
        source = json.loads(path.read_text(encoding="utf-8"))
        if source.get("dialect") == body.DIALECT:
            compile_result = body.compile_file(str(path))
            match compile_result:
                case body.CompiledBody(graph, _, _):
                    pass
                case body.RejectedBody(obstructions):
                    return BodyGeometryLoadObstruction(path, obstructions)
                case _ as unreachable:
                    assert_never(unreachable)
        else:
            graph = source
        evaluated = engine.evaluate(graph, res=mesh_resolution)
        match evaluated:
            case engine.EvaluatedMorphology(vertices=vertices, faces=faces):
                return Geometry(
                    vertices=np.asarray(vertices, dtype=np.float64),
                    faces=np.asarray(faces, dtype=np.int64),
                )
            case engine.RejectedSurfaceFormation(obstructions):
                return SurfaceFormationGeometryLoadObstruction(
                    path,
                    obstructions,
                )
            case _ as unreachable:
                assert_never(unreachable)
    loaded = trimesh.load(path, force="scene")
    mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError("source did not decode to a non-empty triangle mesh")
    return Geometry(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare GOLEM candidate and reference silhouettes."
    )
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--brief-id", required=True)
    parser.add_argument("--res", type=int, default=_DEFAULT_MESH_RESOLUTION)
    parser.add_argument("--size", type=int, default=_DEFAULT_IMAGE_SIZE)
    return parser


def _surface_formation_text(
    obstruction: SurfaceFormationGeometryLoadObstruction,
) -> str:
    return json.dumps(
        {
            "obstructions": tuple(
                {
                    "obstruction": type(member).__name__,
                    **asdict(member),
                }
                for member in obstruction.obstructions
            ),
            "path": str(obstruction.path),
            "status": "surface_formation_obstructed",
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def main(argv: list[str]) -> int:
    args = _argument_parser().parse_args(argv[1:])
    result = evaluate_sources(
        args.brief_id,
        args.candidate,
        args.reference,
        args.res,
        args.size,
    )
    match result:
        case GeometryLoadFailure(path, reason):
            print(f"silhouette-load-failure {path}: {reason}", file=sys.stderr)
            return 2
        case BodyGeometryLoadObstruction(path, obstructions):
            rejection = body.RejectedBody(obstructions)
            print(
                f"silhouette-load-failure {path}: "
                f"{body.rejected_body_text(rejection)}",
                file=sys.stderr,
            )
            return 2
        case SurfaceFormationGeometryLoadObstruction() as obstruction:
            print(_surface_formation_text(obstruction), file=sys.stderr)
            return 2
        case SilhouetteScore() as report:
            print(render_report(report))
            print(f"{report.mean_intersection_over_union:.12f}")
            return 0
        case _ as unreachable:
            assert_never(unreachable)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
