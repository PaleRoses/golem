"""Frozen corpus, formatting, and CLI effects for symmetry evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import trimesh

from golem.paths import PILOTS as _PILOTS, REHEARSAL as _REHEARSAL
from golem.senses.symmetry_algebra import (
    PITCH,
    aligned_symmetry_iou,
    legacy_symmetry_iou,
    mesh_symmetry_distance,
)
from golem.senses.symmetry_compile import lower_artifact_source
from golem.senses.symmetry_types import (
    ArtifactSource,
    GraphArtifactSource,
    MeshArtifactSource,
    SymmetryObstruction,
    SymmetrySourceObstruction,
    SymmetrySurfaceFormationObstruction,
)


GENERATED = "2026-07-12"
V4_RES = 190


CORPUS: tuple[ArtifactSource, ...] = (
    GraphArtifactSource("golem_v4@190", _PILOTS / "golem_v4.json", V4_RES),
    MeshArtifactSource("bulwark_initial", _REHEARSAL / "blender" / "bulwark_initial.glb"),
    MeshArtifactSource("bulwark_r1", _REHEARSAL / "blender" / "bulwark_r1.glb"),
    MeshArtifactSource("bulwark_r2", _REHEARSAL / "blender" / "bulwark_r2.glb"),
    MeshArtifactSource("bulwark_r3", _REHEARSAL / "blender" / "bulwark_r3.glb"),
    MeshArtifactSource(
        "bulwark_r3_metaball",
        _REHEARSAL / "blender" / "bulwark_r3_metaball.glb",
    ),
    MeshArtifactSource("blender_knight", _REHEARSAL / "knight" / "blender_knight.glb"),
)


def compute_corpus() -> tuple[tuple[str, float, float], ...] | SymmetryObstruction:
    lowered = tuple((source, lower_artifact_source(source)) for source in CORPUS)
    obstruction = next(
        (
            result
            for _source, result in lowered
            if isinstance(
                result,
                (
                    SymmetrySourceObstruction,
                    SymmetrySurfaceFormationObstruction,
                ),
            )
        ),
        None,
    )
    return (
        obstruction
        if obstruction is not None
        else tuple(
            (
                source.label,
                legacy_symmetry_iou(mesh),
                aligned_symmetry_iou(mesh),
            )
            for source, mesh in lowered
            if isinstance(mesh, trimesh.Trimesh)
        )
    )


def format_table(rows: Sequence[tuple[str, float, float]]) -> str:
    label_width = 20
    header = (
        f"GOLEM v03 bilateral-symmetry side-by-side  (generated {GENERATED})",
        "",
        f"Metrics (mirror plane x=0, PITCH={PITCH:.3f}):",
        "  legacy  = pilots/judge surface-shell voxel IoU; grid centered on x=0.",
        "            Reproduces conformance unrounded_symmetry_iou verbatim.",
        "  aligned = NEW solid voxel IoU; grid offset so x=0 is a cell boundary.",
        "  delta   = aligned - legacy.",
        "Values rounded to 4 decimals for cross-platform stability.",
        "",
        f"{'artifact':<{label_width}}  {'legacy':>7}  {'aligned':>7}  {'delta':>7}",
        f"{'-' * label_width}  {'-' * 7}  {'-' * 7}  {'-' * 7}",
    )
    body = tuple(
        f"{label:<{label_width}}  {legacy:>7.4f}  {aligned:>7.4f}  "
        f"{aligned - legacy:>+7.4f}"
        for label, legacy, aligned in rows
    )
    return "\n".join((*header, *body)) + "\n"


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("--res", type=int, default=V4_RES)
    parser.add_argument("--table", action="store_true")
    return parser


def _single_source(path: Path, resolution: int) -> ArtifactSource:
    return (
        GraphArtifactSource(path.name, path, resolution)
        if path.suffix.lower() == ".json"
        else MeshArtifactSource(path.name, path)
    )


def _surface_formation_text(
    obstruction: SymmetrySurfaceFormationObstruction,
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
    args = _argument_parser().parse_args(argv)
    if args.table:
        result = compute_corpus()
        if isinstance(result, SymmetrySourceObstruction):
            sys.stderr.write(f"{result.path}: {result.reason}\n")
            return 2
        if isinstance(result, SymmetrySurfaceFormationObstruction):
            sys.stderr.write(_surface_formation_text(result) + "\n")
            return 2
        sys.stdout.write(format_table(result))
        return 0
    if args.source is None:
        _argument_parser().print_help()
        return 0
    lowered = lower_artifact_source(_single_source(args.source, args.res))
    if isinstance(lowered, SymmetrySourceObstruction):
        sys.stderr.write(f"{lowered.path}: {lowered.reason}\n")
        return 2
    if isinstance(lowered, SymmetrySurfaceFormationObstruction):
        sys.stderr.write(_surface_formation_text(lowered) + "\n")
        return 2
    legacy = legacy_symmetry_iou(lowered)
    aligned = aligned_symmetry_iou(lowered)
    mesh_distance = mesh_symmetry_distance(lowered)
    sys.stdout.write(
        "\n".join(
            (
                f"artifact : {args.source.name}",
                f"  legacy  symmetry IoU (shell, centered) : {legacy:.6f}",
                f"  aligned symmetry IoU (solid, boundary) : {aligned:.6f}",
                f"  delta                                  : {aligned - legacy:+.6f}",
                "  mesh second opinion : "
                f"within_frac={mesh_distance['within_frac']:.4f} "
                f"rms={mesh_distance['rms']:.5f} max={mesh_distance['max']:.5f}",
            )
        )
        + "\n"
    )
    return 0
