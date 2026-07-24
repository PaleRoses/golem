"""Effectful lowering from closed artifact sources to meshes."""

from __future__ import annotations

import json
from pathlib import Path

import trimesh

from golem.kernel import engine
from golem.senses.symmetry_types import (
    ArtifactSource,
    GraphArtifactSource,
    MeshArtifactSource,
    MeshLoadResult,
    SymmetrySourceObstruction,
    SymmetrySurfaceFormationObstruction,
)


def load_graph_mesh(json_path: Path, res: int = 190) -> MeshLoadResult:
    try:
        graph = json.loads(Path(json_path).read_text())
        evaluated = engine.evaluate(graph, res=res)
        return (
            SymmetrySurfaceFormationObstruction(
                Path(json_path),
                evaluated.obstructions,
            )
            if isinstance(evaluated, engine.RejectedSurfaceFormation)
            else trimesh.Trimesh(
                vertices=evaluated.vertices,
                faces=evaluated.faces,
                process=True,
            )
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as failure:
        return SymmetrySourceObstruction(Path(json_path), str(failure))


def load_scene_mesh(mesh_path: Path) -> MeshLoadResult:
    try:
        loaded = trimesh.load(mesh_path, force="mesh")
    except (OSError, TypeError, ValueError) as failure:
        return SymmetrySourceObstruction(Path(mesh_path), str(failure))
    return (
        loaded
        if isinstance(loaded, trimesh.Trimesh) and not loaded.is_empty
        else SymmetrySourceObstruction(
            Path(mesh_path), "source did not decode to a non-empty triangle mesh"
        )
    )


def lower_artifact_source(source: ArtifactSource) -> MeshLoadResult:
    match source:
        case GraphArtifactSource(path=path, resolution=resolution):
            return load_graph_mesh(path, resolution)
        case MeshArtifactSource(path=path):
            return load_scene_mesh(path)


def load_corpus_mesh(kind: str, path: Path, extra: int) -> MeshLoadResult:
    source = (
        GraphArtifactSource(path.name, path, extra)
        if kind == "json"
        else MeshArtifactSource(path.name, path)
        if kind == "glb"
        else None
    )
    return (
        lower_artifact_source(source)
        if source is not None
        else SymmetrySourceObstruction(path, f"unknown corpus kind {kind!r}")
    )
