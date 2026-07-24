"""Closed artifact-source and obstruction carriers for symmetry evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import trimesh

from golem.kernel.engine.types import SurfaceFormationObstruction


@dataclass(frozen=True)
class GraphArtifactSource:
    label: str
    path: Path
    resolution: int


@dataclass(frozen=True)
class MeshArtifactSource:
    label: str
    path: Path


type ArtifactSource = GraphArtifactSource | MeshArtifactSource


@dataclass(frozen=True)
class SymmetrySourceObstruction:
    path: Path
    reason: str


@dataclass(frozen=True)
class SymmetrySurfaceFormationObstruction:
    path: Path
    obstructions: tuple[SurfaceFormationObstruction, ...]


type SymmetryObstruction = (
    SymmetrySourceObstruction | SymmetrySurfaceFormationObstruction
)
type MeshLoadResult = trimesh.Trimesh | SymmetryObstruction
