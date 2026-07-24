"""Public bilateral-symmetry surface and executable entrypoint."""

from __future__ import annotations

import sys

from golem.senses.symmetry_algebra import (
    PITCH,
    aligned_symmetry_iou,
    legacy_symmetry_iou,
    mesh_symmetry_distance,
)
from golem.senses.symmetry_compile import (
    load_corpus_mesh,
    load_graph_mesh,
    load_scene_mesh,
)
from golem.senses.symmetry_harness import (
    CORPUS,
    GENERATED,
    V4_RES,
    compute_corpus,
    format_table,
    main,
)
from golem.senses.symmetry_types import (
    GraphArtifactSource,
    MeshArtifactSource,
    SymmetryObstruction,
    SymmetrySourceObstruction,
    SymmetrySurfaceFormationObstruction,
)


__all__ = [
    "CORPUS",
    "GENERATED",
    "GraphArtifactSource",
    "MeshArtifactSource",
    "PITCH",
    "SymmetrySourceObstruction",
    "SymmetrySurfaceFormationObstruction",
    "SymmetryObstruction",
    "V4_RES",
    "aligned_symmetry_iou",
    "compute_corpus",
    "format_table",
    "legacy_symmetry_iou",
    "load_corpus_mesh",
    "load_graph_mesh",
    "load_scene_mesh",
    "main",
    "mesh_symmetry_distance",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
