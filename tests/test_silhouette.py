"""Tests for the missing GOLEM ``SilhouetteScore`` perception relation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from golem import paths as _paths
from golem.assembly.core import AcceptedAssembly, compile_assembly
from golem.assembly.mounts import concretize_mirrors
from golem.kernel import body, engine
from golem.senses import silhouette


_BRIEF_ID = "knight-calibration"


def _geometry(mesh: trimesh.Trimesh) -> silhouette.Geometry:
    return silhouette.Geometry(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
    )


def test_canonical_view_vocabulary_is_total_and_stable() -> None:
    views = tuple(silhouette.CanonicalView)
    assert tuple(view.value for view in views) == (
        "front",
        "three-quarter",
        "side",
        "back",
    )
    assert tuple(view.azimuth_degrees for view in views) == (0, 40, 90, 180)
    assert len({view.value for view in views}) == len(views)


def test_identical_geometry_has_perfect_cover_score() -> None:
    geometry = _geometry(trimesh.creation.icosphere(subdivisions=2))
    report = silhouette.score_geometry(
        geometry,
        geometry,
        brief_id=_BRIEF_ID,
        image_size=160,
    )
    assert report.brief_id == _BRIEF_ID
    assert silhouette.render_report(report).splitlines()[0] == (
        f"brief_id={_BRIEF_ID}"
    )
    assert report.mean_intersection_over_union == pytest.approx(1.0)
    assert report.mean_boundary_f1 == pytest.approx(1.0)


def test_distinct_global_shapes_do_not_score_as_equal() -> None:
    sphere = _geometry(trimesh.creation.icosphere(subdivisions=2))
    box = _geometry(trimesh.creation.box(extents=(1.5, 0.8, 0.6)))
    report = silhouette.score_geometry(
        sphere,
        box,
        brief_id=_BRIEF_ID,
        image_size=160,
    )
    assert report.mean_intersection_over_union < 0.90
    assert report.mean_boundary_f1 < 0.90


def test_missing_source_is_an_addressed_obstruction() -> None:
    missing = _paths.SPECS / "does-not-exist.json"
    result = silhouette.load_geometry(missing, mesh_resolution=80)
    assert result == silhouette.GeometryLoadFailure(
        path=missing,
        reason="source does not exist",
    )


@pytest.mark.slow
def test_vasculature_does_not_fabricate_structural_surface_displacement() -> None:
    assembly_path = _paths.REHEARSAL / "reforge" / "knight_separated.json"
    assembly_spec = json.loads(assembly_path.read_text(encoding="utf-8"))
    visual_spec = {
        key: value for key, value in assembly_spec.items() if key != "service"
    }
    assembly = compile_assembly(visual_spec, assembly_path.parent)
    assert isinstance(assembly, AcceptedAssembly)
    supported = next(
        record
        for record in assembly.records
        if record.record_id == "body" and record.stratum == "solid"
    )
    compiled = body.compile_file(str(_paths.SPECS / "knight_body.json"))
    assert isinstance(compiled, body.CompiledBody)
    unreinforced_graph = concretize_mirrors(compiled.graph)
    reference = engine.evaluate(
        unreinforced_graph, res=supported.resolution
    )
    assert np.array_equal(supported.vertices, reference.vertices)
    assert np.array_equal(supported.faces, reference.faces)
    assert "reinforcement" not in supported.report
