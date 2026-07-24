"""Tests for the M0 mirror-plane-aligned symmetry metrics and the golden-text
harness. Fast synthetic cases run by default; the frozen-corpus recomputation
is marked ``slow`` (run it with ``-m slow`` / skip it with ``-m 'not slow'``)."""

from __future__ import annotations

import re

import numpy as np
import pytest
import trimesh

import engine
from golem import goldentext
from golem.senses import symmetry

# --------------------------------------------------------------------------- #
# Synthetic fixtures.                                                          #
# --------------------------------------------------------------------------- #

# A small part-graph, mirror-symmetric by construction, that crosses x = 0.
# Compiled through marching cubes it carries the same sub-voxel asymmetry that
# depresses the legacy metric on the real corpus -- the perfect fast stand-in.
_MIRROR_GRAPH = {
    "name": "test_mirror",
    "blend": 0.05,
    "parts": [
        {"id": "torso", "type": "gencyl",
         "spine": [[0, 0.5, 0.0], [0, 1.4, 0.1]], "radii": [0.45, 0.5]},
        {"id": "arm", "type": "gencyl", "mirror": True,
         "spine": [[0.4, 1.3, 0.0], [0.9, 0.7, 0.2]], "radii": [0.22, 0.18]},
        {"id": "shoulder", "type": "blob", "mirror": True,
         "center": [0.45, 1.3, 0.0], "size": [0.3, 0.3, 0.3]},
    ],
}


@pytest.fixture(scope="module")
def symmetric_mesh() -> trimesh.Trimesh:
    vertices, faces, *_ = engine.evaluate(_MIRROR_GRAPH, res=64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


@pytest.fixture(scope="module")
def asymmetric_mesh() -> trimesh.Trimesh:
    # A box shoved well off the mirror plane: unambiguously asymmetric.
    mesh = trimesh.creation.box(extents=(1.4, 0.9, 0.6))
    mesh.apply_translation((0.31, 0.0, 0.0))
    return mesh


@pytest.fixture(scope="module")
def exact_symmetric_primitive() -> trimesh.Trimesh:
    # An icosphere centered on x=0 is exactly symmetric down to its vertices.
    return trimesh.creation.icosphere(subdivisions=4, radius=0.7)


# --------------------------------------------------------------------------- #
# Voxel metrics: the new metric fixes symmetric geometry, catches asymmetry.   #
# --------------------------------------------------------------------------- #

def test_aligned_beats_legacy_on_symmetric_geometry(symmetric_mesh):
    legacy = symmetry.legacy_symmetry_iou(symmetric_mesh)
    aligned = symmetry.aligned_symmetry_iou(symmetric_mesh)
    # The defect is real on this mesh: the legacy grid-misaligned shell reads
    # below the 0.995 gate...
    assert legacy < 0.995
    # ...while the aligned solid metric clears it, and strictly beats legacy.
    assert aligned >= 0.995
    assert aligned > legacy


def test_asymmetric_scores_low_under_both(asymmetric_mesh):
    legacy = symmetry.legacy_symmetry_iou(asymmetric_mesh)
    aligned = symmetry.aligned_symmetry_iou(asymmetric_mesh)
    # Both metrics must clearly reject asymmetry -- the new metric is not blind.
    assert legacy < 0.9
    assert aligned < 0.9
    # And far below the >= 0.995 floor a symmetric artifact reaches.
    assert aligned < 0.95


def test_exact_symmetric_primitive_scores_perfect(exact_symmetric_primitive):
    # Neither metric is broken on trivially symmetric geometry.
    assert symmetry.legacy_symmetry_iou(exact_symmetric_primitive) == 1.0
    assert symmetry.aligned_symmetry_iou(exact_symmetric_primitive) == 1.0


# --------------------------------------------------------------------------- #
# Mesh-based second opinion: agrees directionally, also not blind.            #
# --------------------------------------------------------------------------- #

def test_mesh_second_opinion_separates_symmetric_from_asymmetric(
    symmetric_mesh, asymmetric_mesh
):
    # within_frac has a sample-density floor (nearest-neighbour distance is
    # bounded below by sample spacing), so use enough samples and lean on rms,
    # which separates the two by ~40x here.
    sym = symmetry.mesh_symmetry_distance(symmetric_mesh, samples=20000)
    asym = symmetry.mesh_symmetry_distance(asymmetric_mesh, samples=20000)
    # Symmetric: almost every reflected sample has a near neighbour, tiny rms.
    assert sym["within_frac"] > 0.95
    assert sym["rms"] < 0.02
    # Asymmetric: reflected samples land far from the cloud.
    assert asym["within_frac"] < 0.6
    assert asym["rms"] > 0.05
    assert asym["rms"] > sym["rms"] * 5


def test_mesh_second_opinion_is_deterministic(symmetric_mesh):
    a = symmetry.mesh_symmetry_distance(symmetric_mesh, samples=4000, seed=7)
    b = symmetry.mesh_symmetry_distance(symmetric_mesh, samples=4000, seed=7)
    assert a == b


# --------------------------------------------------------------------------- #
# Golden-text harness self-test: missing fails, create, match, refuse-overwrite.
# --------------------------------------------------------------------------- #

def test_goldentext_write_once_discipline(tmp_path, monkeypatch):
    name = "selftest.txt"
    text = "hello golden\n"

    # 1. Missing golden, no GOLDEN_CREATE -> hard failure.
    monkeypatch.delenv("GOLDEN_CREATE", raising=False)
    with pytest.raises(goldentext.GoldenMissing):
        goldentext.assert_matches_golden(name, text, golden_dir=tmp_path)
    assert not (tmp_path / name).exists()

    # 2. GOLDEN_CREATE=1 -> creates it and passes.
    monkeypatch.setenv("GOLDEN_CREATE", "1")
    goldentext.assert_matches_golden(name, text, golden_dir=tmp_path)
    assert (tmp_path / name).read_text() == text

    # 3. Second run against the existing golden -> matches (idempotent).
    goldentext.assert_matches_golden(name, text, golden_dir=tmp_path)

    # 4. Overwrite attempt (different text, even with GOLDEN_CREATE=1) -> refuse,
    #    and the file on disk is left untouched.
    with pytest.raises(AssertionError) as excinfo:
        goldentext.assert_matches_golden(name, "different\n", golden_dir=tmp_path)
    assert not isinstance(excinfo.value, goldentext.GoldenMissing)
    assert (tmp_path / name).read_text() == text


# --------------------------------------------------------------------------- #
# Frozen-corpus table -- heavy; recompute and check against the checked-in    #
# golden within 1e-3, plus the exact legacy obstruction value.                #
# --------------------------------------------------------------------------- #

def _parse_golden_rows(text: str) -> dict[str, tuple[float, float]]:
    rows: dict[str, tuple[float, float]] = {}
    for line in text.splitlines():
        m = re.match(r"^(\S+)\s+([01]\.\d{4})\s+([01]\.\d{4})\s+[+-]\d\.\d{4}$", line)
        if m:
            rows[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return rows


@pytest.mark.slow
def test_corpus_table_matches_golden():
    rows = symmetry.compute_corpus()
    by_label = {label: (legacy, aligned) for label, legacy, aligned in rows}

    # The preregistered obstruction reproduces exactly (float64).
    assert by_label["golem_v4@190"][0] == pytest.approx(
        0.9799926081064433, abs=1e-9
    )

    # Every mirror-symmetric-by-construction artifact clears the new gate.
    for label, (_legacy, aligned) in by_label.items():
        assert aligned >= 0.995, f"{label} aligned {aligned} < 0.995"

    # Legacy readings sit in / near the documented 0.958-0.980 defect band.
    for label, (legacy, _aligned) in by_label.items():
        assert 0.95 <= legacy <= 0.985, f"{label} legacy {legacy} out of band"

    # Rendered table is byte-exact against the checked-in golden...
    table = symmetry.format_table(rows)
    goldentext.assert_matches_golden("symmetry_side_by_side.txt", table)

    # ...and the recomputed values agree with the golden to 1e-3.
    golden = (goldentext.GOLDEN_DIR / "symmetry_side_by_side.txt").read_text()
    parsed = _parse_golden_rows(golden)
    assert parsed  # golden actually contained rows
    for label, (legacy, aligned) in by_label.items():
        g_legacy, g_aligned = parsed[label]
        assert legacy == pytest.approx(g_legacy, abs=1e-3)
        assert aligned == pytest.approx(g_aligned, abs=1e-3)
