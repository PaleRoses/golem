"""Tests for the balance-solution -> frozen plated-renderer lowering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image, ImageSequence

from golem.kernel import sheaf
from golem.senses import elemental_render


_SEAM_RGBA = np.asarray((168, 85, 247, 255), dtype=np.uint8)
_PLATE_RGBA = np.asarray((45, 52, 61, 255), dtype=np.uint8)


@pytest.fixture
def balance_problem() -> sheaf.BalanceProblem:
    return sheaf.BalanceProblem(
        problem_id=sheaf.BalanceProblemId("elemental-render-test"),
        domain=sheaf.CellularComplex(
            complex_id="three-plate-cover",
            cell_centers=(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
            ),
            adjacencies=(
                sheaf.CellAdjacency(sheaf.CellId(0), sheaf.CellId(1), 1.0),
                sheaf.CellAdjacency(sheaf.CellId(1), sheaf.CellId(2), 1.0),
            ),
        ),
        symmetric_conductances=(
            sheaf.SymmetricConductance(
                sheaf.BalanceInterfaceId("left-rune"),
                sheaf.CellId(0),
                sheaf.CellId(1),
                2.0,
            ),
            sheaf.SymmetricConductance(
                sheaf.BalanceInterfaceId("right-rune"),
                sheaf.CellId(1),
                sheaf.CellId(2),
                1.0,
            ),
        ),
        fixed_boundaries=(
            sheaf.FixedBoundary(
                sheaf.BoundaryId("source"), sheaf.CellId(0), 1.0
            ),
            sheaf.FixedBoundary(
                sheaf.BoundaryId("terminal"), sheaf.CellId(2), 0.0
            ),
        ),
    )


@pytest.fixture
def mesh_path(tmp_path: Path) -> Path:
    vertices = np.asarray(
        (
            (0.20, -0.15, 0.0),
            (0.45, 0.15, 0.0),
            (0.45, -0.15, 0.0),
            (1.80, -0.15, 0.0),
            (1.55, -0.15, 0.0),
            (1.55, 0.15, 0.0),
            (0.25, 0.25, 0.0),
            (0.45, 0.55, 0.0),
            (0.45, 0.25, 0.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 1, 2), (3, 4, 5), (6, 7, 8)), dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.face_colors = np.asarray(
        (_SEAM_RGBA, _SEAM_RGBA, _PLATE_RGBA), dtype=np.uint8
    )
    path = tmp_path / "three-plate.glb"
    mesh.export(path)
    return path


@pytest.fixture
def scene(
    mesh_path: Path, balance_problem: sheaf.BalanceProblem
) -> elemental_render.ElementalRenderScene:
    result = elemental_render.solve_elemental_scene(mesh_path, balance_problem)
    assert isinstance(result, elemental_render.AcceptedElementalRender)
    return result.value


def test_scene_glues_frozen_faces_to_the_authoritative_balance_cover(
    scene: elemental_render.ElementalRenderScene,
) -> None:
    assert scene.frozen_vertices.shape == (9, 3)
    assert scene.render_vertices.shape == (9, 3)
    assert scene.faces.shape == (3, 3)
    assert scene.frozen_face_colors.shape == (3, 4)
    assert scene.face_cell_interfaces.shape == (3, 2)
    assert scene.balance_interface_faces.tolist() == [True, True, False]
    assert scene.normalized_face_field == pytest.approx((5.0 / 6.0, 1.0 / 3.0, 5.0 / 6.0))
    assert scene.normalized_face_flux == pytest.approx((1.0, 1.0, 1.0))
    assert scene.solution.maximum_normalized_residual <= 1.0e-10
    assert not scene.frozen_vertices.flags.writeable
    assert not scene.normalized_face_flux.flags.writeable


def test_balance_interfaces_are_the_only_active_seam_overlap(
    scene: elemental_render.ElementalRenderScene,
) -> None:
    cell_count = scene.solution.problem.domain.cell_count
    face_keys = (
        scene.face_cell_interfaces[:, 0] * cell_count
        + scene.face_cell_interfaces[:, 1]
    )
    balance_keys = frozenset(
        int(flux.left_cell) * cell_count + int(flux.right_cell)
        for flux in scene.solution.interface_fluxes
        if isinstance(flux, sheaf.SymmetricConductanceFlux)
    )
    assert frozenset(face_keys[scene.balance_interface_faces]) == balance_keys


def test_material_derivation_is_total_over_every_mesh_face(
    scene: elemental_render.ElementalRenderScene,
) -> None:
    materials = elemental_render.derive_face_materials(scene, phase=0.375)
    assert materials.shape == scene.frozen_face_colors.shape
    assert materials.dtype == np.uint8
    assert np.array_equal(materials[:, 3], scene.frozen_face_colors[:, 3])


def test_rune_relief_raises_only_balance_interface_vertices(
    scene: elemental_render.ElementalRenderScene,
) -> None:
    displacement = np.linalg.norm(
        scene.render_vertices - scene.frozen_vertices, axis=1
    )
    raised_vertices = np.unique(
        scene.faces[scene.balance_interface_faces].reshape(-1)
    )
    assert np.array_equal(np.flatnonzero(displacement > 0.0), raised_vertices)
    assert np.allclose(displacement[raised_vertices], 0.08)


def test_material_pulse_changes_only_balance_interface_seams(
    scene: elemental_render.ElementalRenderScene,
) -> None:
    first = elemental_render.derive_face_materials(scene, phase=0.0)
    later = elemental_render.derive_face_materials(scene, phase=0.5)
    changed_from_control = np.any(first != scene.frozen_face_colors, axis=1)
    assert np.any(changed_from_control)
    assert not np.any(changed_from_control & ~scene.balance_interface_faces)
    assert np.any(
        first[scene.balance_interface_faces]
        != later[scene.balance_interface_faces]
    )


def test_invalid_balance_problem_is_a_typed_rejection(
    mesh_path: Path, balance_problem: sheaf.BalanceProblem
) -> None:
    invalid = sheaf.BalanceProblem(
        problem_id=sheaf.BalanceProblemId("unanchored-render-problem"),
        domain=balance_problem.domain,
        symmetric_conductances=balance_problem.symmetric_conductances,
    )
    result = elemental_render.solve_elemental_scene(mesh_path, invalid)
    assert isinstance(result, elemental_render.RejectedElementalRender)
    assert result.obstructions == (
        elemental_render.BalanceSolveElementalRenderObstruction(
            invalid.problem_id,
            sheaf.UnanchoredBalanceComponentObstruction(invalid.problem_id, 0),
        ),
    )


def test_missing_mesh_is_an_addressed_typed_rejection(
    tmp_path: Path, balance_problem: sheaf.BalanceProblem
) -> None:
    missing = tmp_path / "does-not-exist.glb"
    result = elemental_render.solve_elemental_scene(missing, balance_problem)
    assert isinstance(result, elemental_render.RejectedElementalRender)
    assert isinstance(
        result.obstructions[0],
        elemental_render.ElementalRenderMeshLoadObstruction,
    )
    assert result.obstructions[0].path == missing


def test_turntable_cover_is_finite_and_deterministic(
    scene: elemental_render.ElementalRenderScene,
) -> None:
    first = elemental_render.render_elemental_frames(
        scene, frame_count=2, image_size=96
    )
    second = elemental_render.render_elemental_frames(
        scene, frame_count=2, image_size=96
    )
    assert len(first) == 2
    assert tuple(map(lambda frame: frame.size, first)) == ((96, 96), (96, 96))
    assert all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(first, second)
    )


def test_turntable_writer_round_trips_timing_and_background(
    tmp_path: Path, scene: elemental_render.ElementalRenderScene
) -> None:
    frames = elemental_render.render_elemental_frames(
        scene, frame_count=2, image_size=96
    )
    output = tmp_path / "elemental.gif"
    result = elemental_render.write_turntable(frames, output)
    assert result == elemental_render.AcceptedElementalRender(output)
    with Image.open(output) as animation:
        decoded = tuple(
            frame.convert("RGB").copy()
            for frame in ImageSequence.Iterator(animation)
        )
        assert animation.info["duration"] == 110
        assert animation.info["loop"] == 0
    assert len(decoded) == 2
    assert all(
        map(lambda frame: frame.getpixel((0, 0)) == (244, 244, 244), decoded)
    )


def test_empty_turntable_is_a_typed_rejection(tmp_path: Path) -> None:
    output = tmp_path / "empty.gif"
    result = elemental_render.write_turntable((), output)
    assert result == elemental_render.RejectedElementalRender(
        (
            elemental_render.EmptyElementalTurntableObstruction(output),
        )
    )
