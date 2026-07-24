"""Lower one solved balance section into the canonical plated renderer.

``golem.kernel.sheaf`` owns the domain, balance law, solution, and interface
fluxes.  This module owns only the derived appearance view: it glues frozen
mesh faces to the problem's cell cover, normalizes the authoritative section
and interface fluxes for colour, and delegates projection and Lambert shading
to ``pilots/render.py``.  It does not reconstruct an elemental-flow algebra or
preserve the retired scalar-field JSON dialect.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from math import fsum, pi
from pathlib import Path
from typing import assert_never

import numpy as np
import trimesh
from numpy.typing import NDArray
from PIL import Image
from scipy.spatial import cKDTree

from golem.kernel import sheaf
from golem.senses.model import immutable_array

from render import render_view  # pilots, frozen renderer


_DEFAULT_FRAME_COUNT = 24
_DEFAULT_IMAGE_SIZE = 420
_FRAME_DURATION_MILLISECONDS = 110
_SEAM_COLOR = np.asarray((168, 85, 247), dtype=np.uint8)
_RUNE_RELIEF = 0.08
_SOURCE_EMISSION = np.asarray((250.0, 240.0, 255.0), dtype=np.float64)
_TERMINAL_EMISSION = np.asarray((48.0, 238.0, 250.0), dtype=np.float64)


@dataclass(frozen=True)
class ElementalRenderScene:
    """Immutable appearance view derived from one accepted balance solution."""

    frozen_vertices: NDArray[np.float64]
    render_vertices: NDArray[np.float64]
    faces: NDArray[np.int64]
    frozen_face_colors: NDArray[np.uint8]
    face_cell_interfaces: NDArray[np.int64]
    seam_faces: NDArray[np.bool_]
    balance_interface_faces: NDArray[np.bool_]
    normalized_face_field: NDArray[np.float64]
    normalized_face_flux: NDArray[np.float64]
    solution: sheaf.BalanceSolution


@dataclass(frozen=True)
class BalanceSolveElementalRenderObstruction:
    problem_id: sheaf.BalanceProblemId
    obstruction: sheaf.Obstruction


@dataclass(frozen=True)
class ElementalRenderMeshLoadObstruction:
    path: Path
    reason: str


@dataclass(frozen=True)
class ElementalRenderMeshKindObstruction:
    path: Path
    actual_kind: str


@dataclass(frozen=True)
class ElementalRenderMeshGlueObstruction:
    path: Path
    reason: str


@dataclass(frozen=True)
class ElementalRenderFaceColorObstruction:
    path: Path
    expected_shape: tuple[int, int]
    actual_shape: tuple[int, ...]


@dataclass(frozen=True)
class ElementalRenderCellCoverObstruction:
    problem_id: sheaf.BalanceProblemId
    cell_count: int
    minimum_cell_count: int


@dataclass(frozen=True)
class EmptyElementalTurntableObstruction:
    path: Path


@dataclass(frozen=True)
class ElementalRenderWriteObstruction:
    path: Path
    reason: str


type ElementalRenderObstruction = (
    BalanceSolveElementalRenderObstruction
    | ElementalRenderMeshLoadObstruction
    | ElementalRenderMeshKindObstruction
    | ElementalRenderMeshGlueObstruction
    | ElementalRenderFaceColorObstruction
    | ElementalRenderCellCoverObstruction
    | EmptyElementalTurntableObstruction
    | ElementalRenderWriteObstruction
)


@dataclass(frozen=True)
class AcceptedElementalRender[value]:
    value: value


@dataclass(frozen=True)
class RejectedElementalRender:
    obstructions: tuple[ElementalRenderObstruction, ...]


type ElementalRenderResult[value] = (
    AcceptedElementalRender[value] | RejectedElementalRender
)


def solve_elemental_scene(
    mesh_path: Path,
    problem: sheaf.BalanceProblem,
) -> ElementalRenderResult[ElementalRenderScene]:
    """Solve the canonical balance problem and glue its section to one mesh."""
    solution_result = sheaf.solve_balance(problem)
    if isinstance(solution_result, sheaf.Rejected):
        return RejectedElementalRender(
            tuple(
                BalanceSolveElementalRenderObstruction(
                    problem.problem_id, obstruction
                )
                for obstruction in solution_result.obstructions
            )
        )
    solution = solution_result.value
    domain = solution.problem.domain
    if domain.cell_count < 2:
        return RejectedElementalRender(
            (
                ElementalRenderCellCoverObstruction(
                    problem.problem_id,
                    domain.cell_count,
                    2,
                ),
            )
        )
    try:
        loaded = trimesh.load_mesh(mesh_path, force="mesh", process=False)
    except (OSError, ValueError, TypeError, IndexError) as failure:
        return RejectedElementalRender(
            (ElementalRenderMeshLoadObstruction(mesh_path, str(failure)),)
        )
    if not isinstance(loaded, trimesh.Trimesh):
        return RejectedElementalRender(
            (
                ElementalRenderMeshKindObstruction(
                    mesh_path, type(loaded).__name__
                ),
            )
        )
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    face_colors = np.asarray(loaded.visual.face_colors, dtype=np.uint8)
    expected_color_shape = (len(faces), 4)
    if face_colors.shape != expected_color_shape:
        return RejectedElementalRender(
            (
                ElementalRenderFaceColorObstruction(
                    mesh_path, expected_color_shape, face_colors.shape
                ),
            )
        )
    try:
        return AcceptedElementalRender(
            _derive_elemental_scene(
                loaded, vertices, faces, face_colors, solution
            )
        )
    except (ValueError, TypeError, IndexError) as failure:
        return RejectedElementalRender(
            (ElementalRenderMeshGlueObstruction(mesh_path, str(failure)),)
        )


def _derive_elemental_scene(
    loaded: trimesh.Trimesh,
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    face_colors: NDArray[np.uint8],
    solution: sheaf.BalanceSolution,
) -> ElementalRenderScene:
    """Derive the immutable face view after both effects have succeeded."""
    domain = solution.problem.domain
    face_cell_interfaces = np.sort(
        np.asarray(
            cKDTree(
                np.asarray(domain.cell_centers, dtype=np.float64)
            ).query(vertices[faces[:, 0]], k=2)[1],
            dtype=np.int64,
        ),
        axis=1,
    )
    cell_count = domain.cell_count
    face_interface_keys = (
        face_cell_interfaces[:, 0] * cell_count + face_cell_interfaces[:, 1]
    )
    interface_fluxes = _aggregate_interface_fluxes(solution.interface_fluxes)
    balance_interface_keys = np.fromiter(
        (
            int(left) * cell_count + int(right)
            for (left, right), _ in interface_fluxes
        ),
        dtype=np.int64,
        count=len(interface_fluxes),
    )
    interface_values = np.fromiter(
        map(lambda interface_flux: interface_flux[1], interface_fluxes),
        dtype=np.float64,
        count=len(interface_fluxes),
    )
    seam_faces = np.all(face_colors[:, :3] == _SEAM_COLOR, axis=1)
    balance_interface_faces = seam_faces & np.isin(
        face_interface_keys, balance_interface_keys
    )
    normalized_cell_field = _normalize_values(
        np.asarray(
            tuple(map(lambda value: value.value, solution.field.values)),
            dtype=np.float64,
        ),
        solution.solver_config.normalization_floor,
    )
    normalized_face_field = 0.5 * (
        normalized_cell_field[face_cell_interfaces[:, 0]]
        + normalized_cell_field[face_cell_interfaces[:, 1]]
    )
    normalized_face_flux = _face_flux(
        face_interface_keys,
        balance_interface_keys,
        interface_values,
        solution.solver_config.normalization_floor,
    )
    rune_vertices = np.unique(faces[balance_interface_faces].reshape(-1))
    raised_vertices = np.isin(
        np.arange(len(vertices), dtype=np.int64), rune_vertices
    )
    render_vertices = vertices + (
        raised_vertices[:, None]
        * np.asarray(loaded.vertex_normals, dtype=np.float64)
        * _RUNE_RELIEF
    )
    return ElementalRenderScene(
        frozen_vertices=immutable_array(vertices),
        render_vertices=immutable_array(render_vertices),
        faces=immutable_array(faces),
        frozen_face_colors=immutable_array(face_colors),
        face_cell_interfaces=immutable_array(face_cell_interfaces),
        seam_faces=immutable_array(seam_faces),
        balance_interface_faces=immutable_array(balance_interface_faces),
        normalized_face_field=immutable_array(normalized_face_field),
        normalized_face_flux=immutable_array(normalized_face_flux),
        solution=solution,
    )


def derive_face_materials(
    scene: ElementalRenderScene,
    phase: float,
) -> NDArray[np.uint8]:
    """Derive a travelling balance-interface pulse for one phase."""
    face_field = scene.normalized_face_field[:, None]
    travelling_wave = np.power(
        0.5 + 0.5 * np.cos(2.0 * pi * (face_field + phase)), 4.0
    )
    face_signal = (
        scene.balance_interface_faces[:, None]
        * (0.16 + 0.84 * travelling_wave)
        * (0.35 + 0.65 * np.sqrt(scene.normalized_face_flux[:, None]))
    )
    emission = (
        face_field * _SOURCE_EMISSION
        + (1.0 - face_field) * _TERMINAL_EMISSION
    )
    signal = np.clip(6.0 * face_signal, 0.0, 1.0)
    frozen_rgb = scene.frozen_face_colors[:, :3].astype(np.float64)
    derived_rgb = np.rint(frozen_rgb + signal * (emission - frozen_rgb)).astype(
        np.uint8
    )
    return np.concatenate((derived_rgb, scene.frozen_face_colors[:, 3:]), axis=1)


def render_elemental_frame(
    scene: ElementalRenderScene,
    azimuth_degrees: float,
    phase: float,
    image_size: int = _DEFAULT_IMAGE_SIZE,
) -> Image.Image:
    """Render one balance-conditioned frame through the canonical renderer."""
    return render_view(
        scene.render_vertices,
        scene.faces,
        azimuth_degrees,
        elev_deg=12,
        size=image_size,
        face_colors=derive_face_materials(scene, phase),
    )


def render_elemental_frames(
    scene: ElementalRenderScene,
    frame_count: int = _DEFAULT_FRAME_COUNT,
    image_size: int = _DEFAULT_IMAGE_SIZE,
) -> tuple[Image.Image, ...]:
    """Render a synchronized turntable and balance-field cycle."""
    positions = (
        tuple(np.linspace(0.0, 1.0, frame_count, endpoint=False))
        if frame_count > 0
        else ()
    )
    return tuple(
        render_elemental_frame(
            scene,
            azimuth_degrees=360.0 * position,
            phase=position,
            image_size=image_size,
        )
        for position in positions
    )


def write_turntable(
    frames: tuple[Image.Image, ...], path: Path
) -> ElementalRenderResult[Path]:
    """Write one deterministic looping GIF or return a typed rejection."""
    if not frames:
        return RejectedElementalRender(
            (EmptyElementalTurntableObstruction(path),)
        )
    first, remaining = frames[0], frames[1:]
    try:
        first.save(
            path,
            save_all=True,
            append_images=remaining,
            duration=_FRAME_DURATION_MILLISECONDS,
            loop=0,
            disposal=2,
        )
    except (OSError, ValueError) as failure:
        return RejectedElementalRender(
            (ElementalRenderWriteObstruction(path, str(failure)),)
        )
    return AcceptedElementalRender(path)


def render_obstruction(obstruction: ElementalRenderObstruction) -> str:
    """Render one addressed appearance obstruction without erasing its type."""
    match obstruction:
        case BalanceSolveElementalRenderObstruction(problem_id, cause):
            return (
                f"BalanceSolveElementalRender @{problem_id}: "
                f"{sheaf.render_obstruction(cause)}"
            )
        case ElementalRenderMeshLoadObstruction(path, reason):
            return f"ElementalRenderMeshLoad @{path}: {reason}"
        case ElementalRenderMeshKindObstruction(path, actual_kind):
            return (
                f"ElementalRenderMeshKind @{path}: expected Trimesh, "
                f"got {actual_kind}"
            )
        case ElementalRenderMeshGlueObstruction(path, reason):
            return f"ElementalRenderMeshGlue @{path}: {reason}"
        case ElementalRenderFaceColorObstruction(path, expected, actual):
            return (
                f"ElementalRenderFaceColor @{path}: expected {expected}, "
                f"got {actual}"
            )
        case ElementalRenderCellCoverObstruction(
            problem_id, cell_count, minimum_cell_count
        ):
            return (
                f"ElementalRenderCellCover @{problem_id}: expected at least "
                f"{minimum_cell_count} cells, got {cell_count}"
            )
        case EmptyElementalTurntableObstruction(path):
            return f"EmptyElementalTurntable @{path}"
        case ElementalRenderWriteObstruction(path, reason):
            return f"ElementalRenderWrite @{path}: {reason}"
        case _ as unreachable:
            assert_never(unreachable)


def _interface_pair(
    interface_flux: sheaf.BalanceInterfaceFlux,
) -> tuple[sheaf.CellId, sheaf.CellId]:
    match interface_flux:
        case sheaf.SymmetricConductanceFlux(left_cell=left, right_cell=right):
            return (left, right) if int(left) < int(right) else (right, left)
        case sheaf.DirectedTransportFlux(
            upstream_cell=upstream, downstream_cell=downstream
        ):
            return (
                (upstream, downstream)
                if int(upstream) < int(downstream)
                else (downstream, upstream)
            )
        case _ as unreachable:
            assert_never(unreachable)


def _absolute_interface_flux(
    interface_flux: sheaf.BalanceInterfaceFlux,
) -> float:
    match interface_flux:
        case sheaf.SymmetricConductanceFlux(left_to_right_flux=flux):
            return abs(flux)
        case sheaf.DirectedTransportFlux(upstream_to_downstream_flux=flux):
            return abs(flux)
        case _ as unreachable:
            assert_never(unreachable)


def _aggregate_interface_fluxes(
    interface_fluxes: tuple[sheaf.BalanceInterfaceFlux, ...],
) -> tuple[tuple[tuple[sheaf.CellId, sheaf.CellId], float], ...]:
    keyed_fluxes = tuple(
        sorted(
            map(
                lambda interface_flux: (
                    _interface_pair(interface_flux),
                    _absolute_interface_flux(interface_flux),
                ),
                interface_fluxes,
            ),
            key=lambda item: tuple(map(int, item[0])),
        )
    )
    return tuple(
        (pair, fsum(map(lambda item: item[1], grouped)))
        for pair, grouped in groupby(keyed_fluxes, key=lambda item: item[0])
    )


def _face_flux(
    face_interface_keys: NDArray[np.int64],
    interface_keys: NDArray[np.int64],
    interface_values: NDArray[np.float64],
    normalization_floor: float,
) -> NDArray[np.float64]:
    if not interface_values.size:
        return np.zeros(face_interface_keys.shape, dtype=np.float64)
    positions = np.searchsorted(interface_keys, face_interface_keys)
    bounded_positions = np.minimum(positions, len(interface_keys) - 1)
    face_values = np.where(
        interface_keys[bounded_positions] == face_interface_keys,
        interface_values[bounded_positions],
        0.0,
    )
    maximum_flux = float(np.max(interface_values, initial=0.0))
    return (
        face_values / maximum_flux
        if maximum_flux > normalization_floor
        else np.zeros(face_values.shape, dtype=np.float64)
    )


def _normalize_values(
    values: NDArray[np.float64], normalization_floor: float
) -> NDArray[np.float64]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = maximum - minimum
    return (
        (values - minimum) / span
        if span > normalization_floor
        else np.zeros(values.shape, dtype=np.float64)
    )
