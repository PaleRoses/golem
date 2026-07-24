from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import trimesh
from numpy.typing import NDArray
from PIL import Image, ImageSequence
from scipy import ndimage

import control_creature
import engine
import hand_pack
import judge


ROOT = Path(__file__).resolve().parents[1]
PILOTS = ROOT / "pilots"
OUTPUTS = ROOT / "outputs"
GOLDENS = ROOT / "conformance" / "golden"

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type MeshData = tuple[NDArray[np.float64], NDArray[np.int64]]


class SealedObstruction(StrEnum):
    RAW_SYMMETRY_GATE = (
        "raw-symmetry-gate: 0.9799926081064433 is below the declared 0.98 minimum"
    )
    RENDER_PIXEL_DRIFT = (
        "render-pixel-drift: tied painter depths lack a canonical secondary order"
    )
    PLATED_GLB_ORDERING = (
        "plated-glb-ordering: colored-face export order is not canonical"
    )


@dataclass(frozen=True)
class PilotArtifacts:
    root: Path
    stdout: str


def read_json(path: Path) -> JsonValue:
    return cast(JsonValue, json.loads(path.read_text()))


def read_json_object(path: Path) -> dict[str, JsonValue]:
    value = read_json(path)
    assert isinstance(value, dict)
    return value


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_equivalent(left: JsonValue, right: JsonValue) -> bool:
    if left is None or isinstance(left, (bool, str)):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-14)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            map(lambda pair: json_equivalent(*pair), zip(left, right, strict=True))
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            map(lambda key: json_equivalent(left[key], right[key]), left)
        )
    return False


def run_pilot(
    destination: Path,
    script: str,
    arguments: tuple[str, ...] = (),
) -> PilotArtifacts:
    pilot_root = Path(shutil.copytree(PILOTS, destination / "pilots"))
    (pilot_root / "out").mkdir()
    completed = subprocess.run(
        (sys.executable, script, *arguments),
        cwd=pilot_root,
        env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    return PilotArtifacts(root=pilot_root, stdout=completed.stdout)


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)
    return mesh


def canonical_colored_faces(mesh: trimesh.Trimesh) -> NDArray[np.integer]:
    payload = np.hstack((mesh.faces, mesh.visual.face_colors))
    return payload[np.lexsort(payload.T[::-1])]


def decoded_frames(path: Path) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        return np.stack(
            tuple(
                map(
                    lambda frame: np.asarray(frame.convert("RGB"), dtype=np.uint8),
                    ImageSequence.Iterator(image),
                )
            )
        )


def unrounded_symmetry_iou(mesh: trimesh.Trimesh) -> float:
    _, voxel_grid = judge.occupancy(mesh)
    filled = ndimage.binary_fill_holes(np.asarray(voxel_grid.matrix, dtype=bool))
    voxel_indices = tuple(
        np.round((voxel_grid.points - voxel_grid.translation) / judge.PITCH)
        .astype(int)
        .T
    )
    points = voxel_grid.points[filled[voxel_indices]]
    occupied = frozenset(map(tuple, np.round(points / judge.PITCH).astype(int)))
    mirrored = frozenset(map(lambda point: (-point[0], point[1], point[2]), occupied))
    return len(occupied & mirrored) / len(occupied | mirrored)


@pytest.fixture(scope="session")
def v4_mesh_data() -> MeshData:
    vertices, faces, *_ = engine.evaluate(read_json_object(PILOTS / "golem_v4.json"), res=190)
    return vertices, faces


@pytest.fixture(scope="session")
def constrained_mesh(v4_mesh_data: MeshData) -> trimesh.Trimesh:
    vertices, faces = v4_mesh_data
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


@pytest.fixture(scope="session")
def control_mesh() -> trimesh.Trimesh:
    return control_creature.build()


@pytest.fixture(scope="session")
def constrained_symmetry_iou(constrained_mesh: trimesh.Trimesh) -> float:
    return unrounded_symmetry_iou(constrained_mesh)


@pytest.fixture(scope="session")
def mounted_artifacts(tmp_path_factory: pytest.TempPathFactory) -> PilotArtifacts:
    return run_pilot(tmp_path_factory.mktemp("mounted"), "mount_hands.py")


@pytest.fixture(scope="session")
def mounted_mesh(mounted_artifacts: PilotArtifacts) -> trimesh.Trimesh:
    vertices, faces, *_ = engine.evaluate(
        read_json_object(mounted_artifacts.root / "golem_hands.json"),
        res=190,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


@pytest.fixture(scope="session")
def plated_artifacts(tmp_path_factory: pytest.TempPathFactory) -> PilotArtifacts:
    return run_pilot(tmp_path_factory.mktemp("plated"), "plates.py")


@pytest.fixture(scope="session")
def rendered_artifacts(tmp_path_factory: pytest.TempPathFactory) -> PilotArtifacts:
    return run_pilot(
        tmp_path_factory.mktemp("rendered"),
        "render.py",
        ("golem_v4.json", "v4", "190", "gif"),
    )


@pytest.mark.parametrize(
    ("spec_name", "golden_name"),
    (("hand_v1.json", "hand_v1.txt"), ("hand_v2.json", "hand_v2.txt")),
)
def test_hand_report_golden(
    spec_name: str,
    golden_name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report, _ = hand_pack.run_pack(read_json_object(PILOTS / spec_name))
    hand_pack.print_report(report)
    assert capsys.readouterr().out == (GOLDENS / golden_name).read_text()


def test_v4_coherence_oracle(v4_mesh_data: MeshData) -> None:
    vertices, faces = v4_mesh_data
    assert engine.coherence_report(vertices, faces) == {
        "components": 1,
        "grid_dust_slivers": 0,
        "watertight_main": True,
        "largest_component_volume_share": 1.0,
        "faces": 174236,
    }


def test_mount_oracle(mounted_artifacts: PilotArtifacts) -> None:
    assert mounted_artifacts.stdout == (
        "ground solve: lowest point -0.230, lifted by 0.250\n"
        "forearm re-aimed at junction [1.28, 1.05, 0.76]\n"
        "parts: 14\n"
    )
    assert json_equivalent(
        read_json(mounted_artifacts.root / "golem_hands.json"),
        read_json(PILOTS / "golem_hands.json"),
    )
    assert sha256((mounted_artifacts.root / "golem_hands.json").read_bytes()) == (
        "2eb74cda86c62c8c014549ab4bb5aac99c7b8b8621187641ecd83ef555782628"
    )


def test_body_glb_is_byte_exact(constrained_mesh: trimesh.Trimesh) -> None:
    assert constrained_mesh.export(file_type="glb") == (OUTPUTS / "golem.glb").read_bytes()


def test_mounted_glb_is_byte_exact(mounted_mesh: trimesh.Trimesh) -> None:
    assert mounted_mesh.export(file_type="glb") == (OUTPUTS / "golem_hands.glb").read_bytes()


def test_plate_complex_oracle(plated_artifacts: PilotArtifacts) -> None:
    assert plated_artifacts.stdout == (
        "plates: 110 | seam pairs: 324 | seam faces: 38197 / 174236\n"
    )
    assert (plated_artifacts.root / "out" / "plate_complex.json").read_bytes() == (
        OUTPUTS / "plate_complex.json"
    ).read_bytes()


def test_plated_glb_semantic_oracle(plated_artifacts: PilotArtifacts) -> None:
    generated = load_mesh(plated_artifacts.root / "out" / "golem_plated.glb")
    frozen = load_mesh(OUTPUTS / "golem_plated.glb")
    assert np.array_equal(generated.vertices, frozen.vertices)
    assert np.array_equal(canonical_colored_faces(generated), canonical_colored_faces(frozen))


def test_plated_glb_drift_is_exactly_current_obstruction(
    plated_artifacts: PilotArtifacts,
) -> None:
    assert sha256(
        (plated_artifacts.root / "out" / "golem_plated.glb").read_bytes()
    ) == "b122704a289e9c5f1a36a65ee2539dc2ead249c2d55c743e66f460d8792b0fad"
    assert sha256((OUTPUTS / "golem_plated.glb").read_bytes()) == (
        "23adc421ec870abda26e6be928db40023981207a1086d10218b79ca26cad522e"
    )


def test_control_scorecard_oracle(
    constrained_mesh: trimesh.Trimesh,
    control_mesh: trimesh.Trimesh,
    capsys: pytest.CaptureFixture[str],
) -> None:
    judge.report(constrained_mesh, "CONSTRAINED (part graph + solver, golem_v4)")
    judge.report(control_mesh, "CONTROL (free primitives code, round 4)")
    assert capsys.readouterr().out == (
        "== CONSTRAINED (part graph + solver, golem_v4)\n"
        "   solid connected components: 1  (sizes: [251251])\n"
        "   mesh shells: 1\n"
        "   watertight single solid: True\n"
        "   bilateral symmetry IoU: 0.980\n"
        "== CONTROL (free primitives code, round 4)\n"
        "   solid connected components: 1  (sizes: [183664])\n"
        "   mesh shells: 35\n"
        "   watertight single solid: False\n"
        "   bilateral symmetry IoU: 0.998\n"
    )


def test_raw_symmetry_oracle(constrained_symmetry_iou: float) -> None:
    assert constrained_symmetry_iou == pytest.approx(0.9799926081064433)


@pytest.mark.xfail(reason=SealedObstruction.RAW_SYMMETRY_GATE.value, strict=True)
def test_raw_symmetry_meets_declared_gate(constrained_symmetry_iou: float) -> None:
    assert constrained_symmetry_iou >= 0.98


@pytest.mark.xfail(reason=SealedObstruction.RENDER_PIXEL_DRIFT.value, strict=True)
def test_reference_render_is_pixel_exact(rendered_artifacts: PilotArtifacts) -> None:
    assert all(
        map(
            lambda name: np.array_equal(
                decoded_frames(rendered_artifacts.root / "out" / name),
                decoded_frames(OUTPUTS / name),
            ),
            ("v4_views.png", "v4_turntable.gif"),
        )
    )


def test_reference_render_drift_is_exactly_current_obstruction(
    rendered_artifacts: PilotArtifacts,
) -> None:
    generated_views = decoded_frames(rendered_artifacts.root / "out" / "v4_views.png")
    frozen_views = decoded_frames(OUTPUTS / "v4_views.png")
    generated_turntable = decoded_frames(
        rendered_artifacts.root / "out" / "v4_turntable.gif"
    )
    frozen_turntable = decoded_frames(OUTPUTS / "v4_turntable.gif")
    assert generated_views.shape == frozen_views.shape
    assert generated_turntable.shape == frozen_turntable.shape
    assert int(np.any(generated_views != frozen_views, axis=-1).sum()) == 4
    assert int(np.abs(generated_views.astype(int) - frozen_views.astype(int)).max()) == 41
    assert int(np.any(generated_turntable != frozen_turntable, axis=-1).sum()) == 5
    assert int(
        np.abs(generated_turntable.astype(int) - frozen_turntable.astype(int)).max()
    ) == 41
    assert np.array_equal(generated_turntable[1:], frozen_turntable[1:])


@pytest.mark.xfail(reason=SealedObstruction.PLATED_GLB_ORDERING.value, strict=True)
def test_plated_glb_is_byte_exact(plated_artifacts: PilotArtifacts) -> None:
    assert (plated_artifacts.root / "out" / "golem_plated.glb").read_bytes() == (
        OUTPUTS / "golem_plated.glb"
    ).read_bytes()
