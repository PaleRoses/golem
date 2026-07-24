from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from golem.assembly import AcceptedAssembly, PinnedResolution, RejectedAssembly
from golem.assembly.exchange import export_scene
from golem.assembly.obstructions import ElementSurfaceDetailObstruction
from golem.cli.compile import compile_spec
from golem.kernel.engine.types import SurfaceDetailRule


_AMPLITUDE = 0.012
_GRAPH = {
    "name": "surface-detail-probe",
    "appearance_material": "gambeson_dark",
    "blend": 0.025,
    "parts": [
        {
            "id": "torso",
            "type": "blob",
            "center": [0.0, 0.0, 0.0],
            "size": [0.32, 0.42, 0.24],
        },
        {
            "id": "pectoral",
            "type": "blob",
            "center": [0.18, 0.16, 0.17],
            "size": [0.22, 0.18, 0.15],
            "mirror": True,
        },
        {
            "id": "deltoid",
            "type": "blob",
            "center": [0.34, 0.15, 0.02],
            "size": [0.17, 0.19, 0.17],
            "mirror": True,
        },
    ],
}


def _payload(surface_detail: object | None = None) -> dict[str, object]:
    return {
        **_GRAPH,
        **(
            {"surface_detail": surface_detail}
            if surface_detail is not None
            else {}
        ),
    }


def _compile(
    tmp_path: Path,
    name: str,
    payload: dict[str, object],
) -> AcceptedAssembly | RejectedAssembly:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = compile_spec(path, PinnedResolution(60))
    assert isinstance(result, (AcceptedAssembly, RejectedAssembly))
    return result


def test_opt_in_curvature_detail_is_signed_bounded_and_topology_preserving(
    tmp_path: Path,
) -> None:
    baseline = _compile(tmp_path, "baseline", _payload())
    detailed = _compile(
        tmp_path,
        "detailed",
        _payload(
            {
                "kind": "curvature",
                "amplitude": _AMPLITUDE,
                "scale": 0.11,
            }
        ),
    )
    assert isinstance(baseline, AcceptedAssembly)
    assert isinstance(detailed, AcceptedAssembly)
    baseline_record = baseline.records[0]
    detailed_record = detailed.records[0]
    displacement = np.linalg.norm(
        detailed_record.vertices - baseline_record.vertices,
        axis=1,
    )
    detail_report = detailed_record.report["surface_detail"]
    assert np.array_equal(detailed_record.faces, baseline_record.faces)
    assert np.count_nonzero(displacement > 1.0e-12) > 0
    assert float(np.max(displacement)) <= _AMPLITUDE + 1.0e-12
    assert detail_report["maximum_displacement"] <= _AMPLITUDE
    assert detail_report["convex_vertices"] > 0
    assert detail_report["concave_vertices"] > 0
    assert detailed_record.report["watertight_main"] is True
    assert detailed_record.report["components"] == 1


def test_zero_amplitude_surface_detail_is_byte_identical(
    tmp_path: Path,
) -> None:
    baseline = _compile(tmp_path, "baseline", _payload())
    zero = _compile(
        tmp_path,
        "zero",
        _payload(
            {
                "kind": "curvature",
                "amplitude": 0.0,
                "scale": 0.11,
            }
        ),
    )
    assert isinstance(baseline, AcceptedAssembly)
    assert isinstance(zero, AcceptedAssembly)
    baseline_record = baseline.records[0]
    zero_record = zero.records[0]
    assert baseline_record.vertices.tobytes() == zero_record.vertices.tobytes()
    assert baseline_record.faces.tobytes() == zero_record.faces.tobytes()
    assert baseline_record.report == zero_record.report
    assert baseline_record.violations == zero_record.violations
    baseline_path = tmp_path / "baseline.glb"
    zero_path = tmp_path / "zero.glb"
    export_scene(baseline, str(baseline_path))
    export_scene(zero, str(zero_path))
    assert baseline_path.read_bytes() == zero_path.read_bytes()
    assert Path(f"{baseline_path}.materials.json").read_bytes() == Path(
        f"{zero_path}.materials.json"
    ).read_bytes()


def test_surface_detail_failures_are_typed_obstructions(
    tmp_path: Path,
) -> None:
    rejected = _compile(
        tmp_path,
        "invalid",
        _payload(
            {
                "kind": "curvature",
                "amplitude": -0.01,
                "scale": 0.0,
            }
        ),
    )
    assert isinstance(rejected, RejectedAssembly)
    assert len(rejected.obstructions) == 1
    obstruction = rejected.obstructions[0]
    assert isinstance(obstruction, ElementSurfaceDetailObstruction)
    assert tuple(item.address for item in obstruction.obstructions) == (
        "/surface_detail/amplitude",
        "/surface_detail/scale",
    )
    assert tuple(item.rule for item in obstruction.obstructions) == (
        SurfaceDetailRule.FINITE_NON_NEGATIVE_AMPLITUDE,
        SurfaceDetailRule.FINITE_POSITIVE_SCALE,
    )
    assert tuple(item.authored for item in obstruction.obstructions) == (
        -0.01,
        0.0,
    )
    assert tuple(item.required for item in obstruction.obstructions) == (
        {"finite": True, "minimum": 0.0},
        {"finite": True, "exclusive_minimum": 0.0},
    )
    oversized = _compile(
        tmp_path,
        "oversized",
        _payload(
            {
                "kind": "curvature",
                "amplitude": 0.001,
                "scale": 100.0,
            }
        ),
    )
    assert isinstance(oversized, RejectedAssembly)
    scale_obstruction = oversized.obstructions[0]
    assert isinstance(scale_obstruction, ElementSurfaceDetailObstruction)
    assert tuple(item.address for item in scale_obstruction.obstructions) == (
        "/surface_detail/scale",
    )
    (scale_failure,) = scale_obstruction.obstructions
    assert scale_failure.rule is SurfaceDetailRule.SCALE_MESH_DIAGONAL
    assert scale_failure.authored == 100.0
    assert scale_failure.required["maximum"] > 0.0
