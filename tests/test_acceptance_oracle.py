from __future__ import annotations

import json
from pathlib import Path

import pytest

from golem.cli.parser import main
from golem.paths import KERNEL_ROOT
from golem.session.protocol import start_session, submit
from golem.session.state import empty_spec


_VERDIGRIS_BURIED_FACE_LINE = {
    "id": "jade_belly_waist",
    "kind": "face_line",
    "part": "waist_line",
    "axis": "x",
    "width": 0.08,
    "face_normal": [0.0, -1.0, 0.0],
    "emit": ["band"],
    "material": "gambeson_dark",
}


def _write_spec(
    tmp_path: Path,
    name: str,
    document: dict[str, object],
) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_check_and_transaction_reject_the_same_assembly_surface(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = empty_spec("assembly-oracle-probe")
    candidate = {
        **base,
        "conduits": [
            {
                "id": "impossible-groove",
                "kind": "axial_loop",
                "part": "seed",
                "t": 0.5,
                "width": 0.04,
                "depth": 0.05,
                "emit": ["groove"],
            }
        ],
    }
    candidate_path = _write_spec(
        tmp_path,
        "assembly-oracle-probe.json",
        candidate,
    )

    check_exit_code = main(("check", str(candidate_path)))
    check_output = capsys.readouterr()
    submission = submit(start_session(base, tmp_path), candidate)

    assert check_exit_code == 1
    assert not submission.verdict.accepted
    assert check_output.err == ""
    assert "ASSEMBLY acceptance: REJECTED" in check_output.out
    assert "EmissionPitch" in check_output.out


def test_verdigris_buried_face_line_has_one_accepted_oracle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base_path = (
        KERNEL_ROOT / "rehearsal" / "verdigris-trial" / "base.json"
    )
    base = json.loads(base_path.read_text(encoding="utf-8"))
    # Evidence law (Wall 006): undeclared contacts block acceptance. The
    # sealed base carries eight blend_ambiguity observations the retired
    # severity-only law let ride; the candidate declares them as the
    # authored junctions they are (the base artifact itself stays sealed).
    blend_declarations = [
        ["dorsal_blade_thorax", "neck_base"],
        ["dorsal_blade_tail_root", "tail_mid"],
        ["dorsal_blade_tail_mid", "tail_taper"],
        ["haunch_mass", "wing_humerus"],
        ["fore_cannon", "wing_humerus"],
        ["dorsal_blade_thorax", "neck_rise"],
        ["wing_humerus", "tail_root"],
        ["dorsal_blade_thorax", "wing_humerus"],
    ]
    contract = {
        **base["contract"],
        "attach": [*base["contract"]["attach"], *blend_declarations],
    }
    candidate = {
        **base,
        "contract": contract,
        "conduits": [*base["conduits"], _VERDIGRIS_BURIED_FACE_LINE],
    }
    candidate_path = _write_spec(
        tmp_path,
        "verdigris-buried-face-line.json",
        candidate,
    )

    check_exit_code = main(("check", str(candidate_path)))
    check_output = capsys.readouterr()
    submission = submit(
        start_session(base, base_path.parent),
        candidate,
    )

    assert check_exit_code == 0
    assert submission.verdict.accepted
    assert check_output.err == ""
    assert "ASSEMBLY acceptance: ACCEPTED" in check_output.out
    assert "EmptySurfaceConduitObstruction" not in check_output.out
