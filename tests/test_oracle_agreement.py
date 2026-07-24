"""Oracle-agreement gates (wave-5, R10/R11): C15 poisoned engine, C16 witness.

C15 — the poisoned-engine agreement gate. ``compile_assembly`` is poisoned at
the dotted binding each oracle actually reaches (the house mock.patch seam
discipline): ``golem.session.protocol`` covers session and check (check
evaluates through the protocol), ``golem.cli.compile`` covers compile and
look (look calls ``compile_spec``). Every oracle must then surface the SAME
typed ``engine.fault`` witness — code, seam address, exception kind, message
digest, message — and the sentinel kind/message is asserted inside each
output so the agreement cannot pass with the poison unfired. This is the
regression net that would have caught wyvern wall 003 in minutes.

C16 — Cinderwake (``specs/cinderwake.json``, wyvern-trial lineage) as a
permanent oracle-agreement witness: cold ``check``, cold ``session``, and
``look`` must all descend to acceptance, pinned to the append-only
``rehearsal/wyvern-trial/verdict-04.json`` lineage (read, never written).
The three cold runs stay in THIS file in THIS order: the first pays the
~9s cold compile, the in-process engine caches then serve the others —
every test stays under the 10-second law while each still runs with
``GOLEM_CIRCUIT_CACHE_COLD=1`` (read at call time, so no disk-cache row is
consulted).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from golem.cli import check as check_cli
from golem.cli import compile as compile_cli
from golem.cli.parser import main
from golem.session import protocol as session_protocol
from golem.session.fault import (
    ASSEMBLY_COMPILE_SEAM,
    CHECK_DECODE_SEAM,
    ENGINE_FAULT_CODE,
    project_engine_fault,
)
from golem.session.state import empty_spec

_ROOT = Path(__file__).parents[1]
_CINDERWAKE = _ROOT / "specs" / "cinderwake.json"
_WYVERN_TRIAL = _ROOT / "rehearsal" / "wyvern-trial"

_POISON_MESSAGE = "oracle-agreement gate sentinel: assembly seam poisoned"
_POISON_DECODE_MESSAGE = "oracle-agreement gate sentinel: decode seam poisoned"
_POISON_FAULT = project_engine_fault(
    ASSEMBLY_COMPILE_SEAM,
    RuntimeError(_POISON_MESSAGE),
).to_json()


def _poisoned_compile_assembly(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError(_POISON_MESSAGE)


def _poisoned_decode_graph(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError(_POISON_DECODE_MESSAGE)


@pytest.fixture
def poisoned_assembly_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        session_protocol,
        "compile_assembly",
        _poisoned_compile_assembly,
    )
    monkeypatch.setattr(
        compile_cli,
        "compile_assembly",
        _poisoned_compile_assembly,
    )


@pytest.fixture
def poison_spec(tmp_path: Path) -> Path:
    spec = tmp_path / "poison.json"
    spec.write_text(json.dumps(empty_spec("poison")) + "\n")
    return spec


def _fault_jsons(payload: str, prefix: str) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line.removeprefix(prefix))
        for line in payload.splitlines()
        if line.startswith(prefix)
    )


def test_poisoned_engine_all_oracles_agree_on_one_typed_fault(
    poisoned_assembly_seam: None,
    poison_spec: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Session: the fault arrives as an engine.fault diagnostic whose witness
    # IS the canonical fault JSON.
    session_rc = main(("session", str(poison_spec), str(poison_spec)))
    session_out = capsys.readouterr()
    assert session_rc == 1
    verdict = json.loads(session_out.out)
    assert verdict["status"] == "rejected"
    session_faults = tuple(
        diagnostic["witness"]
        for diagnostic in verdict["diagnostics"]
        if diagnostic["code"] == ENGINE_FAULT_CODE
    )
    assert session_faults, "session swallowed the poisoned seam"

    # Check: the acceptance block renders the same diagnostic JSON.
    check_rc = main(("check", str(poison_spec)))
    check_out = capsys.readouterr()
    assert check_rc == 1
    check_faults = tuple(
        diagnostic["witness"]
        for diagnostic in _fault_jsons(
            check_out.out, f"REJECTED [{ENGINE_FAULT_CODE}] "
        )
    )
    assert check_faults, "check swallowed the poisoned seam"

    # Compile: the retired raw-escape path now refuses in type.
    compile_rc = main(("compile", str(poison_spec)))
    compile_out = capsys.readouterr()
    assert compile_rc == 1
    compile_faults = _fault_jsons(
        compile_out.err, f"REJECTED [{ENGINE_FAULT_CODE}] "
    )
    assert compile_faults, "compile swallowed the poisoned seam"
    assert "Traceback" not in compile_out.err

    # Look: typed refusal, never the wall-003 traceback.
    look_rc = main(("look", str(poison_spec), "--out", str(poison_spec.parent)))
    look_out = capsys.readouterr()
    assert look_rc == 1
    look_faults = _fault_jsons(
        look_out.err, f"REJECTED [{ENGINE_FAULT_CODE}] "
    )
    assert look_faults, "look swallowed the poisoned seam"
    assert "Traceback" not in look_out.err

    # The gate: one fault, identical across every oracle — and it is the
    # sentinel, so the agreement is proven fired, never vacuous.
    for emitted in (*session_faults, *check_faults, *compile_faults, *look_faults):
        assert emitted == _POISON_FAULT
    assert _POISON_FAULT["exception_kind"] == "RuntimeError"
    assert _POISON_FAULT["message"] == _POISON_MESSAGE
    assert _POISON_FAULT["seam"] == ASSEMBLY_COMPILE_SEAM


def test_check_decode_exception_projects_typed_fault(
    monkeypatch: pytest.MonkeyPatch,
    poison_spec: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(check_cli, "decode_graph", _poisoned_decode_graph)
    expected_fault = project_engine_fault(
        CHECK_DECODE_SEAM,
        RuntimeError(_POISON_DECODE_MESSAGE),
    ).to_json()

    assert main(("check", str(poison_spec))) == 1
    emitted = capsys.readouterr()
    faults = _fault_jsons(
        emitted.err,
        f"REJECTED [{ENGINE_FAULT_CODE}] {poison_spec}: ",
    )

    assert faults == (expected_fault,)
    assert faults[0]["code"] == ENGINE_FAULT_CODE
    assert faults[0]["seam"] == CHECK_DECODE_SEAM
    assert faults[0]["message"] == _POISON_DECODE_MESSAGE
    assert "Traceback" not in emitted.err


def test_cinderwake_check_accepts_cold(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GOLEM_CIRCUIT_CACHE_COLD", "1")
    assert main(("check", str(_CINDERWAKE), "--all")) == 0
    out = capsys.readouterr().out
    assert "ASSEMBLY acceptance: ACCEPTED" in out
    assert "REJECTED" not in out


def test_cinderwake_session_agrees_with_the_lineage(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GOLEM_CIRCUIT_CACHE_COLD", "1")
    rc = main(
        (
            "session",
            str(_WYVERN_TRIAL / "base.json"),
            str(_CINDERWAKE),
            "--all",
        )
    )
    assert rc == 0
    verdict = json.loads(capsys.readouterr().out)

    lineage = json.loads(
        (_WYVERN_TRIAL / "verdict-04.json").read_text(encoding="utf-8")
    )
    assert verdict["status"] == lineage["status"] == "accepted"
    for key in ("total", "expected", "unexpected"):
        assert verdict["diagnostics_summary"][key] == (
            lineage["diagnostics_summary"][key]
        )
    assert verdict["diagnostics_summary"]["unexpected"] == 0


def test_cinderwake_look_renders_cold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GOLEM_CIRCUIT_CACHE_COLD", "1")
    assert main(("look", str(_CINDERWAKE), "--out", str(tmp_path))) == 0
    rendered = {path.name for path in tmp_path.glob("*.png")}
    lineage = {
        path.name for path in (_WYVERN_TRIAL / "look-final").glob("*.png")
    }
    assert rendered == lineage != set()
    assert "Traceback" not in capsys.readouterr().err
