"""M3 acceptance tests for the session attachment grammar (ExecPlan M3 gate).

The M3 surface (``golem/session/core.py``) is ShapeAssembly adapted to bones:
every placement is relative to named structure -- an *anchor* ``<bone>:<t>``,
a parent-frame direction, a bone-local offset, a mirror, an array declaration,
or a ``squeeze`` that SOLVES a bone to span two anchors and records a
landmark-distance clause so the gluing stays two-sided and live. No op accepts
a world coordinate.

The evidence here freezes the from-empty biped authored through that grammar
(``tests/data/biped_session.ops``) and pins the lessons the plan made
permanent:

  1. the scripted biped receipts, byte-exact (golem.goldentext), behind a
     behavioral spine (contract 0 FAIL / 3 pass, the solved ``belt`` bone, no
     REJECT/ERROR);
  2. the file authors ZERO world coordinates -- a static grammar check;
  3. squeeze liveness (L-10): a clause on a landmarked station PASSES, then
     FAILS when the far anchor's geometry moves -- the two-sidedness is live;
  4. the perturbation-validity property (ShapeAssembly's lesson): scaling any
     authored length never crashes the compile -- every clause stays
     measurable;
  5. the grammar's refusals: world-coordinate fields, coincident anchors,
     squeeze across a mirror, a bad array template, and inline clauses on a
     file-referenced contract.

Goldens are write-once: establish once with GOLDEN_CREATE=1, byte-compare
forever.

History note: the first golden establishment exposed a repl tokenizer defect
(``rest.split(None, 4)`` broke the space-bearing ``[0.2, 0.9]`` list, so the
scripted array line never applied). The tokenizer now uses
``json.JSONDecoder.raw_decode`` and the golden was deliberately deleted and
re-established WITH the dorsal array applied -- that re-establishment is
recorded in the ExecPlan Decision Log.
"""

from __future__ import annotations

import random
import subprocess
import sys

import pytest

from golem import paths as _paths
from golem.goldentext import assert_matches_golden
from golem.session import core
from golem.session import ops as _ops
from golem.session import repl as _repl

_SCRIPT = _paths.KERNEL_ROOT / "tests" / "data" / "biped_session.ops"


# --------------------------------------------------------------------------- #
# Harness: the scripted run (subprocess, byte-exact) and an in-process replay. #
# --------------------------------------------------------------------------- #
def _run_session() -> str:
    """Run the scripted biped exactly as an operator would -- a subprocess,
    minimal environment, no wallclock or randomness in the output stream (the
    pattern of ``test_session_receipt``)."""
    proc = subprocess.run(
        [sys.executable, "-m", "golem.session.repl",
         "--new", "biped", "--script", str(_SCRIPT)],
        cwd=_paths.KERNEL_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"},
    )
    assert proc.stderr == "", proc.stderr
    return proc.stdout


def _script_lines() -> list[str]:
    """The executable lines of the ops script (comments and blanks dropped)."""
    lines = []
    for raw in _SCRIPT.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _replay(skip_outline: bool = True) -> core.Session:
    """A fresh in-process biped, driven through the SAME dispatcher the repl
    uses (so it is byte-for-byte what the operator would see). Outline lines
    are pure senses and are skipped by default for speed."""
    s = core.Session.new("biped")
    for line in _script_lines():
        if skip_outline and line.startswith("outline"):
            continue
        _repl.execute(s, line)
    return s


def _clause(session: core.Session, clause_id: str) -> dict | None:
    for r in session.state.cache.records:
        if r.get("id") == clause_id:
            return r
    return None


# --------------------------------------------------------------------------- #
# 1. The scripted biped receipts, byte-exact behind a behavioral spine.        #
# --------------------------------------------------------------------------- #
def test_biped_session_receipts_are_golden() -> None:
    out = _run_session()
    # Behavioral spine BEFORE freezing: the from-empty biped closes at the
    # documented contract, the ``squeeze`` emitted a solved bone, and no
    # command tripped the survival fence.
    assert "contract 0 FAIL / 3 pass" in out
    assert "add bone skeleton/belt" in out          # the solved squeeze bone
    assert "REJECT" not in out and "ERROR" not in out
    assert "add array skeleton/spine/dorsal" in out  # the ONE-txn declaration
    assert_matches_golden("session_biped_receipts.txt", out)


def test_receipts_are_deterministic() -> None:
    assert _run_session() == _run_session()


# --------------------------------------------------------------------------- #
# 2. No world coordinates: a static check on the authored ops file.            #
# --------------------------------------------------------------------------- #
def test_biped_ops_use_no_world_coordinates() -> None:
    # Only these verbs may open a line; ``add contact`` is the single permitted
    # raw ``add`` (placement is a named group, not a coordinate).
    allowed = ("envelope", "attach", "remove", "reflect", "array", "squeeze",
               "add contact", "set", "outline")
    for raw in _SCRIPT.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):        # comments are exempt
            continue
        assert '"world"' not in line, f"world-coordinate key in: {line!r}"
        assert '"center":' not in line, f"authored center in: {line!r}"
        assert '"spine": [' not in line, f"authored spine array in: {line!r}"
        assert line.startswith(allowed), f"non-grammar line: {line!r}"


# --------------------------------------------------------------------------- #
# 3. Squeeze liveness: a landmark clause that PASSES, then FAILS on a move.     #
# --------------------------------------------------------------------------- #
def test_squeeze_clause_is_live() -> None:
    # The scripted ``belt`` squeeze lands B at spine:0.22 -- NOT a landmarked
    # station (t in {0, 0.5, 1}) -- so it records no clause.
    s = _replay()
    assert "squeeze_belt" not in {r.get("id") for r in s.state.cache.records}

    # Squeeze a strap onto a landmarked station (neck:1 -> the bone's tail).
    # NOTE (brief adaptation): the brief squeezes to clavicle:1, but the
    # scripted clavicle is reflected, and squeeze refuses a mirrored anchor
    # ("not solvable one-sided"); ``neck`` is the nearest unmirrored bone whose
    # tail is a landmarked station, so the strap glues spine:0 -> neck:1.
    fresh = _replay()
    fresh.squeeze("strap", "spine:0", "neck:1", {})
    strap = _clause(fresh, "squeeze_strap")
    assert strap is not None, "landmarked squeeze must record a clause"
    assert strap["metric"] == "landmark_distance"
    assert strap["status"] == "pass"                 # solved to land on target
    assert set(strap["offenders"]) == {"landmark:strap/tail", "landmark:neck/tail"}

    # Move the FAR anchor's geometry: growing neck slides neck/tail away from
    # the strap tail the solve pinned -- the clause goes red. The gluing is
    # live in BOTH directions, which is the whole point of recording it.
    fresh.do_line_op({"op": "set", "addr": "skeleton/neck@length", "value": 0.3})
    strap_moved = _clause(fresh, "squeeze_strap")
    assert strap_moved is not None
    assert strap_moved["status"] == "fail"
    assert fresh.state.cache.error is None           # a fail is not a crash


# --------------------------------------------------------------------------- #
# 4. Perturbation validity: scaling any authored length never crashes.         #
# --------------------------------------------------------------------------- #
def _perturbation_property(iterations: int) -> None:
    s = _replay()
    lengths = {b["id"]: b["length"] for b in s.state.spec["skeleton"]["bones"]}
    bone_ids = sorted(lengths)
    # "unmeasurable" is only tolerated for clauses ALREADY unmeasurable at
    # baseline (the biped's contract is fully measurable, so this set is empty).
    baseline_unmeasurable = {r.get("id") for r in s.state.cache.records
                             if r["status"] == "unmeasurable"}
    rng = random.Random(11)
    for _ in range(iterations):
        bone = rng.choice(bone_ids)
        factor = rng.uniform(0.8, 1.2)
        new_len = round(lengths[bone] * factor, 4)
        s.do_line_op({"op": "set", "addr": f"skeleton/{bone}@length",
                      "value": new_len})
        assert s.state.cache.error is None, (
            f"perturbing {bone} -> {new_len} broke the compile: "
            f"{s.state.cache.error}"
        )
        for r in s.state.cache.records:
            status = r["status"]
            ok = (status in ("pass", "fail")
                  or (status == "unmeasurable"
                      and r.get("id") in baseline_unmeasurable))
            assert ok, (
                f"perturbing {bone} -> {new_len} left clause "
                f"{r.get('id')!r} in unexpected status {status!r}"
            )
        s.undo()                                     # back to the authored biped


def test_perturbation_validity_property_fast() -> None:
    _perturbation_property(25)


@pytest.mark.slow
def test_perturbation_validity_property() -> None:
    _perturbation_property(200)


# --------------------------------------------------------------------------- #
# 5. Refusals: the boundaries the grammar defends.                             #
# --------------------------------------------------------------------------- #
def test_attach_grammar_rejections() -> None:
    s = _replay()

    # A world-coordinate field is forbidden even under a valid anchor.
    with pytest.raises(_ops.Reject):
        s.attach_bone("bad", "spine:0", {"world": [0, 0, 0], "length": 0.2})

    # Coincident anchors leave nothing to squeeze.
    with pytest.raises(_ops.Reject):
        s.squeeze("bad", "spine:0.5", "spine:0.5", {})

    # A mirrored bone (thigh is reflected) cannot be squeezed one-sided.
    with pytest.raises(_ops.Reject):
        s.squeeze("bad", "thigh:0", "spine:0", {})

    # An array template must be a blob or box, never a gencyl.
    with pytest.raises(_ops.Reject):
        s.array("spine", "bad", 3, [0.2, 0.9],
                {"kind": "gencyl", "size": [0.1, 0.1, 0.1]})

    # Envelope refuses to write inline clauses onto a file-referenced contract.
    # The canonical knight is now a separated underbody and deliberately owns
    # no armor-era contract, so this boundary gets a minimal direct witness.
    knight = core.Session.new("file-contract-probe")
    knight.state.spec["contract"] = "frozen_contract.json"
    with pytest.raises(_ops.Reject):
        knight.envelope({"heads": 7})


# --------------------------------------------------------------------------- #
# 6. The array grammar works when invoked correctly (the scripted line is       #
#    defeated only by repl tokenization -- see the module docstring).           #
# --------------------------------------------------------------------------- #
def test_array_declaration_is_one_transaction() -> None:
    s = _replay()
    out = s.array("spine", "lateral_studs", 3, [0.3, 0.7],
                  {"kind": "blob", "size": [0.02, 0.02, 0.02],
                   "offset": [0.14, 0, 0]})
    assert "add array skeleton/spine/lateral_studs" in out
    assert s.state.cache.error is None
    spine = next(b for b in s.state.spec["skeleton"]["bones"] if b["id"] == "spine")
    assert any(a.get("name") == "lateral_studs" for a in spine.get("arrays", []))
    # the scripted dorsal array is already present via the repl path
    assert any(a.get("name") == "dorsal" for a in spine.get("arrays", []))
