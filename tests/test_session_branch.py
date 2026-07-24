"""M4 acceptance tests for session branch-lite (ExecPlan M4 gate).

Branch-lite (``golem/session/core.py``) forks state+journal naively; ``merge``
replays a branch's journal suffix onto the current branch iff the touched
record addresses are disjoint (R-19 in miniature), otherwise it refuses and
names the overlap. The pilot is deep-copy where the Haskell kernel will share
structure, but the SURFACE -- fork, edit, compare, refuse-on-overlap, merge,
undo-across-merge -- is what M4 pins, and what these tests exercise on the
from-empty biped:

  1. a three-branch stance tournament: fork lean-fwd / lean-back / neutral off
     a shared trunk, compare two, list all four (one current marker), merge the
     untouched neutral (nothing to replay), then merge lean-fwd into an
     untouched main (disjoint -> applies; the pose lands in main's spec);
  2. overlap refusal: two branches that both edit ``skeleton/spine`` cannot
     merge -- the refusal names the scope;
  3. undo across a merge: the merged op is an ordinary journaled transaction,
     so ``undo`` reverts it and the state stays compilable;
  4. determinism: branch names, ``branches()`` and ``compare()`` are pure
     functions of the construction (AX-8: the symbolic surface is bit-exact).

The biped is replayed in-process through the repl dispatcher (outline lines,
being pure senses, are skipped for speed) so each test starts from the exact
creature the M3 golden froze.
"""

from __future__ import annotations

import pytest

from golem import paths as _paths
from golem.session import core
from golem.session import ops as _ops
from golem.session import repl as _repl

_SCRIPT = _paths.KERNEL_ROOT / "tests" / "data" / "biped_session.ops"


def _script_lines() -> list[str]:
    lines = []
    for raw in _SCRIPT.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _replay(skip_outline: bool = True) -> core.Session:
    """A fresh in-process biped on branch ``main`` (the M3 creature)."""
    s = core.Session.new("biped")
    for line in _script_lines():
        if skip_outline and line.startswith("outline"):
            continue
        _repl.execute(s, line)
    return s


# --------------------------------------------------------------------------- #
# 1. Three-branch stance tournament: fork, edit, compare, list, merge.         #
# --------------------------------------------------------------------------- #
def test_three_branch_stance_tournament() -> None:
    s = _replay()

    # branch() switches the current branch to the new fork; edits land there.
    b1 = s.branch("lean fwd")
    assert b1 == "b1-lean-fwd"
    s.do_line_op({"op": "set", "addr": "pose/spine@pitch", "value": 6})
    s.checkout("main")
    b2 = s.branch("lean back")
    s.do_line_op({"op": "set", "addr": "pose/spine@pitch", "value": -6})
    s.checkout("main")
    b3 = s.branch("neutral")                          # no edit

    # compare(): the two leans differ only in a pose angle, which changes
    # neither the depth-2 structure nor a contract status, so compare reports
    # the no-differences line -- an outcome the M4 gate accepts. Either way it
    # is a real, header-bearing diff report.
    report = s.compare(b1, b2)
    lines = report.splitlines()
    assert lines[0] == f"compare {b1} <> {b2}"
    assert ("no contract or structural differences" in report) or (len(lines) > 1)

    # branches(): four branches (main + three forks), exactly one current mark.
    listing = s.branches()
    assert len(listing.splitlines()) == 4
    assert listing.count("*") == 1

    # Merging the untouched neutral fork has nothing to replay.
    s.checkout("main")
    assert "nothing to merge" in s.merge(b3)

    # Merging lean-fwd is disjoint: main is untouched since the fork, so the
    # pose edit replays cleanly and lands in main's authored spec.
    merged = s.merge(b1)
    assert "merge[" in merged and "pose/spine" in merged
    assert s.state.spec["pose"]["joints"]["spine"]["pitch"] == 6
    assert s.state.cache.error is None


# --------------------------------------------------------------------------- #
# 2. Overlap refusal: two edits to one record cannot merge.                     #
# --------------------------------------------------------------------------- #
def test_merge_refuses_overlapping_scopes() -> None:
    s = _replay()
    b = s.branch("x")
    s.do_line_op({"op": "set", "addr": "skeleton/spine@length", "value": 0.36})
    s.checkout("main")
    s.do_line_op({"op": "set", "addr": "skeleton/spine@length", "value": 0.38})
    with pytest.raises(_ops.Reject) as excinfo:
        s.merge(b)
    assert "skeleton/spine" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 3. Undo across a merge: the merged op reverts, the state stays compilable.    #
# --------------------------------------------------------------------------- #
def test_undo_across_merge() -> None:
    s = _replay()
    b = s.branch("lean fwd")
    s.do_line_op({"op": "set", "addr": "pose/spine@pitch", "value": 6})
    s.checkout("main")
    s.merge(b)
    assert s.state.spec["pose"]["joints"].get("spine") == {"pitch": 6}

    s.undo()                                          # revert the merged op
    assert s.state.spec["pose"]["joints"].get("spine") is None
    assert s.state.cache.error is None


# --------------------------------------------------------------------------- #
# 4. Determinism: the branch surface is a pure function of its construction.    #
# --------------------------------------------------------------------------- #
def test_branch_outputs_are_deterministic() -> None:
    def build() -> tuple[str, str]:
        s = _replay()
        b1 = s.branch("lean fwd")
        s.do_line_op({"op": "set", "addr": "pose/spine@pitch", "value": 6})
        s.checkout("main")
        b2 = s.branch("lean back")
        s.do_line_op({"op": "set", "addr": "pose/spine@pitch", "value": -6})
        s.checkout("main")
        s.branch("neutral")
        return s.branches(), s.compare(b1, b2)

    first_branches, first_compare = build()
    second_branches, second_compare = build()
    assert isinstance(first_branches, str) and isinstance(first_compare, str)
    assert first_branches == second_branches
    assert first_compare == second_compare
