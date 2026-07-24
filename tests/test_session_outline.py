"""M2 acceptance tests for the session outline sense (ExecPlan M2 gate).

The plan (docs/implementation/plans/creature/golem-v04-session-pilot.md, M2)
binds ``outline(scope, depth, aspect)`` to spec R-22: deterministic (same
state, same bytes); size-bounded by a declared per-depth token budget using
proprio's ``est_tokens``, truncating whole lines with a final ``... +K more @
<scope>`` address-carrying line; every content line address-bearing;
increasing depth only adds content lines (convergence).

Coverage here, one test per binding property:

* two separated-underbody outlines frozen byte-exact as write-once goldens
  (structure d1/d2);
* determinism across two freshly opened sessions for every (depth, aspect);
* budget: est_tokens(outline) <= declared budget + the final line's cost + 1,
  on both the knight and a minimal ``new`` session;
* convergence: depth-1 content lines subset depth-2 subset depth-3;
* address-bearing: every non-indented content line carries an address token;
* scope: an address prefix confines the outline to that subtree;
* the ``recent`` aspect digests the journal (an op then its undo).

Goldens are write-once (golem.goldentext): establish once with GOLDEN_CREATE=1,
byte-compare forever.
"""

from __future__ import annotations

from golem import paths as _paths
from golem.goldentext import assert_matches_golden
from golem.senses.proprio import est_tokens
from golem.session import core
from golem.session import outline as _outline

_KNIGHT = str(_paths.SPECS / "knight_body.json")

# Address tokens R-22 requires on every content line of a survey outline.
_ADDR_TOKENS = ("@ ", "skeleton/", "props/", "mounts/", "contacts/", "pose", "meta")


def _knight() -> core.Session:
    """A pristine session opened on the canonical separated underbody."""
    return core.Session.open(_KNIGHT)


def _content_lines(text: str) -> set[str]:
    """Content lines of an outline: everything but the ``... +K more`` truncation
    lines (which start with ``"... "`` and are excluded from convergence)."""
    return {ln for ln in text.splitlines() if not ln.startswith("... ")}


# --------------------------------------------------------------------------- #
# 1. Goldens: knight structure outline at depths 1 and 2, byte-exact.          #
# --------------------------------------------------------------------------- #
def test_knight_structure_outlines_are_golden() -> None:
    s = _knight()
    d1 = s.outline("", 1, "structure")
    d2 = s.outline("", 2, "structure")
    # Behavioral spine before byte-freezing: the survey is rooted, addressed,
    # and depth-2 exposes the level-2 bones depth-1 folds away.
    assert "@ skeleton/root" in d1
    assert "props/" not in d1
    assert "skeleton/spine_upper" not in d1  # level 2, hidden at depth 1
    assert "skeleton/spine_upper" in d2
    assert "skeleton/shin" in d2
    assert_matches_golden("session_knight_outline_d1.txt", d1)
    assert_matches_golden("session_knight_outline_d2.txt", d2)


# --------------------------------------------------------------------------- #
# 2. Determinism: two fresh sessions, byte-identical for every (depth, aspect).#
# --------------------------------------------------------------------------- #
def test_outline_is_deterministic_across_fresh_sessions() -> None:
    a, b = _knight(), _knight()
    for depth in (1, 2, 3):
        for aspect in ("structure", "constraint"):
            assert a.outline("", depth, aspect) == b.outline("", depth, aspect), (
                f"outline diverged at depth {depth} aspect {aspect!r}"
            )


# --------------------------------------------------------------------------- #
# 3. Budget: est_tokens(outline) within the declared per-depth budget, plus a  #
#    tolerance equal to the cost of the final (possibly truncation) line.       #
# --------------------------------------------------------------------------- #
def test_outline_respects_token_budget() -> None:
    sessions = {"knight": _knight(), "probe": core.Session.new("probe")}
    for label, s in sessions.items():
        for depth in (1, 2, 3):
            text = s.outline("", depth, "structure")
            lines = text.splitlines()
            last = lines[-1] if lines else ""
            budget = _outline._BUDGET[depth]
            tol = est_tokens(last) + 1  # the final line rides on top of the budget
            assert est_tokens(text) <= budget + tol, (
                f"{label} depth {depth}: {est_tokens(text)} tokens "
                f"> budget {budget} + tol {tol}"
            )


# --------------------------------------------------------------------------- #
# 4. Convergence: increasing depth only adds content lines.                     #
# --------------------------------------------------------------------------- #
def test_structure_depth_is_convergent() -> None:
    s = _knight()
    d1 = _content_lines(s.outline("", 1, "structure"))
    d2 = _content_lines(s.outline("", 2, "structure"))
    d3 = _content_lines(s.outline("", 3, "structure"))
    assert d1 <= d2, f"depth-1 lines not a subset of depth-2: {d1 - d2}"
    assert d2 <= d3, f"depth-2 lines not a subset of depth-3: {d2 - d3}"


# --------------------------------------------------------------------------- #
# 5. Address-bearing: every non-indented content line carries an address.       #
#    Excluded: blank lines, ``... `` truncation lines, and indented continuation #
#    lines (constraint clause text / bone detail rows begin with whitespace).    #
# --------------------------------------------------------------------------- #
def test_every_content_line_is_address_bearing() -> None:
    s = _knight()
    for aspect in ("structure", "constraint"):
        for depth in (1, 2):
            for ln in s.outline("", depth, aspect).splitlines():
                if not ln.strip():
                    continue
                if ln.startswith("... "):
                    continue
                if ln.startswith(" "):  # indented continuation / detail row
                    continue
                assert any(tok in ln for tok in _ADDR_TOKENS), (
                    f"line without address ({aspect} d{depth}): {ln!r}"
                )


# --------------------------------------------------------------------------- #
# 6. Scope: an address prefix confines the outline to that subtree.             #
# --------------------------------------------------------------------------- #
def test_scope_confines_the_outline() -> None:
    s = _knight()
    thigh = s.outline("skeleton/thigh", 3, "structure")
    assert "skeleton/thigh" in thigh
    assert "skeleton/forearm" not in thigh

    skeleton = s.outline("skeleton", 2, "structure")
    assert "skeleton/spine_lower" in skeleton
    assert "@ meta" not in skeleton


# --------------------------------------------------------------------------- #
# 7. Recent aspect: the journal digest names the transactions that ran.         #
# --------------------------------------------------------------------------- #
def test_recent_aspect_digests_the_journal() -> None:
    s = _knight()
    s.do_line_op({"op": "set", "addr": "skeleton/forearm@length", "value": 0.5})
    s.undo()
    recent = s.outline("", 1, "recent")
    assert "txn 1" in recent
    assert "txn 2" in recent
