"""Receipt tests for the canonical separated-underbody session.

The script (``tests/data/knight_session.ops``) performs and undoes one reach
revision and one shoulder-mass revision.  The behavioral spine requires
addressed anomaly deltas, clean rejection status, and a byte-exact journal
receipt.  Armor and sword are assembly elements now; this test does not summon
their deleted body-era facsimiles.
"""

from __future__ import annotations

import subprocess
import sys

from golem import paths as _paths
from golem.goldentext import assert_matches_golden

_SCRIPT = _paths.KERNEL_ROOT / "tests" / "data" / "knight_session.ops"


def _run_session() -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "golem.session.repl",
         "specs/knight_body.json", "--script", str(_SCRIPT)],
        cwd=_paths.KERNEL_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"},
    )
    assert proc.stderr == "", proc.stderr
    return proc.stdout


def test_knight_session_receipts_are_golden() -> None:
    out = _run_session()
    # Behavioral spine before byte-freezing: both semantic addresses execute,
    # their anomaly deltas reverse, and the journal retains all four txns.
    assert "set skeleton/forearm@length" in out
    assert "set skeleton/clavicle/shoulder@size" in out
    assert "anomaly + cross_plane_fusion @ hand.L~hand.R" in out
    assert "anomaly - cross_plane_fusion @ hand.L~hand.R" in out
    assert "txn 4" in out
    assert "est_tokens:" in out
    assert "REJECT" not in out and "ERROR" not in out
    assert_matches_golden("session_knight_receipts.txt", out)


def test_receipts_are_deterministic() -> None:
    assert _run_session() == _run_session()
