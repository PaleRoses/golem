"""Byte-exact golden-text harness for the golem lane.

Mirrors the discipline of ``conformance/golden/`` without touching it: a golden
is compared byte-for-byte, and it is **write-once**. A golden is only *created*
when the environment variable ``GOLDEN_CREATE=1`` is set, and an existing golden
is **never** overwritten -- if it exists, we always compare (even under
``GOLDEN_CREATE=1``) and a mismatch is a hard failure, not a silent refresh.

    from golem.goldentext import assert_matches_golden
    assert_matches_golden("symmetry_side_by_side.txt", rendered_text)

To (re)establish a golden intentionally, delete it and rerun once with
``GOLDEN_CREATE=1``.
"""

from __future__ import annotations

import os
from pathlib import Path

from golem.paths import GOLDEN as GOLDEN_DIR  # noqa: E402  (golden/)


class GoldenMissing(AssertionError):
    """Raised when a golden is absent and GOLDEN_CREATE is not set."""


def golden_path(name: str, golden_dir: Path | None = None) -> Path:
    return (golden_dir or GOLDEN_DIR) / name


def _creation_enabled() -> bool:
    return os.environ.get("GOLDEN_CREATE") == "1"


def assert_matches_golden(
    name: str,
    text: str,
    golden_dir: Path | None = None,
) -> None:
    """Compare ``text`` byte-exact against ``<golden_dir>/<name>``.

    - If the golden exists: compare. Equal -> pass; differ -> AssertionError.
      This branch runs even when GOLDEN_CREATE=1, so an existing golden is
      never overwritten.
    - If the golden is missing and GOLDEN_CREATE=1: create it, then pass.
    - If the golden is missing otherwise: raise GoldenMissing.
    """
    path = golden_path(name, golden_dir)
    payload = text.encode("utf-8")
    if path.exists():
        expected = path.read_bytes()
        if payload != expected:
            raise AssertionError(
                f"golden mismatch for {name!r} at {path}\n"
                f"--- expected ({len(expected)} bytes) ---\n"
                f"{expected.decode('utf-8', 'replace')}"
                f"--- actual ({len(payload)} bytes) ---\n"
                f"{text}"
                "Goldens are write-once; delete it and rerun with "
                "GOLDEN_CREATE=1 to intentionally re-establish it."
            )
        return
    if _creation_enabled():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return
    raise GoldenMissing(
        f"golden {name!r} missing at {path}; rerun with GOLDEN_CREATE=1 to create it"
    )
