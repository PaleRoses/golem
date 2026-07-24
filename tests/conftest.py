"""Test bootstrap for the living golem suite.

Path setup lives in ``pytest.ini`` (``pythonpath = pilots .``; kept out of
the hash-pinned ``pyproject.toml``) and ``golem.paths`` (the single pilots
bootstrap); this file only registers the ``slow`` marker. Run the fast suite
with ``-m "not slow"``.
"""

from __future__ import annotations


def pytest_configure(config) -> None:  # noqa: ANN001 (pytest hook)
    config.addinivalue_line(
        "markers",
        "slow: heavy frozen-corpus recomputation; skip with -m 'not slow'",
    )
