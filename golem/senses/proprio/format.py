"""Core: token accounting + platform-stable float formatting (leaf, math only)."""

from __future__ import annotations

import math


# --------------------------------------------------------------------------- #
# Token accounting -- ONE definition (Decision-Log).                          #
# --------------------------------------------------------------------------- #
def est_tokens(text: str) -> int:
    """Token estimate for the pinned tokenizer-less env: ceil(chars / 4)."""
    return math.ceil(len(text) / 4)


# --------------------------------------------------------------------------- #
# Float formatting -- <=3 decimals everywhere for cross-platform golden        #
# stability (libm drift on the sandbox; the sealed gate is macOS).             #
# --------------------------------------------------------------------------- #
def _nz(x: float) -> float:
    """Collapse -0.0 to 0.0 so formatting is platform-stable."""
    return 0.0 if x == 0 else x


def f3(x: float) -> str:
    return f"{_nz(round(float(x), 3)):.3f}"


def f2(x: float) -> str:
    return f"{_nz(round(float(x), 2)):.2f}"


def s3(x: float) -> str:
    """Signed, 3 decimals."""
    return f"{_nz(round(float(x), 3)):+.3f}"


def s2(x: float) -> str:
    return f"{_nz(round(float(x), 2)):+.2f}"
