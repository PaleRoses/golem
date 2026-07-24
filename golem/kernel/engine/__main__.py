"""Executable geometry-engine boundary."""

from __future__ import annotations

import sys

from .effect import main


raise SystemExit(main(sys.argv[1:]))
