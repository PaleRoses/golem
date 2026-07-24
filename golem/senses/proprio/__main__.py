"""Entry point for ``python -m golem.senses.proprio`` (delegates to cli.main)."""

from __future__ import annotations

import sys

from golem.senses.proprio.cli import main

raise SystemExit(main(sys.argv[1:]))
