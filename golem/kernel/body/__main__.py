"""``python -m golem.kernel.body`` entry point (delegates to the CLI)."""

from __future__ import annotations

import sys

from golem.kernel.body.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
