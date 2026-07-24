"""Public contract-evaluation surface over frozen proprioceptive senses."""

from __future__ import annotations

import sys

from golem.contracts.api import evaluate_clause, evaluate_pack
from golem.contracts.cli import main
from golem.contracts.metrics import _VIEW_AXES
from golem.contracts.verdicts import evaluate_verdict, evaluate_verdict_pack


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
