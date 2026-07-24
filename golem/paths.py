"""Single source of repo geometry for the ``golem`` package.

Every location the living lane needs is named here, exactly once. The frozen
pilots are plain top-level modules (``engine``, ``render``, ``hand``,
``judge``); this module performs the ONE ``sys.path`` insertion that makes
them importable. No other module in the package may mutate ``sys.path`` --
the impurity is quarantined here, behind this boundary, and never spoken of
again.
"""

from __future__ import annotations

import sys
from pathlib import Path

KERNEL_ROOT = Path(__file__).resolve().parent.parent  # .../golem-kernel
PILOTS = KERNEL_ROOT / "pilots"        # frozen conformance oracle
REHEARSAL = KERNEL_ROOT / "rehearsal"  # experiment evidence
OUTPUTS = KERNEL_ROOT / "outputs"      # pilot-rendered regression targets
SPECS = KERNEL_ROOT / "specs"          # authored creature/intent specs
GOLDEN = KERNEL_ROOT / "golden"        # write-once golden texts

if str(PILOTS) not in sys.path:
    sys.path.insert(0, str(PILOTS))
