"""GOLEM assembly subsystem -- elements composed by mounts (v0.5, D12/D13/D14/D15).

Doctrine (codified from Rosalia's 2026-07-17/18 directives):

An *assembly* is a set of *elements*, each a separate solid compiled in its own
field with its own integrity report and its own material. Cross-element
composition is rigid placement (a *mount*), never smooth-min: armor you put on
is never part of the body; a sword is held, not welded.

Local integrity is necessary and spectacularly insufficient. Before an
assembly is accepted, every pair restricts its final surface onto the other's
signed field. Undeclared pairs must be disjoint; fitted surfaces and bounded
mount contact carry explicit typed laws, obstructions, and receipts.

Elements carry a *role*, and the ontology strata (D13) apply DIFFERENTIALLY:

    creature  : full stack -- morphology, morphology architecture (skeleton or
                part-graph), welds, anatomy-owned closed conduit systems, and
                proprio/contract senses when authored in the body dialect.
    equipment : the lean subset -- morphology + material + anchors + integrity.
                No skeleton, no pose, no balance, no proprioception.

Equipment is authored ONCE in its own canonical frame (a sword: origin at the
grip base, +y along the handle, blade down -y) and placed by its mount's rigid
transform -- what makes equipment reusable across morphologies.

Subsystem shape (D15): every subsystem is a subpackage under ``golem/`` with a
core others hook into -- not a flat module. This package's rooms:

    core.py      the compiler spine: assembly dialect, element roles, per-
                 element compilation at uniform world pitch
    mounts.py    placement: mirror concretization and rigid mounting
    exchange.py  scene export (named-geometry GLB; M3's PBR slots hook here)
    __main__.py  the CLI: ``python -m golem.assembly <spec.json> [--glb out]``

Public API is re-exported here and is the stability surface for tests and
callers; the internal room layout may grow without breaking them.
"""

from golem.assembly.core import (  # noqa: F401
    AcceptedAssembly,
    AssemblyRecord,
    AssemblyResolutionPolicy,
    AssemblyResult,
    AssemblyStratum,
    ElementClearanceObstruction,
    ElementFitLaw,
    ElementFitObstruction,
    ElementFitReceipt,
    ElementRole,
    PinnedResolution,
    RES_MAX,
    RES_MIN,
    RejectedAssembly,
    TargetPitch,
    compile_assembly,
    load_element_graph,
    pitch_res,
)
from golem.assembly.exchange import export_scene  # noqa: F401
from golem.assembly.mounts import concretize_mirrors, mount_graph  # noqa: F401
from golem.assembly.project import project_assembly_obstructions

__all__ = [
    "compile_assembly",
    "AcceptedAssembly",
    "RejectedAssembly",
    "AssemblyResult",
    "AssemblyRecord",
    "AssemblyResolutionPolicy",
    "AssemblyStratum",
    "ElementClearanceObstruction",
    "ElementFitLaw",
    "ElementFitObstruction",
    "ElementFitReceipt",
    "ElementRole",
    "PinnedResolution",
    "TargetPitch",
    "load_element_graph",
    "pitch_res",
    "concretize_mirrors",
    "mount_graph",
    "export_scene",
    "project_assembly_obstructions",
    "RES_MIN",
    "RES_MAX",
]
