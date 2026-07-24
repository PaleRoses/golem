"""Run the deterministic check pipeline and normalize its rejection to a signature."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import assert_never

from golem.assembly import AcceptedAssembly, RejectedAssembly, compile_assembly
from golem.cli.spec import LoadedSpec, assembly_document
from golem.kernel.anatomy import (
    AcceptedAnatomy,
    AcceptedVasculature,
    RejectedAnatomy,
    RejectedVasculature,
    realize_vasculature,
)
from golem.kernel.blame.core import CheckVerdict, ObstructionSignature
from golem.kernel.body import Compiler, CompiledBody, RejectedBody
from golem.kernel.body.report import assertion_records_pass, run_asserts
from golem.kernel.engine.types import Rejected, decode_graph
from golem.senses.model import RejectedSenses


def obstruction_signature(obstruction: object) -> ObstructionSignature:
    """Project any obstruction onto (class name, string-valued locating fields).

    The identity fields are exactly the ``str``/``StrEnum`` members of the
    dataclass — region ids, lanes, phases, predicates, addresses — which is what
    distinguishes *where* a rejection lands. Floats and counts are quantitative
    magnitude, aggregated elsewhere, never part of presence-identity.
    """
    if isinstance(obstruction, ObstructionSignature):
        return obstruction
    kind = type(obstruction).__name__
    if not dataclasses.is_dataclass(obstruction):
        return ObstructionSignature(kind, ())
    identity = tuple(
        sorted(
            (field.name, str(getattr(obstruction, field.name)))
            for field in dataclasses.fields(obstruction)
            if isinstance(getattr(obstruction, field.name), str)
        )
    )
    return ObstructionSignature(kind, identity)


def _signatures(obstructions) -> tuple[ObstructionSignature, ...]:
    return tuple(map(obstruction_signature, obstructions))


def _rejected_verdict(stage: str, obstructions: tuple[object, ...]) -> CheckVerdict:
    return CheckVerdict(
        stage,
        rejected=True,
        resolved=True,
        obstructions=_signatures(obstructions),
        raw_obstructions=obstructions,
    )


def _receipt_violation_signatures(receipt: Mapping[str, object] | None) -> tuple[ObstructionSignature, ...]:
    if not isinstance(receipt, Mapping):
        return ()
    violations = receipt.get("violations", ())
    solved = receipt.get("solved", {})
    limits = solved.get("limit_violations", ()) if isinstance(solved, Mapping) else ()
    return (
        *tuple(
            ObstructionSignature(
                "BodyViolation",
                (
                    ("address", str(violation.get("address", "spec"))),
                    ("rule", str(violation.get("rule", "BodyViolation"))),
                ),
            )
            for violation in violations
            if isinstance(violation, Mapping)
        ),
        *tuple(
            ObstructionSignature(
                "JointLimitViolation", (("bone", str(violation.get("bone", "unknown"))),)
            )
            for violation in limits
            if isinstance(violation, Mapping)
        ),
    )


def check_verdict(spec: Mapping[str, object], spec_dir: Path) -> CheckVerdict:
    """Deterministic body-document verdict, faithful to ``cli.check`` staging.

    A raised exception on a forked spec becomes an *unresolved* verdict rather
    than a crash — the shrinker treats that as failure to reproduce the target.
    """
    try:
        compiled = Compiler(dict(spec), spec_dir=spec_dir).compile()
    except Exception:
        return CheckVerdict("error", rejected=True, resolved=False, obstructions=())
    if isinstance(compiled, RejectedBody):
        return _rejected_verdict("compile", tuple(compiled.obstructions))
    if not isinstance(compiled, CompiledBody):
        return CheckVerdict("error", rejected=True, resolved=False, obstructions=())
    try:
        graph, receipt, anatomy = compiled.graph, compiled.receipt, compiled.anatomy
        decoded = decode_graph(graph)
        if isinstance(decoded, Rejected):
            fatal = tuple(o for o in decoded.obstructions if getattr(o, "fatal", True))
            if fatal:
                return _rejected_verdict("geometry", fatal)
        if isinstance(anatomy, RejectedAnatomy):
            return _rejected_verdict("anatomy", tuple(anatomy.obstructions))
        if isinstance(anatomy, AcceptedAnatomy):
            vascular = realize_vasculature(anatomy, graph)
            if isinstance(vascular, (RejectedAnatomy, RejectedVasculature)):
                return _rejected_verdict("vascular", tuple(vascular.obstructions))
        body_violations = _receipt_violation_signatures(receipt)
        intent = graph.get("intent") if isinstance(graph.get("intent"), Mapping) else {}
        assertion_result = run_asserts(graph, dict(intent))
        match assertion_result:
            case RejectedSenses(obstructions=obstructions):
                return _rejected_verdict("senses", obstructions)
            case (_senses, records):
                pass
            case _ as unreachable:
                assert_never(unreachable)
        assert_signatures = (
            ()
            if assertion_records_pass(records)
            else tuple(
                ObstructionSignature(
                    "AssertionFailure",
                    (("id", str(record.get("id", record.get("address", "?")))),),
                )
                for record in records
                if record.get("status") == "fail"
            )
        )
    except Exception:
        return CheckVerdict("error", rejected=True, resolved=False, obstructions=())
    rejection = (*body_violations, *assert_signatures)
    if rejection:
        return _rejected_verdict("verdict", rejection)
    try:
        source = LoadedSpec(spec_dir / "blame.json", dict(spec))
        assembly = compile_assembly(assembly_document(source), spec_dir)
    except Exception:
        return CheckVerdict("error", rejected=True, resolved=False, obstructions=())
    if isinstance(assembly, RejectedAssembly):
        return _rejected_verdict("assembly", tuple(assembly.obstructions))
    if not isinstance(assembly, AcceptedAssembly):
        return CheckVerdict("error", rejected=True, resolved=False, obstructions=())
    return CheckVerdict("accept", rejected=False, resolved=True, obstructions=())
