from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from golem.cli.model import (
    CommandDescriptor,
    CommandResult,
    CommandRunner,
    PositionalArgument,
    SwitchArgument,
)
from golem.cli.spec import LoadedSpec, SpecSourceObstruction, load_spec
from golem.kernel.blame import check_verdict, interrogate_blame
from golem.session.fault import (
    SESSION_SUBMIT_SEAM,
    EngineFault,
    project_engine_fault,
)
from golem.session.protocol import (
    AuthoringEvidence,
    Diagnostic,
    SessionSubmission,
    TransactionVerdict,
    VerdictDeltas,
    engine_fault_diagnostic,
    start_session,
    submit,
    witness_expected,
)


@dataclass(frozen=True)
class SessionArguments:
    base_path: Path
    input_path: Path
    blame: bool
    all_diagnostics: bool


def _session_arguments(namespace: Namespace) -> SessionArguments:
    return SessionArguments(
        namespace.base,
        namespace.input,
        namespace.blame,
        namespace.all_diagnostics,
    )


def _source_diagnostic(obstruction: SpecSourceObstruction) -> Diagnostic:
    return Diagnostic(
        code=f"session.source.{obstruction.kind}",
        severity="error",
        address=str(obstruction.path),
        predicate="source decodes as a JSON object",
        required="JSON object",
        observed=obstruction.reason,
        witness={
            "kind": obstruction.kind,
            "reason": obstruction.reason,
        },
        contract_refs=("schema",),
    )


def _source_verdict(obstruction: SpecSourceObstruction) -> TransactionVerdict:
    evidence = AuthoringEvidence(
        (_source_diagnostic(obstruction),),
        (),
    )
    return TransactionVerdict(
        evidence,
        VerdictDeltas(),
        (),
        0,
        0,
        (),
        (),
    )


def _fault_verdict(fault: EngineFault) -> TransactionVerdict:
    return TransactionVerdict(
        AuthoringEvidence((engine_fault_diagnostic(fault),), ()),
        VerdictDeltas(),
        (),
        0,
        0,
        (),
        (),
    )


def _render(
    verdict: TransactionVerdict,
    all_diagnostics: bool,
    blame: dict[str, object] | None = None,
) -> str:
    raw_record = verdict.to_json()
    raw_diagnostics = cast(
        tuple[Mapping[str, object], ...],
        raw_record["diagnostics"],
    )
    expected_diagnostics = tuple(
        filter(
            lambda diagnostic: witness_expected(diagnostic.get("witness")),
            raw_diagnostics,
        )
    )
    unexpected_diagnostics = tuple(
        filter(
            lambda diagnostic: not witness_expected(
                diagnostic.get("witness")
            ),
            raw_diagnostics,
        )
    )
    diagnostics = (
        raw_diagnostics if all_diagnostics else unexpected_diagnostics
    )
    record = {
        **raw_record,
        "diagnostics": diagnostics,
        "diagnostics_summary": {
            "total": len(raw_diagnostics),
            "expected": len(expected_diagnostics),
            "unexpected": len(unexpected_diagnostics),
            "shown": len(diagnostics),
            "expected_filtered": not all_diagnostics,
        },
    }
    return (
        json.dumps(
            record if blame is None else {**record, "blame": blame},
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _loaded_submission(base: LoadedSpec, transaction: LoadedSpec) -> SessionSubmission:
    session = start_session(base.payload, base.base_dir)
    return submit(session, transaction.payload)


def _base_rotation_warning(
    base: LoadedSpec,
    submission: SessionSubmission | None,
) -> str:
    return (
        "WARNING: the session CLI is stateless and the submitted document "
        "equals the base document; this fresh invocation has no cross-run "
        "delta. Rotate the last accepted edited document into `base` before "
        "the next invocation.\n"
        if submission is not None
        and submission.verdict.accepted
        and submission.session.document == base.payload
        else ""
    )


def _blame_section(
    submission: SessionSubmission,
    spec_dir: Path,
) -> dict[str, object]:
    if any(
        diagnostic.severity == "error"
        and diagnostic.code.partition(".")[0] in ("session", "document")
        for diagnostic in submission.verdict.evidence.diagnostics
    ):
        return {}
    document = submission.session.document
    probe = check_verdict(document, spec_dir)
    return interrogate_blame(
        document,
        probe.raw_obstructions,
        spec_dir=spec_dir,
        probe_timeout=2.0,
    ).to_json()


def run(namespace: Namespace) -> CommandResult:
    arguments = _session_arguments(namespace)
    base = load_spec(arguments.base_path)
    transaction = load_spec(arguments.input_path)
    obstruction = next(
        (
            source
            for source in (base, transaction)
            if isinstance(source, SpecSourceObstruction)
        ),
        None,
    )
    fault: EngineFault | None = None
    submission: SessionSubmission | None = None
    if (
        obstruction is None
        and isinstance(base, LoadedSpec)
        and isinstance(transaction, LoadedSpec)
    ):
        try:
            submission = _loaded_submission(base, transaction)
        except Exception as failure:
            fault = project_engine_fault(SESSION_SUBMIT_SEAM, failure)
    verdict = (
        _source_verdict(obstruction)
        if obstruction is not None
        else _fault_verdict(fault)
        if fault is not None
        else submission.verdict
        if submission is not None
        else _source_verdict(
            SpecSourceObstruction(
                arguments.input_path,
                "UnreachableSourceState",
                "source loader returned no typed result",
            )
        )
    )
    blame = (
        _blame_section(submission, base.base_dir)
        if arguments.blame
        and submission is not None
        and isinstance(base, LoadedSpec)
        else {}
        if arguments.blame
        else None
    )
    return CommandResult(
        0 if verdict.accepted else 1,
        stdout=_render(verdict, arguments.all_diagnostics, blame),
        stderr=(
            _base_rotation_warning(base, submission)
            if isinstance(base, LoadedSpec)
            else ""
        ),
    )


COMMAND = CommandDescriptor(
    name="session",
    help_line="submit edits or a replacement document through the session journal",
    runner=CommandRunner(
        arguments=(
            PositionalArgument(
                "base",
                "authoritative base body document; each CLI invocation starts "
                "a fresh in-memory session at txn 0. Rotate the last accepted "
                "edited document here before the "
                "next invocation",
                Path,
            ),
            PositionalArgument(
                "input",
                "transaction envelope or complete edited body document. "
                "Minimal envelope: "
                "{\"base_txn\": 0, \"operations\": [{\"op\": \"set\", "
                "\"addr\": \"meta@blend\", \"value\": 0.1}]}. "
                "Replacement envelope: "
                "{\"base_txn\": 0, \"document\": {...}}",
                Path,
            ),
            SwitchArgument(
                ("--blame",),
                "blame",
                "interrogate rejection blame after the canonical verdict",
            ),
            SwitchArgument(
                ("--all",),
                "all_diagnostics",
                "emit the raw diagnostics array, including witness.expected observations",
            ),
        ),
        evaluate=run,
    ),
)
