from __future__ import annotations

import json
import re
import subprocess
from argparse import Namespace
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import assert_never

from golem.cli.model import (
    CommandDescriptor,
    CommandResult,
    CommandRunner,
    OptionArgument,
    PositionalArgument,
)
from golem.cli.spec import LoadedSpec, SpecSourceObstruction, load_spec
from golem.session.protocol import (
    ProtocolSession,
    SessionSubmission,
    TransactionVerdict,
    start_session,
    submit,
)


@dataclass(frozen=True)
class ForgeArguments:
    brief: str
    output_path: Path
    rounds: int
    model: str


@dataclass(frozen=True)
class CandidateSpec:
    payload: dict[str, object]


@dataclass(frozen=True)
class MalformedAuthorReply:
    detail: str


@dataclass(frozen=True)
class MalformedInitialDocument:
    detail: str


type ParsedAuthorReply = CandidateSpec | MalformedAuthorReply
type InitialDocument = dict[str, object] | MalformedInitialDocument
type AuthoringChannel = Callable[[str], str]
type SubmissionRequest = dict[str, object]
type SubmissionChannel = Callable[
    [ProtocolSession, SubmissionRequest],
    SessionSubmission,
]


@dataclass(frozen=True)
class MalformedReplyRound:
    number: int
    obstruction: MalformedAuthorReply


@dataclass(frozen=True)
class SubmittedRound:
    number: int
    verdict: TransactionVerdict


type RoundEntry = MalformedReplyRound | SubmittedRound


@dataclass(frozen=True)
class ActiveForge:
    session: ProtocolSession
    entries: tuple[RoundEntry, ...]


@dataclass(frozen=True)
class AcceptedForge:
    session: ProtocolSession
    entries: tuple[RoundEntry, ...]


type ForgeState = ActiveForge | AcceptedForge


@dataclass(frozen=True)
class ForgeContext:
    arguments: ForgeArguments
    author: AuthoringChannel
    submission: SubmissionChannel


_FENCED_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_VERDICT_FEEDBACK_FIELDS = (
    "status",
    "diagnostics",
    "margins",
    "binding_constraint",
    "deltas",
    "contract_refs",
    "changed_addresses",
)


def _forge_arguments(namespace: Namespace) -> ForgeArguments:
    return ForgeArguments(
        brief=namespace.brief,
        output_path=namespace.output_path,
        rounds=namespace.rounds,
        model=namespace.model,
    )


def _parse_block(block: str, index: int) -> CandidateSpec | str:
    try:
        payload = json.loads(block)
    except json.JSONDecodeError as failure:
        return (
            f"block {index}: {failure.msg} at line {failure.lineno} "
            f"column {failure.colno}"
        )
    return (
        CandidateSpec(payload)
        if isinstance(payload, dict)
        else f"block {index}: expected a JSON object, received {type(payload).__name__}"
    )


def parse_author_reply(reply: str) -> ParsedAuthorReply:
    fenced = tuple(match.group(1) for match in _FENCED_BLOCK.finditer(reply))
    blocks = fenced if fenced else (reply,)
    parsed = tuple(
        _parse_block(block, index) for index, block in enumerate(blocks, 1)
    )
    candidates = tuple(item for item in parsed if isinstance(item, CandidateSpec))
    errors = tuple(item for item in parsed if isinstance(item, str))
    return (
        candidates[0]
        if candidates
        else MalformedAuthorReply("; ".join(errors) or "reply contained no JSON")
    )


def codex_authoring_channel(model: str) -> AuthoringChannel:
    def author(prompt: str) -> str:
        try:
            completed = subprocess.run(
                (
                    "codex",
                    "exec",
                    "-c",
                    'model_reasoning_effort="xhigh"',
                    "-c",
                    f"model={json.dumps(model)}",
                    "-",
                ),
                input=prompt,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as failure:
            return f"codex exec unavailable: {failure}"
        return (
            completed.stdout
            if completed.returncode == 0
            else "\n".join(
                (
                    f"codex exec rejected with exit code {completed.returncode}",
                    completed.stdout,
                    completed.stderr,
                )
            )
        )

    return author


def _verdict_feedback(verdict: TransactionVerdict) -> dict[str, object]:
    record = verdict.to_json()
    return {field: record[field] for field in _VERDICT_FEEDBACK_FIELDS}


def _latest_submitted_round(
    entries: tuple[RoundEntry, ...],
) -> SubmittedRound | None:
    return next(
        (
            entry
            for entry in reversed(entries)
            if isinstance(entry, SubmittedRound)
        ),
        None,
    )


def _latest_wire_obstruction(entries: tuple[RoundEntry, ...]) -> str | None:
    latest = entries[-1] if entries else None
    return (
        latest.obstruction.detail
        if isinstance(latest, MalformedReplyRound)
        else None
    )


def _authoring_prompt(context: ForgeContext, state: ActiveForge) -> str:
    latest = _latest_submitted_round(state.entries)
    feedback = (
        json.dumps(_verdict_feedback(latest.verdict), indent=2)
        if latest is not None
        else "null"
    )
    wire_obstruction = _latest_wire_obstruction(state.entries)
    return "\n".join(
        (
            "Revise the current creature specification against the GOLEM "
            "authoring contract.",
            "Return exactly one complete revised spec as a fenced ```json block.",
            "Do not return a patch, commentary-only answer, or partial document.",
            "Fetch any referenced contract section on demand with:",
            "python -m golem contract SECTION",
            "",
            "CREATURE BRIEF",
            context.arguments.brief,
            "",
            "CURRENT AUTHORITATIVE SPEC",
            "```json",
            json.dumps(state.session.document, indent=2),
            "```",
            "",
            "LATEST TRANSACTION VERDICT FEEDBACK",
            feedback,
            "",
            "LATEST AUTHORING WIRE OBSTRUCTION",
            wire_obstruction or "none",
            "",
        )
    )


def _render_document(document: Mapping[str, object]) -> str:
    return json.dumps(document, indent=2) + "\n"


def _persist_submission(path: Path, submission: SessionSubmission) -> None:
    path.write_text(
        _render_document(submission.session.document),
        encoding="utf-8",
    )


def _advance_round(
    context: ForgeContext,
    state: ForgeState,
    number: int,
) -> ForgeState:
    match state:
        case AcceptedForge():
            return state
        case ActiveForge(session, entries):
            parsed = parse_author_reply(
                context.author(_authoring_prompt(context, state))
            )
            match parsed:
                case MalformedAuthorReply() as obstruction:
                    return ActiveForge(
                        session,
                        (*entries, MalformedReplyRound(number, obstruction)),
                    )
                case CandidateSpec() as candidate:
                    submission = context.submission(
                        session,
                        {
                            "base_txn": session.txn,
                            "document": candidate.payload,
                        },
                    )
                    _persist_submission(context.arguments.output_path, submission)
                    next_entries = (
                        *entries,
                        SubmittedRound(number, submission.verdict),
                    )
                    return (
                        AcceptedForge(submission.session, next_entries)
                        if submission.verdict.accepted
                        else ActiveForge(submission.session, next_entries)
                    )
                case _ as unreachable:
                    assert_never(unreachable)
        case _ as unreachable:
            assert_never(unreachable)


def _diagnostic_codes(verdict: TransactionVerdict) -> tuple[str, ...]:
    diagnostics = verdict.to_json().get("diagnostics", ())
    return tuple(
        str(diagnostic.get("code", "UnknownDiagnostic"))
        for diagnostic in diagnostics
        if isinstance(diagnostic, Mapping)
    )


def _ledger_line(entry: RoundEntry) -> str:
    match entry:
        case MalformedReplyRound(number, obstruction):
            return (
                f"round {number}: author REJECTED - "
                f"[MalformedAuthorReply] {obstruction.detail}"
            )
        case SubmittedRound(number, verdict):
            status = "ACCEPTED" if verdict.accepted else "REJECTED"
            codes = _diagnostic_codes(verdict)
            suffix = f" - {', '.join(codes)}" if codes else ""
            return f"round {number}: session {status}{suffix}"
        case _ as unreachable:
            assert_never(unreachable)


def _final_tally(state: ForgeState) -> str:
    entries = state.entries
    malformed = sum(isinstance(entry, MalformedReplyRound) for entry in entries)
    rejected = sum(
        isinstance(entry, SubmittedRound) and not entry.verdict.accepted
        for entry in entries
    )
    verdict = "ACCEPTED" if isinstance(state, AcceptedForge) else "REJECTED"
    return (
        f"final tally: {verdict}; rounds={len(entries)}; "
        f"malformed_replies={malformed}; session_rejections={rejected}"
    )


def _load_initial_document(path: Path) -> InitialDocument:
    if not path.exists():
        return {}
    source = load_spec(path)
    return (
        source.payload
        if isinstance(source, LoadedSpec)
        else MalformedInitialDocument(f"{source.kind}: {source.reason}")
        if isinstance(source, SpecSourceObstruction)
        else MalformedInitialDocument("source loader returned no typed result")
    )


def forge(
    arguments: ForgeArguments,
    authoring_channel: AuthoringChannel,
    *,
    submission_channel: SubmissionChannel = submit,
) -> CommandResult:
    if arguments.rounds <= 0:
        return CommandResult(
            2,
            stderr="REJECTED [InvalidRoundBudget] --rounds must be positive\n",
        )
    initial_document = _load_initial_document(arguments.output_path)
    if isinstance(initial_document, MalformedInitialDocument):
        return CommandResult(
            1,
            stderr=(
                "REJECTED [MalformedInitialDocument] "
                f"{arguments.output_path}: {initial_document.detail}\n"
            ),
        )
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    context = ForgeContext(arguments, authoring_channel, submission_channel)
    state = reduce(
        lambda current, number: _advance_round(context, current, number),
        range(1, arguments.rounds + 1),
        ActiveForge(
            start_session(initial_document, arguments.output_path.parent),
            (),
        ),
    )
    stdout = "\n".join(
        (*tuple(map(_ledger_line, state.entries)), _final_tally(state), "")
    )
    return CommandResult(
        0 if isinstance(state, AcceptedForge) else 1,
        stdout=stdout,
    )


def run(namespace: Namespace) -> CommandResult:
    arguments = _forge_arguments(namespace)
    return forge(arguments, codex_authoring_channel(arguments.model))


COMMAND = CommandDescriptor(
    name="forge",
    help_line="author a GOLEM creature through structured session transactions",
    runner=CommandRunner(
        arguments=(
            PositionalArgument("brief", "creature authoring brief", str),
            OptionArgument(
                ("--out",),
                "output_path",
                "write the current complete spec JSON",
                Path,
                "FILE.JSON",
                required=True,
            ),
            OptionArgument(
                ("--rounds",),
                "rounds",
                "maximum authoring rounds",
                int,
                "N",
                12,
            ),
            OptionArgument(
                ("--model",),
                "model",
                "Codex model id",
                str,
                "MODEL",
                "gpt-5.6-sol",
            ),
        ),
        evaluate=run,
    ),
)
