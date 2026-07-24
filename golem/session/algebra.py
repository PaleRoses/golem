"""Pure session transitions over branches, journals, and edit programs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

from golem.session import commands, ops, outline
from golem.session.journal import (
    Journal,
    JournalEntry,
    JournalState,
    commit_program,
    empty,
    plan_redo as plan_journal_redo,
    plan_undo as plan_journal_undo,
    touched_addresses,
)
from golem.session.ops import _obstruct
from golem.session.state import CompileOutcome, SessionState

if TYPE_CHECKING:
    from golem.session.core import Session


@dataclass(frozen=True)
class Branch:
    name: str
    purpose: str
    state: SessionState
    journal: JournalState
    base_txn: int


@dataclass(frozen=True)
class SessionModel:
    branches: tuple[Branch, ...]
    current: str


@dataclass(frozen=True)
class PendingTransition:
    branch_name: str
    next_spec: dict
    next_journal: JournalState
    entries: tuple[JournalEntry, ...]
    previous: CompileOutcome


@dataclass(frozen=True)
class BranchCreated:
    model: SessionModel
    name: str


@dataclass(frozen=True)
class MergeEmpty:
    message: str


type TransitionResult = PendingTransition | ops.EditObstructed
type BranchResult = BranchCreated | ops.EditObstructed
type CheckoutResult = SessionModel | ops.EditObstructed
type MergeResult = PendingTransition | MergeEmpty | ops.EditObstructed


def initial(state: SessionState) -> SessionModel:
    return SessionModel((Branch("main", "trunk", state, empty(), 0),), "main")


def branch_named(model: SessionModel, name: str) -> Branch | None:
    return next((branch for branch in model.branches if branch.name == name), None)


def current_branch(model: SessionModel) -> Branch:
    branch = branch_named(model, model.current)
    return branch if branch is not None else model.branches[0]


def _replace_branch(model: SessionModel, replacement: Branch) -> SessionModel:
    return SessionModel(
        tuple(
            replacement if branch.name == replacement.name else branch
            for branch in model.branches
        ),
        model.current,
    )


def plan_program(
    model: SessionModel,
    operations: tuple[ops.Op, ...],
    label_prefix: str = "",
) -> TransitionResult:
    branch = current_branch(model)
    compiled = ops.compile_program(branch.state.spec, operations)
    if isinstance(compiled, ops.EditObstructed):
        return compiled
    journal_batch = commit_program(branch.journal, operations, compiled, label_prefix)
    return PendingTransition(
        branch.name,
        compiled.next_spec,
        journal_batch.state,
        journal_batch.entries,
        branch.state.outcome,
    )


def plan_raw_edit(model: SessionModel, raw: dict) -> TransitionResult:
    decoded = ops.decode_op(raw)
    return decoded if isinstance(decoded, ops.EditObstructed) else plan_program(model, (decoded,))


def _pending_journal_transition(
    model: SessionModel,
    transition,
) -> PendingTransition | ops.EditObstructed:
    if isinstance(transition, ops.EditObstructed):
        return transition
    branch = current_branch(model)
    return PendingTransition(
        branch.name,
        transition.edit.next_spec,
        transition.state,
        (transition.entry,),
        branch.state.outcome,
    )


def plan_undo(model: SessionModel) -> TransitionResult:
    branch = current_branch(model)
    return _pending_journal_transition(model, plan_journal_undo(branch.journal, branch.state.spec))


def plan_redo(model: SessionModel) -> TransitionResult:
    branch = current_branch(model)
    return _pending_journal_transition(model, plan_journal_redo(branch.journal, branch.state.spec))


def accept_transition(
    model: SessionModel,
    pending: PendingTransition,
    outcome: CompileOutcome,
) -> SessionModel:
    branch = branch_named(model, pending.branch_name)
    if branch is None:
        return model
    txn = pending.entries[-1].txn if pending.entries else branch.state.txn
    state = branch.state.transitioned(pending.next_spec, txn, outcome)
    return _replace_branch(
        model,
        Branch(branch.name, branch.purpose, state, pending.next_journal, branch.base_txn),
    )


def create_branch(model: SessionModel, purpose: str) -> BranchResult:
    slug = re.sub(r"[^a-z0-9]+", "-", purpose.lower()).strip("-")[:24] or "branch"
    name = f"b{len(model.branches)}-{slug}"
    source = current_branch(model)
    created = Branch(
        name,
        purpose,
        source.state.fork(),
        source.journal,
        len(source.journal.entries),
    )
    return BranchCreated(SessionModel((*model.branches, created), name), name)


def checkout(model: SessionModel, name: str) -> CheckoutResult:
    return (
        SessionModel(model.branches, name)
        if branch_named(model, name) is not None
        else _obstruct(name, f"no branch {name!r}")
    )


def render_branches(model: SessionModel) -> str:
    def status(branch: Branch) -> str:
        cache = branch.state.cache
        if cache.error is not None:
            return "COMPILE ERROR"
        failures = sum(record["status"] == "fail" for record in cache.records)
        passes = sum(record["status"] == "pass" for record in cache.records)
        return f"{failures} FAIL / {passes} pass"

    return "\n".join(
        f"{'*' if branch.name == model.current else ' '} {branch.name}  "
        f"txns {len(branch.journal.entries)}  {status(branch)}  ({branch.purpose})"
        for branch in model.branches
    )


def compare(model: SessionModel, left_name: str, right_name: str) -> str | ops.EditObstructed:
    left = branch_named(model, left_name)
    right = branch_named(model, right_name)
    missing = left_name if left is None else right_name if right is None else None
    if missing is not None:
        return _obstruct(missing, f"no branch {missing!r}")
    if left is None or right is None:
        return _obstruct("branch", "missing branch")

    def statuses(branch: Branch) -> dict:
        return (
            {}
            if branch.state.cache.error is not None
            else {
                record.get("id", index): record["status"]
                for index, record in enumerate(branch.state.cache.records)
            }
        )

    left_statuses = statuses(left)
    right_statuses = statuses(right)
    status_lines = tuple(
        f"  {key}: {left_statuses.get(key, '-')} <> {right_statuses.get(key, '-')}"
        for key in sorted(set(left_statuses) | set(right_statuses), key=str)
        if left_statuses.get(key, "-") != right_statuses.get(key, "-")
    )
    left_outline = set(
        outline.render(left.state, "", 2, "structure", Journal.from_state(left.journal)).splitlines()
    )
    right_outline = set(
        outline.render(right.state, "", 2, "structure", Journal.from_state(right.journal)).splitlines()
    )
    outline_lines = (
        *(f"  < {line.strip()}" for line in sorted(left_outline - right_outline)),
        *(f"  > {line.strip()}" for line in sorted(right_outline - left_outline)),
    )
    differences = (*status_lines, *outline_lines)
    tail = differences or ("  (no contract or structural differences at depth 2)",)
    return "\n".join((f"compare {left_name} <> {right_name}", *tail))


def plan_merge(model: SessionModel, source_name: str) -> MergeResult:
    source = branch_named(model, source_name)
    if source is None:
        return _obstruct(source_name, f"no branch {source_name!r}")
    if source_name == model.current:
        return _obstruct(source_name, "cannot merge a branch into itself")
    destination = current_branch(model)
    suffix = source.journal.entries[source.base_txn :]
    if not suffix:
        return MergeEmpty(f"merge {source_name}: nothing to merge")
    overlap = tuple(
        sorted(
            touched_addresses(source.journal, source.base_txn)
            & touched_addresses(destination.journal, source.base_txn)
        )
    )
    if overlap:
        return _obstruct(
            ", ".join(overlap),
            f"merge refused: overlapping scopes {list(overlap)}",
        )
    operations = tuple(entry.applied for entry in suffix)
    return plan_program(model, operations, f"merge[{source_name}] ")


@dataclass(frozen=True)
class ExitRequested:
    pass


type CommandResult = str | None | ExitRequested


def interpret_command(session: "Session", command: commands.Command) -> CommandResult:
    match command:
        case commands.EditCommand(operation):
            return session.do_typed_op(operation)
        case commands.UndoCommand():
            return session.undo()
        case commands.RedoCommand():
            return session.redo()
        case commands.HelpCommand():
            return commands.help_text()
        case commands.QuitCommand():
            return ExitRequested()
        case commands.OutlineCommand(scope, depth, aspect):
            return session.outline(scope, depth, aspect.value)
        case commands.RecentCommand(count):
            return session.recent(count)
        case commands.AttachBoneCommand(name, anchor, params):
            return session.attach_bone(name, anchor, params)
        case commands.AttachFleshCommand(bone, name, params):
            return session.attach_flesh(bone, name, params)
        case commands.ReflectCommand(bone):
            return session.reflect(bone)
        case commands.ArrayCommand(bone, name, count, along, template):
            return session.array(bone, name, count, along, template)
        case commands.SqueezeCommand(name, anchor_a, anchor_b, params):
            return session.squeeze(name, anchor_a, anchor_b, params)
        case commands.EnvelopeCommand(params):
            return session.envelope(params)
        case commands.SaveCommand(path):
            session.save_spec(path)
            return f"saved {path}"
        case commands.SaveJournalCommand(path):
            session.save_journal(path)
            return f"journal written {path}"
        case commands.MeshCommand(resolution):
            return session.mesh(resolution)
        case commands.ViewsCommand(path, resolution):
            return session.views(path, resolution)
        case commands.BranchCommand(purpose):
            return session.branch(purpose)
        case commands.BranchesCommand():
            return session.branches()
        case commands.CheckoutCommand(name):
            return session.checkout(name)
        case commands.CompareCommand(left, right):
            return session.compare(left, right)
        case commands.MergeCommand(name):
            return session.merge(name)
        case _ as unreachable:
            assert_never(unreachable)
