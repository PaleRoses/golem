"""Immutable journal algebra with pure undo, redo, replay, and batching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from itertools import chain
from pathlib import Path

from golem.session import ops


class EntryKind(StrEnum):
    OP = "op"
    UNDO = "undo"
    REDO = "redo"


class RepairOperationKind(StrEnum):
    SCALE_RADIUS = "scale_radius"


@dataclass(frozen=True)
class JournalEntry:
    txn: int
    kind: EntryKind
    applied: ops.Op
    inverse: ops.Op
    label: str
    touched_addresses: frozenset[str]

    def to_json(self) -> dict:
        return {
            "txn": self.txn,
            "kind": self.kind.value,
            "applied": ops.to_json(self.applied),
            "inverse": ops.to_json(self.inverse),
            "label": self.label,
            "ts": self.txn,
        }


@dataclass(frozen=True)
class RepairJournalEvent:
    txn: int
    constraint: str
    address: str
    predicate: str
    authored: float
    repaired: float
    operation: RepairOperationKind
    factor: float
    authoring_addresses: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "txn": self.txn,
            "kind": "auto_repair",
            "constraint": self.constraint,
            "address": self.address,
            "predicate": self.predicate,
            "authored": self.authored,
            "repaired": self.repaired,
            "operation": {
                "operation": self.operation.value,
                "factor": self.factor,
            },
            "authoring_addresses": self.authoring_addresses,
            "ts": self.txn,
        }


@dataclass(frozen=True)
class JournalState:
    entries: tuple[JournalEntry, ...] = ()
    undo: tuple[int, ...] = ()
    redo: tuple[int, ...] = ()


@dataclass(frozen=True)
class JournalTransition:
    state: JournalState
    entry: JournalEntry
    edit: ops.EditPlan


@dataclass(frozen=True)
class JournalBatch:
    state: JournalState
    entries: tuple[JournalEntry, ...]


type JournalResult = JournalTransition | ops.EditObstructed


def empty() -> JournalState:
    return JournalState()


def _entry(
    state: JournalState,
    kind: EntryKind,
    applied: ops.Op,
    inverse: ops.Op,
    label: str,
    touched_addresses: frozenset[str],
) -> JournalEntry:
    txn = len(state.entries) + 1
    return JournalEntry(txn, kind, applied, inverse, label, touched_addresses)


def commit_edit(
    state: JournalState,
    applied: ops.Op,
    edit: ops.EditPlan,
    label: str | None = None,
) -> JournalTransition:
    entry = _entry(
        state,
        EntryKind.OP,
        applied,
        edit.inverse,
        label or edit.label,
        edit.touched_addresses,
    )
    return JournalTransition(
        JournalState((*state.entries, entry), (*state.undo, entry.txn), ()),
        entry,
        edit,
    )


@dataclass(frozen=True)
class _BatchAccumulator:
    state: JournalState
    entries: tuple[JournalEntry, ...]


def commit_program(
    state: JournalState,
    applied: tuple[ops.Op, ...],
    program: ops.ProgramPlan,
    label_prefix: str = "",
) -> JournalBatch:
    def commit_pair(
        accumulator: _BatchAccumulator,
        pair: tuple[ops.Op, ops.EditPlan],
    ) -> _BatchAccumulator:
        operation, edit = pair
        transition = commit_edit(
            accumulator.state,
            operation,
            edit,
            f"{label_prefix}{edit.label}",
        )
        return _BatchAccumulator(
            transition.state,
            (*accumulator.entries, transition.entry),
        )

    committed = reduce(
        commit_pair,
        zip(applied, program.edits, strict=True),
        _BatchAccumulator(state, ()),
    )
    return JournalBatch(committed.state, committed.entries)


def plan_undo(state: JournalState, spec: dict) -> JournalResult:
    if not state.undo:
        return ops.EditObstructed((ops.EditObstruction("journal", "nothing to undo"),))
    target_txn = state.undo[-1]
    target = state.entries[target_txn - 1]
    edit = ops.compile_op(spec, target.inverse)
    if isinstance(edit, ops.EditObstructed):
        return edit
    entry = _entry(
        state,
        EntryKind.UNDO,
        target.inverse,
        edit.inverse,
        f"undo txn {target_txn} ({target.label})",
        edit.touched_addresses,
    )
    return JournalTransition(
        JournalState((*state.entries, entry), state.undo[:-1], (*state.redo, target_txn)),
        entry,
        edit,
    )


def plan_redo(state: JournalState, spec: dict) -> JournalResult:
    if not state.redo:
        return ops.EditObstructed((ops.EditObstruction("journal", "nothing to redo"),))
    target_txn = state.redo[-1]
    target = state.entries[target_txn - 1]
    edit = ops.compile_op(spec, target.applied)
    if isinstance(edit, ops.EditObstructed):
        return edit
    entry = _entry(
        state,
        EntryKind.REDO,
        target.applied,
        edit.inverse,
        f"redo txn {target_txn} ({target.label})",
        edit.touched_addresses,
    )
    return JournalTransition(
        JournalState((*state.entries, entry), (*state.undo, target_txn), state.redo[:-1]),
        entry,
        edit,
    )


def recent(state: JournalState, count: int = 5) -> tuple[JournalEntry, ...]:
    return state.entries[-count:]


def touched_addresses(state: JournalState, since_txn: int = 0) -> frozenset[str]:
    return frozenset(
        chain.from_iterable(entry.touched_addresses for entry in state.entries[since_txn:])
    )


def replay_plan(spec: dict, entries: tuple[JournalEntry, ...]) -> ops.ProgramCompileResult:
    return ops.compile_program(spec, tuple(entry.applied for entry in entries))


def _decode_entry(data: dict) -> JournalEntry:
    applied = ops.from_json(data["applied"])
    inverse = ops.from_json(data["inverse"])
    touched = frozenset(
        {applied.addr.record_source, inverse.addr.record_source}
    )
    return JournalEntry(
        data["txn"],
        EntryKind(data["kind"]),
        applied,
        inverse,
        data["label"],
        touched,
    )


class Journal:
    __slots__ = ("_state",)

    def __init__(self, state: JournalState | None = None) -> None:
        self._state = state or empty()

    @classmethod
    def from_state(cls, state: JournalState) -> "Journal":
        return cls(state)

    @property
    def state(self) -> JournalState:
        return self._state

    @property
    def entries(self) -> list[dict]:
        return [entry.to_json() for entry in self._state.entries]

    def fork(self) -> "Journal":
        return Journal(self._state)

    def record(self, kind: str, applied: dict, inverse: dict, label: str) -> dict:
        applied_op = ops.from_json(applied)
        inverse_op = ops.from_json(inverse)
        entry = _entry(
            self._state,
            EntryKind(kind),
            applied_op,
            inverse_op,
            label,
            frozenset({applied_op.addr.record_source, inverse_op.addr.record_source}),
        )
        self._state = JournalState((*self._state.entries, entry), self._state.undo, self._state.redo)
        return entry.to_json()

    def commit_op(self, applied: dict, inverse: dict, label: str) -> dict:
        applied_op = ops.from_json(applied)
        inverse_op = ops.from_json(inverse)
        entry = _entry(
            self._state,
            EntryKind.OP,
            applied_op,
            inverse_op,
            label,
            frozenset({applied_op.addr.record_source, inverse_op.addr.record_source}),
        )
        self._state = JournalState(
            (*self._state.entries, entry),
            (*self._state.undo, entry.txn),
            (),
        )
        return entry.to_json()

    def undoable(self) -> int | None:
        return self._state.undo[-1] if self._state.undo else None

    def apply_undo(self, spec: dict) -> dict:
        result = plan_undo(self._state, spec)
        if isinstance(result, ops.EditObstructed):
            raise ops.Reject(result.first.address, result.first.reason)
        ops.commit_edit_plan(spec, result.edit)
        self._state = result.state
        return result.entry.to_json()

    def apply_redo(self, spec: dict) -> dict:
        result = plan_redo(self._state, spec)
        if isinstance(result, ops.EditObstructed):
            raise ops.Reject(result.first.address, result.first.reason)
        ops.commit_edit_plan(spec, result.edit)
        self._state = result.state
        return result.entry.to_json()

    def save(self, path: str | Path) -> None:
        lines = tuple(json.dumps(entry.to_json(), sort_keys=True) for entry in self._state.entries)
        Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))

    @staticmethod
    def load(path: str | Path) -> list[dict]:
        return [
            json.loads(line)
            for line in Path(path).read_text().splitlines()
            if line.strip()
        ]

    @staticmethod
    def replay(spec: dict, entries: list[dict]) -> dict:
        decoded = tuple(_decode_entry(entry) for entry in entries)
        result = replay_plan(spec, decoded)
        if isinstance(result, ops.EditObstructed):
            raise ops.Reject(result.first.address, result.first.reason)
        ops.commit_edit_plan(spec, result)
        return spec

    def recent(self, n: int = 5) -> list[dict]:
        return [entry.to_json() for entry in recent(self._state, n)]

    def touched_addresses(self, since_txn: int = 0) -> set[str]:
        return set(touched_addresses(self._state, since_txn))
