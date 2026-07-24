"""Pure state transition algebra for closed harness commands."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import assert_never

from golem.conduits.operator.types import (
    AcceptedCommand,
    AddAppendage,
    AddConduit,
    AddIntent,
    Command,
    CommandTransition,
    CompileSurfaceEffect,
    Grow,
    HarnessObstruction,
    HarnessState,
    Help,
    ListParts,
    LoadHelpEffect,
    NoCommand,
    Quit,
    RejectedCommand,
    RemoveDeclaration,
    RenderLast,
    RenderRecordsEffect,
    RenderVascular,
    RequestedEffect,
    ResolvePartsEffect,
    Save,
    SaveSessionEffect,
    SetDeclaration,
    Show,
    ShowVascularReceipt,
    SolveVascular,
    SolveVascularEffect,
)
from golem.kernel.anatomy import vasculature_to_dict


def _declarations(state: HarnessState, key: str) -> tuple[dict, ...]:
    return tuple(state.spec.get(key, ()))


def _replace_declarations(
    state: HarnessState, key: str, declarations: tuple[dict, ...]
) -> HarnessState:
    return replace(state, spec={**state.spec, key: list(declarations)})


def _reject(
    state: HarnessState, kind: str, address: str, reason: str
) -> RejectedCommand:
    return RejectedCommand(state, HarnessObstruction(kind, address, reason))


def step(state: HarnessState, command: Command) -> CommandTransition:
    match command:
        case NoCommand():
            return AcceptedCommand(state, None)
        case ListParts():
            return RequestedEffect(state, ResolvePartsEffect())
        case AddIntent(intent_id, intent_kind, declaration):
            next_state = _replace_declarations(
                state,
                "_intents",
                (*_declarations(state, "_intents"), declaration),
            )
            return AcceptedCommand(
                next_state, f"intent {intent_id} ({intent_kind}) added"
            )
        case AddConduit(conduit_id, declaration):
            next_state = _replace_declarations(
                state,
                "conduits",
                (*_declarations(state, "conduits"), declaration),
            )
            return AcceptedCommand(
                next_state,
                f"conduit {conduit_id} ({declaration['kind']}) declared",
            )
        case AddAppendage(
            appendage_id, anchor_part, anchor_station, declaration
        ):
            next_state = _replace_declarations(
                state,
                "appendages",
                (*_declarations(state, "appendages"), declaration),
            )
            return AcceptedCommand(
                next_state,
                f"appendage {appendage_id} declared at "
                f"{anchor_part}:{anchor_station}",
            )
        case SetDeclaration(
            declaration_kind, declaration_id, key, value
        ):
            declarations = _declarations(state, declaration_kind)
            if not any(
                item.get("id") == declaration_id for item in declarations
            ):
                return _reject(
                    state,
                    "UnknownDeclaration",
                    f"/{declaration_kind}/{declaration_id}",
                    "no declaration at this address",
                )
            revised = tuple(
                {**item, key: value}
                if item.get("id") == declaration_id
                else item
                for item in declarations
            )
            return AcceptedCommand(
                _replace_declarations(state, declaration_kind, revised),
                f"{declaration_kind}.{declaration_id}.{key} = {value}",
            )
        case RemoveDeclaration(
            authored_kind, storage_key, declaration_id
        ):
            declarations = _declarations(state, storage_key)
            revised = tuple(
                item
                for item in declarations
                if item.get("id") != declaration_id
            )
            if len(revised) == len(declarations):
                return _reject(
                    state,
                    "UnknownDeclaration",
                    f"/{storage_key}/{declaration_id}",
                    "no declaration at this address",
                )
            return AcceptedCommand(
                _replace_declarations(state, storage_key, revised),
                f"{authored_kind} {declaration_id} removed",
            )
        case Grow():
            return RequestedEffect(state, CompileSurfaceEffect())
        case SolveVascular():
            return RequestedEffect(state, SolveVascularEffect())
        case ShowVascularReceipt():
            if state.vascular_result is None:
                return _reject(
                    state,
                    "MissingVascularSolve",
                    "/vascular/result",
                    "run 'vascular solve' first",
                )
            return AcceptedCommand(
                state,
                json.dumps(
                    {"vasculature": vasculature_to_dict(state.vascular_result)},
                    indent=2,
                    sort_keys=True,
                ),
            )
        case RenderVascular(prefix):
            if state.vascular_records is None:
                return _reject(
                    state,
                    "MissingVascularSolve",
                    "/vascular/result",
                    "run 'vascular solve' first",
                )
            return RequestedEffect(
                state,
                RenderRecordsEffect(
                    state.vascular_records, prefix, "/vascular/render"
                ),
            )
        case RenderLast(prefix):
            if state.last_records is None:
                return _reject(
                    state,
                    "MissingCompile",
                    "/harness/last_records",
                    "run 'grow' or 'vascular solve' first",
                )
            return RequestedEffect(
                state,
                RenderRecordsEffect(state.last_records, prefix, "/render"),
            )
        case Save(path):
            return RequestedEffect(state, SaveSessionEffect(path))
        case Show():
            return AcceptedCommand(state, json.dumps(state.spec, indent=1))
        case Help():
            return RequestedEffect(state, LoadHelpEffect())
        case Quit():
            return AcceptedCommand(state, "quit", quit=True)
        case unreachable:
            assert_never(unreachable)


__all__ = ["step"]
