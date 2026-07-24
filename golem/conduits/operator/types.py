"""Immutable state, closed commands, results, and effect requests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from golem.assembly.carriers import AssemblyObstruction, AssemblyRecord
from golem.kernel.anatomy import VasculatureResult


DEFAULT_RESOLUTION = 120


@dataclass(frozen=True)
class HarnessObstruction:
    kind: str
    address: str
    reason: str


type CommandObstruction = HarnessObstruction | AssemblyObstruction


@dataclass(frozen=True)
class HarnessState:
    spec: dict
    spec_dir: Path
    resolution: int = DEFAULT_RESOLUTION
    journal: tuple[str, ...] = ()
    last_records: tuple[AssemblyRecord, ...] | None = None
    vascular_records: tuple[AssemblyRecord, ...] | None = None
    vascular_result: VasculatureResult | None = None


@dataclass(frozen=True)
class AcceptedCommand:
    state: HarnessState
    output: str | None
    quit: bool = False


@dataclass(frozen=True)
class RejectedCommand:
    state: HarnessState
    obstruction: CommandObstruction


type CommandResult = AcceptedCommand | RejectedCommand


@dataclass(frozen=True)
class NoCommand:
    pass


@dataclass(frozen=True)
class ListParts:
    pass


@dataclass(frozen=True)
class AddIntent:
    intent_id: str
    intent_kind: str
    declaration: dict


@dataclass(frozen=True)
class AddConduit:
    conduit_id: str
    declaration: dict


@dataclass(frozen=True)
class AddAppendage:
    appendage_id: str
    anchor_part: str
    anchor_station: str
    declaration: dict


@dataclass(frozen=True)
class SetDeclaration:
    declaration_kind: str
    declaration_id: str
    key: str
    value: object


@dataclass(frozen=True)
class RemoveDeclaration:
    authored_kind: str
    storage_key: str
    declaration_id: str


@dataclass(frozen=True)
class Grow:
    pass


@dataclass(frozen=True)
class SolveVascular:
    pass


@dataclass(frozen=True)
class ShowVascularReceipt:
    pass


@dataclass(frozen=True)
class RenderVascular:
    prefix: str


@dataclass(frozen=True)
class RenderLast:
    prefix: str


@dataclass(frozen=True)
class Save:
    path: Path


@dataclass(frozen=True)
class Show:
    pass


@dataclass(frozen=True)
class Help:
    pass


@dataclass(frozen=True)
class Quit:
    pass


type Command = (
    NoCommand
    | ListParts
    | AddIntent
    | AddConduit
    | AddAppendage
    | SetDeclaration
    | RemoveDeclaration
    | Grow
    | SolveVascular
    | ShowVascularReceipt
    | RenderVascular
    | RenderLast
    | Save
    | Show
    | Help
    | Quit
)


@dataclass(frozen=True)
class ResolvePartsEffect:
    pass


@dataclass(frozen=True)
class CompileSurfaceEffect:
    pass


@dataclass(frozen=True)
class SolveVascularEffect:
    pass


@dataclass(frozen=True)
class RenderRecordsEffect:
    records: tuple[AssemblyRecord, ...]
    prefix: str
    address: str


@dataclass(frozen=True)
class SaveSessionEffect:
    path: Path


@dataclass(frozen=True)
class LoadHelpEffect:
    pass


type EffectRequest = (
    ResolvePartsEffect
    | CompileSurfaceEffect
    | SolveVascularEffect
    | RenderRecordsEffect
    | SaveSessionEffect
    | LoadHelpEffect
)


@dataclass(frozen=True)
class RequestedEffect:
    state: HarnessState
    request: EffectRequest


type CommandTransition = CommandResult | RequestedEffect


__all__ = [
    "AcceptedCommand",
    "AddAppendage",
    "AddConduit",
    "AddIntent",
    "Command",
    "CommandObstruction",
    "CommandResult",
    "CommandTransition",
    "CompileSurfaceEffect",
    "DEFAULT_RESOLUTION",
    "EffectRequest",
    "Grow",
    "HarnessObstruction",
    "HarnessState",
    "Help",
    "ListParts",
    "LoadHelpEffect",
    "NoCommand",
    "Quit",
    "RejectedCommand",
    "RemoveDeclaration",
    "RenderLast",
    "RenderRecordsEffect",
    "RenderVascular",
    "RequestedEffect",
    "ResolvePartsEffect",
    "Save",
    "SaveSessionEffect",
    "SetDeclaration",
    "Show",
    "ShowVascularReceipt",
    "SolveVascular",
    "SolveVascularEffect",
]
