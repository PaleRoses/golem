"""Closed REPL command grammar and pure decoder."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

from golem.session import ops


class CommandTag(StrEnum):
    SET = "set"
    UNSET = "unset"
    ADD = "add"
    REMOVE = "remove"
    RENAME = "rename"
    UNDO = "undo"
    REDO = "redo"
    OUTLINE = "outline"
    RECENT = "recent"
    ATTACH = "attach"
    REFLECT = "reflect"
    ARRAY = "array"
    SQUEEZE = "squeeze"
    ENVELOPE = "envelope"
    SAVE = "save"
    JOURNAL = "journal"
    MESH = "mesh"
    VIEWS = "views"
    BRANCH = "branch"
    BRANCHES = "branches"
    CHECKOUT = "checkout"
    COMPARE = "compare"
    MERGE = "merge"
    HELP = "help"
    QUIT = "quit"


class OutlineAspect(StrEnum):
    STRUCTURE = "structure"
    CONSTRAINT = "constraint"
    RECENT = "recent"


@dataclass(frozen=True)
class CommandSpec:
    tag: CommandTag
    usage: str
    summary: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommandLanguage:
    commands: tuple[CommandSpec, ...]
    address_lines: tuple[str, ...]
    add_kind_lines: tuple[str, ...]


LANGUAGE = CommandLanguage(
    (
        CommandSpec(CommandTag.SET, "set ADDR VALUE", "write one authored field; VALUE is JSON or a bareword"),
        CommandSpec(CommandTag.UNSET, "unset ADDR", "delete one authored field"),
        CommandSpec(CommandTag.ADD, "add KIND ADDR JSON", "insert one typed record"),
        CommandSpec(CommandTag.REMOVE, "remove ADDR", "delete one record"),
        CommandSpec(CommandTag.RENAME, "rename ADDR NEWNAME", "rename a bone or flesh and rewrite local references"),
        CommandSpec(CommandTag.UNDO, "undo", "apply and journal the latest inverse"),
        CommandSpec(CommandTag.REDO, "redo", "reapply the latest undone edit"),
        CommandSpec(CommandTag.OUTLINE, "outline [SCOPE] [DEPTH 1-3] [structure|constraint|recent]", "pull a bounded view"),
        CommandSpec(CommandTag.RECENT, "recent [N]", "show the latest transactions"),
        CommandSpec(CommandTag.ATTACH, "attach bone NAME at PARENT:T [JSON] | attach flesh BONE NAME JSON", "construct a relational attachment"),
        CommandSpec(CommandTag.REFLECT, "reflect BONE", "enable live bilateral reflection"),
        CommandSpec(CommandTag.ARRAY, "array BONE NAME N [T0,T1] TEMPLATE-JSON", "author one repeated flesh declaration"),
        CommandSpec(CommandTag.SQUEEZE, "squeeze NAME between A:TA B:TB [JSON]", "solve a bone between two anchors"),
        CommandSpec(CommandTag.ENVELOPE, "envelope JSON", "derive inline proportion clauses"),
        CommandSpec(CommandTag.SAVE, "save PATH", "write canonical authored JSON"),
        CommandSpec(CommandTag.JOURNAL, "journal PATH", "write deterministic JSONL"),
        CommandSpec(CommandTag.MESH, "mesh [RES]", "evaluate an on-demand mesh summary"),
        CommandSpec(CommandTag.VIEWS, "views PATH [RES]", "render a four-view contact sheet"),
        CommandSpec(CommandTag.BRANCH, "branch PURPOSE", "fork the current immutable session value"),
        CommandSpec(CommandTag.BRANCHES, "branches", "list branches"),
        CommandSpec(CommandTag.CHECKOUT, "checkout NAME", "select a branch"),
        CommandSpec(CommandTag.COMPARE, "compare A B", "compare two branch views"),
        CommandSpec(CommandTag.MERGE, "merge NAME", "atomically merge a disjoint suffix"),
        CommandSpec(CommandTag.HELP, "help", "show this grammar"),
        CommandSpec(CommandTag.QUIT, "quit", "leave the session", ("exit",)),
    ),
    (
        "meta@blend | skeleton/root | skeleton/<bone>@length",
        "skeleton/<bone>/<flesh>@field | pose/<bone>@pitch",
        "pose/goals/<id>@target | props/<prop>/<part>@field",
        "mounts/<port> | contacts/<group>",
    ),
    (
        " | ".join(kind.value for kind in ops.AddKind),
        "flesh kind: "
        + " | ".join(kind.value for kind in ops.FleshKind)
        + "; joint dof: "
        + " | ".join(degree.value for degree in ops.DegreeOfFreedom),
    ),
)


@dataclass(frozen=True)
class EmptyCommand:
    pass


@dataclass(frozen=True)
class CommandObstructed:
    output: str


@dataclass(frozen=True)
class EditCommand:
    operation: ops.Op


@dataclass(frozen=True)
class UndoCommand:
    pass


@dataclass(frozen=True)
class RedoCommand:
    pass


@dataclass(frozen=True)
class HelpCommand:
    pass


@dataclass(frozen=True)
class QuitCommand:
    pass


@dataclass(frozen=True)
class OutlineCommand:
    scope: str
    depth: int
    aspect: OutlineAspect


@dataclass(frozen=True)
class RecentCommand:
    count: int


@dataclass(frozen=True)
class AttachBoneCommand:
    name: str
    anchor: str
    params: dict


@dataclass(frozen=True)
class AttachFleshCommand:
    bone: str
    name: str
    params: dict


@dataclass(frozen=True)
class ReflectCommand:
    bone: str


@dataclass(frozen=True)
class ArrayCommand:
    bone: str
    name: str
    count: int
    along: list
    template: dict


@dataclass(frozen=True)
class SqueezeCommand:
    name: str
    anchor_a: str
    anchor_b: str
    params: dict


@dataclass(frozen=True)
class EnvelopeCommand:
    params: dict


@dataclass(frozen=True)
class SaveCommand:
    path: str


@dataclass(frozen=True)
class SaveJournalCommand:
    path: str


@dataclass(frozen=True)
class MeshCommand:
    resolution: int


@dataclass(frozen=True)
class ViewsCommand:
    path: str
    resolution: int


@dataclass(frozen=True)
class BranchCommand:
    purpose: str


@dataclass(frozen=True)
class BranchesCommand:
    pass


@dataclass(frozen=True)
class CheckoutCommand:
    name: str


@dataclass(frozen=True)
class CompareCommand:
    left: str
    right: str


@dataclass(frozen=True)
class MergeCommand:
    name: str


type Command = (
    EditCommand
    | UndoCommand
    | RedoCommand
    | HelpCommand
    | QuitCommand
    | OutlineCommand
    | RecentCommand
    | AttachBoneCommand
    | AttachFleshCommand
    | ReflectCommand
    | ArrayCommand
    | SqueezeCommand
    | EnvelopeCommand
    | SaveCommand
    | SaveJournalCommand
    | MeshCommand
    | ViewsCommand
    | BranchCommand
    | BranchesCommand
    | CheckoutCommand
    | CompareCommand
    | MergeCommand
)
type ParseResult = Command | EmptyCommand | CommandObstructed


def help_text(language: CommandLanguage = LANGUAGE) -> str:
    command_lines = tuple(
        f"  {spec.usage:<68} {spec.summary}" for spec in language.commands
    )
    return "\n".join(
        (
            "COMMANDS",
            *command_lines,
            "",
            "ADDRESS GRAMMAR",
            *(f"  {line}" for line in language.address_lines),
            "",
            "ADD KINDS",
            *(f"  {line}" for line in language.add_kind_lines),
        )
    )


def _spec_for(token: str) -> CommandSpec | None:
    return next(
        (
            spec
            for spec in LANGUAGE.commands
            if token == spec.tag.value or token in spec.aliases
        ),
        None,
    )


def _usage(spec: CommandSpec) -> CommandObstructed:
    return CommandObstructed(f"usage: {spec.usage}")


def _tokens(rest: str, spec: CommandSpec) -> tuple[str, ...] | CommandObstructed:
    try:
        return tuple(shlex.split(rest))
    except ValueError:
        return _usage(spec)


def _exact(rest: str, count: int, spec: CommandSpec) -> tuple[str, ...] | CommandObstructed:
    tokens = _tokens(rest, spec)
    return (
        tokens
        if isinstance(tokens, tuple) and len(tokens) == count
        else tokens
        if isinstance(tokens, CommandObstructed)
        else _usage(spec)
    )


def _json(raw: str, command: str) -> dict | list | int | float | str | bool | None | CommandObstructed:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        return CommandObstructed(f"{command}: params must be valid JSON ({exc})")


def _edit(raw: dict) -> EditCommand | CommandObstructed:
    decoded = ops.decode_op(raw)
    return (
        CommandObstructed(f"REJECT @ {decoded.first.address}: {decoded.first.reason}")
        if isinstance(decoded, ops.EditObstructed)
        else EditCommand(decoded)
    )


def _decode_edit_command(spec: CommandSpec, rest: str) -> ParseResult:
    match spec.tag:
        case CommandTag.SET:
            parts = rest.split(None, 1)
            if len(parts) != 2:
                return _usage(spec)
            address, raw_value = parts
            try:
                value = json.loads(raw_value)
            except json.JSONDecodeError:
                value = raw_value
            return _edit({"op": "set", "addr": address, "value": value})
        case CommandTag.ADD:
            parts = rest.split(None, 2)
            if len(parts) != 3:
                return _usage(spec)
            kind, address, raw_params = parts
            params = _json(raw_params, "add")
            return params if isinstance(params, CommandObstructed) else _edit(
                {"op": "add", "kind": kind, "addr": address, "params": params}
            )
        case CommandTag.UNSET | CommandTag.REMOVE:
            args = _exact(rest, 1, spec)
            if isinstance(args, CommandObstructed):
                return args
            return _edit({"op": spec.tag.value, "addr": args[0]})
        case CommandTag.RENAME:
            args = _exact(rest, 2, spec)
            if isinstance(args, CommandObstructed):
                return args
            return _edit({"op": "rename", "addr": args[0], "to": args[1]})
        case _:
            return _usage(spec)


def _decode_attach(spec: CommandSpec, rest: str) -> ParseResult:
    parts = rest.split(None, 4)
    if parts and parts[0] == "bone":
        if len(parts) < 4 or parts[2] != "at":
            return _usage(spec)
        params = _json(parts[4] if len(parts) == 5 else "{}", "attach")
        return params if isinstance(params, CommandObstructed) else (
            AttachBoneCommand(parts[1], parts[3], params)
            if isinstance(params, dict)
            else CommandObstructed("attach: params must be a JSON object")
        )
    if parts and parts[0] == "flesh":
        flesh_parts = rest.split(None, 3)
        if len(flesh_parts) != 4:
            return _usage(spec)
        params = _json(flesh_parts[3], "attach")
        return params if isinstance(params, CommandObstructed) else (
            AttachFleshCommand(flesh_parts[1], flesh_parts[2], params)
            if isinstance(params, dict)
            else CommandObstructed("attach: params must be a JSON object")
        )
    return _usage(spec)


def _decode_array(spec: CommandSpec, rest: str) -> ParseResult:
    parts = rest.split(None, 3)
    if len(parts) != 4:
        return _usage(spec)
    bone, name, count_raw, tail = parts
    try:
        count = int(count_raw)
        along, index = json.JSONDecoder().raw_decode(tail.strip())
        template = json.loads(tail.strip()[index:])
    except ValueError as exc:
        return CommandObstructed(f"array: N int, ALONG and TEMPLATE JSON ({exc})")
    return (
        ArrayCommand(bone, name, count, along, template)
        if isinstance(along, list) and isinstance(template, dict)
        else CommandObstructed("array: ALONG must be a JSON list and TEMPLATE a JSON object")
    )


def _decode_outline(spec: CommandSpec, rest: str) -> ParseResult:
    args = _tokens(rest, spec)
    if isinstance(args, CommandObstructed) or len(args) > 3:
        return args if isinstance(args, CommandObstructed) else _usage(spec)
    scope = args[0] if args else ""
    try:
        depth = int(args[1]) if len(args) >= 2 else 1
        aspect = OutlineAspect(args[2]) if len(args) >= 3 else OutlineAspect.STRUCTURE
    except ValueError:
        return _usage(spec)
    return OutlineCommand(scope, depth, aspect)


def _decode_numeric_optional(
    spec: CommandSpec,
    rest: str,
    default: int,
) -> int | CommandObstructed:
    args = _tokens(rest, spec)
    if isinstance(args, CommandObstructed) or len(args) > 1:
        return args if isinstance(args, CommandObstructed) else _usage(spec)
    try:
        return int(args[0]) if args else default
    except ValueError:
        return _usage(spec)


def _decode_command(spec: CommandSpec, rest: str) -> ParseResult:
    match spec.tag:
        case CommandTag.SET | CommandTag.UNSET | CommandTag.ADD | CommandTag.REMOVE | CommandTag.RENAME:
            return _decode_edit_command(spec, rest)
        case CommandTag.UNDO:
            return UndoCommand()
        case CommandTag.REDO:
            return RedoCommand()
        case CommandTag.HELP:
            return HelpCommand()
        case CommandTag.QUIT:
            return QuitCommand()
        case CommandTag.OUTLINE:
            return _decode_outline(spec, rest)
        case CommandTag.RECENT:
            count = _decode_numeric_optional(spec, rest, 5)
            return count if isinstance(count, CommandObstructed) else RecentCommand(count)
        case CommandTag.ATTACH:
            return _decode_attach(spec, rest)
        case CommandTag.REFLECT:
            args = _exact(rest, 1, spec)
            return args if isinstance(args, CommandObstructed) else ReflectCommand(args[0])
        case CommandTag.ARRAY:
            return _decode_array(spec, rest)
        case CommandTag.SQUEEZE:
            parts = rest.split(None, 4)
            if len(parts) < 4 or parts[1] != "between":
                return _usage(spec)
            params = _json(parts[4] if len(parts) == 5 else "{}", "squeeze")
            return params if isinstance(params, CommandObstructed) else (
                SqueezeCommand(parts[0], parts[2], parts[3], params)
                if isinstance(params, dict)
                else CommandObstructed("squeeze: params must be a JSON object")
            )
        case CommandTag.ENVELOPE:
            params = _json(rest.strip() or "{}", "envelope")
            return params if isinstance(params, CommandObstructed) else (
                EnvelopeCommand(params)
                if isinstance(params, dict)
                else CommandObstructed("envelope: params must be a JSON object")
            )
        case CommandTag.SAVE:
            args = _exact(rest, 1, spec)
            return args if isinstance(args, CommandObstructed) else SaveCommand(args[0])
        case CommandTag.JOURNAL:
            args = _exact(rest, 1, spec)
            return args if isinstance(args, CommandObstructed) else SaveJournalCommand(args[0])
        case CommandTag.MESH:
            resolution = _decode_numeric_optional(spec, rest, 120)
            return resolution if isinstance(resolution, CommandObstructed) else MeshCommand(resolution)
        case CommandTag.VIEWS:
            args = _tokens(rest, spec)
            if isinstance(args, CommandObstructed) or not 1 <= len(args) <= 2:
                return args if isinstance(args, CommandObstructed) else _usage(spec)
            try:
                resolution = int(args[1]) if len(args) == 2 else 96
            except ValueError:
                return _usage(spec)
            return ViewsCommand(args[0], resolution)
        case CommandTag.BRANCH:
            purpose = rest.strip()
            return BranchCommand(purpose) if purpose else _usage(spec)
        case CommandTag.BRANCHES:
            return BranchesCommand()
        case CommandTag.CHECKOUT:
            args = _exact(rest, 1, spec)
            return args if isinstance(args, CommandObstructed) else CheckoutCommand(args[0])
        case CommandTag.COMPARE:
            args = _exact(rest, 2, spec)
            return args if isinstance(args, CommandObstructed) else CompareCommand(*args)
        case CommandTag.MERGE:
            args = _exact(rest, 1, spec)
            return args if isinstance(args, CommandObstructed) else MergeCommand(args[0])
        case _ as unreachable:
            assert_never(unreachable)


def parse_command(line: str) -> ParseResult:
    stripped = line.strip()
    if not stripped:
        return EmptyCommand()
    parts = stripped.split(None, 1)
    token = parts[0]
    rest = parts[1] if len(parts) == 2 else ""
    spec = _spec_for(token)
    return (
        CommandObstructed(
            f"unknown command {token!r}; type \"help\" for the command reference"
        )
        if spec is None
        else _decode_command(spec, rest)
    )
