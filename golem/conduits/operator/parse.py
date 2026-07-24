"""Pure parser from harness command lines into the closed command sum."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

from golem.conduits.operator.types import (
    AddAppendage,
    AddConduit,
    AddIntent,
    Command,
    Grow,
    HarnessObstruction,
    Help,
    ListParts,
    NoCommand,
    Quit,
    RemoveDeclaration,
    RenderLast,
    RenderVascular,
    Save,
    SetDeclaration,
    Show,
    ShowVascularReceipt,
    SolveVascular,
)


type Converter = Callable[[str], object]
type ParsedCommand = Command | HarnessObstruction


def _vector(token: str) -> list[float]:
    return list(map(float, token.split(",")))


def _parse_tail(
    tokens: tuple[str, ...],
    declaration: dict,
    converters: dict[str, Converter],
) -> dict | HarnessObstruction:
    if not tokens:
        return declaration
    key = tokens[0]
    if key == "mirror":
        return _parse_tail(
            tokens[1:], {**declaration, "mirror": True}, converters
        )
    if key not in converters:
        return HarnessObstruction(
            "UnknownOption",
            f"command/options/{key}",
            f"unknown option {key!r}",
        )
    if len(tokens) < 2:
        return HarnessObstruction(
            "MissingOptionValue",
            f"command/options/{key}",
            f"option {key!r} requires a value",
        )
    try:
        value = converters[key](tokens[1])
    except (TypeError, ValueError) as exception:
        return HarnessObstruction(
            "MalformedOptionValue",
            f"command/options/{key}",
            str(exception),
        )
    return _parse_tail(
        tokens[2:], {**declaration, key: value}, converters
    )


def _parse_intent(arguments: tuple[str, ...]) -> ParsedCommand:
    if len(arguments) < 5 or arguments[0] != "add":
        return HarnessObstruction(
            "MalformedCommand",
            "command/intent",
            "expected: intent add <id> <blob|gencyl> ...",
        )
    intent_id, intent_kind = arguments[1], arguments[2]
    try:
        declaration = (
            {
                "id": intent_id,
                "type": "blob",
                "center": _vector(arguments[3]),
                "size": _vector(arguments[4]),
            }
            if intent_kind == "blob"
            else {
                "id": intent_id,
                "type": "gencyl",
                "spine": [_vector(arguments[3]), _vector(arguments[4])],
                "radii": [float(arguments[5]), float(arguments[6])],
            }
            if intent_kind == "gencyl" and len(arguments) >= 7
            else None
        )
    except (TypeError, ValueError) as exception:
        return HarnessObstruction(
            "MalformedIntent", f"/_intents/{intent_id}", str(exception)
        )
    return (
        AddIntent(intent_id, intent_kind, declaration)
        if declaration is not None
        else HarnessObstruction(
            "UnknownIntentKind",
            f"/_intents/{intent_id}",
            f"intent kind {intent_kind!r}; expected blob or gencyl",
        )
    )


def _parse_conduit(arguments: tuple[str, ...]) -> ParsedCommand:
    if len(arguments) < 4 or arguments[0] != "add":
        return HarnessObstruction(
            "MalformedCommand",
            "command/conduit",
            "expected: conduit add <loop|line> ...",
        )
    kind, conduit_id = arguments[1], arguments[2]
    try:
        if kind == "loop":
            part, station = arguments[3].split(":", maxsplit=1)
            parsed = _parse_tail(
                arguments[4:],
                {
                    "id": conduit_id,
                    "kind": "axial_loop",
                    "part": part,
                    "t": float(station),
                    "width": 0.03,
                    "emit": ["groove", "band"],
                    "material": "emissive_seam",
                },
                {
                    "width": float,
                    "depth": float,
                    "emit": lambda value: value.split(","),
                    "material": str,
                },
            )
        elif kind == "line":
            parsed = _parse_tail(
                arguments[4:],
                {
                    "id": conduit_id,
                    "kind": "face_line",
                    "part": arguments[3],
                    "axis": "x",
                    "width": 0.02,
                    "emit": ["band"],
                    "material": "emissive_seam",
                },
                {
                    "axis": str,
                    "width": float,
                    "depth": float,
                    "normal": _vector,
                    "emit": lambda value: value.split(","),
                    "material": str,
                },
            )
        else:
            return HarnessObstruction(
                "UnknownConduitKind",
                f"/conduits/{conduit_id}/kind",
                f"kind {kind!r}; expected loop or line; vasculature uses 'vascular solve'",
            )
    except (TypeError, ValueError) as exception:
        return HarnessObstruction(
            "MalformedConduit", f"/conduits/{conduit_id}", str(exception)
        )
    if isinstance(parsed, HarnessObstruction):
        return parsed
    declaration = (
        {
            **{key: value for key, value in parsed.items() if key != "normal"},
            "face_normal": parsed["normal"],
        }
        if "normal" in parsed
        else parsed
    )
    return AddConduit(conduit_id, declaration)


def _parse_appendage(arguments: tuple[str, ...]) -> ParsedCommand:
    if (
        len(arguments) < 6
        or arguments[0] != "add"
        or arguments[2] != "anchor"
        or arguments[4] != "intent"
    ):
        return HarnessObstruction(
            "MalformedCommand",
            "command/appendage",
            "expected: appendage add <id> anchor <part>:<t> intent <id,id> ...",
        )
    appendage_id = arguments[1]
    try:
        anchor_part, anchor_station = arguments[3].split(":", maxsplit=1)
        parsed = _parse_tail(
            arguments[6:],
            {
                "id": appendage_id,
                "anchor_part": anchor_part,
                "anchor_t": float(anchor_station),
                "intent_refs": arguments[5].split(","),
            },
            {
                "count": int,
                "seed": int,
                "step": float,
                "influence": float,
                "kill": float,
                "envelope": float,
                "env_min_radius": float,
                "root_radius": float,
                "min_radius": float,
                "tip_taper": float,
                "offset": _vector,
            },
        )
    except (TypeError, ValueError) as exception:
        return HarnessObstruction(
            "MalformedAppendage",
            f"/appendages/{appendage_id}",
            str(exception),
        )
    if isinstance(parsed, HarnessObstruction):
        return parsed
    declaration = (
        {
            **{key: value for key, value in parsed.items() if key != "offset"},
            "root_offset": parsed["offset"],
        }
        if "offset" in parsed
        else parsed
    )
    return AddAppendage(
        appendage_id, anchor_part, anchor_station, declaration
    )


def _parse_set(arguments: tuple[str, ...]) -> ParsedCommand:
    if len(arguments) != 2 or arguments[0].count(".") != 2:
        return HarnessObstruction(
            "MalformedCommand",
            "command/set",
            "expected: set <conduits|appendages>.<id>.<key> <json-value>",
        )
    declaration_kind, declaration_id, key = arguments[0].split(".")
    try:
        value = json.loads(arguments[1])
    except json.JSONDecodeError:
        value = arguments[1]
    return SetDeclaration(
        declaration_kind, declaration_id, key, value
    )


def _parse_remove(arguments: tuple[str, ...]) -> ParsedCommand:
    if len(arguments) != 2:
        return HarnessObstruction(
            "MalformedCommand",
            "command/remove",
            "expected: remove <conduits|appendages|intents> <id>",
        )
    authored_kind, declaration_id = arguments
    return RemoveDeclaration(
        authored_kind,
        "_intents" if authored_kind == "intents" else authored_kind,
        declaration_id,
    )


def _parse_vascular(arguments: tuple[str, ...]) -> ParsedCommand:
    if not arguments or arguments[0] not in ("solve", "receipt", "render"):
        return HarnessObstruction(
            "MalformedCommand",
            "command/vascular",
            "expected: vascular <solve|receipt|render>",
        )
    action = arguments[0]
    if action == "solve":
        return SolveVascular()
    if action == "receipt":
        return ShowVascularReceipt()
    return (
        RenderVascular(arguments[1])
        if len(arguments) == 2
        else HarnessObstruction(
            "MalformedCommand",
            "command/vascular/render",
            "expected: vascular render <out-prefix>",
        )
    )


def parse_command(line: str) -> ParsedCommand:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return NoCommand()
    command_name, *argument_list = stripped.split()
    arguments = tuple(argument_list)
    match command_name:
        case "parts":
            return ListParts()
        case "intent":
            return _parse_intent(arguments)
        case "conduit":
            return _parse_conduit(arguments)
        case "appendage":
            return _parse_appendage(arguments)
        case "set":
            return _parse_set(arguments)
        case "remove":
            return _parse_remove(arguments)
        case "grow":
            return Grow()
        case "vascular":
            return _parse_vascular(arguments)
        case "render" if len(arguments) == 1:
            return RenderLast(arguments[0])
        case "render":
            return HarnessObstruction(
                "MalformedCommand",
                "command/render",
                "expected: render <out-prefix>",
            )
        case "save" if len(arguments) == 1:
            return Save(Path(arguments[0]))
        case "save":
            return HarnessObstruction(
                "MalformedCommand", "command/save", "expected: save <path>"
            )
        case "show":
            return Show()
        case "help":
            return Help()
        case "quit":
            return Quit()
        case unknown_command:
            return HarnessObstruction(
                "UnknownCommand",
                f"command/{unknown_command}",
                f"unknown command {unknown_command!r}; try 'help'",
            )


__all__ = ["ParsedCommand", "parse_command"]
