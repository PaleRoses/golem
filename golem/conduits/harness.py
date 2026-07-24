"""Typed conduit operator harness with a pure command step and thin effects.

Surface trim and appendage authoring operate on element graphs. Vascular solve
accepts only body documents carrying anatomy intent and consumes the accepted
anatomy result. Resolution remains pinned so consecutive receipts describe the
same tessellation. Every expected refusal is an addressed typed obstruction.
"""

from __future__ import annotations

from dataclasses import replace
from functools import reduce
import json
from pathlib import Path
import sys

from golem.assembly.obstructions import (
    ElementSourceObstruction,
    UnknownAppendageIntentObstruction,
)
from golem.conduits.operator.algebra import step
from golem.conduits.operator.effects import Effect, Harness
from golem.conduits.operator.parse import parse_command
from golem.conduits.operator.types import (
    AcceptedCommand,
    Command,
    CommandObstruction,
    CommandResult,
    DEFAULT_RESOLUTION,
    HarnessObstruction,
    HarnessState,
    RejectedCommand,
)
from golem.conduits.surface.types import EmissionPitchObstruction


_HARNESS = Harness()


def _append_journal(state: HarnessState, line: str) -> HarnessState:
    return replace(state, journal=(*state.journal, line))


def run_line(state: HarnessState, line: str) -> CommandResult:
    stripped = line.strip()
    parsed = parse_command(stripped)
    result = (
        RejectedCommand(state, parsed)
        if isinstance(parsed, HarnessObstruction)
        else _HARNESS.execute(state, parsed)
    )
    return (
        replace(result, state=_append_journal(result.state, stripped))
        if isinstance(result, AcceptedCommand) and result.output is not None
        else result
    )


def _obstruction_fields(
    obstruction: CommandObstruction,
) -> tuple[str, str, str]:
    match obstruction:
        case HarnessObstruction(kind, address, reason):
            return kind, address, reason
        case EmissionPitchObstruction() as addressed:
            return addressed.kind, addressed.address, addressed.reason
        case ElementSourceObstruction(element_id, reason):
            return (
                "ElementSourceObstruction",
                f"/elements/{element_id}/source",
                reason,
            )
        case UnknownAppendageIntentObstruction(
            element_id, appendage_id, intent_id
        ):
            return (
                "UnknownAppendageIntentObstruction",
                f"/elements/{element_id}/appendages/{appendage_id}/intent_refs",
                f"unknown intent {intent_id!r}",
            )
        case other:
            return type(other).__name__, "/assembly", repr(other)


def _format_result(line: str, result: CommandResult) -> str | None:
    if isinstance(result, RejectedCommand):
        kind, address, reason = _obstruction_fields(result.obstruction)
        return (
            f"> {line}\nREJECTED [{kind}] "
            f"{address}: {reason}"
        )
    return (
        None
        if result.output is None
        else f"> {line}\n{result.output}"
    )


def _run_script(state: HarnessState, lines: tuple[str, ...]) -> HarnessState:
    def apply_line(current: HarnessState, line: str) -> HarnessState:
        result = run_line(current, line)
        rendered = _format_result(line, result)
        if rendered is not None:
            print(rendered)
        return result.state

    return reduce(apply_line, lines, state)


def _interactive(state: HarnessState) -> int:
    try:
        line = input("golem-conduit> ")
    except EOFError:
        return 0
    result = run_line(state, line)
    rendered = _format_result(line, result)
    if rendered is not None:
        print(rendered.removeprefix(f"> {line}\n"))
    return (
        0
        if isinstance(result, AcceptedCommand) and result.quit
        else _interactive(result.state)
    )


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    path = Path(argv[0])
    state = HarnessState(
        spec=json.loads(path.read_text()),
        spec_dir=path.resolve().parent,
        resolution=(
            int(argv[argv.index("--res") + 1])
            if "--res" in argv
            else DEFAULT_RESOLUTION
        ),
    )
    if "--script" in argv:
        script_path = Path(argv[argv.index("--script") + 1])
        _run_script(state, tuple(script_path.read_text().splitlines()))
        return 0
    return _interactive(state)


__all__ = [
    "AcceptedCommand",
    "Command",
    "CommandResult",
    "Effect",
    "Harness",
    "HarnessObstruction",
    "HarnessState",
    "RejectedCommand",
    "main",
    "parse_command",
    "run_line",
    "step",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
