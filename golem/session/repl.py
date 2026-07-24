"""Thin interactive and scripted IO harness for the closed command grammar."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path

from golem.session import algebra, commands, core, ops


_QUIT = object()
HELP = commands.help_text()


def execute(session: core.Session, line: str):
    parsed = commands.parse_command(line)
    match parsed:
        case commands.EmptyCommand():
            return None
        case commands.CommandObstructed(output):
            return output
        case _:
            try:
                interpreted = algebra.interpret_command(session, parsed)
            except ops.Reject as rejected:
                return f"REJECT @ {rejected.addr}: {rejected.why}"
            return _QUIT if isinstance(interpreted, algebra.ExitRequested) else interpreted


@dataclass(frozen=True)
class _ScriptRun:
    output: tuple[str, ...]
    active: bool


def _script_step(session: core.Session, run: _ScriptRun, line: str) -> _ScriptRun:
    if not run.active:
        return run
    result = execute(session, line)
    rendered = (
        f"golem> {line}\n"
        if result is None or result is _QUIT
        else f"golem> {line}\n{result}\n"
    )
    return _ScriptRun((*run.output, rendered), result is not _QUIT)


def _run_script(session: core.Session, script_path: str | Path) -> None:
    lines = tuple(
        line
        for raw in Path(script_path).read_text().splitlines()
        if (line := raw.strip()) and not line.startswith("#")
    )
    run = reduce(partial(_script_step, session), lines, _ScriptRun((), True))
    sys.stdout.write("".join(run.output))


def _interactive_step(session: core.Session) -> None:
    try:
        line = input("golem> ")
    except EOFError:
        print()
        return
    result = execute(session, line)
    if result is _QUIT:
        return
    if result is not None:
        print(result)
    _interactive_step(session)


def _run_interactive(session: core.Session) -> None:
    print(f"golem session: {session.state.spec.get('name', '(unnamed)')}")
    print('type "help" for the command reference')
    _interactive_step(session)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m golem.session.repl",
        description="Interactive shell and scripted runner for a GOLEM session.",
    )
    parser.add_argument("spec", nargs="?", help="SPEC.json to open")
    parser.add_argument("--new", metavar="NAME", help="start a new empty session named NAME")
    parser.add_argument("--script", metavar="FILE", help="run commands from FILE, then exit")
    parser.add_argument("--journal", metavar="PATH", help="write journal JSONL on exit")
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.new and args.spec:
        parser.error("give either SPEC.json or --new NAME, not both")
    if not args.new and not args.spec:
        parser.error("give a SPEC.json path or --new NAME")
    try:
        session = core.Session.new(args.new) if args.new else core.Session.open(args.spec)
    except ops.Reject as rejected:
        print(f"REJECT @ {rejected.addr}: {rejected.why}", file=sys.stderr)
        return 1
    if args.script:
        _run_script(session, args.script)
    else:
        _run_interactive(session)
    if args.journal:
        session.save_journal(args.journal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
