"""Thin filesystem and console interpreter for contract evaluation."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import assert_never

from golem.contracts.api import evaluate_pack
from golem.senses.model import RejectedSenses, Senses
from golem.senses.project import rejected_senses_text

_HELP = "usage: asserts.py <graph.json> --intent <intent.json> [--json]"


@dataclass(frozen=True)
class CliArguments:
    graph_path: str | None = None
    intent_path: str | None = None
    as_json: bool = False


@dataclass(frozen=True)
class RejectedCliArguments:
    reason: str


type CliArgumentsResult = CliArguments | RejectedCliArguments


def parse_cli_arguments(
    argv: tuple[str, ...], arguments: CliArguments = CliArguments()
) -> CliArgumentsResult:
    match argv:
        case ():
            return arguments
        case ("--json", *rest):
            return parse_cli_arguments(tuple(rest), replace(arguments, as_json=True))
        case ("--intent", intent_path, *rest):
            return parse_cli_arguments(
                tuple(rest),
                replace(arguments, intent_path=intent_path),
            )
        case ("--intent",):
            return RejectedCliArguments("--intent requires a path")
        case (flag, *_) if flag.startswith("-"):
            return RejectedCliArguments(f"unknown option {flag}")
        case (graph_path, *rest) if arguments.graph_path is None:
            return parse_cli_arguments(
                tuple(rest),
                replace(arguments, graph_path=graph_path),
            )
        case (graph_path, *_):
            return RejectedCliArguments(f"unexpected graph path {graph_path}")


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(_HELP)
        return 0
    match parse_cli_arguments(tuple(argv)):
        case RejectedCliArguments(reason):
            print(f"{_HELP}\n{reason}", file=sys.stderr)
            return 2
        case CliArguments(None, _, _):
            print(_HELP, file=sys.stderr)
            return 2
        case CliArguments(_, None, _):
            print(_HELP, file=sys.stderr)
            return 2
        case CliArguments(graph_path, intent_path, as_json):
            from golem.senses import proprio

            graph = json.loads(Path(graph_path).read_text())
            intent = json.loads(Path(intent_path).read_text())
            senses_result = proprio.build_senses(graph, intent)
            match senses_result:
                case RejectedSenses() as rejected:
                    sys.stderr.write(rejected_senses_text(rejected))
                    return 2
                case Senses() as senses:
                    pass
                case _ as unreachable:
                    assert_never(unreachable)
            clauses = (intent.get("asserts") or {}).get("clauses", [])
            records = evaluate_pack(clauses, senses)
            if as_json:
                print(json.dumps(records, indent=2, default=float))
                return 0
            failures = tuple(
                record for record in records if record["status"] == "fail"
            )
            passes = tuple(
                record for record in records if record["status"] == "pass"
            )
            unmeasurable = tuple(
                record
                for record in records
                if record["status"] == "unmeasurable"
            )
            summary = (
                f"ASSERT {len(failures)} FAIL / {len(passes)} pass / "
                f"{len(unmeasurable)} unmeasurable"
            )
            lines = tuple(
                f"  {record['human']}"
                for record in (*failures, *unmeasurable)
            )
            print("\n".join((summary, *lines)))
            return 0
        case _ as unreachable:
            assert_never(unreachable)
