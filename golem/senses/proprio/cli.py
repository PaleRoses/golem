"""CLI Harness for deterministic proprioceptive receipts and pull views."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import assert_never

from golem.contracts import asserts as _asserts
from golem.senses.model import RejectedSenses, Senses
from golem.senses.project import rejected_senses_text
from golem.senses.proprio.anomaly import detect_anomalies
from golem.senses.proprio.build import build_senses
from golem.senses.proprio.render import (
    SketchSurfaceFormationObstruction,
    render_receipt,
    render_section,
    render_sketch,
)


def run(graph: dict, intent: dict | None):
    senses_result = build_senses(graph, intent)
    match senses_result:
        case RejectedSenses() as rejected:
            return rejected
        case Senses() as senses:
            pass
        case _ as unreachable:
            assert_never(unreachable)
    clauses, _pack = _assert_configuration(intent)
    assert_results = _asserts.evaluate_pack(clauses, senses)
    return senses, assert_results, detect_anomalies(senses)


def _assert_configuration(
    intent: dict | None,
) -> tuple[tuple[dict, ...], str]:
    assertions = ((intent or {}).get("asserts") or {})
    return tuple(assertions.get("clauses", ())), str(
        assertions.get("pack", "knight.intent")
    )


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("--intent", type=Path)
    parser.add_argument("--section", action="append", default=[])
    parser.add_argument("--sketch", choices=("front", "side"))
    return parser


def _surface_formation_text(
    obstruction: SketchSurfaceFormationObstruction,
) -> str:
    return json.dumps(
        {
            "obstructions": tuple(
                {
                    "obstruction": type(member).__name__,
                    **asdict(member),
                }
                for member in obstruction.obstructions
            ),
            "status": "surface_formation_obstructed",
            "view": obstruction.view.value,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def main(argv: list[str]) -> int:
    args = _argument_parser().parse_args(argv)
    graph = _load_json(args.graph)
    intent = (
        _load_json(args.intent)
        if args.intent is not None
        else graph.get("intent")
    )
    run_result = run(graph, intent)
    match run_result:
        case RejectedSenses() as rejected:
            sys.stderr.write(rejected_senses_text(rejected))
            return 2
        case (senses, assert_results, anomalies):
            pass
        case _ as unreachable:
            assert_never(unreachable)
    if args.section or args.sketch:
        sketch = render_sketch(senses, args.sketch) if args.sketch else None
        if isinstance(sketch, SketchSurfaceFormationObstruction):
            sys.stderr.write(_surface_formation_text(sketch) + "\n")
            return 2
        chunks = (
            *tuple(render_section(senses, section) for section in args.section),
            *((sketch,) if isinstance(sketch, str) else ()),
        )
        sys.stdout.write("\n".join(chunks))
        return 0
    _clauses, pack = _assert_configuration(intent)
    sys.stdout.write(
        render_receipt(
            senses,
            assert_results,
            anomalies,
            pack=pack,
        )
    )
    return 0
