"""Filesystem and console effects for the geometry engine."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import numpy as np

from .compile import coherence_report, evaluate
from .types import RejectedSurfaceFormation
from .validation import vocab_violations


_ROOT = Path(__file__).resolve().parents[3]
_DELEGATION_CORPUS = (
    ("golem_v4", _ROOT / "pilots" / "golem_v4.json"),
    ("golem_hands", _ROOT / "pilots" / "golem_hands.json"),
    ("golem_knight", _ROOT / "rehearsal" / "knight" / "golem_knight.json"),
)
_DELEGATION_RES = 130
_USAGE = """\
GOLEM geometry engine

python -m golem.kernel.engine --check-delegation [--res N]
python -m golem.kernel.engine <graph.json> [--res N]
"""


def _load_frozen_oracle() -> ModuleType:
    oracle_path = _ROOT / "pilots" / "engine.py"
    specification = importlib.util.spec_from_file_location(
        "_golem_frozen_engine_oracle", oracle_path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load frozen geometry oracle at {oracle_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def check_delegation(res: int = _DELEGATION_RES) -> bool:
    oracle = _load_frozen_oracle()

    def check_case(case: tuple[str, Path]) -> bool:
        name, path = case
        graph = json.loads(path.read_text())
        _, _, _, reference_field, _, _ = oracle.evaluate(graph, res=res)
        evaluated = evaluate(graph, res=res)
        exact = bool(np.array_equal(reference_field, evaluated.field))
        print(f"{name} bit-exact: {exact}")
        return exact

    return all(tuple(map(check_case, _DELEGATION_CORPUS)))


def _resolution_argument(arguments: tuple[str, ...], default: int) -> tuple[int, tuple[str, ...]]:
    if "--res" not in arguments:
        return default, arguments
    index = arguments.index("--res")
    return int(arguments[index + 1]), arguments[:index] + arguments[index + 2 :]


def main(argv: list[str]) -> int:
    arguments = tuple(argv)
    if not arguments or arguments[0] in ("-h", "--help"):
        print(_USAGE)
        return 0
    resolution, positional = _resolution_argument(
        arguments,
        _DELEGATION_RES if arguments[0] == "--check-delegation" else 170,
    )
    if arguments[0] == "--check-delegation":
        return 0 if check_delegation(resolution) else 1
    graph = json.loads(Path(positional[0]).read_text())
    evaluated = evaluate(graph, res=resolution)
    if isinstance(evaluated, RejectedSurfaceFormation):
        print(
            json.dumps(
                tuple(
                    {
                        "obstruction": type(obstruction).__name__,
                        **asdict(obstruction),
                    }
                    for obstruction in evaluated.obstructions
                ),
                indent=2,
            )
        )
        return 1
    print(json.dumps(coherence_report(evaluated.vertices, evaluated.faces), indent=2))
    violations = vocab_violations(graph, res=resolution)
    print(f"vocab violations: {len(violations)}")
    tuple(
        map(
            lambda violation: print(
                f"  [{violation['rule']}] {violation['part']} "
                f"({violation['kind']}): {violation['detail']}"
            ),
            violations,
        )
    )
    return 0
