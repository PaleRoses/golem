"""Frozen authored-state and compile-outcome carriers for sessions."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from golem import paths as _paths
from golem.kernel.body.canonical import _canonical_json
from golem.kernel.body.project import rejected_body_text
from golem.kernel.body.types import BodyObstruction, RejectedBody
from golem.session.fault import EngineFault

if TYPE_CHECKING:
    from golem.contracts.model import Verdict
    from golem.kernel.anatomy import AnatomyResult
    from golem.senses.proprio.anomaly import Anomaly


@dataclass(frozen=True)
class AuthoredState:
    spec: dict
    spec_dir: Path
    txn: int = 0

    def with_spec(self, spec: dict, txn: int) -> "AuthoredState":
        return AuthoredState(spec, self.spec_dir, txn)

    def fork(self) -> "AuthoredState":
        return replace(self, spec=copy.deepcopy(self.spec))

    def canonical(self) -> str:
        return _canonical_json(self.spec)


@dataclass(frozen=True)
class BodyCompileObstruction:
    obstructions: tuple[BodyObstruction, ...]

    def render(self) -> str:
        return rejected_body_text(RejectedBody(self.obstructions))


type CompileObstruction = BodyCompileObstruction | EngineFault


@dataclass(frozen=True)
class Compiled:
    graph: dict
    receipt: dict
    anatomy: "AnatomyResult | None"
    senses: object
    verdicts: tuple["Verdict", ...]
    records: tuple[Mapping[str, object], ...]
    anomalies: tuple["Anomaly", ...]

    @property
    def error(self) -> None:
        return None


@dataclass(frozen=True)
class CompileObstructed:
    obstructions: tuple[CompileObstruction, ...]

    @property
    def graph(self) -> None:
        return None

    @property
    def receipt(self) -> None:
        return None

    @property
    def senses(self) -> None:
        return None

    @property
    def anatomy(self) -> None:
        return None

    @property
    def verdicts(self) -> tuple[()]:
        return ()

    @property
    def records(self) -> tuple[()]:
        return ()

    @property
    def anomalies(self) -> tuple[()]:
        return ()

    @property
    def error(self) -> str:
        return "; ".join(obstruction.render() for obstruction in self.obstructions)


type CompileOutcome = Compiled | CompileObstructed
type CompileCache = CompileOutcome


@dataclass(frozen=True)
class LoadObstruction:
    path: Path
    reason: str


@dataclass(frozen=True)
class LoadObstructed:
    obstructions: tuple[LoadObstruction, ...]


def empty_spec(name: str) -> dict:
    return {
        "name": name,
        "dialect": "body/0.3",
        "emit_target": "v03",
        "blend": 0.006,
        "ground_y": 0.02,
        "ground_tol": 0.02,
        "skeleton": {
            "root": {
                "id": "root",
                "world": [0.0, 1.0, 0.0],
                "flesh": [
                    {
                        "kind": "blob",
                        "name": "seed",
                        "t": 0.0,
                        "size": [0.05, 0.05, 0.05],
                    }
                ],
            },
            "bones": [],
        },
        "pose": {"joints": {}, "goals": []},
        "props": [],
        "mounts": [],
        "contacts": [],
    }


class SessionState:
    __slots__ = ("_authored", "_outcome")

    def __init__(
        self,
        spec: dict,
        spec_dir: Path,
        outcome: CompileOutcome | None = None,
        txn: int = 0,
    ):
        from golem.session.effect import compile_authored

        authored = AuthoredState(spec, Path(spec_dir), txn)
        self._authored = authored
        self._outcome = outcome if outcome is not None else compile_authored(authored)

    @classmethod
    def from_carriers(cls, authored: AuthoredState, outcome: CompileOutcome) -> "SessionState":
        state = cls.__new__(cls)
        state._authored = authored
        state._outcome = outcome
        return state

    @classmethod
    def open(cls, path: str | Path) -> "SessionState":
        from golem.session.effect import load_state

        result = load_state(path)
        if isinstance(result, LoadObstructed):
            obstruction = result.obstructions[0]
            raise OSError(obstruction.reason)
        return result

    @classmethod
    def new(cls, name: str) -> "SessionState":
        return cls(empty_spec(name), _paths.SPECS)

    @property
    def authored(self) -> AuthoredState:
        return self._authored

    @property
    def outcome(self) -> CompileOutcome:
        return self._outcome

    @property
    def spec(self) -> dict:
        return self._authored.spec

    @property
    def spec_dir(self) -> Path:
        return self._authored.spec_dir

    @property
    def cache(self) -> CompileOutcome:
        return self._outcome

    @property
    def txn(self) -> int:
        return self._authored.txn

    @txn.setter
    def txn(self, txn: int) -> None:
        self._authored = replace(self._authored, txn=txn)

    def transitioned(
        self,
        spec: dict,
        txn: int,
        outcome: CompileOutcome,
    ) -> "SessionState":
        return SessionState.from_carriers(self._authored.with_spec(spec, txn), outcome)

    def recompile(self) -> CompileOutcome:
        from golem.session.effect import compile_authored

        self._outcome = compile_authored(self._authored)
        return self._outcome

    def canonical(self) -> str:
        return self._authored.canonical()

    def fork(self) -> "SessionState":
        return SessionState.from_carriers(self._authored.fork(), self._outcome)


type LoadResult = SessionState | LoadObstructed
