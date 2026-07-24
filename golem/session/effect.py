"""Filesystem, compiler, mesh, render, and frame effects for sessions."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from golem.contracts.views import verdict_record
from golem.kernel import body as _body
from golem.kernel.engine.types import (
    RejectedSurfaceFormation,
    SurfaceFormationObstruction,
)
from golem.senses import proprio as _proprio
from golem.senses.proprio.format import f3
from golem.session.fault import BODY_COMPILE_SEAM, project_engine_fault
from golem.session.state import (
    AuthoredState,
    BodyCompileObstruction,
    CompileObstructed,
    CompileOutcome,
    Compiled,
    LoadObstructed,
    LoadObstruction,
    LoadResult,
    SessionState,
)


@dataclass(frozen=True)
class EffectObstruction:
    address: str
    reason: str
    surface_formation: SurfaceFormationObstruction | None = None


@dataclass(frozen=True)
class EffectObstructed:
    obstructions: tuple[EffectObstruction, ...]


@dataclass(frozen=True)
class FramesReady:
    frames: dict


@dataclass(frozen=True)
class EffectText:
    text: str


@dataclass(frozen=True)
class EffectDone:
    pass


type FramesResult = FramesReady | EffectObstructed
type TextResult = EffectText | EffectObstructed
type DoneResult = EffectDone | EffectObstructed


def _surface_formation_effect(
    address: str,
    rejected: RejectedSurfaceFormation,
) -> EffectObstructed:
    return EffectObstructed(
        tuple(
            EffectObstruction(
                address,
                type(obstruction).__name__,
                obstruction,
            )
            for obstruction in rejected.obstructions
        )
    )


def compile_authored(authored: AuthoredState) -> CompileOutcome:
    try:
        compiler = _body.Compiler(copy.deepcopy(authored.spec), spec_dir=authored.spec_dir)
        result = compiler.compile()
        match result:
            case _body.CompiledBody(graph, receipt, anatomy):
                senses, verdicts = _body.run_typed_asserts(graph, graph["intent"])
                anomalies = _proprio.detect_anomalies(senses)
                return Compiled(
                    graph=graph,
                    receipt=receipt,
                    anatomy=anatomy,
                    senses=senses,
                    verdicts=verdicts,
                    records=tuple(map(verdict_record, verdicts)),
                    anomalies=tuple(anomalies),
                )
            case _body.RejectedBody(obstructions):
                return CompileObstructed(
                    (BodyCompileObstruction(obstructions),)
                )
            case _ as unreachable:
                assert_never(unreachable)
    except Exception as exc:
        return CompileObstructed(
            (project_engine_fault(BODY_COMPILE_SEAM, exc),)
        )


def load_state(path: str | Path) -> LoadResult:
    source = Path(path)
    try:
        spec = json.loads(source.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return LoadObstructed((LoadObstruction(source, f"{type(exc).__name__}: {exc}"),))
    authored = AuthoredState(spec, source.parent)
    return SessionState.from_carriers(authored, compile_authored(authored))


def save_spec(state: SessionState, path: str | Path) -> DoneResult:
    target = Path(path)
    try:
        target.write_text(state.canonical() + "\n")
    except OSError as exc:
        return EffectObstructed((EffectObstruction(str(target), str(exc)),))
    return EffectDone()


def save_journal(entries: tuple[dict, ...], path: str | Path) -> DoneResult:
    target = Path(path)
    lines = tuple(json.dumps(entry, sort_keys=True) for entry in entries)
    try:
        target.write_text("\n".join(lines) + ("\n" if lines else ""))
    except OSError as exc:
        return EffectObstructed((EffectObstruction(str(target), str(exc)),))
    return EffectDone()


def bone_frames(state: SessionState) -> FramesResult:
    try:
        compiler = _body.Compiler(copy.deepcopy(state.spec), spec_dir=state.spec_dir)
        compiler.validate()
        compiler.forward_kinematics()
        return FramesReady(compiler.bones)
    except Exception as exc:
        return EffectObstructed(
            (EffectObstruction("skeleton", f"{type(exc).__name__}: {exc}"),)
        )


def mesh_summary(state: SessionState, resolution: int) -> TextResult:
    if isinstance(state.outcome, CompileObstructed):
        return EffectObstructed(
            (EffectObstruction("meta", f"cannot mesh: {state.outcome.error}"),)
        )
    from golem.kernel import engine as ev

    try:
        evaluated = ev.evaluate(state.outcome.graph, res=resolution)
        if isinstance(evaluated, ev.RejectedSurfaceFormation):
            return _surface_formation_effect("mesh", evaluated)
        report = ev.coherence_report(evaluated.vertices, evaluated.faces)
    except Exception as exc:
        return EffectObstructed(
            (EffectObstruction("mesh", f"{type(exc).__name__}: {exc}"),)
        )
    return EffectText(
        f"mesh res {resolution}: {report['components']} component(s), "
        f"watertight {report['watertight_main']}, faces {report['faces']}, "
        f"dust {report['grid_dust_slivers']}, "
        f"volume_share {f3(report['largest_component_volume_share'])}"
    )


def render_views(state: SessionState, path: str, resolution: int) -> TextResult:
    if isinstance(state.outcome, CompileObstructed):
        return EffectObstructed(
            (EffectObstruction("meta", f"cannot render: {state.outcome.error}"),)
        )
    from golem.kernel import engine as ev
    import render as _render

    try:
        evaluated = ev.evaluate(state.outcome.graph, res=resolution)
        if isinstance(evaluated, ev.RejectedSurfaceFormation):
            return _surface_formation_effect(path, evaluated)
        azimuths = (0, 90, 180, 270)
        views = tuple(
            _render.render_view(evaluated.vertices, evaluated.faces, azimuth)
            for azimuth in azimuths
        )
        labels = tuple(f"az {azimuth}" for azimuth in azimuths)
        _render.contact_sheet(views, labels, path)
    except Exception as exc:
        return EffectObstructed(
            (EffectObstruction(path, f"{type(exc).__name__}: {exc}"),)
        )
    return EffectText(
        f"views ({', '.join(labels)}) res {resolution} -> {path}"
    )
