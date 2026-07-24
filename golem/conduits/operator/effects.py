"""Thin compiler, filesystem, renderer, and terminal effect interpreter."""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib.util
from itertools import chain
import json
from pathlib import Path
from typing import Protocol, assert_never

from golem.assembly import (
    AssemblyStratum,
    PinnedResolution,
    RejectedAssembly,
    compile_assembly,
    load_element_graph,
)
from golem.assembly.carriers import AssemblyRecord
from golem.assembly.obstructions import ElementSourceObstruction
from golem.conduits.operator.algebra import step
from golem.conduits.operator.types import (
    AcceptedCommand,
    Command,
    CommandResult,
    EffectRequest,
    HarnessObstruction,
    HarnessState,
    LoadHelpEffect,
    RejectedCommand,
    RenderRecordsEffect,
    RequestedEffect,
    ResolvePartsEffect,
    SaveSessionEffect,
    SolveVascularEffect,
    CompileSurfaceEffect,
)
from golem.kernel.anatomy import AcceptedVasculature
from golem.kernel.body import DIALECT as BODY_DIALECT
from golem.paths import KERNEL_ROOT


def _reject(
    state: HarnessState, kind: str, address: str, reason: str
) -> RejectedCommand:
    return RejectedCommand(state, HarnessObstruction(kind, address, reason))


def _reject_assembly(
    state: HarnessState, result: RejectedAssembly
) -> RejectedCommand:
    obstruction = next(iter(result.obstructions), None)
    return (
        RejectedCommand(state, obstruction)
        if obstruction is not None
        else _reject(
            state,
            "MalformedAssemblyRejection",
            "/assembly",
            "assembly rejected without an obstruction",
        )
    )


def _solid_records(
    records: tuple[AssemblyRecord, ...],
) -> tuple[AssemblyRecord, ...]:
    return tuple(
        record
        for record in records
        if record.stratum is AssemblyStratum.SOLID
    )


def _vocabulary_advisory_lines(
    solid_records: tuple[AssemblyRecord, ...],
) -> tuple[str, ...]:
    violations = tuple(
        chain.from_iterable(record.violations for record in solid_records)
    )
    return (
        f"  advisories vocabulary_violations={len(violations)}",
        *tuple(
            f"  advisory  part={violation['part']} "
            f"rule={violation['rule']} detail={violation['detail']}"
            + (
                f" -> {violation['repair']}"
                if violation.get("repair") is not None
                else ""
            )
            for violation in violations
        ),
    )


class Effect(Protocol):
    def perform(
        self, state: HarnessState, request: EffectRequest
    ) -> CommandResult: ...


@dataclass(frozen=True)
class Harness:
    help_path: Path = Path(__file__).resolve().parents[1] / "HELP.txt"

    def execute(self, state: HarnessState, command: Command) -> CommandResult:
        transition = step(state, command)
        if not isinstance(transition, RequestedEffect):
            return transition
        try:
            return self.perform(transition.state, transition.request)
        except Exception as exception:
            return _reject(
                transition.state,
                "InternalHarnessFailure",
                "/harness",
                f"{type(exception).__name__}: {exception}",
            )

    def perform(
        self, state: HarnessState, request: EffectRequest
    ) -> CommandResult:
        match request:
            case ResolvePartsEffect():
                element_id = str(state.spec.get("name", "element"))
                graph = load_element_graph(
                    {
                        "id": element_id,
                        "role": "creature",
                        **(
                            {"body": state.spec}
                            if state.spec.get("dialect") == BODY_DIALECT
                            else {"graph": state.spec}
                        ),
                    },
                    state.spec_dir,
                )
                if isinstance(graph, ElementSourceObstruction):
                    return _reject_assembly(
                        state, RejectedAssembly((graph,))
                    )
                return AcceptedCommand(
                    state,
                    "\n".join(
                        f"  {part['id']:20s} {part['type']:8s} "
                        + (
                            f"spine={part['spine'][0]}->{part['spine'][-1]}"
                            if part["type"] == "gencyl"
                            else f"center={part['center']}"
                        )
                        + (" mirror" if part.get("mirror") else "")
                        for part in graph["parts"]
                    ),
                )
            case CompileSurfaceEffect():
                element_id = str(state.spec.get("name", "element"))
                graph = load_element_graph(
                    {
                        "id": element_id,
                        "role": "creature",
                        **(
                            {"body": state.spec}
                            if state.spec.get("dialect") == BODY_DIALECT
                            else {"graph": state.spec}
                        ),
                    },
                    state.spec_dir,
                )
                if isinstance(graph, ElementSourceObstruction):
                    return _reject_assembly(
                        state, RejectedAssembly((graph,))
                    )
                compiled = compile_assembly(
                    {
                        "name": element_id,
                        "elements": [
                            {
                                "id": element_id,
                                "role": "creature",
                                "graph": graph,
                                "appearance_material": state.spec.get(
                                    "appearance_material", "obsidian_warden"
                                ),
                            }
                        ],
                    },
                    state.spec_dir,
                    resolution_policy=PinnedResolution(state.resolution),
                )
                if isinstance(compiled, RejectedAssembly):
                    return _reject_assembly(state, compiled)
                solid_records = _solid_records(compiled.records)
                receipt = "\n".join(
                    (
                        f"RECEIPT (pinned res {state.resolution})",
                        *tuple(
                            "  element   "
                            f"components={record.report['components']} "
                            f"watertight={record.report['watertight_main']} "
                            f"dust={record.report['grid_dust_slivers']} "
                            f"faces={record.report['faces']}"
                            for record in solid_records
                        ),
                        "  verdict   OK",
                        *tuple(
                            "  surface   groove-displaced vertices="
                            f"{record.report['surface_conduit_displaced_vertices']}"
                            for record in solid_records
                        ),
                        *tuple(
                            (
                                f"  conduit  {record.record_id}: EMPTY"
                                if record.empty
                                else f"  conduit  {record.record_id}: faces={record.report['faces']}"
                            )
                            for record in compiled.records
                            if record.stratum is AssemblyStratum.CONDUIT
                        ),
                        *_vocabulary_advisory_lines(solid_records),
                    )
                )
                return AcceptedCommand(
                    replace(state, last_records=compiled.records), receipt
                )
            case SolveVascularEffect():
                if state.spec.get("dialect") != BODY_DIALECT:
                    return _reject(
                        state,
                        "MissingAnatomyIntent",
                        "/anatomy",
                        "vascular commands require a body/0.3 document with anatomy/0.1 intent",
                    )
                if "anatomy" not in state.spec:
                    return _reject(
                        state,
                        "MissingAnatomyIntent",
                        "/anatomy",
                        "body has no anatomy/0.1 overall",
                    )
                element_id = str(state.spec.get("name", "body"))
                compiled = compile_assembly(
                    {
                        "name": element_id,
                        "elements": [
                            {
                                "id": element_id,
                                "role": "creature",
                                "body": state.spec,
                                "appearance_material": state.spec.get(
                                    "appearance_material", "gambeson_dark"
                                ),
                            }
                        ],
                    },
                    state.spec_dir,
                    resolution_policy=PinnedResolution(state.resolution),
                )
                if isinstance(compiled, RejectedAssembly):
                    return _reject_assembly(state, compiled)
                result = next(iter(compiled.vasculature), None)
                if not isinstance(result, AcceptedVasculature):
                    return _reject(
                        state,
                        "MissingAcceptedVasculature",
                        "/vascular/result",
                        "accepted body assembly produced no accepted vasculature",
                    )
                receipt = result.receipt
                output = "\n".join(
                    (
                        f"VASCULAR ACCEPTED (pinned res {state.resolution})",
                        (
                            f"  topology nodes={len(result.graph.nodes)} "
                            f"edges={len(result.graph.edges)} "
                            f"terminal_pairs={receipt.terminal_pair_budget}"
                        ),
                        f"  allocation {dict(receipt.terminal_pairs_by_region)}",
                        (
                            f"  flow residual={receipt.maximum_free_cell_residual:.3e} "
                            f"balance={receipt.boundary_balance_error:.3e} "
                            f"delivery_error={receipt.maximum_delivery_relative_error:.3e}"
                        ),
                        *_vocabulary_advisory_lines(
                            _solid_records(compiled.records)
                        ),
                    )
                )
                next_state = replace(
                    state,
                    last_records=compiled.records,
                    vascular_records=compiled.records,
                    vascular_result=result,
                )
                return AcceptedCommand(next_state, output)
            case RenderRecordsEffect(records, prefix, address):
                try:
                    from golem.assembly.carriers import (
                        AcceptedAssembly,
                        VisualAssemblyReceipt,
                    )
                    from golem.assembly.exchange import export_scene
                    from golem.assembly.service import VisualOnlyServiceIntent

                    glb = f"{prefix}.glb"
                    export_scene(
                        AcceptedAssembly(
                            records=records,
                            service=VisualOnlyServiceIntent(),
                            receipt=VisualAssemblyReceipt(),
                        ),
                        glb,
                    )
                    native_path = KERNEL_ROOT / "presentation" / "native.py"
                    module_spec = importlib.util.spec_from_file_location(
                        "golem_native_render", str(native_path)
                    )
                    if module_spec is None or module_spec.loader is None:
                        raise RuntimeError(
                            f"cannot load native renderer at {native_path}"
                        )
                    native = importlib.util.module_from_spec(module_spec)
                    module_spec.loader.exec_module(native)
                    paths = tuple(native.render(glb, prefix))
                    return AcceptedCommand(
                        state,
                        "rendered:\n"
                        + "\n".join(f"  {path}" for path in paths),
                    )
                except Exception as exception:
                    return _reject(
                        state,
                        "RenderFailure",
                        address,
                        f"{type(exception).__name__}: {exception}",
                    )
            case SaveSessionEffect(path):
                try:
                    path.write_text(json.dumps(state.spec, indent=1))
                    Path(f"{path}.journal.txt").write_text(
                        "\n".join(state.journal) + "\n"
                    )
                except OSError as exception:
                    return _reject(
                        state, "SaveFailure", str(path), str(exception)
                    )
                return AcceptedCommand(state, f"saved {path} (+journal)")
            case LoadHelpEffect():
                try:
                    return AcceptedCommand(state, self.help_path.read_text())
                except OSError as exception:
                    return _reject(
                        state,
                        "HelpFailure",
                        str(self.help_path),
                        str(exception),
                    )
            case unreachable:
                assert_never(unreachable)


__all__ = ["Effect", "Harness"]
