from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from golem.assembly import (
    AcceptedAssembly,
    AssemblyRecord,
    AssemblyResolutionPolicy,
    AssemblyStratum,
    PinnedResolution,
    RejectedAssembly,
    TargetPitch,
    compile_assembly,
)
from golem.assembly.__main__ import (
    _assembly_obstruction_lines,
    _coupled_receipt_lines,
    _fit_receipt_lines,
    _output_lines,
)
from golem.assembly.exchange import export_scene
from golem.cli.model import (
    CommandDescriptor,
    CommandResult,
    CommandRunner,
    ExclusiveOptions,
    OptionArgument,
    PositionalArgument,
)
from golem.cli.spec import (
    ASSEMBLY_DOCUMENT,
    BODY_DOCUMENT,
    GRAPH_DOCUMENT,
    SpecSourceObstruction,
    UnrecognizedDocumentObstruction,
    assembly_document,
    classify_document,
    load_spec,
    render_source_obstruction,
    render_unrecognized_document,
)
from golem.session.fault import (
    ASSEMBLY_COMPILE_SEAM,
    EngineFault,
    project_engine_fault,
)

COMPILE_DIALECTS = (BODY_DOCUMENT, GRAPH_DOCUMENT, ASSEMBLY_DOCUMENT)

type CompilationObstruction = (
    SpecSourceObstruction
    | UnrecognizedDocumentObstruction
    | RejectedAssembly
    | EngineFault
)
type CompilationOutcome = AcceptedAssembly | CompilationObstruction


@dataclass(frozen=True)
class CompileArguments:
    spec_path: Path
    resolution: int | None
    pitch: float | None
    output_path: Path | None


def _compile_arguments(namespace: Namespace) -> CompileArguments:
    return CompileArguments(
        spec_path=namespace.spec,
        resolution=namespace.resolution,
        pitch=namespace.pitch,
        output_path=namespace.output_path,
    )


def _resolution_policy(
    arguments: CompileArguments,
) -> AssemblyResolutionPolicy | None:
    return (
        PinnedResolution(arguments.resolution)
        if arguments.resolution is not None
        else TargetPitch(arguments.pitch)
        if arguments.pitch is not None
        else None
    )


def _accepted_lines(
    result: AcceptedAssembly,
    output_path: Path | None,
) -> tuple[str, ...]:
    solid_records = tuple(
        record
        for record in result.records
        if record.stratum is AssemblyStratum.SOLID
    )
    return (
        *_output_lines(result.records),
        f"physical_evidence: {result.receipt.physical_evidence.value}",
        *_fit_receipt_lines(result),
        *_coupled_receipt_lines(result),
        *_violation_lines(solid_records),
        *((f"scene -> {output_path}",) if output_path is not None else ()),
        "assembly verdict: AcceptedAssembly",
    )


def _violation_lines(records: tuple[AssemblyRecord, ...]) -> tuple[str, ...]:
    violations = tuple(
        violation
        for record in records
        for violation in record.violations
    )
    return (
        f"advisories vocabulary_violations={len(violations)}",
        *tuple(
            f"  advisory [{violation['rule']}] "
            f"{violation['part']}: {violation['detail']}"
            + (
                f" -> {violation['repair']}"
                if violation.get("repair") is not None
                else ""
            )
            for violation in violations
        ),
    )


def compile_spec(
    spec_path: Path,
    resolution_policy: AssemblyResolutionPolicy | None = None,
) -> CompilationOutcome:
    source = load_spec(spec_path)
    if isinstance(source, SpecSourceObstruction):
        return source
    unrecognized = classify_document(source, COMPILE_DIALECTS)
    if unrecognized is not None:
        return unrecognized
    try:
        return compile_assembly(
            assembly_document(source),
            source.base_dir,
            resolution_policy=resolution_policy,
        )
    except Exception as failure:
        return project_engine_fault(ASSEMBLY_COMPILE_SEAM, failure)


def compilation_obstruction_result(
    obstruction: CompilationObstruction,
    refusal_line: str | None = None,
) -> CommandResult:
    match obstruction:
        case SpecSourceObstruction():
            return CommandResult(
                1,
                stderr=render_source_obstruction(obstruction),
            )
        case UnrecognizedDocumentObstruction():
            return CommandResult(
                1,
                stderr=render_unrecognized_document(obstruction),
            )
        case RejectedAssembly():
            return CommandResult(
                1,
                stdout="\n".join(
                    (
                        *_assembly_obstruction_lines(obstruction),
                        *((refusal_line,) if refusal_line is not None else ()),
                        "assembly verdict: RejectedAssembly",
                        "",
                    )
                ),
            )
        case EngineFault() as fault:
            return CommandResult(
                1,
                stderr="".join(
                    (
                        fault.render_refusal(),
                        *(
                            (f"{refusal_line}\n",)
                            if refusal_line is not None
                            else ()
                        ),
                    )
                ),
            )
        case _ as unreachable:
            assert_never(unreachable)


def _accepted_result(
    result: AcceptedAssembly,
    arguments: CompileArguments,
) -> CommandResult:
    if arguments.output_path is not None:
        try:
            export_scene(result, str(arguments.output_path))
        except Exception as failure:
            return CommandResult(
                1,
                stdout="\n".join((*_accepted_lines(result, None), "")),
                stderr=(
                    f"REJECTED [SceneExportObstruction] "
                    f"{arguments.output_path}: {type(failure).__name__}: "
                    f"{failure}\n"
                ),
            )
    return CommandResult(
        0,
        stdout="\n".join(
            (*_accepted_lines(result, arguments.output_path), "")
        ),
    )


def run(namespace: Namespace) -> CommandResult:
    arguments = _compile_arguments(namespace)
    result = compile_spec(
        arguments.spec_path,
        _resolution_policy(arguments),
    )
    return (
        _accepted_result(result, arguments)
        if isinstance(result, AcceptedAssembly)
        else compilation_obstruction_result(
            result,
            (
                f"scene refused -> {arguments.output_path} "
                "(typed rejection above)"
                if arguments.output_path is not None
                else None
            ),
        )
    )


COMMAND = CommandDescriptor(
    name="compile",
    help_line="compile a body, graph, or assembly and optionally emit GLB",
    runner=CommandRunner(
        arguments=(
            PositionalArgument(
                "spec",
                "body/0.3 document, raw part graph, or assembly document",
                Path,
            ),
            ExclusiveOptions(
                (
                    OptionArgument(
                        ("--res",),
                        "resolution",
                        "pin the author's grid resolution",
                        int,
                        "N",
                    ),
                    OptionArgument(
                        ("--pitch",),
                        "pitch",
                        "target a world-space grid pitch",
                        float,
                        "P",
                    ),
                )
            ),
            OptionArgument(
                ("--out",),
                "output_path",
                "emit the accepted assembly as a GLB scene",
                Path,
                "FILE.GLB",
            ),
        ),
        evaluate=run,
    ),
)
