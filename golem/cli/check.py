from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import assert_never

from golem.assembly import AcceptedAssembly, RejectedAssembly
from golem.assembly.__main__ import _assembly_obstruction_lines
from golem.cli.model import (
    CommandDescriptor,
    CommandResult,
    CommandRunner,
    PositionalArgument,
    SwitchArgument,
)
from golem.cli.spec import (
    BODY_DOCUMENT,
    GRAPH_DOCUMENT,
    LoadedSpec,
    SpecSourceObstruction,
    classify_document,
    is_body_document,
    load_spec,
    render_source_obstruction,
    render_unrecognized_document,
)
from golem.kernel.anatomy import (
    AcceptedVasculature,
    RejectedAnatomy,
    RejectedVasculature,
    VasculatureResult,
)

CHECK_DIALECTS = (BODY_DOCUMENT, GRAPH_DOCUMENT)
from golem.kernel.body import (
    obstruction_text,
)
from golem.kernel.body.report import assertion_records_pass, run_asserts
from golem.kernel.engine.types import Accepted, GeometryObstruction, Rejected, decode_graph
from golem.senses.model import RejectedSenses, Senses
from golem.senses.project import rejected_senses_text
from golem.senses.proprio import detect_anomalies, render_receipt
from golem.senses.proprio.anomaly import (
    _ANOMALY_CAP,
    _ASSERT_CAP,
    _apply_suppression,
)
from golem.senses.proprio.render import _anatomy_receipt_lines
from golem.session.fault import (
    ASSEMBLY_COMPILE_SEAM,
    CHECK_DECODE_SEAM,
    CHECK_SENSES_SEAM,
    EngineFault,
    project_engine_fault,
)
from golem.session.protocol import (
    AuthoringEvaluation,
    Diagnostic,
    evaluate_authoring_surfaces,
)
from golem.session.state import (
    BodyCompileObstruction,
    CompileObstructed,
    Compiled,
    SessionState,
)


@dataclass(frozen=True)
class CheckArguments:
    spec_path: Path
    all_records: bool


@dataclass(frozen=True)
class CheckGraph:
    graph: dict[str, object]
    body_receipt: Mapping[str, object] | None
    authoring_evaluation: AuthoringEvaluation | None


type VascularFeasibility = VasculatureResult | RejectedAnatomy | None


def _check_arguments(namespace: Namespace) -> CheckArguments:
    return CheckArguments(
        spec_path=namespace.spec,
        all_records=namespace.all_records,
    )


def _assertion_line(
    record: Mapping[str, object],
    suffix_by_rule_id: Mapping[str, tuple[str, ...]],
) -> str:
    rule_id = str(record["id"])
    suffix = suffix_by_rule_id.get(rule_id, ())
    return "  " + str(record["human"]) + (
        "  [also: " + "; ".join(suffix) + "]" if suffix else ""
    )


def _uncapped_receipt(
    receipt: str,
    senses: Senses,
    assertion_records: Sequence[Mapping[str, object]],
    anomalies: Sequence[Mapping[str, object]],
) -> str:
    suppression = _apply_suppression(assertion_records, anomalies)
    failed = tuple(
        record
        for record in assertion_records
        if record.get("status") == "fail"
    )
    unmeasurable = tuple(
        record
        for record in assertion_records
        if record.get("status") == "unmeasurable"
    )
    assertion_tail = (*failed, *unmeasurable)[_ASSERT_CAP:]
    anomaly_tail = suppression.visible_anomalies[_ANOMALY_CAP:]
    suffix = (
        f" ({suppression.suppressed_count} suppressed into asserts)"
        if suppression.suppressed_count
        else ""
    )
    assertion_replacements = (
        (
            (
                f"  +{len(assertion_tail)} more -- proprio(asserts)",
                "\n".join(
                    _assertion_line(record, suppression.suffix_by_rule_id)
                    for record in assertion_tail
                ),
            ),
        )
        if assertion_tail
        else ()
    )
    anomaly_replacements = (
        (
            (
                f"ANOMALY {_ANOMALY_CAP}{suffix}",
                f"ANOMALY {len(suppression.visible_anomalies)}{suffix}",
            ),
            (
                f"  +{len(anomaly_tail)} more -- proprio(anomalies)",
                "\n".join(
                    f"  N{index} {anomaly['det']} @ "
                    f"{anomaly['addr']}: {anomaly['detail']}"
                    for index, anomaly in enumerate(
                        anomaly_tail,
                        _ANOMALY_CAP + 1,
                    )
                ),
            ),
        )
        if anomaly_tail
        else ()
    )
    replacements = (*assertion_replacements, *anomaly_replacements)
    expanded = reduce(
        lambda rendered, replacement: rendered.replace(*replacement, 1),
        replacements,
        receipt,
    )
    underbuilt_carriers = (
        tuple(
            row
            for row in senses.anatomy.get("carrier_rows", ())
            if isinstance(row, Mapping)
            and isinstance(row.get("suggested_scale"), (int, float))
            and float(row["suggested_scale"]) > 1.0 + 1.0e-9
        )
        if isinstance(senses.anatomy, Mapping)
        else ()
    )
    rendered_carrier_count = sum(
        "; suggest scale " in line
        for line in _anatomy_receipt_lines(senses.anatomy)
    )
    carrier_tail = underbuilt_carriers[rendered_carrier_count:]
    carrier_tail_lines = (
        tuple(
            line
            for offset in range(
                0,
                len(carrier_tail),
                rendered_carrier_count,
            )
            for line in _anatomy_receipt_lines(
                {
                    **senses.anatomy,
                    "circuits": (),
                    "carrier_rows": carrier_tail[
                        offset : offset + rendered_carrier_count
                    ],
                }
            )[2:]
        )
        if carrier_tail
        and rendered_carrier_count > 0
        and isinstance(senses.anatomy, Mapping)
        else ()
    )
    return (
        expanded.rstrip("\n")
        + "\n"
        + "\n".join(carrier_tail_lines)
        + "\n"
        if carrier_tail_lines
        else expanded
    )


def _render_check_receipt(
    senses: Senses,
    assertion_records: Sequence[Mapping[str, object]],
    anomalies: Sequence[Mapping[str, object]],
    *,
    pack: str,
    all_records: bool,
) -> str:
    receipt = render_receipt(
        senses,
        list(assertion_records),
        list(anomalies),
        pack=pack,
    )
    return (
        _uncapped_receipt(
            receipt,
            senses,
            assertion_records,
            anomalies,
        )
        if all_records
        else receipt
    )


def _decode_source(source: LoadedSpec) -> CheckGraph | CommandResult:
    try:
        if not is_body_document(source.payload):
            return CheckGraph(source.payload, None, None)
        state = SessionState(source.payload, source.base_dir)
        match state.outcome:
            case Compiled(graph, receipt):
                return CheckGraph(
                    graph,
                    receipt,
                    evaluate_authoring_surfaces(state),
                )
            case CompileObstructed((BodyCompileObstruction(obstructions),)):
                lines = tuple(
                    f"  [{type(obstruction).__name__}] "
                    f"{obstruction_text(obstruction)}"
                    for obstruction in obstructions
                )
                return CommandResult(
                    1,
                    stdout="\n".join(
                        (f"OBSTRUCTIONS {len(lines)}", *lines, "")
                    ),
                )
            case CompileObstructed((EngineFault() as fault,)):
                return CommandResult(
                    1,
                    stderr=(
                        f"REJECTED [engine.fault] {source.path}: "
                        f"{json.dumps(fault.to_json(), sort_keys=True)}\n"
                    ),
                )
            case _ as unreachable:
                assert_never(unreachable)
    except Exception as failure:
        fault = project_engine_fault(CHECK_DECODE_SEAM, failure)
        return CommandResult(
            1,
            stderr=(
                f"REJECTED [engine.fault] {source.path}: "
                f"{json.dumps(fault.to_json(), sort_keys=True)}\n"
            ),
        )


def _geometry_obstruction_line(obstruction: GeometryObstruction) -> str:
    return (
        f"  [{obstruction.rule.value}] {obstruction.part_id} "
        f"({obstruction.kind}): {obstruction.detail}"
    )


def _body_violation_lines(
    receipt: Mapping[str, object] | None,
) -> tuple[str, ...]:
    if receipt is None:
        return ()
    violations = receipt.get("violations", ())
    solved = receipt.get("solved", {})
    limits = (
        solved.get("limit_violations", ())
        if isinstance(solved, Mapping)
        else ()
    )
    return (
        *tuple(
            "  ["
            + str(violation.get("rule", "BodyViolation"))
            + "] "
            + str(violation.get("address", "spec"))
            + ": "
            + str(violation.get("detail", "invalid body declaration"))
            for violation in violations
            if isinstance(violation, Mapping)
        ),
        *tuple(
            "  [joint_limit] skeleton/"
            + str(violation.get("bone", "unknown"))
            + ": "
            + json.dumps(dict(violation), sort_keys=True, default=str)
            for violation in limits
            if isinstance(violation, Mapping)
        ),
    )


def _obstruction_block(lines: tuple[str, ...]) -> str:
    return (
        ""
        if not lines
        else "\n".join(("", f"OBSTRUCTIONS {len(lines)}", *lines, ""))
    )


def _acceptance_diagnostic_line(diagnostic: Diagnostic) -> str:
    return (
        f"REJECTED [{diagnostic.code}] "
        f"{json.dumps(diagnostic.to_json(), sort_keys=True, default=str)}"
    )


def _assembly_acceptance_block(
    evaluation: AuthoringEvaluation | None,
) -> str:
    if evaluation is None:
        return ""
    match evaluation.assembly:
        case AcceptedAssembly():
            return "\nASSEMBLY acceptance: ACCEPTED\n"
        case RejectedAssembly() as rejected:
            return "\n".join(
                (
                    "",
                    "ASSEMBLY acceptance: REJECTED",
                    *_assembly_obstruction_lines(rejected),
                    "",
                )
            )
        case None:
            assembly_diagnostics = tuple(
                diagnostic
                for diagnostic in evaluation.evidence.diagnostics
                if diagnostic.severity == "error"
                and (
                    diagnostic.code.startswith("assembly.")
                    or diagnostic.address == ASSEMBLY_COMPILE_SEAM
                )
            )
            return (
                "\n".join(
                    (
                        "",
                        "ASSEMBLY acceptance: REJECTED",
                        *tuple(
                            map(
                                _acceptance_diagnostic_line,
                                assembly_diagnostics,
                            )
                        ),
                        "",
                    )
                )
                if assembly_diagnostics
                else ""
            )
        case _ as unreachable:
            assert_never(unreachable)


def _vascular_block(result: VascularFeasibility) -> str:
    match result:
        case None:
            return ""
        case AcceptedVasculature(graph, receipt):
            return "\n".join(
                (
                    "",
                    "VASCULAR feasibility: ACCEPTED",
                    "  vascular "
                    f"nodes={len(graph.nodes)} "
                    f"edges={len(graph.edges)} "
                    f"terminal_pairs={receipt.terminal_pair_budget} "
                    f"residual={receipt.maximum_free_cell_residual:.3e} "
                    f"balance={receipt.boundary_balance_error:.3e}",
                    "  allocation "
                    + json.dumps(
                        dict(receipt.terminal_pairs_by_region),
                        sort_keys=True,
                    ),
                    "",
                )
            )
        case RejectedAnatomy(obstructions) | RejectedVasculature(obstructions):
            return "\n".join(
                (
                    "",
                    "VASCULAR feasibility: REJECTED",
                    *_assembly_obstruction_lines(
                        RejectedAssembly(obstructions)
                    ),
                    "",
                )
            )
        case _ as unreachable:
            assert_never(unreachable)


def _check_loaded(source: LoadedSpec, all_records: bool) -> CommandResult:
    decoded_source = _decode_source(source)
    if isinstance(decoded_source, CommandResult):
        return decoded_source
    try:
        decoded_graph = decode_graph(decoded_source.graph)
    except Exception as failure:
        fault = project_engine_fault(CHECK_DECODE_SEAM, failure)
        return CommandResult(
            1,
            stderr=(
                f"REJECTED [engine.fault] {source.path}: "
                f"{json.dumps(fault.to_json(), sort_keys=True)}\n"
            ),
        )
    if isinstance(decoded_graph, Rejected):
        lines = tuple(map(_geometry_obstruction_line, decoded_graph.obstructions))
        return CommandResult(
            1,
            stdout="\n".join((f"OBSTRUCTIONS {len(lines)}", *lines, "")),
        )
    if not isinstance(decoded_graph, Accepted):
        return CommandResult(
            1,
            stderr="REJECTED [MalformedDecodeResult] geometry decoder returned no verdict\n",
        )
    graph = decoded_source.graph
    intent_value = graph.get("intent")
    intent = intent_value if isinstance(intent_value, dict) else {}
    try:
        assertion_result = run_asserts(graph, intent)
        match assertion_result:
            case RejectedSenses() as rejected:
                return CommandResult(
                    1,
                    stderr=rejected_senses_text(rejected),
                )
            case (senses, assertion_records):
                pass
            case _ as unreachable:
                assert_never(unreachable)
        anomalies = detect_anomalies(senses)
        assertions = intent.get("asserts")
        pack = (
            str(assertions.get("pack", "body.contract"))
            if isinstance(assertions, Mapping)
            else "body.contract"
        )
        receipt = _render_check_receipt(
            senses,
            assertion_records,
            anomalies,
            pack=pack,
            all_records=all_records,
        )
    except Exception as failure:
        fault = project_engine_fault(CHECK_SENSES_SEAM, failure)
        return CommandResult(
            1,
            stderr=(
                f"REJECTED [engine.fault] {source.path}: "
                f"{json.dumps(fault.to_json(), sort_keys=True)}\n"
            ),
        )
    geometry_obstructions = decoded_graph.obstructions
    body_obstruction_lines = _body_violation_lines(decoded_source.body_receipt)
    obstruction_lines = (
        *tuple(map(_geometry_obstruction_line, geometry_obstructions)),
        *body_obstruction_lines,
    )
    vascular_result = (
        decoded_source.authoring_evaluation.vasculature
        if decoded_source.authoring_evaluation is not None
        else None
    )
    analytic_check_passed = (
        assertion_records_pass(assertion_records)
        and not any(
            obstruction.fatal for obstruction in geometry_obstructions
        )
        and not body_obstruction_lines
        and not isinstance(
            vascular_result,
            (RejectedAnatomy, RejectedVasculature),
        )
    )
    check_passed = (
        decoded_source.authoring_evaluation.evidence.accepted
        if decoded_source.authoring_evaluation is not None
        else analytic_check_passed
    )
    return CommandResult(
        0 if check_passed else 1,
        stdout=(
            receipt.rstrip("\n")
            + _vascular_block(vascular_result)
            + _assembly_acceptance_block(
                decoded_source.authoring_evaluation
            )
            + _obstruction_block(obstruction_lines)
            + "\n"
        ),
    )


def run(namespace: Namespace) -> CommandResult:
    arguments = _check_arguments(namespace)
    source = load_spec(arguments.spec_path)
    if isinstance(source, SpecSourceObstruction):
        return CommandResult(1, stderr=render_source_obstruction(source))
    unrecognized = classify_document(source, CHECK_DIALECTS)
    return (
        CommandResult(1, stderr=render_unrecognized_document(unrecognized))
        if unrecognized is not None
        else _check_loaded(source, arguments.all_records)
    )


COMMAND = CommandDescriptor(
    name="check",
    help_line="decode, validate, and sense a body or graph without meshing",
    runner=CommandRunner(
        arguments=(
            PositionalArgument(
                "spec",
                "body/0.3 document or raw part graph",
                Path,
            ),
            SwitchArgument(
                ("--all",),
                "all_records",
                "render all failed/unmeasurable assertions, unexpected "
                "anomalies, and suggested carrier scales",
            ),
        ),
        evaluate=run,
    ),
)
