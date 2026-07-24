"""Canonical transaction protocol over the GOLEM session edit algebra."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum, StrEnum
from functools import reduce
from pathlib import Path
from typing import assert_never

from golem.addressing.session_grammar import encode_address_segment
from golem.assembly import (
    AcceptedAssembly,
    ElementFitLaw,
    RejectedAssembly,
    compile_assembly,
)
from golem.assembly.carriers import (
    AssemblyRecord,
    ElementFitReceipt,
)
from golem.assembly.project import project_assembly_obstructions
from golem.cli.spec import LoadedSpec, assembly_document
from golem.contract import section_names
from golem.contracts.model import (
    ClauseIdentity,
    EqualOperator,
    Failed,
    MinimumOperator,
    Passed,
    Unmeasurable,
    UpperBoundOperator,
    Verdict,
    WithinOperator,
)
from golem.contracts.operators import operator_name, operator_value
from golem.kernel.anatomy import (
    AcceptedAnatomy,
    AcceptedVasculature,
    CapsuleEscapeObstruction,
    DemandDeliveryMismatchObstruction,
    InfeasibleBifurcationObstruction,
    RejectedAnatomy,
    RejectedVasculature,
    ReversedVascularFlowObstruction,
    SealedVascularConfig,
    SymmetryMismatchObstruction,
    VascularGluingObstruction,
    VascularIntersectionObstruction,
    VascularSearchBudgetObstruction,
    VascularSolverResidualObstruction,
    VascularSymmetryPredicate,
    anatomy_to_dict,
    project_vascular_obstruction,
    realize_vasculature,
    vasculature_to_dict,
)
from golem.kernel.body import DIALECT as BODY_DIALECT
from golem.kernel.body.project import project_obstruction
from golem.kernel.body.types import BodyObstruction
from golem.kernel.engine.types import Accepted, Rejected, decode_graph
from golem.senses.proprio.anomaly import Anomaly
from golem.session import algebra, ops
from golem.session.diff import diff_specs
from golem.session.effect import compile_authored
from golem.session.journal import (
    JournalEntry,
    JournalState,
    RepairJournalEvent,
    RepairOperationKind,
)
from golem.session.fault import (
    ASSEMBLY_COMPILE_SEAM,
    ENGINE_FAULT_CODE,
    SESSION_EVIDENCE_SEAM,
    EngineFault,
    project_engine_fault,
)
from golem.session.state import (
    BodyCompileObstruction,
    CompileObstructed,
    CompileObstruction,
    Compiled,
    SessionState,
)


SCHEMA = "golem.session.verdict/2"
_SECTION_ORDER = section_names()
_CONTRACT_REFS_BY_DOMAIN: Mapping[str, tuple[str, ...]] = {
    "session": ("schema",),
    "document": ("schema", "primer"),
    "body": ("primer", "schema"),
    "geometry": ("vocabulary", "receipts"),
    "contract": ("receipts",),
    "relation": ("relations", "receipts"),
    "anatomy": ("primer", "schema", "receipts"),
    "vascular": ("envelope", "receipts"),
    "assembly": ("receipts",),
}


def _stable_float(value: float | int) -> float:
    return float(f"{float(value):.12g}")


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list, frozenset, set)):
        return tuple(map(_json_value, value))
    if isinstance(value, float):
        return _stable_float(value) if math.isfinite(value) else str(value)
    item = getattr(value, "item", None)
    return _json_value(item()) if callable(item) else value


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: str
    address: str
    predicate: str
    required: object
    observed: object
    witness: object
    help_operations: tuple[ops.Op, ...] = ()
    contract_refs: tuple[str, ...] = ()
    help_unavailable: str | None = None


    def to_json(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "address": self.address,
            "predicate": self.predicate,
            "required": _json_value(self.required),
            "observed": _json_value(self.observed),
            "witness": _json_value(self.witness),
            "help": {
                "operations": tuple(map(ops.to_json, self.help_operations)),
                **(
                    {"unavailable": self.help_unavailable}
                    if self.help_unavailable is not None
                    else {}
                ),
            },
        }


def witness_expected(witness: object) -> bool:
    return isinstance(witness, Mapping) and witness.get("expected") is True


def diagnostic_expected(diagnostic: Diagnostic) -> bool:
    """The one acceptance, filtering, and counting discriminator.

    A diagnostic is expected iff its witness carries the declared-intent
    marker ``"expected": true``. Rendering and acceptance consume the same
    witness predicate; check consumes ``AuthoringEvidence.accepted``.
    """
    return witness_expected(diagnostic.witness)


class MarginDisposition(StrEnum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    REPAIRED = "repaired"


@dataclass(frozen=True)
class Margin:
    constraint: str
    address: str
    predicate: str
    required: object
    observed: object
    margin: float
    normalized_margin: float
    unit: str | None
    authoring_addresses: tuple[str, ...] = ()
    disposition: MarginDisposition = MarginDisposition.SATISFIED
    active: bool = False
    witness: object = None

    def to_json(self) -> dict[str, object]:
        return {
            "constraint": self.constraint,
            "address": self.address,
            "predicate": self.predicate,
            "required": _json_value(self.required),
            "observed": _json_value(self.observed),
            "margin": _stable_float(self.margin),
            "normalized_margin": _stable_float(self.normalized_margin),
            "unit": self.unit,
            "authoring_addresses": self.authoring_addresses,
            "disposition": self.disposition.value,
            "active": self.active,
            "witness": _json_value(self.witness),
        }


@dataclass(frozen=True)
class AuthoringEvidence:
    diagnostics: tuple[Diagnostic, ...]
    margins: tuple[Margin, ...]
    journal_events: tuple[RepairJournalEvent, ...] = ()

    @property
    def accepted(self) -> bool:
        # Evidence law (Wall 006): acceptance REQUIRES zero unexpected
        # diagnostics -- an undeclared observation blocks acceptance even at
        # warning severity, while an expected (authored-intent) observation
        # rides. Severity never enters this property; it governs the
        # analytic->assembly computation gate instead, so warnings suppress
        # acceptance here without suppressing downstream evidence there.
        return all(map(diagnostic_expected, self.diagnostics))

    @property
    def binding_constraint(self) -> Margin | None:
        return next(
            (margin for margin in self.margins if margin.active),
            None,
        )

    @property
    def contract_refs(self) -> tuple[str, ...]:
        explicit = tuple(
            reference
            for diagnostic in self.diagnostics
            for reference in diagnostic.contract_refs
        )
        margin_domains = tuple(
            margin.constraint.partition(":")[0]
            for margin in self.margins
        )
        requested = frozenset((*explicit, *_contract_refs(margin_domains)))
        return tuple(
            section for section in _SECTION_ORDER if section in requested
        )


@dataclass(frozen=True)
class MarginDelta:
    constraint: str
    address: str
    before: float | None
    after: float | None
    delta: float | None
    before_normalized: float | None
    after_normalized: float | None
    normalized_delta: float | None

    def to_json(self) -> dict[str, object]:
        return {
            "constraint": self.constraint,
            "address": self.address,
            "before": None if self.before is None else _stable_float(self.before),
            "after": None if self.after is None else _stable_float(self.after),
            "delta": None if self.delta is None else _stable_float(self.delta),
            "before_normalized": (
                None
                if self.before_normalized is None
                else _stable_float(self.before_normalized)
            ),
            "after_normalized": (
                None
                if self.after_normalized is None
                else _stable_float(self.after_normalized)
            ),
            "normalized_delta": (
                None
                if self.normalized_delta is None
                else _stable_float(self.normalized_delta)
            ),
        }


@dataclass(frozen=True)
class VerdictDeltas:
    margins: tuple[MarginDelta, ...] = ()
    newly_active: tuple[str, ...] = ()
    newly_inactive: tuple[str, ...] = ()

    def to_json(self) -> dict[str, object]:
        return {
            "margins": tuple(map(MarginDelta.to_json, self.margins)),
            "newly_active": self.newly_active,
            "newly_inactive": self.newly_inactive,
        }


type EvidenceChannel = Callable[[SessionState], AuthoringEvidence]


@dataclass(frozen=True)
class ProtocolSession:
    _model: algebra.SessionModel
    _evidence: AuthoringEvidence
    _evidence_channel: EvidenceChannel
    _ingress_diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def document(self) -> dict[str, object]:
        return copy.deepcopy(algebra.current_branch(self._model).state.spec)

    @property
    def txn(self) -> int:
        return algebra.current_branch(self._model).state.txn

    @property
    def journal(self) -> JournalState:
        return algebra.current_branch(self._model).journal

    @property
    def evidence(self) -> AuthoringEvidence:
        return copy.deepcopy(self._evidence)


@dataclass(frozen=True)
class TransactionVerdict:
    _evidence: AuthoringEvidence
    deltas: VerdictDeltas
    changed_addresses: tuple[str, ...]
    base_txn: int
    txn: int
    operations: tuple[ops.Op, ...]
    journal: tuple[JournalEntry, ...]

    @property
    def accepted(self) -> bool:
        return self._evidence.accepted

    @property
    def evidence(self) -> AuthoringEvidence:
        return copy.deepcopy(self._evidence)

    def to_json(self) -> dict[str, object]:
        return {
            "status": "accepted" if self.accepted else "rejected",
            "diagnostics": tuple(
                map(Diagnostic.to_json, self._evidence.diagnostics)
            ),
            "margins": tuple(map(Margin.to_json, self._evidence.margins)),
            "binding_constraint": (
                self._evidence.binding_constraint.to_json()
                if self._evidence.binding_constraint is not None
                else None
            ),
            "deltas": self.deltas.to_json(),
            "contract_refs": self._evidence.contract_refs,
            "changed_addresses": self.changed_addresses,
            "base_txn": self.base_txn,
            "txn": self.txn,
            "operations": tuple(map(ops.to_json, self.operations)),
            "journal": tuple(
                entry.to_json()
                for entry in (*self.journal, *self._evidence.journal_events)
            ),
            "schema": SCHEMA,
        }


@dataclass(frozen=True)
class SessionSubmission:
    session: ProtocolSession
    verdict: TransactionVerdict


@dataclass(frozen=True)
class AddressedTransaction:
    base_txn: int
    operations: tuple[ops.Op, ...]


@dataclass(frozen=True)
class DocumentTransaction:
    base_txn: int
    document: dict[str, object]


@dataclass(frozen=True)
class RequestObstructed:
    diagnostics: tuple[Diagnostic, ...]
    base_txn: int


type DecodedRequest = AddressedTransaction | DocumentTransaction | RequestObstructed


@dataclass(frozen=True)
class DecodedOperations:
    operations: tuple[ops.Op, ...]


@dataclass(frozen=True)
class OperationsObstructed:
    diagnostics: tuple[Diagnostic, ...]


type OperationsDecodeResult = DecodedOperations | OperationsObstructed


def _contract_refs(domains: Sequence[str]) -> tuple[str, ...]:
    requested = frozenset(
        reference
        for domain in domains
        for reference in _CONTRACT_REFS_BY_DOMAIN.get(domain, ())
    )
    return tuple(
        section for section in _SECTION_ORDER if section in requested
    )


def _diagnostic(
    code: str,
    address: str,
    predicate: str,
    required: object,
    observed: object,
    witness: object,
    domain: str,
    *,
    severity: str = "error",
    help_operations: tuple[ops.Op, ...] = (),
    help_unavailable: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code=code,
        severity=severity,
        address=address,
        predicate=predicate,
        required=required,
        observed=observed,
        witness=witness,
        help_operations=help_operations,
        contract_refs=_contract_refs((domain,)),
        help_unavailable=help_unavailable,
    )


def _request_diagnostic(
    code: str,
    address: str,
    predicate: str,
    required: object,
    observed: object,
) -> Diagnostic:
    return _diagnostic(
        code,
        address,
        predicate,
        required,
        observed,
        {"observed": _json_value(observed)},
        "session",
    )


def _decode_base_txn(
    payload: Mapping[str, object],
    current_txn: int,
    *,
    required: bool,
) -> int | Diagnostic:
    value = payload.get("base_txn")
    if value is None and not required:
        return current_txn
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else _request_diagnostic(
            "session.invalid_base_txn",
            "base_txn",
            "nonnegative integer",
            "integer >= 0",
            value,
        )
    )


def _decode_operations(payload: object) -> OperationsDecodeResult:
    if not isinstance(payload, list):
        return OperationsObstructed(
            (
                _request_diagnostic(
                    "session.invalid_operations",
                    "operations",
                    "array of session operations",
                    "JSON array",
                    type(payload).__name__,
                ),
            ),
        )
    decoded = tuple(map(ops.decode_op, payload))
    obstructions = tuple(
        _diagnostic(
            "session.invalid_operation",
            obstruction.address,
            "operation decodes in the public edit algebra",
            "SetOp | UnsetOp | AddOp | AddAtOp | RemoveOp | RenameOp",
            obstruction.reason,
            {"index": index, "reason": obstruction.reason},
            "session",
        )
        for index, result in enumerate(decoded)
        if isinstance(result, ops.EditObstructed)
        for obstruction in result.obstructions
    )
    return (
        OperationsObstructed(obstructions)
        if obstructions
        else DecodedOperations(
            tuple(
                result
                for result in decoded
                if not isinstance(result, ops.EditObstructed)
            )
        )
    )


def _document_diagnostics(document: object) -> tuple[Diagnostic, ...]:
    if not isinstance(document, dict):
        return (
            _request_diagnostic(
                "session.invalid_document",
                "document",
                "whole document is a JSON object",
                "JSON object",
                type(document).__name__,
            ),
        )
    probe = ops.decode_op(
        {"op": "set", "addr": "meta", "value": document}
    )
    return (
        tuple(
            _request_diagnostic(
                "session.invalid_document",
                obstruction.address,
                "whole document contains only JSON values",
                "JSON object",
                obstruction.reason,
            )
            for obstruction in probe.obstructions
        )
        if isinstance(probe, ops.EditObstructed)
        else ()
    )


def decode_request(payload: object, current_txn: int) -> DecodedRequest:
    if not isinstance(payload, dict):
        return RequestObstructed(
            (
                _request_diagnostic(
                    "session.invalid_request",
                    "request",
                    "transaction envelope or edited document",
                    "JSON object",
                    type(payload).__name__,
                ),
            ),
            current_txn,
        )
    if frozenset(payload) == frozenset({"base_txn", "operations"}):
        base = _decode_base_txn(payload, current_txn, required=True)
        decoded = _decode_operations(payload["operations"])
        diagnostics = (
            *((base,) if isinstance(base, Diagnostic) else ()),
            *(
                decoded.diagnostics
                if isinstance(decoded, OperationsObstructed)
                else ()
            ),
        )
        return (
            RequestObstructed(diagnostics, current_txn)
            if diagnostics
            else AddressedTransaction(
                base if isinstance(base, int) else current_txn,
                decoded.operations
                if isinstance(decoded, DecodedOperations)
                else (),
            )
        )
    if "document" in payload and frozenset(payload) <= frozenset(
        {"base_txn", "document"}
    ):
        base = _decode_base_txn(payload, current_txn, required=False)
        document = payload["document"]
        document_diagnostics = _document_diagnostics(document)
        diagnostics = (
            *((base,) if isinstance(base, Diagnostic) else ()),
            *document_diagnostics,
        )
        return (
            RequestObstructed(diagnostics, current_txn)
            if diagnostics
            else DocumentTransaction(
                base if isinstance(base, int) else current_txn,
                copy.deepcopy(document) if isinstance(document, dict) else {},
            )
        )
    diagnostics = _document_diagnostics(payload)
    return (
        RequestObstructed(diagnostics, current_txn)
        if diagnostics
        else DocumentTransaction(current_txn, copy.deepcopy(payload))
    )


def _margin_scale(required: object, observed: float, margin: float) -> float:
    values = tuple(
        abs(float(value))
        for value in (
            *(required if isinstance(required, (tuple, list)) else (required,)),
            observed,
            margin,
        )
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    return max((*values, 1.0e-12))


def _margin(
    constraint: str,
    address: str,
    predicate: str,
    required: object,
    observed: float,
    margin: float,
    unit: str | None,
    witness: object = None,
    *,
    authoring_addresses: tuple[str, ...] = (),
    disposition: MarginDisposition = MarginDisposition.SATISFIED,
) -> Margin:
    return Margin(
        constraint=constraint,
        address=address,
        predicate=predicate,
        required=required,
        observed=_stable_float(observed),
        margin=_stable_float(margin),
        normalized_margin=_stable_float(
            margin / _margin_scale(required, observed, margin)
        ),
        unit=unit,
        authoring_addresses=authoring_addresses,
        disposition=disposition,
        active=False,
        witness=witness,
    )


def _verdict_identity(verdict: Verdict) -> ClauseIdentity:
    match verdict:
        case Passed(clause, _) | Failed(clause, _, _, _, _):
            return clause.identity
        case Unmeasurable(identity, _):
            return identity
        case _ as unreachable:
            assert_never(unreachable)


def _contract_constraint(
    identities: Sequence[ClauseIdentity],
    index: int,
) -> str:
    identity = identities[index]
    matching = tuple(
        candidate
        for candidate in identities
        if candidate.clause_id == identity.clause_id
    )
    occurrence = sum(
        1
        for candidate in identities[:index]
        if candidate.clause_id == identity.clause_id
    )
    return (
        f"contract:{identity.clause_id}"
        if len(matching) == 1
        else (
            f"contract:{identity.clause_id}:"
            f"{identity.authored_metric}@{occurrence}"
        )
    )


def _writable_contract_addresses(
    sources: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        normalized
        for source in sources
        for normalized in (
            "meta" + source[4:] if source.startswith("spec@") else source,
        )
        if not isinstance(ops.decode_address(normalized), ops.EditObstructed)
    )


def _contract_margin(
    document: dict[str, object],
    verdict: Passed | Failed,
    constraint: str,
) -> Margin:
    match verdict:
        case Passed(clause, measurement):
            disposition = MarginDisposition.SATISFIED
            authoring_addresses = _writable_contract_addresses(
                measurement.offenders
            )
            failure_witness: dict[str, object] = {}
        case Failed(
            clause,
            measurement,
            violation,
            gradient,
            _skeleton_offenders,
        ):
            disposition = MarginDisposition.VIOLATED
            help_evidence = _gradient_help(document, verdict)
            authoring_addresses = help_evidence.authoring_addresses
            failure_witness = {
                "violation": violation,
                "gradient": gradient,
                **(
                    {"help_unavailable": help_evidence.unavailable}
                    if help_evidence.unavailable is not None
                    else {}
                ),
            }
    observed = float(measurement.measured)
    operator = clause.operator
    match operator:
        case UpperBoundOperator(_, value, tolerance):
            slack = value + tolerance - observed
        case MinimumOperator(value, tolerance):
            slack = observed - (value - tolerance)
        case EqualOperator(value, tolerance):
            slack = tolerance - abs(observed - value)
        case WithinOperator(lower, upper, tolerance):
            slack = min(
                observed - (lower - tolerance),
                upper + tolerance - observed,
            )
        case _ as unreachable:
            assert_never(unreachable)
    return _margin(
        constraint,
        measurement.offenders[0] if measurement.offenders else "contract",
        operator_name(operator),
        operator_value(operator),
        observed,
        slack,
        measurement.unit,
        {
            "offenders": measurement.offenders,
            "detail": measurement.detail,
            **failure_witness,
        },
        authoring_addresses=authoring_addresses,
        disposition=disposition,
    )


def _field_value(current: object, token: str | int) -> object:
    if isinstance(token, str) and isinstance(current, Mapping):
        return current.get(token)
    if (
        isinstance(token, int)
        and isinstance(current, list)
        and 0 <= token < len(current)
    ):
        return current[token]
    return None


def _current_field(document: dict[str, object], address: ops.RecordAddress) -> object:
    resolved = ops.resolve_record(document, address)
    if isinstance(resolved, ops.EditObstructed):
        return None
    return reduce(_field_value, address.fields, resolved.record)


@dataclass(frozen=True)
class _GradientHelp:
    operations: tuple[ops.Op, ...]
    unavailable: str | None
    authoring_addresses: tuple[str, ...]


def _gradient_help(
    document: dict[str, object],
    verdict: Failed,
) -> _GradientHelp:
    source = verdict.gradient.knob
    normalized = "meta" + source[4:] if source.startswith("spec@") else source

    def unavailable(reason: str) -> _GradientHelp:
        offenders = ", ".join(verdict.measurement.offenders) or "none"
        return _GradientHelp(
            (),
            f"{reason}; gradient knob {source!r}; measured offenders: {offenders}",
            (),
        )

    address = ops.decode_address(normalized)
    if isinstance(address, ops.EditObstructed):
        return unavailable(
            "gradient knob is derived or is not a writable session address"
        )
    current = _current_field(document, address)
    if (
        isinstance(current, bool)
        or not isinstance(current, (int, float))
        or not math.isfinite(float(current))
    ):
        return unavailable(
            "gradient knob does not resolve to a finite numeric authored field"
        )
    direction = 1.0 if verdict.gradient.direction == "+" else -1.0
    raw = {
        "op": "set",
        "addr": normalized,
        "value": _stable_float(
            float(current) + direction * verdict.gradient.estimate
        ),
    }
    decoded = ops.decode_op(raw)
    if isinstance(decoded, ops.EditObstructed):
        return unavailable("derived set operation does not decode")
    compiled = ops.compile_op(document, decoded)
    if isinstance(compiled, ops.EditObstructed):
        return unavailable("derived set operation does not apply to this document")
    return _GradientHelp((decoded,), None, (normalized,))


def _contract_diagnostic(document: dict[str, object], verdict: Verdict) -> Diagnostic | None:
    match verdict:
        case Passed():
            return None
        case Failed(clause, measurement, violation, gradient, skeleton_offenders):
            help_evidence = _gradient_help(document, verdict)
            return _diagnostic(
                "contract.constraint_failed",
                measurement.offenders[0] if measurement.offenders else "contract",
                operator_name(clause.operator),
                operator_value(clause.operator),
                measurement.measured,
                {
                    "clause": clause.identity.clause_id,
                    "unit": measurement.unit,
                    "offenders": measurement.offenders,
                    "skeleton_offenders": skeleton_offenders,
                    "detail": measurement.detail,
                    "violation": violation,
                    "gradient": gradient,
                },
                "contract",
                help_operations=help_evidence.operations,
                help_unavailable=help_evidence.unavailable,
            )
        case Unmeasurable(identity, obstructions):
            return _diagnostic(
                "contract.unmeasurable",
                identity.clause_id,
                "contract clause is measurable",
                "measurement",
                "unmeasurable",
                {"metric": identity.authored_metric, "obstructions": obstructions},
                "contract",
            )
        case _ as unreachable:
            assert_never(unreachable)


def _relation_margins(receipt: Mapping[str, object]) -> tuple[Margin, ...]:
    solved = receipt.get("solved")
    rows = solved.get("relations", ()) if isinstance(solved, Mapping) else ()
    return tuple(
        _margin(
            f"relation:{row.get('id', index)}",
            str(row.get("owner", "pose/relations")),
            "maximum normalized relation residual",
            1.0,
            float(row["normalized_residual"]),
            1.0 - float(row["normalized_residual"]),
            "normalized_residual",
            row,
            authoring_addresses=(str(row.get("owner", "pose/relations")),),
        )
        for index, row in enumerate(rows)
        if isinstance(row, Mapping)
        and isinstance(row.get("observed"), (int, float))
        and isinstance(row.get("normalized_residual"), (int, float))
        and bool(row.get("satisfied"))
    )


def _body_receipt_diagnostics(receipt: Mapping[str, object]) -> tuple[Diagnostic, ...]:
    solved = receipt.get("solved")
    limits = (
        solved.get("limit_violations", ())
        if isinstance(solved, Mapping)
        else ()
    )
    return (
        *tuple(
            _diagnostic(
                f"body.{violation.get('rule', 'violation')}",
                str(violation.get("address", "body")),
                str(violation.get("rule", "body declaration is valid")),
                violation.get("required"),
                violation.get("observed"),
                violation,
                "body",
            )
            for violation in receipt.get("violations", ())
            if isinstance(violation, Mapping)
        ),
        *tuple(
            _diagnostic(
                "body.joint_limit",
                f"skeleton/"
                f"{encode_address_segment(str(violation.get('bone', 'unknown')))}",
                "posed joint lies within declared limits",
                violation.get("limit"),
                violation.get("observed", violation.get("value")),
                violation,
                "body",
            )
            for violation in limits
            if isinstance(violation, Mapping)
        ),
    )


def _anomaly_diagnostics(anomalies: tuple[Anomaly, ...]) -> tuple[Diagnostic, ...]:
    return tuple(
        _diagnostic(
            f"body.anomaly.{anomaly.kind.value}",
            anomaly.address,
            "derived body senses are non-anomalous",
            "no anomaly",
            anomaly.kind.value,
            anomaly,
            "body",
            severity="warning",
        )
        for anomaly in anomalies
    )


def _explicit_projected_diagnostic(
    domain: str,
    projection: Mapping[str, object],
    *,
    help_operations: tuple[ops.Op, ...] = (),
) -> Diagnostic:
    kind = str(
        projection.get(
            "obstruction",
            projection.get("kind", "Obstruction"),
        )
    )
    contract_domain = str(projection.get("contract_domain", domain))
    return _diagnostic(
        f"{domain}.{kind}",
        str(projection.get("address", domain)),
        str(projection.get("predicate", kind)),
        projection.get("required"),
        projection.get("observed", projection.get("reason")),
        projection,
        contract_domain,
        help_operations=help_operations,
    )


def _body_obstruction_diagnostic(
    body_obstruction: BodyObstruction,
) -> Diagnostic:
    return _explicit_projected_diagnostic(
        "body",
        project_obstruction(body_obstruction),
    )


def _anatomy_diagnostics(result: RejectedAnatomy) -> tuple[Diagnostic, ...]:
    projected = anatomy_to_dict(result)
    obstructions = projected.get("obstructions", ())
    return tuple(
        _explicit_projected_diagnostic("anatomy", obstruction)
        for obstruction in obstructions
        if isinstance(obstruction, Mapping)
    )


def _anatomy_margins(result: AcceptedAnatomy) -> tuple[Margin, ...]:
    return tuple(
        _margin(
            f"anatomy:carrier:{row.bone_id}",
            row.shape_address or row.interface_address,
            row.controlling_constraint.value,
            row.target_minimum_radius,
            row.measured_minimum_radius,
            row.measured_minimum_radius - row.target_minimum_radius,
            "world_unit",
            {
                "interface_address": row.interface_address,
                "normalized_flow": row.normalized_flow,
                "distance_from_pump": row.distance_from_pump,
            },
            authoring_addresses=(
                (row.shape_address,)
                if row.shape_address is not None
                else (row.interface_address,)
            ),
            disposition=(
                MarginDisposition.REPAIRED
                if row.deficit > 0.0
                else MarginDisposition.SATISFIED
            ),
        )
        for row in result.carrier_rows
        if row.measured_minimum_radius is not None
    )


def _anatomy_repair_events(
    result: AcceptedAnatomy,
    txn: int,
) -> tuple[RepairJournalEvent, ...]:
    return tuple(
        RepairJournalEvent(
            txn=txn,
            constraint=f"anatomy:carrier:{row.bone_id}",
            address=row.shape_address or row.interface_address,
            predicate=row.controlling_constraint.value,
            authored=_stable_float(row.measured_minimum_radius),
            repaired=_stable_float(row.target_minimum_radius),
            operation=RepairOperationKind.SCALE_RADIUS,
            factor=_stable_float(row.suggested_scale),
            authoring_addresses=(
                (row.shape_address,)
                if row.shape_address is not None
                else (row.interface_address,)
            ),
        )
        for row in result.carrier_rows
        if row.measured_minimum_radius is not None and row.deficit > 0.0
    )


def _mapping_path(
    value: object,
    *keys: str,
) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return (
        value
        if not keys
        else _mapping_path(value.get(keys[0]), *keys[1:])
    )


def _gluing_witness_regions(
    interrogation: Mapping[str, object],
    hypothesis: str,
) -> tuple[str, ...]:
    witness = _mapping_path(
        interrogation,
        "hypotheses",
        hypothesis,
        "witness",
    )
    sections = (
        witness.get("worst_sections", ())
        if witness is not None
        else ()
    )
    return tuple(
        str(section["region_id"])
        for section in sections
        if isinstance(section, Mapping)
        and isinstance(section.get("region_id"), str)
    )


def _compiled_set_operation(
    document: dict[str, object], address: str, value: object
) -> ops.Op | None:
    decoded = ops.decode_op({"op": "set", "addr": address, "value": value})
    if isinstance(decoded, ops.EditObstructed):
        return None
    compiled = ops.compile_op(document, decoded)
    return None if isinstance(compiled, ops.EditObstructed) else decoded


def _gluing_operations(
    document: dict[str, object],
    projection: Mapping[str, object],
) -> tuple[ops.Op, ...]:
    if projection.get("kind") != "FailedGluing":
        return ()
    interrogation = projection.get("interrogation")
    circulation = _mapping_path(
        document,
        "anatomy",
        "overall",
        "circulation",
    )
    if not isinstance(interrogation, Mapping) or circulation is None:
        return ()
    raw_beds = circulation.get("exchange_beds")
    beds = (
        tuple(raw_beds)
        if isinstance(raw_beds, Sequence)
        and not isinstance(raw_beds, (str, bytes))
        else ()
    )
    bed_regions = tuple(
        str(bed.get("region_id"))
        for bed in beds
        if isinstance(bed, Mapping) and isinstance(bed.get("region_id"), str)
    )
    conditioning_regions = tuple(
        dict.fromkeys(
            _gluing_witness_regions(interrogation, "conditioning")
        )
    )
    pump_region = circulation.get("pump_region_id")
    reanchor_region = next(
        (
            region
            for region in conditioning_regions
            if region not in bed_regions and region != pump_region
        ),
        None,
    )
    reanchor = (
        _compiled_set_operation(
            document,
            "meta@anatomy.overall.circulation.pump_region_id",
            reanchor_region,
        )
        if reanchor_region is not None
        else None
    )
    raw_decay = circulation.get("distance_decay")
    decay = (
        float(raw_decay)
        if isinstance(raw_decay, (int, float))
        and not isinstance(raw_decay, bool)
        and math.isfinite(float(raw_decay))
        and float(raw_decay) > 0.0
        else None
    )
    reroute = (
        _compiled_set_operation(
            document,
            "meta@anatomy.overall.circulation.distance_decay",
            _stable_float((decay + 1.0) / 2.0),
        )
        if decay is not None and not math.isclose(decay, 1.0)
        else None
    )
    required = projection.get("required")
    observed = projection.get("observed")
    reduction = (
        min(0.9, max(0.1, float(required) / float(observed)))
        if isinstance(required, (int, float))
        and not isinstance(required, bool)
        and isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and math.isfinite(float(required))
        and math.isfinite(float(observed))
        and float(required) > 0.0
        and float(observed) > 0.0
        else 0.9
    )
    witness_regions = frozenset(
        (
            *conditioning_regions,
            *_gluing_witness_regions(interrogation, "demand"),
        )
    )
    demand_rows = tuple(
        (index, bed, float(demand))
        for index, bed in enumerate(beds)
        if isinstance(bed, Mapping)
        for demand in (bed.get("demand"),)
        if isinstance(demand, (int, float))
        and not isinstance(demand, bool)
        and math.isfinite(float(demand))
        and float(demand) > 0.0
    )
    matched_rows = tuple(
        sorted(
            (
                row
                for row in demand_rows
                if str(row[1].get("region_id")) in witness_regions
            ),
            key=lambda row: (-row[2], row[0]),
        )[:3]
    )
    fallback_row = max(
        demand_rows,
        key=lambda row: row[2],
        default=None,
    )
    targeted_rows = (
        matched_rows
        if matched_rows
        else ((fallback_row,) if fallback_row is not None else ())
    )
    demand_reductions = tuple(
        operation
        for index, _bed, demand in targeted_rows
        for operation in (
            _compiled_set_operation(
                document,
                (
                    "meta@anatomy.overall.circulation."
                    f"exchange_beds[{index}].demand"
                ),
                _stable_float(demand * reduction),
            ),
        )
        if operation is not None
    )
    return tuple(
        dict.fromkeys(
            operation
            for operation in (reanchor, reroute, *demand_reductions)
            if operation is not None
        )
    )


def _vascular_diagnostics(
    result: RejectedVasculature,
    document: dict[str, object] | None = None,
) -> tuple[Diagnostic, ...]:
    projected = vasculature_to_dict(result)
    obstructions = projected.get("obstructions", ())
    return tuple(
        _explicit_projected_diagnostic(
            "vascular",
            obstruction,
            help_operations=(
                _gluing_operations(document, obstruction)
                if document is not None
                else ()
            ),
        )
        for obstruction in obstructions
        if isinstance(obstruction, Mapping)
    )


def _vascular_authoring_addresses(
    obstruction: object,
    projection: Mapping[str, object],
) -> tuple[str, ...]:
    explicit: tuple[str, ...] = ()
    match obstruction:
        case (
            InfeasibleBifurcationObstruction()
            | VascularSearchBudgetObstruction()
            | CapsuleEscapeObstruction()
            | VascularIntersectionObstruction()
            | ReversedVascularFlowObstruction()
        ) as geometric:
            segments = geometric.failing_segments
        case VascularGluingObstruction() as gluing:
            segments = gluing.failing_segments
            interrogation = projection.get("interrogation")
            raw_addresses = (
                interrogation.get("suspect_addresses", ())
                if isinstance(interrogation, Mapping)
                else ()
            )
            explicit = tuple(
                address
                for address in raw_addresses
                if isinstance(address, str)
            )
        case _:
            segments = ()
    return tuple(
        dict.fromkeys(
            (
                *explicit,
                *(
                    address
                    for segment in segments
                    for address in (
                        *segment.source_host_part_addresses,
                        *segment.source_host_bone_addresses,
                        *segment.target_host_part_addresses,
                        *segment.target_host_bone_addresses,
                    )
                ),
            )
        )
    )


def _vascular_violation_measurement(
    obstruction: object,
    projection: Mapping[str, object],
) -> tuple[str, float, float, float, str] | None:
    config = SealedVascularConfig()
    match obstruction:
        case VascularGluingObstruction(
            gluing_interrogation=interrogation
        ) if interrogation is not None:
            match projection:
                case {
                    "predicate": str(predicate),
                    "required": (int() | float()) as required,
                    "observed": (int() | float()) as observed,
                    "unit": str(unit),
                } if (
                    not isinstance(required, bool)
                    and not isinstance(observed, bool)
                    and float(observed) > float(required)
                ):
                    return (
                        predicate,
                        float(required),
                        float(observed),
                        float(required) - float(observed),
                        unit,
                    )
                case _:
                    return None
        case InfeasibleBifurcationObstruction() as failure:
            return (
                failure.predicate.value,
                failure.required,
                failure.observed,
                failure.observed - failure.required,
                "world_unit",
            )
        case VascularSearchBudgetObstruction() as failure:
            return (
                failure.limiting_predicate.value,
                failure.limiting_required,
                failure.limiting_observed,
                failure.limiting_observed - failure.limiting_required,
                "world_unit",
            )
        case CapsuleEscapeObstruction() as failure:
            return (
                "minimum capsule margin",
                config.wall_clearance,
                failure.margin,
                failure.margin - config.wall_clearance,
                "world_unit",
            )
        case VascularIntersectionObstruction() as failure:
            return (
                "minimum nonincident vessel clearance",
                config.vessel_clearance,
                failure.clearance,
                failure.clearance - config.vessel_clearance,
                "world_unit",
            )
        case SymmetryMismatchObstruction() as failure:
            margin = (
                failure.required - failure.observed
                if failure.predicate
                is VascularSymmetryPredicate.REFLECTION_DISTANCE
                else failure.observed - failure.required
            )
            return (
                failure.predicate.value,
                failure.required,
                failure.observed,
                margin,
                "world_unit",
            )
        case VascularSolverResidualObstruction() as failure:
            return (
                "maximum vascular solver residual",
                failure.tolerance,
                failure.maximum_residual,
                failure.tolerance - failure.maximum_residual,
                "normalized_residual",
            )
        case DemandDeliveryMismatchObstruction() as failure:
            return (
                "maximum demand delivery relative error",
                failure.tolerance,
                failure.relative_error,
                failure.tolerance - failure.relative_error,
                "relative_error",
            )
        case ReversedVascularFlowObstruction() as failure:
            return (
                "vascular edge flow is strictly positive",
                0.0,
                failure.solved_flow,
                failure.solved_flow,
                "flow",
            )
        case _:
            return None


def _vascular_violation_margins(
    result: RejectedVasculature,
) -> tuple[Margin, ...]:
    projected = tuple(map(project_vascular_obstruction, result.obstructions))
    return tuple(
        _margin(
            f"vascular:violation:{type(obstruction).__name__}:{index}",
            str(projection.get("address", "anatomy/vasculature")),
            measurement[0],
            measurement[1],
            measurement[2],
            measurement[3],
            measurement[4],
            projection,
            authoring_addresses=_vascular_authoring_addresses(
                obstruction,
                projection,
            ),
            disposition=MarginDisposition.VIOLATED,
        )
        for index, (obstruction, projection) in enumerate(
            zip(result.obstructions, projected, strict=True)
        )
        for measurement in (
            _vascular_violation_measurement(obstruction, projection),
        )
        if measurement is not None
    )


def _vascular_margins(result: AcceptedVasculature) -> tuple[Margin, ...]:
    receipt = result.receipt
    config = SealedVascularConfig()
    return (
        _margin(
            "vascular:capsule_clearance",
            "anatomy/vasculature",
            "minimum capsule margin",
            config.wall_clearance,
            receipt.minimum_capsule_margin,
            receipt.minimum_capsule_margin - config.wall_clearance,
            "world_unit",
        ),
        _margin(
            "vascular:delivery",
            "anatomy/vasculature",
            "maximum delivery relative error",
            config.delivery_tolerance,
            receipt.maximum_delivery_relative_error,
            config.delivery_tolerance - receipt.maximum_delivery_relative_error,
            "relative_error",
        ),
        _margin(
            "vascular:solver_residual",
            "anatomy/vasculature",
            "maximum free-cell residual",
            config.solver_tolerance,
            receipt.maximum_free_cell_residual,
            config.solver_tolerance - receipt.maximum_free_cell_residual,
            "normalized_residual",
        ),
        _margin(
            "vascular:boundary_balance",
            "anatomy/vasculature",
            "boundary balance error",
            config.solver_tolerance,
            receipt.boundary_balance_error,
            config.solver_tolerance - receipt.boundary_balance_error,
            "normalized_residual",
        ),
        _margin(
            "vascular:symmetry",
            "anatomy/vasculature",
            "maximum bilateral symmetry error",
            1.0e-12,
            receipt.maximum_symmetry_error,
            1.0e-12 - receipt.maximum_symmetry_error,
            "world_unit",
        ),
    )


def _solid_margins(record: AssemblyRecord) -> tuple[Margin, ...]:
    if record.stratum.value != "solid":
        return ()
    components = int(record.report.get("components", 0))
    watertight = record.report.get("watertight_main") is True
    violations = len(record.violations)
    return (
        _margin(
            f"assembly:integrity:{record.record_id}:components",
            record.record_id,
            "exactly one connected component",
            1,
            components,
            1.0 - abs(components - 1),
            "component",
        ),
        _margin(
            f"assembly:integrity:{record.record_id}:watertight",
            record.record_id,
            "watertight main surface",
            True,
            1.0 if watertight else 0.0,
            1.0 if watertight else -1.0,
            "boolean",
        ),
        _margin(
            f"assembly:integrity:{record.record_id}:vocabulary",
            record.record_id,
            "no vocabulary violations",
            0,
            violations,
            1.0 if violations == 0 else -float(violations),
            "violation",
        ),
    )


def _fit_margins(receipt: ElementFitReceipt) -> tuple[Margin, ...]:
    address = f"assembly/{receipt.left_element_id}~{receipt.right_element_id}"
    bounded_contact = receipt.fit_law is ElementFitLaw.BOUNDED_MOUNT_CONTACT
    penetration = (
        ()
        if bounded_contact
        else (
            _margin(
                f"assembly:fit:{receipt.left_element_id}:{receipt.right_element_id}:penetration",
                address,
                receipt.fit_law.value,
                receipt.tolerance_world,
                receipt.maximum_penetration_world,
                receipt.tolerance_world - receipt.maximum_penetration_world,
                "world_unit",
            ),
        )
    )
    certified_clearance = not bounded_contact or receipt.penetrating_sample_count == 0
    clearance = (
        (
            _margin(
                f"assembly:fit:{receipt.left_element_id}:{receipt.right_element_id}:clearance",
                address,
                "clearance within declared maximum",
                receipt.permitted_clearance_world,
                receipt.minimum_clearance_world,
                receipt.permitted_clearance_world
                + receipt.tolerance_world
                - receipt.minimum_clearance_world,
                "world_unit",
            ),
        )
        if receipt.minimum_clearance_world is not None
        and receipt.permitted_clearance_world is not None
        and certified_clearance
        else ()
    )
    fitted = (
        (
            _margin(
                f"assembly:fit:{receipt.left_element_id}:{receipt.right_element_id}:fitted_surface",
                address,
                "minimum fitted surface fraction",
                receipt.required_fitted_surface_fraction,
                receipt.fitted_surface_fraction,
                receipt.fitted_surface_fraction
                - receipt.required_fitted_surface_fraction,
                "fraction",
            ),
        )
        if receipt.fitted_surface_fraction is not None
        and receipt.required_fitted_surface_fraction is not None
        and certified_clearance
        else ()
    )
    return (*penetration, *clearance, *fitted)


def _assembly_margins(result: AcceptedAssembly) -> tuple[Margin, ...]:
    return (
        *tuple(
            margin
            for record in result.records
            for margin in _solid_margins(record)
        ),
        *tuple(
            margin
            for receipt in result.receipt.fit_receipts
            for margin in _fit_margins(receipt)
        ),
    )


def _assembly_diagnostics(result: RejectedAssembly) -> tuple[Diagnostic, ...]:
    return tuple(
        _explicit_projected_diagnostic(
            str(projection.get("diagnostic_domain", "assembly")),
            projection,
        )
        for obstruction in result.obstructions
        for projection in project_assembly_obstructions(obstruction)
    )


def engine_fault_diagnostic(fault: EngineFault) -> Diagnostic:
    return _diagnostic(
        ENGINE_FAULT_CODE,
        fault.seam,
        "engine seam returns a typed verdict, never a raw exception",
        "typed verdict",
        fault.exception_kind,
        fault.to_json(),
        fault.seam.partition(".")[0],
    )


def _compile_obstruction_diagnostics(
    obstruction: CompileObstruction,
) -> tuple[Diagnostic, ...]:
    match obstruction:
        case BodyCompileObstruction(obstructions):
            return tuple(map(_body_obstruction_diagnostic, obstructions))
        case EngineFault() as fault:
            return (engine_fault_diagnostic(fault),)
        case _ as unreachable:
            assert_never(unreachable)


def _body_compile_diagnostics(outcome: CompileObstructed) -> tuple[Diagnostic, ...]:
    return tuple(
        diagnostic
        for obstruction in outcome.obstructions
        for diagnostic in _compile_obstruction_diagnostics(obstruction)
    )


@dataclass(frozen=True)
class _AnalyticEvidence:
    diagnostics: tuple[Diagnostic, ...]
    margins: tuple[Margin, ...]
    journal_events: tuple[RepairJournalEvent, ...] = ()
    vasculature: (
        AcceptedVasculature
        | RejectedVasculature
        | RejectedAnatomy
        | None
    ) = None

    @property
    def accepted(self) -> bool:
        return not any(
            diagnostic.severity == "error"
            for diagnostic in self.diagnostics
        )


def _compiled_evidence(state: SessionState, outcome: Compiled) -> _AnalyticEvidence:
    decoded = decode_graph(outcome.graph)
    geometry_diagnostics = tuple(
        _diagnostic(
            f"geometry.{obstruction.rule.value}",
            obstruction.part_id,
            obstruction.rule.value,
            "geometry rule satisfied",
            obstruction.detail,
            {
                "kind": obstruction.kind,
                "fatal": obstruction.fatal,
                "detail": obstruction.detail,
            },
            "geometry",
            severity="error" if obstruction.fatal else "warning",
        )
        for obstruction in decoded.obstructions
    )
    contract_diagnostics = tuple(
        diagnostic
        for verdict in outcome.verdicts
        for diagnostic in (_contract_diagnostic(state.spec, verdict),)
        if diagnostic is not None
    )
    contract_identities = tuple(map(_verdict_identity, outcome.verdicts))
    contract_margins = tuple(
        _contract_margin(
            state.spec,
            verdict,
            _contract_constraint(contract_identities, index),
        )
        for index, verdict in enumerate(outcome.verdicts)
        if isinstance(verdict, (Passed, Failed))
    )
    base_diagnostics = (
        *geometry_diagnostics,
        *contract_diagnostics,
        *_body_receipt_diagnostics(outcome.receipt),
        *_anomaly_diagnostics(outcome.anomalies),
    )
    base_margins = (
        *contract_margins,
        *_relation_margins(outcome.receipt),
    )
    if isinstance(decoded, Rejected):
        return _AnalyticEvidence(base_diagnostics, base_margins)
    if not isinstance(decoded, Accepted):
        return _AnalyticEvidence(
            (
                *base_diagnostics,
                _diagnostic(
                    "geometry.invalid_decode_result",
                    "graph",
                    "geometry decoder returns Accepted or Rejected",
                    "typed decode result",
                    type(decoded).__name__,
                    {},
                    "geometry",
                ),
            ),
            base_margins,
        )
    match outcome.anatomy:
        case None:
            return _AnalyticEvidence(base_diagnostics, base_margins)
        case RejectedAnatomy() as rejected:
            return _AnalyticEvidence(
                (*base_diagnostics, *_anatomy_diagnostics(rejected)),
                base_margins,
                (),
                rejected,
            )
        case AcceptedAnatomy() as accepted:
            vascular = realize_vasculature(accepted, outcome.graph)
        case _ as unreachable:
            assert_never(unreachable)
    match vascular:
        case RejectedVasculature() as rejected:
            return _AnalyticEvidence(
                (
                    *base_diagnostics,
                    *_vascular_diagnostics(rejected, state.spec),
                ),
                (
                    *base_margins,
                    *_anatomy_margins(accepted),
                    *_vascular_violation_margins(rejected),
                ),
                _anatomy_repair_events(accepted, state.txn),
                rejected,
            )
        case AcceptedVasculature() as realized:
            return _AnalyticEvidence(
                base_diagnostics,
                (
                    *base_margins,
                    *_anatomy_margins(accepted),
                    *_vascular_margins(realized),
                ),
                _anatomy_repair_events(accepted, state.txn),
                realized,
            )
        case _ as unreachable:
            assert_never(unreachable)


def _active_margins(
    margins: tuple[Margin, ...],
    rejected: bool,
) -> tuple[Margin, ...]:
    violated = tuple(
        margin
        for margin in margins
        if margin.disposition is MarginDisposition.VIOLATED
    )
    candidates = (
        violated
        if violated
        else ()
        if rejected
        else tuple(
            margin
            for margin in margins
            if margin.disposition is MarginDisposition.SATISFIED
        )
    )
    minimum = min(
        (margin.normalized_margin for margin in candidates),
        default=None,
    )
    return tuple(
        replace(
            margin,
            active=minimum is not None
            and math.isclose(
                margin.normalized_margin,
                minimum,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ),
        )
        for margin in margins
    )


def _glue_evidence(
    diagnostics: tuple[Diagnostic, ...],
    margins: tuple[Margin, ...],
    journal_events: tuple[RepairJournalEvent, ...] = (),
) -> AuthoringEvidence:
    rejected = not all(map(diagnostic_expected, diagnostics))
    active = _active_margins(margins, rejected)
    return AuthoringEvidence(diagnostics, active, journal_events)


@dataclass(frozen=True)
class AuthoringEvaluation:
    evidence: AuthoringEvidence
    vasculature: (
        AcceptedVasculature
        | RejectedVasculature
        | RejectedAnatomy
        | None
    )
    assembly: AcceptedAssembly | RejectedAssembly | None


def _evaluate_authoring_state(state: SessionState) -> AuthoringEvaluation:
    if state.spec.get("dialect") != BODY_DIALECT:
        return AuthoringEvaluation(
            _glue_evidence(
                (
                    _diagnostic(
                        "document.unsupported_dialect",
                        "meta@dialect",
                        "session authoring document uses the body dialect",
                        BODY_DIALECT,
                        state.spec.get("dialect"),
                        {"dialect": state.spec.get("dialect")},
                        "document",
                    ),
                ),
                (),
            ),
            None,
            None,
        )
    match state.outcome:
        case CompileObstructed() as obstructed:
            return AuthoringEvaluation(
                _glue_evidence(_body_compile_diagnostics(obstructed), ()),
                None,
                None,
            )
        case Compiled() as compiled:
            analytic = _compiled_evidence(state, compiled)
        case _ as unreachable:
            assert_never(unreachable)
    if not analytic.accepted:
        return AuthoringEvaluation(
            _glue_evidence(
                analytic.diagnostics,
                analytic.margins,
                analytic.journal_events,
            ),
            analytic.vasculature,
            None,
        )
    source = LoadedSpec(state.spec_dir / "session.json", state.spec)
    try:
        assembly = compile_assembly(
            assembly_document(source),
            state.spec_dir,
        )
    except Exception as failure:
        return AuthoringEvaluation(
            _glue_evidence(
                (
                    *analytic.diagnostics,
                    engine_fault_diagnostic(
                        project_engine_fault(ASSEMBLY_COMPILE_SEAM, failure)
                    ),
                ),
                analytic.margins,
                analytic.journal_events,
            ),
            analytic.vasculature,
            None,
        )
    match assembly:
        case RejectedAssembly() as rejected:
            return AuthoringEvaluation(
                _glue_evidence(
                    (*analytic.diagnostics, *_assembly_diagnostics(rejected)),
                    analytic.margins,
                    analytic.journal_events,
                ),
                analytic.vasculature,
                rejected,
            )
        case AcceptedAssembly() as accepted:
            return AuthoringEvaluation(
                _glue_evidence(
                    analytic.diagnostics,
                    (*analytic.margins, *_assembly_margins(accepted)),
                    analytic.journal_events,
                ),
                analytic.vasculature,
                accepted,
            )
        case _ as unreachable:
            assert_never(unreachable)


def evaluate_authoring_surfaces(state: SessionState) -> AuthoringEvaluation:
    try:
        return _evaluate_authoring_state(state)
    except Exception as failure:
        return AuthoringEvaluation(
            _glue_evidence(
                (
                    engine_fault_diagnostic(
                        project_engine_fault(SESSION_EVIDENCE_SEAM, failure)
                    ),
                ),
                (),
            ),
            None,
            None,
        )


def evaluate_authoring_state(state: SessionState) -> AuthoringEvidence:
    return evaluate_authoring_surfaces(state).evidence


def _margin_deltas(
    before: tuple[Margin, ...],
    after: tuple[Margin, ...],
) -> tuple[MarginDelta, ...]:
    before_by_id = {margin.constraint: margin for margin in before}
    after_by_id = {margin.constraint: margin for margin in after}
    ordered_ids = tuple(
        dict.fromkeys(
            (
                *(margin.constraint for margin in before),
                *(margin.constraint for margin in after),
            )
        )
    )
    return tuple(
        MarginDelta(
            constraint,
            (
                after_by_id[constraint].address
                if constraint in after_by_id
                else before_by_id[constraint].address
            ),
            (
                before_by_id[constraint].margin
                if constraint in before_by_id
                else None
            ),
            (
                after_by_id[constraint].margin
                if constraint in after_by_id
                else None
            ),
            (
                after_by_id[constraint].margin
                - before_by_id[constraint].margin
                if constraint in before_by_id and constraint in after_by_id
                else None
            ),
            (
                before_by_id[constraint].normalized_margin
                if constraint in before_by_id
                else None
            ),
            (
                after_by_id[constraint].normalized_margin
                if constraint in after_by_id
                else None
            ),
            (
                after_by_id[constraint].normalized_margin
                - before_by_id[constraint].normalized_margin
                if constraint in before_by_id and constraint in after_by_id
                else None
            ),
        )
        for constraint in ordered_ids
        if constraint not in before_by_id
        or constraint not in after_by_id
        or not math.isclose(
            before_by_id[constraint].margin,
            after_by_id[constraint].margin,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            before_by_id[constraint].normalized_margin,
            after_by_id[constraint].normalized_margin,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    )


def _verdict_deltas(
    before: AuthoringEvidence,
    after: AuthoringEvidence,
) -> VerdictDeltas:
    before_active = frozenset(
        margin.constraint for margin in before.margins if margin.active
    )
    after_active = frozenset(
        margin.constraint for margin in after.margins if margin.active
    )
    return VerdictDeltas(
        _margin_deltas(before.margins, after.margins),
        tuple(
            margin.constraint
            for margin in after.margins
            if margin.constraint in after_active - before_active
        ),
        tuple(
            margin.constraint
            for margin in before.margins
            if margin.constraint in before_active - after_active
        ),
    )


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _entry_changed_addresses(entry: JournalEntry) -> tuple[str, ...]:
    match entry.applied:
        case ops.AddOp() | ops.AddAtOp() | ops.RemoveOp() | ops.RenameOp():
            return _unique(
                (
                    entry.applied.addr.source,
                    *tuple(sorted(entry.touched_addresses)),
                )
            )
        case ops.SetOp() | ops.UnsetOp():
            return (entry.applied.addr.source,)
        case _ as unreachable:
            assert_never(unreachable)


def _changed_addresses(entries: tuple[JournalEntry, ...]) -> tuple[str, ...]:
    return _unique(
        tuple(
            address
            for entry in entries
            for address in _entry_changed_addresses(entry)
        )
    )


def _request_rejection(
    session: ProtocolSession,
    diagnostics: tuple[Diagnostic, ...],
    base_txn: int,
) -> SessionSubmission:
    evidence = _glue_evidence(
        (*diagnostics, *session._evidence.diagnostics),
        session._evidence.margins,
        session._evidence.journal_events,
    )
    return SessionSubmission(
        session,
        TransactionVerdict(
            copy.deepcopy(evidence),
            VerdictDeltas(),
            (),
            base_txn,
            session.txn,
            (),
            algebra.current_branch(session._model).journal.entries,
        ),
    )


def start_session(
    document: dict[str, object],
    spec_dir: Path,
    *,
    evidence_channel: EvidenceChannel = evaluate_authoring_state,
) -> ProtocolSession:
    state = SessionState(copy.deepcopy(document), Path(spec_dir))
    ingress_diagnostics = _document_diagnostics(document)
    evidence = (
        _glue_evidence(ingress_diagnostics, ())
        if ingress_diagnostics
        else copy.deepcopy(evidence_channel(state.fork()))
    )
    return ProtocolSession(
        algebra.initial(state),
        evidence,
        evidence_channel,
        ingress_diagnostics,
    )


def submit(
    session: ProtocolSession,
    request: object,
) -> SessionSubmission:
    decoded = decode_request(request, session.txn)
    if isinstance(decoded, RequestObstructed):
        return _request_rejection(
            session,
            decoded.diagnostics,
            decoded.base_txn,
        )
    if decoded.base_txn != session.txn:
        return _request_rejection(
            session,
            (
                _request_diagnostic(
                    "session.stale_base_txn",
                    "base_txn",
                    "transaction is based on the current journal tip",
                    session.txn,
                    decoded.base_txn,
                ),
            ),
            decoded.base_txn,
        )
    if session._ingress_diagnostics:
        return _request_rejection(session, (), decoded.base_txn)
    operations = (
        decoded.operations
        if isinstance(decoded, AddressedTransaction)
        else diff_specs(session.document, decoded.document)
    )
    if isinstance(operations, ops.EditObstructed):
        return _request_rejection(
            session,
            tuple(
                _diagnostic(
                    "session.document_diff_obstructed",
                    obstruction.address,
                    "edited document descends into the session edit algebra",
                    "normalizable document",
                    obstruction.reason,
                    {"reason": obstruction.reason},
                    "session",
                )
                for obstruction in operations.obstructions
            ),
            decoded.base_txn,
        )
    pending = algebra.plan_program(session._model, operations)
    if isinstance(pending, ops.EditObstructed):
        diagnostics = tuple(
            _diagnostic(
                "session.edit_obstructed",
                obstruction.address,
                "operation applies against base transaction",
                "applicable edit",
                obstruction.reason,
                {"reason": obstruction.reason},
                "session",
            )
            for obstruction in pending.obstructions
        )
        return _request_rejection(
            session,
            diagnostics,
            decoded.base_txn,
        )
    normalized_pending = (
        replace(pending, next_spec=copy.deepcopy(decoded.document))
        if isinstance(decoded, DocumentTransaction)
        and pending.next_spec == decoded.document
        else pending
    )
    branch = algebra.current_branch(session._model)
    next_txn = (
        normalized_pending.entries[-1].txn
        if normalized_pending.entries
        else branch.state.txn
    )
    authored = branch.state.authored.with_spec(
        normalized_pending.next_spec,
        next_txn,
    )
    outcome = compile_authored(authored)
    model = algebra.accept_transition(
        session._model,
        normalized_pending,
        outcome,
    )
    state = algebra.current_branch(model).state
    evidence = copy.deepcopy(
        session._evidence_channel(state.fork())
    )
    next_session = ProtocolSession(
        model,
        evidence,
        session._evidence_channel,
    )
    return SessionSubmission(
        next_session,
        TransactionVerdict(
            copy.deepcopy(evidence),
            _verdict_deltas(session._evidence, evidence),
            _changed_addresses(normalized_pending.entries),
            decoded.base_txn,
            state.txn,
            operations,
            algebra.current_branch(model).journal.entries,
        ),
    )


__all__ = [
    "AddressedTransaction",
    "AuthoringEvaluation",
    "AuthoringEvidence",
    "Diagnostic",
    "DocumentTransaction",
    "EvidenceChannel",
    "Margin",
    "MarginDisposition",
    "MarginDelta",
    "ProtocolSession",
    "SCHEMA",
    "SessionSubmission",
    "TransactionVerdict",
    "VerdictDeltas",
    "decode_request",
    "diagnostic_expected",
    "witness_expected",
    "engine_fault_diagnostic",
    "evaluate_authoring_state",
    "evaluate_authoring_surfaces",
    "start_session",
    "submit",
]
