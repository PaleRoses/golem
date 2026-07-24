"""Pure authoring-trial decoding, measurement, and corpus gluing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from itertools import chain
from typing import NewType, assert_never


TrialId = NewType("TrialId", str)
CreatureId = NewType("CreatureId", str)
TransactionId = NewType("TransactionId", str)
ObstructionCode = NewType("ObstructionCode", str)


@dataclass(frozen=True)
class AcceptedTransaction:
    transaction_id: TransactionId
    tokens: int | None = None


@dataclass(frozen=True)
class SurfaceAcceptedTransaction:
    transaction_id: TransactionId
    tokens: int | None = None


@dataclass(frozen=True)
class ObstructedTransaction:
    transaction_id: TransactionId
    obstruction_codes: tuple[ObstructionCode, ...]
    tokens: int | None = None


type AuthoringTransaction = (
    AcceptedTransaction | SurfaceAcceptedTransaction | ObstructedTransaction
)


@dataclass(frozen=True)
class DerivedAcceptanceRounds:
    pass


@dataclass(frozen=True)
class ReportedAcceptanceRounds:
    value: int
    detail: str


@dataclass(frozen=True)
class IncompleteAcceptanceRounds:
    detail: str


type AcceptanceRoundEvidence = (
    DerivedAcceptanceRounds
    | ReportedAcceptanceRounds
    | IncompleteAcceptanceRounds
)


@dataclass(frozen=True)
class VerdictSequenceAcceptance:
    artifact_path: str
    round_evidence: ReportedAcceptanceRounds | IncompleteAcceptanceRounds


@dataclass(frozen=True)
class AuthoringTrial:
    trial_id: TrialId
    creature_id: CreatureId
    transactions: tuple[AuthoringTransaction, ...]
    acceptance_round_evidence: AcceptanceRoundEvidence = (
        DerivedAcceptanceRounds()
    )


class MetricUnavailableCode(StrEnum):
    NO_ACCEPTED_TRANSACTION = "no_accepted_transaction"
    INCOMPLETE_TRANSACTION_TRANSCRIPT = "incomplete_transaction_transcript"
    INCOMPLETE_TOKEN_TRANSCRIPT = "incomplete_token_transcript"
    NO_FOLLOWING_TRANSACTION = "no_following_transaction"


@dataclass(frozen=True)
class AvailableCount:
    value: int


@dataclass(frozen=True)
class UnavailableMetric:
    code: MetricUnavailableCode
    detail: str


type CountMetric = AvailableCount | UnavailableMetric


@dataclass(frozen=True)
class AvailableRate:
    numerator: int
    denominator: int


@dataclass(frozen=True)
class UnavailableRate:
    code: MetricUnavailableCode
    detail: str


type RepairRate = AvailableRate | UnavailableRate


@dataclass(frozen=True)
class ObstructionRepairMetric:
    obstruction_code: ObstructionCode
    observed_transactions: int
    followed_transactions: int
    cleared_by_next_transaction: int
    not_cleared_by_next_transaction: int
    unfollowed_transactions: int
    success_rate: RepairRate


@dataclass(frozen=True)
class AuthoringTrialMetrics:
    trial_id: TrialId
    creature_id: CreatureId
    transaction_count: int
    analyzed_transaction_count: int
    rounds_to_acceptance: CountMetric
    obstruction_repair: tuple[ObstructionRepairMetric, ...]
    tokens_per_accepted_creature: CountMetric


@dataclass(frozen=True)
class AuthoringCorpusMetrics:
    trials: tuple[AuthoringTrialMetrics, ...]
    obstruction_repair: tuple[ObstructionRepairMetric, ...]


class DecodeObstructionCode(StrEnum):
    INVALID_JSON = "invalid_json"
    INVALID_TEXT_ENCODING = "invalid_text_encoding"
    EXPECTED_OBJECT = "expected_object"
    UNEXPECTED_FIELD = "unexpected_field"
    MISSING_FIELD = "missing_field"
    INVALID_SCHEMA_VERSION = "invalid_schema_version"
    INVALID_IDENTIFIER = "invalid_identifier"
    INVALID_TRANSACTIONS = "invalid_transactions"
    INVALID_TRANSACTION_STATUS = "invalid_transaction_status"
    INVALID_OBSTRUCTION_CODES = "invalid_obstruction_codes"
    INVALID_TOKEN_COUNT = "invalid_token_count"
    DUPLICATE_TRANSACTION_ID = "duplicate_transaction_id"
    SESSION_TRANSACTION_SEQUENCE = "session_transaction_sequence"
    INCOMPLETE_SESSION_TRANSACTION = "incomplete_session_transaction"
    SESSION_OBSTRUCTION_STATE_MISMATCH = (
        "session_obstruction_state_mismatch"
    )
    MISSING_ACCEPTANCE_AUTHORITY = "missing_acceptance_authority"
    INVALID_VERDICT_SEQUENCE = "invalid_verdict_sequence"
    INVALID_VERDICT_PAYLOAD = "invalid_verdict_payload"
    INVALID_VERDICT_STATUS = "invalid_verdict_status"
    INVALID_VERDICT_DIAGNOSTICS = "invalid_verdict_diagnostics"
    INVALID_ACCEPTANCE_EVIDENCE = "invalid_acceptance_evidence"
    UNSUPPORTED_ARTIFACT_FORMAT = "unsupported_artifact_format"


@dataclass(frozen=True)
class DecodeObstruction:
    address: str
    code: DecodeObstructionCode
    detail: str


@dataclass(frozen=True)
class RejectedTrial:
    obstructions: tuple[DecodeObstruction, ...]


type TrialDecodeResult = AuthoringTrial | RejectedTrial


@dataclass(frozen=True)
class CorpusMetricObstruction:
    trial_id: TrialId
    detail: str


type CorpusMetricResult = AuthoringCorpusMetrics | CorpusMetricObstruction


@dataclass(frozen=True)
class _DecodedField:
    value: object


type _FieldDecodeResult = _DecodedField | RejectedTrial


@dataclass(frozen=True)
class _SessionObservation:
    transaction_id: TransactionId
    new_obstruction_codes: tuple[ObstructionCode, ...]
    cleared_obstruction_codes: tuple[ObstructionCode, ...]
    failure_count: int
    pass_count: int
    unmeasurable_count: int
    tokens: int


@dataclass(frozen=True)
class _SessionDescent:
    active_obstruction_codes: frozenset[ObstructionCode]
    transactions: tuple[AuthoringTransaction, ...]
    obstructions: tuple[DecodeObstruction, ...]


@dataclass(frozen=True)
class _VerdictObservation:
    artifact_path: str
    transaction_id: TransactionId
    status: _VerdictStatus
    transaction_count: int
    obstruction_codes: tuple[ObstructionCode, ...]


class _VerdictStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


_SESSION_TRANSACTION_START = re.compile(
    r"^golem> (?!recent\b)[^\n]+\n"
    r"(?P<transaction>txn (?P<transaction_id>[1-9][0-9]*)  )",
    re.MULTILINE,
)
_SESSION_CONTRACT = re.compile(
    r"^contract (?P<failures>[0-9]+) FAIL / (?P<passes>[0-9]+) pass"
    r"(?: / (?P<unmeasurable>[0-9]+) unmeasurable)?$",
    re.MULTILINE,
)
_SESSION_TOKENS = re.compile(r"^est_tokens: (?P<tokens>[0-9]+)$", re.MULTILINE)
_SESSION_NEW = re.compile(
    r"^new [0-9]+:\n(?P<records>(?:  [^\n]+\n?)*)",
    re.MULTILINE,
)
_SESSION_CLEARED = re.compile(
    r"^cleared [0-9]+: (?P<codes>[^\n]+)$",
    re.MULTILINE,
)


def evaluate_authoring_trial(trial: AuthoringTrial) -> AuthoringTrialMetrics:
    acceptance_index = next(
        (
            index
            for index, transaction in enumerate(trial.transactions)
            if isinstance(transaction, AcceptedTransaction)
        ),
        None,
    )
    analyzed_transactions = (
        trial.transactions
        if acceptance_index is None
        else trial.transactions[: acceptance_index + 1]
    )
    rounds_to_acceptance = _rounds_to_acceptance(
        trial.acceptance_round_evidence,
        acceptance_index,
    )
    return AuthoringTrialMetrics(
        trial_id=trial.trial_id,
        creature_id=trial.creature_id,
        transaction_count=len(trial.transactions),
        analyzed_transaction_count=len(analyzed_transactions),
        rounds_to_acceptance=rounds_to_acceptance,
        obstruction_repair=_obstruction_repair(analyzed_transactions),
        tokens_per_accepted_creature=_tokens_through_acceptance(
            analyzed_transactions,
            acceptance_index,
        ),
    )


def _rounds_to_acceptance(
    evidence: AcceptanceRoundEvidence,
    acceptance_index: int | None,
) -> CountMetric:
    if acceptance_index is None:
        return UnavailableMetric(
            MetricUnavailableCode.NO_ACCEPTED_TRANSACTION,
            "the trial contains no accepted transaction",
        )
    match evidence:
        case DerivedAcceptanceRounds():
            return AvailableCount(acceptance_index + 1)
        case ReportedAcceptanceRounds(value, _detail):
            return AvailableCount(value)
        case IncompleteAcceptanceRounds(detail):
            return UnavailableMetric(
                MetricUnavailableCode.INCOMPLETE_TRANSACTION_TRANSCRIPT,
                detail,
            )
        case _ as unreachable:
            assert_never(unreachable)


def summarize_authoring_trials(
    trials: tuple[AuthoringTrial, ...],
) -> CorpusMetricResult:
    ordered_trials = tuple(sorted(trials, key=lambda trial: trial.trial_id))
    duplicate_trial_id = next(
        (
            left.trial_id
            for left, right in zip(ordered_trials, ordered_trials[1:])
            if left.trial_id == right.trial_id
        ),
        None,
    )
    if duplicate_trial_id is not None:
        return CorpusMetricObstruction(
            duplicate_trial_id,
            "multiple measured receipts claim the same trial identity",
        )
    trial_metrics = tuple(map(evaluate_authoring_trial, ordered_trials))
    return AuthoringCorpusMetrics(
        trials=trial_metrics,
        obstruction_repair=_glue_obstruction_repair(trial_metrics),
    )


def authoring_trial_payload(trial: AuthoringTrial) -> dict[str, object]:
    common = {
        "creature_id": trial.creature_id,
        "transactions": list(map(_transaction_payload, trial.transactions)),
        "trial_id": trial.trial_id,
    }
    extended = not isinstance(
        trial.acceptance_round_evidence,
        DerivedAcceptanceRounds,
    ) or any(
        isinstance(transaction, SurfaceAcceptedTransaction)
        for transaction in trial.transactions
    )
    return (
        {
            **common,
            "acceptance_round_evidence": (
                _acceptance_round_evidence_payload(
                    trial.acceptance_round_evidence
                )
            ),
            "schema_version": 2,
        }
        if extended
        else {**common, "schema_version": 1}
    )


def authoring_trial_metrics_payload(
    metrics: AuthoringTrialMetrics,
) -> dict[str, object]:
    return {
        "analyzed_transaction_count": metrics.analyzed_transaction_count,
        "creature_id": metrics.creature_id,
        "obstruction_repair": list(
            map(_obstruction_repair_payload, metrics.obstruction_repair)
        ),
        "rounds_to_acceptance": _count_metric_payload(
            metrics.rounds_to_acceptance
        ),
        "tokens_per_accepted_creature": _count_metric_payload(
            metrics.tokens_per_accepted_creature
        ),
        "transaction_count": metrics.transaction_count,
        "trial_id": metrics.trial_id,
    }


def authoring_corpus_metrics_payload(
    metrics: AuthoringCorpusMetrics,
) -> dict[str, object]:
    return {
        "obstruction_repair": list(
            map(_obstruction_repair_payload, metrics.obstruction_repair)
        ),
        "trials": list(map(authoring_trial_metrics_payload, metrics.trials)),
    }


def decode_authoring_trial_json(source: str) -> TrialDecodeResult:
    try:
        payload = json.loads(source)
    except (json.JSONDecodeError, TypeError, ValueError) as failure:
        return RejectedTrial(
            (
                DecodeObstruction(
                    "/",
                    DecodeObstructionCode.INVALID_JSON,
                    str(failure),
                ),
            )
        )
    return decode_authoring_trial_payload(payload)


def decode_authoring_trial_payload(payload: object) -> TrialDecodeResult:
    if not isinstance(payload, dict):
        return _rejected(
            "/",
            DecodeObstructionCode.EXPECTED_OBJECT,
            "expected an authoring-trial object",
        )
    schema_value = payload.get("schema_version")
    schema_fields = (
        ("acceptance_round_evidence",)
        if schema_value == 2
        else ()
    )
    expected_fields = frozenset(
        (
            "schema_version",
            "trial_id",
            "creature_id",
            "transactions",
            *schema_fields,
        )
    )
    field_obstructions = (
        *tuple(
            DecodeObstruction(
                f"/{field}",
                DecodeObstructionCode.MISSING_FIELD,
                "required field is absent",
            )
            for field in sorted(expected_fields - payload.keys())
        ),
        *tuple(
            DecodeObstruction(
                f"/{field}",
                DecodeObstructionCode.UNEXPECTED_FIELD,
                "field is not part of the selected authoring-trial schema",
            )
            for field in sorted(payload.keys() - expected_fields)
        ),
    )
    schema_version = _decode_schema_version(payload.get("schema_version"))
    trial_id = _decode_identifier(payload.get("trial_id"), "/trial_id")
    creature_id = _decode_identifier(
        payload.get("creature_id"), "/creature_id"
    )
    transactions = _decode_transactions(
        payload.get("transactions"),
        allow_surface_acceptance=schema_value == 2,
    )
    acceptance_round_evidence = (
        _decode_acceptance_round_evidence(
            payload.get("acceptance_round_evidence")
        )
        if schema_value == 2
        else _DecodedField(DerivedAcceptanceRounds())
    )
    obstructions = (
        *field_obstructions,
        *_field_obstructions(schema_version),
        *_field_obstructions(trial_id),
        *_field_obstructions(creature_id),
        *_field_obstructions(transactions),
        *_field_obstructions(acceptance_round_evidence),
    )
    if obstructions:
        return RejectedTrial(obstructions)
    match (
        schema_version,
        trial_id,
        creature_id,
        transactions,
        acceptance_round_evidence,
    ):
        case (
            _DecodedField(1 | 2),
            _DecodedField(str() as decoded_trial_id),
            _DecodedField(str() as decoded_creature_id),
            _DecodedField(tuple() as decoded_transactions),
            _DecodedField(
                DerivedAcceptanceRounds()
                | ReportedAcceptanceRounds()
                | IncompleteAcceptanceRounds() as decoded_round_evidence
            ),
        ):
            return AuthoringTrial(
                TrialId(decoded_trial_id),
                CreatureId(decoded_creature_id),
                decoded_transactions,
                decoded_round_evidence,
            )
        case _:
            return _rejected(
                "/",
                DecodeObstructionCode.INVALID_TRANSACTIONS,
                "decoded fields did not form an authoring trial",
            )


def decode_authoring_verdict_sequence(
    sources: tuple[tuple[str, str], ...],
    *,
    trial_id: str,
    creature_id: str,
    acceptance: VerdictSequenceAcceptance,
) -> TrialDecodeResult:
    decoded_trial_id = _decode_identifier(trial_id, "/trial_id")
    decoded_creature_id = _decode_identifier(creature_id, "/creature_id")
    source_obstructions = (
        (
            DecodeObstruction(
                "/artifacts",
                DecodeObstructionCode.INVALID_VERDICT_SEQUENCE,
                "expected a non-empty ordered verdict sequence",
            ),
        )
        if not sources
        else ()
    )
    duplicate_paths = tuple(
        path
        for path in sorted(frozenset(path for path, _source in sources))
        if sum(source_path == path for source_path, _source in sources) > 1
    )
    decoded_observations = tuple(
        _decode_verdict_observation(path, source, index)
        for index, (path, source) in enumerate(sources)
    )
    decode_obstructions = tuple(
        chain.from_iterable(
            _field_obstructions(result) for result in decoded_observations
        )
    )
    observations = tuple(
        result.value
        for result in decoded_observations
        if isinstance(result, _DecodedField)
        and isinstance(result.value, _VerdictObservation)
    )
    acceptance_obstructions = _validate_verdict_acceptance(
        observations,
        acceptance,
    )
    obstructions = (
        *source_obstructions,
        *tuple(
            DecodeObstruction(
                "/artifacts",
                DecodeObstructionCode.INVALID_VERDICT_SEQUENCE,
                f"duplicate verdict artifact path {path!r}",
            )
            for path in duplicate_paths
        ),
        *_field_obstructions(decoded_trial_id),
        *_field_obstructions(decoded_creature_id),
        *decode_obstructions,
        *acceptance_obstructions,
    )
    if obstructions:
        return RejectedTrial(obstructions)
    match decoded_trial_id, decoded_creature_id:
        case (
            _DecodedField(str() as accepted_trial_id),
            _DecodedField(str() as accepted_creature_id),
        ):
            return AuthoringTrial(
                TrialId(accepted_trial_id),
                CreatureId(accepted_creature_id),
                tuple(
                    _verdict_transaction(observation, acceptance)
                    for observation in observations
                ),
                acceptance.round_evidence,
            )
        case _:
            return _rejected(
                "/",
                DecodeObstructionCode.INVALID_VERDICT_SEQUENCE,
                "decoded verdict fields did not form an authoring trial",
            )


def decode_session_receipt(
    source: str,
    *,
    trial_id: str,
    creature_id: str,
) -> TrialDecodeResult:
    decoded_trial_id = _decode_identifier(trial_id, "/trial_id")
    decoded_creature_id = _decode_identifier(creature_id, "/creature_id")
    observations = _decode_session_observations(source)
    field_obstructions = (
        *_field_obstructions(decoded_trial_id),
        *_field_obstructions(decoded_creature_id),
        *_field_obstructions(observations),
    )
    if field_obstructions:
        return RejectedTrial(field_obstructions)
    match decoded_trial_id, decoded_creature_id, observations:
        case (
            _DecodedField(str() as accepted_trial_id),
            _DecodedField(str() as accepted_creature_id),
            _DecodedField(tuple() as accepted_observations),
        ):
            descent = reduce(
                _descend_session_observation,
                accepted_observations,
                _SessionDescent(frozenset(), (), ()),
            )
            return (
                RejectedTrial(descent.obstructions)
                if descent.obstructions
                else AuthoringTrial(
                    TrialId(accepted_trial_id),
                    CreatureId(accepted_creature_id),
                    descent.transactions,
                )
            )
        case _:
            return _rejected(
                "/",
                DecodeObstructionCode.INCOMPLETE_SESSION_TRANSACTION,
                "decoded session fields did not form a trial",
            )


def _decode_verdict_observation(
    artifact_path: str,
    source: str,
    index: int,
) -> _FieldDecodeResult:
    address = f"/artifacts/{index}"
    try:
        payload = json.loads(source)
    except (json.JSONDecodeError, TypeError, ValueError) as failure:
        return _rejected(
            address,
            DecodeObstructionCode.INVALID_JSON,
            str(failure),
        )
    if not isinstance(payload, dict):
        return _rejected(
            address,
            DecodeObstructionCode.INVALID_VERDICT_PAYLOAD,
            "expected a verdict object",
        )
    try:
        status = _VerdictStatus(payload.get("status"))
    except (TypeError, ValueError):
        status = None
    transaction_count = payload.get("txn")
    diagnostics = payload.get("diagnostics")
    field_obstructions = (
        *(
            ()
            if status is not None
            else (
                DecodeObstruction(
                    f"{address}/status",
                    DecodeObstructionCode.INVALID_VERDICT_STATUS,
                    "expected 'accepted' or 'rejected'",
                ),
            )
        ),
        *(
            ()
            if type(transaction_count) is int and transaction_count >= 0
            else (
                DecodeObstruction(
                    f"{address}/txn",
                    DecodeObstructionCode.INVALID_VERDICT_PAYLOAD,
                    "expected a non-negative integer transaction count",
                ),
            )
        ),
        *(
            ()
            if isinstance(diagnostics, list)
            else (
                DecodeObstruction(
                    f"{address}/diagnostics",
                    DecodeObstructionCode.INVALID_VERDICT_DIAGNOSTICS,
                    "expected a diagnostic array",
                ),
            )
        ),
    )
    if field_obstructions:
        return RejectedTrial(field_obstructions)
    if not isinstance(diagnostics, list):
        return _rejected(
            f"{address}/diagnostics",
            DecodeObstructionCode.INVALID_VERDICT_DIAGNOSTICS,
            "diagnostic array disappeared during decoding",
        )
    invalid_diagnostics = tuple(
        diagnostic_index
        for diagnostic_index, diagnostic in enumerate(diagnostics)
        if not isinstance(diagnostic, dict)
        or diagnostic.get("severity") not in ("error", "warning")
        or not isinstance(diagnostic.get("code"), str)
        or not diagnostic.get("code")
    )
    if invalid_diagnostics:
        return RejectedTrial(
            tuple(
                DecodeObstruction(
                    f"{address}/diagnostics/{diagnostic_index}",
                    DecodeObstructionCode.INVALID_VERDICT_DIAGNOSTICS,
                    "expected a coded error or warning diagnostic",
                )
                for diagnostic_index in invalid_diagnostics
            )
        )
    error_codes = tuple(
        ObstructionCode(code)
        for code in sorted(
            frozenset(
                diagnostic["code"]
                for diagnostic in diagnostics
                if isinstance(diagnostic, dict)
                and diagnostic.get("severity") == "error"
                and isinstance(diagnostic.get("code"), str)
            )
        )
    )
    consistency_obstruction = (
        _rejected(
            f"{address}/diagnostics",
            DecodeObstructionCode.INVALID_VERDICT_STATUS,
            "accepted verdict retains error diagnostics",
        )
        if status is _VerdictStatus.ACCEPTED and error_codes
        else (
            _rejected(
                f"{address}/diagnostics",
                DecodeObstructionCode.INVALID_VERDICT_STATUS,
                "rejected verdict has no error diagnostic code",
            )
            if status is _VerdictStatus.REJECTED and not error_codes
            else None
        )
    )
    if consistency_obstruction is not None:
        return consistency_obstruction
    if status is None or type(transaction_count) is not int:
        return _rejected(
            address,
            DecodeObstructionCode.INVALID_VERDICT_PAYLOAD,
            "verdict fields disappeared during decoding",
        )
    return _DecodedField(
        _VerdictObservation(
            artifact_path=artifact_path,
            transaction_id=TransactionId(
                f"{artifact_path}#txn-{transaction_count}"
            ),
            status=status,
            transaction_count=transaction_count,
            obstruction_codes=error_codes,
        )
    )


def _validate_verdict_acceptance(
    observations: tuple[_VerdictObservation, ...],
    acceptance: VerdictSequenceAcceptance,
) -> tuple[DecodeObstruction, ...]:
    matching = tuple(
        observation
        for observation in observations
        if observation.artifact_path == acceptance.artifact_path
    )
    selected = matching[0] if len(matching) == 1 else None
    return (
        *(
            ()
            if len(matching) == 1
            else (
                DecodeObstruction(
                    "/acceptance/artifact_path",
                    DecodeObstructionCode.MISSING_ACCEPTANCE_AUTHORITY,
                    "acceptance artifact must identify exactly one verdict",
                ),
            )
        ),
        *(
            ()
            if selected is None or selected.status is _VerdictStatus.ACCEPTED
            else (
                DecodeObstruction(
                    "/acceptance/artifact_path",
                    DecodeObstructionCode.MISSING_ACCEPTANCE_AUTHORITY,
                    "authoritative acceptance artifact is not accepted",
                ),
            )
        ),
        *(
            ()
            if selected is None
            or not observations
            or selected == observations[-1]
            else (
                DecodeObstruction(
                    "/acceptance/artifact_path",
                    DecodeObstructionCode.INVALID_VERDICT_SEQUENCE,
                    "authoritative acceptance verdict must terminate the sequence",
                ),
            )
        ),
        *(
            ()
            if selected is None
            or not isinstance(
                acceptance.round_evidence,
                ReportedAcceptanceRounds,
            )
            or acceptance.round_evidence.value == selected.transaction_count
            else (
                DecodeObstruction(
                    "/acceptance/round_evidence",
                    DecodeObstructionCode.INVALID_ACCEPTANCE_EVIDENCE,
                    "reported acceptance rounds do not match the authoritative verdict txn",
                ),
            )
        ),
        *(
            ()
            if not isinstance(
                acceptance.round_evidence,
                ReportedAcceptanceRounds,
            )
            or acceptance.round_evidence.value > 0
            else (
                DecodeObstruction(
                    "/acceptance/round_evidence",
                    DecodeObstructionCode.INVALID_ACCEPTANCE_EVIDENCE,
                    "reported acceptance rounds must be positive",
                ),
            )
        ),
        *(
            ()
            if not isinstance(
                acceptance.round_evidence,
                IncompleteAcceptanceRounds,
            )
            or bool(acceptance.round_evidence.detail.strip())
            else (
                DecodeObstruction(
                    "/acceptance/round_evidence",
                    DecodeObstructionCode.INVALID_ACCEPTANCE_EVIDENCE,
                    "incomplete acceptance evidence requires a detail",
                ),
            )
        ),
    )


def _verdict_transaction(
    observation: _VerdictObservation,
    acceptance: VerdictSequenceAcceptance,
) -> AuthoringTransaction:
    if observation.artifact_path == acceptance.artifact_path:
        return AcceptedTransaction(observation.transaction_id)
    if observation.status is _VerdictStatus.ACCEPTED:
        return SurfaceAcceptedTransaction(observation.transaction_id)
    return ObstructedTransaction(
        observation.transaction_id,
        observation.obstruction_codes,
    )


def _tokens_through_acceptance(
    analyzed_transactions: tuple[AuthoringTransaction, ...],
    acceptance_index: int | None,
) -> CountMetric:
    if acceptance_index is None:
        return UnavailableMetric(
            MetricUnavailableCode.NO_ACCEPTED_TRANSACTION,
            "tokens per accepted creature require an accepted transaction",
        )
    missing = tuple(
        transaction.transaction_id
        for transaction in analyzed_transactions
        if transaction.tokens is None
    )
    return (
        UnavailableMetric(
            MetricUnavailableCode.INCOMPLETE_TOKEN_TRANSCRIPT,
            "missing token observations for " + ", ".join(missing),
        )
        if missing
        else AvailableCount(
            sum(
                transaction.tokens
                for transaction in analyzed_transactions
                if transaction.tokens is not None
            )
        )
    )


def _obstruction_repair(
    transactions: tuple[AuthoringTransaction, ...],
) -> tuple[ObstructionRepairMetric, ...]:
    followed = tuple(
        chain.from_iterable(
            map(
                lambda pair: tuple(
                    (
                        code,
                        code not in _transaction_obstruction_codes(pair[1]),
                    )
                    for code in _transaction_obstruction_codes(pair[0])
                ),
                zip(transactions, transactions[1:]),
            )
        )
    )
    unfollowed = (
        _transaction_obstruction_codes(transactions[-1])
        if transactions
        else ()
    )
    codes = tuple(
        sorted(
            frozenset(
                (
                    *(code for code, _cleared in followed),
                    *unfollowed,
                )
            )
        )
    )
    return tuple(
        _repair_metric(code, followed, unfollowed) for code in codes
    )


def _repair_metric(
    code: ObstructionCode,
    followed: tuple[tuple[ObstructionCode, bool], ...],
    unfollowed: tuple[ObstructionCode, ...],
) -> ObstructionRepairMetric:
    outcomes = tuple(
        cleared
        for observed_code, cleared in followed
        if observed_code == code
    )
    cleared_count = sum(outcomes)
    unfollowed_count = sum(observed_code == code for observed_code in unfollowed)
    followed_count = len(outcomes)
    return ObstructionRepairMetric(
        obstruction_code=code,
        observed_transactions=followed_count + unfollowed_count,
        followed_transactions=followed_count,
        cleared_by_next_transaction=cleared_count,
        not_cleared_by_next_transaction=followed_count - cleared_count,
        unfollowed_transactions=unfollowed_count,
        success_rate=(
            AvailableRate(cleared_count, followed_count)
            if followed_count > 0
            else UnavailableRate(
                MetricUnavailableCode.NO_FOLLOWING_TRANSACTION,
                "the obstruction was observed only on the final transaction",
            )
        ),
    )


def _glue_obstruction_repair(
    trials: tuple[AuthoringTrialMetrics, ...],
) -> tuple[ObstructionRepairMetric, ...]:
    local_metrics = tuple(
        chain.from_iterable(trial.obstruction_repair for trial in trials)
    )
    codes = tuple(
        sorted(
            frozenset(metric.obstruction_code for metric in local_metrics)
        )
    )
    return tuple(
        _glue_obstruction_code(code, local_metrics) for code in codes
    )


def _glue_obstruction_code(
    code: ObstructionCode,
    metrics: tuple[ObstructionRepairMetric, ...],
) -> ObstructionRepairMetric:
    matching = tuple(
        metric for metric in metrics if metric.obstruction_code == code
    )
    followed = sum(metric.followed_transactions for metric in matching)
    cleared = sum(
        metric.cleared_by_next_transaction for metric in matching
    )
    unfollowed = sum(metric.unfollowed_transactions for metric in matching)
    return ObstructionRepairMetric(
        obstruction_code=code,
        observed_transactions=followed + unfollowed,
        followed_transactions=followed,
        cleared_by_next_transaction=cleared,
        not_cleared_by_next_transaction=followed - cleared,
        unfollowed_transactions=unfollowed,
        success_rate=(
            AvailableRate(cleared, followed)
            if followed > 0
            else UnavailableRate(
                MetricUnavailableCode.NO_FOLLOWING_TRANSACTION,
                "the obstruction has no followed occurrence in the corpus",
            )
        ),
    )


def _transaction_payload(
    transaction: AuthoringTransaction,
) -> dict[str, object]:
    match transaction:
        case AcceptedTransaction(transaction_id, tokens):
            return {
                "obstruction_codes": [],
                "status": "accepted",
                "transaction_id": transaction_id,
                **({"tokens": tokens} if tokens is not None else {}),
            }
        case SurfaceAcceptedTransaction(transaction_id, tokens):
            return {
                "obstruction_codes": [],
                "status": "surface_accepted",
                "transaction_id": transaction_id,
                **({"tokens": tokens} if tokens is not None else {}),
            }
        case ObstructedTransaction(transaction_id, obstruction_codes, tokens):
            return {
                "obstruction_codes": list(obstruction_codes),
                "status": "obstructed",
                "transaction_id": transaction_id,
                **({"tokens": tokens} if tokens is not None else {}),
            }
        case _ as unreachable:
            assert_never(unreachable)


def _acceptance_round_evidence_payload(
    evidence: AcceptanceRoundEvidence,
) -> dict[str, object]:
    match evidence:
        case DerivedAcceptanceRounds():
            return {"kind": "derived_from_observed_sequence"}
        case ReportedAcceptanceRounds(value, detail):
            return {"detail": detail, "kind": "reported", "value": value}
        case IncompleteAcceptanceRounds(detail):
            return {"detail": detail, "kind": "incomplete"}
        case _ as unreachable:
            assert_never(unreachable)


def _count_metric_payload(metric: CountMetric) -> dict[str, object]:
    match metric:
        case AvailableCount(value):
            return {"status": "available", "value": value}
        case UnavailableMetric(code, detail):
            return {
                "code": code.value,
                "detail": detail,
                "status": "unavailable",
            }
        case _ as unreachable:
            assert_never(unreachable)


def _rate_payload(rate: RepairRate) -> dict[str, object]:
    match rate:
        case AvailableRate(numerator, denominator):
            return {
                "denominator": denominator,
                "numerator": numerator,
                "status": "available",
            }
        case UnavailableRate(code, detail):
            return {
                "code": code.value,
                "detail": detail,
                "status": "unavailable",
            }
        case _ as unreachable:
            assert_never(unreachable)


def _obstruction_repair_payload(
    metric: ObstructionRepairMetric,
) -> dict[str, object]:
    return {
        "cleared_by_next_transaction": metric.cleared_by_next_transaction,
        "followed_transactions": metric.followed_transactions,
        "not_cleared_by_next_transaction": (
            metric.not_cleared_by_next_transaction
        ),
        "observed_transactions": metric.observed_transactions,
        "obstruction_code": metric.obstruction_code,
        "success_rate": _rate_payload(metric.success_rate),
        "unfollowed_transactions": metric.unfollowed_transactions,
    }


def _decode_schema_version(value: object) -> _FieldDecodeResult:
    return (
        _DecodedField(value)
        if type(value) is int and value in (1, 2)
        else _rejected(
            "/schema_version",
            DecodeObstructionCode.INVALID_SCHEMA_VERSION,
            "expected schema version 1 or 2",
        )
    )


def _decode_acceptance_round_evidence(value: object) -> _FieldDecodeResult:
    address = "/acceptance_round_evidence"
    if not isinstance(value, dict):
        return _rejected(
            address,
            DecodeObstructionCode.INVALID_ACCEPTANCE_EVIDENCE,
            "expected an acceptance-round evidence object",
        )
    match value.get("kind"):
        case "derived_from_observed_sequence":
            return (
                _DecodedField(DerivedAcceptanceRounds())
                if frozenset(value) == frozenset(("kind",))
                else _rejected(
                    address,
                    DecodeObstructionCode.INVALID_ACCEPTANCE_EVIDENCE,
                    "derived evidence accepts only the kind field",
                )
            )
        case "reported":
            rounds = value.get("value")
            detail = value.get("detail")
            return (
                _DecodedField(ReportedAcceptanceRounds(rounds, detail))
                if frozenset(value)
                == frozenset(("kind", "value", "detail"))
                and type(rounds) is int
                and rounds > 0
                and isinstance(detail, str)
                and bool(detail.strip())
                else _rejected(
                    address,
                    DecodeObstructionCode.INVALID_ACCEPTANCE_EVIDENCE,
                    "reported evidence requires a positive integer value and non-empty detail",
                )
            )
        case "incomplete":
            detail = value.get("detail")
            return (
                _DecodedField(IncompleteAcceptanceRounds(detail))
                if frozenset(value) == frozenset(("kind", "detail"))
                and isinstance(detail, str)
                and bool(detail.strip())
                else _rejected(
                    address,
                    DecodeObstructionCode.INVALID_ACCEPTANCE_EVIDENCE,
                    "incomplete evidence requires a non-empty detail",
                )
            )
        case _:
            return _rejected(
                address,
                DecodeObstructionCode.INVALID_ACCEPTANCE_EVIDENCE,
                "unknown acceptance-round evidence kind",
            )


def _decode_identifier(value: object, address: str) -> _FieldDecodeResult:
    return (
        _DecodedField(value)
        if isinstance(value, str) and value and value == value.strip()
        else _rejected(
            address,
            DecodeObstructionCode.INVALID_IDENTIFIER,
            "expected a non-empty string without surrounding whitespace",
        )
    )


def _decode_transactions(
    value: object,
    *,
    allow_surface_acceptance: bool = False,
) -> _FieldDecodeResult:
    if not isinstance(value, list) or not value:
        return _rejected(
            "/transactions",
            DecodeObstructionCode.INVALID_TRANSACTIONS,
            "expected a non-empty transaction array",
        )
    decoded = tuple(
        _decode_transaction(
            transaction,
            index,
            allow_surface_acceptance=allow_surface_acceptance,
        )
        for index, transaction in enumerate(value)
    )
    obstructions = tuple(
        chain.from_iterable(_field_obstructions(result) for result in decoded)
    )
    transactions = tuple(
        result.value
        for result in decoded
        if isinstance(result, _DecodedField)
        and isinstance(
            result.value,
            (
                AcceptedTransaction,
                SurfaceAcceptedTransaction,
                ObstructedTransaction,
            ),
        )
    )
    duplicate_ids = tuple(
        transaction_id
        for transaction_id in sorted(
            frozenset(
                transaction.transaction_id for transaction in transactions
            )
        )
        if sum(
            transaction.transaction_id == transaction_id
            for transaction in transactions
        )
        > 1
    )
    duplicate_obstructions = tuple(
        DecodeObstruction(
            "/transactions",
            DecodeObstructionCode.DUPLICATE_TRANSACTION_ID,
            f"duplicate transaction id {transaction_id!r}",
        )
        for transaction_id in duplicate_ids
    )
    return (
        RejectedTrial((*obstructions, *duplicate_obstructions))
        if obstructions or duplicate_obstructions
        else _DecodedField(transactions)
    )


def _decode_transaction(
    value: object,
    index: int,
    *,
    allow_surface_acceptance: bool,
) -> _FieldDecodeResult:
    address = f"/transactions/{index}"
    if not isinstance(value, dict):
        return _rejected(
            address,
            DecodeObstructionCode.EXPECTED_OBJECT,
            "expected a transaction object",
        )
    required_fields = frozenset(
        ("transaction_id", "status", "obstruction_codes")
    )
    allowed_fields = required_fields | {"tokens"}
    field_obstructions = (
        *tuple(
            DecodeObstruction(
                f"{address}/{field}",
                DecodeObstructionCode.MISSING_FIELD,
                "required field is absent",
            )
            for field in sorted(required_fields - value.keys())
        ),
        *tuple(
            DecodeObstruction(
                f"{address}/{field}",
                DecodeObstructionCode.UNEXPECTED_FIELD,
                "field is not part of transaction schema version 1",
            )
            for field in sorted(value.keys() - allowed_fields)
        ),
    )
    transaction_id = _decode_identifier(
        value.get("transaction_id"), f"{address}/transaction_id"
    )
    status = _decode_transaction_status(
        value.get("status"),
        address,
        allow_surface_acceptance=allow_surface_acceptance,
    )
    obstruction_codes = _decode_obstruction_codes(
        value.get("obstruction_codes"), address
    )
    tokens = _decode_tokens(value.get("tokens"), address, "tokens" in value)
    obstructions = (
        *field_obstructions,
        *_field_obstructions(transaction_id),
        *_field_obstructions(status),
        *_field_obstructions(obstruction_codes),
        *_field_obstructions(tokens),
    )
    if obstructions:
        return RejectedTrial(obstructions)
    match transaction_id, status, obstruction_codes, tokens:
        case (
            _DecodedField(str() as decoded_transaction_id),
            _DecodedField(str() as decoded_status),
            _DecodedField(tuple() as decoded_codes),
            _DecodedField(int() | None as decoded_tokens),
        ):
            if decoded_status in ("accepted", "surface_accepted") and decoded_codes:
                return _rejected(
                    f"{address}/obstruction_codes",
                    DecodeObstructionCode.INVALID_OBSTRUCTION_CODES,
                    "accepted surface results cannot retain obstructions",
                )
            if decoded_status == "obstructed" and not decoded_codes:
                return _rejected(
                    f"{address}/obstruction_codes",
                    DecodeObstructionCode.INVALID_OBSTRUCTION_CODES,
                    "obstructed transactions require at least one code",
                )
            match decoded_status:
                case "accepted":
                    transaction: AuthoringTransaction = AcceptedTransaction(
                        TransactionId(decoded_transaction_id), decoded_tokens
                    )
                case "surface_accepted":
                    transaction = SurfaceAcceptedTransaction(
                        TransactionId(decoded_transaction_id), decoded_tokens
                    )
                case "obstructed":
                    transaction = ObstructedTransaction(
                        TransactionId(decoded_transaction_id),
                        decoded_codes,
                        decoded_tokens,
                    )
                case _ as unreachable:
                    assert_never(unreachable)
            return _DecodedField(transaction)
        case _:
            return _rejected(
                address,
                DecodeObstructionCode.INVALID_TRANSACTIONS,
                "decoded fields did not form a transaction",
            )


def _decode_transaction_status(
    value: object,
    address: str,
    *,
    allow_surface_acceptance: bool,
) -> _FieldDecodeResult:
    accepted_statuses = (
        ("accepted", "obstructed", "surface_accepted")
        if allow_surface_acceptance
        else ("accepted", "obstructed")
    )
    return (
        _DecodedField(value)
        if value in accepted_statuses
        else _rejected(
            f"{address}/status",
            DecodeObstructionCode.INVALID_TRANSACTION_STATUS,
            "expected one of " + ", ".join(map(repr, accepted_statuses)),
        )
    )


def _decode_obstruction_codes(
    value: object, address: str
) -> _FieldDecodeResult:
    if not isinstance(value, list):
        return _rejected(
            f"{address}/obstruction_codes",
            DecodeObstructionCode.INVALID_OBSTRUCTION_CODES,
            "expected an array of obstruction-code strings",
        )
    invalid_indices = tuple(
        index
        for index, code in enumerate(value)
        if not isinstance(code, str) or not code or code != code.strip()
    )
    duplicates = tuple(
        code
        for code in sorted(
            frozenset(code for code in value if isinstance(code, str))
        )
        if value.count(code) > 1
    )
    obstructions = (
        *tuple(
            DecodeObstruction(
                f"{address}/obstruction_codes/{index}",
                DecodeObstructionCode.INVALID_OBSTRUCTION_CODES,
                "expected a non-empty code without surrounding whitespace",
            )
            for index in invalid_indices
        ),
        *tuple(
            DecodeObstruction(
                f"{address}/obstruction_codes",
                DecodeObstructionCode.INVALID_OBSTRUCTION_CODES,
                f"duplicate obstruction code {code!r}",
            )
            for code in duplicates
        ),
    )
    return (
        RejectedTrial(obstructions)
        if obstructions
        else _DecodedField(
            tuple(ObstructionCode(code) for code in value)
        )
    )


def _decode_tokens(
    value: object, address: str, present: bool
) -> _FieldDecodeResult:
    return (
        _DecodedField(None)
        if not present
        else (
            _DecodedField(value)
            if type(value) is int and value >= 0
            else _rejected(
                f"{address}/tokens",
                DecodeObstructionCode.INVALID_TOKEN_COUNT,
                "expected a non-negative integer",
            )
        )
    )


def _decode_session_observations(source: str) -> _FieldDecodeResult:
    starts = tuple(_SESSION_TRANSACTION_START.finditer(source))
    if not starts:
        return _rejected(
            "/transactions",
            DecodeObstructionCode.INCOMPLETE_SESSION_TRANSACTION,
            "no session transactions were found",
        )
    blocks = tuple(
        source[
            start.start() : end.start() if end is not None else len(source)
        ]
        for start, end in zip(starts, (*starts[1:], None))
    )
    decoded = tuple(
        _decode_session_observation(block, index)
        for index, block in enumerate(blocks)
    )
    obstructions = tuple(
        chain.from_iterable(_field_obstructions(result) for result in decoded)
    )
    observations = tuple(
        result.value
        for result in decoded
        if isinstance(result, _DecodedField)
        and isinstance(result.value, _SessionObservation)
    )
    sequence = tuple(
        int(observation.transaction_id) for observation in observations
    )
    expected_sequence = tuple(range(1, len(observations) + 1))
    sequence_obstructions = (
        (
            DecodeObstruction(
                "/transactions",
                DecodeObstructionCode.SESSION_TRANSACTION_SEQUENCE,
                f"expected transaction ids {expected_sequence}, observed {sequence}",
            ),
        )
        if not obstructions and sequence != expected_sequence
        else ()
    )
    return (
        RejectedTrial((*obstructions, *sequence_obstructions))
        if obstructions or sequence_obstructions
        else _DecodedField(observations)
    )


def _decode_session_observation(
    block: str, index: int
) -> _FieldDecodeResult:
    address = f"/transactions/{index}"
    transaction_match = _SESSION_TRANSACTION_START.search(block)
    contract_match = _SESSION_CONTRACT.search(block)
    tokens_match = _SESSION_TOKENS.search(block)
    missing = tuple(
        name
        for name, match in (
            ("transaction", transaction_match),
            ("contract", contract_match),
            ("est_tokens", tokens_match),
        )
        if match is None
    )
    if missing:
        return _rejected(
            address,
            DecodeObstructionCode.INCOMPLETE_SESSION_TRANSACTION,
            "missing " + ", ".join(missing),
        )
    new_codes = tuple(
        ObstructionCode(line.strip().split(maxsplit=1)[0])
        for match in _SESSION_NEW.finditer(block)
        for line in match.group("records").splitlines()
        if line.strip()
    )
    cleared_codes = tuple(
        ObstructionCode(code.strip())
        for match in _SESSION_CLEARED.finditer(block)
        for code in match.group("codes").split(",")
        if code.strip()
    )
    duplicate_codes = tuple(
        code
        for code in sorted(frozenset((*new_codes, *cleared_codes)))
        if new_codes.count(code) > 1 or cleared_codes.count(code) > 1
    )
    overlapping_codes = tuple(
        sorted(frozenset(new_codes) & frozenset(cleared_codes))
    )
    code_obstructions = (
        *tuple(
            DecodeObstruction(
                address,
                DecodeObstructionCode.SESSION_OBSTRUCTION_STATE_MISMATCH,
                f"duplicate obstruction delta {code!r}",
            )
            for code in duplicate_codes
        ),
        *tuple(
            DecodeObstruction(
                address,
                DecodeObstructionCode.SESSION_OBSTRUCTION_STATE_MISMATCH,
                f"obstruction {code!r} is both new and cleared",
            )
            for code in overlapping_codes
        ),
    )
    if code_obstructions:
        return RejectedTrial(code_obstructions)
    if transaction_match is None or contract_match is None or tokens_match is None:
        return _rejected(
            address,
            DecodeObstructionCode.INCOMPLETE_SESSION_TRANSACTION,
            "required session records disappeared during decoding",
        )
    return _DecodedField(
        _SessionObservation(
            transaction_id=TransactionId(
                transaction_match.group("transaction_id")
            ),
            new_obstruction_codes=new_codes,
            cleared_obstruction_codes=cleared_codes,
            failure_count=int(contract_match.group("failures")),
            pass_count=int(contract_match.group("passes")),
            unmeasurable_count=int(contract_match.group("unmeasurable") or 0),
            tokens=int(tokens_match.group("tokens")),
        )
    )


def _descend_session_observation(
    descent: _SessionDescent,
    observation: _SessionObservation,
) -> _SessionDescent:
    unknown_clears = frozenset(observation.cleared_obstruction_codes) - (
        descent.active_obstruction_codes
    )
    active = (
        descent.active_obstruction_codes
        - frozenset(observation.cleared_obstruction_codes)
    ) | frozenset(observation.new_obstruction_codes)
    blocking_count = observation.failure_count + observation.unmeasurable_count
    state_obstructions = (
        *tuple(
            DecodeObstruction(
                f"/transactions/{observation.transaction_id}",
                DecodeObstructionCode.SESSION_OBSTRUCTION_STATE_MISMATCH,
                f"cleared unknown obstruction {code!r}",
            )
            for code in sorted(unknown_clears)
        ),
        *(
            (
                DecodeObstruction(
                    f"/transactions/{observation.transaction_id}",
                    DecodeObstructionCode.SESSION_OBSTRUCTION_STATE_MISMATCH,
                    f"contract reports {blocking_count} blockers but reconstructed {len(active)} codes",
                ),
            )
            if blocking_count != len(active)
            else ()
        ),
        *(
            (
                DecodeObstruction(
                    f"/transactions/{observation.transaction_id}",
                    DecodeObstructionCode.MISSING_ACCEPTANCE_AUTHORITY,
                    "a zero-blocker session receipt requires at least one passing contract clause",
                ),
            )
            if blocking_count == 0 and observation.pass_count == 0
            else ()
        ),
    )
    transaction = (
        AcceptedTransaction(observation.transaction_id, observation.tokens)
        if blocking_count == 0 and observation.pass_count > 0
        else ObstructedTransaction(
            observation.transaction_id,
            tuple(sorted(active)),
            observation.tokens,
        )
    )
    return _SessionDescent(
        active_obstruction_codes=active,
        transactions=(*descent.transactions, transaction),
        obstructions=(*descent.obstructions, *state_obstructions),
    )


def _transaction_obstruction_codes(
    transaction: AuthoringTransaction,
) -> tuple[ObstructionCode, ...]:
    match transaction:
        case AcceptedTransaction() | SurfaceAcceptedTransaction():
            return ()
        case ObstructedTransaction(obstruction_codes=obstruction_codes):
            return obstruction_codes
        case _ as unreachable:
            assert_never(unreachable)


def _field_obstructions(
    result: _FieldDecodeResult,
) -> tuple[DecodeObstruction, ...]:
    return result.obstructions if isinstance(result, RejectedTrial) else ()


def _rejected(
    address: str,
    code: DecodeObstructionCode,
    detail: str,
) -> RejectedTrial:
    return RejectedTrial((DecodeObstruction(address, code, detail),))
