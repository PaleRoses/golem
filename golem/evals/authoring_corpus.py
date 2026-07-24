"""Content-addressed persistence for authoring-surface evaluation receipts."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import assert_never

from golem.evals.authoring_surface import (
    AuthoringCorpusMetrics,
    AuthoringTrial,
    AuthoringTrialMetrics,
    CorpusMetricObstruction,
    DecodeObstruction,
    DecodeObstructionCode,
    RejectedTrial,
    TrialDecodeResult,
    VerdictSequenceAcceptance,
    authoring_corpus_metrics_payload,
    authoring_trial_metrics_payload,
    authoring_trial_payload,
    decode_authoring_trial_json,
    decode_authoring_trial_payload,
    decode_authoring_verdict_sequence,
    decode_session_receipt,
    evaluate_authoring_trial,
    summarize_authoring_trials,
)


class SourceFormat(StrEnum):
    AUTHORING_TRIAL_JSON_V1 = "authoring_trial_json_v1"
    AUTHORING_VERDICT_SEQUENCE_V1 = "authoring_verdict_sequence_v1"
    SESSION_RECEIPT_V1 = "session_receipt_v1"
    LEGACY_ARTIFACT_BUNDLE = "legacy_artifact_bundle"


@dataclass(frozen=True)
class SourceArtifact:
    path: str
    sha256: str


@dataclass(frozen=True)
class MeasuredReceiptResult:
    trial: AuthoringTrial
    metrics: AuthoringTrialMetrics


@dataclass(frozen=True)
class UnparseableReceiptResult:
    obstructions: tuple[DecodeObstruction, ...]


type ReceiptResult = MeasuredReceiptResult | UnparseableReceiptResult


@dataclass(frozen=True)
class CorpusReceipt:
    source_format: SourceFormat
    source_artifacts: tuple[SourceArtifact, ...]
    result: ReceiptResult


class CorpusIOObstructionCode(StrEnum):
    SOURCE_READ_FAILURE = "source_read_failure"
    RECEIPT_DIRECTORY_FAILURE = "receipt_directory_failure"
    RECEIPT_WRITE_FAILURE = "receipt_write_failure"
    RECEIPT_COLLISION = "receipt_collision"
    CORPUS_READ_FAILURE = "corpus_read_failure"
    INVALID_RECEIPT_ADDRESS = "invalid_receipt_address"
    INVALID_RECEIPT_PAYLOAD = "invalid_receipt_payload"
    CONFLICTING_TRIAL_IDENTITY = "conflicting_trial_identity"
    INVALID_COMMAND = "invalid_command"


@dataclass(frozen=True)
class CorpusIOObstruction:
    address: str
    code: CorpusIOObstructionCode
    detail: str


@dataclass(frozen=True)
class StoredCorpusReceipt:
    receipt_sha256: str
    path: Path
    created: bool


type StoreReceiptResult = StoredCorpusReceipt | CorpusIOObstruction


@dataclass(frozen=True)
class LoadedSourceArtifact:
    artifact: SourceArtifact
    content: bytes


type SourceArtifactResult = LoadedSourceArtifact | CorpusIOObstruction


@dataclass(frozen=True)
class RejectedSourceBundle:
    obstructions: tuple[CorpusIOObstruction, ...]


type SourceBundleResult = (
    tuple[LoadedSourceArtifact, ...] | RejectedSourceBundle
)


@dataclass(frozen=True)
class RejectedCorpus:
    obstructions: tuple[CorpusIOObstruction, ...]


type CorpusLoadResult = tuple[CorpusReceipt, ...] | RejectedCorpus


@dataclass(frozen=True)
class UnparseableCorpusSource:
    source_format: SourceFormat
    source_artifacts: tuple[SourceArtifact, ...]
    obstructions: tuple[DecodeObstruction, ...]


@dataclass(frozen=True)
class AuthoringCorpusSummary:
    metrics: AuthoringCorpusMetrics
    unparseable_sources: tuple[UnparseableCorpusSource, ...]


type CorpusSummaryResult = AuthoringCorpusSummary | RejectedCorpus


@dataclass(frozen=True)
class _RecordJsonCommand:
    source_path: Path
    corpus_directory: Path


@dataclass(frozen=True)
class _RecordSessionCommand:
    source_path: Path
    trial_id: str
    creature_id: str
    corpus_directory: Path


@dataclass(frozen=True)
class _SummarizeCommand:
    corpus_directory: Path


type _Command = _RecordJsonCommand | _RecordSessionCommand | _SummarizeCommand
type _CommandDecodeResult = _Command | CorpusIOObstruction


def build_corpus_receipt(
    source_format: SourceFormat,
    source_artifacts: tuple[SourceArtifact, ...],
    decoded: TrialDecodeResult,
) -> CorpusReceipt:
    return CorpusReceipt(
        source_format=source_format,
        source_artifacts=tuple(
            sorted(source_artifacts, key=lambda artifact: artifact.path)
        ),
        result=(
            UnparseableReceiptResult(decoded.obstructions)
            if isinstance(decoded, RejectedTrial)
            else MeasuredReceiptResult(
                decoded,
                evaluate_authoring_trial(decoded),
            )
        ),
    )


def build_unparseable_corpus_receipt(
    source_artifacts: tuple[SourceArtifact, ...],
    obstructions: tuple[DecodeObstruction, ...],
) -> CorpusReceipt:
    return CorpusReceipt(
        source_format=SourceFormat.LEGACY_ARTIFACT_BUNDLE,
        source_artifacts=tuple(
            sorted(source_artifacts, key=lambda artifact: artifact.path)
        ),
        result=UnparseableReceiptResult(obstructions),
    )


def corpus_receipt_payload(receipt: CorpusReceipt) -> dict[str, object]:
    return {
        "result": _receipt_result_payload(receipt.result),
        "schema_version": 1,
        "source": {
            "artifacts": list(
                map(_source_artifact_payload, receipt.source_artifacts)
            ),
            "format": receipt.source_format.value,
        },
    }


def corpus_receipt_bytes(receipt: CorpusReceipt) -> bytes:
    return _stable_json_bytes(corpus_receipt_payload(receipt))


def corpus_receipt_sha256(receipt: CorpusReceipt) -> str:
    return sha256(corpus_receipt_bytes(receipt)).hexdigest()


def append_corpus_receipt(
    corpus_directory: Path,
    receipt: CorpusReceipt,
) -> StoreReceiptResult:
    receipt_bytes = corpus_receipt_bytes(receipt)
    receipt_digest = sha256(receipt_bytes).hexdigest()
    receipt_path = corpus_directory / f"{receipt_digest}.json"
    try:
        corpus_directory.mkdir(parents=True, exist_ok=True)
    except OSError as failure:
        return CorpusIOObstruction(
            str(corpus_directory),
            CorpusIOObstructionCode.RECEIPT_DIRECTORY_FAILURE,
            str(failure),
        )
    try:
        with receipt_path.open("xb") as receipt_file:
            receipt_file.write(receipt_bytes)
        return StoredCorpusReceipt(receipt_digest, receipt_path, True)
    except FileExistsError:
        try:
            existing = receipt_path.read_bytes()
        except OSError as failure:
            return CorpusIOObstruction(
                str(receipt_path),
                CorpusIOObstructionCode.CORPUS_READ_FAILURE,
                str(failure),
            )
        return (
            StoredCorpusReceipt(receipt_digest, receipt_path, False)
            if existing == receipt_bytes
            else CorpusIOObstruction(
                str(receipt_path),
                CorpusIOObstructionCode.RECEIPT_COLLISION,
                "content-addressed path contains different bytes",
            )
        )
    except OSError as failure:
        return CorpusIOObstruction(
            str(receipt_path),
            CorpusIOObstructionCode.RECEIPT_WRITE_FAILURE,
            str(failure),
        )


def record_authoring_trial_json(
    source_path: Path,
    corpus_directory: Path,
    *,
    source_label: str | None = None,
) -> StoreReceiptResult:
    loaded = read_source_artifact(source_path, source_label=source_label)
    if isinstance(loaded, CorpusIOObstruction):
        return loaded
    decoded = _decode_text_source(
        loaded.content,
        decode_authoring_trial_json,
    )
    return append_corpus_receipt(
        corpus_directory,
        build_corpus_receipt(
            SourceFormat.AUTHORING_TRIAL_JSON_V1,
            (loaded.artifact,),
            decoded,
        ),
    )


def record_session_receipt(
    source_path: Path,
    corpus_directory: Path,
    *,
    trial_id: str,
    creature_id: str,
    source_label: str | None = None,
) -> StoreReceiptResult:
    loaded = read_source_artifact(source_path, source_label=source_label)
    if isinstance(loaded, CorpusIOObstruction):
        return loaded
    try:
        source = loaded.content.decode("utf-8")
    except UnicodeDecodeError as failure:
        decoded: TrialDecodeResult = RejectedTrial(
            (
                DecodeObstruction(
                    "/",
                    DecodeObstructionCode.INVALID_TEXT_ENCODING,
                    str(failure),
                ),
            )
        )
    else:
        decoded = decode_session_receipt(
            source,
            trial_id=trial_id,
            creature_id=creature_id,
        )
    return append_corpus_receipt(
        corpus_directory,
        build_corpus_receipt(
            SourceFormat.SESSION_RECEIPT_V1,
            (loaded.artifact,),
            decoded,
        ),
    )


def record_authoring_verdict_sequence(
    verdict_sources: tuple[tuple[Path, str], ...],
    corpus_directory: Path,
    *,
    context_sources: tuple[tuple[Path, str], ...] = (),
    trial_id: str,
    creature_id: str,
    acceptance: VerdictSequenceAcceptance,
) -> StoreReceiptResult:
    if not verdict_sources:
        return CorpusIOObstruction(
            "/source/verdicts",
            CorpusIOObstructionCode.SOURCE_READ_FAILURE,
            "verdict sources must be non-empty",
        )
    loaded = read_source_artifacts((*verdict_sources, *context_sources))
    if isinstance(loaded, RejectedSourceBundle):
        return loaded.obstructions[0]
    verdict_artifacts = loaded[: len(verdict_sources)]
    decoded_sources = tuple(map(_decode_verdict_source, verdict_artifacts))
    text_obstructions = tuple(
        result
        for result in decoded_sources
        if isinstance(result, DecodeObstruction)
    )
    decoded = (
        RejectedTrial(text_obstructions)
        if text_obstructions
        else decode_authoring_verdict_sequence(
            tuple(
                result
                for result in decoded_sources
                if isinstance(result, tuple)
            ),
            trial_id=trial_id,
            creature_id=creature_id,
            acceptance=acceptance,
        )
    )
    return append_corpus_receipt(
        corpus_directory,
        build_corpus_receipt(
            SourceFormat.AUTHORING_VERDICT_SEQUENCE_V1,
            tuple(result.artifact for result in loaded),
            decoded,
        ),
    )


def record_unparseable_artifact_bundle(
    source_paths: tuple[Path, ...],
    corpus_directory: Path,
    *,
    source_labels: tuple[str, ...] | None = None,
    obstruction: DecodeObstruction,
) -> StoreReceiptResult:
    labels = (
        source_labels
        if source_labels is not None
        else tuple(map(str, source_paths))
    )
    if len(source_paths) != len(labels) or not source_paths:
        return CorpusIOObstruction(
            "/source/artifacts",
            CorpusIOObstructionCode.SOURCE_READ_FAILURE,
            "artifact paths and labels must be non-empty and have equal length",
        )
    loaded = read_source_artifacts(tuple(zip(source_paths, labels)))
    if isinstance(loaded, RejectedSourceBundle):
        return loaded.obstructions[0]
    return append_corpus_receipt(
        corpus_directory,
        build_unparseable_corpus_receipt(
            tuple(artifact.artifact for artifact in loaded),
            (obstruction,),
        ),
    )


def read_source_artifact(
    source_path: Path,
    *,
    source_label: str | None = None,
) -> SourceArtifactResult:
    try:
        content = source_path.read_bytes()
    except OSError as failure:
        return CorpusIOObstruction(
            str(source_path),
            CorpusIOObstructionCode.SOURCE_READ_FAILURE,
            str(failure),
        )
    return LoadedSourceArtifact(
        SourceArtifact(
            path=source_label if source_label is not None else str(source_path),
            sha256=sha256(content).hexdigest(),
        ),
        content,
    )


def read_source_artifacts(
    sources: tuple[tuple[Path, str], ...],
) -> SourceBundleResult:
    loaded = tuple(
        read_source_artifact(path, source_label=label)
        for path, label in sources
    )
    obstructions = tuple(
        result for result in loaded if isinstance(result, CorpusIOObstruction)
    )
    return (
        RejectedSourceBundle(obstructions)
        if obstructions
        else tuple(
            result
            for result in loaded
            if isinstance(result, LoadedSourceArtifact)
        )
    )


def load_corpus_receipt(path: Path) -> CorpusReceipt | CorpusIOObstruction:
    try:
        content = path.read_bytes()
    except OSError as failure:
        return CorpusIOObstruction(
            str(path),
            CorpusIOObstructionCode.CORPUS_READ_FAILURE,
            str(failure),
        )
    content_digest = sha256(content).hexdigest()
    if path.name != f"{content_digest}.json":
        return CorpusIOObstruction(
            str(path),
            CorpusIOObstructionCode.INVALID_RECEIPT_ADDRESS,
            f"expected filename {content_digest}.json",
        )
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError) as failure:
        return CorpusIOObstruction(
            str(path),
            CorpusIOObstructionCode.INVALID_RECEIPT_PAYLOAD,
            str(failure),
        )
    decoded = _decode_corpus_receipt_payload(payload, path)
    return (
        CorpusIOObstruction(
            str(path),
            CorpusIOObstructionCode.INVALID_RECEIPT_PAYLOAD,
            "receipt bytes are not canonical stable JSON",
        )
        if isinstance(decoded, CorpusReceipt)
        and corpus_receipt_bytes(decoded) != content
        else decoded
    )


def load_authoring_corpus(corpus_directory: Path) -> CorpusLoadResult:
    try:
        receipt_paths = tuple(sorted(corpus_directory.glob("*.json")))
    except OSError as failure:
        return RejectedCorpus(
            (
                CorpusIOObstruction(
                    str(corpus_directory),
                    CorpusIOObstructionCode.CORPUS_READ_FAILURE,
                    str(failure),
                ),
            )
        )
    decoded = tuple(map(load_corpus_receipt, receipt_paths))
    obstructions = tuple(
        result for result in decoded if isinstance(result, CorpusIOObstruction)
    )
    return (
        RejectedCorpus(obstructions)
        if obstructions
        else tuple(
            result for result in decoded if isinstance(result, CorpusReceipt)
        )
    )


def summarize_authoring_corpus(
    receipts: tuple[CorpusReceipt, ...],
) -> CorpusSummaryResult:
    measured_receipts = tuple(
        receipt
        for receipt in receipts
        if isinstance(receipt.result, MeasuredReceiptResult)
    )
    metric_result = summarize_authoring_trials(
        tuple(
            receipt.result.trial
            for receipt in measured_receipts
            if isinstance(receipt.result, MeasuredReceiptResult)
        )
    )
    if isinstance(metric_result, CorpusMetricObstruction):
        return RejectedCorpus(
            (
                CorpusIOObstruction(
                    f"/trials/{metric_result.trial_id}",
                    CorpusIOObstructionCode.CONFLICTING_TRIAL_IDENTITY,
                    metric_result.detail,
                ),
            )
        )
    measured_source_keys = frozenset(
        _source_key(receipt) for receipt in measured_receipts
    )
    unparseable = tuple(
        sorted(
            (
                UnparseableCorpusSource(
                    receipt.source_format,
                    receipt.source_artifacts,
                    receipt.result.obstructions,
                )
                for receipt in receipts
                if isinstance(receipt.result, UnparseableReceiptResult)
                and _source_key(receipt) not in measured_source_keys
            ),
            key=lambda source: tuple(
                artifact.path for artifact in source.source_artifacts
            ),
        )
    )
    return AuthoringCorpusSummary(metric_result, unparseable)


def summarize_authoring_corpus_directory(
    corpus_directory: Path,
) -> CorpusSummaryResult:
    loaded = load_authoring_corpus(corpus_directory)
    return (
        loaded
        if isinstance(loaded, RejectedCorpus)
        else summarize_authoring_corpus(loaded)
    )


def authoring_corpus_summary_payload(
    summary: AuthoringCorpusSummary,
) -> dict[str, object]:
    return {
        "metrics": authoring_corpus_metrics_payload(summary.metrics),
        "schema_version": 1,
        "unparseable_sources": list(
            map(_unparseable_source_payload, summary.unparseable_sources)
        ),
    }


def main(arguments: tuple[str, ...] | None = None) -> int:
    decoded = _decode_command(
        tuple(sys.argv[1:]) if arguments is None else arguments
    )
    if isinstance(decoded, CorpusIOObstruction):
        sys.stderr.buffer.write(
            _stable_json_bytes(_corpus_obstruction_payload(decoded))
        )
        return 2
    match decoded:
        case _RecordJsonCommand(source_path, corpus_directory):
            result: StoreReceiptResult | CorpusSummaryResult = (
                record_authoring_trial_json(source_path, corpus_directory)
            )
        case _RecordSessionCommand(
            source_path,
            trial_id,
            creature_id,
            corpus_directory,
        ):
            result = record_session_receipt(
                source_path,
                corpus_directory,
                trial_id=trial_id,
                creature_id=creature_id,
            )
        case _SummarizeCommand(corpus_directory):
            result = summarize_authoring_corpus_directory(corpus_directory)
        case _ as unreachable:
            assert_never(unreachable)
    match result:
        case StoredCorpusReceipt() as stored:
            sys.stdout.buffer.write(
                _stable_json_bytes(_stored_receipt_payload(stored))
            )
            return 0
        case AuthoringCorpusSummary() as summary:
            sys.stdout.buffer.write(
                _stable_json_bytes(authoring_corpus_summary_payload(summary))
            )
            return 0
        case CorpusIOObstruction() as obstruction:
            sys.stderr.buffer.write(
                _stable_json_bytes(_corpus_obstruction_payload(obstruction))
            )
            return 2
        case RejectedCorpus(obstructions):
            sys.stderr.buffer.write(
                _stable_json_bytes(
                    {
                        "obstructions": list(
                            map(_corpus_obstruction_payload, obstructions)
                        ),
                        "status": "obstructed",
                    }
                )
            )
            return 2
        case _ as unreachable:
            assert_never(unreachable)


def _decode_text_source(
    content: bytes,
    decoder: Callable[[str], TrialDecodeResult],
) -> TrialDecodeResult:
    try:
        source = content.decode("utf-8")
    except UnicodeDecodeError as failure:
        return RejectedTrial(
            (
                DecodeObstruction(
                    "/",
                    DecodeObstructionCode.INVALID_TEXT_ENCODING,
                    str(failure),
                ),
            )
        )
    return decoder(source)


def _decode_verdict_source(
    loaded: LoadedSourceArtifact,
) -> tuple[str, str] | DecodeObstruction:
    try:
        source = loaded.content.decode("utf-8")
    except UnicodeDecodeError as failure:
        return DecodeObstruction(
            f"/artifacts/{loaded.artifact.path}",
            DecodeObstructionCode.INVALID_TEXT_ENCODING,
            str(failure),
        )
    return loaded.artifact.path, source


def _decode_command(arguments: tuple[str, ...]) -> _CommandDecodeResult:
    match arguments:
        case ("record-json", source_path, corpus_directory):
            return _RecordJsonCommand(Path(source_path), Path(corpus_directory))
        case (
            "record-session",
            source_path,
            trial_id,
            creature_id,
            corpus_directory,
        ):
            return _RecordSessionCommand(
                Path(source_path),
                trial_id,
                creature_id,
                Path(corpus_directory),
            )
        case ("summary", corpus_directory):
            return _SummarizeCommand(Path(corpus_directory))
        case _:
            return CorpusIOObstruction(
                "/arguments",
                CorpusIOObstructionCode.INVALID_COMMAND,
                "expected 'record-json SOURCE CORPUS', 'record-session SOURCE TRIAL_ID CREATURE_ID CORPUS', or 'summary CORPUS'",
            )


def _decode_corpus_receipt_payload(
    payload: object,
    path: Path,
) -> CorpusReceipt | CorpusIOObstruction:
    if not isinstance(payload, dict) or frozenset(payload) != frozenset(
        ("schema_version", "source", "result")
    ):
        return _invalid_receipt(path, "expected receipt schema version 1 fields")
    if payload.get("schema_version") != 1:
        return _invalid_receipt(path, "expected schema_version 1")
    source = payload.get("source")
    result = payload.get("result")
    if not isinstance(source, dict) or frozenset(source) != frozenset(
        ("format", "artifacts")
    ):
        return _invalid_receipt(path, "invalid source record")
    try:
        source_format = SourceFormat(source.get("format"))
    except (TypeError, ValueError):
        return _invalid_receipt(path, "unknown source format")
    artifacts = _decode_source_artifacts(source.get("artifacts"), path)
    if isinstance(artifacts, CorpusIOObstruction):
        return artifacts
    receipt_result = _decode_receipt_result(result, path)
    return (
        receipt_result
        if isinstance(receipt_result, CorpusIOObstruction)
        else CorpusReceipt(source_format, artifacts, receipt_result)
    )


def _decode_source_artifacts(
    payload: object,
    path: Path,
) -> tuple[SourceArtifact, ...] | CorpusIOObstruction:
    if not isinstance(payload, list) or not payload:
        return _invalid_receipt(path, "source artifacts must be non-empty")
    valid = all(
        isinstance(artifact, dict)
        and frozenset(artifact) == frozenset(("path", "sha256"))
        and isinstance(artifact.get("path"), str)
        and bool(artifact.get("path"))
        and isinstance(artifact.get("sha256"), str)
        and len(artifact.get("sha256")) == 64
        for artifact in payload
    )
    return (
        tuple(
            SourceArtifact(artifact["path"], artifact["sha256"])
            for artifact in payload
        )
        if valid
        else _invalid_receipt(path, "invalid source artifact record")
    )


def _decode_receipt_result(
    payload: object,
    path: Path,
) -> ReceiptResult | CorpusIOObstruction:
    if not isinstance(payload, dict):
        return _invalid_receipt(path, "invalid receipt result")
    match payload.get("status"):
        case "measured":
            if frozenset(payload) != frozenset(
                ("status", "trial", "metrics")
            ):
                return _invalid_receipt(path, "invalid measured result fields")
            decoded = decode_authoring_trial_payload(payload.get("trial"))
            if isinstance(decoded, RejectedTrial):
                return _invalid_receipt(
                    path,
                    "stored normalized trial does not decode: "
                    + "; ".join(
                        obstruction.code.value
                        for obstruction in decoded.obstructions
                    ),
                )
            metrics = evaluate_authoring_trial(decoded)
            return (
                MeasuredReceiptResult(decoded, metrics)
                if payload.get("metrics")
                == authoring_trial_metrics_payload(metrics)
                else _invalid_receipt(
                    path,
                    "stored metrics do not match the normalized trial",
                )
            )
        case "unparseable":
            if frozenset(payload) != frozenset(("status", "obstructions")):
                return _invalid_receipt(
                    path, "invalid unparseable result fields"
                )
            obstructions = _decode_import_obstructions(
                payload.get("obstructions"), path
            )
            return (
                obstructions
                if isinstance(obstructions, CorpusIOObstruction)
                else UnparseableReceiptResult(obstructions)
            )
        case _:
            return _invalid_receipt(path, "unknown receipt result status")


def _decode_import_obstructions(
    payload: object,
    path: Path,
) -> tuple[DecodeObstruction, ...] | CorpusIOObstruction:
    if not isinstance(payload, list) or not payload:
        return _invalid_receipt(path, "import obstructions must be non-empty")
    valid = all(
        isinstance(obstruction, dict)
        and frozenset(obstruction) == frozenset(("address", "code", "detail"))
        and isinstance(obstruction.get("address"), str)
        and isinstance(obstruction.get("detail"), str)
        for obstruction in payload
    )
    if not valid:
        return _invalid_receipt(path, "invalid import obstruction record")
    try:
        return tuple(
            DecodeObstruction(
                obstruction["address"],
                DecodeObstructionCode(obstruction["code"]),
                obstruction["detail"],
            )
            for obstruction in payload
        )
    except (TypeError, ValueError):
        return _invalid_receipt(path, "unknown import obstruction code")


def _receipt_result_payload(result: ReceiptResult) -> dict[str, object]:
    match result:
        case MeasuredReceiptResult(trial, metrics):
            return {
                "metrics": authoring_trial_metrics_payload(metrics),
                "status": "measured",
                "trial": authoring_trial_payload(trial),
            }
        case UnparseableReceiptResult(obstructions):
            return {
                "obstructions": list(
                    map(_decode_obstruction_payload, obstructions)
                ),
                "status": "unparseable",
            }
        case _ as unreachable:
            assert_never(unreachable)


def _source_artifact_payload(artifact: SourceArtifact) -> dict[str, str]:
    return {"path": artifact.path, "sha256": artifact.sha256}


def _decode_obstruction_payload(
    obstruction: DecodeObstruction,
) -> dict[str, str]:
    return {
        "address": obstruction.address,
        "code": obstruction.code.value,
        "detail": obstruction.detail,
    }


def _unparseable_source_payload(
    source: UnparseableCorpusSource,
) -> dict[str, object]:
    return {
        "obstructions": list(
            map(_decode_obstruction_payload, source.obstructions)
        ),
        "source": {
            "artifacts": list(
                map(_source_artifact_payload, source.source_artifacts)
            ),
            "format": source.source_format.value,
        },
    }


def _source_key(receipt: CorpusReceipt) -> tuple[str, ...]:
    return tuple(
        artifact.sha256
        for artifact in sorted(
            receipt.source_artifacts,
            key=lambda artifact: artifact.path,
        )
    )


def _stable_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _invalid_receipt(path: Path, detail: str) -> CorpusIOObstruction:
    return CorpusIOObstruction(
        str(path),
        CorpusIOObstructionCode.INVALID_RECEIPT_PAYLOAD,
        detail,
    )


def _stored_receipt_payload(
    stored: StoredCorpusReceipt,
) -> dict[str, object]:
    return {
        "created": stored.created,
        "path": str(stored.path),
        "receipt_sha256": stored.receipt_sha256,
        "status": "stored",
    }


def _corpus_obstruction_payload(
    obstruction: CorpusIOObstruction,
) -> dict[str, str]:
    return {
        "address": obstruction.address,
        "code": obstruction.code.value,
        "detail": obstruction.detail,
        "status": "obstructed",
    }


if __name__ == "__main__":
    raise SystemExit(main())
