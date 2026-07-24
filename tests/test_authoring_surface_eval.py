from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from golem.evals.authoring_corpus import (
    AuthoringCorpusSummary,
    CorpusIOObstruction,
    CorpusIOObstructionCode,
    CorpusReceipt,
    MeasuredReceiptResult,
    RejectedCorpus,
    SourceArtifact,
    SourceFormat,
    StoredCorpusReceipt,
    UnparseableReceiptResult,
    append_corpus_receipt,
    authoring_corpus_summary_payload,
    build_corpus_receipt,
    build_unparseable_corpus_receipt,
    corpus_receipt_bytes,
    corpus_receipt_sha256,
    load_corpus_receipt,
    load_authoring_corpus,
    main,
    record_authoring_trial_json,
    record_authoring_verdict_sequence,
    record_session_receipt,
    record_unparseable_artifact_bundle,
    summarize_authoring_corpus,
    summarize_authoring_corpus_directory,
)
from golem.evals.authoring_surface import (
    AcceptedTransaction,
    AuthoringCorpusMetrics,
    AuthoringTrial,
    AvailableCount,
    AvailableRate,
    CreatureId,
    DecodeObstruction,
    DecodeObstructionCode,
    IncompleteAcceptanceRounds,
    MetricUnavailableCode,
    ObstructedTransaction,
    ObstructionCode,
    RejectedTrial,
    ReportedAcceptanceRounds,
    TransactionId,
    TrialId,
    SurfaceAcceptedTransaction,
    UnavailableMetric,
    UnavailableRate,
    VerdictSequenceAcceptance,
    authoring_trial_metrics_payload,
    decode_authoring_trial_json,
    decode_authoring_trial_payload,
    decode_authoring_verdict_sequence,
    decode_session_receipt,
    evaluate_authoring_trial,
)
from golem.paths import GOLDEN
from golem.paths import KERNEL_ROOT


SCREE_TRIAL = KERNEL_ROOT / "rehearsal" / "scree-maiden-trial"
VERDIGRIS_TRIAL = KERNEL_ROOT / "rehearsal" / "verdigris-trial"
SCREE_ACCEPTANCE_DETAIL = (
    "FRICTION.md reports approximately ten pre-session check rounds without "
    "an ordered verdict ledger; those rounds and their obstruction transitions "
    "are omitted, and verdict-00-identity-hound.json is a nontrial protocol probe."
)
VERDIGRIS_ACCEPTANCE_DETAIL = (
    "round count comes from verdict-08.json /txn, corroborated by the FRICTION.md "
    "sol-handoff; repair metrics use verdict-01..08 only because other narrated "
    "check and probe activity lacks transaction-aligned verdicts."
)


def _repo_label(path: Path) -> str:
    return path.relative_to(KERNEL_ROOT).as_posix()


def _verdict_paths(directory: Path, final_index: int) -> tuple[Path, ...]:
    return tuple(
        directory / f"verdict-{index:02d}.json"
        for index in range(1, final_index + 1)
    )


def _verdict_text_sources(
    paths: tuple[Path, ...],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (_repo_label(path), path.read_text(encoding="utf-8")) for path in paths
    )


def _verdict_file_sources(
    paths: tuple[Path, ...],
) -> tuple[tuple[Path, str], ...]:
    return tuple((path, _repo_label(path)) for path in paths)


def _trial(
    trial_id: str = "trial-a",
    creature_id: str = "wyrm",
) -> AuthoringTrial:
    return AuthoringTrial(
        TrialId(trial_id),
        CreatureId(creature_id),
        (
            ObstructedTransaction(
                TransactionId("txn-1"),
                (
                    ObstructionCode("WidthObstruction"),
                    ObstructionCode("BalanceObstruction"),
                ),
                10,
            ),
            ObstructedTransaction(
                TransactionId("txn-2"),
                (ObstructionCode("BalanceObstruction"),),
                20,
            ),
            AcceptedTransaction(TransactionId("txn-3"), 30),
            AcceptedTransaction(TransactionId("polish"), 1000),
        ),
    )


def _artifact(label: str = "trial.json", content: bytes = b"trial") -> SourceArtifact:
    return SourceArtifact(label, sha256(content).hexdigest())


def test_authoring_metrics_stop_at_first_acceptance_and_keep_exact_rates() -> None:
    metrics = evaluate_authoring_trial(_trial())

    assert metrics.transaction_count == 4
    assert metrics.analyzed_transaction_count == 3
    assert metrics.rounds_to_acceptance == AvailableCount(3)
    assert metrics.tokens_per_accepted_creature == AvailableCount(60)
    assert metrics.obstruction_repair[0].obstruction_code == "BalanceObstruction"
    assert metrics.obstruction_repair[0].observed_transactions == 2
    assert metrics.obstruction_repair[0].success_rate == AvailableRate(1, 2)
    assert metrics.obstruction_repair[1].obstruction_code == "WidthObstruction"
    assert metrics.obstruction_repair[1].success_rate == AvailableRate(1, 1)


def test_missing_token_observation_obstructs_only_token_metric() -> None:
    source = _trial()
    trial = AuthoringTrial(
        source.trial_id,
        source.creature_id,
        (
            source.transactions[0],
            ObstructedTransaction(
                TransactionId("txn-2"),
                (ObstructionCode("BalanceObstruction"),),
            ),
            *source.transactions[2:],
        ),
    )

    metrics = evaluate_authoring_trial(trial)

    assert metrics.rounds_to_acceptance == AvailableCount(3)
    assert metrics.obstruction_repair[0].success_rate == AvailableRate(1, 2)
    assert isinstance(metrics.tokens_per_accepted_creature, UnavailableMetric)
    assert (
        metrics.tokens_per_accepted_creature.code
        is MetricUnavailableCode.INCOMPLETE_TOKEN_TRANSCRIPT
    )
    assert "txn-2" in metrics.tokens_per_accepted_creature.detail


def test_unaccepted_trial_retains_terminal_obstruction_evidence() -> None:
    trial = AuthoringTrial(
        TrialId("blocked"),
        CreatureId("chimera"),
        (
            ObstructedTransaction(
                TransactionId("txn-1"),
                (ObstructionCode("UnsupportedVocabulary"),),
                11,
            ),
        ),
    )

    metrics = evaluate_authoring_trial(trial)

    assert isinstance(metrics.rounds_to_acceptance, UnavailableMetric)
    assert (
        metrics.rounds_to_acceptance.code
        is MetricUnavailableCode.NO_ACCEPTED_TRANSACTION
    )
    assert isinstance(metrics.tokens_per_accepted_creature, UnavailableMetric)
    assert metrics.obstruction_repair[0].unfollowed_transactions == 1
    assert metrics.obstruction_repair[0].success_rate == UnavailableRate(
        MetricUnavailableCode.NO_FOLLOWING_TRANSACTION,
        "the obstruction was observed only on the final transaction",
    )


def test_authoring_trial_json_decoder_is_strict_and_accumulative() -> None:
    decoded = decode_authoring_trial_json(
        json.dumps(
            {
                "schema_version": 1,
                "trial_id": "trial-a",
                "creature_id": "wyrm",
                "transactions": [
                    {
                        "transaction_id": "same",
                        "status": "obstructed",
                        "obstruction_codes": ["Width", "Width"],
                    },
                    {
                        "transaction_id": "same",
                        "status": "accepted",
                        "obstruction_codes": ["StillBlocked"],
                        "ornament": True,
                    },
                ],
                "canonical_spec": "specs/knight_body.json",
            }
        )
    )

    assert isinstance(decoded, RejectedTrial)
    codes = tuple(obstruction.code for obstruction in decoded.obstructions)
    assert DecodeObstructionCode.UNEXPECTED_FIELD in codes
    assert DecodeObstructionCode.INVALID_OBSTRUCTION_CODES in codes
    assert DecodeObstructionCode.DUPLICATE_TRANSACTION_ID not in codes
    assert all("canonical path" not in obstruction.detail for obstruction in decoded.obstructions)


def test_authoring_trial_json_round_trips_without_spec_paths() -> None:
    payload = {
        "schema_version": 1,
        "trial_id": "trial-json",
        "creature_id": "manticore",
        "transactions": [
            {
                "transaction_id": "first",
                "status": "obstructed",
                "obstruction_codes": ["ClearanceObstruction"],
                "tokens": 40,
            },
            {
                "transaction_id": "second",
                "status": "accepted",
                "obstruction_codes": [],
                "tokens": 25,
            },
        ],
    }

    decoded = decode_authoring_trial_json(json.dumps(payload))

    assert isinstance(decoded, AuthoringTrial)
    assert authoring_trial_metrics_payload(evaluate_authoring_trial(decoded)) == {
        "analyzed_transaction_count": 2,
        "creature_id": "manticore",
        "obstruction_repair": [
            {
                "cleared_by_next_transaction": 1,
                "followed_transactions": 1,
                "not_cleared_by_next_transaction": 0,
                "observed_transactions": 1,
                "obstruction_code": "ClearanceObstruction",
                "success_rate": {
                    "denominator": 1,
                    "numerator": 1,
                    "status": "available",
                },
                "unfollowed_transactions": 0,
            }
        ],
        "rounds_to_acceptance": {"status": "available", "value": 2},
        "tokens_per_accepted_creature": {
            "status": "available",
            "value": 65,
        },
        "transaction_count": 2,
        "trial_id": "trial-json",
    }


def test_complete_session_receipt_decodes_first_acceptance_and_tokens() -> None:
    decoded = decode_session_receipt(
        (GOLDEN / "session_biped_receipts.txt").read_text(encoding="utf-8"),
        trial_id="golden-session-biped",
        creature_id="biped",
    )

    assert isinstance(decoded, AuthoringTrial)
    metrics = evaluate_authoring_trial(decoded)
    assert metrics.transaction_count == 16
    assert metrics.analyzed_transaction_count == 10
    assert metrics.rounds_to_acceptance == AvailableCount(10)
    assert metrics.tokens_per_accepted_creature == AvailableCount(469)
    repair_by_code = {
        metric.obstruction_code: metric for metric in metrics.obstruction_repair
    }
    assert repair_by_code["env_heads_tall"].success_rate == AvailableRate(1, 7)
    assert repair_by_code["env_balanced"].success_rate == AvailableRate(1, 9)


def test_post_spine_verdict_sequences_preserve_local_passes_and_evidence_limits() -> None:
    scree_paths = _verdict_paths(SCREE_TRIAL, 6)
    verdigris_paths = _verdict_paths(VERDIGRIS_TRIAL, 8)
    scree = decode_authoring_verdict_sequence(
        _verdict_text_sources(scree_paths),
        trial_id="post-spine-scree-maiden",
        creature_id="scree-maiden",
        acceptance=VerdictSequenceAcceptance(
            _repo_label(scree_paths[-1]),
            IncompleteAcceptanceRounds(SCREE_ACCEPTANCE_DETAIL),
        ),
    )
    verdigris = decode_authoring_verdict_sequence(
        _verdict_text_sources(verdigris_paths),
        trial_id="post-spine-verdigris",
        creature_id="verdigris-drake",
        acceptance=VerdictSequenceAcceptance(
            _repo_label(verdigris_paths[-1]),
            ReportedAcceptanceRounds(11, VERDIGRIS_ACCEPTANCE_DETAIL),
        ),
    )

    assert isinstance(scree, AuthoringTrial)
    assert isinstance(verdigris, AuthoringTrial)
    assert sum(
        isinstance(transaction, SurfaceAcceptedTransaction)
        for transaction in scree.transactions
    ) == 2
    assert sum(
        isinstance(transaction, SurfaceAcceptedTransaction)
        for transaction in verdigris.transactions
    ) == 3
    scree_metrics = evaluate_authoring_trial(scree)
    verdigris_metrics = evaluate_authoring_trial(verdigris)
    assert scree_metrics.transaction_count == 6
    assert scree_metrics.rounds_to_acceptance == UnavailableMetric(
        MetricUnavailableCode.INCOMPLETE_TRANSACTION_TRANSCRIPT,
        SCREE_ACCEPTANCE_DETAIL,
    )
    scree_repairs = {
        metric.obstruction_code: metric
        for metric in scree_metrics.obstruction_repair
    }
    assert scree_repairs["vascular.CapsuleEscape"].success_rate == (
        AvailableRate(1, 2)
    )
    assert scree_repairs[
        "assembly.EmptySurfaceConduitObstruction"
    ].success_rate == AvailableRate(1, 1)
    assert verdigris_metrics.transaction_count == 8
    assert verdigris_metrics.rounds_to_acceptance == AvailableCount(11)
    verdigris_repairs = {
        metric.obstruction_code: metric
        for metric in verdigris_metrics.obstruction_repair
    }
    assert verdigris_repairs[
        "assembly.EmptySurfaceConduitObstruction"
    ].success_rate == AvailableRate(1, 2)
    assert verdigris_repairs[
        "assembly.SolidIntegrityObstruction"
    ].success_rate == AvailableRate(1, 1)
    assert verdigris_repairs["vascular.Intersection"].success_rate == (
        AvailableRate(1, 1)
    )
    assert isinstance(
        verdigris_metrics.tokens_per_accepted_creature,
        UnavailableMetric,
    )


def test_verdict_sequence_recording_hashes_context_and_round_trips_v2_trial(
    tmp_path: Path,
) -> None:
    verdigris_paths = _verdict_paths(VERDIGRIS_TRIAL, 8)
    stored = record_authoring_verdict_sequence(
        _verdict_file_sources(verdigris_paths),
        tmp_path,
        context_sources=(
            (
                VERDIGRIS_TRIAL / "FRICTION.md",
                _repo_label(VERDIGRIS_TRIAL / "FRICTION.md"),
            ),
        ),
        trial_id="post-spine-verdigris",
        creature_id="verdigris-drake",
        acceptance=VerdictSequenceAcceptance(
            _repo_label(verdigris_paths[-1]),
            ReportedAcceptanceRounds(11, VERDIGRIS_ACCEPTANCE_DETAIL),
        ),
    )

    assert isinstance(stored, StoredCorpusReceipt)
    loaded = load_corpus_receipt(stored.path)
    assert isinstance(loaded, CorpusReceipt)
    assert loaded.source_format is SourceFormat.AUTHORING_VERDICT_SEQUENCE_V1
    assert len(loaded.source_artifacts) == 9
    assert isinstance(loaded.result, MeasuredReceiptResult)
    assert loaded.result.metrics.rounds_to_acceptance == AvailableCount(11)
    assert isinstance(
        decode_authoring_trial_payload(
            json.loads(stored.path.read_text(encoding="utf-8"))["result"][
                "trial"
            ]
        ),
        AuthoringTrial,
    )


def test_zero_clause_session_probe_does_not_fabricate_acceptance() -> None:
    decoded = decode_session_receipt(
        (GOLDEN / "session_knight_receipts.txt").read_text(encoding="utf-8"),
        trial_id="golden-session-knight-revision-probe",
        creature_id="obsidian-knight-underbody",
    )

    assert isinstance(decoded, RejectedTrial)
    assert all(
        obstruction.code is DecodeObstructionCode.MISSING_ACCEPTANCE_AUTHORITY
        for obstruction in decoded.obstructions
    )


def test_content_addressed_receipt_is_idempotent_and_refuses_collision(
    tmp_path: Path,
) -> None:
    receipt = build_corpus_receipt(
        SourceFormat.AUTHORING_TRIAL_JSON_V1,
        (_artifact(),),
        _trial(),
    )
    first = append_corpus_receipt(tmp_path, receipt)
    second = append_corpus_receipt(tmp_path, receipt)

    assert isinstance(first, StoredCorpusReceipt)
    assert isinstance(second, StoredCorpusReceipt)
    assert first.receipt_sha256 == corpus_receipt_sha256(receipt)
    assert first.created is True
    assert second.created is False
    assert first.path.read_bytes() == corpus_receipt_bytes(receipt)

    first.path.write_bytes(b"counterfeit\n")
    collision = append_corpus_receipt(tmp_path, receipt)

    assert collision == CorpusIOObstruction(
        str(first.path),
        CorpusIOObstructionCode.RECEIPT_COLLISION,
        "content-addressed path contains different bytes",
    )
    assert first.path.read_bytes() == b"counterfeit\n"


def test_loaded_receipt_recomputes_metrics_and_checks_content_address(
    tmp_path: Path,
) -> None:
    receipt = build_corpus_receipt(
        SourceFormat.AUTHORING_TRIAL_JSON_V1,
        (_artifact(),),
        _trial(),
    )
    stored = append_corpus_receipt(tmp_path, receipt)
    assert isinstance(stored, StoredCorpusReceipt)

    loaded = load_corpus_receipt(stored.path)

    assert loaded == receipt
    renamed = tmp_path / "not-the-content-address.json"
    renamed.write_bytes(stored.path.read_bytes())
    rejected = load_corpus_receipt(renamed)
    assert isinstance(rejected, CorpusIOObstruction)
    assert rejected.code is CorpusIOObstructionCode.INVALID_RECEIPT_ADDRESS


def test_corpus_glues_repairs_and_keeps_unparseable_sources() -> None:
    second_trial = AuthoringTrial(
        TrialId("trial-b"),
        CreatureId("gryphon"),
        (
            ObstructedTransaction(
                TransactionId("txn-1"),
                (ObstructionCode("BalanceObstruction"),),
                4,
            ),
            AcceptedTransaction(TransactionId("txn-2"), 6),
        ),
    )
    receipts = (
        build_corpus_receipt(
            SourceFormat.AUTHORING_TRIAL_JSON_V1,
            (_artifact("a.json", b"a"),),
            _trial(),
        ),
        build_corpus_receipt(
            SourceFormat.AUTHORING_TRIAL_JSON_V1,
            (_artifact("b.json", b"b"),),
            second_trial,
        ),
        build_unparseable_corpus_receipt(
            (_artifact("legacy.md", b"summary"),),
            (
                DecodeObstruction(
                    "/",
                    DecodeObstructionCode.UNSUPPORTED_ARTIFACT_FORMAT,
                    "summary has no transaction ledger",
                ),
            ),
        ),
    )

    summary = summarize_authoring_corpus(receipts)

    assert isinstance(summary, AuthoringCorpusSummary)
    assert tuple(metric.trial_id for metric in summary.metrics.trials) == (
        "trial-a",
        "trial-b",
    )
    balance = summary.metrics.obstruction_repair[0]
    assert balance.obstruction_code == "BalanceObstruction"
    assert balance.success_rate == AvailableRate(2, 3)
    assert len(summary.unparseable_sources) == 1
    assert summary.unparseable_sources[0].source_artifacts[0].path == "legacy.md"


def test_conflicting_measured_trial_identity_obstructs_corpus_summary() -> None:
    receipts = (
        build_corpus_receipt(
            SourceFormat.AUTHORING_TRIAL_JSON_V1,
            (_artifact("a.json", b"a"),),
            _trial(),
        ),
        build_corpus_receipt(
            SourceFormat.AUTHORING_TRIAL_JSON_V1,
            (_artifact("b.json", b"b"),),
            _trial(creature_id="different-creature"),
        ),
    )

    summary = summarize_authoring_corpus(receipts)

    assert isinstance(summary, RejectedCorpus)
    assert summary.obstructions[0].code is CorpusIOObstructionCode.CONFLICTING_TRIAL_IDENTITY


def test_record_boundaries_persist_measured_and_unparseable_results(
    tmp_path: Path,
) -> None:
    trial_path = tmp_path / "trial.json"
    trial_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trial_id": "recorded",
                "creature_id": "basilisk",
                "transactions": [
                    {
                        "transaction_id": "1",
                        "status": "accepted",
                        "obstruction_codes": [],
                        "tokens": 9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus"

    measured = record_authoring_trial_json(
        trial_path,
        corpus,
        source_label="trials/basilisk.json",
    )
    unparseable = record_unparseable_artifact_bundle(
        (GOLDEN / "session_knight_receipts.txt",),
        corpus,
        source_labels=("golden/session_knight_receipts.txt",),
        obstruction=DecodeObstruction(
            "/",
            DecodeObstructionCode.UNSUPPORTED_ARTIFACT_FORMAT,
            "revision probe has no creature acceptance authority",
        ),
    )

    assert isinstance(measured, StoredCorpusReceipt)
    assert isinstance(unparseable, StoredCorpusReceipt)
    summary = summarize_authoring_corpus_directory(corpus)
    assert isinstance(summary, AuthoringCorpusSummary)
    assert summary.metrics.trials[0].tokens_per_accepted_creature == AvailableCount(9)
    assert len(summary.unparseable_sources) == 1


def test_module_entry_point_records_and_summarizes_without_cli_registry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trial_path = tmp_path / "trial.json"
    trial_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trial_id": "module-entry",
                "creature_id": "sphinx",
                "transactions": [
                    {
                        "transaction_id": "1",
                        "status": "accepted",
                        "obstruction_codes": [],
                        "tokens": 13,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus"

    record_exit = main(("record-json", str(trial_path), str(corpus)))
    record_output = json.loads(capsys.readouterr().out)
    summary_exit = main(("summary", str(corpus)))
    summary_output = json.loads(capsys.readouterr().out)

    assert record_exit == 0
    assert record_output["status"] == "stored"
    assert summary_exit == 0
    assert summary_output["metrics"]["trials"][0][
        "tokens_per_accepted_creature"
    ] == {"status": "available", "value": 13}
    assert "spec" not in json.dumps(summary_output)


def test_session_recording_is_deterministic_for_explicit_trial_metadata(
    tmp_path: Path,
) -> None:
    first = record_session_receipt(
        GOLDEN / "session_biped_receipts.txt",
        tmp_path,
        trial_id="session-biped",
        creature_id="biped",
        source_label="golden/session_biped_receipts.txt",
    )
    second = record_session_receipt(
        GOLDEN / "session_biped_receipts.txt",
        tmp_path,
        trial_id="session-biped",
        creature_id="biped",
        source_label="golden/session_biped_receipts.txt",
    )

    assert isinstance(first, StoredCorpusReceipt)
    assert isinstance(second, StoredCorpusReceipt)
    assert first.receipt_sha256 == second.receipt_sha256
    assert (first.created, second.created) == (True, False)


def test_checked_in_corpus_is_content_addressed_and_honest() -> None:
    corpus_directory = KERNEL_ROOT / "golem" / "evals" / "corpus" / "receipts"

    loaded = load_authoring_corpus(corpus_directory)

    assert not isinstance(loaded, RejectedCorpus)
    assert len(loaded) == 15
    assert all(
        path.name == f"{sha256(path.read_bytes()).hexdigest()}.json"
        for path in corpus_directory.glob("*.json")
    )
    summary = summarize_authoring_corpus(loaded)
    assert isinstance(summary, AuthoringCorpusSummary)
    assert len(summary.metrics.trials) == 3
    assert len(summary.unparseable_sources) == 12
    measured = {
        metric.trial_id: metric for metric in summary.metrics.trials
    }
    baseline = measured["golden-session-biped-from-empty"]
    assert baseline.rounds_to_acceptance == AvailableCount(10)
    assert baseline.tokens_per_accepted_creature == AvailableCount(469)
    baseline_repairs = {
        metric.obstruction_code: metric
        for metric in baseline.obstruction_repair
    }
    assert baseline_repairs["env_heads_tall"].success_rate == AvailableRate(
        1, 7
    )
    assert baseline_repairs["env_balanced"].success_rate == AvailableRate(1, 9)
    scree = measured["post-spine-scree-maiden"]
    assert scree.rounds_to_acceptance == UnavailableMetric(
        MetricUnavailableCode.INCOMPLETE_TRANSACTION_TRANSCRIPT,
        SCREE_ACCEPTANCE_DETAIL,
    )
    assert isinstance(scree.tokens_per_accepted_creature, UnavailableMetric)
    verdigris = measured["post-spine-verdigris"]
    assert verdigris.rounds_to_acceptance == AvailableCount(11)
    assert isinstance(
        verdigris.tokens_per_accepted_creature,
        UnavailableMetric,
    )
    measured_receipts = tuple(
        receipt
        for receipt in loaded
        if isinstance(receipt.result, MeasuredReceiptResult)
    )
    verdict_receipts = tuple(
        receipt
        for receipt in measured_receipts
        if receipt.source_format
        is SourceFormat.AUTHORING_VERDICT_SEQUENCE_V1
    )
    assert len(verdict_receipts) == 2
    # Sealed artifacts stay pinned to disk; FRICTION.md journals are
    # append-only living documents whose hashes are recording-time provenance.
    assert all(
        artifact.sha256
        == sha256((KERNEL_ROOT / artifact.path).read_bytes()).hexdigest()
        for receipt in verdict_receipts
        for artifact in receipt.source_artifacts
        if not artifact.path.endswith("FRICTION.md")
    )
    assert all(
        len(artifact.sha256) == 64 and (KERNEL_ROOT / artifact.path).exists()
        for receipt in verdict_receipts
        for artifact in receipt.source_artifacts
    )
    scree_receipt = next(
        receipt
        for receipt in verdict_receipts
        if isinstance(receipt.result, MeasuredReceiptResult)
        and receipt.result.trial.trial_id == "post-spine-scree-maiden"
    )
    assert {
        artifact.path for artifact in scree_receipt.source_artifacts
    } >= {
        "rehearsal/scree-maiden-trial/FRICTION.md",
        "rehearsal/scree-maiden-trial/verdict-00-identity-hound.json",
    }
    metric_payload = authoring_corpus_summary_payload(summary)["metrics"]
    assert "timestamp" not in json.dumps(metric_payload)
    assert "wall_clock" not in json.dumps(metric_payload)
