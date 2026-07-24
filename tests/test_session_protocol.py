from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from golem.cli import session as session_cli
from golem.cli.parser import main
from golem.session import algebra, ops, protocol as session_protocol
from golem.session.fault import (
    ASSEMBLY_COMPILE_SEAM,
    SESSION_EVIDENCE_SEAM,
    project_engine_fault,
)
from golem.session.protocol import (
    AuthoringEvidence,
    Diagnostic,
    Margin,
    ProtocolSession,
    TransactionVerdict,
    VerdictDeltas,
    _verdict_deltas,
    start_session,
    submit,
)
from golem.session.journal import RepairJournalEvent, RepairOperationKind
from golem.session.state import AuthoredState, Compiled, SessionState, empty_spec


def _margin(
    constraint: str,
    value: float,
    normalized: float,
    active: bool,
) -> Margin:
    return Margin(
        constraint=constraint,
        address=f"meta@{constraint[-1]}",
        predicate="minimum",
        required=0.0,
        observed=value,
        margin=value,
        normalized_margin=normalized,
        unit="probe",
        active=active,
        witness={"value": value},
    )


def _probe_evidence(state: SessionState) -> AuthoringEvidence:
    edited = state.spec.get("x") == 1
    margins = (
        _margin("contract:a", 4.0 if edited else 1.0, 0.4 if edited else 0.1, not edited),
        _margin("contract:b", 2.0 if edited else 6.0, 0.2 if edited else 0.6, edited),
    )
    return AuthoringEvidence((), margins)


def _probe_session(tmp_path: Path) -> ProtocolSession:
    return start_session(
        {"x": 0},
        tmp_path,
        evidence_channel=_probe_evidence,
    )


def _expected_cli_record(
    verdict: TransactionVerdict,
    *,
    all_diagnostics: bool,
) -> dict[str, object]:
    raw = verdict.to_json()
    diagnostics = tuple(raw["diagnostics"])
    expected = tuple(
        diagnostic
        for diagnostic in diagnostics
        if diagnostic["witness"].get("expected") is True
    )
    unexpected = tuple(
        diagnostic
        for diagnostic in diagnostics
        if diagnostic["witness"].get("expected") is not True
    )
    shown = diagnostics if all_diagnostics else unexpected
    return {
        **raw,
        "diagnostics": shown,
        "diagnostics_summary": {
            "total": len(diagnostics),
            "expected": len(expected),
            "unexpected": len(unexpected),
            "shown": len(shown),
            "expected_filtered": not all_diagnostics,
        },
    }


def _contract_clause(identifier: str, minimum: float) -> dict[str, object]:
    return {
        "id": identifier,
        "scope": ["part:seed"],
        "metric": "height",
        "operator": "minimum",
        "value": minimum,
        "unit": "world_unit",
        "tolerance": 0.0,
    }


@pytest.mark.parametrize(
    "submission_request",
    (
        {
            "base_txn": 0,
            "operations": [
                {"op": "set", "addr": "meta@x", "value": 1},
            ],
        },
        {"base_txn": 0, "document": {"x": 1}},
        {"document": {"x": 1}},
        {"x": 1},
    ),
)
def test_all_input_shapes_normalize_into_the_same_journal(
    tmp_path: Path,
    submission_request: dict[str, object],
) -> None:
    submission = submit(_probe_session(tmp_path), submission_request)

    assert submission.session.document == {"x": 1}
    assert submission.session.txn == 1
    assert submission.verdict.accepted
    assert submission.verdict.to_json()["operations"] == (
        {"op": "set", "addr": "meta@x", "value": 1},
    )
    assert submission.verdict.to_json()["journal"][0]["applied"] == {
        "op": "set",
        "addr": "meta@x",
        "value": 1,
    }
    assert submission.verdict.to_json()["journal"][0]["inverse"] == {
        "op": "set",
        "addr": "meta@x",
        "value": 0,
    }
    assert submission.verdict.to_json()["changed_addresses"] == ("meta@x",)


def test_addressed_transaction_canonicalizes_equivalent_address_spelling(
    tmp_path: Path,
) -> None:
    opening = json.loads(
        (Path(__file__).parents[1] / "specs" / "knight_body.json").read_text()
    )
    submission = submit(
        start_session(opening, tmp_path, evidence_channel=_probe_evidence),
        {
            "base_txn": 0,
            "operations": [
                {
                    "op": "set",
                    "addr": "skeleton/%66orearm@length",
                    "value": 0.41,
                },
            ],
        },
    )
    record = submission.verdict.to_json()

    assert submission.session.document != opening
    assert record["operations"] == (
        {
            "op": "set",
            "addr": "skeleton/forearm@length",
            "value": 0.41,
        },
    )
    assert record["journal"][0]["applied"]["addr"] == (
        "skeleton/forearm@length"
    )
    assert record["changed_addresses"] == ("skeleton/forearm@length",)


def test_raw_reserved_identity_spelling_and_file_edit_share_canonical_ops(
    tmp_path: Path,
) -> None:
    base = empty_spec("probe")
    bone = {
        "id": "a#b",
        "parent": "root",
        "length": 0.2,
        "rest_dir": [0.0, 1.0, 0.0],
        "joint": {"dof": "fixed"},
    }
    opening = {
        **base,
        "skeleton": {**base["skeleton"], "bones": [bone]},
    }
    edited = {
        **opening,
        "skeleton": {
            **opening["skeleton"],
            "bones": [{**bone, "length": 0.3}],
        },
    }
    addressed = submit(
        start_session(opening, tmp_path, evidence_channel=_probe_evidence),
        {
            "base_txn": 0,
            "operations": [
                {
                    "op": "set",
                    "addr": "skeleton/a#b@length",
                    "value": 0.3,
                }
            ],
        },
    ).verdict.to_json()
    file_edited = submit(
        start_session(opening, tmp_path, evidence_channel=_probe_evidence),
        edited,
    ).verdict.to_json()

    assert addressed["operations"] == file_edited["operations"] == (
        {
            "op": "set",
            "addr": "skeleton/a%23b@length",
            "value": 0.3,
        },
    )
    assert addressed["journal"] == file_edited["journal"]
    assert addressed["changed_addresses"] == file_edited["changed_addresses"]


def test_plain_document_may_contain_transaction_field_names(tmp_path: Path) -> None:
    submission = submit(
        _probe_session(tmp_path),
        {"x": 1, "base_txn": "authored", "operations": {"notes": True}},
    )
    record = submission.verdict.to_json()

    assert submission.session.document == {
        "x": 1,
        "base_txn": "authored",
        "operations": {"notes": True},
    }
    assert submission.session.txn == 3
    assert record["status"] == "accepted"
    assert record["operations"] == (
        {"op": "set", "addr": "meta@x", "value": 1},
        {"op": "set", "addr": "meta@base_txn", "value": "authored"},
        {
            "op": "set",
            "addr": "meta@operations",
            "value": {"notes": True},
        },
    )


def test_protocol_session_exposes_only_detached_authoring_views(
    tmp_path: Path,
) -> None:
    session = _probe_session(tmp_path)
    document = session.document
    evidence = session.evidence
    document["x"] = 9
    evidence.margins[0].witness["value"] = 9

    assert not hasattr(session, "model")
    assert session.document == {"x": 0}
    assert session.txn == 0
    assert session.journal.entries == ()
    assert session.evidence.margins[0].witness == {"value": 1.0}


def test_evidence_channel_shares_only_derived_outcome_and_returns_no_aliases(
    tmp_path: Path,
) -> None:
    provided = AuthoringEvidence(
        (),
        (_margin("contract:a", 1.0, 0.1, True),),
    )
    channel = Mock(return_value=provided)
    session = start_session(
        {"x": 0},
        tmp_path,
        evidence_channel=channel,
    )
    retained = channel.call_args.args[0]
    authoritative = algebra.current_branch(session._model).state

    assert retained is not authoritative
    assert retained.spec is not authoritative.spec
    assert retained.outcome is authoritative.outcome
    assert session._evidence is not provided
    assert session._evidence.margins[0].witness is not provided.margins[0].witness

    submission = submit(session, {"x": 1})
    next_authoritative = algebra.current_branch(submission.session._model).state
    next_retained = channel.call_args_list[1].args[0]

    assert next_retained.spec is not next_authoritative.spec
    assert next_retained.outcome is next_authoritative.outcome
    assert submission.verdict._evidence is not submission.session._evidence
    assert (
        submission.verdict._evidence.margins[0].witness
        is not submission.session._evidence.margins[0].witness
    )


def test_non_json_plain_document_is_a_typed_request_obstruction(
    tmp_path: Path,
) -> None:
    session = _probe_session(tmp_path)
    submission = submit(session, {"x": object()})
    record = submission.verdict.to_json()

    assert submission.session is session
    assert submission.session.document == {"x": 0}
    assert submission.session.txn == 0
    assert record["status"] == "rejected"
    assert record["diagnostics"][0]["code"] == "session.invalid_document"
    assert record["operations"] == ()
    assert record["journal"] == ()


def test_nonfinite_addressed_value_is_a_strict_json_obstruction(
    tmp_path: Path,
) -> None:
    session = _probe_session(tmp_path)
    submission = submit(
        session,
        {
            "base_txn": 0,
            "operations": [
                {"op": "set", "addr": "meta@x", "value": float("nan")},
            ],
        },
    )
    record = submission.verdict.to_json()

    assert submission.session is session
    assert record["status"] == "rejected"
    assert record["diagnostics"][0]["code"] == "session.invalid_operation"
    assert record["operations"] == ()
    assert record["journal"] == ()
    assert json.loads(json.dumps(record, allow_nan=False))["status"] == "rejected"


def test_nonfinite_plain_document_is_a_strict_json_obstruction(
    tmp_path: Path,
) -> None:
    session = _probe_session(tmp_path)
    submission = submit(session, {"x": float("nan")})
    record = submission.verdict.to_json()

    assert submission.session is session
    assert submission.session.document == {"x": 0}
    assert record["diagnostics"][0]["code"] == "session.invalid_document"
    assert record["operations"] == ()
    assert json.loads(json.dumps(record, allow_nan=False))["status"] == "rejected"


def test_nonfinite_session_base_cannot_be_repaired_past_the_journal(
    tmp_path: Path,
) -> None:
    session = start_session({"x": float("nan")}, tmp_path)
    submission = submit(session, {"x": 0})
    record = submission.verdict.to_json()

    assert not session.evidence.accepted
    assert session.evidence.diagnostics[0].code == "session.invalid_document"
    assert submission.session is session
    assert submission.session.txn == 0
    assert record["operations"] == ()
    assert record["journal"] == ()
    assert json.loads(json.dumps(record, allow_nan=False))["status"] == "rejected"


@pytest.mark.parametrize(
    ("graph", "accepted", "severity", "code"),
    (
        (
            {
                "parts": [
                    {
                        "id": "core",
                        "type": "blob",
                        "center": [0.0, 0.0, 0.0],
                        "size": [0.3, 0.4, 0.2],
                        "rot": [2.0, 0.0, 0.0, 0.0],
                    }
                ]
            },
            True,
            "warning",
            "geometry.R-quat-nonunit",
        ),
        (
            {"parts": []},
            False,
            "error",
            "geometry.R-malformed-graph",
        ),
    ),
)
def test_geometry_obstruction_fatality_controls_protocol_acceptance(
    tmp_path: Path,
    graph: dict[str, object],
    accepted: bool,
    severity: str,
    code: str,
) -> None:
    outcome = Compiled(graph, {}, None, object(), (), (), ())
    state = SessionState.from_carriers(
        AuthoredState(graph, tmp_path),
        outcome,
    )
    evidence = session_protocol._compiled_evidence(state, outcome)

    assert evidence.accepted is accepted
    assert evidence.diagnostics[0].severity == severity
    assert evidence.diagnostics[0].code == code


def test_unexpected_assembly_exception_is_a_structured_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_untyped_assembly(*_arguments: object) -> object:
        raise RuntimeError("assembly probe")

    monkeypatch.setattr(
        session_protocol,
        "compile_assembly",
        reject_untyped_assembly,
    )
    session = start_session(empty_spec("probe"), tmp_path)

    fault = project_engine_fault(
        ASSEMBLY_COMPILE_SEAM,
        RuntimeError("assembly probe"),
    )
    assert not session.evidence.accepted
    assert session.evidence.diagnostics[0].to_json() == {
        "code": "engine.fault",
        "severity": "error",
        "address": "assembly.compile",
        "predicate": "engine seam returns a typed verdict, never a raw exception",
        "required": "typed verdict",
        "observed": "RuntimeError",
        "witness": fault.to_json(),
        "help": {"operations": ()},
    }


def test_unexpected_default_evidence_exception_is_a_structured_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_untyped_geometry(*_arguments: object) -> object:
        raise RuntimeError("geometry probe")

    monkeypatch.setattr(
        session_protocol,
        "decode_graph",
        reject_untyped_geometry,
    )
    session = start_session(empty_spec("probe"), tmp_path)

    fault = project_engine_fault(
        SESSION_EVIDENCE_SEAM,
        RuntimeError("geometry probe"),
    )
    assert not session.evidence.accepted
    assert session.evidence.diagnostics[0].to_json() == {
        "code": "engine.fault",
        "severity": "error",
        "address": "session.evidence",
        "predicate": "engine seam returns a typed verdict, never a raw exception",
        "required": "typed verdict",
        "observed": "RuntimeError",
        "witness": fault.to_json(),
        "help": {"operations": ()},
    }


def test_custom_evidence_channel_contract_violations_are_not_swallowed(
    tmp_path: Path,
) -> None:
    def invalid_channel(_state: SessionState) -> AuthoringEvidence:
        raise RuntimeError("custom channel probe")

    with pytest.raises(RuntimeError, match="custom channel probe"):
        start_session(
            empty_spec("probe"),
            tmp_path,
            evidence_channel=invalid_channel,
        )


def test_verdict_schema_and_reasoning_order_are_pinned(tmp_path: Path) -> None:
    submission = submit(
        _probe_session(tmp_path),
        {"base_txn": 0, "document": {"x": 1}},
    )
    record = submission.verdict.to_json()

    assert tuple(record) == (
        "status",
        "diagnostics",
        "margins",
        "binding_constraint",
        "deltas",
        "contract_refs",
        "changed_addresses",
        "base_txn",
        "txn",
        "operations",
        "journal",
        "schema",
    )
    assert tuple(record["margins"][0]) == (
        "constraint",
        "address",
        "predicate",
        "required",
        "observed",
        "margin",
        "normalized_margin",
        "unit",
        "authoring_addresses",
        "disposition",
        "active",
        "witness",
    )
    assert tuple(record["deltas"]) == (
        "margins",
        "newly_active",
        "newly_inactive",
    )
    assert tuple(record["deltas"]["margins"][0]) == (
        "constraint",
        "address",
        "before",
        "after",
        "delta",
        "before_normalized",
        "after_normalized",
        "normalized_delta",
    )
    assert record["schema"] == "golem.session.verdict/2"


def test_repair_events_are_distinct_verdict_journal_entries(tmp_path: Path) -> None:
    repair = RepairJournalEvent(
        txn=1,
        constraint="anatomy:carrier:shin",
        address="skeleton/shin/flesh[0]@size",
        predicate="circulation_clearance",
        authored=0.02,
        repaired=0.04,
        operation=RepairOperationKind.SCALE_RADIUS,
        factor=2.0,
        authoring_addresses=("skeleton/shin/flesh[0]@size",),
    )

    def evidence_channel(_state: SessionState) -> AuthoringEvidence:
        return AuthoringEvidence((), (), (repair,))

    record = submit(
        start_session({"x": 0}, tmp_path, evidence_channel=evidence_channel),
        {"x": 1},
    ).verdict.to_json()

    assert tuple(entry["kind"] for entry in record["journal"]) == (
        "op",
        "auto_repair",
    )
    assert record["journal"][1] == repair.to_json()


def test_base_transaction_delta_reports_margin_and_active_set_changes(
    tmp_path: Path,
) -> None:
    record = submit(
        _probe_session(tmp_path),
        {"base_txn": 0, "document": {"x": 1}},
    ).verdict.to_json()

    assert record["deltas"]["margins"] == (
        {
            "constraint": "contract:a",
            "address": "meta@a",
            "before": 1.0,
            "after": 4.0,
            "delta": 3.0,
            "before_normalized": 0.1,
            "after_normalized": 0.4,
            "normalized_delta": 0.3,
        },
        {
            "constraint": "contract:b",
            "address": "meta@b",
            "before": 6.0,
            "after": 2.0,
            "delta": -4.0,
            "before_normalized": 0.6,
            "after_normalized": 0.2,
            "normalized_delta": -0.4,
        },
    )
    assert record["deltas"]["newly_active"] == ("contract:b",)
    assert record["deltas"]["newly_inactive"] == ("contract:a",)
    assert record["binding_constraint"]["constraint"] == "contract:b"


def test_normalized_margin_movement_is_reported_when_raw_slack_is_stable() -> None:
    before = AuthoringEvidence(
        (),
        (
            _margin("contract:a", 1.0, 0.1, True),
            _margin("contract:b", 1.0, 0.2, False),
        ),
    )
    after = AuthoringEvidence(
        (),
        (
            _margin("contract:a", 1.0, 0.3, False),
            _margin("contract:b", 1.0, 0.2, True),
        ),
    )

    deltas = _verdict_deltas(before, after).to_json()

    assert deltas["margins"] == (
        {
            "constraint": "contract:a",
            "address": "meta@a",
            "before": 1.0,
            "after": 1.0,
            "delta": 0.0,
            "before_normalized": 0.1,
            "after_normalized": 0.3,
            "normalized_delta": 0.2,
        },
    )
    assert deltas["newly_active"] == ("contract:b",)
    assert deltas["newly_inactive"] == ("contract:a",)


def test_stale_base_is_a_typed_obstruction_and_does_not_advance(
    tmp_path: Path,
) -> None:
    session = _probe_session(tmp_path)
    submission = submit(
        session,
        {
            "base_txn": 4,
            "operations": [
                {"op": "set", "addr": "meta@x", "value": 1},
            ],
        },
    )
    record = submission.verdict.to_json()

    assert submission.session is session
    assert submission.session.document == {"x": 0}
    assert submission.session.txn == 0
    assert record["status"] == "rejected"
    assert tuple(record["diagnostics"][0]) == (
        "code",
        "severity",
        "address",
        "predicate",
        "required",
        "observed",
        "witness",
        "help",
    )
    assert record["diagnostics"][0]["code"] == "session.stale_base_txn"
    assert record["operations"] == ()
    assert record["journal"] == ()


def test_rejected_document_is_still_normalized_and_journaled(
    tmp_path: Path,
) -> None:
    base = empty_spec("probe")
    submission = submit(
        start_session(base, tmp_path),
        {
            "base_txn": 0,
            "document": {
                **base,
                "mystery_section": {"nested": True},
            },
        },
    )
    record = submission.verdict.to_json()

    assert record["status"] == "rejected"
    assert submission.session.txn == 1
    assert submission.session.document["mystery_section"] == {"nested": True}
    assert record["operations"] == (
        {
            "op": "set",
            "addr": "meta@mystery_section",
            "value": {"nested": True},
        },
    )
    assert record["changed_addresses"] == ("meta@mystery_section",)
    assert record["diagnostics"][0]["code"] == (
        "body.UnknownBodySpecKeyObstruction"
    )


@pytest.mark.parametrize(
    ("section", "malformed"),
    (
        ("blend", "bad"),
        ("pose", []),
        ("skeleton", []),
    ),
)
def test_malformed_file_edits_reach_the_journal_and_body_compiler(
    tmp_path: Path,
    section: str,
    malformed: object,
) -> None:
    base = empty_spec("probe")
    submission = submit(
        start_session(base, tmp_path),
        {**base, section: malformed},
    )
    record = submission.verdict.to_json()

    assert record["status"] == "rejected"
    assert submission.session.txn == 1
    assert submission.session.document[section] == malformed
    expected_address = f"meta@{section}"
    expected_value = malformed
    assert record["operations"] == (
        {"op": "set", "addr": expected_address, "value": expected_value},
    )
    assert record["changed_addresses"] == (expected_address,)
    first = record["diagnostics"][0]
    if section == "blend":
        # The scalar blend decode gap raises inside the body compiler; the
        # oracle boundary projects it to the typed engine fault (R11),
        # addressed at the seam that raised.
        assert first["code"] == "engine.fault"
        assert first["address"] == "body.compile"
    else:
        assert first["code"].startswith("body.")


def test_success_carries_margins_and_binding_constraint(tmp_path: Path) -> None:
    session = start_session(empty_spec("probe"), tmp_path)

    assert session.evidence.accepted
    assert session.evidence.margins
    assert session.evidence.binding_constraint is not None
    assert session.evidence.binding_constraint.active
    assert all(margin.margin >= 0.0 for margin in session.evidence.margins)


def test_duplicate_contract_ids_keep_distinct_margins_and_binding_deltas(
    tmp_path: Path,
) -> None:
    base = {
        **empty_spec("probe"),
        "contract": {
            "contract": "probe",
            "clauses": [
                _contract_clause("same", 0.01),
                _contract_clause("same", 0.02),
            ],
        },
    }
    session = start_session(base, tmp_path)
    edited = {
        **base,
        "contract": {
            "contract": "probe",
            "clauses": [
                _contract_clause("same", 0.03),
                _contract_clause("same", 0.02),
            ],
        },
    }
    record = submit(session, edited).verdict.to_json()
    before = tuple(
        (margin.constraint, margin.active)
        for margin in session.evidence.margins
        if margin.constraint.startswith("contract:")
    )
    after = tuple(
        (margin["constraint"], margin["active"])
        for margin in record["margins"]
        if margin["constraint"].startswith("contract:")
    )

    assert before == (
        ("contract:same:height@0", False),
        ("contract:same:height@1", True),
    )
    assert after == (
        ("contract:same:height@0", True),
        ("contract:same:height@1", False),
    )
    assert record["deltas"]["newly_active"] == (
        "contract:same:height@0",
    )
    assert record["deltas"]["newly_inactive"] == (
        "contract:same:height@1",
    )


def test_unrelated_contract_insertion_does_not_renumber_unique_margin_identity(
    tmp_path: Path,
) -> None:
    base = {
        **empty_spec("probe"),
        "contract": {
            "contract": "probe",
            "clauses": [_contract_clause("floor", 0.01)],
        },
    }
    edited = {
        **base,
        "contract": {
            "contract": "probe",
            "clauses": [
                _contract_clause("other", 0.001),
                _contract_clause("floor", 0.01),
            ],
        },
    }
    session = start_session(base, tmp_path)
    record = submit(session, edited).verdict.to_json()

    assert tuple(
        margin.constraint
        for margin in session.evidence.margins
        if margin.constraint.startswith("contract:")
    ) == ("contract:floor",)
    assert tuple(
        margin["constraint"]
        for margin in record["margins"]
        if margin["constraint"].startswith("contract:")
    ) == ("contract:other", "contract:floor")
    assert tuple(
        delta["constraint"] for delta in record["deltas"]["margins"]
    ) == ("contract:other",)


def test_root_glued_add_inverse_preserves_inserted_changed_address(
    tmp_path: Path,
) -> None:
    base = empty_spec("probe")
    record = submit(
        start_session(base, tmp_path, evidence_channel=_probe_evidence),
        {
            "base_txn": 0,
            "operations": [
                {
                    "op": "add",
                    "kind": "array",
                    "addr": "skeleton/root",
                    "params": {
                        "name": "probe",
                        "n": 1,
                        "template": {"kind": "blob"},
                    },
                }
            ],
        },
    ).verdict.to_json()

    assert record["journal"][0]["inverse"]["addr"] == "meta"
    assert record["changed_addresses"] == (
        "skeleton/root",
        "skeleton/root/probe",
    )


def test_root_glued_remove_inverse_preserves_parent_changed_address(
    tmp_path: Path,
) -> None:
    base = empty_spec("probe")
    document = {
        **base,
        "skeleton": {
            **base["skeleton"],
            "bones": [
                {
                    "id": "orphan",
                    "length": 0.2,
                    "rest_dir": [0.0, 1.0, 0.0],
                    "joint": {"dof": "fixed"},
                }
            ],
        },
    }
    record = submit(
        start_session(document, tmp_path, evidence_channel=_probe_evidence),
        {
            "base_txn": 0,
            "operations": [
                {"op": "remove", "addr": "skeleton/orphan"},
            ],
        },
    ).verdict.to_json()

    assert record["journal"][0]["inverse"]["addr"] == "meta"
    assert record["changed_addresses"] == (
        "skeleton/orphan",
        "skeleton/root",
    )


def test_rename_changed_addresses_include_old_and_new_record_identities(
    tmp_path: Path,
) -> None:
    base = empty_spec("probe")
    document = {
        **base,
        "skeleton": {
            **base["skeleton"],
            "bones": [
                {
                    "id": "old",
                    "parent": "root",
                    "length": 0.2,
                    "rest_dir": [0.0, 1.0, 0.0],
                    "joint": {"dof": "fixed"},
                }
            ],
        },
    }
    record = submit(
        start_session(document, tmp_path, evidence_channel=_probe_evidence),
        {
            "base_txn": 0,
            "operations": [
                {"op": "rename", "addr": "skeleton/old", "to": "new"},
            ],
        },
    ).verdict.to_json()

    assert record["changed_addresses"] == (
        "skeleton/old",
        "skeleton/new",
    )


def test_failed_contract_exposes_round_trippable_help_operations(
    tmp_path: Path,
) -> None:
    document = {
        **empty_spec("probe"),
        "contract": {
            "contract": "probe",
            "clauses": [
                {
                    "id": "height_floor",
                    "scope": ["part:seed"],
                    "metric": "height",
                    "operator": "minimum",
                    "value": 1.0,
                    "unit": "world_unit",
                    "tolerance": 0.0,
                    "knob": "skeleton/root/seed@size[1]",
                    "direction": "+",
                }
            ],
        },
    }
    session = start_session(document, tmp_path)
    diagnostic = session.evidence.diagnostics[0]
    raw_operations = diagnostic.to_json()["help"]["operations"]

    assert diagnostic.code == "contract.constraint_failed"
    assert diagnostic.contract_refs == ("receipts",)
    assert raw_operations
    decoded = tuple(map(ops.decode_op, raw_operations))
    assert all(not isinstance(operation, ops.EditObstructed) for operation in decoded)
    compiled = ops.compile_program(
        document,
        tuple(
            operation
            for operation in decoded
            if not isinstance(operation, ops.EditObstructed)
        ),
    )
    assert isinstance(compiled, ops.ProgramPlan)


def test_session_help_teaches_envelopes_stateless_rotation_and_raw_diagnostics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(("session", "--help"))
    captured = capsys.readouterr()

    assert stopped.value.code == 0
    assert captured.err == ""
    assert '{"base_txn": 0, "operations": [' in captured.out
    assert '{"base_txn": 0, "document": {...}}' in captured.out
    assert "fresh in-memory session at txn 0" in captured.out
    assert "Rotate the last accepted" in captured.out
    assert "edited document here before the next invocation" in captured.out
    assert "--all" in captured.out
    assert "witness.expected" in captured.out


def test_session_render_filters_expected_diagnostics_by_default() -> None:
    verdict = TransactionVerdict(
        AuthoringEvidence(
            (
                Diagnostic(
                    code="body.anomaly.expected",
                    severity="warning",
                    address="part:a~b",
                    predicate="declared contact remains expected",
                    required=True,
                    observed=True,
                    witness={"expected": True},
                ),
                Diagnostic(
                    code="body.anomaly.unexpected",
                    severity="warning",
                    address="part:c~d",
                    predicate="undeclared contact is visible",
                    required=False,
                    observed=True,
                    witness={"expected": False},
                ),
            ),
            (),
        ),
        VerdictDeltas(),
        (),
        0,
        0,
        (),
        (),
    )

    filtered = json.loads(session_cli._render(verdict, False))
    complete = json.loads(session_cli._render(verdict, True))

    assert tuple(
        diagnostic["code"] for diagnostic in filtered["diagnostics"]
    ) == ("body.anomaly.unexpected",)
    assert filtered["diagnostics_summary"] == {
        "total": 2,
        "expected": 1,
        "unexpected": 1,
        "shown": 1,
        "expected_filtered": True,
    }
    assert tuple(
        diagnostic["code"] for diagnostic in complete["diagnostics"]
    ) == ("body.anomaly.expected", "body.anomaly.unexpected")
    assert complete["diagnostics_summary"]["expected_filtered"] is False


def test_session_cli_emits_only_structured_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base_path = tmp_path / "base.json"
    transaction_path = tmp_path / "transaction.json"
    base_path.write_text(json.dumps(empty_spec("probe")), encoding="utf-8")
    transaction_path.write_text(
        json.dumps(
            {
                "base_txn": 0,
                "operations": [
                    {"op": "set", "addr": "meta@blend", "value": 0.007},
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(("session", str(base_path), str(transaction_path)))
    captured = capsys.readouterr()
    record = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert record["status"] == "accepted"
    assert record["changed_addresses"] == ["meta@blend"]
    assert record["binding_constraint"] is not None
    assert "blame" not in record
    expected_verdict = submit(
        start_session(json.loads(base_path.read_text()), tmp_path),
        json.loads(transaction_path.read_text()),
    ).verdict
    expected = _expected_cli_record(
        expected_verdict,
        all_diagnostics=False,
    )
    assert captured.out == json.dumps(
        expected,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def test_session_cli_blame_flag_attaches_post_verdict_section(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_path = tmp_path / "base.json"
    transaction_path = tmp_path / "transaction.json"
    base_path.write_text(json.dumps(empty_spec("probe")), encoding="utf-8")
    transaction_path.write_text(json.dumps(empty_spec("probe")), encoding="utf-8")
    expected_blame = {
        "ProbeObstruction(address=meta@blend)": {
            "blamed_addresses": ("meta@blend",),
            "budget": {"spent": 1, "limit": 512, "exhausted": False},
            "disposition": "localized",
        }
    }
    monkeypatch.setattr(
        session_cli,
        "_blame_section",
        lambda _submission, _spec_dir: expected_blame,
    )

    exit_code = main(("session", str(base_path), str(transaction_path), "--blame"))
    captured = capsys.readouterr()
    record = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == (
        "WARNING: the session CLI is stateless and the submitted document "
        "equals the base document; this fresh invocation has no cross-run "
        "delta. Rotate the last accepted edited document into `base` before "
        "the next invocation.\n"
    )
    assert record["blame"] == json.loads(json.dumps(expected_blame))
    assert record["schema"] == "golem.session.verdict/2"


def test_session_cli_blame_does_not_interrogate_request_rejection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_path = tmp_path / "base.json"
    transaction_path = tmp_path / "transaction.json"
    base_path.write_text(json.dumps(empty_spec("probe")), encoding="utf-8")
    transaction_path.write_text(
        json.dumps({"base_txn": "stale", "operations": []}),
        encoding="utf-8",
    )

    def forbidden_probe(_document, _spec_dir):
        raise AssertionError("request obstructions have no authored-body blame")

    monkeypatch.setattr(session_cli, "check_verdict", forbidden_probe)

    exit_code = main(("session", str(base_path), str(transaction_path), "--blame"))
    captured = capsys.readouterr()
    record = json.loads(captured.out)

    assert exit_code == 1
    assert captured.err == ""
    assert record["diagnostics"][0]["code"] == "session.invalid_base_txn"
    assert record["blame"] == {}


def test_session_cli_rejects_nonfinite_base_with_strict_json_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base_path = tmp_path / "base.json"
    transaction_path = tmp_path / "transaction.json"
    base_path.write_text(
        json.dumps({**empty_spec("probe"), "blend": float("nan")}),
        encoding="utf-8",
    )
    transaction_path.write_text(
        json.dumps(empty_spec("probe")),
        encoding="utf-8",
    )

    exit_code = main(("session", str(base_path), str(transaction_path)))
    captured = capsys.readouterr()
    record = json.loads(captured.out)

    assert exit_code == 1
    assert captured.err == ""
    assert record["status"] == "rejected"
    assert record["diagnostics"][0]["code"] == (
        "session.source.NonFiniteJsonNumber"
    )
