"""Acceptance law (evidence-law Wall 006): a session verdict is accepted IFF
its evidence carries zero unexpected diagnostics, classified by the one
``witness.expected`` discriminator the CLI counts and filters by. Severity
never enters acceptance; it governs the analytic->assembly computation gate
only. Receipt: trial 4 verdict-43 reported ``status: accepted`` with six
undeclared anomaly warnings riding through.
"""
from __future__ import annotations

from pathlib import Path

from golem.assembly.obstructions import (
    ElementSurfaceFormationObstruction,
    MaskedIntegrityEvidence,
)
from golem.assembly.project import project_assembly_obstructions
from golem.senses.proprio.anomaly import Anomaly, AnomalyKind
from golem.kernel.engine.types import (
    SkinProjectionUnsatisfied,
    SkinRelaxationObstruction,
    SkinRootMultiplicity,
)
from golem.session.protocol import (
    AuthoringEvidence,
    Diagnostic,
    Margin,
    MarginDisposition,
    TransactionVerdict,
    VerdictDeltas,
    diagnostic_expected,
    witness_expected,
    start_session,
    _glue_evidence,
)
from golem.session.state import empty_spec


def _anomaly_diagnostic(*, expected: bool) -> Diagnostic:
    anomaly = Anomaly(
        kind=AnomalyKind.unintended_fusion,
        pair=("chest_mass", "shoulder_sling"),
        gap=0.0,
        magnitude=0.42,
        detail="probe",
        address="chest_mass~shoulder_sling",
        expected=expected,
    )
    return Diagnostic(
        code=f"body.anomaly.{anomaly.kind.value}",
        severity="warning",
        address=anomaly.address,
        predicate="derived body senses are non-anomalous",
        required="no anomaly",
        observed=anomaly.kind.value,
        witness=anomaly,
    )


def _verdict(evidence: AuthoringEvidence) -> TransactionVerdict:
    return TransactionVerdict(
        _evidence=evidence,
        deltas=VerdictDeltas(),
        changed_addresses=(),
        base_txn=0,
        txn=1,
        operations=(),
        journal=(),
    )


def test_unexpected_warning_blocks_acceptance() -> None:
    # The verdict-43 shape: six undeclared anomaly warnings rode through an
    # "accepted" status under the retired severity-only law.
    evidence = AuthoringEvidence(
        diagnostics=(_anomaly_diagnostic(expected=False),),
        margins=(),
    )
    verdict = _verdict(evidence)

    assert not evidence.accepted
    assert not verdict.accepted
    assert verdict.to_json()["status"] == "rejected"


def test_expected_observation_rides() -> None:
    evidence = AuthoringEvidence(
        diagnostics=(_anomaly_diagnostic(expected=True),),
        margins=(),
    )
    verdict = _verdict(evidence)

    assert evidence.accepted
    assert verdict.accepted
    assert verdict.to_json()["status"] == "accepted"


def test_warning_rejection_does_not_bind_a_satisfied_margin() -> None:
    margin = Margin(
        constraint="contract:height_floor",
        address="whole",
        predicate="height is above the floor",
        required=1.0,
        observed=2.0,
        margin=1.0,
        normalized_margin=1.0,
        unit="world_unit",
    )
    rejected = _glue_evidence(
        (_anomaly_diagnostic(expected=False),),
        (margin,),
    )
    accepted = _glue_evidence(
        (_anomaly_diagnostic(expected=True),),
        (margin,),
    )

    assert not rejected.accepted
    assert rejected.binding_constraint is None
    assert not rejected.margins[0].active
    assert accepted.accepted
    assert accepted.binding_constraint is not None
    assert accepted.binding_constraint.disposition is MarginDisposition.SATISFIED


def test_error_diagnostics_are_unexpected_and_block() -> None:
    diagnostic = Diagnostic(
        code="geometry.R-malformed-graph",
        severity="error",
        address="geometry",
        predicate="geometry rule satisfied",
        required="well-formed graph",
        observed="malformed",
        witness={"rule": "R-malformed-graph"},
    )
    evidence = AuthoringEvidence(diagnostics=(diagnostic,), margins=())

    assert not diagnostic_expected(diagnostic)
    assert not evidence.accepted


def test_clean_evidence_is_accepted() -> None:
    assert AuthoringEvidence(diagnostics=(), margins=()).accepted


def test_expected_discriminator_survives_json_projection() -> None:
    witnesses: tuple[object, ...] = (
        _anomaly_diagnostic(expected=True).witness,
        _anomaly_diagnostic(expected=False).witness,
        {"expected": True},
        {"expected": False},
        {"rule": "no-expected-key"},
        None,
        "opaque",
    )
    for witness in witnesses:
        diagnostic = Diagnostic(
            code="probe",
            severity="warning",
            address="probe",
            predicate="probe",
            required="probe",
            observed="probe",
            witness=witness,
        )
        projected_witness = diagnostic.to_json()["witness"]

        assert witness_expected(projected_witness) is witness_expected(witness)
        assert diagnostic_expected(diagnostic) is witness_expected(
            projected_witness
        )


def test_failed_contract_exposes_slack_and_honest_help_unavailability(
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
                }
            ],
        },
    }

    session = start_session(document, tmp_path)
    evidence = session.evidence
    diagnostic = next(
        item
        for item in evidence.diagnostics
        if item.code == "contract.constraint_failed"
    )
    help_record = diagnostic.to_json()["help"]
    margin = next(
        item
        for item in evidence.margins
        if item.constraint == "contract:height_floor"
    )

    assert help_record["operations"] == ()
    assert "not a writable session address" in help_record["unavailable"]
    assert margin.disposition is MarginDisposition.VIOLATED
    assert margin.margin < 0.0
    assert margin.normalized_margin < 0.0
    assert margin.active
    assert margin.authoring_addresses == ()
    assert evidence.binding_constraint is margin


def test_surface_formation_rejection_exposes_local_crossing_slack() -> None:
    crossing_distances = ((0.04, 0.08, 0.12),)
    nested = SkinRelaxationObstruction(
        ("forelimb", "hindlimb"),
        SkinProjectionUnsatisfied(
            SkinRootMultiplicity(
                (17,),
                17,
                3,
                3,
                crossing_distances,
            )
        ),
    )
    obstruction = ElementSurfaceFormationObstruction(
        "body",
        (nested,),
        vocabulary_violations=(
            {
                "part": "shin",
                "rule": "R-8-feature-size",
                "detail": "probe",
            },
        ),
        masked_integrity=MaskedIntegrityEvidence(
            ("component_count", "watertight"),
            "surface formation rejected before a mesh exists",
        ),
    )

    (projection,) = project_assembly_obstructions(obstruction)

    assert projection["predicate"] == (
        "each local flesh-to-scaffold projection segment crosses the skin "
        "offset field exactly once"
    )
    assert projection["required"] == (
        "one crossing per local projection segment"
    )
    assert projection["observed"] == {
        "multiple_crossing_segments": 1,
        "minimum_crossing_count": 3,
        "maximum_crossing_count": 3,
        "crossing_distances_world": crossing_distances,
    }
    assert projection["vocabulary_violations"] == (
        {
            "part": "shin",
            "rule": "R-8-feature-size",
            "detail": "probe",
        },
    )
    assert projection["masked_integrity"] == {
        "checks": ("component_count", "watertight"),
        "reason": obstruction.masked_integrity.reason,
    }
