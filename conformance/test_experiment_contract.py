from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict, cast

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
PHASE_MINUS_ONE = ROOT / "phase-minus-one"
ORACLE_SEMANTIC_CHECK_IDS = (
    "handReportGoldenV1",
    "handReportGoldenV2",
    "v4Coherence",
    "mount",
    "bodyGlbByteExact",
    "mountedGlbByteExact",
    "plateComplex",
    "platedGlbSemantic",
    "platedGlbDriftBounded",
    "controlScorecard",
    "rawSymmetryObserved",
    "renderDriftBounded",
)


class ClauseSpec(TypedDict):
    scope: list[str]
    metric: str
    operator: str


class RevisionSpec(TypedDict):
    id: str
    targets: list[ClauseSpec]
    preserve: list[ClauseSpec]


class InitialBriefSpec(TypedDict):
    identityCriteria: list[ClauseSpec]


class CreatureSpec(TypedDict):
    id: str
    phase: str
    initialBrief: InitialBriefSpec
    revisions: list[RevisionSpec]


class FamilySplit(TypedDict):
    calibration: list[str]
    heldOut: list[str]


class MetricDefinition(TypedDict):
    id: str
    definition: str
    multiScopeLaw: str
    valueShape: str


class PreserveProfile(TypedDict):
    clauses: list[ClauseSpec]


class MeasurementContract(TypedDict):
    requiredSurfaceScopes: list[str]
    groundPlaneY: float
    surfaceSamplesPerSurface: int


class FamilySpec(TypedDict):
    split: FamilySplit
    creatures: list[CreatureSpec]
    preserveProfiles: list[PreserveProfile]
    metricSemantics: list[MetricDefinition]
    measurementContract: MeasurementContract


class PathHash(TypedDict):
    path: str
    sha256: str


class AuthoritySpec(TypedDict):
    sourceHashes: list[PathHash]
    artifactHashes: list[PathHash]


class ExperimentManifest(TypedDict):
    authority: AuthoritySpec
    family: FamilySpec


def object_from_unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = tuple(key for key, _value in pairs)
    assert len(keys) == len(frozenset(keys))
    return dict(pairs)


def read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(), object_pairs_hook=object_from_unique_pairs)
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def read_manifest() -> ExperimentManifest:
    return cast(ExperimentManifest, read_json_object(PHASE_MINUS_ONE / "manifest.json"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def definition_errors(
    schema: dict[str, object],
    definition_name: str,
    instance: object,
) -> tuple[object, ...]:
    definition_schema: dict[str, object] = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": f"#/$defs/{definition_name}",
    }
    return tuple(
        Draft202012Validator(
            definition_schema,
            format_checker=Draft202012Validator.FORMAT_CHECKER,
        ).iter_errors(instance)
    )


def manifest_clause_references(manifest: ExperimentManifest) -> frozenset[str]:
    family = manifest["family"]
    profile_references = tuple(
        f"/family/preserveProfiles/{profile_index}/clauses/{clause_index}"
        for profile_index, profile in enumerate(family["preserveProfiles"])
        for clause_index, _clause in enumerate(profile["clauses"])
    )
    initial_references = tuple(
        f"/family/creatures/{creature_index}/initialBrief/identityCriteria/{clause_index}"
        for creature_index, creature in enumerate(family["creatures"])
        for clause_index, _clause in enumerate(
            creature["initialBrief"]["identityCriteria"]
        )
    )
    revision_references = tuple(
        f"/family/creatures/{creature_index}/revisions/{revision_index}/{kind}/{clause_index}"
        for creature_index, creature in enumerate(family["creatures"])
        for revision_index, revision in enumerate(creature["revisions"])
        for kind in ("targets", "preserve")
        for clause_index, _clause in enumerate(revision[kind])
    )
    return frozenset((*profile_references, *initial_references, *revision_references))


def test_phase_minus_one_manifest_satisfies_closed_schema() -> None:
    schema = read_json_object(PHASE_MINUS_ONE / "schema.json")
    manifest = read_manifest()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(
        schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    ).validate(manifest)


def test_phase_minus_one_corpus_is_exactly_two_plus_six_by_three() -> None:
    manifest = read_manifest()
    family = manifest["family"]
    creatures = family["creatures"]
    creature_ids = tuple(map(lambda creature: creature["id"], creatures))
    revision_ids = tuple(
        revision["id"]
        for creature in creatures
        for revision in creature["revisions"]
    )
    calibration_ids = tuple(
        creature["id"] for creature in creatures if creature["phase"] == "calibration"
    )
    held_out_ids = tuple(
        creature["id"] for creature in creatures if creature["phase"] == "held_out"
    )
    assert (len(creature_ids), len(set(creature_ids))) == (8, 8)
    assert (len(revision_ids), len(set(revision_ids))) == (24, 24)
    assert calibration_ids == tuple(family["split"]["calibration"])
    assert held_out_ids == tuple(family["split"]["heldOut"])
    assert tuple(map(lambda creature: len(creature["revisions"]), creatures)) == (3,) * 8


def test_phase_minus_one_authority_hashes_match_frozen_files() -> None:
    manifest = read_manifest()
    authority = manifest["authority"]
    pinned = tuple(authority["sourceHashes"]) + tuple(authority["artifactHashes"])
    assert all(
        map(
            lambda entry: sha256((PHASE_MINUS_ONE / entry["path"]).resolve())
            == entry["sha256"],
            pinned,
        )
    )


def test_phase_minus_one_defines_every_used_metric_exactly_once() -> None:
    family = read_manifest()["family"]
    creatures = family["creatures"]
    profile_clauses = tuple(
        clause
        for profile in family["preserveProfiles"]
        for clause in profile["clauses"]
    )
    creature_clauses = tuple(
        clause
        for creature in creatures
        for clause in (
            *creature["initialBrief"]["identityCriteria"],
            *(
                clause
                for revision in creature["revisions"]
                for clause in (
                    *revision["targets"],
                    *revision["preserve"],
                )
            ),
        )
    )
    used_metrics = frozenset(
        clause["metric"] for clause in (*profile_clauses, *creature_clauses)
    )
    defined_metrics = tuple(
        definition["id"] for definition in family["metricSemantics"]
    )
    combined_metrics = frozenset(
        definition["id"]
        for definition in family["metricSemantics"]
        if definition["multiScopeLaw"] == "combined"
    )
    relational_metrics = frozenset(
        definition["id"]
        for definition in family["metricSemantics"]
        if definition["multiScopeLaw"] == "relational_pair"
    )
    shape_by_metric = {
        definition["id"]: definition["valueShape"]
        for definition in family["metricSemantics"]
    }
    vector_operators = frozenset(("max_distance", "max_angle"))
    vector_clauses = tuple(
        clause
        for clause in (*profile_clauses, *creature_clauses)
        if clause["operator"] in vector_operators
    )
    scalar_clauses = tuple(
        clause
        for clause in (*profile_clauses, *creature_clauses)
        if clause["operator"] not in vector_operators
    )
    relational_clauses = tuple(
        clause
        for clause in (*profile_clauses, *creature_clauses)
        if clause["metric"] in relational_metrics
    )
    assert len(defined_metrics) == len(set(defined_metrics))
    assert used_metrics == frozenset(defined_metrics)
    assert combined_metrics == frozenset(("combined_length", "combined_scale"))
    assert relational_metrics == frozenset(
        ("attachment_error", "wrist_junction_error")
    )
    assert all(shape_by_metric[clause["metric"]] == "vec3s" for clause in vector_clauses)
    assert all(shape_by_metric[clause["metric"]] == "scalars" for clause in scalar_clauses)
    assert all(
        len(tuple(scope for scope in clause["scope"] if scope != "ground_contact"))
        == 2
        for clause in relational_clauses
    )


def test_phase_minus_one_schema_rejects_premature_verdict_artifacts() -> None:
    schema = read_json_object(PHASE_MINUS_ONE / "schema.json")
    premature_verdict = {
        "recordType": "VerdictEvidence",
        "schemaVersion": 1,
        "evidenceId": "fabricated-pass",
        "manifestSha256": "0" * 64,
        "status": "Passed",
    }
    assert tuple(
        Draft202012Validator(
            schema,
            format_checker=Draft202012Validator.FORMAT_CHECKER,
        ).iter_errors(premature_verdict)
    )


def test_phase_minus_one_clause_references_are_exactly_the_manifest_paths() -> None:
    schema = read_json_object(PHASE_MINUS_ONE / "schema.json")
    definitions = cast(dict[str, object], schema["$defs"])
    clause_reference = cast(dict[str, object], definitions["ClauseReference"])
    schema_references = cast(list[str], clause_reference["enum"])
    expected_references = manifest_clause_references(read_manifest())
    assert len(schema_references) == len(set(schema_references))
    assert frozenset(schema_references) == expected_references
    assert (
        "/family/creatures/0/initialBrief/identityCriteria/999"
        not in schema_references
    )


def test_phase_minus_one_measurement_sidecar_has_one_closed_surface_cover() -> None:
    schema = read_json_object(PHASE_MINUS_ONE / "schema.json")
    definitions = cast(dict[str, object], schema["$defs"])
    sidecar = cast(dict[str, object], definitions["MeasurementSidecarEvidence"])
    sidecar_properties = cast(dict[str, object], sidecar["properties"])
    scopes = cast(dict[str, object], sidecar_properties["scopes"])
    scope_properties = cast(dict[str, object], scopes["properties"])
    required_scopes = cast(list[str], scopes["required"])
    contract = read_manifest()["family"]["measurementContract"]
    assert tuple(required_scopes) == tuple(contract["requiredSurfaceScopes"])
    assert frozenset(scope_properties) == frozenset(required_scopes)
    assert "ground_contact" not in scope_properties
    assert sidecar_properties["groundPlaneY"] == {"const": contract["groundPlaneY"]}
    surface = cast(dict[str, object], definitions["SurfaceMeasurement"])
    surface_properties = cast(dict[str, object], surface["properties"])
    samples = cast(dict[str, object], surface_properties["surfaceSamples"])
    assert samples["minItems"] == contract["surfaceSamplesPerSurface"]
    assert samples["maxItems"] == contract["surfaceSamplesPerSurface"]
    bilateral = cast(dict[str, object], definitions["BilateralScopeMeasurement"])
    bilateral_properties = cast(dict[str, object], bilateral["properties"])
    appendage = cast(dict[str, object], definitions["AppendageInstanceMeasurement"])
    assert frozenset(bilateral_properties) == frozenset(("left", "right"))
    assert "surface" in cast(list[str], appendage["required"])


def test_phase_minus_one_utc_timestamps_reject_untyped_strings_and_offsets() -> None:
    schema = read_json_object(PHASE_MINUS_ONE / "schema.json")
    assert not definition_errors(schema, "UtcTimestamp", "2026-07-12T07:15:30Z")
    assert definition_errors(schema, "UtcTimestamp", "garbage")
    assert definition_errors(schema, "UtcTimestamp", "2026-07-12T00:15:30-07:00")
    assert definition_errors(schema, "UtcTimestamp", "2026-02-31T07:15:30Z")


def test_phase_minus_one_oracle_success_cannot_contain_a_failed_check() -> None:
    schema = read_json_object(PHASE_MINUS_ONE / "schema.json")
    definitions = cast(dict[str, object], schema["$defs"])
    oracle = cast(dict[str, object], definitions["OracleEvidence"])
    success_variant = cast(list[dict[str, object]], oracle["oneOf"])[0]
    success_properties = cast(dict[str, object], success_variant["properties"])
    semantic_checks = cast(dict[str, object], success_properties["semanticChecks"])
    assert tuple(cast(list[str], semantic_checks["required"])) == ORACLE_SEMANTIC_CHECK_IDS
    check_observation = {
        "passed": True,
        "expected": "zero violations",
        "observed": "zero violations",
    }
    semantic_success = {
        "recordType": "OracleEvidence",
        "schemaVersion": 1,
        "evidenceId": "oracle-replay",
        "manifestSha256": "0" * 64,
        "verifiedAt": "2026-07-12T07:15:30Z",
        "status": "SemanticPassSealedObstructed",
        "semanticChecks": {
            check_id: check_observation for check_id in ORACLE_SEMANTIC_CHECK_IDS
        },
        "sealedObstructions": {
            "rawSymmetryGate": {
                "expected": ">=0.98",
                "observed": "0.9799926081064433",
            },
            "renderPixelDrift": {
                "expected": "pixel-exact",
                "observed": "four PNG pixels and five GIF pixels differ",
            },
            "platedGlbOrdering": {
                "expected": "byte-exact",
                "observed": "semantic colored-face match with different face serialization",
            },
        },
    }
    contradictory_success = {
        **semantic_success,
        "semanticChecks": {
            **cast(dict[str, object], semantic_success["semanticChecks"]),
            "handReportGoldenV2": {**check_observation, "passed": False},
        },
    }
    incomplete_success = {
        **semantic_success,
        "semanticChecks": {
            check_id: check_observation
            for check_id in ORACLE_SEMANTIC_CHECK_IDS
            if check_id != "handReportGoldenV2"
        },
    }
    assert not definition_errors(schema, "OracleEvidence", semantic_success)
    assert definition_errors(schema, "OracleEvidence", contradictory_success)
    assert definition_errors(schema, "OracleEvidence", incomplete_success)


def test_phase_minus_one_stage_product_enforces_budgets_without_fake_artifacts() -> None:
    schema = read_json_object(PHASE_MINUS_ONE / "schema.json")
    metrics = {
        "wallSeconds": 700,
        "activeSeconds": 650,
        "renderRounds": 3,
        "humanInterventions": 0,
        "humanAuthoringSeconds": 0,
        "directMeshWorkSeconds": 0,
        "authoritativeMeshEdits": 0,
    }
    unsupported_stage = {
        "metrics": metrics,
        "outcome": {
            "tag": "UnsupportedVocabulary",
            "proposalEvidenceId": "missing-vocabulary",
            "description": "No frozen-pilot expression exists.",
        },
    }
    over_budget_stage = {
        **unsupported_stage,
        "metrics": {**metrics, "wallSeconds": 721},
    }
    invented_stage_id = {**unsupported_stage, "stageId": "r99"}
    wrong_timeout_budget = {
        "metrics": metrics,
        "outcome": {"tag": "TimedOut", "budgetSeconds": 999},
    }
    receipt_only_completion = {
        "metrics": metrics,
        "outcome": {
            "tag": "Completed",
            "measurementEvidenceId": "sentinel-r1-measurement",
            "artifacts": [
                {"path": "receipt.json", "sha256": "0" * 64, "kind": "receipt"}
            ],
        },
    }
    completed_stage = {
        "metrics": metrics,
        "outcome": {
            "tag": "Completed",
            "measurementEvidenceId": "sentinel-r1-measurement",
            "sourceArtifact": {
                "path": "sentinel-r1.json",
                "sha256": "0" * 64,
                "kind": "source",
            },
            "outputMesh": {
                "path": "sentinel-r1.glb",
                "sha256": "0" * 64,
                "kind": "mesh",
            },
            "viewsArtifact": {
                "path": "sentinel-r1.png",
                "sha256": "0" * 64,
                "kind": "render",
            },
            "turntableArtifact": {
                "path": "sentinel-r1.webp",
                "sha256": "0" * 64,
                "kind": "turntable",
            },
        },
    }
    blocked_r1 = {
        "outcome": {"tag": "BlockedByPriorStage", "priorSlot": "initial"}
    }
    future_block = {"outcome": {"tag": "BlockedByPriorStage", "priorSlot": "r2"}}
    lawful_blocked_sequence = {
        "initial": unsupported_stage,
        "r1": blocked_r1,
        "r2": blocked_r1,
        "r3": blocked_r1,
    }
    impossible_continuation = {
        "initial": unsupported_stage,
        "r1": completed_stage,
        "r2": completed_stage,
        "r3": completed_stage,
    }
    assert not definition_errors(
        schema,
        "AttemptedRevisionStageEvidence",
        unsupported_stage,
    )
    assert definition_errors(
        schema,
        "AttemptedRevisionStageEvidence",
        over_budget_stage,
    )
    assert definition_errors(
        schema,
        "AttemptedRevisionStageEvidence",
        invented_stage_id,
    )
    assert definition_errors(
        schema,
        "AttemptedRevisionStageEvidence",
        wrong_timeout_budget,
    )
    assert not definition_errors(schema, "BlockedByInitialStageEvidence", blocked_r1)
    assert definition_errors(schema, "BlockedByInitialStageEvidence", future_block)
    assert not definition_errors(
        schema,
        "StageSequenceEvidence",
        lawful_blocked_sequence,
    )
    assert definition_errors(
        schema,
        "StageSequenceEvidence",
        impossible_continuation,
    )
    assert definition_errors(
        schema,
        "AttemptedRevisionStageEvidence",
        receipt_only_completion,
    )
    assert not definition_errors(
        schema,
        "AttemptedRevisionStageEvidence",
        completed_stage,
    )


def test_phase_minus_one_blind_diagnosis_cannot_leak_arm_specific_proposal() -> None:
    schema = read_json_object(PHASE_MINUS_ONE / "schema.json")
    technical = {
        "recordType": "TechnicalJudgeEvidence",
        "schemaVersion": 1,
        "evidenceId": "technical-sentinel-a",
        "manifestSha256": "0" * 64,
        "judgeId": "judge-a",
        "blindId": "p01",
        "visualEvidenceId": "visual-sentinel-a",
        "recordedAt": "2026-07-12T07:20:30Z",
        "technical": {
            "cleanupSeconds": 600,
            "meshSurgeryRequired": True,
            "blockingDefects": ["collapsed wrist topology"],
        },
        "rigging": {"tag": "NotAssessable", "reasons": ["static mesh only"]},
        "engineReadiness": {
            "tag": "NotAssessable",
            "reasons": ["no engine round trip"],
        },
    }
    arm_leaking_diagnosis = {
        **technical,
        "technical": {
            **cast(dict[str, object], technical["technical"]),
            "quarantineProposalEvidenceIds": ["rewrite-wrist-topology"],
        },
    }
    assert not definition_errors(schema, "TechnicalJudgeEvidence", technical)
    assert definition_errors(
        schema,
        "TechnicalJudgeEvidence",
        arm_leaking_diagnosis,
    )


def test_phase_minus_one_visual_choices_are_the_closed_creature_lineup() -> None:
    schema = read_json_object(PHASE_MINUS_ONE / "schema.json")
    visual = {
        "recordType": "VisualJudgeEvidence",
        "schemaVersion": 1,
        "evidenceId": "visual-sentinel-a",
        "manifestSha256": "0" * 64,
        "judgeId": "judge-a",
        "blindId": "p01",
        "sealedAt": "2026-07-12T07:15:30Z",
        "lineupCreatureId": "sentinel",
        "appeal": 5,
        "briefFidelity": 6,
        "confidence": 4,
    }
    invented_choice = {**visual, "lineupCreatureId": "dragon"}
    assert not definition_errors(schema, "VisualJudgeEvidence", visual)
    assert definition_errors(schema, "VisualJudgeEvidence", invented_choice)


def test_phase_minus_one_artifact_refinements_reject_wrong_surfaces() -> None:
    schema = read_json_object(PHASE_MINUS_ONE / "schema.json")
    mesh = {"path": "sentinel.glb", "sha256": "0" * 64, "kind": "mesh"}
    render = {**mesh, "path": "sentinel.png", "kind": "render"}
    blend = {**mesh, "path": "warden-master.blend", "kind": "blend"}
    blind_assignment = {
        "creatureId": "sentinel",
        "armId": "golem_pilot",
        "meshSha256": "0" * 64,
        "viewsSha256": "1" * 64,
        "turntableSha256": "2" * 64,
    }
    unglued_assignment = {
        "creatureId": "sentinel",
        "armId": "golem_pilot",
        "artifactSha256": "0" * 64,
    }
    assert not definition_errors(schema, "MeshArtifactReference", mesh)
    assert definition_errors(schema, "MeshArtifactReference", render)
    assert not definition_errors(schema, "BlendArtifactReference", blend)
    assert definition_errors(schema, "BlendArtifactReference", mesh)
    assert not definition_errors(schema, "BlindAssignment", blind_assignment)
    assert definition_errors(schema, "BlindAssignment", unglued_assignment)
