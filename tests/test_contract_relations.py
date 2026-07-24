from __future__ import annotations

import jsonschema
import pytest

from golem.contract import body_schema, render_section, section_names
from golem.kernel.body.relations import (
    RELATION_ALIASES,
    RELATION_CONVENTIONS,
    SUPPORTED_RELATION_SCOPES,
    ConventionId,
    RelationKind,
    RelationSolvePolicy,
)
from golem.kernel.body.relation_solve import DeepeningStage, RelationalSolveConfig
from golem.kernel.body.types import DIALECT as BODY_DIALECT, _DOF


def test_relations_section_is_taught_first() -> None:
    names = section_names()
    assert "relations" in names
    assert names.index("relations") == names.index("primer") + 1
    assert names.index("relations") < names.index("vocabulary")


def test_relations_section_renders_live_pose_kinds_and_aliases() -> None:
    rendered = render_section("relations")
    pose_block = rendered.split("POSE RELATIONS")[1].split(
        "PLACEMENT VERSUS ASSERTION"
    )[0]
    for kind in RelationKind:
        if kind is RelationKind.ATTACH_AT_NAMED_SITE:
            assert kind.value not in pose_block
            continue
        assert kind.value in pose_block
    for alias, kind in RELATION_ALIASES.items():
        assert alias in rendered
        if kind is RelationKind.ATTACH_AT_NAMED_SITE:
            assert alias not in pose_block


def test_relations_section_renders_live_capability_and_policies() -> None:
    rendered = render_section("relations")
    for kind, scopes in SUPPORTED_RELATION_SCOPES.items():
        if kind is RelationKind.ATTACH_AT_NAMED_SITE:
            continue
        for scope in scopes:
            assert scope.value in rendered
    for policy in RelationSolvePolicy:
        assert policy.value in rendered


def test_relations_section_renders_live_conventions_and_stages() -> None:
    rendered = render_section("relations")
    for kind in RELATION_CONVENTIONS:
        for convention in RELATION_CONVENTIONS[kind]:
            assert convention.value in rendered
    for convention in ConventionId:
        assert convention.value in rendered
    for stage in DeepeningStage:
        assert stage.value in rendered


def test_relations_section_projects_live_solver_constants() -> None:
    config = RelationalSolveConfig()
    rendered = render_section("relations")
    for value in (
        config.position_tolerance,
        config.angle_tolerance_deg,
        config.singular_value_tolerance,
        config.maximum_evaluations,
        config.maximum_active_set_resolves,
    ):
        assert str(value) in rendered


def test_relations_section_names_every_obstruction() -> None:
    rendered = render_section("relations")
    for name in (
        "MalformedRelationObstruction",
        "DuplicateRelationIdObstruction",
        "UnknownRelationKindObstruction",
        "UnresolvedRelationSelectorObstruction",
        "UnsupportedRelationSelectorObstruction",
        "UnreferenceableDerivedGeometryObstruction",
        "FixedPlacementConflictObstruction",
        "GoalRelationAuthorityObstruction",
        "UnderconstrainedRelationObstruction",
        "RelationalSolveExhaustedObstruction",
        "RelationalSolverBudgetObstruction",
        "BranchCycleObstruction",
        "BranchSearchBudgetObstruction",
    ):
        assert name in rendered


def test_relations_teach_above_below_as_assertions_not_positioners() -> None:
    rendered = render_section("relations")
    assert "PLACEMENT VERSUS ASSERTION" in rendered
    block = rendered.split("PLACEMENT VERSUS ASSERTION")[1].split(
        "TWO-REFERENCE RELATIONS"
    )[0]
    assert "verified assertions" in block
    assert "never move a bone" in block
    assert "FixedPlacementConflictObstruction" in block
    assert "RelationalSolveExhaustedObstruction" in block


def test_relations_state_mirror_sagittal_plane_requirement() -> None:
    rendered = render_section("relations")
    assert "sagittal plane" in rendered
    assert "back-axis" in rendered


# -- between (two-reference kind) ------------------------------------------- #
def test_between_surfaces_in_every_projected_section() -> None:
    for section in ("relations", "vocabulary", "schema"):
        assert RelationKind.BETWEEN.value in render_section(section)
    between_aliases = [
        alias
        for alias, kind in RELATION_ALIASES.items()
        if kind is RelationKind.BETWEEN
    ]
    assert between_aliases, "expected `between` to expose lexical aliases"
    for alias in between_aliases:
        for section in ("relations", "vocabulary", "schema"):
            assert alias in render_section(section), (section, alias)


def test_relations_section_teaches_between_as_two_reference_kind() -> None:
    rendered = render_section("relations")
    assert "TWO-REFERENCE RELATIONS" in rendered
    block = rendered.split("TWO-REFERENCE RELATIONS")[1].split(
        "SELECTOR CAPABILITY"
    )[0]
    assert "references" in block
    for convention in RELATION_CONVENTIONS[RelationKind.BETWEEN]:
        assert convention.value in block
    assert "RelationalSolveExhaustedObstruction" in block
    assert '"kind": "between"' in block
    assert '"references"' in block


def test_between_capability_line_projects_subject_and_reference_scopes() -> None:
    rendered = render_section("relations")
    capability = rendered.split("SELECTOR CAPABILITY")[1].split("SOLVE POLICY")[0]
    line = next(
        row
        for row in capability.splitlines()
        if row.startswith(f"- {RelationKind.BETWEEN.value}:")
    )
    for scope in SUPPORTED_RELATION_SCOPES[RelationKind.BETWEEN]:
        assert scope.value in line
    assert "subject" in line


def test_primer_teaches_relations_before_the_fixed_escape_hatch() -> None:
    rendered = render_section("primer")
    assert "RELATIONAL PLACEMENT" in rendered
    assert "escape hatch" in rendered
    assert rendered.index("consequence") < rendered.index("escape hatch")


def test_vocabulary_renders_live_relation_closed_sets_and_sugar() -> None:
    rendered = render_section("vocabulary")
    assert "RelationKind:" in rendered
    assert "RelationSolvePolicy:" in rendered
    assert "ConventionId:" in rendered
    assert "surface sugar" in rendered
    for alias, kind in RELATION_ALIASES.items():
        assert f"{alias}->{kind.value}" in rendered


def test_receipts_document_the_reference_b_key() -> None:
    rendered = render_section("receipts")
    assert "reference_b" in rendered


def test_schema_attach_has_two_disjoint_forms() -> None:
    forms = body_schema()["$defs"]["attach"]["oneOf"]
    assert len(forms) == 2
    named_site, fixed = forms
    assert named_site["required"] == ["at"]
    assert named_site["additionalProperties"] is False
    assert set(fixed["properties"]) == {"t", "offset"}
    assert fixed["additionalProperties"] is False


def _relation_branches() -> tuple[dict[str, object], dict[str, object]]:
    branches = body_schema()["$defs"]["relation"]["oneOf"]
    single = next(b for b in branches if "reference" in b["properties"])
    between = next(b for b in branches if "references" in b["properties"])
    return single, between


def test_schema_relation_single_branch_sources_enums_from_live_types() -> None:
    single, _ = _relation_branches()
    assert single["properties"]["solve"]["enum"] == [
        policy.value for policy in RelationSolvePolicy
    ]
    kind_enum = single["properties"]["kind"]["enum"]
    assert RelationKind.ATTACH_AT_NAMED_SITE.value not in kind_enum
    assert RelationKind.BETWEEN.value not in kind_enum
    for kind in RelationKind:
        if kind in (RelationKind.ATTACH_AT_NAMED_SITE, RelationKind.BETWEEN):
            continue
        assert kind.value in kind_enum
    assert single["additionalProperties"] is False
    assert "distance" in single["properties"]


def test_schema_relation_between_branch_is_two_reference_and_peer_solved() -> None:
    _, between = _relation_branches()
    assert between["properties"]["kind"]["enum"][0] == RelationKind.BETWEEN.value
    references = between["properties"]["references"]
    assert references["minItems"] == 2 and references["maxItems"] == 2
    assert between["required"] == ["id", "kind", "subject", "references", "solve"]
    assert between["properties"]["solve"]["enum"] == [
        RelationSolvePolicy.SUBJECT.value,
        RelationSolvePolicy.NEGOTIATE.value,
    ]
    assert "reference" not in between["properties"]
    assert "distance" not in between["properties"]
    assert between["additionalProperties"] is False


def _document(
    attach: dict[str, object] | None = None,
    relations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    dof = next(iter(_DOF))
    forearm: dict[str, object] = {
        "id": "forearm",
        "parent": "upper_arm",
        "length": 0.42,
        "rest_dir": [0.1, 0.0, 1.0],
        "joint": {"dof": dof},
    }
    if attach is not None:
        forearm["attach"] = attach
    document: dict[str, object] = {
        "name": "probe",
        "dialect": BODY_DIALECT,
        "skeleton": {
            "root": {"id": "root", "world": [0.0, 0.0, 0.0]},
            "bones": [forearm],
        },
    }
    if relations is not None:
        document["pose"] = {"relations": relations}
    return document


def test_schema_accepts_both_attach_forms_and_rejects_mixed() -> None:
    schema = body_schema()
    jsonschema.validate(_document({"at": "landmark:upper_arm/wrist"}), schema)
    jsonschema.validate(_document({"t": 1.0, "offset": [0.0, 0.0, 0.0]}), schema)
    jsonschema.validate(_document({}), schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            _document({"at": "landmark:upper_arm/wrist", "t": 1.0}), schema
        )


def test_schema_accepts_typed_pose_relations() -> None:
    schema = body_schema()
    jsonschema.validate(
        _document(
            relations=[
                {
                    "id": "head_above_chest",
                    "kind": "above",
                    "subject": "part:head",
                    "reference": "part:chest",
                    "distance": 0.04,
                    "solve": "subject",
                }
            ]
        ),
        schema,
    )
    for bad in (
        {
            "id": "bad",
            "kind": "not_a_relation",
            "subject": "part:head",
            "reference": "part:chest",
            "solve": "subject",
        },
        {
            "id": "bad_policy",
            "kind": "above",
            "subject": "part:head",
            "reference": "part:chest",
            "solve": "whatever",
        },
    ):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(_document(relations=[bad]), schema)


def test_schema_accepts_between_with_two_references() -> None:
    schema = body_schema()
    for solve in ("subject", "negotiate"):
        jsonschema.validate(
            _document(
                relations=[
                    {
                        "id": "collar",
                        "kind": "between",
                        "subject": "bone:neck",
                        "references": [
                            "landmark:torso/head",
                            "landmark:skull/tail",
                        ],
                        "solve": solve,
                    }
                ]
            ),
            schema,
        )
    for alias in (
        alias
        for alias, kind in RELATION_ALIASES.items()
        if kind is RelationKind.BETWEEN
    ):
        jsonschema.validate(
            _document(
                relations=[
                    {
                        "id": "collar",
                        "kind": alias,
                        "subject": "bone:neck",
                        "references": [
                            "landmark:torso/head",
                            "landmark:skull/tail",
                        ],
                        "solve": "subject",
                    }
                ]
            ),
            schema,
        )


def test_schema_rejects_malformed_between_and_plural_single_reference() -> None:
    schema = body_schema()
    bad_records = (
        # singular reference on between
        {
            "id": "collar",
            "kind": "between",
            "subject": "bone:neck",
            "reference": "landmark:torso/head",
            "solve": "subject",
        },
        # wrong reference count
        {
            "id": "collar",
            "kind": "between",
            "subject": "bone:neck",
            "references": ["landmark:torso/head"],
            "solve": "subject",
        },
        # reference policy on between (peers, no reference side)
        {
            "id": "collar",
            "kind": "between",
            "subject": "bone:neck",
            "references": ["landmark:torso/head", "landmark:skull/tail"],
            "solve": "reference",
        },
        # distance is not lawful on between
        {
            "id": "collar",
            "kind": "between",
            "subject": "bone:neck",
            "references": ["landmark:torso/head", "landmark:skull/tail"],
            "solve": "subject",
            "distance": 0.1,
        },
        # plural references on a single-reference kind
        {
            "id": "al",
            "kind": "aligned",
            "subject": "bone:b",
            "references": ["bone:a", "bone:c"],
            "solve": "subject",
        },
    )
    for record in bad_records:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(_document(relations=[record]), schema)


def test_schema_named_site_alias_is_rejected_as_pose_relation_kind() -> None:
    single, between = _relation_branches()
    for alias, kind in RELATION_ALIASES.items():
        if kind is RelationKind.ATTACH_AT_NAMED_SITE:
            assert alias not in single["properties"]["kind"]["enum"]
            assert alias not in between["properties"]["kind"]["enum"]
