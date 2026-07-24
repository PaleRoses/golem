"""Tests for the total pure projection of relation obstructions to text and JSON."""

from __future__ import annotations

import json

import pytest

from golem.addressing.scope import ScopeKind
from golem.addressing.session import Address
from golem.addressing.session_grammar import parse_address
from golem.kernel.body.project import (
    obstruction_text,
    project_obstruction,
    project_rejected_body,
    rejected_body_text,
)
from golem.kernel.body.relations import (
    BranchCycleObstruction,
    BranchSearchBudgetObstruction,
    DuplicateRelationIdObstruction,
    FixedPlacementConflictObstruction,
    GoalRelationAuthorityObstruction,
    MalformedRelationObstruction,
    RelationalSolveExhaustedObstruction,
    RelationalSolverBudgetObstruction,
    UnderconstrainedRelationObstruction,
    UnknownRelationKindObstruction,
    UnreferenceableDerivedGeometryObstruction,
    UnresolvedRelationSelectorObstruction,
    UnsupportedRelationSelectorObstruction,
)
from golem.kernel.body.types import (
    EyeRule,
    MalformedEyeObstruction,
    MalformedMuscleBulkObstruction,
    MalformedMuscleDeclarationObstruction,
    MuscleBulkRule,
    MuscleDeclarationRule,
    RejectedBody,
)

_EVERY_OBSTRUCTION = (
    MalformedRelationObstruction("pose/relations/0", "mixed form"),
    DuplicateRelationIdObstruction("pose/relations/r", "r"),
    UnknownRelationKindObstruction("pose/relations/r", "abov", ("above",)),
    UnresolvedRelationSelectorObstruction("pose/relations/r/subject", "part:ghost", ("part:head",)),
    UnsupportedRelationSelectorObstruction("pose/relations/r/subject", "landmark:x",
                                           (ScopeKind.BONE, ScopeKind.PART)),
    UnreferenceableDerivedGeometryObstruction("pose/relations/r", "region:x", "post-boolean"),
    FixedPlacementConflictObstruction("fc", "part:foot", "part:chest", 0.05, -0.01),
    GoalRelationAuthorityObstruction("ov", "reach", ("arm",)),
    UnderconstrainedRelationObstruction("skeleton/head", 3, 1, 2, ("skeleton/head/t0",)),
    RelationalSolveExhaustedObstruction("skeleton/head", "local", 0, "ha", 0.05, -0.02, 750.0, 2),
    RelationalSolverBudgetObstruction("skeleton/head", 2000, 2000, 12.5),
    BranchCycleObstruction("ha", (("a~b",), ("b~a",)), 0.05, -0.01),
    BranchSearchBudgetObstruction("ha", 8, 8, 3.2),
)


@pytest.mark.parametrize("obstruction", _EVERY_OBSTRUCTION, ids=lambda o: type(o).__name__)
def test_project_obstruction_names_its_type_and_serializes(obstruction):
    projected = project_obstruction(obstruction)
    assert projected["obstruction"] == type(obstruction).__name__
    # Pure projection is JSON-total: no dataclass, enum, or tuple resists serialization.
    encoded = json.dumps(projected)
    assert type(obstruction).__name__ in encoded


@pytest.mark.parametrize("obstruction", _EVERY_OBSTRUCTION, ids=lambda o: type(o).__name__)
def test_obstruction_text_is_total_and_nonempty(obstruction):
    text = obstruction_text(obstruction)
    assert isinstance(text, str) and text


def test_unsupported_scope_kinds_project_as_strings():
    projected = project_obstruction(_EVERY_OBSTRUCTION[4])
    assert projected["supported_scope_kinds"] == ["bone", "part"]


def test_exhausted_text_carries_the_actionable_numbers():
    text = obstruction_text(_EVERY_OBSTRUCTION[9])
    assert "skeleton/head" in text and "'ha'" in text and "local" in text


def test_project_rejected_body_wraps_every_obstruction():
    rejected = RejectedBody((_EVERY_OBSTRUCTION[0], _EVERY_OBSTRUCTION[6]))
    projected = project_rejected_body(rejected)
    assert projected["rejected"] == "body"
    assert len(projected["obstructions"]) == 2
    assert json.dumps(projected)


def test_rejected_body_text_joins_one_line_per_obstruction():
    rejected = RejectedBody((_EVERY_OBSTRUCTION[0], _EVERY_OBSTRUCTION[6]))
    lines = rejected_body_text(rejected).splitlines()
    assert len(lines) == 2
    assert all(line for line in lines)


@pytest.mark.parametrize(
    ("rule", "authored", "predicate", "required"),
    (
        (
            MuscleDeclarationRule.DEFINITION,
            1.2,
            "muscle definition is finite and lies within the closed unit interval",
            (0.0, 1.0),
        ),
        (
            MuscleDeclarationRule.FINITE_BLEND,
            -0.1,
            "muscle blend is finite and non-negative",
            {"finite": True, "minimum": 0.0},
        ),
    ),
)
def test_quantitative_muscle_declaration_rules_project_exact_laws(
    rule: MuscleDeclarationRule,
    authored: float,
    predicate: str,
    required: object,
) -> None:
    projected = project_obstruction(
        MalformedMuscleDeclarationObstruction(
            "muscles/0/value",
            "value",
            rule,
            authored,
        )
    )

    assert projected["predicate"] == predicate
    assert projected["required"] == required
    assert projected["observed"] == authored


def test_positive_muscle_bulk_dimension_projects_strict_lower_bound() -> None:
    projected = project_obstruction(
        MalformedMuscleBulkObstruction(
            "muscles/0/bulk/absolute/width",
            "bicep",
            MuscleBulkRule.FINITE_POSITIVE_DIMENSION,
            -1.0,
        )
    )

    assert projected["predicate"] == (
        "muscle bulk dimension is finite and strictly positive"
    )
    assert projected["required"] == {
        "finite": True,
        "exclusive_minimum": 0.0,
    }
    assert projected["observed"] == -1.0


def test_synthetic_muscle_field_restricts_to_nearest_valid_session_address() -> None:
    projected = project_obstruction(
        MalformedMuscleDeclarationObstruction(
            "muscles/0/sections|bulk|mirror_of",
            "sections|bulk|mirror_of",
            MuscleDeclarationRule.PROFILE_FORM,
            {},
        )
    )

    assert projected["address"] == "meta@muscles[0]"
    assert isinstance(parse_address(projected["address"]), Address)


def test_malformed_eye_projects_authored_and_required_evidence() -> None:
    projected = project_obstruction(
        MalformedEyeObstruction(
            address="/eyes/2/brow_ridge/size",
            rule=EyeRule.FINITE_POSITIVE_VECTOR3,
            authored=(0.2, -0.1, 0.3),
            required={
                "arity": 3,
                "finite": True,
                "exclusive_minimum": 0.0,
            },
        )
    )

    assert projected["address"] == "meta@eyes[2].brow_ridge.size"
    assert projected["required"] == {
        "arity": 3,
        "finite": True,
        "exclusive_minimum": 0.0,
    }
    assert projected["observed"] == (0.2, -0.1, 0.3)
