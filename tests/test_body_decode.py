from __future__ import annotations

import json
from pathlib import Path

import pytest

from golem.cli import main
from golem.kernel.body.compile import Compiler
from golem.kernel.body.project import obstruction_text, project_rejected_body
from golem.kernel.body.types import (
    FleshCompositionRule,
    MalformedFleshCompositionObstruction,
    MalformedBodySpecSectionObstruction,
    MisplacedBodySpecFieldObstruction,
    MissingBodySpecSectionObstruction,
    RejectedBody,
    UnknownBodySpecKeyObstruction,
)


def _minimal_body(**overrides: object) -> dict[str, object]:
    return {
        "name": "decode-probe",
        "dialect": "body/0.3",
        "skeleton": {
            "root": {"id": "root", "world": [0.0, 0.0, 0.0]},
            "bones": [],
        },
        **overrides,
    }


def _rejection(spec: dict[str, object]) -> RejectedBody:
    result = Compiler(spec).compile()
    assert isinstance(result, RejectedBody)
    return result


def test_unknown_top_level_intent_is_typed_and_suggests_contract() -> None:
    rejected = _rejection(
        _minimal_body(intent={"asserts": {"clauses": [{"kind": "probe"}]}})
    )

    (obstruction,) = rejected.obstructions
    assert obstruction == UnknownBodySpecKeyObstruction(
        address="spec/intent",
        authored_key="intent",
        closest_valid_candidates=("contract",),
    )
    assert obstruction_text(obstruction) == (
        "spec/intent: unknown top-level body key 'intent' (closest: contract)"
    )


def test_relations_inside_a_bone_pose_names_the_legal_parent() -> None:
    bone = {
        "id": "spine",
        "parent": "root",
        "length": 0.5,
        "rest_dir": [0.0, 1.0, 0.0],
        "pose": {"relations": [{"id": "lost-obligation"}]},
    }
    rejected = _rejection(
        _minimal_body(
            skeleton={
                "root": {"id": "root", "world": [0.0, 0.0, 0.0]},
                "bones": [bone],
            }
        )
    )

    (obstruction,) = rejected.obstructions
    assert obstruction == MisplacedBodySpecFieldObstruction(
        address="skeleton/spine/pose/relations",
        field="relations",
        authored_parent="bone pose",
        legal_parent="top-level pose",
        legal_address="pose/relations",
    )
    assert obstruction_text(obstruction) == (
        "skeleton/spine/pose/relations: relations is not valid inside a bone pose;"
        " move to top-level pose/relations"
    )


def test_skeleton_typo_is_missing_not_an_implementation_exception(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = {
        "name": "decode-probe",
        "dialect": "body/0.3",
        "skeletn": {
            "root": {"id": "root", "world": [0.0, 0.0, 0.0]},
            "bones": [],
        },
    }
    rejected = _rejection(spec)
    projected = project_rejected_body(rejected)
    spec_path = tmp_path / "skeletn.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    exit_code = main(("check", str(spec_path)))
    captured = capsys.readouterr()

    assert rejected.obstructions == (
        UnknownBodySpecKeyObstruction("spec/skeletn", "skeletn", ("skeleton",)),
        MissingBodySpecSectionObstruction("skeleton", "skeleton", "top-level body"),
    )
    assert tuple(
        obstruction["obstruction"] for obstruction in projected["obstructions"]
    ) == (
        "UnknownBodySpecKeyObstruction",
        "MissingBodySpecSectionObstruction",
    )
    assert exit_code == 1
    assert captured.err == ""
    assert "UnknownBodySpecKeyObstruction" in captured.out
    assert "MissingBodySpecSectionObstruction" in captured.out
    assert "KeyError" not in captured.out
    assert "TypeError" not in captured.out


def test_present_non_object_skeleton_is_malformed_not_missing() -> None:
    rejected = _rejection(_minimal_body(skeleton=[]))

    assert rejected.obstructions == (
        MalformedBodySpecSectionObstruction(
            "skeleton", "skeleton", "object", "list"
        ),
    )


@pytest.mark.parametrize(
    ("field", "value", "rule", "required", "predicate"),
    (
        (
            "blend",
            "heavy",
            FleshCompositionRule.FINITE_NON_NEGATIVE_BLEND,
            {"finite": True, "minimum": 0.0},
            "flesh blend is finite and non-negative",
        ),
        (
            "blend",
            -0.01,
            FleshCompositionRule.FINITE_NON_NEGATIVE_BLEND,
            {"finite": True, "minimum": 0.0},
            "flesh blend is finite and non-negative",
        ),
        (
            "operator",
            "round",
            FleshCompositionRule.OPERATOR,
            ("blend", "chamfer", "crease", "local_blend"),
            "flesh operator belongs to the closed composition vocabulary",
        ),
    ),
)
def test_flesh_composition_is_strictly_decoded(
    field: str,
    value: object,
    rule: FleshCompositionRule,
    required: object,
    predicate: str,
) -> None:
    bone = {
        "id": "slab",
        "parent": "root",
        "length": 0.5,
        "rest_dir": [0.0, 1.0, 0.0],
        "flesh": [
            {
                "kind": "box",
                "name": "facet",
                "size": [0.1, 0.1, 0.1],
                field: value,
            }
        ],
    }
    rejected = _rejection(
        _minimal_body(
            skeleton={
                "root": {"id": "root", "world": [0.0, 0.0, 0.0]},
                "bones": [bone],
            }
        )
    )
    (obstruction,) = rejected.obstructions
    assert obstruction == MalformedFleshCompositionObstruction(
        address=f"skeleton/slab/flesh[0]/{field}",
        rule=rule,
        authored_value=value,
        required=required,
    )
    projected = project_rejected_body(rejected)["obstructions"][0]
    assert projected["predicate"] == predicate
    assert projected["address"] == f"skeleton/slab/flesh[0]/{field}"
    assert rule.value in obstruction_text(obstruction)
