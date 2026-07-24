"""Tests for the between relation (ShapeAssembly squeeze transliteration)."""

from __future__ import annotations

import pytest

from golem.kernel.body.compile import Compiler
from golem.kernel.body.relations import (
    AcceptedRelationDecode,
    MalformedRelationObstruction,
    RejectedRelationDecode,
    RelationKind,
    RelationalSolveExhaustedObstruction,
    UnsupportedRelationSelectorObstruction,
    decode_relations,
)
from golem.kernel.body.types import CompiledBody, RejectedBody

_ROOT = {"id": "root", "world": [0, 0, 0]}


def _decode_one(relation: dict):
    return decode_relations([relation])


def _between(**overrides) -> dict:
    return {
        "id": "span",
        "kind": "between",
        "subject": "bone:crossbar",
        "references": ["landmark:pillar_a/tail", "landmark:pillar_b/tail"],
        "solve": "subject",
        **overrides,
    }


# -- decode laws ----------------------------------------------------------- #
def test_between_decodes_with_two_landmark_references():
    result = _decode_one(_between())
    assert isinstance(result, AcceptedRelationDecode)
    declaration = result.declarations[0]
    assert declaration.kind is RelationKind.BETWEEN
    assert declaration.reference_selector == "landmark:pillar_a/tail"
    assert declaration.reference_b_selector == "landmark:pillar_b/tail"
    assert declaration.reference_b is not None


def test_between_aliases_absorb_to_canonical_kind():
    for alias in ("spanning", "bridges", "squeezed_between"):
        result = _decode_one(_between(kind=alias))
        assert isinstance(result, AcceptedRelationDecode)
        declaration = result.declarations[0]
        assert declaration.kind is RelationKind.BETWEEN
        assert declaration.authored_kind == alias


def test_between_rejects_singular_reference_key():
    relation = _between()
    del relation["references"]
    relation["reference"] = "landmark:pillar_a/tail"
    result = _decode_one(relation)
    assert isinstance(result, RejectedRelationDecode)
    obstruction = result.obstructions[0]
    assert isinstance(obstruction, MalformedRelationObstruction)
    assert "references" in obstruction.reason


@pytest.mark.parametrize("references", [[], ["landmark:pillar_a/tail"],
                                        ["landmark:a/tail", "landmark:b/tail", "landmark:c/tail"]])
def test_between_requires_exactly_two_references(references):
    result = _decode_one(_between(references=references))
    assert isinstance(result, RejectedRelationDecode)
    obstruction = result.obstructions[0]
    assert isinstance(obstruction, MalformedRelationObstruction)
    assert "exactly two references" in obstruction.reason


def test_between_rejects_reference_solve_policy():
    result = _decode_one(_between(solve="reference"))
    assert isinstance(result, RejectedRelationDecode)
    obstruction = result.obstructions[0]
    assert isinstance(obstruction, MalformedRelationObstruction)
    assert "negotiate" in obstruction.reason and "subject" in obstruction.reason


def test_single_reference_kinds_reject_plural_references_key():
    result = _decode_one(
        {"id": "al", "kind": "aligned", "subject": "bone:b",
         "references": ["bone:a", "bone:c"], "solve": "subject"}
    )
    assert isinstance(result, RejectedRelationDecode)
    obstruction = result.obstructions[0]
    assert isinstance(obstruction, MalformedRelationObstruction)
    assert "exactly one reference" in obstruction.reason


def test_between_subject_must_be_a_bone():
    result = _decode_one(_between(subject="part:crossf"))
    assert isinstance(result, RejectedRelationDecode)
    assert isinstance(
        result.obstructions[0], UnsupportedRelationSelectorObstruction
    )


def test_between_references_must_be_landmarks():
    result = _decode_one(
        _between(references=["bone:pillar_a", "landmark:pillar_b/tail"])
    )
    assert isinstance(result, RejectedRelationDecode)
    assert isinstance(
        result.obstructions[0], UnsupportedRelationSelectorObstruction
    )


# -- solve ------------------------------------------------------------------ #
def _pillars(separation: float, crossbar: dict) -> dict:
    return {
        "name": "span",
        "skeleton": {
            "root": _ROOT,
            "bones": [
                {"id": "pillar_a", "parent": "root", "attach": {"t": 0},
                 "length": 0.4, "rest_dir": [0, 1, 0]},
                {"id": "pillar_b", "parent": "root",
                 "attach": {"t": 0, "offset": [separation, 0, 0]},
                 "length": 0.4, "rest_dir": [0, 1, 0]},
                crossbar,
            ],
        },
        "pose": {"relations": [_between()]},
    }


def _row(compiled: CompiledBody, relation_id: str) -> dict:
    rows = compiled.receipt["solved"]["relations"]
    return next(row for row in rows if row["id"] == relation_id)


def test_between_satisfied_at_seed_selects_forward_convention():
    result = Compiler(_pillars(0.3, {
        "id": "crossbar", "parent": "root",
        "attach": {"at": "landmark:pillar_a/tail"},
        "length": 0.3, "rest_dir": [1, 0, 0],
    })).compile()
    assert isinstance(result, CompiledBody)
    row = _row(result, "span")
    assert row["canonical_kind"] == "between"
    assert row["convention"] == "between_forward"
    assert row["satisfied"]
    assert row["reference_b"] == "landmark:pillar_b/tail"


def test_between_selects_reversed_convention_from_swapped_references():
    spec = _pillars(0.3, {
        "id": "crossbar", "parent": "root",
        "attach": {"at": "landmark:pillar_a/tail"},
        "length": 0.3, "rest_dir": [1, 0, 0],
    })
    spec["pose"]["relations"] = [
        _between(references=["landmark:pillar_b/tail", "landmark:pillar_a/tail"])
    ]
    result = Compiler(spec).compile()
    assert isinstance(result, CompiledBody)
    row = _row(result, "span")
    assert row["convention"] == "between_reversed"
    assert row["satisfied"]


def test_between_swings_the_subject_onto_the_far_site():
    result = Compiler(_pillars(0.3, {
        "id": "crossbar", "parent": "root",
        "attach": {"at": "landmark:pillar_a/tail"},
        "length": 0.3, "rest_dir": [1, 0.25, 0],
    })).compile()
    assert isinstance(result, CompiledBody)
    row = _row(result, "span")
    assert row["satisfied"]
    assert row["observed"] <= 1.0e-3


def test_between_span_mismatch_is_honest_exhaustion():
    result = Compiler(_pillars(0.45, {
        "id": "crossbar", "parent": "root",
        "attach": {"at": "landmark:pillar_a/tail"},
        "length": 0.15, "rest_dir": [1, 0, 0],
    })).compile()
    assert isinstance(result, RejectedBody)
    obstruction = result.obstructions[0]
    assert isinstance(obstruction, RelationalSolveExhaustedObstruction)
    assert obstruction.relation_id == "span"
    assert obstruction.observed > 0.01


def test_single_reference_rows_carry_no_reference_b():
    result = Compiler({
        "name": "s",
        "skeleton": {"root": _ROOT, "bones": [
            {"id": "a", "parent": "root", "attach": {"t": 0},
             "length": 0.4, "rest_dir": [1, 0, 0]},
            {"id": "b", "parent": "root",
             "attach": {"at": "landmark:a/tail"},
             "length": 0.4, "rest_dir": [1, 0, 0]},
        ]},
        "pose": {"relations": [
            {"id": "al", "kind": "aligned", "subject": "bone:b",
             "reference": "bone:a", "solve": "subject"},
        ]},
    }).compile()
    assert isinstance(result, CompiledBody)
    assert _row(result, "al")["reference_b"] is None
