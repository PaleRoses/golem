"""Tests for the relational solve threaded through the body compiler."""

from __future__ import annotations

import numpy as np
import pytest

from golem.kernel.body.compile import Compiler
from golem.kernel.body.relations import (
    FixedPlacementConflictObstruction,
    GoalRelationAuthorityObstruction,
    MalformedRelationObstruction,
    RelationalSolveExhaustedObstruction,
)
from golem.kernel.body.types import CompiledBody, RejectedBody

_ROOT = {"id": "root", "world": [0, 0, 0]}


def _compile(skeleton_bones: list[dict], relations: list[dict] | None = None,
             goals: list[dict] | None = None):
    pose: dict = {}
    if relations is not None:
        pose["relations"] = relations
    if goals is not None:
        pose["goals"] = goals
    spec = {"name": "rel", "skeleton": {"root": _ROOT, "bones": skeleton_bones}}
    if pose:
        spec["pose"] = pose
    return Compiler(spec).compile()


def _row(compiled: CompiledBody, relation_id: str) -> dict:
    rows = compiled.receipt["solved"]["relations"]
    return next(row for row in rows if row["id"] == relation_id)


# -- feasible typed measurements ------------------------------------------ #
def test_coincident_shared_named_site_yields_coincident_heads():
    spec = {"name": "s", "skeleton": {"root": _ROOT, "bones": [
        {"id": "spine", "parent": "root", "attach": {"t": 0}, "length": 0.4, "rest_dir": [0, 0, 1]},
        {"id": "a", "parent": "root", "attach": {"at": "landmark:spine/tail"}, "length": 0.2,
         "rest_dir": [1, 0, 0], "flesh": [{"kind": "blob", "name": "af", "size": [0.1, 0.1, 0.1], "t": 1.0}]},
        {"id": "b", "parent": "root", "attach": {"at": "landmark:spine/tail"}, "length": 0.2,
         "rest_dir": [0, 1, 0], "flesh": [{"kind": "blob", "name": "bf", "size": [0.1, 0.1, 0.1], "t": 1.0}]},
    ]}}
    comp = Compiler(spec)
    result = comp.compile()
    assert isinstance(result, CompiledBody)
    assert np.allclose(comp.bones["a"]["head"], comp.bones["b"]["head"])


def test_above_is_satisfied_when_the_seed_clears_the_reference():
    result = _compile(
        [
            {"id": "chest", "parent": "root", "attach": {"t": 0}, "length": 0.6, "rest_dir": [0, 1, 0],
             "flesh": [{"kind": "blob", "name": "chest", "size": [0.2, 0.2, 0.2], "t": 0.5}]},
            {"id": "head", "parent": "root", "attach": {"at": "landmark:chest/tail"}, "length": 0.2,
             "rest_dir": [0, 1, 0], "flesh": [{"kind": "blob", "name": "headf", "size": [0.1, 0.1, 0.1], "t": 0.5}]},
        ],
        [{"id": "ha", "kind": "above", "subject": "part:headf", "reference": "part:chest",
          "distance": 0.05, "solve": "subject"}],
    )
    assert isinstance(result, CompiledBody)
    row = _row(result, "ha")
    assert row["canonical_kind"] == "above" and row["satisfied"]
    assert row["observed"] >= row["required"]


def test_below_is_satisfied_when_the_seed_clears_the_reference():
    result = _compile(
        [
            {"id": "top", "parent": "root", "attach": {"t": 0, "offset": [0, 0.6, 0]}, "length": 0.4,
             "rest_dir": [0, 1, 0], "flesh": [{"kind": "blob", "name": "topf", "size": [0.2, 0.2, 0.2], "t": 0.5}]},
            {"id": "bot", "parent": "root", "attach": {"at": "landmark:root/head"}, "length": 0.2,
             "rest_dir": [0, 1, 0], "flesh": [{"kind": "blob", "name": "botf", "size": [0.1, 0.1, 0.1], "t": 0.5}]},
        ],
        [{"id": "bl", "kind": "below", "subject": "part:botf", "reference": "part:topf",
          "distance": 0.05, "solve": "subject"}],
    )
    assert isinstance(result, CompiledBody)
    assert _row(result, "bl")["satisfied"]


def test_aligned_selects_antiparallel_convention_from_opposite_axes():
    result = _compile(
        [
            {"id": "a", "parent": "root", "attach": {"t": 0}, "length": 0.4, "rest_dir": [1, 0, 0]},
            {"id": "b", "parent": "root", "attach": {"at": "landmark:a/tail"}, "length": 0.4, "rest_dir": [-1, 0, 0]},
        ],
        [{"id": "al", "kind": "aligned", "subject": "bone:b", "reference": "bone:a", "solve": "subject"}],
    )
    assert isinstance(result, CompiledBody)
    row = _row(result, "al")
    assert row["convention"] == "aligned_antiparallel" and row["satisfied"]


def test_perpendicular_is_satisfied_for_orthogonal_axes():
    result = _compile(
        [
            {"id": "a", "parent": "root", "attach": {"t": 0}, "length": 0.4, "rest_dir": [1, 0, 0]},
            {"id": "b", "parent": "root", "attach": {"at": "landmark:a/tail"}, "length": 0.4, "rest_dir": [0, 0, 1]},
        ],
        [{"id": "pp", "kind": "perpendicular_to", "subject": "bone:b", "reference": "bone:a", "solve": "subject"}],
    )
    assert isinstance(result, CompiledBody)
    row = _row(result, "pp")
    assert row["canonical_kind"] == "perpendicular_to" and row["satisfied"]


def _mirror_bones(**extra_left) -> list[dict]:
    return [
        {"id": "spine", "parent": "root", "attach": {"t": 0}, "length": 0.3, "rest_dir": [0, 0, 1]},
        {"id": "arm_l", "parent": "spine", "attach": {"t": 1.0}, "length": 0.3, "rest_dir": [-1, 0, 0], **extra_left},
        {"id": "arm_r", "parent": "spine", "attach": {"t": 1.0}, "length": 0.3, "rest_dir": [1, 0, 0]},
    ]


def test_mirror_of_is_satisfied_for_a_sagittal_pair():
    result = _compile(
        _mirror_bones(),
        [{"id": "mr", "kind": "mirror_of", "subject": "bone:arm_r", "reference": "bone:arm_l",
          "solve": "negotiate"}],
    )
    assert isinstance(result, CompiledBody)
    assert _row(result, "mr")["satisfied"]


def test_solved_frames_stay_orthonormal_and_right_handed():
    comp = Compiler({"name": "m", "skeleton": {"root": _ROOT, "bones": _mirror_bones()},
                     "pose": {"relations": [{"id": "mr", "kind": "mirror_of", "subject": "bone:arm_r",
                                             "reference": "bone:arm_l", "solve": "negotiate"}]}})
    assert isinstance(comp.compile(), CompiledBody)
    for bid in ("arm_l", "arm_r"):
        frame = comp.bones[bid]["R"]
        assert np.allclose(frame @ frame.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(frame), 1.0, atol=1e-9)


# -- rejection paths ------------------------------------------------------ #
def _first(result: RejectedBody):
    assert isinstance(result, RejectedBody)
    return result.obstructions[0]


def test_fixed_attachment_conflict_reports_required_and_observed():
    result = _compile(
        [
            {"id": "chest", "parent": "root", "attach": {"t": 0}, "length": 0.6, "rest_dir": [0, 1, 0],
             "flesh": [{"kind": "blob", "name": "chest", "size": [0.2, 0.2, 0.2], "t": 0.5}]},
            {"id": "foot", "parent": "root", "attach": {"t": 0}, "length": 0.1, "rest_dir": [0, 1, 0],
             "flesh": [{"kind": "blob", "name": "footf", "size": [0.05, 0.05, 0.05], "t": 0.5}]},
        ],
        [{"id": "fc", "kind": "above", "subject": "part:footf", "reference": "part:chest",
          "distance": 0.05, "solve": "subject"}],
    )
    obstruction = _first(result)
    assert isinstance(obstruction, FixedPlacementConflictObstruction)
    assert obstruction.relation_id == "fc"
    assert obstruction.required == 0.05
    assert obstruction.observed != obstruction.required


def test_over_constrained_relation_exhausts_at_its_deepest_frontier():
    # head pinned at chest/tail (below the reference) AND required above -> infeasible.
    result = _compile(
        [
            {"id": "chest", "parent": "root", "attach": {"t": 0}, "length": 0.4, "rest_dir": [0, 1, 0],
             "flesh": [{"kind": "blob", "name": "chest", "size": [0.2, 0.2, 0.2], "t": 0.5}]},
            {"id": "head", "parent": "chest", "attach": {"at": "landmark:chest/tail"}, "length": 0.2,
             "rest_dir": [0, 1, 0], "flesh": [{"kind": "blob", "name": "headf", "size": [0.1, 0.1, 0.1], "t": 0.5}]},
        ],
        [{"id": "ha", "kind": "above", "subject": "part:headf", "reference": "part:chest",
          "distance": 0.05, "solve": "subject"}],
    )
    obstruction = _first(result)
    assert isinstance(obstruction, RelationalSolveExhaustedObstruction)
    assert obstruction.stage == "local"
    assert obstruction.attempted_evaluations >= 1


def test_relation_overlapping_a_goal_chain_is_rejected():
    result = _compile(
        [
            {"id": "chest", "parent": "root", "attach": {"t": 0}, "length": 0.4, "rest_dir": [0, 1, 0],
             "flesh": [{"kind": "blob", "name": "chest", "size": [0.2, 0.2, 0.2], "t": 0.5}]},
            {"id": "arm", "parent": "root", "attach": {"at": "landmark:chest/tail"}, "length": 0.3,
             "rest_dir": [1, 0, 0], "flesh": [{"kind": "blob", "name": "hand", "size": [0.1, 0.1, 0.1], "t": 1.0}]},
        ],
        [{"id": "ov", "kind": "above", "subject": "part:hand", "reference": "part:chest",
          "distance": 0.05, "solve": "subject"}],
        goals=[{"id": "reach", "chain": ["arm"]}],
    )
    obstruction = _first(result)
    assert isinstance(obstruction, GoalRelationAuthorityObstruction)
    assert obstruction.goal_id == "reach"


def test_explicit_mirror_of_plus_mirror_true_rejects_double_ownership():
    result = _compile(
        _mirror_bones(mirror=True),
        [{"id": "mr", "kind": "mirror_of", "subject": "bone:arm_r", "reference": "bone:arm_l",
          "solve": "negotiate"}],
    )
    obstruction = _first(result)
    assert isinstance(obstruction, MalformedRelationObstruction)


# -- legacy invariance ---------------------------------------------------- #
def test_non_relational_document_emits_no_relations_receipt_key():
    result = _compile([
        {"id": "spine", "parent": "root", "attach": {"t": 0}, "length": 0.4, "rest_dir": [0, 0, 1],
         "flesh": [{"kind": "blob", "name": "trunk", "size": [0.2, 0.2, 0.2], "t": 0.5}]},
        {"id": "arm", "parent": "spine", "attach": {"t": 1.0}, "length": 0.3, "rest_dir": [1, 0, 0]},
    ])
    assert isinstance(result, CompiledBody)
    assert "relations" not in result.receipt["solved"]
