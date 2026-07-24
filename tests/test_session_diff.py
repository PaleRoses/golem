from __future__ import annotations

import json
from pathlib import Path

import pytest

from golem.kernel.body.canonical import _canonical_json
from golem.session import effect, ops
from golem.session.diff import diff_specs
from golem.session.state import AuthoredState, CompileObstructed


_KNIGHT = Path(__file__).parents[1] / "specs" / "knight_body.json"


def _knight() -> dict:
    return json.loads(_KNIGHT.read_text())


def _compile_diff(previous: dict, current: dict) -> tuple[tuple[ops.Op, ...], ops.ProgramPlan]:
    operations = diff_specs(previous, current)
    assert not isinstance(operations, ops.EditObstructed)
    compiled = ops.compile_program(previous, operations)
    assert isinstance(compiled, ops.ProgramPlan)
    assert _canonical_json(compiled.next_spec) == _canonical_json(current)
    return operations, compiled


def test_meta_set_descends_through_arbitrary_document_fields_and_inverts() -> None:
    previous = _knight()
    operation = ops.decode_op(
        {
            "op": "set",
            "addr": "meta@anatomy.overall.session_probe",
            "value": {"enabled": True},
        }
    )
    assert isinstance(operation, ops.SetOp)
    compiled = ops.compile_program(previous, (operation,))
    assert isinstance(compiled, ops.ProgramPlan)
    assert compiled.next_spec["anatomy"]["overall"]["session_probe"] == {
        "enabled": True
    }
    restored = ops.compile_program(compiled.next_spec, (compiled.edits[0].inverse,))
    assert isinstance(restored, ops.ProgramPlan)
    assert _canonical_json(restored.next_spec) == _canonical_json(previous)


def test_document_diff_uses_stable_record_address_for_field_edit() -> None:
    previous = _knight()
    current = {
        **previous,
        "skeleton": {
            **previous["skeleton"],
            "bones": [
                {**bone, "length": 0.41} if bone.get("id") == "forearm" else bone
                for bone in previous["skeleton"]["bones"]
            ],
        },
    }
    operations, _compiled = _compile_diff(previous, current)
    assert tuple(map(ops.to_json, operations)) == (
        {
            "op": "set",
            "addr": "skeleton/forearm@length",
            "value": 0.41,
        },
    )


def test_document_diff_uses_remove_and_add_at_for_stable_collection_edits() -> None:
    previous = _knight()
    probe = {
        "kind": "blob",
        "name": "probe",
        "t": 0.5,
        "size": [0.02, 0.02, 0.02],
    }
    current = {
        **previous,
        "skeleton": {
            **previous["skeleton"],
            "bones": [
                {
                    **bone,
                    "flesh": [
                        probe,
                        *(
                            record
                            for record in bone.get("flesh", [])
                            if record.get("name") != "hand"
                        ),
                    ],
                }
                if bone.get("id") == "forearm"
                else bone
                for bone in previous["skeleton"]["bones"]
            ],
        },
    }
    operations, _compiled = _compile_diff(previous, current)
    assert tuple(map(ops.to_json, operations)) == (
        {
            "op": "remove",
            "addr": "skeleton/forearm/hand",
        },
        {
            "op": "add_at",
            "kind": "flesh",
            "addr": "skeleton/forearm",
            "params": probe,
            "index": 0,
        },
    )


def test_document_diff_inverse_preserves_literal_selector_spelling() -> None:
    previous = _knight()
    probe = {"kind": "gencyl", "name": "flesh[0]"}
    current = {
        **previous,
        "skeleton": {
            **previous["skeleton"],
            "bones": [
                {**bone, "flesh": [*bone.get("flesh", []), probe]}
                if bone.get("id") == "forearm"
                else bone
                for bone in previous["skeleton"]["bones"]
            ],
        },
    }
    operations, compiled = _compile_diff(previous, current)

    assert tuple(map(ops.to_json, operations)) == (
        {
            "op": "add_at",
            "kind": "flesh",
            "addr": "skeleton/forearm",
            "params": probe,
            "index": len(
                next(
                    bone["flesh"]
                    for bone in previous["skeleton"]["bones"]
                    if bone.get("id") == "forearm"
                )
            ),
        },
    )
    assert ops.to_json(compiled.edits[0].inverse) == {
        "op": "remove",
        "addr": "skeleton/forearm/flesh%5B0%5D",
    }
    restored = ops.compile_program(
        compiled.next_spec,
        (compiled.edits[0].inverse,),
    )
    assert isinstance(restored, ops.ProgramPlan)
    assert _canonical_json(restored.next_spec) == _canonical_json(previous)


def test_unsupported_section_glues_with_nearest_meta_set() -> None:
    previous = _knight()
    current_anatomy = {
        **previous["anatomy"],
        "overall": {
            **previous["anatomy"]["overall"],
            "session_probe": {"enabled": True},
        },
    }
    current = {**previous, "anatomy": current_anatomy}
    operations, _compiled = _compile_diff(previous, current)
    assert tuple(map(ops.to_json, operations)) == (
        {
            "op": "set",
            "addr": "meta@anatomy",
            "value": current_anatomy,
        },
    )


def test_document_diff_glues_top_level_key_order_exactly() -> None:
    previous = _knight()
    keys = tuple(previous)
    reordered = (*keys[1:], keys[0])
    current = {key: previous[key] for key in reordered}
    operations, _compiled = _compile_diff(previous, current)
    assert operations
    assert all(operation.addr.record_source == "meta" for operation in operations)


def test_unknown_top_level_key_normalizes_before_compiler_rejection() -> None:
    previous = _knight()
    current = {
        **previous,
        "mystery_section": {"nested": {"value": 1}},
    }
    operations, compiled = _compile_diff(previous, current)
    assert tuple(map(ops.to_json, operations)) == (
        {
            "op": "set",
            "addr": "meta@mystery_section",
            "value": {"nested": {"value": 1}},
        },
    )
    outcome = effect.compile_authored(
        AuthoredState(compiled.next_spec, _KNIGHT.parent)
    )
    assert isinstance(outcome, CompileObstructed)
    assert "unknown top-level body key 'mystery_section'" in outcome.error


def test_malformed_top_level_shape_normalizes_before_compiler_rejection() -> None:
    previous = _knight()
    current = {**previous, "pose": []}
    operations, compiled = _compile_diff(previous, current)

    assert tuple(map(ops.to_json, operations)) == (
        {"op": "set", "addr": "meta@pose", "value": []},
    )
    outcome = effect.compile_authored(
        AuthoredState(compiled.next_spec, _KNIGHT.parent)
    )
    assert isinstance(outcome, CompileObstructed)
    assert "malformed section 'pose'" in outcome.error


@pytest.mark.parametrize("key", ("foo.bar", "foo@bar", "with space", "1x"))
def test_arbitrary_json_keys_glue_through_root_document_set(key: str) -> None:
    previous = _knight()
    current = {**previous, key: {"nested": True}}
    operations, compiled = _compile_diff(previous, current)

    assert tuple(map(ops.to_json, operations)) == (
        {"op": "set", "addr": "meta", "value": current},
    )
    restored = ops.compile_program(
        compiled.next_spec,
        (compiled.edits[0].inverse,),
    )
    assert isinstance(restored, ops.ProgramPlan)
    assert _canonical_json(restored.next_spec) == _canonical_json(previous)
