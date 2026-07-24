"""M0 acceptance tests for the GOLEM session op/journal layer.

These pin the M0 gate spelled out across ``golem/session/{ops,journal,state}``:
every mutation flows through :func:`ops.apply_op`, which returns the concrete
inverse, so undo/redo/replay are correct by construction and the canonical
bytes are the only equality that matters. The suite exercises, in order:

  1. a seeded property -- 20 random valid op sequences, applied through the
     journal, undo-all restores the OPENING canonical bytes byte-for-byte;
  2. replay of a journal that interleaves undo/redo reproduces the FINAL bytes;
  3. per-op-type inverse round-trips (set, set-on-new-key, add/remove flesh,
     add/remove bone, rename-with-ref-rewrite, pose auto-create);
  4. rejections are validate-before-mutate: the spec is byte-untouched;
  5. the JSONL journal round-trips through disk and replays to the same bytes;
  6. an end-to-end sanity check that a few ops still compile clean.

Canonical bytes come from ``body._canonical_json`` (the same function
``SessionState.canonical`` calls), so tests 1-5 compare bare spec dicts without
paying for a compile -- only test 6 recompiles (opening + one explicit call).
All spec paths resolve through ``golem.paths`` so the suite runs from any cwd.
"""

from __future__ import annotations

import itertools
import json
import random

import pytest

from golem import paths as _paths
from golem.kernel import body
from golem.session import journal, ops
from golem.session.journal import Journal
from golem.session.state import SessionState, empty_spec

_KNIGHT = _paths.SPECS / "knight_body.json"


# --------------------------------------------------------------------------- #
# Helpers: a fresh authored spec, its canonical bytes, and a commit shorthand. #
# --------------------------------------------------------------------------- #
def _fresh() -> dict:
    """A fresh, independent copy of the authored knight spec (no compile)."""
    return json.loads(_KNIGHT.read_text())


def _canon(spec: dict) -> str:
    """The canonical bytes of a bare spec -- what ``SessionState.canonical``
    returns, without constructing a state (so no recompile)."""
    return body._canonical_json(spec)


def _planned_undo(
    spec: dict,
    operation_record: dict[str, object],
) -> tuple[ops.ProgramPlan, journal.JournalTransition]:
    operation = ops.decode_op(operation_record)
    assert not isinstance(operation, ops.EditObstructed)
    program = ops.compile_program(spec, (operation,))
    assert isinstance(program, ops.ProgramPlan)
    committed = journal.commit_program(journal.empty(), (operation,), program)
    undo = journal.plan_undo(committed.state, program.next_spec)
    assert isinstance(undo, journal.JournalTransition)
    return program, undo


def _commit(j: Journal, spec: dict, op: dict) -> dict:
    """Apply ``op`` and journal it as an ordinary transaction."""
    inverse, label = ops.apply_op(spec, op)
    j.commit_op(op, inverse, label)
    return inverse


# --------------------------------------------------------------------------- #
# A pool of valid ops for the seeded property test.                            #
# --------------------------------------------------------------------------- #
_LEN_BONES = ("forearm", "shin", "thigh", "upper_arm")   # every bone has @length
_POSE_BONES = ("upper_arm", "shin")                      # pose auto-creates
_FLESH_BONES = ("forearm", "spine_upper", "helm")        # bones that own flesh


def _pool_op(rng: random.Random, counter) -> list[dict]:
    """One draw from the valid-op pool. Returns a list of ops (a flesh add is
    sometimes paired with its own remove, all other draws are single ops)."""
    kind = rng.choice(("length", "pose", "blend", "flesh"))
    if kind == "length":
        bone = rng.choice(_LEN_BONES)
        return [{"op": "set", "addr": f"skeleton/{bone}@length",
                 "value": round(rng.uniform(0.2, 0.8), 6)}]
    if kind == "pose":
        bone = rng.choice(_POSE_BONES)
        return [{"op": "set", "addr": f"pose/{bone}@pitch",
                 "value": round(rng.uniform(-30, 30), 6)}]
    if kind == "blend":
        return [{"op": "set", "addr": "meta@blend",
                 "value": round(rng.uniform(0.004, 0.01), 6)}]
    # flesh add with a globally unique generated name; sometimes remove it too.
    bone = rng.choice(_FLESH_BONES)
    name = f"probe_{next(counter)}"
    out = [{"op": "add", "kind": "flesh", "addr": f"skeleton/{bone}",
            "params": {"kind": "gencyl", "name": name}}]
    if rng.random() < 0.5:
        out.append({"op": "remove", "addr": f"skeleton/{bone}/{name}"})
    return out


# --------------------------------------------------------------------------- #
# 1. Seeded property: apply a random valid sequence, undo-all, opening bytes.   #
# --------------------------------------------------------------------------- #
def test_property_undo_all_restores_opening_bytes():
    rng = random.Random(7)
    counter = itertools.count()
    opening = _canon(_fresh())
    for _ in range(20):
        spec = _fresh()
        journal = Journal()
        seq: list[dict] = []
        target = rng.randint(1, 10)
        while len(seq) < target:
            seq.extend(_pool_op(rng, counter))
        seq = seq[:10]                              # up to 10 ops per sequence
        for op in seq:
            _commit(journal, spec, op)
        while journal.undoable() is not None:       # undo-all, strict LIFO
            journal.apply_undo(spec)
        assert _canon(spec) == opening


# --------------------------------------------------------------------------- #
# 2. Replay of an undo/redo-interleaved journal reproduces the final bytes.     #
# --------------------------------------------------------------------------- #
def test_replay_reproduces_final_bytes_through_undo_redo():
    spec = _fresh()
    journal = Journal()
    _commit(journal, spec, {"op": "set", "addr": "skeleton/forearm@length", "value": 0.4})
    _commit(journal, spec, {"op": "set", "addr": "pose/upper_arm@pitch", "value": 12.0})
    _commit(journal, spec, {"op": "set", "addr": "meta@blend", "value": 0.008})
    journal.apply_undo(spec)     # undo the blend set (blend back to authored)
    journal.apply_undo(spec)     # undo the pose set  (removes the pose entry)
    journal.apply_redo(spec)     # redo the pose set  (re-adds the pose entry)
    _commit(journal, spec, {"op": "set", "addr": "skeleton/shin@length", "value": 0.5})
    final = _canon(spec)

    # Replay is a plain fold of apply_op over entry["applied"] -- undo and redo
    # entries are ordinary transactions, so a fresh spec reaches the same bytes.
    fresh = _fresh()
    Journal.replay(fresh, journal.entries)
    assert _canon(fresh) == final


# --------------------------------------------------------------------------- #
# 3. Inverse round-trips, one op type per test.                                #
# --------------------------------------------------------------------------- #
def test_inverse_set_number_roundtrips():
    spec = _fresh()
    opening = _canon(spec)
    authored_length = next(
        bone["length"]
        for bone in spec["skeleton"]["bones"]
        if bone["id"] == "forearm"
    )
    inverse = ops.apply_op(spec, {"op": "set", "addr": "skeleton/forearm@length",
                                  "value": 0.37})[0]
    assert _canon(spec) != opening
    assert inverse == {
        "op": "set",
        "addr": "skeleton/forearm@length",
        "value": authored_length,
    }
    ops.apply_op(spec, inverse)
    assert _canon(spec) == opening


def test_inverse_set_on_new_key_is_unset():
    spec = _fresh()
    opening = _canon(spec)
    inverse = ops.apply_op(spec, {"op": "set", "addr": "skeleton/forearm@probe_field",
                                  "value": 3})[0]
    assert inverse == {"op": "unset", "addr": "skeleton/forearm@probe_field"}
    ops.apply_op(spec, inverse)
    assert _canon(spec) == opening


def test_unset_inverse_restores_authored_field_order() -> None:
    spec = _fresh()
    opening = _canon(spec)
    inverse = ops.apply_op(
        spec,
        {"op": "unset", "addr": "skeleton/forearm@length"},
    )[0]

    assert inverse["op"] == "set"
    assert inverse["addr"] == "meta"
    ops.apply_op(spec, inverse)
    assert _canon(spec) == opening


def test_inverse_add_flesh_roundtrips():
    spec = _fresh()
    opening = _canon(spec)
    inverse = ops.apply_op(spec, {"op": "add", "kind": "flesh", "addr": "skeleton/forearm",
                                  "params": {"kind": "gencyl", "name": "probe"}})[0]
    assert inverse == {"op": "remove", "addr": "skeleton/forearm/probe"}
    ops.apply_op(spec, inverse)
    assert _canon(spec) == opening


@pytest.mark.parametrize(
    ("kind", "address", "params", "inverse_address"),
    (
        (
            "flesh",
            "skeleton/root",
            {"kind": "gencyl", "name": "flesh[0]"},
            "skeleton/root/flesh%5B0%5D",
        ),
        (
            "array",
            "skeleton/root",
            {
                "name": "arrays[0]",
                "n": 1,
                "template": {"kind": "blob"},
            },
            "skeleton/root/arrays%5B0%5D",
        ),
        (
            "mount",
            "mounts",
            {
                "port": "mounts[0]",
                "subgrammar": "body",
                "spec": "probe.json",
            },
            "mounts/mounts%5B0%5D",
        ),
        (
            "bone",
            "skeleton/root",
            {
                "id": "child/leaf@bud",
                "length": 0.2,
                "rest_dir": [0, 1, 0],
                "joint": {"dof": "fixed"},
            },
            "skeleton/child%2Fleaf%40bud",
        ),
    ),
)
def test_inverse_add_preserves_literal_selector_spelling(
    kind: str,
    address: str,
    params: dict[str, object],
    inverse_address: str,
) -> None:
    authored = _fresh()
    spec = (
        {
            **authored,
            "skeleton": {
                **authored["skeleton"],
                "root": {**authored["skeleton"]["root"], "arrays": []},
            },
        }
        if kind == "array"
        else authored
    )
    opening = _canon(spec)
    inverse = ops.apply_op(
        spec,
        {"op": "add", "kind": kind, "addr": address, "params": params},
    )[0]

    assert inverse == {"op": "remove", "addr": inverse_address}
    ops.apply_op(spec, inverse)
    assert _canon(spec) == opening


def test_journal_undo_restores_literal_selector_named_addition() -> None:
    spec = _fresh()
    opening = _canon(spec)
    operation = ops.decode_op(
        {
            "op": "add_at",
            "kind": "flesh",
            "addr": "skeleton/root",
            "params": {"kind": "gencyl", "name": "flesh[0]"},
            "index": 1,
        }
    )
    assert isinstance(operation, ops.AddAtOp)
    program = ops.compile_program(spec, (operation,))
    assert isinstance(program, ops.ProgramPlan)
    committed = journal.commit_program(journal.empty(), (operation,), program)
    undo = journal.plan_undo(committed.state, program.next_spec)

    assert isinstance(undo, journal.JournalTransition)
    assert ops.to_json(undo.entry.applied) == {
        "op": "remove",
        "addr": "skeleton/root/flesh%5B0%5D",
    }
    assert _canon(undo.edit.next_spec) == opening


def test_add_inverse_glues_to_document_when_local_remove_leaves_residue() -> None:
    spec = _fresh()
    opening = _canon(spec)
    assert "arrays" not in spec["skeleton"]["root"]

    inverse = ops.apply_op(
        spec,
        {
            "op": "add",
            "kind": "array",
            "addr": "skeleton/root",
            "params": {
                "name": "probe",
                "n": 1,
                "template": {"kind": "blob"},
            },
        },
    )[0]

    assert inverse["op"] == "set"
    assert inverse["addr"] == "meta"
    ops.apply_op(spec, inverse)
    assert _canon(spec) == opening


def test_inverse_remove_flesh_roundtrips():
    spec = _fresh()
    opening = _canon(spec)
    inverse = ops.apply_op(spec, {"op": "remove",
                                  "addr": "skeleton/spine_upper/spine_upper"})[0]
    assert inverse["op"] == "add_at" and inverse["kind"] == "flesh"
    assert inverse["addr"] == "skeleton/spine_upper"
    assert inverse["params"]["name"] == "spine_upper"
    ops.apply_op(spec, inverse)                     # re-insert at the same index
    assert _canon(spec) == opening


def test_inverse_add_bone_roundtrips():
    spec = _fresh()
    opening = _canon(spec)
    inverse = ops.apply_op(spec, {"op": "add", "kind": "bone", "addr": "skeleton/forearm",
                                  "params": {"id": "probe", "length": 0.3,
                                             "rest_dir": [0, 1, 0],
                                             "joint": {"dof": "hinge"}, "flesh": []}})[0]
    assert inverse == {"op": "remove", "addr": "skeleton/probe"}
    ops.apply_op(spec, inverse)
    assert _canon(spec) == opening


def test_inverse_rename_bone_rewrites_refs_and_roundtrips():
    spec = _fresh()
    opening = _canon(spec)
    inverse = ops.apply_op(spec, {"op": "rename", "addr": "skeleton/forearm",
                                  "to": "lower_arm"})[0]
    # The separated underbody has no authored equipment mount or IK goal.  Its
    # forelimb anatomy provenance is the live reference a rename must preserve.
    ids = {b["id"] for b in spec["skeleton"]["bones"]}
    assert "lower_arm" in ids and "forearm" not in ids
    forelimb = next(
        region
        for region in spec["anatomy"]["overall"]["regions"]
        if region["region_id"] == "forelimb"
    )
    assert forelimb["host_bone_id"] == "lower_arm"
    assert inverse == {"op": "rename", "addr": "skeleton/lower_arm", "to": "forearm"}
    ops.apply_op(spec, inverse)
    assert _canon(spec) == opening


@pytest.mark.parametrize(
    ("identifier", "address"),
    (
        ("old-id", "skeleton/old-id"),
        ("old/leaf@bud", "skeleton/old%2Fleaf%40bud"),
    ),
)
def test_rename_inverse_glues_to_document_for_non_renameable_original_ids(
    identifier: str,
    address: str,
) -> None:
    base = empty_spec("probe")
    spec = {
        **base,
        "skeleton": {
            **base["skeleton"],
            "bones": [
                {
                    "id": identifier,
                    "length": 0.2,
                    "rest_dir": [0.0, 1.0, 0.0],
                    "joint": {"dof": "fixed"},
                }
            ],
        },
    }
    program, undo = _planned_undo(
        spec,
        {"op": "rename", "addr": address, "to": "newid"},
    )

    assert ops.to_json(program.edits[0].inverse)["op"] == "set"
    assert ops.to_json(undo.entry.applied)["addr"] == "meta"
    assert _canon(undo.edit.next_spec) == _canon(spec)


def test_remove_inverse_glues_to_document_when_bone_parent_was_absent() -> None:
    base = empty_spec("probe")
    spec = {
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
    program, undo = _planned_undo(
        spec,
        {"op": "remove", "addr": "skeleton/orphan"},
    )

    assert ops.to_json(program.edits[0].inverse)["op"] == "set"
    assert ops.to_json(undo.entry.applied)["addr"] == "meta"
    assert _canon(undo.edit.next_spec) == _canon(spec)


def test_remove_pose_inverse_glues_to_document_when_readding_changes_key_order() -> None:
    base = empty_spec("probe")
    bones = [
        {
            "id": identifier,
            "parent": "root",
            "length": 0.2,
            "rest_dir": [0.0, 1.0, 0.0],
            "joint": {"dof": "fixed"},
        }
        for identifier in ("a", "b", "c")
    ]
    spec = {
        **base,
        "skeleton": {**base["skeleton"], "bones": bones},
        "pose": {
            "joints": {
                "a": {"pitch": 1.0},
                "b": {"pitch": 2.0},
                "c": {"pitch": 3.0},
            },
            "goals": [],
        },
    }
    program, undo = _planned_undo(
        spec,
        {"op": "remove", "addr": "pose/b"},
    )

    assert ops.to_json(program.edits[0].inverse)["op"] == "set"
    assert ops.to_json(undo.entry.applied)["addr"] == "meta"
    assert _canon(undo.edit.next_spec) == _canon(spec)


def test_inverse_pose_autocreate_then_undo_removes_entry():
    spec = _fresh()
    opening = _canon(spec)
    assert "upper_arm" not in spec["pose"]["joints"]
    inverse = ops.apply_op(spec, {"op": "set", "addr": "pose/upper_arm@pitch",
                                  "value": 15.0})[0]
    assert spec["pose"]["joints"]["upper_arm"] == {"pitch": 15.0}
    assert inverse == {"op": "remove", "addr": "pose/upper_arm"}
    ops.apply_op(spec, inverse)                     # undo removes the whole entry
    assert "upper_arm" not in spec["pose"]["joints"]
    assert _canon(spec) == opening


def test_pose_creation_inverse_restores_absent_joint_collection() -> None:
    authored = _fresh()
    spec = {
        **authored,
        "pose": {
            key: value
            for key, value in authored["pose"].items()
            if key != "joints"
        },
    }
    opening = _canon(spec)
    inverse = ops.apply_op(
        spec,
        {"op": "set", "addr": "pose/upper_arm@pitch", "value": 15.0},
    )[0]

    assert inverse["op"] == "set"
    assert inverse["addr"] == "meta"
    ops.apply_op(spec, inverse)
    assert _canon(spec) == opening


# --------------------------------------------------------------------------- #
# 4. Rejections are validate-before-mutate: the bytes never move.              #
# --------------------------------------------------------------------------- #
def test_rejections_leave_bytes_untouched():
    spec = _fresh()
    before = _canon(spec)
    bad_ops = [
        # unknown bone address
        {"op": "set", "addr": "skeleton/nonexistent@length", "value": 0.5},
        # malformed address (empty segment)
        {"op": "remove", "addr": "skeleton//x"},
        # a 3-vector field given a 2-vector
        {"op": "set", "addr": "skeleton/forearm@rest_dir", "value": [1, 0]},
        # bone with children may not be removed
        {"op": "remove", "addr": "skeleton/spine_lower"},
        # duplicate bone id
        {"op": "add", "kind": "bone", "addr": "skeleton/neck",
         "params": {"id": "helm", "length": 0.3, "rest_dir": [0, 1, 0],
                    "joint": {"dof": "hinge"}}},
        # duplicate flesh name on a bone
        {"op": "add", "kind": "flesh", "addr": "skeleton/forearm",
         "params": {"kind": "gencyl", "name": "forearm"}},
        # remove pointed at an @field (use unset)
        {"op": "remove", "addr": "skeleton/forearm@length"},
        # unknown op name
        {"op": "frobnicate", "addr": "skeleton/forearm"},
    ]
    for op in bad_ops:
        with pytest.raises(ops.Reject):
            ops.apply_op(spec, op)
        assert _canon(spec) == before               # untouched after each reject


# --------------------------------------------------------------------------- #
# 5. JSONL journal round-trips through disk and replays to the same bytes.      #
# --------------------------------------------------------------------------- #
def test_journal_jsonl_roundtrip(tmp_path):
    spec = _fresh()
    journal = Journal()
    _commit(journal, spec, {"op": "set", "addr": "skeleton/forearm@length", "value": 0.45})
    _commit(journal, spec, {"op": "add", "kind": "flesh", "addr": "skeleton/helm",
                            "params": {"kind": "box", "name": "crest"}})
    _commit(journal, spec, {"op": "set", "addr": "pose/upper_arm@pitch", "value": 8.0})
    journal.apply_undo(spec)                         # a real undo entry on disk too
    final = _canon(spec)

    path = tmp_path / "session.jsonl"
    journal.save(str(path))
    for line in path.read_text().splitlines():
        if line.strip():
            json.loads(line)                         # every line parses cleanly
    entries = Journal.load(str(path))
    assert entries == journal.entries

    fresh = _fresh()
    Journal.replay(fresh, entries)
    assert _canon(fresh) == final


# --------------------------------------------------------------------------- #
# 6. End-to-end sanity: a few pool ops still recompile without a compile error. #
# --------------------------------------------------------------------------- #
def test_end_to_end_ops_then_recompile_clean():
    state = SessionState.open(_KNIGHT)
    assert state.cache.error is None                 # opens clean
    journal = Journal()
    _commit(journal, state.spec, {"op": "set", "addr": "skeleton/forearm@length",
                                  "value": 0.5})
    _commit(journal, state.spec, {"op": "set", "addr": "pose/upper_arm@pitch",
                                  "value": 10.0})
    _commit(journal, state.spec, {"op": "set", "addr": "meta@blend", "value": 0.007})
    state.recompile()
    assert state.cache.error is None
