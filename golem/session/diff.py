from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial, reduce
from itertools import chain

from golem.addressing.session_grammar import encode_address_segment
from golem.kernel.body.canonical import _canonical_json
from golem.session import ops
from golem.session.ops import _obstruct


__all__ = ("diff_specs",)


type _RawOperation = dict[str, object]
type _RecordAddress = Callable[[str], str]
type _AddAddress = Callable[[dict], str]
type _RecordDescent = Callable[[dict, dict], "_OperationPhases | ops.EditObstructed"]


_MISSING = object()


@dataclass(frozen=True)
class _OperationPhases:
    removals: tuple[_RawOperation, ...] = ()
    edits: tuple[_RawOperation, ...] = ()
    additions: tuple[_RawOperation, ...] = ()

    @property
    def operations(self) -> tuple[_RawOperation, ...]:
        return (*self.removals, *self.edits, *self.additions)


@dataclass(frozen=True)
class _CompiledOperations:
    operations: tuple[ops.Op, ...]
    next_spec: dict


@dataclass(frozen=True)
class _DiffAccumulator:
    spec: dict
    operations: tuple[ops.Op, ...]
    obstruction: ops.EditObstructed | None = None


def _json_bytes(value: object) -> str:
    return json.dumps(value, indent=1, ensure_ascii=False)


def _same_json(left: object, right: object) -> bool:
    return _json_bytes(left) == _json_bytes(right)


def _combine_phases(phases: tuple[_OperationPhases, ...]) -> _OperationPhases:
    return _OperationPhases(
        tuple(chain.from_iterable(phase.removals for phase in phases)),
        tuple(chain.from_iterable(phase.edits for phase in phases)),
        tuple(chain.from_iterable(phase.additions for phase in phases)),
    )


def _field_address(record_address: str, path: tuple[str, ...]) -> str:
    return f"{record_address}@{'.'.join(path)}"


def _field_change(
    record_address: str,
    path: tuple[str, ...],
    previous: object,
    current: object,
) -> tuple[_RawOperation, ...]:
    address = _field_address(record_address, path)
    if current is _MISSING:
        return ({"op": "unset", "addr": address},)
    if previous is _MISSING:
        return ({"op": "set", "addr": address, "value": current},)
    if _same_json(previous, current):
        return ()
    if isinstance(previous, dict) and isinstance(current, dict):
        return _mapping_field_operations(record_address, previous, current, path)
    return ({"op": "set", "addr": address, "value": current},)


def _mapping_field_operations(
    record_address: str,
    previous: Mapping[str, object],
    current: Mapping[str, object],
    prefix: tuple[str, ...] = (),
    excluded: frozenset[str] = frozenset(),
) -> tuple[_RawOperation, ...]:
    keys = tuple(
        key
        for key in dict.fromkeys((*previous.keys(), *current.keys()))
        if key not in excluded
    )
    return tuple(
        chain.from_iterable(
            _field_change(
                record_address,
                (*prefix, key),
                previous.get(key, _MISSING),
                current.get(key, _MISSING),
            )
            for key in keys
        )
    )


def _record_pairs(
    value: object,
    identity_key: str,
    address: str,
) -> tuple[tuple[str, dict], ...] | ops.EditObstructed:
    if not isinstance(value, list):
        return _obstruct(address, "expected an array of records")
    if not all(isinstance(record, dict) for record in value):
        return _obstruct(address, "expected every array member to be an object")
    records = tuple(record for record in value if isinstance(record, dict))
    if not all(isinstance(record.get(identity_key), str) for record in records):
        return _obstruct(address, f"every record needs string field {identity_key!r}")
    pairs = tuple((record[identity_key], record) for record in records)
    identities = tuple(identity for identity, _record in pairs)
    return (
        pairs
        if len(identities) == len(frozenset(identities))
        else _obstruct(address, f"duplicate {identity_key!r} identities")
    )


def _common_record_phases(
    record_address: str,
    identity_key: str,
    excluded: frozenset[str] = frozenset(),
) -> _RecordDescent:
    def descend(previous: dict, current: dict) -> _OperationPhases:
        return _OperationPhases(
            edits=_mapping_field_operations(
                record_address,
                previous,
                current,
                excluded=frozenset({identity_key, *excluded}),
            )
        )

    return descend


def _keyed_collection_phases(
    previous: object,
    current: object,
    identity_key: str,
    collection_address: str,
    add_kind: ops.AddKind,
    record_address: _RecordAddress,
    add_address: _AddAddress,
    descend_record: _RecordDescent,
) -> _OperationPhases | ops.EditObstructed:
    previous_pairs = _record_pairs(previous, identity_key, collection_address)
    current_pairs = _record_pairs(current, identity_key, collection_address)
    if isinstance(previous_pairs, ops.EditObstructed):
        return previous_pairs
    if isinstance(current_pairs, ops.EditObstructed):
        return current_pairs
    previous_by_id = dict(previous_pairs)
    current_by_id = dict(current_pairs)
    common_previous = tuple(
        identity for identity, _record in previous_pairs if identity in current_by_id
    )
    common_current = tuple(
        identity for identity, _record in current_pairs if identity in previous_by_id
    )
    if common_previous != common_current:
        return _obstruct(collection_address, "stable record order changed")
    common_results = tuple(
        descend_record(previous_by_id[identity], current_by_id[identity])
        for identity in common_current
    )
    common_obstructions = tuple(
        chain.from_iterable(
            result.obstructions
            for result in common_results
            if isinstance(result, ops.EditObstructed)
        )
    )
    if common_obstructions:
        return ops.EditObstructed(common_obstructions)
    common_phases = _combine_phases(
        tuple(
            result
            for result in common_results
            if isinstance(result, _OperationPhases)
        )
    )
    removals = tuple(
        {
            "op": "remove",
            "addr": record_address(identity),
        }
        for identity, _record in reversed(previous_pairs)
        if identity not in current_by_id
    )
    additions = tuple(
        {
            "op": "add_at",
            "kind": add_kind.value,
            "addr": add_address(record),
            "params": record,
            "index": index,
        }
        for index, (identity, record) in enumerate(current_pairs)
        if identity not in previous_by_id
    )
    return _OperationPhases(
        (*removals, *common_phases.removals),
        common_phases.edits,
        (*common_phases.additions, *additions),
    )


def _bone_record_phases(previous: dict, current: dict) -> _OperationPhases | ops.EditObstructed:
    bone_id = current.get("id")
    if not isinstance(bone_id, str):
        return _obstruct("skeleton", "bone needs a string id")
    bone_address = f"skeleton/{encode_address_segment(bone_id)}"
    flesh = _keyed_collection_phases(
        previous.get("flesh", []),
        current.get("flesh", []),
        "name",
        f"{bone_address}/flesh",
        ops.AddKind.FLESH,
        lambda name: f"{bone_address}/{encode_address_segment(name)}",
        lambda _record: bone_address,
        lambda old, new: _common_record_phases(
            f"{bone_address}/"
            f"{encode_address_segment(str(new.get('name')))}",
            "name",
        )(old, new),
    )
    arrays = _keyed_collection_phases(
        previous.get("arrays", []),
        current.get("arrays", []),
        "name",
        f"{bone_address}/arrays",
        ops.AddKind.ARRAY,
        lambda name: f"{bone_address}/{encode_address_segment(name)}",
        lambda _record: bone_address,
        lambda old, new: _common_record_phases(
            f"{bone_address}/"
            f"{encode_address_segment(str(new.get('name')))}",
            "name",
        )(old, new),
    )
    obstruction = next(
        (
            result
            for result in (flesh, arrays)
            if isinstance(result, ops.EditObstructed)
        ),
        None,
    )
    if obstruction is not None:
        return obstruction
    nested = _combine_phases(
        tuple(result for result in (flesh, arrays) if isinstance(result, _OperationPhases))
    )
    return _OperationPhases(
        nested.removals,
        (
            *_mapping_field_operations(
                bone_address,
                previous,
                current,
                excluded=frozenset({"id", "flesh", "arrays"}),
            ),
            *nested.edits,
        ),
        nested.additions,
    )


def _skeleton_operations(previous: object, current: object) -> tuple[_RawOperation, ...] | ops.EditObstructed:
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return _obstruct("meta@skeleton", "skeleton section is not an object")
    previous_root = previous.get("root")
    current_root = current.get("root")
    if not isinstance(previous_root, dict) or not isinstance(current_root, dict):
        return _obstruct("skeleton/root", "skeleton root is not an object")
    current_root_id = current_root.get("id", "root")
    bones = _keyed_collection_phases(
        previous.get("bones", []),
        current.get("bones", []),
        "id",
        "skeleton",
        ops.AddKind.BONE,
        lambda identity: f"skeleton/{encode_address_segment(identity)}",
        lambda record: (
            "skeleton/root"
            if record.get("parent", current_root_id) == current_root_id
            else f"skeleton/{encode_address_segment(str(record.get('parent')))}"
        ),
        _bone_record_phases,
    )
    if isinstance(bones, ops.EditObstructed):
        return bones
    return _OperationPhases(
        bones.removals,
        (
            *_mapping_field_operations("skeleton/root", previous_root, current_root),
            *_mapping_field_operations(
                "meta",
                previous,
                current,
                prefix=("skeleton",),
                excluded=frozenset({"root", "bones"}),
            ),
            *bones.edits,
        ),
        bones.additions,
    ).operations


def _joint_phases(previous: object, current: object) -> _OperationPhases | ops.EditObstructed:
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return _obstruct("pose", "pose joints are not an object")
    if not all(isinstance(value, dict) for value in (*previous.values(), *current.values())):
        return _obstruct("pose", "every pose joint must be an object")
    previous_keys = tuple(previous)
    current_keys = tuple(current)
    common_previous = tuple(key for key in previous_keys if key in current)
    common_current = tuple(key for key in current_keys if key in previous)
    if common_previous != common_current:
        return _obstruct("pose", "stable pose order changed")
    return _OperationPhases(
        tuple(
            {
                "op": "remove",
                "addr": f"pose/{encode_address_segment(bone_id)}",
            }
            for bone_id in reversed(previous_keys)
            if bone_id not in current
        ),
        tuple(
            chain.from_iterable(
                _mapping_field_operations(
                    f"pose/{encode_address_segment(bone_id)}",
                    previous[bone_id],
                    current[bone_id],
                )
                for bone_id in common_current
            )
        ),
        tuple(
            {
                "op": "add",
                "kind": ops.AddKind.POSE.value,
                "addr": f"pose/{encode_address_segment(bone_id)}",
                "params": current[bone_id],
            }
            for bone_id in current_keys
            if bone_id not in previous
        ),
    )


def _pose_operations(previous: object, current: object) -> tuple[_RawOperation, ...] | ops.EditObstructed:
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return _obstruct("meta@pose", "pose section is not an object")
    joints = _joint_phases(previous.get("joints", {}), current.get("joints", {}))
    goals = _keyed_collection_phases(
        previous.get("goals", []),
        current.get("goals", []),
        "id",
        "pose/goals",
        ops.AddKind.GOAL,
        lambda identity: f"pose/goals/{encode_address_segment(identity)}",
        lambda _record: "pose/goals",
        lambda old, new: _common_record_phases(
            f"pose/goals/{encode_address_segment(str(new.get('id')))}",
            "id",
        )(old, new),
    )
    obstruction = next(
        (
            result
            for result in (joints, goals)
            if isinstance(result, ops.EditObstructed)
        ),
        None,
    )
    if obstruction is not None:
        return obstruction
    phases = _combine_phases(
        tuple(result for result in (joints, goals) if isinstance(result, _OperationPhases))
    )
    return _OperationPhases(
        phases.removals,
        (
            *_mapping_field_operations(
                "meta",
                previous,
                current,
                prefix=("pose",),
                excluded=frozenset({"joints", "goals"}),
            ),
            *phases.edits,
        ),
        phases.additions,
    ).operations


def _prop_record_phases(previous: dict, current: dict) -> _OperationPhases | ops.EditObstructed:
    prop_id = current.get("id")
    if not isinstance(prop_id, str):
        return _obstruct("props", "prop needs a string id")
    prop_address = f"props/{encode_address_segment(prop_id)}"
    parts = _keyed_collection_phases(
        previous.get("parts", []),
        current.get("parts", []),
        "id",
        f"{prop_address}/parts",
        ops.AddKind.PART,
        lambda identity: f"{prop_address}/{encode_address_segment(identity)}",
        lambda _record: prop_address,
        lambda old, new: _common_record_phases(
            f"{prop_address}/{encode_address_segment(str(new.get('id')))}",
            "id",
        )(old, new),
    )
    if isinstance(parts, ops.EditObstructed):
        return parts
    return _OperationPhases(
        parts.removals,
        (
            *_mapping_field_operations(
                prop_address,
                previous,
                current,
                excluded=frozenset({"id", "parts"}),
            ),
            *parts.edits,
        ),
        parts.additions,
    )


def _props_operations(previous: object, current: object) -> tuple[_RawOperation, ...] | ops.EditObstructed:
    result = _keyed_collection_phases(
        previous,
        current,
        "id",
        "props",
        ops.AddKind.PROP,
        lambda identity: f"props/{encode_address_segment(identity)}",
        lambda _record: "props",
        _prop_record_phases,
    )
    return result if isinstance(result, ops.EditObstructed) else result.operations


def _flat_collection_operations(
    previous: object,
    current: object,
    identity_key: str,
    collection_address: str,
    add_kind: ops.AddKind,
) -> tuple[_RawOperation, ...] | ops.EditObstructed:
    result = _keyed_collection_phases(
        previous,
        current,
        identity_key,
        collection_address,
        add_kind,
        lambda identity: (
            f"{collection_address}/{encode_address_segment(identity)}"
        ),
        lambda _record: collection_address,
        lambda old, new: _common_record_phases(
            f"{collection_address}/"
            f"{encode_address_segment(str(new.get(identity_key)))}",
            identity_key,
        )(old, new),
    )
    return result if isinstance(result, ops.EditObstructed) else result.operations


def _semantic_operations(
    section: str,
    previous: object,
    current: object,
) -> tuple[_RawOperation, ...] | ops.EditObstructed | None:
    match section:
        case "skeleton":
            return _skeleton_operations(previous, current)
        case "pose":
            return _pose_operations(previous, current)
        case "props":
            return _props_operations(previous, current)
        case "mounts":
            return _flat_collection_operations(
                previous,
                current,
                "port",
                "mounts",
                ops.AddKind.MOUNT,
            )
        case "contacts":
            return _flat_collection_operations(
                previous,
                current,
                "group",
                "contacts",
                ops.AddKind.CONTACT,
            )
        case _:
            return None


def _decode_and_compile(
    spec: dict,
    raw_operations: tuple[_RawOperation, ...],
) -> _CompiledOperations | ops.EditObstructed:
    decoded = tuple(map(ops.decode_op, raw_operations))
    obstructions = tuple(
        chain.from_iterable(
            result.obstructions
            for result in decoded
            if isinstance(result, ops.EditObstructed)
        )
    )
    if obstructions:
        return ops.EditObstructed(obstructions)
    operations = tuple(
        result for result in decoded if not isinstance(result, ops.EditObstructed)
    )
    compiled = ops.compile_program(spec, operations)
    return (
        compiled
        if isinstance(compiled, ops.EditObstructed)
        else _CompiledOperations(operations, compiled.next_spec)
    )


def _fallback_operation(section: str, current: object) -> tuple[_RawOperation, ...]:
    return (
        ({"op": "unset", "addr": f"meta@{section}"},)
        if current is _MISSING
        else ({"op": "set", "addr": f"meta@{section}", "value": current},)
    )


def _section_matches(spec: dict, section: str, target: object) -> bool:
    return (
        section not in spec
        if target is _MISSING
        else section in spec and _same_json(spec[section], target)
    )


def _section_step(target: dict, accumulator: _DiffAccumulator, section: str) -> _DiffAccumulator:
    if accumulator.obstruction is not None:
        return accumulator
    previous = accumulator.spec.get(section, _MISSING)
    current = target.get(section, _MISSING)
    if previous is not _MISSING and current is not _MISSING and _same_json(previous, current):
        return accumulator
    semantic = (
        _semantic_operations(section, previous, current)
        if previous is not _MISSING and current is not _MISSING
        else None
    )
    attempted = (
        _decode_and_compile(accumulator.spec, semantic)
        if isinstance(semantic, tuple)
        else semantic
    )
    selected = (
        attempted
        if isinstance(attempted, _CompiledOperations)
        and _section_matches(attempted.next_spec, section, current)
        else _decode_and_compile(
            accumulator.spec,
            _fallback_operation(section, current),
        )
    )
    return (
        _DiffAccumulator(
            selected.next_spec,
            (*accumulator.operations, *selected.operations),
        )
        if isinstance(selected, _CompiledOperations)
        else _DiffAccumulator(
            accumulator.spec,
            accumulator.operations,
            selected,
        )
    )


def _local_top_level_order_is_reconstructible(previous: dict, current: dict) -> bool:
    retained = tuple(key for key in previous if key in current)
    additions = tuple(key for key in current if key not in previous)
    return (*retained, *additions) == tuple(current)


def _rewrite_document(previous: dict, current: dict) -> tuple[ops.Op, ...] | ops.EditObstructed:
    raw_operations = (
        {"op": "set", "addr": "meta", "value": current},
    )
    compiled = _decode_and_compile(previous, raw_operations)
    if isinstance(compiled, ops.EditObstructed):
        return compiled
    return (
        compiled.operations
        if _canonical_json(compiled.next_spec) == _canonical_json(current)
        else _obstruct("meta", "document rewrite did not reproduce canonical bytes")
    )


def diff_specs(previous: dict, current: dict) -> tuple[ops.Op, ...] | ops.EditObstructed:
    if not _local_top_level_order_is_reconstructible(previous, current):
        return _rewrite_document(previous, current)
    sections = (
        *tuple(previous),
        *(section for section in current if section not in previous),
    )
    diff = reduce(
        partial(_section_step, current),
        sections,
        _DiffAccumulator(previous, ()),
    )
    if diff.obstruction is not None:
        return _rewrite_document(previous, current)
    return (
        diff.operations
        if _canonical_json(diff.spec) == _canonical_json(current)
        else _rewrite_document(previous, current)
    )
