"""Typed session edits compiled by descent before one boundary commit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from itertools import chain
from typing import ClassVar, Mapping, assert_never

from golem.addressing.session import (
    Address,
    DottedQualifiedToken,
    IndexedSegment,
    RejectedAddress,
)
from golem.addressing.session_grammar import (
    encode_address_segment,
    field_tokens,
    parse_address,
    parse_dotted_qualified_token,
    parse_indexed_segment,
    render_address,
)
from golem.kernel.body.canonical import _canonical_json
from golem.kernel.engine.types import CompositionOperator
from golem.json_value import is_json_value


_MISSING = object()
_VEC3 = frozenset({"world", "offset", "rest_dir", "axis", "pole", "center"})
_POSE_KEYS = frozenset({"yaw", "pitch", "roll"})


class Reject(Exception):
    def __init__(self, addr: str, why: str):
        self.addr = addr
        self.why = why
        super().__init__(f"reject @ {addr}: {why}")


@dataclass(frozen=True)
class FrozenObject:
    items: tuple[tuple[str, FrozenJson], ...]


type FrozenJson = None | bool | int | float | str | tuple["FrozenJson", ...] | FrozenObject


def _freeze_json(value: object) -> FrozenJson:
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, dict):
        return FrozenObject(tuple((key, _freeze_json(item)) for key, item in value.items()))
    return value


def _thaw_json(value: FrozenJson) -> object:
    match value:
        case FrozenObject(items):
            return {key: _thaw_json(item) for key, item in items}
        case tuple(items):
            return [_thaw_json(item) for item in items]
        case _:
            return value


class AddKind(StrEnum):
    BONE = "bone"
    FLESH = "flesh"
    ARRAY = "array"
    PROP = "prop"
    PART = "part"
    GOAL = "goal"
    CONTACT = "contact"
    MOUNT = "mount"
    POSE = "pose"


class FleshKind(StrEnum):
    GENCYL = "gencyl"
    LOFT = "loft"
    BLOB = "blob"
    BOX = "box"


class ArrayTemplateKind(StrEnum):
    BLOB = "blob"
    BOX = "box"


class DegreeOfFreedom(StrEnum):
    FIXED = "fixed"
    HINGE = "hinge"
    BALL = "ball"


_REQUIRED_ADD_FIELDS: Mapping[AddKind, tuple[str, ...]] = {
    AddKind.BONE: ("id", "length", "rest_dir", "joint"),
    AddKind.FLESH: ("kind", "name"),
    AddKind.ARRAY: ("name", "n", "template"),
    AddKind.PROP: ("id", "anchor", "parts"),
    AddKind.PART: ("id", "type"),
    AddKind.GOAL: ("id", "chain", "end", "target"),
    AddKind.CONTACT: ("group", "parts", "solve"),
    AddKind.MOUNT: ("port", "subgrammar", "spec"),
    AddKind.POSE: (),
}


@dataclass(frozen=True)
class MetaLocation:
    pass


@dataclass(frozen=True)
class SkeletonRootLocation:
    pass


@dataclass(frozen=True)
class BoneLocation:
    bone_id: str


@dataclass(frozen=True)
class NamedSelector:
    name: str


@dataclass(frozen=True)
class FleshIndexSelector:
    index: int


@dataclass(frozen=True)
class ArrayIndexSelector:
    index: int


type FleshSelector = NamedSelector | FleshIndexSelector | ArrayIndexSelector


@dataclass(frozen=True)
class FleshLocation:
    owner: SkeletonRootLocation | BoneLocation
    selector: FleshSelector


@dataclass(frozen=True)
class PoseLocation:
    bone_id: str


@dataclass(frozen=True)
class GoalsLocation:
    pass


@dataclass(frozen=True)
class GoalLocation:
    goal_id: str


@dataclass(frozen=True)
class PropsLocation:
    pass


@dataclass(frozen=True)
class PropLocation:
    prop_id: str


@dataclass(frozen=True)
class PartLocation:
    prop_id: str
    part_id: str


@dataclass(frozen=True)
class MountsLocation:
    pass


@dataclass(frozen=True)
class NamedMountSelector:
    port: str


@dataclass(frozen=True)
class IndexedMountSelector:
    index: int


type MountSelector = NamedMountSelector | IndexedMountSelector


@dataclass(frozen=True)
class MountLocation:
    selector: MountSelector


@dataclass(frozen=True)
class ContactsLocation:
    pass


@dataclass(frozen=True)
class ContactLocation:
    group: str


type RecordLocation = (
    MetaLocation
    | SkeletonRootLocation
    | BoneLocation
    | FleshLocation
    | PoseLocation
    | GoalsLocation
    | GoalLocation
    | PropsLocation
    | PropLocation
    | PartLocation
    | MountsLocation
    | MountLocation
    | ContactsLocation
    | ContactLocation
)


@dataclass(frozen=True)
class RecordAddress:
    source: str
    location: RecordLocation
    fields: tuple[str | int, ...]

    @property
    def record_source(self) -> str:
        return self.source.partition("@")[0]


@dataclass(frozen=True)
class EditObstruction:
    address: str
    reason: str


@dataclass(frozen=True)
class EditObstructed:
    obstructions: tuple[EditObstruction, ...]

    @property
    def first(self) -> EditObstruction:
        return self.obstructions[0]


@dataclass(frozen=True)
class FixedRecord:
    address: RecordAddress
    record: dict


class CollectionKind(StrEnum):
    BONES = "bones"
    FLESH = "flesh"
    ARRAYS = "arrays"
    GOALS = "goals"
    PROPS = "props"
    PARTS = "parts"
    MOUNTS = "mounts"
    CONTACTS = "contacts"


@dataclass(frozen=True)
class ListedRecord:
    address: RecordAddress
    collection: CollectionKind
    owner: SkeletonRootLocation | BoneLocation | PropLocation | None
    index: int
    record: dict


type RecordResolution = FixedRecord | ListedRecord | EditObstructed


@dataclass(frozen=True)
class BonePayload:
    _kind: ClassVar[AddKind] = AddKind.BONE
    identifier: str
    length: int | float
    rest_dir: tuple[int | float, int | float, int | float]
    degree_of_freedom: DegreeOfFreedom
    fields: FrozenObject


@dataclass(frozen=True)
class FleshPayload:
    _kind: ClassVar[AddKind] = AddKind.FLESH
    name: str
    flesh_kind: FleshKind
    fields: FrozenObject


@dataclass(frozen=True)
class ArrayPayload:
    _kind: ClassVar[AddKind] = AddKind.ARRAY
    name: str
    count: int
    along: tuple[int | float, int | float]
    template_kind: ArrayTemplateKind
    fields: FrozenObject


@dataclass(frozen=True)
class _IdentifiedPayload:
    identifier: str
    fields: FrozenObject


@dataclass(frozen=True)
class PropPayload(_IdentifiedPayload):
    _kind: ClassVar[AddKind] = AddKind.PROP


@dataclass(frozen=True)
class PartPayload(_IdentifiedPayload):
    _kind: ClassVar[AddKind] = AddKind.PART


@dataclass(frozen=True)
class GoalPayload(_IdentifiedPayload):
    _kind: ClassVar[AddKind] = AddKind.GOAL


@dataclass(frozen=True)
class ContactPayload:
    _kind: ClassVar[AddKind] = AddKind.CONTACT
    group: str
    fields: FrozenObject


@dataclass(frozen=True)
class MountPayload:
    _kind: ClassVar[AddKind] = AddKind.MOUNT
    port: str
    fields: FrozenObject


@dataclass(frozen=True)
class PosePayload:
    _kind: ClassVar[AddKind] = AddKind.POSE
    fields: FrozenObject


type AddPayload = (
    BonePayload
    | FleshPayload
    | ArrayPayload
    | PropPayload
    | PartPayload
    | GoalPayload
    | ContactPayload
    | MountPayload
    | PosePayload
)

type _IdentifiedPayloadType = (
    type[PropPayload] | type[PartPayload] | type[GoalPayload]
)

_IDENTIFIED_PAYLOAD_TYPES: Mapping[AddKind, _IdentifiedPayloadType] = {
    AddKind.PROP: PropPayload,
    AddKind.PART: PartPayload,
    AddKind.GOAL: GoalPayload,
}


@dataclass(frozen=True)
class SetOp:
    addr: RecordAddress
    value: FrozenJson


@dataclass(frozen=True)
class UnsetOp:
    addr: RecordAddress


@dataclass(frozen=True)
class AddOp:
    kind: AddKind
    addr: RecordAddress
    payload: AddPayload


@dataclass(frozen=True)
class AddAtOp:
    kind: AddKind
    addr: RecordAddress
    payload: AddPayload
    index: int


@dataclass(frozen=True)
class RemoveOp:
    addr: RecordAddress


@dataclass(frozen=True)
class RenameOp:
    addr: RecordAddress
    to: str


type Op = SetOp | UnsetOp | AddOp | AddAtOp | RemoveOp | RenameOp


@dataclass(frozen=True)
class EditPlan:
    next_spec: dict
    inverse: Op
    label: str
    touched_addresses: frozenset[str]


@dataclass(frozen=True)
class ProgramPlan:
    next_spec: dict
    edits: tuple[EditPlan, ...]
    touched_addresses: frozenset[str]


type EditCompileResult = EditPlan | EditObstructed
type ProgramCompileResult = ProgramPlan | EditObstructed


def _obstruct(address: str, reason: str) -> EditObstructed:
    return EditObstructed((EditObstruction(address, reason),))


def _decode_location(address: Address, source: str) -> RecordLocation | EditObstructed:
    segments = address.segments
    match segments:
        case ("meta",):
            return MetaLocation()
        case ("skeleton", "root"):
            return SkeletonRootLocation()
        case ("skeleton", owner, selector):
            owner_location: SkeletonRootLocation | BoneLocation = (
                SkeletonRootLocation() if owner == "root" else BoneLocation(owner)
            )
            indexed = (
                None
                if 2 in address.encoded_segments
                else parse_indexed_segment(selector)
            )
            selected: FleshSelector = (
                FleshIndexSelector(indexed.index)
                if isinstance(indexed, IndexedSegment)
                and indexed.collection == "flesh"
                else ArrayIndexSelector(indexed.index)
                if isinstance(indexed, IndexedSegment)
                and indexed.collection == "arrays"
                else NamedSelector(selector)
            )
            return FleshLocation(owner_location, selected)
        case ("skeleton", bone_id):
            return BoneLocation(bone_id)
        case ("pose", "goals"):
            return GoalsLocation()
        case ("pose", "goals", goal_id):
            return GoalLocation(goal_id)
        case ("pose", bone_id):
            return PoseLocation(bone_id)
        case ("props",):
            return PropsLocation()
        case ("props", prop_id):
            return PropLocation(prop_id)
        case ("props", prop_id, part_id):
            return PartLocation(prop_id, part_id)
        case ("mounts",):
            return MountsLocation()
        case ("mounts", selector):
            indexed = (
                None
                if 1 in address.encoded_segments
                else parse_indexed_segment(selector)
            )
            return MountLocation(
                IndexedMountSelector(indexed.index)
                if isinstance(indexed, IndexedSegment)
                and indexed.collection == "mounts"
                else NamedMountSelector(selector)
            )
        case ("contacts",):
            return ContactsLocation()
        case ("contacts", group):
            return ContactLocation(group)
        case ("skeleton",):
            return _obstruct(source, "skeleton needs /root or /<bone_id>")
        case _:
            return _obstruct(source, "address does not name a session record or collection")


def decode_address(source: object) -> RecordAddress | EditObstructed:
    if not isinstance(source, str):
        return _obstruct(str(source), "address must be a string")
    parsed = parse_address(source)
    match parsed:
        case RejectedAddress(obstructions):
            return _obstruct(source, obstructions[0].reason)
        case Address() as address:
            canonical_source = render_address(address)
            location = _decode_location(address, canonical_source)
            if isinstance(location, EditObstructed):
                return location
            return RecordAddress(
                canonical_source,
                location,
                field_tokens(address),
            )
        case _ as unreachable:
            assert_never(unreachable)


def _payload_json(payload: AddPayload) -> dict:
    value = _thaw_json(payload.fields)
    return value if isinstance(value, dict) else {}


def _json_object(params: object, address: str) -> FrozenObject | EditObstructed:
    if not isinstance(params, dict):
        return _obstruct(address, "params must be a JSON object")
    if not is_json_value(params):
        return _obstruct(address, "params must contain only JSON values")
    frozen = _freeze_json(params)
    return frozen if isinstance(frozen, FrozenObject) else _obstruct(address, "params must be an object")


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_vec(value: object, size: int) -> bool:
    return isinstance(value, list) and len(value) == size and all(is_number(item) for item in value)


def _missing_fields(address: str, kind: AddKind, params: Mapping[str, object]) -> EditObstructed | None:
    missing = tuple(
        EditObstruction(address, f"add {kind.value} missing required field {field!r}")
        for field in _REQUIRED_ADD_FIELDS[kind]
        if field not in params
    )
    return EditObstructed(missing) if missing else None


def _decode_payload(kind: AddKind, params: object, address: str) -> AddPayload | EditObstructed:
    fields = _json_object(params, address)
    if isinstance(fields, EditObstructed):
        return fields
    raw = params if isinstance(params, dict) else {}
    missing = _missing_fields(address, kind, raw)
    if missing is not None:
        return missing
    match kind:
        case AddKind.BONE:
            identifier = raw["id"]
            length = raw["length"]
            rest_dir = raw["rest_dir"]
            joint = raw["joint"]
            if not isinstance(identifier, str):
                return _obstruct(address, "bone id must be a string")
            if not is_number(length) or length <= 0:
                return _obstruct(address, "length must be a positive number")
            if not _is_vec(rest_dir, 3):
                return _obstruct(address, "rest_dir must be a 3-vector of numbers")
            if not isinstance(joint, dict):
                return _obstruct(address, "joint must be an object")
            try:
                degree_of_freedom = DegreeOfFreedom(joint.get("dof"))
            except ValueError:
                return _obstruct(address, "joint.dof must be one of ['ball', 'fixed', 'hinge']")
            return BonePayload(
                identifier,
                length,
                tuple(rest_dir),
                degree_of_freedom,
                fields,
            )
        case AddKind.FLESH:
            name = raw["name"]
            if not isinstance(name, str):
                return _obstruct(address, "flesh name must be a string")
            try:
                flesh_kind = FleshKind(raw["kind"])
            except ValueError:
                return _obstruct(
                    address,
                    f"flesh kind must be one of {[kind.value for kind in FleshKind]}",
                )
            if "operator" in raw:
                try:
                    CompositionOperator(raw["operator"])
                except (TypeError, ValueError):
                    return _obstruct(
                        address,
                        "operator must be one of "
                        f"{[operator.value for operator in CompositionOperator]}",
                    )
            return FleshPayload(name, flesh_kind, fields)
        case AddKind.ARRAY:
            name = raw["name"]
            count = raw["n"]
            template = raw["template"]
            along = raw.get("along", [0.0, 1.0])
            if not isinstance(name, str):
                return _obstruct(address, "array name must be a string")
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                return _obstruct(address, "array n must be an integer >= 1")
            if not _is_vec(along, 2):
                return _obstruct(address, "array along must be [t0, t1]")
            if not isinstance(template, dict):
                return _obstruct(address, "array template must be an object")
            try:
                template_kind = ArrayTemplateKind(template.get("kind"))
            except ValueError:
                return _obstruct(address, "array template kind must be one of ['blob', 'box']")
            return ArrayPayload(name, count, tuple(along), template_kind, fields)
        case AddKind.PROP | AddKind.PART | AddKind.GOAL:
            identifier = raw["id"]
            if not isinstance(identifier, str):
                return _obstruct(address, f"{kind.value} id must be a string")
            return _IDENTIFIED_PAYLOAD_TYPES[kind](identifier, fields)
        case AddKind.CONTACT:
            group = raw["group"]
            if not isinstance(group, str):
                return _obstruct(address, "contact group must be a string")
            return ContactPayload(group, fields)
        case AddKind.MOUNT:
            port = raw["port"]
            if not isinstance(port, str):
                return _obstruct(address, "mount port must be a string")
            return MountPayload(port, fields)
        case AddKind.POSE:
            bad = frozenset(raw) - _POSE_KEYS
            if bad:
                return _obstruct(
                    address,
                    f"pose accepts only {sorted(_POSE_KEYS)}, got {sorted(bad)}",
                )
            if not all(is_number(value) for value in raw.values()):
                return _obstruct(address, "pose angles must be numbers")
            return PosePayload(fields)
        case _ as unreachable:
            assert_never(unreachable)


def _decode_add_kind(raw: object, address: str) -> AddKind | EditObstructed:
    try:
        return AddKind(raw)
    except ValueError:
        return _obstruct(address, f"unknown add kind {raw!r}")


def decode_op(data: object) -> Op | EditObstructed:
    if not isinstance(data, dict):
        return _obstruct(str(data), "op must be a JSON object")
    tag = data.get("op")
    address = decode_address(data.get("addr"))
    if isinstance(address, EditObstructed):
        return address
    match tag:
        case "set":
            if "value" not in data:
                return _obstruct(address.source, "set missing value")
            if not is_json_value(data["value"]):
                return _obstruct(address.source, "set value must be JSON")
            return SetOp(address, _freeze_json(data["value"]))
        case "unset":
            return UnsetOp(address)
        case "add" | "add_at":
            kind = _decode_add_kind(data.get("kind"), address.source)
            if isinstance(kind, EditObstructed):
                return kind
            payload = _decode_payload(kind, data.get("params"), address.source)
            if isinstance(payload, EditObstructed):
                return payload
            if tag == "add_at":
                index = data.get("index")
                if not isinstance(index, int) or isinstance(index, bool):
                    return _obstruct(address.source, "add_at index must be an integer")
                return AddAtOp(kind, address, payload, index)
            return AddOp(kind, address, payload)
        case "remove":
            return RemoveOp(address)
        case "rename":
            target = data.get("to")
            if not isinstance(target, str):
                return _obstruct(address.source, "rename target must be a string")
            return RenameOp(address, target)
        case _:
            return _obstruct(str(data), f"unknown op {tag!r}")


def from_json(data: dict) -> Op:
    result = decode_op(data)
    if isinstance(result, EditObstructed):
        raise Reject(result.first.address, result.first.reason)
    return result


def to_json(op: Op) -> dict:
    match op:
        case SetOp(addr, value):
            return {"op": "set", "addr": addr.source, "value": _thaw_json(value)}
        case UnsetOp(addr):
            return {"op": "unset", "addr": addr.source}
        case AddOp(kind, addr, payload):
            return {
                "op": "add",
                "kind": kind.value,
                "addr": addr.source,
                "params": _payload_json(payload),
            }
        case AddAtOp(kind, addr, payload, index):
            return {
                "op": "add_at",
                "kind": kind.value,
                "addr": addr.source,
                "params": _payload_json(payload),
                "index": index,
            }
        case RemoveOp(addr):
            return {"op": "remove", "addr": addr.source}
        case RenameOp(addr, target):
            return {"op": "rename", "addr": addr.source, "to": target}
        case _ as unreachable:
            assert_never(unreachable)


def _find_index(records: list, key: str, value: object) -> tuple[int, dict] | None:
    return next(
        (
            (index, record)
            for index, record in enumerate(records)
            if isinstance(record, dict) and record.get(key) == value
        ),
        None,
    )


def _skeleton(spec: dict) -> dict:
    value = spec.get("skeleton", {})
    return value if isinstance(value, dict) else {}


def _pose(spec: dict) -> dict:
    value = spec.get("pose", {})
    return value if isinstance(value, dict) else {}


def _records(value: object) -> list:
    return value if isinstance(value, list) else []


def _bone_owner(spec: dict, owner: SkeletonRootLocation | BoneLocation, source: str) -> FixedRecord | ListedRecord | EditObstructed:
    skeleton = _skeleton(spec)
    match owner:
        case SkeletonRootLocation():
            root = skeleton.get("root", {})
            return (
                FixedRecord(RecordAddress(source, owner, ()), root)
                if isinstance(root, dict)
                else _obstruct(source, "skeleton root is not an object")
            )
        case BoneLocation(bone_id):
            bones = _records(skeleton.get("bones", []))
            found = _find_index(bones, "id", bone_id)
            if found is None:
                return _obstruct(source, f"no bone {bone_id!r}")
            index, bone = found
            return ListedRecord(
                RecordAddress(source, owner, ()),
                CollectionKind.BONES,
                None,
                index,
                bone,
            )
        case _ as unreachable:
            assert_never(unreachable)


def _resolve_flesh(spec: dict, address: RecordAddress, location: FleshLocation) -> RecordResolution:
    owner = _bone_owner(spec, location.owner, address.source)
    if isinstance(owner, EditObstructed):
        return owner
    bone = owner.record
    flesh = _records(bone.get("flesh", []))
    arrays = _records(bone.get("arrays", []))
    match location.selector:
        case FleshIndexSelector(index):
            return (
                ListedRecord(address, CollectionKind.FLESH, location.owner, index, flesh[index])
                if 0 <= index < len(flesh) and isinstance(flesh[index], dict)
                else _obstruct(address.source, f"flesh index {index} out of range ({len(flesh)})")
            )
        case ArrayIndexSelector(index):
            return (
                ListedRecord(address, CollectionKind.ARRAYS, location.owner, index, arrays[index])
                if 0 <= index < len(arrays) and isinstance(arrays[index], dict)
                else _obstruct(address.source, f"array index {index} out of range ({len(arrays)})")
            )
        case NamedSelector(name):
            flesh_found = _find_index(flesh, "name", name)
            array_found = _find_index(arrays, "name", name)
            if flesh_found is not None:
                index, record = flesh_found
                return ListedRecord(address, CollectionKind.FLESH, location.owner, index, record)
            if array_found is not None:
                index, record = array_found
                return ListedRecord(address, CollectionKind.ARRAYS, location.owner, index, record)
            return _obstruct(address.source, f"no flesh or array named {name!r}")
        case _ as unreachable:
            assert_never(unreachable)


def resolve_record(spec: dict, address: RecordAddress | str) -> RecordResolution:
    decoded = decode_address(address) if isinstance(address, str) else address
    if isinstance(decoded, EditObstructed):
        return decoded
    location = decoded.location
    match location:
        case MetaLocation():
            return FixedRecord(decoded, spec)
        case SkeletonRootLocation() | BoneLocation():
            return _bone_owner(spec, location, decoded.source)
        case FleshLocation():
            return _resolve_flesh(spec, decoded, location)
        case PoseLocation(bone_id):
            joints = _pose(spec).get("joints", {})
            if not isinstance(joints, dict) or bone_id not in joints or not isinstance(joints[bone_id], dict):
                return _obstruct(decoded.source, f"no pose entry for bone {bone_id!r}")
            return FixedRecord(decoded, joints[bone_id])
        case GoalsLocation() | PropsLocation() | MountsLocation() | ContactsLocation():
            return _obstruct(decoded.source, "address names a collection, not a record")
        case GoalLocation(goal_id):
            goals = _records(_pose(spec).get("goals", []))
            found = _find_index(goals, "id", goal_id)
            if found is None:
                return _obstruct(decoded.source, f"no goal {goal_id!r}")
            index, record = found
            return ListedRecord(decoded, CollectionKind.GOALS, None, index, record)
        case PropLocation(prop_id):
            props = _records(spec.get("props", []))
            found = _find_index(props, "id", prop_id)
            if found is None:
                return _obstruct(decoded.source, f"no prop {prop_id!r}")
            index, record = found
            return ListedRecord(decoded, CollectionKind.PROPS, None, index, record)
        case PartLocation(prop_id, part_id):
            prop_address = RecordAddress(
                f"props/{encode_address_segment(prop_id)}",
                PropLocation(prop_id),
                (),
            )
            prop = resolve_record(spec, prop_address)
            if isinstance(prop, EditObstructed):
                return prop
            parts = _records(prop.record.get("parts", []))
            found = _find_index(parts, "id", part_id)
            if found is None:
                return _obstruct(decoded.source, f"no part {part_id!r} in prop {prop_id!r}")
            index, record = found
            return ListedRecord(decoded, CollectionKind.PARTS, PropLocation(prop_id), index, record)
        case MountLocation(selector):
            mounts = _records(spec.get("mounts", []))
            match selector:
                case IndexedMountSelector(index):
                    return (
                        ListedRecord(decoded, CollectionKind.MOUNTS, None, index, mounts[index])
                        if 0 <= index < len(mounts) and isinstance(mounts[index], dict)
                        else _obstruct(decoded.source, f"mount index {index} out of range")
                    )
                case NamedMountSelector(port):
                    found = _find_index(mounts, "port", port)
                    if found is None:
                        return _obstruct(decoded.source, f"no mount at port {port!r}")
                    index, record = found
                    return ListedRecord(decoded, CollectionKind.MOUNTS, None, index, record)
                case _ as unreachable:
                    assert_never(unreachable)
        case ContactLocation(group):
            contacts = _records(spec.get("contacts", []))
            found = _find_index(contacts, "group", group)
            if found is None:
                return _obstruct(decoded.source, f"no contact group {group!r}")
            index, record = found
            return ListedRecord(decoded, CollectionKind.CONTACTS, None, index, record)
        case _ as unreachable:
            assert_never(unreachable)


def _replace_index(values: list, index: int, value: object) -> list:
    return [*values[:index], value, *values[index + 1 :]]


def _insert_index(values: list, index: int, value: object) -> list:
    return [*values[:index], value, *values[index:]]


def _remove_index(values: list, index: int) -> list:
    return [*values[:index], *values[index + 1 :]]


def _replace_bone_owner(spec: dict, owner: SkeletonRootLocation | BoneLocation, bone: dict) -> dict:
    skeleton = _skeleton(spec)
    match owner:
        case SkeletonRootLocation():
            return {**spec, "skeleton": {**skeleton, "root": bone}}
        case BoneLocation(bone_id):
            bones = _records(skeleton.get("bones", []))
            found = _find_index(bones, "id", bone_id)
            return (
                {**spec, "skeleton": {**skeleton, "bones": _replace_index(bones, found[0], bone)}}
                if found is not None
                else spec
            )
        case _ as unreachable:
            assert_never(unreachable)


def _replace_listed(spec: dict, resolved: ListedRecord, records: list) -> dict:
    match resolved.collection:
        case CollectionKind.BONES:
            skeleton = _skeleton(spec)
            return {**spec, "skeleton": {**skeleton, "bones": records}}
        case CollectionKind.FLESH | CollectionKind.ARRAYS:
            owner = resolved.owner
            if not isinstance(owner, (SkeletonRootLocation, BoneLocation)):
                return spec
            owner_record = _bone_owner(spec, owner, resolved.address.source)
            if isinstance(owner_record, EditObstructed):
                return spec
            key = resolved.collection.value
            return _replace_bone_owner(spec, owner, {**owner_record.record, key: records})
        case CollectionKind.GOALS:
            pose = _pose(spec)
            return {**spec, "pose": {**pose, "goals": records}}
        case CollectionKind.PROPS:
            return {**spec, "props": records}
        case CollectionKind.PARTS:
            owner = resolved.owner
            if not isinstance(owner, PropLocation):
                return spec
            prop = resolve_record(
                spec,
                RecordAddress(
                    f"props/{encode_address_segment(owner.prop_id)}",
                    owner,
                    (),
                ),
            )
            if not isinstance(prop, ListedRecord):
                return spec
            return _replace_listed(
                spec,
                prop,
                _replace_index(_records(spec.get("props", [])), prop.index, {**prop.record, "parts": records}),
            )
        case CollectionKind.MOUNTS:
            return {**spec, "mounts": records}
        case CollectionKind.CONTACTS:
            return {**spec, "contacts": records}
        case _ as unreachable:
            assert_never(unreachable)


def _replace_record(spec: dict, resolved: FixedRecord | ListedRecord, record: dict) -> dict:
    location = resolved.address.location
    match resolved:
        case FixedRecord():
            match location:
                case MetaLocation():
                    return record
                case SkeletonRootLocation():
                    return _replace_bone_owner(spec, location, record)
                case PoseLocation(bone_id):
                    pose = _pose(spec)
                    joints = pose.get("joints", {})
                    current = joints if isinstance(joints, dict) else {}
                    return {**spec, "pose": {**pose, "joints": {**current, bone_id: record}}}
                case _:
                    return spec
        case ListedRecord():
            records = _listed_records(spec, resolved)
            return _replace_listed(spec, resolved, _replace_index(records, resolved.index, record))
        case _ as unreachable:
            assert_never(unreachable)


def _listed_records(spec: dict, resolved: ListedRecord) -> list:
    match resolved.collection:
        case CollectionKind.BONES:
            return _records(_skeleton(spec).get("bones", []))
        case CollectionKind.FLESH | CollectionKind.ARRAYS:
            owner = resolved.owner
            if not isinstance(owner, (SkeletonRootLocation, BoneLocation)):
                return []
            bone = _bone_owner(spec, owner, resolved.address.source)
            return [] if isinstance(bone, EditObstructed) else _records(bone.record.get(resolved.collection.value, []))
        case CollectionKind.GOALS:
            return _records(_pose(spec).get("goals", []))
        case CollectionKind.PROPS:
            return _records(spec.get("props", []))
        case CollectionKind.PARTS:
            owner = resolved.owner
            if not isinstance(owner, PropLocation):
                return []
            prop = resolve_record(
                spec,
                RecordAddress(
                    f"props/{encode_address_segment(owner.prop_id)}",
                    owner,
                    (),
                ),
            )
            return [] if isinstance(prop, EditObstructed) else _records(prop.record.get("parts", []))
        case CollectionKind.MOUNTS:
            return _records(spec.get("mounts", []))
        case CollectionKind.CONTACTS:
            return _records(spec.get("contacts", []))
        case _ as unreachable:
            assert_never(unreachable)


@dataclass(frozen=True)
class FieldUpdate:
    value: object
    previous: object


def _set_field(current: object, tokens: tuple[str | int, ...], value: object, address: str) -> FieldUpdate | EditObstructed:
    if not tokens:
        return _obstruct(address, "op needs an @field on this address")
    token, *rest = tokens
    tail = tuple(rest)
    if isinstance(token, str):
        if not isinstance(current, dict):
            return _obstruct(address, f"no field {token!r} on the way to the target")
        previous = current.get(token, _MISSING)
        if not tail:
            return FieldUpdate({**current, token: value}, previous)
        if previous is _MISSING:
            return _obstruct(address, f"no field {token!r} on the way to the target")
        nested = _set_field(previous, tail, value, address)
        return nested if isinstance(nested, EditObstructed) else FieldUpdate({**current, token: nested.value}, nested.previous)
    if not isinstance(current, list) or not 0 <= token < len(current):
        return _obstruct(address, f"index {token} out of range")
    if not tail:
        return FieldUpdate(_replace_index(current, token, value), current[token])
    nested = _set_field(current[token], tail, value, address)
    return nested if isinstance(nested, EditObstructed) else FieldUpdate(_replace_index(current, token, nested.value), nested.previous)


def _unset_field(current: object, tokens: tuple[str | int, ...], address: str) -> FieldUpdate | EditObstructed:
    if not tokens:
        return _obstruct(address, "op needs an @field on this address")
    token, *rest = tokens
    tail = tuple(rest)
    if isinstance(token, int):
        if not tail:
            return _obstruct(address, "cannot unset a list element (use remove)")
        if not isinstance(current, list) or not 0 <= token < len(current):
            return _obstruct(address, f"index {token} out of range")
        nested = _unset_field(current[token], tail, address)
        return nested if isinstance(nested, EditObstructed) else FieldUpdate(_replace_index(current, token, nested.value), nested.previous)
    if not isinstance(current, dict) or token not in current:
        return _obstruct(address, "cannot unset a missing field" if not tail else f"no field {token!r} on the way to the target")
    if not tail:
        return FieldUpdate({key: value for key, value in current.items() if key != token}, current[token])
    nested = _unset_field(current[token], tail, address)
    return nested if isinstance(nested, EditObstructed) else FieldUpdate({**current, token: nested.value}, nested.previous)


def _check_value(address: str, key: str | int, old: object, new: object) -> EditObstructed | None:
    if isinstance(key, str) and key in _VEC3 and not _is_vec(new, 3):
        return _obstruct(address, f"{key} must be a 3-vector of numbers")
    if old is _MISSING:
        return None
    mismatch = (
        "expected a number"
        if is_number(old) and not is_number(new)
        else "expected a string"
        if isinstance(old, str) and not isinstance(new, str)
        else "expected a list"
        if isinstance(old, list) and not isinstance(new, list)
        else "expected an object"
        if isinstance(old, dict) and not isinstance(new, dict)
        else None
    )
    return _obstruct(address, f"{mismatch}, got {type(new).__name__}") if mismatch else None


def _fmt(value: object) -> str:
    rendered = json.dumps(value)
    return rendered if len(rendered) <= 48 else rendered[:45] + "..."


def _touched(address: RecordAddress, extra: tuple[str, ...] = ()) -> frozenset[str]:
    return frozenset((address.record_source, *extra))


def _compile_set(
    spec: dict,
    op: SetOp,
    validate_inverse: bool,
) -> EditCompileResult:
    address = op.addr
    location = address.location
    value = _thaw_json(op.value)
    if isinstance(location, MetaLocation) and not address.fields:
        return (
            EditPlan(
                value,
                SetOp(address, _freeze_json(spec)),
                "replace authored document",
                frozenset({"meta"}),
            )
            if isinstance(value, dict)
            else _obstruct(
                address.source,
                "root document replacement requires a JSON object",
            )
        )
    if isinstance(location, PoseLocation) and not (
        len(address.fields) == 1 and address.fields[0] in _POSE_KEYS
    ):
        return _obstruct(address.source, f"pose fields are {sorted(_POSE_KEYS)}")
    resolved = resolve_record(spec, address)
    pose = _pose(spec)
    joints = pose.get("joints", {})
    created_pose = (
        isinstance(location, PoseLocation)
        and isinstance(joints, dict)
        and location.bone_id not in joints
    )
    if created_pose:
        bones = _records(_skeleton(spec).get("bones", []))
        if not any(isinstance(bone, dict) and bone.get("id") == location.bone_id for bone in bones):
            return _obstruct(address.source, f"no bone {location.bone_id!r} to pose")
        resolved = FixedRecord(address, {})
    if isinstance(resolved, EditObstructed):
        return resolved
    updated = _set_field(resolved.record, address.fields, value, address.source)
    if isinstance(updated, EditObstructed):
        return updated
    checked = (
        None
        if isinstance(location, MetaLocation) and len(address.fields) == 1
        else _check_value(
            address.source,
            address.fields[-1],
            updated.previous,
            value,
        )
    )
    if checked is not None:
        return checked
    next_spec = _replace_record(spec, resolved, updated.value)
    candidate: Op = (
        RemoveOp(
            RecordAddress(
                f"pose/{encode_address_segment(location.bone_id)}",
                location,
                (),
            )
        )
        if created_pose and isinstance(location, PoseLocation)
        else UnsetOp(address)
        if updated.previous is _MISSING
        else SetOp(address, _freeze_json(updated.previous))
    )
    inverse = (
        _validated_inverse(spec, next_spec, candidate)
        if validate_inverse and created_pose
        else candidate
    )
    label = (
        f"set {address.source} (new pose) = {_fmt(value)}"
        if created_pose
        else f"set {address.source} (new) = {_fmt(value)}"
        if updated.previous is _MISSING
        else f"set {address.source} {_fmt(updated.previous)} -> {_fmt(value)}"
    )
    return EditPlan(next_spec, inverse, label, _touched(address))


def _compile_unset(
    spec: dict,
    op: UnsetOp,
    validate_inverse: bool,
) -> EditCompileResult:
    resolved = resolve_record(spec, op.addr)
    if isinstance(resolved, EditObstructed):
        return resolved
    updated = _unset_field(resolved.record, op.addr.fields, op.addr.source)
    if isinstance(updated, EditObstructed):
        return updated
    next_spec = _replace_record(spec, resolved, updated.value)
    candidate = SetOp(op.addr, _freeze_json(updated.previous))
    inverse = (
        _validated_inverse(spec, next_spec, candidate)
        if validate_inverse
        else candidate
    )
    return EditPlan(
        next_spec,
        inverse,
        f"unset {op.addr.source}",
        _touched(op.addr),
    )


def _payload_identity(payload: AddPayload) -> str | None:
    match payload:
        case BonePayload(identifier=identifier) | _IdentifiedPayload(identifier=identifier):
            return identifier
        case FleshPayload(name=name) | ArrayPayload(name=name):
            return name
        case ContactPayload(group=group):
            return group
        case MountPayload(port=port):
            return port
        case PosePayload():
            return None
        case _ as unreachable:
            assert_never(unreachable)


def _inserted(values: list, op: AddOp | AddAtOp, payload: dict, address: str) -> list | EditObstructed:
    index = op.index if isinstance(op, AddAtOp) else len(values)
    return (
        _insert_index(values, index, payload)
        if 0 <= index <= len(values)
        else _obstruct(address, f"add index {index} out of range")
    )


def _validated_inverse(
    previous_spec: dict,
    next_spec: dict,
    candidate: Op,
) -> Op:
    compiled = _compile_op(next_spec, candidate, False)
    return (
        candidate
        if isinstance(compiled, EditPlan)
        and _canonical_json(compiled.next_spec) == _canonical_json(previous_spec)
        else SetOp(
            RecordAddress("meta", MetaLocation(), ()),
            _freeze_json(previous_spec),
        )
    )


def _compile_add(
    spec: dict,
    op: AddOp | AddAtOp,
    validate_inverse: bool,
) -> EditCompileResult:
    address = op.addr
    if address.fields:
        return _obstruct(address.source, "add takes a record address, not an @field")
    params = _payload_json(op.payload)
    location = address.location
    kind = op.kind
    if op.payload._kind is not kind:
        return _obstruct(
            address.source,
            f"add {kind.value} received the wrong payload carrier",
        )
    match kind:
        case AddKind.BONE:
            if not isinstance(op.payload, BonePayload) or not isinstance(location, (SkeletonRootLocation, BoneLocation)):
                return _obstruct(address.source, "add bone under skeleton/<parent> or skeleton/root")
            parent = _bone_owner(spec, location, address.source)
            if isinstance(parent, EditObstructed) and not isinstance(location, SkeletonRootLocation):
                return parent
            skeleton = _skeleton(spec)
            bones = _records(skeleton.get("bones", []))
            root = skeleton.get("root", {})
            root_id = root.get("id", "root") if isinstance(root, dict) else "root"
            identifiers = frozenset(
                bone.get("id") for bone in bones if isinstance(bone, dict)
            ) | frozenset({root_id})
            if op.payload.identifier in identifiers:
                return _obstruct(address.source, f"bone id {op.payload.identifier!r} already exists")
            inserted = _inserted(
                bones,
                op,
                {**params, "parent": params.get("parent", root_id if isinstance(location, SkeletonRootLocation) else location.bone_id)},
                address.source,
            )
            if isinstance(inserted, EditObstructed):
                return inserted
            new_address = (
                f"skeleton/{encode_address_segment(op.payload.identifier)}"
            )
            next_spec = {**spec, "skeleton": {**skeleton, "bones": inserted}}
        case AddKind.FLESH | AddKind.ARRAY:
            if not isinstance(location, (SkeletonRootLocation, BoneLocation)):
                return _obstruct(address.source, f"add {kind.value} under skeleton/<bone>")
            owner = _bone_owner(spec, location, address.source)
            if isinstance(owner, EditObstructed):
                return owner
            flesh = _records(owner.record.get("flesh", []))
            arrays = _records(owner.record.get("arrays", []))
            identity = _payload_identity(op.payload)
            taken = frozenset(
                record.get("name")
                for record in [*flesh, *arrays]
                if isinstance(record, dict)
            )
            if identity in taken:
                noun = "flesh name" if kind is AddKind.FLESH else "name"
                return _obstruct(
                    address.source,
                    f"{noun} {identity!r} already on this bone",
                )
            target = flesh if kind is AddKind.FLESH else arrays
            inserted = _inserted(target, op, params, address.source)
            if isinstance(inserted, EditObstructed):
                return inserted
            key = "flesh" if kind is AddKind.FLESH else "arrays"
            next_spec = _replace_bone_owner(spec, location, {**owner.record, key: inserted})
            new_address = (
                f"{address.source}/{encode_address_segment(str(identity))}"
            )
        case AddKind.PROP:
            if not isinstance(op.payload, PropPayload) or not isinstance(location, PropsLocation):
                return _obstruct(address.source, "add prop at address 'props'")
            props = _records(spec.get("props", []))
            if _find_index(props, "id", op.payload.identifier) is not None:
                return _obstruct(address.source, f"prop {op.payload.identifier!r} already exists")
            inserted = _inserted(props, op, params, address.source)
            if isinstance(inserted, EditObstructed):
                return inserted
            next_spec = {**spec, "props": inserted}
            new_address = f"props/{encode_address_segment(op.payload.identifier)}"
        case AddKind.PART:
            if not isinstance(op.payload, PartPayload) or not isinstance(location, PropLocation):
                return _obstruct(address.source, "add part under props/<prop_id>")
            prop = resolve_record(spec, address)
            if not isinstance(prop, ListedRecord):
                return prop if isinstance(prop, EditObstructed) else _obstruct(address.source, "no prop")
            parts = _records(prop.record.get("parts", []))
            if _find_index(parts, "id", op.payload.identifier) is not None:
                return _obstruct(address.source, f"part {op.payload.identifier!r} already in prop")
            inserted = _inserted(parts, op, params, address.source)
            if isinstance(inserted, EditObstructed):
                return inserted
            next_spec = _replace_record(spec, prop, {**prop.record, "parts": inserted})
            new_address = (
                f"{address.source}/"
                f"{encode_address_segment(op.payload.identifier)}"
            )
        case AddKind.GOAL:
            if not isinstance(op.payload, GoalPayload) or not isinstance(location, GoalsLocation):
                return _obstruct(address.source, "add goal at address 'pose/goals'")
            pose = _pose(spec)
            goals = _records(pose.get("goals", []))
            if _find_index(goals, "id", op.payload.identifier) is not None:
                return _obstruct(address.source, f"goal {op.payload.identifier!r} already exists")
            inserted = _inserted(goals, op, params, address.source)
            if isinstance(inserted, EditObstructed):
                return inserted
            next_spec = {**spec, "pose": {**pose, "goals": inserted}}
            new_address = (
                f"pose/goals/{encode_address_segment(op.payload.identifier)}"
            )
        case AddKind.CONTACT:
            if not isinstance(op.payload, ContactPayload) or not isinstance(location, ContactsLocation):
                return _obstruct(address.source, "add contact at address 'contacts'")
            contacts = _records(spec.get("contacts", []))
            if _find_index(contacts, "group", op.payload.group) is not None:
                return _obstruct(address.source, f"contact group {op.payload.group!r} already exists")
            inserted = _inserted(contacts, op, params, address.source)
            if isinstance(inserted, EditObstructed):
                return inserted
            next_spec = {**spec, "contacts": inserted}
            new_address = f"contacts/{encode_address_segment(op.payload.group)}"
        case AddKind.MOUNT:
            if not isinstance(op.payload, MountPayload) or not isinstance(location, MountsLocation):
                return _obstruct(address.source, "add mount at address 'mounts'")
            mounts = _records(spec.get("mounts", []))
            inserted = _inserted(mounts, op, params, address.source)
            if isinstance(inserted, EditObstructed):
                return inserted
            next_spec = {**spec, "mounts": inserted}
            new_address = f"mounts/{encode_address_segment(op.payload.port)}"
        case AddKind.POSE:
            if not isinstance(op.payload, PosePayload) or not isinstance(location, PoseLocation):
                return _obstruct(address.source, "add pose at address 'pose/<bone_id>'")
            bones = _records(_skeleton(spec).get("bones", []))
            if _find_index(bones, "id", location.bone_id) is None:
                return _obstruct(address.source, f"no bone {location.bone_id!r} to pose")
            pose = _pose(spec)
            joints_value = pose.get("joints", {})
            joints = joints_value if isinstance(joints_value, dict) else {}
            if location.bone_id in joints:
                return _obstruct(address.source, f"pose entry for {location.bone_id!r} exists (use set)")
            next_spec = {**spec, "pose": {**pose, "joints": {**joints, location.bone_id: params}}}
            new_address = address.source
        case _ as unreachable:
            assert_never(unreachable)
    inverse_address = decode_address(new_address)
    if isinstance(inverse_address, EditObstructed):
        return inverse_address
    candidate = RemoveOp(inverse_address)
    inverse = (
        _validated_inverse(spec, next_spec, candidate)
        if validate_inverse
        else candidate
    )
    return EditPlan(
        next_spec,
        inverse,
        f"add {kind.value} {new_address}",
        frozenset({address.record_source, new_address}),
    )


def _kind_of(resolved: FixedRecord | ListedRecord) -> AddKind | EditObstructed:
    location = resolved.address.location
    match location:
        case FleshLocation():
            return AddKind.ARRAY if isinstance(resolved, ListedRecord) and resolved.collection is CollectionKind.ARRAYS else AddKind.FLESH
        case BoneLocation():
            return AddKind.BONE
        case GoalLocation():
            return AddKind.GOAL
        case PoseLocation():
            return AddKind.POSE
        case PartLocation():
            return AddKind.PART
        case PropLocation():
            return AddKind.PROP
        case MountLocation():
            return AddKind.MOUNT
        case ContactLocation():
            return AddKind.CONTACT
        case _:
            return _obstruct(resolved.address.source, "cannot remove at this address")


def _parent_address(spec: dict, resolved: ListedRecord, kind: AddKind) -> str:
    location = resolved.address.location
    if kind in (AddKind.FLESH, AddKind.ARRAY, AddKind.PART):
        return "/".join(resolved.address.record_source.split("/")[:-1])
    if kind is AddKind.BONE:
        parent = resolved.record.get("parent", "root")
        bones = _records(_skeleton(spec).get("bones", []))
        return (
            f"skeleton/{encode_address_segment(str(parent))}"
            if _find_index(bones, "id", parent) is not None
            else "skeleton/root"
        )
    if kind is AddKind.GOAL:
        return "pose/goals"
    match location:
        case PropLocation():
            return "props"
        case MountLocation():
            return "mounts"
        case ContactLocation():
            return "contacts"
        case _:
            return resolved.address.record_source


def _compile_remove(
    spec: dict,
    op: RemoveOp,
    validate_inverse: bool,
) -> EditCompileResult:
    if op.addr.fields:
        return _obstruct(op.addr.source, "remove takes a record address (use unset for fields)")
    resolved = resolve_record(spec, op.addr)
    if isinstance(resolved, EditObstructed):
        return resolved
    kind = _kind_of(resolved)
    if isinstance(kind, EditObstructed):
        return kind
    if kind is AddKind.POSE and isinstance(op.addr.location, PoseLocation):
        pose = _pose(spec)
        joints_value = pose.get("joints", {})
        joints = joints_value if isinstance(joints_value, dict) else {}
        record = joints.get(op.addr.location.bone_id)
        if not isinstance(record, dict):
            return _obstruct(op.addr.source, f"no pose entry {op.addr.location.bone_id!r}")
        payload = _decode_payload(AddKind.POSE, record, op.addr.source)
        if isinstance(payload, EditObstructed):
            return payload
        next_joints = {key: value for key, value in joints.items() if key != op.addr.location.bone_id}
        next_spec = {**spec, "pose": {**pose, "joints": next_joints}}
        candidate = AddOp(AddKind.POSE, op.addr, payload)
        inverse = (
            _validated_inverse(spec, next_spec, candidate)
            if validate_inverse
            else candidate
        )
        return EditPlan(
            next_spec,
            inverse,
            f"remove pose {op.addr.source}",
            _touched(op.addr),
        )
    if not isinstance(resolved, ListedRecord):
        return _obstruct(op.addr.source, "this record cannot be removed")
    if kind is AddKind.BONE:
        children = tuple(
            bone.get("id")
            for bone in _records(_skeleton(spec).get("bones", []))
            if isinstance(bone, dict) and bone.get("parent") == resolved.record.get("id")
        )
        if children:
            return _obstruct(op.addr.source, f"bone has children {list(children)}; remove them first")
    payload = _decode_payload(kind, resolved.record, op.addr.source)
    if isinstance(payload, EditObstructed):
        return payload
    parent_source = _parent_address(spec, resolved, kind)
    parent_address = decode_address(parent_source)
    if isinstance(parent_address, EditObstructed):
        return parent_address
    records = _listed_records(spec, resolved)
    next_spec = _replace_listed(spec, resolved, _remove_index(records, resolved.index))
    candidate = AddAtOp(kind, parent_address, payload, resolved.index)
    inverse = (
        _validated_inverse(spec, next_spec, candidate)
        if validate_inverse
        else candidate
    )
    return EditPlan(
        next_spec,
        inverse,
        f"remove {kind.value} {op.addr.source}",
        frozenset({op.addr.record_source, parent_source}),
    )


def _renamed_anatomy(anatomy: object, old: str, new: str) -> tuple[object, tuple[str, ...]]:
    if not isinstance(anatomy, dict):
        return anatomy, ()
    overall = anatomy.get("overall")
    if not isinstance(overall, dict):
        return anatomy, ()
    regions = overall.get("regions")
    if not isinstance(regions, list):
        return anatomy, ()
    addresses = tuple(
        f"anatomy/overall/regions[{index}]@host_bone_id"
        for index, region in enumerate(regions)
        if isinstance(region, dict) and region.get("host_bone_id") == old
    )
    renamed = [
        {**region, "host_bone_id": new}
        if isinstance(region, dict) and region.get("host_bone_id") == old
        else region
        for region in regions
    ]
    return {**anatomy, "overall": {**overall, "regions": renamed}}, addresses


def _rename_bone_refs(spec: dict, old: str, new: str) -> tuple[dict, tuple[str, ...]]:
    anatomy, anatomy_addresses = _renamed_anatomy(spec.get("anatomy"), old, new)
    skeleton = _skeleton(spec)
    bones = _records(skeleton.get("bones", []))
    parent_addresses = tuple(
        f"skeleton/{encode_address_segment(str(bone.get('id')))}@parent"
        for bone in bones
        if isinstance(bone, dict) and bone.get("parent") == old
    )
    renamed_bones = [
        {**bone, "parent": new}
        if isinstance(bone, dict) and bone.get("parent") == old
        else bone
        for bone in bones
    ]
    pose = _pose(spec)
    joints_value = pose.get("joints", {})
    joints = joints_value if isinstance(joints_value, dict) else {}
    renamed_joints = (
        {
            **{key: value for key, value in joints.items() if key not in (old, new)},
            new: joints[old],
        }
        if old in joints
        else joints
    )
    joint_addresses = (
        (f"pose/{encode_address_segment(new)}",)
        if old in joints
        else ()
    )
    goals = _records(pose.get("goals", []))
    goal_addresses = tuple(
        chain.from_iterable(
            (
                f"pose/goals/"
                f"{encode_address_segment(str(goal.get('id')))}"
                f"@chain[{index}]"
                for index, bone_id in enumerate(goal.get("chain", []))
                if bone_id == old
            )
            for goal in goals
            if isinstance(goal, dict)
        )
    )
    renamed_goals = [
        {
            **goal,
            "chain": [new if bone_id == old else bone_id for bone_id in goal.get("chain", [])],
        }
        if isinstance(goal, dict)
        else goal
        for goal in goals
    ]
    contacts = _records(spec.get("contacts", []))
    contact_addresses = tuple(
        chain.from_iterable(
            (
                f"contacts/"
                f"{encode_address_segment(str(contact.get('group')))}"
                f"@parts[{index}]"
                for index, part in enumerate(contact.get("parts", []))
                if part == old
            )
            for contact in contacts
            if isinstance(contact, dict)
        )
    )
    renamed_contacts = [
        {
            **contact,
            "parts": [new if part == old else part for part in contact.get("parts", [])],
        }
        if isinstance(contact, dict)
        else contact
        for contact in contacts
    ]
    mounts = _records(spec.get("mounts", []))
    mount_addresses = tuple(
        f"mounts/"
        f"{encode_address_segment(new + str(mount.get('port', ''))[len(old):])}"
        "@port"
        for mount in mounts
        if isinstance(mount, dict) and str(mount.get("port", "")).split("/")[0] == old
    )
    renamed_mounts = [
        {
            **mount,
            "port": "/".join((new, *str(mount.get("port", "")).split("/")[1:])),
        }
        if isinstance(mount, dict) and str(mount.get("port", "")).split("/")[0] == old
        else mount
        for mount in mounts
    ]
    next_spec = {
        **spec,
        **({"anatomy": anatomy} if "anatomy" in spec else {}),
        "skeleton": {**skeleton, "bones": renamed_bones},
        **(
            {"pose": {**pose, "joints": renamed_joints, "goals": renamed_goals}}
            if "pose" in spec
            else {}
        ),
        **({"contacts": renamed_contacts} if "contacts" in spec else {}),
        **({"mounts": renamed_mounts} if "mounts" in spec else {}),
    }
    return next_spec, (
        *anatomy_addresses,
        *parent_addresses,
        *joint_addresses,
        *goal_addresses,
        *contact_addresses,
        *mount_addresses,
    )


def _rename_flesh_refs(spec: dict, old: str, new: str) -> tuple[dict, tuple[str, ...]]:
    contacts = _records(spec.get("contacts", []))
    addresses = tuple(
        chain.from_iterable(
            (
                f"contacts/"
                f"{encode_address_segment(str(contact.get('group')))}"
                f"@parts[{index}]"
                for index, part in enumerate(contact.get("parts", []))
                if part == old
            )
            for contact in contacts
            if isinstance(contact, dict)
        )
    )
    renamed = [
        {
            **contact,
            "parts": [new if part == old else part for part in contact.get("parts", [])],
        }
        if isinstance(contact, dict)
        else contact
        for contact in contacts
    ]
    return (
        {**spec, "contacts": renamed} if "contacts" in spec else spec,
        addresses,
    )


def _compile_rename(
    spec: dict,
    op: RenameOp,
    validate_inverse: bool,
) -> EditCompileResult:
    if op.addr.fields:
        return _obstruct(op.addr.source, "rename takes a record address")
    if not isinstance(parse_dotted_qualified_token(op.to), DottedQualifiedToken):
        return _obstruct(op.addr.source, f"bad name {op.to!r}")
    resolved = resolve_record(spec, op.addr)
    if isinstance(resolved, EditObstructed):
        return resolved
    kind = _kind_of(resolved)
    if isinstance(kind, EditObstructed):
        return kind
    if kind is AddKind.BONE and isinstance(resolved, ListedRecord):
        old = resolved.record.get("id")
        if not isinstance(old, str):
            return _obstruct(op.addr.source, "bone has no string id")
        identifiers = frozenset(
            bone.get("id")
            for bone in _records(_skeleton(spec).get("bones", []))
            if isinstance(bone, dict)
        )
        if op.to in identifiers:
            return _obstruct(op.addr.source, f"bone {op.to!r} already exists")
        renamed_record_spec = _replace_record(spec, resolved, {**resolved.record, "id": op.to})
        next_spec, refs = _rename_bone_refs(renamed_record_spec, old, op.to)
        inverse_address = decode_address(
            f"skeleton/{encode_address_segment(op.to)}"
        )
    elif kind in (AddKind.FLESH, AddKind.ARRAY) and isinstance(resolved, ListedRecord):
        old = resolved.record.get("name")
        if not isinstance(old, str):
            return _obstruct(op.addr.source, "flesh has no string name")
        renamed_record_spec = _replace_record(spec, resolved, {**resolved.record, "name": op.to})
        next_spec, refs = _rename_flesh_refs(renamed_record_spec, old, op.to)
        parent = "/".join(op.addr.record_source.split("/")[:-1])
        inverse_address = decode_address(
            f"{parent}/{encode_address_segment(op.to)}"
        )
    else:
        return _obstruct(op.addr.source, f"rename not supported for {kind.value}")
    if isinstance(inverse_address, EditObstructed):
        return inverse_address
    label = f"rename {op.addr.source} -> {op.to}" + (
        f" (rewrote {len(refs)} refs)" if refs else ""
    )
    candidate = RenameOp(inverse_address, old)
    inverse = (
        _validated_inverse(spec, next_spec, candidate)
        if validate_inverse
        else candidate
    )
    return EditPlan(
        next_spec,
        inverse,
        label,
        _touched(op.addr, (inverse_address.record_source, *refs)),
    )


def _compile_op(
    spec: dict,
    op: Op,
    validate_inverse: bool,
) -> EditCompileResult:
    match op:
        case SetOp():
            return _compile_set(spec, op, validate_inverse)
        case UnsetOp():
            return _compile_unset(spec, op, validate_inverse)
        case AddOp() | AddAtOp():
            return _compile_add(spec, op, validate_inverse)
        case RemoveOp():
            return _compile_remove(spec, op, validate_inverse)
        case RenameOp():
            return _compile_rename(spec, op, validate_inverse)
        case _ as unreachable:
            assert_never(unreachable)


def compile_op(spec: dict, op: Op) -> EditCompileResult:
    return _compile_op(spec, op, True)


def compile_edit(spec: dict, raw: dict | Op) -> EditCompileResult:
    op = decode_op(raw) if isinstance(raw, dict) else raw
    return op if isinstance(op, EditObstructed) else compile_op(spec, op)


@dataclass(frozen=True)
class _ProgramAccumulator:
    spec: dict
    edits: tuple[EditPlan, ...]
    obstruction: EditObstructed | None


def _compile_program_step(accumulator: _ProgramAccumulator, op: dict | Op) -> _ProgramAccumulator:
    if accumulator.obstruction is not None:
        return accumulator
    result = compile_edit(accumulator.spec, op)
    return (
        _ProgramAccumulator(accumulator.spec, accumulator.edits, result)
        if isinstance(result, EditObstructed)
        else _ProgramAccumulator(result.next_spec, (*accumulator.edits, result), None)
    )


def compile_program(spec: dict, operations: tuple[dict | Op, ...]) -> ProgramCompileResult:
    compiled = reduce(_compile_program_step, operations, _ProgramAccumulator(spec, (), None))
    if compiled.obstruction is not None:
        return compiled.obstruction
    return ProgramPlan(
        compiled.spec,
        compiled.edits,
        frozenset(chain.from_iterable(edit.touched_addresses for edit in compiled.edits)),
    )


def commit_edit_plan(spec: dict, plan: EditPlan | ProgramPlan) -> None:
    spec.clear()
    spec.update(plan.next_spec)


def apply_op(spec: dict, op: dict) -> tuple[dict, str]:
    result = compile_edit(spec, op)
    if isinstance(result, EditObstructed):
        raise Reject(result.first.address, result.first.reason)
    commit_edit_plan(spec, result)
    return to_json(result.inverse), result.label
