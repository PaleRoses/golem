"""Typed eye authoring, bone-local placement, and morphology projection."""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np

from golem.kernel.body.canonical import _r, _rvec
from golem.kernel.body.decode import (
    _finite_number,
    _malformed,
    _rotation_payload,
    _unknown_field_obstructions,
    _vector3,
)
from golem.kernel.body.geometry import placed_center
from golem.kernel.body.types import (
    AcceptedEyes,
    BrowRidge,
    DuplicateEyeIdObstruction,
    EyeCompileResult,
    EyeGlobe,
    EyeHostFrame,
    EyeObstruction,
    EyePartCollisionObstruction,
    EyeRule,
    EyeSocket,
    EyeSpec,
    MalformedEyeObstruction,
    NonGlossyEyeMaterialObstruction,
    RejectedEyes,
    UnknownEyeHostObstruction,
    UnknownEyeMaterialObstruction,
)
from golem.materials import (
    AcceptedMaterial,
    AppearanceMaterialId,
    decode_appearance_material,
)


_EYE_ID_SYNTAX = "[A-Za-z][A-Za-z0-9_.-]*"
_EYE_ID = re.compile(rf"{_EYE_ID_SYNTAX}\Z")
_MAXIMUM_EYE_ROUGHNESS = 0.2
_EYE_FIELDS = frozenset(
    (
        "id",
        "host_bone_id",
        "t",
        "offset",
        "radius",
        "mirror",
        "appearance_material",
        "socket",
        "brow_ridge",
    )
)
_SOCKET_FIELDS = frozenset(("radius", "depth"))
_BROW_FIELDS = frozenset(("offset", "size", "round", "blend"))


@dataclass(frozen=True)
class _EyeSections:
    globe: EyeGlobe
    carve: dict
    brow: dict | None
    provenance: tuple[str, str] | None


def _positive_number(value: object) -> bool:
    return _finite_number(value) and float(value) > 0.0


def _non_negative_number(value: object) -> bool:
    return _finite_number(value) and float(value) >= 0.0


def _material_obstructions(
    value: object,
    address: str,
) -> tuple[
    UnknownEyeMaterialObstruction | NonGlossyEyeMaterialObstruction,
    ...,
]:
    decoded = decode_appearance_material(value)
    return (
        (UnknownEyeMaterialObstruction(address, value),)
        if not isinstance(decoded, AcceptedMaterial)
        else (
            (
                NonGlossyEyeMaterialObstruction(
                    address,
                    decoded.material.material_id.value,
                    decoded.material.roughness,
                    _MAXIMUM_EYE_ROUGHNESS,
                ),
            )
            if decoded.material.roughness > _MAXIMUM_EYE_ROUGHNESS
            else ()
        )
    )


def _brow_obstructions(
    value: object,
    address: str,
) -> tuple[MalformedEyeObstruction, ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        return (
            MalformedEyeObstruction(
                address=address,
                rule=EyeRule.OBJECT,
                authored=value,
                required="object",
            ),
        )
    size = _vector3(value.get("size"))
    blend = value.get("blend")
    return (
        *_unknown_field_obstructions(
            MalformedEyeObstruction,
            EyeRule.KNOWN_FIELD,
            value,
            _BROW_FIELDS,
            address,
        ),
        *_malformed(
            MalformedEyeObstruction,
            _vector3(value.get("offset")) is not None,
            f"{address}/offset",
            EyeRule.FINITE_VECTOR3,
            value.get("offset"),
            {"arity": 3, "finite": True},
        ),
        *_malformed(
            MalformedEyeObstruction,
            size is not None and min(size) > 0.0,
            f"{address}/size",
            EyeRule.FINITE_POSITIVE_VECTOR3,
            value.get("size"),
            {"arity": 3, "finite": True, "exclusive_minimum": 0.0},
        ),
        *_malformed(
            MalformedEyeObstruction,
            _non_negative_number(value.get("round", 0.0)),
            f"{address}/round",
            EyeRule.FINITE_NON_NEGATIVE_NUMBER,
            value.get("round", 0.0),
            {"finite": True, "minimum": 0.0},
        ),
        *_malformed(
            MalformedEyeObstruction,
            blend is None or _non_negative_number(blend),
            f"{address}/blend",
            EyeRule.FINITE_NON_NEGATIVE_NUMBER,
            blend,
            {"finite": True, "minimum": 0.0},
        ),
    )


def _socket_obstructions(
    value: object,
    eye_radius: object,
    address: str,
) -> tuple[MalformedEyeObstruction, ...]:
    if not isinstance(value, Mapping):
        return (
            MalformedEyeObstruction(
                address=address,
                rule=EyeRule.OBJECT,
                authored=value,
                required="object",
            ),
        )
    socket_radius = value.get("radius")
    depth = value.get("depth", 0.0)
    return (
        *_unknown_field_obstructions(
            MalformedEyeObstruction,
            EyeRule.KNOWN_FIELD,
            value,
            _SOCKET_FIELDS,
            address,
        ),
        *_malformed(
            MalformedEyeObstruction,
            _positive_number(socket_radius),
            f"{address}/radius",
            EyeRule.FINITE_POSITIVE_NUMBER,
            socket_radius,
            {"finite": True, "exclusive_minimum": 0.0},
        ),
        *_malformed(
            MalformedEyeObstruction,
            not (
                _positive_number(socket_radius)
                and _positive_number(eye_radius)
            )
            or float(socket_radius) > float(eye_radius),
            f"{address}/radius",
            EyeRule.SOCKET_RADIUS_EXCEEDS_EYE_RADIUS,
            socket_radius,
            {"exclusive_minimum": eye_radius},
        ),
        *_malformed(
            MalformedEyeObstruction,
            _non_negative_number(depth),
            f"{address}/depth",
            EyeRule.FINITE_NON_NEGATIVE_NUMBER,
            depth,
            {"finite": True, "minimum": 0.0},
        ),
        *_malformed(
            MalformedEyeObstruction,
            not (
                _positive_number(socket_radius)
                and _non_negative_number(depth)
            )
            or float(depth) < float(socket_radius),
            f"{address}/depth",
            EyeRule.SOCKET_DEPTH_BELOW_RADIUS,
            depth,
            {"exclusive_maximum": socket_radius},
        ),
    )


def _entry_obstructions(
    value: object,
    index: int,
    hosts: Mapping[str, EyeHostFrame],
    occupied_part_ids: frozenset[str],
) -> tuple[EyeObstruction, ...]:
    address = f"/eyes/{index}"
    if not isinstance(value, Mapping):
        return (
            MalformedEyeObstruction(
                address=address,
                rule=EyeRule.OBJECT,
                authored=value,
                required="object",
            ),
        )
    eye_id = value.get("id")
    host_bone_id = value.get("host_bone_id")
    offset = _vector3(value.get("offset"))
    mirror = value.get("mirror", False)
    appearance_material = value.get(
        "appearance_material",
        AppearanceMaterialId.EYE_GLOSS.value,
    )
    host = hosts.get(host_bone_id) if isinstance(host_bone_id, str) else None
    parameter = value.get("t", 0.0)
    authored_center = (
        placed_center(
            {"t": parameter, "offset": offset},
            np.asarray(host.origin, dtype=np.float64),
            np.asarray(host.rotation, dtype=np.float64),
            host.length,
        )
        if host is not None and _finite_number(parameter) and offset is not None
        else None
    )
    base_derived_ids = (
        (
            f"eye.{eye_id}.socket",
            *(
                (f"eye.{eye_id}.brow",)
                if value.get("brow_ridge") is not None
                else ()
            ),
        )
        if isinstance(eye_id, str)
        else ()
    )
    derived_ids = tuple(
        derived_id
        for base_id in base_derived_ids
        for derived_id in (
            (base_id, f"{base_id}_m")
            if mirror is True
            else (base_id,)
        )
    )
    return (
        *_unknown_field_obstructions(
            MalformedEyeObstruction,
            EyeRule.KNOWN_FIELD,
            value,
            _EYE_FIELDS,
            address,
        ),
        *_malformed(
            MalformedEyeObstruction,
            isinstance(eye_id, str) and _EYE_ID.fullmatch(eye_id) is not None,
            f"{address}/id",
            EyeRule.IDENTIFIER,
            eye_id,
            _EYE_ID_SYNTAX,
        ),
        *(
            ()
            if isinstance(host_bone_id, str) and host_bone_id in hosts
            else (
                UnknownEyeHostObstruction(
                    f"{address}/host_bone_id",
                    str(host_bone_id),
                    tuple(
                        difflib.get_close_matches(
                            str(host_bone_id),
                            tuple(hosts),
                            n=3,
                        )
                    ),
                ),
            )
        ),
        *_malformed(
            MalformedEyeObstruction,
            _finite_number(value.get("t", 0.0)),
            f"{address}/t",
            EyeRule.FINITE_NUMBER,
            value.get("t", 0.0),
            {"finite": True},
        ),
        *_malformed(
            MalformedEyeObstruction,
            offset is not None,
            f"{address}/offset",
            EyeRule.FINITE_VECTOR3,
            value.get("offset"),
            {"arity": 3, "finite": True},
        ),
        *_malformed(
            MalformedEyeObstruction,
            isinstance(mirror, bool),
            f"{address}/mirror",
            EyeRule.BOOLEAN,
            mirror,
            "boolean",
        ),
        *_malformed(
            MalformedEyeObstruction,
            mirror is not True
            or authored_center is None
            or authored_center[0] > 0.0,
            f"{address}/offset",
            EyeRule.CANONICAL_MIRROR_SIDE,
            None if authored_center is None else float(authored_center[0]),
            {"coordinate": "world_x", "exclusive_minimum": 0.0},
        ),
        *_malformed(
            MalformedEyeObstruction,
            _positive_number(value.get("radius")),
            f"{address}/radius",
            EyeRule.FINITE_POSITIVE_NUMBER,
            value.get("radius"),
            {"finite": True, "exclusive_minimum": 0.0},
        ),
        *_material_obstructions(
            appearance_material,
            f"{address}/appearance_material",
        ),
        *_socket_obstructions(
            value.get("socket"),
            value.get("radius"),
            f"{address}/socket",
        ),
        *_brow_obstructions(
            value.get("brow_ridge"),
            f"{address}/brow_ridge",
        ),
        *tuple(
            EyePartCollisionObstruction(address, part_id)
            for part_id in derived_ids
            if part_id in occupied_part_ids
        ),
    )


def _duplicate_obstructions(
    entries: tuple[object, ...],
) -> tuple[DuplicateEyeIdObstruction, ...]:
    instances = tuple(
        (index, instance_id)
        for index, entry in enumerate(entries)
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str)
        for instance_id in (
            (str(entry["id"]), f"{entry['id']}_m")
            if entry.get("mirror") is True
            else (str(entry["id"]),)
        )
    )
    return tuple(
        DuplicateEyeIdObstruction(f"/eyes/{index}/id", instance_id)
        for position, (index, instance_id) in enumerate(instances)
        if instance_id in tuple(
            prior_id for _prior_index, prior_id in instances[:position]
        )
    )


def _decode_brow(value: object) -> BrowRidge | None:
    if not isinstance(value, Mapping):
        return None
    blend = value.get("blend")
    return BrowRidge(
        offset=tuple(map(float, value["offset"])),
        size=tuple(map(float, value["size"])),
        round_radius=float(value.get("round", 0.0)),
        blend=float(blend) if blend is not None else None,
    )


def _decode_eye(value: Mapping[str, object]) -> EyeSpec:
    socket = cast(Mapping[str, object], value["socket"])
    appearance = cast(
        AcceptedMaterial,
        decode_appearance_material(
            value.get(
                "appearance_material",
                AppearanceMaterialId.EYE_GLOSS.value,
            )
        ),
    )
    return EyeSpec(
        eye_id=str(value["id"]),
        host_bone_id=str(value["host_bone_id"]),
        parameter=float(value.get("t", 0.0)),
        offset=tuple(map(float, value["offset"])),
        radius=float(value["radius"]),
        mirror=bool(value.get("mirror", False)),
        appearance_material=appearance.material.material_id,
        socket=EyeSocket(
            radius=float(socket["radius"]),
            depth=float(socket.get("depth", 0.0)),
        ),
        brow_ridge=_decode_brow(value.get("brow_ridge")),
    )


def _place_eye(
    eye: EyeSpec,
    index: int,
    host: EyeHostFrame,
) -> _EyeSections:
    origin = np.asarray(host.origin, dtype=np.float64)
    rotation = np.asarray(host.rotation, dtype=np.float64)
    center = placed_center(
        {"t": eye.parameter, "offset": eye.offset},
        origin,
        rotation,
        host.length,
    )
    socket_center = center - rotation[:, 2] * eye.socket.depth
    mirror = {"mirror": True} if eye.mirror else {}
    carve = {
        "id": f"eye.{eye.eye_id}.socket",
        "type": "blob",
        "center": _rvec(socket_center),
        "size": [_r(eye.socket.radius)] * 3,
        **mirror,
    }
    brow = eye.brow_ridge
    brow_part = (
        None
        if brow is None
        else {
            "id": f"eye.{eye.eye_id}.brow",
            "type": "box",
            "center": _rvec(center + rotation @ np.asarray(brow.offset)),
            "size": [_r(value) for value in brow.size],
            "round": _r(brow.round_radius),
            **_rotation_payload(rotation),
            **mirror,
            **({"blend": _r(brow.blend)} if brow.blend is not None else {}),
        }
    )
    return _EyeSections(
        globe=EyeGlobe(
            eye_id=eye.eye_id,
            center=tuple(map(float, _rvec(center))),
            radius=_r(eye.radius),
            mirror=eye.mirror,
            appearance_material=eye.appearance_material,
        ),
        carve=carve,
        brow=brow_part,
        provenance=(
            (f"eye.{eye.eye_id}.brow", f"eyes[{index}]/brow_ridge")
            if brow_part is not None
            else None
        ),
    )


def compile_eyes(
    payload: object,
    hosts: Mapping[str, EyeHostFrame],
    occupied_part_ids: frozenset[str],
) -> EyeCompileResult:
    if payload is None:
        return AcceptedEyes((), (), (), ())
    if not isinstance(payload, (list, tuple)):
        return RejectedEyes(
            (
                MalformedEyeObstruction(
                    address="/eyes",
                    rule=EyeRule.EYES_ARRAY,
                    authored=payload,
                    required="array",
                ),
            )
        )
    entries = tuple(payload)
    obstructions = (
        *tuple(
            obstruction
            for index, entry in enumerate(entries)
            for obstruction in _entry_obstructions(
                entry,
                index,
                hosts,
                occupied_part_ids,
            )
        ),
        *_duplicate_obstructions(entries),
    )
    if obstructions:
        return RejectedEyes(obstructions)
    eyes = tuple(
        _decode_eye(entry)
        for entry in entries
        if isinstance(entry, Mapping)
    )
    sections = tuple(
        _place_eye(eye, index, hosts[eye.host_bone_id])
        for index, eye in enumerate(eyes)
    )
    return AcceptedEyes(
        globes=tuple(section.globe for section in sections),
        carves=tuple(section.carve for section in sections),
        brow_parts=tuple(
            section.brow for section in sections if section.brow is not None
        ),
        provenance=tuple(
            section.provenance
            for section in sections
            if section.provenance is not None
        ),
    )


def eye_globe_graph(globe: EyeGlobe) -> dict:
    return {
        "name": f"eye.{globe.eye_id}",
        "blend": 0.0,
        "parts": [
            {
                "id": globe.eye_id,
                "type": "blob",
                "center": list(globe.center),
                "size": [globe.radius] * 3,
                **({"mirror": True} if globe.mirror else {}),
            }
        ],
    }
