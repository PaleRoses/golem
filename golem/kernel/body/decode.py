"""Shared strict-decode primitives for body-part authoring vocabularies."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, TypeVar

import numpy as np

from golem.kernel.body.canonical import _rvec
from golem.kernel.body.contacts import _is_identity_quat
from golem.kernel.body.linalg import mat_to_quat
from golem.kernel.body.types import Vector3

_Obstruction = TypeVar("_Obstruction")
_Duplicate = TypeVar("_Duplicate")
_Collision = TypeVar("_Collision")


class _AuthoredEntry(Protocol):
    address: str
    declaration: Mapping[str, object]


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _number_array(value: object) -> tuple[float, ...] | None:
    return (
        tuple(map(float, value))
        if isinstance(value, (list, tuple)) and all(map(_finite_number, value))
        else None
    )


def _vector3(value: object) -> Vector3 | None:
    return (
        tuple(map(float, value))
        if isinstance(value, (list, tuple))
        and len(value) == 3
        and all(map(_finite_number, value))
        else None
    )


def _positive_vector3(value: object) -> Vector3 | None:
    vector = _vector3(value)
    return vector if vector is not None and min(vector) > 0.0 else None


def _malformed(
    obstruction: Callable[..., _Obstruction],
    valid: bool,
    address: str,
    rule: object,
    authored: object,
    required: object,
) -> tuple[_Obstruction, ...]:
    return (
        ()
        if valid
        else (
            obstruction(
                address=address,
                rule=rule,
                authored=authored,
                required=required,
            ),
        )
    )


def _unknown_field_obstructions(
    obstruction: Callable[..., _Obstruction],
    rule: object,
    declaration: Mapping[str, object],
    allowed: frozenset[str],
    address: str,
) -> tuple[_Obstruction, ...]:
    return tuple(
        obstruction(
            address=f"{address}/{field}",
            rule=rule,
            authored=field,
            required=tuple(sorted(allowed)),
        )
        for field in declaration
        if field not in allowed
    )


def _duplicate_obstructions(
    duplicate: Callable[[str, str], _Duplicate],
    collision: Callable[[str, str], _Collision],
    entries: Sequence[_AuthoredEntry],
    occupied_part_ids: frozenset[str],
) -> tuple[_Duplicate | _Collision, ...]:
    identified = tuple(
        (entry.address, str(entry.declaration["id"]))
        for entry in entries
        if isinstance(entry.declaration.get("id"), str)
    )
    return (
        *tuple(
            duplicate(f"{address}/id", entity_id)
            for index, (address, entity_id) in enumerate(identified)
            if entity_id in tuple(prior_id for _prior_address, prior_id in identified[:index])
        ),
        *tuple(
            collision(f"{address}/id", entity_id)
            for address, entity_id in identified
            if entity_id in occupied_part_ids
        ),
    )


def _rotation_payload(rotation: np.ndarray) -> dict[str, list[float]]:
    quaternion = mat_to_quat(rotation)
    return {} if _is_identity_quat(quaternion) else {"rot": _rvec(quaternion)}
