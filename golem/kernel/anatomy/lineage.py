"""Skeleton addressing, lineage, and numeric predicate primitives."""

from __future__ import annotations

import math
from itertools import groupby, takewhile

from golem.kernel.body.types import BONE_ROLES


def _interface_address(
    left_bone_id: str, right_bone_id: str, bones: dict[str, dict]
) -> str:
    child_bone_id = (
        right_bone_id
        if bones[right_bone_id].get("parent") == left_bone_id
        else left_bone_id
    )
    return f"skeleton/{child_bone_id}/joint"


def _assembly_path(
    bones: dict[str, dict], source_bone_id: str, target_bone_id: str
) -> tuple[str, ...] | None:
    source_lineage = _rooted_lineage(bones, source_bone_id, frozenset())
    target_lineage = _rooted_lineage(bones, target_bone_id, frozenset())
    if source_lineage is None or target_lineage is None:
        return None
    shared_count = len(
        tuple(
            takewhile(
                lambda pair: pair[0] == pair[1],
                zip(source_lineage, target_lineage),
            )
        )
    )
    return (
        (
            *tuple(reversed(source_lineage[shared_count - 1 :])),
            *target_lineage[shared_count:],
        )
        if shared_count > 0
        else None
    )


def _rooted_lineage(
    bones: dict[str, dict], bone_id: str, visited: frozenset[str]
) -> tuple[str, ...] | None:
    bone = bones.get(bone_id)
    if not isinstance(bone, dict) or bone_id in visited:
        return None
    parent = bone.get("parent")
    if parent is None:
        return (bone_id,)
    if not isinstance(parent, str):
        return None
    parent_lineage = _rooted_lineage(bones, parent, visited | {bone_id})
    return (*parent_lineage, bone_id) if parent_lineage is not None else None


def _terminal_bones(
    bones: dict[str, dict], ordered_bones: tuple[str, ...]
) -> frozenset[str]:
    # The perfused forest: role-declared handle bones (R3) are contracted out
    # -- a handle neither demands an exchange bed as a leaf nor revokes its
    # parent's terminality; a perfused child's effective parent is its nearest
    # perfused ancestor, so a handle mid-chain (a routing waypoint) leaves the
    # chain's terminality intact.
    perfused = frozenset(
        bone_id
        for bone_id in ordered_bones
        if bones[bone_id].get("role") not in BONE_ROLES
    )

    def effective_parent(bone_id: str) -> str | None:
        parent = bones[bone_id].get("parent")
        while parent is not None and parent not in perfused:
            parent_record = bones.get(parent)
            parent = (
                parent_record.get("parent")
                if isinstance(parent_record, dict)
                else None
            )
        return parent

    parent_bones = frozenset(
        parent
        for bone_id in perfused
        for parent in (effective_parent(bone_id),)
        if parent is not None
    )
    return perfused - parent_bones


def _duplicate_values(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        value
        for value, grouped_values in groupby(sorted(values))
        if len(tuple(grouped_values)) > 1
    )


def _is_finite_positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _is_unit_interval_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _is_finite_pair(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(
            isinstance(component, (int, float))
            and not isinstance(component, bool)
            and math.isfinite(float(component))
            for component in value
        )
    )


def _is_bone_cross_section(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and all(map(_is_finite_positive_number, value))
    )
