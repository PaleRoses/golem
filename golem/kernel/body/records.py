"""Canonical bone-record projection (wave-3, K5).

A compiled bone record is placement (head/R/frames — each constructor's own
solve) layered over an authored payload projected exactly once, here. Both
record constructors — kinematics (``forward_kinematics`` root +
``_place_bone``) and geometry (``fk_table`` root + ``_bone_record``) — draw
from :func:`project_bone_record`, so a new dialect field is threaded in this
one function and rides to every consumer. An authored key outside the known
vocabulary fails loudly (``UnthreadedBoneFieldError``) instead of vanishing
from a per-constructor key whitelist.
"""

from __future__ import annotations

from typing import Mapping


class UnthreadedBoneFieldError(ValueError):
    """An authored bone key that neither the projection nor the placement
    machinery knows — the compile-time guard against silent field drops."""


# Placement-consumed: read by frame/head solving, never carried verbatim.
_PLACEMENT_KEYS = frozenset(
    (
        "id",
        "parent",
        "world",
        "length",
        "rest_dir",
        "twist_deg",
        "joint",
        "mirror",
        "attach",
    )
)

# Projected: carried onto every compiled record by this one function.
_PROJECTED_KEYS = frozenset(
    ("flesh", "carves", "ports", "arrays", "bone_radii", "role")
)

KNOWN_BONE_KEYS: frozenset[str] = _PLACEMENT_KEYS | _PROJECTED_KEYS


def project_bone_record(
    bone: Mapping, *, parent_mirrored: bool, is_root: bool
) -> dict:
    """Project the authored payload of one bone declaration.

    Placement fields (``head``, ``R``, ``R_rest``, ``rest_dir``,
    ``twist_deg``, ``local_translation``, ``rest_direction``) belong to the
    calling constructor and are layered over this projection; everything an
    author declares that must *ride* the record is projected here, once.
    """
    unknown = sorted(set(bone) - KNOWN_BONE_KEYS)
    if unknown:
        raise UnthreadedBoneFieldError(
            f"skeleton/{bone.get('id', '?')}: authored bone keys {unknown} have "
            "no projection — thread the field once in records.py (shared by "
            "kinematics and geometry) or register it as placement-consumed"
        )
    joint = bone.get("joint", {})
    return {
        "id": bone["id"],
        "parent": bone.get("parent"),
        "length": 0.0 if is_root else float(bone["length"]),
        "mirrored": (
            False
            if is_root
            else bool(bone.get("mirror", False)) or parent_mirrored
        ),
        "dof": joint.get("dof", "fixed"),
        "limits": joint.get("limits", {}) or {},
        "flesh": bone.get("flesh", []),
        "carves": bone.get("carves", []),
        "ports": bone.get("ports", []),
        "arrays": bone.get("arrays", []),
        "bone_radii": bone.get("bone_radii"),
        "role": bone.get("role"),
        "is_root": is_root,
    }
