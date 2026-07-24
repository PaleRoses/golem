from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import assert_never, cast

import numpy as np

from golem.kernel.body.geometry import (
    BodyGeometryView,
    Tree,
    ancestors,
    build_tree,
    geometry_view,
    lca,
    seed_params,
)
from golem.kernel.body.linalg import _angle_between, _normalize
from golem.kernel.body.relations import (
    Attachment,
    FixedLocalAttachment,
    NamedSiteAttachment,
    decode_attachment,
)


class QuadrupedMetric(StrEnum):
    SHOULDER_ANGLE = "shoulder_angle"
    STIFLE_ANGLE = "stifle_angle"
    HOCK_ANGLE = "hock_angle"
    STANCE_WIDTH_TO_CHEST_DEPTH = "stance_width_to_chest_depth"
    TOPLINE_LEVELNESS = "topline_levelness"
    NECK_SET = "neck_set"


@dataclass(frozen=True)
class CanonicalRange:
    minimum: float
    maximum: float
    unit: str

    def contains(self, value: float) -> bool:
        return self.minimum <= value <= self.maximum

    def nearest(self, value: float) -> float:
        return min(max(value, self.minimum), self.maximum)


CANONICAL_QUADRUPED_RANGES: Mapping[QuadrupedMetric, CanonicalRange] = (
    MappingProxyType(
        {
            QuadrupedMetric.SHOULDER_ANGLE: CanonicalRange(
                110.0, 145.0, "degrees"
            ),
            QuadrupedMetric.STIFLE_ANGLE: CanonicalRange(
                125.0, 150.0, "degrees"
            ),
            QuadrupedMetric.HOCK_ANGLE: CanonicalRange(
                120.0, 160.0, "degrees"
            ),
            QuadrupedMetric.STANCE_WIDTH_TO_CHEST_DEPTH: CanonicalRange(
                0.5, 1.1, "ratio"
            ),
            QuadrupedMetric.TOPLINE_LEVELNESS: CanonicalRange(
                0.0, 10.0, "degrees"
            ),
            QuadrupedMetric.NECK_SET: CanonicalRange(
                25.0, 55.0, "degrees"
            ),
        }
    )
)


@dataclass(frozen=True)
class WithinCanonicalRange:
    metric: QuadrupedMetric
    observed: float
    canonical_range: CanonicalRange
    bone_ids: tuple[str, ...]


@dataclass(frozen=True)
class OutsideCanonicalRange:
    metric: QuadrupedMetric
    observed: float
    canonical_range: CanonicalRange
    bone_ids: tuple[str, ...]
    deviation: float


@dataclass(frozen=True)
class UnmeasurableQuadrupedDiagnostic:
    metric: QuadrupedMetric
    canonical_range: CanonicalRange
    bone_ids: tuple[str, ...]
    reason: str


type QuadrupedDiagnostic = (
    WithinCanonicalRange
    | OutsideCanonicalRange
    | UnmeasurableQuadrupedDiagnostic
)


@dataclass(frozen=True)
class RestDirectionAdjustment:
    bone_id: str
    current_direction: tuple[float, float, float]
    proposed_direction: tuple[float, float, float]
    delta: tuple[float, float, float]
    diagnostics: tuple[QuadrupedMetric, ...]


@dataclass(frozen=True)
class IncompatibleRestDirectionObstruction:
    bone_id: str
    diagnostics: tuple[QuadrupedMetric, ...]


type RepairProposalObstruction = IncompatibleRestDirectionObstruction


@dataclass(frozen=True)
class QuadrupedRepairProposal:
    adjustments: tuple[RestDirectionAdjustment, ...]
    obstructions: tuple[RepairProposalObstruction, ...] = ()


@dataclass(frozen=True)
class QuadrupedPriorReport:
    subject_name: str
    diagnostics: tuple[QuadrupedDiagnostic, ...]
    proposal: QuadrupedRepairProposal


@dataclass(frozen=True)
class MalformedQuadrupedSkeletonObstruction:
    address: str
    reason: str


@dataclass(frozen=True)
class UnresolvedQuadrupedRoleObstruction:
    role: str
    reason: str


@dataclass(frozen=True)
class NonQuadrupedTopologyObstruction:
    limb_bone_ids: tuple[str, ...]
    physical_limb_count: int
    reason: str


type QuadrupedPriorObstruction = (
    MalformedQuadrupedSkeletonObstruction
    | UnresolvedQuadrupedRoleObstruction
    | NonQuadrupedTopologyObstruction
)


@dataclass(frozen=True)
class RejectedQuadrupedPrior:
    obstructions: tuple[QuadrupedPriorObstruction, ...]


type QuadrupedPriorResult = QuadrupedPriorReport | RejectedQuadrupedPrior


@dataclass(frozen=True)
class _LimbChain:
    host_bone_id: str
    bones: tuple[str, ...]
    mirrored: bool
    cranial_score: float


@dataclass(frozen=True)
class _QuadrupedRoles:
    head_bone_id: str
    torso_bone_id: str
    forelimbs: tuple[_LimbChain, ...]
    hindlimbs: tuple[_LimbChain, ...]
    neck_bone_id: str


@dataclass(frozen=True)
class _QuadrupedContext:
    subject_name: str
    skeleton: Mapping[str, object]
    tree: Tree
    view: BodyGeometryView
    roles: _QuadrupedRoles


@dataclass(frozen=True)
class _LocalRestDirection:
    bone_id: str
    current_direction: tuple[float, float, float]
    proposed_direction: tuple[float, float, float]
    diagnostic: QuadrupedMetric


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_vector(value: object) -> tuple[float, float, float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
        or not all(map(_is_finite_number, value))
    ):
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _bone_shape_obstructions(
    bone: object,
    index: int,
    known_ids: frozenset[str],
) -> tuple[MalformedQuadrupedSkeletonObstruction, ...]:
    address = f"skeleton/bones/{index}"
    if not isinstance(bone, Mapping):
        return (
            MalformedQuadrupedSkeletonObstruction(address, "expected an object"),
        )
    bone_id = bone.get("id")
    parent = bone.get("parent")
    rest_direction = _finite_vector(bone.get("rest_dir"))
    length = bone.get("length")
    return (
        (() if isinstance(bone_id, str) and bone_id else (
            MalformedQuadrupedSkeletonObstruction(
                f"{address}/id", "expected a non-empty string"
            ),
        ))
        + (() if isinstance(parent, str) and parent in known_ids else (
            MalformedQuadrupedSkeletonObstruction(
                f"{address}/parent", f"unknown parent {parent!r}"
            ),
        ))
        + (() if rest_direction is not None and math.dist(rest_direction, (0.0, 0.0, 0.0)) > 1.0e-12 else (
            MalformedQuadrupedSkeletonObstruction(
                f"{address}/rest_dir", "expected a finite nonzero 3-vector"
            ),
        ))
        + (() if _is_finite_number(length) and float(length) > 0.0 else (
            MalformedQuadrupedSkeletonObstruction(
                f"{address}/length", "expected a finite positive number"
            ),
        ))
    )


def _reaches_root(
    bone_id: str,
    root_id: str,
    parent_by_id: Mapping[str, str],
    visited: frozenset[str] = frozenset(),
) -> bool:
    return (
        True
        if bone_id == root_id
        else False
        if bone_id in visited or bone_id not in parent_by_id
        else _reaches_root(
            parent_by_id[bone_id],
            root_id,
            parent_by_id,
            visited | {bone_id},
        )
    )


def _decode_attachments(
    bones: tuple[Mapping[str, object], ...],
) -> tuple[
    Mapping[str, Attachment],
    tuple[MalformedQuadrupedSkeletonObstruction, ...],
]:
    decoded = tuple(
        (
            str(bone["id"]),
            decode_attachment(
                bone.get("attach"),
                f"skeleton/{bone['id']}/attach",
            ),
        )
        for bone in bones
    )
    obstructions = tuple(
        MalformedQuadrupedSkeletonObstruction(
            f"skeleton/{bone_id}/attach",
            f"{type(result).__name__}: {result}",
        )
        for bone_id, result in decoded
        if not isinstance(result, (FixedLocalAttachment, NamedSiteAttachment))
    )
    return (
        {
            bone_id: result
            for bone_id, result in decoded
            if isinstance(result, (FixedLocalAttachment, NamedSiteAttachment))
        },
        obstructions,
    )


def _raw_regions(payload: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    anatomy = payload.get("anatomy")
    overall = anatomy.get("overall") if isinstance(anatomy, Mapping) else None
    regions = overall.get("regions") if isinstance(overall, Mapping) else None
    return (
        tuple(region for region in regions if isinstance(region, Mapping))
        if isinstance(regions, Sequence) and not isinstance(regions, (str, bytes))
        else ()
    )


def _region_hosts(
    regions: tuple[Mapping[str, object], ...],
    kind: str,
    tree: Tree,
) -> tuple[str, ...]:
    return tuple(
        host
        for region in regions
        for host in (region.get("host_bone_id"),)
        if region.get("kind") == kind
        and isinstance(host, str)
        and host in tree.records
    )


def _bone_tail(view: BodyGeometryView, bone_id: str) -> np.ndarray:
    record = view.table[bone_id]
    return np.asarray(record["head"], dtype=np.float64) + (
        np.asarray(record["R"], dtype=np.float64)[:, 2]
        * float(record["length"])
    )


def _terminal_bones(tree: Tree) -> tuple[str, ...]:
    return tuple(
        bone_id
        for bone_id in tree.order
        if bone_id != tree.root and not tree.children.get(bone_id, ())
    )


def _infer_limb_hosts(tree: Tree, view: BodyGeometryView) -> tuple[str, ...]:
    root_y = float(view.table[tree.root]["head"][1])
    return tuple(
        bone_id
        for bone_id in _terminal_bones(tree)
        if float(_bone_tail(view, bone_id)[1]) < root_y - 0.1
    )


def _infer_head_host(
    tree: Tree,
    view: BodyGeometryView,
    limb_hosts: tuple[str, ...],
) -> str | None:
    root_point = np.asarray(view.table[tree.root]["head"], dtype=np.float64)
    candidates = tuple(
        bone_id
        for bone_id in _terminal_bones(tree)
        if bone_id not in limb_hosts
    )
    return max(
        candidates,
        key=lambda bone_id: float(
            np.linalg.norm((_bone_tail(view, bone_id) - root_point)[[0, 2]])
        ),
        default=None,
    )


def _flesh_depth(flesh: object) -> float | None:
    if not isinstance(flesh, Mapping):
        return None
    kind = flesh.get("kind")
    size = _finite_vector(flesh.get("size"))
    radii = flesh.get("radii")
    if kind in ("blob", "box") and size is not None:
        return 2.0 * size[1]
    if (
        kind == "gencyl"
        and isinstance(radii, Sequence)
        and not isinstance(radii, (str, bytes))
        and radii
        and all(map(_is_finite_number, radii))
    ):
        return 2.0 * max(map(float, radii))
    return None


def _bone_chest_depth(tree: Tree, bone_id: str) -> float | None:
    flesh = tree.records[bone_id].get("flesh", ())
    depths = tuple(
        depth
        for item in flesh
        for depth in (_flesh_depth(item),)
        if depth is not None
    )
    return max(depths, default=None)


def _infer_torso_host(tree: Tree, head_bone_id: str) -> str | None:
    candidates = tuple(reversed(ancestors(tree, head_bone_id)))
    measured = tuple(
        (bone_id, depth)
        for bone_id in candidates
        for depth in (_bone_chest_depth(tree, bone_id),)
        if depth is not None
    )
    return max(measured, key=lambda item: item[1], default=(None, 0.0))[0]


def _path_after(tree: Tree, ancestor: str, descendant: str) -> tuple[str, ...]:
    path = tuple(reversed(ancestors(tree, descendant)))
    return (
        path[path.index(ancestor) + 1 :]
        if ancestor in path
        else ()
    )


def _cranial_horizontal(
    view: BodyGeometryView,
    torso_bone_id: str,
    head_bone_id: str,
) -> np.ndarray | None:
    torso = np.asarray(view.table[torso_bone_id]["head"], dtype=np.float64)
    head = _bone_tail(view, head_bone_id)
    horizontal = np.array([head[0] - torso[0], 0.0, head[2] - torso[2]])
    return (
        _normalize(horizontal)
        if float(np.linalg.norm(horizontal)) > 1.0e-12
        else None
    )


def _limb_chain(
    host_bone_id: str,
    head_bone_id: str,
    cranial: np.ndarray,
    tree: Tree,
    view: BodyGeometryView,
) -> _LimbChain:
    axial_ancestor = lca(tree, host_bone_id, head_bone_id)
    bones = _path_after(tree, axial_ancestor, host_bone_id)
    branch_head = np.asarray(view.table[bones[0]]["head"], dtype=np.float64)
    return _LimbChain(
        host_bone_id=host_bone_id,
        bones=bones,
        mirrored=bool(view.table[host_bone_id]["mirrored"]),
        cranial_score=float(branch_head @ cranial),
    )


def _physical_limb_count(chains: tuple[_LimbChain, ...]) -> int:
    return sum(2 if chain.mirrored else 1 for chain in chains)


def _resolve_roles(
    payload: Mapping[str, object],
    tree: Tree,
    view: BodyGeometryView,
) -> _QuadrupedRoles | tuple[QuadrupedPriorObstruction, ...]:
    regions = _raw_regions(payload)
    authored_limbs = _region_hosts(regions, "limb", tree)
    limb_hosts = authored_limbs or _infer_limb_hosts(tree, view)
    authored_heads = _region_hosts(regions, "head", tree)
    head_bone_id = (
        authored_heads[0]
        if len(authored_heads) == 1
        else _infer_head_host(tree, view, limb_hosts)
    )
    if head_bone_id is None:
        return (
            UnresolvedQuadrupedRoleObstruction(
                "head", "no unique cranial terminal could be derived"
            ),
        )
    authored_torsos = _region_hosts(regions, "torso", tree)
    torso_bone_id = (
        authored_torsos[0]
        if len(authored_torsos) == 1
        else _infer_torso_host(tree, head_bone_id)
    )
    if torso_bone_id is None:
        return (
            UnresolvedQuadrupedRoleObstruction(
                "torso", "no chest-bearing axial bone could be derived"
            ),
        )
    cranial = _cranial_horizontal(view, torso_bone_id, head_bone_id)
    if cranial is None:
        return (
            UnresolvedQuadrupedRoleObstruction(
                "cranial_axis", "head and torso have no horizontal separation"
            ),
        )
    chains = tuple(
        _limb_chain(host, head_bone_id, cranial, tree, view)
        for host in limb_hosts
    )
    short_chains = tuple(
        chain.host_bone_id for chain in chains if len(chain.bones) < 3
    )
    if short_chains:
        return (
            NonQuadrupedTopologyObstruction(
                short_chains,
                _physical_limb_count(chains),
                "each limb requires three rest-direction bones",
            ),
        )
    ordered = tuple(
        sorted(chains, key=lambda chain: chain.cranial_score, reverse=True)
    )
    midpoint = len(ordered) // 2
    forelimbs, hindlimbs = ordered[:midpoint], ordered[midpoint:]
    physical_count = _physical_limb_count(chains)
    if (
        len(ordered) not in (2, 4)
        or physical_count != 4
        or _physical_limb_count(forelimbs) != 2
        or _physical_limb_count(hindlimbs) != 2
    ):
        return (
            NonQuadrupedTopologyObstruction(
                tuple(chain.host_bone_id for chain in chains),
                physical_count,
                "expected two forelimbs and two hindlimbs",
            ),
        )
    axial_ancestor = lca(tree, torso_bone_id, head_bone_id)
    neck_path = (
        _path_after(tree, torso_bone_id, head_bone_id)
        if axial_ancestor == torso_bone_id
        else _path_after(tree, axial_ancestor, head_bone_id)
    )
    neck_candidates = tuple(
        bone_id for bone_id in neck_path if bone_id != head_bone_id
    )
    if not neck_candidates:
        return (
            UnresolvedQuadrupedRoleObstruction(
                "neck", "head has no distinct cervical rest-direction bone"
            ),
        )
    return _QuadrupedRoles(
        head_bone_id=head_bone_id,
        torso_bone_id=torso_bone_id,
        forelimbs=forelimbs,
        hindlimbs=hindlimbs,
        neck_bone_id=neck_candidates[0],
    )


def _decode_context(
    payload: object,
) -> _QuadrupedContext | tuple[QuadrupedPriorObstruction, ...]:
    if not isinstance(payload, Mapping):
        return (
            MalformedQuadrupedSkeletonObstruction("/", "expected an object"),
        )
    skeleton = payload.get("skeleton")
    if not isinstance(skeleton, Mapping):
        return (
            MalformedQuadrupedSkeletonObstruction(
                "/skeleton", "expected an object"
            ),
        )
    root = skeleton.get("root")
    raw_bones = skeleton.get("bones")
    if not isinstance(root, Mapping):
        return (
            MalformedQuadrupedSkeletonObstruction(
                "/skeleton/root", "expected an object"
            ),
        )
    if (
        not isinstance(raw_bones, Sequence)
        or isinstance(raw_bones, (str, bytes))
    ):
        return (
            MalformedQuadrupedSkeletonObstruction(
                "/skeleton/bones", "expected an array"
            ),
        )
    root_id = root.get("id")
    root_world = _finite_vector(root.get("world"))
    root_obstructions = (
        (() if isinstance(root_id, str) and root_id else (
            MalformedQuadrupedSkeletonObstruction(
                "/skeleton/root/id", "expected a non-empty string"
            ),
        ))
        + (() if root_world is not None else (
            MalformedQuadrupedSkeletonObstruction(
                "/skeleton/root/world", "expected a finite 3-vector"
            ),
        ))
    )
    if root_obstructions:
        return root_obstructions
    root_id = cast(str, root_id)
    bones = tuple(bone for bone in raw_bones if isinstance(bone, Mapping))
    container_obstructions = tuple(
        MalformedQuadrupedSkeletonObstruction(
            f"/skeleton/bones/{index}", "expected an object"
        )
        for index, bone in enumerate(raw_bones)
        if not isinstance(bone, Mapping)
    )
    bone_ids = tuple(
        bone_id
        for bone in bones
        for bone_id in (bone.get("id"),)
        if isinstance(bone_id, str) and bone_id
    )
    duplicate_ids = tuple(
        bone_id
        for bone_id, count in Counter((root_id, *bone_ids)).items()
        if count > 1
    )
    duplicate_obstructions = tuple(
        MalformedQuadrupedSkeletonObstruction(
            f"/skeleton/{bone_id}", "duplicate bone id"
        )
        for bone_id in duplicate_ids
    )
    known_ids = frozenset((root_id, *bone_ids))
    shape_obstructions = tuple(
        obstruction
        for index, bone in enumerate(bones)
        for obstruction in _bone_shape_obstructions(bone, index, known_ids)
    )
    if container_obstructions or duplicate_obstructions or shape_obstructions:
        return (
            *container_obstructions,
            *duplicate_obstructions,
            *shape_obstructions,
        )
    typed_bones = tuple(bones)
    parent_by_id = {
        str(bone["id"]): str(bone["parent"])
        for bone in typed_bones
    }
    disconnected = tuple(
        bone_id
        for bone_id in bone_ids
        if not _reaches_root(bone_id, root_id, parent_by_id)
    )
    if disconnected:
        return tuple(
            MalformedQuadrupedSkeletonObstruction(
                f"/skeleton/{bone_id}", "bone ancestry does not reach the root"
            )
            for bone_id in disconnected
        )
    attachments, attachment_obstructions = _decode_attachments(typed_bones)
    if attachment_obstructions:
        return attachment_obstructions
    body_view = {"skeleton": skeleton}
    tree = build_tree(body_view)
    parameters = seed_params(body_view, {}, attachments)
    view = geometry_view(body_view, {}, attachments, parameters)
    roles = _resolve_roles(payload, tree, view)
    if isinstance(roles, tuple):
        return roles
    name = payload.get("name")
    return _QuadrupedContext(
        subject_name=name if isinstance(name, str) and name else "quadruped",
        skeleton=skeleton,
        tree=tree,
        view=view,
        roles=roles,
    )


def _unique_bones(bone_ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(bone_ids))


def _diagnostic(
    metric: QuadrupedMetric,
    observed: float,
    bone_ids: tuple[str, ...],
) -> QuadrupedDiagnostic:
    canonical_range = CANONICAL_QUADRUPED_RANGES[metric]
    rounded = round(float(observed), 6)
    return (
        WithinCanonicalRange(metric, rounded, canonical_range, _unique_bones(bone_ids))
        if canonical_range.contains(observed)
        else OutsideCanonicalRange(
            metric,
            rounded,
            canonical_range,
            _unique_bones(bone_ids),
            round(abs(observed - canonical_range.nearest(observed)), 6),
        )
    )


def _unmeasurable(
    metric: QuadrupedMetric,
    bone_ids: tuple[str, ...],
    reason: str,
) -> UnmeasurableQuadrupedDiagnostic:
    return UnmeasurableQuadrupedDiagnostic(
        metric,
        CANONICAL_QUADRUPED_RANGES[metric],
        _unique_bones(bone_ids),
        reason,
    )


def _joint_angle(view: BodyGeometryView, proximal: str, distal: str) -> float:
    proximal_axis = np.asarray(view.table[proximal]["R"], dtype=np.float64)[:, 2]
    distal_axis = np.asarray(view.table[distal]["R"], dtype=np.float64)[:, 2]
    return 180.0 - _angle_between(proximal_axis, distal_axis)


def _mean(values: tuple[float, ...]) -> float:
    return math.fsum(values) / len(values)


def _angular_diagnostic(
    metric: QuadrupedMetric,
    chains: tuple[_LimbChain, ...],
    proximal_index: int,
    distal_index: int,
    view: BodyGeometryView,
) -> QuadrupedDiagnostic:
    pairs = tuple(
        (chain.bones[proximal_index], chain.bones[distal_index])
        for chain in chains
    )
    return _diagnostic(
        metric,
        _mean(tuple(_joint_angle(view, proximal, distal) for proximal, distal in pairs)),
        tuple(bone_id for pair in pairs for bone_id in pair),
    )


def _stance_x_sites(chain: _LimbChain, view: BodyGeometryView) -> tuple[float, ...]:
    x = float(_bone_tail(view, chain.host_bone_id)[0])
    return (-abs(x), abs(x)) if chain.mirrored else (x,)


def _stance_diagnostic(context: _QuadrupedContext) -> QuadrupedDiagnostic:
    roles = context.roles
    chains = (*roles.forelimbs, *roles.hindlimbs)
    sites = tuple(
        x
        for chain in chains
        for x in _stance_x_sites(chain, context.view)
    )
    chest_depth = _bone_chest_depth(context.tree, roles.torso_bone_id)
    bone_ids = tuple(chain.host_bone_id for chain in chains)
    return (
        _unmeasurable(
            QuadrupedMetric.STANCE_WIDTH_TO_CHEST_DEPTH,
            bone_ids,
            "torso has no measurable blob, box, or gencyl chest depth",
        )
        if chest_depth is None or chest_depth <= 1.0e-12
        else _diagnostic(
            QuadrupedMetric.STANCE_WIDTH_TO_CHEST_DEPTH,
            (max(sites) - min(sites)) / chest_depth,
            bone_ids,
        )
    )


def _mean_tail(
    chains: tuple[_LimbChain, ...],
    view: BodyGeometryView,
) -> np.ndarray:
    return np.mean(
        np.asarray(tuple(_bone_tail(view, chain.bones[0]) for chain in chains)),
        axis=0,
    )


def _topline_diagnostic(context: _QuadrupedContext) -> QuadrupedDiagnostic:
    roles = context.roles
    fore = _mean_tail(roles.forelimbs, context.view)
    hind = _mean_tail(roles.hindlimbs, context.view)
    horizontal = math.hypot(float(fore[0] - hind[0]), float(fore[2] - hind[2]))
    bone_ids = tuple(
        chain.bones[0] for chain in (*roles.forelimbs, *roles.hindlimbs)
    )
    return (
        _unmeasurable(
            QuadrupedMetric.TOPLINE_LEVELNESS,
            bone_ids,
            "forequarter and hindquarter have no horizontal separation",
        )
        if horizontal <= 1.0e-12
        else _diagnostic(
            QuadrupedMetric.TOPLINE_LEVELNESS,
            math.degrees(math.atan2(abs(float(fore[1] - hind[1])), horizontal)),
            bone_ids,
        )
    )


def _neck_diagnostic(context: _QuadrupedContext) -> QuadrupedDiagnostic:
    neck = context.roles.neck_bone_id
    axis = np.asarray(context.view.table[neck]["R"], dtype=np.float64)[:, 2]
    horizontal = math.hypot(float(axis[0]), float(axis[2]))
    return (
        _unmeasurable(
            QuadrupedMetric.NECK_SET,
            (neck,),
            "neck axis is vertical and has no horizontal set",
        )
        if horizontal <= 1.0e-12
        else _diagnostic(
            QuadrupedMetric.NECK_SET,
            math.degrees(math.atan2(float(axis[1]), horizontal)),
            (neck,),
        )
    )


def _diagnostics(context: _QuadrupedContext) -> tuple[QuadrupedDiagnostic, ...]:
    roles = context.roles
    return (
        _angular_diagnostic(
            QuadrupedMetric.SHOULDER_ANGLE,
            roles.forelimbs,
            0,
            1,
            context.view,
        ),
        _angular_diagnostic(
            QuadrupedMetric.STIFLE_ANGLE,
            roles.hindlimbs,
            0,
            1,
            context.view,
        ),
        _angular_diagnostic(
            QuadrupedMetric.HOCK_ANGLE,
            roles.hindlimbs,
            1,
            2,
            context.view,
        ),
        _stance_diagnostic(context),
        _topline_diagnostic(context),
        _neck_diagnostic(context),
    )


def _direction_tuple(value: object) -> tuple[float, float, float]:
    vector = cast(tuple[float, float, float], _finite_vector(value))
    normalized = _normalize(np.asarray(vector, dtype=np.float64))
    return (float(normalized[0]), float(normalized[1]), float(normalized[2]))


def _joint_target_direction(
    current: tuple[float, float, float],
    anatomical_angle: float,
) -> tuple[float, float, float] | None:
    deflection = math.radians(180.0 - anatomical_angle)
    radial = math.hypot(current[0], current[1])
    if radial <= 1.0e-12:
        return None
    scale = math.sin(deflection) / radial
    target = np.asarray(
        (current[0] * scale, current[1] * scale, math.cos(deflection)),
        dtype=np.float64,
    )
    normalized = _normalize(target)
    return (float(normalized[0]), float(normalized[1]), float(normalized[2]))


def _angle_repairs(
    diagnostic: QuadrupedDiagnostic,
    chains: tuple[_LimbChain, ...],
    proximal_index: int,
    distal_index: int,
    context: _QuadrupedContext,
) -> tuple[_LocalRestDirection, ...]:
    if not isinstance(diagnostic, OutsideCanonicalRange):
        return ()
    candidates = tuple(
        (
            chain.bones[distal_index],
            _joint_angle(
                context.view,
                chain.bones[proximal_index],
                chain.bones[distal_index],
            ),
        )
        for chain in chains
    )
    return tuple(
        _LocalRestDirection(
            bone_id,
            current,
            target,
            diagnostic.metric,
        )
        for bone_id, observed in candidates
        for current in (
            _direction_tuple(context.tree.records[bone_id]["rest_dir"]),
        )
        for target in (
            _joint_target_direction(
                current,
                diagnostic.canonical_range.nearest(observed),
            ),
        )
        if target is not None
        and not diagnostic.canonical_range.contains(observed)
    )


def _world_target_local(
    bone_id: str,
    target_world: np.ndarray,
    context: _QuadrupedContext,
) -> tuple[float, float, float]:
    parent_id = context.tree.parent[bone_id]
    parent_frame = (
        np.eye(3)
        if parent_id is None
        else np.asarray(context.view.table[parent_id]["R"], dtype=np.float64)
    )
    target = _normalize(parent_frame.T @ _normalize(target_world))
    return (float(target[0]), float(target[1]), float(target[2]))


def _stance_repairs(
    diagnostic: QuadrupedDiagnostic,
    context: _QuadrupedContext,
) -> tuple[_LocalRestDirection, ...]:
    if not isinstance(diagnostic, OutsideCanonicalRange):
        return ()
    ratio = diagnostic.observed
    factor = diagnostic.canonical_range.nearest(ratio) / ratio
    chains = (*context.roles.forelimbs, *context.roles.hindlimbs)
    return tuple(
        _LocalRestDirection(
            bone_id,
            _direction_tuple(context.tree.records[bone_id]["rest_dir"]),
            _world_target_local(
                bone_id,
                np.asarray(
                    (
                        world_axis[0] * factor,
                        world_axis[1],
                        world_axis[2],
                    ),
                    dtype=np.float64,
                ),
                context,
            ),
            diagnostic.metric,
        )
        for chain in chains
        for bone_id in (chain.bones[0],)
        for world_axis in (
            np.asarray(context.view.table[bone_id]["R"], dtype=np.float64)[:, 2],
        )
    )


def _topline_repairs(
    diagnostic: QuadrupedDiagnostic,
    context: _QuadrupedContext,
) -> tuple[_LocalRestDirection, ...]:
    if not isinstance(diagnostic, OutsideCanonicalRange):
        return ()
    fore = _mean_tail(context.roles.forelimbs, context.view)
    hind = _mean_tail(context.roles.hindlimbs, context.view)
    horizontal = math.hypot(float(fore[0] - hind[0]), float(fore[2] - hind[2]))
    signed_delta = float(fore[1] - hind[1])
    allowed_delta = math.tan(
        math.radians(diagnostic.canonical_range.maximum)
    ) * horizontal
    excess = math.copysign(max(0.0, abs(signed_delta) - allowed_delta), signed_delta)
    sections = (
        *((chain, -0.5) for chain in context.roles.forelimbs),
        *((chain, 0.5) for chain in context.roles.hindlimbs),
    )
    return tuple(
        _LocalRestDirection(
            bone_id,
            _direction_tuple(context.tree.records[bone_id]["rest_dir"]),
            _world_target_local(
                bone_id,
                np.asarray(
                    (
                        world_axis[0],
                        world_axis[1] + sign * excess / max(length, 1.0e-12),
                        world_axis[2],
                    ),
                    dtype=np.float64,
                ),
                context,
            ),
            diagnostic.metric,
        )
        for chain, sign in sections
        for bone_id in (chain.bones[0],)
        for world_axis in (
            np.asarray(context.view.table[bone_id]["R"], dtype=np.float64)[:, 2],
        )
        for length in (float(context.view.table[bone_id]["length"]),)
    )


def _neck_repairs(
    diagnostic: QuadrupedDiagnostic,
    context: _QuadrupedContext,
) -> tuple[_LocalRestDirection, ...]:
    if not isinstance(diagnostic, OutsideCanonicalRange):
        return ()
    bone_id = context.roles.neck_bone_id
    world_axis = np.asarray(context.view.table[bone_id]["R"], dtype=np.float64)[:, 2]
    horizontal = np.asarray((world_axis[0], 0.0, world_axis[2]), dtype=np.float64)
    if float(np.linalg.norm(horizontal)) <= 1.0e-12:
        return ()
    target_angle = math.radians(
        diagnostic.canonical_range.nearest(diagnostic.observed)
    )
    target_world = (
        _normalize(horizontal) * math.cos(target_angle)
        + np.array([0.0, math.sin(target_angle), 0.0])
    )
    return (
        _LocalRestDirection(
            bone_id,
            _direction_tuple(context.tree.records[bone_id]["rest_dir"]),
            _world_target_local(bone_id, target_world, context),
            diagnostic.metric,
        ),
    )


def _local_repairs(
    diagnostics: tuple[QuadrupedDiagnostic, ...],
    context: _QuadrupedContext,
) -> tuple[_LocalRestDirection, ...]:
    by_metric = {diagnostic.metric: diagnostic for diagnostic in diagnostics}
    roles = context.roles
    return (
        *_angle_repairs(
            by_metric[QuadrupedMetric.SHOULDER_ANGLE],
            roles.forelimbs,
            0,
            1,
            context,
        ),
        *_angle_repairs(
            by_metric[QuadrupedMetric.STIFLE_ANGLE],
            roles.hindlimbs,
            0,
            1,
            context,
        ),
        *_angle_repairs(
            by_metric[QuadrupedMetric.HOCK_ANGLE],
            roles.hindlimbs,
            1,
            2,
            context,
        ),
        *_stance_repairs(
            by_metric[QuadrupedMetric.STANCE_WIDTH_TO_CHEST_DEPTH], context
        ),
        *_topline_repairs(
            by_metric[QuadrupedMetric.TOPLINE_LEVELNESS], context
        ),
        *_neck_repairs(by_metric[QuadrupedMetric.NECK_SET], context),
    )


def _glue_bone_repairs(
    bone_id: str,
    local_repairs: tuple[_LocalRestDirection, ...],
) -> RestDirectionAdjustment | IncompatibleRestDirectionObstruction:
    sections = tuple(
        repair for repair in local_repairs if repair.bone_id == bone_id
    )
    summed = np.asarray(
        tuple(
            math.fsum(repair.proposed_direction[index] for repair in sections)
            for index in range(3)
        ),
        dtype=np.float64,
    )
    diagnostics = tuple(dict.fromkeys(repair.diagnostic for repair in sections))
    if float(np.linalg.norm(summed)) <= 1.0e-12:
        return IncompatibleRestDirectionObstruction(bone_id, diagnostics)
    current = sections[0].current_direction
    proposed_array = _normalize(summed)
    proposed = tuple(float(component) for component in proposed_array)
    delta = tuple(proposed[index] - current[index] for index in range(3))
    return RestDirectionAdjustment(
        bone_id,
        current,
        proposed,
        delta,
        diagnostics,
    )


def _repair_proposal(
    diagnostics: tuple[QuadrupedDiagnostic, ...],
    context: _QuadrupedContext,
) -> QuadrupedRepairProposal:
    local_repairs = _local_repairs(diagnostics, context)
    bone_ids = tuple(
        bone_id
        for bone_id in context.tree.order
        if any(repair.bone_id == bone_id for repair in local_repairs)
    )
    glued = tuple(
        _glue_bone_repairs(bone_id, local_repairs) for bone_id in bone_ids
    )
    return QuadrupedRepairProposal(
        adjustments=tuple(
            adjustment
            for adjustment in glued
            if isinstance(adjustment, RestDirectionAdjustment)
            and math.dist(
                adjustment.current_direction,
                adjustment.proposed_direction,
            )
            > 1.0e-9
        ),
        obstructions=tuple(
            obstruction
            for obstruction in glued
            if isinstance(obstruction, IncompatibleRestDirectionObstruction)
        ),
    )


def diagnose_quadruped_prior(payload: object) -> QuadrupedPriorResult:
    context = _decode_context(payload)
    if isinstance(context, tuple):
        return RejectedQuadrupedPrior(context)
    diagnostics = _diagnostics(context)
    return QuadrupedPriorReport(
        context.subject_name,
        diagnostics,
        _repair_proposal(diagnostics, context),
    )


def _diagnostic_record(diagnostic: QuadrupedDiagnostic) -> dict[str, object]:
    match diagnostic:
        case WithinCanonicalRange(metric, observed, target, bone_ids):
            return {
                "id": metric.value,
                "status": "pass",
                "measured": observed,
                "target": {
                    "minimum": target.minimum,
                    "maximum": target.maximum,
                },
                "unit": target.unit,
                "bone_ids": bone_ids,
            }
        case OutsideCanonicalRange(
            metric, observed, target, bone_ids, deviation
        ):
            return {
                "id": metric.value,
                "status": "fail",
                "measured": observed,
                "target": {
                    "minimum": target.minimum,
                    "maximum": target.maximum,
                },
                "unit": target.unit,
                "deviation": deviation,
                "bone_ids": bone_ids,
            }
        case UnmeasurableQuadrupedDiagnostic(metric, target, bone_ids, reason):
            return {
                "id": metric.value,
                "status": "unmeasurable",
                "measured": None,
                "target": {
                    "minimum": target.minimum,
                    "maximum": target.maximum,
                },
                "unit": target.unit,
                "bone_ids": bone_ids,
                "reason": reason,
            }
        case _ as unreachable:
            assert_never(unreachable)


def _proposal_record(proposal: QuadrupedRepairProposal) -> dict[str, object]:
    return {
        "status": (
            "obstructed"
            if proposal.obstructions
            else "proposed"
            if proposal.adjustments
            else "empty"
        ),
        "adjustments": tuple(
            {
                "bone_id": adjustment.bone_id,
                "address": f"skeleton/{adjustment.bone_id}/rest_dir",
                "current_direction": adjustment.current_direction,
                "proposed_direction": adjustment.proposed_direction,
                "delta": adjustment.delta,
                "diagnostics": tuple(
                    metric.value for metric in adjustment.diagnostics
                ),
            }
            for adjustment in proposal.adjustments
        ),
        "obstructions": tuple(
            {
                "kind": type(obstruction).__name__,
                "bone_id": obstruction.bone_id,
                "diagnostics": tuple(
                    metric.value for metric in obstruction.diagnostics
                ),
            }
            for obstruction in proposal.obstructions
        ),
    }


def _obstruction_record(
    obstruction: QuadrupedPriorObstruction,
) -> dict[str, object]:
    match obstruction:
        case MalformedQuadrupedSkeletonObstruction(address, reason):
            return {
                "kind": type(obstruction).__name__,
                "address": address,
                "reason": reason,
            }
        case UnresolvedQuadrupedRoleObstruction(role, reason):
            return {
                "kind": type(obstruction).__name__,
                "role": role,
                "reason": reason,
            }
        case NonQuadrupedTopologyObstruction(
            limb_bone_ids, physical_limb_count, reason
        ):
            return {
                "kind": type(obstruction).__name__,
                "limb_bone_ids": limb_bone_ids,
                "physical_limb_count": physical_limb_count,
                "reason": reason,
            }
        case _ as unreachable:
            assert_never(unreachable)


def quadruped_prior_to_dict(result: QuadrupedPriorResult) -> dict[str, object]:
    match result:
        case RejectedQuadrupedPrior(obstructions):
            return {
                "status": "rejected",
                "obstructions": tuple(map(_obstruction_record, obstructions)),
            }
        case QuadrupedPriorReport(subject_name, diagnostics, proposal):
            return {
                "status": "accepted",
                "subject": subject_name,
                "diagnostics": tuple(map(_diagnostic_record, diagnostics)),
                "proposal": _proposal_record(proposal),
            }
        case _ as unreachable:
            assert_never(unreachable)
