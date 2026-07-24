"""Analytic constructed-geometry queries, skeleton topology, and the relational evaluator."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from golem.addressing.scope import Bone, Landmark, Part, Scope
from golem.kernel.body.gencyl import _gencyl_stations, _loft_sections
from golem.kernel.body.linalg import _frame_from_axis, _normalize, _pose_matrix, _rodrigues
from golem.kernel.body.relation_solve import Evaluator, GeometryView, VariableKind, VariableSpec
from golem.kernel.body.relations import Attachment, FixedLocalAttachment, NamedSiteAttachment
from golem.kernel.body.types import LoftSectionObstruction, _GencylStationObstruction
from golem.kernel.body.records import project_bone_record


@dataclass(frozen=True)
class BoneParam:
    translation: tuple[float, float, float] | None
    swing: tuple[float, float]
    twist: float


def placed_center(
    declaration: Mapping,
    origin: np.ndarray,
    rotation: np.ndarray,
    length: float,
) -> np.ndarray:
    parameter = float(declaration.get("t", 1.0))
    offset = np.asarray(
        declaration.get("offset", (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    return origin + rotation @ (
        np.asarray((0.0, 0.0, parameter * length)) + offset
    )


# -- skeleton topology ---------------------------------------------------- #


@dataclass(frozen=True)
class Tree:
    root: str
    parent: Mapping[str, str | None]
    children: Mapping[str, tuple[str, ...]]
    order: tuple[str, ...]
    records: Mapping[str, Mapping]


def build_tree(spec: Mapping) -> Tree:
    skeleton = spec["skeleton"]
    root = skeleton["root"]
    rid = root["id"]
    by_parent: dict[str, list[str]] = {}
    records: dict[str, Mapping] = {rid: root}
    for bone in skeleton["bones"]:
        by_parent.setdefault(bone["parent"], []).append(bone["id"])
        records[bone["id"]] = bone
    parent = {rid: None} | {
        bone["id"]: bone["parent"] for bone in skeleton["bones"]
    }

    def walk(bid: str) -> tuple[str, ...]:
        return (bid, *tuple(
            node for child in by_parent.get(bid, ()) for node in walk(child)
        ))

    order = walk(rid)
    children = {
        bid: tuple(by_parent.get(bid, ())) for bid in order
    }
    return Tree(root=rid, parent=parent, children=children, order=order, records=records)


def ancestors(tree: Tree, bid: str) -> tuple[str, ...]:
    chain: tuple[str, ...] = ()
    current: str | None = bid
    while current is not None:
        chain = chain + (current,)
        current = tree.parent.get(current)
    return chain


def lca(tree: Tree, a: str, b: str) -> str:
    a_chain = set(ancestors(tree, a))
    for node in ancestors(tree, b):
        if node in a_chain:
            return node
    return tree.root


def subtree_depths(tree: Tree, node: str) -> Mapping[str, int]:
    def walk(bid: str, depth: int) -> tuple[tuple[str, int], ...]:
        return ((bid, depth), *tuple(
            pair
            for child in tree.children.get(bid, ())
            for pair in walk(child, depth + 1)
        ))

    return dict(walk(node, 0))


# -- authored-flesh identity --------------------------------------------- #


def _flesh_id(bone_id: str, flesh: Mapping, index: int) -> str:
    return flesh.get("name", f"{bone_id}.flesh{index}")


def flesh_owner_map(tree: Tree) -> Mapping[str, tuple[str, Mapping]]:
    return {
        _flesh_id(bid, flesh, index): (bid, flesh)
        for bid in tree.order
        for index, flesh in enumerate(tree.records[bid].get("flesh", ()))
    }


def port_owner_map(tree: Tree) -> Mapping[str, str]:
    return {
        port["name"]: bid
        for bid in tree.order
        for port in tree.records[bid].get("ports", ())
    }


def landmark_owner(name: str, tree: Tree, ports: Mapping[str, str]) -> str | None:
    head, separator, _ = name.partition("/")
    if separator and head in tree.records:
        return head
    if name in ports:
        return ports[name]
    return tree.records.get(name) and name or None


def constructed_identifiers(tree: Tree) -> tuple[str, ...]:
    flesh = tuple(flesh_owner_map(tree))
    landmarks = tuple(
        f"{bid}/{suffix}"
        for bid in tree.order
        for suffix in ("head", "tail", "center")
    )
    ports = tuple(port_owner_map(tree))
    return (*tree.order, *flesh, *landmarks, *ports)


# -- parameterized forward kinematics ------------------------------------ #


def _swing_basis(rest_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = (
        np.array([0.0, 0.0, 1.0])
        if abs(float(rest_local[2])) < 0.9
        else np.array([1.0, 0.0, 0.0])
    )
    t1 = _normalize(np.cross(reference, rest_local))
    t2 = _normalize(np.cross(rest_local, t1))
    return t1, t2


def _swung_direction(rest_dir: Sequence[float], swing: tuple[float, float]) -> np.ndarray:
    rest_local = _normalize(np.asarray(rest_dir, dtype=np.float64))
    t1, t2 = _swing_basis(rest_local)
    return _normalize(rest_local + swing[0] * t1 + swing[1] * t2)


def _fixed_local_translation(attach: FixedLocalAttachment, parent_length: float) -> np.ndarray:
    return np.asarray(attach.offset, dtype=np.float64) + np.array(
        [0.0, 0.0, attach.t * parent_length]
    )


def _bone_record(
    bone: Mapping,
    parent: Mapping,
    attach: Attachment,
    param: BoneParam,
    pose_joints: Mapping,
) -> Mapping:
    parent_R = np.asarray(parent["R"], dtype=np.float64)
    parent_head = np.asarray(parent["head"], dtype=np.float64)
    if isinstance(attach, FixedLocalAttachment):
        local = _fixed_local_translation(attach, float(parent["length"]))
    else:
        local = np.asarray(
            param.translation
            if param.translation is not None
            else (0.0, 0.0, float(parent["length"])),
            dtype=np.float64,
        )
    head = parent_head + parent_R @ local
    rest_dir = bone.get("rest_dir", (0.0, 0.0, 1.0))
    swung = _swung_direction(rest_dir, param.swing)
    z_world = parent_R @ swung
    twist = float(bone.get("twist_deg", 0.0)) + param.twist
    R_rest = _frame_from_axis(z_world, parent_R[:, 1], parent_R[:, 0], twist)
    R = R_rest
    dof = bone.get("joint", {}).get("dof", "fixed")
    mirrored = bool(bone.get("mirror", False)) or bool(parent["mirrored"])
    pose = pose_joints.get(bone["id"])
    if pose and dof != "fixed" and not mirrored:
        R = R_rest @ _pose_matrix(
            float(pose.get("yaw", 0.0)),
            float(pose.get("pitch", 0.0)),
            float(pose.get("roll", 0.0)),
        )
    return {
        **project_bone_record(
            bone, parent_mirrored=bool(parent["mirrored"]), is_root=False
        ),
        "R": R,
        "R_rest": R_rest,
        "head": head,
        "twist_deg": twist,
        "rest_dir": np.asarray(rest_dir, dtype=np.float64),
        "local_translation": local,
        "rest_direction": swung,
    }


def fk_table(
    spec: Mapping,
    pose_joints: Mapping,
    attachments: Mapping[str, Attachment],
    params: Mapping[str, BoneParam],
) -> dict[str, dict]:
    tree = build_tree(spec)
    root = spec["skeleton"]["root"]
    table: dict[str, dict] = {
        tree.root: {
            **project_bone_record(root, parent_mirrored=False, is_root=True),
            "R": np.eye(3),
            "R_rest": np.eye(3),
            "head": np.asarray(root["world"], dtype=np.float64),
            "twist_deg": 0.0,
            "rest_dir": np.array([0.0, 0.0, 1.0]),
            "local_translation": np.zeros(3),
            "rest_direction": np.array([0.0, 0.0, 1.0]),
        }
    }
    default = BoneParam(translation=None, swing=(0.0, 0.0), twist=0.0)
    for bid in tree.order[1:]:
        bone = tree.records[bid]
        table[bid] = dict(
            _bone_record(
                bone,
                table[bone["parent"]],
                attachments.get(bid, FixedLocalAttachment()),
                params.get(bid, default),
                pose_joints,
            )
        )
    return table


# -- seeding -------------------------------------------------------------- #


def _port_frame(rec: Mapping, port: Mapping) -> dict:
    t = float(port.get("t", 1.0))
    origin = np.asarray(rec["head"], dtype=np.float64) + np.asarray(
        rec["R"], dtype=np.float64
    ) @ np.array([0.0, 0.0, t * float(rec["length"])])
    z = np.asarray(rec["R"], dtype=np.float64)[:, 2]
    y = np.asarray(rec["R"], dtype=np.float64)[:, 1]
    twist = float(port.get("twist_deg", 0.0))
    if twist:
        y = _rodrigues(y, z, twist)
    y = _normalize(y)
    x = np.cross(y, z)
    return {"origin": origin, "R": np.stack([_normalize(x), y, z], axis=1)}


def landmark_points(table: Mapping[str, Mapping]) -> dict[str, np.ndarray]:
    points: dict[str, np.ndarray] = {}
    for bid, rec in table.items():
        head = np.asarray(rec["head"], dtype=np.float64)
        tail = head + np.asarray(rec["R"], dtype=np.float64) @ np.array(
            [0.0, 0.0, float(rec["length"])]
        )
        points[f"{bid}/head"] = head
        points[f"{bid}/tail"] = tail
        points[f"{bid}/center"] = (head + tail) / 2.0
        for port in rec.get("ports", ()):
            frame = _port_frame(rec, port)
            points[port["name"]] = frame["origin"]
            points[f"{bid}/{port['name']}"] = frame["origin"]
    return points


def seed_params(
    spec: Mapping,
    pose_joints: Mapping,
    attachments: Mapping[str, Attachment],
) -> dict[str, BoneParam]:
    tree = build_tree(spec)
    table = fk_table(spec, pose_joints, attachments, {})
    points = landmark_points(table)
    seeds: dict[str, BoneParam] = {}
    for bid in tree.order[1:]:
        attach = attachments.get(bid, FixedLocalAttachment())
        if isinstance(attach, NamedSiteAttachment):
            site = points.get(attach.site.name)
            parent = table[tree.parent[bid]]
            parent_R = np.asarray(parent["R"], dtype=np.float64)
            parent_head = np.asarray(parent["head"], dtype=np.float64)
            local = (
                parent_R.T @ (site - parent_head)
                if site is not None
                else np.array([0.0, 0.0, float(parent["length"])])
            )
            seeds[bid] = BoneParam(
                translation=(float(local[0]), float(local[1]), float(local[2])),
                swing=(0.0, 0.0),
                twist=0.0,
            )
        else:
            seeds[bid] = BoneParam(translation=None, swing=(0.0, 0.0), twist=0.0)
    return seeds


# -- geometry view -------------------------------------------------------- #


def _flesh_extreme_y(flesh: Mapping, head: np.ndarray, R: np.ndarray, length: float) -> tuple[float, float]:
    if flesh["kind"] == "gencyl":
        station_result = _gencyl_stations(flesh)
        if isinstance(station_result, _GencylStationObstruction):
            return (math.inf, -math.inf)
        radii = flesh["radii"]
        lows = tuple(
            float((head + R @ np.array([0.0, 0.0, s * length]))[1]) - float(r)
            for s, r in zip(station_result, radii)
        )
        highs = tuple(
            float((head + R @ np.array([0.0, 0.0, s * length]))[1]) + float(r)
            for s, r in zip(station_result, radii)
        )
        return (min(lows), max(highs))
    if flesh["kind"] == "loft":
        section_result = _loft_sections(dict(flesh))
        if isinstance(section_result, LoftSectionObstruction):
            return (math.inf, -math.inf)
        bounds = tuple(
            (
                float(
                    (
                        head
                        + R
                        @ np.asarray(
                            (0.0, 0.0, section.station * length),
                            dtype=np.float64,
                        )
                    )[1]
                ),
                math.hypot(section.width, section.depth),
            )
            for section in section_result
        )
        return (
            min(center_y - radius for center_y, radius in bounds),
            max(center_y + radius for center_y, radius in bounds),
        )
    t = float(flesh.get("t", 1.0))
    offset = np.asarray(flesh.get("offset", (0.0, 0.0, 0.0)), dtype=np.float64)
    center = head + R @ (np.array([0.0, 0.0, t * length]) + offset)
    size = np.asarray(flesh["size"], dtype=np.float64)
    if flesh["kind"] == "box":
        extra = float(flesh.get("round", 0.0))
        e = float((np.abs(R) @ (size + extra))[1])
    else:
        e = float(math.sqrt(float(np.sum((R[1, :] * size) ** 2))))
    return (float(center[1]) - e, float(center[1]) + e)


def _flesh_center(flesh: Mapping, head: np.ndarray, R: np.ndarray, length: float) -> np.ndarray:
    if flesh["kind"] == "gencyl":
        station_result = _gencyl_stations(flesh)
        if isinstance(station_result, _GencylStationObstruction):
            return head
        points = [head + R @ np.array([0.0, 0.0, s * length]) for s in station_result]
        return np.mean(np.asarray(points), axis=0)
    if flesh["kind"] == "loft":
        section_result = _loft_sections(dict(flesh))
        if isinstance(section_result, LoftSectionObstruction):
            return head
        return np.mean(
            np.asarray(
                tuple(
                    head
                    + R
                    @ np.asarray(
                        (0.0, 0.0, section.station * length),
                        dtype=np.float64,
                    )
                    for section in section_result
                )
            ),
            axis=0,
        )
    t = float(flesh.get("t", 1.0))
    offset = np.asarray(flesh.get("offset", (0.0, 0.0, 0.0)), dtype=np.float64)
    return head + R @ (np.array([0.0, 0.0, t * length]) + offset)


@dataclass(frozen=True)
class BodyGeometryView:
    table: Mapping[str, Mapping]
    tree: Tree
    flesh_map: Mapping[str, tuple[str, Mapping]]
    ports: Mapping[str, str]
    _points: Mapping[str, np.ndarray]

    def _owner(self, scope: Scope) -> str:
        match scope:
            case Bone(bone_id):
                return bone_id
            case Part(part_id):
                return self.flesh_map[part_id][0]
            case Landmark(name):
                return landmark_owner(name, self.tree, self.ports) or self.tree.root
            case _:
                return self.tree.root

    def point(self, scope: Scope) -> np.ndarray:
        match scope:
            case Landmark(name):
                return np.asarray(
                    self._points.get(name, self.table[self.tree.root]["head"]),
                    dtype=np.float64,
                )
            case Part(part_id):
                bid, flesh = self.flesh_map[part_id]
                rec = self.table[bid]
                return _flesh_center(
                    flesh,
                    np.asarray(rec["head"], dtype=np.float64),
                    np.asarray(rec["R"], dtype=np.float64),
                    float(rec["length"]),
                )
            case _:
                return np.asarray(self.table[self._owner(scope)]["head"], dtype=np.float64)

    def axis(self, scope: Scope) -> np.ndarray:
        return np.asarray(self.frame(scope), dtype=np.float64)[:, 2]

    def frame(self, scope: Scope) -> np.ndarray:
        match scope:
            case Landmark(name):
                owner = landmark_owner(name, self.tree, self.ports)
                bare = name.partition("/")[2] or name
                rec = self.table.get(owner)
                if rec is not None:
                    for port in rec.get("ports", ()):
                        if port["name"] == bare:
                            return _port_frame(rec, port)["R"]
                return np.asarray(self.table[owner or self.tree.root]["R"], dtype=np.float64)
            case _:
                return np.asarray(self.table[self._owner(scope)]["R"], dtype=np.float64)

    def extent_sites_y(self, scope: Scope) -> tuple[tuple[str, float], ...]:
        match scope:
            case Part(part_id):
                fleshes = ((part_id, *self.flesh_map[part_id]),)
            case Bone(bone_id):
                depths = subtree_depths(self.tree, bone_id)
                fleshes = tuple(
                    (_flesh_id(bid, flesh, index), bid, flesh)
                    for bid in depths
                    for index, flesh in enumerate(self.table[bid].get("flesh", ()))
                )
            case _:
                fleshes = ()
        sites: tuple[tuple[str, float], ...] = ()
        for fid, bid, flesh in fleshes:
            rec = self.table[bid]
            low, high = _flesh_extreme_y(
                flesh,
                np.asarray(rec["head"], dtype=np.float64),
                np.asarray(rec["R"], dtype=np.float64),
                float(rec["length"]),
            )
            sites = sites + ((f"{fid}#lo", low), (f"{fid}#hi", high))
        return sites or ((f"{self._owner(scope)}#o", float(self.point(scope)[1])),)

    def poses(self) -> Mapping[str, tuple[np.ndarray, np.ndarray]]:
        return {
            bid: (
                np.asarray(rec["head"], dtype=np.float64),
                np.asarray(rec["head"], dtype=np.float64)
                + np.asarray(rec["R"], dtype=np.float64)
                @ np.array([0.0, 0.0, float(rec["length"])]),
            )
            for bid, rec in self.table.items()
        }


def geometry_view(
    spec: Mapping,
    pose_joints: Mapping,
    attachments: Mapping[str, Attachment],
    params: Mapping[str, BoneParam],
) -> BodyGeometryView:
    tree = build_tree(spec)
    table = fk_table(spec, pose_joints, attachments, params)
    return BodyGeometryView(
        table=table,
        tree=tree,
        flesh_map=flesh_owner_map(tree),
        ports=port_owner_map(tree),
        _points=landmark_points(table),
    )


# -- evaluator ------------------------------------------------------------ #


def make_evaluator(
    spec: Mapping,
    pose_joints: Mapping,
    attachments: Mapping[str, Attachment],
    glued: Mapping[str, BoneParam],
    variables: tuple[VariableSpec, ...],
) -> Evaluator:
    def evaluate(values: tuple[float, ...]) -> GeometryView:
        overlaid = {bid: param for bid, param in glued.items()}
        families: dict[tuple[str, VariableKind], list[float]] = {}
        for spec_var, value in zip(variables, values):
            families.setdefault((spec_var.bone_id, spec_var.kind), []).append(float(value))
        for (bid, kind), slots in families.items():
            base = overlaid.get(bid, BoneParam(None, (0.0, 0.0), 0.0))
            if kind is VariableKind.TRANSLATION:
                overlaid[bid] = replace(base, translation=(slots[0], slots[1], slots[2]))
            elif kind is VariableKind.SWING:
                overlaid[bid] = replace(base, swing=(slots[0], slots[1]))
            else:
                overlaid[bid] = replace(base, twist=slots[0])
        return geometry_view(spec, pose_joints, attachments, overlaid)

    return evaluate


__all__ = [
    "BodyGeometryView",
    "BoneParam",
    "Tree",
    "ancestors",
    "build_tree",
    "constructed_identifiers",
    "flesh_owner_map",
    "fk_table",
    "geometry_view",
    "landmark_owner",
    "landmark_points",
    "lca",
    "make_evaluator",
    "port_owner_map",
    "seed_params",
    "subtree_depths",
]
