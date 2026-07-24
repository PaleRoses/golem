"""Compiler phase: rest frames, pose, forward kinematics, landmarks, ground-lift."""

from __future__ import annotations

import difflib
import functools
import math

import numpy as np

from golem.addressing.scope import Bone, Landmark, Part, Scope
from golem.kernel.body.canonical import _r, _rvec
from golem.kernel.body.contacts import _matches
from golem.kernel.body.gencyl import _gencyl_stations, _loft_sections
from golem.kernel.body.geometry import (
    BoneParam,
    Tree,
    _fixed_local_translation,
    _swung_direction,
    ancestors,
    build_tree,
    constructed_identifiers,
    fk_table,
    flesh_owner_map,
    landmark_owner,
    landmark_points,
    lca,
    make_evaluator,
    port_owner_map,
    seed_params,
    subtree_depths,
)
from golem.kernel.body.linalg import (
    _frame_from_axis,
    _normalize,
    _pose_matrix,
    _rodrigues,
)
from golem.kernel.body.records import project_bone_record
from golem.kernel.body.relation_solve import (
    AcceptedLocalSolve,
    DeepeningStage,
    Frontier,
    LocalRelationProblem,
    LocalSolveReceipt,
    RelationalSolveConfig,
    VariableKind,
    VariableSpec,
    _cumulative_frontiers,
    measure_relation,
    select_conventions,
    solve_local,
)
from golem.kernel.body.relations import (
    Attachment,
    FixedLocalAttachment,
    GoalRelationAuthorityObstruction,
    MalformedRelationObstruction,
    NamedSiteAttachment,
    RelationDeclaration,
    RelationKind,
    RelationSolvePolicy,
    UnresolvedRelationSelectorObstruction,
)
from golem.kernel.body.types import (
    LoftSectionObstruction,
    _GencylStationObstruction,
    RelationalSolveReceipt,
    SolvedLocalPose,
)

_TRANSLATION_KINDS = frozenset(
    (
        RelationKind.ATTACH_AT_NAMED_SITE,
        RelationKind.COINCIDENT,
        RelationKind.ABOVE,
        RelationKind.BELOW,
    )
)
_SWING_KINDS = frozenset((RelationKind.ALIGNED, RelationKind.PERPENDICULAR_TO))


def _unlocked_families(kind: RelationKind) -> tuple[VariableKind, ...]:
    if kind is RelationKind.MIRROR_OF:
        return (VariableKind.TRANSLATION, VariableKind.SWING, VariableKind.TWIST)
    if kind is RelationKind.BETWEEN:
        return (VariableKind.TRANSLATION, VariableKind.SWING)
    if kind in _SWING_KINDS:
        return (VariableKind.SWING,)
    return (VariableKind.TRANSLATION,)


def _scope_owner(
    scope: Scope, tree: Tree, flesh_map, ports
) -> str | None:
    match scope:
        case Bone(bone_id):
            return bone_id if bone_id in tree.records else None
        case Part(part_id):
            owner = flesh_map.get(part_id)
            return owner[0] if owner is not None else None
        case Landmark(name):
            return landmark_owner(name, tree, ports)
        case _:
            return None


def _yielding_owners(
    declaration: RelationDeclaration, tree: Tree, flesh_map, ports
) -> tuple[str, ...]:
    subject = _scope_owner(declaration.subject, tree, flesh_map, ports)
    reference = _scope_owner(declaration.reference, tree, flesh_map, ports)
    reference_b = (
        _scope_owner(declaration.reference_b, tree, flesh_map, ports)
        if declaration.reference_b is not None
        else None
    )
    match declaration.solve:
        case RelationSolvePolicy.SUBJECT:
            owners = (subject,)
        case RelationSolvePolicy.REFERENCE:
            owners = (reference,)
        case _:
            owners = (subject, reference, reference_b)
    return tuple(owner for owner in owners if owner is not None)


def _assign_node(
    declaration: RelationDeclaration, tree: Tree, flesh_map, ports
) -> str:
    subject = _scope_owner(declaration.subject, tree, flesh_map, ports) or tree.root
    reference = _scope_owner(declaration.reference, tree, flesh_map, ports) or tree.root
    match declaration.solve:
        case RelationSolvePolicy.SUBJECT:
            return subject
        case RelationSolvePolicy.REFERENCE:
            return reference
        case _:
            node = lca(tree, subject, reference)
            if declaration.reference_b is not None:
                node = lca(
                    tree,
                    node,
                    _scope_owner(declaration.reference_b, tree, flesh_map, ports)
                    or tree.root,
                )
            return node


def _family_specs(
    bone_id: str, family: VariableKind, param: BoneParam
) -> tuple[VariableSpec, ...]:
    if family is VariableKind.TRANSLATION:
        translation = param.translation or (0.0, 0.0, 0.0)
        return tuple(
            VariableSpec(f"skeleton/{bone_id}/t{axis}", bone_id, family, float(translation[axis]))
            for axis in range(3)
        )
    if family is VariableKind.SWING:
        return tuple(
            VariableSpec(f"skeleton/{bone_id}/s{axis}", bone_id, family, float(param.swing[axis]))
            for axis in range(2)
        )
    return (
        VariableSpec(f"skeleton/{bone_id}/w", bone_id, family, float(param.twist)),
    )


def _node_stages(
    node: str,
    declarations: tuple[RelationDeclaration, ...],
    tree: Tree,
    flesh_map,
    ports,
    attachments: dict[str, Attachment],
    glued: dict[str, BoneParam],
) -> tuple[tuple[VariableSpec, ...], tuple[Frontier, ...]]:
    depths = subtree_depths(tree, node)
    collected: dict[str, tuple[VariableSpec, DeepeningStage, int]] = {}
    for declaration in declarations:
        for bone_id in _yielding_owners(declaration, tree, flesh_map, ports):
            if bone_id == tree.root or bone_id not in depths:
                continue
            depth = depths[bone_id]
            for family in _unlocked_families(declaration.kind):
                if family is VariableKind.TRANSLATION and not isinstance(
                    attachments.get(bone_id), NamedSiteAttachment
                ):
                    continue
                stage = (
                    DeepeningStage.LOCAL
                    if depth == 0
                    else (
                        DeepeningStage.TRANSLATION
                        if family is VariableKind.TRANSLATION
                        else DeepeningStage.PARAMETER
                    )
                )
                for spec in _family_specs(
                    bone_id, family, glued.get(bone_id, BoneParam(None, (0.0, 0.0), 0.0))
                ):
                    collected[spec.address] = (spec, stage, depth)
    order_index = {bid: index for index, bid in enumerate(tree.order)}
    ordered = sorted(
        collected.values(),
        key=lambda entry: (order_index[entry[0].bone_id], entry[0].address),
    )
    base = tuple(spec for spec, stage, _ in ordered if stage is DeepeningStage.LOCAL)
    frontiers: tuple[Frontier, ...] = ()
    for stage in (DeepeningStage.TRANSLATION, DeepeningStage.PARAMETER):
        depths_present = sorted(
            {depth for _, entry_stage, depth in ordered if entry_stage is stage}
        )
        for depth in depths_present:
            frontiers = frontiers + (
                Frontier(
                    stage,
                    depth,
                    tuple(
                        spec
                        for spec, entry_stage, entry_depth in ordered
                        if entry_stage is stage and entry_depth == depth
                    ),
                ),
            )
    return base, frontiers


def _glue(
    glued: dict[str, BoneParam],
    variables: tuple[VariableSpec, ...],
    solution: tuple[float, ...],
) -> dict[str, BoneParam]:
    families: dict[tuple[str, VariableKind], list[float]] = {}
    for spec, value in zip(variables, solution):
        families.setdefault((spec.bone_id, spec.kind), []).append(float(value))
    updated = dict(glued)
    for (bone_id, kind), slots in families.items():
        current = updated.get(bone_id, BoneParam(None, (0.0, 0.0), 0.0))
        if kind is VariableKind.TRANSLATION:
            updated[bone_id] = BoneParam((slots[0], slots[1], slots[2]), current.swing, current.twist)
        elif kind is VariableKind.SWING:
            updated[bone_id] = BoneParam(current.translation, (slots[0], slots[1]), current.twist)
        else:
            updated[bone_id] = BoneParam(current.translation, current.swing, slots[0])
    return updated


def _resolution_obstruction(
    declaration: RelationDeclaration,
    tree: Tree,
    flesh_map,
    ports,
    points,
    identifiers: tuple[str, ...],
) -> UnresolvedRelationSelectorObstruction | None:
    roles: tuple[tuple[Scope, str, str], ...] = (
        (declaration.subject, declaration.subject_selector, "subject"),
        (
            declaration.reference,
            declaration.reference_selector,
            "references/0" if declaration.reference_b is not None else "reference",
        ),
    )
    if declaration.reference_b is not None:
        roles = roles + (
            (
                declaration.reference_b,
                declaration.reference_b_selector or "",
                "references/1",
            ),
        )
    for scope, selector, suffix in roles:
        resolved = True
        match scope:
            case Bone(bone_id):
                resolved = bone_id in tree.records
            case Part(part_id):
                resolved = part_id in flesh_map
            case Landmark(name):
                resolved = name in points or landmark_owner(name, tree, ports) is not None
        if not resolved:
            return UnresolvedRelationSelectorObstruction(
                f"{declaration.address}/{suffix}",
                selector,
                tuple(difflib.get_close_matches(selector, identifiers, n=3)),
            )
    return None


class _KinematicsMixin:
    # -- mirror propagation ----------------------------------------------- #
    def _is_mirrored(self, bid: str) -> bool:
        return self.bones[bid]["mirrored"]

    # -- relational placement: deferred hierarchical solve ---------------- #
    def solve_relations(
        self,
        explicit: tuple[RelationDeclaration, ...],
        attachments: dict[str, Attachment],
    ) -> tuple[dict[str, SolvedLocalPose], RelationalSolveReceipt | None, tuple]:
        spec = self.spec
        pose_joints = (spec.get("pose", {}) or {}).get("joints", {}) or {}
        tree = build_tree(spec)
        flesh_map = flesh_owner_map(tree)
        ports = port_owner_map(tree)
        identifiers = constructed_identifiers(tree)
        seeds = seed_params(spec, pose_joints, attachments)
        declarations = self._synthesized_attachments(attachments, tree) + explicit

        points = landmark_points(fk_table(spec, pose_joints, attachments, seeds))
        resolution = tuple(
            obstruction
            for declaration in declarations
            for obstruction in (
                _resolution_obstruction(
                    declaration, tree, flesh_map, ports, points, identifiers
                ),
            )
            if obstruction is not None
        )
        if resolution:
            return {}, None, resolution

        by_node: dict[str, list[RelationDeclaration]] = {}
        for declaration in declarations:
            by_node.setdefault(
                _assign_node(declaration, tree, flesh_map, ports), []
            ).append(declaration)

        config = RelationalSolveConfig()
        glued = dict(seeds)
        sections: list[LocalSolveReceipt] = []

        def post_order(
            state: tuple[dict[str, BoneParam], tuple[LocalSolveReceipt, ...], tuple],
            node: str,
        ) -> tuple[dict[str, BoneParam], tuple[LocalSolveReceipt, ...], tuple]:
            glue_state, receipts, failures = state
            if failures:
                return state
            owned = tuple(by_node.get(node, ()))
            if not owned:
                return state
            base, frontiers = _node_stages(
                node, owned, tree, flesh_map, ports, attachments, glue_state
            )
            full = _cumulative_frontiers(base, frontiers)[-1][2]
            evaluator = make_evaluator(spec, pose_joints, attachments, glue_state, full)
            if not full:
                view = evaluator(())
                choices = select_conventions(owned, view, config)
                conventions = {choice.relation_id: choice.selected for choice in choices}
                measurements = tuple(
                    measure_relation(
                        declaration,
                        conventions.get(declaration.relation_id),
                        (None, None),
                        view,
                        config,
                    )
                    for declaration in owned
                )
                if all(measurement.satisfied for measurement in measurements):
                    return (
                        glue_state,
                        receipts
                        + (
                            LocalSolveReceipt(
                                owner_address=f"skeleton/{node}",
                                measurements=measurements,
                                conventions=choices,
                                stage=DeepeningStage.LOCAL,
                                depth=0,
                                variable_count=0,
                                rank=0,
                                active_site_history=(),
                                evaluations=0,
                                maximum_normalized_residual=max(
                                    (m.normalized_residual for m in measurements),
                                    default=0.0,
                                ),
                            ),
                        ),
                        failures,
                    )
            problem = LocalRelationProblem(
                owner_address=f"skeleton/{node}",
                declarations=owned,
                evaluator=evaluator,
                base_variables=base,
                frontiers=frontiers,
            )
            result = solve_local(problem, config)
            if isinstance(result, AcceptedLocalSolve):
                return (
                    _glue(glue_state, result.variables, result.solution),
                    receipts + (result.receipt,),
                    failures,
                )
            return glue_state, receipts, (result,)

        glued, section_tuple, failures = functools.reduce(
            post_order, reversed(tree.order), (glued, tuple(sections), ())
        )
        if failures:
            return {}, None, failures

        solved = self._solved_poses(tree, attachments, glued)
        receipt = RelationalSolveReceipt(
            sections=section_tuple, declarations=declarations
        )
        return solved, receipt, ()

    def _synthesized_attachments(
        self, attachments: dict[str, Attachment], tree: Tree
    ) -> tuple[RelationDeclaration, ...]:
        return tuple(
            RelationDeclaration(
                relation_id=f"attach:{bid}",
                kind=RelationKind.ATTACH_AT_NAMED_SITE,
                authored_kind="attach_at_named_site",
                subject=Landmark(f"{bid}/head"),
                subject_selector=f"landmark:{bid}/head",
                reference=attach.site,
                reference_selector=attach.selector,
                solve=RelationSolvePolicy.SUBJECT,
                distance=0.0,
                address=f"skeleton/{bid}/attach",
            )
            for bid in tree.order[1:]
            for attach in (attachments.get(bid),)
            if isinstance(attach, NamedSiteAttachment)
        )

    def _solved_poses(
        self, tree: Tree, attachments: dict[str, Attachment], glued: dict[str, BoneParam]
    ) -> dict[str, SolvedLocalPose]:
        solved: dict[str, SolvedLocalPose] = {}
        for bid in tree.order[1:]:
            param = glued.get(bid, BoneParam(None, (0.0, 0.0), 0.0))
            attach = attachments.get(bid, FixedLocalAttachment())
            movable = isinstance(attach, NamedSiteAttachment)
            if not (movable or param.swing != (0.0, 0.0) or param.twist != 0.0):
                continue
            bone = tree.records[bid]
            parent_length = float(tree.records[bone["parent"]].get("length", 0.0))
            local = (
                (param.translation or (0.0, 0.0, parent_length))
                if movable
                else tuple(_fixed_local_translation(attach, parent_length))
            )
            rest_direction = _swung_direction(bone.get("rest_dir", (0.0, 0.0, 1.0)), param.swing)
            solved[bid] = SolvedLocalPose(
                bone_id=bid,
                local_translation=tuple(float(component) for component in local),
                rest_direction=tuple(float(component) for component in rest_direction),
                twist=float(bone.get("twist_deg", 0.0)) + param.twist,
            )
        return solved

    def _relation_validation(
        self,
        explicit: tuple[RelationDeclaration, ...],
        attachments: dict[str, Attachment],
    ) -> tuple:
        tree = build_tree(self.spec)
        flesh_map = flesh_owner_map(tree)
        ports = port_owner_map(tree)
        goals = (self.spec.get("pose", {}) or {}).get("goals", []) or []
        chains = tuple(
            (goal.get("id", ""), frozenset(goal.get("chain", []))) for goal in goals
        )

        def mirrored(bone_id: str) -> bool:
            return any(
                tree.records[node].get("mirror", False)
                for node in ancestors(tree, bone_id)
            )

        obstructions: tuple = ()
        for declaration in explicit:
            movable = frozenset(
                _yielding_owners(declaration, tree, flesh_map, ports)
            )
            for goal_id, chain in chains:
                overlap = movable & chain
                if overlap:
                    obstructions = obstructions + (
                        GoalRelationAuthorityObstruction(
                            declaration.relation_id, goal_id, tuple(sorted(overlap))
                        ),
                    )
            if declaration.kind is RelationKind.MIRROR_OF:
                for owner in (
                    _scope_owner(declaration.subject, tree, flesh_map, ports),
                    _scope_owner(declaration.reference, tree, flesh_map, ports),
                ):
                    if owner is not None and mirrored(owner):
                        obstructions = obstructions + (
                            MalformedRelationObstruction(
                                declaration.address,
                                f"mirror_of participant {owner!r} is under"
                                " mirror:true; sagittal symmetry has one owner",
                            ),
                        )
        for bid, attach in attachments.items():
            if isinstance(attach, NamedSiteAttachment):
                for goal_id, chain in chains:
                    if bid in chain:
                        obstructions = obstructions + (
                            GoalRelationAuthorityObstruction(
                                f"attach:{bid}", goal_id, (bid,)
                            ),
                        )
        return obstructions

    # -- steps 2-4: rest frames + pose + FK (single DFS) ------------------ #
    def forward_kinematics(self) -> None:
        sk = self.spec["skeleton"]
        root = sk["root"]
        pose_joints = (self.spec.get("pose", {}) or {}).get("joints", {}) or {}

        # root record: world axes at root.world.
        rid = root["id"]
        self.bones[rid] = {
            **project_bone_record(root, parent_mirrored=False, is_root=True),
            "R": np.eye(3),
            "head": np.array(root["world"], dtype=np.float64),
            "twist_deg": 0.0,
        }
        self.order.append(rid)

        # children indexed by parent, in authored order (deterministic DFS).
        by_parent: dict[str, list[dict]] = {}
        for b in sk["bones"]:
            by_parent.setdefault(b["parent"], []).append(b)

        def walk(bid: str) -> None:
            for b in by_parent.get(bid, []):
                self._place_bone(b, pose_joints)
                self.order.append(b["id"])
                walk(b["id"])

        walk(rid)

        # landmark table: every joint head/tail/center + ports (pre-lift).
        self._record_landmarks()

    def _place_bone(self, b: dict, pose_joints: dict) -> None:
        parent = self.bones[b["parent"]]
        Rp = parent["R"]
        attach = b.get("attach", {"t": 1.0})
        t = float(attach.get("t", 1.0))
        offset = np.array(attach.get("offset", [0, 0, 0]), dtype=np.float64)
        head = parent["head"] + Rp @ (offset + np.array([0, 0, t * parent["length"]]))

        rest_dir = np.array(b["rest_dir"], dtype=np.float64) if "rest_dir" in b \
            else np.array([0, 0, 1], dtype=np.float64)
        z_world = Rp @ _normalize(rest_dir)
        up_hint = Rp[:, 1]        # parent's +y in world
        fallback = Rp[:, 0]       # parent's +x in world
        twist = float(b.get("twist_deg", 0.0))
        R_rest = _frame_from_axis(z_world, up_hint, fallback, twist)

        mirrored = bool(b.get("mirror", False)) or parent["mirrored"]
        dof = b.get("joint", {}).get("dof", "fixed")
        limits = b.get("joint", {}).get("limits", {}) or {}

        # explicit pose (fence 3/4): fixed => any pose is a violation; else apply
        # R_pose in the rest-local frame; asymmetric pose on a mirrored subtree
        # is a violation (T2).
        R_bone = R_rest
        pose = pose_joints.get(b["id"])
        if pose:
            if dof == "fixed":
                self._viol("pose_on_fixed", f"skeleton/{b['id']}/joint",
                           f"pose {pose} authored on a fixed joint")
            elif mirrored:
                self._viol("asymmetric_pose_on_mirror", f"skeleton/{b['id']}",
                           f"explicit pose {pose} on mirrored subtree bone "
                           f"{b['id']!r} breaks symmetry; use split_mirror")
            else:
                yaw = float(pose.get("yaw", 0.0))
                pitch = float(pose.get("pitch", 0.0))
                roll = float(pose.get("roll", 0.0))
                R_bone = R_rest @ _pose_matrix(yaw, pitch, roll)
                self._check_pose_limits(b["id"], dof, limits, yaw, pitch, roll)

        solved = self._solved_params.get(b["id"])
        if solved is not None:
            head = parent["head"] + Rp @ np.array(solved.local_translation, dtype=np.float64)
            z_world = Rp @ np.array(solved.rest_direction, dtype=np.float64)
            R_rest = _frame_from_axis(z_world, up_hint, fallback, solved.twist)
            R_bone = R_rest

        self.bones[b["id"]] = {
            **project_bone_record(b, parent_mirrored=parent["mirrored"], is_root=False),
            "R": R_bone, "R_rest": R_rest,
            "head": head,
            "twist_deg": twist, "rest_dir": rest_dir,
        }

    def _check_pose_limits(self, bid, dof, limits, yaw, pitch, roll) -> None:
        """Fence 3: record + report, never enforce."""
        checks = {"yaw": yaw, "pitch": pitch, "roll": roll}
        if dof == "hinge":
            checks = {"pitch": pitch}  # single flexion axis
        for k, v in checks.items():
            rng = limits.get(k)
            if rng and not (rng[0] <= v <= rng[1]):
                over = v - rng[1] if v > rng[1] else rng[0] - v
                self.limit_violations.append({
                    "bone": bid, "axis": k, "value": _r(v),
                    "limit": list(rng), "residual": _r(abs(over)),
                    "source": "pose",
                })

    def _record_landmarks(self) -> None:
        for bid in self.order:
            rec = self.bones[bid]
            head = rec["head"]
            tail = head + rec["R"] @ np.array([0, 0, rec["length"]])
            self.landmarks[f"{bid}/head"] = _rvec(head)
            self.landmarks[f"{bid}/tail"] = _rvec(tail)
            self.landmarks[f"{bid}/center"] = _rvec((head + tail) / 2.0)
            for port in rec.get("ports", []):
                pf = self._port_frame(rec, port)
                self.landmarks[port["name"]] = _rvec(pf["origin"])
                self.landmarks[f"{bid}/{port['name']}"] = _rvec(pf["origin"])

    def _port_frame(self, bone_rec: dict, port: dict) -> dict:
        t = float(port.get("t", 1.0))
        origin = bone_rec["head"] + bone_rec["R"] @ np.array([0, 0, t * bone_rec["length"]])
        twist = float(port.get("twist_deg", 0.0))
        # port frame = bone frame twisted about +z (fingers +z, back +y, palm -y).
        z = bone_rec["R"][:, 2]
        y = bone_rec["R"][:, 1]
        if twist:
            y = _rodrigues(y, z, twist)
        y = _normalize(y)
        x = np.cross(y, z)
        R = np.stack([_normalize(x), y, z], axis=1)
        return {"origin": origin, "R": R}

    # -- step 5: ground-lift over support-group flesh --------------------- #
    def ground_lift_body(self) -> None:
        support_ids = self._contact_group_parts("support")
        low = math.inf
        for bid in self.order:
            rec = self.bones[bid]
            for i, fl in enumerate(rec["flesh"]):
                pid = self._flesh_id(bid, fl, i)
                if not any(_matches(pid, bid, pat) for pat in support_ids):
                    continue
                low = min(low, self._flesh_low_y(rec, fl))
        if low == math.inf:
            self.ground_lift = 0.0
            return
        self.ground_lift = self.ground_y - low
        shift = np.array([0.0, self.ground_lift, 0.0])
        for bid in self.order:
            self.bones[bid]["head"] = self.bones[bid]["head"] + shift
        # re-record landmarks in the lifted frame.
        self.landmarks = {}
        self._record_landmarks()

    def _flesh_low_y(self, rec: dict, fl: dict) -> float:
        head = rec["head"]
        R = rec["R"]
        L = rec["length"]
        if fl["kind"] == "gencyl":
            radii = fl["radii"]
            station_result = _gencyl_stations(fl)
            if isinstance(station_result, _GencylStationObstruction):
                return math.inf
            samples = station_result
            low = math.inf
            for s, r in zip(samples, radii):
                p = head + R @ np.array([0, 0, s * L])
                low = min(low, float(p[1]) - float(r))
            return low
        if fl["kind"] == "loft":
            section_result = _loft_sections(fl)
            if isinstance(section_result, LoftSectionObstruction):
                return math.inf
            return min(
                float(
                    (
                        head
                        + R
                        @ np.asarray(
                            (0.0, 0.0, section.station * L),
                            dtype=np.float64,
                        )
                    )[1]
                )
                - math.hypot(section.width, section.depth)
                for section in section_result
            )
        t = float(fl.get("t", 1.0))
        offset = np.array(fl.get("offset", [0, 0, 0]), dtype=np.float64)
        c = head + R @ (np.array([0, 0, t * L]) + offset)
        size = np.array(fl["size"], dtype=np.float64)
        if fl["kind"] == "box":
            # a rotated box's corner reaches its AABB: sum |R[1,j]| (size+round).
            extra = float(fl.get("round", 0.0))
            e = float((np.abs(R) @ (size + extra))[1])
        else:
            # a rotated ELLIPSOID's true lowest surface point is
            # center_y - sqrt(sum_j (R[1,j] size_j)^2)  (NOT the looser AABB
            # sum|R[1,j]|size_j) -- this matches the surface proprio samples,
            # so the support group plants where the balance sensor sees it.
            e = float(math.sqrt(float(np.sum((R[1, :] * size) ** 2))))
        return float(c[1]) - e

    def _named_port_frame(self, ref: str | None):
        if not ref:
            return None
        # ref like "forearm/wrist"
        if "/" in ref:
            bid, pname = ref.split("/", 1)
        else:
            bid, pname = None, ref
        for b in self.order:
            rec = self.bones[b]
            for port in rec.get("ports", []):
                if port["name"] == pname and (bid is None or b == bid):
                    return self._port_frame(rec, port)
        return None

    def _port_bone(self, port_ref: str) -> str:
        bid = port_ref.split("/", 1)[0] if "/" in port_ref else None
        if bid:
            return bid
        pname = port_ref
        for b in self.order:
            for port in self.bones[b].get("ports", []):
                if port["name"] == pname:
                    return b
        return self.order[0]
