"""Compiler phase: reach-goal resolution (fence 1 -- one analytic two-bone solve)."""

from __future__ import annotations

import json
import math

import numpy as np

from golem.addressing.core import Landmark, parse_scope

from golem.kernel.body.canonical import _r, _rvec
from golem.kernel.body.contacts import _matches
from golem.kernel.body.linalg import _angle_between, _frame_from_axis, _normalize


class _GoalsMixin:
    # -- step 7: resolve reach goals (fence 1) ---------------------------- #
    def resolve_goals(self) -> None:
        goals = (self.spec.get("pose", {}) or {}).get("goals", []) or []
        for goal in goals:
            self._solve_goal(goal)

    def _solve_goal(self, goal: dict) -> None:
        chain = goal["chain"]
        if len(chain) != 2:
            self._viol("bad_goal_chain", f"pose/goals/{goal['id']}",
                       "exactly two bones per reach chain (fence 1)")
            return
        b0, b1 = self.bones[chain[0]], self.bones[chain[1]]

        # circularity guard: a reach chain containing a ground-contact part is a
        # compile violation, never an iteration (T4).
        ground_ids = self._contact_group_parts("support") + \
            self._contact_group_parts("planted")
        for bid in chain:
            for i, fl in enumerate(self.bones[bid]["flesh"]):
                pid = self._flesh_id(bid, fl, i)
                if any(_matches(pid, bid, pat) for pat in ground_ids):
                    self._viol("reach_chain_grounded", f"pose/goals/{goal['id']}",
                               f"reach chain bone {bid!r} carries ground-contact "
                               f"flesh {pid!r}: circular (T4)")

        S = b0["head"].copy()
        a = b0["length"]
        b = b1["length"]
        T = self._resolve_target(goal.get("target"), mirrored=b1["mirrored"])
        if T is None:
            self._viol("bad_goal_target", f"pose/goals/{goal['id']}",
                       f"target {goal.get('target')!r} unresolved")
            return

        # Mirrored chains solve once on the +x side; the engine instances the
        # solved arm onto -x via the mirror flag (a near-midline prop target such
        # as the greatsword grip is gripped by both hands by construction).

        diff = T - S
        d = float(np.linalg.norm(diff))
        n = diff / d if d > 1e-12 else np.array([0.0, 0.0, 1.0])
        pole = np.array(goal.get("pole", [0, 0, 1]), dtype=np.float64)

        rec = {"id": goal["id"], "chain": list(chain), "root": _rvec(S),
               "target": _rvec(T), "reach": _r(d), "max_reach": _r(a + b)}

        if d >= a + b - 1e-12:
            # unreachable: clamp at full extension aimed at target (fence 1).
            E = S + a * n
            W = S + (a + b) * n
            gap = d - (a + b)
            rec.update({"reachable": False, "elbow": _rvec(E), "wrist": _rvec(W),
                        "gap": _r(gap), "residual": _r(gap)})
            grad = [
                {"knob": f"skeleton/{chain[0]}@length", "direction": "+",
                 "estimate": _r(max(0.0, gap))},
                {"knob": f"skeleton/{chain[1]}@length", "direction": "+",
                 "estimate": _r(max(0.0, gap))},
            ]
            self._viol("goal_unreachable", f"pose/goals/{goal['id']}",
                       f"reach {d:.4f} > max {a + b:.4f} (gap {gap:.4f}); "
                       f"clamped at full extension", gap=_r(gap), gradient=grad)
            self.clamps.append({"goal": goal["id"], "gap": _r(gap)})
            # Fence 1: the chain STILL clamps at full extension aimed at the
            # target (the arm points at the goal, just too short) -- apply it.
            self._apply_ik_frames(chain, S, E, W)
        else:
            cos_a = (a * a + d * d - b * b) / (2 * a * d)
            alpha = math.acos(float(np.clip(cos_a, -1.0, 1.0)))
            u = pole - (pole @ n) * n
            if float(np.linalg.norm(u)) < 1e-9:
                # pole parallel to reach: deterministic perpendicular fallback.
                ref = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
                u = ref - (ref @ n) * n
            u = _normalize(u)
            ua_dir = math.cos(alpha) * n + math.sin(alpha) * u
            E = S + a * ua_dir
            fa_dir = _normalize(T - E)
            W = E + b * fa_dir
            interior = _angle_between(E - S, T - E)
            rec.update({"reachable": True, "elbow": _rvec(E), "wrist": _rvec(W),
                        "elbow_interior_deg": _r(interior),
                        "flexion_deg": _r(180.0 - interior),
                        "residual": _r(float(np.linalg.norm(W - T)))})
            self._apply_ik_frames(chain, S, E, W)
            self._check_ik_limits(goal["id"], chain, ua_dir, fa_dir, interior)

        self.solved_goals.append(rec)

    def _apply_ik_frames(self, chain, S, E, W) -> None:
        """Override the two chain bones' world frames from the IK solution and
        re-record their landmarks + any port frames (re-run FK for this chain
        only)."""
        b0, b1 = self.bones[chain[0]], self.bones[chain[1]]
        parent0 = self.bones[b0["parent"]]
        R0 = _frame_from_axis(E - S, parent0["R"][:, 1], parent0["R"][:, 0],
                              float(b0.get("twist_deg", 0.0)))
        b0["head"] = S
        b0["R"] = R0
        R1 = _frame_from_axis(W - E, R0[:, 1], R0[:, 0],
                              float(b1.get("twist_deg", 0.0)))
        b1["head"] = E
        b1["R"] = R1
        for bid in (chain[0], chain[1]):
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

    def _check_ik_limits(self, gid, chain, ua_dir, fa_dir, interior) -> None:
        """Fence 3: solved angles outside declared limits are applied and
        REPORTED with residuals (never clamped)."""
        b0, b1 = self.bones[chain[0]], self.bones[chain[1]]
        # forearm hinge flexion vs its declared pitch limits.
        flex = 180.0 - interior
        rng = (b1.get("limits") or {}).get("pitch")
        if rng and not (rng[0] <= flex <= rng[1]):
            over = flex - rng[1] if flex > rng[1] else rng[0] - flex
            self.limit_violations.append({
                "bone": chain[1], "axis": "pitch(flexion)", "value": _r(flex),
                "limit": list(rng), "residual": _r(abs(over)),
                "source": f"ik:{gid}",
            })
        # upper_arm deviation from rest direction (coarse ball-joint check).
        rest_dir_world = b0.get("R_rest", b0["R"])[:, 2]
        dev = _angle_between(ua_dir, rest_dir_world)
        lims = b0.get("limits") or {}
        max_lim = max((abs(x) for rng in lims.values() for x in rng), default=180.0)
        if dev > max_lim + 1e-9:
            self.limit_violations.append({
                "bone": chain[0], "axis": "ball_deviation", "value": _r(dev),
                "limit": _r(max_lim), "residual": _r(dev - max_lim),
                "source": f"ik:{gid}",
            })

    def _resolve_target(self, spec: str | None, mirrored: bool):
        # Reach-goal targets share the ``landmark:`` colon-scope decode with the
        # assert/schematic surface, delegated to F1.  A landmark key carries any
        # authored ``@view`` verbatim -- the table holds only unprojected keys --
        # so a view-qualified target resolves exactly as before (a miss).
        # ``world:<coords>`` is a literal-point target grammar F1's ``Scope`` has
        # no arm for (``World`` is the whole scene, not a coordinate); it is
        # rejected there by construction and decoded locally.
        if not spec:
            return None
        match parse_scope(spec):
            case Landmark(name, view):
                key = name if view is None else f"{name}@{view}"
                coords = self.landmarks.get(key)
                return (
                    np.array(coords, dtype=np.float64)
                    if coords is not None
                    else None
                )
            case _:
                if spec.startswith("world:"):
                    return np.array(
                        json.loads(spec[len("world:"):]), dtype=np.float64
                    )
                return None
