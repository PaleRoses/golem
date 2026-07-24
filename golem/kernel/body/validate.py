"""Compiler phase: skeleton validation (step 1)."""

from __future__ import annotations

from golem.kernel.body.gencyl import (
    _gencyl_station_violations,
    _loft_section_violations,
)
from golem.kernel.body.types import BONE_ROLES, DIALECT, FLESH_ROLES, _DOF


class _ValidateMixin:
    # -- step 1: validate ------------------------------------------------- #
    def validate(self) -> None:
        sk = self.spec.get("skeleton", {})
        root = sk.get("root", {})
        bones = sk.get("bones", [])
        seen = {root.get("id")}
        if self.spec.get("dialect") != DIALECT:
            self._viol("bad_dialect", "spec/dialect",
                       f"dialect {self.spec.get('dialect')!r} != {DIALECT!r}")
        self.violations.extend(_gencyl_station_violations(self.spec))
        self.violations.extend(_loft_section_violations(self.spec))
        self._validate_roles(root)
        for b in bones:
            self._validate_roles(b)
            bid = b["id"]
            if bid in seen:
                self._viol("dup_id", f"skeleton/{bid}", f"duplicate id {bid!r}")
            seen.add(bid)
            parent = b.get("parent")
            if parent not in seen and parent != root.get("id"):
                # parent must be declared before child (tree, DFS-authorable)
                if parent not in {x["id"] for x in bones}:
                    self._viol("bad_parent", f"skeleton/{bid}",
                               f"parent {parent!r} is not a declared bone/root")
            dof = b.get("joint", {}).get("dof")
            if dof not in _DOF:
                self._viol("bad_dof", f"skeleton/{bid}/joint",
                           f"dof {dof!r} not in {_DOF}")
            limits = b.get("joint", {}).get("limits")
            if limits:
                for k, rng in limits.items():
                    if not (isinstance(rng, (list, tuple)) and len(rng) == 2
                            and rng[0] <= rng[1]):
                        self._viol("bad_limits", f"skeleton/{bid}/joint/limits",
                                   f"limit {k}={rng} ill-formed (need [lo<=hi])")

    def _validate_roles(self, rec: dict) -> None:
        """Ontology roles are closed vocabulary: an unknown value is a hard
        violation, never a silently-ignored key (R3/R4)."""
        bid = rec.get("id", "?")
        role = rec.get("role")
        if role is not None and role not in BONE_ROLES:
            self._viol("bad_bone_role", f"skeleton/{bid}/role",
                       f"role {role!r} not in {BONE_ROLES}")
        for i, fl in enumerate(rec.get("flesh", ())):
            flesh_role = fl.get("role") if isinstance(fl, dict) else None
            if flesh_role is not None and flesh_role not in FLESH_ROLES:
                self._viol("bad_flesh_role", f"skeleton/{bid}/flesh[{i}]/role",
                           f"role {flesh_role!r} not in {FLESH_ROLES}")
