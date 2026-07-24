"""Compiler phase: assemble the M2 intent block + the compile-receipt sidecar."""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from pathlib import Path

from golem.kernel.body.canonical import _r
from golem.kernel.body.contacts import _matches
from golem.kernel.body.gencyl import _gencyl_stations, _loft_sections
from golem.kernel.body.types import (
    BONE_ROLES,
    FLESH_ROLE_NON_CARRIER,
    LoftSectionObstruction,
)


class _IntentMixin:
    # -- step 9: assemble intent + receipt -------------------------------- #
    def build_intent(
        self,
        anatomy: dict[str, object] | None = None,
        *,
        parts: list[dict] | None = None,
        provenance: dict[str, str] | None = None,
    ) -> dict:
        emitted_parts = self.parts if parts is None else parts
        emitted_provenance = self.provenance if provenance is None else provenance
        contract = self._load_contract()
        clauses = contract.get("clauses", []) if contract else []
        ground_parts = self._ground_intent_parts(emitted_parts)
        attach = self._derive_attach() + self._authored_attach(
            contract, emitted_parts
        )
        midline = self._authored_midline(contract, emitted_parts)
        intent = {
            "name": self.spec.get("name", "?"),
            "txn": 1,
            "note": "compiled by v03/body.py (M3)",
            "ground": {"parts": ground_parts,
                       "tolerance": float(self.spec.get("ground_tol", 0.02))},
            "attach": attach,
            "landmarks": self.landmarks,
            "provenance": dict(emitted_provenance),
            "asserts": {"pack": contract.get("contract", "body.contract")
                        if contract else "body.contract",
                        "clauses": clauses},
        }
        # M4 (optional): pass an authored pose schematic through to the intent
        # block verbatim. Pure passthrough -- no solver obligation, no compiler
        # fence touched -- and a strict no-op when the spec carries none (so
        # knight_body.json's compile / digest is byte-unchanged).
        schematic = self.spec.get("schematic")
        if schematic:
            intent["schematic"] = schematic
        # Midline declarations ride the intent block only when authored, so
        # specs without the channel compile byte-identically to before.
        if midline:
            intent["midline"] = midline
        roles = self._declared_roles()
        if roles:
            intent["roles"] = roles
        if anatomy is not None:
            intent["anatomy"] = anatomy
        return intent

    def _declared_roles(self) -> dict[str, list[str]]:
        """Authored ontology-role declarations (R3), keyed by role value:
        handle bone ids under "handle", non-carrier flesh part ids under
        "non_carrier". Strictly absent when nothing is declared so role-free
        specs compile byte-identically."""
        roles: dict[str, list[str]] = {}
        handles = sorted(
            bid
            for bid in self.order
            if self.bones[bid].get("role") in BONE_ROLES
        )
        if handles:
            roles["handle"] = handles
        non_carrier = sorted(
            self._flesh_id(bid, fl, i)
            for bid in self.order
            for i, fl in enumerate(self.bones[bid]["flesh"])
            if isinstance(fl, dict)
            and fl.get("role") == FLESH_ROLE_NON_CARRIER
        )
        non_carrier = sorted((*non_carrier, *self._web_non_carrier))
        if non_carrier:
            roles["non_carrier"] = non_carrier
        return roles

    def _ground_intent_parts(self, parts: list[dict]) -> list[str]:
        """Concrete emitted part ids for the declared support/planted groups."""
        pats = self._contact_group_parts("support") + self._contact_group_parts("planted")
        out: list[str] = []
        emitted = {p["id"] for p in parts}
        for bid in self.order:
            rec = self.bones[bid]
            for i, fl in enumerate(rec["flesh"]):
                pid = self._flesh_id(bid, fl, i)
                if any(_matches(pid, bid, pat) for pat in pats):
                    out.append(pid)
        for pid in emitted:
            if any(_matches(pid, pid.split("/")[0], pat) for pat in pats):
                if pid not in out:
                    out.append(pid)
        return sorted(set(out))

    def _primary_flesh_ids(self) -> dict[str, str]:
        """Bone id -> its primary (first) flesh part id, for fleshed bones."""
        return {
            bid: self._flesh_id(bid, self.bones[bid]["flesh"][0], 0)
            for bid in self.order
            if self.bones[bid]["flesh"]
        }

    def _web_anchor_flesh(
        self, prim: Mapping[str, str]
    ) -> dict[str, tuple[str, ...]]:
        """Web id -> the primary flesh ids of its anchor bones, in authored
        anchor order: the exact endpoints of the derived anchor-to-web pairs.
        One minting site, consumed by the derived attach channel and by the
        attach-rejection neighborhood (B15) alike."""
        return {
            web_id: tuple(
                prim[bone_id] for bone_id in anchor_bones if bone_id in prim
            )
            for web_id, anchor_bones in self._web_anchors
        }

    def _declarable_part_ids(self, parts: list[dict]) -> frozenset[str]:
        """The contract vocabulary's writable part-id namespace: every emitted
        part plus every spanning-web id. Webs are flesh topology (R10) --
        proprio promotes them into the sensed part set and names them in
        anomaly addresses, so the contract channels admit the same ids the
        senses can speak (R12). One owner: the web namespace is decoded by
        webs.py and held as ``self._web_anchors``; this reads that single
        record, never a re-derivation."""
        return frozenset(
            (
                *(str(part["id"]) for part in parts),
                *(web_id for web_id, _anchors in self._web_anchors),
            )
        )

    def _derive_attach(self) -> list[list[str]]:
        """Attach pairs from parent-child adjacency of fleshed bones + mounts +
        prop internal chains + hand->gripped prop landmark. Used by proprio for
        anomaly classification only (not correctness)."""
        prim = self._primary_flesh_ids()
        pairs: list[list[str]] = []
        for bid in self.order:
            rec = self.bones[bid]
            parent = rec["parent"]
            # bone-internal flesh chain (multi-flesh bones)
            fls = rec["flesh"]
            for i in range(1, len(fls)):
                pairs.append([self._flesh_id(bid, fls[0], 0),
                              self._flesh_id(bid, fls[i], i)])
            if parent is None:
                continue
            # nearest fleshed ancestor
            anc = parent
            while anc is not None and anc not in prim:
                anc = self.bones[anc]["parent"]
            if anc in prim and bid in prim:
                pairs.append([prim[anc], prim[bid]])
        # mounts: forearm primary <-> hand palm
        for mount in self.spec.get("mounts", []):
            pb = self._port_bone(mount["port"])
            if pb in prim:
                pairs.append([prim[pb], "hand_palm"])
        # spanning webs (R10): the membrane MEETING each anchor bone's flesh
        # is the authored intent, never an accident -- declare every anchor
        # pair so proprio reads the fusion as the topology it is.
        for web_id, anchor_parts in self._web_anchor_flesh(prim).items():
            pairs.extend([anchor_part, web_id] for anchor_part in anchor_parts)
        # prop internal adjacency (declared as authored order chain)
        for prop in self.spec.get("props", []):
            pid = prop["id"]
            plist = prop.get("parts", [])
            for i in range(1, len(plist)):
                pairs.append([f"{pid}/{plist[i-1]['id']}", f"{pid}/{plist[i]['id']}"])
        return [list(p) for p in pairs]

    def _authored_attach(
        self, contract: dict | None, parts: list[dict]
    ) -> list[list[str]]:
        """Author-declared attach pairs from the contract block. The derived
        pairs cover skeletal adjacency; fusion that is intentional but not
        skeletal (muscle bellies sinking into bone masses, a brow ridge
        meeting its host flesh) is declared here so proprio reads it as
        intent, not accident. Strictly additive when absent."""
        if not contract:
            return []
        authored = contract.get("attach", [])
        if not isinstance(authored, list):
            self._viol(
                "bad_contract_attach",
                "spec/contract/attach",
                "attach must be a list of part-id pairs, "
                f"got {type(authored).__name__}",
            )
            return []
        known = self._declarable_part_ids(parts)
        pairs: list[list[str]] = []
        for index, entry in enumerate(authored):
            address = f"spec/contract/attach/{index}"
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 2
                or not all(isinstance(pid, str) and pid for pid in entry)
            ):
                self._viol(
                    "bad_contract_attach",
                    address,
                    f"attach entry must be two part ids, got {entry!r}",
                )
                continue
            pair = [str(entry[0]), str(entry[1])]
            unknown = [pid for pid in pair if pid not in known]
            if unknown:
                self._viol(
                    "bad_contract_attach",
                    address,
                    f"unknown part ids {unknown} "
                    f"({self._attach_neighborhood(unknown, known)})",
                )
                continue
            pairs.append(pair)
        return pairs

    def _attach_neighborhood(
        self, unknown: list[str], known: frozenset[str]
    ) -> str:
        """B15: an unknown-id rejection enumerates the legal neighborhood of
        each offending id -- the closest declarable ids (A6's
        enumerate-the-legal-set law, in the difflib shape the web-anchor
        rejection already uses) and, when the offender reaches for a spanning
        web, the anchor flesh that sensed interface derives from: the writable
        endpoints of the derived anchor-to-web pairs."""
        anchor_flesh = self._web_anchor_flesh(self._primary_flesh_ids())
        return "; ".join(
            self._id_neighborhood(pid, known, anchor_flesh) for pid in unknown
        )

    @staticmethod
    def _id_neighborhood(
        pid: str,
        known: frozenset[str],
        anchor_flesh: Mapping[str, tuple[str, ...]],
    ) -> str:
        closest = tuple(difflib.get_close_matches(pid, sorted(known), n=3))
        web_notes = tuple(
            f"web {web_id!r} derives from anchor flesh "
            f"[{', '.join(anchor_parts)}]"
            for web_id, anchor_parts in anchor_flesh.items()
            if web_id in closest and anchor_parts
        )
        return "; ".join(
            (
                f"{pid!r}: closest declarable "
                f"{', '.join(closest) or 'none'}",
                *web_notes,
            )
        )

    def _authored_midline(
        self, contract: dict | None, parts: list[dict]
    ) -> list[str]:
        """Author-declared midline parts: mirrored flesh whose .L/.R instances
        are MEANT to meet across x=0 (a fused haunch, a joined hip cap).
        Undeclared, proprio reads the crossing as accidental cross-plane
        fusion; declared, it is intent. Unknown ids are hard violations,
        exactly like the attach channel. Strictly additive when absent."""
        if not contract:
            return []
        declared = contract.get("midline", [])
        if not isinstance(declared, list):
            self._viol(
                "bad_contract_midline",
                "spec/contract/midline",
                "midline must be a list of part ids, "
                f"got {type(declared).__name__}",
            )
            return []
        known = self._declarable_part_ids(parts)
        out: list[str] = []
        for index, entry in enumerate(declared):
            address = f"spec/contract/midline/{index}"
            if not isinstance(entry, str) or not entry:
                self._viol(
                    "bad_contract_midline",
                    address,
                    f"midline entry must be a part id, got {entry!r}",
                )
                continue
            if entry not in known:
                self._viol(
                    "bad_contract_midline",
                    address,
                    f"unknown part id {entry!r}",
                )
                continue
            out.append(entry)
        return out

    def _load_contract(self) -> dict | None:
        cref = self.spec.get("contract")
        if not cref:
            return None
        if isinstance(cref, dict):
            # v0.4 session dialect: an inline contract (same shape as the
            # file: {"contract": name, "clauses": [...]}). Additive; file
            # references behave exactly as before.
            return cref
        path = self.spec_dir / cref
        try:
            return json.loads(Path(path).read_text())
        except Exception as exc:  # noqa: BLE001
            self._viol("bad_contract", "spec/contract", f"{exc}")
            return None

    def _gencyl_station_receipt_rows(self) -> tuple[dict, ...]:
        return tuple(
            {
                "part_id": self._flesh_id(bone_id, flesh, index),
                "address": f"skeleton/{bone_id}/flesh[{index}]/stations",
                "stations": station_result,
            }
            for bone_id in self.order
            for index, flesh in enumerate(self.bones[bone_id]["flesh"])
            if flesh.get("kind") == "gencyl" and "stations" in flesh
            for station_result in (_gencyl_stations(flesh),)
            if isinstance(station_result, tuple)
        )

    def _relations_receipt(self) -> list[dict]:
        receipt = self._relational_receipt
        by_id = {
            declaration.relation_id: declaration
            for declaration in receipt.declarations
        }
        return [
            {
                "id": measurement.relation_id,
                "authored_kind": declaration.authored_kind if declaration else None,
                "canonical_kind": str(declaration.kind) if declaration else None,
                "subject": declaration.subject_selector if declaration else None,
                "reference": declaration.reference_selector if declaration else None,
                "reference_b": declaration.reference_b_selector if declaration else None,
                "owner": section.owner_address,
                "convention": (
                    str(measurement.convention)
                    if measurement.convention is not None
                    else None
                ),
                "stage": str(section.stage),
                "depth": section.depth,
                "required": _r(measurement.required),
                "observed": _r(measurement.observed),
                "normalized_residual": _r(measurement.normalized_residual),
                "satisfied": measurement.satisfied,
                "variable_count": section.variable_count,
                "rank": section.rank,
                "active_sites": [list(sites) for sites in section.active_site_history],
            }
            for section in receipt.sections
            for measurement in section.measurements
            for declaration in (by_id.get(measurement.relation_id),)
        ]

    def _web_receipt_rows(self) -> tuple[dict, ...]:
        return tuple(
            {
                "id": web_id,
                "address": f"webs[{index}]",
                "anchors": [f"skeleton/{bone_id}" for bone_id in anchor_bones],
            }
            for index, (web_id, anchor_bones) in enumerate(self._web_anchors)
        )

    def _loft_section_receipt_rows(self) -> tuple[dict, ...]:
        return tuple(
            {
                "part_id": self._flesh_id(bone_id, flesh, index),
                "address": f"skeleton/{bone_id}/flesh[{index}]/sections",
                "sections": tuple(
                    {
                        "station": section.station,
                        "width": section.width,
                        "depth": section.depth,
                        "exponent": section.exponent,
                        "roll": section.roll,
                    }
                    for section in section_result
                ),
            }
            for bone_id in self.order
            for index, flesh in enumerate(self.bones[bone_id]["flesh"])
            if flesh.get("kind") == "loft"
            for section_result in (_loft_sections(flesh),)
            if not isinstance(section_result, LoftSectionObstruction)
        )

    def build_receipt(
        self,
        anatomy: dict[str, object] | None = None,
        *,
        parts: list[dict] | None = None,
        provenance: dict[str, str] | None = None,
    ) -> dict:
        emitted_parts = self.parts if parts is None else parts
        emitted_provenance = self.provenance if provenance is None else provenance
        station_rows = self._gencyl_station_receipt_rows()
        loft_rows = self._loft_section_receipt_rows()
        web_rows = self._web_receipt_rows()
        roles = self._declared_roles()
        solved = {
            "goals": self.solved_goals,
            "ground_lift": _r(self.ground_lift),
            "limit_violations": self.limit_violations,
            "clamps": self.clamps,
        }
        if self._relational_receipt is not None:
            solved["relations"] = self._relations_receipt()
        receipt = {
            "name": self.spec.get("name", "?"),
            "target": self.target,
            "landmarks": self.landmarks,
            "solved": solved,
            "emitted": [p["id"] for p in emitted_parts],
            "provenance": dict(emitted_provenance),
            "violations": self.violations,
            "provenance_addresses": sorted(set(emitted_provenance.values())),
            **(
                {"gencyl_stations": station_rows}
                if station_rows
                else {}
            ),
            **({"loft_sections": loft_rows} if loft_rows else {}),
            **({"webs": web_rows} if web_rows else {}),
            **({"roles": roles} if roles else {}),
        }
        return (
            {**receipt, "anatomy": anatomy}
            if anatomy is not None
            else receipt
        )
