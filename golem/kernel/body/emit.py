"""Compiler phase: emit world props, body flesh, and mounted hand sub-grammars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from golem.paths import PILOTS as _PILOTS
from golem.kernel import engine as ev  # quat helpers, box/rot-aware field

import hand as _hand  # noqa: E402  (frozen hand sub-grammar; compile_graph)

from golem.kernel.body.anatomy_bridge import _tissue_envelopes_by_address
from golem.kernel.body.canonical import _r, _rvec
from golem.kernel.body.contacts import _is_identity_quat, _mount_pose
from golem.kernel.body.gencyl import _gencyl_stations, _loft_sections
from golem.kernel.body.geometry import placed_center
from golem.kernel.body.linalg import _angle_between, _normalize, mat_to_quat
from golem.kernel.body.types import (
    FLESH_ROLE_NON_CARRIER,
    LoftSectionObstruction,
    _GencylStationObstruction,
)

if TYPE_CHECKING:
    from golem.kernel.anatomy import AnatomyResult, TissueEnvelopeSection


class _EmitMixin:
    # -- step 6: place world props ---------------------------------------- #
    def place_props(self) -> None:
        for pi, prop in enumerate(self.spec.get("props", [])):
            anchor = prop.get("anchor", {})
            kind = anchor.get("kind", "world_ground")
            if kind == "world_ground":
                xz = anchor.get("xz", [0.0, 0.0])
                axis = _normalize(anchor.get("axis", [0, 1, 0]))
                # local +y along axis; tip (local origin) at (xz, ground_y).
                y_ax = axis
                # deterministic frame around y_ax
                ref = np.array([0.0, 0.0, 1.0]) if abs(y_ax[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
                x_ax = _normalize(np.cross(ref, y_ax)) if abs(y_ax[1]) <= 0.9 else np.array([1.0, 0.0, 0.0])
                if abs(y_ax[1]) > 0.9:
                    R = np.eye(3)  # axis == +y: identity local frame
                else:
                    z_ax = np.cross(x_ax, y_ax)
                    R = np.stack([x_ax, y_ax, z_ax], axis=1)
                tip = np.array([xz[0], self.ground_y, xz[1]], dtype=np.float64)
                self._emit_prop(prop, pi, R, tip)
            elif kind == "port":
                # rides the FK chain: transform by the named port world frame.
                port_ref = anchor.get("port")
                pf = self._named_port_frame(port_ref)
                if pf is None:
                    self._viol("bad_prop_port", f"props[{pi}]",
                               f"port {port_ref!r} not found")
                    continue
                self._emit_prop(prop, pi, pf["R"], pf["origin"])
            else:
                self._viol("bad_anchor", f"props[{pi}]/anchor",
                           f"unknown anchor kind {kind!r}")

    def _emit_prop(self, prop: dict, pi: int, R: np.ndarray, t: np.ndarray) -> None:
        propid = prop["id"]
        for part in prop.get("parts", []):
            world = self._transform_local_part(part, R, t, scale=1.0)
            world["id"] = f"{propid}/{part['id']}"
            self.parts.append(world)
            self.provenance[world["id"]] = f"props[{pi}]/parts/{part['id']}"
        for lm, local in prop.get("landmarks", {}).items():
            wp = R @ np.array(local, dtype=np.float64) + t
            self.landmarks[f"{propid}/{lm}"] = _rvec(wp)

    def _transform_local_part(self, part: dict, R: np.ndarray, t: np.ndarray,
                              scale: float) -> dict:
        """Transform a local-frame part (prop / mounted hand) into world.
        Reproduces ``mount_hands.transform_parts`` semantics: world = R @
        (local * scale) + t for centers; spine points likewise."""
        q: dict = {"id": part["id"], "type": part["type"]}
        if part["type"] == "gencyl":
            spine = np.array(part["spine"], dtype=np.float64) * scale
            q["spine"] = [_rvec(R @ p + t) for p in spine]
            q["radii"] = [_r(r * scale) for r in part["radii"]]
        else:  # blob / box
            c = np.array(part["center"], dtype=np.float64) * scale
            q["center"] = _rvec(R @ c + t)
            q["size"] = [_r(s * scale) for s in part["size"]]
            if part["type"] == "box":
                q["round"] = _r(part.get("round", 0.0) * scale)
            if "rot" in part:
                Rp = R @ ev.quat_to_matrix(ev.read_quat(part["rot"]))
                q["rot"] = _rvec(mat_to_quat(Rp)) if self.target == "v03" else None
                if self.target != "v03":
                    del q["rot"]
                    self._viol("blob_rot_dropped", part["id"],
                               "oriented prop part demoted axis-aligned (v02)")
        if "blend" in part:
            q["blend"] = _r(part["blend"])
        if "operator" in part:
            q["operator"] = part["operator"]
        return q

    # -- step 8: emit body flesh + mounts --------------------------------- #
    def emit_parts(self, anatomy: "AnatomyResult | None" = None) -> None:
        tissue_envelopes = _tissue_envelopes_by_address(anatomy)
        for bid in self.order:
            rec = self.bones[bid]
            for i, fl in enumerate(rec["flesh"]):
                self._emit_flesh(
                    rec,
                    fl,
                    i,
                    tissue_envelope=tissue_envelopes.get(
                        f"skeleton/{bid}/flesh[{i}]"
                    ),
                )
            # v0.4 arrays: ONE authored declaration expands to n placements
            # along the bone (blob/box templates only); provenance points at
            # the declaration, so a later edit to n/along/template is a
            # one-field edit. Absent means absent (byte-identical goldens).
            for ai, arr in enumerate(rec.get("arrays", ())):
                n = int(arr.get("n", 0))
                t0, t1 = (arr.get("along") or [0.0, 1.0])
                for k in range(n):
                    t = t0 if n == 1 else t0 + (t1 - t0) * k / (n - 1)
                    fl = {**arr["template"],
                          "name": f"{arr['name']}.{k}", "t": _r(float(t))}
                    self._emit_flesh(
                        rec, fl, i,
                        addr=f"skeleton/{bid}/arrays[{ai}]")
        for mi, mount in enumerate(self.spec.get("mounts", [])):
            self._emit_mount(mount, mi)

    def _emit_flesh(
        self,
        rec: dict,
        fl: dict,
        i: int,
        addr: str | None = None,
        tissue_envelope: "TissueEnvelopeSection | None" = None,
    ) -> None:
        bid = rec["id"]
        pid = self._flesh_id(bid, fl, i)
        addr = addr or f"skeleton/{bid}/flesh[{i}]"
        head, R, L = rec["head"], rec["R"], rec["length"]
        mirrored = rec["mirrored"]
        kind = fl["kind"]

        if kind == "gencyl":
            part = self._flesh_gencyl_part(fl, pid, head, R, L, tissue_envelope)
        elif kind == "loft":
            part = self._flesh_loft_part(fl, pid, head, R, L, tissue_envelope)
        elif kind == "blob":
            part = self._flesh_blob_part(fl, pid, head, R, L)
        elif kind == "box":
            part = self._flesh_box_part(fl, pid, bid, head, R, L)
        else:
            self._viol("bad_flesh_kind", pid, f"unknown flesh kind {kind!r}")
            return
        if part is None:
            return

        if mirrored:
            part["mirror"] = True
        if "blend" in fl:
            part["blend"] = _r(fl["blend"])
        if "operator" in fl:
            part["operator"] = fl["operator"]
        self.parts.append(part)
        if fl.get("role") == FLESH_ROLE_NON_CARRIER:
            # Membrane role (R3): emitted geometry, but no carrier
            # provenance -- a route bone left with only non-carrier flesh is
            # honestly rejected (MissingProvenance), never inflated.
            part["role"] = FLESH_ROLE_NON_CARRIER
        else:
            self.provenance[pid] = addr

    def _flesh_gencyl_part(
        self, fl, pid, head, R, L,
        tissue_envelope: "TissueEnvelopeSection | None",
    ) -> dict | None:
        span_start, span_end = tuple(
            map(float, fl.get("span", (0.0, 1.0)))
        )
        uses_tissue_envelope = (
            self.target == "v03" and tissue_envelope is not None
        )
        major_axis = (
            max(
                station.half_size[0]
                for station in tissue_envelope.stations
            )
            >= max(
                station.half_size[1]
                for station in tissue_envelope.stations
            )
            if uses_tissue_envelope
            else True
        )
        major_index, minor_index = (
            (0, 1) if major_axis else (1, 0)
        )
        radii = (
            tuple(
                station.half_size[major_index]
                for station in tissue_envelope.stations
            )
            if uses_tissue_envelope
            else fl["radii"]
        )
        station_result = (
            tuple(
                span_start
                + station.parameter * (span_end - span_start)
                for station in tissue_envelope.stations
            )
            if uses_tissue_envelope
            else _gencyl_stations(fl)
        )
        if isinstance(station_result, _GencylStationObstruction):
            return None
        samples = station_result
        spine = [head + R @ np.array([0, 0, s * L]) for s in samples]
        part = {"id": pid, "type": "gencyl",
                "spine": [_rvec(p) for p in spine],
                "radii": [_r(r) for r in radii]}
        if uses_tissue_envelope:
            profile_up = (
                R[:, 1]
                if major_index == 0
                else -R[:, 0]
            )
            part["profile"] = {
                "n": tuple(2.0 for _station in tissue_envelope.stations),
                "aspect": tuple(
                    _r(
                        station.half_size[minor_index]
                        / station.half_size[major_index]
                    )
                    for station in tissue_envelope.stations
                ),
                "offset": tuple(
                    (
                        _r(station.offset[major_index]),
                        _r(station.offset[minor_index]),
                    )
                    for station in tissue_envelope.stations
                ),
                "up": _rvec(profile_up),
            }
        elif "profile" in fl:
            part["profile"] = fl["profile"]
        return part

    def _flesh_loft_part(
        self, fl, pid, head, R, L,
        tissue_envelope: "TissueEnvelopeSection | None",
    ) -> dict | None:
        section_result = _loft_sections(fl)
        if isinstance(section_result, LoftSectionObstruction):
            return None
        sections = section_result
        authored_stations = np.asarray(
            tuple(section.station for section in sections), dtype=np.float64
        )
        uses_tissue_envelope = (
            self.target == "v03" and tissue_envelope is not None
        )
        stations = (
            authored_stations[0]
            + np.asarray(
                tuple(
                    station.parameter for station in tissue_envelope.stations
                ),
                dtype=np.float64,
            )
            * (authored_stations[-1] - authored_stations[0])
            if uses_tissue_envelope
            else authored_stations
        )

        def interpolate(attribute: str) -> np.ndarray:
            return np.interp(
                stations,
                authored_stations,
                np.asarray(
                    tuple(getattr(section, attribute) for section in sections),
                    dtype=np.float64,
                ),
            )

        authored_widths = interpolate("width")
        authored_depths = interpolate("depth")
        widths = (
            np.maximum(
                authored_widths,
                np.asarray(
                    tuple(
                        station.half_size[0]
                        for station in tissue_envelope.stations
                    ),
                    dtype=np.float64,
                ),
            )
            if uses_tissue_envelope
            else authored_widths
        )
        depths = (
            np.maximum(
                authored_depths,
                np.asarray(
                    tuple(
                        station.half_size[1]
                        for station in tissue_envelope.stations
                    ),
                    dtype=np.float64,
                ),
            )
            if uses_tissue_envelope
            else authored_depths
        )
        spine = tuple(
            head + R @ np.asarray((0.0, 0.0, station * L))
            for station in stations
        )
        profile = {
            "n": [_r(value) for value in interpolate("exponent")],
            "depth": [_r(value) for value in depths],
            "roll": [_r(value) for value in interpolate("roll")],
            "up": _rvec(R[:, 1]),
            **(
                {
                    "offset": [
                        [_r(value) for value in station.offset]
                        for station in tissue_envelope.stations
                    ]
                }
                if uses_tissue_envelope
                else {}
            ),
        }
        return {
            "id": pid,
            "type": "gencyl",
            "spine": [_rvec(point) for point in spine],
            "radii": [_r(value) for value in widths],
            "profile": profile,
        }

    def _flesh_blob_part(self, fl, pid, head, R, L) -> dict:
        c = placed_center(fl, head, R, L)
        part = {"id": pid, "type": "blob", "center": _rvec(c),
                "size": [_r(s) for s in fl["size"]]}
        if self.target == "v03":
            q = mat_to_quat(R)
            if not _is_identity_quat(q):
                part["rot"] = _rvec(q)
        else:
            # v02: drop orientation WITH a quantified violation.
            q = mat_to_quat(R)
            if not _is_identity_quat(q):
                ang = _angle_between(R[:, 2], np.array([0, 0, 1]))
                self._viol("blob_rot_dropped", pid,
                           f"blob orientation ({ang:.1f} deg) dropped for "
                           f"v02 target", degrees=_r(ang))
        return part

    def _flesh_box_part(self, fl, pid, bid, head, R, L) -> dict | None:
        if self.target != "v03":
            self._viol("box_in_v02", pid,
                       f"box flesh {pid!r} on bone {bid!r} illegal for "
                       f"v02 emit_target")
            return None
        c = placed_center(fl, head, R, L)
        part = {"id": pid, "type": "box", "center": _rvec(c),
                "size": [_r(s) for s in fl["size"]],
                "round": _r(fl.get("round", 0.0))}
        q = mat_to_quat(R)
        if not _is_identity_quat(q):
            part["rot"] = _rvec(q)
        return part

    def _emit_mount(self, mount: dict, mi: int) -> None:
        pf = self._named_port_frame(mount["port"])
        if pf is None:
            self._viol("bad_mount_port", f"mounts[{mi}]",
                       f"port {mount['port']!r} not found")
            return
        hand_spec = None
        for cand in (self.spec_dir / mount["spec"], _PILOTS / mount["spec"]):
            if Path(cand).exists():
                hand_spec = json.loads(Path(cand).read_text())
                break
        if hand_spec is None:
            self._viol("bad_mount_spec", f"mounts[{mi}]",
                       f"hand spec {mount['spec']!r} not found in spec dir or pilots")
            return
        pose = mount["pose"]
        if isinstance(pose, str):
            pose = _hand.POSES[pose]
        else:
            pose = _mount_pose(pose)
        scale = float(mount.get("scale", 1.0))
        gap = float(mount.get("gap", 0.02))
        blend = float(mount.get("blend", self.blend))
        hand_graph = _hand.compile_graph(hand_spec, pose, blend=blend, wrist_stub=False)

        R = pf["R"]
        z_port = R[:, 2]
        palm_len = hand_spec["palm"]["length"]
        # junction (proximal wrist) sits at the port; palm center is +z of it.
        palm_center = pf["origin"] + z_port * (palm_len / 2.0 * scale + gap)
        mirrored = self.bones[self._port_bone(mount["port"])]["mirrored"]
        for part in hand_graph["parts"]:
            world = self._transform_local_part(part, R, palm_center, scale)
            world["id"] = f"hand_{part['id']}"
            if mirrored:
                world["mirror"] = True
            self.parts.append(world)
            self.provenance[world["id"]] = f"mounts[{mi}]/hand_{part['id']}"
