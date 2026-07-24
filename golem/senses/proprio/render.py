"""Effect: pushed receipt, pull-only section tables, opt-in ASCII silhouette."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import assert_never

import numpy as np

from golem.kernel import engine as ev
from golem.kernel.engine.types import SurfaceFormationObstruction
from golem.senses.model import FusionBand, NoSupport, Senses, Supported
from golem.senses.proprio.anomaly import (
    _ANOM_TAG,
    _ANOMALY_CAP,
    _ASSERT_CAP,
    _apply_suppression,
)
from golem.senses.proprio.format import f2, f3, s2, s3
from golem.senses.proprio.posture import GROUND_Y
from golem.senses.raster import CanonicalView


def render_receipt(
    senses: Senses, assert_results: list, anomalies: list, *, pack: str
) -> str:
    """The PUSHED receipt (fence a): header + GLOBAL + assert failures +
    anomalies. No tables. No wall-clock (determinism)."""
    g = senses.global_dims
    composition_overrides = sum(
        1
        for part in senses.graph.get("parts", ())
        if isinstance(part, Mapping)
        and ("blend" in part or "operator" in part)
    )
    blend_scale = f"k={f3(g.k)}" + (
        f" (global; {composition_overrides} part override(s))"
        if composition_overrides
        else ""
    )
    lines: list[str] = []
    lines.append(
        f"PROPRIO {senses.name} txn:{senses.txn} | "
        f"{senses.n_parts} parts ({senses.n_mirror} mirrored, "
        f"{senses.n_instances} instances) | no mesh"
    )
    c = g.centroid
    lines.append(
        f"GLOBAL  bbox {f2(g.W)}w x {f2(g.H)}h x {f2(g.D)}d | "
        f"H/W {f2(g.HW)} | centroid ({f2(c[0])},{f2(c[1])},{s2(c[2])}) | "
        f"{blend_scale} | predicted {senses.n_components} component(s)"
    )

    fails = [r for r in assert_results if r["status"] == "fail"]
    passes = [r for r in assert_results if r["status"] == "pass"]
    unmeas = [r for r in assert_results if r["status"] == "unmeasurable"]
    suffix, visible_anoms, n_suppressed = _apply_suppression(assert_results, anomalies)

    lines.append("")
    header = f"ASSERT  {len(fails)} FAIL / {len(passes)} pass"
    if unmeas:
        header += f" / {len(unmeas)} unmeasurable"
    header += f"  [pack:{pack}]"
    lines.append(header)
    shown = 0
    for r in fails + unmeas:
        if shown >= _ASSERT_CAP:
            break
        line = "  " + r["human"]
        if r["id"] in suffix:
            line += "  [also: " + "; ".join(suffix[r["id"]]) + "]"
        lines.append(line)
        shown += 1
    overflow = len(fails) + len(unmeas) - shown
    if overflow > 0:
        lines.append(f"  +{overflow} more -- proprio(asserts)")

    lines.append("")
    total_visible = len(visible_anoms)
    shown_anoms = visible_anoms[:_ANOMALY_CAP]
    head = f"ANOMALY {len(shown_anoms)}"
    if n_suppressed:
        head += f" ({n_suppressed} suppressed into asserts)"
    lines.append(head)
    for i, an in enumerate(shown_anoms, 1):
        lines.append(f"  N{i} {_ANOM_TAG[an['det']]} @ {an['addr']}: {an['detail']}")
    rem = total_visible - len(shown_anoms)
    if rem > 0:
        lines.append(f"  +{rem} more -- proprio(anomalies)")
    return "\n".join((*lines, *_anatomy_receipt_lines(senses.anatomy))) + "\n"


def _anatomy_receipt_lines(anatomy: object) -> tuple[str, ...]:
    if not isinstance(anatomy, Mapping):
        return ()
    if anatomy.get("status") == "rejected":
        records = tuple(
            record
            for record in anatomy.get("obstructions", ())
            if isinstance(record, Mapping)
        )
        return (
            "",
            f"ANATOMY REJECTED | {len(records)} obstruction(s)",
            *tuple(
                "  "
                + f"{record.get('kind', 'AnatomyObstruction')} "
                + f"@ {record.get('address', 'skeleton')}: "
                + str(record.get("reason", "invalid"))
                for record in records
            ),
        )
    circuits = tuple(
        circuit
        for circuit in anatomy.get("circuits", ())
        if isinstance(circuit, Mapping)
    )
    underbuilt = tuple(
        row
        for row in anatomy.get("carrier_rows", ())
        if isinstance(row, Mapping)
        and isinstance(row.get("suggested_scale"), (int, float))
        and float(row["suggested_scale"]) > 1.0 + 1.0e-9
    )
    overall = anatomy.get("overall")
    pump = overall.get("pump") if isinstance(overall, Mapping) else None
    return (
        "",
        f"CIRCULATION pump "
        f"{pump.get('host_bone_id', 'unknown') if isinstance(pump, Mapping) else 'unknown'} | "
        f"{sum(map(lambda circuit: int(bool(circuit.get('is_complete'))), circuits))}/"
        f"{len(circuits)} closed | {len(underbuilt)} suggested carrier scales | "
        f"coverage {f3(float(anatomy.get('terminal_coverage', 0.0)))} | "
        f"leverage {f3(float(anatomy.get('authoring_leverage', 0.0)))}x",
        *tuple(
            "  "
            + str(circuit.get("tissue", "tissue")).upper()
            + f" {f3(float(circuit.get('flow_fraction', 0.0)))} | "
            + "PUMP > ARTERY > ARTERIOLE > CAPILLARY > VENULE > VEIN | "
            + " > ".join(map(str, circuit.get("bones", ())))
            for circuit in circuits
        ),
        *tuple(
            "  "
            + str(row.get("controlling_constraint", "circulation_clearance")).upper()
            + " @ "
            + str(row.get("interface_address", "skeleton"))
            + ": radius "
            + f"{f3(float(row.get('measured_minimum_radius', 0.0)))} < "
            + f"{f3(float(row.get('target_minimum_radius', 0.0)))}; "
            + f"suggest scale {row.get('shape_address', 'shape')} x"
            + f"{f3(float(row.get('suggested_scale', 1.0)))}"
            for row in underbuilt[:12]
        ),
    )


def render_section(senses: Senses, section: str) -> str:
    if section == "parts":
        return _section_parts(senses)
    if section == "fusion":
        return _section_fusion(senses)
    if section == "ground":
        return _section_ground(senses)
    if section == "balance":
        return _section_balance(senses)
    if section == "bands":
        return _section_bands(senses)
    raise ValueError(f"unknown section {section!r}")


def _canonical_addr(pid: str, senses: Senses) -> str:
    return f"{pid}.L" if senses.parts[pid].mirror else pid


def _section_parts(senses: Senses) -> str:
    lines = ["PARTS (canonical side; x[lo,hi] y[lo,hi] z[lo,hi])"]
    for pid, pd in senses.parts.items():
        lo, hi = pd.bbox_lo, pd.bbox_hi
        if pd.mirror:
            addr = pd.instances[0]
            lo, hi = senses.inst_data[addr].lo, senses.inst_data[addr].hi
        name = _canonical_addr(pid, senses)
        lines.append(
            f" {name:<13}[{s2(lo[0])},{s2(hi[0])}][{s2(lo[1])},{s2(hi[1])}]"
            f"[{s2(lo[2])},{s2(hi[2])}]"
        )
    return "\n".join(lines) + "\n"


def _section_fusion(senses: Senses) -> str:
    lines = ["FUSION (FUSED gap<0; ~BLEND 0<=gap<k; mirror pairs shown once)"]
    for pid, pd in senses.parts.items():
        rows = senses.fusion_rows[pid]
        best: dict[str, tuple[float, FusionBand]] = {}
        for row in rows:
            if row.pb not in best or row.g < best[row.pb][0]:
                best[row.pb] = (row.g, row.band)
        if not best and pid not in senses.cross_plane:
            continue
        parts_txt = []
        for partner, (g, band) in sorted(best.items(), key=lambda kv: kv[1][0]):
            tag = "~" if band is FusionBand.BLEND else ""
            parts_txt.append(f"{tag}{partner} {s3(g)}")
        if pid in senses.cross_plane:
            parts_txt.append(f"{pid}.L~{pid}.R {s3(senses.cross_plane[pid])}")
        lines.append(f" {_canonical_addr(pid, senses)}: " + " | ".join(parts_txt))
    return "\n".join(lines) + "\n"


def _section_ground(senses: Senses) -> str:
    decl = " ".join(f"{pid} {s3(c)}" for pid, c in senses.ground.items())
    near = " ".join(
        f"{pid} {s3(c)} (near)" for pid, c in senses.near_ground.items()
    )
    lines = [
        f"GROUND (ground_y={f2(GROUND_Y)}; positive = floating; declared: "
        f"{', '.join(senses.ground_decl)})",
        " " + decl + ((" | " + near) if near else ""),
    ]
    return "\n".join(lines) + "\n"


def _section_balance(senses: Senses) -> str:
    match senses.balance:
        case NoSupport():
            return "BALANCE  no declared support contacts\n"
        case Supported() as b:
            c = senses.global_dims.centroid
            verdict = "INSIDE" if b.inside else "OUTSIDE"
            return (
                f"BALANCE  centroid ({f2(c[0])},{f2(c[1])},{s2(c[2])}); support hull "
                f"x[{s2(b.hull_x[0])},{s2(b.hull_x[1])}] "
                f"z[{s2(b.hull_z[0])},{s2(b.hull_z[1])}] -> {verdict}, "
                f"margin x {s3(b.margin_x)} z {s3(b.margin_z_fwd)} fwd / "
                f"{s3(b.margin_z_aft)} aft\n"
            )
        case _ as unreachable:
            assert_never(unreachable)


def _section_bands(senses: Senses) -> str:
    lines = ["BANDS (10 slabs, top-down: y[lo,hi] front-width | side-depth | z-mid)"]
    for row in reversed(senses.bands):
        lines.append(
            f" y[{f2(row.ylo)},{f2(row.yhi)}] "
            f"{f2(row.width)}|{f2(row.depth)}|{s2(row.zmid)}"
        )
    return "\n".join(lines) + "\n"


_SKETCH_COLS = 28
_SKETCH_ROWS = 14
_SKETCH_RAY = 24


@dataclass(frozen=True)
class SketchSurfaceFormationObstruction:
    view: CanonicalView
    obstructions: tuple[SurfaceFormationObstruction, ...]


type SketchRenderResult = str | SketchSurfaceFormationObstruction


def render_sketch(senses: Senses, view: str) -> SketchRenderResult:
    graph = senses.graph
    lo, hi = senses.raw_lo, senses.raw_hi
    if view == "front":
        canonical_view = CanonicalView.FRONT
        h_lo, h_hi, hi_ax = lo[0], hi[0], 0
        v_lo, v_hi, v_ax = lo[1], hi[1], 1
        r_lo, r_hi, r_ax = lo[2], hi[2], 2
        hlabel = "x"
    elif view == "side":
        canonical_view = CanonicalView.SIDE
        h_lo, h_hi, hi_ax = lo[2], hi[2], 2
        v_lo, v_hi, v_ax = lo[1], hi[1], 1
        r_lo, r_hi, r_ax = lo[0], hi[0], 0
        hlabel = "z"
    else:
        raise ValueError(f"unknown sketch view {view!r}")

    cols, rows, nray = _SKETCH_COLS, _SKETCH_ROWS, _SKETCH_RAY
    cellw = (h_hi - h_lo) / cols
    cellh = (v_hi - v_lo) / rows
    rays = np.linspace(r_lo, r_hi, nray)

    header = (
        f"{view}, {hlabel}[{s2(h_lo)},{s2(h_hi)}] y[{s2(v_lo)},{s2(v_hi)}], "
        f"cell {f3(cellw)}w x {f3(cellh)}h (NOT square)"
    )
    horizontal = h_lo + (np.arange(cols) + 0.5) * cellw
    vertical = v_lo + (rows - 1 - np.arange(rows) + 0.5) * cellh
    vertical_grid, horizontal_grid, ray_grid = np.meshgrid(
        vertical, horizontal, rays, indexing="ij"
    )

    def coordinate_grid(axis: int) -> np.ndarray:
        return (
            horizontal_grid
            if axis == hi_ax
            else vertical_grid
            if axis == v_ax
            else ray_grid
        )

    points = np.stack(tuple(map(coordinate_grid, range(3))), axis=-1).reshape((-1, 3))
    sampled = ev.sample_graph_field(graph, points)
    if isinstance(sampled, ev.RejectedSurfaceFormation):
        return SketchSurfaceFormationObstruction(
            canonical_view,
            sampled.obstructions,
        )
    field = sampled.reshape((rows, cols, nray))
    raster = np.min(field, axis=2) <= 0.0
    lines = tuple(
        map(
            lambda row: "".join(map(lambda inside: "#" if inside else ".", row)),
            raster,
        )
    )
    return "\n".join((header, *lines)) + "\n"
