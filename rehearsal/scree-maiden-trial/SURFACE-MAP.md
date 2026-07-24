# SURFACE MAP — an author's atlas of the GOLEM session surface

Charted during the scree-maiden trial (first full authoring pass on the new
session surface). Every claim receipted by a command, a verdict fragment, or a
source line. FRICTION.md holds the outrage; this file holds the cartography.

---

## 1. Teaching topology — where each needed fact actually came from

| # | Fact needed to author | Where it came from | Surface that SHOULD have taught it |
|---|---|---|---|
| 1 | Transaction envelope shape (`{base_txn, operations}` / `{document}` / bare document) | `golem/session/protocol.py::decode_request` (source) | `session --help` (names "envelope or document", shows neither) |
| 2 | Verdict JSON anatomy (status/diagnostics/margins/binding_constraint/deltas) | empirical: `session specs/vigil_hound_demo.json specs/vigil_hound_demo.json` → verdict-00 | a `--schema` flag or contract primer section |
| 3 | "accepted" tolerates 148 diagnostics; check's `ANOMALY 0` == verdict diagnostics filtered on `witness.expected` | diffing check output vs verdict-00 diagnostics | both tools (they claim to measure the same thing, report different numbers) |
| 4 | Palette keys are a CLOSED enum (`AppearanceMaterialId`, 10 ids); palette re-tints, never mints | `UnknownEyeMaterialObstruction`, then `golem/materials/core.py:25` | the obstruction (names bad value, not allowed set); the schema |
| 5 | Eyes require a GLOSSY material (only `eye_gloss` qualifies) | `NonGlossyEyeMaterialObstruction` exists in `kernel/body/eyes.py:140` | the eye schema |
| 6 | blob `size` and loft `width`/`depth` are HALF-extents (radii) | empirical probe: one blob `[0.1,0.2,0.3]` → bbox `0.20×0.40×0.60` | the dialect schema — one word ("half-extents") would do |
| 7 | `mirror: true` on a bone mirrors the whole subtree (`_m` instances, `:negative:`/`:positive:` in vascular ids) | hound attach list naming `_m` parts + probe (2 parts → 3 instances) | skeleton schema |
| 8 | `rest_dir` composes in the PARENT frame; child frame = derived (`z=parent_R@dir`, y from parent-y projection, `x=y×z`) | `kernel/body/geometry.py:196-199` + `linalg.py:47-64`, after 3 cross_plane_fusion anomalies | nothing on any surface; it is the single most load-bearing law of multi-bone authoring |
| 9 | Carrier clearance radius = min over loft sections of min(width,depth), vs `max(0.04, carrier_radius_scale×flow)` | hound margins (verdict-00) read against hound section values | circulation receipts |
| 10 | Anomaly law: overlap undeclared → `unintended_fusion`; `0 ≤ gap < min(blend_r)` undeclared → `blend_ambiguity`; declared+overlap → `deep_burial expected`; declared+gap≥reach → `missing_fusion`; same-part L~R overlap → `cross_plane_fusion` (midline-declarable) | `senses/proprio/anomaly.py:143-275` + verdict-01/02 witnesses | the contract primer |
| 11 | Same-bone and parent-child-bone flesh pairs are auto-`expected` (graph distance ≤1); 2+ links needs authored attach | verdict-01: 22 `expected:true` with attach=[] | the attach channel documentation |
| 12 | CapsuleEscape is measured against AUTHORED flesh (auto-scale edits cannot absorb it) | experiment: tissue_envelope floor fired `scale skull ×1.167`, escape bit-identical | the CIRCULATION "edits" line |
| 13 | Vascular corridors route through REGION flesh, can leave the bone corridor, and need "aerial" cover | node-position extraction (escape probe): cranial path at y≈0.83 over a bone at y≈0.62 | nothing; discovered by six null experiments + source bypass |
| 14 | Escape margin law: `margin = −max(union_SDF(samples)) − radius`; chamfer corners round analytically | `kernel/anatomy/geometry.py:150-181` | the obstruction (edge_id+margin only) |
| 15 | Exchange-bed cover set = own route/region flesh; thorax flesh cannot cover a leg bed | plastron on thorax: escape bit-identical across 2 positions | nothing |
| 16 | Exact leg aiming: `local = parent_Rᵀ @ desired_world` | own frame probe (`fk_table` dump of every leg bone frame) | nothing — the kernel has no introspection for authored-vs-actual world dirs |
| 17 | look renders use FIXED catalogue recipes; `appearance_palette` is export-path only | `senses/orthographic.py:100-141` + `assembly/compile.py:26-47` + two A/B renders | the palette schema (implies visible effect) |
| 18 | CLI session is stateless; delta discipline (base rotation) is the author's job | process miss (entry 007) | the CLI (could print "base_txn: 0 (fresh session)" or accept a journal file) |
| 19 | `look` refuses to render until full acceptance; no bypass flag | `render refused -> ... (typed rejection above)` | `look --help` |

**Teaching-topology summary:** of 19 required facts, 2 came from verdicts
directly, 1 from an error message, 5 from empirical probes, 8 from kernel
source, 3 from exemplar imitation. The authored surface (help text, schema,
contract primer, verdicts) taught roughly 15% of what the creature required.

## 2. Margin-to-knob map

| Reported margin (constraint) | Editable knob it maps to | Fidelity |
|---|---|---|
| `anatomy:carrier:<bone>` → `skeleton/<bone>/flesh[0]` | min section min-dim of that bone's FIRST flesh vs `max(0.04, scale×flow)` | GOOD — address names the exact flesh; hound shows `scale ×N` suggestions |
| `vascular:capsule_clearance` (accepted) | the union-SDF minimum over all bed capsules — nearest approach of any capsule to any surface | ORPHANED as steering: one scalar for ~1300 edges; no address, no edge id, no part |
| `vascular:delivery` / `solver_residual` / `boundary_balance` / `symmetry` | solver internals; author knob = demands + carrier scale only | informational |
| `assembly:integrity:*` | whole-spec properties (single component, watertight, vocabulary) | informational |
| **REJECTED vascular** (CapsuleEscape/VascularIntersection) | **NO margin emitted at all** — only `edge_id` + deficit in diagnostics | ORPHAN — see door catalogue D1 |
| binding_constraint | min normalized margin among EMITTED margins | MISLEADING when rejection channels emit no margins (named `anatomy:carrier:fore_shin` margin 0 while capsule escapes rejected; entry 004) |

Orphan margin #1: `binding_constraint` during vascular rejection — names a
wall you are NOT nearest, because the rejecting channel emits no margins.
Orphan margin #2: `vascular:capsule_clearance` at acceptance (mine: norm
0.0002) — the true thinnest margin, reported with no location. To act on it
you must re-derive it via internals (`_validate_vascular_geometry`).

## 3. Session machinery vs whole-file edits — what actually accelerated

| Step | Tool used | Verdict |
|---|---|---|
| Whole-creature first draft | whole-file write | only option — session ops are per-record |
| Material enum fix | whole-file edit + `check` | fast, obvious |
| Geometry semantics (half-extents) | probe spec + `check` | probe was the teacher |
| Leg re-aim (2 rounds) | whole-file edit + `check` + frame probe | session deltas USELESS here (changed 8 addresses at once; deltas listed addresses, not geometry) |
| Attach declarations (35 pairs) | verdict-02/03 diagnostic dump → one whole-file edit | verdict was the teacher (full anomaly list w/ gaps) |
| Cranial + foot escapes | escape probe (internals) + whole-file edit | session verdict gave edge_id+margin only; the probe gave coordinates |
| Delta chain (verdict-01→06) | `session base.json spec` per edit round | GENUINELY USEFUL twice: (a) v01→v02 confirmed leg widening moved carrier margins +0.02 and newly_inactive freed two constraints; (b) v03 diagnostics listed the exact attach candidates with gaps. Useless for multi-knob geometry rounds (changed_addresses ≠ world positions) |

Where addressed ops would have won: the attach pass (35 `add` ops, one
journal entry, invertible). I used whole-file edit instead because the CLI
session is stateless — ops would have to be replayed against base.json each
run anyway, so the envelope buys nothing over a document transaction at CLI
level. The REPL (`golem.session.repl`) is the stateful surface; the CLI
protocol gives verdicts, not sessions.

## 4. Hidden couplings — the dependency sketch no surface admits

```
anatomy.circulation.carrier_radius_scale ─┐
anatomy.circulation.distance_decay ───────┤→ per-bone required carrier radius
skeleton.<bone>.flesh[0] sections ────────┘  (flesh[0] ONLY; later flesh exempt)

skeleton.<bone>.rest_dir ──composes-in──> parent frame ──> child world dir,
   child frame axes, loft width/depth orientation, blob size axes, eye offsets
   (one girdle edit silently re-aims the entire leg + re-orients every loft)

anatomy.regions[].host_bone_id ──> exchange-bed flesh cover set
   (bed capsules must hide inside OWN region/route flesh; cover from other
    bones' flesh — even the pump's — does not count)

skeleton.mirror:true ──> _m parts, :negative:/:positive: vascular instances,
   L~R pairs eligible for cross_plane_fusion (clearable only via
   contract.midline or geometry)

contract.attach ──> anomaly suppression AND missing_fusion obligation
   (a declaration is a promise of contact: declare a pair the geometry
    separates and it flips to missing_fusion — declarations must be written
    AFTER the geometry they describe, entry 005)

senses/proprio gap law ──> blend window = min(blend_r of pair):
   gap<0 undeclared→fusion, 0≤gap<k undeclared→ambiguity, gap≥k→clean void

appearance_palette ──> GLB/mesh vertex colors ONLY
senses/orthographic  ──> fixed 10-recipe catalogue (palette inert)
   (the same spec has two different colors on two surfaces)
```

## 5. Door catalogue — every wall, the door, and what the verdict should have said

**D1. CapsuleEscapeObstruction (cranial, fore_talons, hind_talons — 3 separate sieges)**
Wall: `REJECTED [CapsuleEscapeObstruction] {"edge_id":"exchange:...s:26>...r:28","margin":-0.0072}`.
Door found: bypass the surface — realize the graph via
`kernel.anatomy.realize.allocation` internals, print the failing segment's
endpoints, add flesh cover at THOSE coordinates (poll plate over the skull-head
notch; exact leg re-aim; talon enlarged past its chamfer corner).
Verdict SHOULD say: `CapsuleEscape @ segment (−0.055,0.832,0.470)→(−0.055,0.832,0.589), 0.009 outside nearest surface (hump_plate front edge / poll_crest rear edge); nearest parts: poll_crest (0.014), hump_plate (0.031); suggested: extend poll_crest −z by ≥0.012 or add cover on skeleton/skull`.
Structural note: `RejectedVasculature` carries ONLY obstructions — the graph is
discarded at rejection, so the door is unreachable from the verdict by
construction. Keep the geometry in the rejection.

**D2. cross_plane_fusion ×3 (legs across the midline)**
Wall: `cross_plane_fusion @ fore_knee.L~fore_knee.R gap −0.017` with knees
authored at ±0.17 — geometrically "impossible" as authored.
Door: frame law (geometry.py:196) — child dirs compose in parent frames; my
girdle's 51° x/z mix swung every child dir inward. Pure-lateral girdle +
`local = parent_Rᵀ @ desired_world` (exact locals from an fk_table probe).
Verdict SHOULD say: `fore_knee instances overlap across x=0; fore_shin world dir (−0.39,−0.40,0.83) differs from authored rest_dir (0.39,−0.92,0) — rest_dir composes in the parent frame; see fore_girdle frame (z=+x)`. A composed-vs-authored dir diff on every bone would have cost the kernel one line and saved me three walls.

**D3. Closed material vocabulary**
Wall: `unknown eye appearance material 'eye_amber'` (one obstruction at a time).
Door: palette re-tints of the 10 enum ids.
Obstruction SHOULD say: `unknown material 'eye_amber'; allowed: obsidian_warden, obsidian_deep, steel_violet, emissive_seam, eye_gloss (glossy; required for eyes), vascular_*, gambeson_dark, neutral_gray; 5 unknown ids present in this document`.

**D4. Half-extent geometry semantics**
Wall: every flesh twice authored size; bbox 0.98w vs intended 0.40w.
Door: probe blob (size[0.1,0.2,0.3] → bbox 0.2×0.4×0.6).
Schema SHOULD say: `size: half-extents [hx,hy,hz]` — one word.

**D5. Rejected-render refusal**
Wall: `look` on a vascular-rejected spec: `render refused`, exit 1. No visual
debugging of a failing creature precisely when you need to see it.
Door: reach acceptance first, render after.
`look --help` SHOULD admit the gate; a `--force` (render accepted geometry,
print rejection banner) would make escape-hunting visual.

**D6. Tint invisibility**
Wall: palette re-tints render as slot defaults (black stripe at two values;
violet emissive).
Door: none on the look surface — the 10 catalogue recipes are the whole
vocabulary (orthographic.py:100). Palette honored on export path only.
Renderer SHOULD either honor `record.surface_color` or stamp `palette ignored
in look renders` once per invocation.

**D7. binding_constraint during rejection**
Wall: binding = `anatomy:carrier:fore_shin` (margin 0) while 5 capsule escapes
reject the spec.
Door: read the raw diagnostics array; ignore binding during rejection.
Verdict SHOULD mark rejection-channel diagnostics as first-class margins
(negative by construction) so binding can name the true wall.

**D8. noise floor**
Wall: 148 diagnostics on an ACCEPTED exemplar; `check` says ANOMALY 0.
Door: filter on `witness.expected` / severity — discovered by diffing.
Verdict SHOULD carry a summary line: `diagnostics: 148 (0 unexpected)`; check
and session should agree on what "ANOMALY" counts.

---

## Appendix — probes built during the trial (kept as /tmp, reproducible)

- `/tmp/probe_blob.json`, `/tmp/probe_loft.json` — semantics probes (size,
  width, mirror).
- `/tmp/escape_probe.py` — realizes the vascular graph from a spec and prints
  every rejecting edge's endpoints, radius, margin. THE door-opener.
- `/tmp/frame_probe.py` — dumps every bone's actual world head/tail/dir +
  frame axes via `fk_table`. Leg-aiming oracle.
- `/tmp/part_probe.py` — prints compiled part centers/sizes from the body
  graph (where flesh ACTUALLY landed).
