# Bulwark dry-run bake-off — NON-EVIDENTIARY REHEARSAL

Date: 2026-07-12. Operator: one AI session (contaminated: read the protocol,
manifest, and both arms' work; no fresh sessions, no blinding, no independent
judges, no custodian). This document licenses **no** Phase -1 conclusion. It is
a rehearsal of tooling and a cheap directional signal on the calibration
creature `bulwark` only. The Phase -1 experiment status remains `not_run`.

## Setup

Both arms attempted the same brief ("broadest, lowest Obsidian Warden") and the
three sealed bulwark revisions. To isolate revision economics from design
taste, the Blender arm **inherited GOLEM's part layout** as named procedural
control objects (30 parts, 60 objects with mirrors), unioned by voxel remesh
(0.028) + smooth, exported to GLB per stage. Both arms' outputs were judged by
the identical sandbox pipeline (trimesh occupancy/watertight/symmetry + the
pilots' renderer as a neutral presentation). GOLEM ran the frozen pilots
unmodified in a pinned-ish sandbox env (numpy 2.2.6, not the locked 2.5.1).
Blender was the user's real 5.1.1 via MCP.

## Wall-clock per stage (operator round-trips included)

| stage | GOLEM | Blender | ratio G/B | budget |
|---|---|---|---|---|
| initial | 147 s | 185 s | 0.79 | 1800 s |
| r1 shoulders +20 % / +15 % | 20 s | 102 s | 0.196 | 720 s |
| r2 forearms −12 %, replant | 13 s | 45 s | 0.289 | 720 s |
| r3 face −18 % / +10 % fwd | 2 s | 51 s | 0.039 | 720 s |

Revision ratios: median **0.196** (gate ≤ 0.50), nearest-rank p75 **0.289**
(gate ≤ 0.67), n = 3 — GOLEM clears the revision-economy gates by a wide
margin *in this rehearsal*. Hardware differs between arms (sandbox vs. Mac);
times are dominated by operator iteration, not compute.

## Targets and preserves

Both arms landed every target **exactly** (span +20.00 %, mass +15.00 %,
forearm ratio 0.880 with contact error 0.0000, width −18.00 %, projection
+10.00 %). GOLEM's preserved scopes were byte-identical when untouched.
Blender's preserves were verified only from control values and per-object
world bboxes — scoped *mesh* verification on the unioned artifact was not
implemented, which is precisely the verifier-sidecar work the real protocol
demands. GOLEM targets were likewise verified from source parameters, which
the protocol forbids as paired evidence.

## Integrity

All eight artifacts (4 stages × 2 arms): 1 solid component, 1 shell,
watertight. Face counts ~92–166 k.

**Metric defect found:** raw bilateral symmetry IoU measured 0.976–0.979
(GOLEM, res 130) and 0.958–0.970 (Blender) despite mirror-symmetry by
construction in both arms. The voxel grid is not aligned to the mirror plane,
so the ≥ 0.98 family-integrity gate as currently implemented would spuriously
fail both arms — the same disease as the preregistered 0.9799926 obstruction.
The measurement, not the geometry, is the problem. Fix (grid aligned/centered
on x = 0, or mesh-based symmetric-distance metric) should be a preregistered
amendment before any real run.

## Failure rounds (honest count)

- GOLEM: none. 2 render-feedback rounds on the initial, ≤ 1 per revision.
- Blender: 1 failed revision round (r1: object transforms had been folded into
  mesh data at build; `location/scale` edits silently no-oped or scaled about
  the world origin — caught by in-scene verification, cost a rebuild), plus
  1 stale-depsgraph verification round (r3). Both are the tool-surface traps
  the experiment exists to price.

## Aesthetics (unblinded, worthless as evidence)

GOLEM's smooth-min blending reads as a carved monolith (family look); the
Blender voxel-union + smooth keeps a "fused balloon animal" quality and the
fingers stay visually separate. See `side_by_side.png`. A real verdict needs
the three blinded technical artists.

### Post-hoc polish round: metaballs (145 s, excluded from ratios)

The user judged the voxel-union final "night and day" uglier than GOLEM's.
Because a crippled baseline cannot be allowed to donate a victory, the Blender
arm received one post-hoc aesthetics round: the post-r3 part state rebuilt as
**metaballs** (218 elements, stiffness 3.0, calibrated radius factors 1/0.643
ball, 1/0.635 ellipsoid), converted and exported (`bulwark_r3_metaball.glb`).
Result: 1 component, watertight, symmetry IoU 0.9763 (better than voxel
remesh — the polygonizer grid is more symmetric), and a visibly organic,
family-plausible surface. See row 3 of `side_by_side.png`.

Consequences, honestly priced:

- A strong Blender arm should use metaballs (or equivalent implicit surfacing)
  from stage 0, not as an afterthought. Round-1 surfacing was a weak-baseline
  artifact; the real run's calibration gate exists to catch exactly this.
- The metaball field **inflates** composite surfaces beyond the calibrated
  single-element factor: bounds grew ~1.7 % in x, and the lowest point dipped
  to y = −0.006, i.e. ground-contact error 0.026 vs the ≤ 0.02 family clause —
  a marginal integrity fail. Exact percentage targets verified on the
  polygonal stages do NOT automatically transfer to the metaball resurface;
  a consistent metaball-from-stage-0 arm would have to re-tune per stage.
  This is the substantive contrast with GOLEM, where the authored parameters
  and the measured surface cannot drift apart.

## What this rehearsal does and does not say

Does: the frozen pilots run end-to-end; the revision workflow is real; the
budgets are generous; the protocol's mechanics (stage causality, per-stage
artifacts, integrity checks) are executable; GOLEM's symbolic revisions are
5–25× cheaper than scripted-Blender revisions for parameter-expressible edits.

Does not: qualify the Blender baseline (calibration judging not done), test
held-out generalization, test independent Blender design from brief, produce
any admissible evidence. The Blender arm here was a scripting agent driving
procedural controls — a strong human artist in the GUI is a different baseline.
No revision in this family forced a non-symbolic edit, so the
`NonSymbolicEditRequired` failure mode remains unexercised.

## Files

- `golem/` — stage JSONs, metrics, timings, view sheets (GOLEM arm)
- `blender/` — stage GLBs, mesh-judge JSONs, timings, view sheets
- `side_by_side.png` — final artifacts, both arms, neutral renderer
