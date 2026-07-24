# FRICTION JOURNAL — scree-maiden authoring trial

First authoring trial on the new session surface. Creature: low-slung slate-slab
quadruped, triangular maul-wedge head, moss-tinted spine. Author: Blue Rose.
Convention: entries appended contemporaneously, timestamp order, exact command +
verdict fragment recorded at the moment of confusion. Verdict JSONs saved
append-only as `verdict-NN.json` beside this file.

---

## 000 — setup (before first command)

- Read the commission. Stated protocol: `uv run python -m golem session <base.json> <txn.json>`
  returns a JSON verdict with margins / binding_constraint / deltas since base_txn.
- `session --help` says positional args are `base` and `input`, where input is
  "transaction envelope or complete edited body document". Friction #0 already:
  the commission says `<txn.json>`, the help says `input` can be an envelope OR
  a whole document — the shape of a "transaction envelope" is nowhere shown.
  No `--help` example, no schema pointer. I will have to dig in golem/session/
  or rehearsal/session-knight/ for an envelope exemplar before I can even run
  the protocol once. The surface names a door but not its handle.

## 001 — first contact with the session surface (probe on the exemplar)

Command: `uv run python -m golem session specs/vigil_hound_demo.json specs/vigil_hound_demo.json`
(identity document transaction, saved as verdict-00-identity-hound.json)

WEIRD, three things at once:
1. Verdict says `"status": "accepted"` while carrying **148 diagnostics**. `golem check`
   on the same file says `ANOMALY 0`. The delta: declared-attach anomalies are emitted
   as diagnostics with `witness.expected: true` and severity "warning", and `check`
   filters those to zero. So "ANOMALY 0" and "accepted" do NOT mean "no diagnostics" —
   the verdict's noise floor is ~150 entries for a clean spec. Nothing on any surface
   tells you to filter on `expected`/`severity`; you discover it by diffing two tools
   that claim to measure the same thing.
2. Margins contain NEGATIVE margins (skull -0.013, fore_shin -0.005, tail_tip -0.028)
   on `circulation_clearance` — i.e. spec is out of contract on 7 carriers — yet
   VASCULAR feasibility is ACCEPTED and status accepted. `check` shows these as
   `CIRCULATION_CLEARANCE ... scale skeleton/X/flesh[0] x1.481` "edits" (7 edits):
   the kernel silently re-scales flesh to force clearance and the margin records the
   PRE-repair deficit. So a negative margin is not a violation and not a fix
   instruction; it is a fossil of an auto-repair. "Active: true" marks ONLY the single
   minimum margin (tail_tip), so 6 other negative margins read `active: false` —
   which looks exactly like "fine". The vocabulary lies by omission.
3. `binding_constraint` = the single min-margin entry (tail_tip). Naming ONE wall when
   seven are negative means the steering signal the commission promised ("the binding
   constraint names the wall you are nearest") collapses a vector into a scalar and
   hides the rest.

Also noted: identity transaction yields `txn: 0, deltas: {margins:[],...}` — deltas
only exist against a different base, so my loop must keep a checkpoint base file and
diff the working spec against it. The CLI session is stateless (fresh session per
invocation; base_txn always 0), so "deltas since base_txn" = "deltas vs base file".
That is workable but undocumented; learned it by reading golem/session/protocol.py
+ cli/session.py source, not from any help text.

## 002 — closed material vocabulary, hidden behind the palette (first obstruction)

Command: `uv run python -m golem check specs/scree_maiden.json`
Verdict fragment: `OBSTRUCTIONS 1 / [UnknownEyeMaterialObstruction] /eyes/0/appearance_material: unknown eye appearance material 'eye_amber'`

WEIRD: I authored palette entries named `slate_body`, `moss_spine`, `talon_horn`,
`eye_amber` — the way the hound palette LOOKS like it works (it has keys like
"obsidian_warden" with custom tints, so I assumed palette keys are free names).
They are not. `AppearanceMaterialId` is a closed 10-member enum
(obsidian_warden, obsidian_deep, steel_violet, emissive_seam, eye_gloss,
vascular_supply_gold, vascular_return_violet, vascular_exchange_cyan,
gambeson_dark, neutral_gray); the palette RE-TINTS known ids, it does not mint
new ones. The dialect's only way to make a slate-blue creature is to hijack
"neutral_gray" and redefine its tint — vocabulary the schema hides: nothing in
the spec shape hints the keys are enum-bound; the hound exemplar never
demonstrates it because its names happen to coincide with the enum.
HARD: the check stops at the FIRST obstruction (eye material) instead of
listing every unknown material at once (my base material, four conduit
materials, and the eye would all fail) — one-at-a-time obstruction reporting
means N round trips for what is one mistake class. Also: eyes additionally
require a GLOSSY material (NonGlossyEyeMaterialObstruction exists) — eye_gloss
is effectively the only lawful eye material id. Discovered by reading
golem/materials/core.py; the error message names the bad value but not the
allowed set.
Fix mapping: slate_body→neutral_gray (retinted), slate_dark→obsidian_deep,
moss_spine→gambeson_dark, talon_horn→steel_violet, eye_amber→eye_gloss.

## 003 — first full check: geometry semantics probe (blob size is HALF-extent)

Command: `uv run python -m golem check specs/scree_maiden.json` (first full draft)
Verdict fragments: `bbox 0.98w x 0.96h x 1.80d`, `ANOMALY 12 (+85 more)`,
`VASCULAR feasibility: REJECTED` (CapsuleEscapeObstruction cranial x3,
VascularIntersectionObstruction fore/hind_talons L~R x8).

WEIRD: my hand-computed silhouette was ~0.40 wide / 1.60 deep; the kernel
measured 0.98 / 1.80. Every fusion gap was ~-0.10 to -0.16 where my drawing
said -0.02 to -0.06. A factor of two, everywhere, in the direction of bloat.
Probe: /tmp/probe_blob.json, one blob size [0.1,0.2,0.3] → bbox
0.20 x 0.40 x 0.60. CONFIRMED: blob `size` is per-axis HALF-extent (a radius
triple), and loft `width`/`depth` likewise (hound carrier margin "observed"
0.1 == min section min-dim 0.1, only consistent with radii).
STRANGE SYSTEM: nothing in the dialect names this. "size" [0.1,...] producing a
0.2-wide box is a convention you can only learn by probing or by reading the
mesh code. The hound exemplar never forces the realization because its author
already knew. This is exactly the class of invariant a schema should state.
Also: VascularIntersectionObstruction edges name `negative`/`positive` sides —
that's L/R mirror instances colliding across the midline; the vocabulary
(negative/positive rather than left/right or _m) is internal and unexplained.

## 004 — verdict-01: binding constraint names the wrong wall

Command: `uv run python -m golem session rehearsal/scree-maiden-trial/base.json specs/scree_maiden.json`
Saved: verdict-01.json. Verdict fragment:
`status: rejected`; 5 `vascular.CapsuleEscape` diagnostics:
cranial margin -0.006989956; fore_talons ± margins -0.008766110 and
-0.003634012. BUT `binding_constraint.constraint` =
`anatomy:carrier:fore_shin` with required=0.04 observed=0.04 margin=0,
while the actual rejecting edges are in skull + fore_foot exchange beds.

This is the promised margin-steering surface failing at the exact moment it is
needed. Rejected vascular outcomes emit **no vascular margins** at all; only
carrier margins survive. The binding selector then chooses an unrelated zero
carrier margin (fore_shin; fore_foot/hind_shin/hind_foot are also zero), not the
rejecting CapsuleEscape wall. The diagnostic gives a negative margin and an
internal edge id (`exchange:vascular:cranial:center:s:24>...r:24`) but no
operation, no host part address, no statement of WHICH cross-section escaped,
no mapping from deficit to a spec knob. I can see the wall (-0.00699) but not a
door. I have to infer "thicken host bone flesh" from anatomy expertise and the
hound exemplar.

The full anomaly verdict is also STRANGE: with contract.attach=[] it reports 22
`deep_burial expected:true` entries. That expectation is not authored intent;
it is implicit skeleton adjacency / same-bone ownership. Meanwhile a lap two
links away is `unintended_fusion`. The hidden law is graph distance, not visual
anatomy. No receipt names that rule. I extracted all 44 unexpected pairs from
raw JSON because `check` prints 12 then `+32 more` with no verbosity flag.

## 005 — the frame-composition law (why my legs crossed the midline)

Receipt: kernel/body/geometry.py:196-199 —
`swung = _swung_direction(rest_dir, param.swing)`; `z_world = parent_R @ swung`;
`R_rest = _frame_from_axis(z_world, parent_R[:,1], parent_R[:,0], twist)`.
And linalg.py:47-64: child frame +z = parent_R @ rest_dir; +y = parent's +y
projected orthogonal to +z (fallback parent +x); +x = y cross z.

So every `rest_dir` is authored in the PARENT'S frame, and the child's own
frame (which its children's rest_dirs, its loft width/depth axes, and its blob
size axes are all expressed in) is DERIVED, not authored. My fore/hind girdles
used rest_dir [0.78,-0.06,±0.6231] — a 51° x/z mix — which rotated the girdle
frame so that child dirs authored as (0,-0.8,-0.6) acquired world x ≈ -0.52:
both forelegs aimed INWARD at each other. That is the entire
cross_plane_fusion finding (fore_knee/fore_thigh_slab/fore_shin_facet L~R
overlap) — not a stance-width problem at all. My earlier mirror probe
(girdle + blob flesh, no grandchild BONE) structurally could not reveal this:
flesh follows the bone's own frame, but a child bone's direction composes
through it. I probed the wrong level.

STRANGE SYSTEM: this law is load-bearing for EVERY multi-bone chain and no
surface states it — not the schema, not the contract primer, not any verdict.
The hound exemplar obeys it silently (its girdle x/z mix is why its leg
rest_dirs look "odd" — they are compensated for frame rotation; I read them as
world vectors and copied the compensation without the cause). Knowledge
source: kernel source + an external hint; the surface itself gave only three
cryptic cross_plane_fusion lines naming a plane, not a mechanism.

Fix (verified against linalg.py): pure-lateral girdles rest_dir [1,0,0],
length 0.19 → girdle frame is a clean axis swap (local+z→world+x,
local+y→world+y, local+x→world−z), so a desired world dir (wx,wy,wz) is
authored local (−wz,wy,wx). Thorax pitch (~11°) leaves a small sagittal
residual only. Girdle blob sizes must be axis-swapped too
(size = [world_z, world_y, world_x] half-extents). Legs re-aimed:
fore_thigh [0.60,-0.80,0], fore_shin [0.39,-0.92,0], fore_foot [-0.24,-0.97,0],
hind_thigh [-0.78,-0.62,0], hind_shin [0.44,-0.90,0], hind_foot [-0.41,-0.91,0].

Sequencing lesson: my attach list was computed against the OLD geometry;
declaring pairs that the re-aim separates would flip them to missing_fusion
(declared-but-out-of-reach). So: geometry first, attach=[] re-check, THEN
declare exactly what the fresh anomaly list shows as genuine in-window laps.
The declaration channel punishes stale knowledge — it cannot be written
ahead of the geometry it describes.

## 006 — the cranial CapsuleEscape saga: six nulls, then coordinates

Wall: `REJECTED [CapsuleEscapeObstruction] {"edge_id": "exchange:vascular:cranial:center:s:N>...r:N", "margin": -0.007..-0.009}`.

Experiment matrix (each a full check run):
1. widen wedge mid/tip sections +33%          -> margin BIT-IDENTICAL (-0.00699)
2. tissue_envelope.minimum_radius 0.07        -> auto-edit fired (scale skull x1.167),
                                                 escape IDENTICAL => escape is evaluated on
                                                 AUTHORED flesh, auto-repair cannot absorb it
3. cranial demand 0.05 -> 0.03                -> IDENTICAL => demand-independent
4. eye move + eye REMOVAL                     -> IDENTICAL => not the sockets
5. joint kink flatten (skull -0.34 -> -0.15)  -> MOVED (-0.00745) => butt-adjacent
6. butt deepen + poll enlarge (advisory)      -> MOVED WORSE (-0.00897) => butt cover
                                                 pushes the path OUT; localization inverted

HARD: the verdict's only datum is edge_id + margin. The edge id encodes
internal station numbering (s:24->s:26->s:28->s:14 across runs — the indices
RENUMBER as the bed adapts, so you cannot even track one capsule across
experiments). Six experiments of inference-by-null. The door only opened when
I bypassed the surface entirely: realize the graph via
kernel.anatomy.realize.allocation internals and print the failing segment's
ENDPOINTS (probe saved as /tmp/escape_probe.py). Result: the cranial center
path rides at y~0.825-0.833 OVER the skull head (the bone centerline there is
y~0.62), crossing the notch between hump_plate's front edge (z 0.447) and the
poll plate — a region with NO flesh. The corridor follows the region's flesh
topology (through the poll plate on the skull), not the bone spine; nothing
anywhere says a bed path can leave the bone corridor and need AERIAL cover.
Fix: poll_crest z half-extent 0.13 -> 0.17, closing the notch: ACCEPTED.
Also banked for the MAP: RejectedVasculature carries ONLY obstructions — the
graph is discarded at rejection, so the verdict STRUCTURALLY cannot name the
door; the escape edge's radius was 0.00002 (a line, not a vessel) — the
escape is about path cover, not capsule bulk.

## 007 — process miss: base checkpoint rotation

I rotated edits (eye move, envelope add/remove, demand, butt, poll x3) without
rotating rehearsal base.json after verdict-02, so the delta chain v02->v03 is
not incremental. Recovered by accepting verdict-03 as cumulative-vs-v01 and
rotating base after EVERY verdict from here. The CLI session being stateless
means delta discipline is entirely on the author; nothing warns you.

## 008 — the exact re-aim: frame probe kills the frog-sprawl AND the fore escape

After the first standing render showed floating, splayed legs (front view:
"frog-sprawl"), I stopped hand-solving offsets and wrote /tmp/frame_probe.py —
walk the skeleton with the kernel's OWN fk_table and print each leg bone's
actual world head/tail/dir plus frame axes. Receipts:
  fore_shin world_dir=(-0.39,-0.395,0.832)  (authored intent: (0,-0.92,-0.39))
  fore_foot world_dir=(0.894,-0.379,0.239)  tail y=0.26 — FOOT IN MID-AIR
  hind_foot world_dir=(-0.819,-0.569,0.072) tail y=0.14 — same
The advisory's "pure-lateral girdle" conversion was exact only for girdle
CHILDREN; shin/foot compose through the thigh's already-rotated frame, so each
subsequent bone inherited an x-scramble. Inversion law: local = parent_R^T @
desired_world — solved exact locals per bone (fore_shin [0,-0.065,0.999],
fore_foot [0,-0.581,0.813], hind_shin [0,-0.978,0.215], hind_foot
[0,0.778,0.639]). One edit: feet land at y≈0.035 both sides, bbox height jumps
0.87->0.98 (feet finally BELOW the body), length:height lands 1.55:1 (brief
~1.6:1), fore_talons escape VANISHES (the corridor now has a straight run down
a real leg). ANOMALY 0 on the same pass.
WEIRD: the escaping segment I had been chasing sat at the OLD foot head
(0.085,0.312,0.385) — the bed was telling me the foot was in the wrong place,
and no surface said so. The vascular bed is the best proprioceptive sense the
kernel offers; it diagnoses geometry the skeleton receipt will not.

## 009 — hind talon corner: the last wall

hind_talons escape at (±0.1776,0.0138,-0.232->-0.210): the bed's distal turn
poking 0.002 out of the talon's rounded rear-bottom corner. Read
_certified_segment_capsule_margins (kernel/anatomy/geometry.py:181): margin =
-max(union_SDF(samples)) - radius, certified by a Lipschitz bound over segment
samples — bounding-box arithmetic cannot adjudicate; the chamfer corner
rounding is analytic. Enlarged talon [0.05,0.048,0.07] offset back-down:
ACCEPTED. Lesson for the MAP: at capsule-escape scale, flesh is not its bbox —
corner rounding is load-bearing; extend PAST the corner, not to it.

## 010 — tint wall: look renders ignore appearance_palette entirely

Wanted: slate blue-grey body, moss-green spine accents. Authored palette
re-tints of closed ids. Renders: body light grey, spine stripe near-BLACK at
gambeson_dark [0.16,0.30,0.10]; identical black at [0.3,0.6,0.15]; emissive_seam
renders VIOLET regardless of palette. Source receipt: senses/orthographic.py:100
_material_rgb -> materials/decode.py:59 resolve_appearance_material -> FIXED
catalogue recipes (materials/core.py: neutral_gray base (0.40,0.41,0.44) —
exactly the render's body color). assembly/compile.py apply_appearance_palette
sets record.surface_color, but the orthographic renderer never reads it. So:
the 10 catalogue colors are the whole look-render vocabulary; palette tints
serve other consumers (GLB/mesh vertex color path, surface.py:610). Moss-green
is UNREACHABLE in look renders today — a surface gap, not an author error.
Shipped: palette intent documented in-spec ([0.16,0.30,0.10] on gambeson_dark),
render shows the catalogue's dark seam. Also: flesh entries cannot carry
appearance_material (schema census: eyes, top-level, plate seams only), so
per-part tint routing exists ONLY through conduits — which the renderer then
flattens to slot defaults. Two layers of the same wall.

## 011 — acceptance

check exit 0 | ASSERT 0 FAIL / 0 pass | ANOMALY 0 | VASCULAR ACCEPTED
(0 circulation edits — no auto-repairs needed; every carrier radius authored
above requirement). verdict-06 accepted. Binding: vascular:capsule_clearance
norm 0.0002 — razor-thin but lawful; all carrier margins 0.20-0.67.
Renders: look-final/{front,side,top}.png — she stands: wide planted stance,
boar mass, hump-to-poll ridge, down-tipped wedge.
