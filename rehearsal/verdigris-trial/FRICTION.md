# FRICTION JOURNAL — verdigris (winged drake) authoring trial

Trial 2. Author carries the scree-maiden scars: frame composition, escape
cures, half-extents, base rotation, closed material enum, look-renderer tint
law. NEW WALLS ONLY are recorded here — trial-1 frictions live in
rehearsal/scree-maiden-trial/. Verdicts saved append-only as verdict-NN.json.

## 000 — protocol deltas from trial 1 (scars applied, not walls)

- Building in STAGES this time: (a) skeleton + carrier flesh[0] only, vascular
  first; (b) decorative flesh (blades/horns/claws/membrane); (c) attach pass;
  (d) tint. Trial 1 authored everything at once and debugged the superposition.
- Frame probe (fk_table dump) runs BEFORE flesh on every multi-bone chain
  (legs, wings, S-neck); exact locals solved via local = parent_R^T @ world.
- Escape probe (vascular graph endpoint extraction) loaded from trial 1.
- Wing pose decision pending receipts: half-furled planned (struts near body
  mass = escape-safe); full-spread only if early vascular probes say the finger
  terminals are honestly reachable.

## 001 — the frame law's second bite: the tail folds (axial -z inversion)

Wall: 4 beds rejected at seed with BIT-IDENTICAL ReservedLaneClearance -6e-05
(supply/return capsule pairs 4v1/8v7/9v10/4v1), invariant to host fattening,
chest deepening, and demand halving (candidate counts moved; margin never).

The paired-corridor dump (lane probe, after breaking two instrumentation
walls — see 002) showed the caudal corridor ZIGZAGGING:
z -0.12 -> -0.22 -> -0.52 -> -0.24 -> -0.5 -> -0.38. My tail bones:
tail_a local (0,-0.05,-0.9988) in the pelvis (world) frame -> world -z ✓.
tail_b local (0,-0.10,-0.995) in tail_a's frame — tail_a's frame z-axis IS its
world dir (0,-0.05,-0.9988), so tail_b world ≈ -0.995 * (0,-0.05,-0.9988) ≈
(0,+0.04,+0.99) — pointing FORWARD back into the body. Each -z local inverts a
generation: tail_b forward, tail_c backward, tail_d forward. The tail folds
through the torso; the bed's lanes overlap by construction; every flesh/demand
knob was orthogonal to the failure.

TRIAL-1 LAW, GENERALIZED: rest_dir composes in the parent frame — I knew this
for LIMBS (lateral girdles) and missed it in the AXIAL chain, where
"near-axis ≈ near-world" holds only for +z continuations. A chain continuing
straight needs local ≈ +z FOREVER; authoring the world direction (-z) into
each child is the fold. The silent trap: tail_a is correct either way (root
frame IS world), so the error only shows at generation 2+.
Fix pattern: continuing children get local (0, droop, +0.99x); direction
changes get explicit solves via the frame probe.

## 002 — instrumentation walls: lru priming + spawn pool (MAP candidates)

Two layers made the seed search unpatchable for four probes:
1. allocation._realize_vasculature_cached is @lru_cache(maxsize=8), and
   start_session EVALUATES the authoring state (realizing the vasculature)
   BEFORE my probe installs patches — the memo then serves the explicit
   realize call. Receipt: patched _realize_vasculature_cached fired,
   _realize_vasculature_uncached never did. Fix: patch before start_session
   AND alloc._realize_vasculature_cached.cache_clear().
2. lineage._solve_parallel dispatches >1 circuit over a SPAWNED
   ProcessPoolExecutor when estimated work > 256 pairs; children re-import
   modules fresh — parent monkeypatches do not exist there. At exactly 256 the
   path is sequential, but the floor is invisible from any verdict surface.
   Force serial in probes:
   lin._solve_missing = lambda i: tuple(map(lin._solve_one_circuit, i))
3. GOLEM_CIRCUIT_CACHE_COLD=1 bypasses the disk store
   (lineage.py:337, .golem-cache/) — required for iteration, absent from
   check --help and every receipt. Identical rejection output across spec
   edits is the smell; the cache key covers the body graph, so cross-edit
   identity is real signal, but cross-run identity within one spec is cache.

## 003 — the verdict named a wall; only capsule coordinates named the fold

Commands:
`uv run python -m golem check specs/verdigris.json`
and the runtime-only capsule probe
`GOLEM_CIRCUIT_CACHE_COLD=1 uv run python - specs/verdigris.json < /tmp/pair_probe.py`.

Verdict fragment:
`ReservedLaneClearance required=1e-08 observed=-6e-05 supply_capsule=4 return_capsule=1`
for both `fore_toes` and `wing_tips`. The margin and integer indices did not
identify a body location. After instrumenting the discarded candidate trees,
the fore failure was a centerline crossing in the forward thorax prefix; moving
`fore_girdle.attach.t 0.70 -> 0.58` cleared it. Moving
`wing_root.attach.t 0.52 -> 0.40` advanced the wing failure to
`observed=-2.52947e-05 supply_capsule=6 return_capsule=7`.

The next coordinate receipt exposed the real wing wall: the supply humerus
`(0.14,0.533,0.290)->(0.213,0.724,-0.010)` crossed the return forearm
`(0.212,0.718,-0.014)->(0.245,0.881,0.215)` at the elbow. The authored
half-furl reversed world-z from -0.83 to +0.80 in one joint. Re-solving
`wing_fore` to world `(0.08,0.94,0.33)` kept the wrist spike above the shoulder
but softened the fold; every `ReservedLaneClearance` seed obstruction cleared.

WEIRD/HARD: the public obstruction gives capsule indices in a transient tree
that is discarded on rejection. It provides no endpoints and no mapping to a
bone, so “move the branch backward” versus “soften the elbow” is invisible
without importing kernel internals, disabling two cache layers, forcing serial
solve, and logging candidate-tree capsule coordinates.

## 004 — cranial escape moved from “skull” to the S-neck after coordinates

Command:
`uv run python - specs/verdigris.json < /tmp/escape_probe.py`

Verdict fragments:
`CapsuleEscapeObstruction` margins `-0.0105073`, `-0.0135306`,
`-0.0106973`; edge ids all say `exchange:vascular:cranial:center`.
Coordinates:

- `(0.0168,0.3618,0.6019)->(0.0168,0.3618,0.6607)`
- `(-0.0151,0.5003,0.5873)->(-0.0151,0.5003,0.6753)`
- `(-0.0027,0.3727,0.5654)->(-0.0027,0.3727,0.6972)`

The edge ids imply the skull bed, but all three segments lie around the
neck-a/neck-b junction (`drake_wedge` begins near world z=0.83). As in the
Maiden's poll-crest wall, the graph follows region flesh topology rather than
the named host bone centerline. The verdict names neither the missing covering
part nor the nearest surface. The honest door is a small nuchal poll/collar at
the measured junction, not arbitrary skull inflation.

## 005 — `check` accepted the body that `session` rejected

Commands:

- `uv run python -m golem check specs/verdigris.json`
- `uv run python -m golem session rehearsal/verdigris-trial/base.json specs/verdigris.json`

`check` receipt: `VASCULAR feasibility: ACCEPTED`, `5/5 closed`, `0 edits`,
`coverage 1.000`. It printed no hard obstruction and returned through the
vascular summary. The session receipt (`verdict-01.json`) instead returned
`status: rejected` with one error:
`assembly.EmptySurfaceConduitObstruction` at
`verdigris-drake__jade_belly_waist`.

WEIRD: the two advertised authoring paths do not exercise the same acceptance
surface. A creature can satisfy the explicit DONE headline under `check` yet
fail the new protocol for an assembly-only surface detail. The binding
constraint still says `anatomy:carrier:skull margin=0`, unrelated to the
rejecting conduit.

HARD: the diagnostic has no observed value, no source address
(`conduits[jade_belly_waist]`), no reason the embedding was empty, and
`help.operations=[]`. Source inspection revealed `face_line` ignores
`face_normal`; it selects union-mesh vertices near the named part's own SDF.
Thus a lawful conduit can become empty when its host part is buried inside the
union, a distant coupling neither schema nor verdict names. The honest repair
is to host the rear belly line on exposed `haunch_mass`, not tune the ignored
normal or invent width guesses.

## 006 — decorative horn bones silently become vascular organs

Command:
`uv run python -m golem check specs/verdigris.json`

Verdict:
`MissingTerminalRegion @ skeleton/horn: terminal skeleton region has no
capillary exchange bed` plus `NonterminalExchangeRegion @
anatomy/regions/cranial: exchange host 'skull' is not terminal`.

I authored a mirrored `horn` child bone solely to orient a swept-back loft.
That changed the circulation ontology: every skeleton leaf is treated as a
terminal region requiring an exchange bed, and an existing exchange host
becomes illegal as soon as decorative descendants make it nonterminal.
Neither body schema nor the check help says “bones are perfused terminal
organs, not transform handles.”

Fix: keep `skull` terminal and encode the paired horn blades as two chamfered
flesh elements on it. This loses tapered/curved bone-frame expressiveness but
does not lie to the vascular model. Requirement: the dialect needs
non-anatomical transform handles or an explicit `decorative` bone role; authors
should not have to choose between honest circulation and orientable geometry.

## 007 — thin membrane is carrier flesh; the dialect inflates it

Command:
`uv run python -m golem check specs/verdigris.json`

After adding narrow gathered-membrane lofts to `wing_fore` and
`wing_finger_a`, check reported `2 edits`:
`CIRCULATION_CLEARANCE @ skeleton/wing_fore/joint: radius 0.018 < 0.040;
scale skeleton/wing_fore/flesh[1] x2.222` and
`wing_finger_a ... radius 0.012 < 0.040 ... x3.333`.

STRANGE: every flesh element on a circulation-route bone is treated as a
carrier cross-section, including decorative membrane beside an already lawful
carrier `flesh[0]`. There is no non-carrier/decorative flesh role, so a true
thin membrane cannot coexist with the wing-tip exchange route without silent
inflation.

Honest adaptation per the commission: retain the accepted half-furled strut
pose and thicken the two gathered panels to the 0.040 carrier floor, describing
them as strut-and-web ridges rather than pretending they are thin sheets.
Also reverted `hind_girdle` widening: moving the terminal chain laterally
reopened `WallContainment observed=-0.00732363`; the accepted 0.17 socket
geometry stays.

## 008 — integrity reports only a count; the two violations were eye brows

Commands:

- `uv run python -m golem session rehearsal/verdigris-trial/base.json specs/verdigris.json`
- `uv run python -m golem compile specs/verdigris.json`

Both reported only `SolidIntegrityObstruction` with
`vocabulary_violation_count=2`; neither named a rule or part. A direct
compile at the inferred default resolution (`PinnedResolution(155)`) retained
the advisory records and exposed both mirrored violations:
`eye.verdigris.brow{,_m}`, rule `R-8-sharp-edge`,
`round=0.016000 < pitch=0.022616`, repair `round >= 0.022616`.

WEIRD: accepted pinned compilation prints/retains rule-level repairs, while the
normal target-pitch path turns the same records into a hard integrity failure
and discards every actionable detail from CLI/session output. Fix:
`brow_ridge.round 0.016 -> 0.024`.

## sol handoff

### 009 — a planar shared-trunk fork has no author-level waypoint

Command:
`GOLEM_CIRCUIT_CACHE_COLD=1 .venv/bin/python -m golem check specs/verdigris.json`

The three final intersections were all on `x=0`. Coordinate instrumentation
showed the wing return corridor leaving the pump cover across whichever axial
shared lane occupied the same sagittal plane:

- positive pump common crossed the cranial/fore supply turn at clearance
  `-0.000240569547438`;
- negative pump common crossed the caudal supply trunk at clearance
  `-0.000247588459599`.

Small `wing_root.attach.t` changes moved the witness but could not change this
topology. A direct `attach.offset.x` nudge split formerly glued midline nodes;
the reflected local wing trees then crossed one another as nonincident edges.
The body dialect exposes no vascular waypoint, so neither planar re-aim nor a
blind lateral root shift can express “share the trunk, then diverge.”

The accepted descent inserts one nonterminal `wing_waypoint` at the existing
`thorax/head` shared node, then gives its mirrored tail a `0.002` lateral
departure before `wing_root`. The old wing-root tail and every distal wing
frame remain fixed by re-expressing the child in the waypoint frame and
shortening only that hidden carrier segment `0.140 -> 0.138`. The socket center
is restored exactly with local offset `z=-0.0002`.

### 010 — routing bones require flesh provenance; sharp shared turns swap lanes

A fleshless waypoint was rejected as:
`MissingProvenanceObstruction region_id=wing_tips bone_id=wing_waypoint`.
Giving it a lawful `0.040`-radius carrier loft cleared provenance, but a purely
planar waypoint still failed seed construction with
`ReservedLaneClearance observed=-6e-05`. At the thorax-head anchor the failing
indices were supply capsule 4 (shared caudal turn into the anchor) and return
capsule 5 (wing departure from the anchor): transported paired-lane normals
crossed at the sharp turn.

The same `0.002` mirrored lateral departure cleared the lane swap. The carrier
is genuinely bilateral and fused at the central chest, so and only so it is
declared in `contract.midline`. Its observed contacts with `waist_line` and
`wing_humerus` are declared as exact attach pairs; no blanket declaration was
added.

Form receipt: a temporary body-surface render of pre-surgery versus final,
with anatomy intent removed equally from both, was pixel-identical in front,
side, and top (`difference bbox=None`, RMS `0.0`). The compiled socket-center
delta was `0.0`; maximum distal wing frame delta was below `4.5e-16`.

### 011 — `check` exit zero does not imply anomaly zero

The first vascularly accepted `0.002` waypoint candidate returned check exit
`0` while printing `ANOMALY 8`. Exit status covers assertions, fatal geometry,
and typed vascular acceptance, but not visible proprio anomalies. Completion
must read the headline and clear each observed pair explicitly; process status
alone is not the advertised acceptance surface.

### 012 — sandboxed `uv` dies before GOLEM starts

Both normal and offline/cache-local invocations of
`uv run python -m golem ...` panicked in Rust
`system-configuration-0.6.1/src/dynamic_store.rs` with
`Attempted to create a NULL object` under the managed macOS sandbox. All GOLEM
checks, sessions, and the final look therefore used the already-resolved
`.venv/bin/python` interpreter. This is an execution-wrapper wall, not a model
rejection; the direct interpreter receipts are authoritative for this handoff.

## wave-3 amendment

Lane KIMI-A implementation journal (dialect ontology roles: `handle` bones,
`non_carrier` flesh). New walls hit while amending, not while authoring.

### A1 — authored bone fields do not ride to the law layer

Wall: the anatomy terminal law reads `self.bones`, but those records are
REBUILT in kinematics.py (`_place_bone`, root record) with a fixed key
whitelist — an authored `role` key is silently dropped before reaching
`_terminal_bones`. Worse, a SECOND parallel constructor exists
(geometry.py `_bone_record` + `fk_table` root) used by geometry views, so one
authored field has three record shapes to thread and no surface names any of
them. Found by grep (`"mirrored"` as the tracer key), not by any receipt.
The authoring surface implies "the bone declaration flows to the law"; the
compiler actually re-derives the record twice and drops what it does not
know. Any future ontology field hits the same wall.

### A2 — the carrier floor's "inflation" is advisory, but the surface says
"edits"

Receipt: `suggested_scale` has NO geometry consumer (grep: only
project.py renders it and render.py prints it). The membrane "inflation" of
FRICTION-007 is an unapplied suggestion row — yet check prints it under
`N edits` with `scale skeleton/... x2.222`, which reads as an applied
mutation (trial 1 read it the same way: "the kernel silently re-scales
flesh"). The K2 fix (exclude `non_carrier` flesh from `_minimum_carrier_sample`,
withhold its carrier provenance so parity raises the EXISTING
MissingProvenanceObstruction) therefore needed no geometry repair — only
sampling and provenance law. The word "edits" for a suggestion list is a
teaching-surface lie (K4 territory); recorded here because two trials
misread it as applied repair.

### A3 — union-verification hazard: a pre-existing suite failure masquerades
as lane damage

`tests/test_cli_contract.py::test_check_distinguishes_closed_intent_from_infeasible_vasculature`
fails on this tree independent of the KIMI-A edits (verified by stashing the
lane's files and re-running): the committed `failing_segments` verdict
payloads (SOL-A ground) outgrew the test's exact key-set assertion
`{left_edge_id, right_edge_id, clearance}`. Any lane running the full suite
hits it and must attribute it before proceeding; attribution cost one
stash-verify cycle.

### A4 — the two bone records disagree on solved-placement semantics

K5 could not lawfully replace both record constructors with one full
constructor: `geometry.py::_bone_record` stores solved `twist_deg` (authored
twist plus `BoneParam.twist`), while `kinematics.py::_place_bone` stores the
authored twist even when a relational solve changes the frame. Flattening that
difference while fixing field loss would have silently changed a placement
contract.

The safe cut is `body/records.py::project_bone_record`: one canonical
projection owns every authored payload field that must ride a compiled bone
record (`flesh`, `ports`, `arrays`, `bone_radii`, `role`, plus common identity
and joint fields); each evaluator layers its own solved placement fields over
it. Both root and child paths in kinematics and `fk_table` now use it. An
authored key outside the projected/placement vocabulary raises
`UnthreadedBoneFieldError` in both consumers instead of disappearing from a
whitelist. Tests compare the full projected payload across both tables and
exercise the loud failure on an invented future field.

### A5 — K6 vocabulary already existed below an incomplete validation fence

The declaration did not need inventing: body flesh already emitted `blend` and
`operator`, the engine already implemented per-part radii plus
`blend`/`chamfer`/`crease`, and `contract/schema.py` already exposed both in
`common_flesh`. The missing layer was strict body decode. A string blend crashed
later in `_r`; a negative blend passed through; explicit bad operators reached
engine rejection only after body emission. A soft `validate.py` violation was
not honest because compilation continues after violations — guarding emission
would have silently substituted the global blend and changed the authored
form.

K6 therefore adds `MalformedFleshCompositionObstruction` to the pre-emission
body decode gate: present blends must be finite numbers >= 0 (booleans do not
count), and operators belong to the closed engine vocabulary. Invalid form is
rejected at `skeleton/<bone>/flesh[i]/<field>`; it is never compiled as a
different creature. Absence remains byte-exact: the minimal no-declaration
probe retains graph SHA-256
`092d7f12cd5fde4a1f6ba4219f46a0ce2adbd16e2bc68a9af3dc781f5e007ea8`.

CONTRACT STITCH FOR FABLE/KIMI-B (forbidden territory in this lane):
`contract/schema.py` currently declares `common_flesh.blend` as merely
`{\"type\":\"number\"}`. Add `minimum: 0` so the published schema agrees with
the kernel's strict finite/non-negative decode. The operator enum is already
correct; the primer already states the three operator semantics.

### A6 — GLOBAL `k` was already lying about five shipped overrides

The first full touched-file run broke the byte-exact knight proprio golden:
`rehearsal/knight/golem_knight.json` already contains five part-level
composition overrides, but GLOBAL printed only `k=0.123` as though every edge
used that radius. The golden failure was the proof, not collateral damage.
GLOBAL now preserves the exact old line when no override exists and reports
`k=<global> (global; N part override(s))` when they do.

Senses also treated `operator:\"crease\"` as if it had the selected smoothing
radius. That made hard-union edges eligible for blend-reach fusion explanations
they cannot realize. Crease instances now carry predicted blend reach `0.0`;
blend and chamfer retain their selected radius.

### A7 — A2's “no consumer” premise went stale, but “edits” remains false

Concurrent wave-3 work added consumers after A2 was written:
`golem/kernel/anatomy/project.py` projects `suggested_scale` as a
`suggestion:{operation:scale_radius,factor}` payload, and
`golem/session/protocol.py` can turn it into a `SCALE_RADIUS` repair operation.
Neither consumer means the body geometry was mutated when the proprio receipt
prints. The honest distinction is now “consumable advisory suggestion, not an
applied edit,” not “has no consumer.”

K7 changes only the permitted receipt surface: `N edits` becomes
`N suggested carrier scales`, and each row says `suggest scale ...`. No anatomy
payload or session protocol was touched.

### A8 — blocker: crease is an incoming-edge law, not an intrinsic zero radius

A6's first repair was wrong in one declaration order. It set every crease
instance's `blend_r` to zero and kept the old symmetric
`min(left.blend_r,right.blend_r)` pair rule. The engine does not compose
symmetrically: it folds instances in declaration order, and the later/incoming
part owns the junction's operator and selected radius. Therefore
`[A operator=crease, B operator=blend]` is blended by B, but the first repair
predicted zero reach from A and could emit a false `missing_fusion`.

Resolution: `InstanceDatum.blend_r` again retains the raw selected part radius.
One centralized `_incoming_blend_radius` maps instances to their declaration
/ fold order, selects the later instance, and returns zero only when that
incoming instance declares `operator:crease`. Fusion prefiltering, fusion-band
classification, blend-ambiguity detection, and declared-attach blend reach all
use the same helper; attachment argument order cannot reverse the law.

The paired ratchet is explicit:

- `[crease, blend]` keeps positive incoming blend reach, enters the observed
  pair set, and produces no `missing_fusion` inside reach.
- `[blend, crease]` has zero incoming reach and produces
  `missing_fusion` for the same positive gap.

The no-declaration graph hash remains
`092d7f12cd5fde4a1f6ba4219f46a0ce2adbd16e2bc68a9af3dc781f5e007ea8`.
All KIMI-C touched test files pass: `102 passed` (plus the isolated byte-pin
test).

### A9 — material identity is record-scoped; one creature cannot burn in one
region

Lane KIMI-D1 implementation journal (material classes, R8). The palette
keyspace is `AppearanceMaterialId`, and material ids attach per RECORD — one
per element, plus the vascular strata — never per part. The crystalline-drake
target (obsidian hide, burning mouth, membrane wings as ONE creature) has no
lawful single-element encoding: the mouth cannot carry `emissive_seam` while
the hide carries `obsidian_warden`, because a part cannot carry an appearance
material at all. The probe that tried (three elements, one per material
region) died on `ElementFitObstruction`: the fit law demands disjoint
elements, and a glowing maw recessed into a head is a penetration, not a
contact. The class vocabulary therefore composes with assemblies of
spatially-separated parts, not with material-regions-of-one-body. That is an
authoring-surface wall R8 did not name: classes color records, and the
dialect has no part-level material address. D2's protrusion vocabulary will
hit the same wall the moment a crystal ridge wants its own class.

### A10 — the renderer fence forced an outside compositor; the oracle cannot
share its depth law

`golem/senses/` is forbidden territory for this lane, but the shading
attenuation emissive must ignore lives in
`senses/orthographic.py::_render_view`. The honest cut was composable from
`cli/look.py` alone: the shaded pipeline already runs through the
deterministic painter (`raster._render_flat_view`, exact face colors), so the
class path adds flat id-color passes through the SAME painter at the SAME
render size and camera — per-pixel nearest-face attribution with zero
renderer edits, occlusion-correct for emissive-behind-opaque and
translucent-over-opaque. The unshaded mode cannot follow: it renders through
the hash-frozen pilot oracle (`pilots/render.py`), whose internal framing and
z-law are opaque, so no id-color pass can be aligned with it from outside.
Class-aware unshaded renders therefore also use the painter — a rasterizer
divergence that exists ONLY when classes are declared (absent classes take
the legacy path byte-identically; 12 pinned PNG hashes on knight_body and
scree_maiden prove it). The divergence and the single-layer translucent
collapse are named in a `golem.material_classes` PNG text chunk, per the
lane's honest-receipt law. Adjacent hazard: the class path fits the subject
by encoding coverage (`winner >= 0`) while the legacy path fits by shaded
body mask; the two bboxes can differ by a row where a rim-lit pixel lands
exactly on background (244,244,244). Tests must never compare pixels across
the two paths — assert relationally within one render.

### A11 — glTF carriers quantize what the vocabulary expresses

Two fidelity limits surfaced at the export seam, both absorbed without lying:
(a) core glTF `emissiveFactor` is capped at 1.0, so authored intensity above
1 has no honest core carrier — the export reuses the catalogue's existing
`component * min(strength, 1)` convention, `look` pre-scales the linear tint
and lets the single sRGB seam clip, and the sidecar preserves the full
authored intensity (KHR_materials_emissive_strength is the honest carrier;
trimesh 4.12.2 does not expose it). (b) trimesh round-trips `baseColorFactor`
alpha through uint8 — authored opacity 0.45 exports as 115/255 — while
`COLOR_0` alpha carries the exact byte. Tests pin both honest representatives
rather than pretending the float survives.

CONTRACT STITCH FOR FABLE/KIMI-B (forbidden territory in this lane):
`contract/schema.py`'s `appearancePalette` def still describes only
`tint_linear_rgb` + `variation`. It must document the `class` key, the
per-class legal field sets (opaque: tint/variation; emissive:
tint/intensity, intensity > 0 finite, default 1.0; translucent:
tint/variation/opacity, opacity REQUIRED in [0,1]), the unknown-class typed
rejection naming `("emissive","opaque","translucent")`, and the deliberate
law that a local element `surface_color` REJECTS the class key
(unknown-field) — classes are palette-entry vocabulary only.

### A12 — carve promotion (R9): the law stack already tolerated the void the vocabulary lacked

Promoting the eye-socket carve privilege into an authored bone-local `carves`
vocabulary was almost pure vocabulary, because the geometry law was already
total: the engine subtracts every `graph["carves"]` instance with the hard
`max(host, -carve)` difference, and both the mesh coherence report and the
vascular material masks read that authoritative carved field. Threading a new
bone key still bit exactly where A1 predicted — `records.py::project_bone_record`
raises `UnthreadedBoneFieldError` for any authored key it does not project, so
`carves` had to be registered in `_PROJECTED_KEYS` before the compiler would
accept it. The canonical projection (A1/A4's cure) paid off: one line threaded
the field to every FK consumer with no per-constructor whitelist to chase.

Two law claims from the dispatch turned out to already hold, and honesty
required proving that rather than adding ceremony:
(a) VASCULAR. A carve that guts a carrier does NOT need new anatomy
containment code. An off-centre slab carve through the box host leaves the
lumen intact but splits `load_bearing_solid`, and
`derive_vascular_material_masks` already rejects it with
`ChannelInducedDisconnectionObstruction(component_count>1)`. The pin lives in
`tests/test_anatomy.py`. The residual wound worth recording: this rejection
lands at the material-mask boundary in the assembly/coupled path, far from the
authored carve declaration — the session analytic stage accepts
`realize_vasculature` first because vascular CONTAINMENT (`_union_sdf` over
positive parts in `golem/kernel/anatomy/realize/`) is still carve-blind. The
solver wastes a search inside the cavity before a downstream stage catches the
broken anatomy. Making containment itself carve-aware is a real, separate
ownership decision in anatomy territory (sol's law-generalization lane), not
this promotion.
(b) SENSES. Anomaly detection never observed carves at all: every pair gap is
measured from positive-part SDFs. A close lip/jaw pair therefore reported
`blend_ambiguity` ("no void; may bridge at mesh") even when an authored oral
cavity guarantees no bridge. That WAS a false anomaly at a mouth. The fix is
a minimum field-aware guard in the `blend_ambiguity` branch
(`_carve_severs_bridge`): find the closest cross-pair of surface samples, and
if the authored carve field (the compiler's own `graph["carves"]`, sampled via
the existing `field_at` wrapper — no second SDF) fully occupies that neck
segment, the advisory is suppressed. An off-corridor carve does not silence a
real bridge. Carve-free specs never enter the guard, so every pre-existing
proprio pin is byte-identical.

One vocabulary taste note: carve kinds are the closed set the engine carve
path already evaluates — `("gencyl","blob","box")` — with no string fallback,
so an unknown kind names the whole legal tuple. Bone-local placement reuses
`placed_center` and the bone rotation exactly as flesh does, so mirroring is
inherited from the bone record with zero new mirror machinery; the authored
cheek pit and its sagittal reflection flip solid→void symmetric to the last
emitted digit.

### A13 — spanning-flesh promotion (R10): the membrane was vocabulary; the topology was already law

Promoting the engine's frozen `SpanningWeb` into an authored top-level `webs`
section was the R9 shape repeated one level up: a multi-anchor record (never
duplicated into bone records, so the K5 projection stays untouched), strict
decode with typed obstructions naming the legal shape per R4, and
`graph["webs"]` emitted only when authored — byte-identical otherwise
(pinned against `specs/eye_demo_head.json` and an explicit `webs: []`).

Four honest findings, in ascending order of interest:

(a) ROLE SEMANTICS BIT ON FIRST LANDING. The first `compile_webs` emitted
carrier provenance for EVERY web, including `"role":"non_carrier"` ones —
quietly violating the same role law (R3) the flesh channel obeys. The test
that caught it (`test_non_carrier_web_carries_nothing`) is the argument for
writing the role test before the happy path, not after.

(b) THE MIRROR LAW IS EASY TO OVER-PROBE. My first atomic-mirror test probed
the sheet edge at exactly the authored half-extent, inside the one-sided
finger flesh's blend reach; the field there is dominated by the unmirrored
fingers, so authored/reflected probes disagreed and the "mirror" looked
broken. The mirror was never wrong — the probe measured the blend, not the
reflection. Sheet-interior probes with the web creased (`blend: 0.0`)
compare equal to the last digit. Lesson: when a symmetry assertion fails,
check what the field is actually composed of at the probe before blaming
the symmetry.

(c) THE (A-1)^2 TRAP IN CONSUMERS. The proprio sampler's triangulation
helper paired spine intervals and radii intervals through two independent
`for` clauses — a cartesian product that is accidentally correct for exactly
2 anchors ((A-1)^2 = 1), which is every fixture I had written. For the
headline case (the drake's 3+-finger wing) it would have duplicated
triangles and mis-assigned every half-extent, silently, in all four
consumers (sampler, bounds, min-radius, volume/centroid). A reviewer caught
what my tests structurally could not; the pin now asserts the sample count
(2 faces × 7 barycentric × 2 triangles × (S-1) × (A-1)) and that a vertex
sample carries its own pair's station radius, never a mis-paired one. If
your fixtures all share one arity, your tests cannot see arity bugs.

(d) AERIAL COVER HAS TWO HALVES, AND ONLY ONE IS MINE. The senses half is
landed and provable: a web joins the instance fold in authored order after
parts (matching the engine's union), so contiguity counts it as flesh
topology between its anchors (2 components without, 1 with, `fused_adj`
naming the web on both anchors), and the compiler declares web↔anchor-flesh
attach pairs so the membrane meeting its anchor bones reads as intent
(`deep_burial`, expected) — never `unintended_fusion`. The VASCULAR half is
the same deferred-ownership species as A12's carve-blind containment:
`golem/kernel/anatomy/realize/` derives containment from `graph["parts"]`
plus `skeleton/{bone}/`-prefixed provenance, so a carrier web's provenance
(which deliberately names the multi-anchor declaration `webs[i]
anchors=(...)` rather than impersonating a bone-local address) does not yet
rescue an anchor bone from `MissingProvenanceObstruction`, and `_union_sdf`
over part dicts is web-blind. Wing venation as lawful vasculature — R10's
"exchange-bed and vessel paths may ride it" — needs the anatomy/assembly
consumption generalization, which is sol's law-generalization lane, exactly
as carve containment was. Recorded here so it is a decision, not an
oversight.

One vocabulary note for the contract stitch (KIMI-B territory, not mine):
top-level `webs` is now dialect vocabulary — `BODY_SPEC_TOP_LEVEL_KEYS`
admits it and `webs.py` owns the strict decode — so the schema/primer prose
must teach the authored shape (`{id, anchors: [{bone, span?, stations?,
radii}], mirror?, blend?, operator?, role?}`), the station-arity law
(rejection naming per-anchor counts, never silent pad/truncate), the atomic
mirror, and the all-anchors attribution discipline.
