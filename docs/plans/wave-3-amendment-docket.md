# Wave 3 — Authoring-Surface Amendment Docket

Harvested from two full authoring trials (Scree Maiden, Verdigris — both ACCEPTED) run
against the wave-2 session spine. Receipts live in `rehearsal/scree-maiden-trial/FRICTION.md`
(12 entries), `rehearsal/scree-maiden-trial/SURFACE-MAP.md`, and
`rehearsal/verdigris-trial/FRICTION.md` (8 Kimi entries + sol handoff section).

Every element below is a defect the surface inflicted on a competent author. The lanes are
file-disjoint so they dispatch in parallel. Fable's lane is the rulings written inline here,
merge review, and union verification — no fable execution.

---

## Standing rulings (fable)

**R1 — One acceptance oracle.** The session path's semantics are canonical. `check` accepting
what `session` rejects is a false green of the exact species the vascular-floor campaign
cured; the cure extends: every surface the transaction path evaluates runs in `check`, or
`check` prints a typed notice naming the surfaces it did not run. No third semantics.

**R2 — Verdicts must be able to name the door.** A rejection that discards the geometry which
caused it is structurally incapable of guiding repair. Obstruction payloads carry enough to
localize (segment coordinates, host part addresses, spec addresses) — prose renderers may
summarize, but the payload never forgets.

**R3 — Roles are contract vocabulary, not compilation accidents.** If the kernel treats a bone
as a perfused organ or inflates flesh to a carrier floor, that classification must be
authorable and visible — never a silent default. Absence of a role declaration keeps today's
behavior; the roles are opt-in vocabulary, not a migration.

**R4 — Vocabulary must not lie by omission.** Margins, enums, and mirror ids that appear in
verdicts must be resolvable by the author from the spec and the documented contract alone.

---

## Lane SOL-A — Verdict truthfulness (codex, gpt-5.6-sol, xhigh)

Territory: `golem/kernel/anatomy/` obstruction types & vascular verdict payloads,
`golem/session/`, `golem/cli/session.py`, session-side rendering. Does not touch assembly
conduit machinery or `golem/cli/check.py` pipeline order (lane SOL-B's ground).

- **A1. RejectedVasculature carries the wound.** (Maiden FRICTION-003/007, Verdigris 001–004.)
  Rejection currently discards the solved graph; the verdict cannot name which segment failed
  where, against which host part. Amendment: rejection payload retains failing-segment
  geometry (endpoint coordinates, bone/part addresses on both sides) and surfaces it through
  session diagnostics. Per R2.
- **A2. Binding constraint honest under rejection.** (Verdigris 002.) Rejecting channels emit
  no margins, so the selector names an unrelated satisfied wall; multiple violations collapse
  vector→scalar. Amendment: rejecting channels emit their violated margins; binding
  constraint selects among violations when any exist; report all violations, not the min.
- **A3. Auto-repair fossils distinguished.** (Maiden 009.) Negative margins with
  `active:false` are records of repairs already applied, indistinguishable from live
  violations. Amendment: margins carry a `repaired` disposition (or equivalent) separating
  "the kernel fixed this silently" from "this binds you"; silent repairs become first-class
  journal events.
- **A4. Margins name their knobs.** (Maiden SURFACE-MAP.) A margin that names no spec address
  is an orphan — the author cannot know which declaration moves it. Amendment: each margin
  carries the authored address(es) whose change moves it, where derivable.
- **A5. SolidIntegrity detail retained.** (Verdigris 008.) The kernel computes rule id, part,
  and applied repair; the CLI/session render a bare count. Amendment: surface the detail the
  kernel already holds.
- **A6. Batch same-class obstructions; errors carry the legal set.** (Maiden 004/005.)
  One-obstruction-per-round-trip on closed-enum mistakes (materials) costs N transactions per
  mistake class, and the diagnostic names the illegal value but not the allowed vocabulary.
  Amendment: same-class obstructions report together; closed-enum diagnostics enumerate the
  legal ids.

## Lane SOL-B — One acceptance oracle (codex, gpt-5.6-sol, xhigh)

Territory: `golem/cli/check.py` pipeline, assembly/exchange conduit surface
(`golem/kernel/assembly/`), shared oracle plumbing. Does not touch session verdict payload
shapes (SOL-A's ground).

- **B1. check ≡ session.** (Verdigris 005 — `check` ACCEPTED a spec `session` rejected with
  `EmptySurfaceConduitObstruction`.) Amendment per R1: the assembly-stage surfaces the
  transaction path runs also run under `check`; if any surface is genuinely
  transaction-only, `check` says so by name instead of staying silent.
- **B2. face_line honors face_normal, or says why not.** (Verdigris 005/006.) A buried host
  face yields a silently empty conduit because face_line ignores the declared normal.
  Amendment: honor the normal in face-line extraction, or emit a typed diagnostic naming the
  burial coupling — never an empty conduit with no explanation.

## Lane KIMI-A — Dialect ontology roles (omp, kimi-code/k3)

Territory: body/0.3 contract & intent (`golem/body/`, contract schema, intent channels),
compile-side role threading, `golem/senses/` expectation updates. Does not touch vascular
obstruction payloads or assembly conduits (sol's grounds).

- **K1. Decorative / transform-handle bone role.** (Verdigris 006/007.) Bones exist today
  only as perfused terminal organs; a waypoint bone added purely to reframe a child becomes
  a vascular organ demanding an exchange bed. Amendment per R3: an authorable bone role
  (`decorative` or `handle`) that exempts the bone from terminal-region/exchange-bed law
  while keeping frame composition; senses expectations updated to match.
- **K2. Non-carrier flesh role.** (Maiden 011, Verdigris 007.) Every flesh instance is a
  vascular carrier; thin membranes silently inflate ×2.2–3.3 to the 0.04 carrier floor.
  Amendment per R3: an authorable non-carrier flesh role that keeps authored dimensions and
  carries no conduits — with honest rejection (not inflation) if a route thereby lacks a
  lawful carrier.

## Lane KIMI-B — Teaching surface (omp, kimi-code/k3; queued behind KIMI-A, same brain)

Territory: contract/schema prose, CLI help text, `golem/cli/check.py` verbosity flags
(coordinate with SOL-B's landing; dispatch after both A-lanes merge).

- **K3. Hidden laws stated where authors look.** (Maiden SURFACE-MAP "teaching topology
  ~15%".) Half-extent semantics; closed 10-id material enum + palette-is-export-only;
  frame-composition law (rest_dir in parent frame, axial −z inversion); aerial-cover law for
  exchange-bed paths; `:negative:`/`:positive:` mirror vocabulary mapped to authored ids.
  All stated in the contract/schema the author actually reads. Per R4.
- **K4. Session protocol self-describing.** (Maiden 001/002, Verdigris 001.) Transaction
  envelope shape documented in `--help` with an exemplar; `check` truncation ("+32 more")
  gets a verbosity/JSON flag; accepted-verdict noise floor (148 expected diagnostics)
  filtered by default with the raw set behind a flag; stateless base-rotation discipline
  documented, with a warning when base == input.

## Wave-3b queue (after the above merge; not dispatched now)

- **Q1. Blame integration.** Wire `extract_blame` into session verdicts for well-posed
  obstruction classes; honest decline for the ill-posed taxonomy; group-MUS over
  limb+region+bed units.

  **R5 — seam ruling (fable).** Blame is a *post-verdict interrogation*, never part of
  `submit`: verdicts stay cheap and pure, and a blameless verdict is byte-identical to
  today's. Opt-in via `golem session ... --blame` (and only there), which attaches an
  optional `blame` section to the verdict JSON, keyed by obstruction signature. Each entry
  carries: blamed authored addresses, budget accounting (spent/limit/exhausted), and a
  well-posedness disposition — `localized`, or `declined: <taxon>` naming the ill-posedness
  taxon (coupled-unit, absence-degenerate, global-solve, pathological-fork) honestly instead
  of emitting garbage minimal sets. Group-MUS runs as a coarse pre-pass over
  limb+region+bed units, refining to declarations only inside implicated units. The
  section is strictly absent when the flag is absent (schema stability per R4's spirit).
- **Q2. Eval corpus recording.** Parse both trials' verdict sequences into
  `golem/evals/corpus/` receipts as post-spine ratchet datapoints (baseline: acceptance
  txn 10, repair rates 1/7 and 1/9).
- **Q3. Import hygiene.** Verify the eager-skimage import that broke bare `--help` outside
  the venv is cured; cure it if not.

---

## Wave 3c — Naturalness campaign (authorized 2026-07-22; reference: crystalline drake image)

**R6 — The author must see what she authored.** `appearance_palette` being export-path-only
while the verify loop renders catalogue clay is the revalidation disease on the appearance
channel: authored appearance is invisible to the authoring loop. `golem look` honors the
palette. Sight precedes vocabulary — no material-algebra expansion ships before the
renderer can show it.

**R7 — Sharpness is vocabulary, not a defect.** The global `blend` field must not be the
only voice on form transitions. A crease/per-part-blend declaration is opt-in dialect: a
spec that declares none compiles byte-identical (same discipline as roles). Strict decode;
senses' GLOBAL blend-scale reporting stays honest about overrides.

### Lane SOL-C — Open the eye (codex, xhigh)

Territory: `golem/cli/look.py`, `golem/kernel/body/eyes.py`, `golem/materials/projection.py`
/ `surface.py`, render tests. No decode/schema changes.

- **C1.** `golem look` honors `appearance_palette` for the existing 10-id vocabulary in all
  views (`--view all`, shaded and flat); catalogue recipes remain the fallback when no
  palette entry exists. Verified against `specs/scree_maiden.json` (moss/slate must finally
  be visible) and `specs/verdigris.json`.

### Lane KIMI-C — Form fundamentals (omp, kimi-code/k3)

Territory: `golem/kernel/body/` (kinematics, geometry, types, validate), the SDF blend
evaluation path, `golem/senses/proprio/render.py`, their tests. Forbidden: `golem/contract/`
and `golem/cli/` (KIMI-B live there), `golem/materials/`, anatomy vascular, session.

- **K5. Kill the record-rebuild bug factory.** Bone records are rebuilt through key
  whitelists in `kinematics.py` AND `geometry.py::fk_table`; new dialect fields silently
  vanish unless threaded twice. One canonical bone-record projection both consumers draw
  from — a field is threaded once or fails loudly.
- **K6. Crease vocabulary (per R7).** Per-part blend override or junction crease
  declaration so body/0.3 can say "sharp"; papercraft/crystalline forms stop being sanded
  into pebbles. Opt-in, byte-identical when absent.
- **K7. Advisory-as-edits lie.** `suggested_scale` rows render under the word "edits" with
  no consumer; rename the rendering to what it is (suggestions), per R4.

**R8 — Material classes, not a longer enum (fable design, governs D1).** The vocabulary
grows by *class*, closed and typed: `opaque` (today's 10 catalogue ids become presets of
it), `emissive` (linear-rgb color + intensity), `translucent` (opacity in [0,1]). Strict
decode; unknown class or malformed param is a typed rejection naming the legal set. Threading
is decode → look shading → export, one seam each: in `look`, emissive surfaces are
shading-exempt (rendered at full authored luminance) and translucent surfaces composite
with authored opacity — honest orthographic receipts, not a lighting engine; in export,
emissive maps to glTF emissive factors and translucent to alpha-blend mode. A spec
declaring no material classes compiles and renders byte-identical. Sight and safe
field-threading (R6, K5) are landed prerequisites.

### Queued 3c-b (after SOL-C + KIMI-C land)

- **D1 (sol). Material classes per fable design:** closed *class* vocabulary — opaque
  (presets = today's 10 ids), emissive (color, intensity), translucent (opacity) — with
  typed parameters; strict decode; threaded decode → look shading → export. Needs R6 lane
  landed (sight) and K5 landed (field threading is safe).
- **D2 (Kimi). Protrusion/growth vocabulary:** repeated sharp growths emitted along a bone
  or surface line (dorsal crystal ridges); composes with K6 creases and D1 emissive.
### Trial 3 — the breakthrough measurement (fires when D2 lands)

The amendments are claims until an author survives them. Trial 3 commissions the reference
creature itself — the crystalline drake (obsidian hide, emissive mouth/eye-trenches,
translucent membrane wings, sharp dorsal crystal ridges) — authored cold against the
amended surface, using everything wave 3 built: door-naming rejections, `--blame`,
crease operators, roles, visible palette, emissive/translucent classes, protrusions.

Measured against the ratchet corpus (baselines: golden biped txn 10, Maiden ~16 rounds
equivalent, Verdigris txn 11; repair rates 1/7–1/9 pre-spine):

- rounds-to-acceptance — target: meaningfully below Verdigris's 11.
- repair-rate per obstruction code — door-naming verdicts + blame should push repairs
  toward first-attempt.
- friction journal — count of "had to read kernel source to proceed" events; Maiden had
  many, target approaches zero (the teaching surface now carries the laws).
- Recorded into `golem/evals/corpus/` like its siblings; honest unavailability where the
  transcript is incomplete.

Deliverable to Rosalia: look receipts in authored palette AND the GLB export viewed in a
real renderer (Blender) — emissive and translucency only sing off-receipt.

---

## Wave 4 — Morphology breakthrough (designed 2026-07-22; dispatch after 3c-b lands)

The audit finding: appearance classes (3c) are paint; two GEOMETRY citizens gate the
reference creature. One is a promotion, one is genuinely new.

**R9 — Carve is vocabulary, not an eyes privilege (mouths, nostrils, sockets).** The engine
already owns subtraction (`sdf_difference`, `graph.carves`, mirrored carve instancing) and
the eyes subsystem proves the entire law stack tolerates cavities — solid integrity,
anomaly, vascular all pass over carved sockets today. Amendment: an authored carve role on
parts (strict decode, closed to the same mirror vocabulary), with the laws generalized from
the eyes precedent: vascular clearance treats cavity walls as walls; integrity stays
watertight-aware; senses learn "authored cavity ≠ missing flesh." Articulated jaws already
exist (joint dof); carve + crease lips + emissive interior = the mouth.

**R10 — Spanning flesh: the membrane citizen (wings, frills, flukes, dewlaps).** Every
flesh kind (gencyl/loft/blob) anchors to ONE bone; `span` is a range along it. A membrane
is a surface between TWO OR MORE bone curves, evaluated after FK so pose folds it for
free. This is the only genuinely new geometry citizen, and all its prerequisites landed in
wave 3: `non_carrier` thinness (K2), `crease` edges (K6), translucency (D1), canonical
record threading (K5). Fable's ownership rulings, binding on the design:
  - A spanning flesh instance is owned by a NEW multi-anchor record (not duplicated into
    each bone's record); its anchors are ordered bone references, and mirroring mirrors the
    anchor tuple atomically.
  - Senses attribute its anomalies to the instance itself, naming all anchors — never
    arbitrarily to one bone.
  - The aerial-cover law extends: a web IS flesh topology, so exchange-bed and vessel
    paths may ride it (wing venation as lawful vasculature) — carrier eligibility governed
    by the same role vocabulary as any flesh (`non_carrier` webs carry nothing).
  - Absent from a spec ⇒ byte-identical compile (standing discipline).

Sequencing: R9 (carve promotion) first — smaller, proves law generalization; then R10.
Executor split per the standing routing: fable seam design (done, above), Kimi decode/
vocabulary + senses semantics, sol engine/assembly plumbing and law generalization —
file territories partitioned at dispatch time. Trial 3 moves AFTER wave 4: the reference
drake needs membranes and a mouth, not just their paint.

- **D3 (fable ruling, executor TBD). `check --format=json` is DEFERRED, not declined.**
  Sol's takeover report framed the omission as "no canonical JSON owner for the mixed
  receipt surface" — that is a description of the disease, not a reason. The owner is the
  transaction verdict: `check` already consumes the session-owned acceptance evaluation
  (R1); when its receipt becomes a *rendering* of that structured verdict, JSON output
  falls out for free instead of being a second serialization to maintain. Sequenced after
  the prose receipt is fully derivable from verdict/2 — never as a parallel format.
