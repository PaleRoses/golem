# GOLEM Phase -1: frozen-pilot revision trial

## Decision under test

This protocol tests one narrow claim before any GOLEM kernel infrastructure is
founded: within one bounded stylized family, does the frozen symbolic pilot retain
blind aesthetic parity with a strong procedural-DCC agent while making realistic
revisions substantially cheaper and less destructive?

The primary estimand is **warm-start revision economics**. Historical work spent
creating the Python pilots is unknown and is therefore `NotMeasured`, not silently
priced at zero. Time spent producing and qualifying the Blender master is measured
separately and is excluded from the primary revision-time ratio. Both figures remain
visible in the final receipt.

Passing this protocol licenses investigation of kernel Phases 0-1. It does not prove
generality beyond this family, pack accretion, riggability, or engine readiness.
The three sealed-output obstructions recorded in `../CONFORMANCE.md` remain failures
of the v0.2 replay claim, but they do not answer the commercial revision question and
therefore do not invalidate this trial. Any *additional* oracle mismatch does.

## Existing semantic owners

Phase -1 reuses the frozen pilots directly:

- `../pilots/engine.py` owns the `gencyl`/`blob` part graph, SDF composition,
  meshing, and coherence report.
- `../pilots/golem_hands.json` is the warm-start family master.
- `../pilots/hand.py` and `../pilots/hand_pack.py` own hand compilation and the
  20-to-0 hand oracle.
- `../pilots/mount_hands.py` owns the recorded two-sided wrist composition.
- `../pilots/plates.py` owns the fixed seed-7, 110-cell appearance pass.
- `../pilots/render.py` owns the deterministic pilot render.
- `../pilots/judge.py` owns occupancy components, shell/watertightness, and
  symmetry-IoU measurement.

The experiment adds no creature IR, pack engine, solver, runner, adapter, or result
service. The manifest is experiment metadata, not another kernel. Pilot source and
golden artifacts are hash-pinned in `manifest.json` and MUST NOT change during
calibration or held-out work.

## Arms

### Frozen GOLEM pilot

The operator may edit only trial-local copies of the existing body/hand JSON dialect
and invoke the frozen pilots. It may not edit pilot source, add a new part kind,
hand-author vertices, or repair an exported mesh in a DCC. A request that cannot be
expressed is `UnsupportedVocabulary`; a request that requires a vertex/edge/face or
sculpt edit is `NonSymbolicEditRequired`. Both are measured failures, not excuses to
change the substrate mid-trial.

### Blender procedural-agent baseline

The baseline is Blender **5.1.1**, with the exact build identity captured before the
first session. It may use the full native surface: Geometry Nodes, modifiers, mirror,
voxel remesh, booleans, sculpt, Python, materials, and a persistent `.blend` master.
Direct mesh work is allowed and timed. This is deliberately not the old loose-
primitive control.

Before held-out briefs are revealed, the Blender master MUST pass both calibration
creatures: all six calibration revisions within the ordinary revision budget, at
least six named procedural control groups, one watertight output per calibration
creature, symmetry IoU at least 0.98, and mean appeal and brief-fidelity scores of at
least 5.0/7 from at least two independent calibration judges per creature. Those
judges did not author the baseline, and their receipts bind the exact final mesh,
view sheet, and turntable hashes. Failure is
`BaselineQualificationFailed`, making the experiment `Inconclusive`; a crippled
baseline cannot donate a victory to GOLEM.

Calibration repair/setup time is recorded separately. The qualified `.blend`, exact
build identity, model identity/version, and content-addressed tool-instruction,
sampling-setting, and operator-prompt source artifacts are then frozen.

## Site, cover, and held-out split

The site is the bounded **Obsidian Wardens** family: bilateral, faceless, plated,
five-digit golems with the existing family-master part inventory. Hair, wings,
membranes, articulated animation, detailed faces, new materials, and new part kinds
are outside the cover.

The eight local contexts are fixed in `manifest.json`:

- calibration: `bulwark`, `climber`;
- held out: `sentinel`, `delver`, `lancer`, `porter`, `strider`, `oracle`.

Every creature has one initial brief and exactly three sealed revisions. Initial
briefs carry non-gating post-run `identityCriteria`; revisions carry gating target
and preserve clauses. Family-integrity clauses apply at every stage. A revision
descends only when its target clauses pass and its restrictions onto all preserved
regions agree; disagreement is an addressed failed clause, never a cosmetic success.

`manifest.json.family.metricSemantics` is the closed measurement dictionary for
those clauses. A metric ID without exactly one definition is a contract failure;
operators and judges do not invent measurement meanings during the run.

Calibration briefs may be seen before the tool surfaces freeze. Held-out briefs and
revision text remain sealed until the corresponding prior artifact has been hashed.
Calibration scores never enter decision gates. Family-rank identity criteria are
computed only after all eight initial artifacts exist; they are diagnostics for the
lineup, never initial-stage completion conditions and never prerequisites for
baseline qualification.

## Session and contamination controls

1. Both arms use the same model identity, model version, sampling parameters, and
   non-tool system instructions. Only tool-specific operating instructions differ.
2. A fresh model session is opened for each creature/arm. Its initial build and three
   revisions remain in that session; no context crosses creatures or arms.
3. The arms never see each other's sources, artifacts, receipts, conversations, or
   measurements before the verdict is sealed.
4. Held-out arm order is paired and counterbalanced exactly as declared in the
   manifest: three creatures run GOLEM first and three Blender first.
5. Heavy arm runs do not overlap. They use the same workstation; hardware, library,
   Blender build, model identity/version, content-addressed prompt and sampling
   artifacts, and start/end timestamps are evidence.
6. A revision is released only after the prior source and visible artifacts are
   content-hashed. If a prior stage is obstructed, every dependent slot is recorded
   as `BlockedByPriorStage { priorSlot }` without metrics or fabricated artifacts.
   No retroactive repair is admitted.
7. Human concierge work is mechanical only. Any human-authored geometric decision,
   parameter value, node arrangement, or mesh edit is timed as human intervention.
8. Pack or vocabulary ideas discovered during held-out work enter quarantine records.
   They cannot affect any trial. Offline pack-proposal authoring begins only after all
   held-out artifacts are frozen.
9. The blind-code mapping is held by a custodian who does not judge. Judges cannot
   access the manifest, construction logs, source filenames, or mapping until their
   signed evidence is frozen.

## Budgets and stage completion

- Initial creature: 1,800 wall-clock seconds and at most six render-feedback rounds.
- Each revision: 720 wall-clock seconds and at most three render-feedback rounds.
- Solver, render, model, and tool latency remain inside wall time. Paused external
  outages require an addressed `ResourceContention` or `ProtocolDeviation`; they are
  not subtracted by recollection.

Each trial is the closed product `initial x r1 x r2 x r3`; duplicate or invented
stage IDs do not exist. The schema caps initial/revision wall time and render rounds
at their distinct budgets. Every attempted stage records active, human-authoring,
and direct-mesh-work seconds; a downstream blocked slot records only its typed cause.
A stage is `Completed` only when its outcome owns
content-addressed source/artifacts and a measurement-sidecar ID. Obstructed outcomes
never fabricate an artifact. `TimedOut`, `UnsupportedVocabulary`,
`NonSymbolicEditRequired`, and `ValidationFailed` count against decision gates.
`OracleMismatch`, `BlindingBreach`, typed `MissingEvidence`, missing measurement, or
an unqualified baseline produce `Inconclusive` instead of an invented score.

## Analysis arithmetic

The post-run analyst follows these rules without discretionary substitutions:

- Average the three-or-more judge scores per blind artifact. For each held-out
  creature subtract the Blender artifact mean from the GOLEM artifact mean, then
  average the six paired differences for appeal, brief fidelity, rigging readiness,
  and engine readiness. `NotAssessable` and a missing final artifact score 1; they
  are not dropped.
- A lineup identification succeeds when a strict majority of judges select the
  artifact's actual creature. Count successes only for GOLEM held-out finals; a
  missing final is incorrect.
- Pair revision wall times by creature and revision ID. Compute the median and the
  nearest-rank p75 of the eighteen GOLEM/Blender ratios: sort ascending and select
  the fourteenth value. Every outcome uses its positive recorded wall time; timeouts
  and other failed outcomes remain in the population. A `BlockedByPriorStage`
  revision has no attempted-stage metrics and contributes the full 720-second
  revision budget to this arithmetic.
- For cleanup, first take the median judge estimate per artifact, then the median
  across the six creatures per arm. Missing GOLEM finals use 2,700 seconds. A
  missing Blender final is a validity failure and makes the experiment
  `Inconclusive`; it is never silently imputed. The cleanup disadvantage is GOLEM
  median minus Blender median. An addressed obstructed GOLEM outcome is an observed
  absence and receives the conservative imputations above; an absent trial receipt
  is `MissingEvidence` and therefore `Inconclusive`.
- A failed family-integrity clause is critical. A revision is one noncritical
  regression if any of its revision-specific preserve clauses fails, regardless of
  how many such clauses fail. A target-clause failure is not a regression; it makes
  that revision incomplete.
- Human burden, direct-mesh-work seconds, and authoritative mesh-edit counts sum
  held-out revision stages per arm; a human-authored revision is one distinct slot
  with `humanInterventions > 0` or `humanAuthoringSeconds > 0`. The GOLEM decision
  gates use its own totals.
- Count typed `NonSymbolicEditRequired` outcomes by distinct held-out revision slot.
  Separately, a final artifact requires mesh surgery when a strict majority of its
  technical judges say so; count those artifacts by revealed GOLEM blind ID. A final
  diagnosis is never retroactively attributed to a revision.
- Vocabulary count is the number of distinct quarantined proposal IDs. Proposal
  authoring time is summed globally and per originating creature; the per-creature
  median includes zeros. Calibration/setup costs remain separately reported.

## Common evaluation and blind judging

Before held-out work, the coordinator freezes one arm-neutral Blender evaluation
contract: the exact build identity, `.blend` scene, import/render script, four camera
IDs, 1024-square resolution, and 36-frame turntable. Its content-addressed
`EvaluationContractEvidence` is referenced by every trial and by the blind-map
commitment. Both arms' exported meshes are imported into that scene with the same
crop, lighting, background, and Obsidian Warden material. `pilots/render.py` remains
an oracle owner; it is not misrepresented as an arbitrary-mesh comparison renderer.
Generator metadata and filenames are stripped from judge-visible copies. Geometry is
not repaired or normalized beyond reversible presentation transforms.

Before blinding, each arm emits the common measurement sidecar declared in the
manifest: singleton surfaces, separate left/right surfaces for every bilateral
scope, ten separately surfaced digit chains, and the closed junction/contact
landmark product. Aggregate bilateral surfaces are derived views, never evidence.
Finger chains contain four points and four radii; thumb chains contain three of each,
matching the frozen three-phalange/finger and two-phalange/thumb grammar.
`ground_contact` is not counterfeit geometry: the sidecar records the preregistered
`groundPlaneY = 0.02` separately.

Surface samples are verifier-owned, never arm-authored. For each singleton or named
instance surface the verifier canonicalizes and lexicographically sorts float64
triangles, then emits exactly 4,096 area-weighted sqrt-barycentric samples using
NumPy 2.5.1 PCG64 with seed `7319 + 100 * scopeOrdinal + instanceOrdinal`, exactly
as frozen in `measurementContract`. Geometry
comparison normalizes only by the prior-stage scoped bounding-box diagonal.

Metric descent is fixed. Remove `ground_contact` from the surface cover. For
`all_scopes`, evaluate every remaining scope independently in manifest order. For
`combined`, concatenate the declared scope instances first and emit one result. For
`relational_pair`, require exactly two non-ground scopes and emit the current
left/right junction-distance scalars between their corresponding named instances.
Definitions naming bounding boxes or surface samples read the singleton surface or
each exact bilateral/digit instance surface independently; definitions naming
centerlines, radii, endpoints, or landmarks read the same named instances/product.
A metric emits the tagged shape frozen in the manifest:
`Scalars` preserves scalar/component order, while `Vec3s` preserves each vector as a
vector rather than laundering it into three unrelated numbers.

Operator descent is also fixed. `equal`, `minimum`, and `maximum` use the current
stage. `increase`, `increase_percent`, `decrease_percent`, `max_delta_percent`,
`max_distance`, and `max_angle` compare with the immediately preceding stage at the
same scope/instance keys. The non-gating `family_rank_desc` identity criteria compare
the eight initial artifacts only after construction and rank the arithmetic mean of
each metric's scalar values, descending. `equal` uses
absolute error <= tolerance; minima/maxima admit tolerance on the failing side;
increase/decrease targets admit absolute target error <= tolerance; maximum-delta,
distance, and angle admit value + tolerance; rank passes when its absolute distance
from the requested rank is <= tolerance. Percent comparison with a zero reference
is `MissingMeasurement`, never infinity. Scalar operators apply componentwise;
`max_distance` takes Euclidean distance between corresponding `Vec3s`, and
`max_angle` takes their unsigned angle. Every component/pair must pass. Thus a union
cannot hide a failed preserved scope and a vector cannot pass by componentwise fraud.

An independent verifier checks all landmarks and instances against the visible mesh
and evaluates clause metrics using only those output-geometry definitions. Clause
results may reference only the concrete paths enumerated in `schema.json`; an
invented array index is invalid evidence. GOLEM source parameters and Blender
control values are forbidden as paired measurements. A missing, invalid, or
source-only sidecar is `MissingMeasurement`, not a guessed score.

Three independent technical artists, none of whom authored the protocol or baseline,
judge the twelve held-out final artifacts in randomized order. After artifact hashes
freeze, the non-judging custodian draws a seed and applies the manifest's exact PCG64
permutation to the canonical held-out-creature x arm order. The ordered mapping and
seed are sealed into the exact `p01`-through-`p12` product as immutable
`BlindingCommitmentEvidence`. Every slot binds the final mesh, four-view sheet, and
turntable SHA-256 separately, gluing the visual and technical surfaces to the same
completed outcome. A separate
`BlindingRevealEvidence` links the receipts only after every judgment is frozen.

1. **Visual-only phase:** four views and turntable only. Each judge selects one of all
   eight lineup descriptions and records appeal (1-7), brief fidelity (1-7), and
   confidence (1-5) in a write-once `VisualJudgeEvidence` receipt.
2. **Technical phase:** only after that visual receipt is sealed, the judge receives
   the anonymous static mesh. A separate `TechnicalJudgeEvidence` links the visual
   receipt and records estimated static-mesh cleanup seconds, whether
   direct mesh surgery is required, blocking static defects, rigging-readiness score
   and estimated preparation time, and engine-readiness score and estimated
   preparation time. `NotAssessable` is a typed answer and scores as 1 for the paired
   comparison. These are explicitly expert estimates, not production receipts: a
   GLB existing proves serialization, while an opinion about topology does not prove
   a rig or engine round trip.

Blind technical receipts contain no arm-specific proposal link: that would reveal
the arm or require mutating sealed evidence. After reveal, every GOLEM technical
receipt with `meshSurgeryRequired = true` MUST appear in the `sourceEvidenceIds` of
at least one `QuarantineProposalEvidence` on the existing
`propose(vocab | rewrite, payload) -> quarantine` slow path; an uncovered diagnosis
is `MissingEvidence`. Blender diagnoses create no counterfeit GOLEM proposal. The
mesh edit never becomes authoritative state; the post-reveal symbolic proposal and
authoring cost remain separately inspectable.

## Exact verdict gates

Validity gates fail to `Inconclusive`; decision gates fail to `Failed`. `Passed`
requires every decision criterion below after both validity gates pass.

### Validity

- Semantic oracle mismatch count = 0, including: hand v1 = 20 violations, hand v2 =
  0, plate complex = 110 cells / 324 adjacencies, control = 35 shells / one occupied
  solid / not watertight, and v4 = one post-dust component / watertight. The three
  preregistered sealed-output obstructions remain reported separately; any new
  sealed mismatch is an `OracleMismatch`.
- Blender qualification: 6/6 calibration revisions completed within budget, at least
  six named procedural control groups, two watertight calibration finals, and minimum
  calibration symmetry IoU >= 0.98; at least two independent judges per creature
  give the Blender calibration finals mean appeal >= 5.0 and mean brief fidelity >=
  5.0.
- Held-out baseline completeness: all eighteen Blender revision-stage receipts and
  exactly six Blender final artifacts exist. A missing baseline observation makes
  the experiment `Inconclusive`, not conveniently favorable to either arm.

### Held-out decision

- Completion: all 18/18 GOLEM revisions completed and exactly six GOLEM final
  artifacts recorded.
- Aesthetic parity: GOLEM-Blender paired mean >= -0.5 on both 1-7 appeal and brief-
  fidelity scales; majority lineup identification correct for at least 5/6 GOLEM
  creatures.
- Production-assessment guard: GOLEM-Blender paired mean >= -1.0 on both 1-7 expert
  rigging-readiness and engine-readiness estimates. This remains an assessment, not
  a claim that either production pipeline ran.
- Revision economy: GOLEM/Blender median wall-time ratio <= 0.50 and p75 ratio <=
  0.67.
- Integrity: all six GOLEM finals pass component=1, watertight, and symmetry-IoU >=
  0.98.
- Regression: zero critical regressions and at most two held-out revisions with any
  noncritical preserve-clause failure.
- Human burden: at most two human-authored revisions and at most 1,800 total human-
  authoring seconds.
- Mesh surgery: zero direct mesh edits in an authoritative GOLEM result, zero GOLEM
  revision slots ending `NonSymbolicEditRequired`, and at most two GOLEM final
  artifacts whose judges require direct mesh surgery by strict majority.
- Cleanup: median GOLEM static cleanup estimate <= 2,700 seconds and no more than 900
  seconds worse than Blender's median.
- Accretion burden: at most two distinct vocabulary proposals; post-run offline pack-
  proposal authoring <= 7,200 seconds total and median <= 900 seconds per creature.

## Evidence and authority

`schema.json` is Draft 2020-12 and defines manifests, oracle receipts, trials,
measurement sidecars, the shared evaluation contract, separate visual/technical
judge receipts, separate blind commitment/reveal receipts, baseline qualification,
quarantine proposals, and typed experiment obstructions as closed tagged
products/sums.
Every run file records `manifestSha256`; every referenced artifact records its own
SHA-256. Evidence files are write-once: a correction creates a new evidence ID and
sets `supersedesEvidenceId`. Files are never edited in place.

The schema validates local evidence shape only. It does not compute a verdict or
pretend to prove cross-file referential integrity. After all external observations
exist, an independent analyst MUST manually audit IDs, timestamps, hashes, stage and
clause references, blinding completeness, and arm compatibility, then calculate the
preregistered gates above in a signed analysis report. No `verdict.json` type or
result ships now; founding a scoring engine before one real run would be precisely
the infrastructure-first error this phase exists to prevent.

No run evidence or verdict ships with this protocol. Empty scorecards would be
decoration; fabricated outcomes would be fraud.
