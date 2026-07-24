# GOLEM Architecture — Subsystem Ownership Map

Purpose: prevent re-invention. Every capability below already exists at the
cited location. Before writing new machinery, find its owner here and extend
through the owner's seam. One owner per surface; derived views stay derived;
new vocabulary is opt-in and its absence compiles byte-identically.

How to read: each entry states what the package OWNS (its single authority),
its TYPED SURFACE (the verdict/obstruction families it emits), its SEAMS
(who consumes it and through what boundary), and DO-NOT-REBUILD (existing
machinery a newcomer would plausibly duplicate). `file.py:NN` citations are
line anchors; symbols are authoritative.

**IN FLIGHT — do not deep-document or re-architect these internals.** The
*Surface-Formation Adoption Docket* (trial-3-era campaign,
`docs/plans/surface-formation-adoption-docket.md`) owns three open fronts:
(1) the engine's union composition law (`kernel/engine/compile.py` composition
seam, `algebra.py` — legacy `smin` stays the byte-identical default for
undeclared specs); (2) radius-faithful skeleton-integral muscle formation
(`kernel/body/myology.py` remains the authoring owner; the solver is new);
(3) skin as its own solved layer. Their obstruction taxonomies
(`MuscleFormationObstruction`, `SkinRelaxationObstruction`, …) are docket
law, not yet code. Extend them only through the docket.

---

## Leaf utilities

### golem/paths.py, golem/json_value.py, golem/goldentext.py

- **OWNS.**
  - `paths.py` — all repo-geometry constants (`KERNEL_ROOT`, `PILOTS`,
    `REHEARSAL`, `OUTPUTS`, `SPECS`, `GOLDEN`, `paths.py:16-21`) and the
    *single* quarantined pilots bootstrap: it inserts `PILOTS` into
    `sys.path` once (`paths.py:23-24`); no other module may mutate `sys.path`.
    Imported for side effect by `golem/__init__.py:19`.
  - `json_value.py` — the canonical recursive JSON-value predicate
    `is_json_value` (`json_value.py:6-18`): finite floats, string keys,
    recursive lists/dicts; everything else rejected.
  - `goldentext.py` — the byte-exact write-once golden harness:
    `assert_matches_golden` (`goldentext.py:36-67`), `GoldenMissing`,
    creation gated on `GOLDEN_CREATE=1`. Existing goldens are *never*
    overwritten; mismatch is a hard `AssertionError` with byte diff.
- **Typed surface.** `GoldenMissing` refusal; otherwise predicates, not verdicts.
- **Seams.** `paths` is consumed by assembly/ingest, cli, contract, body/emit,
  senses harnesses, session/state, evals. `json_value` by `cli/spec.py:8` and
  `session/ops.py:28`. `goldentext` by the session golden tests.
- **DO-NOT-REBUILD.** A second repo-root locator, a second JSON-value
  definition, a goldens "update mode" — the write-once discipline *is* the
  byte-exactness pin the whole tree leans on.

### golem/addressing

- **OWNS.** The three total address grammars and their immutable carriers:
  parametric bone anchors (`<bone_id>:<t>`, `anchor_grammar.py`, bounds
  `[-0.5, 1.5]`, default `t=1.0`), assertion/qualified scopes (`scope.py`
  closed `ScopeKind` — whole/world/part/bone/landmark/chain/contact/element/
  port/mount/region — parsed by `scope_grammar.parse_scope`:29), and session
  addresses (URL-encoded segments + fieldpath, `session_grammar.parse_address`:41).
  `cell.py` owns spatial cell identity: `CellId`, `GridDims`, row-major
  `flatten`/`unflatten`.
- **Typed surface.** `AnchorObstruction`/`RejectedAnchor`, `ScopeObstruction`/
  `RejectedScope`, `AddressObstruction`/`RejectedAddress` — every grammar is
  total: malformed input returns a typed rejection, never throws.
- **Seams.** `core.py` is a compatibility facade re-exporting all three
  grammars (no semantics of its own). Consumers: kernel/body (goals, myology,
  relations, geometry, kinematics), contracts (clauses/metrics/views),
  session (ops/diff/coalgebra/outline/protocol), mechanics/model,
  assembly/project, senses/proprio/fusion.
- **DO-NOT-REBUILD.** The assertion-vs-qualified token dispatch
  (`scope_grammar.py:103-170`), encoded-segment preservation in session
  addresses (`session_grammar.py:124-160`), anchor range/default law, the
  row-major cell-index algebra. Parse through these grammars; never hand-roll
  string splitting on addresses.

---

## kernel/ — solvers and compilers

### golem/kernel/engine — the sole production authority for part-graph geometry

- **OWNS.** The closed geometry graph contract and its field realization:
  primitives (`GencylPart`, `BlobPart`, `BoxPart`, `SpanningWeb`,
  `types.py:197-258`), the closed operator vocabulary `CompositionOperator =
  BLEND | CHAMFER | CREASE` (`types.py:152-156`), `GeometryGraph(blend, parts,
  webs, carves)` (`types.py:261-266`), analytic SDFs (`algebra.py`: `sdf_gencyl`,
  `sdf_blob`, `sdf_box`, `smin`, `compose_union`, `sdf_difference`), mirrored
  instance expansion (`part_instances`, `carve_part_instances`,
  `web_instances`, `algebra.py:169-179`), and mesh lowering with a
  content-addressed field cache (`compile.py:224-310,370-545`).
- **Typed surface.** `GeometryObstruction` / `GeometryRule` (`types.py:100-141`)
  covering composition, profile, quaternion, web, feature, and malformed
  rules; `decode_graph` (`types.py:1186`) → `Accepted`/`Rejected`;
  `require_accepted` raises `GeometryDecodeFailure` (`types.py:1261`).
- **Seams.** Consumed by kernel/body (compile, myology, webs), kernel/anatomy
  (geometry, material/carve), assembly, plates/projection, conduits/surface,
  senses/proprio/fusion, session/ops+protocol, kernel/blame/verdict,
  contract/primer. Field composition seam: `sample_graph_field_checked`
  (`compile.py:483`) unions part+web instances via `compose_union` then
  subtracts carve instances via `subtract_carve_field` (`compile.py:528`).
- **DO-NOT-REBUILD.** SDF primitives, the blend/chamfer/crease vocabulary,
  carve subtraction (`sdf_difference` + mirrored `carve_part_instances`),
  prepared-web SDF (`algebra.py:343-376`), the field cache key/salt/prune
  machinery. **IN FLIGHT:** the union composition law is Phase 1 of the
  Surface-Formation Adoption Docket — opt-in operators join the vocabulary;
  undeclared specs keep the legacy `smin` path byte-identically.
- **Status.** The `golem/__init__.py` layout labels this vocabulary
  *quarantined*; the frozen pilots (`engine`, `render`, `hand`, `judge`)
  remain plain top-level modules made resolvable by `paths.py`.

### golem/kernel/body — the body/0.3 stance compiler

- **OWNS.** One deterministic pass from skeleton spec to flat part-graph +
  embedded M2 intent block + compile-receipt sidecar (`compile.py` `Compiler`,
  `compile_file`:299-510). Bones are pure scaffolding; flesh attaches in
  bone-local coordinates. Phases: validate → gencyl/loft stations → kinematics
  (rest frames, pose, FK, landmarks, ground-lift) → reach-goal IK → relations
  → anatomy bridge → emit → intent/receipt.
- **Typed surface.** `BodyCompileResult = CompiledBody | RejectedBody`;
  `BodyObstruction` union (`types.py:610-697`): spec-decode (`types.py:63-114`),
  relation (`relations.py:125-243` — fixed-placement conflict, underconstrained,
  solve-exhausted, branch budget), muscle (`types.py:264-397`), carve
  (`types.py:429-452`), web (`types.py:531-581`), eye (`types.py:635-676`),
  loft (`types.py:117-164`). Solve evidence: `RelationalSolveReceipt`,
  `LocalSolveReceipt` (`relation_solve.py:30-122`).
- **Seams.** `assembly/ingest.py:27` consumes `Compiler`/`CompiledBody`;
  kernel/blame/verdict runs it as an oracle stage; cli/spec, session,
  conduits/operator pin its `DIALECT`; anatomy consumes `BONE_ROLES` and
  gencyl types; `anatomy_bridge.py` is the lazy (no-eager-import) boundary
  *into* kernel/anatomy. Contract evaluation runs through `report.py`
  (`run_typed_asserts`:17-31 → senses/proprio + contracts/verdicts).
- **DO-NOT-REBUILD.**
  - **Canonical bone-record projection** — `records.py:project_bone_record:46-82`
    with `KNOWN_BONE_KEYS` + `UnthreadedBoneFieldError`. Both record
    constructors (kinematics FK and geometry `fk_table`) draw from it; thread
    any new dialect field *here*, never in a constructor whitelist.
  - **Attach topology** — `intent.py`: `_derive_attach`:109-151 (skeletal,
    mount, prop, web topology) + `_authored_attach`:155 (declared attach
    pairs, additive) + `_declared_roles`:67-90.
  - **Myology** — muscle declaration ADTs and machinery (`myology.py`:
    `BaseMuscleDeclaration`/`NaturalMuscleDeclaration`/`MirroredMuscleDeclaration`,
    anchor/bulk/profile resolution, mirror tolerance, `compile_myology`).
    **IN FLIGHT:** myology's declaration descent is the authoring seam for
    the docket's Phase 2 muscle formation; the solver lands elsewhere.
  - **Closed vocabularies** — carves (`CarveKind` gencyl/blob/box,
    `carves.py:compile_carves`:349), webs (multi-anchor spanning membranes,
    `webs.py:compile_webs`), eyes (socket/brow/material laws,
    `eyes.py:compile_eyes`:505, `eye_globe_graph`:562), relations
    (`relations.py:24-119` kinds/aliases/policies; `relation_solve.py`
    residual algebra + progressive deepening), roles (`types.py:51-60`).
  - **Numeric floor** — `canonical.py` byte-freezable rounding, `linalg.py`
    rotation/frame/quaternion primitives, `gencyl.py` loft/station validation,
    `goals.py` two-bone analytic IK, `decode.py` shared strict-decode helpers.
  - **Projection** — `project.py:project_obstruction`:681 total obstruction →
    text/JSON projection. Every new obstruction family joins it.
- **Attribution caveat.** The anomaly classifier (`blend_ambiguity`,
  `cross_plane_fusion`) and posture/support margins are **senses/proprio**,
  not body. Body supplies intent topology; proprio classifies.

### golem/kernel/anatomy — vascular feasibility and circuit realization live HERE

- **OWNS.** The `anatomy/0.1` whole-body descent (`descent.py:derive_anatomy` —
  regions, pump, exchange beds, myotendinous paths, carrier rows) and the
  entire closed vasculature pipeline: realization (`realize/`), physical
  hydraulics (`hydraulics/`), and vascular material masks (`material/`).
  Circulation services tissue but never dictates the exterior.
- **Typed surface.** `AnatomyInputObstruction`/`AnatomyObstruction`
  (`vocabulary.py`); `VasculatureObstruction` (`graph.py` —
  `InfeasibleBifurcationObstruction`, `VascularSearchBudgetObstruction`,
  capsule escape/intersection, symmetry mismatch, gluing, solver residual,
  delivery mismatch, reversed flow); `PhysicalHydraulicsObstruction`
  (`hydraulics/types.py`); `VascularMaterialObstruction` (`material/types.py`).
  Accepted/Rejected result pairs per stratum; `VascularFlowReceipt`.
- **Seams.** kernel/body reaches it only through `anatomy_bridge.py` (lazy).
  assembly consumes accepted products broadly (carriers, coupled, hydraulics,
  thermal, mechanics, mounts). cli/check imports its verdict types.
  `project.py:project_anatomy_obstruction/project_vascular_obstruction` is
  the projection boundary.
- **DO-NOT-REBUILD.** The `realize/` pipeline end-to-end: terminal allocation
  with Sobol sites + content-addressed store and `lru_cache`
  (`allocation.py`), the `CircuitConstructor` seam + `GreedyCCOConstructor`
  (`constructor.py`), staged CCO search (`cco.py`), bifurcation feasibility
  predicates (`bifurcation.py`), macro corridors (`corridor.py`), packed
  capsule broadphase clearance (`clearance.py`), certified capsule margins +
  bilateral symmetry (`geometry.py`), sheaf-balance lowering (`balance.py`),
  Poiseuille/Reynolds physical solve (`hydraulics/solve.py`), and sealed
  lumen/wall material masks (`material/carve.py` — tiny channels rejected
  *before* rasterization). Structural cost fields and flow relaxation
  (`realize/grow.py` — its relaxation solve is the sole numeric quarantine).

### golem/kernel/mechanics — small-strain isotropic linear thermoelasticity

- **OWNS.** The D25/M7 constitutive claim, deliberately narrow: structured
  volumetric/SDF-derived domains, conforming trilinear hexes, 2×2×2 Gauss,
  follower pressure traction. It consumes an already-derived domain and
  already-resolved properties; it owns no body, mesh language, material
  catalogue, or vascular graph. `embedded_channel/` owns analytic
  capsule/thick-cylinder pressure evidence descended from canonical
  vasculature edges.
- **Typed surface.** `MechanicsObstruction` closed union
  (`obstructions.py:11-166` — validation, rigid-body mode, linear-solver,
  nonfinite/residual/force-balance, strain/displacement/yield limits, and
  the buckling family); `MechanicsResult = AcceptedMechanics |
  RejectedMechanics`, `BucklingResult` (`results.py:55-135`);
  `EmbeddedChannelObstruction` union (`embedded_channel/model.py:109-162`).
- **Seams.** Exactly one caller-side seam: `assembly/mechanics/compile.py`
  declares supports/forces/pressure/temperature intent and invokes
  `solve_mechanics` + `evaluate_embedded_channel_mechanics`; rejection
  translates to `RejectedAssembly`. anatomy/realize/allocation references
  `CellMechanics` (TYPE_CHECKING only).
- **DO-NOT-REBUILD.** The validation pipeline (`validate.py:46-404`),
  Q1 shape/constitutive/Voigt/von-Mises kernels (`element.py`), assembly
  kernels incl. follower-pressure tangent (`assemble.py`), constrained sparse
  solve + rigid-body detection (`linear_solve.py`), verdict/force-balance/
  acceptance limits (`verdict.py`), linearized buckling spectrum
  (`buckling.py:43-390`), embedded-channel volume bounds and burst-margin
  law (`embedded_channel/evaluate.py`, `validate.py`).

### golem/kernel/sheaf — the one linear-balance algebra

- **OWNS.** Immutable cellular balance: typed local terms (symmetric
  conductance, directed transport, source, fixed boundary, ambient exchange)
  glued into one sparse global Laplacian-style solve → authoritative `Section`
  + every interface flux, cell residual, boundary supply, global imbalance.
  Hydraulics and thermal analysis are *local interpreters* of this algebra;
  neither owns a parallel matrix system.
- **Typed surface.** Obstruction union in `result.py:25-151` (empty/invalid/
  duplicate terms, degenerate interfaces, conflicting boundaries, unanchored
  components, matrix failures, non-convergence, residual, imbalance,
  unsupported heat-transfer regime); closed enums in `vocabulary.py`
  (`BalanceTermKind`, `HeatTransferCorrelationKind`, …).
- **Seams.** The `BalanceInterpreter` protocol (`interpreter.py:32-50`) is the
  extension seam. Interpreters: `thermal.py` (reference), anatomy
  (`balance.py`, `hydraulics/solve.py`), assembly thermal stack,
  assembly/service environment. evals/eval_elemental_flow drives it via CLI.
- **DO-NOT-REBUILD.** `solve_balance` (`solve.py:43`) and its canonicalization/
  validation/assembly stages, the Nusselt/correlation algebra (`nusselt.py`,
  `correlation.py`). There is no separate linalg/spectra package — the
  documented "Field → Section Laplacian" *is* this solver; do not invent a
  second one.

### golem/kernel/blame — the WHY layer

- **OWNS.** Deterministic interrogation of rejections: approximate MUS
  (minimal unsatisfiable subset) over authored declarations. Spec +
  obstruction list → typed blame result, with deletion-based greedy
  1-minimization and near-miss aggregation.
- **Typed surface.** `DeclarationFamily`, `AuthoredUnitKind`, `IllPosedTaxon`
  (`core.py:10-37`), `AuthoredAddress`, `ObstructionSignature`,
  `CheckVerdict`, `BlameResult`, `BlameInterrogation`
  (`interrogate.py:50-66`), `NearMissDistribution`.
- **Seams.** One external consumer: `golem/cli/session.py` (`--blame`,
  `_blame_section`:150-170) via `check_verdict` + `interrogate_blame`.
  `verdict.py` runs the staged body→geometry→anatomy→vascular→assertions→
  assembly pipeline as its oracle.
- **DO-NOT-REBUILD.** `interrogate_blame` (the query API — ask blame *why*
  before building any new diagnostic explainer), `obstruction_signature`
  (identity projection), `extract_blame`/`shrink_target` (candidate
  exhaustion + minimization with budget accounting), `authored_addresses`/
  `restrict` (declaration enumeration and spec forking).

---

## Authoring and orchestration

### golem/assembly — elements composed by mounts, never smooth-min

- **OWNS.** The assembly dialect and its orchestration: independently compiled
  elements (creature = full stack; equipment = morphology + material +
  anchors + integrity), rigid placement via mounts, pairwise fit laws, and
  the coupled vascular/hydraulic/thermal/mechanics descent with authoritative
  receipts. Entry: `compile_assembly` (`core.py:51-105`, cached at
  `core.py:108-120`). Kernel packages own the primitive solvers; assembly
  owns composition and cross-element law.
- **Typed surface.** `AssemblyResult = AcceptedAssembly | RejectedAssembly`
  (`carriers.py:267-271`); the obstruction families (`obstructions.py:22-336`);
  closed `ElementFitLaw` (disjoint / surface_clearance /
  bounded_mount_contact, `obstructions.py:77-113`); receipts —
  `VisualAssemblyReceipt`, `CoupledPerformanceReceipt`, `ElementFitReceipt`,
  thermal/mechanics/vascular-sizing receipts (`carriers.py:123-251`);
  `CoupledUnevaluatedPhysics` records explicitly unmodeled physics.
- **Seams.** Consumed by `cli/compile.py`, `session/protocol.py`, and its own
  `__main__`. Consumes kernel/body, kernel/anatomy, kernel/mechanics,
  kernel/sheaf, materials, plates, conduits. Render export:
  `exchange.export_scene` (`exchange.py:82-104`).
- **DO-NOT-REBUILD.** Source digest/snapshot ingestion cache
  (`ingest.py:37-399`), mirror concretization + rigid graph/vascular
  transforms (`mounts.py:38-72,226-295`), pair-fit verdict algebra
  (`fit_algebra.py`), immutability sealing (`freeze.py`), coupled fixed-point
  / radius sizing (`core.py:280-430`), obstruction projection
  (`project.py:project_assembly_obstructions`), and the **glTF material
  export law** (`exchange.py:_pbr_material`:36 — emissive factors, alpha
  BLEND only for translucent). Note: `DECOMPOSITION_CAMPAIGN.md:56-63`
  constrains `core.py` to remain the re-export hub with patch-sensitive
  internals physically resident.
- **Status.** `mechanics_assembly.py` is a dead legacy compatibility shim
  ("stable import surface", zero importers) kept for import-path stability —
  route through `assembly/mechanics/` directly.

### golem/assembly/service — typed service intent (not a daemon)

- **OWNS.** The pure, total, applicative decoder for M7 coupled-substrate
  operating intent: semantic addresses, loads, supports, joints, heat,
  environment, pump, material, evidence cases (`service/model.py:75-220`).
  Malformed/incompatible declarations descend to addressed obstructions; only
  a fully compatible section glues into `ServiceIntent`.
- **Typed surface.** `service/outcome.py:23-121`: `MalformedServiceIntent`,
  missing unit scale, unknown/invalid semantic address, contradictory
  support, missing heat sink, unsupported constitutive model, unknown
  material/validity, duplicate case; `AcceptedServiceIntent |
  RejectedServiceIntent`.
- **Seams.** Decoded by `assembly/core.py:127-141` against known semantic
  addresses; consumed by assembly/mechanics and thermal lowering.
- **DO-NOT-REBUILD.** `decode_service_intent` (`service/intent.py:35-46`) and
  its fail-closed address/material validation. No second intent layer.

### golem/session — transactional authored-body sessions

- **OWNS.** The canonical authoring protocol: typed edit grammar, immutable
  journal with undo/redo/replay, branch-aware transitions, and the
  transaction verdict evidence chain. `protocol.py` is the public boundary;
  `ops` owns edits; `journal`/`algebra` own state transitions; `diff` lowers
  whole documents into the same edit algebra.
- **Typed surface.** `TransactionVerdict`, `SessionSubmission`,
  `AddressedTransaction`/`DocumentTransaction`, `RequestObstructed`,
  `Diagnostic`, `Margin`/`MarginDisposition`, `AuthoringEvidence`,
  `MarginDelta`/`VerdictDeltas` (`protocol.py`); edit ADT `SetOp | UnsetOp |
  AddOp | AddAtOp | RemoveOp | RenameOp` + `EditObstruction`/`EditPlan`/
  `ProgramPlan` (`ops.py`); `JournalEntry`/`RepairJournalEvent`
  (`journal.py`); `Compiled`/`CompileObstructed` (`state.py`).
- **Seams.** cli/forge and cli/session drive `ProtocolSession`/`submit`;
  cli/check uses `evaluate_authoring_surfaces`; the protocol itself calls
  `assembly.compile_assembly` and kernel/body for evidence.
- **DO-NOT-REBUILD.** The edit-op algebra and its decode/compile
  (`ops.py:decode_op/compile_program`), journal commit/undo/redo/replay
  (`journal.py`), whole-document `diff_specs`, branch planning
  (`algebra.py`), the receipt/outline projections (`receipt.py`,
  `outline.py`), and the effect boundary (`effect.py`).

---

## Perception, appearance, surfaces

### golem/senses — the perception plane

- **OWNS.** Everything derived from compiled geometry for measurement and
  rendering: `proprio` (assembled proprioceptive senses), `silhouette`
  (cross-artifact calibration), `symmetry` (bilateral metrics),
  `elemental_render` (appearance view of a solved sheaf section),
  `orthographic`/`raster` (the canonical renderers).
- **Typed surface.** `Senses` immutable carrier (`model.py`); `AnomalyKind` +
  `Anomaly` + `detect_anomalies` (`proprio/anomaly.py:23-29` — owns
  `blend_ambiguity`, `cross_plane_fusion`); `SilhouetteScore`/`ViewScore` +
  load obstructions (`silhouette.py`); `SymmetrySourceObstruction`
  (`symmetry_types.py`); `Accepted/RejectedElementalRender` with 8
  obstruction variants (`elemental_render.py`); orthographic/raster accepted/
  rejected pairs.
- **Seams.** contracts/asserts and body/report consume `proprio.build_senses`;
  cli/look renders through `orthographic`; evals consume silhouette,
  elemental flow/visual, symmetry harnesses.
- **DO-NOT-REBUILD.** Proprio geometry sampling/fusion/posture
  (`proprio/geometry.py`, `fusion.py`, `posture.py` — `GROUND_Y`, centroid,
  support margins, bands), the anomaly detector, receipt rendering
  (`proprio/render.py:render_receipt`), `symmetry_algebra` mesh folds,
  raster algebra (`raster.py` masks/IoU). **Tint law is NOT here** — it lives
  in `materials/surface.py`; `orthographic.py:_material_rgb` only resolves
  material + bounded emissive + gamma.

### golem/materials — the single typed material catalogue

- **OWNS.** Immutable visual + physical material evidence
  (`core.py:MATERIAL_CATALOGUE`:187). Appearance coatings are annotations
  with a neutral-gray fallback; solids/fluids/gases are constitutive and
  their decoders fail closed. `surface.py` owns the class vocabulary —
  `OPAQUE | EMISSIVE | TRANSLUCENT` with typed params — and the **tint law**:
  authored tint × seeded variation (`derive_linear_vertex_rgb`:777,
  `derive_vertex_colors`:806).
- **Typed surface.** `UnknownMaterialObstruction`, `AcceptedMaterial |
  RejectedMaterial` (`core.py:170-186`); `SurfaceColorObstruction` /
  `AppearancePaletteObstruction` with closed rule enums (`surface.py:64-197`).
- **Seams.** assembly (compile seeding, exchange export, obstructions),
  assembly/service (physical decoders), cli/look, body/eyes (appearance IDs),
  contract (vocabulary/schema), plates (seam material), evals. glTF
  emissive/alpha export law lives downstream in `assembly/exchange.py`.
- **DO-NOT-REBUILD.** The catalogue registry + neutral fallback, strict
  class-aware surface decoders (`decode_surface_color`:413,
  `decode_appearance_palette`:554), deterministic seeding
  (`seed_surface_color`:646), linear→sRGB face projection
  (`projection.py`), canonical sidecar payload.

### golem/plates — the derived plated-surface cover

- **OWNS.** Policy-driven surface segmentation: a `PlatePolicy` restricts
  authoritative morphology into local cells; seams and relief are derived
  views, never a second mesh-authoring language. Layouts:
  `SURFACE_VORONOI | MORPHOLOGY_ANISOTROPIC` (`core.py:11`).
- **Typed surface.** `PlatePolicyObstruction`/`RejectedPlatePolicy`,
  `PlateSurfaceObstruction` + accepted/rejected (`core.py:52-128`), closed
  `PlatePolicyRule`/`PlateSurfaceRule`.
- **Seams.** `assembly/compile.py` decodes policy and derives surfaces;
  assembly obstructions/project re-export its types; evals/eval_plate_authoring
  measures it; contract/schema exposes `PlateLayout`.
- **DO-NOT-REBUILD.** Deterministic farthest-point seeding
  (`geometry.py:farthest_point_seed_indices`), Voronoi/morphology-axis cell
  assignment, circulation-gated active seam selection (`active_seam_faces`),
  adjacency/area algebra, seam geometry + report projections
  (`projection.py`), the strict bounded policy decoder (`decode.py` —
  4..512 cells, finite nonnegatives, registered seam material).

### golem/conduits — surface trim, growth, vascular views, command harness

- **OWNS.** Four semantic subpackages (`__init__.py:1-15` is the authoritative
  split): `surface` — authored trim/rune embeddings, groove displacement,
  band strata; `growth` — seeded appendage centerlines and host-fusing part
  sequences (no circulatory semantics); `vascular` — *derived* diagnostic
  tube meshes from anatomy-owned accepted graphs only; `operator` — the
  closed command algebra + effectful terminal harness. Root-level modules
  (`core`, `embed`, `emit`, `grow`, `appendage`, `harness`) are legacy import
  paths re-exporting these owners.
- **Typed surface.** `EmissionPitchObstruction` (`surface/types.py:11` —
  displacement bounded by `0.9 * pitch`, violations reject rather than emit
  incomparable meshes); `GrowthTermination` closed variants
  (`growth/types.py:11`); `HarnessObstruction`, `AcceptedCommand |
  RejectedCommand`, closed `Command`/`Effect` ADTs (`operator/types.py`).
- **Seams.** `assembly/compile.py` applies conduits, expands appendages, and
  emits vascular stratum meshes. Vascular intent, topology, radii, flow, and
  lumen ownership remain in kernel/anatomy — conduits consumes accepted
  results only. The terminal authoring contract (`HELP.txt`) is explicit:
  authors declare body/anatomy vascular *intent* (skeleton, perfusion
  regions, pump, exchange beds, demands), never vessel geometry; conduit
  commands are surface trim/runes (`add loop`/`add line`, `grow`,
  `vascular solve/receipt/render`).
- **DO-NOT-REBUILD.** Analytic station embeddings (`surface/embed.py`
  axial-loop/face-line), the pitch safety gate, seeded space-colonization
  growth with radius fold and chain consolidation (`growth/grow.py`,
  `appendage.py`), the command algebra `step` + parse + effect interpreter
  (`operator/algebra.py`, `parse.py`, `effects.py`).

---

## Contracts, instruments, entry points

### golem/contracts — postural contracts (runtime verdicts)

- **OWNS.** The executable typed assertion algebra over frozen proprioceptive
  senses: clause lowering, metric registry, operator algebra, authoritative
  verdict evaluation.
- **Typed surface.** `ContractObstruction` union (`model.py` — scope,
  malformed clause, unknown metric, unsupported operator, missing sense,
  invalid scope shape, invalid descriptor); closed `MetricKind` and
  `OperatorKind`; `Verdict = Passed | Failed | Unmeasurable`; `Accepted |
  Rejected` clause results.
- **Seams.** `kernel/body/report.py` evaluates packs over compiled senses;
  `senses/proprio/cli.py`, `session/effect.py`, `session/protocol.py` consume
  verdicts/records; `contracts/cli.py` is the standalone boundary
  (`asserts.py` is the public surface).
- **DO-NOT-REBUILD.** `evaluate_verdict`/`evaluate_verdict_pack`
  (`verdicts.py` — the authoritative pipeline), `lower_clause`
  (`clauses.py`), the metric descriptor registry (`metrics.py`),
  `lower_operator`/`apply_operator` (`operators.py`), `verdict_record`
  projections (`views.py`).

### golem/contract — the authoring contract (publication, not runtime)

- **OWNS.** The live-authored specification documents: seven rendered
  sections — primer, relations, vocabulary, exemplar, receipts, envelope,
  schema (`registry.py:SECTIONS`, `section_by_name`/`render_section`) — plus
  `body_schema` JSON-schema (`schema.py`) and closed vocabulary value
  projections (`vocabulary.py` — element roles, appearance materials,
  assembly strata).
- **Typed surface.** None of its own; it *documents* the obstruction
  families owned elsewhere (primer, receipts, envelope — e.g. the
  `InfeasibleBifurcationObstruction` vs `VascularSearchBudgetObstruction`
  trichotomy).
- **Seams.** `cli/contract.py` lists/renders sections; `session/protocol.py`
  references section names; internal registry modules compose.
- **DO-NOT-REBUILD.** A second schema or vocabulary source — `schema.py` and
  `vocabulary.py` are the canonical projections agents author against. The
  exemplar renders `specs/knight_body.json` and marks fixed-coordinate
  attach as the legacy escape hatch (`exemplar.py:77`).

### golem/evals — executable instruments

- **OWNS.** Sealed measurement, never beauty oracles: `harness.py` owns the
  typed outcome (`AcceptedEvaluation`, `EvaluationObstruction`,
  `ProcessReceipt`), stable JSON rendering, guard exit law, and the
  subprocess boundary. Instruments: `eval_anatomy` (control-vs-tissue
  construction effect), `eval_elemental_flow` (balance-interface
  capabilities), `eval_elemental_visual` (rendered-output legibility),
  `eval_plate_authoring` (plate-cover compass), `eval_vascular_rivals`
  (rival constructors on identical terms), `authoring_surface` +
  `authoring_corpus` (authoring-trial metrics, content-addressed receipts).
- **Typed surface.** Per-instrument typed evaluation records +
  `EvaluationObstruction(address, reason)`; corpus `CorpusIOObstruction`/
  `RejectedCorpus`; `DecodeObstructionCode`/`MetricUnavailableCode`.
- **Seams.** Run via `python -m golem.evals.<name>` or `cli/eval.py`
  (lazy wiring on canonical/spec predicates). Human blind preference remains
  the appeal authority; these instruments only prevent degenerate wins.
- **DO-NOT-REBUILD.** The shared harness outcome/render/exit law — a new
  instrument plugs into `harness.py`, never into ad-hoc prints.

### golem/cli — the forge boundary

- **OWNS.** The lazy command registry (`registry.py` — exactly: check,
  compile, contract, eval, forge, look, pose-prior, session) and per-command
  wiring; `golem/__main__.py` dispatches `cli.main`.
- **Seams.** check → body/anatomy/engine/proprio/session evidence; compile →
  assembly + `exchange.export_scene`; contract → golem/contract sections;
  eval → golem/evals; forge/session → session protocol (+ kernel/blame for
  `--blame`); look → compile + senses orthographic + materials; pose-prior →
  anatomy quadruped diagnostics.
- **DO-NOT-REBUILD.** Command descriptors and the registry cache — new
  surface area enters as a registered command reusing these wirings, not as
  a parallel entry point.

---

## Cross-cutting laws

1. **One owner per surface.** No parallel authoring surface, no second
   geometry/material/balance/verdict authority. Extensions join the existing
   owner's closed vocabulary; derived views stay derived.
2. **Absence is byte-exact.** New vocabulary is opt-in; undeclared specs
   compile byte-identically (pinned goldens, graph SHAs, PNG hashes).
3. **Typed obstructions, never silent fallback.** Widening a radius, scaling
   a neighbor, or accepting a non-converged iterate is forbidden — reject
   with the witness.
4. **Total decoders.** Malformed input returns addressed obstructions;
   decoders never throw.
5. **Projection is owned.** Every obstruction family joins its package's
   projection (`body/project.py`, `assembly/project.py`,
   `anatomy/project.py`); human/JSON views derive from the authoritative
   typed value, never the reverse.
