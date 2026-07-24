# GOLEM — A Language-Native Creature Construction Kernel

**Design Document v0.2 — 2026-07-08**
Codename `GOLEM` (Geometry Of Language-Editable Morphology) is a placeholder; rename freely.

*v0.2 changes (from independent operator-model review of v0.1): determinism partitioned into symbolic/numeric surfaces (L-13 → AX-8, R-21, A-8); orientation added as a first-class pull sense (L-14 → §8.5, R-22); membrane/watertight contradiction closed (L-15 → R-23, §13.2).*

---

## 0. Preamble: who this document is for and how to read it

This document is written to be consumed primarily by **agents** — LLM builders who will implement the kernel, and LLM operators who will construct creatures with it. Humans direct, set taste, and own the product; the hands on the substrate are models. Every design decision below is therefore evaluated against one question: *does this make the reasoning creature — the LLM operator — more able to sense, say, and rely on things?*

An unusual note on authorship: this spec was drafted by an LLM that spent a working session operating three successive prototypes of this substrate (the "pilots," §2). The requirements in the Feedback (§8) and Articulation (§9) sections are not projections of what an operator might need; they are reports of what the drafting model did need, observed at the moment of needing it. Where a pilot taught a lesson, it is recorded in the **Lessons Register** (§16) with an ID like `L-3`, and the normative rule it produced cites that ID.

**Normative language.** MUST / MUST NOT / SHOULD / MAY as in RFC 2119. Numbered rules (`R-*`) are binding on implementations. Rule IDs are accretion-ordered and stable; textual position does not imply sequence. Axioms (`AX-*`) are binding on the design itself: a change that violates an axiom is a different system, not a version of this one.

**Reference artifacts.** A working single-file Python pilot exists and functions as an *executable specification and conformance oracle*:

| artifact | role |
|---|---|
| `engine.py` | part-graph → SDF field → mesh; coherence metrics |
| `render.py` | deterministic software renderer; contact sheets, turntables |
| `golem_v4.json` | reference creature #1 (body plan, 14 parts) |
| `hand.py`, `hand_v2.json` | sub-grammar + FK compiler; reference sub-creature |
| `hand_pack.py` | reference constraint pack (predicates + pose tests) |
| `plates.py`, `plate_complex.json` | reference surface cell complex (appearance layer 2) |
| `mount_hands.py` | reference port composition (with recorded failures) |
| `control_creature.py`, `judge.py` | A/B control condition + identical-judging harness |

Implementing agents MUST treat the outputs of these artifacts as golden regression targets (§15) until the kernel supersedes them, and SHOULD read them before reading §5–§9: they are small, and they are the spec in miniature.

---

## 1. Thesis

Text-to-3D exists and is not this. Diffusion tools produce opaque meshes an operator cannot read, address, or incrementally edit; code-driven tools (Blender scripting and its research descendants) hand the operator a continuous substrate in which every emitted coordinate is an unverifiable claim, errors compound silently, and the result is shell soup. Both are tools built for a *visual-native* constructor and retrofitted for a linguistic one.

This kernel inverts the assumption. The constructor is a language model. Therefore:

> The ground truth of a creature is a set of **relations** — named, typed, diffable symbols. Geometry is *compiled* from the relations by deterministic solvers. Pictures are rendered from geometry as one sense among several, and proposals may arrive from images, but nothing visual is ever the substrate.

The operator emits only discrete, local, symbolic decisions. Everything continuous — blending, surfacing, field interpolation, contact, equilibrium — is solved, never generated. The system's obligation in return is a sensory stack that makes every consequence of a symbolic decision observable at a granularity the operator can act on, at a latency that keeps the loop tight, with every error carrying an address.

The one-line justification for building this at all, measured in the pilots: **the model is the constant; the tool surface is the variable.** The same model, same reference image, and same iteration budget produced a shapeless mitt in one substrate and a pack-passing five-fingered hand in another (`L-8`). Intelligence was not the bottleneck. Observability was.

---

## 2. Empirical grounding: the pilots and what they measured

Three pilots were run against a reference image set (an obsidian golem, a furred ape, a crystal-winged dragon — chosen because they exercise three orthogonal appearance systems: tessellation+emissive, groom-dominant, spar+membrane).

**Pilot 1 — body plan (Layer 0).** A 5-construct JSON part-graph DSL (generalized cylinder, blob, mirror flag, typed spine points, smooth-min blend) plus ~250 lines of solver/renderer. Four iterations, every edit a small named delta ("elbow x 1.45→1.55"). Result: watertight single-component game-exportable GLB whose source of truth is a 90-line JSON file. The coherence metric (connected components) caught two defects the renders did not show: a detached sliver (v2) and symmetric claw-tip dust below grid feature size (v4) — the latter became a permanent invariant (`L-2`).

**Pilot 2 — controlled comparison + blind judging + first appearance layer.** (a) Control condition: same model, same reference, same four rounds, free primitive-placement code (LL3M/Blender-style). Judged by identical metrics: control also reached one connected solid and near-perfect hand-authored symmetry — but as 35 unmerged shells (not watertight, not game-ready), with no taper available in the substrate, and each fix adding filler geometry (shell count rose every round). (b) Six context-free judge agents matched both conditions' renders to the correct lineup description at 0.9 confidence; blind quality scores were statistically indistinguishable (3.3 vs 4.0 on n=3 each). Honest conclusion, now axiomatic: **at the maquette layer the substrate's wins are mechanical — integrity, symmetry-by-construction, diff economics — not aesthetic; the aesthetic separation must come from solver-shaped layers above** (`L-5`, `L-6`). (c) Plate tessellation: 110 Voronoi cells over the golem surface with relief/groove displacement and emissive seams, exported alongside the mesh as a plate adjacency graph — the first live surface cell complex, and the first artifact where "merge plates 41 and 67" is a legal sentence.

**Pilot 3 — the hand pack.** A hand sub-grammar (palm + fingers/phalanges/splay + opposed thumb, forward kinematics, compiled to the same part-graph dialect) plus a constraint pack: 8 anatomical predicate families and 3 behavioral test poses (open / fist / point), all evaluated on the spec in milliseconds with no meshing. A deliberately naive spec (equal fingers, no splay, parallel thumb — the unconstrained-LLM default) drew 20 named, quantified violations; one revision authored directly against the report went to zero; renders confirmed. The naive hand passed `point` and failed `fist` — behavioral tests expose defects static checks cannot (`L-9`). Port-mounting the hand on the golem failed twice until the forearm re-aimed itself at the hand's wrist junction — attachment is a constraint between two chains, not a transform of one (`L-10`).

Numbers an implementer should retain: L0 symbolic checks ran in **milliseconds**; mesh evaluation at res=190 in **seconds**; a full perceptual round (render + judge agents) in **tens of seconds**. That three-decade latency spread *is* the sensory architecture (§8).

---

## 3. Axioms

**AX-1 (Language-native).** The authoritative state of a creature is relational and textual. Every fact the system knows MUST be representable as rows; every row MUST be addressable and printable. Geometry, images, and compiled assets are derived artifacts. A datum that exists only in a mesh, a texture, or a float array is not state; it is exhaust.

**AX-2 (Solved, not generated).** The operator emits discrete, local, symbolic decisions. Continuous consequences are computed by deterministic solvers. No workflow may require the operator to author raw coordinates whose global consequences it cannot check symbolically. (The operator MAY author *anchor* coordinates — spine control points, seeds, boundary conditions — because these are checkable one at a time.)

**AX-3 (Observability is the binding constraint).** Every past failure of LLM 3D generation this project examined reduces to the model being unable to see what it did wrong at a granularity it could act on. Design effort allocates to senses before capabilities. A feature that adds expressive power without a matching check is deferred until the check exists.

**AX-4 (Information-rich, veto-poor constraint).** Constraints are tiered (§7). The set of hard invariants is deliberately thin; the broad constraint mass informs and never blocks. Rationale from pilot 3: the naive hand converged in one round *because* all twenty violations were simultaneously visible and none of them rejected the artifact. A scalar reward or a hard rejection carries ~1 bit; a violation report carries the whole gradient.

**AX-5 (Validity-total design space).** Under the invariant tier, every reachable state is a live, compilable, renderable creature. There is no broken intermediate state. (Spore's editor demonstrated the human version of this in 2008; it matters more for autoregressive constructors, because it converts open-loop generation into search inside a closed space.)

**AX-6 (The ratchet).** Know-how accumulates as machine-checkable artifacts — pack predicates, pose tests, rewrite rules, conformance cases — never merely in prompts or weights. Every diagnosed failure MUST have a path to becoming a permanent check. Quality is cumulative infrastructure, not per-run heroics. This is the only known mechanism for the 90→95% climb (`L-2`, `L-8`).

**AX-7 (Freedom as legible deviation).** The operator may deviate from any norm, per scope, by declaration with a stated reason. Deviation is recorded in the artifact. Weirdness is a choice the system can distinguish from a mistake.

**AX-8 (Determinism and replay — two surfaces).** Determinism is partitioned, because bit-exactness and parallel floating point cannot coexist, and pretending otherwise quietly serializes the kernel (L-13). The **symbolic surface** — relations, journal, canonical serialization, receipts, addresses, L0/L1 evaluation — MUST be bit-exact under replay on any conformant build. The **numeric surface** — solver outputs: meshes, fields, weights, renders — MUST be *tolerance-conformant*: reproducible within each solver's declared tolerance (R-21), with a **sealed mode** (fixed iteration order, no parallel reduction, reference kernels) that IS bit-exact and is what golden conformance runs use. Every state is reconstructible from the journal: symbolically exactly, numerically within tolerance (exactly under sealed mode). Speculation, undo, and multi-agent merge ride the symbolic surface and are unaffected by numeric-surface variation.

**AX-9 (The visual proposes; the symbolic disposes).** Generative visual models, reference images, and scans MAY propose — region graphs, palettes, silhouettes, part hypotheses. Proposals enter the world only by transcription into relations, where the sensory stack can check them. Nothing visual is ever authoritative.

**AX-10 (Agent-first ergonomics).** Interfaces are optimized for token-efficiency, diffability, and quotability: compact canonical serializations, stable ordering, small deltas, addresses everywhere. Human ergonomics (viewports, gizmos) are views built on top and are out of kernel scope.

---
## 4. System overview: five planes

```
                       ┌─────────────────────────────────────────────┐
   operator agents ──► │  STATE PLANE      relations + deltas (z-sets)│ ◄── proposals
   (edit language)     │  Part Port Attach Field Cell Section Pose …  │     (image→region
                       └──────┬───────────────────────────┬──────────┘      transcription)
                              │ deltas                    │ deltas
                       ┌──────▼──────────┐        ┌───────▼──────────┐
                       │  VIEW PLANE     │        │  SOLVER PLANE    │
                       │  incremental    │        │  SDF/smin, mesh, │
                       │  constraints,   │        │  sheaf Laplacian,│
                       │  reachability,  │        │  FK/contact,     │
                       │  budgets, eqsat │        │  skin weights    │
                       │  (DBSP circuits)│        │  (deterministic) │
                       └──────┬──────────┘        └───────┬──────────┘
                              │ violations w/ addresses   │ meshes, fields
                       ┌──────▼───────────────────────────▼──────────┐
                       │  PERCEPTION PLANE  renders, silhouette IoU,  │
                       │  scrutiny-weighted judges, calibration       │
                       └──────────────────────┬───────────────────────┘
                                              │ compiled assets
                                       ┌──────▼──────────┐
                                       │  EXCHANGE PLANE │  glTF/GLB, skeleton+
                                       │                 │  weights, shader/VFX
                                       └─────────────────┘  annotations, LODs
```

The **state plane** is the only writable surface. The **view plane** maintains derived truth incrementally — every constraint, budget, and query is a materialized view over state relations; edit cost scales with delta size, never creature size. The **solver plane** turns symbols into continuous geometry deterministically. The **perception plane** is the slow, expensive sense that calibrates the fast ones. The **exchange plane** is the one-way membrane to engines and DCC tools.

**R-1.** All writes go through the state plane as transactions of relation deltas. Views, solvers, perception, and exchange MUST be pure functions of state (plus declared seeds).

**R-2.** The view plane MUST be incremental (differential dataflow / DBSP or equivalent): recomputation proportional to changed rows. Full recomputes are permitted only as conformance checks against the incremental result.

---

## 5. State plane: the kernel schema

Notation: `Relation(key fields | value fields)`. All rows carry `(creature_id, txn_id)` provenance implicitly. Serialization is canonical JSON Lines with stable field order (AX-10). Types: `id` (opaque stable identifier), `addr` (hierarchical address, §5.1), `frame` (position + quaternion + scale), `expr` (pure expression in the kernel expression language, §5.4).

### 5.1 Addressing

Every editable datum has exactly one canonical address:

```
creature://<creature>/<part-path>[#<component>][@<field>]
  creature://golem/arm.L/hand/finger.index/seg1@radius
  creature://golem/torso#cell/plate41
  creature://golem/~fur@direction!singularity.crown
```

**R-3.** Every violation, judge note, journal entry, and solver diagnostic MUST carry at least one address. An unaddressed error is a kernel bug. (Rationale: `L-1`, `L-4` — the pilots' errors were fixable in one round exactly when they arrived with an address and a magnitude.)

**R-4.** Renames preserve identity: addresses are aliases over stable `id`s; the journal records alias history. Operators SHOULD refer by address; the kernel resolves.

### 5.2 Core relations (v1, normative)

**Structure.**

```
Part(part_id | kind, params, frame_expr, region_id?, meta)
    kind ∈ {gencyl, blob, membrane, scatter, subgraph_instance, …extensible §9.4}
    gencyl params: spine: [vec3…] (anchors), radii: [r…]   # pilot dialect, kept
    blob   params: center: vec3, size: vec3
Port(port_id | owner_part, name, iface)                    # iface: §6.1
Attach(attach_id | port_a, port_b, solve_policy, status)   # bidirectional, §6.2
Symmetry(sym_id | kind=bilateral|radial(n), scope=region|whole, plane/axis)
Region(region_id | name, definition_expr, parent_region?)  # semantic, addressable
```

**Appearance.**

```
Cell(cell_id | complex_id, seed, kind)                     # e.g. plate, scale, patch
CellAdj(complex_id, cell_a, cell_b | seam_measure)         # the base complex; sheaf lives here
Complex(complex_id | surface_of: region_id, policy_expr)   # seeding/density policy, not raw cells
Field(field_id | domain: region|complex, type: scalar|dir|frame|color, boundary_exprs,
      singularities: [(addr, kind, params)…])              # operator writes BCs + singularities ONLY
Section(section_id | field_id, cell_id | value)            # solver-filled; derived but persisted
Material(mat_id | region|cell scope, model, params, emissive?, vfx_tag?)
```

**Constraint & function.** (details §7)

```
Pack(pack_id | name, version, scope_expr, provenance)
Rule(rule_id | pack_id, tier ∈ {invariant, norm, brief}, predicate_expr, message_template)
Deviation(dev_id | rule_id, scope_addr, reason, declared_by)
CoreOrgan(core_id | part_id, metabolism, capacity)
Consumer(cons_id | part_id, kind=mouth|intake|…, rate)
Budget(budget_id | scope, resource, cap)                    # complexity economy
```

**Behavior.**

```
Joint(joint_id | at_port, dof, limits, axes)                # typed at ports
Pose(pose_id | name, pack_id?)  PoseJoint(pose_id, joint_id | angles)
TestPose(test_id | pack_id, pose_id, checks_expr)           # behavioral tests
Clip(clip_id | body_plan_class, source, retarget_policy)    # animation library hook
```

**Process.**

```
Journal(txn_id | parent_txn, author_agent, ops, timestamp_in)   # append-only; AX-8
Branch(branch_id | base_txn, purpose)                            # speculation, §10.1
Brief(brief_id | reference_images, target_exprs, style_notes)    # scored, never blocking
VocabEntry(vocab_id | kind_name, compiler_ref, verification_ref, status ∈ {quarantine, trusted})
RewriteRule(rw_id | lhs_expr, rhs_expr, context_guard_expr, equivalence_class, provenance)  # §10.2
```

**Derived (views, read-only).**

```
Violation(rule_id, scope_addr | magnitude, range, suggestion, since_txn)
BudgetUse(budget_id | used, remaining)
Reachability(core_id, cons_id | path_exists, path)
CoherenceReport(creature | components, watertight, symmetry_iou, dust)
TestResult(test_id | pass, violations)
SilhouetteScore(brief_id, view | iou)
JudgeScore(brief_id | match_rate, quality_mean, n, calibration_ref)
```

**R-5.** The derived relations above MUST exist with at least these fields; implementations MAY extend. `CoherenceReport` MUST distinguish real components from sub-feature-size dust (`L-2`).

### 5.3 What the operator may write

The writable surface is deliberately narrow: `Part` anchors and params, `Port/Attach`, `Symmetry`, `Region`, `Complex` *policies*, `Field` *boundary conditions and singularities*, `Material`, `Pack` selection, `Deviation`, `Budget`, `Pose/PoseJoint`, `Brief`, `Branch`, and vocabulary/rewrite proposals into quarantine. Everything else is solved or derived.

**R-6.** The kernel MUST reject (invariant tier) any transaction that writes directly to solver-owned relations (`Section` values, meshes, weights). The operator's motor system is anchors, policies, boundary conditions, names, and declarations — nothing else. (AX-2 made enforceable.)

### 5.4 Expression language

A small, total, pure expression language for predicates, policies, guards, and derived scopes (comparisons, arithmetic, vector ops, quantifiers over relations, region algebra). It MUST be: deterministic, terminating (no general recursion), serializable as text, and evaluable both row-at-a-time (L0 checks) and incrementally (view plane). It SHOULD compile to the same relational algebra the view plane executes, so a `Rule.predicate_expr` is simultaneously documentation, an L0 check, and a DBSP circuit. (This is the egglog-flavored unification point: datalog-style rules, e-matching, and constraint views over one relational core.)

---

## 6. Geometry model and solver plane

### 6.1 Part kinds (v1)

`gencyl` (spine anchors + per-anchor radii — the workhorse; taper is native, `L-7`), `blob` (ellipsoid), `membrane` (surface spanned between named spines/ports, carrying material/VFX annotation only — never modeled volumetrically), `scatter` (instanced sub-parts over a region driven by a direction/density field — crystals, spikes), `subgraph_instance` (a packaged sub-grammar behind a port, §6.3).

**R-7 (Frames are explicit).** Every part kind, sub-grammar, and port MUST document its local frame convention in its `VocabEntry`, and the kernel MUST carry frames in `Attach` solving. Rationale: `L-10` — both mounting failures in pilot 3 were frame mistakes, and they were only diagnosable because the frame convention was written down.

**R-8 (Invariant: minimum feature size).** No solvable configuration may produce features below `k·`(finest evaluation pitch). Enforced symbolically (tip radii, seam widths) before meshing. From `L-2`.

**R-9 (Symmetry is a constraint, not a copy).** `Symmetry` rows cause instancing at solve time; there are no hand-mirrored duplicates in state. The control condition (35 shells, hand-negated x-coordinates) is the documented anti-pattern (`L-6`).

**R-23 (Open surfaces vs the watertight invariant).** Watertightness and the components=1 check apply to the **volumetric solid** — the SDF-compiled body — only. Membranes are a distinct geometry class: 2-manifolds with boundary, excluded from the solid's watertight/component checks and governed by their own invariants: every membrane boundary curve MUST anchor (within solver tolerance) to body surface or spar parts; free-floating membranes are invariant violations; anchoring participates in creature-connectivity, so a detached wing still fails. R-8 applies to membrane compile thickness. (Closes the crystal-winged-dragon contradiction, L-15.)

### 6.2 Solvers (v1 set, all deterministic)

1. **Field assembly + meshing**: per-part SDFs, smooth-min union with per-part blend, adaptive-resolution marching cubes (octree refinement near small features; the pilot's uniform grid is the conformance baseline).
2. **Contact/placement solve**: generalize the pilot's ground solve (measure penetration/clearance along a support direction, shift to kiss) into a constrained least-squares placement pass. `L-11`: two lines of measure-and-lift previewed this; the kernel version handles multiple simultaneous contacts.
3. **FK/pose solve**: typed joints at ports; pose application is pure FK in v1; IK enters only as a solver (operator states goals, never joint floats).
4. **Sheaf relaxation** (phase 3+): fill `Section` values by diffusion on the cellular sheaf Laplacian with operator boundary conditions/singularities pinned — fur direction, plate size grading, color fields. Failure to converge or obstruction residue is reported *per cell*, addressed.
5. **Skinning**: skeleton derived from the part graph (spines are bones), heat-diffusion weights; blendshape hooks reserved for the face sub-grammar.

**R-10.** Solver diagnostics are first-class: every solver MUST emit residuals/obstructions into `Violation` with addresses, not logs.

**R-21 (Solver determinism declaration).** Every solver MUST declare its tolerance model (metric + bound) and provide a sealed-mode implementation (AX-8). Conformance runs sealed; production runs fast. Fast-mode output drifting beyond declared tolerance from sealed output is non-conformant. Symbolic-surface checks MUST be robust to numeric variation within tolerance — topological counts and banded comparisons, never float equality; a predicate that flips on last-bit differences is a kernel bug.

### 6.3 Sub-grammars and the composition protocol

A **sub-grammar** is a packaged creature-fragment: relations + compiler (spec → parts), its constraint pack, its test poses, and a **port signature**. The hand pilot is the reference instance: `hand.py` (compiler) + `hand_v2.json` (spec) + `hand_pack.py` (pack) behind a `wrist` port.

Port signature (the `iface` in `Port`):

```
iface = { frame_convention, cross_section: profile_expr, joint: Joint?,
          load_rating, material_continuity, scale_band: [lo, hi] }
```

**R-11 (Bidirectional gluing).** `Attach` is a constraint satisfied by *both* chains. The solve MUST be allowed to modify anchors on both sides within declared freedoms (`solve_policy`: which side yields on position/orientation/scale, or `negotiate`). Mounting-as-transform-of-one-side is non-conformant. From `L-10`: the hand mount succeeded only when the forearm re-aimed at the hand's wrist junction.

**R-12 (Scope isolation).** A sub-grammar's norms bind inside its boundary; only the port signature crosses. Composition of packs along ports follows sheaf gluing: interface agreement is exactly the shared restriction, nothing more.

---
## 7. Constraint system

### 7.1 Tiers

| tier | may block? | who authors | examples | operator relation |
|---|---|---|---|---|
| **invariant** | yes — transaction rejected | kernel + distilled lessons | watertight-compilable, connected, min feature size, no writes to solver-owned relations, budget hard caps | cannot deviate |
| **norm** | never — reports | packs (anatomy, structure, style systems, metabolisms) | phalanx ratios, splay fan, taper monotonicity, support/load, plate seam alignment | may deviate by declaration (AX-7) |
| **brief** | never — scores | operator + director | silhouette IoU vs reference, palette targets, style notes, judge quality | tunes search, never gates |

**R-13.** The invariant tier MUST stay thin: adding an invariant requires demonstrating that violating states are *never* legitimate creatures (the pilot count after three pilots: five invariants). Everything else is a norm. AX-4 is the arbiter in disputes.

**R-14 (Violation format).** Every violation MUST carry: `rule_id`, address, measured value, permitted range, magnitude (normalized), and — when the predicate admits it — a suggested direction. Canonical example, verbatim from pilot 3, because its form is why convergence took one round: `phalanx_ratio @ index/seg1: ratio 1.00 outside [0.55, 0.90]`. A violation is an instruction, not a critique (`L-4`).

### 7.2 Packs

A pack is the unit of accumulated know-how: `Pack` + `Rule` rows + `TestPose` rows + docs, versioned, with provenance. Packs bind to scopes (regions, sub-grammars, whole creature) and compose by gluing (R-12). Conflicts between overlapping packs resolve by: explicit precedence declaration, else more-specific-scope wins, else the conflict itself is surfaced as a violation on the packs (never silently).

**Functional packs** (metabolisms) make validity teleological: a `biotic` pack requires `Reachability(core, mouth)`; an `elemental` pack requires a core and support but no consumer; `undead`, `mechanical`, etc. Metabolisms are swappable per creature — the reference golem has no mouth, and that is a metabolism choice, not a violation. Budgets (part count, energy, polycount-at-LOD) are functional norms with hard caps only at the invariant tier's outer bound.

**R-15 (Accretion protocol — the ratchet made operational).**
1. A failure is diagnosed (by operator, judge, or human director).
2. A candidate `Rule`/`TestPose` reproducing the failure is authored (usually by the operator agent that hit it).
3. It is verified offline by a verification agent against the regression corpus: catches the failure, fires on no golden artifact (or the golden artifact is updated with cause).
4. It ships in a pack version bump; the journal links rule → originating failure.
Every kernel release note MUST list rules accreted since the last. A quarter with zero accreted rules is a process failure, not a sign of completeness.

### 7.3 Behavioral tests

Test poses are the unit tests of bodies (`L-9`). Packs SHOULD ship pose suites sized to the scrutiny of their region (§8.4): hands ship fist/splay/point; quadruped legs ship stance/stride/crouch; wings ship fold/spread. Checks run on chains (capsule-capsule distance, contact gaps, joint limits) *without meshing*. Pose suites double as animation acceptance: a clip retargeted onto a body plays back through the same checks as a streaming sequence of poses.

---

## 8. Feedback: the sensory stack

The pilots measured a three-decade latency spread across senses; the architecture is built around it.

| level | sense | latency target | trigger |
|---|---|---|---|
| **L0** | symbolic predicates on specs/anchors (no meshing) | ≤ 10 ms | every edit, synchronously |
| **L1** | incremental views: constraints, budgets, reachability, coherence | ≤ 100 ms, ∝ delta size | every transaction commit |
| **L2** | solver-derived: pose suites, contact residuals, sheaf obstructions | ≤ 2 s | on demand + on relevant deltas |
| **L3** | perceptual: renders, silhouette IoU, judge panels | seconds–minutes | at operator checkpoints; scrutiny-allocated |

**R-16.** L0/L1 results MUST be delivered in the same turn as the edit (synchronous transaction receipt). The receipt for any transaction is: applied ops, new violations, cleared violations, budget deltas. Nothing else. Token-frugal, quotable, diffable (AX-10).

**R-17 (Perception is calibration, not the loop).** Inner-loop decisions run on L0–L2. L3 exists to catch what predicates cannot (gestalt, proportion, appeal) and to *calibrate* cheap surrogates: silhouette IoU and predicate weights SHOULD be periodically regressed against judge scores. When L3 catches something L0–L2 missed, that is an accretion trigger (R-15), not just a fix.

### 8.1 Renders as a sense

Deterministic renderer (the pilot's software renderer is the conformance baseline), canonical views + turntables, contact-sheet format with labeled azimuths. Depth, normal, and silhouette channels are part of the sense, not just beauty shots. Every render is journaled with its txn so any perception is reproducible (AX-8).

### 8.2 Judge protocol

Blind, context-free judge agents; N ≥ 3 per question; lineup identification (match render to one of ≥ 8 descriptions) for recognizability; scalar quality with rubric for appeal; majority/mean with recorded confidence. Judges MUST NOT see the constructing conversation (pilot 2's judges are the reference implementation). Costs are budgeted by the scrutiny field.

### 8.3 The journal as a sense organ

The append-only journal is queryable by the operator: "what did I change since the fist test last passed," "which edits historically preceded `fist_contact` regressions." Distiller agents run over journals to propose pack rules (R-15 step 2 automation).

### 8.4 Scrutiny field

A scalar field over regions (default priors: face ≫ hands ≫ silhouette edges ≫ bulk), operator- and director-editable. Allocates: L3 budget, pose-suite density, predicate strictness bands, mesh resolution refinement, and eqsat extraction weights. This is how 95% is affordable *where it counts* without paying for it everywhere.

### 8.5 Orientation: the outline sense

L0–L3 are all delta and error senses; none of them tells a fresh operator session *what is here*. Raw rows are not a gestalt and journal replay is not a summary — v0.1 review correctly flagged this as the one gap contradicting the doc's own thesis that observability is the binding constraint (L-14). The kernel therefore provides **outline**, a pull-based sense: canonical, deterministic compression of state to text at controllable depth and scope.

`outline(scope_addr, depth, aspect) → bounded text`. Aspects: `structure` (part tree, key params, counts, port status), `constraint` (active packs, violation summary by rule, standing deviations), `appearance` (complexes, fields, materials by region), `budget`, `recent(txn)` (digest of change since a transaction). Every line quotes at least one address, so any outline entry is a handle the operator can zoom into (deeper outline) or act on (edit ops) without translation.

**R-22 (Outline).** Outline MUST be: deterministic (same state, same text — symbolic surface); size-bounded (declared token budget per depth, enforced); address-bearing on every line (R-3 extended); and convergent (increasing depth reproduces full canonical state). Fresh operator sessions SHOULD begin with `outline(creature, 2, structure)` + `outline(creature, 1, constraint)` + the active brief. Receipts stay delta-only (R-16); orientation is pull, never push.

---

## 9. Articulation: the motor system

### 9.1 Edit language

Transactions of typed ops over the writable surface (§5.3):

```
add(kind, addr, params) · remove(addr) · set(addr@field, value)
rename(addr, name) · instance(vocab_id | subgraph, at_port)
attach(port_a, port_b, solve_policy) · declare(deviation | budget | brief)
pose(pose_id, {joint: angles}) · branch(purpose) · merge(branch, policy)
propose(vocab | rewrite, payload)          # → quarantine
```

**R-18.** Ops address symbols, never vertices. The kernel MUST provide *no* vertex-level op. (The temptation returns at the face; resist it there too — blendshapes are symbols.)

**R-19 (Diff-native).** Every op has a printable inverse; every transaction a compact canonical serialization. Two agents' concurrent transactions merge by z-set addition when disjoint in scope; overlapping scopes require a lease (§12.2).

### 9.2 Fields are edited by their handles

Boundary conditions, seeds/density policies, and singularity placement — never raw values. The fur field on a creature is authored as: direction BCs at named regions + explicitly placed singularities (crown, elbows) + a groom pack; the solver fills 100k values. Topology guarantees singularities exist (hairy-ball); the kernel's job is that they exist *where declared* and nowhere else, and the check is a view.

### 9.3 Poses and animation

Poses are relations (§5.2) and double as tests (§7.3). Retargeting: clips are indexed by body-plan class (biped/quadruped/winged/serpentine); the retarget solver maps clip bones onto part-graph spines by class correspondence; playback streams through pose checks (L2). Spore's retargeting demonstrated feasibility on user morphologies in 2008; this kernel's stronger typing (joints at ports with limits) makes the problem easier, not harder.

### 9.4 Vocabulary growth (the slow path)

New part kinds, new sub-grammars, and new rewrite rules arrive as `propose(...)` into quarantine: code + frame doc (R-7) + verification harness. A verification agent must demonstrate: determinism, invariant-compatibility, conformance-suite neutrality, and at least one golden artifact. Promotion to `trusted` is a human-visible event. The fast path (daily creature work) is purely compositional over trusted vocabulary. This is the sanctioned door that replaces "the LLM writes Blender Python" — open, but guarded (`L-8` rationale).

---

## 10. Style and search: the 95% machinery

### 10.1 Speculation

Branches are cheap (z-sets; AX-8): the operator forks K variants of a scoped decision (a hand, a plate policy, a silhouette change), runs L0–L2 on all, spends L3 on survivors, merges the winner. Tournament-over-own-tests is the default idiom for any decision the operator is unsure of. The kernel MUST support ≥ dozens of live branches per creature with per-branch view maintenance.

### 10.2 Equality saturation

Two sanctioned uses, one guarded frontier:

1. **Canonicalization (articulation maintenance).** Rewrites that factor structure: near-mirror pairs → `Symmetry` rows; repeated chains → parameterized arrays; swallowed blobs → removed. Runs after sessions; keeps the artifact maximally *sayable*. Precedent: Szalinski (CSG decompilation via egg). Equivalence here is exact-geometry-within-solver-tolerance; extraction cost = editability (fewest params, most structure).
2. **Style-space search (equivalence relaxed to brief-level).** Rewrite families like `plate ↔ n·smaller plates + seams`, `ridge ↔ spike row`, licensed *by context*: region semantics, active packs, scrutiny band. The e-graph then holds stylistic variants compactly; extraction with aesthetic surrogate costs (calibrated per §8) makes globally consistent detailing decisions no greedy edit sequence can. This is the designated mechanism for coherent fine detail at scale.
3. **The guarded frontier.** Approximate equivalence is not transitive; naive merging under ε-equality is unsound. Style-space eqsat therefore REQUIRES context-sensitive machinery: scoped rule licensing, context-indexed equivalence classes, and drift bounds on rewrite chains. This kernel treats that machinery as a pluggable engine with the interface `RewriteRule(lhs, rhs, context_guard, equivalence_class)` + saturation/extraction calls, implemented over the same relational substrate (relational e-matching; egglog-style datalog+eqsat unification, incremental over the view plane). Where the host research library (context-sensitive eqsat) lands is exactly here.

### 10.3 Appearance layers as repeated pattern

Every appearance system is the same triple — `Complex` (cells) + `Field`s + a pack — instantiated with different physics: plates (built; reference `plates.py`), scales (smaller cells, overlap rule), fur/groom (direction field + singularities + clumping pack), crystal/spike scatter (growth field + scatter kind), membranes/VFX (annotation-only materials compiled to engine effects). Implementing agents SHOULD build plates→scales→groom in that order; each reuses the previous machinery.

### 10.4 Image conditioning

Reference images compile to briefs: segmentation → proposed `Region` graph + target silhouettes per canonical view + palette targets. A proposal agent MAY pre-fill a body-plan hypothesis (transcribed parts, quarantine-style); the operator accepts/edits symbolically. Score, never gate (AX-9).

---

## 11. Exchange plane

Compile targets (v1): **GLB/glTF** with mesh, skeleton (spines→bones), skin weights, per-cell vertex colors/materials, and a sidecar JSON carrying the addressable structure (parts, cells, fields) so downstream tools can round-trip references. **Engine annotations**: material/VFX tags (`membrane: arc-lightning`) compile to target-engine shader/particle presets per an adapter registry. **LOD ladder**: decimation with cell-boundary preservation; budgets are norms (§7.2). Print/CAD export MAY come later; nothing in the kernel presumes it.

**R-20.** Exchange is one-way: nothing re-enters state except through transcription (AX-9). The sidecar JSON is regenerated, never hand-edited.

---
## 12. Multi-agent operating model

### 12.1 Roles

| role | writes | reads | notes |
|---|---|---|---|
| **operator** | state plane (edit language) | receipts, views, renders | constructs creatures; the kernel's primary user |
| **verifier** | quarantine promotions, conformance results | proposals, corpus | gate for vocabulary/rewrites (§9.4) |
| **judge** | JudgeScore | renders + lineup only | context-free by construction (§8.2) |
| **distiller** | pack rule candidates | journals | mines failures → R-15 pipeline |
| **proposal** | quarantined transcriptions | briefs, images | image→region/body-plan hypotheses |
| **director (human)** | briefs, scrutiny, taste vetoes | everything | owns "should," never needs to touch "how" |

Builder agents implementing this spec are a seventh, temporary role; their contract is §14.

### 12.2 Concurrency

Optimistic transactions; disjoint-scope merges are automatic (R-19). Overlapping intent takes a **region lease** (advisory lock on an address subtree, journaled). Long speculative work happens on branches; merges are transactions like any other and pass the same L0/L1 gates. Judges and distillers never contend — they are read-only over snapshots.

### 12.3 Session shape (operator's reference loop)

1. Read brief + scrutiny; open branch.
2. Body plan pass: parts/ports/symmetry at low mesh res; L0/L1 receipts each edit; L2 pose smoke tests; one L3 checkpoint against silhouette targets.
3. Sub-grammar passes (hands, face, feet) inside their scopes with their packs.
4. Appearance passes: complexes + fields + materials, scrutiny-ordered.
5. Behavioral pass: full pose suites, clip retarget check.
6. Tournament any low-confidence decisions (§10.1); eqsat canonicalize (§10.2.1).
7. L3 judge panel; declare deviations for intentional weirdness; merge; compile.

The pilots ran a compressed version of this loop by hand; the kernel's job is to make each step one instruction.

---

## 13. Conformance and metrics

### 13.1 Golden corpus (v0, from the pilots)

`golem_v4.json` (body plan), `hand_v2.json` under `hand_pack` (0 violations; 20 on `hand_v1.json` — the report text itself is a golden output), `golem_hands.json` (composition; forearm re-aim expected), `plate_complex.json` (110 cells / 324 adjacencies at seed 7), the control-condition metrics (35 shells / 1 solid / not watertight), and all pilot renders (pixel-exact under the reference renderer, AX-8).

### 13.2 Kernel acceptance metrics

Integrity: components=1 (post-dust) and watertight on the volumetric solid (R-23); membrane anchoring violations = 0; symmetry-IoU ≥ 0.98 under declared `Symmetry`. Geometry conformance runs sealed (R-21, AX-8); fast-mode outputs are additionally checked against declared tolerances. Latency: L0 ≤ 10 ms, L1 ∝ delta and ≤ 100 ms at 10³-part creatures, incremental==full-recompute equality checks. Loop economics: violations-to-clear-per-round ≥ 10 (information-rich receipts, AX-4); rounds-to-pack-pass on the naive-hand replay ≤ 2. Perceptual: lineup pass at N=8 distractors; judge quality tracked longitudinally per pack version — the metric that must climb with accretion is *quality at fixed rounds*, not rounds.

### 13.3 The standing A/B

The control condition (free primitive code, same model, same rounds) is rerun quarterly against the current kernel on a rotating reference image. If the kernel does not beat the control on integrity *and* judge quality by growing margins as packs accrete, AX-6 is failing in practice and the accretion pipeline gets audited. The pilot's honest result — mechanical wins, aesthetic tie at Layer 0 — is the baseline to beat (`L-5`).

---

## 14. Build plan (for implementing agents)

Phases are dependency-ordered; each has a hard acceptance gate. Do not start N+1 before N's gate passes. At every phase, the Python pilot artifacts are the oracle: same inputs → equivalent outputs (geometry within meshing tolerance, reports semantically identical).

**Phase 0 — Relational core + journal.** State relations (§5.2), canonical serialization, transactions, journal/replay, branches as z-sets. Gate: replay any pilot JSON as transactions; branch/merge property tests; AX-8 determinism harness.

**Phase 1 — Solver baseline + receipts.** Port `engine.py` semantics (gencyl/blob/smin/meshing) behind the solver interface; CoherenceReport view; L0 expression evaluator; transaction receipts (R-16). Gate: golden corpus geometry + `CoherenceReport` matches pilot outputs including the v2-sliver and claw-dust cases (`L-2`).

**Phase 2 — Constraint kernel.** Packs/rules/tiers/deviations as relations; incremental Violation views (DBSP); hand pack ported to `Rule`/`TestPose` rows evaluated by the kernel (not bespoke Python). Gate: naive-hand replay produces the 20 golden violations with R-14 formatting; L1 latency budget on synthetic 10³-part creatures.

**Phase 3 — Composition + behavior.** Ports, ifaces, bidirectional attach solve (R-11), sub-grammar packaging, pose relations + suites, contact solve. Gate: golem-hand mount succeeds *without* manual forearm re-aim (the solver negotiates it); `mount_hands.py`'s recorded failures reproduce as violations when solve_policy forbids negotiation.

**Phase 4 — Appearance substrate.** Complexes, fields, sections, sheaf relaxation solver, plates end-to-end from `Complex.policy` (not a script); scales; groom v0 (direction field + singularity declarations + validation view). Gate: plate golden artifact reproduced from policy; declared-singularity check catches an undeclared cowlick seeded adversarially.

**Phase 5 — Perception + judges + calibration.** Deterministic renderer parity, silhouette scoring vs briefs, judge harness, scrutiny-driven allocation, surrogate calibration loop. Gate: pilot-2 judging replicates; calibration measurably improves surrogate→judge correlation on a held-out set.

**Phase 6 — Search.** Branch tournaments as one instruction; eqsat canonicalization rule set (symmetry factoring, array factoring) with editability-cost extraction; style-rewrite interface stubbed for the context-sensitive engine (§10.2.3). Gate: canonicalizer factors a hand authored as four literal fingers into an array + symmetry without geometry change; tournament API survives 50-branch stress.

**Phase 7 — Exchange.** GLB + skeleton + weights + sidecar; engine annotation adapters; LOD. Gate: golem-with-hands round-trips into a game engine scene with skeleton intact and cells addressable via sidecar.

Parallelization notes for a builder-agent fleet: Phases 1–2 admit heavy intra-phase parallelism (solvers vs views); Phase 4's three appearance systems are independent after the sheaf solver lands; conformance harness construction (§13) can proceed from day one and SHOULD be its own agent.

---

## 15. Risks and open problems (ranked by expected pain)

1. **The aesthetic ceiling is still a hypothesis.** Pilot 2's honest result: blind judges saw no quality difference at Layer 0. The bet is that solver-shaped layers (fields, plates, groom) + accreted packs + eqsat detailing move judge quality decisively. Falsifiable via §13.3; if falsified, the kernel remains a superior *integrity and iteration* substrate, but the thesis needs revision.
2. **Faces.** Scrutiny's global maximum; blob+cylinder will not carry it. Plan: dedicated face sub-grammar (FACS-parameterized blendshape symbols) as its own pack-heavy vocabulary entry. Until it exists, creatures ship stylized or masked faces. Do not improvise faces from the body vocabulary.
3. **Approximate-equivalence soundness in style eqsat.** Known-unsound naively (§10.2.3). Gated on the context-sensitive engine; ship canonicalization first, which needs no relaxation.
4. **Organic smoothness at the ape tier.** Smooth-min blends read "maquette." Sheaf-relaxed muscle/volume fields over the skeleton are the planned answer; unproven. The ape reference is the acceptance target.
5. **Retarget quality.** Typed joints make retargeting tractable, not good. Budget animator-taste accretion (clip-quality packs) the same way as anatomy.
6. **Pose-check scaling.** Naive capsule-pair checks are O(n²); BVH or the view plane's incremental joins must hold the L2 budget at wing/tail part counts.
7. **Mesh-resolution economics.** Uniform grids waste 10–100× at creature scale; adaptive meshing (Phase 1) is load-bearing for L2 latency, not a nicety.
8. **Judge drift and gaming.** Surrogates calibrated to judges can be Goodharted by search (§10). Rotate judge rubrics/lineups; keep the human director's veto in the loop for style briefs.

## 16. Lessons register (pilot → rule)

L-13 through L-15 entered via independent operator-model review of v0.1 — the register accepts review findings exactly as it accepts pilot failures.

| id | lesson (observed) | encoded at |
|---|---|---|
| L-1 | Named-delta edits converge in ~1 round when errors carry address+magnitude+range | R-3, R-14 |
| L-2 | Sub-grid features silently shed dust; metrics must separate dust from detachment; min-feature must be symbolic and pre-mesh | R-5, R-8 |
| L-3 | Coherence metrics catch what renders hide (v2 sliver) | §8 L1, R-5 |
| L-4 | A violation formatted as instruction ("1.00 outside [0.55,0.90]") is directly actionable; critiques are not | R-14 |
| L-5 | At Layer 0, substrate wins are mechanical (integrity/symmetry/diffs), not aesthetic — blind judges tied | §13.3, Risk 1 |
| L-6 | Unconstrained substrates fix seams by adding geometry: shell count grows monotonically; symmetry survives only by authoring discipline | R-9, §2 |
| L-7 | Uniform-radius primitives cannot taper; taper must be native to the workhorse kind | §6.1 |
| L-8 | Same model, same rounds: mitt vs pack-passing hand — the pack is the difference; know-how must live in checkable artifacts | AX-6, R-15 |
| L-9 | Behavioral tests expose what statics cannot (naive hand passed `point`, failed `fist`) | §7.3 |
| L-10 | Attachment is a two-sided constraint; frame conventions must be written down (both mount failures were frame errors) | R-7, R-11 |
| L-11 | "Measure and lift" ground contact previews a real placement solver; contact belongs in the solver plane | §6.2.2 |
| L-12 | Blind context-free judge agents are cheap, fast (≈13 s, ≈16k tokens each), and consistent (6/6 @ 0.9) — perception can be a commodity sense | §8.2 |
| L-13 | Bit-exact determinism across parallel FP reduction, CG solves, FFI, and GPU is a tar pit; demanding it quietly serializes everything — determinism must partition into a bit-exact symbolic surface and a tolerance-conformant numeric surface with a sealed mode | AX-8, R-21, A-8 |
| L-14 | A sensory stack of only delta/error senses leaves a fresh session with no token-efficient gestalt; orientation (canonical depth-controlled outline) must be a first-class pull sense | §8.5, R-22 |
| L-15 | Membranes as open surfaces contradicted the watertight invariant as written; the invariant scopes to the volumetric solid and membranes get anchoring invariants of their own | R-23, §13.2 |

## 17. Glossary

**Operator** — the LLM constructing creatures. **Pack** — versioned bundle of rules + pose tests bound to a scope. **Port** — typed attachment interface carrying a frame, profile, and joint. **Sub-grammar** — packaged fragment (compiler + pack + tests) behind a port signature. **Complex/Cell/Section** — surface decomposition, its units, and per-cell field values (the sheaf's base, stalks' carrier, and sections). **Scrutiny field** — spatial allocation of perceptual budget and strictness. **Metabolism** — a functional pack making validity teleological. **Receipt** — synchronous L0/L1 result of a transaction. **Ratchet** — the accretion protocol (R-15). **Brief** — scored targets from director/reference; never blocking. **Quarantine** — where proposed vocabulary lives until verified.

---

*End of v0.2 main text. v0.2's own amendments arrived exactly this way — three findings from an independent operator-model review of v0.1, entered as L-13…L-15. The next revision belongs to the agents who try to build Phase 0–1 against it.*

---

# Appendix A — Haskell realization notes (normative for builders)

The kernel will be implemented in Haskell, alongside the host research libraries (computational sheaves; DBSP-style differential dataflow with circuits; context-sensitive equality saturation). These notes bind implementation decisions where language choice interacts with the axioms.

**A-1 (Types for builders, text for operators).** The operator is an LLM and never sees a Haskell type. Type sophistication is internal armor: it MUST NOT leak into receipts, addresses, violation messages, or serialized relations. When a design choice trades internal type elegance against operator-surface simplicity, the operator wins. Corollary: port interfaces (`iface`, §6.3) are value-level records validated by smart constructors, not type-level programs — a mount failure must be a `Violation` row the operator can read, not a compile error the operator never sees.

**A-2 (Purity is AX-8).** The kernel core is a pure function:

```haskell
step :: KernelState -> Transaction -> Either Rejection (KernelState, Receipt)
```

All solvers, meshers, renderers, and view updates live inside `step` or pure functions it calls. The IO shell does exactly: journal persistence, artifact file emission, agent transport. Randomness is a seeded generator carried in state. Wallclock never enters the core. Determinism tests (same journal ⇒ bit-identical state and artifacts) are Phase 0 gates, and purity makes them cheap rather than heroic.

**A-3 (Branches are persistent data structures).** `KernelState` uses persistent maps (HAMTs); a branch is a pointer, a merge is z-set addition over deltas. Haskell's structural sharing makes §10.1's "dozens of live branches" a memory non-event rather than an engineering project. This is the single largest language dividend: AX-8 + immutability gives speculation nearly free.

**A-4 (Z-sets and the view plane).**

```haskell
newtype ZSet a = ZSet (Map a Weight)   -- Weight = Int; Abelian group
type Transaction = ZSet Row            -- Row: sum over relation row types
```

Relations are one ADT per relation (typed rows), with a `Row` sum for the journal and canonical serialization (stable field order, sorted keys — AX-10). The view plane compiles `Rule.predicate_expr` to circuits in the DBSP library; the *same* expression AST (a GADT `Expr t`, total by construction: no general recursion, §5.4) is interpreted row-at-a-time for L0. One AST, two evaluators, conformance-checked against each other (incremental == full recompute, §13.2).

**A-5 (Geometry performance).** Grids on unboxed arrays (`massiv` or equivalent) with data parallelism; marching cubes pure; adaptive octree refinement per R-8's feature-size knowledge (the symbolic layer already knows where small features are — use it to refine, don't discover smallness numerically). FFI to C for hot loops is permitted only behind referentially transparent interfaces with golden tests; GPU (Accelerate or FFI) is a Phase-5+ optimization, never a semantic dependency.

**A-6 (Sheaf machinery).** Complexes are data (face posets from `Cell`/`CellAdj`); a sheaf assigns stalk dimensions and restriction maps (concrete matrices); the Laplacian assembles sparsely; sections solve by CG with operator boundary conditions pinned. The categorical formulation (functor from the face poset) may organize the host library; the kernel consumes matrices and returns per-cell residuals as addressed `Violation`s (R-10). Obstruction is a number with an address, not a theorem.

**A-7 (Vocabulary growth without dynamic loading).** New part kinds and rewrites arrive as *data*: terms in the kernel's SDF-combinator / expression DSL (deep embedding), not compiled Haskell. Quarantine verification is then decidable mechanically — totality, bounds, feature-size compatibility — and the "LLM extends its own language" loop (§9.4) never requires GHC at runtime. Compiled extensions ship only with kernel releases. This resolves the classic Haskell plugin problem by design rather than by tooling.

**A-8 (Concurrency).** The core is sequential per creature (transactions are cheap; contention is rare by §12.2 leases); parallelism spends itself where it pays — solver-plane data parallelism and multi-branch evaluation (`parMap` over pure branch evaluations). STM only in the shell (lease table, agent sessions). Parallelism lives on AX-8's numeric surface only: `parMap` reduction order, CG iteration, FFI and eventual GPU kernels may vary bits within declared tolerances, and sealed mode disables them for conformance. Nothing on the symbolic surface may depend on numeric-surface bits except through tolerance-aware predicates (R-21).

**A-9 (Testing discipline).** `tasty-golden` over the pilot corpus (§13.1); property tests (hedgehog): disjoint-scope merge commutativity, op-inverse round-trips, incremental/full agreement, determinism-under-replay. The naive-hand 20-violation report is a golden *text* artifact: receipts are part of the specified surface, not incidental output.

**A-10 (Module map → phases).** `Golem.Relation` / `Golem.ZSet` / `Golem.Journal` (Phase 0) · `Golem.Expr` / `Golem.Solver.Field`, `.Mesh` (Phase 1) · `Golem.Constraint` / `Golem.Pack` (Phase 2) · `Golem.Port` / `Golem.Pose` / `Golem.Solver.Contact` (Phase 3) · `Golem.Complex` / `Golem.Sheaf` (Phase 4) · `Golem.Percept` / `Golem.Judge` (Phase 5) · `Golem.Search.Branch` / `Golem.Search.EqSat` (Phase 6, hosting the context-sensitive engine) · `Golem.Exchange` (Phase 7).

---

# Appendix B — Prior-art autopsy: Blender (inspiration, not lineage)

Blender is the most instructive existing system because it solved, over thirty years and for a *human visual operator*, three of the same problems this kernel solves for a linguistic one. Study it for the shapes of those solutions; do not import its assumptions. Sources: the [dependency graph developer documentation](https://developer.blender.org/docs/features/core/depsgraph/), the [attributes & fields design post](https://code.blender.org/2021/08/attributes-and-fields/), and the [2016 depsgraph proposal](https://code.blender.org/2016/12/dependency-graph-proposal/).

## B.1 What to steal

**Fields and attribute domains (Geometry Nodes) — steal the concept, upgrade the semantics.** Blender's attributes are per-element data on typed domains (point/edge/face/corner); its fields are *functions* passed backward through the node graph and evaluated lazily per domain element, composable into new fields, with automatic domain interpolation. Their stated win is precisely ours: the artist can "change the topology of the geometry while the function flow still works as expected." That is the industrial validation of §5.2's `Field`/`Section` design — fields as functions over addressable domains survive re-meshing, which is why GOLEM's operator authors boundary conditions and singularities rather than values. Two upgrades ours makes: GOLEM domains are *semantic* cells (plates, regions), not only mesh-topological elements, so field addresses mean something to a language model; and GOLEM fields are solver-filled under constraints (sheaf relaxation) rather than merely lazily evaluated — a field can be *wrong* in our system, and wrongness has an address.

**The depsgraph — steal the goal, change the granularity.** Blender's dependency graph exists so that edits recompute "only what was dependent on the modified value," with copy-on-write separation of source data from evaluated results so threaded evaluation never mutates originals. That is the ancestor of the view plane, and its copy-on-write discipline is our immutability-by-construction (A-3) discovered independently under C++ constraints. The lesson in its limitations: node granularity sets the floor of edit latency. Blender's graph is entity/component-grained, so small edits can trigger large re-evaluations; GOLEM's view plane is row-grained (R-2) because the operator's edits are row-sized. Their modernization backlog is our starting requirement.

**The modifier stack — the ancestor of solved-not-generated.** Non-destructive, ordered procedures over untouched source data is AX-2 in embryo. GOLEM generalizes the linear stack into the solver plane (a DAG of deterministic passes) and makes the source data relational.

**DNA/RNA — the cost of forward compatibility, paid early.** Blender's self-describing data layer bought thirty years of file compatibility and a uniform reflection API (which the Python layer and animation drivers ride). GOLEM's equivalents are the canonical relation serialization and versioned schema (§5) — and the anti-lesson sits beside the lesson: `.blend` is a binary memory dump, undiffable and unquotable. GOLEM's state is text because the operator thinks in text (AX-1, AX-10).

## B.2 Anti-patterns (the assumptions not to import)

**The scripting surface is imperative mutation against live shared state.** `bpy` edits the scene in place: no transactions, no receipts, no validation tier, consequences discovered by looking at the viewport. This is precisely the substrate in which "LLM writes Blender Python" fails — the pilot's control condition reproduced the failure shape in miniature (35 shells, seams fixed by adding geometry, `L-6`). GOLEM's edit language (§9.1) is the deliberate negation: transactional, validated, receipted.

**No constraint tier exists at all.** Nothing in Blender knows a hand should taper or a body should be one component; correctness is outsourced to the artist's eyes at 60 fps. Reasonable when the operator has eyes; fatal when the operator has receipts. The entire §7/§8 apparatus is the part Blender never needed to build.

**Weak durable addressability.** RNA paths exist, but names are mutable, IDs are not stable across undo/rename in the way R-3/R-4 demand, and historical undo is snapshot-shaped rather than journal-shaped. A system whose operator *refers to things in sentences* lives or dies on stable addresses; a system whose operator points and clicks does not.

**bmesh is a vertex-level motor system.** Superb for a human sculptor's direct manipulation; it is the exact capability R-18 forbids the operator, because vertex-level freedom is where autoregressive error compounds.

**UI-first economics.** Blender optimizes for frames-per-second under a mouse; its known "spaghetti graph" complaint is the visual analogue of context overflow. GOLEM optimizes for tokens-per-decision and receipt quotability (AX-10). Same discipline, different currency.

## B.3 Summary judgment

Blender demonstrates that procedural, dependency-driven, per-domain-field geometry systems work at industrial scale and thirty-year time horizons — and that all three pillars were shaped end-to-end by the assumption of a human visual operator. GOLEM re-derives those pillars for a linguistic operator and adds the two layers Blender's assumption made unnecessary: a constraint/receipt system (because our operator cannot glance) and a text-native, journaled, addressable state (because our operator speaks). Houdini's attribute-driven proceduralism is the adjacent deeper prior art on fields and is worth a later, similar autopsy; nothing in it changes the judgment above.

*Appendices v0.2 — same amendment discipline as the main text: builder disagreements enter §16 as lessons.*
