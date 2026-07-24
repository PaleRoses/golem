# AIDL Method Transliteration for GOLEM body/0.3

Status: design only

This plan transliterates the method described by Jones et al. in [A Solver-Aided Hierarchical Language for LLM-Driven CAD Design](https://arxiv.org/abs/2502.09819). It does not copy or depend on the authors' implementation.

## 1. Decision

GOLEM will add a relational placement layer to body/0.3. Authors state relationships over the existing selector vocabulary; the compiler defers equation construction until the complete skeleton tree and its constructed sites are known; a deterministic hierarchical least-squares descent derives the absolute bone poses consumed by the existing body compiler.

The change is additive:

- Existing attach objects containing t and offset remain valid and become fixed local-translation constraints.
- A document with no relational attachments and no pose relations takes the current path, does not import SciPy for placement, and emits byte-identical graph and receipt data.
- The existing body frame convention remains authoritative: bones extend along local +z and frames are built by the minimal-rotation look-frame logic in golem/kernel/body/linalg.py.
- The exact two-bone reach solver in golem/kernel/body/goals.py remains its own semantic owner. The first cut rejects relations that attempt to control a goal-owned chain rather than allowing two solvers to silently overwrite one another.
- Body relations operate only on constructed pre-boolean geometry. Meshes, SDF unions, assembly solids, plates, conduits, vascular results, and other post-solve products are not referenceable.

The vascular CCO will separately gain staged relaxation. It will retry fixed corridors, then movable synthetic waypoints, then a wider backtracking search. InfeasibleBifurcationObstruction will mean that the configured finite search space was exhausted. A search-budget cutoff will have a different obstruction and will not counterfeit infeasibility.

## 2. Source authority and ownership

The new behavior must strengthen these live owners rather than inventing parallel systems.

| Concern | Current semantic owner | Decision |
|---|---|---|
| Selector ADTs and parsing | golem/addressing/scope.py and golem/addressing/scope_grammar.py | Reuse Bone, Part, Landmark, parse_scope, and render_scope directly. Do not add a second reference grammar. golem/addressing/core.py and __init__.py remain compatibility surfaces, not new owners. |
| Parametric bone anchors | golem/addressing/anchor.py and anchor_grammar.py | Preserve Anchor for t-based authoring. Relational attachment targets are Scope selectors, not a widened Anchor string. |
| Attachment and frame equations | golem/kernel/body/kinematics.py | Preserve child.head = parent.head + parent.R applied to local_translation, with local_translation derived from offset + [0, 0, t * parent.length] for fixed attachments. |
| Minimal-rotation frames | golem/kernel/body/linalg.py | Reuse _frame_from_axis. The relational solver derives local translations, axes, and twist; it does not create a second frame convention. |
| Body compilation order | golem/kernel/body/compile.py | Insert relation decoding and solving before authoritative forward kinematics and emission. |
| Existing reach IK | golem/kernel/body/goals.py | Retain as the sole owner of legacy pose.goals. Reject overlapping relation authority in this cut. |
| Body receipts | golem/kernel/body/intent.py | Project accepted relation measurements into the existing solved receipt block. No empty block on legacy documents. |
| Direct body CLI | golem/kernel/body/cli.py | Exhaustively match the typed body result, render rejection through the body projector, write no graph or receipt on rejection, and return nonzero. |
| Session compile outcome | golem/session/state.py and effect.py | Preserve body obstruction values inside CompileObstructed instead of converting them to exception_type/message. Unexpected legacy exceptions remain a distinct session obstruction variant. |
| Silhouette body loading | golem/senses/silhouette.py | Exhaustively match compile_file and return a typed geometry-load obstruction before mesh evaluation. |
| Evaluation compile boundaries | golem/evals/eval_anatomy.py and eval_plate_authoring.py | Exhaustively match body rejection and deterministically project its complete evidence into the existing terminal EvaluationObstruction protocol. |
| Body authoring contract | golem/contract/{schema,vocabulary,primer,receipts,envelope,registry}.py | Extend these owners and add one registered relations section. Do not create another contract service. |
| Check tier | golem/cli/check.py | Propagate typed body relation rejection and run the solve before sensing, still without meshing. |
| Assembly body ingestion | golem/assembly/ingest.py | Propagate typed body rejection while preserving the body compiler as the placement owner. |
| Assembly compilation | golem/assembly/core.py and golem/assembly/compile.py | core.py remains the compiler hub; compile.py remains element meshing, plate, conduit, and vascular-record derivation. It consumes an already solved body graph and does not own body placement. |
| Obstruction rendering | golem/assembly/__main__.py | Its generic frozen-dataclass rendering already preserves fields. Reuse it rather than writing a second formatter. |
| Vascular topology and obstructions | golem/kernel/anatomy/graph.py and project.py | Preserve the in-flight enriched InfeasibleBifurcationObstruction fields: phase, lane, predicate, required, observed, and candidate evidence. |
| Vascular corridor construction | golem/kernel/anatomy/realize/corridor.py and parity.py | Strengthen the existing skeleton-derived corridor and per-circuit descent. |
| Vascular insertion | golem/kernel/anatomy/realize/cco.py and bifurcation.py | Replace Optional failure and greedy continuation with typed candidate rejection and bounded backtracking. Reuse _SeedSearch and strengthen _InsertionBatch. |
| Vascular limits | golem/kernel/anatomy/vocabulary.py | Add sealed search budgets and waypoint bounds here, projected by the contract envelope. |

Four absent concepts are justified:

1. golem/kernel/body/relations.py: the closed relation vocabulary, frozen normalized declarations, accepted/rejected decode results, and body relation obstruction ADTs. It is consumed by validation, solving, schema/contract projection, and receipt rendering.
2. golem/kernel/body/relation_solve.py: the pure hierarchical descent and the single SciPy boundary. No existing module owns general relational placement.
3. golem/kernel/body/geometry.py: pure analytic projection of constructed pre-boolean flesh sites and directional support. This consolidates existing placement logic split between emit._placed_center and kinematics._flesh_low_y and supplies the third relation-solver call site. The old helpers are deleted or rewired in the same change; no delegation wrapper remains.
4. golem/kernel/body/project.py: a total pure projection from RejectedBody and every BodyCompileObstruction constructor to canonical dictionaries and stable text. The direct body CLI, check CLI, session outcome, assembly ingestion, silhouette loader, and evaluation adapters all require the same projection. Centralizing this function prevents six incompatible string renderers without creating a second semantic owner; the ADT in types.py remains authoritative.

No general Python expression IR is added. GOLEM_KERNEL_SPEC.md section 5.4 already assigns a total expression language to the broader relational kernel, while COLLAPSE_CAMPAIGN.md explicitly defers the greenfield selector/cascade language. This plan therefore exposes typed numeric relation fields only. Internal residual expressions may use finite min/max branches, but they are not a parallel public Expr DSL.

## 3. Authoring surface

### 3.1 Relational attachments

The existing bone attach field gains a second, disjoint form:

    {
      "id": "forearm",
      "parent": "upper_arm",
      "attach": {
        "at": "landmark:upper_arm/wrist"
      },
      "length": 0.42,
      "rest_dir": [0.1, 0.0, 1.0],
      "joint": {"dof": "hinge"}
    }

The two forms are:

| Form | Meaning |
|---|---|
| {"t": number?, "offset": [x,y,z]?} | FixedLocalAttachment. Defaults remain t=1 and offset=[0,0,0]. The resulting local translation is immutable during every deepening stage. |
| {"at": "landmark:..."} | NamedSiteAttachment. The child head is movable and must coincide with the selected constructed site. The authored rest direction remains the initial orientation. |

A mixed object containing at together with t or offset is rejected as MalformedRelationObstruction. It would otherwise create two placement owners in one record.

Two children whose at values normalize to the same selector share the same site section. Their child heads therefore receive identical coincidence equations; coincidence is implicit in the shared reference rather than recomputed coordinates.

There is no second top-level attach_at relation syntax. The attach field is already the attachment owner, so duplicating it would be decorative debt.

### 3.2 Explicit pose relations

Explicit relations live under the existing pose owner:

    {
      "pose": {
        "joints": {},
        "goals": [],
        "relations": [
          {
            "id": "head_above_chest",
            "kind": "above",
            "subject": "part:head",
            "reference": "part:chest",
            "distance": 0.04,
            "solve": "subject"
          },
          {
            "id": "shoulder_pair",
            "kind": "mirror_of",
            "subject": "bone:arm_right",
            "reference": "bone:arm_left",
            "solve": "negotiate"
          },
          {
            "id": "wrist_axis",
            "kind": "perpendicular_to",
            "subject": "landmark:forearm/wrist",
            "reference": "bone:forearm",
            "solve": "subject"
          }
        ]
      }
    }

Each declaration has:

- id: stable non-empty relation identity, unique within pose.relations.
- kind: a canonical relation identifier or an accepted alias.
- subject and reference: strings parsed by the existing Scope grammar.
- solve: one of subject, reference, or negotiate.
- distance: finite non-negative world-unit separation for above/below; absent means zero.

solve is a closed policy:

- subject: only the subject side may yield.
- reference: only the reference side may yield.
- negotiate: either side may yield, subject to fixed attachments, goal ownership, and hierarchical deepening.

No policy can unlock a FixedLocalAttachment. Absolute authoring is fixed because the user explicitly chose coordinates.

### 3.3 Referenceability

Relation selectors refine existing Scope values by capability:

| Scope | Constructed meaning | Allowed uses |
|---|---|---|
| Bone | The authored bone structure, its frame, and the finite constructed geometry in its subtree | above, below, mirror_of, aligned, perpendicular_to |
| Part | A named authored flesh primitive before SDF union or meshing | above, below, mirror_of, coincident at its analytic center, aligned/perpendicular_to by its bone frame |
| Landmark | A bone head, tail, center, or named skeletal port; ports carry a frame | attach target, coincident, aligned, perpendicular_to |

Part resolution uses authored flesh identity and golem/kernel/body/geometry.py. It never asks the compiled boolean surface which face, edge, or component survived.

The following existing scopes are rejected in body relations: Whole, World, Chain, Contact, Element, Port, Mount, and Region. The assembly-qualified Port scope is not reused for body ports; body ports already appear as skeletal landmarks such as landmark:forearm/wrist. Prop landmarks are excluded from this cut because prop placement currently occurs after skeletal FK and allowing them would introduce a cyclic placement owner.

Unknown or unsupported selectors produce typed obstructions with the original selector, the relation address, the supported scope kinds, and closest valid constructed identifiers when available.

## 4. Closed relation vocabulary and aliases

The canonical semantic set is:

| Canonical id | Residual meaning | Accepted aliases |
|---|---|---|
| attach_at_named_site | child head equals a named skeletal site | attach_at, attached_at, attach_to_site |
| above | minimum subject y is at least maximum reference y plus distance | over |
| below | maximum subject y is at most minimum reference y minus distance | under |
| mirror_of | subject pose is the sagittal reflection of reference across world x=0 | mirrored_from, reflection_of |
| perpendicular_to | selected local +z axes have zero dot product | perpendicular, orthogonal, orthogonal_to |
| coincident | selected points have zero separation | coincident_with, coincides_with |
| aligned | selected local +z axes are parallel under the inferred direction convention | align, aligned_with, parallel_to |

Resolution order is:

1. Normalize surrounding whitespace, case, and hyphen versus underscore spelling.
2. Match a canonical id.
3. Match an explicit alias and return its canonical RelationKind.
4. Otherwise use difflib.get_close_matches with n=3 over canonical ids and alias keys.
5. Map every alias match back to its canonical id, preserve similarity order, and deduplicate.

Suggestions always print canonical ids. An exact alias is accepted without warning, while the solve receipt records authored_kind and canonical_kind. This preserves a closed semantic set while tolerating lexical variance.

The existing _closest_intent_candidates in golem/assembly/compile.py remains local. It suggests dynamic author-defined appendage intent ids and has no alias semantics; merging that behavior with the closed relation vocabulary would create a vague utility that owns no common law.

## 5. Relational semantics

Every relation lowers to normalized residual components and a measurement projection:

| Relation | Solver residual | required | observed |
|---|---|---|---|
| attach_at_named_site | child_head - target_point | 0 distance | Euclidean distance |
| coincident | subject_point - reference_point | 0 distance | Euclidean distance |
| above | max(0, reference_max_y + distance - subject_min_y) | distance | subject_min_y - reference_max_y |
| below | max(0, subject_max_y + distance - reference_min_y) | distance | reference_min_y - subject_max_y |
| aligned | chosen signed axis difference | 0 degrees | angle under chosen convention |
| perpendicular_to | axis_subject dot axis_reference | 90 degrees | unsigned axis angle |
| mirror_of | position and frame difference after sagittal reflection | 0 distance and 0 degrees | maximum positional/angular deviation |

Position and angular residuals are normalized by separate sealed tolerances before entering least_squares. Acceptance checks the original, unnormalized measurements.

The solver variables are existing body-placement parameters, not a second pose model:

- local_translation: the three parent-frame coordinates used by kinematics.py;
- swing: two tangent coordinates that change the bone's local +z direction from its authored rest_dir;
- twist: one coordinate consumed by _frame_from_axis;
- no scale or length variables in this cut;
- no variables for fixed attachments;
- no variables for a pose.goals-owned chain.

Subject/reference policy and relation kind decide which coordinates are eligible. For example, above/below unlock translation, aligned/perpendicular_to unlock swing, and mirror_of unlocks translation, swing, and twist. A value may be an initial condition without being a fixed condition; rest_dir and twist_deg are seeds when their bone is an eligible yielding subject.

Ruling (orchestrator, 2026-07-21): above/below are verified assertions, not positioners. The attach field is the single placement owner — a NamedSiteAttachment's synthesized coincidence determines all three translation coordinates, so an above/below hinge can only be slack-satisfied or conflict (FixedPlacementConflict for a fixed bone; RelationalSolveExhausted for a named-site bone in a genuine tug-of-war). This is intended: vertical placement freedom is expressed by site choice, and orientation freedom by swing/twist under aligned/perpendicular_to/mirror_of. A placement form that frees only the vertical coordinate would create a second translation owner and is rejected. The contract teaches above/below as intent assertions the solver verifies with quantitative evidence.

## 6. Deferred hierarchical solve

### 6.1 Site and cover

The site is the finalized skeleton tree. For each root or bone b, define U_b as b together with its descendant subtree.

- Local sections are immutable maps from bone ids in U_b to solved local translations and frame parameters.
- A relation is assigned to the lowest common ancestor of its subject and reference owners.
- The overlap between U_b and a child cover U_c is the child boundary frame plus the child's solved internal constraints.
- Compatibility means that restricting the parent solution to the child boundary preserves every constraint previously accepted in U_c within tolerance.
- A failed compatibility check is a typed obstruction, not an exception or a global partial commit.

This is the GOLEM form of AIDL's post-order recursive solve: descend to local sections, reconcile overlaps, then glue one authoritative solved placement.

### 6.2 Finalization before equation construction

decode_relations performs only syntax, vocabulary, and selector-shape validation. Equation construction is deferred until:

- the root and every bone are indexed;
- parent/child relationships and lowest common ancestors are known;
- authored flesh ids, bone landmarks, and ports are known;
- the provisional legacy FK establishes initial geometry;
- relation aliases and direction conventions have canonical ids.

The compiler then constructs each local problem from the finalized topology. No residual captures a half-built tree.

### 6.3 Solve order

The solve is a post-order fold over the skeleton:

1. Solve each child section.
2. Treat solved child sections as constants.
3. Solve the current node's direct geometry and its LCA-owned relations.
4. If accepted, glue the current section.
5. If rejected, deepen the eligible variable frontier without modifying any fixed datum.

The immutable accumulator is either AcceptedRelationalSection or RejectedRelationalSection. Sibling obstructions are accumulated applicatively; one failed branch does not erase measurements from another.

### 6.4 Iterative deepening

Each local problem uses the following exact stage order:

1. Local: free only the current structure's eligible translation, swing, and twist coordinates. Descendants are constants.
2. Translation depth d: add eligible translation coordinates for descendant roots exactly d edges below the current node. Moving a child translates its whole solved section rigidly. Re-add every ancestor relation whose restriction includes that child boundary.
3. Parameter depth d: after all translation depths fail, add eligible swing and twist coordinates for descendants at depth d. Re-add the descendant's own relation set as well as affected ancestor relations because internal geometry can now change.
4. Exhausted: fail only after every eligible depth in both stages has been attempted.

The frontier is cumulative within a stage and resets between translation and parameter deepening, matching AIDL's preference for moving a solved child before changing its internal geometry.

Fixed attachments, the root world placement, and goal-owned chains never enter a frontier. A relation requiring one of them to move reports FixedPlacementConflictObstruction or GoalRelationAuthorityObstruction with the unsatisfied measurement.

### 6.5 Numeric solver

Use scipy.optimize.least_squares as the single numeric interpreter:

- method is trf;
- loss is linear;
- variable and residual ordering is canonical DFS order followed by relation id;
- the initial vector is derived solely from authored geometry;
- there are no random restarts;
- tolerances, maximum evaluations, and maximum active-set re-solves live in a frozen RelationalSolveConfig;
- SciPy is imported only inside relation_solve.py and only when at least one relation is present.

least_squares is preferred over root because GOLEM must support overdetermined systems, inequality hinge residuals, and explicit rank diagnosis with one solver contract.

Residual construction, variable projection, rank analysis, active-set selection, and receipt derivation remain pure functions over frozen carriers. A single run_least_squares boundary converts one immutable LocalRelationProblem into one immutable SolverAttempt; no compiler state is exposed to SciPy and no trial iterate becomes authoritative before acceptance and gluing.

Every trial returns a frozen attempt receipt. The solver result is accepted only when all original relation measurements satisfy their typed tolerances and every reopened child constraint remains satisfied.

### 6.6 Under- and over-constraint diagnosis

The displacement-from-initial quantity is a tie-breaker after constraint satisfaction, never a residual that hides missing intent.

- Under-constraint: compute the rank of the unregularized accepted Jacobian using a sealed singular-value tolerance. If rank is below the number of eligible variables and the null space changes an authoritative pose, return UnderconstrainedRelationObstruction with variable_count, rank, free_dimension_count, and the affected addresses.
- Exact fixed conflict: evaluate fixed-only equality and inequality components before SciPy. Contradictory fixed measurements return FixedPlacementConflictObstruction immediately.
- Nonlinear exhaustion: if all deepening stages finish with unsatisfied residuals, return RelationalSolveExhaustedObstruction. Do not label it mathematically infeasible unless the fixed contradiction check proved that narrower fact.
- Evaluation cap: return RelationalSolverBudgetObstruction with required evaluation budget and observed evaluations. Do not fabricate a pose from the best partial iterate.

## 7. Ambiguity by nearest satisfaction

Any relation with more than one direction or orientation convention exposes a finite ConventionId set.

Before solving:

1. Evaluate every convention against the provisional constructed geometry.
2. Normalize each convention's residual by the same typed tolerances used for acceptance.
3. Select the least residual norm.
4. Break a tolerance-level tie by canonical ConventionId order.
5. Freeze the selected convention for the entire hierarchical solve.

Examples:

- aligned chooses parallel versus anti-parallel.
- A full-frame perpendicular relation chooses the signed quarter-turn closest to the initial frames.
- mirror_of chooses the proper SO(3) frame projection closest to the initial orientation after reflection, never an improper reflected matrix.

The receipt records relation id, candidate convention ids, initial residuals, selected id, and whether a tie-break was needed. Conventions never switch mid-solve; allowing that would introduce a hidden discontinuity.

above and below always mean world +y and world -y respectively. They are not ambiguous in body/0.3.

## 8. Finite min/max active-set iteration

Bone and Part structural extents are derived from finite constructed pre-boolean geometry. above and below therefore contain min/max branches.

For each local solve:

1. Evaluate every min/max expression at the initial geometry.
2. Select the active constructed site by canonical selector order on ties.
3. Compile only those active branches into the numeric residual.
4. Solve.
5. Re-evaluate the original min/max expressions.
6. If an active site changed, compile the new branch set and solve from the previous result.
7. Accept only when the active set is stable and the original expression is satisfied.

Seen active sets are stored immutably. Repetition without satisfaction returns BranchCycleObstruction. Reaching the sealed active-set budget returns BranchSearchBudgetObstruction. Neither is mislabeled as geometric inconsistency.

## 9. Authoritative products and typed obstructions

### 9.1 Body result

golem/kernel/body/types.py gains:

- SolvedLocalPose: bone id, local translation, normalized rest direction, and twist.
- RelationalSolveReceipt: per-relation measurement receipts, chosen conventions, deepest translation/parameter frontier, Jacobian rank, active-set history, and maximum normalized residual.
- RejectedBody: a non-empty tuple of BodyCompileObstruction.
- BodyCompileResult = CompiledBody | RejectedBody.

SolvedLocalPose is an internal solved carrier, not a second authoring IR. kinematics.py consumes it to build the existing bones table. Landmarks, emitted parts, anatomy, senses, and meshes remain derived views.

The body relation obstruction union includes at least:

- MalformedRelationObstruction(address, reason)
- DuplicateRelationIdObstruction(address, relation_id)
- UnknownRelationKindObstruction(address, authored_kind, closest_valid_candidates)
- UnresolvedRelationSelectorObstruction(address, selector, closest_valid_candidates)
- UnsupportedRelationSelectorObstruction(address, selector, supported_scope_kinds)
- UnreferenceableDerivedGeometryObstruction(address, selector, reason)
- FixedPlacementConflictObstruction(relation_id, subject_address, reference_address, required, observed)
- GoalRelationAuthorityObstruction(relation_id, goal_id, bone_addresses)
- UnderconstrainedRelationObstruction(owner_address, variable_count, rank, free_dimension_count, parameter_addresses)
- RelationalSolveExhaustedObstruction(owner_address, stage, depth, relation_id, required, observed, maximum_normalized_residual, attempted_evaluations)
- RelationalSolverBudgetObstruction(owner_address, required_evaluations, observed_evaluations, best_residual)
- BranchCycleObstruction(relation_id, active_sets, required, observed)
- BranchSearchBudgetObstruction(relation_id, required_active_sets, observed_active_sets, best_residual)

Every obstruction has at least one authoring address and a quantitative required/observed pair where a geometric obligation exists.

### 9.2 Boundary propagation

- Public compiler API: Compiler.compile and compile_file both return BodyCompileResult. No partial convenience function unwraps RejectedBody, raises it, or preserves the old success-only signature.
- Direct check: cli/check.py matches RejectedBody and renders its typed obstructions. It does not catch a relation failure as Exception.
- Direct body CLI: body/cli.py matches RejectedBody before graph access, projects every obstruction, emits no output artifacts, and exits nonzero.
- Assembly ingestion: ingest.py wraps body failures as ElementBodyObstruction(element_id, obstructions), preserving body addresses and measurements.
- Assembly compilation: carriers.py includes ElementBodyObstruction in AssemblyObstruction. assembly/core.py's existing descent accumulates it.
- CLI rendering: assembly/__main__.py's existing dataclass projection renders the nested typed fields. No new string-only failure surface is added.
- Session: session/state.py changes CompileObstruction from one exception-shaped record into a closed union of BodyCompileObstruction and UnexpectedCompileException. compile_authored matches RejectedBody directly; it catches only the pre-existing exceptional failure classes at the effect boundary. CompileObstructed retains the original body constructor and fields, while its human error view derives text through body/project.py.
- Silhouette loading: senses/silhouette.py changes its body JSON load path to a GeometryLoadResult. RejectedBody maps to BodyGeometryLoadObstruction carrying the original tuple; main renders the pure projection and returns nonzero. It is not converted into ValueError.
- Evaluations: eval_anatomy.py and eval_plate_authoring.py match RejectedBody before reading graph or anatomy. Their externally fixed EvaluationObstruction remains terminal, but its reason is canonical JSON from body/project.py, not an exception string; no body evidence is discarded.

Existing informative body violations remain informative. A relational decode or solve failure is blocking because there is no authoritative pose to emit.

### 9.3 Receipts

For relational documents only, body receipt solved gains a relations entry containing:

- authored and canonical relation ids;
- resolved selector addresses;
- selected conventions;
- deepest frontier used;
- required, observed, and normalized residual measurements;
- rank and variable counts;
- stable min/max active sites.

The full solved frame table is not duplicated into the receipt. The existing landmark table is the derived coordinate view, and the solved placement carrier is the compiler authority.

Legacy documents emit no empty relations key.

## 10. Compile and check integration

The body path becomes:

1. validate the body and accumulate ordinary informative violations;
2. decode relation declarations and refine existing Scope selectors;
3. build provisional legacy FK solely as the initial geometry;
4. solve and glue the relational placement;
5. run authoritative FK from SolvedLocalPose;
6. perform the existing ground lift;
7. place props;
8. run legacy two-bone goals on disjoint chains;
9. derive anatomy;
10. emit constructed parts;
11. build intent and receipt.

Validation rejects any relation whose movable closure overlaps a pose.goals chain. This keeps step 8 from invalidating a solved relation. A later design may lower reach goals into the same relation algebra, but this plan does not install a compatibility shim or allow two owners.

check already calls Compiler.compile before senses and calls realize_vasculature without meshing. The relation solve therefore belongs in the check tier automatically. compile reaches the same body path through assembly/ingest.py before assembly/compile.py meshes the solved graph.

Performance requirements:

- A legacy no-relation body has no placement-solver work and preserves current latency.
- A representative 100-bone, 200-relation body must complete the relation solve within 500 ms on the sealed CI host.
- The complete check command for that mesh-free fixture must remain below 1.0 s after warm import.
- Each local solve and each whole-body descent carries a deterministic evaluation-count budget in addition to the wall-clock gate.
- No broad global solve is permitted as a fallback.

## 11. Staged vascular CCO relaxation

### 11.1 Root cause in the live source

The current realization has three separate search losses:

1. corridor.py constructs one macro corridor. Synthetic points use one fixed pump detour and one fixed 2 percent endpoint waypoint;
2. cco._insert_terminal_cco searches four nearest split edges, then at most eight more;
3. cco._insert_feasible_terminal_sequence selects the first feasible next terminal and never backtracks when that locally valid choice causes a later dead end.

That is search failure, not proof of infeasibility. The enriched obstruction payload already improves evidence, but a typed payload cannot redeem the wrong verdict.

There is also an in-flight integration seam: graph.py already requires the enriched InfeasibleBifurcationObstruction fields while cco.py still contains an old two-argument construction. The obstruction-enrichment lane must finish and freeze its API before this lane starts; no one edits those files concurrently.

### 11.2 Candidate result algebra

bifurcation._evaluate_bifurcation_candidate stops returning None. It returns:

- FeasibleBifurcation(cost, tree), or
- RejectedBifurcationCandidate(obstruction).

The rejection uses the existing phase, lane, predicate, required, observed, terminal index, capsule, split-edge, candidate-point, and attempt-count evidence.

cco._InsertionBatch is strengthened and reused as the immutable accumulation of feasible candidates, rejected candidate obstructions, checked ancestors, and escaped ancestor margins. No second candidate batch type is introduced.

Wider search is a total recursive descent over immutable candidate tuples and returns a closed SearchResult ADT. There is no mutable global queue, hidden incumbent, or exception-based backtrack channel; pruning evidence travels in the returned section and is therefore reproducible in the final obstruction or receipt.

### 11.3 Relaxation stages

Per-circuit realization tries the following ordered stages:

#### Stage 0: fixed corridor

- Preserve the current skeleton-derived medial corridor.
- Preserve fixed pump/interface and skeletal landmark points.
- Try the current nearest edge ordering.
- If a complete supply/return pair succeeds, stop. Existing successful results remain unchanged.

#### Stage 1: synthetic waypoint adjustment

- Keep every constructed skeletal landmark and pump interface fixed.
- Unlock only solver-created corridor points: the pump detour and off-bone transition waypoints.
- Generate a deterministic finite local stencil in the transported tangent/normal frame using bounds from SealedVascularConfig.
- Reject candidate corridors that cannot derive a transverse frame, leave containment, violate lane clearance, or invert waypoint order.
- For every admissible adjusted corridor, rerun paired-corridor construction, terminal siting, and local tree construction.

This stage changes the route, not anatomy ownership, terminal demand, vessel radii, or fixed interfaces.

#### Stage 2: wider search

- Enumerate all eligible insertion edges rather than truncating at twelve.
- Search terminal insertion order with deterministic cost-ordered depth-first backtracking.
- Backtrack over the reserved return seed and the supply seed rather than committing to the first locally feasible continuation.
- Try combined admissible waypoint adjustments after single-waypoint candidates.
- Use lower-bound branch cost and previously proved ancestor escape sets to prune only branches that cannot improve or cannot regain containment.

The stage ends in one of three results:

- Accepted complete supply/return trees.
- InfeasibleBifurcationObstruction only when the candidate queue is empty and every configured corridor, seed, edge, and insertion continuation has been examined.
- VascularSearchBudgetObstruction when the sealed state budget is reached while candidates remain.

VascularSearchBudgetObstruction carries region, phase, lane, required state budget, observed evaluated states, remaining queue size, and the best limiting predicate measurement. It is a distinct VasculatureObstruction and a rejected compile, but it does not claim geometry is infeasible.

### 11.4 Final infeasibility evidence

When exhaustive search fails, choose the reported limiting geometric obstruction by:

1. maximum normalized deficit from required to observed;
2. earliest growth phase;
3. canonical lane order;
4. terminal index;
5. split-edge index.

The final InfeasibleBifurcationObstruction retains that predicate's required and observed values and carries the total attempted candidate count. A separate SEARCH_EXHAUSTED summary may be projected in the receipt, but it must not replace the useful limiting measurement.

Accepted VascularFlowReceipt gains search_effort_by_region entries containing the deepest stage used and evaluated state count. This makes successful relaxation visible without exposing rejected intermediate branches as authoring state.

### 11.5 Check-tier bound

Staged vascular relaxation is failure-driven:

- Stage 0 success has no widened-search cost.
- Stage 1 runs only after Stage 0 failure.
- Stage 2 runs only after both earlier stages fail.
- Sealed state and corridor-candidate budgets are part of SealedVascularConfig and rendered in the feasible-envelope contract.
- A budget obstruction is preferred to crossing the sub-second check budget and lying about infeasibility.

The existing content-addressed cache in anatomy/realize/allocation.py remains the effect boundary for repeated identical realizations.

## 12. Mirror interaction

The current mirror machinery remains authoritative:

- mirror: true solves one canonical authored subtree and the engine reflects emitted geometry across world x=0.
- Relations inside that canonical subtree solve once and are inherited by the reflected instance.
- An implicit reflected instance has no selector identity and cannot be a relation endpoint.
- mirror_of relates two explicitly authored structures. It must not be combined with mirror: true on either related subtree; that double ownership is rejected.
- The reflection of a frame is projected back to a proper right-handed SO(3) frame before comparison.
- The existing asymmetric explicit-pose-on-mirror violation remains.
- Vascular bilateral realization continues to solve the representative side and reflect exact circuit geometry after the staged search.

## 13. Contract changes

### golem/contract/schema.py

- Extend bone.attach with disjoint fixed and named-site forms.
- Add typed pose.relations items, canonical solve policies, finite distance, and non-empty ids/selectors.
- Keep body/0.3 as the dialect.
- Continue to source enum values from live RelationKind and RelationSolvePolicy definitions.

### golem/contract/relations.py

Add a registered RELATIONAL PLACEMENT section containing:

- the canonical relation table and aliases;
- selector capability rules;
- fixed versus relational attachment examples;
- solve-policy semantics;
- hierarchy and iterative-deepening behavior;
- ambiguity and min/max active-set rules;
- the prohibition on derived boolean references;
- the goal-chain non-overlap rule.

### golem/contract/vocabulary.py

Render RelationKind, RelationSolvePolicy, canonical aliases, and convention ids from live types. Aliases are explicitly labeled surface sugar.

### golem/contract/primer.py

Replace the current description of attach as only t/offset with both forms. State that fixed coordinates are immutable and relations are solved before emission.

Primacy ruling (orchestrator, 2026-07-21): the contract teaches relations FIRST. Named-site attachment and pose relations are presented as the canonical authoring surface; fixed t/offset is presented afterward as the escape hatch for the rare case where the author genuinely means a coordinate. Mechanical additivity (legacy byte identity) is unchanged — the inversion is pedagogical order and framing in primer, schema ordering, and the relations section, not runtime behavior.

### golem/contract/receipts.py

Document the relational solved block, typed body rejection, required/observed measurements, rank diagnostics, and vascular search-budget distinction.

### golem/contract/envelope.py

Replace the claim that twelve split edges define bifurcation feasibility. Project the live fixed/waypoint/wider-search stages, sealed search budgets, and the distinction between exhausted infeasibility and budget obstruction.

### golem/contract/exemplar.py

Keep the live knight fixed-attachment reading honest until a canonical spec actually uses relations. Do not fabricate a relational excerpt from a legacy file. The new relations section owns the relational example.

### golem/contract/registry.py

Register the new relations section directly. No barrel or re-export facade is added.

## 14. Test strategy

All new tests use synthetic body/anatomy fixtures in tests. No pilot, conformance, golden, or generated artifact is edited.

### Relation decoding and vocabulary

- Both attach forms decode; mixed forms reject.
- Canonical ids and every explicit alias resolve to the same RelationKind.
- Misspelled canonical and alias forms suggest deduplicated canonical ids.
- Unknown selectors retain the source address and nearest constructed ids.
- Unsupported assembly/post-boolean scopes return typed obstructions.

### Geometry and relation laws

- Existing no-relation body graph and receipt remain byte-identical.
- Two named-site attachments sharing one selector produce coincident heads.
- coincident, above, below, aligned, perpendicular_to, and mirror_of satisfy their typed measurements.
- Fixed attachment conflicts report required and observed values without moving the fixed bone.
- A relation requiring one translation-deepening level succeeds without unlocking child parameters.
- A relation requiring parameter deepening rechecks child constraints before gluing.
- Exhaustion reports the deepest attempted frontier.
- Jacobian rank deficiency reports under-constraint.
- Nearest-satisfaction selects opposite conventions from opposite initial geometries and remains deterministic on ties.
- A min/max active-site change triggers re-solve and satisfies the original expression.
- Active-set cycling and budget exhaustion have distinct obstructions.
- A relation overlapping pose.goals is rejected before either solver commits.

### Mirror laws

- Relations on a mirror:true canonical subtree emit exact reflected geometry.
- An implicit mirror instance cannot be selected.
- Explicit mirror_of plus mirror:true rejects double ownership.
- Mirrored frames remain orthonormal and right-handed.

### Vascular staged relaxation

- A fixture where the first feasible terminal choice dead-ends succeeds through backtracking.
- A fixed-corridor failure succeeds after one synthetic waypoint adjustment.
- A twelve-edge failure succeeds when all eligible edges are considered.
- Exhaustive failure returns enriched InfeasibleBifurcationObstruction with phase, lane, predicate, required, observed, and attempted count.
- State-budget exhaustion returns VascularSearchBudgetObstruction and never InfeasibleBifurcationObstruction.
- Repeated solves produce identical topology, stage, and rounded receipts.
- Final clearance, containment, flow, and bilateral symmetry validation still run after relaxed success.

### Boundary and contract

- check renders typed body relation obstructions and never an exception traceback.
- body CLI writes neither graph nor receipt after rejection and returns nonzero.
- compile preserves the same nested body evidence through ElementBodyObstruction.
- session CompileObstructed retains the original BodyCompileObstruction constructor and quantitative fields; its text is derived.
- silhouette body loading returns a typed geometry-load obstruction and never accesses graph on RejectedBody.
- anatomy and plate evaluations match RejectedBody and include the canonical body obstruction projection in their terminal evaluation obstruction.
- Every production call to Compiler.compile and compile_file is exhaustively matched; a static search has no success-only graph access.
- Contract schema validates both old and new forms.
- The relations contract section and alias table are rendered from live types.
- The feasible envelope prints live staged-search constants.

### Performance and validation order

Narrow gates first:

    uv run --frozen pytest tests/test_body_relations_decode.py
    uv run --frozen pytest tests/test_body_relation_solve.py
    uv run --frozen pytest tests/test_vascular_relaxation.py
    uv run --frozen pytest tests/test_contract_relations.py tests/test_cli_contract.py

Then:

    uv run --frozen pytest tests -m "not slow"
    uv run --frozen pytest tests
    uv run --frozen pytest

The last command is the frozen conformance canary. It is validation only; conformance files remain untouched.

A pinned-runner latency test records warm check latency for the representative relational fixture and fails at 1.0 s. Deterministic evaluation-count assertions accompany it so numeric work growth is caught even where wall-clock tests are noisy.

## 15. Dispatch lanes and file ownership

No two active lanes own the same file. Dependency checkpoints are explicit.

### Gate 0: finish the existing obstruction-enrichment lane

Owner: existing lane only.

Files:

- golem/kernel/anatomy/graph.py
- golem/kernel/anatomy/project.py
- tests/test_anatomy.py
- tests/test_assembly.py
- tests/test_cli_contract.py
- tests/test_conduits_vascular.py
- tests/test_elemental_render.py
- tests/test_service.py

Gate:

- enriched InfeasibleBifurcationObstruction constructors and projections agree;
- phase, lane, predicate, required, and observed are frozen for consumers.

BODY-RELATIONS may run concurrently because its file set is disjoint. BODY-BOUNDARIES and VASCULAR-RELAXATION wait for this gate. On closure, graph.py and project.py transfer to VASCULAR-RELAXATION; tests/test_anatomy.py and tests/test_cli_contract.py transfer to BODY-BOUNDARIES. The other four test files are released but remain outside every later lane. Thus the listed files never have concurrent owners.

### Lane BODY-RELATIONS

Files:

- new golem/kernel/body/relations.py
- new golem/kernel/body/relation_solve.py
- new golem/kernel/body/geometry.py
- new golem/kernel/body/project.py
- golem/kernel/body/compile.py
- golem/kernel/body/kinematics.py
- golem/kernel/body/emit.py
- golem/kernel/body/types.py
- golem/kernel/body/intent.py
- golem/kernel/body/validate.py
- golem/kernel/body/__init__.py
- new tests/test_body_relations_decode.py
- new tests/test_body_relation_solve.py
- new tests/test_body_geometry.py
- new tests/test_body_relation_projection.py

It may import golem/addressing/scope.py and scope_grammar.py but does not edit them.

Gate:

- relation laws, hierarchy, active-set iteration, typed body result, and legacy byte identity pass.

### Lane BODY-BOUNDARIES

Starts after BODY-RELATIONS freezes BodyCompileResult and Gate 0 releases tests/test_anatomy.py and tests/test_cli_contract.py.

Files:

- golem/cli/check.py
- golem/kernel/body/cli.py
- golem/assembly/ingest.py
- golem/assembly/carriers.py
- golem/assembly/obstructions.py
- golem/session/state.py
- golem/session/effect.py
- golem/senses/silhouette.py
- golem/evals/eval_anatomy.py
- golem/evals/eval_plate_authoring.py
- tests/test_cli_contract.py
- tests/test_silhouette.py
- tests/test_session_receipt.py
- tests/test_anatomy.py
- tests/test_plates.py
- new tests/test_body_relation_boundaries.py

It does not edit golem/assembly/compile.py, core.py, or __main__.py unless a failing typed-propagation test proves the generic existing descent/rendering insufficient. That exception requires a serial ownership transfer, not concurrent editing.

Gate:

- every production body compile caller exhaustively handles BodyCompileResult;
- direct check, body CLI, assembly compile, session, silhouette loading, and evaluations preserve or canonically project typed relation evidence as specified;
- no success-only compatibility wrapper or exception-based unwrapping remains.

### Lane VASCULAR-RELAXATION

Starts after Gate 0.

Files:

- golem/kernel/anatomy/realize/cco.py
- golem/kernel/anatomy/realize/bifurcation.py
- golem/kernel/anatomy/realize/corridor.py
- golem/kernel/anatomy/realize/parity.py
- golem/kernel/anatomy/realize/carriers.py
- golem/kernel/anatomy/vocabulary.py
- golem/kernel/anatomy/graph.py and project.py only after the Gate 0 owner releases them
- new tests/test_vascular_relaxation.py

The lane must preserve the enriched obstruction fields rather than replacing them with a new traceback or string.

Gate:

- success through each relaxation stage, honest budget obstruction, exhaustive infeasibility, and final geometry validation pass.

### Lane CONTRACT

Starts after RelationKind, RelationSolvePolicy, and SealedVascularConfig fields are frozen.

Files:

- new golem/contract/relations.py
- golem/contract/schema.py
- golem/contract/vocabulary.py
- golem/contract/primer.py
- golem/contract/receipts.py
- golem/contract/envelope.py
- golem/contract/exemplar.py
- golem/contract/registry.py
- golem/contract/__init__.py only if the section must be exposed by the existing public surface
- new tests/test_contract_relations.py

Gate:

- live types, schema, prose, and receipts agree; no fabricated exemplar.

### Integration gate

The orchestrator alone runs the broad living suite, conformance canary, latency receipt, and changed-path audit. Forbidden paths remain untouched:

- pilots/
- conformance/
- pytest.ini
- pyproject.toml
- uv.lock

## 16. Risk register

| Risk | Consequence | Mitigation / acceptance rule |
|---|---|---|
| Under-constrained relations | Solver returns an arbitrary-looking pose that merely stayed near the seed | Rank the unregularized Jacobian and reject pose-changing null spaces with parameter addresses. |
| Over-constrained or nonlinear nonconvergent systems | A generic infeasible label overstates what was proved | Prove fixed contradictions separately; otherwise report solve exhaustion or budget obstruction with worst required/observed measurements. |
| SciPy and BLAS nondeterminism | Last-bit drift or equal-minimum branch changes break golden receipts | Canonical variable order, no randomness, single method, sealed tolerances, convention tie-breaks, rounded/banded numeric assertions, and exact topology/identifier goldens. |
| Min/max branch churn | A solution satisfies a pruned branch but not the original relation | Re-evaluate original expressions, iterate finite active sets, detect cycles, and reject budget exhaustion distinctly. |
| Mirror double ownership | mirror:true and mirror_of produce duplicate or contradictory symmetry | Solve canonical mirror subtrees once; reject implicit-mirror references and mirror_of on mirror:true participants. |
| Legacy goal overlap | goals.py overwrites a relation-solved frame | Reject overlapping movable closures in this cut; do not silently order two authorities. |
| Selector ambiguity | A relation resolves a similarly named but wrong site | Exact Scope parsing, exact constructed-id lookup, typed failure, canonical nearest suggestions only. |
| Boolean/topology references | A post-solve face or union component changes identity across edits | Permit only pre-boolean Bone, Part, and skeletal Landmark constructions; reject every derived result scope. |
| Hierarchical local minimum | A local solve blocks a globally satisfiable arrangement | Translation then parameter deepening, affected-constraint reopening, nearest-seed objective only as tie-break, and explicit exhaustion evidence. |
| Global fallback temptation | Performance collapses and local distinctions disappear | No global solve. A budget obstruction is the only permitted terminal result when local search cannot finish in tier. |
| Vascular branch explosion | Full terminal-order search violates check latency | Cost-ordered backtracking, proved ancestor pruning, sealed state budget, failure-driven stages, cached identical realizations, and a distinct search-budget obstruction. |
| In-flight obstruction API drift | Vascular work races the current payload lane | Gate 0 freezes graph.py/project.py first; file ownership transfers serially. |
| Public compile-result widening | A caller reads graph or anatomy from RejectedBody, or flattens typed evidence into a generic exception | Treat the production caller search as a gate, assign every caller to BODY-BOUNDARIES, and forbid success-only unwrappers. Preserve the ADT through internal boundaries; use the one pure projector only at terminal text/JSON protocols. |
| Legacy receipt churn | Empty solver blocks break byte goldens | Strict no-op path and omission of empty relation receipt data. |

## 17. Acceptance

The transliteration is complete only when all of the following hold:

1. A body/0.3 author can replace absolute attachment arithmetic with shared named-site references and explicit canonical relations.
2. Legacy t/offset attachments remain fixed, valid, and byte-identical when no relations are present.
3. The solver constructs equations only after the tree and constructed sites are finalized.
4. Post-order local solves, translation deepening, parameter deepening, overlap rechecks, and final gluing are observable in typed receipts.
5. Ambiguous conventions are selected by nearest initial satisfaction and recorded.
6. Min/max branch pruning is rechecked against the original expressions.
7. Relation failures carry addresses and quantitative required/observed evidence.
8. check remains mesh-free and sub-second for the pinned representative fixture.
9. Vascular CCO backtracks and relaxes synthetic waypoints before declaring infeasibility.
10. A vascular search budget cutoff is never rendered as InfeasibleBifurcationObstruction.
11. Mirror behavior remains exact and single-owned.
12. The contract schema, vocabulary, primer, relations section, receipts, and feasible envelope all describe the live implementation.
13. Compiler.compile and compile_file have no production caller that assumes success without exhaustively matching BodyCompileResult.
14. No forbidden or generated artifact is edited.

## 18. Addendum: the `between` relation (ShapeAssembly squeeze transliteration)

Authored by the orchestrator, 2026-07-21. Deferred until the BODY-RELATIONS lane freezes; executed by the orchestrator personally. This closes the one vocabulary gap identified against ShapeAssembly: their `squeeze` operator, which places a part in simultaneous contact with two others, has no expression in the section 4 vocabulary — every current kind takes exactly one reference.

### Authoring surface

```json
{"id": "collar", "kind": "between", "subject": "skeleton/neck",
 "references": ["landmark:skeleton/torso/head", "landmark:skeleton/skull/tail"],
 "solve": "negotiate"}
```

`between` is the first two-reference kind. `RelationDeclaration` widens with an optional `reference_b` / `reference_b_selector` pair (default `None`); single-reference kinds leave it `None` and their decode paths are untouched. A `between` declaration missing a second reference, or any other kind carrying one, is a `MalformedRelationObstruction` at decode ("between requires exactly two references" / "kind X admits exactly one reference"). Aliases absorbed into the canonical kind: `spanning`, `bridges`, `squeezed_between`.

### Residual semantics

The subject bone's two endpoints bind to the two reference sites:

- component 1: `|head(subject) − site(reference_a)| / position_tolerance`
- component 2: `|tail(subject) − site(reference_b)| / position_tolerance`

Both components enter the same least-squares block; no hinge, no angular term. Correction at implementation (orchestrator, 2026-07-21): section 5's cut excludes length variables, and that ruling wins. In this cut `between` unlocks translation and swing on its yielding owners; when the references' separation differs from the subject's authored length, no variable can close both endpoint gaps and the honest outcome is RelationalSolveExhausted carrying the residual gap as evidence. A length variable under the PARAMETER stage is the natural extension in the cut that introduces parameter variables. No new solver machinery is required — `between` is two coincident components sharing one subject. Solve policy `reference` is Malformed at decode: the two references are peers with no single yielding side. Subject scope is Bone only; both references are Landmark sites, authored under a plural `references` key.

### Convention

Endpoint assignment is genuinely ambiguous: head-at-A/tail-at-B versus the reverse. Two new `ConventionId` members, `BETWEEN_FORWARD` and `BETWEEN_REVERSED`, selected by nearest initial satisfaction under the section 8 rule (frozen before the first solve, tie broken by canonical order, recorded in the receipt). No other convention interacts with `between`.

### Measure semantics

`measure_relation` reports `required = 0.0` (both endpoint gaps closed) and `observed = max(gap_head, gap_tail)` in world units, with the chosen convention applied before measurement.

### Boundaries

Mirror: a mirrored `between` reflects both references through the sagittal plane and swaps the convention (`FORWARD` ↔ `REVERSED`), consistent with section 12's single-owner rule. Vascular, contract, and boundary-propagation behavior are unchanged; the contract's relations section gains one vocabulary row and one worked example after the implementation lands, in the same change.
