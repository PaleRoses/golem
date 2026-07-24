# Surface-Formation Adoption Docket — trial-3-era campaign

Mission: assess and implement the three ranked adoptions from
`docs/plans/surface-formation-literature.md` §8 (read that document FIRST —
it names the papers, the per-entry adaptation constraints, and the seams:
`golem/kernel/body/myology.py` declaration descent, `golem/kernel/engine/algebra.py`
polynomial smin, the composition seam in `golem/kernel/engine/compile.py`).
The literature report is the spec; this docket is the law.

Work the phases IN ORDER. Each phase: a short written assessment first
(append to `docs/plans/surface-formation-adoption-journal.md` — design
decisions, seam findings, deviations from the literature report and why),
then implementation, then verification, then the next phase.

## Standing rulings

- **R-A. Absence is byte-exact.** Every new law and every new vocabulary item
  is opt-in. A spec that declares nothing new compiles byte-identically:
  pinned goldens, graph SHA pins, and PNG hash pins stay untouched and green.
  This is the house pattern (wave-3: webs/roles/classes all land this way).
- **R-B. One owner per surface.** No parallel authoring surface, no second
  geometry authority. Extensions live in the existing contract organs: the
  engine's closed operator vocabulary, myology's declaration descent, the
  graph contract. Derived views stay derived.
- **R-C. Typed obstructions, never silent fallback.** Use the taxonomy the
  literature report names (`FieldScaleUncalibrated`, `NoCompatibleOverlap`,
  `BlendSupportEscaped`, `GradientDegeneracy`, `TopologyChanged`,
  `IntegralRadiusUnsatisfied`, `CanalEnvelopeObstruction`,
  `MuscleFormationObstruction`, `SkinRelaxationObstruction`, ...). Widening a
  radius, scaling a neighbor, or accepting a non-converged iterate as final
  is forbidden — reject with the witness instead.
- **R-D. Refusal list is law.** No FEM runtime, no neural/learned components,
  no interactive-editor metaphors, no scan-derived templates (report §8
  "what not to adopt first"). Borrow layer topology and cache discipline only.
- **R-E. A concurrent authoring trial is running in this tree.** Another
  agent is authoring a creature under `specs/` and `rehearsal/wyvern-trial/`.
  Do NOT touch `specs/` or `rehearsal/`. R-A is what keeps its trial sound.
- **R-F. Test discipline.** Tests assert semantic structure (laws, residuals,
  obstruction types, parity outside active regions), not regression numbers.
  Never weaken an existing assertion. No single test over 10 seconds.
- **R-G. No git operations.** Leave the tree dirty; the operator owns commits.
- **R-H. Environment.** Invoke everything via `.venv/bin/python` (never
  `uv run` — it panics under the sandbox). Full-union verification command:
  `.venv/bin/python -m pytest tests/ -q --ignore=tests/conformance`
  (baseline this morning: 969 passed, 0 failed).

## Phase 1 — Certified local composition family at the union seam

Replace the LAW at the engine's union composition, not the graph or meshing.
Adopt: gradient-aware binary operators (Gourmel et al. TOG 2013),
overlap-localized clean-union/blend behavior (Bernhardt et al. CGF 2010),
bounded-support witnesses (Pasko SMI 2002 / Fryazinov C&G 2011). Opt-in via
the graph contract's operator vocabulary (alongside blend/chamfer/crease);
undeclared specs take the legacy smin path byte-identically (R-A).
Local field normalization with certified gradient bounds or
`FieldScaleUncalibrated`; no blending without declared-iso overlap
(`NoCompatibleOverlap`); active-region support witness carried as derived
evidence and verified (`BlendSupportEscaped`); parity proof outside active
regions on identical fields. Acceptance: full union green + a small benchmark
comparing new-law vs legacy smin on identical field pairs (correctness parity
outside support, bulge suppression inside), recorded in the journal.

## Phase 2 — Radius-faithful skeleton-integral formation for muscle groups

A small CLOSED formation family: finite centerline segments, bounded junction
polygons, branch-free variable-radius canal intervals — analytic evaluation
only, every other skeleton shape rejected at decode with a typed obstruction
naming the legal family. SCALIS-style scale normalization and prescribed-radius
law (Zanni et al. CGF 2013); convolution closed forms (Bloomenthal '91,
McCormack & Sherstyuk '98); bulge-free preflight ratio check as a validator
(Bloomenthal '97). The muscle stays authored intent in `myology.py` (origin,
insertion, sections, bulk policy); the solver derives field, volume, local
frame, and contact envelope, consumed through the existing pure sampling
boundary — a separate integral-field representation, never smuggled into
smin. Solved receipt carries prescribed-radius residual, volume residual,
curvature/envelope bounds, branch compatibility. Failures:
`MuscleFormationObstruction` / `IntegralRadiusUnsatisfied` /
`CanalEnvelopeObstruction` with interval and bound witnesses. Rest-pose only.
At branches, hand off to Phase 1's junction operators.

## Phase 3 — Skin as its own solved layer

Consumes a solved flesh boundary from Phase 2 (and the legacy union where no
formation is declared). Static rest-pose solve: canonical skeleton-to-surface
correspondence from the formation fields, implicit projection to a declared
skin iso-value/offset (Vaillant et al. SIGGRAPH 2013), bounded tangential
relaxation with a sealed iteration budget and convergence verdict
(iso-surface tracking SIGGRAPH Asia 2014 decomposition:
field update -> normal projection -> tangential relaxation, each with its own
law; Turner/Thalmann and Wilhelms/Van Gelder energies as the static
vocabulary). The skin result is a derived view of flesh plus its own proof —
containment, offset, stretch, foldover, projection multiplicity, convergence
all exposed; non-convergence is `SkinRelaxationObstruction`, never a final
mesh with a log warning. No dynamics, no frame state in pure compilation.

## Verification

After Phase 3: the full union suite green; all byte-exactness pins intact;
the journal records per-phase assessment, deviations, benchmark results, and
every typed obstruction actually exercised by a test.
