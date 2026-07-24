# Replace sampled skin crossings with an exact segment certificate

This ExecPlan is a living document maintained under
`/Users/bluerose/Developer/pale-meridian/docs/llm-guidance/PLANS.md`.

## Purpose / Big Picture

The skin acceptance law currently infers sheet identity from seventeen field
samples along each flesh-to-scaffold segment. Two roots between adjacent
samples are invisible, so the law can accept a segment without actually
proving that it meets exactly one sheet. After this change, every segment is
partitioned at grid-cell boundaries, its trilinear field restriction is
recovered as an exact cubic on each partition, and distinct real root
locations are counted directly. The focused skin-formation tests demonstrate
the old blind spot and the exact tangency behavior.

## Progress

- [x] (2026-07-23) Read the trilinear sampler, crossing helpers, production and rehearsal callers, sealed evidence vocabulary, and focused tests.
- [x] (2026-07-23) Derive the vectorized cell partition, inverse Vandermonde, closed-form degree ladder, half-open gluing law, and location deduplication rule.
- [x] (2026-07-23) Replace the sampled implementation and rewire every caller without a compatibility surface.
- [x] (2026-07-23) Add the thin-slab, tangency, and shared-boundary regression pins.
- [x] (2026-07-23) Run `.venv/bin/python -m pytest tests/test_skin_formation.py -q`; 36 tests passed.

## Surprises & Discoveries

- Observation: `root_tolerance` does not determine the location-deduplication
  tolerance because its construction floors the value at `1e-12`.
  Evidence: `golem/kernel/engine/compile.py` computes
  `max(1e-12, policy.minimum_gradient * maximum_pitch)`, so the policy gradient
  cannot be recovered when the floor binds.
- Observation: the private crossing function also has two rehearsal consumers.
  Evidence: `rehearsal/skin-siege/diagnose_multiplicity.py` calls it for both
  scaffold and relaxed segments.
- Observation: applying the policy-derived location tolerance as an
  unconditional global quotient erased genuine close-fold roots.
  Evidence: the Knight fixture has exact analytic root pairs separated by
  `0.00898` and `0.00327` world units while the policy-derived tolerance is
  `0.05960`; preserving distinct analytic roots restored the typed
  multiplicity obstruction.
- Observation: a roundoff-sized cubic coefficient can survive the
  scale-relative degree mask and make raw trigonometric coordinates
  catastrophically inaccurate.
  Evidence: a convex gencyl interval produced a candidate with Horner residual
  `-0.01126` against root tolerance `1.17e-9`; two safeguarded Newton
  compositions followed by value certification delete the phantom while
  retaining the endpoint root.
- Observation: exact local roots still require numerical overlap descent.
  Evidence: an adversarial perfect-square tangency initially split into two
  analytic coordinates `1.91e-7` apart, and one shared-boundary root initially
  appeared at both `0.4999999999999787` and `0.5`.

## Decision Log

- Decision: add `minimum_gradient` as a required argument to the two crossing
  views while deleting `sample_count`.
  Rationale: exact location deduplication requires
  `root_tolerance / minimum_gradient / segment_length`; binding the helper to
  `SEALED_SKIN_POLICY` would falsify custom-policy behavior.
  Date/Author: 2026-07-23, Blue Rose.
- Decision: place all polynomial discovery and gluing in one private
  root-parameter function, with counts and world distances remaining derived
  views.
  Rationale: the two public-private helpers ask the same semantic question;
  duplicating Cardano and deduplication logic would create competing numerical
  authorities.
  Date/Author: 2026-07-23, Blue Rose.
- Decision: treat branch-selected analytic roots as authoritative local
  sections after numerical polish and value certification, and apply the
  policy-derived location tolerance only while reconciling auxiliary
  sample/tangency representatives.
  Rationale: a numerical glue radius may identify duplicate representations
  on overlaps; it may not quotient two distinct roots of one local polynomial
  into a single sheet location.
  Date/Author: 2026-07-23, Blue Rose.
- Decision: canonicalize value-near boundary roots to the half-open node and
  numerically split tangencies to the certified stationary point; use a
  cube-root-epsilon cap when grouping floating cubic representations.
  Rationale: cubic multiple-root conditioning scales as the cube root of
  machine epsilon, while the policy-derived location tolerance remains the
  outer semantic bound. Nearest-boundary selection prevents that outer bound
  from erasing distinct interior analytic roots.
  Date/Author: 2026-07-23, Blue Rose.

## Outcomes & Retrospective

The seventeen-point heuristic and its `sample_count` parameter are gone from
both crossing views. Counts and distances now descend from one vectorized
piecewise-cubic root owner. The thin off-grid slab proves three roots where the
historical uniform samples expose one; the adversarial tangency pin proves a
numerically split double root is one location; the shared-boundary pin proves
half-open ownership under floating reconstruction; and the established Knight
and convex acceptance witnesses remain intact. Every focused skin-formation
test passes. The main numerical lesson is that local polynomial exactness still
requires certified gluing: raw closed forms and a global proximity quotient
are both too coarse to own topology.

## Context and Orientation

`golem/kernel/engine/compile.py` owns `_trilinear_samples`, the authoritative
evaluation of the grid field, and the two crossing views
`_skin_segment_crossings` and `_skin_segment_crossing_distances`. The skin
solver calls those functions before constructing `SkinRootMultiplicity` or
`SkinRootAbsent`. Those dataclasses retain their sealed `sample_count` evidence
field as a policy echo. `tests/test_skin_formation.py` owns the focused
behavioral pins. `rehearsal/skin-siege/diagnose_multiplicity.py` is a direct
diagnostic consumer whose signature must remain live after the old parameter
is deleted.

Each grid cell is a region where trilinear interpolation has one polynomial
law. Restricting that law to a straight segment produces a cubic in the local
segment coordinate. Grid-plane intersections therefore form a cover of each
segment by cubic sections. Half-open section ownership and tolerance-based
location gluing turn the local roots into one authoritative global root set.

## Plan of Work

In `golem/kernel/engine/compile.py`, construct every segment's grid-plane
parameters with broadcast integer plane ranges, pad the ragged axis ranges
with parameter one, sort the boundaries, and mask zero-width intervals.
Evaluate four fixed nodes per active interval with `_trilinear_samples`, apply
the constant inverse Vandermonde with one `einsum`, and select cubic,
quadratic, linear, constant, or riding-sheet behavior by scale-relative
coefficient masks.

Produce real roots with vectorized Cardano, trigonometric cubic, stable
quadratic, and linear formulae. Add value-near interpolation nodes and
value-near stationary points, suppress duplicate shared boundaries through
half-open interval ownership, collapse adjacent riding cells, then sort and
deduplicate global parameters using the policy-derived distance tolerance.
Zero-length segments produce no candidates. Derive counts directly and derive
world distances by multiplying the sorted parameters by segment length.

Delete `sample_count` from both helper signatures and all callers. Preserve
the separate absence-evidence resampling and every sealed evidence field.

In `tests/test_skin_formation.py`, preserve the existing three-vertex
`(3, 3, 3)` assertion under the new signature. Add one field whose narrow
negative slab creates two roots between adjacent points of the historical
seventeen-point grid and one later root, then assert that historical sampling
sees one while the exact certificate sees three. Add a bilinear field whose
diagonal restriction is a perfect square and assert one tangential root.

## Concrete Steps

Work from
`/Users/bluerose/Developer/pale-meridian/potentialimprovements/golem-kernel`.
Edit only the live source, focused test, direct rehearsal caller, and this
plan. Do not run Git commands. Validate only with:

    .venv/bin/python -m pytest tests/test_skin_formation.py -q

The expected result is a zero exit status with every test in that file passing.

The final transcript is:

    36 passed, 79 warnings in 8.08s

## Validation and Acceptance

Acceptance requires the existing multiple-crossing test to retain counts
`(3, 3, 3)`, the thin-slab test to prove exact count three while the historical
uniform samples expose only one crossing, and the tangency test to count one
root location. The shared-boundary test must count the boundary root once. A
source search must show no `sample_count` parameter or
seventeen-point sampling inside the two replaced helpers. The absence gate's
separate evidence sampling and the sealed evidence dataclasses must remain
unchanged.

## Idempotence and Recovery

The calculation is pure NumPy over its inputs and writes no persisted state.
The focused test command can be repeated without repository changes. If a
numerical branch fails, repair the shared exact parameter owner rather than
restoring or aliasing the sampled implementation.

## Artifacts and Notes

The required behavior is pinned in `tests/test_skin_formation.py`. The warnings
in the final run are NumPy 2.5 deprecations emitted by scikit-image marching
cubes, outside the changed root certificate.

## Interfaces and Dependencies

No new dependency is required. `_trilinear_samples` remains the field-value
owner. `_skin_segment_crossings` returns one NumPy integer count per input
segment. `_skin_segment_crossing_distances` retains its tuple-of-tuples
world-distance shape. Both require `minimum_gradient` and `root_tolerance`;
neither accepts `sample_count`.
