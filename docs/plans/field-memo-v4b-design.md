# V4b — Per-section sampled reuse: the edit-loop cure (design, fable, 2026-07-23)

Commission (Rosalia): take a geometry edit at seal resolution from ~25s
to low single-digit seconds. Executors: constrained sol (xhigh) cuts
compile.py; opus writes the test file in parallel; fable reviews, runs
binding gates, and measures. Inherits every ruling of
`docs/plans/field-memo-v4-design.md` (R-V1..R-V3, E5 cold law,
rejection non-caching, atomic stores, mtime-LRU pruning).

## Verdict correction: the certificates are re-aimed

The V4 verdict promised Lipschitz/interval certificates would "narrow
the recompose to cells the edit can provably reach." Reading the fold
refutes the target: composition is ELEMENTWISE (`np.minimum` chains +
masked pairwise corrections, compile.py:950-955), so once per-section
arrays are in hand the full-grid recompose is cheap NumPy — there is
nothing worth narrowing. The certificates' true application is
tile-pruned SYMBOLIC evaluation (the libfive/Lipschitz-pruning shape),
which attacks the miss/first-build cost — that is Prong D, a separate
commission if wanted. This design is Prong C alone and delivers the
edit loop without certificates.

## Measured anatomy (receipts, fable's read)

`_sample_graph_field_certified_fold` (compile.py:1399): prepares
symbolic sections (`_prepared_composition_section`, :438 — consumes
ONLY the instance), then reduces `merge` (:1437) over instances. Per
section, `merge` samples `section.evaluate` and (when present)
`section.gradient.evaluate` over the FULL points array — this is the
entire expensive term — then calls `_compose_certified_field` (:798).
Composition consumes: the accumulated field array, prior sections'
SAMPLED arrays (`compose_pair`, :884 — `prior.field`,
`prior.gradient`), prepared symbolic sections (only for
`_compatible_overlaps` probes, :829, and gradient magnitude bounds,
:918 — both resolution-independent), operator, radius, points, domain,
topology_shape. Carves (:1490) sample `part_sdf_checked` full-grid and
`sdf_difference` elementwise. Nothing in composition is a grid-wide
reduction feeding arithmetic; `local_topology_*` receive
`topology_shape=None` on the production pass.

Therefore: the production pass factors EXACTLY into
(1) GATHER — per-section/per-carve full-grid symbolic sampling, and
(2) COMPOSE — deterministic arithmetic over those arrays plus cheap
symbolic probes. Gather is memoizable per section; compose is not worth
memoizing.

## The law

- Memo unit: one positive section's sampled `(field, gradient|None)`
  pair, and one carve instance's sampled field. Keys are
  composition-independent: a section's sampled arrays depend on its OWN
  geometry only, never on operators, blends, order, or other sections.
- Certification pass: NEVER touches the section memo — always fully
  symbolic (all evidence, overlap law, calibration fresh; the low-res
  grid is cheap). Production pass (topology_shape None): gather consults
  the memo per entry; misses evaluate symbolically.
- Cold law (E5): `_field_cache_bypassed()` gates the section memo
  identically — cold mode neither reads nor writes `sections/`.
- Rejection law: if the production fold rejects, NOTHING is stored —
  section arrays computed before the rejection are discarded (collect
  pending writes, flush only on an accepted `_ComposedField`).
- Layering with V4a: composed-snapshot hit returns early as today. On a
  composed miss, cert pass runs symbolic, production gathers through
  the section memo, and BOTH stores are populated on acceptance.
- Bitwise pin (non-negotiable): for any edit, the warm-assembled
  production field is `np.array_equal` to the cold full symbolic
  recompute. If it is not, the cut is wrong; the pin is never loosened.

## Keys and store

`_points_digest(points) -> str` — sha256 over `points.tobytes()` +
shape + dtype; extracted so `_composed_snapshot_key` and section keys
share ONE computation per call.

`_section_sample_key(geometry, mirrored, points_digest) -> str` —
sha256 over pickled payload: the instance's SHAPE-canonical geometry,
`mirrored`, the points digest, and `_field_source_salt()`. Carve
entries use the same function (a carve is a `(Part, mirrored)`
instance whose sample has no gradient).
SHAPE-canonical is adjudicated (fable, 2026-07-23, resolving a
cross-lane ambiguity): the payload names EXACTLY the fields the
evaluator consumes — kind/type and geometric parameters — and EXCLUDES
the part identifier (`part_id` never reaches `evaluate`; keying it
would refuse duplicate collapse and miss on renames for nothing).
ALSO EXCLUDED BY LAW, with the governance argument: `operator`,
per-instance `blend`, `graph.blend`, domain bounds, fold order, and
the certification-resolution constant govern COMPOSITION, not
`section.evaluate`'s output over fixed points. Obligation C-1 verifies
the full exclusion set before it stands; test 7 pins the duplicate
collapse, test 1 pins the composition exclusions.

Store: `sections/` sibling of `composed/` under the field-cache root
(`_section_sample_root() -> Path`), one entry = one `.npz`
(`np.savez`, uncompressed, `allow_pickle=False`; float64 exactly —
lossy anything is banned by the bitwise pin), written `tmp +
os.replace`, pruned by the existing `_prune_field_cache(root, cap)`
with `_SECTION_CACHE_MAX_BYTES` (default 32 GiB, env override
`GOLEM_SECTION_CACHE_MAX_BYTES`). Separate root + cap so section churn
can never evict composed snapshots. Loads: `np.load` per entry; the OS
page cache is the RAM layer — no bespoke in-process LRU until receipts
demand one.

## Pinned API (law for both agents; tests are written against these)

- `_points_digest(points) -> str`
- `_section_sample_root() -> Path`
- `_section_sample_key(geometry, mirrored, points_digest) -> str`
- `_load_section_sample(key) -> _SectionSample | None`
- `_store_section_sample(key, sample) -> None`
- `_SectionSample` — field + gradient|None (a small frozen dataclass)
- `_sample_section_arrays(section, points, chunk_size, executor)
  -> tuple[np.ndarray, np.ndarray | None]` — the EXTRACTED symbolic
  gather for one positive section (the spy seam for tests)
- `_sample_carve_array(part, mirrored, points, executor) -> np.ndarray
  | MuscleFormationObstruction` — likewise for carves

The fold's `merge`/`subtract_carve_field` route ALL symbolic sampling
through these two functions on both passes; the memo consult wraps them
on the production warm path only.

## Obligations (sol discharges with file:line receipts in its report)

- C-1 (key exclusion): verify `_prepared_composition_section` (:438)
  and the evaluators it builds consume NOTHING beyond the instance
  `(geometry, mirrored)` — no graph.blend, no bounds, no operator. If
  ANY excluded input leaks into `evaluate`, it enters the key; report
  the leak either way.
- C-2 (compose purity): enumerate every input to
  `_compose_certified_field` and the carve reduce; confirm each is a
  sampled array, a scalar derived from (graph, bounds), a prepared
  symbolic section, or topology_shape — i.e. the compose phase is a
  pure function of gather outputs + cheap recomputation. Any grid-wide
  reduction feeding arithmetic refutes the design; STOP and report.
- C-3 (chunking determinism): confirm `_sample_sdf` chunked evaluation
  is a pure concatenation of per-chunk elementwise evaluation (bitwise
  vs whole-array); flag any evaluator with cross-point coupling.
- C-4 (store hygiene): atomic writes, prune-after-flush, cold gate
  before ANY store contact, rejection discard, duplicate-key sections
  (identical geometry twice in a graph) collapse to one entry and two
  hits.
- C-5 (behavior identity): the refactor changes WHEN arrays are
  computed, never WHAT is computed — same chunk sizes, same executor
  routing, same rejection short-circuits in fold order (a section
  memo hit must not reorder or skip law checks that today run before a
  later section's rejection fires).

## Tests (opus, new file `tests/test_field_section_memo.py`)

Fixture: patch `_section_sample_root` and `_composed_snapshot_root` and
`_field_cache_root` to tmp; `monkeypatch.delenv` the cold env (the
fourth-confession hygiene law); production grids res ≤ 24 with res
chosen UNEQUAL to `_LOCAL_TOPOLOGY_CERTIFICATION_RESOLUTION` (assert
inequality) so spies can tell passes apart by points count; every test
< 10s.

1. Key law: same geometry+mirrored+points → same key; perturbed
   geometry → new key; mirrored flip → new key; different points →
   new key; salt monkeypatch → new key. Operator and per-part blend
   perturbations change the COMPOSED snapshot key but NOT the section
   key (pins the lawful exclusion).
2. Warm edit loop: build a 3-part graph (memo populates on
   acceptance), edit ONE part, re-compose. Spy on
   `_sample_section_arrays`/`_sample_carve_array`: at production
   points-count, ONLY the edited section is sampled symbolically;
   cert-pass calls (different points count) cover all sections both
   times.
3. THE bitwise pin: the warm-assembled edited field is
   `np.array_equal` to the cold full recompute of the edited graph.
4. Cold law: with the env set, two identical calls sample everything
   symbolically twice and `sections/` is never created.
5. Rejection law: a graph that rejects mid-fold (containment overlap)
   leaves `sections/` empty — including entries for sections evaluated
   BEFORE the rejecting section.
6. Carve edit: editing only a carve re-samples only the carve at
   production points; all positive sections hit.
7. Duplicate geometry: two identical instances yield one store entry;
   warm pass samples symbolically zero times at production points.

## Measurement (fable's hand, post-review)

Extend the probe family with
`rehearsal/skin-siege/measure_field_edit.py`: Sable Stride formed at
res 242 — (a) cold build (populates both stores), (b) EDIT one part
(small center nudge), time the production re-evaluation warm vs the
25.36s miss baseline and the 0.228s hit floor; report section count,
store bytes, and per-phase split (gather-hit IO vs edited-section
symbolic vs compose). Receipt:
`rehearsal/skin-siege/measure-field-edit.json`, numbers into docket V7.

## Fences

- sol (xhigh, workspace-write): `golem/kernel/engine/compile.py` ONLY.
  algebra.py and everything else read-only — if the seam genuinely
  demands another file, STOP and record why in the report instead of
  cutting. NO git subcommands of any kind. NO full test suite; verify
  with `.venv/bin/python -m py_compile` and, at most, self-authored
  inline probes at res ≤ 24. Report (obligations C-1..C-5 with
  file:line receipts, deviations, refutations) to
  `docs/plans/field-memo-v4b-impl-report.md`.
- opus: `tests/test_field_section_memo.py` ONLY; `py_compile` only
  (impl lands in parallel — symbols may not exist yet); NO pytest, NO
  git. Design API names are law; where the impl drifts, the design
  wins and sol conforms.
- Scoped pytest runs, binding gates (union + R10, cold), and
  measurement are fable's hand at review. Neither agent commits.

## Verdict (fable, 2026-07-24)

LANDED and CERTIFIED. Implementation (compile.py cut + role cure) and
tests (`tests/test_field_section_memo.py`, 7 pins + cross-role
amendment) reviewed at source; scoped suites 13/13 warm AND cold;
binding gates union **1088/0 cold** (317.83s) + R10 **5/5 cold**
(21.18s). The campaign's key catch stands recorded in the impl report:
the role-blind key collision (positive sections sample prepared
evaluators WITH gradients, carves sample `part_sdf_checked` without —
a shared key namespace assumed an unproven bitwise equivalence and a
gradient arity, warm/cold divergence on a carve-then-positive hit),
cured with the keyword-only `role` key component.

Measurement (`rehearsal/skin-siege/measure_field_edit.py`, Sable
Stride formed, bounds-stable `pelvic_basin` 0.98 shrink — deviation
from the design's "center nudge", verified bitwise bounds-stable; 8 of
Sable's 30 parts admit such an edit):

| res | miss | edit warm | recompose floor | hit | store |
|-----|------|-----------|-----------------|-----|-------|
| 121 | 4.84s | **0.62s** (7.7x) | 0.587s | 0.028s | 2.52 GB |
| 192 | 22.32s | **2.55s** (8.7x) | 2.28s | 0.104s | 10.08 GB |

Store and floor scale cubically (ratios 4.00 / 3.88 vs the theoretical
3.99): the floor is load-bandwidth-bound, and the edited section's
symbolic resample adds only ~0.27s at 192 (a lone miss inherits the
whole executor). Receipts:
`rehearsal/skin-siege/measure-field-edit-res{121,192}.json`.

**res 242 (seal) is BLOCKED ON DISK, not on the mechanism.** Exact
bill: Sable = 43 positive instances (30 parts + 13 mirrored twins,
zero webs) x 432.5 MB + carve = 18.3 GiB store, ~19.6 GiB peak with
transients — against 18 GiB free on this machine after purging the uv
cache and the legacy field-cache entries (~2.1 GiB reclaimed; the
remaining candidates, hie-bios 8.3 GiB and the bazel repo caches, are
Rosalia's to authorize). Extrapolating the measured cubic floor:
edit_warm at 242 ~= 4.6-5.2s vs the measured 25.36s miss (~5x) — the
commission's "low single-digit seconds" holds at its upper edge. The
`_SECTION_CACHE_MAX_BYTES` default of 32 GiB exceeds this machine's
practical free disk; right-sizing it (or the disk) is an open ruling.

Confessions (probe, fable's own): v1 convicted every bounds-stable
candidate — `hasattr(decoded, "obstructions")` matches the `Accepted`
wrapper too (types.py:1887); cured with `require_accepted` +
`GeometryDecodeFailure`. v1 also leaked its 18 GB tempdir store on the
ENOSPC crash; probes now surrender their stores in `finally`.

The bounds-stability caveat is now measured law, not conjecture: the
authoring grid derives from `graph_bounds_checked`, so a hull-moving
edit shifts every point and lawfully misses every section key. Prong D
(tile-pruned symbolic evaluation via Lipschitz/interval certificates,
attacking the miss itself) remains uncommissioned — her call.
