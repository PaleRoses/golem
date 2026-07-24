# V4 — Incremental field revalidation (design, fable, 2026-07-23)

Target: the 20.50s flesh-field evaluation at res 242 (velocity docket V7,
entry 015 receipts). Scope is EXACTLY this stage — no downstream stage,
no gate, no diagnostic changes. Rulings R-V1..R-V3 bind everything below.

## Measured anatomy of the cost

`_sample_graph_field_certified_checked` (compile.py:1470) runs the
certified fold TWICE per call: once at
`_LOCAL_TOPOLOGY_CERTIFICATION_RESOLUTION` over the full domain (the
certification pass — produces ALL topology evidence), once at the
requested res-242 points with `topology_shape=None` (the production pass —
its evidence is REPLACED by the certification pass's at the end). The fold
(`_sample_graph_field_certified_fold`, 1344) evaluates every positive
section's field AND gradient symbolically over the full points array
(`sample(section.evaluate, ...)`, chunked through the executor), composes
pairwise with overlap assessment, then subtracts carve parts full-grid.
The V1 per-part disk cache (`_sampled_part_field`, 1220) is NOT on this
path — nothing here is reused across transactions. That is the whole
20.50s: it is paid on untouched anatomy and on every local edit alike.

## The law: content-addressed staged memos ARE the invalidation algebra

No edit-reachability mapping, no carried evidence across moved ground
(R-V2; the revalidation disease). A stage's memo key is a canonical
fingerprint of its complete governing inputs; a hit is a proof of
input-identity, a miss recomputes. The session edit algebra's role is
observability (V7 receipts: per-stage wall clock + hit/miss), never
invalidation.

Inherited law, non-negotiable:
- E3 lesson / R-V2: keys name the FULL dependency set. A memo an edit can
  invalidate without changing its key is forbidden.
- E5 lesson: the cold check (`_field_cache_bypassed()`, same
  `GOLEM_CIRCUIT_CACHE_COLD` env) runs BEFORE any lookup and gates reads
  AND writes. Cold mode touches no store.
- R-V1: warm results never seal. Binding gates run cold; acceptance
  re-derives on the cold path.
- Rejections are NEVER memoized. Only composed positive results store.
- Atomic writes via `tmp + os.replace`, pruning by the existing
  mtime-LRU (`_prune_field_cache`, parameterized by root), matching the
  established store idiom.

## Prong A — composed-field snapshot memo (untouched anatomy → ~0)

Memo boundary: `_sample_graph_field_certified_checked`. On entry compute
the snapshot key; on hit return the stored result; on miss run both folds
as today and store.

Key (sha256 over pickled canonical payload), enumerated exactly:
1. the canonical graph geometry: every positive instance's Part /
   SpanningWeb canonical form WITH its mirrored flag, in fold order;
   every instance's `operator` and per-instance `blend`;
2. the carve instance sequence (part + mirrored, in order);
3. `graph.blend` and the domain bounds from `graph_bounds_checked`
   (diagonal → global blend radius depends on them);
4. `_LOCAL_TOPOLOGY_CERTIFICATION_RESOLUTION`;
5. sha256 of `points.tobytes()` + points.shape + dtype (the exact
   production grid — ~0.25s on 340MB, negligible against 20.5s;
   compute ONCE and reuse for Prong B keys);
6. `_field_source_salt()` (code drift).

Payload: the returned `_ComposedField`'s CONSUMED projection. Obligation
A-1 (agent verifies, reports with receipts): trace both call sites
(compile.py:3756, compile.py:4047) and every downstream consumer — if
`.sections` (per-section sampled arrays) are unread after the wrapper
returns, store `field` + `evidence` and reconstruct with `sections=()`;
if ANY consumer reads sections, store them (no silent structural
weakening). Storage: subdirectory `composed/` under the existing field
cache root with its own byte cap (default 4 GiB — snapshots are ~113MB
at res 242), same pruning law.

Alongside each snapshot, store the canonical graph payload itself (the
key's input, not just its hash) and register the snapshot in a small
per-points-hash index (`composed/<points-hash>.index`, most-recent-first,
atomic rewrite). This index is what Prong B diffs against.

## Prong B — dirty-region splice on local edits (the edit-loop win)

On a Prong A miss, before full recompute, consult the index for the most
recent predecessor snapshot with the SAME points-hash and salt. If one
exists, diff the canonical graphs section-by-section (identity by section
IDENTIFIER, the same identity `_compatible_overlaps` uses; carves and
operators included).

Hardening (upgrade pass, adversarial self-review):
- FOLD ORDER IS LAW. The fold is a `reduce`; floating-point composition
  is order-dependent, so the unchanged sections must appear in an
  IDENTICAL relative order to the predecessor's. Any reordering of
  surviving sections → inadmissible, full recompute. (The bitwise pin
  would convict a violation; this rule prevents it.)
- DOMAIN GUARD. `graph_bounds_checked` bounds and `graph.blend` must be
  identical to the predecessor's — the global blend radius derives from
  the domain diagonal, so a bounds- or blend-changing edit dirties every
  section. Differ → inadmissible, full recompute. (The points-hash
  usually catches bounds drift when the caller's grid derives from
  bounds; this guard is the belt to that suspender, not a substitute.)

Admissibility is a CERTIFICATE, never an assumption:
- The certification pass (low-res, full domain) ALWAYS recomputes fully —
  it is cheap and it carries ALL evidence and every global check
  (overlap compatibility, topology, calibration). No evidence is ever
  carried from the predecessor.
- Obligation B-1 (agent verifies before building; this gates the whole
  prong): the fold's law checks (`_compatible_overlaps`, calibration,
  support-conflict) are resolution-independent — they probe PREPARED
  symbolic sections, not the sampled grid. If any check depends on the
  production grid, the splice is inadmissible as designed; STOP and
  report rather than approximate (R-V3).
- SOUNDNESS LANDMINE (why naive splice is wrong): a raw SDF has GLOBAL
  support — `min` composition lets a moved part change the argmin at
  cells arbitrarily far away. A splice is admissible ONLY if every
  CHANGED section (old and new form) is bounded-support: outside its
  support envelope its field provably cannot influence any compose op in
  use (the localized-morphology envelope logic in
  `_sampled_part_field_uncached` is the reference — radial bound plus
  blend padding). Obligation B-2: state and verify the exact
  non-influence criterion per operator (union/smooth-min and
  LOCAL_BLEND) against the section evaluators actually in use. Sections
  that cannot certify bounded support (the `case _` fallback family,
  webs) make any edit touching them inadmissible.
- Inadmissible for ANY reason → full recompute (today's path), and the
  session receipt says so in type (`splice: full-recompute <reason>`),
  never silently.

Admissible splice: dirty region = union over all changed/added/removed
sections of (old envelope ∪ new envelope), intersected with the domain;
re-run the ENTIRE production fold (all sections, same reduce order, same
chunk sizes, including carve subtraction) restricted to the dirty points;
splice into a copy of the predecessor's field array; store as a new
snapshot under the current key. Cost scales with the dirty fraction — a
local limb edit is 1–5% of the domain.

The determinism pin that makes this honest: for a local edit, the
spliced field must be `np.array_equal` (BITWISE) to the cold full
recompute — same arithmetic on dirty cells, untouched bytes elsewhere.
If bitwise equality cannot be achieved, the splice is wrong; fix the
splice, never loosen the pin to allclose (honest-tests law).

## What this does not touch

Marching cubes, skin formation, anomaly senses, session acceptance
semantics, any diagnostic or obstruction type, the sealed
`SEALED_SKIN_POLICY`, the R10 gates. The memo returns byte-identical
fields; downstream is oblivious by construction.

## Tests (new file `tests/test_field_memo.py`)

1. Key completeness (R-V2 pins): perturb EACH governing input in
   isolation — one part's parameter, a mirrored flag, an operator, a
   per-part blend, `graph.blend`, a carve, the grid axes, the
   certification resolution constant (monkeypatched), the source salt —
   and assert the snapshot key changes; identical inputs → identical key.
2. Warm equality: second call with identical inputs returns a result
   whose `field` is `np.array_equal` to the first and whose evidence is
   equal; and the expensive evaluators are NOT invoked (count calls via
   monkeypatch).
3. Cold law: with `GOLEM_CIRCUIT_CACHE_COLD=1`, two calls both invoke
   the evaluators, and the store directory is neither read nor written.
4. Rejection non-caching: a graph that rejects composes no snapshot.
5. Splice bitwise pin: build a small multi-part graph, snapshot, apply a
   local edit to one bounded-support part, and assert the spliced field
   is `np.array_equal` to a cold full recompute of the edited graph;
   assert the receipt records a splice (not full recompute).
6. Splice inadmissibility: an edit touching a non-bounded-support
   section falls back to full recompute WITH the typed reason.
7. Global-support adversary: construct the argmin-at-a-distance case —
   an edit whose influence provably escapes its naive envelope under
   `min` composition — and assert the engine either certifies the true
   (larger) envelope or refuses the splice; the bitwise pin must hold
   either way. This is the pin that convicts naive splicing.

Scoped run: `tests/test_field_memo.py` plus
`tests/test_surface_formation.py` only. Never the full suite; never git.

## Measurement (fable's hand, post-review)

Re-run the entry-015 harness (rehearsal/skin-siege measurement path,
res-242 trial anatomy): untouched-anatomy transaction and one-part-edit
transaction, before/after, receipts into the velocity docket V7 table.

## Execution fences

- Agent A (opus): compile.py only — Prongs A and B, obligations A-1,
  B-1, B-2 discharged with receipts in its report. If B-1 or B-2 fails
  verification, land Prong A alone and report the refutation; Prong A
  is independently valuable and independently testable.
- Agent B (opus): `tests/test_field_memo.py` only, written against the
  API pinned here: `_composed_snapshot_key(graph, points) -> str`,
  `_composed_snapshot_root() -> Path`, snapshot store/load private
  helpers named `_load_composed_snapshot` / `_store_composed_snapshot`,
  splice entry `_spliced_or_recomputed(...)` internal to the wrapper.
  Where implementation drifts from these names, the DESIGN names win —
  Agent A conforms to them. Agent B runs in parallel with Agent A, so
  Agent A's symbols may not exist yet: author the tests, verify syntax
  with `py_compile` ONLY, and do NOT run pytest — the scoped runs happen
  at review, after both agents land. Test grids stay small (res ≤ 48;
  no test may exceed 10s).
- Neither agent touches session/, cli/, or any other engine module.
  Neither commits. Both write receipts (what was verified, how) into
  their final reports.

## Verdict (fable, 2026-07-23, post-execution)

**Prong A — LANDED.** Agent A (opus-4.8) cut it; fable reviewed at
source: the six-part key is complete against every governing input
(pinned by `test_snapshot_key_names_every_governing_input`), the cold
check runs before any store contact (E5 law), rejections never store,
writes are `tmp + os.replace` under the 4 GiB `composed/` cap with the
existing mtime-LRU prune. Obligation A-1 discharged: `.sections` is
consumed only inside the fold — both call sites and every downstream
consumer read `field` + `evidence` alone, so snapshots reconstruct with
`sections=()`.

**Prong B — REFUTED via the designed escape hatch.** Obligation B-2
failed verification, and the refutation was verified at source by fable:
the certified fold composes RAW UNCLIPPED SDFs under a global min
(`hard = np.minimum(accumulated, arriving)`, algebra.py:510;
`composed = hard + correction` — the correction is masked, the min base
is NOT). The localized-morphology envelope logic this design cited as
the bounded-support reference clips only on the LEGACY path. Under an
unclipped global min, the only sound influence set of a changed part is
its Voronoi cell — domain-filling, so no bounded-support certificate
exists for the operators in use; forcing compact support would change
field values (a banned behavior change). Tests 5–6 of the plan fell
with the prong (5 converted to full-recompute-and-store semantics, 6
deleted with no hollow replacement); test 7 landed as the permanent
conviction pin
`test_global_support_adversary_convicts_naive_envelope_splicing`, which
constructs the argmin-at-a-distance case and proves the change escapes
any naive envelope. Naive splicing on this fold is convicted forever;
any future splice proposal must first repeal this pin's geometry.

**Tests:** 6/6 in `tests/test_field_memo.py`, green under BOTH the warm
env and `GOLEM_CIRCUIT_CACHE_COLD=1` (the fixture pins the ambient cold
env off — warm-path semantics are the subject; cold law has its own
dedicated test that sets the env deliberately).

**Honest successor (surfaced, NOT commissioned — scope is Rosalia's):**
the edit-loop win Prong B could not deliver is reachable as per-section
sampled-field reuse — memo each section's sampled field+gradient at the
production grid, re-evaluate ONLY the edited section symbolically, and
recompose the cheap NumPy fold over all cells. Bitwise-sound trivially
(same reduce, same operands), but the price is heavy per-section storage
(~N_sections × field bytes per grid).

Measurement receipt: `rehearsal/skin-siege/measure-field-snapshot.json`
(numbers in the velocity docket V7 table).
