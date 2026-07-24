# V4b — Per-section sampled-field memo: implementation report

Executor: opus (constrained cut). Scope: `golem/kernel/engine/compile.py`
only. Verified with `.venv/bin/python -m py_compile` and self-authored
inline probes at grid resolution ≤ 16. No git, no pytest, no commit.

Verdicts: **C-1 CONFIRMED · C-2 CONFIRMED · C-3 CONFIRMED · C-4 discharged ·
C-5 discharged.** The cut landed in full; no soundness obligation refuted,
so the escape hatch (report-not-approximate) was not triggered. One review
amendment landed after the cut — fable's cross-role key-collision finding,
recorded as deviation 5.

All file:line receipts below are post-edit line numbers in
`golem/kernel/engine/compile.py` unless another file is named.

---

## What landed

New surface (all pinned-API names honored exactly):

- `_SectionSample` frozen dataclass — `field` + `gradient|None` (compile.py:182).
- `_SectionGather` frozen dataclass — fold accumulator carrying the compose
  `result` plus the tuple of pending `(key, sample)` writes (compile.py:214).
- `_SECTION_CACHE_SUBDIR = "sections"`, `_SECTION_CACHE_MAX_BYTES`
  (default `1 << 35` = 32 GiB, env override `GOLEM_SECTION_CACHE_MAX_BYTES`)
  (compile.py:104-106).
- `_section_sample_root()` — `field/sections`, sibling of `field/composed`
  (compile.py:1098).
- `_points_digest(points)` — sha256 over `points.tobytes()` + shape + dtype
  (compile.py:1102); reused inside `_composed_snapshot_key` (compile.py:1125).
- `_section_shape_signature(geometry)` (compile.py:1157) +
  `_section_sample_key(geometry, mirrored, points_digest, *, role)` (compile.py:1183).
- `_load_section_sample` / `_store_section_sample` / `_flush_section_samples`
  (compile.py:1205 / 1216 / 1231).
- `_sample_section_arrays(section, points, chunk_size, executor)` (compile.py:1513)
  and `_sample_carve_array(part, mirrored, points, executor)` (compile.py:1538) —
  the extracted symbolic gather seam; **all** symbolic sampling on **both**
  passes routes through these two (positive sections via `gather_section`,
  compile.py:1582; carves via `gather_carve`, compile.py:1659).
- `_prune_field_cache` gained a `pattern="*.pkl"` param (compile.py:1051-1054);
  section store passes `"*.npz"` (compile.py:1228).

Fold rewrite (`_sample_graph_field_certified_fold`, compile.py:1551): the
`sample` closure is gone; `merge` and `subtract_carve_field` now (a) consult
the section memo on the production warm path only, and (b) thread pending
writes through the `_SectionGather` accumulator, flushing on acceptance.
Signature and external contract unchanged — the V4a `_fold_spy`
(`tests/test_field_memo.py:30`) forwards `real(graph, points, topology_shape,
executor)` positionally, and that four-argument shape is preserved verbatim.

---

## C-1 (key exclusion) — CONFIRMED

`_prepared_composition_section` (compile.py:454) consumes only the instance
`(geometry, mirrored)`: it calls `prepared_web_sdf(geometry, mirrored)`,
`prepared_part_sdf(geometry, mirrored)`, `prepared_part_gradient(geometry,
mirrored)`, `_instance_bounds(geometry, mirrored)`, and reads
`geometry.part_id` for the identifier. No `graph.blend`, no domain bounds, no
operator.

Every evaluator whose output *is* the sample was read at source and
destructures only shape fields plus `mirrored` — never `.blend`, never
`.operator`:

- `prepared_part_sdf` (algebra.py:2063) — matches on `formation`, `spine`,
  `radii`, `profile`, `center`, `size`, `rotation`, `round_radius`.
- `prepared_part_gradient` (algebra.py:2301) — same field set.
- `prepared_web_sdf` (algebra.py:789) — reads only `web.anchors`.
- `part_sdf_checked` (algebra.py:1802, the carve evaluator) — same shape-only
  field set.

Therefore the sampled `(field, gradient)` over fixed points is a pure function
of shape geometry + `mirrored`, and excluding `operator`, per-instance
`blend`, `graph.blend`, domain bounds, fold order, the certification
resolution, **and the `part_id` label** from the key is **sound**.

Per fable's key-law adjudication, `_section_sample_key` (compile.py:1183) does
NOT pickle the whole `Part`/`SpanningWeb` dataclass (which carries `part_id`,
`operator`, `blend`, `obstructions`). Instead `_section_shape_signature`
(compile.py:1157) builds a discriminated payload from **exactly** the
evaluator-governing fields this C-1 verification enumerated, and nothing else:

- `("gencyl", formation, spine, radii, profile)`
- `("blob", center, size, rotation)`
- `("box", center, size, round_radius, rotation)`
- `("web", anchors)`

The full key payload prefixes a `role` discriminator (`"positive"`/`"carve"`;
deviation 5, fable's review) ahead of this signature, then appends `mirrored`,
the points digest, and the salt. `part_id` never reaches
`evaluate`, so it is structurally absent from the key — two identical-shape
parts with different ids collapse to one store entry (verified empirically),
and a rename never misses. No excluded input leaks into `evaluate`; the
evaluators consume nothing beyond kind + geometric parameters + `mirrored`, so
no additional field enters the key.

## C-2 (compose purity) — CONFIRMED

Inputs to `_compose_certified_field` (compile.py:814) enumerated, each
classified:

- `state` (`_ComposedField|None`) — `.field` composed array + `.sections`
  (sampled `_SampledCompositionSection`s) + evidence → gather outputs.
- `incoming` (`_SampledCompositionSection`) — sampled `field`/`gradient` +
  `prepared` symbolic section → gather output + prepared symbolic.
- `operator` — scalar from `geometry.operator`.
- `radius` — scalar from `graph.blend`/`geometry.blend` × diagonal (diagonal
  from bounds).
- `points` — the fixed grid; `domain` (`_Bounds`) — from
  `graph_bounds_checked`.
- `topology_shape` — `None` on the production pass.

Internal operations touch only those: `compose_union` (legacy + certified
paths) is arithmetic over sampled arrays and certificate scalars;
`_compatible_overlaps` and the gradient-magnitude certificates are
resolution-independent symbolic probes (9³ grids, recomputed every pass, never
memoized); `clean = np.minimum(state.field, incoming.field)` and the
`reduce(np.minimum, …)` chain (compile.py:966-969) are **elementwise**, not
reductions. `_local_composition` (algebra.py:363-575) is elementwise per point;
its only whole-grid reductions — `np.max(outside_delta)`, `np.median` over the
two-element certificate bounds, `np.count_nonzero(active_mask)`,
`np.max(|correction|)` — feed **rejection guards and diagnostic evidence,
never the field values**. `local_topology_obstruction`/`local_topology_counts`
short-circuit to `None` when `shape is None` (`_component_count`,
algebra.py:320, returns before touching the array), so the production pass does
no topology work.

No grid-wide reduction feeds field arithmetic. The compose phase is a pure
function of gather outputs + cheap recomputation. Design not refuted.

## C-3 (chunking determinism) — CONFIRMED

`_sample_sdf` (compile.py, unchanged) returns `field(points)` for a single
chunk, else `np.concatenate(executor.map(field, chunks))`. `executor.map`
preserves submission order, so concatenation is deterministic. Every evaluator
above is elementwise per point (no cross-point normalization, sorting, or
length-dependent coupling; the `(N,3)@(3,3)` matmuls in blob/box are per-row),
so `field(whole)` equals `concat(field(chunk_i))` bitwise. The refactor does
not alter `_sample_sdf` usage — `_sample_section_arrays`/`_sample_carve_array`
reproduce the prior `sample`-closure and carve logic exactly (same chunk-size
selection: `_WEB_FIELD_CHUNK_SIZE` for `SpanningWeb`, `_FIELD_CHUNK_SIZE`
otherwise). A stored entry is the byte-for-byte output of a chunked symbolic
eval; a warm hit returns those bytes; a fresh symbolic recompute of the same
shape over the same points chunks identically. The bitwise pin therefore holds
by construction and was confirmed empirically (below).

## C-4 (store hygiene) — discharged

- **Atomic writes**: `_store_section_sample` (compile.py:1216) writes
  `{key}.{pid}.npz.tmp` then `os.replace` to `{key}.npz`. Tmp names end in
  `.tmp`, so the `*.npz` prune glob never sees a half-written file.
- **float64, no pickle, uncompressed**: `np.savez` (not `_compressed`) of the
  raw float64 arrays; loads via `np.load(..., allow_pickle=False)`. `gradient`
  is stored as a second archive member only when present; absence is detected
  by `"gradient" in archive.files` (compile.py:1211) — no sentinel, no lossy
  conversion.
- **Prune after flush**: `_prune_field_cache(root, _SECTION_CACHE_MAX_BYTES,
  "*.npz")` runs inside each store, mtime-LRU, on the separate `sections/` root
  with its own cap — section churn cannot evict composed snapshots.
- **Cold gate before ANY store contact**: `memo_active = topology_shape is None
  and not _field_cache_bypassed()` (compile.py:1579). When cold, no load, no
  pending, flush guarded by `if memo_active` — `_section_sample_root()` is
  never called, so `sections/` is never created. Probe confirmed.
- **Rejection discard**: pending accumulates in `_SectionGather.pending`;
  `_flush_section_samples` runs only after both the positive and carve folds
  yield an accepted `_ComposedField` (compile.py:1702). A rejection anywhere
  returns before the flush, discarding pending — including entries for sections
  sampled before the rejecting one. Probe: rejecting graph leaves `sections/`
  empty.
- **Duplicate-key collapse**: identical geometry twice yields one `.npz`
  (same key, `os.replace` overwrite) and, on the warm pass, zero symbolic
  samples with two hits. Probe confirmed (1 entry, 0 warm samples).

## C-5 (behavior identity) — discharged

The refactor changes **when** arrays are computed, never **what**. The fold
stays *interleaved* (gather inside `merge`/`subtract_carve_field`, compose
inside `merge`) rather than physically gathering all sections up front,
precisely so the rejection short-circuit is byte-identical: `merge` returns the
accumulator unchanged when `result` is already `RejectedSurfaceFormation`
(compile.py:1612), so no section is sampled after an earlier section rejects —
exactly as before. Chunk sizes, executor routing, and compose-fold law-check
order are untouched; a memo hit returns arrays bitwise-equal to the symbolic
recompute and never reorders or skips a compose law check (compose always runs
in full). The certification pass (`topology_shape` not `None`) has
`memo_active` false, so it samples symbolically exactly as before, touches no
store, and flushes nothing.

Empirical (probes, res 16, production grid ≠ certification res 17):

- Warm shape-edit of one part re-samples **only** the edited section at
  production points (1 of 2); the other section hits.
- **Bitwise pin**: warm-assembled edited field `np.array_equal` cold full
  symbolic recompute — holds for both a shape edit and a blend-only edit.
- Blend-only edit re-samples **zero** sections (all hit — direct confirmation
  of the operator/blend exclusion) while the composed field correctly reflects
  the edited blend.
- Cold env: two identical calls create no `sections/`.
- Rejecting graph: `sections/` empty.
- Duplicate geometry: one store entry, zero warm samples.

---

## Deviations from the design (all forced, none loosen a law)

1. **`_prune_field_cache` gained a `pattern` parameter.** The design says
   "pruned by the existing `_prune_field_cache(root, cap)`", but that function
   globs `*.pkl` (composed/part caches) and section entries are `.npz`. Without
   a pattern the prune would never see section files, silently unbounding the
   cache. Added `pattern="*.pkl"` (backward-compatible; existing callers keep
   the default) and pass `"*.npz"` for sections. Tmp files are named
   `…​.npz.tmp` so the `*.npz` glob excludes them.

2. **`_points_digest` is not threaded into the fold; it is computed once per
   pass, not once per call.** The design asked for "computed once per call" by
   reusing the digest across `_composed_snapshot_key` and the section keys. The
   V4a `_fold_spy` (`tests/test_field_memo.py:30`, unmodifiable under the
   fences) monkeypatches `_sample_graph_field_certified_fold` with a frozen
   four-arg signature and forwards positionally; a fifth `points_digest`
   parameter would raise `TypeError` under the spy. So the fold computes its
   own digest on the production pass (compile.py:1580), and
   `_composed_snapshot_key` computes another. Net: at most **two** digests per
   top-level edit (composed key + production fold), never per-section — the
   per-section blow-up the design targets is fully eliminated.
   `_composed_snapshot_key` still calls `_points_digest` internally
   (compile.py:1125) and carries an optional `points_digest` param for
   forward-compatibility (no current caller passes it; the V4a two-arg call
   site is preserved).

3. **`_section_sample_root` is not `@cache`d** (unlike `_composed_snapshot_root`).
   The V4a fixture (`tests/test_field_memo.py:50-59`) patches only
   `_field_cache_root`, not `_section_sample_root`; a cached root would go
   stale across V4a's per-test `tmp_path`s. Deriving it live from
   `_field_cache_root()` (itself `@cache`d) each call is correct and cheap
   (a path join). The V4b fixture patches `_section_sample_root` directly, so
   caching would be moot there anyway.

4. **Section key excludes `part_id` and is built from evaluator-governing
   fields, not a whole-dataclass pickle.** This now *matches* fable's sharpened
   key-law adjudication rather than deviating from the original text. C-1 proved
   the evaluators read only kind + geometric parameters + `mirrored` — never
   `part_id`, `operator`, `blend`, or `obstructions` — so `_section_shape_signature`
   (compile.py:1157) encodes exactly that per-kind field set and nothing else.
   Excluding `part_id` makes duplicate-geometry collapse structural: two
   identical-shape parts with different ids yield one store entry (the decoder
   permits duplicate positive-section ids when no skin layer is present —
   `types.py:1864` gates the uniqueness check on `skin.value is not None`).
   Soundness is preserved because the fresh `CompositionSectionId` identifier
   always comes from re-preparation (`_prepared_composition_section`), never from
   the cached array — a warm hit reuses only the field/gradient bytes, and
   compose sees the correct per-instance identifier. Building the payload from
   named fields (rather than `replace`-ing sentinels into the dataclass) also
   guarantees a future non-evaluator field added to `Part` cannot silently leak
   into the key.

5. **`_section_sample_key` carries a keyword-only `role` discriminator
   (`Literal["positive", "carve"]`, no default), prefixed into the key payload.**
   Origin: fable's review; landed **jointly** — fable's hand supplied the
   `Literal` import, the signature, the rationale comment, and the
   `gather_section` call site; this executor's edit supplied the `gather_carve`
   call site. The two edits raced on `compile.py`, were reconciled, and the
   merged form is `py_compile`-clean (an amendment, not a self-directed
   deviation). The shape-signature key of deviation 4 was role-blind: a positive
   section and
   a carve instance with the same geometry shared one key. That rested soundness
   on two premises nobody proved — (a) that `prepared_part_sdf`'s closures
   (algebra.py:2063) and `part_sdf_checked`'s inline paths (algebra.py:1802) are
   bitwise-identical for every shape family forever, and (b) gradient arity: a
   carve-stored entry (field only) hit by a gradient-demanding `local_blend`
   positive would return `gradient=None` and warm-reject
   (`FieldScaleUncalibrated`) what cold accepts, a warm/cold divergence the
   bitwise pin forbids outright. The source salt cures cross-*version* drift,
   never same-version cross-*role* collision. Cure: `role` is prefixed into the
   payload (compile.py:1194); `gather_section` passes `role="positive"`
   (compile.py:1594), `gather_carve` passes `role="carve"` (compile.py:1668).
   The namespaces split cleanly — carve entries only ever serve carves, positive
   entries only positives — so no cross-path value-equality theorem is assumed
   and the arity asymmetry is structurally unreachable. The absence of a default
   forces every call site to name its role, so a future caller cannot silently
   reintroduce the collision. Cost: a shape used in both roles stores twice —
   negligible and honest.

## Fences honored

Modified only `golem/kernel/engine/compile.py` and created this report. No
other file touched; `algebra.py`/`types.py` read-only. No git subcommand run.
No pytest / full suite run — verification was `py_compile` (clean) plus inline
probes at res ≤ 16 (each well under 10 s). Nothing committed. The deviation-5
role amendment was a reconciled joint edit — the team lead's hand landed most of
it on `compile.py` concurrently, this executor's edit landed the `gather_carve`
call site; every other change to that file is this executor's alone. The
out-of-scope
`_skin_formation_sections` vertex sampler (compile.py:~1810) and the legacy
`_evaluate_legacy_checked` region (compile.py:~4030) were identified and left
untouched — the section memo is scoped to the certified fold's production pass
only.
