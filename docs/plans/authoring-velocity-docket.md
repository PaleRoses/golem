# Authoring-Velocity Docket (wave 6) — numbers sealed, pending fleet-close ratification

Goal restated (Rosalia, 2026-07-23): iterate the golem until it yields
high-quality 3D creatures on short notice. Trial 4 priced the current loop:
~9 hours and 44 verdicts for one cursorial quadruped, two authors, one hang
that ate 2h45m of dead air. This docket legislates the cure from receipts,
not aspiration. Every element cites the receipt that convicts the current
law. Status: V1 mechanism LANDED and V7 numbers SEALED 2026-07-23
(skin-siege journal entry 015, `rehearsal/skin-siege/measure-warm-start.json`,
commit d417b2e687); remaining elements ratify at fleet close.

## The indictment (receipts)

- Trial 4 (`rehearsal/formation-trial/FRICTION.md`, sealed): 60 -> 0
  unexpected across 21 luna verdicts + 22 closer verdicts, ~9h wall.
- Latency ledger (`docs/plans/evidence-law-journal.md`,
  `docs/plans/skin-siege-journal.md`): flesh-field eval 20.50s at
  resolution 242 is the dominant verdict term; the trial paid ~82x per
  transaction with the skin token on; the ray stencil paid 9.69s where the
  siege's descent pays 1-2.6s.
- Wall 005 (masking): the verdict-40/41 pair burned a full author cycle
  discovering evidence the kernel already possessed and dropped.
- Wall 006 (loose acceptance): "accepted" with unexpected>0 invited authors
  to stop early and forced re-litigation at visual QA (Wall 007).
- Minion thesis (trial-4 close): the surface carries weak authors TO the
  threshold, not through it — empty `help.operations` on rejections is the
  measured gap (live example: the blend-bad decode fault renders
  `help.operations: []`).
- Severance (2026-07-23): end-loaded journals lose everything; the
  contemporaneous-journal law (already in force) is a velocity measure —
  zero harvest was lost across an infrastructure restart and a quota death.

## Elements

### V1 — Warm-start authoring mode (skin first, then the principle) — LANDED

The siege landed a mechanism STRONGER than the commissioned warm-seed:
`_SKIN_MEMO` in `engine/compile.py` — a bounded (2-entry) in-process memo
whose fingerprint is SHA-256 over problem + sealed policy +
domain/resolution/pitch + the EXACT flesh field/vertices/faces bytes. A hit
returns the prior certified sheet byte-exactly with ZERO solve, so the
warm-seed's seed-dependence soundness hole never opens; any changed
governing input misses and solves cold. `GOLEM_CIRCUIT_CACHE_COLD=1`
bypasses the memo, so acceptance oracles seal on canonical cold solves by
construction. Pinned by
`test_authoring_skin_memo_reuses_unchanged_governing_inputs`.

Receipts (`measure-warm-start.json`, seven samples, res 64, the accepted
quadruped): formed-solve median 0.9499s cold -> 0.0366s on memo hit
(25.94x); formed/legacy overhead 29.99x -> 1.156x. The trial's ~82x skin
tax is dead for unchanged-input transactions; the residual 15.6% is
fingerprint hashing and transport, not solving.

The principle now legislated for every expensive stage (field eval,
assembly): memo keyed by canonical governing-input fingerprints at the
seam — never revalidation by recomputation. The cold solve remains the
canonical seal at acceptance. No gate is ever weakened in warm mode.

### V2 — Starter lineages

A session may open from any accepted lineage receipt (spec + verdict +
evidence fingerprint) instead of an empty base. The seed is an accepted
STRUCTURE with its certification already priced — the author pays only for
the delta. No curated content library; the shelf is the set of sealed
rehearsal lineages, and it grows only by acceptance.

### V3 — Tournament authoring

Receipts: one luna hung for 2h45m; five parallel kimi lanes self-organized
through a restart and a quota death with zero lost work. For descent-shaped
authoring (N transactions toward 0 unexpected), run K cheap authors in
parallel lanes from the same base with disjoint verdict lineages;
first-to-accept (or best margin at a time box) wins; the losers' FRICTION
entries are harvested regardless — walls found by losers are still walls.

### V4 — Incremental revalidation via content-addressed staged memos — LANDED (Prong A); naive splicing CONVICTED

Design + verdict: `docs/plans/field-memo-v4-design.md`. Law correction
against this element's original framing: content-addressed memo keys ARE
the invalidation algebra — the session edit algebra's role is
observability (per-stage wall clock, hit/miss in receipts), never
invalidation; no edit-reachability mapping exists or may exist (R-V2).

Landed mechanism: composed-field snapshot memo at
`_sample_graph_field_certified_checked` — sha256 key over the complete
governing set (canonical positive instances with mirrored flags,
operators, and blends in fold order; carve sequence; `graph.blend` +
domain bounds; certification resolution; points bytes/shape/dtype;
source salt), snapshots under `composed/` with a 4 GiB mtime-LRU cap and
atomic writes, the cold check before any store contact, rejections never
memoized.

Prong B (dirty-region splice for the edit loop) REFUTED via the design's
own escape hatch: the certified fold min-composes RAW UNCLIPPED SDFs
(algebra.py `normalized_hard`/`normalized_smooth`), so a changed part's
sound influence set is its Voronoi cell — domain-filling; no
bounded-support certificate exists for the operators in use.
`test_global_support_adversary_convicts_naive_envelope_splicing` pins
the conviction permanently.

Prong C (V4b, per-section sampled reuse) LANDED 2026-07-24
(`docs/plans/field-memo-v4b-design.md`): the production pass factors
exactly into gather (per-section full-grid symbolic sampling) +
compose (elementwise arithmetic over cached arrays); sections are
content-addressed by SHAPE-canonical geometry + mirrored +
points-digest + salt + ROLE (the role-blind key collision — carve
entries lack gradients positive sections demand — was caught in review
and cured); certification pass never memoized; cold/rejection laws
inherited. Edited-anatomy receipts
(`rehearsal/skin-siege/measure-field-edit-res{121,192}.json`, Sable,
bounds-stable pelvic_basin shrink): res 192 miss 22.32s -> edit warm
2.55s (8.7x, floor 2.28s); res 121 miss 4.84s -> 0.62s (7.7x). Store
and floor scale cubically; extrapolated edit at seal res 242 ~=
4.6-5.2s vs the measured 25.36s miss (~5x) — the res-242 receipt
itself is blocked on ~20 GB free disk (store bill: 43 positive
instances x 432.5 MB). Hull-moving edits shift the derived grid and
lawfully miss every key — measured law. Prong D (tile-pruned symbolic
evaluation, attacking the miss) remains uncommissioned.

Receipts (`rehearsal/skin-siege/measure-field-snapshot.json`, Sable
Stride formed, res 242): miss median 25.36s (fold + one-time snapshot
write; uncached baseline 20.50s, so a miss pays ~+24% once per content),
hit median 0.228s — 111.1x. Gates: binding union 1081/0 cold + R10 5/5
cold; `tests/test_field_memo.py` 6/6 under warm AND cold envs.

### V5 — Machine-drafted friction skeletons

Machines for receipts, minds for interiority: when a verdict lands a NEW
obstruction kind (or the author invokes a wall), the session appends a
skeleton FRICTION entry — verdict id, codes, addresses, timestamps,
diagnostic deltas — contemporaneously. The author owes only the prose
interior (what was tried, what it felt like, what the law should learn).
An entry the machine started cannot be lost to a hang.

### V6 — Help density floor

Every rejection carries either at least one compile-checked
`help.operations` candidate or a typed statement of why no lawful operation
exists. The evidence lane lands the mechanism for vascular/contract/anomaly;
this element extends the floor to every diagnostic code and adds the law
test: no rejection path may emit an empty help block without the typed
justification. Margins/slack accompany every violated predicate
(required/observed/binding_constraint non-null).

### V7 — The latency budget

Authoring-mode verdict latency is a sealed budget, measured and reported
per transaction in the session receipt (wall-clock per stage, memo
hit/miss). The post-V1 measured floor (entry 015): a skin-bearing
transaction with unchanged governing inputs pays 1.156x legacy (0.0366s vs
0.0317s at res 64) — the skin term is settled. The field term is settled
by V4's composed-field snapshot memo: 20.50s -> 0.228s on unchanged
anatomy at res 242 (111.1x; a miss pays 25.36s once per content —
receipt `rehearsal/skin-siege/measure-field-snapshot.json`). The
EDITED-anatomy term is settled by V4b's per-section memo (2026-07-24):
a bounds-stable edit pays 0.62s at res 121 / 2.55s at res 192
(7.7x/8.7x vs its own miss; extrapolated ~5s at seal res 242, receipt
blocked on disk); hull-moving edits still pay the full fold — that and
the first build are Prong D territory. The budget is set from these
measured floors, not invented. A transaction that blows the budget is
itself a reportable event (the authoring loop's Wall entry, not a silent
stall).

## Rulings (R-V series, proposed)

- R-V1: warm results never seal. Acceptance re-derives cold; oracle
  agreement (R10) is asserted on the cold path only.
- R-V2: memo keys are canonical-quotient fingerprints of governing inputs;
  a memo that can be invalidated by an edit it does not key on is unsound
  and forbidden (the revalidation disease in cache clothing).
- R-V3: no velocity mechanism may weaken a gate, drop a diagnostic, or
  reorder acceptance semantics. Velocity is bought with reuse and
  parallelism, never with rigor.
- R-V4: every lane in a tournament writes a contemporaneous journal; the
  harvest is the union of all lanes' walls, not the winner's alone.

## Sequencing

1. Fleet close + merge review + R10 gate (in flight, wave 5).
2. V1 skin memo/warm — DONE (landed + measured 2026-07-23); V7 skin term
   settled at 1.156x, field-eval term handed to V4.
3. V4 content-addressed field memo — DONE (Prong A landed + measured
   2026-07-23: 20.5s -> 0.228s untouched anatomy; splicing convicted;
   Prong C per-section memo landed + measured 2026-07-24: bounds-stable
   edit 2.55s at res 192, ~5s extrapolated at seal res).
4. V5 + V6 (authoring-surface quality; cheap, high leverage for weak
   authors).
5. V2 + V3 (trial 5 runs as the first tournament, from a starter lineage).
