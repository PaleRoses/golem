# Golem Decomposition Campaign

Pure **structural** refactor of the `golem/` Python package. Three aims, in the user's words:
compress repeated code · establish non-flat folder structures · decompose thousand-line
elements into coherent, non-duplicative subsystems. **NO behaviour change. NO bug hunting.**
The motive is reasoning: flat peer-sprawl and multi-thousand-line god-files cannot be reasoned
about, only endured. Clean categorical decomposition is the cure.

## Role-naming scheme (categorical module decomposition)

Every carved module is named by its algebraic role, never by flat peer accident:

| role            | holds                                                      |
|-----------------|-----------------------------------------------------------|
| `Core`/`Types`  | carriers — dataclasses, enums, NewTypes, unions, consts   |
| `Algebra`       | folds / reducers / analysers / interpreters               |
| `Coalgebra`     | builders / generators / unfolds                           |
| `Compile`/`Lower` | problem-lowering, verdict derivation                    |
| `Effect`/`Harness` | IO, CLI, rendering, orchestration roots                |

Exemplars already in-tree: `golem/materials/core.py`, `golem/addressing/`, `golem/session/`.

## Surface-preservation rule (zero consumer edits)

Each monolith `foo.py` becomes a package `foo/` whose `__init__.py` is a **pure re-export
façade**: it re-exports the exact public `__all__` so every `from golem.x.foo import Y` and
`from golem.x import foo` resolves unchanged. Consumers, `pilots/`, and `conformance/` are
**never** edited. The one exception is `assembly/core.py` (see Lane B) — a `mock.patch`
constraint forces it to stay a *module* + re-export hub, extracting to *siblings*.

## Frozen — never touch (oracle corpus)

`pilots/` · `conformance/` · `outputs/` · any `golden/` · `phase-minus-one/` ·
`pyproject.toml` · `uv.lock`. Editing these forfeits the oracle.

## Non-target files (leave untouched this stage)

`kernel/engine.py` (lower-vocabulary layer; the `_read_only_array` twin stays put for Stage 2),
`assembly/exchange.py`, `assembly/mounts.py`, `senses/model.py`, `senses/silhouette.py`,
`senses/symmetry.py`, `senses/elemental_render.py`, and every parent-package `__init__.py`
of a shared directory (`kernel/__init__.py`, `assembly/__init__.py`, `senses/__init__.py`).

## The seven lanes (Stage 1 — disjoint file ownership, fully parallel)

Each lane reads its persisted design and owns ONLY its file-set. Carve = verbatim code moves
into role modules; apply ONLY the design-sanctioned behaviour-preserving internal dedups; SKIP
any dedup the design flags byte-risky / golden-feeding / optional (co-locate, don't merge).
NO cross-file dedup (C1–C10) — that is Stage 2.

Design files live under the session scratchpad: `…/scratchpad/design-<lane>.md`.

- **Lane A — anatomy** · `kernel/anatomy.py` → `kernel/anatomy/` package (+ `realize/` sub-package).
  Carve pattern: package-replace (create package, delete `anatomy.py`).
  Landmine: `node_index` is a DOF ordinal → `sheaf.CellId`; keep the string-id↔DOF boundary.
  Re-export the 10 test-pinned privates through `__init__` (no test edit). `__init__` import-light,
  preserve every lazy import.
- **Lane B — assembly/core** · `assembly/core.py` STAYS the module `golem.assembly.core`
  (spine + re-export hub); extract leaf clusters to **sibling** `.py` files under `assembly/`.
  Carve pattern: sibling-extract (`core.py` shrinks, is NOT deleted).
  DECISIVE CONSTRAINT: `test_coupled_assembly.py` `mock.patch`es four internals by dotted string
  (`golem.assembly.core._realize_element_vasculature`, `_compile_coupled_assembly`,
  `_evaluate_compiled_for_physics`, `_solve_sized_coupled_assembly`). Those four AND their
  bare-name callers stay physically defined in `core.py`, or the patch silently no-ops.
  Landmine: `_ResolvedMaterialDomain.cell_id_by_index` bridge stays whole in `voxel.py`.
- **Lane C — service** · `assembly/service.py` → `assembly/service/` package (package-replace).
  Cross-file landmine ruled hands-off: `service.SemanticAddress` vs `addressing.Scope` are
  textual twins but semantically distinct (element-qualification, field order, subset). Do NOT
  touch that this campaign.
- **Lane D — mechanics** · `kernel/mechanics.py` → `kernel/mechanics/` package
  (+ `embedded_channel/` sub-package). Package-replace.
  Landmine: `CellIndex` is a geometric triple with NO flatten projection — the STALE comment at
  `addressing/core.py:524-526` ("Mechanics' CellIndex becomes a projection…") is DEAD; the real
  import is `GridDims` alone. Do not wire `CellIndex` to `addressing.CellId` or `sheaf.CellId`.
- **Lane E — sheaf** · `kernel/sheaf.py` → `kernel/sheaf/` package. Package-replace.
  Landmine: `sheaf.CellId` is a dense DOF ordinal — a distinct NewType with a warning docstring;
  NEVER merged with `addressing.core.CellId`. Keep the warning comment first in `vocabulary.py`.
- **Lane F — body** · `kernel/body.py` → `kernel/body/` package. Package-replace, TWO tiers:
  Tier 1 free lift-and-shift (module-level helpers → role modules), Tier 2 `Compiler` phase
  mixins (verbatim method moves, MRO-preserving — `Compiler(_ValidateMixin, _KinematicsMixin,
  _GoalsMixin, _EmitMixin, _IntentMixin)`). The `Compiler` is a quarantined imperative island;
  do NOT purity-lift it (separate future transmutation). Landmine: `_canonical_json` verbatim,
  re-exported despite the underscore (crossed by `session/state.py`, `test_session_ops.py`).
- **Lane G — proprio** · `senses/proprio.py` → `senses/proprio/` package. Package-replace.
  `senses/model.py` already a sibling — untouched. Keep the `asserts` import confined to `cli.py`;
  add `proprio/__main__.py` to preserve `python -m golem.senses.proprio`. Do NOT edit
  `contracts/asserts.py` (its `_nz/_f3/_s3` twin is Stage-2 C5).

## Landmine register (consolidated)

1. **Three cell-identity schemes coexist BY DESIGN** — `addressing.CellId` (flat spatial),
   `mechanics.CellIndex` (geometric triple), `sheaf.CellId` (dense DOF ordinal). Merging any pair
   is unsound solver corruption. The only legitimate crossing is `assembly/voxel.py`'s
   `cell_id_by_index` bridge (`enumerate(solid_cells)`), which stays whole.
2. **`addressing/core.py:524-526` comment is STALE/DEAD** — ignore it for the mechanics carve.
3. **`mock.patch` physical-definition** — Lane B's four patched fns + callers stay in `core.py`.
4. **`_canonical_json`** (body) and **10 anatomy privates** and **proprio test-reached privates**
   are re-exported through their `__init__` so NO test file is edited.
5. **`service.SemanticAddress` vs `addressing.Scope`** — hands-off (behaviour-adjacent).
6. **Byte-risky internal dedups** (body landmark/flesh-center, proprio `-0.0` formatting) — SKIP
   this stage; the reorganization is the guaranteed win, risky compression waits.

## Rules of engagement (workers)

- NEVER run `uv` / `pytest` / `git` / any build. You edit files only. Verification is the
  orchestrator's job (two-tier gate below). Agents that build panic and nuke worktrees.
- Own ONLY your file-set. If you believe you must touch a file outside it (a sibling package,
  `engine.py`, a parent `__init__.py`, a test, a frozen dir), STOP and report instead.
- No shims, adapters, or transitional projections. Delete-and-replace in one pass.
- Honest: never weaken an assertion or fabricate. Preserve behaviour exactly.
- Report your model ID (self-declared) in your final summary.

## Verify gate (orchestrator only, after each stage)

1. **Freeze canary**: `uv run pytest` at package root — 25 passed / 3 strict-xfail.
2. **Behavioural**: `uv run pytest tests` — `-m "not slow"` = 452, slow = 12, total **464 / 0**.
3. **Golden corpora** byte-exact (`outputs/**/golden/`).

## Stage 2 (deferred — cross-file compression, against the decomposed tree)

Wave-0 shared homes then coordinated call-site migration: `senses/_raster.py` (C1/C2/C7),
`kernel/_geometry.py` (C3/C8/C9), `golem/_arrays.py` (C4), `golem/_format.py` (C5),
`golem/_numbers.py` (C6), `evals/_frame_eval.py`+`evals/_cli.py` (C1/C10). C8 (orthonormal frame
thresholds 0.9/0.99/0.999) is the sole landmine — parameterize per caller, never unify the
threshold. Plus the byte-risky internal dedups skipped in Stage 1, plus the `asserts` C5 twin.
