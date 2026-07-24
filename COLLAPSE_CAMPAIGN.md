# GOLEM — Python Collapse Campaign

*A behavior-preserving collapse of the living `golem/` package: delete accidental
complexity, and tighten what remains toward the typed idiom the codebase already
practices in `golem/materials/core.py`. This is the **Python collapse**. The
greenfield selector/cascade engine — the total `Expr` scope-language with two
evaluators, the equality-saturation style search — is **deferred to the Haskell
kernel** (Appendix A of the spec) and is deliberately out of scope here.*

## Context

GOLEM has become "a CSS cascade engine with the cascade removed": it names symbols
and resolves values, but the machinery that should carry that resolution is
fragmented. The compute plane is genuinely clean (mechanics, sheaf, anatomy,
body/engine — pure solvers that refuse conflicts with typed obstructions). The rot
is not in the essential complexity; it is in four **tooling gaps** where Python was
written below the discipline the house style already knows how to keep. None of
them requires a language switch to close. They require deletion and typing.

The oracle stays frozen. Only `golem/`, `specs/`, and `tests/` change. `pilots/`,
`conformance/`, `outputs/`, `golden/`, `phase-minus-one/` are never edited.

## The four gaps

| Gap | Disease | Cure (Python-expressible) | House exemplar |
|---|---|---|---|
| **G1** | Untyped `dict` blackboards passed between modules; every consumer re-parses defensively. `build_senses` is a 28-key bag (`senses/proprio.py:314`); the op record is a bare dict (`session/ops.py:35`). Silent `{gencyl,blob,box}` fall-throughs corrupt centroids (`proprio.py:571`). | Frozen dataclasses + PEP-695 tagged unions + `match`/`typing.assert_never`. | `materials/core.py` (`MaterialDecodeResult`, `StrEnum`, `Accepted`/`Rejected`) |
| **G2** | "One algebra, many interpreters" promised (`sheaf.py:8-9`) but half-built and welded: the thermal interpreter is fused into the generic balance core, and a **second interpreter (hydraulics) already lives out-of-module** in `anatomy.py`, hand-wired. Zero `Protocol` exists. | A `BalanceInterpreter` `Protocol` (`lower`/`lift`); thermal + hydraulic become registered interpreters over the generic `solve_balance`. | `sheaf.solve_balance` (`:899`) is already the domain-agnostic core |
| **G3** | ~28 incompatible "how do I name a thing" grammars. The canonical `parse_addr` (`ops.py:84`) is re-implemented as opaque string-surgery in ≥4 places; cell identity has two incompatible schemes; three disjoint colon-prefix scope grammars. | One `Address`/`Scope`/`Anchor` ADT + one first-order matcher module the fragmented sites delegate to. **First-order, not a GADT** — the GADT is the deferred Haskell layer. | `assembly/service.py:482 decode_semantic_address` (already conformant) |
| **G4** | Mutable-spec + deepcopy-undo. `SessionState.spec` mutated in place; undo replays inverse ops embedding `deepcopy`; **every transaction deepcopies the whole spec and full-recompiles** (`state.py:93`). The revalidation-by-recomputation disease. | Typed `Op` ADT (round-trips to identical JSONL) + a vendored stdlib HAMT for structural sharing. No new dependency. | — |

**Dependency constraint:** `pyproject.toml` + `uv.lock` are SHA-256-pinned in the
frozen `phase-minus-one/manifest.json`. `pyrsistent`/`immutables` are therefore
**out**. G4's persistent map is a vendored pure-Python HAMT inside `golem/session/`.

## Five workstreams

- **WS0 — Typing infrastructure.** Standardize on stdlib `typing.assert_never`;
  scoped strict type-check (orchestrator-run) on every touched module. Precondition,
  not deletion.
- **WS1 — G1.** `build_senses` → frozen `Senses` dataclass with typed sub-records
  (`InstanceDatum`, `PartDatum`, `Band`, a `Supported | NoSupport` union for
  `balance`). Consumers (`asserts.py`, `session/receipt.py`, `body.py`,
  `eval_anatomy.py`) drop defensive `.get`/`[...]` and read typed fields.
- **WS2 — G2.** Extract `BalanceInterpreter` Protocol; reduce `solve_thermal_balance`
  to a `ThermalInterpreter`; refactor the two hydraulic problems in `anatomy.py`
  (`:3470`, `:3646`) to the same Protocol. **Coefficients move verbatim** — this is
  reorganization, not re-derivation. Public signatures stay byte-stable.
- **WS3 — G3.** `golem/addressing/` — the `Address`/`Scope`/`Anchor` ADT + matcher +
  `GridDims`. The ~28 sites collapse into four constructor families and
  delegate. Ships with a round-trip property test: `render(parse(s)) == s`.
  **Wave-1 correction (cell identity):** the "two incompatible cell schemes" is
  *essential, not accidental*. `sheaf.CellId` is a dense solver DOF ordinal
  (`enumerate(solid_cells)` in `assembly/core.py:521`, consumed as `values[int(id)]`);
  `mechanics.CellIndex`/F1 `flatten` is a sparse geometric grid position. They diverge
  on any hollow voxel — merging them corrupts the solver. Ruling: keep them **distinct
  types**; F1's `flatten`/`unflatten` have no consumer (only `GridDims.contains` is
  load-bearing); the geometric↔DOF bridge stays the `enumerate(solid_cells)` map.
- **WS4 — G4.** *WS4a:* typed `Op` ADT with `to_json`/`from_json` to the identical
  dict (JSONL replay is a wire contract). *WS4b (optional, last, gated):* vendored
  HAMT `PMap` replaces deepcopy-the-world; canonical bytes stay produced solely by
  `body._canonical_json`, so the M0 byte-equality property is untouched.
  **RULING 2026-07-20: WS4b never shipped — production never adopted `PMap`; the
  dormant subsystem (`session/pmap.py` + its tests) was deleted in the repair
  campaign. Session state remains dict + deepcopy behind immutable carriers.**

## Dispatch topology

The naive "one agent per gap" is impossible: G3 entangles the files every other gap
owns. Resolution — a **foundation wave of new modules only**, then a **parallel
migration wave with disjoint per-file ownership**, all targeting the frozen
foundation interfaces.

### Wave 0 — Foundation (new files only; no call-site churn; independently testable)

| Module | New path | Contents |
|---|---|---|
| **F1** | `golem/addressing/core.py` | `Address`/`Scope`/`Anchor` ADT + first-order matcher + unified `CellId` (WS3 core) |
| **F2** | `golem/senses/model.py` | frozen `Senses` + sub-record unions (WS1 type surface; `build_senses` still returns dict until Wave 1) |
| **F3** | `golem/session/pmap.py` | vendored stdlib HAMT `PMap` + tests (WS4b core; built and battle-tested in isolation before it backs the session) |

### Wave 1 — Parallel migration lanes (disjoint source-file ownership)

| Lane | Source files | Slices |
|---|---|---|
| **SESSION** | `session/{ops,journal,core,state,receipt,outline,repl}.py` + `tests/test_session_*` | WS4a, WS4b, WS3(session addrs), WS1(receipt consumer) |
| **SENSES** | `senses/proprio.py`, `contracts/asserts.py`, `evals/eval_anatomy.py` + `tests/test_{proprio,asserts,schematic}` | WS1(producer+consumers), WS3(`parse_scope`, `_scope_part`) |
| **BODY** | `kernel/body.py` + `tests/test_body.py` | WS1(senses consumer), WS3(landmark/world/port surgery) |
| **KERNEL** | `kernel/sheaf.py`, `kernel/anatomy.py`, `conduits/*` + their tests | WS2(Protocol seam), WS3(`CellId`, address surgery) |
| **MECH** | `kernel/mechanics.py` + `tests/test_mechanics.py` | WS3(`CellIndex` ↔ `CellId`) |

Every Wave-1 lane depends only on frozen Wave-0 interfaces, never on another lane —
that is what makes them genuinely parallel. `anatomy.py` does **not** consume the
`senses` blackboard (grep-confirmed), so KERNEL is independent of F2.

## Verification (orchestrator-run only)

Workers never run `uv`, `pytest`, or `git` — they produce code to spec; the
orchestrator verifies. Two-tier gate:

1. **Freeze canary — `uv run pytest`** (sealed; imports only `pilots/`). Invariant
   under correct `golem/` edits. A red here means a frozen file or a pinned hash was
   touched — check `git diff --name-only ⊆ golem/ specs/ tests/` first. Steady state:
   25 passed, 3 strict-xfail.
2. **Behavioral gate — `uv run pytest tests`** (living). The real regression gate.
   `-m "not slow"` first, full pass before accepting a wave.

Checkpoints: **CF** (foundation modules' isolated tests green, both suites at
baseline) → **C1..C5** (each lane's owned test subset in its own worktree) → **CW1**
(integrate all lanes, both tiers full). The wave — not the file — is the green
checkpoint granularity, because the blackboard and op-record interfaces flip
atomically (no shim permitted). Bisection is by disjoint test-file ownership: a red
`test_proprio` names SENSES, a red `test_sheaf` names KERNEL, a red in `conformance/`
names a freeze violation.

## Risk register

| Risk | Mitigation |
|---|---|
| **G3 breaks a string-surgery consumer** (a provenance key or contract scope silently shifts). | Round-trip property test in Wave 0 over each site's real input corpus; migrate as delegation, not rewrite; keep `Anchor` distinct from `Scope`. |
| **G4b persistent-map diverges from deepcopy** (aliased subtree, different canonical bytes). | Canonical bytes stay from `body._canonical_json` (sorts keys); randomized op-sequence equivalence test (`PMap` vs `deepcopy` → identical bytes); WS4b is last, optional, aborted if not bit-clean. |
| **G2 seam shifts a solved field.** | Move coefficients verbatim; keep public signatures stable; gate on exact-float `test_sheaf`/`test_conduits_vascular`/`test_anatomy`. |
| **Non-atomic blackboard migration** (a consumer left reading `senses["…"]`). | F2 freezes the type in Wave 0; all consumers migrate in the same wave, verified together at CW1. |
| **Worker panic from running builds** damaging a worktree. | Worker prompts forbid `uv`/`pytest`/`git`; each brief is self-contained (anchors + target types + delete list). |
| **Frozen-file drift.** | Freeze canary + pre-integration assertion that changed set ⊆ `golem/ specs/ tests/`. |

## Sequencing

1. **WS0 + Wave 0 foundation (F1, F2, F3)** — first, always. Defining the interfaces
   once (with round-trip + isolation tests) is what converts a serial, conflict-prone
   campaign into a parallel one.
2. **Wave 1, all five lanes in parallel** — WS1, WS2, WS3-migrations, WS4a. Within
   SESSION, WS4a before WS4b.
3. **WS4b last, and gated** — deepest change, highest divergence risk, least
   behavioral payoff (the suite already passes; mesh-free recompute is already the
   design). Ship WS4a and stop if the equivalence test isn't bit-clean.

**If scope must be cut**, the highest-value core is F1 + F2 + SENSES/BODY/
SESSION-WS4a + KERNEL-WS2 — the blackboard, the interpreter seam, and the scope
grammars, deferring only the persistent-map perf work and the deferred-to-Haskell
stalk vectorization.

## Rules of engagement for worker agents

- Edit **only** files named in your lane. Never touch `pilots/`, `conformance/`,
  `outputs/`, `golden/`, `phase-minus-one/`, `pyproject.toml`, `uv.lock`.
- **Never** run `uv`, `pytest`, `git`, or any build. Produce code to spec; the
  orchestrator verifies.
- **Delete and replace in one pass.** No adapters, shims, or transitional
  projections beside the old path.
- Mirror the house idiom in `materials/core.py`: frozen dataclasses, `StrEnum`,
  PEP-695 tagged unions, `match` + `typing.assert_never` on every closed vocabulary.
