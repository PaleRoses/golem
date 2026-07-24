# Promote the GOLEM session algebra to the public authoring protocol

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept current while implementation proceeds.

This document follows `../../docs/llm-guidance/PLANS.md` from the `potentialimprovements/golem-kernel` directory. The user explicitly forbids commits for this work, so the repository-wide instruction to commit during an ExecPlan is superseded for this execution.

## Purpose / Big Picture

GOLEM already has an addressed edit algebra, inverse edits, an append-only journal, undo and redo, and branch-aware transition planning under `golem/session/`. The public forge loop ignores it. Forge currently asks a model for a whole JSON document, writes that document, scrapes human `check` output, separately compiles the assembly, and resends the complete 77 KB contract on every round.

After this change, one canonical session protocol accepts addressed operation transactions, whole-document replacements, and ordinary edited body files. Whole documents and files are normalized into the existing operation algebra before validation. Every reply is structured JSON with stable diagnostic codes, exact addresses, quantitative evidence, machine-applicable edit operations when a lawful operation can be constructed, contract section references, exact changed addresses, feasibility margins, the nearest binding constraint, and differential margin or active-constraint feedback relative to the submitted base transaction. Forge becomes a small policy loop over this protocol and tells the model to fetch named contract sections through the existing `golem contract` command only when needed.

The visible acceptance demonstration is:

    .venv/bin/python -m golem session BASE.json INPUT.json

The command prints one JSON verdict to stdout. A whole-document input and an equivalent addressed-operation input must produce the same final canonical document and the same validation authority. The full repository test suite must remain green.

## Progress

- [x] (2026-07-22 04:53Z) Read `golem/session/`, `golem/cli/check.py`, `golem/cli/forge.py`, `golem/kernel/body/report.py`, `golem/contract/`, the CLI registry, assembly gate, and relevant tests.
- [x] (2026-07-22 04:53Z) Recorded the existing owners and the missing normalization, structured-verdict, margin, delta, and forge-gluing laws.
- [x] (2026-07-22 04:53Z) Established a focused baseline: 26 tests passed across session ops, branches, forge, and typed body-relation boundaries.
- [x] (2026-07-22 05:16Z) Strengthened `MetaLocation` so arbitrary document fields lower through the existing operation algebra.
- [x] (2026-07-22 05:16Z) Added pure whole-document descent with semantic-address edits, section gluing, and canonical-byte replay proofs.
- [x] (2026-07-22 05:16Z) Added the structured protocol, typed diagnostics, lawful help operations, feasibility margins, deterministic binding selection, and base deltas.
- [x] (2026-07-22 05:16Z) Registered `golem session` and pinned top-level, diagnostic, margin, and delta field order.
- [x] (2026-07-22 05:16Z) Deleted forge's prose/check/compile/contract lane and rewired it to one protocol submission channel.
- [x] (2026-07-22 08:24Z) Completed the read-only architectural closure audit after plural diagnostic descent and rich carrier repair. Final validation: 811 passed; independent closure cone: 399 passed; quantitative carrier cone: 59 passed.

## Surprises & Discoveries

- Observation: `MetaLocation` already resolves to the complete authored document and `_replace_record` already glues a rewritten document. Only `_META_KEYS` prevents top-level sections such as `anatomy`, `eyes`, `muscles`, `appearance_palette`, and `surface_detail` from using `SetOp` or `UnsetOp`.
  Evidence: `golem/session/ops.py` functions `resolve_record`, `_replace_record`, and `_compile_set`.

- Observation: a multi-operation program is compiled atomically but receives one journal transaction number per operation. The protocol therefore reports the last committed journal transaction as the new head while preserving every operation and inverse entry.
  Evidence: `golem/session/journal.py` functions `commit_program` and `_entry`.

- Observation: current session compile outcomes flatten contract verdicts into dictionaries and discard typed operators and tolerances. Passing margins cannot be reconstructed lawfully from those records.
  Evidence: `golem/kernel/body/report.py::run_asserts`, `golem/contracts/views.py::verdict_record`, and `golem/contracts/operators.py::apply_operator`.

- Observation: forge acceptance is stricter than session compilation. It requires analytic `check` acceptance and then `compile_spec -> AcceptedAssembly`. Replacing this with `effect.compile_authored` would silently weaken the product.
  Evidence: `golem/cli/forge.py::_advance_round` and `golem/cli/compile.py::compile_spec`.

- Observation: forge renders all contract sections into 77,028 bytes and resends them each round. Twelve rounds resend at least 924,336 contract bytes before current specs and history.
  Evidence: `golem/cli/forge.py::render_authoring_contract` and `_authoring_prompt`; live rendered-size measurement.

- Observation: `uv` cannot initialize in this sandbox because its configured cache is read-only and its macOS system-configuration probe panics even with a writable cache. The checked-in `.venv` executes the same pinned Python environment successfully.
  Evidence: failed `uv run` attempts and the passing `.venv/bin/pytest` baseline.

- Observation: inverse operations for locally destructive rewrites are lawful only when replayed against the post-edit document. When a local inverse cannot reconstruct the exact prior document, the existing root `SetOp(meta)` is the authoritative glued inverse rather than a fabricated local edit.
  Evidence: `golem/session/ops.py::_validated_inverse` and inverse replay pins in `tests/test_session_ops.py`.

- Observation: assembly rejection is a cover, not a scalar. Body, plate, color, palette, detail, anatomy, vascular, mechanics, hydraulics, and sheaf failures must descend to one diagnostic per nested obstruction while preserving native contract domains and route-invariant codes.
  Evidence: `golem/assembly/project.py::project_assembly_obstructions`, `golem/session/protocol.py::_assembly_diagnostics`, and `tests/test_session_diagnostic_semantics.py`.

- Observation: permissive input requires rich local carriers. Eye, muscle, plate, surface, palette, detail, thermal, buckling, hydraulic, embedded-channel, constitutive, and material-fraction constructors now retain typed rules, authored values, thresholds, scale, and provenance before projection; prose cannot recover evidence after the fact.
  Evidence: the obstruction ADTs and construction sites under `golem/kernel/body/`, `golem/plates/`, `golem/materials/`, `golem/kernel/engine/`, `golem/kernel/mechanics/`, and `golem/kernel/anatomy/hydraulics/`.

- Observation: protocol evidence must be detached from compiler-owned mappings and JSON must reject non-finite authored input while still serializing non-finite diagnostic observations as strings. Duplicate contract clause labels also require their full typed identity to remain distinct.
  Evidence: `golem/session/protocol.py`, `golem/json_value.py`, `golem/session/state.py`, and protocol isolation/non-finite/duplicate-identity pins.

- Observation: the current living suite contains 811 tests. The checkout contains concurrent work, so the difference from the user's stated 676-test baseline is not attributed solely to this change; no existing test was weakened or removed by this implementation.
  Evidence: `.venv/bin/pytest -q tests` completed with `811 passed`.

- Observation: the protocol preserves the old forge cost shape rather than adding a validation pass. Analytic evidence compiles once, and full assembly validation runs only after analytic acceptance, matching the former check-then-compile conjunction while deleting repeated contract prompt payloads.
  Evidence: `golem/session/protocol.py::evaluate_authoring_state`, `golem/cli/forge.py::_advance_round`, and the absence of check or compile channels in forge.

- Observation: retaining every prior evidence value would duplicate derived state and grow the authoring session without serving the accepted optimistic-concurrency law. Stale bases are rejected, so only the current evidence is required; the journal remains the sole provenance owner.
  Evidence: `golem/session/protocol.py::ProtocolSession` stores one current `AuthoringEvidence`, while `JournalState.entries` retains operations and inverses.

## Decision Log

- Decision: strengthen `MetaLocation` and the existing `Op` ADT; do not add `ReplaceDocumentOp`, JSON Patch, a parallel IR, or a forge-only patch language.
  Rationale: the current algebra already has lawful `SetOp`, `UnsetOp`, `AddAtOp`, `RemoveOp`, inverses, and atomic program compilation. The missing behavior is normalization and full document coverage, not another edit system.
  Date/Author: 2026-07-22, Blue Rose.

- Decision: whole-document and plain-file normalization share one pure diff function. It descends through stable semantic identities where the existing op algebra supports them and glues at the nearest stable top-level section otherwise.
  Rationale: bones, flesh, arrays, goals, props, parts, mounts, and contacts already have typed addresses and insertion/removal laws. Unsupported body sections must remain editable, but a single root replacement would erase useful locality.
  Date/Author: 2026-07-22, Blue Rose.

- Decision: unknown top-level keys are still journaled and passed to the body compiler.
  Rationale: permissive input requires the compiler's existing `UnknownBodySpecKeyObstruction` and closest-key evidence, not an artificial protocol rejection before semantic validation.
  Date/Author: 2026-07-22, Blue Rose.

- Decision: the canonical protocol aggregates the existing analytic checker authority and the existing full assembly authority. Forge gates only on that aggregate verdict.
  Rationale: this preserves the old forge conjunction without a second gate or prose adapter.
  Date/Author: 2026-07-22, Blue Rose.

- Decision: contract margins are derived from typed `Verdict` and operator carriers before compatibility records flatten them. Other margins are derived only where existing receipts retain both the measured value and a real threshold.
  Rationale: a fabricated margin is worse than an omitted one. The protocol will not infer missing witnesses or thresholds from prose.
  Date/Author: 2026-07-22, Blue Rose.

- Decision: active constraints are the constraints tied for the smallest normalized nonnegative slack in an accepted verdict. The binding constraint is the first active constraint in deterministic source order. Deltas compare margin values by stable constraint id and compare active-id sets.
  Rationale: this gives “nearest to violation” and newly active/inactive semantics without inventing a solver-specific active-set definition for unrelated constraint families.
  Date/Author: 2026-07-22, Blue Rose.

- Decision: protocol JSON is emitted in deliberate insertion order and never with `sort_keys=True`. Author model generation remains unconstrained; only decoded request and verdict wires are checked.
  Rationale: the user requires reasoning-first field order and explicitly forbids grammar-constraining author generation.
  Date/Author: 2026-07-22, Blue Rose.

- Decision: the first public CLI is a one-shot `golem session BASE INPUT` command. Long-lived callers such as forge retain the immutable protocol state between submissions; the reply exposes normalized operations and journal entries.
  Rationale: this proves all three input shapes and the public wire without inventing a daemon, database, lock manager, or persistence format outside the stated scope.
  Date/Author: 2026-07-22, Blue Rose.

## Outcomes & Retrospective

The public protocol now accepts addressed operation envelopes, whole-document envelopes with optional current-base selection, and ordinary edited document objects. Every document path descends through `diff_specs` into the existing `Op` ADT and append-only journal. Semantic rejection advances the journal and preserves the directly editable document; request, stale-base, diff, and edit obstructions leave the protocol state unchanged.

`TransactionVerdict.to_json()` is the sole wire projection. It emits reasoning evidence before transaction bookkeeping: status, diagnostics, margins, binding constraint, deltas, contract references, changed addresses, base and head transactions, normalized operations, the journal with inverses, then the schema id. Accepted bodies carry assembly integrity margins at minimum; typed contract, relation, anatomy, vascular, and assembly-fit owners contribute additional margins only when their real thresholds exist. Failed scalar contract gradients become help operations only after decoding and compiling through the public edit algebra.

Forge is now a policy fold over `start_session` and `submit`. The old check channel, assembly compile gate, full-contract concatenation, prefix scraper, prose transcript, and sorted candidate serializer are gone. Its prompt carries the current authoritative document, the latest compact verdict projection, and the existing on-demand `golem contract SECTION` command. Model generation remains unconstrained.

Validation receipts:

    .venv/bin/pytest -q tests/test_session_protocol.py tests/test_session_diff.py tests/test_session_ops.py tests/test_addressing.py tests/test_forge.py tests/test_cli_contract.py tests/test_typed_evidence.py tests/test_session_diagnostic_semantics.py tests/test_body_relation_projection.py
    277 passed

    Independent architectural closure cone
    399 passed

    Independent quantitative carrier cone
    59 passed

    .venv/bin/pytest -q tests
    811 passed, 164 warnings in 108.64s

The only environmental limitation is the previously recorded `uv` macOS `dynamic_store` panic. The checked-in pinned `.venv` ran the entire living suite successfully. No commit was created.

## Context and Orientation

`golem/session/ops.py` owns the operation algebra. `SetOp`, `UnsetOp`, `AddOp`, `AddAtOp`, `RemoveOp`, and `RenameOp` carry typed addresses and frozen JSON payloads. `decode_op` and `to_json` are the total wire codec. `compile_program` performs a pure left-to-right descent over an immutable next-document value and either returns a `ProgramPlan` with inverses or a typed `EditObstructed` value.

`golem/session/journal.py` owns append-only operation entries, inverse operations, undo, redo, replay, and touched conflict scopes. `golem/session/algebra.py` owns branches and plans an operation program against the current branch before `accept_transition` glues a compiled outcome into that branch. `golem/session/core.py` is currently only a human shell: it raises `Reject` and returns prose receipts. The new public protocol must use the lower algebra directly, not parse or wrap that prose.

`golem/kernel/body/report.py` currently builds proprioceptive senses and compatibility assertion records. The authoritative typed assertion algebra is in `golem/contracts/model.py`, `golem/contracts/verdicts.py`, and `golem/contracts/operators.py`. Passing margins require the typed operator value and tolerance, so the report boundary must expose typed verdicts before deriving existing records.

`golem/cli/check.py` expresses analytic acceptance: body compilation or raw graph decoding, geometry decoding, contract assertions, body and joint-limit violations, and anatomy/vascular feasibility. `golem/cli/compile.py` and `golem/assembly/` express the final assembly gate. Forge currently requires both. The protocol must compose these authorities directly and retain their typed evidence.

`golem/contract/registry.py` is the only contract-section registry. Its exact section names are `primer`, `relations`, `vocabulary`, `exemplar`, `receipts`, `envelope`, and `schema`. A protocol diagnostic references only names from this registry. Forge tells the authoring model to run `python -m golem contract SECTION`; it never embeds another contract copy.

## Plan of Work

First, remove `_META_KEYS` as an artificial completeness fence in `golem/session/ops.py`. A set or unset rooted at `meta` may target any valid JSON field path, including an unknown authored key. Preserve existing record-specific checks for pose and typed additions. Add a root-document set only as a total fallback for a key that cannot be represented by the existing field grammar.

Add a pure specification diff beside the operation algebra. It compares two JSON body documents and emits the existing operations. Stable keyed collections use their domain identities and existing typed additions or removals. Ordinary field changes use `SetOp` or `UnsetOp`. If a structural edit cannot be expressed lawfully through those local operations, the diff emits one `SetOp` for the nearest stable top-level section. Compile the candidate operation tuple against the base as the final normalization proof; if local descent fails, retry the section gluing fallback. Tests prove that applying the normalized program to the base yields the edited document's canonical bytes and that semantic addresses, rather than list indices, appear where domain ids exist.

Expose typed contract verdicts from `golem/kernel/body/report.py` and retain them in session compile outcomes. Build protocol diagnostics from existing body, geometry, contract, anatomy, vascular, and assembly projections. Diagnostics use one stable wire shape: code, severity, address, predicate, required, observed, witness, help operations. A help operation is emitted only after it decodes and compiles through `golem.session.ops`; otherwise the operations list is empty.

Build feasibility margins from evidence whose threshold is present: typed contract operators, accepted body relation residuals, anatomy carrier-radius rows, accepted vascular tolerances and capsule clearance, assembly integrity, and assembly fit receipts. Every margin has a stable id, address, predicate, required, observed, signed slack, normalized slack, unit, and witness. Select active and binding constraints deterministically. Compare the new margin index with the current base evidence to produce moved margins and newly active or inactive constraint ids.

Add the immutable protocol state and submission function under `golem/session/`. The request decoder accepts an addressed operation envelope, a document-replacement envelope, or a plain body document. It checks optimistic `base_txn` against the current journal head, normalizes document inputs through the same diff, plans and journals the operation program atomically, compiles the authored state, runs analytic validation, runs full assembly validation only after analytic acceptance, replaces the current derived evidence, and returns the next protocol state plus the structured verdict. Semantic rejection remains journaled; edit-algebra obstruction leaves the state unchanged.

Add `golem/cli/session.py` and register it in `golem/cli/registry.py`. The CLI loads the base and input through the existing JSON source boundary and prints only ordered JSON on stdout. Source and request failures are represented by the same verdict schema rather than traceback text.

Finally, delete forge's check channel, compile gate, contract concatenation, transcript renderers, prose prefix scraper, and sorted candidate serializer. Forge retains only author reply decoding, round-budget policy, immutable round reduction, protocol submission, current-spec persistence, concise ledger rendering, and prompt construction from the latest structured feedback plus on-demand contract references.

## Concrete Steps

All commands run from `potentialimprovements/golem-kernel`.

1. Edit the session operation owner and add focused normalization tests.

       .venv/bin/pytest -q tests/test_session_ops.py tests/test_session_protocol.py

2. Add typed verdict and margin projection, then run assertion, vascular, body-boundary, and protocol tests.

       .venv/bin/pytest -q tests/test_asserts.py tests/test_body_relation_boundaries.py tests/test_vascular_relaxation.py tests/test_session_protocol.py

3. Register the CLI and rewire forge.

       .venv/bin/pytest -q tests/test_forge.py tests/test_cli_contract.py tests/test_session_protocol.py

4. Run all existing session tests and the full living suite.

       .venv/bin/pytest -q tests/test_session_ops.py tests/test_session_receipt.py tests/test_session_outline.py tests/test_session_attach.py tests/test_session_branch.py tests/test_session_protocol.py
       .venv/bin/pytest tests

The requested canonical command is `uv run pytest tests`. If `uv` remains unusable because of the sandboxed cache and macOS configuration panic, the `.venv/bin/pytest tests` run is the same pinned environment and the limitation must be stated explicitly.

## Validation and Acceptance

Acceptance requires all of the following observable behavior.

An addressed request containing `base_txn` and operations applies through `ops.compile_program`, returns the normalized operations unchanged, exposes journal entries with inverses, and advances the head transaction. A stale `base_txn` returns a stable rejection code and leaves the document and journal unchanged.

A whole-document envelope and a plain edited body file both normalize through the same diff. Their normalized operations replay to the edited document's canonical bytes. Unknown body keys are journaled and appear as the body compiler's structured diagnostic rather than a diff-layer rejection.

Every verdict has the pinned top-level field order. Every diagnostic has the pinned field order and a stable address. Every help operation round-trips through `ops.decode_op`; no textual repair hint is mislabeled as an operation.

An accepted document returns at least one quantitative feasibility margin and a binding constraint. Editing a measured constraint changes the corresponding margin delta. When the nearest constraint changes, the prior constraint appears in `newly_inactive` and the new one appears in `newly_active`.

Forge no longer imports or calls `check.run`, `compile_spec`, contract rendering, or assembly/check prose helpers. Its second-round prompt contains the latest structured diagnostic, delta, and contract section names; it contains neither the full live contract nor prior prose prefixes. Candidate key order is not alphabetized. Forge accepts only when the aggregate session verdict includes an accepted assembly gate.

No existing test is weakened or removed. The complete `tests` suite is green.

## Idempotence and Recovery

All transformations are pure until the existing compiler or CLI filesystem boundary. A failed operation plan returns the unchanged protocol state. A semantic rejection is intentionally a journaled authored state and can be revised by the next transaction. Tests and CLI commands may be rerun without generated source edits. No migration, branch change, commit, or generated-artifact patch is part of this work.

## Artifacts and Notes

The protocol wire is represented by frozen Python carriers plus deterministic `to_json` projections. The authoritative document remains the ordinary body JSON specification. The normalized operation journal is provenance and replay capability; it does not replace or hide the directly editable spec.

## Interfaces and Dependencies

No dependency is added. The implementation reuses Python 3.12 dataclasses and structural pattern matching, the existing `golem.session` algebra, the existing contract verdict ADT, the existing anatomy and body projectors, and the existing assembly compiler.

The public Python interfaces introduced by this plan are concrete and minimal:

    start_session(document, spec_dir) -> ProtocolSession
    submit(session, request) -> SessionSubmission
    decode_request(payload, current_txn) -> TransactionRequest | ProtocolObstructed
    diff_specs(base, edited) -> tuple[Op, ...] | EditObstructed

`SessionSubmission` carries the next immutable protocol state and one `TransactionVerdict`. `TransactionVerdict.to_json()` is the sole public wire projection. Forge imports this Python boundary directly; it never shells out to or parses `golem session`.
