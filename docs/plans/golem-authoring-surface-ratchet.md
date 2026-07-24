# Build the permanent authoring-surface evaluation ratchet

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept current while the work proceeds. This document follows `../../docs/llm-guidance/PLANS.md` from the `potentialimprovements/golem-kernel` repository root.

## Purpose / Big Picture

GOLEM can currently judge fixed visual or mechanical cases, but it cannot compare the cost of authoring creatures across trials without coupling the evaluation to canonical paths or manually reading prose. After this change, any authoring trial that supplies an ordered transaction history can be measured for rounds to first acceptance, next-transaction repair success by obstruction code, and total transcript tokens consumed through acceptance. Each source bundle produces one deterministic, content-addressed JSON receipt. Re-running the same input is idempotent, and corpus summaries are derived from append-only receipts rather than becoming a second source of truth.

The behavior is visible by running the focused tests in `tests/test_authoring_surface_eval.py`, recording a trial through `python -m golem.evals.authoring_corpus`, and summarizing a receipt directory. The checked-in initial corpus also records which historical repository artifacts can be measured and which cannot.

## Progress

- [x] (2026-07-21) Located `pilots/judge.py`, `golem/evals/harness.py`, `golem/cli/eval.py`, `phase-minus-one/schema.json`, and the rehearsal/trial artifact families.
- [x] (2026-07-21) Established that the generic transaction-level authoring metrics and append-only corpus owner are absent.
- [x] (2026-07-21) Implemented the immutable authoring-trial model, total decoders, metrics, receipt model, and filesystem boundary under `golem/evals/`.
- [x] (2026-07-21) Generated thirteen deterministic first-corpus receipts: one measured session transcript and twelve typed unparseable historical source bundles.
- [x] (2026-07-21) Added focused tests for metric laws, decoding, deterministic content addresses, append-only behavior, corpus gluing, and repository corpus receipts.
- [x] (2026-07-21) Ran the focused, tracked-suite, and unrestricted dirty-tree validation cones and recorded their exact outcomes below.

## Surprises & Discoveries

- Observation: The file named in the request, `pilots/judge.py`, does apply one voxel occupancy, connectivity, watertightness, shell, and symmetry procedure to both conditions, but its executable edge hardcodes `golem_v4.json` and `control_creature.build()`.
  Evidence: The shared `report(mesh, name)` function is condition-independent; only the `if __name__ == "__main__"` block chooses those two canonical sources.

- Observation: The living eval CLI has a second, distinct canonical coupling.
  Evidence: `golem/cli/eval.py::_is_anatomy_case` and `_is_elemental_flow_case` compare resolved input paths with `specs/knight_body.json` and `specs/elemental_field.json`; the anatomy evaluator also owns canonical body and reference paths internally.

- Observation: `phase-minus-one/schema.json::TrialEvidence` cannot own the requested ratchet without preserving the coupling being removed.
  Evidence: It admits only eight Obsidian Warden creature IDs, two arm IDs, and the exact stage sequence `initial`, `r1`, `r2`, `r3`. Its stage metrics include render rounds and elapsed seconds but no ordered transaction obstruction codes or transcript token counts.

- Observation: Historical rehearsal artifacts are mostly final snapshots or prose summaries rather than machine-complete transaction histories.
  Evidence: `rehearsal/REHEARSAL.md`, the harness probe reports, `rehearsal/session-knight/OPERATOR_NOTES.md`, and `rehearsal/reforge/README.md` name counts and failures in prose; the corresponding files do not preserve a transaction-aligned verdict, obstruction-code, and token record.

- Observation: `golden/session_biped_receipts.txt` is a deterministic transaction transcript with transaction IDs, contract status, typed contract codes, clear events, and `est_tokens` for every transaction.
  Evidence: Its contract becomes fully measurable and passing at transaction 10, and its preceding compact deltas are sufficient to reconstruct the active contract-obstruction section exactly.

- Observation: The unrestricted test tree changed materially during validation because parallel lanes added source and tests while the commands were running.
  Evidence: The first full run collected 793 tests and transiently failed while `project_anatomy_obstruction` was absent; the function appeared in the live dirty source before the focused rerun. The final full run collected 799 tests and failed only two untracked `tests/test_session_diagnostic_semantics.py` cases against the separately owned session projection work.

## Decision Log

- Decision: Keep `golem/evals/harness.py` as the owner of ordinary evaluation rendering and subprocess receipts; add a distinct authoring-trial metric owner rather than widening that harness into persistence and historical decoding.
  Rationale: Stable evaluator rendering and cross-trial transaction economics are different laws. Combining them would give a small rendering boundary responsibility for corpus identity, storage, and trial semantics.
  Date/Author: 2026-07-21 / Blue Rose

- Decision: Do not modify `golem/cli/eval.py` or add a compatibility adapter there.
  Rationale: The requested reusable authoring harness is source-format driven, not spec-path driven, and the task limits changes to additive eval-plane files plus tests. A direct module entry point avoids preserving or extending canonical spec dispatch.
  Date/Author: 2026-07-21 / Blue Rose

- Decision: Represent every unavailable metric and every import failure explicitly instead of using `null`, zero, or inferred success.
  Rationale: Missing acceptance, incomplete token coverage, a terminal obstruction with no next transaction, and a lossy prose artifact are distinct typed obstructions. Collapsing them would fabricate outcomes.
  Date/Author: 2026-07-21 / Blue Rose

- Decision: Use rational repair rates as integer numerator and denominator, never floating-point percentages.
  Rationale: The values remain exact, deterministic, and composable across corpus gluing.
  Date/Author: 2026-07-21 / Blue Rose

- Decision: Persist only content-addressed per-source receipts; calculate corpus summaries on demand.
  Rationale: Receipts are append-only authority. A stored mutable summary would be a derived second owner and would eventually lie.
  Date/Author: 2026-07-21 / Blue Rose

- Decision: Do not repair the two unrestricted-suite failures in the parallel session lane.
  Rationale: The user explicitly forbade changes under `golem/session/`, the failing test file is untracked parallel work, and the tracked pre-existing suite plus this eval lane is green. Crossing that boundary would trade honest validation for scope corruption.
  Date/Author: 2026-07-21 / Blue Rose

## Outcomes & Retrospective

The eval plane now owns a source-format-independent transaction model, exact first-acceptance and obstruction-repair metrics, partial token evidence, strict generic JSON and session-receipt decoders, content-addressed append-only receipts, and derived corpus gluing. The initial corpus contains thirteen receipts: `golden/session_biped_receipts.txt` is measured at ten rounds and 469 estimated receipt tokens through acceptance; twelve other historical bundles retain typed unparseable outcomes rather than invented metrics.

Validation completed with `15 passed` in the focused file and `699 passed` for every tracked pre-existing test plus the new eval test. The unrestricted dirty-tree command completed at `797 passed, 2 failed`; both failures are in untracked `tests/test_session_diagnostic_semantics.py` against the forbidden parallel session lane. No requested or owned file failed. No commit was created.

## Context and Orientation

`pilots/judge.py` is the frozen pilot geometry judge. Its `report` function applies identical metrics to arbitrary meshes, but its executable block loads one canonical GOLEM graph and one canonical control builder. `golem/evals/harness.py` defines accepted evaluation results, a generic obstruction, stable JSON rendering, and a subprocess receipt. `golem/cli/eval.py` chooses registered evaluators, but two predicates recognize cases by exact canonical paths. The existing evaluators under `golem/evals/eval_*.py` measure geometry, flow, plates, or vascular constructors rather than the authoring process.

The preregistered experiment schema in `phase-minus-one/schema.json` owns its sealed Warden experiment, not general authoring trials. The requested authoring surface needs an ordered sequence of transactions. Each transaction has an identity, an accepted or obstructed verdict, the complete set of active obstruction codes returned after that transaction, and optionally the transcript token count attributable to that transaction. The first accepted transaction determines rounds to acceptance. Token cost is available only when every transaction through that acceptance has a token count. An obstruction occurrence is repaired when that code is absent from the immediately following transaction; a final occurrence with no following transaction is recorded as unfollowed rather than failed.

The first corpus audit covers machine-complete session receipt transcripts and the historical trial-like bundles under `rehearsal/`, `pilots/`, and `golden/`. A historical bundle is measured only when its source format establishes the required facts. Otherwise its content-addressed receipt stores a typed unparseable result and the exact source hashes.

## Plan of Work

Add `golem/evals/authoring_surface.py` as the pure semantic owner. Define immutable trial, transaction, accepted/obstructed verdict, metric availability, per-creature metrics, obstruction repair counts, and evaluation result types. Provide a strict version-one JSON decoder and a decoder for deterministic GOLEM session receipt transcripts. Both decoders return either a trial or accumulated typed decode obstructions. Implement first-acceptance descent, token-coverage validation, exact next-transaction repair counts, and corpus-level gluing without filesystem effects.

Add `golem/evals/authoring_corpus.py` as the thin effects boundary. It reads source bytes, computes SHA-256 source references, constructs measured or unparseable receipts, serializes stable JSON, derives the receipt content address, and creates the receipt file without overwriting any existing path. Its module entry point records one source in either generic JSON or session-receipt format and renders a corpus summary. Any filesystem or decode failure becomes a typed obstruction and a nonzero exit rather than an exception-shaped success.

Add `golem/evals/corpus/README.md` and generated receipt files under `golem/evals/corpus/receipts/`. The README defines the metric and receipt schema and inventories the historical bundles considered. Generate receipts through the new boundary rather than patching their bytes manually. Supported sources become measured entries; lossy summaries become unparseable entries with source hashes and exact obstruction codes.

Add `tests/test_authoring_surface_eval.py`. Tests will construct small trials directly, decode both generic JSON and session receipts, prove first-acceptance and token-prefix behavior, prove next-transaction repair accounting including unfollowed occurrences, prove exact corpus aggregation, prove deterministic receipt names and idempotent append behavior, prove refusal to overwrite mismatched bytes, exercise the module entry point, and verify every checked-in corpus receipt has the content hash named by its filename.

## Concrete Steps

From `/Users/bluerose/Developer/pale-meridian/potentialimprovements/golem-kernel`, implement the two eval modules and focused tests. Run:

    uv run pytest tests/test_authoring_surface_eval.py

Generate the repository corpus receipts only after those focused tests pass. Re-run the focused test so it checks the generated corpus. Then run:

    uv run pytest tests

The focused suite must pass with deterministic receipt bytes, and the full test suite must retain every pre-existing passing test plus the new authoring-surface tests.

## Validation and Acceptance

Acceptance requires a generic JSON trial with two obstructed transactions followed by one accepted transaction to report three rounds to acceptance; repair counts grouped by obstruction code; and a token value equal to the exact sum through transaction three when all three token observations exist. Removing one token observation must make only the token metric unavailable. Removing the final accepted transaction must make rounds and tokens unavailable without erasing repair evidence.

Recording the same source and metadata twice into a temporary corpus must return the same content address and preserve one file. A pre-existing file at that address with different bytes must return a typed collision obstruction and leave the file untouched. Loading multiple measured receipts must glue repair numerators and denominators by obstruction code while retaining per-trial creature metrics. Conflicting measured receipts for one trial identity must obstruct the summary.

Every checked-in historical receipt filename must equal the SHA-256 digest of its canonical JSON bytes. At least the complete biped session transcript should produce measured rounds and token evidence. Historical prose/snapshot bundles lacking transaction-aligned evidence must appear as unparseable, with no invented metric values.

## Idempotence and Recovery

Receipt creation uses content-addressed filenames and exclusive creation. Re-running an identical record operation is safe and returns the existing receipt. No operation rewrites or deletes a receipt. If implementation fails before a receipt is created, retrying is safe. If a hash-named path contains different bytes, stop with the collision obstruction; do not repair it by overwriting the corpus.

The task must not commit, change branches, or touch `golem/cli/forge.py`, `golem/cli/check.py`, `golem/session/`, the anomaly classifier, or `golem/kernel/body/`.

## Artifacts and Notes

The initial historical import inventory and exact measured/unparseable outcomes are recorded in `golem/evals/corpus/README.md`. The receipt directory contains one measured entry and twelve unparseable entries, all validated by filename/content SHA-256 equality in `tests/test_authoring_surface_eval.py`.

## Interfaces and Dependencies

`golem/evals/authoring_surface.py` will expose immutable source-format-independent domain types, `decode_authoring_trial_json`, `decode_session_receipt`, `evaluate_authoring_trial`, and `summarize_authoring_trials`. The implementation uses only the Python standard library.

`golem/evals/authoring_corpus.py` will expose source artifact and corpus receipt types, receipt construction, stable serialization, append-only persistence, receipt loading, corpus summarization, and `main`. The module entry point will support explicit source format, trial identity, creature identity, and corpus directory arguments. No spec path selects semantics.

The receipt JSON schema version is one. Receipt payloads contain source artifact path and SHA-256, source format, measured or unparseable result, and no wall-clock fields. Measured results contain per-trial metrics with explicit available/unavailable tags and exact repair counts. The receipt address is the SHA-256 of the complete stable JSON payload bytes; the hash is carried by the filename rather than recursively embedded in the payload.

Revision note, 2026-07-21: Created after source inspection established the two canonical eval couplings, the narrower sealed Phase -1 owner, and the historical artifact limitations.

Revision note, 2026-07-21: Closed implementation and validation with the generated corpus inventory, focused and tracked-suite green receipts, and the explicit parallel-session dirty-tree obstruction.
