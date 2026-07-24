# Wave 5 Journal — Lane C (Total oracles)

Contemporaneous journal for C14/C15/C16. Walls land when hit. Law: `.venv/bin/python`;
oracles via `GOLEM_CIRCUIT_CACHE_COLD=1 .venv/bin/python -m golem ...`; no git.

## 000 — reconnaissance, before surgery

- Docket read in full. Rulings R10-R13 govern; my elements are C14 (typed
  EngineFault projection), C15 (poisoned-engine agreement gate), C16
  (Cinderwake permanent witness). Territory: `golem/cli/check.py`,
  `golem/cli/look.py`, session CLI exception boundaries, the
  `unexpected_compile_exception` classifier. Verdict payload shapes (Lane A)
  and senses/contract (Lane B) are not mine.
- Originating receipt: `rehearsal/wyvern-trial/FRICTION.md` 003 — naked
  `TypeError` through `check` as `assembly.unexpected_compile_exception`;
  `look` crashed in traceback on the same call.
- Exception-boundary census (pre-change):
  - `session/effect.py::compile_authored` — `except Exception` ->
    `CompileObstructed((UnexpectedCompileException,))` (body.compile seam).
  - `session/protocol.py::_evaluate_authoring_state` — `except Exception`
    around `compile_assembly` -> `assembly.unexpected_compile_exception`.
  - `session/protocol.py::evaluate_authoring_surfaces` — outer
    `except Exception` -> `session.unexpected_evidence_exception`.
  - `cli/check.py::_decode_source` — catch-all `except Exception` ->
    `REJECTED [Type] path: msg` stderr (untyped); matches
    `UnexpectedCompileException` for the body.compile seam.
  - `cli/check.py::_check_loaded` — catch-all around
    asserts/senses/anomalies -> same untyped stderr.
  - `cli/compile.py::compile_spec` — NO boundary: `compile_assembly`
    exceptions escape raw. This is the wall-003 `look` crash vector.
  - `cli/look.py::run` — NO boundary: render/write exceptions escape raw.
  - `cli/session.py::run` — NO boundary around `start_session`/`submit`.
- Advisory noted: no existing mock.patch harness file exists; the "seam
  discipline" is patch-by-dotted-binding at the seam the oracle actually
  reaches, and the gate MUST prove the poison fired (sentinel kind/message
  asserted inside the verdicts), or all three oracles "agree" vacuously.
- Advisory noted: Lane C boundary is exception projection only; I render
  whatever obstruction shapes Lanes A/B hand me and do not reshape verdict
  payloads.
- Cold baseline on the pre-change tree (surface-formation already landed):
  - `check specs/cinderwake.json --all`: 9.1s, exit 0,
    `ASSEMBLY acceptance: ACCEPTED`.
  - `session rehearsal/wyvern-trial/base.json specs/cinderwake.json`: 8.7s,
    exit 0, `accepted`, diagnostics 58 expected / 0 unexpected — matches
    `verdict-04.json`. The lineage is consistent going in.

## Design decisions (pre-implementation, recorded once)

- One owner: new module `golem/session/fault.py` owns the `EngineFault`
  record (seam, exception_kind, message, derived sha256 message_digest),
  the seam-address constants, `project_engine_fault`, and the canonical
  `engine.fault` code. `session/state.py` already owns the compile-outcome
  union and imports it; CLI modules import the same owner. No second
  projection anywhere.
- `UnexpectedCompileException` is deleted from the union, not aliased:
  `type CompileObstruction = BodyCompileObstruction | EngineFault`.
- One typed code `engine.fault` for every seam; the seam address
  (`body.compile`, `assembly.compile`, `session.evidence`, `session.submit`,
  `check.senses`, `check.decode`, `orthographic.render`, `orthographic.write`)
  distinguishes origin. The full witness `{code, seam, exception_kind,
  message_digest, message}` is identical across session/check/look for the
  same fault — that equality is what C15 asserts.
- `cli/compile.py::compile_spec` gains the boundary (returns `EngineFault`
  in its obstruction union); `compilation_obstruction_result` renders it.
  This totals both `compile` and `look` at the assembly seam — R11 says no
  CLI oracle may surface a raw exception, and the seam is shared.
- Session diagnostic projection keeps the house shape (code/severity/
  address/predicate/required/observed/witness/help); only the retired
  catch-all codes change. Address = seam; witness = canonical fault JSON.

## 001 — inheritance inventory, second session (2026-07-23 ~16:05)

- Read the evidence-law journal and the LIVE protocol.py before any cut, per
  the pool fence. The evidence lane's Wall-006 claim is present
  (`diagnostic_expected` free function; `accepted` requires
  `unexpected == 0`). The contamination-flagged duplicate `expected` property
  on `Diagnostic` (~line 139) is still in the tree — THE EVIDENCE LANE
  DELETES IT; I do not touch it. My work needs ZERO protocol.py edits: the
  assembly.compile (line ~1578) and session.evidence (~1621) boundaries and
  `engine_fault_diagnostic` are landed and correct.
- Tree inventory vs. design decisions: `session/fault.py` born as designed;
  `session/state.py` union is `BodyCompileObstruction | EngineFault` with
  `UnexpectedCompileException` fully deleted (no alias anywhere; only docs
  and sealed receipts mention it); `session/effect.py` body.compile boundary
  landed; `cli/check.py` decode+senses boundaries landed; `cli/look.py`
  assembly/render/write boundaries landed; `cli/session.py` session.submit
  boundary landed. The sibling's blend-bad test update stands
  (tests/test_session_protocol.py:753-757 asserts engine.fault @
  body.compile) — verified by reading, not re-derived.
- REMAINING C14 HOLE: `cli/compile.py::compile_spec` still has NO boundary —
  `compile_assembly` exceptions escape raw, and `CompilationObstruction`
  lacks `EngineFault`. This is also look's assembly seam (look calls
  `compile_spec`), so look's landed boundary at run() compensates, but the
  `compile` oracle itself still dies in traceback. CUT NOW, per the recorded
  design: boundary in `compile_spec`, `EngineFault` in the union, rendered
  via `fault.render_refusal()` in `compilation_obstruction_result`. Seam
  address `assembly.compile` — the same engine call, the same address, so
  C15's cross-oracle witness equality holds.
- C15 gate shape: poison `compile_assembly` at the dotted binding each
  oracle reaches — `golem.session.protocol.compile_assembly` (covers session
  AND check, which evaluates through the protocol) and
  `golem.cli.compile.compile_assembly` (covers compile AND look, which calls
  compile_spec). Sentinel RuntimeError; assert its kind/message inside every
  oracle's typed output so agreement cannot pass vacuously. New test file
  `tests/test_oracle_agreement.py` — mine alone.
- C16: cinderwake agreement in the same file, cold via
  `monkeypatch.setenv("GOLEM_CIRCUIT_CACHE_COLD", "1")` (the env is read at
  call time in kernel/engine/compile.py:1002, so in-process cold works).
  Split per-oracle so no test approaches the 10s budget; agreement is
  enforced by pinning every oracle to the SAME expectation: accepted, exit
  0, matching the verdict-04 lineage (0 unexpected diagnostics).
- `rehearsal/` lineage READ-ONLY honored: tests read verdict-04.json /
  check-accepted.txt as oracles, never write.

## 002 — close-out verification (fable, 2026-07-23 ~16:10)

The second session was severed by Kimi quota exhaustion (403, billing cycle)
AFTER landing its cuts but BEFORE verifying or journaling them. Verified
under fable's own hand:

- Landed cuts inventoried: `cli/compile.py` boundary (+26/−5, the last C14
  hole — `compile_spec` now returns `EngineFault` in its obstruction union)
  and `tests/test_oracle_agreement.py` (C15 poisoned-seam gate + C16
  Cinderwake permanent witness, three cold oracles sharing one in-process
  compile to stay under the 10s law).
- One defect repaired by fable (mechanical, no assertion touched): the C16
  session test passed `--all-diagnostics`; the CLI flag is `--all`
  (dest `all_diagnostics`). One-token fix at test_oracle_agreement.py:166.
- Receipts: `tests/test_oracle_agreement.py tests/test_session_protocol.py`
  — 44 passed, 11.91s. The C15 gate proves the poison fired (sentinel
  kind/message asserted inside every oracle's typed output; identical fault
  JSON across session/check/compile/look; no Traceback in stderr). C16 pins
  cold session to the verdict-04 lineage (accepted, unexpected 0) and cold
  look to the look-final PNG set.

Lane C is COMPLETE: C14 (every seam boundary landed, union closed,
`UnexpectedCompileException` deleted not aliased), C15, C16 — implemented
across three sessions, verified green.
