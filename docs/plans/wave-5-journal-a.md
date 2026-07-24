# WAVE 5 JOURNAL — Lane A (gluing gets an authoring address)

Lane: A14, A15, A16 of `docs/plans/wave-5-amendment-docket.md`. Rulings R10-R13
govern; R13 is the law being repaired. Contemporaneous: wall entries land when
hit. Direct interpreter only: `.venv/bin/python`. No git (R-G).

## 000 — severance reconstruction, before any surgery

This lane has burned three sessions; the record so the fourth starts here:

- **Original severed wave (15:02-15:07 window):** no evidence it touched Lane A
  territory. `golem/kernel/sheaf/` and `golem/kernel/blame/` are byte-clean
  against HEAD (`git status` shows no modifications, no untracked files);
  anatomy mtimes predate the window (surface-formation campaign, Jul 23 02:40).
- **15:15 relaunch (`kimi-wave5-a.relaunch.log`, session 019f910c-1e25):**
  launched with a TRUNCATED prompt — the severance preamble only, the lane
  commission after the `---` never arrived (visible verbatim in the log's first
  user message). It made ZERO edit/write tool calls across 7MB of log, wandered
  read-only, and died ~15:29 having echoed Lane B's completion summary. It did
  not open this journal. No partial Lane A edits exist anywhere.
- **15:32 relaunch2 (this session, 019f911b-88f5):** launched with a fully
  EMPTY prompt (`$PRE` unset, `kimi-wave5-a.md` already deleted from the
  dispatch dir — reconstruction from process table + log ownership). Lane
  identity recovered from the dispatch wrapper filename
  (`kimi-wave5-a.relaunch2.log`) and the docket's lane roster.

**Inherited tree state (verified, not assumed):**

- Lane B (Kimi, B14/B15/B16): COMPLETE — `docs/plans/wave-5-journal-b.md`
  entry 003, union 1047/1, FRICTION-002 closed cold. Territory `intent.py` +
  `tests/test_webs.py` carries its landed edits.
- Lane C (C14/C15/C16): IN FLIGHT — `docs/plans/wave-5-journal-c.md`, edits in
  `golem/cli/{check,look,session}.py`, `golem/session/{fault,effect,state,
  protocol}.py`, `tests/test_session_protocol.py`. Its in-flight
  `engine.fault` conversion accounts for the 1 known union failure
  (`test_session_protocol.py[blend-bad]`). NOT my territory; I do not touch
  CLI exception plumbing, and I read Lane C's `protocol.py` diff before any
  edit to that shared file.
- Evidence-law / skin-siege lane: IN FLIGHT —
  `docs/plans/evidence-law-journal.md` (Wall 005/006), edits in
  `golem/kernel/anatomy/realize/{allocation,bifurcation}.py`,
  `golem/kernel/engine/{algebra,compile}.py`, `rehearsal/formation-trial/`.
  Its Wall 006 ruling touches `session/protocol.py::AuthoringEvidence.accepted`
  — a second sibling on the shared file.
- Lane A territory (`sheaf/`, `blame/`, vascular gluing payloads in
  `anatomy/balance.py` + `anatomy/graph.py`, `anatomy/project.py` rendering):
  UNTOUCHED. A14/A15/A16 are UNSTARTED. This journal is the first Lane A
  artifact.

**The wall (FRICTION-001, `rehearsal/wyvern-trial/FRICTION.md` 001, verdict-02):**
`vascular.FailedGluing @ closed-vascular-field` with
`reason="NonConvergentBalance @closed-vascular-balance: solver_info=10920"`,
`binding_constraint=null`, `help.operations=[]`. R13: a numerical code is not
an address; the kernel owns `kernel/blame` and a gluing failure that does not
invoke it withholds evidence.

## 001 — recon: where the evidence dies

The collapse chain, verified by reading every link:

1. `sheaf/solve.py::_solve_free_operator` — cg returns `info != 0`; the site
   has `problem`, `free_operator`, `free_source`, `config`, and the stop
   iterate in scope, and mints `NonConvergentBalanceObstruction(problem_id,
   solver_info)` — two fields, everything else discarded. (Directed-transport
   path uses splu + refinement, `solver_info` always 0 — untouched.)
2. `anatomy/balance.py::_ClosedVascularInterpreter.lift` — flattens every
   sheaf obstruction through `sheaf.render_obstruction` into
   `VascularGluingObstruction("closed-vascular-field", <string>)`. The
   carrier (graph + node_index) is IN SCOPE here — cell→node→region→edge
   projection is computable at this seam and nowhere above it.
3. `anatomy/project.py` — projects to `{"kind": "FailedGluing", address,
   reason}`; `session/protocol.py::_explicit_projected_diagnostic` renders
   `vascular.FailedGluing` with `help_operations=()` (default) and
   `_vascular_violation_measurement` has no gluing case, so no margin exists
   and `binding_constraint` is null. FRICTION-001 reproduced exactly.
4. `blame/interrogate.py::well_posedness_disposition` —
   `VascularGluingObstruction` is `DeclinedDisposition(GLOBAL_SOLVE)`:
   the organ is invoked (`--blame`) and declines empty. That is R13's
   "withholding evidence it could have produced". No test pins this
   disposition for gluing (test_blame.py is clean).

Pre-solve topology is already proven: `sheaf/validate.py` rejects unanchored
components (`UnanchoredBalanceComponentObstruction`, csgraph
connected_components) before the solve. So on NonConvergent the topology
channel reports what is PROVEN plus the weakest conductance bridges
(near-separations), not a disconnection claim.

Key numeric fact: `SealedSolverConfig.maximum_iteration_factor=10`, and
FRICTION-001's `solver_info=10920` is exactly `1092 cells * 10` — the wyvern
failure is budget exhaustion under cg with default `relative_tolerance=1e-13`
on a Poiseuille network (conductance ~ r^4 spread). scipy cg `info > 0` is
the iteration count at stop, and cg still returns the stop iterate — so the
stop residual `r = b - A x_stop` is one matvec away.

Authorable discharge knobs (the lawful moves A15 may name), from
`specs/cinderwake.json` + `anatomy/decode.py`:
`anatomy/overall/circulation/{exchange_beds/<i>/demand,
exchange_beds/<i>/tissue_envelope/minimum_radius, carrier_radius_scale,
distance_decay, pump_region_id}`. Conditioning is driven by the conductance
SPREAD, so `carrier_radius_scale` (uniform scale) does NOT discharge it;
`distance_decay` (spread compressor), `minimum_radius` floors (raise the
small end), and demand/routing (network shape) do.

## 002 — mechanism design (pre-implementation, recorded once)

One owner per surface; strictly additive; accepted paths byte-identical and
touch zero new work (interrogation is computed only on the failure branch).

- **sheaf (owns the algebra):** new `sheaf/interrogate.py` —
  `BalanceInterrogation` = four typed hypothesis witnesses:
  `topology` (component count + weakest conductance bridges),
  `conditioning` (Gershgorin row-sum bound, diagonal spread, worst-K cells),
  `demand` (top-K stop-residual cells + max residual vs tolerance),
  `budget` (cap, iterations used, rtol, exhausted flag).
  `NonConvergentBalanceObstruction` gains `interrogation: ... | None = None`.
  `render_obstruction` unchanged — every existing pin and historical verdict
  renders byte-identical. solve.py threads `free_cells` so witnesses name
  global cell ids.
- **anatomy (owns the vascular lowering):** `VascularGluingObstruction` gains
  `failing_segments=()` and `gluing_interrogation=None`, both defaulted — the
  other seven minting sites (calibration, pump-outlet, invalid delivery,
  materialize, parity x3, allocation) are byte-untouched. The lift projects
  cell ids → node ids → `VascularNode.region_id`, and suspect interfaces
  (which ARE edge ids in this lowering) → `vascular_edge_geometry` segments.
- **localization (one existing owner):** `_localized_rejection`
  (realize/allocation.py) gains `VascularGluingObstruction` in its geometric
  union — the wave-3 mechanism that maps segments to host part/bone
  addresses. Sibling (evidence-law) has 9 lines in this file; my edit is the
  union member only, no shared logic touched.
- **projection:** `anatomy/project.py` FailedGluing case adds
  `interrogation` + `failing_segments` keys ONLY when the payload is present.
- **blame consumption:** `interrogate.py` — a gluing obstruction carrying a
  decomposition is no longer an empty decline: the BlameEntry names the
  projected region/interface addresses. Disposition vocabulary gains an
  honest variant (MUS was not run; a solve-level decomposition was).
- **session rendering (shared file; Lane C + evidence-law diffs read
  first):** `_vascular_diagnostics` threads the document and, for gluing
  with payload, emits `help.operations` (A15) — candidate `SetOp`s on the
  discharge knobs above, each compile-checked against the document exactly
  like `_gradient_operation` does; only lawful ops survive.
  `_vascular_authoring_addresses` + `_vascular_violation_measurement` gain
  the gluing case (A16): margin predicate "free-cell residual within sealed
  tolerance at solver stop", required=tolerance, observed=max stop residual
  → VIOLATED → `binding_constraint` non-null.

Wall entries land below as hit.

## 003 — A14/A15/A16 landed

- `sheaf/solve.py` now computes `BalanceInterrogation` only when cg returns
  nonzero info. Accepted solves execute the same branch and allocate nothing
  new. Failure cost is one sparse connected-components pass, one COO view,
  one sparse row-sum, one stop-residual matvec, and bounded (K=3) witness
  selection. `render_obstruction` deliberately ignores the optional payload:
  `NonConvergentBalance @<id>: solver_info=<n>` is byte-identical.
- Engineered 40-cell alternating-conductance law: cg exhausts exactly 40/40
  iterations; topology reports one validated component + three weakest
  interfaces; local conductance ratios identify the 1e12 sections;
  stop residual 3.265e-10 is named against 1e-10; budget says exhausted.
  A numerical correction landed during smoke: diagonal spread uses the
  actual nonempty diagonal min/max (no synthetic `initial=1`).
- `_ClosedVascularInterpreter.lift` projects global cell ids to `node_id` +
  `region_id`, interface ids to real vascular edges, and bottlenecks to
  `VascularSegmentGeometry`. `_localized_rejection` reuses the sole wave-3
  host part/bone locator; no second addressing convention.
- FailedGluing projection is additive only when interrogation exists:
  legacy `{kind,address,reason}` remains exact. Enriched witnesses carry four
  named hypotheses, section/edge addresses, localized host addresses, and
  residual predicate/required/observed.
- Blame gains `decomposed: global-solve`: honest distinction from both
  declaration-level `localized` MUS and empty `declined: global-solve`.
  It does not run speculative deletion probes; it consumes the solve-level
  decomposition and names the projected suspect addresses.
- Cinderwake's real authored document proves A15 operation addresses compile:
  `distance_decay 0.86 -> 0.93` (re-route/conditioning) and
  `exchange_beds[3].demand 0.03 -> 0.01` (demand reduction derived from
  tolerance/residual ratio). Plain gluing emits no ceremonial op; enriched
  gluing is non-empty.
- A16 emits one violated margin: maximum stop residual required 1e-10,
  observed 3e-10, margin -2e-10, authoring addresses
  `[anatomy/regions/wing_tips, host part, host bone]`. `_glue_evidence`
  activates it, so `binding_constraint` is non-null.
- Shared-file discipline held: the first protocol.py patch was rejected stale
  while Lane C/evidence-law moved the file; the edit was re-read and landed
  as per-function hunks. allocation.py sibling diff was at the entry seam;
  Lane A changed only the localization union member.

Semantic laws: four new tests (sheaf failure decomposition, anatomy lift
projection, session help+binding+legacy byte identity, blame consumption) —
4 passed in 0.41s. Focused Lane A union (`test_sheaf`, `test_blame`,
`test_session_diagnostic_semantics`, `test_vascular_relaxation`) — 73 passed.
Full union and cold oracle receipts land in the next entry.

## 004 — review repairs and final receipts

Four independent closeout reviews ran against the settled implementation.
Python correctness found no blocker. The material findings from simplicity,
performance, and architecture were all repaired rather than waved away:

- `_gluing_operations` now uses one typed mapping-path traversal and one
  witness-region extractor instead of three parallel JSON walks. Demand help
  is bounded to at most three witness-region beds, ranked by authored demand;
  when no witness region owns a bed, one maximum-demand bed is the principled
  fallback. Every candidate still passes decode + compile against the real
  document. This removes the review's unbounded help list and O(beds^2)
  compile-copy fanout while preserving the non-empty-help law.
- `_gluing_obstruction` binds `node_by_id` once and threads it through
  bottleneck/segment projection; the failure path no longer rebuilds the
  O(nodes) dictionary up to fifteen times.
- The solve-rejection branch in `realize/allocation.py` now passes through
  `_localized_rejection`, closing the production host-address bypass.
  The wiring law is pinned by
  `test_solver_rejection_passes_through_host_address_localization`.
- Session margins now consume anatomy's projected `suspect_addresses` instead
  of re-deriving region addresses. A regionless witness is pinned to
  `closed-vascular-field/nodes/<node_id>` in both blame and the active margin;
  there is one address owner again.

Final Lane A semantic set (including production localization, solver
decomposition, anatomy lift, blame consumption, matched/fallback lawful help,
regionless address identity, binding margin, and legacy byte identity):
5 passed in 0.42s. Final focused union (`test_sheaf`, `test_blame`,
`test_session_diagnostic_semantics`, `test_vascular_relaxation`):
74 passed in 27.58s. The changed source/test slice also passed `py_compile`.

Final full non-conformance union: **1061 passed / 3 failed** in 180.60s.
All three failures are the concurrent skin-formation lane's in-flight
expectation/API drift in `tests/test_skin_formation.py`
(`fully_covered_knight...`, parameter `root-absent`, and
`multiple_roots_without_tie_breaking`); no Lane A test failed.

Cold oracle receipts on the actual witness:

- `golem check specs/cinderwake.json --all`: ANOMALY 0, vascular ACCEPTED,
  residual 3.851e-13, balance 4.236e-13, assembly ACCEPTED.
- The exact FRICTION-001 session command,
  `golem session rehearsal/wyvern-trial/base.json specs/cinderwake.json --all`,
  is now accepted with 58 expected / 0 unexpected diagnostics.
- `golem look specs/cinderwake.json --out
  /tmp/golem-lane-a-cinderwake-look-0723` produced front/side/top PNGs.

The repaired current Cinderwake no longer reproduces the historical
`NonConvergentBalance`; therefore no counterfeit `verdict-05` was appended.
The null-to-rich rejection proof rests honestly on the engineered
non-convergence law through the real solve/lift/blame/session layers plus
compile-checking against Cinderwake's real authored document. Architectural
closure audit verdict: accepted — solve evidence has one sheaf owner, domain
projection/localization one anatomy owner, blame consumes rather than
rebuilds, session derives help and the binding margin, accepted paths perform
zero new work, and no parallel explainer or address system remains.
