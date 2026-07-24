# Wave 5 — Authoring-Surface Amendment Docket (the Cinderwake harvest)

Harvested from authoring trial 3 (Cinderwake, true bipedal wyvern — ACCEPTED,
`env_balanced +0.027` against `>= +0.020`, 58 expected / 0 unexpected
diagnostics). Receipts: `rehearsal/wyvern-trial/FRICTION.md` (4 entries),
`rehearsal/wyvern-trial/verdict-01..04.json` (append-only),
`rehearsal/wyvern-trial/check-accepted.txt`. The wave-3 vocabulary held:
webs, `handle` roles, `non_carrier` flesh, carves, and material classes were
all exercised by a competent author and none of them wounded him. The three
walls below are what did.

Every element is a defect the surface inflicted on that author. Lanes are
file-disjoint and dispatch in parallel. Fable's lane is the rulings inline
here, merge review, and union verification — no fable execution.

**Sequencing (absolute):** Wave 5 dispatches only after the surface-formation
campaign (`docs/plans/surface-formation-adoption-docket.md`) has landed, passed
Fable's merge review, and Cinderwake has been re-verified cold. See R10. The
known-open A13d wall (web-blind vasculature) remains OUT of this wave — it is
its own future campaign, not an amendment.

---

## Standing rulings (fable; numbering continues the house series)

**R10 — One surgeon per kernel.** Wall 003 was manufactured, not discovered:
the canonical session accepted while a cold check rejected because a concurrent
campaign had half-rewritten `compose_union` under the running trial. Authoring
trials may run during kernel surgery only because R-A/R-E fence them; a second
*kernel-mutating* campaign may not. Henceforth kernel surgeries serialize, and
every campaign merge closes with an oracle-agreement gate: cold `session`,
cold `check`, and `look` on a pinned witness spec must descend to the same
verdict before the next surgery opens.

**R11 — Oracles are total functions.** A naked `TypeError` reached the author
through `check` as a catch-all `unexpected_compile_exception`, and `look`
crashed outright on the same call. No CLI oracle may ever surface a raw
exception or an untyped catch-all. Every engine exception at an oracle
boundary projects to one typed obstruction, identical across all three
oracles. `look` refuses in type; it does not die in traceback.

**R12 — Diagnostics speak the authorable language.** Proprioception named
`cinderwake_wing_membrane` in an actionable `unintended_fusion` address;
`contract.attach` then rejected that exact id as unknown. A diagnostic that
names an interface the author cannot declare is itself a defect. Law: every
part id appearing in an actionable diagnostic is either declarable in the
contract vocabulary or accompanied by its authored ancestors (the anchors from
which it was derived), so the author always holds a writable address.

**R13 — Gluing failures carry blame, not solver codes.** `FailedGluing @
closed-vascular-field` collapsed to `NonConvergentBalance: solver_info=10920`
with `binding_constraint=null` and `help.operations=[]`. A numerical code is
not an address (per R2, verdicts must name the door). The kernel already owns
an interrogation organ (`kernel/blame`, approximate-MUS); a gluing failure
that does not invoke it is withholding evidence it could have produced.

---

## Lane A — Gluing gets an authoring address (codex, gpt-5.6-sol, ULTRA)

Territory: `golem/kernel/sheaf/` balance solve verdict path, `golem/kernel/blame/`
consumption, vascular gluing obstruction payloads and their session rendering.
Does not touch CLI exception plumbing (Lane C) or senses/contract vocabulary
(Lane B).

- **A14. NonConvergentBalance is interrogated, never reported raw.** (Wyvern
  FRICTION-001.) On non-convergence, run blame interrogation over the failed
  balance problem and decompose into typed hypotheses, each with a witness:
  topology (circuit/disconnection witness), conditioning (bound plus the worst
  local sections), demand (the infeasible demand subset), budget (iteration
  and tolerance state at stop). Payload carries suspect sections and overlaps
  projected to body addresses (segments, hosts, spec addresses) per R2.
- **A15. help.operations populated from the hypothesis.** Each hypothesis maps
  to the lawful authoring moves that can discharge it (re-route, re-anchor,
  demand reduction, topology repair). An empty operations list on a rejection
  is treated as a rendering defect.
- **A16. binding_constraint honest under gluing rejection.** Extends wave-3
  A2: the gluing channel emits its violated obligation rather than `null`.

## Lane B — Sensed interfaces become declarable (Kimi via omp, kimi-code/k3)

Territory: `golem/senses/proprio/` anomaly address construction,
`golem/contract/` attach vocabulary validation and its diagnostics. Does not
touch the balance solver (Lane A) or oracle exception boundaries (Lane C).
Kimi built the wave-3 vocabulary; these are her walls to mend.

- **B14. Anomaly addresses closed under authorable vocabulary.** (Wyvern
  FRICTION-002.) Ruling R12 fixes the law; the mechanism is Kimi's choice
  under R-B (one owner per surface): either `contract.attach` admits derived
  interface ids (webs) as endpoints, or derived-part anomaly addresses carry
  their authored anchor ancestors and the attach rejection names them as the
  writable alternative. No third id namespace.
- **B15. bad_contract_attach enumerates the legal neighborhood.** Extends
  wave-3 A6: rejection of an unknown id lists the declarable ids adjacent to
  the offending one (the anchors and parts the sensed interface derives from),
  not merely the unknown-id complaint.
- **B16. Law test.** Every part id emitted in any anomaly of the full test
  menagerie is provably declarable-or-ancestored; the test enumerates the
  emitted address space, not a hand-picked sample.

## Lane C — Total oracles (codex, gpt-5.6-sol, xhigh)

Territory: `golem/cli/check.py`, `golem/cli/look.py`, session CLI exception
boundaries, the `unexpected_compile_exception` classifier. Does not touch
verdict payload shapes (Lane A) or senses/contract (Lane B).

- **C14. Typed EngineFault projection.** (Wyvern FRICTION-003.) Engine
  exceptions crossing an oracle boundary become one typed obstruction
  (exception kind, message digest, seam address), identical in `session`,
  `check`, and `look`. The catch-all `unexpected_compile_exception` is
  retired in favor of the typed projection; `look` renders a typed refusal
  with whatever partial evidence is lawful, never a traceback.
- **C15. Poisoned-engine agreement gate.** A conformance harness patches one
  compile seam to raise (preserving the existing mock.patch seam discipline)
  and asserts all three oracles return the same typed verdict. This is
  wave-3 B1 extended to the exception plane, and it is the regression net
  that would have caught wall 003 in minutes.
- **C16. Cinderwake as witness spec.** `specs/cinderwake.json` joins the
  oracle-agreement gate: cold `session`, cold `check`, and `look` must agree
  on it, and its `rehearsal/wyvern-trial/verdict-*.json` lineage remains
  append-only. The wyvern is now a permanent inhabitant of the test
  menagerie, not a retired trial.

---

## Verification (fable, at merge)

Full union `.venv/bin/python -m pytest tests/ -q --ignore=tests/conformance`
against the post-surface-formation baseline; all byte-exactness pins intact;
the poisoned-engine agreement gate green; Cinderwake replayed cold through all
three oracles with identical verdicts; every new obstruction variant exercised
by a semantic test (R-F: never weaken, none over 10 seconds). R-G: no git —
the tree stays dirty until the operator says otherwise.
