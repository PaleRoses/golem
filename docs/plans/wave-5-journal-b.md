# WAVE 5 JOURNAL — Lane B (sensed interfaces become declarable)

Lane: B14, B15, B16 of `docs/plans/wave-5-amendment-docket.md`. Rulings R10-R13
govern; R-B mechanism choice is mine. Contemporaneous: wall entries land when
hit. Direct interpreter only: `.venv/bin/python`.

## 000 — orientation, before surgery

- The wall (FRICTION-002), reproduced cold before any edit: cinderwake with
  `contract.attach = [["wing_forearm", "cinderwake_wing_membrane"], ...]`
  compiles to `CompiledBody` carrying
  `bad_contract_attach: unknown part ids ['cinderwake_wing_membrane']` at
  `spec/contract/attach/0` and `/1` — and the refused pairs are silently
  dropped from the intent block.
- The schism is exact: `_derive_attach` (kernel/body/intent.py) already mints
  `[anchor_primary_flesh, web_id]` pairs into the same intent channel, and
  proprio's `_instance_list`/`_part_data` fold `graph["webs"]` into the sensed
  part topology, so anomaly addresses name web ids. `_authored_attach` builds
  `known` from the emitted `parts` argument only — webs never enter it.
- Mechanism choice under R-B: option 1 — `contract.attach` (and the sibling
  `midline` channel) admit web ids as endpoints. Grounds: web ids are authored
  vocabulary (`spec/webs[].id`, strict decode in webs.py, collision-checked
  against part ids by `WebPartCollisionObstruction`), not a derived namespace;
  `self._web_anchors` is the single decoded source, populated at
  compile.py before `build_intent` runs. Option 2 (ancestor decoration of
  anomaly addresses) would add a second address shape to every proprio
  consumer for zero law gain — the id is already authorable, the validator
  simply refused to know it. One owner per surface: webs.py owns the web
  namespace; intent.py consumes the same `_web_anchors` record it already
  uses for `_derive_attach`. No third id namespace.
- Menagerie scan: cinderwake is the only web-bearing spec; its contract was
  scrubbed of membrane ids during the trial (FRICTION-002's workaround), so
  widening the declarable set changes zero existing accepted outputs.

## 001 — B14/B15 landed in the single intent owner

- `_declarable_part_ids(parts)` is now the one writable part-id set consumed
  by both `_authored_attach` and `_authored_midline`: emitted parts plus the
  decoded spanning-web ids already held in `self._web_anchors`. No derived
  third namespace, no `senses/proprio` edit, no `specs/cinderwake.json` edit.
- `_primary_flesh_ids` and `_web_anchor_flesh` preserve the existing derived
  attach product while making the anchor-flesh endpoints available to the
  rejection surface. The derived anchor-to-web pairs remain the same law.
- B15's `bad_contract_attach` detail now names up to three closest declarable
  ids and, when the closest legal id is a web, the authored anchor flesh from
  which that sensed interface derives. The witness addresses remain
  `spec/contract/attach/<index>`.
- Synthetic law tests cover the original wall without touching Lane C's
  pinned witness spec: a non-anchor finger can declare `wing.membrane` in
  `contract.attach`; a mirrored membrane can be declared in `contract.midline`;
  and `wing.membran` rejects with `closest declarable wing.membrane` plus
  `anchor flesh [finger1, finger2]`.

## 002 — B16 menagerie closure

- `test_tracked_menagerie_anomaly_addresses_are_declarable` enumerates every
  tracked `specs/*.json` body/0.3 document — 40 body specs at this revision —
  compiles each one, builds senses, detects every anomaly, expands both the
  structured `pair` and rendered `addr` address space, strips only instance
  suffixes `.L`/`.R`, and requires every emitted part id to be contract
  declarable.
- The same test requires any emitted web id to carry authored skeleton anchor
  rows in the compile receipt. Cinderwake is explicitly pinned: its
  non-carrier `cinderwake_wing_membrane` appears in three expected
  `deep_burial` anomaly addresses and is now inside the declarable set, with
  receipt anchors `skeleton/wing_fan_leading`,
  `skeleton/wing_fan_middle`, and `skeleton/wing_fan_trailing`.
- Runtime: 1.52s for the full menagerie law test, below the R-F 10-second
  ceiling.

## Receipts

- Focused web/contract union: `tests/test_webs.py tests/test_cli_contract.py`
  — 75 passed.
- Proprio classifier union: `tests/test_proprio.py` — 30 passed.
- Combined Lane B union: 105 passed.
- Broad non-slow union: 1034 passed, 13 deselected, 1 failed. The failure is
  Lane C's in-flight exception-plane conversion, not this lane:
  `tests/test_session_protocol.py::test_malformed_file_edits_reach_the_journal_and_body_compiler[blend-bad]`
  still expects a `body.*` code while the partially landed Lane C tree returns
  `engine.fault`.
- Cold Cinderwake check (`GOLEM_CIRCUIT_CACHE_COLD=1`): 35 parts, 19 mirrored,
  54 instances; 0/2 contract failures; `ANOMALY 0`; vascular accepted
  (1090 nodes, 1346 edges, 256 terminal pairs, residual 3.851e-13, balance
  4.236e-13); assembly accepted. The checked-in Cinderwake spec was not
  modified.

## 003 — severance continuation (same lane, new session)

The 15:0x session was severed by an infrastructure restart after the Receipts
block above was written; this session (15:15+) verified the inherited state
rather than re-deriving it, and added the end-to-end oracle evidence for the
original wall that the synthetic law tests only approximate.

- Inherited state verified green, unchanged: `tests/test_webs.py` 44 passed
  (1.83s), `tests/test_cli_contract.py` + `tests/test_proprio.py` 61 passed,
  full union 1047 passed / 1 failed — the same Lane C in-flight
  `engine.fault` conversion failure recorded above, still not this lane.
- FRICTION-002 end-to-end, cold (`GOLEM_CIRCUIT_CACHE_COLD=1`), probe specs
  in /tmp only: the pinned cinderwake plus
  `[["wing_forearm","cinderwake_wing_membrane"],
   ["wing_digit_lead","cinderwake_wing_membrane"]]` in `contract.attach`
  checks exit 0 with NO `bad_contract_attach` — the sensed web id is now
  writable vocabulary. The typo probe (`cinderwake_wing_membran`) rejects
  exit 1 at `spec/contract/attach/45` naming
  `closest declarable cinderwake_wing_membrane` and
  `web 'cinderwake_wing_membrane' derives from anchor flesh
   [wing_fan_leading_strut, wing_fan_middle_strut, wing_fan_trailing_strut]`
  (B15's neighborhood, oracle-rendered).
- Correction, recorded so it is never misread: the positive probe reports
  `ANOMALY 2` — two `missing_fusion @ wing_*~cinderwake_wing_membrane`
  (gaps 0.069/0.061) — and those are UNEXPECTED anomalies
  (`_missing_fusion_anomaly` never sets `expected`), not expected ones.
  `ACCEPTED` rode through only because acceptance is still severity-only;
  the sibling evidence-law lane's Wall 006 repair (accepted requires
  `unexpected == 0`) is in flight in this same tree. The declaration is
  honest: the trial's workaround geometry genuinely does not fuse the
  membrane to those bones, so the declared intent is reported unrealized.
  The membrane pairs were therefore NOT re-added to `specs/cinderwake.json`
  (strictly additive is false under the tightened acceptance law); restoring
  the real declaration requires restoring fusion geometry — trial territory,
  not Lane B. Probes remain in /tmp as evidence; the pinned spec is
  byte-untouched.
- Lane B is complete: B14 (attach + midline admit web ids), B15 (rejection
  enumerates the legal neighborhood with anchor ancestry), B16 (menagerie
  address-space closure law) all implemented, tested, and oracle-verified.

## Addendum — duplicate-dispatch collision record (15:37, third session)

A third kimi-code/k3 session (severance-recovery prompt, started ~15:16)
independently verified this lane before discovering the live 15:15+ sibling's
001–003 above, and briefly wedged its own duplicate 001/Lane-status block
between 000 and 001; this addendum replaces it, restoring a single record.
Its independent receipts, recorded once and then yielded: full living union
`pytest tests/ -q` 1047 passed with one transient `.golem-cache/field`
FileNotFoundError flake — since explained as concurrent pytest runs sharing
the cache directory, not a semantic failure (passes in isolation). Its
mechanism audit agreed with 001–003 on every point (`_declarable_part_ids`
ownership, `_web_anchor_flesh` single minting site, byte-identical derived
pairs). Lane B close-out stands as written by the 15:15+ sibling; this
session makes no further edits to this lane's files or this journal.
