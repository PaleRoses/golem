# Authoring-surface evaluation corpus

This directory is GOLEM's append-only authoring-surface ratchet. Files under `receipts/` are authoritative content-addressed receipts. A corpus summary is a derived view and is never stored here.

## Recording a trial

The generic version-one input is independent of creature dialect, spec path, renderer, and acceptance implementation:

```json
{
  "schema_version": 1,
  "trial_id": "trial-wyrm-001",
  "creature_id": "wyrm",
  "transactions": [
    {
      "transaction_id": "txn-1",
      "status": "obstructed",
      "obstruction_codes": ["ClearanceObstruction"],
      "tokens": 120
    },
    {
      "transaction_id": "txn-2",
      "status": "accepted",
      "obstruction_codes": [],
      "tokens": 80
    }
  ]
}
```

`transaction_id`, `status`, and `obstruction_codes` are required. `tokens` is optional and must be a non-negative integer when present. An accepted transaction cannot retain obstruction codes; an obstructed transaction must carry at least one. The ordered array is the trial order. The first accepted transaction is authoritative for acceptance economics; later polish transactions remain in the normalized trial but do not inflate rounds or tokens through acceptance.

Verdict-sequence sources use normalized trial schema version two. They distinguish a local `surface_accepted` checkpoint from the terminal trial-wide `accepted` verdict, and carry typed acceptance-round evidence. A reported round count must equal the authoritative verdict's `txn`; an incomplete ledger makes rounds unavailable with the omission stated verbatim rather than turning an approximation into an integer.

Record and summarize with the eval-plane module directly:

    python -m golem.evals.authoring_corpus record-json TRIAL.json CORPUS_DIRECTORY
    python -m golem.evals.authoring_corpus record-session RECEIPTS.txt TRIAL_ID CREATURE_ID CORPUS_DIRECTORY
    python -m golem.evals.authoring_corpus summary CORPUS_DIRECTORY

Multi-artifact verdict sequences are recorded through `record_authoring_verdict_sequence`. Its ordered verdict inputs are decoded; contextual artifacts such as `FRICTION.md` are hashed into the same receipt without being mistaken for machine transactions.

The session adapter accepts GOLEM's deterministic transaction receipt format. It reconstructs the complete active contract-obstruction section from `new` and `cleared` deltas, requires blocker counts to agree, and treats a zero-blocker result as acceptance only when at least one contract clause passes. A zero-clause revision probe is not an accepted creature by vacuity.

## Metric schema

Each measured receipt contains the normalized trial and these derived fields:

- `rounds_to_acceptance`: `{"status":"available","value":N}` for the one-based index of the first accepted transaction, or for a verdict sequence the exact terminal verdict `txn` named by reported evidence. An incomplete historical prefix produces `incomplete_transaction_transcript`; no accepted transaction produces `no_accepted_transaction`.
- `obstruction_repair`: one record per obstruction code. `observed_transactions` counts transaction results containing that active code. A followed occurrence succeeds when the immediately following transaction no longer contains the code. `success_rate` is the exact integer rational `numerator / denominator`; an occurrence on the final analyzed transaction is `unfollowed_transactions`, not a counterfeit failure.
- `tokens_per_accepted_creature`: the sum of transaction token observations through first acceptance. It is unavailable when there is no acceptance or any transaction in that prefix lacks tokens.
- `transaction_count`: all normalized transactions in the source.
- `analyzed_transaction_count`: the prefix through first acceptance, or the entire sequence when acceptance never occurs.

Unavailable metrics are records with `status`, stable `code`, and `detail`; they are never `null` or zero-filled. Metric payloads contain no wall-clock fields. Repair rates use integers rather than floating-point percentages.

## Receipt schema and append law

A receipt has `schema_version`, `source`, and `result`. `source` contains a closed source-format tag plus one or more repository-relative artifact paths and SHA-256 hashes. A measured result contains the normalized trial and recomputable metrics. An unparseable result contains typed decode obstructions. Loading a measured receipt decodes the normalized trial and recomputes its metrics; mismatches obstruct the corpus.

The filename is the SHA-256 of the complete stable JSON receipt bytes. Recording uses exclusive creation. Repeating the same input returns the existing receipt; different bytes at the same address produce `receipt_collision` and are never overwritten. Multiple measured receipts claiming one `trial_id` obstruct corpus gluing. If later decoder support can measure a source that previously produced an unparseable receipt, the new measured receipt is appended and dominates that source's obsolete unparseable view without deleting history.

## Initial source audit

The initial corpus deliberately distinguishes transaction evidence from final artifacts and prose.

| Receipt | Source bundle | Result |
|---|---|---|
| `fad8dd57ebd421fa20a4dce4a1d62693a8a99feb54d8509e0d6034118e6c2232` | `golden/session_biped_receipts.txt` | Measured: biped first accepts at transaction 10; 469 estimated receipt tokens through acceptance; `env_heads_tall` repairs 1/7 and `env_balanced` repairs 1/9. |
| `6d6580f796f62760c39b2f35a57091b0072734f9188541c352f27103da722d44` | Scree Maiden `verdict-00..06` plus `FRICTION.md` | Measured verdict suffix: `vascular.CapsuleEscape` repairs 1/2 and `assembly.EmptySurfaceConduitObstruction` repairs 1/1. Total rounds are unavailable because the roughly ten pre-session checks have no ordered verdict ledger; tokens are absent. |
| `faaa08043e304545716474cd7c4c50c31db34a1c4b95d3bb3187debf771011d0` | Verdigris `verdict-01..08` plus `FRICTION.md` | Measured: authoritative acceptance at transaction 11; `assembly.EmptySurfaceConduitObstruction` repairs 1/2, `assembly.SolidIntegrityObstruction` repairs 1/1, and `vascular.Intersection` repairs 1/1; tokens are absent. |
| `3c106bbeed3dfc414e14855ea55374b707560b80a5ae66b5787d4d343513bf87` | `golden/session_knight_receipts.txt` | Unparseable as creature acceptance: all four transactions have zero contract clauses, so acceptance has no authority. |
| `b11e9b73337f99c6f084f0b734c1929e2909f930d7cf560683ea7d821d3ce2d5` | `pilots/judge.py`, `control_creature.py`, `golem_v1.json` through `golem_v4.json` | Unparseable: canonical snapshots plus a final geometry judge, no transaction verdict/obstruction/token history. |
| `965afc7b998340be54e3f9c27ffeed6f787d00029d1fa3afe73f0dadec8e189e` | Bulwark GOLEM rehearsal stages | Unparseable: snapshots, timings, metrics, and prose round bounds are not a transaction ledger. |
| `9fd6321064dba79c51bb66aa34d85f0a86009b057093a9396775b77304374abe` | Bulwark Blender rehearsal stages | Unparseable: mesh receipts, timings, and prose failures lack ordered authoring transactions and token data. |
| `a56f7015897aebdb2564e9422aa041471d211147f04c1748ef2a848d19fc8239` | Climber arena builder and four stage specs | Unparseable: no transaction outcomes, acceptance receipt, obstruction codes, or transcript. |
| `a4fd8b28da5fe0f3327f08fe41434b6fc5a7d02ab6778c20b81f22db6c4257b2` | Initial GOLEM knight summary/spec/mesh receipt | Unparseable: six render rounds are reported only in aggregate. |
| `f7b0700e76ca24d8da6aa733c652c6b82acaa035820362d909b891bd2aab0e3b` | Initial Blender knight summary/mesh receipt | Unparseable: six render rounds are reported only in aggregate. |
| `d7607b174b88e226f1381616e709eac3908dd008b0c87e6e944403dcdb355e17` | Session-knight README, operator notes, final ops, final spec | Unparseable: the ops file is a flattened final script and prose estimates are not transaction receipts. |
| `893d0e680c88aaeae4a2873f1d888eac09fd409709168a04afcf48851c832a0a` | Harness probe A report/final | Unparseable: prose run ledger, no machine acceptance/obstruction/token stream. |
| `f46e95bc70e8009dcc8347d42d62f703eac3828880969c15b9b3799fa317ab72` | Harness probe B report/final | Unparseable: prose ledger contains a raw exception string rather than a stable obstruction code and no tokens. |
| `b9b0f675fff479b92eaec6155985e2bf6129197ccf29fafbdc7a74e64f1d56c9` | Reforge README and final v2 spec | Unparseable: seven rounds are summarized, not preserved as ordered verdicts with obstruction and token observations. |
| `68d02d7188d1e072b05794d722116e41e313d4e292fe3a12caec2af2cfdb779c` | Phase -1 manifest and protocol | Unparseable by design: preregistered and explicitly not run, hence specification rather than trial data. |

Binary meshes and renders are derived views. They add no transaction evidence and are not duplicated into source bundles merely to make the hashes look busier.
