# FRICTION JOURNAL — Ashfall wolf authoring trial

Trial 5, attempt 2. Authoring from zero under the blind-authoring law.
Creature: Ashfall, a cursorial quadruped wolf with deep chest, tucked waist,
long muzzle, four digitigrade legs, and straight tail intent. No quarantined
specification or prior creature design was consulted.

Verdicts are append-only as `verdict-NN.json`. The canonical oracle is
`session`: accepted status AND `diagnostics_summary.unexpected == 0`.

## 000 — setup

Inherited scars applied silently: direct `.venv/bin/python`, half-extents,
parent-frame rest directions, fresh base rotation, closed appearance ids,
exact final contacts, and check/session comparison.

Protocol breach note: I initially authored the complete child flesh draft
before probing the chains, then stripped it, probed the bare skeleton, and
re-added carrier flesh. The tail was also first added with flesh before its
tail-chain probe, then corrected to a perfused tail and re-probed. The
commissioned sequence was not followed on the first pass; the reset now uses
bare-chain frame receipts before each current flesh decision.

Authored a new wolf spec at `specs/ashfall.json` with original ids and
proportions. First probe follows.

## 001 — formed skin requires every declared region to be referenced

Command: `GOLEM_CIRCUIT_CACHE_COLD=1 .venv/bin/python -m golem check specs/ashfall.json --all`

Verdict fragment: `OBSTRUCTIONS 1 [SkinFormationAnatomyObstruction] anatomy/integument_layers: skin formation anatomy rejected (UnreferencedAnatomyRegionObstruction, UnreferencedAnatomyRegionObstruction)`.

The formed-skin gate wrapped the underlying region-reference errors and only reported their count. I had declared useful descriptive limb regions, but the vascular contract only references the pump and exchange-bed regions; the remaining limb regions are unreferenced and therefore illegal. I will keep the anatomy region set to the pump plus exchange hosts and give each of those one uniform formed layer, rather than inventing circulation beds for descriptive regions.


## 002 — canonical stop: inherited reserved-lane wall remains

Command: `GOLEM_CIRCUIT_CACHE_COLD=1 .venv/bin/python -m golem session specs/ashfall.json specs/ashfall.json --all`

Verdict fragment: `status=rejected`, `diagnostics_summary.unexpected=16`; hard errors were `ashfall_head_region ReservedLaneClearance observed=-3.83854740504e-05`, `ashfall_hind_region WallContainment observed=5.13461278056e-05`, and `ashfall_tail_region ReservedLaneClearance observed=-4.03224511053e-05`.

The session receipt now names the failing segments and their authored torso/pelvic addresses, but the two reserved-lane failures are the inherited shared-midline topology wall already recorded by the scar journals; tiny lateral departures moved the margin but did not clear it, while larger departures made the solve pathological and damaged the silhouette. The hind return corridor is positive on this pass, so its containment is not the blocker. The required formed-skin stage is never reached because vascular realization rejects first. I am stopping rather than declaring a non-vascular tail, inventing a prop, or copying another creature's route. This is an honest blind-trial failure on the amended surface, not an imitation.

## 003 — recorded lane cures move the wall but do not yet clear it

Command: `GOLEM_CIRCUIT_CACHE_COLD=1 .venv/bin/python -m golem session rehearsal/wolf-trial/base.json specs/ashfall.json --all` (saved as `verdict-06.json`).

Verdict fragment: `status=rejected`, `ashfall_head_region ReservedLaneClearance observed=-3.83854740504e-05`, `ashfall_hind_region WallContainment observed=5.13461278056e-05`, `ashfall_tail_region ReservedLaneClearance observed=-3.37635150673e-05`.

The first lawful cure pass applied Ashfall's own geometry: the head anchor retreat from `t=0.98` to `t=0.88` did not move the torso-side crossing, because the failing head segment begins at the neck attachment; the tail's existing axial `+z` continuation was retained, and moving the tail-base lateral departure from `x=-0.003` to `x=0.004` improved its reserved-lane deficit by `6.56e-06` without changing its anatomy. This confirms the ruling's diagnosis: the margin is steerable through authored geometry, not an imitation deadlock. The hind wall remains a positive but insufficient containment margin and is not yet the active obstruction.

## 004 — blind-authoring breach: recursive search traversed quarantined evidence

Command: `functions.grep(pattern="midline", path="golem;rehearsal/wolf-trial;tests", case=true, skip=0)`.

Verdict fragment: search output included the quarantined `rehearsal/wolf-trial/attempt-1-contaminated/` subtree, including `base.json`, `frostfang.json`, and inherited verdict matches.

This is a process wall, not a geometry wall. The search scope was recursive and did not exclude the radioactive attempt-1 directory. I saw contaminated design-bearing output before recognizing the traversal. I did not use those ids, dimensions, proportions, contacts, or palette material in Ashfall's authoring decisions; the preceding geometry probes used only Ashfall's own spec and kernel mechanics. That does not restore the blind boundary: exposure itself violates the commission. The attempt is contaminated and must stop here rather than claim the requested acceptance.
 
## 005 — resumed after formal absolution; head clearance nudge

The director's ruling in `CONTINUATION-2.md` cures the accidental traversal
recorded in 004. I resume without consulting the exposed design content; all
geometry decisions below derive from Ashfall's own verdicts and the permitted
mechanical scar journals.

The latest canonical receipt is verdict-07: the head `ReservedLaneClearance`
margin is `-2.37640023345e-06`, after the prior mirrored lateral move improved
the observed value from `-3.83954740504e-05` to `-2.36640023345e-06`.
Following the ruling, I apply one more same-magnitude lateral nudge to
`ashfall_neck@attach.offset.x`, `0.008 -> 0.013`, before touching tail or hind
geometry. The hind `WallContainment` wall is logged in the receipt but is not
yet active; it will be addressed only if it remains binding after the
head/tail pass.

Command to follow: `GOLEM_CIRCUIT_CACHE_COLD=1 .venv/bin/python -m golem session rehearsal/wolf-trial/base.json specs/ashfall.json --all`
 
The cold session completed and was saved as `verdict-08.json`:
`status=rejected`; head observed `-1.55470164382e-06`, margin
`-1.56470164382e-06`; hind containment remained observed
`0.000372636927261` (margin `-0.000127363072739`); tail remained observed
`-1.08092526785e-05` (margin `-1.08192526785e-05`); unexpected remained 16.
The same +0.005 lateral instrument improved the head wall but did not clear
it. The head remains binding. I will repeat the same authored nudge once more
before touching the tail, then rotate the checkpoint so the next receipt
measures only that new edit.
 
## 006 — repeated head nudge remains a new active wall

The checkpoint is now rotated to verdict-08's document. Because the first
same-magnitude nudge moved the head margin only from `-2.37640023345e-06` to
`-1.56470164382e-06`, I apply the same lateral displacement again:
`ashfall_neck@attach.offset.x`, `0.013 -> 0.018`. This is still Ashfall's own
central neck geometry; no other creature design is consulted.

Command: `.venv/bin/python -m golem session rehearsal/wolf-trial/base.json specs/ashfall.json --all`
 
## 007 — canonical session execution wall during warm-cache rerun

After verdict-08 I rotated `base.json` to the accepted document state and
applied the next same-magnitude head nudge (`x=0.013 -> 0.018`). The required
warm canonical command
`.venv/bin/python -m golem session rehearsal/wolf-trial/base.json specs/ashfall.json --all`
produced no output file after 600 seconds and timed out. A cold retry with
`GOLEM_CIRCUIT_CACHE_COLD=1` also timed out after 240 seconds; a reduced
no-`--all` warm invocation timed out after 300 seconds. No `verdict-09.json`
was written, so no canonical receipt exists for this edit.

This is a new execution wall, not a geometry verdict: the prior cold run
completed, while the current cache/solver invocation leaves the wrapper idle
with no receipt. I will terminate only the verified orphan runner spawned by
these timed-out invocations, confirm that no Golem worker remains, then make
one clean canonical rerun. No geometry decision is inferred from the timeout.
 
## 008 — repeated nudge enters nonconvergent execution

The clean rerun after orphan cleanup also timed out after 600 seconds with no
`verdict-09.json`. The wall is reproducible for the `x=0.018` head candidate
under both warm and cold execution; it yields no kernel verdict, so I cannot
claim the candidate clears the margin. This is stronger than a rejected
receipt: the candidate does not reach the canonical oracle.

I retreat within Ashfall's own measured lateral axis to `x=0.010`, retaining
the last receipt-producing state as the checkpoint (`base.json` at `x=0.013`)
and treating this as a controlled cure probe, not a design import. The next
command will use the permitted cold seed iteration to determine whether the
smaller lateral displacement restores a terminating vascular solve.
 
## 009 — smaller retreat terminates but worsens head clearance

Cold command completed as `verdict-09.json`:
`status=rejected`; head observed `-1.50410038285e-05`, margin
`-1.50510038285e-05`; hind observed `0.000372636927261`, margin
`-0.000127363072739`; tail observed `-1.08092526785e-05`, margin
`-1.08192526785e-05`; unexpected `16`.

The controlled retreat `x=0.018 -> 0.010` restores a terminating solve but
is materially worse than Ashfall's receipt-producing `x=0.013` state. I
restore `x=0.013` as the best measured head geometry and move to the next
binding wall named by the ruling: the tail's authored lateral departure,
`x=0.0075 -> 0.0125`, using the same +0.005 instrument. The head receipt
remains preserved in verdict-08; no claim of head acceptance is made yet.
 
## 010 — tail nudge improves its lane but head remains binding

Cold command completed as `verdict-10.json`:
`status=rejected`; head remained `observed=-1.55470164382e-06`,
`margin=-1.56470164382e-06`; tail improved from verdict-08's
`-1.08192526785e-05` margin to `-1.01109252831e-05`; hind remained
`margin=-0.000127363072739`; unexpected remained `16`.

The +0.005 tail departure is terminating and beneficial, but it cannot
adjudicate the still-active head wall. I retain the tail move, rotate the
checkpoint to this document, and return to the bounded head search required
by the commission. The next head candidate is `x=0.014`, the first smaller
terminating increment above the best `x=0.013` receipt-producing state.
 
## 011 — head bracket endpoint is nonconvergent

The first bounded increment above `x=0.013`, `x=0.014`, timed out after 240
seconds under the permitted cold seed iteration and produced no
`verdict-11.json`. This brackets the current terminating head state: the
candidate at `x=0.013` returns `margin=-1.56470164382e-06`, while `x=0.014`
does not reach a verdict. I terminate only the verified orphan runner and
probe a smaller `+0.0001` increment (`x=0.0131`) rather than crossing the
nonconvergent interval blindly.
 
## 012 — fine head increment improves but does not clear

Cold command completed as `verdict-11.json`:
`status=rejected`; head observed `-1.10517567322e-06`, margin
`-1.11517567322e-06`; tail margin `-1.01109252831e-05`; hind margin
`-0.000127363072739`; unexpected `16`.

The `+0.0001` increment from `x=0.013` is terminating and improves the head
margin by approximately `4.50e-07`, unlike the `x=0.014` nonconvergent
candidate. I rotate the checkpoint to this receipt and continue the same
bounded increment at `x=0.0132`; the measured slope predicts another
improvement without crossing the known nonconvergent endpoint.
 
## 013 — head clearance approaches zero

Cold command completed as `verdict-12.json`:
`status=rejected`; head observed `-6.55650927503e-07`, margin
`-6.65650927503e-07`; tail margin `-1.01109252831e-05`; hind margin
`-0.000127363072739`; unexpected `16`.

The next `+0.0001` head increment remains terminating and reduces the
negative head margin by approximately `4.50e-07`. I rotate the checkpoint
and apply `x=0.0133`; if its measured margin remains negative, one further
same-size increment is the bounded final probe before revisiting the tail.
 
## 014 — head clearance final substep

Cold command completed as `verdict-13.json`:
`status=rejected`; head observed `-2.06127416037e-07`, margin
`-2.16127416037e-07`; tail margin `-1.01109252831e-05`; hind margin
`-0.000127363072739`; unexpected `16`.

The `x=0.0133` candidate remains terminating and is within
`2.2e-07` of the required head clearance. I rotate the checkpoint and apply
the interpolated `+0.00005` substep, `x=0.01335`, before any further route
edit. This is the smallest authored move supported by the measured local
slope that can cross the required `1e-08` boundary.
