# OPERATOR_NOTES — Obsidian Knight session

Frank account of authoring a heavily-armored two-handed-greatsword knight in the
session REPL, working only from HELP.txt. Six renders used (r1–r6, budget limit),
~30 tool calls. Final: `knight_final.json`, `knight.ops`, renders r1–r6.

## What the tool made EASY

- **Receipts are the killer feature.** Every command prints quantified anomalies
  (`unintended_fusion gap -0.171`, `cross_plane_fusion overlap 0.095 across x=0`,
  `floating_contact clearance -0.920`, `missing_fusion gap 0.041`, `blend_ambiguity`)
  plus contract deltas and est_tokens — all FREE and fast. I diagnosed and fixed the
  sword-through-torso and legs-crossing-centerline bugs entirely from receipt numbers
  before ever rendering. Most of my real work happened without spending renders.
- **`reflect` for symmetry.** One command (`reflect clavicle.L`, `reflect hip.L`)
  mirrored whole subtrees *including flesh*, live, as a constraint. Strict bilateral
  symmetry was free and unbreakable by construction — I never authored a right side.
- **`world_dir` is reliable.** World-frame direction, auto-converted to parent frame.
  Every bone I placed with `world_dir` went exactly where I expected. This is the
  trustworthy primitive.
- **`envelope` gave measurable targets.** heads-tall / balance-margin / single-component
  as live clauses with numbers each run (`6.174 heads, target [6.5,7.5]`,
  `centroid margin -0.007`). Great for steering; I dialed heads 4.87 → 6.5+ by watching
  the number.
- **Anchor-by-station (`PARENT:T`) mental model** is clean once you accept "no positions."

## What FOUGHT me

- **`offset` is parent-frame with no way to know the frame's axes.** My single biggest
  time sink. `offset:[0,0,0.30]` to push the sword forward and `offset:[0.16,...]` to
  spread the hips both silently went to the *wrong* world axes (receipts showed the
  sword still fused to abdomen/tasset, legs still crossed x=0). There is no introspection
  for "which local axis is forward/lateral." **Workaround:** dedicated stub bones placed
  with `world_dir` — a fleshless `swordmount` (world_dir `[0,0,1]`) to get the sword in
  front, and a `hip.L` cross-bar (world_dir `[1,0,0]`) to get true lateral hip
  separation. Both worked first try once I stopped trusting `offset`. (~2 wasted rounds.)
- **No world-position / bounding-box query.** `outline` reports structure (length, dof,
  flesh sizes) but never where anything *is*. I inferred geometry from anomaly gap
  magnitudes and from renders. A `where`/bbox readout would have saved at least one render.
- **Flesh box axis-mapping is undocumented and bone-orientation dependent.** For the
  crossguard I wanted width side-to-side (world X); `size:[0.36,0.05,0.07]` instead made
  it stick *forward* (I caught this reasoning about the az90 side view — a lateral bar
  would foreshorten to a dot, but I saw a forward bar). On the downward sword bone,
  local-X→worldZ and local-Z→worldX. Moving the 0.36 into local-Z fixed it. Had to
  reverse-engineer this from one render. (1 round.)
- **The heads-tall "head unit" is opaque.** Shrinking the helm's Y dimension moved the
  metric *not at all* (6.423 → 6.423). It keys off the largest helm dimension or the head
  bone length, not height. I burned two nudges before switching strategy to lengthening
  the legs to raise the numerator. (2 rounds.)
- **Single-component integrity is emergent from girth, not explicit.** My sword only
  counted as one component because the fat round blade happened to touch the torso. When
  I swapped to a realistic thin flat blade it split into 2 components (`post_dust_components
  2, target 1`) — the sword-to-body connection is implicit mesh proximity, with no
  weld/join op. Cost render r5; reverted to the round blade. This means my hands are
  *near* the pommel but I could not confirm they mechanically grip it.

## What I reached for that DID NOT EXIST

- A world-space position or bounding-box query (`outline … geometry`).
- Per-bone frame/axis introspection (which local axis is forward/lateral) — I guessed + rendered.
- An explicit weld/join to force two flesh pieces into one component.
- A stance/measurement readout (e.g., "how wide are the feet").
- **Mounted hands via the mount subgrammar:** HELP's `add mount` example needs a spec file
  (`hand_v2.json`) and a port I could define, but no port mechanism is documented beyond the
  example and no hand spec existed. Per the rules I used simple flesh gauntlet mitts instead.
  (Contamination fences respected — no source-diving.)
- Comments in the ops file (unknown if supported; avoided them, so the growing script is
  un-annotated and harder to maintain).

## Rounds per fix (roughly)
- Sword in front (offset fail → swordmount stub): 2 rounds, diagnosed from receipts.
- Legs crossing centerline (offset fail → hip cross-bar bone): 2 rounds.
- Stocky → ~7-heads proportions: ~4 receipt-only nudges (4.87→5.74→6.17→6.32→6.42→pass).
- Crossguard orientation (local-Z width): 1 round, verified in r3.
- Heads metric strategy switch (helm-Y is inert → lengthen legs): 2 rounds.
- Flat blade attempt (broke integrity → revert): 1 round, cost render r5.

## Self-score: appeal/fidelity — 5 / 7
It unambiguously reads as an armored knight at a glance — distinct crested helm above
broad pauldrons, breastplate/tasset, greaves and sabatons, strict symmetry, ~6.5 heads,
visible lateral crossguard, and a greatsword planted point-down in front with the blade
reaching the ground — but it is blocky and nutcracker-stiff, the two-handed grip on the
pommel isn't crisply legible, and the blade is a round taper rather than a true flat
greatsword blade, so it lands as "clearly correct and on-brief" rather than "handsome."
