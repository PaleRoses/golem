# Knight probe — initial-design bake-off (NON-EVIDENTIARY)

2026-07-12. Two isolated subagents (same model), same brief ("Obsidian Knight,"
planted greatsword, strict bilateral symmetry), 6 render rounds each, run
concurrently. GOLEM arm: frozen pilots, DSL only (~7.7 min). Blender arm: live
Blender 5.1.1, metaballs (~10.2 min). Neutral renders via the pilots' renderer.
Unblinded judging by Rosalia + coordinator Claude; labels were visible; n = 1;
knight is outside the Warden family cover. Signal, not evidence.

## Scorecard (1–7)

| judge | GOLEM appeal | GOLEM fidelity | Blender appeal | Blender fidelity |
|---|---|---|---|---|
| Claude | 3.5 | 4 | 5 | 6 |
| Rosalia | 2.5 | low (~2–3) | 4 | 4 |

Verdict: **Blender arm won initial design**, both judges, decisive margins.
GOLEM's knight hunches ("old man with a cane" per its own operator), pauldrons
swallow the helm, sword reads as a cane/drip. Blender's reads as a knight
immediately: upright, distinct helm, visible crossguard, blade to ground.

## Integrity

Both: 1 component, watertight. Symmetry IoU 0.9934 (GOLEM) / 0.9835 (Blender)
— both above the 0.98 gate. GOLEM 152k faces; Blender 12.5k.

## Interpretation

Bulwark tested revisions (GOLEM 5–25× cheaper, exact targets). The knight
tested initial design from a brief — and there the DSL handicapped the
operator: posture must be composed as raw coordinate arithmetic, no
manipulators, every look a full recompile. GOLEM's 16-part source also shows
the vocabulary straining: `blade_flat` (a squashed blob faking a flat blade
cross-section, because gencyls are round) is a proto-UnsupportedVocabulary /
quarantine-proposal moment. If the design gap holds across held-out Wardens,
Phase -1's aesthetic-parity gate (paired mean >= -0.5) is where GOLEM dies,
even while winning revision economics.

## Files

`golem_knight.json` (16-part source), `golem_knight_views.png`,
`blender_knight.glb`, `blender_knight_views.png`, `*_mesh.json` (integrity),
`knight_side_by_side.png`.
