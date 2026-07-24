# FRICTION JOURNAL — frostfang wolf authoring trial

Trial 5. Author: Blue Rose. Creature: Frostfang, a cursorial wolf with a deep chest, tucked waist, four digitigrade legs, straight tail, and long muzzle.

Inherited scars are applied silently from the scree-maiden, verdigris, cinderwake, and Sable Stride formation journals: direct `.venv/bin/python`; blob and loft dimensions are half-extents; parent-frame rest directions require frame probes; axial continuation uses local `+z`; mirrored limb topology remains authoritative; carrier flesh must clear the 0.040 floor; `session` is canonical and requires `status=accepted` plus `diagnostics_summary.unexpected=0`; `check` must agree; palette ids are closed; appearance palette does not control look-render catalogue colors; base rotates after every verdict.

## 000 — staged descent

The first authored surface is skeleton plus carrier flesh only. Decorative flesh, muscles, eyes, attachments, skin formation, and tint are withheld until the vascular and four-foot support surfaces are accepted. The wolf deliberately reuses the already-proven carrier dimensions and frame-safe stance rather than reopening known walls.
 
## 001 — stage-1 contract references withheld decorative flesh

Command: `GOLEM_CIRCUIT_CACHE_COLD=1 .venv/bin/python -m golem check specs/frostfang.json`

Verdict fragment: `OBSTRUCTIONS 1 [bad_contract_midline] spec/contract/midline/5: unknown part id 'muzzle'`.

The staged carrier document still carried the final silhouette's `muzzle` midline address even though the decorative muzzle flesh was deliberately withheld. This is a new sequencing wall: the contract validates all part references before the surface can reach the vascular result, so a future-stage name cannot appear in an earlier transaction. I removed only the withheld name; no geometry or kernel law was changed.
 
## 002 — formed skin cannot precede the carrier-stage surface

Command: `GOLEM_CIRCUIT_CACHE_COLD=1 .venv/bin/python -m golem check specs/frostfang.json`

Verdict fragment: `ASSEMBLY acceptance: REJECTED [ElementSurfaceFormationObstruction] ... surface formation rejected before a mesh exists`.

The carrier-only stage inherited the final skin formation declaration, but the formed surface has no valid enclosing decorative surface yet. This is a genuinely new stage-order wall rather than the cured quadruped global-ray wall: the layer is being asked to form before the creature has its final flesh topology. I removed only the `formation` keys for the carrier checkpoint and will restore `static_implicit_relaxation` after decorative flesh and muscles exist.
 
## 003 — formed skin needs the authored contact graph

Command: `.venv/bin/python -m golem check specs/frostfang.json`

Verdict fragment: `ASSEMBLY acceptance: REJECTED [ElementSurfaceFormationObstruction] ... locality_radius=0.033930879557616964, maximum_displacement=0.0467511317377961`.

With the muzzle, carrier anchors, and four skeleton-integral muscles present, vascular acceptance remained clean (`4/4 closed`), but formed skin still rejected. The accepted Sable surface carries its explicit attach graph; the formation stage therefore depends on the final intended contact topology, not merely the presence of flesh. I am moving to the commissioned attach pass, preserving the geometry and recording the dependency rather than weakening the skin declaration.
 
The attach pass cleared every visible anomaly (`ANOMALY 0`) but left the same formation obstruction bit-identical. Comparing the accepted formed quadruped shows its only remaining structural surface is the eye record; the wolf's withheld cranial detail is therefore the next honest stage, not a skin relaxation workaround.
 
## 004 — correction: the formed Sable baseline is a rejected partial

The prior sentence misattributed the formed-skin failure to the missing attach graph. I checked the primary receipt before making a geometry edit: `docs/plans/skin-siege-journal.md:501-512` records `sable_stride_formed.json` itself as `status rejected`, `38 expected / 1 unexpected surface_formation.SkinRelaxationObstruction`, with no skin-family lever clearing the six dorsal-fold witnesses. The attach pass did clear the wolf's anomalies, but it was not the cure for this obstruction.

Command: `.venv/bin/python -m golem check specs/frostfang.json`

Verdict fragment: `ASSEMBLY acceptance: REJECTED [ElementSurfaceFormationObstruction] ... maximum_displacement=0.0467511317377961 ...`.

This is the newly exposed wall: the amended quadruped ray law is fixed, but this wolf's uniform offset still cannot cover a dorsal self-contact fold. The help surface supplied no operation for the obstruction. I will redesign the authored surface around the fold or remove only the formation opt-in under an explicit honesty receipt; I will not claim the rejected Sable partial as a lawful baseline.
 
## 005 — formation removal is not a commissioned completion

The first accepted session (`verdict-02.json`) was only an intermediate receipt: removing every `formation` opt-in made `check` and `session` agree, but violates this trial's explicit requirement to author uniform formed skin where the wolf honestly wants it. The side render confirmed the semantic loss—the body is a narrow unformed carrier silhouette, not a covered wolf.

I am reopening formation on the same digitigrade skeleton and will strip Sable-specific dorsal decoration incrementally. The target is not the old Formation Trial escape; it is a new formed-skin acceptance with `static_implicit_relaxation`, `status=accepted`, `unexpected=0`, and a matching `check`.
