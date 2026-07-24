# Operator experience report - decorating "the acolyte"

Operator: fresh, first contact with the tool. Documentation available: HELP text only.
Result: a hooded acolyte with an engraved hem ring, an engraved collar, a glowing
violet rune girdle at the waist, and a central engraved+glowing seam. 4 conduits, no
appendages, no vascular. Final saved to final.json. Receipt clean throughout
(components=1 watertight=True dust=0 verdict OK, no violations).

Harness runs used: 5 of 8 (parts; declare+grow+show; render1; render2; grow+save).
Renders used: 2 of 2.

## What I actually built
| id | kind | where | emit | material | verified? |
|----|------|-------|------|----------|-----------|
| hem | axial_loop | robe:0.04 (base) | groove | gambeson_dark | by render only |
| collar | axial_loop | robe:0.9 (chest/neck) | groove | steel_violet | by render only |
| rune_ring | axial_loop | robe:0.55 (waist) | band | emissive_seam | receipt 1034 faces + render |
| rune_line | face_line | robe axis y +z | groove,band | emissive_seam | receipt 233 faces, NOT visible in render |

## Top friction points

### 1. The receipt is silent on grooves - I could not confirm my two engraved trims landed.
The receipt printed a faces=N line only for the two emit-band conduits (rune_ring,
rune_line). My two emit-groove conduits (hem, collar) produced NO receipt line at all.
HELP frames faces=N as THE signal that a band "found surface" - grooves get no
equivalent. So for the hem trim (a headline requirement of my brief) I had zero cheap
confirmation it engraved anything; I had to spend a render to find out. For a tool whose
pitch is "growing is cheap, render only to confirm," the cheap path left me blind on
exactly the engraving features I was asked to make. Biggest confusion of the session.

### 2. Face count is not visibility - a feature can "find surface" and still be invisible.
rune_line reported 206 then 233 faces (per HELP that means it found surface and is fine).
But it is effectively invisible in both hero renders: it runs up the front mid-plane,
exactly where the figure's clasped hands/arms sit, and it does not surface on the lower
skirt where it should show. The receipt reported a healthy face count while the feature
contributes nothing a viewer sees. No signal for occlusion, screen coverage, or "is this
doing anything." I burned my second render confirming a change that INCREASED the face
count (206->233) made no visible difference.

### 3. No way to learn part orientation without dumping the whole element.
HELP says "t in [0,1] along the spine (0 = first anchor)" - but "first anchor" is
meaningless until you can see coordinates. parts only lists id + kind. I had to run show,
which dumps the ENTIRE element JSON, just to learn the robe spine runs bottom->top (so
t=0 is the hem). I guessed hem at t=0.04 and got lucky. A parts-with-bounds, or a
station robe:0.04 query returning world position, would remove the guess.

## Things I reached for that do not exist
- A groove face-count / landed confirmation in the receipt (see #1).
- Extent/offset control on face_line. It spans the full mid-plane; I wanted the rune only
  on the visible lower skirt, but could not limit its range or nudge it off the occluded
  center. My only lever was normal, which picks a side, not a span.
- A scoped show (show conduits / show <id>). show dumps everything; noisy when I only
  wanted to sanity-check my four declarations.
- Material previews. The tag list gives names but no color/finish. I guessed emissive_seam
  = violet glow (right) and picked steel_violet / gambeson_dark for grooves blind. A
  one-line swatch per tag would let me pick tastefully without a render.
- Documentation of where render output lands; I passed an absolute prefix to be safe.

## What the receipt failed to tell me
- Whether grooves landed (no line at all for them).
- Whether any band is actually visible vs occluded/subtle (only a raw face count).
- Anything about placement/orientation, so I could not reason about t without show.
- Undocumented fields appeared (dust=0, verdict OK) that HELP never mentions - benign and
  self-explanatory, but minor doc drift for a first-timer trusting HELP as gospel.

## What worked well
- declare -> grow -> receipt loop is genuinely fast (~3s) and every command was accepted
  first try; HELP syntax matched real behavior exactly. Rejections-as-answers,
  journaling, and "nothing corrupts until save" made me feel safe to experiment.
- watertight / components / violation reporting is clear and confidence-building - I never
  wondered if I had broken the solid.
- emit groove,band composed cleanly, and save confirmed with a byte count + journal.

## What I would change
1. Print a faces (or "carved N cells") line for groove conduits too - parity with bands,
   so the cheap loop can confirm every feature.
2. Add a visibility/coverage hint per conduit (screen-visible faces, or an occlusion
   warning) so face count stops masquerading as effectiveness.
3. Make parts show spine endpoints / bbox, or add a station <part>:<t> position query, so
   orientation is not a guess or a full JSON dump.
4. Give face_line a start/end (t-range) and lateral offset.
5. Add show <id> / show conduits, document render output paths, add material swatches.

## Usability score: 3 / 5
Core loop is fast, safe, and honest about the solid - a good foundation. But the receipt
left me blind on 3 of my 4 features (both grooves unconfirmable; one band invisible),
forcing me to spend BOTH budgeted renders learning things the cheap path should have
surfaced. Orientation discovery required dumping the whole element. Very usable once you
know the geometry; noticeably rough for a genuine first-timer.
