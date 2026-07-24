# Reforge record — obsidian_knight_v2 under the v0.5 `profile` vocabulary

**Honest header: unblinded, n=1, authored by the coordinating model (fable) in a single session (2026-07-17), judged by its own author against render sheets. This is a demonstration record and vocabulary-word receipt, NOT evidence of aesthetic superiority. No fresh operator, no blinded judges, no contract pack. Signal, not proof.**

## What this is

A ground-up re-authoring of the session knight as a raw v0.5 part-graph, executing Rosalia's 2026-07-17 directive: *eliminate the box as a massing primitive; make each element use the form it actually has.* Every armor mass is a `profile`d gencyl (swept superellipse cross-section — the new quarantine word, `specs/vocabulary.json`); the two blobs that remain are genuinely round things (pauldrons, pommel, joints). Zero `box` parts. The greatsword blade is genuinely flat (base cross-section ~0.17 x 0.051) and is joined to the gauntlets by an explicit authored bead part (`grip_weld`) — the weld *semantics* of the v0.5 plan's M1, demonstrated manually; the dialect/session weld *machinery* remains unbuilt.

Final state after 7 compile-render-judge rounds: **1 watertight component, 0 dust, 0 vocabulary violations** at res 200, with a flat welded blade — the configuration that was impossible on 2026-07-17 morning (the session probe's thin blade split the creature; the operator shipped a round-fat blade instead).

## Artifacts

- `knight_v2.json` — the spec (raw v0.5 dialect: gencyl/blob + profile + rot; no boxes)
- `knight_v2.glb` — exchange mesh (205k faces, res 200)
- `knight_v2_final_sheet.png` — presentation-shaded hero views (15/55 azimuth)
- `knight_v2_fourviews.png` — canonical 0/40/90/180 strip
- (two loose duplicate renders formerly parked here were deleted 2026-07-18 once deletion was enabled)

Renders are the sandbox demonstration shading (specular/fresnel/tonemap painter's renderer), NOT the Phase -1 Blender contract. Same caveat as `rehearsal/first-light` in the v0.5 plan.

## Findings worth keeping (fed back into the v0.5 plan's Surprises)

1. **Interior-sign cap bug** (fixed during implementation): the first profiled-gencyl SDF returned 0 (not negative) inside the solid because the clamped-spine decomposition discarded the axial sign; marching cubes produced 44 components and 4,544 dust slivers. Fixed with a signed axial distance in the extrusion combination. Regression test: `tests/test_profile.py::test_profile_interior_is_negative`.
2. **Void-bubble components**: when two surfaces approach within ~1 grid cell and the smooth-min half-bridges the gap, marching cubes seals a sub-cell interior pocket that `coherence_report` counts as a real component (the observed bubbles had 112 faces — well above the 10-face dust threshold). The sword hovering 3.3 cm from the belly at blend k~0.04 produced two such bubbles. This is a new, undocumented failure mode of the components=1 integrity check — relevant to the M1 weld design and to the dust-filter definition (R-5).
3. **Feature-size law vindicated**: the blade tip at 2*min(r*aspect) = 0.018 < 2*pitch aliased into island fragments exactly as `R-8-feature-size` predicted; thickening the tip cleared both the violation and the fragments. The checker earns its keep.
4. **`rot` non-unit catch**: a hand-typed gauntlet quaternion drifted |q|-1 = 2.3e-4 and was caught by `R-quat-nonunit`, normalized on read, then corrected at source.

## Addendum (same evening): the element split

Rosalia's second directive — armor is worn, never grown; elements must be separate — was demonstrated on the same knight (`knight_v2_elements.json`, `knight_v2_assembly.glb`, `assembly_sheet.png`): three solids (harness / helm / sword), each compiled in its own field with its own integrity report (all components=1, watertight), its own material in one GLB scene. The sword is HELD — the `grip_weld` bead was deleted, because a held object was always a category error for welding; welds are intra-element (grown structure) only. Doctrine and consequences recorded as D12 in the v0.5 plan: per-element integrity, element-scoped materials (resolving the plan's D6 multi-primitive tension), the fit/clearance check as future cross-element machinery, and the honest open item — this knight's harness still conflates body and worn plate; a true underbody is unresolved.

## Addendum 2 (2026-07-18): the assembly subsystem, real

`golem/assembly.py` makes the element doctrine machinery: an assembly dialect with differential roles (creature = full stack; equipment = lean subset — morphology, material, anchors, integrity, and *forbidden* the body dialect per D14), canonical-frame equipment placed by rigid mounts, bit-exact mirror concretization, per-element meshing at uniform world pitch (the greathelm: 160k faces at uniform res → 14k at identical fidelity), and named-geometry GLB scene export. The knight here is recut through it: `knight_assembly.json` + `elements/{harness,greathelm,obsidian_greatsword}.json` — the sword and helm are now *reusable*: authored once in their own frames (a sword's origin is its grip), mountable on any morphology. `assembly_v2_sheet.png` is the compiled result; `tests/test_assembly.py` (8 tests) is the verification. One defect (D14 guard ordered after the load, unreachable in the common mistake case) was caught by the test dispatch and fixed before shipping.

## Addendum 3 (2026-07-18): the body beneath the armor

`knight_separated.json` closes the underbody question: a true **body** creature element (bare morphology in dark gambeson — head, torso, limbs, one watertight solid, complete on its own) with armor as equipment elements over it (torso_plate, pauldron ×2, arm_plate ×2, leg_plate ×2, helm, sword — ten instances from six unique specs via the new `mirror_x` mounted reflection, exactness SDF-proven in `tests/test_mount_mirror.py`). `separation_proof_sheet.png`: armored | body-only | three-quarter. Renders are the native presentation backend (`presentation/native.py`), which per D16 is the authoritative judged sense — Blender is demoted to an optional never-yet-run cross-check. Fit is by authored clearance margins; the fit/clearance check remains future machinery.

## Addendum 4 (2026-07-20): body-first active specimen

The armor experiment above remains available, including `golem/plates/`, the
fit laws, tests, and reusable armor element sources. It is no longer the
canonical visual target. `knight_separated.json` now instantiates only the
body and its held sword; all eight worn-armor instances and their service
assignments are absent. Current coupled receipts and renders therefore expose
the body instead of laundering its defects beneath equipment. The older
armored files in this directory are historical experiment artifacts, not the
current acceptance surface.

## Verification

`tests/test_profile.py` (6 tests: delegation bit-exactness, mirror exactness, golden blade mesh, interior sign, all violation rules, circle-profile agreement with the frozen engine) — green in the sandbox alongside `tests/test_engine.py` (20 tests). Golden: `golden/profile_blade_v1.json`. Registry entry: `specs/vocabulary.json` kind `profile`, status `quarantine` (promotion is Rosalia's act). Living-suite session/body golden tests show 10 pre-existing byte-exact failures under the sandbox interpreter (Python 3.10/aarch64 vs the sealed 3.12/macOS goldens) with exact HEAD parity (10 failed before these changes, 10 after) — platform drift, not regression; the authoritative run remains `uv run pytest tests` on macOS.
