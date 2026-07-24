# First light — the appearance chain, run end to end

**Honest header: machinery record, not evidence. Renders are the fallback backend (deterministic painter renderer); the Blender backend is written but has never executed (no Blender in the authoring sandbox — its first successful run on a real machine is its acceptance). No blinded judging has occurred.**

## What ran (2026-07-18)

Tag → material record (`golem/materials/`) → real glTF PBR slot + JSON sidecar (`golem/assembly/exchange.py`) → presentation backend (`presentation/fallback.py`, driven by `presentation/contract.py`'s pinned CONTRACT, consuming ONLY the GLB + sidecar — the D3 fence that presentation imports nothing from `golem` is enforced *as a test*). The knight assembly rendered with per-element materials: obsidian harness (roughness 0.24), deeper obsidian helm, metallic steel-violet sword (metallic 0.75 — dimmed diffuse, tinted specular). `firstlight_sheet.png` is the result; the sword visibly reads as a different substance than the armor, which is the entire point of M3.

## Verification

44 tests green: `tests/test_materials.py` (5), `tests/test_presentation_fallback.py` (5 — including render determinism and the D3 source fence), plus the 34 prior (assembly/profile/engine). PBR slots survive trimesh reload with correct factors.

## Addendum (2026-07-18, later): conduits — the armature of intention

`knight_conduits_sheet.png` + `knight_conduits.glb`: the same separated knight with five one-line conduit declarations (cuirass rim and waist trim rings, tasset hem groove, helm visor line, blade fuller-rune), embedded analytically by `golem/conduits/` at compile time, emitted as engraved grooves plus lifted emissive inlay bands carrying their own material records — zero renderer changes. The first GOLEM artifact whose decoration is subordinated to authored structure. Re-embedding stability across meshing resolutions is a passing test, not a hope (`tests/test_conduits.py`). Next words declared, not shipped: geodesic surface paths (flip-geodesics), stripe fields, grown vein networks, session ops.

## Historical addendum (2026-07-18): D22 vascular reinforcement

`body_supported.glb` / `body_supported_sheet.png` and `knight_supported.glb` / `knight_supported_sheet.png` record the superseded D22 proof: a raw seeded vascular declaration, an open tree, and a radius-derived support proxy. They remain experiment evidence, not the current interface. D24 deleted that declaration surface rather than preserving a counterfeit compatibility command.

## Accepted addendum (2026-07-18): hierarchical closed vasculature

The authoritative D24 proof is now `../vascular/`. Sparse `anatomy/0.1` intent deterministically generates 256 paired terminal sites and a closed gold-supply / cyan-exchange / violet-return graph, solves its normalized Poiseuille field, and emits visible diagnostic meshes. The canonical receipt records 1,062 nodes, 1,316 edges, residual `5.502e-11`, boundary balance `8.937e-13`, maximum per-bed delivery error `1.347e-5`, exact bilateral symmetry, zero nonincident intersections, and certified capsule margin `5.000e-4`. D26 deleted the former bounded surface-displacement proxy: it changed render vertices rather than a load-bearing domain and therefore could not support a strength claim.

## Outstanding, honestly

The Blender leg (`presentation/blender_contract.py`) is written-not-run. Per-part material tags inside a body-dialect creature (plan M3's dialect half) and silhouette-mask scoring (D7) are not built. The plated golem and legacy artifacts have not been re-rendered through this path. The `texture_set` hook (generated PBR map sets, CHORD-class) is reserved and empty.
