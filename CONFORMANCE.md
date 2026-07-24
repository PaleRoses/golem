# GOLEM pilot conformance reproduction

Reproduced on 2026-07-11 with the Python 3.12.12 environment sealed by
`pyproject.toml` and `uv.lock`.

## Run

```sh
cd /Users/bluerose/Developer/pale-meridian/potentialimprovements/golem-kernel
PYTHONDONTWRITEBYTECODE=1 uv run pytest
```

Result after adding the Phase -1 contract: **25 passed, 3 strict xfailed** in
32.52 seconds. Thirteen contract tests validate the closed evidence schema, the
2-calibration + 6-held-out corpus with exactly 24 unique revisions, all pinned
pilot/output/harness hashes, exact clause paths, lawful stage causality, the common
measurement cover, blinding bundles, UTC receipts, and typed artifact refinements.
The expected obstructions are:

- raw constrained symmetry IoU is `0.9799926081064433`, below `0.98` by
  `0.0000073918935567`;
- the v4 reference render differs by four PNG pixels and five GIF pixels;
- the plated GLB has equivalent vertices and colored faces but a different
  serialized face order.

The semantic oracle reproduced:

- `hand_v1`: 15 anatomy + 5 fist violations, total 20;
- `hand_v2`: total 0;
- v4 at resolution 190: one component, no dust, watertight, 174,236 faces;
- mount: ground `-0.230`, lift `0.250`, negotiated junction
  `[1.28, 1.05, 0.76]`, 14 parts;
- plates: 110 cells, 324 adjacencies, 38,197 seam faces;
- constrained/control: one/one solid component, one/35 mesh shells,
  watertight/not-watertight, displayed symmetry `0.980`/`0.998`.

## Preserved report goldens

The report goldens were established from the present pilot sources with:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python pilots/hand_pack.py pilots/hand_v1.json > conformance/golden/hand_v1.txt
PYTHONDONTWRITEBYTECODE=1 uv run python pilots/hand_pack.py pilots/hand_v2.json > conformance/golden/hand_v2.txt
```

They are **new conformance fixtures**, not historical evidence that the
original pilot emitted the same text.

| Fixture | SHA-256 |
|---|---|
| `hand_v1.txt` | `ffcc7abb5c31eb7bcb819f6b8069c0e3b6ccd19ce1ff175f1c12ce81a4f23298` |
| `hand_v2.txt` | `b71b83ac5e0ccfb82f2c28fb9d8d28d58e5c58f218b3cb35b802a45d5f0038c8` |

## Artifact comparison

| Artifact | Regenerated SHA-256 | Preserved SHA-256 | Result |
|---|---|---|---|
| `plate_complex.json` | `af11dd22b3dd1186a2bdc5077fecf1fc88950409ef6bac96b49280dad5da3666` | same | byte-exact |
| `golem.glb` | `d406dc84d11e9e996a901de07afd36c69851d6543d7cac289172181328e2b3ab` | same | byte-exact |
| `golem_hands.glb` | `2fcb38d2197f24ec3c600b76929dadf63ac6b6da70934ee48c02342e29d84e1c` | same | byte-exact |
| `golem_plated.glb` | `b122704a289e9c5f1a36a65ee2539dc2ead249c2d55c743e66f460d8792b0fad` | `23adc421ec870abda26e6be928db40023981207a1086d10218b79ca26cad522e` | semantic match; ordering obstruction |
| `v4_views.png` | `dc76140444e5b12405296b5ac4ace98b23e59316e9fddeb246bccf7e362a877b` | `6da2315728346ac6201c15d88f69be3067c28b69c040d0a2e693731bf024bc72` | four pixels differ |
| `v4_turntable.gif` | `f48a95294c45969d4d7e611b36d4d5e70c3e3ffd76e32fd7f5c897f62955371f` | `004cfa40cbd14dd4291886622f0e763e30cb2f57c7c242329fe7c68765bac456` | five pixels differ |

`golem.glb` and `golem_hands.glb` required inferred export calls because the
pilot documentation supplies no production command for them. Their exact
matches therefore validate the artifacts, not the missing recipe.
