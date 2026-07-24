"""Grown appendages (D21): a growth declaration anchored to a station of an
authored element grows into intent clouds, is enveloped with appendage-grade
taper, and its parts are FUSED into the element's own graph before meshing --
an antler welds by nature, so the element's ``components == 1`` integrity
check must survive the graft.

The suite covers the regime law of ``golem.conduits.appendage``:

  * fusion -- expand + evaluate is one watertight solid (the antler is not a
    separate component; it smooth-mins into the host);
  * mirror -- ``mirror: true`` is exact BY CONSTRUCTION: the -x parts are
    bit-exact reflections of the +x parts (designed symmetry, not stochastic
    near-symmetry). See ``test_appendage_mirror_symmetry_exact`` for the one
    place the literal field-flip spec had to be refined -- documented inline;
  * determinism -- seeded growth in authored order is reproducible to the byte;
  * the anchor guard -- an unknown anchor part is a named ValueError;
  * purity -- expansion is a COPY: the declaration-free graph carries no
    ``appendages`` key and the caller's input dict is never mutated;
  * the assembly hook -- ``compile_assembly`` runs the expansion pre-mesh, and
    the grafted element carries the antler's added surface.

Meshing resolutions are fixed and noted inline; the heaviest case (the fused
solid at res 90) runs in a couple of seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import engine as _e  # frozen pilots -- coherence_report
from golem.assembly.core import AcceptedAssembly, compile_assembly
from golem.assembly.mounts import concretize_mirrors
from golem.conduits.appendage import expand_appendages
from golem.kernel import engine as E


# --------------------------------------------------------------------------- #
# Fixtures: an x-symmetric vertical 'trunk' with one antler declaration        #
# anchored at its tip (t = 1.0), grown into a single +x intent cloud.          #
# --------------------------------------------------------------------------- #
def _decl(seed: int = 3, mirror: bool = False) -> dict:
    d = {
        "id": "antler", "anchor_part": "trunk", "anchor_t": 1.0,
        "intent": [{"id": "reach", "type": "gencyl",
                    "spine": [[0.05, 0.5, 0.0], [0.3, 0.85, 0.05]],
                    "radii": [0.06, 0.09]}],
        "count": 50, "seed": seed, "step": 0.04, "influence": 0.30,
        "kill": 0.05, "root_radius": 0.02, "min_radius": 0.008,
        "envelope": 1.7, "env_min_radius": 0.013, "tip_taper": 0.5,
        "blend": 0.02,
    }
    if mirror:
        d["mirror"] = True
    return d


def _host(seed: int = 3, mirror: bool = False, with_appendage: bool = True) -> dict:
    """A single x-symmetric gencyl trunk on the mirror plane, optionally
    carrying one antler declaration anchored at its tip."""
    g = {
        "name": "host", "blend": 0.02,
        "parts": [{"id": "trunk", "type": "gencyl",
                   "spine": [[0, 0, 0], [0, 0.5, 0]], "radii": [0.12, 0.10]}],
    }
    if with_appendage:
        g["appendages"] = [_decl(seed=seed, mirror=mirror)]
    return g


# --------------------------------------------------------------------------- #
# 1. The antler FUSES into the host: one watertight solid.                     #
# --------------------------------------------------------------------------- #
def test_appendage_fuses_into_single_watertight_solid():
    """Expand the antler onto the trunk, evaluate the composed field, and the
    coherence report is a single watertight component -- the grown appendage
    is smooth-min'd into the host, not a detached piece. (res 90.)"""
    graph = expand_appendages(_host())
    evaluated = E.evaluate(graph, res=90)
    rep = _e.coherence_report(evaluated.vertices, evaluated.faces)
    assert rep["components"] == 1, f"antler did not fuse: {rep['components']} components"
    assert rep["watertight_main"] is True


# --------------------------------------------------------------------------- #
# 2. Mirror is EXACT by construction.                                          #
# --------------------------------------------------------------------------- #
def test_appendage_mirror_symmetry_exact():
    """``mirror: true`` grows +x once and the engine reflects it. Exactness is
    proven where it actually holds -- at the part level -- and the composed
    field is x-symmetric to float precision everywhere the two halves do not
    blend.

    Refinement of the literal spec (documented, not silently dropped): with an
    x-symmetric host the padded ``graph_bounds`` are x-symmetric to the bit
    (``lo.x == -hi.x``), yet ``np.array_equal(F, F[::-1])`` -- and even
    ``allclose(atol=1e-12)`` on the FULL field -- does NOT hold. The residual
    (~8e-4 here) is not bounds-padding drift: it is confined to a few grid
    cells of the mirror plane, where smooth-min's NON-associativity bites --
    reflection swaps the fold order of every mirrored (+x, -x) pair, and smin
    is order-sensitive only where two surfaces interpenetrate (the antler is
    rooted on x=0, so its halves meet there). So we assert:

      (a) x-symmetric bounds -- the construction premise;
      (b) EXACT part-level reflection: every -x appendage part's SDF equals its
          +x twin's reflected SDF to 0.0 (this is the real 'exact by
          construction' proof, the same technique as test_mount_mirror);
      (c) the field is symmetric to 1e-12 AWAY from the mirror seam; and
      (d) the seam residual is small but real -- documenting why (a)+(c), not
          the literal full-field equality, is the honest statement. (res 64.)
    """
    graph = concretize_mirrors(expand_appendages(_host(mirror=True)))

    # (a) An x-symmetric host + mirrored appendage -> x-symmetric padded bounds.
    lo, hi = E.graph_bounds(graph)
    assert lo[0] == -hi[0]

    # (b) Every mirrored appendage part is a bit-exact reflection of its twin.
    base = {p["id"]: p for p in graph["parts"] if not p["id"].endswith("_m")}
    mirrored = [p for p in graph["parts"] if p["id"].endswith("_m")]
    assert mirrored, "concretize produced no mirrored parts"
    rng = np.random.RandomState(11)
    pts = rng.uniform(-0.6, 1.2, size=(4000, 3))
    reflection_residuals = tuple(
        float(
            np.max(
                np.abs(
                    E.part_sdf(pts, mirrored_part)
                    - E.part_sdf(
                        pts * np.array([-1.0, 1.0, 1.0]),
                        base[mirrored_part["id"][:-2]],
                    )
                )
            )
        )
        for mirrored_part in mirrored
    )
    assert max(reflection_residuals) == 0.0

    # (c) The composed field is x-symmetric to float precision off the seam.
    evaluated = E.evaluate(graph, res=64)
    F = evaluated.field
    xs = np.linspace(evaluated.lower[0], evaluated.upper[0], 64)
    off_seam = np.abs(xs) > 0.12  # the smin interpenetration band is |x| ~< 0.05
    assert np.allclose(F[off_seam], F[::-1][off_seam], atol=1e-12, rtol=0.0)

    # (d) The full-field literal spec fails ONLY at the mirror seam, and only
    #     by the smooth-min blend -- small, deterministic, and real.
    seam_residual = float(np.max(np.abs(F - F[::-1, :, :])))
    assert 1e-6 < seam_residual < 1e-2


# --------------------------------------------------------------------------- #
# 3. Expansion is deterministic; the seed drives the growth.                   #
# --------------------------------------------------------------------------- #
def test_appendage_deterministic():
    """Same input dict -> byte-identical expanded graph (seeded growth in
    authored order). A different seed grows different parts."""
    a = expand_appendages(_host(seed=3))
    b = expand_appendages(_host(seed=3))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    c = expand_appendages(_host(seed=7))
    assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True)


# --------------------------------------------------------------------------- #
# 4. An unknown anchor part is a named ValueError.                             #
# --------------------------------------------------------------------------- #
def test_appendage_unknown_anchor_raises():
    """Anchoring to a part id that is not in the graph raises ValueError naming
    the missing part."""
    graph = _host()
    graph["appendages"][0]["anchor_part"] = "ghost_limb"
    with pytest.raises(ValueError, match="ghost_limb"):
        expand_appendages(graph)


# --------------------------------------------------------------------------- #
# 5. Expansion is a COPY: no leaked declaration, no mutated input.             #
# --------------------------------------------------------------------------- #
def test_appendage_leaves_declaration_free_graph():
    """The expanded graph carries no ``appendages`` key (the declaration is
    consumed into fused parts), and the caller's input dict is untouched -- its
    parts list is the same length and its ``appendages`` key still present."""
    original = _host()
    n_parts_before = len(original["parts"])

    expanded = expand_appendages(original)

    assert "appendages" not in expanded
    assert len(expanded["parts"]) > n_parts_before  # antler parts were fused in
    # Input dict is not mutated.
    assert "appendages" in original
    assert len(original["parts"]) == n_parts_before


# --------------------------------------------------------------------------- #
# 6. The assembly hook expands appendages pre-mesh (the antler adds surface).  #
# --------------------------------------------------------------------------- #
def test_assembly_hook_runs_appendages():
    """``compile_assembly`` runs ``expand_appendages`` before meshing: a lone
    creature element carrying an inline antler compiles to a single watertight
    solid whose face count exceeds the same host meshed WITHOUT the antler at
    the same world pitch (the graft added surface).

    Pitch note: at the assembly's uniform world pitch the RES_MIN clamp must
    not distort the comparison -- a tiny host-alone box clamped up to RES_MIN
    is over-resolved and can out-face the (larger, honestly-resolved) grafted
    element. Pitch 0.016 keeps both elements above the clamp so 'same pitch'
    genuinely means 'same spacing', and the added antler surface reads as more
    faces. (The task's nominal 0.03 clamps both to RES_MIN=60 and inverts the
    comparison -- see the reported RES_MIN finding.)
    """
    pitch = 0.016
    spec_grafted = {
        "name": "kit", "pitch": pitch,
        "elements": [{"id": "creature", "role": "creature",
                      "graph": _host(), "appearance_material": "neutral_gray"}],
    }
    spec_bare = {
        "name": "kit", "pitch": pitch,
        "elements": [{"id": "creature", "role": "creature",
                      "graph": _host(with_appendage=False),
                      "appearance_material": "neutral_gray"}],
    }

    grafted_result = compile_assembly(spec_grafted, Path("."))
    bare_result = compile_assembly(spec_bare, Path("."))
    assert isinstance(grafted_result, AcceptedAssembly)
    assert isinstance(bare_result, AcceptedAssembly)
    grafted = grafted_result.records[0]
    bare = bare_result.records[0]

    assert grafted.report["components"] == 1
    assert grafted.report["watertight_main"] is True
    assert grafted.report["faces"] > bare.report["faces"], (
        f"antler added no surface: grafted {grafted.report['faces']} "
        f"<= bare {bare.report['faces']}"
    )
