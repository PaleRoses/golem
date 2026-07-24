"""Tests for the GOLEM v0.3 wrapper engine (``engine``): quarantined ``box``
and ``rot`` vocabulary.

Fast synthetic + analytic cases run by default. The frozen-corpus delegation
check and the golden-artifact meshes are marked ``slow`` (heavy marching-cubes
recompute); run them with ``-m slow`` (chunk with ``-k`` to stay under a 45s
cap) or skip with ``-m 'not slow'``.

Meshing resolutions are fixed and noted inline; the delegation corpus uses
res 130 (comfortably under the bash cap while still exercising the full field).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from golem import paths as _paths
import engine  # frozen (pilots), the delegation reference
from golem.kernel import engine as ev
from golem.kernel.engine import compile as engine_compile
from golem.kernel.engine.algebra import part_sdf_checked, prepared_part_sdf
from golem.kernel.engine.types import Accepted, decode_part

_GOLDEN = _paths.GOLDEN

_DELEGATION_RES = 130  # noted in the brief; keeps each corpus run under the cap.


# --------------------------------------------------------------------------- #
# Small deterministic point cloud for analytic / property SDF checks.          #
# --------------------------------------------------------------------------- #
def _sample_points(n=4000, seed=0, scale=1.5):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n, 3))


# --------------------------------------------------------------------------- #
# 1. Delegation bit-exactness -- v0.2 graphs are bit-identical.               #
# --------------------------------------------------------------------------- #
_CORPUS = (
    ("golem_v4", _paths.PILOTS / "golem_v4.json"),
    ("golem_hands", _paths.PILOTS / "golem_hands.json"),
    ("golem_knight", _paths.REHEARSAL / "knight" / "golem_knight.json"),
)


def test_delegation_bit_exact_fast():
    """A tiny v0.2 graph is bit-identical through both engines (cheap gate that
    runs in the default suite; the full frozen corpus is the ``slow`` test)."""
    graph = {
        "name": "tiny",
        "blend": 0.05,
        "parts": [
            {"id": "torso", "type": "gencyl",
             "spine": [[0, 0.4, 0.0], [0, 1.2, 0.1]], "radii": [0.4, 0.45]},
            {"id": "shoulder", "type": "blob", "mirror": True,
             "center": [0.4, 1.1, 0.0], "size": [0.28, 0.28, 0.28]},
        ],
    }
    _, _, _, F_ref, _, _ = engine.evaluate(graph, res=48)
    evaluated = ev.evaluate(graph, res=48)
    assert np.array_equal(F_ref, evaluated.field)


@pytest.mark.slow
@pytest.mark.parametrize("name,path", _CORPUS, ids=[c[0] for c in _CORPUS])
def test_delegation_bit_exact_corpus(name, path):
    """Every frozen v0.2 corpus graph compiles bit-identically (F arrays AND
    coherence reports) through engine as through the frozen engine."""
    graph = json.loads(Path(path).read_text())
    v_ref, f_ref, _, F_ref, lo_ref, hi_ref = engine.evaluate(graph, res=_DELEGATION_RES)
    evaluated = ev.evaluate(graph, res=_DELEGATION_RES)
    assert np.array_equal(F_ref, evaluated.field), f"{name}: field array differs"
    assert np.array_equal(lo_ref, evaluated.lower) and np.array_equal(
        hi_ref, evaluated.upper
    )
    assert np.array_equal(v_ref, evaluated.vertices) and np.array_equal(
        f_ref, evaluated.faces
    )
    rep_ref = engine.coherence_report(v_ref, f_ref)
    rep_new = ev.coherence_report(evaluated.vertices, evaluated.faces)
    assert rep_ref == rep_new, f"{name}: coherence report differs"


@pytest.mark.slow
def test_check_delegation_cli_all_true():
    """The ``--check-delegation`` entry point reports True for every file."""
    assert ev.check_delegation(res=_DELEGATION_RES) is True


# --------------------------------------------------------------------------- #
# 2. Box determinism -- identical field on repeat compilation.                #
# --------------------------------------------------------------------------- #
def _box_graph():
    return {
        "name": "box_probe",
        "blend": 0.02,
        "parts": [
            {"id": "slab", "type": "box", "center": [0.0, 0.5, 0.0],
             "size": [0.30, 0.10, 0.18], "round": 0.03},
            {"id": "post", "type": "gencyl",
             "spine": [[0, 0.0, 0], [0, 0.5, 0]], "radii": [0.06, 0.06]},
        ],
    }


def _carved_graph():
    return {
        **_box_graph(),
        "carves": [
            {
                "id": "socket",
                "type": "blob",
                "mirror": True,
                "center": [0.18, 0.5, 0.15],
                "size": [0.12, 0.08, 0.12],
            },
        ],
    }


def test_box_field_is_deterministic():
    graph = _box_graph()
    F1 = ev.evaluate(graph, res=72).field
    F2 = ev.evaluate(graph, res=72).field
    h1 = hashlib.sha256(np.ascontiguousarray(F1)).hexdigest()
    h2 = hashlib.sha256(np.ascontiguousarray(F2)).hexdigest()
    assert h1 == h2
    assert np.array_equal(F1, F2)


def test_evaluated_morphology_keeps_field_bounds_and_pitch_together():
    evaluated = ev.evaluate(_box_graph(), res=48)
    assert isinstance(evaluated, ev.EvaluatedMorphology)
    assert evaluated.field.shape == (48, 48, 48)
    assert evaluated.resolution == 48
    assert evaluated.maximum_pitch == max(evaluated.world_pitch)
    assert not evaluated.field.flags.writeable
    assert not evaluated.vertices.flags.writeable


def test_sample_graph_field_preserves_ordered_smooth_union() -> None:
    graph = _box_graph()
    points = _sample_points(n=512, seed=31)
    lower, upper = ev.graph_bounds(graph)
    diagonal = float(np.linalg.norm(upper - lower))
    global_blend = float(graph["blend"]) * diagonal
    slab, post = graph["parts"]
    expected = ev.smin(
        ev.part_sdf(points, slab),
        ev.part_sdf(points, post),
        global_blend,
    )

    assert np.array_equal(ev.sample_graph_field(graph, points), expected)


def test_crease_union_has_no_smooth_bulge_overshoot() -> None:
    parts = [
        {
            "id": "left",
            "type": "blob",
            "center": [-1.0, 0.0, 0.0],
            "size": [1.0, 1.0, 1.0],
        },
        {
            "id": "right",
            "type": "blob",
            "center": [1.0, 0.0, 0.0],
            "size": [1.0, 1.0, 1.0],
        },
    ]
    seam_probe = np.asarray([[0.0, 0.1, 0.0]])
    primitive_union = np.minimum(
        ev.part_sdf(seam_probe, parts[0]),
        ev.part_sdf(seam_probe, parts[1]),
    )
    crease = ev.sample_graph_field(
        {
            "name": "crease",
            "blend": 0.2,
            "parts": [parts[0], {**parts[1], "operator": "crease"}],
        },
        seam_probe,
    )
    blended = ev.sample_graph_field(
        {"name": "blend", "blend": 0.2, "parts": parts},
        seam_probe,
    )

    assert np.array_equal(crease, primitive_union)
    assert crease[0] > 0.0
    assert blended[0] < 0.0


def test_chamfer_union_dispatches_through_the_checked_operator() -> None:
    graph = _box_graph()
    points = _sample_points(n=128, seed=311)
    lower, upper = ev.graph_bounds(graph)
    radius = float(graph["blend"]) * float(np.linalg.norm(upper - lower))
    first, second = graph["parts"]
    expected = ev.compose_union(
        ev.LegacyCompositionProblem(
            accumulated=ev.part_sdf(points, first),
            incoming=ev.part_sdf(points, second),
            operator=ev.CompositionOperator.CHAMFER,
            radius=radius,
        )
    )
    assert isinstance(expected, ev.AcceptedComposition)
    authored = {
        **graph,
        "parts": [first, {**second, "operator": "chamfer"}],
    }

    assert np.array_equal(ev.sample_graph_field(authored, points), expected.field)


@pytest.mark.parametrize(
    "operator",
    (ev.CompositionOperator.CHAMFER, ev.CompositionOperator.CREASE),
)
def test_neutral_regional_composition_matches_masked_gluing(operator) -> None:
    resolution = 5
    region = (slice(1, 3), slice(0, 3), slice(2, 5))
    field = np.linspace(
        -0.8,
        1.2,
        resolution**3,
        dtype=np.float64,
    ).reshape((resolution,) * 3)
    sampled = np.linspace(
        -0.4,
        0.6,
        18,
        dtype=np.float64,
    ).reshape((2, 3, 3))
    padding = engine_compile._regional_padding(region, resolution)
    padded = np.pad(sampled, padding, constant_values=0.0)
    mask = np.pad(
        np.ones(sampled.shape, dtype=bool),
        padding,
        constant_values=False,
    )
    composed = ev.compose_union(
        ev.LegacyCompositionProblem(field, padded, operator, 0.2)
    )
    assert isinstance(composed, ev.AcceptedComposition)
    expected = np.where(mask, composed.field, field)

    observed = engine_compile._compose_sampled_region(
        field,
        sampled,
        region,
        operator,
        0.2,
        resolution,
        3.0,
    )

    assert np.array_equal(observed, expected)


@pytest.mark.parametrize(
    "raw_part",
    (
        {
            "id": "plain",
            "type": "gencyl",
            "spine": [
                [-0.2, 0.0, 0.1],
                [0.0, 0.6, 0.0],
                [0.3, 1.0, -0.1],
            ],
            "radii": [0.18, 0.24, 0.12],
        },
        {
            "id": "profiled",
            "type": "gencyl",
            "spine": [
                [-0.2, 0.0, 0.1],
                [0.0, 0.6, 0.0],
                [0.3, 1.0, -0.1],
            ],
            "radii": [0.18, 0.24, 0.12],
            "profile": {
                "n": [4.0, 4.0, 4.0],
                "aspect": [0.5, 0.5, 0.5],
                "up": [0.0, 0.0, 1.0],
            },
        },
        {
            "id": "blob",
            "type": "blob",
            "center": [0.1, 0.4, -0.2],
            "size": [0.3, 0.2, 0.4],
        },
        {
            "id": "box",
            "type": "box",
            "center": [-0.1, 0.2, 0.3],
            "size": [0.3, 0.2, 0.1],
            "round": 0.04,
        },
    ),
    ids=("plain-gencyl", "profiled-gencyl", "blob", "box"),
)
def test_prepared_part_sampling_is_bit_exact(raw_part) -> None:
    decoded = decode_part(raw_part)
    assert isinstance(decoded, Accepted)
    points = _sample_points(n=1024, seed=7331)
    prepared = prepared_part_sdf(decoded.value)
    assert callable(prepared)

    assert np.array_equal(
        prepared(points),
        part_sdf_checked(points, decoded.value),
    )


def test_unknown_composition_operator_is_a_typed_obstruction() -> None:
    graph = {
        **_box_graph(),
        "parts": [
            {**_box_graph()["parts"][0], "operator": "mystery"},
            _box_graph()["parts"][1],
        ],
    }

    assert {
        (violation["part"], violation["rule"])
        for violation in ev.vocab_violations(graph, res=48)
    } == {("slab", "R-composition-operator")}


@pytest.mark.parametrize("operator", ("chamfer", "crease"))
def test_raster_field_uses_the_same_composition_operator(operator: str) -> None:
    base = _box_graph()
    graph = {
        **base,
        "parts": [base["parts"][0], {**base["parts"][1], "operator": operator}],
    }
    evaluated = ev.evaluate(graph, res=20)
    axes = tuple(
        np.linspace(lower, upper, evaluated.resolution)
        for lower, upper in zip(evaluated.lower, evaluated.upper, strict=True)
    )
    grid = np.meshgrid(*axes, indexing="ij")
    points = np.stack(tuple(axis.ravel() for axis in grid), axis=1)

    assert np.array_equal(
        evaluated.field,
        ev.sample_graph_field(graph, points).reshape(evaluated.field.shape),
    )


def test_carves_descend_after_the_ordered_union_without_changing_bounds() -> None:
    graph = _carved_graph()
    positive_graph = _box_graph()
    points = _sample_points(n=512, seed=41)
    socket = graph["carves"][0]
    positive_field = ev.sample_graph_field(positive_graph, points)
    expected = np.maximum(
        np.maximum(positive_field, -ev.part_sdf(points, socket)),
        -ev.part_sdf(points, socket, mirrored=True),
    )

    assert np.array_equal(ev.graph_bounds(graph), ev.graph_bounds(positive_graph))
    assert np.array_equal(ev.sample_graph_field(graph, points), expected)
    assert np.array_equal(
        ev.sample_graph_field({**positive_graph, "carves": []}, points),
        positive_field,
    )


def test_evaluate_checked_applies_the_same_carve_descent_as_point_sampling() -> None:
    graph = _carved_graph()
    evaluated = ev.evaluate(graph, res=32)
    axes = tuple(
        np.linspace(evaluated.lower[index], evaluated.upper[index], 32)
        for index in range(3)
    )
    grids = np.meshgrid(*axes, indexing="ij", copy=False)
    points = np.stack(grids, axis=-1).reshape((-1, 3))
    expected = ev.sample_graph_field(graph, points).reshape((32,) * 3)

    assert np.array_equal(evaluated.field, expected)


# --------------------------------------------------------------------------- #
# 3. Box analytic spot-checks -- closed-form distances to 1e-12.              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("r", [0.0, 0.1])
def test_box_sdf_matches_closed_form(r):
    """Unit box (core half-extents [1,1,1]) at origin; sample face centers,
    edge midpoints, corners and exterior offsets. Rounded distance = raw
    distance to the core box minus ``r``."""
    size = [1.0, 1.0, 1.0]
    t = 0.37  # arbitrary exterior offset
    cases = [
        # (point, raw distance to the CORE box surface)
        ([0.0, 0.0, 0.0], -1.0),                 # deep interior
        ([1.0, 0.0, 0.0], 0.0),                  # face center (on core face)
        ([1.0, 1.0, 0.0], 0.0),                  # edge midpoint (on core edge)
        ([1.0, 1.0, 1.0], 0.0),                  # corner (on core corner)
        ([1.0 + t, 0.0, 0.0], t),                # outside a face
        ([1.0 + t, 1.0 + t, 0.0], np.sqrt(2) * t),   # outside an edge
        ([1.0 + t, 1.0 + t, 1.0 + t], np.sqrt(3) * t),  # outside a corner
        ([0.5, 0.25, -0.75], -0.25),             # interior (nearest face at z=-1)
    ]
    pts = np.array([c[0] for c in cases], dtype=np.float64)
    expected = np.array([c[1] - r for c in cases], dtype=np.float64)
    got = ev.sdf_box(pts, [0.0, 0.0, 0.0], size, round=r)
    assert np.allclose(got, expected, atol=1e-12, rtol=0.0)


def test_box_sdf_via_part_dispatch_matches_direct():
    """part_sdf routes a box through sdf_box with center subtracted."""
    part = {"id": "b", "type": "box", "center": [0.2, 0.5, -0.1],
            "size": [0.3, 0.4, 0.2], "round": 0.05}
    pts = _sample_points(seed=3)
    direct = ev.sdf_box(pts, part["center"], part["size"], round=part["round"])
    routed = ev.part_sdf(pts, part)
    assert np.array_equal(direct, routed)


# --------------------------------------------------------------------------- #
# 4. Rot rotation-invariance -- rotate the part, counter-rotate the grid.     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "part",
    [
        {"id": "b", "type": "box", "center": [0.2, 0.5, -0.1],
         "size": [0.3, 0.4, 0.2], "round": 0.05},                    # no base rot
        {"id": "b", "type": "box", "center": [0.2, 0.5, -0.1],
         "size": [0.3, 0.4, 0.2], "round": 0.05,
         "rot": [0.9238795325112867, 0.0, 0.3826834323650898, 0.0]},  # 45deg yaw
        {"id": "e", "type": "blob", "center": [-0.15, 0.3, 0.25],
         "size": [0.35, 0.2, 0.28],
         "rot": [0.8446231020639574, 0.19134171618254486,
                 0.4619397662556434, 0.19134171618254486]},           # base rot
    ],
    ids=["box_norot", "box_yaw45", "blob_rot"],
)
def test_rot_rotation_invariance(part):
    """d_rotated(R_g p) == d(p): rotate the part by a fixed g and counter-rotate
    the sample grid -> identical per-point SDF to 1e-12."""
    g = ev.read_quat([0.7071067811865476, 0.5, 0.3, 0.4])  # arbitrary unit quat
    Rg = ev.quat_to_matrix(g)
    base_rot = part.get("rot")
    rotated = dict(part)
    rotated["center"] = (Rg @ np.asarray(part["center"], float)).tolist()
    rotated["rot"] = ev.quat_mul(g, ev.read_quat(base_rot) if base_rot
                                 else np.array([1.0, 0.0, 0.0, 0.0])).tolist()
    pts = _sample_points(seed=11)
    pts_rot = pts @ Rg.T  # R_g p for each row
    d0 = ev.part_sdf(pts, part)
    d1 = ev.part_sdf(pts_rot, rotated)
    assert np.allclose(d0, d1, atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# 5. Rot mirror consistency -- (w,x,-y,-z) reflection rule.                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "part",
    [
        {"id": "b", "type": "box", "center": [0.6, 0.5, -0.1],
         "size": [0.3, 0.4, 0.2], "round": 0.05,
         "rot": [0.9061274463528878, 0.25056280708573159,
                 0.3106172175509606, 0.13529902503654924]},
        {"id": "e", "type": "blob", "center": [0.45, 0.3, 0.25],
         "size": [0.35, 0.2, 0.28],
         "rot": [0.8446231020639574, 0.19134171618254486,
                 0.4619397662556434, 0.19134171618254486]},
    ],
    ids=["box", "blob"],
)
def test_rot_mirror_consistency(part):
    """The mirrored oriented instance sampled at x-reflected points equals the
    original: part_sdf(part, p) == part_sdf(part, M p, mirrored=True), M =
    diag(-1,1,1). Verifies center.x -> -center.x AND (w,x,y,z)->(w,x,-y,-z)."""
    pts = _sample_points(seed=5)
    reflected = pts * np.array([-1.0, 1.0, 1.0])
    d_orig = ev.part_sdf(pts, part, mirrored=False)
    d_mirror = ev.part_sdf(reflected, part, mirrored=True)
    assert np.allclose(d_orig, d_mirror, atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# 6. rot on gencyl -> reported violation, never applied.                      #
# --------------------------------------------------------------------------- #
def test_rot_on_gencyl_reported_and_not_applied():
    rot = [0.9238795325112867, 0.0, 0.3826834323650898, 0.0]
    gencyl_rot = {"id": "arm", "type": "gencyl",
                  "spine": [[0.2, 1.0, 0.0], [0.6, 0.5, 0.2]],
                  "radii": [0.15, 0.1], "rot": rot}
    gencyl_plain = {k: v for k, v in gencyl_rot.items() if k != "rot"}
    graph = {"name": "g", "blend": 0.04, "parts": [gencyl_rot]}

    # Reported as a FENCE violation.
    viols = ev.vocab_violations(graph, res=64)
    rules = {(v["part"], v["rule"]) for v in viols}
    assert ("arm", "FENCE-rot-on-gencyl") in rules

    # SDF path must NOT apply the rotation: identical to the plain gencyl.
    pts = _sample_points(seed=9)
    d_rot = ev.part_sdf(pts, gencyl_rot)
    d_plain = ev.part_sdf(pts, gencyl_plain)
    assert np.array_equal(d_rot, d_plain)
    # And identical to the frozen engine's own gencyl evaluation.
    assert np.array_equal(d_rot, engine.part_sdf(pts, gencyl_plain))


def test_nonunit_quat_reported_but_used():
    part = {"id": "b", "type": "box", "center": [0, 0.5, 0],
            "size": [0.3, 0.3, 0.3], "round": 0.02, "rot": [2.0, 0.0, 0.0, 0.0]}
    graph = {"name": "g", "blend": 0.03, "parts": [part]}
    viols = ev.vocab_violations(graph, res=64)
    assert any(v["rule"] == "R-quat-nonunit" for v in viols)
    # |q|=2 is the identity direction once normalized -> behaves like no rotation.
    pts = _sample_points(seed=2)
    d = ev.part_sdf(pts, part)
    d_identity = ev.sdf_box(pts, part["center"], part["size"], round=part["round"])
    assert np.allclose(d, d_identity, atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# 7. Feature-size check -- a sliver box trips both R-8 violations.            #
# --------------------------------------------------------------------------- #
def test_sliver_box_triggers_both_feature_size_violations():
    graph = {
        "name": "sliver",
        "blend": 0.02,
        "parts": [{"id": "sliver", "type": "box", "center": [0, 0.4, 0],
                   "size": [0.4, 0.4, 0.002], "round": 0.0}],
    }
    res = 64
    viols = [v for v in ev.vocab_violations(graph, res=res) if v["part"] == "sliver"]
    rules = {v["rule"] for v in viols}
    assert "R-8-feature-size" in rules, f"missing feature-size in {rules}"
    assert "R-8-sharp-edge" in rules, f"missing sharp-edge in {rules}"


def test_carve_primitives_contribute_authored_and_derived_obstructions() -> None:
    graph = {
        "name": "carve_obstruction",
        "parts": [
            {
                "id": "host",
                "type": "blob",
                "center": [0.0, 0.0, 0.0],
                "size": [1.0, 1.0, 1.0],
            },
        ],
        "carves": [
            {
                "id": "socket",
                "type": "box",
                "center": [0.2, 0.0, 0.8],
                "size": [0.2, 0.2, 0.002],
                "round": 0.0,
                "rot": [2.0, 0.0, 0.0, 0.0],
            },
        ],
    }
    rules = {
        violation["rule"]
        for violation in ev.vocab_violations(graph, res=64)
        if violation["part"] == "socket"
    }

    assert {
        "R-quat-nonunit",
        "R-8-feature-size",
        "R-8-sharp-edge",
    } <= rules


def test_well_formed_box_has_no_feature_size_violations():
    """The greatsword's boxes are thick enough at res 130 to be clean."""
    graph = json.loads((_GOLDEN / "greatsword_v1.json").read_text())
    assert ev.vocab_violations(graph, res=130) == []


# --------------------------------------------------------------------------- #
# Golden artifact: greatsword_v1.                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_greatsword_golden():
    """Standalone planted greatsword (res 130): 1 watertight component, bbox at
    the analytic TOTAL extents (core + round), flat blade cross-section."""
    graph = json.loads((_GOLDEN / "greatsword_v1.json").read_text())
    res = 130
    evaluated = ev.evaluate(graph, res=res)
    verts, faces = evaluated.vertices, evaluated.faces
    pitch = evaluated.maximum_pitch

    rep = ev.coherence_report(verts, faces)
    assert rep["components"] == 1
    assert rep["watertight_main"] is True

    # TOTAL (round-inclusive) extents: x = 2*(0.34+0.03)=0.74,
    # y = pommel_top(1.76) - blade_bottom(0.66-0.66=0.0) = 1.76,
    # z = 2*(0.07+0.03)=0.20. (Core-only x/z would be 0.68 / 0.14.)
    expected_total = np.array([0.74, 1.76, 0.20])
    extents = verts.max(axis=0) - verts.min(axis=0)
    assert np.allclose(extents, expected_total, atol=pitch), (
        f"extents {extents} vs expected {expected_total} (pitch {pitch})"
    )

    # Flat blade: the box core cross-section is width(x)/thickness(z) >= 4 --
    # a flat profile a round gencyl cannot express.
    blade = next(p for p in graph["parts"] if p["id"] == "blade")
    flatness = blade["size"][0] / blade["size"][2]
    assert flatness >= 4.0, f"blade flatness {flatness} < 4"


# --------------------------------------------------------------------------- #
# Golden artifact: lean_marker (rotated box).                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_marker_golden():
    """Single yaw-then-pitch box (res 112): the 8 stored world corners are exact
    SDF zero-crossings (round=0), a mesh vertex lands within one pitch of each,
    and the mesh is a single watertight component. Also re-derives the corners
    from (center,size,rot) to validate the stored values."""
    graph = json.loads((_GOLDEN / "lean_marker.json").read_text())
    part = graph["parts"][0]
    stored = np.array(graph["_expected_corners"], dtype=np.float64)
    assert stored.shape == (8, 3)

    # Re-derive corners from the schema and confirm the stored ones.
    R = ev.quat_to_matrix(ev.read_quat(part["rot"]))
    c = np.asarray(part["center"], float)
    s = np.asarray(part["size"], float)
    signs = np.array([(sx, sy, sz) for sx in (-1, 1)
                      for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    derived = c + (signs * s) @ R.T
    assert np.allclose(derived, stored, atol=1e-12, rtol=0.0)

    # round=0 -> each stored corner is an exact zero-crossing of the SDF.
    sdf_at_corners = ev.part_sdf(stored, part)
    assert np.allclose(sdf_at_corners, 0.0, atol=1e-12)

    res = 112
    evaluated = ev.evaluate(graph, res=res)
    verts, faces = evaluated.vertices, evaluated.faces
    pitch = evaluated.maximum_pitch

    # A meshed zero-crossing (vertex) lands within one grid pitch of each corner.
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(verts).query(stored)
    assert (dist <= pitch).all(), f"corner-to-vertex dists {dist} > pitch {pitch}"

    rep = ev.coherence_report(verts, faces)
    assert rep["components"] == 1
    assert rep["watertight_main"] is True
