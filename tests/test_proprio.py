"""Tests for the M2 proprioception stack (golem/senses/proprio.py).

Fast unit tests run by default; the one test that meshes (to CALIBRATE the
mesh-free component predictor against real marching-cubes connectivity) is
marked ``slow``. Meshing lives ONLY in that test -- proprio itself never meshes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from golem import paths as _paths
import engine
from golem import goldentext
from golem.kernel import engine as kernel_engine
from golem.kernel.engine.types import (
    Accepted,
    GencylPart,
    GeometryObstruction,
    IntegralRadiusUnsatisfied,
    MuscleFormationEvidence,
    MuscleFormationObstruction,
    decode_part,
)
from golem.senses import proprio
from golem.senses.model import RejectedSenses, Senses
from golem.senses.proprio.geometry import _part_shape, _part_volume_centroid

_KNIGHT = _paths.REHEARSAL / "knight" / "golem_knight.json"
_INTENT = _paths.GOLDEN / "knight_intent.json"


def _knight():
    return json.loads(_KNIGHT.read_text())


def _intent():
    return json.loads(_INTENT.read_text())


def _formed_proprio_part(
    *,
    mirror: bool = False,
    profile: dict[str, object] | None = None,
    formation: str = "skeleton_integral",
) -> dict[str, object]:
    return {
        "id": "formed",
        "type": "gencyl",
        "spine": ((0.2, 0.0, 0.0), (0.2, 1.0, 0.0), (0.2, 2.0, 0.0)),
        "radii": (0.2, 0.3, 0.2),
        "formation": formation,
        **({"mirror": True} if mirror else {}),
        **({"profile": profile} if profile is not None else {}),
    }


def _formation_evidence(
    part: dict[str, object],
    *,
    mirrored: bool = False,
) -> MuscleFormationEvidence:
    decoded = decode_part(part)

    assert isinstance(decoded, Accepted)
    assert isinstance(decoded.value, GencylPart)
    evidence = kernel_engine.solve_muscle_formation(
        decoded.value,
        mirrored=mirrored,
    )
    assert isinstance(evidence, MuscleFormationEvidence)
    return evidence


def _formed_surface_bounds(
    evidence: MuscleFormationEvidence,
) -> tuple[np.ndarray, np.ndarray]:
    origins = np.asarray(tuple(frame.origin for frame in evidence.frames))
    radii = np.asarray(
        (
            evidence.intervals[0].lower_radius,
            *(interval.upper_radius for interval in evidence.intervals),
        )
    )[:, None]
    return np.min(origins - radii, axis=0), np.max(origins + radii, axis=0)


# --------------------------------------------------------------------------- #
# Gap classification / burial on two analytically-known spheres.               #
# --------------------------------------------------------------------------- #
def _two_spheres(r1, r2, dist):
    return {
        "name": "two_spheres", "blend": 0.05,
        "parts": [
            {"id": "a", "type": "blob", "center": [0, 0, 0], "size": [r1, r1, r1]},
            {"id": "b", "type": "blob", "center": [float(dist), 0, 0],
             "size": [r2, r2, r2]},
        ],
    }


def test_gap_matches_analytic_sphere_distance():
    # Separated spheres close enough to survive the bbox prefilter: analytic
    # surface gap = dist - r1 - r2 (an upper bound; sampling error ~h^2/8R).
    senses = proprio.build_senses(_two_spheres(0.3, 0.3, 0.8), None)
    gap = senses.inst_gaps[("a", "b")]
    assert gap == pytest.approx(0.8 - 0.6, abs=0.01)  # 0.2


def test_gap_classification_bands():
    # Overlapping -> FUSED (gap < 0).
    s = proprio.build_senses(_two_spheres(0.3, 0.2, 0.4), None)
    assert s.inst_gaps[("a", "b")] < 0  # dist 0.4 < 0.5 sum of radii
    # Far apart survives prefilter only if within 3*k; place beyond -> unreported.
    s2 = proprio.build_senses(_two_spheres(0.3, 0.3, 5.0), None)
    assert ("a", "b") not in s2.inst_gaps  # bbox-prefiltered away


def test_burial_depth_formula():
    # Smaller sphere (r=0.2) overlapping larger (r=0.3), centers 0.4 apart.
    # analytic gap = 0.4 - 0.5 = -0.1; burial = -gap / min_r = 0.1/0.2 = 0.5.
    s = proprio.build_senses(_two_spheres(0.3, 0.2, 0.4), None)
    gap = s.inst_gaps[("a", "b")]
    min_r = min(s.inst_data["a"].min_r, s.inst_data["b"].min_r)
    burial = -gap / min_r
    assert burial == pytest.approx(0.5, abs=0.05)


def _girdle_graph():
    # torso--girdle--limb, collinear and overlapping: torso~limb is a distance-2
    # girdle (girdle fused to both ends, torso fused to limb past it).
    return {
        "name": "fused_girdle",
        "blend": 0.05,
        "parts": [
            {"id": "torso", "type": "blob", "center": [0, 0, 0],
             "size": [0.3, 0.3, 0.3]},
            {"id": "girdle", "type": "blob", "center": [0.2, 0, 0],
             "size": [0.3, 0.3, 0.3]},
            {"id": "limb", "type": "blob", "center": [0.4, 0, 0],
             "size": [0.3, 0.3, 0.3]},
        ],
    }


def _dets_for(senses, pair):
    return tuple(
        anomaly["det"]
        for anomaly in proprio.detect_anomalies(senses)
        if anomaly["pair"] == pair
    )


def test_declared_girdle_attachment_chain_is_articulated_fusion():
    # A distance-2 girdle DECLARED as an expected articulation softens to
    # articulated_fusion. Declaration -- an authorial design intention -- is the
    # discriminator; the girdle shape alone never absolves.
    senses = proprio.build_senses(
        _girdle_graph(),
        {
            "attach": (("torso", "girdle"), ("girdle", "limb")),
            "expected_articulations": (("torso", "limb"),),
        },
    )

    assert "girdle" in senses.fused_adj["torso"]
    assert "limb" in senses.fused_adj["girdle"]
    assert _dets_for(senses, ("limb", "torso")) == ("articulated_fusion",)


def test_undeclared_and_far_fusions_remain_unintended():
    # 1) The SAME distance-2 girdle, but UNDECLARED: with no expected_articulation
    # the girdle shape does not soften it -- it stays unintended_fusion. This is
    # the exact shape of the knight's chin-on-chest (visor_ridge~torso through a
    # fused helm), which must remain a defect.
    undeclared = proprio.build_senses(
        _girdle_graph(),
        {"attach": (("torso", "girdle"), ("girdle", "limb"))},
    )
    assert _dets_for(undeclared, ("limb", "torso")) == ("unintended_fusion",)

    # 2) A fully-fused DECLARED chain three attachments long (a-b-c-d, every
    # consecutive link fused). Its endpoints sit at declared-tree distance 3 --
    # past the girdle pattern -- so they stay unintended even if one tried to
    # declare them. Distance is a necessary structural gate; declaration is the
    # sufficient one.
    chained = proprio.build_senses(
        {
            "name": "fused_chain",
            "blend": 0.05,
            "parts": [
                {"id": "a", "type": "blob", "center": [0.00, 0, 0],
                 "size": [0.3, 0.3, 0.3]},
                {"id": "b", "type": "blob", "center": [0.15, 0, 0],
                 "size": [0.3, 0.3, 0.3]},
                {"id": "c", "type": "blob", "center": [0.30, 0, 0],
                 "size": [0.3, 0.3, 0.3]},
                {"id": "d", "type": "blob", "center": [0.45, 0, 0],
                 "size": [0.3, 0.3, 0.3]},
            ],
        },
        {
            "attach": (("a", "b"), ("b", "c"), ("c", "d")),
            "expected_articulations": (("a", "d"),),
        },
    )
    assert "b" in chained.fused_adj["a"]
    assert "d" in chained.fused_adj["c"]
    assert _dets_for(chained, ("a", "d")) == ("unintended_fusion",)

    # 3) No declared attachment at all: the fused pair has no intermediary to
    # articulate through, so it stays unintended_fusion.
    separate = proprio.build_senses(
        {
            "name": "separate_limbs",
            "blend": 0.05,
            "parts": [
                {"id": "left_limb", "type": "blob", "center": [0, 0, 0],
                 "size": [0.3, 0.3, 0.3]},
                {"id": "right_limb", "type": "blob", "center": [0.4, 0, 0],
                 "size": [0.3, 0.3, 0.3]},
            ],
        },
        None,
    )
    assert separate.attach == ()
    assert _dets_for(separate, ("left_limb", "right_limb")) == (
        "unintended_fusion",
    )


# --------------------------------------------------------------------------- #
# Declaration channels: blend-reach exemption + midline.                        #
# --------------------------------------------------------------------------- #
def test_declared_pair_within_blend_reach_is_declared_fusion():
    # Two spheres 0.7 apart (gap ~0.1) under a fat blend kernel (k >> gap):
    # the undeclared near-miss reads blend_ambiguity -- the honest negative, a
    # REAL observation -- but the SAME geometry with the pair declared is
    # declared fusion: the declaration states the blend union the kernel
    # bridges at mesh, so neither blend_ambiguity nor missing_fusion fires.
    graph = {**_two_spheres(0.3, 0.3, 0.7), "blend": 0.5}
    undeclared = proprio.build_senses(graph, None)
    assert _dets_for(undeclared, ("a", "b")) == ("blend_ambiguity",)
    declared = proprio.build_senses(graph, {"attach": (("a", "b"),)})
    assert _dets_for(declared, ("a", "b")) == ()


def test_declared_pair_beyond_blend_reach_still_missing_fusion():
    # The exemption is reach-bounded, never a blanket silencer: a pair
    # declared but truly out of blend reach (a void the kernel cannot close)
    # still reports missing_fusion.
    senses = proprio.build_senses(
        _two_spheres(0.3, 0.3, 1.0), {"attach": (("a", "b"),)}
    )
    assert _dets_for(senses, ("a", "b")) == ("missing_fusion",)


def test_crease_then_blend_uses_incoming_blend_reach():
    graph = _two_spheres(0.3, 0.3, 0.7)
    graph["blend"] = 0.5
    graph["parts"][0]["operator"] = "crease"
    graph["parts"][1]["operator"] = "blend"
    senses = proprio.build_senses(
        graph, {"attach": (("a", "b"),)}
    )
    assert senses.inst_data["a"].blend_r == senses.inst_data["b"].blend_r > 0.0
    assert ("a", "b") in senses.inst_gaps
    assert _dets_for(senses, ("a", "b")) == ()


def test_blend_then_crease_has_zero_incoming_reach():
    graph = _two_spheres(0.3, 0.3, 0.7)
    graph["blend"] = 0.5
    graph["parts"][0]["operator"] = "blend"
    graph["parts"][1]["operator"] = "crease"
    senses = proprio.build_senses(
        graph, {"attach": (("a", "b"),)}
    )
    assert senses.inst_data["a"].blend_r == senses.inst_data["b"].blend_r > 0.0
    assert ("a", "b") not in senses.inst_gaps
    assert _dets_for(senses, ("a", "b")) == ("missing_fusion",)


def test_local_blend_requires_iso_overlap_instead_of_bridging_at_distance():
    graph = _two_spheres(0.3, 0.3, 0.7)
    graph["blend"] = 0.5
    graph["parts"][1]["operator"] = "local_blend"
    senses = proprio.build_senses(
        graph, {"attach": (("a", "b"),)}
    )
    assert senses.inst_data["a"].blend_r == senses.inst_data["b"].blend_r > 0.0
    assert ("a", "b") not in senses.inst_gaps
    assert _dets_for(senses, ("a", "b")) == ("missing_fusion",)


def test_global_blend_scale_reports_part_overrides():
    graph = _two_spheres(0.3, 0.3, 0.7)
    graph["parts"][0]["blend"] = 0.01
    graph["parts"][1]["operator"] = "crease"
    senses = proprio.build_senses(graph, None)
    receipt = proprio.render_receipt(senses, [], [], pack="override-probe")
    assert (
        f"k={senses.global_dims.k:.3f} (global; 2 part override(s))"
        in receipt
    )


def _midline_graph():
    # mirror:true blob straddling x=0: the .L/.R instances overlap across the
    # plane -- deliberately-symmetric anatomy.
    return {
        "name": "midline_probe", "blend": 0.05,
        "parts": [
            {"id": "haunch", "type": "blob", "center": [0.05, 0, 0],
             "size": [0.15, 0.15, 0.15], "mirror": True},
        ],
    }


def test_undeclared_midline_crossing_is_cross_plane_fusion():
    # Honest negative: an undeclared mirror crossing stays a visible anomaly.
    senses = proprio.build_senses(_midline_graph(), None)
    crossing = [
        anomaly
        for anomaly in proprio.detect_anomalies(senses)
        if anomaly["det"] == "cross_plane_fusion"
    ]
    assert len(crossing) == 1
    assert crossing[0]["addr"] == "haunch.L~haunch.R"
    assert not crossing[0]["expected"]
    _suffix, visible, _suppressed = proprio._apply_suppression((), crossing)
    assert len(visible) == 1


def test_declared_midline_crossing_reports_as_expected():
    # The declaration keeps the observation (the geometry DOES cross the
    # plane) but marks it intent: expected, hence absent from the visible set.
    senses = proprio.build_senses(_midline_graph(), {"midline": ("haunch",)})
    crossing = [
        anomaly
        for anomaly in proprio.detect_anomalies(senses)
        if anomaly["det"] == "cross_plane_fusion"
    ]
    assert len(crossing) == 1
    assert crossing[0]["expected"]
    _suffix, visible, _suppressed = proprio._apply_suppression((), crossing)
    assert tuple(visible) == ()


# --------------------------------------------------------------------------- #
# Bands interval-union.                                                         #
# --------------------------------------------------------------------------- #
def test_union_len_disjoint_and_overlapping():
    assert proprio._union_len([(0.0, 1.0), (2.0, 3.0)]) == pytest.approx(2.0)
    assert proprio._union_len([(0.0, 2.0), (1.0, 3.0)]) == pytest.approx(3.0)
    assert proprio._union_len([]) == 0.0
    assert proprio._union_len([(-1.0, 1.0)]) == pytest.approx(2.0)


def test_bands_shoulder_shelf_present():
    # The knight's front-width column must jump at the shoulder shelf (the
    # 'swallowed helm' signal): a slab with width ~1.96 sits directly below the
    # narrow head slab (~0.5).
    s = proprio.build_senses(_knight(), _intent())
    widths = [round(b.width, 2) for b in s.bands]  # bottom..top
    assert max(widths) == pytest.approx(1.96, abs=0.02)
    assert min(widths) < 0.7  # the head slab


# --------------------------------------------------------------------------- #
# Analytic centroid vs numeric integration (single simple part).               #
# --------------------------------------------------------------------------- #
def test_formed_proprio_views_descend_from_formation_evidence():
    part = _formed_proprio_part()
    evidence = _formation_evidence(part)
    senses = proprio.build_senses(
        {"name": "formed-proprio", "blend": 0.01, "parts": (part,)},
        None,
    )

    assert isinstance(senses, Senses)
    datum = senses.inst_data["formed"]
    sampled_field = kernel_engine.part_sdf(datum.samples, part)
    expected_lower, expected_upper = _formed_surface_bounds(evidence)
    assert isinstance(sampled_field, np.ndarray)
    assert np.max(np.abs(sampled_field)) == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_array_equal(datum.lo, expected_lower)
    np.testing.assert_array_equal(datum.hi, expected_upper)
    assert np.any(datum.lo > evidence.contact_lower)
    assert np.any(datum.hi < evidence.contact_upper)
    assert datum.min_r == min(
        min(interval.lower_radius, interval.upper_radius)
        for interval in evidence.intervals
    )
    solved_volume, solved_centroid = _part_volume_centroid(evidence)
    assert solved_volume == evidence.solved_volume
    np.testing.assert_array_equal(solved_centroid, evidence.centroid)
    np.testing.assert_allclose(
        senses.global_dims.centroid,
        evidence.centroid,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_formed_proprio_glues_mirror_evidence_by_instance():
    part = _formed_proprio_part(mirror=True)
    left_evidence = _formation_evidence(part)
    right_evidence = _formation_evidence(part, mirrored=True)
    senses = proprio.build_senses(
        {"name": "formed-mirror-proprio", "blend": 0.01, "parts": (part,)},
        None,
    )

    assert isinstance(senses, Senses)
    left = senses.inst_data["formed.L"]
    right = senses.inst_data["formed.R"]
    left_field = kernel_engine.part_sdf(left.samples, part)
    right_field = kernel_engine.part_sdf(right.samples, part, mirrored=True)
    left_lower, left_upper = _formed_surface_bounds(left_evidence)
    right_lower, right_upper = _formed_surface_bounds(right_evidence)
    assert isinstance(left_field, np.ndarray)
    assert isinstance(right_field, np.ndarray)
    assert np.max(np.abs(left_field)) == pytest.approx(0.0, abs=1.0e-12)
    assert np.max(np.abs(right_field)) == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_array_equal(left.lo, left_lower)
    np.testing.assert_array_equal(left.hi, left_upper)
    np.testing.assert_array_equal(right.lo, right_lower)
    np.testing.assert_array_equal(right.hi, right_upper)
    np.testing.assert_array_equal(
        right_evidence.centroid,
        np.asarray(left_evidence.centroid) * (-1.0, 1.0, 1.0),
    )
    np.testing.assert_allclose(
        senses.global_dims.centroid,
        (0.0, 1.0, 0.0),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_formed_proprio_rejects_unsatisfied_radius_without_fallback():
    part = _formed_proprio_part(
        profile={
            "n": 2.0,
            "depth": (0.3, 0.3, 0.3),
            "up": (1.0, 0.0, 0.0),
        }
    )
    result = proprio.build_senses(
        {"name": "rejected-formed-proprio", "blend": 0.01, "parts": (part,)},
        None,
    )

    assert isinstance(result, RejectedSenses)
    assert len(result.obstructions) == 1
    obstruction = result.obstructions[0]
    assert isinstance(obstruction, MuscleFormationObstruction)
    assert isinstance(obstruction.failure, IntegralRadiusUnsatisfied)
    assert obstruction.failure.residual > obstruction.failure.tolerance


def test_formed_proprio_preserves_decode_obstruction():
    part = _formed_proprio_part(formation="numerical_guesswork")
    result = proprio.build_senses(
        {"name": "malformed-formed-proprio", "blend": 0.01, "parts": (part,)},
        None,
    )

    assert isinstance(result, RejectedSenses)
    assert len(result.obstructions) == 1
    assert isinstance(result.obstructions[0], GeometryObstruction)
    assert result.obstructions[0].detail.endswith(
        "got 'numerical_guesswork'"
    )


def test_centroid_analytic_vs_numeric_frustum():
    # A tapered vertical gencyl frustum; integrate its interior numerically and
    # compare the y-centroid to the analytic frustum-sum.
    part = {"id": "c", "type": "gencyl",
            "spine": [[0, 0, 0], [0, 1.0, 0]], "radii": [0.3, 0.15]}
    v, cen = proprio._part_volume_centroid(_part_shape(part, False))
    # numeric: sample a dense grid, keep interior points (sdf<0), average.
    n = 90
    xs = np.linspace(-0.4, 0.4, n)
    ys = np.linspace(-0.1, 1.1, n)
    zs = np.linspace(-0.4, 0.4, n)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    d = engine.part_sdf(pts, part)
    inside = pts[d < 0]
    y_numeric = inside[:, 1].mean()
    # analytic ignores the spherical caps; a frustum leans toward the fat end.
    assert cen[1] == pytest.approx(y_numeric, abs=0.05)
    assert cen[1] < 0.5  # fat end at y=0 pulls the centroid below the midpoint


# --------------------------------------------------------------------------- #
# Mirror compression: 16 rows, 22 instances on the knight.                     #
# --------------------------------------------------------------------------- #
def test_mirror_compression_row_and_instance_count():
    s = proprio.build_senses(_knight(), _intent())
    assert len(s.parts) == 16       # one row per declared part
    assert s.n_instances == 22      # 10 single + 6 mirrored*2
    assert s.n_mirror == 6


# --------------------------------------------------------------------------- #
# Fused-adjacency predicted component count (mesh-free) == actual (meshed).     #
# This single test MAY mesh at low res to calibrate the predictor.             #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_component_prediction_matches_meshed_connectivity():
    fused = _two_spheres(0.3, 0.3, 0.45)   # overlapping -> one blob
    split = _two_spheres(0.3, 0.3, 1.4)    # apart -> two blobs
    for graph, expected in ((fused, 1), (split, 2)):
        s = proprio.build_senses(graph, None)
        assert s.n_components == expected, "predictor"
        verts, faces, *_ = engine.evaluate(graph, res=64)
        actual = engine.coherence_report(verts, faces)["components"]
        assert actual == expected, "meshed"
        assert s.n_components == actual  # calibration


def test_knight_predicted_single_component():
    s = proprio.build_senses(_knight(), _intent())
    assert s.n_components == 1


# --------------------------------------------------------------------------- #
# Suppression rule: an anomaly on a failing assert's pair folds onto it.        #
# --------------------------------------------------------------------------- #
def test_suppression_folds_anomaly_into_failing_assert():
    assert_results = [
        {"id": "A2", "status": "fail", "part_scopes": ["helm", "torso"]},
    ]
    anomalies = [
        {"det": "deep_burial", "pair": ("helm", "torso"), "gap": -0.234,
         "mag": 0.92, "detail": "x", "addr": "helm~torso", "expected": True},
        {"det": "unintended_fusion", "pair": ("grip", "blade"), "gap": -0.06,
         "mag": 1.0, "detail": "y", "addr": "grip~blade"},
    ]
    suffix, visible, n_supp = proprio._apply_suppression(assert_results, anomalies)
    assert n_supp == 1
    assert "A2" in suffix and "burial" in suffix["A2"][0]
    assert [a["addr"] for a in visible] == ["grip~blade"]  # the uncovered one


def test_expected_declared_burial_dropped_when_uncovered():
    # A declared deep fusion with no failing assert covering it is intended:
    # it is neither shown nor counted as suppressed.
    assert_results = []
    anomalies = [
        {"det": "deep_burial", "pair": ("torso", "chest_plate"), "gap": -0.35,
         "mag": 1.7, "detail": "x", "addr": "torso~chest_plate", "expected": True},
    ]
    suffix, visible, n_supp = proprio._apply_suppression(assert_results, anomalies)
    assert tuple(visible) == () and n_supp == 0 and dict(suffix) == {}


# --------------------------------------------------------------------------- #
# Overflow rule: anomalies past the cap collapse to '+N more'.                  #
# --------------------------------------------------------------------------- #
def test_anomaly_overflow_rule():
    # Under the enriched intent the knight declares every intended fused join,
    # so the visible anomalies land at exactly the cap (12): all are shown as
    # N1..N12 with no N13 and, crucially, NOTHING is hidden in overflow -- the
    # two real defects that used to sit at N12 / in '+6 more' (the gauntlet
    # cross-plane lump and chin-on-chest visor_ridge~torso) are both surfaced.
    s, ar, an = proprio.run(_knight(), _intent())
    receipt = proprio.render_receipt(s, ar, an, pack="knight.intent")
    assert " N12 " in receipt and " N13 " not in receipt
    assert "more -- proprio(anomalies)" not in receipt
    assert "cross_plane_fusion @ gauntlet.L~gauntlet.R" in receipt
    assert "unintended_fusion @ torso~visor_ridge" in receipt

    # The overflow MECHANISM itself (independent of any one creature): more than
    # _ANOMALY_CAP visible anomalies collapse the tail to '+N more'.
    many = [
        {"det": "blend_ambiguity", "pair": (f"a{i}", f"b{i}"), "gap": 0.01,
         "mag": 1.0 - 0.01 * i, "detail": "synthetic", "addr": f"a{i}~b{i}",
         "expected": False}
        for i in range(proprio._ANOMALY_CAP + 3)
    ]
    over = proprio.render_receipt(s, [], many, pack="knight.intent")
    assert " N12 " in over and " N13 " not in over
    assert "+3 more -- proprio(anomalies)" in over


# --------------------------------------------------------------------------- #
# Replay golden: pushed receipt is byte-exact, <=600 tokens, names the three    #
# losses by rule id, and contains NO pull tables.                              #
# --------------------------------------------------------------------------- #
def test_replay_receipt_golden_and_budget():
    s, ar, an = proprio.run(_knight(), _intent())
    receipt = proprio.render_receipt(s, ar, an, pack="knight.intent")

    # byte-exact against the frozen golden (write-once).
    goldentext.assert_matches_golden("knight_proprio.txt", receipt)

    # token budget (Decision-Log: tokens = ceil(chars/4)).
    assert proprio.est_tokens(receipt) <= 600

    # the three losses are NAMED by rule id with measured values + addresses.
    assert "A2 neck_gap @ helm~torso" in receipt        # hunch (burial)
    assert "A3 head_exposure @ helm~pauldron" in receipt  # swallowed helm
    assert "A4 span @ pauldron" in receipt                # swallowed helm
    assert "A5 plant_offset" in receipt                   # cane-sword
    assert "A6 sword_length" in receipt                   # cane-sword
    fails = [r for r in ar if r["status"] == "fail"]
    passes = [r for r in ar if r["status"] == "pass"]
    assert len(fails) == 6 and len(passes) == 1

    # tables are PULL-ONLY: none of them appear in the pushed receipt.
    for table in ("PARTS (", "FUSION (", "GROUND (", "BALANCE ", "BANDS ("):
        assert table not in receipt


def test_default_cli_output_has_no_tables_but_sections_do(tmp_path):
    # default render == receipt (no tables); each --section renders its table.
    s, ar, an = proprio.run(_knight(), _intent())
    receipt = proprio.render_receipt(s, ar, an, pack="knight.intent")
    assert "PARTS (" not in receipt
    assert proprio.render_section(s, "parts").startswith("PARTS (")
    assert proprio.render_section(s, "fusion").startswith("FUSION (")
    assert proprio.render_section(s, "bands").startswith("BANDS (")


# --------------------------------------------------------------------------- #
# Determinism: two runs are byte-identical.                                    #
# --------------------------------------------------------------------------- #
def test_determinism_two_runs_identical():
    r1 = proprio.render_receipt(*proprio.run(_knight(), _intent()), pack="knight.intent")
    r2 = proprio.render_receipt(*proprio.run(_knight(), _intent()), pack="knight.intent")
    assert r1 == r2


# --------------------------------------------------------------------------- #
# Sketch: opt-in only, cell-aspect header + nonempty raster, not in default.    #
# --------------------------------------------------------------------------- #
def test_sketch_header_and_raster():
    s = proprio.build_senses(_knight(), _intent())
    sketch = proprio.render_sketch(s, "front")
    lines = sketch.splitlines()
    assert lines[0].startswith("front, ")
    assert "cell" in lines[0] and "NOT square" in lines[0]
    body = "\n".join(lines[1:])
    assert "#" in body                       # nonempty raster
    assert len(lines) == 1 + proprio._SKETCH_ROWS

    # the sketch is NEVER in the pushed receipt.
    receipt = proprio.render_receipt(*proprio.run(_knight(), _intent()), pack="knight.intent")
    assert "NOT square" not in receipt
