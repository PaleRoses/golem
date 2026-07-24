"""Tests for the M2 postural-contract layer (golem/contracts/asserts.py).

asserts never measures: every case feeds a hand-built senses record (or the real
knight senses from proprio) and checks the comparison, the failure record shape
(fable-a sec.4), operators incl. ``within``, the ``unmeasurable`` status, and
single-dimension gradients (R-14).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from golem import paths as _paths
from golem.contracts import asserts
from golem.senses import proprio
from golem.senses.model import GlobalDims, PartDatum, Senses, Supported

_KNIGHT = _paths.REHEARSAL / "knight" / "golem_knight.json"
_INTENT = _paths.GOLDEN / "knight_intent.json"


def _mini_senses() -> Senses:
    # A hand-built senses record exercising the assert arithmetic in isolation.
    # Load-bearing here: parts, ground, declared_pair_gap, landmarks. The balance
    # union preserves the fixture's stated support (its margins are unread by
    # these clauses); every remaining field is neutral filler no tested metric
    # touches.
    return Senses(
        name="mini", txn=0, n_parts=2, n_mirror=1, n_instances=3,
        global_dims=GlobalDims(W=1.0, H=2.905, D=1.0, HW=2.905,
                               centroid=np.zeros(3), k=0.05),
        raw_lo=np.array([-0.98, 1.96, -0.31]),
        raw_hi=np.array([0.98, 2.92, 0.41]),
        parts={
            "pauldron": PartDatum(bbox_lo=np.array([-0.98, 1.96, -0.31]),
                                  bbox_hi=np.array([0.98, 2.56, 0.41]),
                                  mirror=True,
                                  instances=["pauldron.L", "pauldron.R"],
                                  min_r=0.30),
            "helm": PartDatum(bbox_lo=np.array([-0.27, 2.28, -0.16]),
                              bbox_hi=np.array([0.27, 2.92, 0.38]),
                              mirror=False, instances=["helm"], min_r=0.255),
        },
        inst_data={}, inst_gaps={},
        declared_pair_gap={("helm", "torso"): -0.234},
        fusion_rows={}, cross_plane={}, n_components=1, fused_adj={},
        ground={"blade": 0.003}, near_ground={}, ground_decl=[], ground_tol=0.02,
        balance=Supported(hull_x=(-0.98, 0.98), hull_z=(-0.31, 0.41),
                          centroid_xz=(0.0, 0.0), margin_x=0.5, margin_z=0.16,
                          margin_z_fwd=0.16, margin_z_aft=0.16, inside=True),
        bands=[],
        landmarks={"pommel": [0, 1.60, 0.74], "blade_tip": [0, 0.095, 0.76],
                   "sword_plant": [0, 0.023, 0.76], "toe_line": [0, 0.02, 0.41]},
        schematic={}, attach=[], provenance={}, anatomy=None, graph={},
    )


# --------------------------------------------------------------------------- #
# Operators, including the banded 'within'.                                     #
# --------------------------------------------------------------------------- #
def test_operator_maximum_fail_and_magnitude():
    s = _mini_senses()
    c = {"id": "A4", "scope": ["part:pauldron"], "metric": "span",
         "operator": "maximum", "value": 1.60, "unit": "world_unit"}
    r = asserts.evaluate_clause(c, s)
    assert r["status"] == "fail"
    assert r["measured"] == pytest.approx(1.96, abs=1e-6)
    # magnitude = relative overshoot (measured-bound)/bound (fable-a 2.66 rule).
    assert r["magnitude"] == pytest.approx((1.96 - 1.60) / 1.60, abs=1e-3)


def test_operator_within_below_and_inside():
    s = _mini_senses()
    below = {"id": "A2", "scope": ["part:helm", "part:torso"], "metric": "neck_gap",
             "operator": "within", "value": [0.02, 0.12], "unit": "world_unit"}
    r = asserts.evaluate_clause(below, s)
    assert r["status"] == "fail"
    assert r["measured"] == pytest.approx(-0.234, abs=1e-6)
    assert r["gradient"][0]["direction"] == "+"   # need to raise into the band

    # A measured value inside the band passes.
    inside = replace(
        s, declared_pair_gap={**s.declared_pair_gap, ("helm", "torso"): 0.06}
    )
    r2 = asserts.evaluate_clause(below, inside)
    assert r2["status"] == "pass"


def test_operator_minimum():
    s = _mini_senses()
    c = {"id": "A3", "scope": ["part:helm", "part:pauldron"],
         "metric": "head_exposure", "operator": "minimum", "value": 0.85,
         "unit": "helm-heights"}
    r = asserts.evaluate_clause(c, s)
    assert r["status"] == "fail"
    # No author knob given -> metric-level correction: raise exposure (+).
    assert r["gradient"][0]["direction"] == "+"


# --------------------------------------------------------------------------- #
# Relational metrics need exactly two scopes; plant_offset is z, not y.         #
# --------------------------------------------------------------------------- #
def test_plant_offset_is_forward_z():
    s = _mini_senses()
    c = {"id": "A5", "scope": ["landmark:sword_plant", "landmark:toe_line"],
         "metric": "plant_offset", "operator": "maximum", "value": 0.15}
    r = asserts.evaluate_clause(c, s)
    assert r["measured"] == pytest.approx(0.76 - 0.41, abs=1e-6)  # z, not y
    assert r["status"] == "fail"


def test_sword_length_landmark_distance():
    s = _mini_senses()
    c = {"id": "A6", "scope": ["landmark:pommel", "landmark:blade_tip"],
         "metric": "sword_length", "operator": "within", "value": [2.47, 3.05]}
    r = asserts.evaluate_clause(c, s)
    assert r["measured"] == pytest.approx(
        float(np.linalg.norm(np.array([0, 1.60, 0.74]) - np.array([0, 0.095, 0.76]))),
        abs=1e-3)  # measured is rounded to 3 decimals
    assert r["status"] == "fail"


# --------------------------------------------------------------------------- #
# Unmeasurable status (never a silent pass).                                    #
# --------------------------------------------------------------------------- #
def test_unmeasurable_missing_landmark():
    s = _mini_senses()
    c = {"id": "X", "scope": ["landmark:pommel", "landmark:ghost"],
         "metric": "landmark_distance", "operator": "maximum", "value": 0.05}
    r = asserts.evaluate_clause(c, s)
    assert r["status"] == "unmeasurable"
    assert r["missing"] == "landmark:ghost"


def test_unmeasurable_missing_part():
    s = _mini_senses()
    c = {"id": "Y", "scope": ["part:nonexistent"], "metric": "height",
         "operator": "maximum", "value": 1.0}
    r = asserts.evaluate_clause(c, s)
    assert r["status"] == "unmeasurable"
    assert r["missing"] == "part:nonexistent"


def test_unmeasurable_unknown_metric():
    s = _mini_senses()
    c = {"id": "Z", "scope": ["whole"], "metric": "vibe",
         "operator": "maximum", "value": 1.0}
    r = asserts.evaluate_clause(c, s)
    assert r["status"] == "unmeasurable"
    assert r["missing"] == "metric:vibe"


# --------------------------------------------------------------------------- #
# Failure record shape (fable-a sec.4) + single-dimension gradient.            #
# --------------------------------------------------------------------------- #
def test_failure_record_shape_and_single_dimension_gradient():
    s = _mini_senses()
    c = {"id": "A4", "scope": ["part:pauldron"], "metric": "span",
         "operator": "maximum", "value": 1.60, "unit": "world_unit",
         "knob": "pauldron.center.x", "direction": "-"}
    r = asserts.evaluate_clause(c, s)
    for key in ("clause", "status", "measured", "target", "unit",
                "magnitude", "offenders", "detail", "gradient", "human"):
        assert key in r
    assert len(r["gradient"]) == 1                       # single-dimension only
    g = r["gradient"][0]
    assert set(g) == {"knob", "direction", "estimate"}
    assert g["knob"] == "pauldron.center.x" and g["direction"] == "-"
    assert "@" in r["human"] and "target" in r["human"]


def test_passing_assert_reports_measured():
    s = _mini_senses()
    c = {"id": "A1", "scope": ["part:blade", "contact:planted"],
         "metric": "contact_error", "operator": "maximum", "value": 0.005}
    r = asserts.evaluate_clause(c, s)
    assert r["status"] == "pass"
    assert r["measured"] == pytest.approx(0.003, abs=1e-6)  # a pass is information


# --------------------------------------------------------------------------- #
# Integration against the real knight senses -> the three losses are named.    #
# --------------------------------------------------------------------------- #
def test_knight_pack_names_three_losses():
    graph = json.loads(_KNIGHT.read_text())
    intent = json.loads(_INTENT.read_text())
    senses = proprio.build_senses(graph, intent)
    results = asserts.evaluate_pack(intent["asserts"]["clauses"], senses)
    by_id = {r["id"]: r for r in results}
    assert by_id["A2"]["status"] == "fail"    # hunch: buried helm / no neck void
    assert by_id["A3"]["status"] == "fail"    # swallowed helm: head exposure
    assert by_id["A4"]["status"] == "fail"    # swallowed helm: shoulder span
    assert by_id["A5"]["status"] == "fail"    # cane-sword: forward plant
    assert by_id["A6"]["status"] == "fail"    # cane-sword: short length
    assert by_id["A1"]["status"] == "pass"    # blade DOES reach the ground here
    # every failing assert carries exactly one single-dimension gradient.
    for r in results:
        if r["status"] == "fail":
            assert len(r["gradient"]) == 1
