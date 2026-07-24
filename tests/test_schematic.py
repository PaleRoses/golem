"""Tests for the M4 pose-schematic layer (``schematic_offset`` metric).

M4 adds ONE metric family: ``schematic_offset`` compares a compiled joint's
projection into a named canonical view (front / side) against the schematic's
hand-authored 2D target for that joint, and reports the Euclidean 2D offset.

Coordinate convention (asserts._VIEW_AXES): WORLD UNITS, front (u,v)=(x,y),
side (u,v)=(z,y). These tests pin: projection correctness for front + side, a
passing schematic-backed contract on the REAL compiled knight body, a seeded
failure that names the offending joint's skeleton-space address + a single
gradient, and the three ``unmeasurable`` paths (unknown view; joint absent from
the schematic; nonexistent landmark). ``asserts`` never measures -- projection
is pure index selection on the senses landmark table.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from golem import paths as _paths
from golem.contracts import asserts
from golem.kernel import body
from golem.senses import proprio
from golem.senses.model import GlobalDims, NoSupport, Senses

_KNIGHT_BODY = _paths.SPECS / "knight_body.json"
_SCHEMATIC = _paths.SPECS / "knight_schematic.json"


# --------------------------------------------------------------------------- #
# Helpers.                                                                     #
# --------------------------------------------------------------------------- #
def _mini_senses() -> Senses:
    """A hand-built senses record: a landmark at (1,2,3) plus a schematic whose
    front/side targets equal its projection, and a couple of off/missing cases.
    Only schematic + landmarks are load-bearing; the rest is neutral filler."""
    return Senses(
        name="mini", txn=0, n_parts=0, n_mirror=0, n_instances=0,
        global_dims=GlobalDims(W=1.0, H=1.0, D=1.0, HW=1.0,
                               centroid=np.zeros(3), k=0.05),
        raw_lo=np.zeros(3), raw_hi=np.zeros(3),
        parts={}, inst_data={}, inst_gaps={}, declared_pair_gap={},
        fusion_rows={}, cross_plane={}, n_components=1, fused_adj={},
        ground={}, near_ground={}, ground_decl=[], ground_tol=0.02,
        balance=NoSupport(), bands=[],
        landmarks={"j": [1.0, 2.0, 3.0], "k": [0.3, 0.4, 9.0]},
        schematic={
            "front": {"j": [1.0, 2.0], "k": [0.0, 0.0], "ghost/tail": [0.0, 0.0]},
            "side": {"j": [3.0, 2.0]},
        },
        attach=[], provenance={}, anatomy=None, graph={},
    )


def _clause(addr, value=0.01, **extra):
    c = {"id": addr, "scope": [f"landmark:{addr}"], "metric": "schematic_offset",
         "operator": "maximum", "value": value, "unit": "world_unit",
         "tolerance": 0}
    c.update(extra)
    return c


def _compiled():
    graph = body.compile_file(str(_KNIGHT_BODY)).graph
    schem = json.loads(_SCHEMATIC.read_text())
    return graph, schem


def _run(graph, schematic, clauses):
    merged = dict(graph["intent"])          # carries compiled landmarks + provenance
    merged["schematic"] = schematic
    merged["asserts"] = {"pack": "knight.schematic", "clauses": clauses}
    senses = proprio.build_senses(graph, merged)
    return asserts.evaluate_pack(clauses, senses)


# --------------------------------------------------------------------------- #
# Projection correctness: front picks (x,y); side picks (z,y).                 #
# --------------------------------------------------------------------------- #
def test_projection_front_is_xy():
    s = _mini_senses()
    r = asserts.evaluate_clause(_clause("j@front"), s)
    assert r["status"] == "pass"
    assert r["measured"] == pytest.approx(0.0, abs=1e-9)   # (x,y)=(1,2) == target


def test_projection_side_is_zy():
    s = _mini_senses()
    r = asserts.evaluate_clause(_clause("j@side"), s)
    assert r["status"] == "pass"
    assert r["measured"] == pytest.approx(0.0, abs=1e-9)   # (z,y)=(3,2) == target


def test_projection_axis_selection_is_exact():
    # landmark k=(0.3,0.4,9); front target (0,0) -> offset = |(0.3,0.4)| = 0.5,
    # proving front ignores z entirely (z=9 does not leak in).
    s = _mini_senses()
    r = asserts.evaluate_clause(_clause("k@front", value=0.05), s)
    assert r["status"] == "fail"
    assert r["measured"] == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------------- #
# Unmeasurable paths (never a silent pass).                                    #
# --------------------------------------------------------------------------- #
def test_unmeasurable_unknown_view():
    s = _mini_senses()
    r = asserts.evaluate_clause(_clause("j@top"), s)
    assert r["status"] == "unmeasurable"
    assert r["missing"] == "view:top"


def test_unmeasurable_joint_absent_from_schematic():
    s = _mini_senses()
    r = asserts.evaluate_clause(_clause("helm/tail@front"), s)   # not in front view
    assert r["status"] == "unmeasurable"
    assert r["missing"] == "schematic:front/helm/tail"


def test_unmeasurable_nonexistent_landmark():
    # 'ghost/tail' IS in the schematic front view but has no landmark -> the
    # measurement, not the target, is missing.
    s = _mini_senses()
    r = asserts.evaluate_clause(_clause("ghost/tail@front"), s)
    assert r["status"] == "unmeasurable"
    assert r["missing"] == "landmark:ghost/tail"


# --------------------------------------------------------------------------- #
# body.py passes an authored schematic through into the compiled intent.        #
# --------------------------------------------------------------------------- #
def test_body_passthrough_of_authored_schematic():
    spec = {
        "name": "probe", "dialect": "body/0.3", "emit_target": "v03",
        "blend": 0.05, "ground_y": 0.02,
        "skeleton": {"root": {"id": "root", "world": [0, 0, 0]}, "bones": [
            {"id": "b0", "parent": "root", "attach": {"t": 0}, "length": 1.0,
             "rest_dir": [0, 1, 0], "joint": {"dof": "ball"},
             "flesh": [{"kind": "gencyl", "name": "b0", "span": [0, 1],
                        "radii": [0.1, 0.1]}]}]},
        "schematic": {"front": {"b0/tail": [0.0, 1.0]}, "side": {"b0/tail": [0.0, 1.0]}},
    }
    comp = body.Compiler(spec, spec_dir=_paths.SPECS)
    graph = comp.compile().graph
    assert graph["intent"]["schematic"] == spec["schematic"]  # verbatim passthrough
    # and it is measurable end-to-end: b0/tail is at world (0,1,0).
    r = asserts.evaluate_clause(_clause("b0/tail@front"),
                                proprio.build_senses(graph, graph["intent"]))
    assert r["status"] == "pass"


def test_schematic_absent_is_noop():
    # A body spec with NO schematic yields an intent with no schematic key, and
    # senses.schematic is an empty dict (the schema addition is a pure no-op).
    graph, _ = _compiled()  # knight_body.json carries no schematic
    assert "schematic" not in graph["intent"]
    senses = proprio.build_senses(graph, graph["intent"])
    assert senses.schematic == {}


# --------------------------------------------------------------------------- #
# Worked example: the schematic-backed contract PASSES on the real compiled     #
# knight body (front + side, every joint), transcribed from its compiled pose.  #
# --------------------------------------------------------------------------- #
def test_worked_example_passes_on_real_compiled_body():
    graph, schem = _compiled()
    clauses = schem["asserts"]["clauses"]
    recs = _run(graph, schem["schematic"], clauses)
    assert all(r["status"] == "pass" for r in recs)
    assert len(recs) == len(clauses) == 22
    # every projection lands on its transcribed target within rounding.
    assert max(r["measured"] for r in recs) <= 0.005


# --------------------------------------------------------------------------- #
# Seeded failure: perturb ONE front target; the offset FAILS naming that        #
# joint's skeleton-space address, with measured vs target + one gradient. The    #
# address round-trip holds: the named joint's bone exists in the authored spec.  #
# --------------------------------------------------------------------------- #
def test_seeded_failure_names_skeleton_joint_with_offset():
    graph, schem = _compiled()
    clauses = schem["asserts"]["clauses"]
    perturbed = copy.deepcopy(schem["schematic"])
    ft = perturbed["front"]["forearm/tail"]
    perturbed["front"]["forearm/tail"] = [round(ft[0] + 0.30, 3), ft[1]]

    recs = _run(graph, perturbed, clauses)
    fails = [r for r in recs if r["status"] == "fail"]
    assert len(fails) == 1                                   # only the seeded joint
    f = fails[0]
    assert f["id"] == "front:forearm/tail"
    assert f["measured"] == pytest.approx(0.30, abs=1e-3)    # measured offset
    assert f["target"] == {"operator": "maximum", "value": 0.01}
    assert len(f["gradient"]) == 1                           # single-dimension (R-14)
    # the human line names the joint's skeleton-space address + measured/target.
    assert "forearm/tail" in f["human"]
    assert "0.300" in f["human"] and "0.010" in f["human"]
    # provenance passthrough rendered the offender in skeleton space.
    assert f["skeleton_offenders"] == ["landmark:forearm/tail"]
    # round-trip: the named joint's bone is a REAL bone in the authored spec.
    spec = json.loads(_KNIGHT_BODY.read_text())
    bones = {b["id"] for b in spec["skeleton"]["bones"]}
    assert f["id"].split(":", 1)[1].split("/", 1)[0] == "forearm"
    assert "forearm" in bones


# --------------------------------------------------------------------------- #
# Frozen worked-example report (write-once golden): pass count + seeded fail.    #
# --------------------------------------------------------------------------- #
def test_schematic_report_names_canonical_underbody():
    graph, schem = _compiled()
    clauses = schem["asserts"]["clauses"]
    recs = _run(graph, schem["schematic"], clauses)
    passes = [r for r in recs if r["status"] == "pass"]
    maxoff = max(r["measured"] for r in recs)

    perturbed = copy.deepcopy(schem["schematic"])
    ft = perturbed["front"]["forearm/tail"]
    perturbed["front"]["forearm/tail"] = [round(ft[0] + 0.30, 3), ft[1]]
    fail = next(r for r in _run(graph, perturbed, clauses) if r["status"] == "fail")

    views = ",".join(schem["schematic"].keys())
    njoints = len(schem["schematic"]["front"])
    report = "\n".join([
        f"SCHEMATIC {graph['name']} | views {views} | joints {njoints} | "
        f"clauses {len(clauses)}",
        f"PASS  {len(passes)}/{len(clauses)} schematic_offset "
        f"(max offset {maxoff:.3f}, bound <= 0.010)",
        "SEED  perturb front:forearm/tail target u +0.300",
        "  " + fail["human"],
        "  detail: " + fail["detail"],
    ]) + "\n"
    assert report.startswith(
        "SCHEMATIC obsidian-knight-underbody | views front,side | "
        "joints 11 | clauses 22\n"
    )
    assert "PASS  22/22 schematic_offset" in report
    assert "front:forearm/tail" in report
    assert "proj (0.106,0.558) vs target (0.406,0.558)" in report
