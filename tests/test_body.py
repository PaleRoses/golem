"""Tests for the M3 stance compiler (``golem/kernel/body.py``).

The compiler is a single deterministic pass ``pose -> FK -> ground-lift ->
prop placement -> one analytic reach solve -> emit -> assert``. These tests pin
the fences from the ExecPlan Decision Log (exactly one two-bone reach solve;
no masses in the receipt; joint limits reported never enforced; dof in
{fixed,hinge,ball}; rot never on gencyl), the frame convention (R-7), the
mirror rule + asymmetric-pose violation, the address round-trip (fable-c's
"compiler, not trapdoor" line), and the knight smoke acceptance (one watertight
component, contract clean, revision probe). Meshing lives ONLY in the single
``slow`` test that confirms the mesh-free component prediction; the compiler
never meshes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from golem import paths as _paths
from golem.kernel import body
from golem.kernel import engine as ev
from golem.kernel.body.geometry import fk_table, seed_params
from golem.kernel.body.records import UnthreadedBoneFieldError
from golem.kernel.body.relations import decode_attachment
from golem.senses import proprio

_KNIGHT_BODY = _paths.SPECS / "knight_body.json"


# --------------------------------------------------------------------------- #
# Helpers: compile a spec dict; build tiny synthetic skeletons.                #
# --------------------------------------------------------------------------- #
def _compile(spec: dict):
    comp = body.Compiler(spec, spec_dir=_paths.SPECS)
    compiled = comp.compile()
    return comp, compiled.graph, compiled.receipt


def _knight_spec() -> dict:
    return json.loads(_KNIGHT_BODY.read_text())


def _one_bone(rest_dir, *, dof="ball", limits=None, mirror=False, pose=None):
    bone = {"id": "b0", "parent": "root", "attach": {"t": 0}, "length": 1.0,
            "rest_dir": rest_dir, "joint": {"dof": dof},
            "flesh": [{"kind": "gencyl", "name": "b0", "span": [0, 1],
                       "radii": [0.1, 0.1]}]}
    if limits is not None:
        bone["joint"]["limits"] = limits
    if mirror:
        bone["mirror"] = True
    spec = {"name": "probe", "dialect": "body/0.3", "emit_target": "v03",
            "blend": 0.05, "ground_y": 0.02,
            "skeleton": {"root": {"id": "root", "world": [0, 0, 0]},
                         "bones": [bone]}}
    if pose is not None:
        spec["pose"] = {"joints": {"b0": pose}}
    return spec


def _geometry_bones(spec: dict) -> dict[str, dict]:
    attachments = {
        bone["id"]: decode_attachment(
            bone.get("attach"), f"skeleton/{bone['id']}/attach"
        )
        for bone in spec["skeleton"]["bones"]
    }
    return fk_table(spec, {}, attachments, seed_params(spec, {}, attachments))


def _replace_one_bone_flesh(spec: dict, flesh: dict) -> dict:
    skeleton = spec["skeleton"]
    bone = skeleton["bones"][0]
    return {
        **spec,
        "skeleton": {
            **skeleton,
            "bones": ({**bone, "flesh": (flesh,)},),
        },
    }


def _two_bone_reach(target: str, pole=(0, 0, 1), chain=("up", "fore")):
    bones = [
        {"id": "up", "parent": "root", "attach": {"t": 0}, "length": 0.5,
         "rest_dir": [1, 0, 0], "joint": {"dof": "ball"},
         "flesh": [{"kind": "gencyl", "name": "up", "span": [0, 1],
                    "radii": [0.1, 0.1]}]},
        {"id": "fore", "parent": "up", "attach": {"t": 1}, "length": 0.5,
         "rest_dir": [1, 0, 0],
         "joint": {"dof": "hinge", "axis": "local_x", "limits": {"pitch": [0, 150]}},
         "flesh": [{"kind": "gencyl", "name": "fore", "span": [0, 1],
                    "radii": [0.1, 0.1]}]},
    ]
    return {"name": "reach", "dialect": "body/0.3", "emit_target": "v03",
            "blend": 0.05, "ground_y": 0.02,
            "skeleton": {"root": {"id": "root", "world": [0, 0, 0]}, "bones": bones},
            "pose": {"joints": {}, "goals": [
                {"id": "g", "chain": list(chain), "end": "port:none",
                 "target": target, "pole": list(pole)}]}}


# --------------------------------------------------------------------------- #
# Frame convention (R-7): local +z is the proximal->distal axis.               #
# --------------------------------------------------------------------------- #
def test_frame_local_z_is_bone_axis_and_fk_tail():
    comp, _, _ = _compile(_one_bone([0, 1, 0]))
    # A bone whose rest_dir is world +y has local +z pointing straight up.
    np.testing.assert_allclose(comp.bones["b0"]["R"][:, 2], [0, 1, 0], atol=1e-9)
    # FK: tail = head + R @ [0,0,length] = one unit up from the origin.
    assert comp.landmarks["b0/tail"] == [0.0, 1.0, 0.0]


def test_frame_is_orthonormal_right_handed():
    comp, _, _ = _compile(_one_bone([0.3, 1.0, 0.2]))
    R = comp.bones["b0"]["R"]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)   # orthonormal
    assert float(np.linalg.det(R)) == pytest.approx(1.0, abs=1e-9)  # right-handed


# --------------------------------------------------------------------------- #
# FK determinism: a compile is a pure function of its spec, byte-for-byte.      #
# --------------------------------------------------------------------------- #
def test_fk_determinism_byte_identical():
    a = _compile(_knight_spec())[1]
    b = _compile(_knight_spec())[1]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_bone_record_projection_is_shared_by_kinematics_and_geometry():
    spec = _one_bone([1, 0, 0])
    bone = spec["skeleton"]["bones"][0]
    bone["role"] = "handle"
    bone["bone_radii"] = [0.03, 0.04]
    bone["ports"] = [{"name": "tip", "t": 1.0}]
    compiler = body.Compiler(spec, spec_dir=_paths.SPECS)
    compiled = compiler.compile()
    assert isinstance(compiled, body.CompiledBody)
    geometry_bones = _geometry_bones(spec)
    projected_keys = (
        "id",
        "parent",
        "length",
        "mirrored",
        "dof",
        "limits",
        "flesh",
        "ports",
        "arrays",
        "bone_radii",
        "role",
        "is_root",
    )
    assert {
        bone_id: {key: compiler.bones[bone_id][key] for key in projected_keys}
        for bone_id in ("root", "b0")
    } == {
        bone_id: {key: geometry_bones[bone_id][key] for key in projected_keys}
        for bone_id in ("root", "b0")
    }


def test_unknown_bone_field_fails_loudly_in_both_record_consumers():
    spec = _one_bone([1, 0, 0])
    spec["skeleton"]["bones"][0]["future_dialect_field"] = "unthreaded"
    with pytest.raises(UnthreadedBoneFieldError, match="future_dialect_field"):
        body.Compiler(spec, spec_dir=_paths.SPECS).compile()
    with pytest.raises(UnthreadedBoneFieldError, match="future_dialect_field"):
        _geometry_bones(spec)


def test_undeclared_flesh_composition_is_byte_identical():
    _, graph, _ = _compile(_one_bone([1, 0, 0]))
    assert hashlib.sha256(body._canonical_json(graph).encode()).hexdigest() == (
        "092d7f12cd5fde4a1f6ba4219f46a0ce2adbd16e2bc68a9af3dc781f5e007ea8"
    )
    assert not [
        key
        for part in graph["parts"]
        for key in ("blend", "operator")
        if key in part
    ]


def test_flesh_composition_declarations_emit_exactly():
    spec = _one_bone([1, 0, 0])
    spec["skeleton"]["bones"][0]["flesh"] = [
        {
            "kind": "gencyl",
            "name": "slab_blend",
            "span": [0, 1],
            "radii": [0.1, 0.1],
            "blend": 0.01,
        },
        {
            "kind": "gencyl",
            "name": "dorsal_crease",
            "span": [0, 1],
            "radii": [0.06, 0.06],
            "operator": "crease",
        },
    ]
    _, graph, _ = _compile(spec)
    parts = {part["id"]: part for part in graph["parts"]}
    assert parts["slab_blend"]["blend"] == 0.01
    assert "operator" not in parts["slab_blend"]
    assert parts["dorsal_crease"]["operator"] == "crease"
    assert "blend" not in parts["dorsal_crease"]


def test_nonuniform_gencyl_stations_derive_exact_spine_and_profile() -> None:
    flesh = {
        "kind": "gencyl",
        "name": "sectioned",
        "span": [-0.1, 1.1],
        "stations": [-0.1, 0.18, 0.62, 1.1],
        "radii": [0.08, 0.14, 0.1, 0.06],
        "profile": {
            "n": [2.2, 2.4, 2.8, 3.0],
            "aspect": [0.72, 0.78, 0.68, 0.62],
            "up": [0, 0, 1],
        },
    }
    _, graph, receipt = _compile(
        _replace_one_bone_flesh(_one_bone([0, 1, 0]), flesh)
    )
    part = graph["parts"][0]
    assert part["spine"] == [
        [0.0, -0.1, 0.0],
        [0.0, 0.18, 0.0],
        [0.0, 0.62, 0.0],
        [0.0, 1.1, 0.0],
    ]
    assert part["radii"] == flesh["radii"]
    assert part["profile"] == flesh["profile"]
    assert receipt["violations"] == []
    assert receipt["gencyl_stations"] == (
        {
            "part_id": "sectioned",
            "address": "skeleton/b0/flesh[0]/stations",
            "stations": (-0.1, 0.18, 0.62, 1.1),
        },
    )


def test_absent_gencyl_stations_preserve_uniform_emission_byte_exactly() -> None:
    original = _one_bone([0, 1, 0])
    flesh = original["skeleton"]["bones"][0]["flesh"][0]
    explicit = _replace_one_bone_flesh(
        original,
        {**flesh, "stations": [0.0, 1.0]},
    )
    assert json.dumps(_compile(original)[1], sort_keys=True) == json.dumps(
        _compile(explicit)[1], sort_keys=True
    )


def test_loft_sections_lower_to_anisotropic_profile() -> None:
    flesh = {
        "kind": "loft",
        "name": "keeled-tuck",
        "operator": "crease",
        "sections": [
            {
                "station": 0.0,
                "width": 0.28,
                "depth": 0.18,
                "exponent": 6.0,
                "roll": 35.0,
            },
            {
                "station": 0.42,
                "width": 0.38,
                "depth": 0.22,
                "exponent": 8.0,
                "roll": 45.0,
            },
            {
                "station": 1.0,
                "width": 0.16,
                "depth": 0.1,
                "exponent": 4.0,
                "roll": 0.0,
            },
        ],
    }
    compiler, graph, receipt = _compile(
        _replace_one_bone_flesh(_one_bone([0, 1, 0]), flesh)
    )
    part = graph["parts"][0]
    assert part["type"] == "gencyl"
    assert part["operator"] == "crease"
    assert part["spine"] == [
        [0.0, 0.0, 0.0],
        [0.0, 0.42, 0.0],
        [0.0, 1.0, 0.0],
    ]
    assert part["radii"] == [0.28, 0.38, 0.16]
    assert part["profile"]["depth"] == [0.18, 0.22, 0.1]
    assert part["profile"]["n"] == [6.0, 8.0, 4.0]
    assert part["profile"]["roll"] == [35.0, 45.0, 0.0]
    np.testing.assert_allclose(
        part["profile"]["up"], compiler.bones["b0"]["R"][:, 1]
    )
    assert part["radii"][1] > part["profile"]["depth"][1]
    assert receipt["violations"] == []
    assert receipt["loft_sections"][0]["part_id"] == "keeled-tuck"
    assert receipt["loft_sections"][0]["sections"][1] == {
        "station": 0.42,
        "width": 0.38,
        "depth": 0.22,
        "exponent": 8.0,
        "roll": 45.0,
    }


@pytest.mark.parametrize(
    ("sections", "address_suffix", "detail"),
    (
        ((), "/sections", "at least two"),
        (
            (
                {
                    "station": 0.0,
                    "width": 0.2,
                    "depth": 0.1,
                    "exponent": 4.0,
                    "roll": 0.0,
                },
                {
                    "station": 0.0,
                    "width": 0.1,
                    "depth": 0.08,
                    "exponent": 4.0,
                    "roll": 0.0,
                },
            ),
            "/sections[1]",
            "strictly increasing",
        ),
        (
            (
                {
                    "station": 0.0,
                    "width": 0.2,
                    "depth": 0.1,
                    "exponent": 4.0,
                    "roll": 0.0,
                },
                {
                    "station": 1.0,
                    "width": 0.1,
                    "depth": 0.0,
                    "exponent": 4.0,
                    "roll": 0.0,
                },
            ),
            "/sections[1]",
            "depth must be positive",
        ),
        (
            (
                {
                    "station": 0.0,
                    "width": 0.2,
                    "depth": 0.1,
                    "exponent": 4.0,
                    "roll": 0.0,
                },
                {
                    "station": 1.0,
                    "width": 0.1,
                    "depth": 0.08,
                    "exponent": 13.0,
                    "roll": 0.0,
                },
            ),
            "/sections[1]",
            "exponent must be in [2, 12]",
        ),
    ),
)
def test_invalid_loft_sections_are_addressed_violations(
    sections: object, address_suffix: str, detail: str
) -> None:
    flesh = {
        "kind": "loft",
        "name": "invalid-loft",
        "sections": sections,
    }
    _, graph, receipt = _compile(
        _replace_one_bone_flesh(_one_bone([0, 1, 0]), flesh)
    )
    violation = next(
        item
        for item in receipt["violations"]
        if item["rule"] == "bad_loft_sections"
    )
    assert violation["address"].endswith(address_suffix)
    assert detail in violation["detail"]
    assert graph["parts"] == []


def test_legacy_gencyl_omits_opt_in_loft_and_operator_surface() -> None:
    _, graph, receipt = _compile(_one_bone([0, 1, 0]))
    assert "operator" not in graph["parts"][0]
    assert "profile" not in graph["parts"][0]
    assert "loft_sections" not in receipt


def test_mirrored_nonuniform_gencyl_preserves_station_profile_order() -> None:
    flesh = {
        "kind": "gencyl",
        "name": "mirrored-section",
        "span": [0.0, 1.0],
        "stations": [0.0, 0.2, 0.7, 1.0],
        "radii": [0.07, 0.13, 0.09, 0.05],
        "profile": {
            "n": [2.0, 2.4, 2.8, 3.0],
            "aspect": [0.8, 0.72, 0.66, 0.6],
            "up": [0, 0, 1],
        },
    }
    _, graph, _ = _compile(
        _replace_one_bone_flesh(
            _one_bone([0.7, 1.0, 0.2], mirror=True), flesh
        )
    )
    part = graph["parts"][0]
    points = np.asarray(
        ((0.08, 0.2, 0.01), (0.14, 0.55, 0.03), (-0.06, 0.8, -0.02))
    )
    reflected = points * np.asarray((-1.0, 1.0, 1.0))
    np.testing.assert_allclose(
        ev.part_sdf(points, part, mirrored=True),
        ev.part_sdf(reflected, part, mirrored=False),
        atol=1.0e-12,
    )
    assert part["radii"] == flesh["radii"]
    assert part["profile"] == flesh["profile"]


@pytest.mark.parametrize(
    ("stations", "detail"),
    (
        ([0.0, 1.0], "identical arity"),
        ([0.0, 0.5, 0.5], "strictly increasing"),
        ([0.0, float("nan"), 1.0], "finite numbers"),
        ([0.1, 0.5, 1.0], "span endpoints"),
        ("not-an-array", "must be an array"),
    ),
)
def test_invalid_gencyl_stations_are_addressed_violations(
    stations: object, detail: str
) -> None:
    flesh = {
        "kind": "gencyl",
        "name": "invalid",
        "span": [0.0, 1.0],
        "stations": stations,
        "radii": [0.1, 0.12, 0.08],
    }
    _, graph, receipt = _compile(
        _replace_one_bone_flesh(_one_bone([0, 1, 0]), flesh)
    )
    violation = next(
        item
        for item in receipt["violations"]
        if item["rule"] == "bad_gencyl_stations"
    )
    assert violation["address"] == "skeleton/b0/flesh[0]/stations"
    assert detail in violation["detail"]
    assert graph["parts"] == []


@pytest.mark.parametrize(
    ("flesh", "detail"),
    (
        (
            {"kind": "gencyl", "name": "bad-span", "span": [1.0, 0.0],
             "radii": [0.1, 0.08]},
            "strictly increasing",
        ),
        (
            {"kind": "gencyl", "name": "bad-radius", "span": [0.0, 1.0],
             "radii": [0.1, float("nan")]},
            "finite positive",
        ),
        (
            {"kind": "gencyl", "name": "bad-profile", "span": [0.0, 1.0],
             "radii": [0.1, 0.08],
             "profile": {"n": [2.0, 2.5, 3.0], "aspect": 0.7}},
            "profile.n",
        ),
    ),
)
def test_uniform_gencyl_anchor_malformations_are_typed_before_sampling(
    flesh: dict, detail: str
) -> None:
    _, graph, receipt = _compile(
        _replace_one_bone_flesh(_one_bone([0, 1, 0]), flesh)
    )
    violation = next(
        item
        for item in receipt["violations"]
        if item["rule"] == "bad_gencyl_stations"
    )
    assert detail in violation["detail"]
    assert graph["parts"] == []


# --------------------------------------------------------------------------- #
# Mirror rule (T2): mirrored subtree emits mirror:true; an asymmetric pose on   #
# a mirrored subtree is a compile violation naming the joint.                   #
# --------------------------------------------------------------------------- #
def test_mirrored_subtree_emits_mirror_flag():
    _, graph, _ = _compile(_one_bone([1, 0, 0], mirror=True))
    arm = next(p for p in graph["parts"] if p["id"] == "b0")
    assert arm.get("mirror") is True


def test_asymmetric_pose_on_mirror_is_violation():
    comp, _, _ = _compile(
        _one_bone([1, 0, 0], mirror=True, limits={"pitch": [-90, 90]},
                  pose={"pitch": 20}))
    rules = {v["rule"]: v for v in comp.violations}
    assert "asymmetric_pose_on_mirror" in rules
    assert rules["asymmetric_pose_on_mirror"]["address"] == "skeleton/b0"


# --------------------------------------------------------------------------- #
# Joint typology fence: dof outside {fixed,hinge,ball} is a violation.          #
# --------------------------------------------------------------------------- #
def test_unknown_dof_is_violation():
    comp, _, _ = _compile(_one_bone([0, 1, 0], dof="slider"))
    assert any(v["rule"] == "bad_dof" for v in comp.violations)


# --------------------------------------------------------------------------- #
# Fixed-joint pose is a violation (fence 4).                                    #
# --------------------------------------------------------------------------- #
def test_pose_on_fixed_joint_is_violation():
    comp, _, _ = _compile(_one_bone([0, 1, 0], dof="fixed", pose={"pitch": 15}))
    v = next(v for v in comp.violations if v["rule"] == "pose_on_fixed")
    assert v["address"] == "skeleton/b0/joint"


# --------------------------------------------------------------------------- #
# Joint limits are REPORTED, never ENFORCED (fence 3).                          #
# --------------------------------------------------------------------------- #
def test_limit_violation_reported_but_pose_applied():
    comp, _, _ = _compile(
        _one_bone([0, 1, 0], limits={"pitch": [-10, 10]}, pose={"pitch": 40}))
    # Reported:
    lv = comp.limit_violations
    assert len(lv) == 1
    assert lv[0]["bone"] == "b0" and lv[0]["axis"] == "pitch"
    assert lv[0]["value"] == 40.0 and lv[0]["residual"] == 30.0
    # NOT enforced: the frame reflects the full authored 40deg tilt, not a clamp
    # to the 10deg limit.
    z = comp.bones["b0"]["R"][:, 2]
    tilt = np.degrees(np.arccos(np.clip(z @ np.array([0.0, 1.0, 0.0]), -1, 1)))
    assert float(tilt) == pytest.approx(40.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# The ONE analytic two-bone reach solve (fence 1).                             #
# --------------------------------------------------------------------------- #
def test_reach_reachable_lands_on_target():
    comp, _, receipt = _compile(_two_bone_reach("world:[0.6, 0.3, 0.2]"))
    g = comp.solved_goals[0]
    assert g["reachable"] is True
    assert g["residual"] == pytest.approx(0.0, abs=1e-6)   # wrist == target
    # solved angle recorded in the receipt (fence 1).
    assert "flexion_deg" in receipt["solved"]["goals"][0]


def test_reach_unreachable_clamps_full_extension_with_violation():
    comp, _, _ = _compile(_two_bone_reach("world:[5.0, 0.3, 0.2]"))
    g = comp.solved_goals[0]
    assert g["reachable"] is False
    assert g["gap"] > 0.0
    # clamped at full extension: |wrist - root| == a + b (0.5 + 0.5).
    W = np.array(g["wrist"])
    assert float(np.linalg.norm(W)) == pytest.approx(1.0, abs=1e-6)
    v = next(v for v in comp.violations if v["rule"] == "goal_unreachable")
    assert v["gap"] > 0.0
    # the gradient names skeleton length knobs (single-dimension, per R-14).
    knobs = {g_["knob"] for g_ in v["gradient"]}
    assert "skeleton/up@length" in knobs and "skeleton/fore@length" in knobs
    assert comp.clamps and comp.clamps[0]["goal"] == "g"


def test_reach_chain_must_be_exactly_two_bones():
    # Fence 1: no simultaneous / multi-bone solve. A 3-bone chain is rejected,
    # and nothing is solved.
    spec = _two_bone_reach("world:[0.6,0.3,0.2]", chain=("up", "fore", "extra"))
    comp, _, _ = _compile(spec)
    assert any(v["rule"] == "bad_goal_chain" for v in comp.violations)
    assert comp.solved_goals == []


# --------------------------------------------------------------------------- #
# rot is NEVER emitted on a gencyl (spines already carry orientation).         #
# --------------------------------------------------------------------------- #
def test_no_rot_on_any_emitted_gencyl():
    _, graph, _ = _compile(_knight_spec())
    for p in graph["parts"]:
        if p["type"] == "gencyl":
            assert "rot" not in p, p["id"]


# --------------------------------------------------------------------------- #
# Receipt contents: landmark table, solved angles, ground lift -- and NO mass. #
# --------------------------------------------------------------------------- #
def test_receipt_contents_and_no_mass_block():
    _, _, receipt = _compile(_knight_spec())
    assert receipt["landmarks"]                         # landmark table present
    assert "ground_lift" in receipt["solved"]           # ground lift recorded
    # The separated underbody authors no equipment reach goal. Analytic reach
    # semantics remain pinned by the synthetic tests above; the body receipt
    # truthfully carries an empty derived goal section.
    assert receipt["solved"]["goals"] == []
    # Fence 2: the compiler emits NO masses/volume/centroid/support anywhere in
    # the receipt (balance lives in proprio, computed from emitted parts).
    blob = json.dumps(receipt).lower()
    for banned in ("mass", "volume", "centroid", "support"):
        assert banned not in blob, banned


# --------------------------------------------------------------------------- #
# Address round-trip (fable-c: "compiler, not trapdoor"). A forced assert       #
# failure names a skeleton-space field that EXISTS in the authored spec.        #
# --------------------------------------------------------------------------- #
def test_address_round_trip_names_real_skeleton_field():
    from golem.contracts import asserts
    spec = _knight_spec()
    comp, graph, _ = _compile(spec)
    senses = proprio.build_senses(graph, graph["intent"])
    # Force a failure on a part-scoped clause (impossible head height).
    clause = {"id": "force", "scope": ["part:head"],
              "metric": "height", "operator": "minimum", "value": 5.0,
              "unit": "world_unit", "tolerance": 0}
    r = asserts.evaluate_clause(clause, senses)
    assert r["status"] == "fail"
    skel = r["skeleton_offenders"]
    assert skel  # provenance rewrote the offenders into skeleton space
    # every skeleton address resolves to a bone (or root) + flesh index that
    # exists in the authored spec.
    bones = {b["id"]: b for b in spec["skeleton"]["bones"]}
    bones[spec["skeleton"]["root"]["id"]] = spec["skeleton"]["root"]
    for addr in skel:
        assert addr.startswith("skeleton/"), addr
        _, bid, fleshref = addr.split("/", 2)
        assert bid in bones, bid
        idx = int(fleshref[fleshref.index("[") + 1:fleshref.index("]")])
        assert 0 <= idx < len(bones[bid]["flesh"]), addr
    # and the human line carries the skeleton addresses.
    assert "skeleton/helm/flesh[0]" in r["human"]


# --------------------------------------------------------------------------- #
# Knight smoke acceptance: one predicted component + contract clean.           #
# --------------------------------------------------------------------------- #
def test_knight_compiles_clean_one_component():
    comp, graph, _ = _compile(_knight_spec())
    assert comp.violations == []
    senses, records = body.run_asserts(graph, graph["intent"])
    assert senses.n_components == 1                     # mesh-free prediction
    fails = [r for r in records if r["status"] == "fail"]
    unmeas = [r for r in records if r["status"] == "unmeasurable"]
    passes = [r for r in records if r["status"] == "pass"]
    assert fails == [] and unmeas == []
    assert len(passes) == len(graph["intent"]["asserts"]["clauses"])


@pytest.mark.slow
def test_knight_meshes_to_one_watertight_component():
    # The single test that MAY mesh: confirm the mesh-free component prediction
    # against real marching-cubes connectivity (res <= 130, per the plan).
    _, graph, _ = _compile(_knight_spec())
    g = {"name": graph["name"], "blend": graph["blend"], "parts": graph["parts"]}
    evaluated = ev.evaluate(g, res=120)
    rep = ev.coherence_report(evaluated.vertices, evaluated.faces)
    assert rep["components"] == 1
    assert rep["watertight_main"] is True
    assert float(rep["largest_component_volume_share"]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Revision probe: forearm.length *= 0.88 re-passes everything (bulwark lesson). #
# --------------------------------------------------------------------------- #
def test_revision_probe_forearm_shorten_still_passes():
    original = _knight_spec()
    baseline, _baseline_graph, _baseline_receipt = _compile(original)
    revised_bones = tuple(
        ({**bone, "length": round(bone["length"] * 0.88, 6)}
         if bone["id"] == "forearm" else bone)
        for bone in original["skeleton"]["bones"]
    )
    spec = {
        **original,
        "skeleton": {**original["skeleton"], "bones": list(revised_bones)},
    }
    comp, graph, receipt = _compile(spec)
    senses, records = body.run_asserts(graph, graph["intent"])
    fails = [r for r in records if r["status"] != "pass"]
    assert fails == []                                  # full contract re-passes
    assert comp.landmarks["forearm/tail"] != baseline.landmarks["forearm/tail"]
    assert receipt["anatomy"]["status"] == "accepted"


# --------------------------------------------------------------------------- #
# Digest remains a deterministic derived view of the canonical underbody.       #
# --------------------------------------------------------------------------- #
def test_compile_digest_is_deterministic_underbody_provenance():
    def digest() -> str:
        _, graph, receipt = _compile(_knight_spec())
        senses, records = body.run_asserts(graph, graph["intent"])
        return body.render_digest(graph, receipt, senses, records)

    first = digest()
    assert first == digest()
    assert first.startswith("BODY obsidian-knight-underbody")
    assert "head <- skeleton/helm/flesh[0]" in first
    assert "greatsword" not in first and "breastplate" not in first


# --------------------------------------------------------------------------- #
# Ontology roles (wave-3, R3): closed vocabulary, hard violations on unknown  #
# values -- an unrecognized role is never a silently-ignored key.             #
# --------------------------------------------------------------------------- #
def test_unknown_bone_role_is_violation():
    spec = _one_bone([1, 0, 0])
    spec["skeleton"]["bones"][0]["role"] = "decorative"
    comp, _, _ = _compile(spec)
    violation = next(v for v in comp.violations if v["rule"] == "bad_bone_role")
    assert violation["address"] == "skeleton/b0/role"
    assert "decorative" in violation["detail"]


def test_unknown_flesh_role_is_violation():
    spec = _one_bone([1, 0, 0])
    spec["skeleton"]["bones"][0]["flesh"][0]["role"] = "decorative"
    comp, _, _ = _compile(spec)
    violation = next(
        v for v in comp.violations if v["rule"] == "bad_flesh_role"
    )
    assert violation["address"] == "skeleton/b0/flesh[0]/role"
    assert "decorative" in violation["detail"]


def test_declared_roles_are_lawful_and_visible_in_intent():
    spec = _one_bone([1, 0, 0])
    spec["skeleton"]["bones"][0]["role"] = "handle"
    spec["skeleton"]["bones"][0]["flesh"].append(
        {"kind": "gencyl", "name": "membrane", "span": [0, 1],
         "radii": [0.02, 0.02], "role": "non_carrier"}
    )
    comp, graph, receipt = _compile(spec)
    assert not [
        v for v in comp.violations
        if v["rule"] in ("bad_bone_role", "bad_flesh_role")
    ]
    assert graph["intent"]["roles"] == {
        "handle": ["b0"],
        "non_carrier": ["membrane"],
    }
    assert receipt["roles"] == graph["intent"]["roles"]


def test_role_free_spec_carries_no_roles_channel():
    comp, graph, receipt = _compile(_one_bone([1, 0, 0]))
    assert "roles" not in graph["intent"]
    assert "roles" not in receipt
    assert not [
        v for v in comp.violations
        if v["rule"] in ("bad_bone_role", "bad_flesh_role")
    ]
