"""Tests for the pure constructed-geometry backing of the relational solve."""

from __future__ import annotations

import numpy as np

from golem.addressing.scope import Bone, Landmark, Part
from golem.kernel.body.geometry import (
    ancestors,
    build_tree,
    constructed_identifiers,
    fk_table,
    flesh_owner_map,
    geometry_view,
    landmark_owner,
    landmark_points,
    lca,
    port_owner_map,
    seed_params,
    subtree_depths,
)
from golem.kernel.body.relations import decode_attachment


def _attachments(spec: dict) -> dict:
    return {
        bone["id"]: decode_attachment(bone.get("attach"), f"skeleton/{bone['id']}/attach")
        for bone in spec["skeleton"]["bones"]
    }


def _limb_spec() -> dict:
    return {
        "name": "limb",
        "skeleton": {
            "root": {"id": "root", "world": [0, 0, 0]},
            "bones": [
                {"id": "spine", "parent": "root", "attach": {"t": 0}, "length": 0.4,
                 "rest_dir": [0, 0, 1],
                 "flesh": [{"kind": "blob", "name": "trunk", "size": [0.2, 0.2, 0.2], "t": 0.5}]},
                {"id": "upper", "parent": "spine", "attach": {"t": 1.0}, "length": 0.3,
                 "rest_dir": [1, 0, 0],
                 "ports": [{"name": "elbow", "t": 1.0}]},
                {"id": "lower", "parent": "upper", "attach": {"at": "landmark:upper/tail"},
                 "length": 0.3, "rest_dir": [1, 0, 0],
                 "flesh": [{"kind": "blob", "name": "hand", "size": [0.1, 0.1, 0.1], "t": 1.0}]},
            ],
        },
    }


def test_build_tree_orders_depth_first_and_indexes_parents():
    tree = build_tree(_limb_spec())
    assert tree.root == "root"
    assert tree.order == ("root", "spine", "upper", "lower")
    assert tree.parent == {"root": None, "spine": "root", "upper": "spine", "lower": "upper"}
    assert tree.children["spine"] == ("upper",)


def test_ancestors_lca_and_subtree_depths():
    tree = build_tree(_limb_spec())
    assert ancestors(tree, "lower") == ("lower", "upper", "spine", "root")
    assert lca(tree, "lower", "upper") == "upper"
    assert lca(tree, "lower", "spine") == "spine"
    assert lca(tree, "upper", "upper") == "upper"
    assert subtree_depths(tree, "spine") == {"spine": 0, "upper": 1, "lower": 2}


def test_owner_maps_resolve_flesh_ports_and_landmarks():
    tree = build_tree(_limb_spec())
    flesh = flesh_owner_map(tree)
    ports = port_owner_map(tree)
    assert flesh["hand"][0] == "lower"
    assert flesh["trunk"][0] == "spine"
    assert ports == {"elbow": "upper"}
    assert landmark_owner("upper/tail", tree, ports) == "upper"
    assert landmark_owner("elbow", tree, ports) == "upper"
    assert landmark_owner("nonexistent/tail", tree, ports) is None


def test_constructed_identifiers_cover_bones_flesh_and_landmarks():
    identifiers = constructed_identifiers(build_tree(_limb_spec()))
    for expected in ("spine", "hand", "trunk", "upper/tail", "lower/head", "elbow"):
        assert expected in identifiers


def test_fk_table_matches_named_site_head_to_its_landmark():
    spec = _limb_spec()
    attach = _attachments(spec)
    seeds = seed_params(spec, {}, attach)
    table = fk_table(spec, {}, attach, seeds)
    points = landmark_points(table)
    # lower is a named-site child seeded onto upper/tail; its head must coincide.
    assert np.allclose(table["lower"]["head"], points["upper/tail"])


def test_geometry_view_extent_sites_bracket_the_center():
    spec = _limb_spec()
    attach = _attachments(spec)
    view = geometry_view(spec, {}, attach, seed_params(spec, {}, attach))
    sites = dict(view.extent_sites_y(Part("trunk")))
    assert set(sites) == {"trunk#lo", "trunk#hi"}
    assert sites["trunk#lo"] < sites["trunk#hi"]
    center_y = float(view.point(Part("trunk"))[1])
    assert sites["trunk#lo"] <= center_y <= sites["trunk#hi"]


def test_geometry_view_axis_is_unit_bone_direction():
    spec = _limb_spec()
    attach = _attachments(spec)
    view = geometry_view(spec, {}, attach, seed_params(spec, {}, attach))
    axis = view.axis(Bone("upper"))
    assert np.isclose(np.linalg.norm(axis), 1.0)
    # upper is rest_dir [1,0,0] off a spine oriented along world +z; frame maps it to world +x.
    assert np.allclose(axis, [1.0, 0.0, 0.0], atol=1e-9)


def test_landmark_scope_point_resolves_named_site():
    spec = _limb_spec()
    attach = _attachments(spec)
    view = geometry_view(spec, {}, attach, seed_params(spec, {}, attach))
    table = fk_table(spec, {}, attach, seed_params(spec, {}, attach))
    assert np.allclose(view.point(Landmark("upper/tail")), landmark_points(table)["upper/tail"])
