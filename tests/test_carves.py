"""Authored carve vocabulary (wave-4, R9).

A carve is a negative primitive declared bone-locally, mirrored by the same
bone vocabulary as flesh, and subtracted through the single existing
``GeometryGraph.carves`` channel the eye sockets already use. These tests pin
the strict decode surface, the single-channel merge, bone-inherited mirroring,
the cavity field sign, watertight integrity over an authored cavity, the
mouth-anomaly law (an authored cavity is a real void, never a false mesh
bridge), and byte-for-byte identity when no carve is declared.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from golem.kernel import body
from golem.kernel import engine
from golem.senses import proprio


_DEMO_SPEC = "specs/eye_demo_head.json"


def _read_demo_spec() -> dict:
    return json.loads(open(_DEMO_SPEC, encoding="utf-8").read())


def _plain_head() -> dict:
    """The eye-demo head with a single skull blob and no eyes or contract."""
    spec = _read_demo_spec()
    spec["name"] = "carve-head"
    spec.pop("eyes", None)
    spec.pop("contract", None)
    spec["skeleton"]["root"]["flesh"] = [
        {
            "kind": "blob",
            "name": "skull",
            "t": 0.0,
            "offset": [0.0, 0.0, 0.0],
            "size": [0.19, 0.18, 0.21],
        }
    ]
    return spec


def _require_compiled(spec: dict) -> body.CompiledBody:
    result = body.Compiler(spec, spec_dir=None).compile()
    assert isinstance(result, body.CompiledBody), getattr(
        result, "obstructions", ()
    )
    return result


def _compile_with_carves(carves: object) -> body.BodyCompileResult:
    spec = _plain_head()
    spec["skeleton"]["root"]["carves"] = carves
    return body.Compiler(spec, spec_dir=None).compile()


def _positive_graph(graph: dict) -> dict:
    return {key: value for key, value in graph.items() if key != "carves"}


def _canonical_body_product(compiled: body.CompiledBody) -> bytes:
    return json.dumps(
        {"graph": compiled.graph, "receipt": compiled.receipt},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


# --------------------------------------------------------------------------- #
# Single-channel promotion.                                                     #
# --------------------------------------------------------------------------- #
def test_authored_carve_and_eye_socket_share_one_carve_channel() -> None:
    spec = _read_demo_spec()
    spec["skeleton"]["root"]["carves"] = [
        {
            "id": "mouth.slot",
            "kind": "box",
            "t": 0.0,
            "offset": [0.0, -0.08, 0.16],
            "size": [0.06, 0.02, 0.06],
            "round": 0.005,
        }
    ]
    compiled = _require_compiled(spec)
    carve_ids = tuple(carve["id"] for carve in compiled.graph["carves"])
    assert carve_ids == ("mouth.slot", "eye.watcher.socket")
    assert compiled.graph["intent"]["provenance"]["mouth.slot"] == (
        "skeleton/head/carves[0]"
    )


def test_authored_cavity_changes_interior_to_exterior() -> None:
    compiled = _require_compiled(
        {
            **_plain_head(),
            "skeleton": {
                **_plain_head()["skeleton"],
                "root": {
                    **_plain_head()["skeleton"]["root"],
                    "carves": [
                        {
                            "id": "mouth.slot",
                            "kind": "box",
                            "t": 0.0,
                            "offset": [0.0, -0.08, 0.16],
                            "size": [0.06, 0.02, 0.06],
                            "round": 0.005,
                        }
                    ],
                },
            },
        }
    )
    center = np.asarray([compiled.graph["carves"][0]["center"]])
    positive = _positive_graph(compiled.graph)
    assert float(engine.sample_graph_field(positive, center)[0]) < 0.0
    assert float(engine.sample_graph_field(compiled.graph, center)[0]) > 0.0


def test_bone_mirror_reflects_the_carve_atomically() -> None:
    spec = _plain_head()
    spec["skeleton"]["bones"] = [
        {
            "id": "cheek",
            "parent": "head",
            "length": 0.12,
            "mirror": True,
            "attach": {"t": 0.0, "offset": [0.10, 0.0, 0.05]},
            "rest_dir": [1.0, 0.0, 0.3],
            "flesh": [
                {
                    "kind": "blob",
                    "name": "cheek",
                    "t": 0.5,
                    "offset": [0.0, 0.0, 0.0],
                    "size": [0.06, 0.06, 0.06],
                }
            ],
            "carves": [
                {
                    "id": "cheek.pit",
                    "kind": "blob",
                    "t": 0.5,
                    "offset": [0.0, 0.0, 0.0],
                    "size": [0.025, 0.025, 0.025],
                }
            ],
        }
    ]
    compiled = _require_compiled(spec)
    pit = next(
        carve for carve in compiled.graph["carves"] if carve["id"] == "cheek.pit"
    )
    assert pit["mirror"] is True
    center = np.asarray([pit["center"]])
    reflected = np.asarray(
        [[-pit["center"][0], pit["center"][1], pit["center"][2]]]
    )
    positive = _positive_graph(compiled.graph)
    assert float(engine.sample_graph_field(positive, center)[0]) < 0.0
    assert float(engine.sample_graph_field(positive, reflected)[0]) < 0.0
    assert float(engine.sample_graph_field(compiled.graph, center)[0]) > 0.0
    assert float(engine.sample_graph_field(compiled.graph, reflected)[0]) > 0.0


def test_gencyl_carve_channel_descends_along_the_bone() -> None:
    spec = _plain_head()
    spec["skeleton"]["root"]["length"] = 0.2
    spec["skeleton"]["root"]["carves"] = [
        {
            "id": "throat",
            "kind": "gencyl",
            "span": [0.1, 0.6],
            "radii": [0.02, 0.03, 0.02],
        }
    ]
    compiled = _require_compiled(spec)
    throat = compiled.graph["carves"][0]
    assert throat["type"] == "gencyl"
    assert len(throat["spine"]) == len(throat["radii"]) == 3


# --------------------------------------------------------------------------- #
# Strict decode: every malformation names the legal vocabulary.                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("carves", "rule"),
    (
        (
            [{"id": "c", "kind": "sphere", "size": [0.02, 0.02, 0.02]}],
            body.CarveRule.KIND,
        ),
        (
            [{"id": "c", "kind": "blob", "size": [0.0, 0.02, 0.02]}],
            body.CarveRule.FINITE_POSITIVE_VECTOR3,
        ),
        (
            [{"id": "c", "kind": "blob", "size": [0.02, 0.02, 0.02], "depth": 1}],
            body.CarveRule.KNOWN_FIELD,
        ),
        (
            [{"id": "1bad", "kind": "blob", "size": [0.02, 0.02, 0.02]}],
            body.CarveRule.IDENTIFIER,
        ),
        (
            [
                {
                    "id": "c",
                    "kind": "gencyl",
                    "span": [0.0, 1.0],
                    "radii": [0.01, 0.01],
                    "stations": [0.0, 0.5, 1.0],
                }
            ],
            body.CarveRule.MATCHING_ARITY,
        ),
        (
            [
                {
                    "id": "c",
                    "kind": "gencyl",
                    "span": [0.0, 1.0],
                    "radii": [0.01, 0.01],
                    "stations": [0.1, 0.9],
                }
            ],
            body.CarveRule.STATIONS_MATCH_SPAN,
        ),
        ("not-an-array", body.CarveRule.CARVES_ARRAY),
        ([42], body.CarveRule.OBJECT),
    ),
)
def test_malformed_carve_is_a_typed_rejection(
    carves: object, rule: body.CarveRule
) -> None:
    result = _compile_with_carves(carves)
    assert isinstance(result, body.RejectedBody)
    obstruction = result.obstructions[0]
    assert isinstance(obstruction, body.MalformedCarveObstruction)
    assert obstruction.rule is rule


def test_unknown_carve_kind_enumerates_the_legal_set() -> None:
    result = _compile_with_carves(
        [{"id": "c", "kind": "cone", "size": [0.02, 0.02, 0.02]}]
    )
    assert isinstance(result, body.RejectedBody)
    obstruction = result.obstructions[0]
    assert obstruction.rule is body.CarveRule.KIND
    assert obstruction.required == body.CARVE_KINDS == ("gencyl", "blob", "box")


def test_duplicate_carve_id_is_a_typed_rejection() -> None:
    result = _compile_with_carves(
        [
            {"id": "twin", "kind": "blob", "size": [0.02, 0.02, 0.02]},
            {"id": "twin", "kind": "box", "size": [0.02, 0.02, 0.02]},
        ]
    )
    assert isinstance(result, body.RejectedBody)
    assert isinstance(result.obstructions[0], body.DuplicateCarveIdObstruction)


def test_carve_colliding_with_positive_part_is_a_typed_rejection() -> None:
    result = _compile_with_carves(
        [{"id": "skull", "kind": "blob", "size": [0.02, 0.02, 0.02]}]
    )
    assert isinstance(result, body.RejectedBody)
    assert isinstance(
        result.obstructions[0], body.CarvePartCollisionObstruction
    )


# --------------------------------------------------------------------------- #
# The mouth law: an authored cavity is a real void, never a false bridge.       #
# --------------------------------------------------------------------------- #
def _mouth_spec() -> dict:
    spec = _plain_head()
    spec["name"] = "mouth-head"
    spec["skeleton"]["root"]["flesh"] = [
        {
            "kind": "blob",
            "name": "skull",
            "t": 0.0,
            "offset": [0.0, 0.0, 0.0],
            "size": [0.19, 0.18, 0.21],
        },
        {
            "kind": "blob",
            "name": "lip.upper",
            "t": 0.0,
            "offset": [0.0, -0.04, 0.19],
            "size": [0.05, 0.02, 0.03],
            "blend": 0.5,
        },
        {
            "kind": "blob",
            "name": "lip.lower",
            "t": 0.0,
            "offset": [0.0, -0.10, 0.19],
            "size": [0.05, 0.02, 0.03],
            "blend": 0.5,
        },
    ]
    return spec


def _lip_anomalies(compiled: body.CompiledBody) -> tuple[str, ...]:
    senses = proprio.build_senses(compiled.graph, compiled.graph["intent"])
    pair = tuple(sorted(("lip.upper", "lip.lower")))
    return tuple(
        anomaly["det"]
        for anomaly in proprio.detect_anomalies(senses)
        if anomaly["pair"] == pair
    )


def test_close_lips_without_a_cavity_read_as_a_possible_bridge() -> None:
    assert _lip_anomalies(_require_compiled(_mouth_spec())) == (
        "blend_ambiguity",
    )


def test_authored_oral_cavity_is_not_a_false_bridge_anomaly() -> None:
    spec = _mouth_spec()
    spec["skeleton"]["root"]["carves"] = [
        {
            "id": "oral.cavity",
            "kind": "box",
            "t": 0.0,
            "offset": [0.0, -0.07, 0.19],
            "size": [0.06, 0.02, 0.04],
            "round": 0.0,
        }
    ]
    assert _lip_anomalies(_require_compiled(spec)) == ()


def test_off_corridor_carve_does_not_silence_a_real_bridge() -> None:
    spec = _mouth_spec()
    spec["skeleton"]["root"]["carves"] = [
        {
            "id": "brow.dimple",
            "kind": "box",
            "t": 0.0,
            "offset": [0.0, 0.12, 0.16],
            "size": [0.02, 0.02, 0.02],
            "round": 0.0,
        }
    ]
    assert _lip_anomalies(_require_compiled(spec)) == ("blend_ambiguity",)


# --------------------------------------------------------------------------- #
# Integrity stays watertight-aware over an authored cavity.                      #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_authored_cavity_meshes_to_one_watertight_component() -> None:
    spec = _plain_head()
    spec["skeleton"]["root"]["carves"] = [
        {
            "id": "mouth.slot",
            "kind": "box",
            "t": 0.0,
            "offset": [0.0, -0.08, 0.16],
            "size": [0.06, 0.02, 0.06],
            "round": 0.005,
        }
    ]
    compiled = _require_compiled(spec)
    graph = {
        "name": compiled.graph["name"],
        "blend": compiled.graph["blend"],
        "parts": compiled.graph["parts"],
        "carves": compiled.graph["carves"],
    }
    evaluated = engine.evaluate(graph, res=96)
    report = engine.coherence_report(evaluated.vertices, evaluated.faces)
    assert report["components"] == 1
    assert report["watertight_main"] is True
    assert float(report["largest_component_volume_share"]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Absence is byte-identical: the vocabulary never taxes a carve-free spec.       #
# --------------------------------------------------------------------------- #
def test_absent_and_explicit_empty_carves_preserve_body_product_bytes() -> None:
    baseline = _plain_head()
    explicit_empty = copy.deepcopy(baseline)
    explicit_empty["skeleton"]["root"]["carves"] = []
    assert _canonical_body_product(
        _require_compiled(explicit_empty)
    ) == _canonical_body_product(_require_compiled(baseline))
    assert "carves" not in _require_compiled(baseline).graph
