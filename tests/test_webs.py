"""Authored spanning-web vocabulary (wave-4, R10).

A spanning web is ONE multi-anchor record — a membrane stretched between two
or more authored bones — compiled into the engine's frozen ``SpanningWeb``
channel (``graph["webs"]``). These tests pin the strict decode surface with
typed obstructions naming the legal shape, the station-arity law (unequal
anchor station counts are a rejection naming the per-anchor counts, never a
silent pad or truncate), the atomic mirror of the whole anchor tuple, senses
attribution naming ALL anchor bones, the aerial-cover law (a web IS flesh
topology between its anchors for contiguity), the carrier/non-carrier role
semantics, and byte-for-byte identity when no web is declared.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from golem.kernel import body
from golem.kernel import engine
from golem.senses import proprio
from golem.senses.model import RejectedSenses


_DEMO_SPEC = "specs/eye_demo_head.json"
_SPECS = Path(__file__).parents[1] / "specs"
_REFLECTION = np.asarray((-1.0, 1.0, 1.0))


def _wing_spec() -> dict:
    """Two fleshed finger bones on a bare root; the membrane spans them."""
    return {
        "name": "web-wing",
        "dialect": "body/0.3",
        "blend": 0.02,
        "skeleton": {
            "root": {"id": "thorax", "world": [0.0, 0.4, 0.0]},
            "bones": [
                {
                    "id": "f1",
                    "parent": "thorax",
                    "length": 0.3,
                    "attach": {"t": 0.0, "offset": [0.05, 0.0, 0.0]},
                    "rest_dir": [1.0, 0.0, 0.0],
                    "flesh": [
                        {
                            "kind": "gencyl",
                            "name": "finger1",
                            "span": [0.0, 1.0],
                            "radii": [0.02, 0.015],
                        }
                    ],
                },
                {
                    "id": "f2",
                    "parent": "thorax",
                    "length": 0.3,
                    "attach": {"t": 0.0, "offset": [0.05, 0.0, 0.08]},
                    "rest_dir": [1.0, 0.0, 0.15],
                    "flesh": [
                        {
                            "kind": "gencyl",
                            "name": "finger2",
                            "span": [0.0, 1.0],
                            "radii": [0.02, 0.015],
                        }
                    ],
                },
            ],
        },
    }


def _web(**overrides: object) -> dict:
    return {
        "id": "wing.membrane",
        "anchors": [
            {"bone": "f1", "radii": [0.004, 0.004]},
            {"bone": "f2", "radii": [0.004, 0.004]},
        ],
        **overrides,
    }


def _compile_with_webs(webs: object) -> body.BodyCompileResult:
    spec = _wing_spec()
    spec["webs"] = webs
    return body.Compiler(spec, spec_dir=None).compile()


def _require_compiled(spec: dict) -> body.CompiledBody:
    result = body.Compiler(spec, spec_dir=None).compile()
    assert isinstance(result, body.CompiledBody), getattr(
        result, "obstructions", ()
    )
    return result


def _compiled_web(compiled: body.CompiledBody) -> dict:
    return compiled.graph["webs"][0]


def _positive_graph(graph: dict) -> dict:
    return {key: value for key, value in graph.items() if key != "webs"}


def _canonical_body_product(compiled: body.CompiledBody) -> bytes:
    return json.dumps(
        {"graph": compiled.graph, "receipt": compiled.receipt},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


# --------------------------------------------------------------------------- #
# Promotion: one multi-anchor record into the engine's web channel.             #
# --------------------------------------------------------------------------- #
def test_authored_web_composes_anchor_bones_into_the_ordered_tuple() -> None:
    compiled = _require_compiled({**_wing_spec(), "webs": [_web()]})
    web = _compiled_web(compiled)
    assert web["id"] == "wing.membrane"
    assert len(web["anchors"]) == 2
    first, second = web["anchors"]
    # Anchor curves are sampled along the FK'd bone axes: f1 runs +x from its
    # attach offset; f2's rest_dir bends its axis toward +z.
    assert first["spine"] == [[0.05, 0.4, 0.0], [0.35, 0.4, 0.0]]
    assert second["spine"][0] == [0.05, 0.4, 0.08]
    assert second["spine"][1][2] > 0.08
    assert first["radii"] == second["radii"] == [0.004, 0.004]


def test_provenance_names_the_declaration_and_every_anchor_bone() -> None:
    compiled = _require_compiled({**_wing_spec(), "webs": [_web()]})
    address = compiled.graph["intent"]["provenance"]["wing.membrane"]
    assert address.startswith("webs[0]")
    assert "skeleton/f1" in address
    assert "skeleton/f2" in address


def test_receipt_row_names_the_web_and_all_anchors() -> None:
    compiled = _require_compiled({**_wing_spec(), "webs": [_web()]})
    (row,) = compiled.receipt["webs"]
    assert row["id"] == "wing.membrane"
    assert row["address"] == "webs[0]"
    assert row["anchors"] == ["skeleton/f1", "skeleton/f2"]


def test_membrane_field_fills_the_sheet_between_the_fingers() -> None:
    compiled = _require_compiled({**_wing_spec(), "webs": [_web()]})
    web = _compiled_web(compiled)
    left = np.asarray(web["anchors"][0]["spine"])
    right = np.asarray(web["anchors"][1]["spine"])
    midsurface = (left[0] + right[0]) / 2.0
    probe = np.asarray([midsurface])
    assert float(engine.sample_graph_field(compiled.graph, probe)[0]) < 0.0
    assert (
        float(engine.sample_graph_field(_positive_graph(compiled.graph), probe)[0])
        > 0.0
    )


def test_mirror_reflects_the_complete_anchor_tuple_as_one_instance() -> None:
    compiled = _require_compiled(
        {**_wing_spec(), "webs": [_web(mirror=True, blend=0.0, operator="crease")]}
    )
    web = _compiled_web(compiled)
    assert web["mirror"] is True
    left = np.asarray(web["anchors"][0]["spine"])
    right = np.asarray(web["anchors"][1]["spine"])
    # Sheet-interior probes, far from the one-sided finger flesh: here the web
    # field dominates cleanly, so authored side and reflection must agree to
    # the last digit (one atomic reflected tuple, never per-anchor).
    center = (left[0] + left[-1] + right[0] + right[-1]) / 4.0
    probes = np.asarray(
        [
            center,
            center + np.asarray((0.0, 0.002, 0.0)),
            center + np.asarray((0.0, -0.003, 0.0)),
        ]
    )
    sampled = engine.sample_graph_field(
        compiled.graph, np.concatenate((probes, probes * _REFLECTION))
    )
    assert np.array_equal(sampled[:3], sampled[3:])


def test_per_web_blend_and_operator_ride_the_emitted_record() -> None:
    compiled = _require_compiled(
        {**_wing_spec(), "webs": [_web(blend=0.0, operator="crease")]}
    )
    web = _compiled_web(compiled)
    assert web["blend"] == 0.0
    assert web["operator"] == "crease"


# --------------------------------------------------------------------------- #
# Strict decode: every malformation names the legal vocabulary.                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("webs", "rule"),
    (
        ([{**_web(), "depth": 1}], body.WebRule.KNOWN_FIELD),
        ([{**_web(), "id": "1bad"}], body.WebRule.IDENTIFIER),
        ([{**_web(), "anchors": "nope"}], body.WebRule.ANCHORS_ARRAY),
        (
            [{"id": "w", "anchors": [{"bone": "f1", "radii": [0.004, 0.004]}]}],
            body.WebRule.ANCHOR_ARITY,
        ),
        ([{"id": "w", "anchors": [42, 43]}], body.WebRule.ANCHOR_OBJECT),
        (
            [
                {
                    "id": "w",
                    "anchors": [
                        {"bone": "f1", "radii": [0.004, 0.004], "skin": 1},
                        {"bone": "f2", "radii": [0.004, 0.004]},
                    ],
                }
            ],
            body.WebRule.KNOWN_FIELD,
        ),
        (
            [
                {
                    "id": "w",
                    "anchors": [
                        {"bone": "f1", "radii": [0.004, 0.004], "span": [1.0, 0.0]},
                        {"bone": "f2", "radii": [0.004, 0.004]},
                    ],
                }
            ],
            body.WebRule.FINITE_STRICT_PAIR,
        ),
        (
            [
                {
                    "id": "w",
                    "anchors": [
                        {"bone": "f1", "radii": [0.0, 0.004]},
                        {"bone": "f2", "radii": [0.004, 0.004]},
                    ],
                }
            ],
            body.WebRule.FINITE_POSITIVE_ARRAY,
        ),
        (
            [
                {
                    "id": "w",
                    "anchors": [
                        {
                            "bone": "f1",
                            "radii": [0.004, 0.004],
                            "stations": [0.5, 0.25],
                        },
                        {"bone": "f2", "radii": [0.004, 0.004]},
                    ],
                }
            ],
            body.WebRule.FINITE_STRICT_ARRAY,
        ),
        (
            [
                {
                    "id": "w",
                    "anchors": [
                        {
                            "bone": "f1",
                            "radii": [0.004, 0.004],
                            "stations": [0.0, 0.5, 1.0],
                        },
                        {"bone": "f2", "radii": [0.004, 0.004]},
                    ],
                }
            ],
            body.WebRule.MATCHING_ARITY,
        ),
        (
            [
                {
                    "id": "w",
                    "anchors": [
                        {
                            "bone": "f1",
                            "radii": [0.004, 0.004],
                            "stations": [0.1, 0.9],
                        },
                        {"bone": "f2", "radii": [0.004, 0.004]},
                    ],
                }
            ],
            body.WebRule.STATIONS_MATCH_SPAN,
        ),
        (
            [{**_web(), "mirror": "yes"}],
            body.WebRule.BOOLEAN,
        ),
        (
            [{**_web(), "blend": -0.1}],
            body.WebRule.FINITE_NON_NEGATIVE_NUMBER,
        ),
        (
            [{**_web(), "operator": "smooth"}],
            body.WebRule.OPERATOR,
        ),
        (
            [{**_web(), "role": "carrier"}],
            body.WebRule.ROLE,
        ),
        ("not-an-array", body.WebRule.WEBS_ARRAY),
        ([42], body.WebRule.OBJECT),
    ),
)
def test_malformed_web_is_a_typed_rejection(
    webs: object, rule: body.WebRule
) -> None:
    result = _compile_with_webs(webs)
    assert isinstance(result, body.RejectedBody)
    obstruction = result.obstructions[0]
    assert isinstance(obstruction, body.MalformedWebObstruction)
    assert obstruction.rule is rule


def test_unknown_operator_enumerates_the_legal_set() -> None:
    result = _compile_with_webs([{**_web(), "operator": "smooth"}])
    assert isinstance(result, body.RejectedBody)
    obstruction = result.obstructions[0]
    assert obstruction.rule is body.WebRule.OPERATOR
    assert tuple(obstruction.required) == (
        "blend",
        "chamfer",
        "crease",
        "local_blend",
    )


def test_unknown_role_enumerates_the_legal_set() -> None:
    result = _compile_with_webs([{**_web(), "role": "carrier"}])
    assert isinstance(result, body.RejectedBody)
    obstruction = result.obstructions[0]
    assert obstruction.rule is body.WebRule.ROLE
    assert tuple(obstruction.required) == ("non_carrier",)


def test_unknown_anchor_bone_names_closest_candidates() -> None:
    result = _compile_with_webs(
        [
            {
                "id": "w",
                "anchors": [
                    {"bone": "f1", "radii": [0.004, 0.004]},
                    {"bone": "f12", "radii": [0.004, 0.004]},
                ],
            }
        ]
    )
    assert isinstance(result, body.RejectedBody)
    obstruction = result.obstructions[0]
    assert isinstance(obstruction, body.UnknownWebAnchorObstruction)
    assert obstruction.bone == "f12"
    assert obstruction.closest_valid_candidates


def test_duplicate_web_id_is_a_typed_rejection() -> None:
    result = _compile_with_webs([_web(), _web()])
    assert isinstance(result, body.RejectedBody)
    assert isinstance(result.obstructions[0], body.DuplicateWebIdObstruction)


def test_web_colliding_with_positive_part_is_a_typed_rejection() -> None:
    result = _compile_with_webs([_web(id="finger1")])
    assert isinstance(result, body.RejectedBody)
    assert isinstance(result.obstructions[0], body.WebPartCollisionObstruction)


# --------------------------------------------------------------------------- #
# The station-arity law: rejection names the mismatch, never silent padding.    #
# --------------------------------------------------------------------------- #
def test_unequal_anchor_station_counts_are_a_typed_rejection() -> None:
    result = _compile_with_webs(
        [
            {
                "id": "w",
                "anchors": [
                    {"bone": "f1", "radii": [0.004, 0.004, 0.003]},
                    {"bone": "f2", "radii": [0.004, 0.004]},
                ],
            }
        ]
    )
    assert isinstance(result, body.RejectedBody)
    obstruction = result.obstructions[0]
    assert isinstance(obstruction, body.MalformedWebObstruction)
    assert obstruction.rule is body.WebRule.STATION_ARITY
    assert obstruction.authored == {
        "anchors[0]": {"bone": "f1", "stations": 3},
        "anchors[1]": {"bone": "f2", "stations": 2},
    }
    assert "identical station counts" in str(obstruction.required)


def test_equal_station_counts_with_explicit_stations_compile() -> None:
    compiled = _require_compiled(
        {
            **_wing_spec(),
            "webs": [
                {
                    "id": "wing.membrane",
                    "anchors": [
                        {
                            "bone": "f1",
                            "span": [0.0, 1.0],
                            "stations": [0.0, 0.5, 1.0],
                            "radii": [0.004, 0.005, 0.004],
                        },
                        {"bone": "f2", "radii": [0.004, 0.005, 0.004]},
                    ],
                }
            ],
        }
    )
    web = _compiled_web(compiled)
    assert [len(anchor["spine"]) for anchor in web["anchors"]] == [3, 3]
    assert web["anchors"][1]["radii"] == [0.004, 0.005, 0.004]


# --------------------------------------------------------------------------- #
# Senses: the membrane names ALL its anchors and IS the topology between them.  #
# --------------------------------------------------------------------------- #
def test_web_anchor_fusion_is_declared_intent_never_an_anomaly() -> None:
    compiled = _require_compiled({**_wing_spec(), "webs": [_web()]})
    senses = proprio.build_senses(compiled.graph, compiled.graph["intent"])
    web_anomalies = tuple(
        anomaly
        for anomaly in proprio.detect_anomalies(senses)
        if "wing.membrane" in anomaly["pair"]
    )
    assert web_anomalies
    assert all(anomaly["expected"] for anomaly in web_anomalies)
    assert all(
        anomaly["det"] != "unintended_fusion" for anomaly in web_anomalies
    )


def test_web_is_flesh_topology_between_its_anchors_for_contiguity() -> None:
    with_web = _require_compiled({**_wing_spec(), "webs": [_web()]})
    without_web = _require_compiled(_wing_spec())
    senses_with = proprio.build_senses(with_web.graph, with_web.graph["intent"])
    senses_without = proprio.build_senses(
        without_web.graph, without_web.graph["intent"]
    )
    assert senses_without.n_components == 2
    assert senses_with.n_components == 1
    assert senses_with.fused_adj["finger1"] == frozenset({"wing.membrane"})
    assert senses_with.fused_adj["finger2"] == frozenset({"wing.membrane"})


def test_membrane_anomaly_attribution_names_every_anchor_bone() -> None:
    compiled = _require_compiled({**_wing_spec(), "webs": [_web()]})
    senses = proprio.build_senses(compiled.graph, compiled.graph["intent"])
    attribution = senses.provenance["wing.membrane"]
    assert "skeleton/f1" in attribution and "skeleton/f2" in attribution


# --------------------------------------------------------------------------- #
# R12: every sensed interface id is writable in the contract vocabulary.        #
# --------------------------------------------------------------------------- #
def test_contract_attach_admits_a_spanning_web_id() -> None:
    spec = copy.deepcopy(_wing_spec())
    spec["skeleton"]["bones"].append(
        {
            "id": "f3",
            "parent": "thorax",
            "length": 0.3,
            "attach": {"t": 0.0, "offset": [0.05, 0.0, -0.08]},
            "rest_dir": [1.0, 0.0, -0.15],
            "joint": {"dof": "fixed"},
            "flesh": [
                {
                    "kind": "gencyl",
                    "name": "finger3",
                    "span": [0.0, 1.0],
                    "radii": [0.02, 0.015],
                }
            ],
        }
    )
    spec["webs"] = [_web()]
    spec["contract"] = {
        "contract": "body.contract",
        "clauses": [],
        "attach": [["finger3", "wing.membrane"]],
    }

    compiler = body.Compiler(spec, spec_dir=None)
    compiled = compiler.compile()

    assert isinstance(compiled, body.CompiledBody)
    assert not [
        violation
        for violation in compiler.violations
        if violation["rule"] == "bad_contract_attach"
    ]
    assert ["finger3", "wing.membrane"] in compiled.graph["intent"]["attach"]


def test_contract_midline_admits_a_mirrored_spanning_web_id() -> None:
    spec = copy.deepcopy(_wing_spec())
    spec["webs"] = [_web(mirror=True)]
    spec["contract"] = {
        "contract": "body.contract",
        "clauses": [],
        "midline": ["wing.membrane"],
    }

    compiler = body.Compiler(spec, spec_dir=None)
    compiled = compiler.compile()

    assert isinstance(compiled, body.CompiledBody)
    assert not [
        violation
        for violation in compiler.violations
        if violation["rule"] == "bad_contract_midline"
    ]
    assert compiled.graph["intent"]["midline"] == ["wing.membrane"]


def test_unknown_attach_id_near_web_names_the_legal_neighborhood() -> None:
    spec = copy.deepcopy(_wing_spec())
    spec["webs"] = [_web()]
    spec["contract"] = {
        "contract": "body.contract",
        "clauses": [],
        "attach": [["finger1", "wing.membran"]],
    }

    compiler = body.Compiler(spec, spec_dir=None)
    compiled = compiler.compile()

    assert isinstance(compiled, body.CompiledBody)
    violation = next(
        violation
        for violation in compiler.violations
        if violation["rule"] == "bad_contract_attach"
    )
    assert violation["address"] == "spec/contract/attach/0"
    assert "closest declarable wing.membrane" in violation["detail"]
    assert (
        "web 'wing.membrane' derives from anchor flesh [finger1, finger2]"
        in violation["detail"]
    )


def _anomaly_address_part_ids(anomaly: object) -> frozenset[str]:
    address_ids = tuple(
        token.removesuffix(".L").removesuffix(".R")
        for token in anomaly["addr"].split("~")
    )
    return frozenset((*anomaly["pair"], *address_ids))


def test_tracked_menagerie_anomaly_addresses_are_declarable() -> None:
    compiled_rows: list[tuple[str, int, int]] = []
    emitted_by_spec: dict[str, frozenset[str]] = {}
    for path in sorted(_SPECS.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("dialect") != "body/0.3":
            continue
        compiled = body.Compiler(payload, spec_dir=path.parent).compile()
        assert isinstance(compiled, body.CompiledBody), path.name
        senses = proprio.build_senses(compiled.graph, compiled.graph["intent"])
        assert not isinstance(senses, RejectedSenses), path.name
        declarable = frozenset(
            str(part["id"]) for part in compiled.graph["parts"]
        ) | frozenset(
            str(web["id"]) for web in compiled.graph.get("webs", ())
        )
        web_ancestors = {
            str(row["id"]): tuple(map(str, row["anchors"]))
            for row in compiled.receipt.get("webs", ())
        }
        anomalies = proprio.detect_anomalies(senses)
        emitted = frozenset(
            part_id
            for anomaly in anomalies
            for part_id in _anomaly_address_part_ids(anomaly)
        )
        undeclarable = emitted - declarable
        assert not undeclarable, (path.name, sorted(undeclarable))
        for web_id in emitted & frozenset(web_ancestors):
            anchors = web_ancestors[web_id]
            assert anchors
            assert all(anchor.startswith("skeleton/") for anchor in anchors)
        compiled_rows.append((path.name, len(declarable), len(anomalies)))
        emitted_by_spec[path.name] = emitted

    assert len(compiled_rows) >= 40
    assert "cinderwake.json" in emitted_by_spec
    assert "cinderwake_wing_membrane" in emitted_by_spec["cinderwake.json"]


# --------------------------------------------------------------------------- #
# Carrier eligibility is the same role vocabulary as any flesh.                  #
# --------------------------------------------------------------------------- #
def test_carrier_web_carries_provenance() -> None:
    compiled = _require_compiled({**_wing_spec(), "webs": [_web()]})
    assert "wing.membrane" in compiled.graph["intent"]["provenance"]
    assert "roles" not in compiled.graph["intent"]


def test_non_carrier_web_carries_nothing() -> None:
    compiled = _require_compiled(
        {**_wing_spec(), "webs": [_web(role="non_carrier")]}
    )
    assert "wing.membrane" not in compiled.graph["intent"]["provenance"]
    assert compiled.graph["intent"]["roles"] == {
        "non_carrier": ["wing.membrane"]
    }
    assert _compiled_web(compiled)["role"] == "non_carrier"


# --------------------------------------------------------------------------- #
# Rejections render as text (the projection surface covers the new carriers).   #
# --------------------------------------------------------------------------- #
def test_web_rejection_projects_to_text_and_json() -> None:
    result = _compile_with_webs([{**_web(), "role": "carrier"}])
    assert isinstance(result, body.RejectedBody)
    text = body.rejected_body_text(result)
    assert "malformed web" in text
    projected = body.project_rejected_body(result)
    assert projected["obstructions"][0]["predicate"] == (
        "web role belongs to the closed flesh-role vocabulary"
    )


def test_carve_rejection_projects_to_text() -> None:
    spec = _wing_spec()
    spec["skeleton"]["root"]["carves"] = [
        {"id": "c", "kind": "cone", "size": [0.02, 0.02, 0.02]}
    ]
    result = body.Compiler(spec, spec_dir=None).compile()
    assert isinstance(result, body.RejectedBody)
    assert "malformed carve" in body.rejected_body_text(result)


# --------------------------------------------------------------------------- #
# Senses triangulation: consecutive anchors pair station-by-station.            #
# --------------------------------------------------------------------------- #
def test_three_anchor_web_triangulates_pairwise_station_intervals() -> None:
    from golem.senses.proprio import geometry as proprio_geometry

    record = {
        "id": "triple",
        "anchors": [
            {
                "spine": [[0.0, 0.4, 0.0], [0.2, 0.4, 0.0], [0.4, 0.4, 0.0]],
                "radii": [0.011, 0.012, 0.013],
            },
            {
                "spine": [[0.0, 0.4, 0.1], [0.2, 0.4, 0.1], [0.4, 0.4, 0.1]],
                "radii": [0.021, 0.022, 0.023],
            },
            {
                "spine": [[0.0, 0.4, 0.2], [0.2, 0.4, 0.2], [0.4, 0.4, 0.2]],
                "radii": [0.031, 0.032, 0.033],
            },
        ],
    }
    shape = proprio_geometry._part_shape(record, False)
    samples = proprio_geometry._sample_web(shape)
    # (A-1) anchor pairs x (S-1) station intervals x 2 triangles x 7
    # barycentric weights x 2 sheet faces. A cartesian spine/radii pairing
    # would emit (A-1)^2 intervals and mis-assign every half-extent.
    assert samples.shape == (2 * 2 * 2 * 7 * 2, 3)
    # Vertex sample of the second anchor pair's second interval carries the
    # SECOND pair's station radius (0.022 along the +y sheet normal) ...
    on_sheet = np.asarray((0.2, 0.4 + 0.022, 0.1))
    assert np.min(np.linalg.norm(samples - on_sheet, axis=1)) < 1.0e-12
    # ... and never a mis-paired radius from the first anchor pair.
    mis_paired = np.asarray((0.2, 0.4 + 0.012, 0.1))
    assert np.min(np.linalg.norm(samples - mis_paired, axis=1)) > 1.0e-9


# --------------------------------------------------------------------------- #
# Absence is byte-identical: the vocabulary never taxes a web-free spec.         #
# --------------------------------------------------------------------------- #
def test_absent_and_explicit_empty_webs_preserve_body_product_bytes() -> None:
    baseline = _wing_spec()
    explicit_empty = copy.deepcopy(baseline)
    explicit_empty["webs"] = []
    assert _canonical_body_product(
        _require_compiled(explicit_empty)
    ) == _canonical_body_product(_require_compiled(baseline))
    assert "webs" not in _require_compiled(baseline).graph


def test_web_free_demo_spec_compiles_byte_identically() -> None:
    baseline = json.loads(open(_DEMO_SPEC, encoding="utf-8").read())
    explicit_empty = copy.deepcopy(baseline)
    explicit_empty["webs"] = []
    assert _canonical_body_product(
        _require_compiled(explicit_empty)
    ) == _canonical_body_product(_require_compiled(baseline))
    assert "webs" not in _require_compiled(baseline).graph
