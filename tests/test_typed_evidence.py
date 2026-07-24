from __future__ import annotations

import json
from functools import cache

from golem import paths
from golem.assembly import (
    ElementFitLaw,
    ElementFitObstruction,
    project_assembly_obstructions,
)
from golem.assembly.obstructions import UnknownAppendageIntentObstruction
from golem.contracts.model import Failed, Passed, Unmeasurable
from golem.contracts.views import verdict_record
from golem.kernel.anatomy import AcceptedAnatomy
from golem.kernel.body import run_asserts, run_typed_asserts
from golem.session.effect import compile_authored
from golem.session.state import AuthoredState, Compiled


@cache
def _compiled_session() -> Compiled:
    payload = json.loads((paths.SPECS / "knight_body.json").read_text())
    payload["contract"] = "knight_contract.json"
    outcome = compile_authored(AuthoredState(payload, paths.SPECS))
    assert isinstance(outcome, Compiled)
    return outcome


def test_typed_asserts_own_verdicts_and_legacy_records_are_a_projection() -> None:
    compiled = _compiled_session()
    _, verdicts = run_typed_asserts(compiled.graph, compiled.graph["intent"])
    _, records = run_asserts(compiled.graph, compiled.graph["intent"])

    assert verdicts
    assert all(
        isinstance(verdict, (Passed, Failed, Unmeasurable))
        for verdict in verdicts
    )
    assert verdicts == compiled.verdicts
    assert records == list(map(verdict_record, verdicts))


def test_session_compilation_retains_anatomy_and_typed_verdicts() -> None:
    compiled = _compiled_session()

    assert isinstance(compiled.anatomy, AcceptedAnatomy)
    assert compiled.verdicts
    assert compiled.records == tuple(map(verdict_record, compiled.verdicts))


def test_assembly_obstruction_projection_is_ordered_and_json_total() -> None:
    obstruction = ElementFitObstruction(
        left_element_id="body",
        right_element_id="sword",
        fit_law=ElementFitLaw.DISJOINT,
        penetrating_sample_count=3,
        outside_contact_sample_count=3,
        maximum_penetration_world=0.02,
        tolerance_world=0.001,
        contact_element_id=None,
        contact_radius_world=None,
    )

    (projected,) = project_assembly_obstructions(obstruction)

    assert tuple(projected) == (
        "obstruction",
        "left_element_id",
        "right_element_id",
        "fit_law",
        "penetrating_sample_count",
        "outside_contact_sample_count",
        "maximum_penetration_world",
        "tolerance_world",
        "contact_element_id",
        "contact_radius_world",
        "address",
        "predicate",
        "required",
        "observed",
    )
    assert projected["obstruction"] == "ElementFitObstruction"
    assert projected["fit_law"] == "disjoint"
    assert projected["address"] == "assembly/body~sword"
    assert projected["required"] == 0
    assert projected["observed"] == 3
    assert json.loads(json.dumps(projected))["fit_law"] == "disjoint"


def test_assembly_obstruction_projection_preserves_tuple_evidence() -> None:
    obstruction = UnknownAppendageIntentObstruction(
        element_id="body",
        appendage_id="missing",
        intent_id="grasp",
        closest_valid_candidates=("left_hand", "right_hand"),
    )

    (projected,) = project_assembly_obstructions(obstruction)

    assert projected["closest_valid_candidates"] == (
        "left_hand",
        "right_hand",
    )
    assert json.loads(json.dumps(projected))["closest_valid_candidates"] == [
        "left_hand",
        "right_hand",
    ]
