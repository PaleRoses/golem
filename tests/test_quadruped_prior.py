from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from golem.cli import main
from golem.kernel.anatomy.quadruped import (
    OutsideCanonicalRange,
    QuadrupedMetric,
    QuadrupedPriorReport,
    WithinCanonicalRange,
    diagnose_quadruped_prior,
)


def _blob(name: str, size: tuple[float, float, float]) -> dict[str, object]:
    return {
        "kind": "blob",
        "name": name,
        "size": size,
        "t": 0.0,
    }


def _bone(
    bone_id: str,
    parent: str,
    attach: dict[str, object],
    rest_direction: tuple[float, float, float],
    length: float,
    mirrored: bool = False,
) -> dict[str, object]:
    return {
        "id": bone_id,
        "parent": parent,
        "attach": attach,
        "rest_dir": rest_direction,
        "length": length,
        "mirror": mirrored,
        "joint": {"dof": "fixed"},
        "flesh": (
            {
                "kind": "gencyl",
                "name": f"{bone_id}_flesh",
                "radii": (0.06, 0.05),
            },
        ),
    }


def _quadruped_spec(girdle_lateral_component: float) -> dict[str, object]:
    girdle_forward_component = math.sqrt(
        1.0 - girdle_lateral_component * girdle_lateral_component
    )
    shoulder_direction = (
        0.0,
        -math.sin(math.radians(50.0)),
        math.cos(math.radians(50.0)),
    )
    stifle_direction = (
        0.0,
        -math.sin(math.radians(45.0)),
        math.cos(math.radians(45.0)),
    )
    hock_direction = (
        0.0,
        -math.sin(math.radians(35.0)),
        math.cos(math.radians(35.0)),
    )
    return {
        "name": "canonical-quadruped",
        "dialect": "body/0.3",
        "skeleton": {
            "root": {
                "id": "core",
                "world": (0.0, 0.8, 0.0),
                "flesh": (_blob("chest", (0.2, 0.25, 0.35)),),
            },
            "bones": (
                _bone(
                    "neck",
                    "core",
                    {"t": 0.0, "offset": (0.0, 0.0, 0.5)},
                    (0.0, 0.5, math.sqrt(0.75)),
                    0.3,
                ),
                _bone(
                    "head",
                    "neck",
                    {"t": 1.0},
                    (0.0, 0.0, 1.0),
                    0.2,
                ),
                _bone(
                    "fore_girdle",
                    "core",
                    {"t": 0.0, "offset": (0.0, 0.0, 0.35)},
                    (girdle_lateral_component, 0.0, girdle_forward_component),
                    0.25,
                    mirrored=True,
                ),
                _bone(
                    "fore_upper",
                    "fore_girdle",
                    {"t": 1.0},
                    shoulder_direction,
                    0.3,
                ),
                _bone(
                    "fore_foot",
                    "fore_upper",
                    {"t": 1.0},
                    (0.0, -0.5, math.sqrt(0.75)),
                    0.2,
                ),
                _bone(
                    "hind_girdle",
                    "core",
                    {"t": 0.0, "offset": (0.0, 0.0, -0.35)},
                    (girdle_lateral_component, 0.0, girdle_forward_component),
                    0.25,
                    mirrored=True,
                ),
                _bone(
                    "hind_upper",
                    "hind_girdle",
                    {"t": 1.0},
                    stifle_direction,
                    0.3,
                ),
                _bone(
                    "hind_foot",
                    "hind_upper",
                    {"t": 1.0},
                    hock_direction,
                    0.2,
                ),
            ),
        },
        "anatomy": {
            "dialect": "anatomy/0.1",
            "overall": {
                "regions": (
                    {
                        "region_id": "thoracic",
                        "kind": "torso",
                        "host_bone_id": "core",
                    },
                    {
                        "region_id": "cranial",
                        "kind": "head",
                        "host_bone_id": "head",
                    },
                    {
                        "region_id": "forelimb",
                        "kind": "limb",
                        "host_bone_id": "fore_foot",
                    },
                    {
                        "region_id": "hindlimb",
                        "kind": "limb",
                        "host_bone_id": "hind_foot",
                    },
                ),
                "circulation": {
                    "kind": "closed_vascular",
                    "pump_region_id": "thoracic",
                    "carrier_radius_scale": 0.12,
                    "distance_decay": 0.86,
                    "exchange_beds": (
                        {
                            "region_id": "cranial",
                            "tissue": "neural_tissue",
                            "demand": 0.1,
                        },
                        {
                            "region_id": "forelimb",
                            "tissue": "skeletal_muscle",
                            "demand": 0.1,
                        },
                        {
                            "region_id": "hindlimb",
                            "tissue": "skeletal_muscle",
                            "demand": 0.1,
                        },
                    ),
                },
            },
        },
    }


def _diagnostic_by_metric(
    report: QuadrupedPriorReport,
) -> dict[QuadrupedMetric, object]:
    return {diagnostic.metric: diagnostic for diagnostic in report.diagnostics}


def _apply_proposal(
    spec: dict[str, object],
    report: QuadrupedPriorReport,
) -> dict[str, object]:
    skeleton = spec["skeleton"]
    assert isinstance(skeleton, dict)
    adjustments = {
        adjustment.bone_id: adjustment
        for adjustment in report.proposal.adjustments
    }
    bones = skeleton["bones"]
    assert isinstance(bones, tuple)
    return {
        **spec,
        "skeleton": {
            **skeleton,
            "bones": tuple(
                {
                    **bone,
                    "rest_dir": adjustments[bone["id"]].proposed_direction,
                }
                if bone["id"] in adjustments
                else bone
                for bone in bones
            ),
        },
    }


def test_canonical_quadruped_has_six_passing_diagnostics_and_empty_proposal() -> None:
    result = diagnose_quadruped_prior(_quadruped_spec(0.45))
    assert isinstance(result, QuadrupedPriorReport)
    assert tuple(diagnostic.metric for diagnostic in result.diagnostics) == tuple(
        QuadrupedMetric
    )
    assert all(
        isinstance(diagnostic, WithinCanonicalRange)
        for diagnostic in result.diagnostics
    )
    assert result.proposal.adjustments == ()
    assert result.proposal.obstructions == ()


def test_splayed_quadruped_is_diagnosed_and_delta_narrows_the_stance() -> None:
    result = diagnose_quadruped_prior(_quadruped_spec(0.98))
    assert isinstance(result, QuadrupedPriorReport)
    diagnostics = _diagnostic_by_metric(result)
    stance = diagnostics[QuadrupedMetric.STANCE_WIDTH_TO_CHEST_DEPTH]
    assert isinstance(stance, OutsideCanonicalRange)
    assert stance.observed > stance.canonical_range.maximum
    stance_adjustments = tuple(
        adjustment
        for adjustment in result.proposal.adjustments
        if QuadrupedMetric.STANCE_WIDTH_TO_CHEST_DEPTH
        in adjustment.diagnostics
    )
    assert {adjustment.bone_id for adjustment in stance_adjustments} == {
        "fore_girdle",
        "hind_girdle",
    }
    assert all(
        abs(adjustment.proposed_direction[0])
        < abs(adjustment.current_direction[0])
        for adjustment in stance_adjustments
    )
    repaired = diagnose_quadruped_prior(_apply_proposal(_quadruped_spec(0.98), result))
    assert isinstance(repaired, QuadrupedPriorReport)
    repaired_stance = _diagnostic_by_metric(repaired)[
        QuadrupedMetric.STANCE_WIDTH_TO_CHEST_DEPTH
    ]
    assert isinstance(repaired_stance, (WithinCanonicalRange, OutsideCanonicalRange))
    assert repaired_stance.observed < stance.observed


def test_pose_prior_cli_projects_typed_diagnostics_and_data_only_deltas(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "splayed.json"
    spec_path.write_text(json.dumps(_quadruped_spec(0.98)), encoding="utf-8")
    exit_code = main(("pose-prior", str(spec_path)))
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    by_id = {diagnostic["id"]: diagnostic for diagnostic in report["diagnostics"]}
    assert exit_code == 0
    assert captured.err == ""
    assert report["status"] == "accepted"
    assert by_id["stance_width_to_chest_depth"]["status"] == "fail"
    assert report["proposal"]["status"] == "proposed"
    assert all(
        adjustment["address"].endswith("/rest_dir")
        and adjustment["current_direction"] != adjustment["proposed_direction"]
        for adjustment in report["proposal"]["adjustments"]
    )
