from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from golem.kernel import body, engine
from golem.kernel.engine.algebra import (
    part_sdf_checked,
    prepared_part_gradient,
    solve_muscle_formation,
)
from golem.kernel.engine.types import (
    Accepted,
    CanalEnvelopeObstruction,
    GencylPart,
    IntegralRadiusUnsatisfied,
    MuscleFormationEvidence,
    MuscleFormationKind,
    MuscleFormationObstruction,
    UnsupportedMuscleFormation,
    decode_part,
)


_ROOT = Path(__file__).resolve().parents[1]
_MINIMAL_FIXTURE = _ROOT / "specs" / "quadruped_minimal_muscle_demo.json"
_REFLECTION = np.asarray((-1.0, 1.0, 1.0))


def _raw_formed_gencyl(
    *,
    part_id: str = "formed",
    spine: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
    ),
    radii: tuple[float, ...] = (0.4, 0.4),
    profile: dict[str, object] | None = None,
    operator: str | None = None,
) -> dict[str, object]:
    return {
        "id": part_id,
        "type": "gencyl",
        "spine": spine,
        "radii": radii,
        "formation": MuscleFormationKind.SKELETON_INTEGRAL.value,
        **({"profile": profile} if profile is not None else {}),
        **({"operator": operator} if operator is not None else {}),
    }


def _formed_part(**overrides: object) -> GencylPart:
    decoded = decode_part(_raw_formed_gencyl(**overrides))

    assert isinstance(decoded, Accepted)
    assert isinstance(decoded.value, GencylPart)
    return decoded.value


def _accepted_formation(
    part: GencylPart,
    *,
    mirrored: bool = False,
) -> MuscleFormationEvidence:
    result = solve_muscle_formation(part, mirrored=mirrored)

    assert isinstance(result, MuscleFormationEvidence)
    return result


def _scaled_part(part: GencylPart, scale: float) -> GencylPart:
    return replace(
        part,
        spine=tuple(
            tuple(scale * coordinate for coordinate in point)
            for point in part.spine
        ),
        radii=tuple(scale * radius for radius in part.radii),
    )


def _analytic_canal_volume(part: GencylPart) -> float:
    intervals = tuple(
        (math.dist(lower, upper), lower_radius, upper_radius)
        for lower, upper, lower_radius, upper_radius in zip(
            part.spine[:-1],
            part.spine[1:],
            part.radii[:-1],
            part.radii[1:],
            strict=True,
        )
    )
    frustums = sum(
        math.pi
        * length
        * (
            lower_radius * lower_radius
            + lower_radius * upper_radius
            + upper_radius * upper_radius
        )
        / 3.0
        for length, lower_radius, upper_radius in intervals
    )
    caps = (
        2.0
        * math.pi
        * (part.radii[0] ** 3 + part.radii[-1] ** 3)
        / 3.0
    )
    return frustums + caps


def _central_difference_gradient(
    part: GencylPart,
    points: np.ndarray,
    *,
    mirrored: bool = False,
    step: float = 1.0e-6,
) -> np.ndarray:
    basis = step * np.eye(3)
    return np.column_stack(
        tuple(
            (
                part_sdf_checked(points + direction, part, mirrored)
                - part_sdf_checked(points - direction, part, mirrored)
            )
            / (2.0 * step)
            for direction in basis
        )
    )


def test_skeleton_integral_field_preserves_side_radius_end_caps_and_sign() -> None:
    part = _formed_part()
    points = np.asarray(
        (
            (0.2, 1.0, 0.0),
            (0.4, 1.0, 0.0),
            (0.8, 1.0, 0.0),
            (0.0, -0.2, 0.0),
            (0.0, -0.4, 0.0),
            (0.0, -0.8, 0.0),
        )
    )
    field = part_sdf_checked(points, part)
    evidence = _accepted_formation(part)

    assert field[0] < 0.0
    assert field[1] == pytest.approx(0.0, abs=1.0e-12)
    assert field[2] > 0.0
    assert field[3] < 0.0
    assert field[4] == pytest.approx(0.0, abs=1.0e-12)
    assert field[5] > 0.0
    assert evidence.maximum_radius_residual == pytest.approx(0.0, abs=1.0e-12)
    assert evidence.centroid == pytest.approx((0.0, 1.0, 0.0))


def test_skeleton_integral_analytic_gradient_matches_field_difference() -> None:
    part = _formed_part(radii=(0.3, 0.5))
    points = np.asarray(
        (
            (0.55, 0.5, 0.0),
            (0.2, 1.4, 0.1),
            (0.0, -0.55, 0.0),
        )
    )
    prepared = prepared_part_gradient(part)

    assert prepared is not None
    np.testing.assert_allclose(
        prepared.evaluate(points),
        _central_difference_gradient(part, points),
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    assert 0.0 < prepared.magnitude_bounds[0] <= prepared.magnitude_bounds[1]


def test_skeleton_integral_uniform_scaling_covaries_field_bounds_and_volume() -> None:
    part = _formed_part(
        spine=((0.2, -0.5, 0.1), (0.2, 1.5, 0.1)),
        radii=(0.25, 0.45),
    )
    scale = 3.25
    scaled = _scaled_part(part, scale)
    points = np.asarray(
        (
            (0.6, 0.4, 0.1),
            (0.2, -0.8, 0.1),
            (-0.1, 1.7, 0.3),
        )
    )
    evidence = _accepted_formation(part)
    scaled_evidence = _accepted_formation(scaled)

    np.testing.assert_allclose(
        part_sdf_checked(points * scale, scaled),
        scale * part_sdf_checked(points, part),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        scaled_evidence.contact_lower,
        scale * np.asarray(evidence.contact_lower),
    )
    np.testing.assert_allclose(
        scaled_evidence.contact_upper,
        scale * np.asarray(evidence.contact_upper),
    )
    np.testing.assert_allclose(
        scaled_evidence.centroid,
        scale * np.asarray(evidence.centroid),
    )
    assert scaled_evidence.solved_volume == pytest.approx(
        scale**3 * evidence.solved_volume
    )
    assert scaled_evidence.target_volume == pytest.approx(
        scale**3 * evidence.target_volume
    )


def test_skeleton_integral_is_invariant_under_linear_interval_subdivision() -> None:
    unsplit = _formed_part(
        spine=((0.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
        radii=(0.2, 0.6),
    )
    split = _formed_part(
        spine=((0.0, 0.0, 0.0), (0.0, 0.8, 0.0), (0.0, 2.0, 0.0)),
        radii=(0.2, 0.36, 0.6),
    )
    points = np.asarray(
        (
            (0.1, -0.15, 0.0),
            (0.25, 0.4, 0.1),
            (0.5, 1.2, 0.0),
            (0.0, 2.5, 0.0),
        )
    )
    unsplit_evidence = _accepted_formation(unsplit)
    split_evidence = _accepted_formation(split)

    np.testing.assert_allclose(
        part_sdf_checked(points, split),
        part_sdf_checked(points, unsplit),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert split_evidence.solved_volume == pytest.approx(
        unsplit_evidence.solved_volume
    )
    assert len(split_evidence.junctions) == 1
    assert split_evidence.junctions[0].vertices


def test_skeleton_integral_evidence_has_exact_volume_and_right_handed_frames() -> None:
    part = _formed_part(
        spine=((0.2, 0.0, -0.1), (0.2, 1.0, -0.1), (0.2, 3.0, -0.1)),
        radii=(0.2, 0.4, 0.3),
    )
    evidence = _accepted_formation(part)

    assert evidence.target_volume == pytest.approx(_analytic_canal_volume(part))
    assert evidence.solved_volume == pytest.approx(evidence.target_volume)
    assert evidence.volume_residual == pytest.approx(0.0, abs=1.0e-12)
    assert evidence.maximum_curvature == pytest.approx(0.0, abs=1.0e-12)
    assert evidence.branch_compatible
    assert all(frame.determinant > 0.0 for frame in evidence.frames)
    assert all(
        np.linalg.det(
            np.column_stack(
                (frame.width_axis, frame.depth_axis, frame.tangent)
            )
        )
        == pytest.approx(frame.determinant)
        for frame in evidence.frames
    )


def test_skeleton_integral_mirror_preserves_field_and_right_handed_evidence() -> None:
    part = _formed_part(
        spine=((0.3, 0.0, 0.1), (0.3, 2.0, 0.1)),
        radii=(0.25, 0.4),
    )
    points = np.asarray(
        (
            (0.6, 0.5, 0.1),
            (0.2, 1.1, 0.3),
            (0.3, -0.3, 0.1),
        )
    )
    evidence = _accepted_formation(part)
    mirrored = _accepted_formation(part, mirrored=True)

    np.testing.assert_allclose(
        part_sdf_checked(points * _REFLECTION, part, True),
        part_sdf_checked(points, part),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        tuple(frame.origin for frame in mirrored.frames),
        np.asarray(tuple(frame.origin for frame in evidence.frames)) * _REFLECTION,
    )
    assert mirrored.mirrored
    assert all(frame.determinant > 0.0 for frame in mirrored.frames)
    assert mirrored.contact_lower[0] == pytest.approx(-evidence.contact_upper[0])
    assert mirrored.contact_upper[0] == pytest.approx(-evidence.contact_lower[0])


def test_skeleton_integral_rejects_anisotropic_radius_with_typed_residual() -> None:
    part = _formed_part(
        profile={
            "n": 2.0,
            "depth": (0.3, 0.3),
            "up": (1.0, 0.0, 0.0),
        }
    )
    result = solve_muscle_formation(part)

    assert isinstance(result, MuscleFormationObstruction)
    assert isinstance(result.failure, IntegralRadiusUnsatisfied)
    assert result.failure.residual > result.failure.tolerance


def test_skeleton_integral_rejects_singular_radius_slope_with_interval_witness() -> None:
    part = _formed_part(
        spine=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        radii=(0.1, 1.0),
    )
    result = solve_muscle_formation(part)

    assert isinstance(result, MuscleFormationObstruction)
    assert isinstance(result.failure, CanalEnvelopeObstruction)
    assert result.failure.interval_index == 0
    assert result.failure.envelope_ratio >= result.failure.required_upper_bound
    assert result.failure.support_scale * abs(
        result.failure.radius_derivative
    ) == pytest.approx(result.failure.envelope_ratio)


@pytest.mark.parametrize(
    "part",
    (
        _formed_part(
            profile={
                "n": 4.0,
                "depth": (0.4, 0.4),
                "up": (1.0, 0.0, 0.0),
            }
        ),
        _formed_part(
            spine=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.2, 2.0, 0.0)),
            radii=(0.3, 0.3, 0.3),
        ),
    ),
    ids=("nonquadratic", "noncollinear"),
)
def test_skeleton_integral_rejects_shapes_outside_the_closed_family(
    part: GencylPart,
) -> None:
    result = solve_muscle_formation(part)

    assert isinstance(result, MuscleFormationObstruction)
    assert isinstance(result.failure, UnsupportedMuscleFormation)
    assert result.failure.reason
    assert result.failure.legal_family


def test_skeleton_integral_point_sampling_is_exactly_the_evaluated_raster() -> None:
    graph = {
        "name": "formed-raster",
        "blend": 0.01,
        "parts": [_raw_formed_gencyl()],
    }
    evaluated = engine.evaluate(graph, res=12)

    assert isinstance(evaluated, engine.EvaluatedMorphology)
    axes = tuple(
        np.linspace(lower, upper, evaluated.resolution)
        for lower, upper in zip(
            evaluated.lower,
            evaluated.upper,
            strict=True,
        )
    )
    points = np.stack(
        tuple(axis.ravel() for axis in np.meshgrid(*axes, indexing="ij")),
        axis=1,
    )
    sampled = engine.sample_graph_field(graph, points)

    assert isinstance(sampled, np.ndarray)
    assert np.array_equal(evaluated.field, sampled.reshape(evaluated.field.shape))
    assert len(evaluated.formation_evidence) == 1


def test_invalid_formed_graph_sampling_returns_typed_obstructions() -> None:
    graph = {
        "name": "invalid-formed",
        "blend": 0.01,
        "parts": [
            _raw_formed_gencyl(
                spine=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                radii=(0.1, 1.0),
            )
        ],
    }
    evaluated = engine.evaluate(graph, res=10)
    sampled = engine.sample_graph_field(graph, np.asarray(((0.0, 0.5, 0.0),)))

    assert isinstance(evaluated, engine.RejectedSurfaceFormation)
    assert isinstance(sampled, engine.RejectedSurfaceFormation)
    assert all(
        len(result.obstructions) == 1
        and isinstance(result.obstructions[0], MuscleFormationObstruction)
        and isinstance(
            result.obstructions[0].failure,
            CanalEnvelopeObstruction,
        )
        for result in (evaluated, sampled)
    )


def test_body_compilation_carries_formed_receipts_and_mirror_evidence() -> None:
    spec = json.loads(_MINIMAL_FIXTURE.read_text(encoding="utf-8"))
    left, right, *remaining = spec["muscles"]
    formed_spec = {
        **spec,
        "muscles": (
            {
                **left,
                "bulk": {"absolute": {"width": 0.06, "depth": 0.06}},
                "formation": MuscleFormationKind.SKELETON_INTEGRAL.value,
                "operator": "crease",
            },
            right,
            *remaining,
        ),
    }
    result = body.Compiler(formed_spec, spec_dir=_ROOT / "specs").compile()

    assert isinstance(result, body.CompiledBody)
    rows = {row["id"]: row for row in result.receipt["myology"]}
    parts = {part["id"]: part for part in result.graph["parts"]}
    left_evidence = rows["left_quadriceps"]["formation"]
    right_evidence = rows["right_quadriceps"]["formation"]
    assert parts["left_quadriceps"]["formation"] == "skeleton_integral"
    assert parts["right_quadriceps"]["formation"] == "skeleton_integral"
    assert left_evidence["maximum_radius_residual"] == pytest.approx(0.0)
    assert right_evidence["maximum_radius_residual"] == pytest.approx(0.0)
    assert not left_evidence["mirrored"]
    assert right_evidence["mirrored"]
    assert left_evidence["solved_volume"] == pytest.approx(
        right_evidence["solved_volume"]
    )


def test_two_formed_muscles_handoff_through_certified_local_blend() -> None:
    graph = {
        "name": "formed-local-handoff",
        "blend": 0.01,
        "parts": (
            _raw_formed_gencyl(
                part_id="left",
                spine=((-0.15, 0.0, 0.0), (-0.15, 1.5, 0.0)),
                radii=(0.3, 0.3),
            ),
            _raw_formed_gencyl(
                part_id="right",
                spine=((0.15, 0.0, 0.0), (0.15, 1.5, 0.0)),
                radii=(0.3, 0.3),
                operator="local_blend",
            ),
        ),
    }
    result = engine.evaluate(graph, res=14)

    assert isinstance(result, engine.EvaluatedMorphology)
    assert len(result.composition_evidence) == 1
    assert len(result.formation_evidence) == 2
    assert result.composition_evidence[0].incoming.part_id == "right"
