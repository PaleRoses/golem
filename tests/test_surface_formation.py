from __future__ import annotations

from dataclasses import replace

import numpy as np

from golem.kernel import engine as ev


_ACCUMULATED = ev.CompositionSectionId("accumulated", False)
_INCOMING = ev.CompositionSectionId("incoming", False)


def _opposed_planar_problem() -> ev.CertifiedLocalCompositionProblem:
    axis = np.linspace(-0.4, 0.4, 17)
    points = np.column_stack(
        (axis, np.zeros_like(axis), np.zeros_like(axis))
    )
    gradient = np.tile(np.asarray([1.0, 0.0, 0.0]), (axis.size, 1))
    support = ev.CompositionSupportRegion(
        _ACCUMULATED,
        (-0.2, -0.1, -0.1),
        (0.2, 0.1, 0.1),
    )
    return ev.CertifiedLocalCompositionProblem(
        accumulated=axis,
        incoming=-axis,
        radius=0.4,
        accumulated_gradient=gradient,
        incoming_gradient=-gradient,
        accumulated_gradient_certificate=ev.GradientMagnitudeCertificate(
            1.0,
            1.0,
        ),
        incoming_gradient_certificate=ev.GradientMagnitudeCertificate(
            1.0,
            1.0,
        ),
        accumulated_section=_ACCUMULATED,
        incoming_section=_INCOMING,
        overlaps=(
            ev.CertifiedOverlapWitness(
                (0.0, 0.0, 0.0),
                0.0,
                axis.size,
                support,
            ),
        ),
        sample_points=points,
        domain_lower=(-1.0, -1.0, -1.0),
        domain_upper=(1.0, 1.0, 1.0),
    )


def _local_blob_graph(
    *,
    blend: float = 0.02,
    center: float = 0.0,
) -> dict[str, object]:
    return {
        "name": "certified_local_blend",
        "blend": blend,
        "parts": [
            {
                "id": "left",
                "type": "blob",
                "center": [center - 0.2, 0.0, 0.0],
                "size": [0.5, 0.5, 0.5],
            },
            {
                "id": "right",
                "type": "blob",
                "center": [center + 0.2, 0.0, 0.0],
                "size": [0.5, 0.5, 0.5],
                "operator": "local_blend",
            },
        ],
    }


def _profiled_gencyl(part_id: str, x: float) -> dict[str, object]:
    return {
        "id": part_id,
        "type": "gencyl",
        "spine": [[x, -0.8, 0.0], [x, 0.8, 0.0]],
        "radii": [0.5, 0.5],
        "profile": {
            "n": 4.0,
            "aspect": 0.8,
            "up": [0.0, 0.0, 1.0],
        },
    }


def test_local_blend_carries_support_and_suppresses_seam_bulge() -> None:
    problem = _opposed_planar_problem()
    result = ev.compose_union(problem)

    assert isinstance(result, ev.AcceptedComposition)
    assert result.evidence is not None
    evidence = result.evidence
    hard = np.minimum(problem.accumulated, problem.incoming)
    support_lower = np.asarray(evidence.support.lower)
    support_upper = np.asarray(evidence.support.upper)
    inside = np.all(
        (problem.sample_points >= support_lower)
        & (problem.sample_points <= support_upper),
        axis=1,
    )
    local_displacement = float(np.max(hard - result.field))
    legacy_displacement = float(
        np.max(hard - ev.smin(problem.accumulated, problem.incoming, 0.4))
    )

    assert evidence.incoming == _INCOMING
    assert evidence.overlapping == (_ACCUMULATED,)
    assert evidence.support.active_sample_count > 0
    assert evidence.support.regions == (
        problem.overlaps[0].support,
    )
    assert evidence.maximum_outside_support_delta == 0.0
    assert np.array_equal(result.field[~inside], hard[~inside])
    assert 0.0 < local_displacement < legacy_displacement


def test_local_blend_normalizes_asymmetric_operand_scales() -> None:
    problem = _opposed_planar_problem()
    baseline = ev.compose_union(problem)
    scaled = ev.compose_union(
        replace(
            problem,
            accumulated=2.0 * problem.accumulated,
            incoming=5.0 * problem.incoming,
            accumulated_gradient=2.0 * problem.accumulated_gradient,
            incoming_gradient=5.0 * problem.incoming_gradient,
            accumulated_gradient_certificate=(
                ev.GradientMagnitudeCertificate(2.0, 2.0)
            ),
            incoming_gradient_certificate=(
                ev.GradientMagnitudeCertificate(5.0, 5.0)
            ),
        )
    )

    assert isinstance(baseline, ev.AcceptedComposition)
    assert isinstance(scaled, ev.AcceptedComposition)
    assert baseline.evidence is not None
    assert scaled.evidence is not None
    baseline_hard = np.minimum(problem.accumulated, problem.incoming)
    scaled_hard = np.minimum(
        2.0 * problem.accumulated,
        5.0 * problem.incoming,
    )

    assert scaled.evidence.support == baseline.evidence.support
    assert np.allclose(
        (scaled_hard - scaled.field) / 2.0,
        baseline_hard - baseline.field,
    )


def test_local_blend_verdict_is_query_invariant_and_raster_exact() -> None:
    graph = _local_blob_graph()
    evaluated = ev.evaluate(graph, res=18)

    assert isinstance(evaluated, ev.EvaluatedMorphology)
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
    sampled = ev.sample_graph_field(graph, points)
    query_sets = (
        np.asarray([[0.0, 0.4, 0.0]]),
        np.asarray([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        np.asarray([[8.0, 9.0, 10.0]]),
    )

    assert isinstance(sampled, np.ndarray)
    assert np.array_equal(
        evaluated.field,
        sampled.reshape(evaluated.field.shape),
    )
    assert all(
        isinstance(ev.sample_graph_field(graph, query), np.ndarray)
        for query in query_sets
    )
    assert len(evaluated.composition_evidence) == 1


def test_local_blend_is_exact_hard_union_when_support_is_empty() -> None:
    graph = _local_blob_graph(blend=0.0, center=10.0)
    evaluated = ev.evaluate(graph, res=18)

    assert isinstance(evaluated, ev.EvaluatedMorphology)
    assert len(evaluated.composition_evidence) == 1
    evidence = evaluated.composition_evidence[0]
    assert evidence.support.lower is None
    assert evidence.support.upper is None
    assert evidence.support.active_sample_count == 0


def test_local_blend_rejects_containment_without_an_iso_intersection() -> None:
    graph = {
        "name": "contained",
        "blend": 0.02,
        "parts": [
            {
                "id": "outer",
                "type": "blob",
                "center": [0.0, 0.0, 0.0],
                "size": [1.0, 1.0, 1.0],
            },
            {
                "id": "inner",
                "type": "blob",
                "center": [0.0, 0.0, 0.0],
                "size": [0.3, 0.3, 0.3],
                "operator": "local_blend",
            },
        ],
    }
    result = ev.sample_graph_field(graph, np.asarray([[0.0, 0.0, 0.0]]))

    assert isinstance(result, ev.RejectedSurfaceFormation)
    assert len(result.obstructions) == 1
    assert isinstance(result.obstructions[0], ev.NoCompatibleOverlap)
    assert result.obstructions[0].minimum_joint_residual >= 0.0
    assert result.obstructions[0].closest_point is not None


def test_local_pair_support_does_not_change_a_disjoint_prior_section() -> None:
    local = {
        "name": "pair_locality",
        "blend": 0.01,
        "parts": [
            {
                "id": "overlap",
                "type": "blob",
                "center": [-1.0, 0.0, 0.0],
                "size": [0.8, 0.8, 0.8],
            },
            {
                "id": "disjoint",
                "type": "blob",
                "center": [1.4, 0.0, 0.0],
                "size": [0.4, 0.4, 0.4],
                "operator": "crease",
            },
            {
                "id": "incoming",
                "type": "blob",
                "center": [-0.5, 0.0, 0.0],
                "size": [0.8, 0.8, 0.8],
                "operator": "local_blend",
            },
        ],
    }
    clean = {
        **local,
        "parts": [
            *local["parts"][:-1],
            {**local["parts"][-1], "operator": "crease"},
        ],
    }
    probes = np.asarray(
        [[1.0, 0.0, 0.0], [1.1, 0.2, 0.0], [1.4, 0.0, 0.0]]
    )
    local_field = ev.sample_graph_field(local, probes)
    clean_field = ev.sample_graph_field(clean, probes)

    assert isinstance(local_field, np.ndarray)
    assert isinstance(clean_field, np.ndarray)
    assert np.array_equal(local_field, clean_field)


def test_local_blend_rejects_uncalibrated_active_profile() -> None:
    graph = {
        "name": "uncalibrated_profile",
        "blend": 0.01,
        "parts": [
            _profiled_gencyl("profile", 0.0),
            {
                "id": "incoming",
                "type": "blob",
                "center": [0.35, 0.0, 0.0],
                "size": [0.5, 0.5, 0.5],
                "operator": "local_blend",
            },
        ],
    }
    result = ev.sample_graph_field(graph, np.asarray([[0.2, 0.0, 0.0]]))

    assert isinstance(result, ev.RejectedSurfaceFormation)
    assert any(
        isinstance(obstruction, ev.FieldScaleUncalibrated)
        for obstruction in result.obstructions
    )


def test_distant_uncalibrated_section_does_not_poison_a_local_pair() -> None:
    graph = {
        "name": "local_descent",
        "blend": 0.005,
        "parts": [
            _profiled_gencyl("distant_profile", 4.0),
            {
                "id": "left",
                "type": "blob",
                "center": [-0.2, 0.0, 0.0],
                "size": [0.5, 0.5, 0.5],
                "operator": "crease",
            },
            {
                "id": "right",
                "type": "blob",
                "center": [0.2, 0.0, 0.0],
                "size": [0.5, 0.5, 0.5],
                "operator": "local_blend",
            },
        ],
    }

    assert isinstance(
        ev.sample_graph_field(graph, np.asarray([[0.0, 0.4, 0.0]])),
        np.ndarray,
    )


def test_local_blend_rejects_a_sharp_edge_gradient_degeneracy() -> None:
    graph = {
        "name": "sharp_edge",
        "blend": 0.0,
        "parts": [
            {
                "id": "left",
                "type": "box",
                "center": [0.0, 0.0, 0.0],
                "size": [1.0, 1.0, 1.0],
            },
            {
                "id": "right",
                "type": "box",
                "center": [1.5, 1.5, 0.0],
                "size": [0.5, 0.5, 1.0],
                "operator": "local_blend",
            },
        ],
    }
    result = ev.sample_graph_field(graph, np.asarray([[1.0, 1.0, 0.0]]))

    assert isinstance(result, ev.RejectedSurfaceFormation)
    assert any(
        isinstance(obstruction, ev.GradientDegeneracy)
        for obstruction in result.obstructions
    )


def test_local_blend_rejects_support_outside_the_domain() -> None:
    result = ev.sample_graph_field(
        _local_blob_graph(blend=0.12),
        np.asarray([[0.0, 0.4, 0.0]]),
    )

    assert isinstance(result, ev.RejectedSurfaceFormation)
    assert any(
        isinstance(obstruction, ev.BlendSupportEscaped)
        for obstruction in result.obstructions
    )


def test_local_blend_rejects_overlapping_local_support_sections() -> None:
    graph = {
        "name": "support_conflict",
        "blend": 0.01,
        "parts": [
            {
                "id": "left",
                "type": "blob",
                "center": [-0.45, 0.0, 0.0],
                "size": [0.8, 0.8, 0.8],
            },
            {
                "id": "right",
                "type": "blob",
                "center": [0.45, 0.0, 0.0],
                "size": [0.8, 0.8, 0.8],
                "operator": "crease",
            },
            {
                "id": "incoming",
                "type": "blob",
                "center": [0.0, 0.0, 0.0],
                "size": [0.8, 0.8, 0.8],
                "operator": "local_blend",
            },
        ],
    }
    result = ev.sample_graph_field(graph, np.asarray([[0.0, 0.7, 0.0]]))

    assert isinstance(result, ev.RejectedSurfaceFormation)
    assert any(
        isinstance(obstruction, ev.LocalSectionsIncompatible)
        for obstruction in result.obstructions
    )


def test_local_blend_rejects_a_topology_change() -> None:
    gradient = np.tile(np.asarray([1.0, 0.0, 0.0]), (3, 1))
    support = ev.CompositionSupportRegion(
        _ACCUMULATED,
        (-1.0, -1.0, -2.0),
        (1.0, 1.0, 2.0),
    )
    result = ev.compose_union(
        ev.CertifiedLocalCompositionProblem(
            accumulated=np.asarray([-0.1, 0.05, 0.3]),
            incoming=np.asarray([0.3, 0.05, -0.1]),
            radius=0.8,
            accumulated_gradient=gradient,
            incoming_gradient=-gradient,
            accumulated_gradient_certificate=(
                ev.GradientMagnitudeCertificate(1.0, 1.0)
            ),
            incoming_gradient_certificate=(
                ev.GradientMagnitudeCertificate(1.0, 1.0)
            ),
            accumulated_section=_ACCUMULATED,
            incoming_section=_INCOMING,
            overlaps=(
                ev.CertifiedOverlapWitness(
                    (0.0, 1.0, 0.0),
                    0.0,
                    3,
                    support,
                ),
            ),
            sample_points=np.asarray(
                [
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            domain_lower=(-2.0, -2.0, -2.0),
            domain_upper=(2.0, 2.0, 2.0),
            topology_shape=(1, 1, 3),
        )
    )

    assert isinstance(result, ev.RejectedSurfaceFormation)
    assert result.obstructions == (
        ev.TopologyChanged(
            incoming=_INCOMING,
            clean_component_count=2,
            composed_component_count=1,
        ),
    )
