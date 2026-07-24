"""Spanning-web geometry laws at the engine boundary."""

from __future__ import annotations

import numpy as np

import engine as frozen_engine
from golem.kernel import engine


_REFLECTION = np.asarray((-1.0, 1.0, 1.0))


def _flat_web(**overrides: object) -> dict:
    return {
        "id": "wing-web",
        "anchors": [
            {
                "spine": [[-1.0, 0.0, 0.0], [-1.0, 1.0, 0.0]],
                "radii": [0.1, 0.1],
            },
            {
                "spine": [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                "radii": [0.1, 0.1],
            },
        ],
        **overrides,
    }


def _host_part() -> dict:
    return {
        "id": "host",
        "type": "blob",
        "center": [0.0, 0.5, -2.0],
        "size": [0.2, 0.2, 0.2],
    }


def test_parallel_anchor_curves_form_a_flat_sheet_with_half_extent_thickness() -> None:
    points = np.asarray(
        (
            (0.0, 0.5, 0.0),
            (0.0, 0.5, 0.1),
            (0.0, 0.5, -0.1),
            (0.0, 0.5, 0.15),
        )
    )

    assert np.allclose(
        engine.sdf_web(points, _flat_web()),
        np.asarray((-0.1, 0.0, 0.0, 0.05)),
        atol=1e-12,
        rtol=0.0,
    )


def test_curved_ordered_anchors_loft_the_midsurface_between_each_pair() -> None:
    web = {
        "id": "curved-web",
        "anchors": [
            {
                "spine": [[-1.0, 0.0, 0.0], [-1.0, 1.0, 0.4], [-1.0, 2.0, 0.0]],
                "radii": [0.05, 0.05, 0.05],
            },
            {
                "spine": [[0.0, 0.0, 0.1], [0.0, 1.0, 0.6], [0.0, 2.0, 0.1]],
                "radii": [0.05, 0.05, 0.05],
            },
            {
                "spine": [[1.0, 0.0, 0.0], [1.0, 1.0, 0.8], [1.0, 2.0, 0.0]],
                "radii": [0.05, 0.05, 0.05],
            },
        ],
    }
    consecutive_pair_midpoints = np.asarray(
        ((-0.5, 1.0, 0.5), (0.5, 1.0, 0.7))
    )

    assert np.allclose(
        engine.sdf_web(consecutive_pair_midpoints, web),
        np.asarray((-0.05, -0.05)),
        atol=1e-12,
        rtol=0.0,
    )


def test_mirrored_web_instance_reflects_the_complete_anchor_tuple() -> None:
    web = {
        "id": "right-wing-web",
        "mirror": True,
        "anchors": [
            {
                "spine": [[0.4, 0.0, 0.0], [0.4, 1.0, 0.2]],
                "radii": [0.08, 0.1],
            },
            {
                "spine": [[1.2, 0.0, 0.1], [1.4, 1.0, 0.4]],
                "radii": [0.08, 0.1],
            },
        ],
    }
    authored_points = np.asarray(
        ((0.8, 0.5, 0.16), (1.0, 0.25, 0.12), (1.5, 0.5, 0.2))
    )
    reflected_points = authored_points * _REFLECTION
    graph = {
        "parts": [_host_part()],
        "blend": 0.0,
        "webs": [web],
    }
    sampled = engine.sample_graph_field(
        graph, np.concatenate((authored_points, reflected_points))
    )

    assert np.array_equal(sampled[:3], sampled[3:])


def test_incoming_web_operator_governs_the_union_fold() -> None:
    host = {
        "id": "edge-host",
        "type": "blob",
        "center": [-0.5, 0.5, 0.0],
        "size": [0.5, 0.5, 0.5],
    }
    web = {
        **_flat_web(operator="crease"),
        "anchors": [
            {
                "spine": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "radii": [0.1, 0.1],
            },
            {
                "spine": [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                "radii": [0.1, 0.1],
            },
        ],
    }
    seam_probe = np.asarray(((0.0, 0.5, 0.1),))
    graph = {"parts": [host], "webs": [web], "blend": 0.2}
    expected = np.minimum(
        engine.part_sdf(seam_probe, host),
        engine.sdf_web(seam_probe, web),
    )
    blended = engine.sample_graph_field(
        {**graph, "webs": [{**web, "operator": "blend"}]},
        seam_probe,
    )

    assert np.array_equal(engine.sample_graph_field(graph, seam_probe), expected)
    assert blended[0] < expected[0]


def test_web_shape_failures_are_typed_obstructions() -> None:
    graph = {
        "parts": [_host_part()],
        "webs": [
            {
                "id": "malformed-web",
                "anchors": [
                    {
                        "spine": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        "radii": [0.1, 0.1],
                    },
                    {
                        "spine": [
                            [1.0, 0.0, 0.0],
                            [1.0, 0.5, 0.0],
                            [1.0, 1.0, 0.0],
                        ],
                        "radii": [0.1, 0.1, 0.1],
                    },
                ],
            }
        ],
    }

    assert set(
        map(
            lambda obstruction: obstruction["rule"],
            engine.vocab_violations(graph, res=18),
        )
    ) == {"R-web-station-arity"}


def test_raster_compile_uses_the_same_web_descent_as_point_sampling() -> None:
    graph = {
        "parts": [_host_part()],
        "blend": 0.0,
        "webs": [_flat_web(operator="crease")],
    }
    evaluated = engine.evaluate(graph, res=18)
    axes = tuple(
        map(
            lambda bounds: np.linspace(*bounds, evaluated.resolution),
            zip(evaluated.lower, evaluated.upper, strict=True),
        )
    )
    points = np.stack(
        np.meshgrid(*axes, indexing="ij", copy=False), axis=-1
    ).reshape((-1, 3))

    assert np.array_equal(
        evaluated.field,
        engine.sample_graph_field(graph, points).reshape(evaluated.field.shape),
    )


def test_graph_without_webs_compiles_byte_identically() -> None:
    graph = {
        "name": "absence-pin",
        "blend": 0.03,
        "parts": [
            {
                "id": "body",
                "type": "gencyl",
                "spine": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.2]],
                "radii": [0.25, 0.3],
            },
            {
                "id": "shoulder",
                "type": "blob",
                "mirror": True,
                "center": [0.35, 0.8, 0.1],
                "size": [0.2, 0.2, 0.2],
            },
        ],
    }
    frozen_vertices, frozen_faces, frozen_normals, frozen_field, frozen_lower, frozen_upper = (
        frozen_engine.evaluate(graph, res=32)
    )
    evaluated = engine.evaluate(graph, res=32)
    explicit_empty = engine.evaluate({**graph, "webs": []}, res=32)

    assert np.array_equal(evaluated.vertices, frozen_vertices)
    assert np.array_equal(evaluated.faces, frozen_faces)
    assert np.array_equal(evaluated.normals, frozen_normals)
    assert np.array_equal(evaluated.field, frozen_field)
    assert np.array_equal(evaluated.lower, frozen_lower)
    assert np.array_equal(evaluated.upper, frozen_upper)
    assert evaluated.vertices.tobytes() == explicit_empty.vertices.tobytes()
    assert evaluated.faces.tobytes() == explicit_empty.faces.tobytes()
    assert evaluated.normals.tobytes() == explicit_empty.normals.tobytes()
    assert evaluated.field.tobytes() == explicit_empty.field.tobytes()
