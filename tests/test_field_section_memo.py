from __future__ import annotations

import numpy as np
import pytest

from golem.kernel.engine import compile as gk


def _decode(graph: dict):
    return gk.require_accepted(gk.decode_graph(graph))


def _domain_grid(graph, res: int) -> np.ndarray:
    lower, upper = gk.graph_bounds_checked(graph)
    axes = tuple(
        np.linspace(float(lower[index]), float(upper[index]), res)
        for index in range(3)
    )
    return gk._grid_points(axes)


def _production_grid(graph, res: int) -> np.ndarray:
    assert res != gk._LOCAL_TOPOLOGY_CERTIFICATION_RESOLUTION
    return _domain_grid(graph, res)


def _certification_count() -> int:
    return gk._LOCAL_TOPOLOGY_CERTIFICATION_RESOLUTION ** 3


def _compose(graph, points: np.ndarray):
    return gk._sample_graph_field_certified_checked(graph, points)


def _first_instance(graph_dict: dict):
    (geometry, mirrored), *_rest = gk._positive_instances(_decode(graph_dict))
    return geometry, mirrored


def _section_spy(monkeypatch) -> list[tuple[int, str]]:
    calls: list[tuple[int, str]] = []
    real = gk._sample_section_arrays

    def spy(section, points, *args, **kwargs):
        calls.append(
            (
                int(np.asarray(points).reshape((-1, 3)).shape[0]),
                section.identifier.part_id,
            )
        )
        return real(section, points, *args, **kwargs)

    monkeypatch.setattr(gk, "_sample_section_arrays", spy)
    return calls


def _carve_spy(monkeypatch) -> list[tuple[int, str]]:
    calls: list[tuple[int, str]] = []
    real = gk._sample_carve_array

    def spy(part, mirrored, points, *args, **kwargs):
        calls.append(
            (
                int(np.asarray(points).reshape((-1, 3)).shape[0]),
                part.part_id,
            )
        )
        return real(part, mirrored, points, *args, **kwargs)

    monkeypatch.setattr(gk, "_sample_carve_array", spy)
    return calls


def _part_ids_at(calls: list[tuple[int, str]], count: int) -> list[str]:
    return [part_id for sampled, part_id in calls if sampled == count]


def _store_is_empty(root) -> bool:
    return not root.exists() or not any(
        path.is_file() for path in root.rglob("*")
    )


@pytest.fixture
def stores(tmp_path, monkeypatch):
    composed = tmp_path / "composed"
    field = tmp_path / "field"
    sections = tmp_path / "sections"
    monkeypatch.setattr(gk, "_composed_snapshot_root", lambda: composed)
    monkeypatch.setattr(gk, "_field_cache_root", lambda: field)
    monkeypatch.setattr(gk, "_section_sample_root", lambda: sections)
    # Warm-path semantics are the subject; the ambient binding-gate cold env
    # must not decide which path the section memo exercises.
    monkeypatch.delenv(gk._FIELD_CACHE_COLD_ENV, raising=False)
    return composed, field, sections


def _key_graph(
    *,
    part_operator: str = "blend",
    part_blend: float | None = None,
    part_center: tuple[float, float, float] = (-0.2, 0.0, 0.0),
    graph_blend: float = 0.03,
) -> dict:
    alpha: dict[str, object] = {
        "id": "alpha",
        "type": "blob",
        "center": list(part_center),
        "size": [0.6, 0.6, 0.6],
        "operator": part_operator,
    }
    if part_blend is not None:
        alpha["blend"] = part_blend
    return {
        "name": "key_graph",
        "blend": graph_blend,
        "parts": [
            alpha,
            {
                "id": "beta",
                "type": "blob",
                "center": [0.2, 0.0, 0.0],
                "size": [0.6, 0.6, 0.6],
                "operator": "blend",
            },
        ],
    }


def _pair() -> dict:
    return {
        "name": "pair",
        "blend": 0.02,
        "parts": [
            {
                "id": "alpha",
                "type": "blob",
                "center": [-0.2, 0.0, 0.0],
                "size": [0.5, 0.5, 0.5],
                "operator": "blend",
            },
            {
                "id": "beta",
                "type": "blob",
                "center": [0.2, 0.0, 0.0],
                "size": [0.5, 0.5, 0.5],
                "operator": "local_blend",
            },
        ],
    }


def _trio(*, base_size: float = 0.5) -> dict:
    return {
        "name": "trio",
        "blend": 0.02,
        "parts": [
            {
                "id": "base",
                "type": "blob",
                "center": [0.0, 0.0, 0.0],
                "size": [base_size, base_size, base_size],
                "operator": "blend",
            },
            {
                "id": "bump",
                "type": "blob",
                "center": [0.4, 0.0, 0.0],
                "size": [0.5, 0.5, 0.5],
                "operator": "local_blend",
            },
            {
                "id": "spacer",
                "type": "blob",
                "center": [0.0, 0.0, 2.5],
                "size": [0.15, 0.15, 0.15],
                "operator": "blend",
            },
        ],
    }


def _carved(*, carve_y: float = 0.4) -> dict:
    graph = _pair()
    graph["name"] = "carved"
    graph["carves"] = [
        {
            "id": "carve",
            "type": "blob",
            "center": [0.0, carve_y, 0.0],
            "size": [0.25, 0.25, 0.25],
        }
    ]
    return graph


def _twins(*, graph_blend: float = 0.02) -> dict:
    def twin(part_id: str) -> dict:
        return {
            "id": part_id,
            "type": "blob",
            "center": [0.1, 0.0, 0.0],
            "size": [0.5, 0.5, 0.5],
            "operator": "blend",
        }

    return {
        "name": "twins",
        "blend": graph_blend,
        "parts": [twin("left"), twin("right")],
    }


def _containment_reject() -> dict:
    return {
        "name": "contained",
        "blend": 0.02,
        "parts": [
            {
                "id": "outer",
                "type": "blob",
                "center": [0.0, 0.0, 0.0],
                "size": [1.0, 1.0, 1.0],
                "operator": "blend",
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


def test_section_key_names_geometry_and_excludes_composition(monkeypatch) -> None:
    reference_graph = _key_graph()
    decoded = _decode(reference_graph)
    geometry, mirrored = _first_instance(reference_graph)
    points = _domain_grid(decoded, 12)
    digest = gk._points_digest(points)
    other_digest = gk._points_digest(_domain_grid(decoded, 14))

    reference = gk._section_sample_key(geometry, mirrored, digest, role="positive")

    twin_geometry, twin_mirrored = _first_instance(_key_graph())
    assert (
        gk._section_sample_key(twin_geometry, twin_mirrored, digest, role="positive")
        == reference
    )

    moved_geometry, moved_mirrored = _first_instance(
        _key_graph(part_center=(-0.25, 0.0, 0.0))
    )
    assert (
        gk._section_sample_key(moved_geometry, moved_mirrored, digest, role="positive")
        != reference
    )

    assert (
        gk._section_sample_key(geometry, not mirrored, digest, role="positive")
        != reference
    )

    assert (
        gk._section_sample_key(geometry, mirrored, other_digest, role="positive")
        != reference
    )

    # Role quarantines the store across the sampler boundary: a carve entry
    # carries no gradient and is sampled by part_sdf_checked, not
    # prepared_part_sdf, so identical geometry under a different role must key
    # elsewhere or a positive section would warm-reject what cold accepts.
    assert (
        gk._section_sample_key(geometry, mirrored, digest, role="carve")
        != reference
    )

    # Operator governs composition, not section.evaluate's output over fixed
    # points: it must move the composed snapshot key but never the section key.
    operator_graph = _key_graph(part_operator="crease")
    operator_geometry, operator_mirrored = _first_instance(operator_graph)
    assert (
        gk._section_sample_key(
            operator_geometry, operator_mirrored, digest, role="positive"
        )
        == reference
    )
    assert (
        gk._composed_snapshot_key(_decode(operator_graph), points)
        != gk._composed_snapshot_key(decoded, points)
    )

    # Per-part blend is likewise excluded from the section key and named by the
    # composed key.
    blend_graph = _key_graph(part_blend=0.05)
    blend_geometry, blend_mirrored = _first_instance(blend_graph)
    assert (
        gk._section_sample_key(blend_geometry, blend_mirrored, digest, role="positive")
        == reference
    )
    assert (
        gk._composed_snapshot_key(_decode(blend_graph), points)
        != gk._composed_snapshot_key(decoded, points)
    )

    # Salt last: monkeypatching it mutates every subsequent key computation.
    monkeypatch.setattr(gk, "_field_source_salt", lambda: b"\x07" * 32)
    assert (
        gk._section_sample_key(geometry, mirrored, digest, role="positive")
        != reference
    )


def test_warm_edit_resamples_only_the_edited_section(stores, monkeypatch) -> None:
    base = _decode(_trio(base_size=0.5))
    edited = _decode(_trio(base_size=0.6))
    points = _production_grid(base, 16)
    production = points.shape[0]
    certification = _certification_count()

    section_calls = _section_spy(monkeypatch)
    carve_calls = _carve_spy(monkeypatch)

    first = _compose(base, points)
    assert isinstance(first, gk._ComposedField)
    mark = len(section_calls)

    second = _compose(edited, points)
    assert isinstance(second, gk._ComposedField)

    first_pass = section_calls[:mark]
    second_pass = section_calls[mark:]

    assert sorted(_part_ids_at(first_pass, certification)) == ["base", "bump", "spacer"]
    assert sorted(_part_ids_at(first_pass, production)) == ["base", "bump", "spacer"]

    assert sorted(_part_ids_at(second_pass, certification)) == ["base", "bump", "spacer"]
    assert _part_ids_at(second_pass, production) == ["base"]

    assert carve_calls == []


def test_warm_edited_field_is_bitwise_cold_recompute(stores, monkeypatch) -> None:
    base = _decode(_trio(base_size=0.5))
    edited = _decode(_trio(base_size=0.6))
    points = _production_grid(base, 16)

    predecessor = _compose(base, points)
    assert isinstance(predecessor, gk._ComposedField)

    warm = _compose(edited, points)

    monkeypatch.setenv(gk._FIELD_CACHE_COLD_ENV, "1")
    cold = _compose(edited, points)

    assert isinstance(warm, gk._ComposedField)
    assert isinstance(cold, gk._ComposedField)
    assert np.array_equal(warm.field, cold.field)


def test_cold_mode_resamples_everything_and_never_writes_sections(
    stores, monkeypatch
) -> None:
    _composed, _field, sections = stores
    monkeypatch.setenv(gk._FIELD_CACHE_COLD_ENV, "1")
    graph = _decode(_pair())
    points = _production_grid(graph, 16)
    production = points.shape[0]

    section_calls = _section_spy(monkeypatch)

    first = _compose(graph, points)
    mark = len(section_calls)
    second = _compose(graph, points)

    assert isinstance(first, gk._ComposedField)
    assert isinstance(second, gk._ComposedField)
    assert sorted(_part_ids_at(section_calls[:mark], production)) == ["alpha", "beta"]
    assert sorted(_part_ids_at(section_calls[mark:], production)) == ["alpha", "beta"]
    assert _store_is_empty(sections)


def test_rejected_graph_leaves_the_section_store_empty(stores) -> None:
    _composed, _field, sections = stores
    graph = _decode(_containment_reject())
    points = _production_grid(graph, 16)

    result = _compose(graph, points)

    assert isinstance(result, gk.RejectedSurfaceFormation)
    assert _store_is_empty(sections)


def test_carve_edit_resamples_only_the_carve(stores, monkeypatch) -> None:
    base = _decode(_carved(carve_y=0.4))
    edited = _decode(_carved(carve_y=0.5))
    points = _production_grid(base, 16)
    production = points.shape[0]

    section_calls = _section_spy(monkeypatch)
    carve_calls = _carve_spy(monkeypatch)

    first = _compose(base, points)
    assert isinstance(first, gk._ComposedField)
    section_mark = len(section_calls)
    carve_mark = len(carve_calls)

    warm = _compose(edited, points)
    assert isinstance(warm, gk._ComposedField)

    assert _part_ids_at(section_calls[section_mark:], production) == []
    assert _part_ids_at(carve_calls[carve_mark:], production) == ["carve"]

    monkeypatch.setenv(gk._FIELD_CACHE_COLD_ENV, "1")
    cold = _compose(edited, points)
    assert isinstance(cold, gk._ComposedField)
    assert np.array_equal(warm.field, cold.field)


def test_duplicate_geometry_collapses_to_one_section_entry(
    stores, monkeypatch
) -> None:
    _composed, _field, sections = stores
    first_graph = _decode(_twins(graph_blend=0.02))
    warm_graph = _decode(_twins(graph_blend=0.05))
    points = _production_grid(first_graph, 16)
    production = points.shape[0]

    first = _compose(first_graph, points)
    assert isinstance(first, gk._ComposedField)
    assert len(list(sections.rglob("*.npz"))) == 1

    section_calls = _section_spy(monkeypatch)
    warm = _compose(warm_graph, points)
    assert isinstance(warm, gk._ComposedField)
    assert _part_ids_at(section_calls, production) == []
