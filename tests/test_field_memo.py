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


def _compose(graph, points: np.ndarray):
    return gk._sample_graph_field_certified_checked(graph, points)


def _fold_spy(monkeypatch) -> list[tuple[int, object]]:
    calls: list[tuple[int, object]] = []
    real = gk._sample_graph_field_certified_fold

    def spy(graph, points, topology_shape=None, executor=None):
        calls.append(
            (int(np.asarray(points).reshape((-1, 3)).shape[0]), topology_shape)
        )
        return real(graph, points, topology_shape, executor)

    monkeypatch.setattr(gk, "_sample_graph_field_certified_fold", spy)
    return calls


def _production_counts(calls: list[tuple[int, object]]) -> list[int]:
    return [count for count, shape in calls if shape is None]


def _store_is_empty(root) -> bool:
    return not root.exists() or not any(
        path.is_file() for path in root.rglob("*")
    )


@pytest.fixture
def stores(tmp_path, monkeypatch):
    composed = tmp_path / "composed"
    field = tmp_path / "field"
    monkeypatch.setattr(gk, "_composed_snapshot_root", lambda: composed)
    monkeypatch.setattr(gk, "_field_cache_root", lambda: field)
    # Warm-path semantics are the subject; the ambient binding-gate cold
    # env must not decide which path these tests exercise.
    monkeypatch.delenv(gk._FIELD_CACHE_COLD_ENV, raising=False)
    return composed, field


def _local_pair(
    *,
    graph_blend: float = 0.02,
    base_center: float = -0.2,
    base_size: float = 0.5,
    base_operator: str = "blend",
    bump_center: float = 0.2,
    bump_blend: float | None = None,
) -> dict:
    bump: dict[str, object] = {
        "id": "bump",
        "type": "blob",
        "center": [bump_center, 0.0, 0.0],
        "size": [0.5, 0.5, 0.5],
        "operator": "local_blend",
    }
    if bump_blend is not None:
        bump["blend"] = bump_blend
    return {
        "name": "local_pair",
        "blend": graph_blend,
        "parts": [
            {
                "id": "base",
                "type": "blob",
                "center": [base_center, 0.0, 0.0],
                "size": [base_size, base_size, base_size],
                "operator": base_operator,
            },
            bump,
        ],
    }


def _base_key_graph() -> dict:
    return {
        "name": "memo_key",
        "blend": 0.03,
        "parts": [
            {
                "id": "base",
                "type": "blob",
                "center": [-0.2, 0.0, 0.0],
                "size": [0.6, 0.6, 0.6],
                "operator": "blend",
            },
            {
                "id": "bump",
                "type": "blob",
                "center": [0.2, 0.0, 0.0],
                "size": [0.6, 0.6, 0.6],
                "operator": "local_blend",
                "blend": 0.03,
            },
        ],
        "carves": [
            {
                "id": "carve",
                "type": "blob",
                "center": [0.0, 0.5, 0.0],
                "size": [0.3, 0.3, 0.3],
            }
        ],
    }


def _containment_reject_graph() -> dict:
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


def _adversary(*, base_size: float) -> dict:
    return {
        "name": "adversary",
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


def test_snapshot_key_names_every_governing_input(monkeypatch) -> None:
    base_graph = _decode(_base_key_graph())
    points = _domain_grid(base_graph, 12)

    def key(graph_dict: dict) -> str:
        return gk._composed_snapshot_key(_decode(graph_dict), points)

    reference = key(_base_key_graph())
    assert reference == key(_base_key_graph())

    part_center = _base_key_graph()
    part_center["parts"][0]["center"] = [-0.25, 0.0, 0.0]
    assert key(part_center) != reference

    mirror_flag = _base_key_graph()
    mirror_flag["parts"][0]["mirror"] = True
    assert key(mirror_flag) != reference

    operator = _base_key_graph()
    operator["parts"][0]["operator"] = "crease"
    assert key(operator) != reference

    part_blend = _base_key_graph()
    part_blend["parts"][1]["blend"] = 0.05
    assert key(part_blend) != reference

    graph_blend = _base_key_graph()
    graph_blend["blend"] = 0.06
    assert key(graph_blend) != reference

    carve = _base_key_graph()
    carve["carves"][0]["center"] = [0.0, 0.6, 0.0]
    assert key(carve) != reference

    assert (
        gk._composed_snapshot_key(base_graph, _domain_grid(base_graph, 14))
        != reference
    )

    certification = gk._LOCAL_TOPOLOGY_CERTIFICATION_RESOLUTION
    monkeypatch.setattr(
        gk, "_LOCAL_TOPOLOGY_CERTIFICATION_RESOLUTION", certification + 2
    )
    assert key(_base_key_graph()) != reference
    monkeypatch.setattr(
        gk, "_LOCAL_TOPOLOGY_CERTIFICATION_RESOLUTION", certification
    )
    assert key(_base_key_graph()) == reference

    monkeypatch.setattr(gk, "_field_source_salt", lambda: b"\x01" * 32)
    assert key(_base_key_graph()) != reference


def test_warm_call_returns_bitwise_field_without_reevaluating(
    stores, monkeypatch
) -> None:
    graph = _decode(_local_pair())
    points = _domain_grid(graph, 16)
    calls = _fold_spy(monkeypatch)

    first = _compose(graph, points)
    after_first = len(calls)
    second = _compose(graph, points)

    assert isinstance(first, gk._ComposedField)
    assert isinstance(second, gk._ComposedField)
    assert after_first >= 1
    assert len(calls) == after_first
    assert np.array_equal(first.field, second.field)
    assert first.evidence == second.evidence


def test_cold_mode_recomputes_and_never_touches_the_store(
    stores, monkeypatch
) -> None:
    composed, _field = stores
    monkeypatch.setenv(gk._FIELD_CACHE_COLD_ENV, "1")
    graph = _decode(_local_pair())
    points = _domain_grid(graph, 16)
    calls = _fold_spy(monkeypatch)

    first = _compose(graph, points)
    after_first = len(calls)
    second = _compose(graph, points)

    assert isinstance(first, gk._ComposedField)
    assert isinstance(second, gk._ComposedField)
    assert after_first >= 1
    assert len(calls) > after_first
    assert _store_is_empty(composed)


def test_rejected_graph_writes_no_snapshot(stores) -> None:
    composed, _field = stores
    graph = _decode(_containment_reject_graph())
    points = _domain_grid(graph, 12)

    result = _compose(graph, points)

    assert isinstance(result, gk.RejectedSurfaceFormation)
    assert _store_is_empty(composed)


def test_edited_graph_recomputes_fully_and_stores_new_snapshot(
    stores, monkeypatch
) -> None:
    composed, _field = stores
    base = _decode(_local_pair(bump_blend=0.03))
    edited = _decode(_local_pair(bump_blend=0.02))
    points = _domain_grid(base, 20)
    full = points.shape[0]

    predecessor = _compose(base, points)
    assert isinstance(predecessor, gk._ComposedField)

    calls = _fold_spy(monkeypatch)
    warm = _compose(edited, points)
    production = _production_counts(calls)

    monkeypatch.setenv(gk._FIELD_CACHE_COLD_ENV, "1")
    cold = _compose(edited, points)

    assert isinstance(warm, gk._ComposedField)
    assert isinstance(cold, gk._ComposedField)
    assert full in production
    assert np.array_equal(warm.field, cold.field)

    base_key = gk._composed_snapshot_key(base, points)
    edited_key = gk._composed_snapshot_key(edited, points)
    assert base_key != edited_key
    snapshots = sorted(path.name for path in composed.glob("*.pkl"))
    assert snapshots == sorted((f"{base_key}.pkl", f"{edited_key}.pkl"))


def test_global_support_adversary_convicts_naive_envelope_splicing(
    stores, monkeypatch
) -> None:
    base = _decode(_adversary(base_size=0.5))
    edited = _decode(_adversary(base_size=0.6))
    points = _domain_grid(base, 20)

    predecessor = _compose(base, points)
    warm = _compose(edited, points)

    monkeypatch.setenv(gk._FIELD_CACHE_COLD_ENV, "1")
    cold = _compose(edited, points)

    assert isinstance(predecessor, gk._ComposedField)
    assert isinstance(warm, gk._ComposedField)
    assert isinstance(cold, gk._ComposedField)

    lower, upper = gk.graph_bounds_checked(edited)
    diagonal = float(np.linalg.norm(np.asarray(upper) - np.asarray(lower)))
    blend_radius = edited.blend * diagonal
    size = np.asarray([0.6, 0.6, 0.6])
    center = np.zeros(3)
    extent = size * (1.0 + 2.0 * blend_radius / float(np.min(size)))
    margin = 0.15
    envelope_lower = center - extent - margin
    envelope_upper = center + extent + margin
    outside = np.any(
        (points < envelope_lower) | (points > envelope_upper), axis=1
    )
    changed = predecessor.field != cold.field

    # global-min argmin-at-a-distance: growing the base lowers its raw SDF, so
    # under the fold's unclipped min it captures cells far beyond any bounded
    # support envelope a naive splice would trust.
    assert np.any(changed & outside)
    assert np.array_equal(warm.field, cold.field)
