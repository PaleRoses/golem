"""Seeded space-colonization as an unfold with a post-order radius fold."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, reduce
import math

import numpy as np

from golem.conduits.growth.types import (
    AppendageChain,
    ContinueGrowth,
    GrowthNetwork,
    GrowthParameters,
    GrowthSection,
    GrowthState,
    GrowthTermination,
    TerminatedGrowth,
)
from golem.kernel import engine


@dataclass(frozen=True)
class _InteriorSamplingState:
    accepted_batches: tuple[np.ndarray, ...] = ()
    accepted_count: int = 0
    evaluated_batches: int = 0


def _read_only_array(values, *, dtype=None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


def _interior_points(
    parts: tuple[dict, ...],
    count: int,
    seed: int,
    margin: float = 0.02,
    max_batches: int = 200,
) -> np.ndarray:
    lower, upper = engine.graph_bounds({"parts": list(parts)})
    batch_size = max(count * 4, 256)

    def consume_candidate_prefix(
        candidates: np.ndarray, state: _InteriorSamplingState
    ) -> _InteriorSamplingState:
        if (
            state.accepted_count >= count
            or state.evaluated_batches * batch_size >= candidates.shape[0]
        ):
            return state
        batch_start = state.evaluated_batches * batch_size
        batch = candidates[batch_start : batch_start + batch_size]
        interior = batch[_part_union_field(parts, batch) < -margin]
        return consume_candidate_prefix(
            candidates,
            _InteriorSamplingState(
                accepted_batches=(*state.accepted_batches, interior),
                accepted_count=state.accepted_count + interior.shape[0],
                evaluated_batches=state.evaluated_batches + 1,
            ),
        )

    def collect(state: _InteriorSamplingState) -> np.ndarray:
        if state.accepted_count >= count:
            return _read_only_array(
                (
                    np.concatenate(state.accepted_batches, axis=0)[:count]
                    if state.accepted_batches
                    else np.empty((0, 3), dtype=np.float64)
                )
            )
        if state.evaluated_batches >= max_batches:
            if not state.accepted_batches:
                raise ValueError(
                    "appendage growth found no interior intent volume"
                )
            return _read_only_array(
                np.concatenate(state.accepted_batches, axis=0)[:count]
            )
        next_batch_count = min(
            max_batches,
            1
            if state.evaluated_batches == 0
            else 2 * state.evaluated_batches,
        )
        candidate_prefix = np.random.RandomState(seed).uniform(
            lower, upper, size=(next_batch_count * batch_size, 3)
        )
        return collect(consume_candidate_prefix(candidate_prefix, state))

    return collect(_InteriorSamplingState())


def _part_union_field(
    parts: tuple[dict, ...], points: np.ndarray
) -> np.ndarray:
    return reduce(
        np.minimum,
        (
            engine.part_sdf(points, part, mirrored)
            for part, mirrored in engine.instances({"parts": list(parts)})
        ),
        np.full(points.shape[0], np.inf, dtype=np.float64),
    )


def _inside_parts(
    parts: tuple[dict, ...], point: np.ndarray, margin: float
) -> bool:
    return bool(_part_union_field(parts, point[None, :])[0] < -margin)


def _candidate_branch(
    parameters: GrowthParameters,
    node_array: np.ndarray,
    influenced_attractors: np.ndarray,
    influenced_nearest: np.ndarray,
    node_index: int,
) -> tuple[np.ndarray, int] | None:
    selected = influenced_attractors[influenced_nearest == node_index]
    directions = selected - node_array[node_index]
    normalized_directions = directions / np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), 1.0e-12
    )
    mean_direction = normalized_directions.sum(axis=0)
    magnitude = float(np.linalg.norm(mean_direction))
    if magnitude < 1.0e-9:
        return None
    unit_direction = mean_direction / magnitude
    full_step = node_array[node_index] + parameters.step * unit_direction
    half_step = node_array[node_index] + 0.5 * parameters.step * unit_direction
    accepted_point = (
        full_step
        if _inside_parts(
            parameters.parts, full_step, parameters.interior_margin
        )
        else half_step
        if _inside_parts(
            parameters.parts, half_step, parameters.interior_margin
        )
        else None
    )
    return (
        (_read_only_array(accepted_point), node_index)
        if accepted_point is not None
        else None
    )


def _advance_growth(
    parameters: GrowthParameters, state: GrowthState
) -> GrowthSection:
    if state.section >= parameters.maximum_sections:
        return TerminatedGrowth(state, GrowthTermination.MAXIMUM_SECTIONS)
    if not state.alive.any():
        return TerminatedGrowth(state, GrowthTermination.ATTRACTORS_EXHAUSTED)
    living_attractors = state.attractors[state.alive]
    distances = np.linalg.norm(
        living_attractors[:, None, :] - state.nodes[None, :, :], axis=2
    )
    nearest = np.argmin(distances, axis=1)
    influenced_mask = (
        distances[np.arange(living_attractors.shape[0]), nearest]
        < parameters.influence
    )
    if not influenced_mask.any():
        return TerminatedGrowth(
            state, GrowthTermination.NO_INFLUENCED_ATTRACTORS
        )
    influenced_attractors = living_attractors[influenced_mask]
    influenced_nearest = nearest[influenced_mask]
    candidates = tuple(
        candidate
        for node_index in map(int, np.unique(influenced_nearest))
        for candidate in (
            _candidate_branch(
                parameters,
                state.nodes,
                influenced_attractors,
                influenced_nearest,
                node_index,
            ),
        )
        if candidate is not None
    )
    if not candidates:
        return TerminatedGrowth(state, GrowthTermination.NO_INTERIOR_BRANCH)
    next_nodes = _read_only_array(
        np.concatenate(
            (
                state.nodes,
                np.asarray(tuple(point for point, _parent in candidates)),
            ),
            axis=0,
        )
    )
    next_parents = _read_only_array(
        np.concatenate(
            (
                state.parents,
                np.asarray(
                    tuple(parent for _point, parent in candidates),
                    dtype=np.int64,
                ),
            )
        ),
        dtype=np.int64,
    )
    living_indices = np.flatnonzero(state.alive)
    nearest_next_distance = np.linalg.norm(
        state.attractors[living_indices][:, None, :]
        - next_nodes[None, :, :],
        axis=2,
    ).min(axis=1)
    surviving_indices = living_indices[
        nearest_next_distance >= parameters.kill
    ]
    next_alive = _read_only_array(
        np.isin(
            np.arange(state.attractors.shape[0]),
            surviving_indices,
            assume_unique=True,
        ),
        dtype=bool,
    )
    return ContinueGrowth(
        GrowthState(
            nodes=next_nodes,
            parents=next_parents,
            attractors=state.attractors,
            alive=next_alive,
            section=state.section + 1,
        )
    )


def _unfold_growth(
    parameters: GrowthParameters, state: GrowthState
) -> TerminatedGrowth:
    section = _advance_growth(parameters, state)
    match section:
        case ContinueGrowth(next_state):
            return _unfold_growth(parameters, next_state)
        case TerminatedGrowth():
            return section


def _children_by_parent(parents: np.ndarray) -> tuple[tuple[int, ...], ...]:
    node_indices = np.arange(parents.shape[0], dtype=np.int64)
    parent_order = np.argsort(parents, kind="stable")
    sorted_parents = parents[parent_order]
    starts = np.searchsorted(sorted_parents, node_indices, side="left")
    ends = np.searchsorted(sorted_parents, node_indices, side="right")
    return tuple(
        tuple(map(int, parent_order[start:end]))
        for start, end in zip(starts, ends)
    )


def _murray_radii(
    parents: np.ndarray,
    minimum_radius: float,
    root_radius: float,
    murray_exponent: float,
) -> np.ndarray:
    children = _children_by_parent(parents)

    @cache
    def radius_at(node_index: int) -> float:
        child_indices = children[node_index]
        return (
            min(
                math.fsum(
                    radius_at(child_index) ** murray_exponent
                    for child_index in child_indices
                )
                ** (1.0 / murray_exponent),
                root_radius,
            )
            if child_indices
            else minimum_radius
        )

    return _read_only_array(
        tuple(map(radius_at, range(parents.shape[0]))), dtype=np.float64
    )


def grow_appendage_network(
    parts: list[dict] | tuple[dict, ...],
    root: np.ndarray,
    count: int = 220,
    seed: int = 7,
    influence: float = 0.30,
    kill: float = 0.055,
    step: float = 0.045,
    max_iter: int = 400,
    min_radius: float = 0.014,
    root_radius: float = 0.034,
    murray_m: float = 3.0,
) -> GrowthNetwork:
    authored_parts = tuple(parts)
    attractors = _interior_points(authored_parts, count, seed)
    initial = GrowthState(
        nodes=_read_only_array(
            np.asarray(root, dtype=np.float64)[None, :]
        ),
        parents=_read_only_array((-1,), dtype=np.int64),
        attractors=attractors,
        alive=_read_only_array(
            np.ones(attractors.shape[0], dtype=bool), dtype=bool
        ),
    )
    completed = _unfold_growth(
        GrowthParameters(
            parts=authored_parts,
            influence=influence,
            kill=kill,
            step=step,
            maximum_sections=max_iter,
        ),
        initial,
    )
    return GrowthNetwork(
        nodes=completed.state.nodes,
        parents=completed.state.parents,
        radii=_murray_radii(
            completed.state.parents,
            min_radius,
            root_radius,
            murray_m,
        ),
        termination=completed.termination,
    )


def _chain_indices(
    children: tuple[tuple[int, ...], ...], current: int
) -> tuple[int, ...]:
    successors = children[current]
    return (
        (current, *_chain_indices(children, successors[0]))
        if len(successors) == 1
        else (current,)
    )


def _dp_simplify(
    points: np.ndarray, radii: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= 2:
        return points, radii
    segment = points[-1] - points[0]
    segment_length = float(np.linalg.norm(segment))
    interior_points = points[1:-1]
    distances = (
        np.linalg.norm(interior_points - points[0], axis=1)
        if segment_length < 1.0e-12
        else np.linalg.norm(
            interior_points
            - (
                points[0]
                + np.clip(
                    ((interior_points - points[0]) @ segment)
                    / (segment_length * segment_length),
                    0.0,
                    1.0,
                )[:, None]
                * segment
            ),
            axis=1,
        )
    )
    farthest_interior_index = int(np.argmax(distances))
    if float(distances[farthest_interior_index]) <= tolerance:
        return points[[0, -1]], radii[[0, -1]]
    split_index = farthest_interior_index + 1
    left_points, left_radii = _dp_simplify(
        points[: split_index + 1], radii[: split_index + 1], tolerance
    )
    right_points, right_radii = _dp_simplify(
        points[split_index:], radii[split_index:], tolerance
    )
    return (
        np.concatenate((left_points[:-1], right_points), axis=0),
        np.concatenate((left_radii[:-1], right_radii), axis=0),
    )


def consolidate_appendage_chains(
    nodes: np.ndarray,
    parents: np.ndarray,
    radii: np.ndarray,
    simplify_tol: float = 0.02,
) -> tuple[AppendageChain, ...]:
    children = _children_by_parent(parents)
    sources = tuple(
        node_index
        for node_index in range(nodes.shape[0])
        if parents[node_index] < 0 or len(children[node_index]) > 1
    )
    edge_chains = tuple(
        (source, *_chain_indices(children, first_child))
        for source in sources
        for first_child in children[source]
    )

    def simplify_chain(indices: tuple[int, ...]) -> AppendageChain:
        index_array = np.asarray(indices, dtype=np.int64)
        spine, chain_radii = _dp_simplify(
            nodes[index_array], radii[index_array], simplify_tol
        )
        return {"spine": spine, "radii": chain_radii}

    return tuple(map(simplify_chain, edge_chains))


def _smooth_spine_once(spine: np.ndarray) -> np.ndarray:
    return (
        spine
        if spine.shape[0] <= 2
        else np.concatenate(
            (
                spine[:1],
                0.5 * spine[1:-1] + 0.25 * (spine[:-2] + spine[2:]),
                spine[-1:],
            ),
            axis=0,
        )
    )


def _compiled_chain_part(
    ordinal: int,
    chain: AppendageChain,
    radius_mean: float,
    minimum_radius_mean: float,
    maximum_radius_mean: float,
    envelope: float,
    minimum_radius: float,
    tip_taper: float,
    smooth_iterations: int,
    envelope_tip: float | None,
) -> dict:
    spine = reduce(
        lambda current, _iteration: _smooth_spine_once(current),
        range(max(0, smooth_iterations)),
        np.asarray(chain["spine"], dtype=np.float64),
    )
    raw_radii = np.asarray(chain["radii"], dtype=np.float64)
    steady_radii = 0.45 * raw_radii + 0.55 * float(np.mean(raw_radii))
    envelope_factor = (
        envelope_tip
        + (envelope - envelope_tip)
        * (radius_mean - minimum_radius_mean)
        / (maximum_radius_mean - minimum_radius_mean)
        if envelope_tip is not None
        and maximum_radius_mean > minimum_radius_mean + 1.0e-12
        else envelope
    )
    enveloped_radii = np.maximum(
        steady_radii * envelope_factor, minimum_radius
    )
    tapered_radii = np.concatenate(
        (enveloped_radii[:-1], enveloped_radii[-1:] * tip_taper)
    )
    return {
        "id": f"limb_{ordinal}",
        "type": "gencyl",
        "spine": tuple(tuple(map(float, point)) for point in spine),
        "radii": tuple(map(float, tapered_radii)),
    }


def appendage_envelope_graph(
    chains: list[AppendageChain] | tuple[AppendageChain, ...],
    envelope: float = 2.8,
    min_radius: float = 0.028,
    tip_taper: float = 0.75,
    blend: float = 0.03,
    smooth_iters: int = 2,
    envelope_tip: float | None = None,
) -> dict:
    authored_chains = tuple(chains)
    if not authored_chains:
        raise ValueError("no chains to envelope")
    radius_means = tuple(
        float(np.mean(chain["radii"])) for chain in authored_chains
    )
    minimum_radius_mean = min(radius_means)
    maximum_radius_mean = max(radius_means)
    parts = tuple(
        _compiled_chain_part(
            ordinal,
            chain,
            radius_means[ordinal],
            minimum_radius_mean,
            maximum_radius_mean,
            envelope,
            min_radius,
            tip_taper,
            smooth_iters,
            envelope_tip,
        )
        for ordinal, chain in enumerate(authored_chains)
    )
    return {"name": "grown_form", "blend": blend, "parts": parts}


__all__ = [
    "appendage_envelope_graph",
    "consolidate_appendage_chains",
    "grow_appendage_network",
]
