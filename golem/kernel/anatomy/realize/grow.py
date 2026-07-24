"""Flow-relaxation vascular grower: vasculature as adaptive transport physics.

A rival to greedy CCO terminal insertion.  Each lane's supply or return tree is
grown as a Physarum-style adaptive flow network: a conductance field over a
discretization of the contained flesh domain carries conserved flow from the
pump interface to the bed terminals, relaxes on the network Laplacian
(Jacobi-preconditioned conjugate gradient), and reinforces along Murray-optimal
channels until the network anneals toward a tree.  The converged flow selects a
rooted spanning tree whose radii the shared materializer derives by Murray's
law.  The proposal is handed to the identical judgment layer as CCO's; a failed
grow reports the same typed, quantitative obstructions, never a bare
non-convergence.

The relaxation solve is the sole numeric quarantine; everything around it is a
pure fold from the circuit domain to a local vascular tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from golem.kernel.anatomy.graph import (
    InfeasibleBifurcationObstruction,
    RejectedVasculature,
    VascularGrowthPhase,
    VascularInfeasibilityPredicate,
    VascularLane,
    VascularNodeKind,
)
from golem.kernel.anatomy.realize.carriers import _LocalVascularTree
from golem.kernel.anatomy.realize.clearance import _validate_vascular_geometry
from golem.kernel.anatomy.realize.materialize import (
    _glue_circuit_sections,
    _materialize_circuit_geometry,
    _merge_circuit_geometry,
    _reflect_circuit_geometry,
)
from golem.kernel.anatomy.realize.parity import _corridor_terminal_data

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from golem.kernel.anatomy.realize.carriers import (
        _CircuitConstructionDomain,
        _CircuitGeometry,
        _StructuralCostIndex,
    )
    from golem.kernel.anatomy.vocabulary import (
        CirculationCircuit,
        SealedVascularConfig,
    )


@dataclass(frozen=True)
class _GrownLane:
    tree: _LocalVascularTree
    iterations: int
    final_residual: float
    active_edge_count: int


@dataclass(frozen=True)
class _GrowStats:
    supply_iterations: int
    return_iterations: int
    supply_residual: float
    return_residual: float
    warm_started: bool


_LAST_GROW_STATS: dict[str, _GrowStats] = {}
_LAST_GROWN_GEOMETRY: dict[str, "_CircuitGeometry"] = {}
_WARM_FIELD: dict[
    str, tuple["NDArray[np.float64]", "NDArray[np.float64]"]
] = {}


@dataclass(frozen=True)
class FlowRelaxationGrower:
    """Grow each circuit's vasculature by relaxing an adaptive flow network."""

    steiner_samples: int = 256
    neighbor_count: int = 10
    iterations: int = 160
    reinforcement_exponent: float = 4.0 / 3.0
    adaptation_rate: float = 0.2
    prune_fraction: float = 0.04
    convergence_tolerance: float = 1.0e-4
    solve_tolerance: float = 1.0e-10
    warm_start: bool = False

    def construct(
        self,
        circuit: CirculationCircuit,
        domain: _CircuitConstructionDomain,
        config: SealedVascularConfig,
        structural_cost_index: _StructuralCostIndex | None,
    ) -> _CircuitGeometry | RejectedVasculature:
        return _grow_circuit(circuit, domain, config, self)


def last_grow_stats(region_id: str) -> _GrowStats | None:
    return _LAST_GROW_STATS.get(region_id)


def last_grown_geometry(region_id: str) -> "_CircuitGeometry | None":
    return _LAST_GROWN_GEOMETRY.get(region_id)


def reset_warm_field() -> None:
    _WARM_FIELD.clear()


def _grow_circuit(
    circuit: CirculationCircuit,
    domain: _CircuitConstructionDomain,
    config: SealedVascularConfig,
    grower: FlowRelaxationGrower,
) -> _CircuitGeometry | RejectedVasculature:
    _LAST_GROWN_GEOMETRY.pop(domain.region_id, None)
    terminal_data = _corridor_terminal_data(
        domain.paired_corridor,
        domain.territory_parts,
        domain.representative_count,
        domain.region_id,
        config,
    )
    if isinstance(terminal_data, InfeasibleBifurcationObstruction):
        return RejectedVasculature((terminal_data,))
    if not isinstance(terminal_data, tuple):
        return RejectedVasculature((terminal_data,))
    terminal_pairs, _exchange_obligations = terminal_data
    supply = _grow_lane(
        domain.paired_corridor.supply,
        tuple(pair.supply for pair in terminal_pairs),
        VascularNodeKind.SUPPLY_TERMINAL,
        VascularLane.SUPPLY,
        domain,
        config,
        grower,
        (),
    )
    if isinstance(supply, InfeasibleBifurcationObstruction):
        return RejectedVasculature((supply,))
    returning = _grow_lane(
        domain.paired_corridor.returning,
        tuple(pair.returning for pair in terminal_pairs),
        VascularNodeKind.RETURN_TERMINAL,
        VascularLane.RETURN,
        domain,
        config,
        grower,
        _reserved_trunk(supply.tree),
    )
    if isinstance(returning, InfeasibleBifurcationObstruction):
        return RejectedVasculature((returning,))
    _LAST_GROW_STATS[domain.region_id] = _GrowStats(
        supply.iterations,
        returning.iterations,
        supply.final_residual,
        returning.final_residual,
        grower.warm_start,
    )
    geometry = _materialize_grown_geometry(
        circuit, domain.mirrored, supply.tree, returning.tree, config
    )
    _LAST_GROWN_GEOMETRY[domain.region_id] = geometry
    obstructions = _grown_geometry_obstructions(geometry, domain, config)
    return (
        RejectedVasculature(obstructions) if obstructions else geometry
    )


def _grown_geometry_obstructions(
    geometry: _CircuitGeometry,
    domain: _CircuitConstructionDomain,
    config: SealedVascularConfig,
) -> tuple[object, ...]:
    glued = _glue_circuit_sections((geometry,))
    if isinstance(glued, RejectedVasculature):
        return glued.obstructions
    validation = _validate_vascular_geometry(
        glued, domain.containment_parts, config
    )
    return validation.obstructions


def _grow_lane(
    spine: tuple[tuple[float, float, float], ...],
    sinks: tuple[tuple[float, float, float], ...],
    terminal_kind: VascularNodeKind,
    lane: VascularLane,
    domain: _CircuitConstructionDomain,
    config: SealedVascularConfig,
    grower: FlowRelaxationGrower,
    reserved: tuple[tuple[float, float, float], ...],
) -> _GrownLane | InfeasibleBifurcationObstruction:
    if len(spine) < 2 or not sinks:
        return _search_exhausted(domain.region_id, lane, None)
    anchor = spine[-1]
    steiner = _sample_steiner_nodes(domain, config, grower)
    relaxed = _relax_flow_network(
        anchor,
        steiner,
        sinks,
        reserved,
        domain,
        config,
        grower,
        f"{domain.region_id}/{lane.value}",
    )
    if isinstance(relaxed, InfeasibleBifurcationObstruction):
        return relaxed
    positions, parents_into_grown, iterations, residual, active_edges = relaxed
    tree = _assemble_lane_tree(
        spine,
        positions,
        parents_into_grown,
        sinks,
        terminal_kind,
        domain.macro_node_count,
        domain.per_terminal_flow,
    )
    return _GrownLane(tree, iterations, residual, active_edges)


def _search_exhausted(
    region_id: str,
    lane: VascularLane,
    terminal_index: int | None,
) -> InfeasibleBifurcationObstruction:
    return InfeasibleBifurcationObstruction(
        region_id=region_id,
        phase=VascularGrowthPhase.SEED,
        lane=lane,
        predicate=VascularInfeasibilityPredicate.SEARCH_EXHAUSTED,
        required=1.0,
        observed=0.0,
        terminal_index=terminal_index,
    )


def _sample_steiner_nodes(
    domain: _CircuitConstructionDomain,
    config: SealedVascularConfig,
    grower: FlowRelaxationGrower,
) -> tuple[tuple[float, float, float], ...]:
    import math

    import numpy as np
    from scipy.stats import qmc

    from golem.kernel import engine as geometry
    from golem.kernel.anatomy.geometry import _union_sdf

    parts = domain.containment_parts
    lo, hi = geometry.graph_bounds({"parts": list(parts)}, pad=0.0)
    requested = max(grower.steiner_samples, 3 * domain.representative_count)
    exponent = max(1, int(math.ceil(math.log2(max(2, requested)))))
    unit = qmc.Sobol(d=3, scramble=False).random_base2(exponent)
    candidates = qmc.scale(unit, lo, hi)
    depth = config.wall_clearance + domain.maximum_tree_radius
    inside = candidates[_union_sdf(candidates, parts) <= -depth]
    return tuple(tuple(map(float, point)) for point in inside)


def _relax_flow_network(
    anchor: tuple[float, float, float],
    steiner: tuple[tuple[float, float, float], ...],
    sinks: tuple[tuple[float, float, float], ...],
    reserved: tuple[tuple[float, float, float], ...],
    domain: _CircuitConstructionDomain,
    config: SealedVascularConfig,
    grower: FlowRelaxationGrower,
    warm_key: str,
) -> (
    tuple[
        tuple[tuple[float, float, float], ...],
        tuple[int, ...],
        int,
        float,
        int,
    ]
    | InfeasibleBifurcationObstruction
):
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    from scipy.sparse.linalg import cg

    node_positions = np.asarray((anchor, *steiner, *sinks), dtype=np.float64)
    node_count = node_positions.shape[0]
    anchor_index = 0
    sink_indices = np.arange(node_count - len(sinks), node_count)
    edges = _knn_edge_index(node_positions, grower.neighbor_count)
    contained_edges = _contained_edges(
        node_positions,
        edges,
        domain.containment_parts,
        config.wall_clearance + domain.maximum_tree_radius,
    )
    contained_edges = _lane_clear_edges(
        node_positions,
        contained_edges,
        reserved,
        sink_indices,
        config.vessel_clearance + 2.0 * domain.maximum_tree_radius,
    )
    if contained_edges.shape[0] == 0:
        return _search_exhausted(domain.region_id, _lane_of(warm_key), 0)
    left = contained_edges[:, 0]
    right = contained_edges[:, 1]
    lengths = np.maximum(
        np.linalg.norm(
            node_positions[left] - node_positions[right], axis=1
        ),
        1.0e-9,
    )
    midpoints = 0.5 * (node_positions[left] + node_positions[right])
    demand = np.zeros(node_count, dtype=np.float64)
    demand[anchor_index] = 1.0
    demand[sink_indices] = -1.0 / len(sinks)
    conductivity = _initial_conductivity(midpoints, warm_key, grower)

    def solve(cond: "NDArray[np.float64]") -> "NDArray[np.float64]":
        conductance = cond / lengths
        weight = np.concatenate((conductance, conductance))
        rows = np.concatenate((left, right))
        cols = np.concatenate((right, left))
        adjacency = csr_matrix(
            (weight, (rows, cols)), shape=(node_count, node_count)
        )
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        laplacian = csr_matrix(
            (np.concatenate((degree, -weight)),
             (np.concatenate((np.arange(node_count), rows)),
              np.concatenate((np.arange(node_count), cols)))),
            shape=(node_count, node_count),
        )
        free = np.arange(node_count) != anchor_index
        reduced = laplacian[free][:, free]
        rhs = demand[free]
        jacobi = reduced.diagonal()
        preconditioner = None
        if np.all(jacobi > 0.0):
            from scipy.sparse import diags
            from scipy.sparse.linalg import LinearOperator

            inverse = diags(1.0 / jacobi)
            preconditioner = LinearOperator(
                reduced.shape, matvec=inverse.dot
            )
        solution, _info = cg(
            reduced,
            rhs,
            rtol=grower.solve_tolerance,
            maxiter=2000,
            M=preconditioner,
        )
        pressure = np.zeros(node_count, dtype=np.float64)
        pressure[free] = solution
        return pressure

    def descend(
        cond: "NDArray[np.float64]", step: int, residual: float
    ) -> tuple["NDArray[np.float64]", int, float]:
        if step >= grower.iterations or (
            step > 0 and residual < grower.convergence_tolerance
        ):
            return cond, step, residual
        pressure = solve(cond)
        flux = np.abs(
            (cond / lengths) * (pressure[left] - pressure[right])
        )
        scaled = flux / max(float(np.max(flux)), 1.0e-30)
        reinforced = scaled**grower.reinforcement_exponent
        updated = (
            1.0 - grower.adaptation_rate
        ) * cond + grower.adaptation_rate * reinforced
        change = float(np.max(np.abs(updated - cond)))
        return descend(updated, step + 1, change)

    converged, iterations, residual = descend(conductivity, 0, 1.0)
    _WARM_FIELD[warm_key] = (midpoints, converged)
    threshold = grower.prune_fraction * float(np.max(converged))
    active = converged >= threshold
    active_left = left[active]
    active_right = right[active]
    if active_left.shape[0] == 0:
        return _search_exhausted(domain.region_id, _lane_of(warm_key), 0)
    tree_weight = lengths[active] / np.maximum(converged[active], 1.0e-30)
    graph = csr_matrix(
        (
            np.concatenate((tree_weight, tree_weight)),
            (
                np.concatenate((active_left, active_right)),
                np.concatenate((active_right, active_left)),
            ),
        ),
        shape=(node_count, node_count),
    )
    labels = connected_components(graph, directed=False)[1]
    unreachable = tuple(
        int(index)
        for position, index in enumerate(sink_indices)
        if labels[index] != labels[anchor_index]
    )
    if unreachable:
        return _search_exhausted(
            domain.region_id,
            _lane_of(warm_key),
            int(np.where(sink_indices == unreachable[0])[0][0]),
        )
    _distances, predecessors = dijkstra(
        graph,
        directed=False,
        indices=anchor_index,
        return_predecessors=True,
    )
    return _prune_to_tree(
        node_positions, predecessors, anchor_index, sink_indices, iterations,
        residual, int(active_left.shape[0]),
    )


def _lane_of(warm_key: str) -> VascularLane:
    return (
        VascularLane.RETURN
        if warm_key.endswith(VascularLane.RETURN.value)
        else VascularLane.SUPPLY
    )


def _initial_conductivity(
    midpoints: "NDArray[np.float64]",
    warm_key: str,
    grower: FlowRelaxationGrower,
) -> "NDArray[np.float64]":
    import numpy as np
    from scipy.spatial import cKDTree

    warm = _WARM_FIELD.get(warm_key)
    if not grower.warm_start or warm is None:
        return np.ones(midpoints.shape[0], dtype=np.float64)
    old_midpoints, old_conductivity = warm
    _distance, nearest = cKDTree(old_midpoints).query(
        midpoints, k=1, workers=1
    )
    return old_conductivity[nearest]


def _knn_edge_index(
    node_positions: "NDArray[np.float64]", neighbor_count: int
) -> "NDArray[np.int64]":
    import numpy as np
    from scipy.spatial import cKDTree

    count = node_positions.shape[0]
    k = min(neighbor_count + 1, count)
    tree = cKDTree(node_positions)
    _distance, neighbors = tree.query(node_positions, k=k, workers=1)
    sources = np.repeat(np.arange(count), k)
    targets = neighbors.reshape(-1)
    ordered = np.sort(np.stack((sources, targets), axis=1), axis=1)
    distinct = ordered[ordered[:, 0] != ordered[:, 1]]
    return np.unique(distinct, axis=0)


def _reserved_trunk(
    tree: _LocalVascularTree,
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        position
        for position, ordinal in zip(
            tree.positions, tree.terminal_ordinals, strict=True
        )
        if ordinal is None
    )


def _lane_clear_edges(
    node_positions: "NDArray[np.float64]",
    edges: "NDArray[np.int64]",
    reserved: tuple[tuple[float, float, float], ...],
    sink_indices: "NDArray[np.int64]",
    clearance: float,
) -> "NDArray[np.int64]":
    import numpy as np

    if not reserved or edges.shape[0] == 0:
        return edges
    obstacle = np.asarray(reserved, dtype=np.float64)
    left = node_positions[edges[:, 0]]
    right = node_positions[edges[:, 1]]
    clear = np.ones(edges.shape[0], dtype=bool)
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        samples = (1.0 - t) * left + t * right
        gaps = np.linalg.norm(
            samples[:, None, :] - obstacle[None, :, :], axis=2
        )
        clear &= gaps.min(axis=1) >= clearance
    sink_set = set(int(index) for index in sink_indices)
    incident = np.array(
        [
            int(edge[0]) in sink_set or int(edge[1]) in sink_set
            for edge in edges
        ],
        dtype=bool,
    )
    return edges[clear | incident]


def _contained_edges(
    node_positions: "NDArray[np.float64]",
    edges: "NDArray[np.int64]",
    containment_parts: tuple[dict, ...],
    depth: float,
) -> "NDArray[np.int64]":
    import numpy as np

    from golem.kernel.anatomy.geometry import _union_sdf

    left = node_positions[edges[:, 0]]
    right = node_positions[edges[:, 1]]
    fractions = (0.2, 0.4, 0.5, 0.6, 0.8)
    samples = np.concatenate(
        tuple((1.0 - t) * left + t * right for t in fractions)
    )
    field = _union_sdf(samples, containment_parts).reshape(len(fractions), -1)
    inside = np.all(field <= -depth, axis=0)
    return edges[inside]


def _prune_to_tree(
    node_positions: "NDArray[np.float64]",
    predecessors: "NDArray[np.int64]",
    anchor_index: int,
    sink_indices: "NDArray[np.int64]",
    iterations: int,
    residual: float,
    active_edge_count: int,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[int, ...],
    int,
    float,
    int,
]:
    import numpy as np

    keep: set[int] = set()

    def climb(node: int) -> None:
        while node != anchor_index and node not in keep and node >= 0:
            keep.add(node)
            node = int(predecessors[node])

    for sink in sink_indices:
        keep.add(int(sink))
        climb(int(predecessors[int(sink)]))
    ordered = tuple(sorted(keep))
    remap = {anchor_index: 0}
    positions: list[tuple[float, float, float]] = [
        tuple(map(float, node_positions[anchor_index]))
    ]
    parents: list[int] = [-1]
    for original in ordered:
        remap[original] = len(positions)
        positions.append(tuple(map(float, node_positions[original])))
        parents.append(-1)
    for original in ordered:
        parent = int(predecessors[original])
        parents[remap[original]] = remap.get(parent, 0)
    return (
        tuple(positions),
        tuple(parents),
        iterations,
        residual,
        active_edge_count,
    )


def _assemble_lane_tree(
    spine: tuple[tuple[float, float, float], ...],
    grown_positions: tuple[tuple[float, float, float], ...],
    grown_parents: tuple[int, ...],
    sinks: tuple[tuple[float, float, float], ...],
    terminal_kind: VascularNodeKind,
    macro_node_count: int,
    per_terminal_flow: float,
) -> _LocalVascularTree:
    spine_count = len(spine)
    sink_ordinal = {
        _rounded(sink): ordinal for ordinal, sink in enumerate(sinks)
    }
    grown_offset = spine_count - 1
    positions = (*spine, *grown_positions[1:])
    parents = (
        -1,
        *tuple(range(spine_count - 1)),
        *tuple(
            grown_offset if parent == 0 else parent + grown_offset
            for parent in grown_parents[1:]
        ),
    )
    kinds = (
        *tuple(
            VascularNodeKind.MACRO_CORRIDOR
            if index < macro_node_count
            else VascularNodeKind.CORRIDOR
            for index in range(spine_count)
        ),
        *tuple(
            terminal_kind
            if _rounded(position) in sink_ordinal
            else VascularNodeKind.BIFURCATION
            for position in grown_positions[1:]
        ),
    )
    terminal_ordinals = (
        *tuple(None for _ in range(spine_count)),
        *tuple(
            sink_ordinal.get(_rounded(position))
            for position in grown_positions[1:]
        ),
    )
    terminal_flows = tuple(
        per_terminal_flow if ordinal is not None else 0.0
        for ordinal in terminal_ordinals
    )
    return _LocalVascularTree(
        positions=positions,
        parents=parents,
        kinds=kinds,
        terminal_flows=terminal_flows,
        terminal_ordinals=terminal_ordinals,
    )


def _rounded(point: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(round(coordinate * 1.0e9) for coordinate in point)


def _materialize_grown_geometry(
    circuit: CirculationCircuit,
    mirrored: bool,
    supply_tree: _LocalVascularTree,
    return_tree: _LocalVascularTree,
    config: SealedVascularConfig,
) -> _CircuitGeometry:
    positive = _materialize_circuit_geometry(
        circuit,
        "positive" if mirrored else "center",
        supply_tree,
        return_tree,
        config,
    )
    return (
        positive
        if not mirrored
        else _merge_circuit_geometry(
            positive, _reflect_circuit_geometry(positive)
        )
    )
