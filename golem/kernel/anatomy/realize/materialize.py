"""Circuit-geometry materialization, reflection, and gluing."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import groupby
from typing import TYPE_CHECKING

from golem.kernel.anatomy.graph import (
    ClosedVascularGraph,
    DisconnectedVascularGraphObstruction,
    RejectedVasculature,
    VascularEdge,
    VascularGluingObstruction,
    VascularNode,
    VascularNodeKind,
    VascularStratum,
)
from golem.kernel.anatomy.realize.carriers import _CircuitGeometry
from golem.kernel.anatomy.realize.tree import _subtree_flow

if TYPE_CHECKING:
    import numpy as np

    from golem.kernel.anatomy.graph import VascularSegment
    from golem.kernel.anatomy.realize.carriers import _LocalVascularTree
    from golem.kernel.anatomy.vocabulary import CirculationCircuit, SealedVascularConfig


def _materialize_circuit_geometry(
    circuit: CirculationCircuit,
    side: str,
    supply_tree: _LocalVascularTree,
    return_tree: _LocalVascularTree,
    config: SealedVascularConfig,
) -> _CircuitGeometry:
    region_id = circuit.capillary_bed.region.region_id
    supply_nodes = _materialize_tree_nodes(
        supply_tree, region_id, side, "s", VascularNodeKind.PUMP_OUTLET
    )
    return_nodes = _materialize_tree_nodes(
        return_tree, region_id, side, "r", VascularNodeKind.PUMP_INLET
    )
    supply_id_by_index = tuple(map(lambda node: node.node_id, supply_nodes))
    return_id_by_index = tuple(map(lambda node: node.node_id, return_nodes))
    supply_edges = _materialize_tree_edges(
        circuit,
        supply_tree,
        supply_id_by_index,
        VascularStratum.SUPPLY,
        config,
    )
    return_edges = _materialize_tree_edges(
        circuit,
        return_tree,
        return_id_by_index,
        VascularStratum.RETURN,
        config,
    )
    supply_terminals = tuple(
        (ordinal, supply_id_by_index[index])
        for index, ordinal in enumerate(supply_tree.terminal_ordinals)
        if ordinal is not None
    )
    return_terminals = tuple(
        (ordinal, return_id_by_index[index])
        for index, ordinal in enumerate(return_tree.terminal_ordinals)
        if ordinal is not None
    )
    return_by_ordinal = dict(return_terminals)
    exchange_edges = tuple(
        VascularEdge(
            edge_id=(
                f"exchange:{supply_node_id}>{return_by_ordinal[ordinal]}"
            ),
            source_node_id=supply_node_id,
            target_node_id=return_by_ordinal[ordinal],
            stage=circuit.capillary_bed,
            stratum=VascularStratum.EXCHANGE,
            radius=config.terminal_radius * config.exchange_radius_ratio,
            target_flow=circuit.effective_demand
            / (
                len(supply_terminals)
                * (2 if side == "positive" else 1)
            ),
            solved_flow=0.0,
        )
        for ordinal, supply_node_id in supply_terminals
    )
    return _CircuitGeometry(
        nodes=(*supply_nodes, *return_nodes),
        edges=(*supply_edges, *return_edges, *exchange_edges),
        supply_terminal_by_ordinal=supply_terminals,
        return_terminal_by_ordinal=return_terminals,
    )


def _materialize_tree_nodes(
    tree: _LocalVascularTree,
    region_id: str,
    side: str,
    lane: str,
    root_kind: VascularNodeKind,
) -> tuple[VascularNode, ...]:
    return tuple(
        VascularNode(
            node_id=(
                "pump:outlet"
                if index == 0 and lane == "s"
                else "pump:inlet"
                if index == 0
                else _derived_node_id(
                    region_id, side, lane, index, position, kind
                )
            ),
            position=position,
            kind=root_kind if index == 0 else kind,
            region_id=None if index == 0 else region_id,
        )
        for index, (position, kind) in enumerate(zip(tree.positions, tree.kinds))
    )


def _derived_node_id(
    region_id: str,
    side: str,
    lane: str,
    index: int,
    position: tuple[float, float, float],
    kind: VascularNodeKind,
) -> str:
    if kind in (
        VascularNodeKind.MACRO_CORRIDOR,
        VascularNodeKind.CORRIDOR,
    ) and abs(position[0]) <= 1.0e-15:
        x, y, z = position
        return f"vascular:shared:{lane}:{x:.12g}:{y:.12g}:{z:.12g}"
    return f"vascular:{region_id}:{side}:{lane}:{index}"


def _materialize_tree_edges(
    circuit: CirculationCircuit,
    tree: _LocalVascularTree,
    node_ids: tuple[str, ...],
    stratum: VascularStratum,
    config: SealedVascularConfig,
) -> tuple[VascularEdge, ...]:
    return tuple(
        _materialize_tree_edge(
            circuit, tree, node_ids, child_index, parent_index, stratum, config
        )
        for child_index, parent_index in enumerate(tree.parents)
        if parent_index >= 0
    )


def _materialize_tree_edge(
    circuit: CirculationCircuit,
    tree: _LocalVascularTree,
    node_ids: tuple[str, ...],
    child_index: int,
    parent_index: int,
    stratum: VascularStratum,
    config: SealedVascularConfig,
) -> VascularEdge:
    flow = _subtree_flow(tree, child_index)
    minimum_terminal_flow = min(
        filter(lambda value: value > 0.0, tree.terminal_flows)
    )
    radius = config.terminal_radius * (
        flow / minimum_terminal_flow
    ) ** (1.0 / config.murray_exponent)
    is_terminal = tree.terminal_ordinals[child_index] is not None
    if stratum is VascularStratum.SUPPLY:
        source, target = node_ids[parent_index], node_ids[child_index]
        stage: VascularSegment = (
            circuit.resistance_arteriole
            if is_terminal
            else circuit.distributing_arteries[-1]
        )
    else:
        source, target = node_ids[child_index], node_ids[parent_index]
        stage = (
            circuit.collecting_venule
            if is_terminal
            else circuit.returning_veins[0]
        )
    return VascularEdge(
        edge_id=f"{stratum.value}:{source}>{target}",
        source_node_id=source,
        target_node_id=target,
        stage=stage,
        stratum=stratum,
        radius=radius,
        target_flow=flow,
        solved_flow=0.0,
    )


def _reflect_circuit_geometry(section: _CircuitGeometry) -> _CircuitGeometry:
    reflected_nodes = tuple(map(_reflect_vascular_node, section.nodes))
    reflected_node_id = {
        original.node_id: reflected.node_id
        for original, reflected in zip(section.nodes, reflected_nodes)
    }
    reflected_edges = tuple(
        replace(
            edge,
            edge_id=(
                f"{edge.stratum.value}:"
                f"{reflected_node_id[edge.source_node_id]}>"
                f"{reflected_node_id[edge.target_node_id]}"
            ),
            source_node_id=reflected_node_id[edge.source_node_id],
            target_node_id=reflected_node_id[edge.target_node_id],
        )
        for edge in section.edges
    )
    return _CircuitGeometry(
        nodes=reflected_nodes,
        edges=reflected_edges,
        supply_terminal_by_ordinal=tuple(
            (ordinal, reflected_node_id[node_id])
            for ordinal, node_id in section.supply_terminal_by_ordinal
        ),
        return_terminal_by_ordinal=tuple(
            (ordinal, reflected_node_id[node_id])
            for ordinal, node_id in section.return_terminal_by_ordinal
        ),
    )


def _reflect_vascular_node(node: VascularNode) -> VascularNode:
    x, y, z = node.position
    reflected_id = (
        node.node_id.replace(":positive:", ":negative:")
        if ":positive:" in node.node_id
        else node.node_id
    )
    return replace(node, node_id=reflected_id, position=(-x, y, z))


def _merge_circuit_geometry(
    left: _CircuitGeometry, right: _CircuitGeometry
) -> _CircuitGeometry:
    return _CircuitGeometry(
        nodes=(*left.nodes, *right.nodes),
        edges=(*left.edges, *right.edges),
        supply_terminal_by_ordinal=(
            *left.supply_terminal_by_ordinal,
            *right.supply_terminal_by_ordinal,
        ),
        return_terminal_by_ordinal=(
            *left.return_terminal_by_ordinal,
            *right.return_terminal_by_ordinal,
        ),
    )


def _glue_circuit_sections(
    sections: tuple[_CircuitGeometry, ...],
) -> ClosedVascularGraph | RejectedVasculature:
    node_groups = tuple(
        tuple(group)
        for _node_id, group in groupby(
            sorted(
                (node for section in sections for node in section.nodes),
                key=lambda node: node.node_id,
            ),
            key=lambda node: node.node_id,
        )
    )
    mismatches = tuple(
        VascularGluingObstruction(
            nodes[0].node_id,
            "local sections disagree on a shared node position",
        )
        for nodes in node_groups
        if any(node.position != nodes[0].position for node in nodes[1:])
    )
    if mismatches:
        return RejectedVasculature(mismatches)
    nodes = tuple(group[0] for group in node_groups)
    edge_groups = tuple(
        tuple(group)
        for _edge_id, group in groupby(
            sorted(
                (edge for section in sections for edge in section.edges),
                key=lambda edge: edge.edge_id,
            ),
            key=lambda edge: edge.edge_id,
        )
    )
    graph = ClosedVascularGraph(
        nodes=nodes,
        edges=tuple(map(_glue_edge_group, edge_groups)),
        pump_outlet_node_id="pump:outlet",
        pump_inlet_node_id="pump:inlet",
    )
    disconnected = _disconnected_node(graph)
    return (
        graph
        if disconnected is None
        else RejectedVasculature(
            (DisconnectedVascularGraphObstruction(disconnected),)
        )
    )


def _glue_edge_group(edges: tuple[VascularEdge, ...]) -> VascularEdge:
    first = edges[0]
    exponent = 3.0
    return replace(
        first,
        radius=math.fsum(edge.radius**exponent for edge in edges)
        ** (1.0 / exponent),
        target_flow=math.fsum(edge.target_flow for edge in edges),
        solved_flow=0.0,
    )


def _disconnected_node(graph: ClosedVascularGraph) -> str | None:
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    node_index = {node.node_id: index for index, node in enumerate(graph.nodes)}
    pairs = np.asarray(
        tuple(
            (node_index[edge.source_node_id], node_index[edge.target_node_id])
            for edge in graph.edges
        ),
        dtype=np.int64,
    )
    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
    adjacency = csr_matrix(
        (np.ones(rows.shape[0]), (rows, cols)),
        shape=(len(graph.nodes), len(graph.nodes)),
    )
    component_count, labels = connected_components(adjacency, directed=False)
    return (
        None
        if component_count == 1
        else next(
            node.node_id
            for index, node in enumerate(graph.nodes)
            if labels[index] != labels[0]
        )
    )
