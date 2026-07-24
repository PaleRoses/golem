"""Local vascular tree segment, capsule, and subtree-flow folds."""

from __future__ import annotations

import math
from functools import cache
from itertools import groupby
from typing import TYPE_CHECKING

from golem.kernel.anatomy.realize.carriers import _SegmentCapsule

if TYPE_CHECKING:
    import numpy as np

    from golem.kernel.anatomy.realize.carriers import _LocalVascularTree
    from golem.kernel.anatomy.vocabulary import SealedVascularConfig


def _local_tree_segments(
    tree: _LocalVascularTree,
) -> tuple[
    tuple[tuple[float, float, float], tuple[float, float, float]], ...
]:
    return tuple(
        (tree.positions[parent], tree.positions[child])
        for child, parent in enumerate(tree.parents)
        if parent >= 0
    )


def _local_tree_capsules(
    tree: _LocalVascularTree,
    config: SealedVascularConfig,
) -> tuple[_SegmentCapsule, ...]:
    return tuple(
        capsule
        for _child, capsule in _local_tree_indexed_capsules(tree, config)
    )


def _local_tree_indexed_capsules(
    tree: _LocalVascularTree,
    config: SealedVascularConfig,
) -> tuple[tuple[int, _SegmentCapsule], ...]:
    minimum_terminal_flow = min(
        filter(lambda value: value > 0.0, tree.terminal_flows)
    )
    subtree_flows = _local_tree_subtree_flows(tree)
    return tuple(
        (
            child,
            _SegmentCapsule(
                tree.positions[parent],
                tree.positions[child],
                config.terminal_radius
                * (subtree_flows[child] / minimum_terminal_flow)
                ** (1.0 / config.murray_exponent),
            ),
        )
        for child, parent in enumerate(tree.parents)
        if parent >= 0
    )


def _ancestor_edge_child_indices(
    parents: tuple[int, ...], node_index: int
) -> tuple[int, ...]:
    parent_index = parents[node_index]
    return (
        ()
        if parent_index < 0
        else (
            node_index,
            *_ancestor_edge_child_indices(parents, parent_index),
        )
    )


def _weighted_geometric_median(anchors, weights, point, iterations: int):
    import numpy as np

    if iterations <= 0:
        return point
    distances = np.maximum(np.linalg.norm(anchors - point, axis=1), 1.0e-12)
    reweighted = weights / distances
    next_point = np.sum(reweighted[:, None] * anchors, axis=0) / np.sum(reweighted)
    return _weighted_geometric_median(
        anchors, weights, next_point, iterations - 1
    )


def _subtree_flow(tree: _LocalVascularTree, root_index: int) -> float:
    return _local_tree_subtree_flows(tree)[root_index]


@cache
def _local_tree_subtree_flows(
    tree: _LocalVascularTree,
) -> tuple[float, ...]:
    grouped = {
        parent: tuple(child for child, _parent in group)
        for parent, group in groupby(
            sorted(enumerate(tree.parents), key=lambda pair: pair[1]),
            key=lambda pair: pair[1],
        )
    }
    children = tuple(
        grouped.get(node_index, ())
        for node_index in range(len(tree.positions))
    )

    @cache
    def descend(node_index: int) -> float:
        return tree.terminal_flows[node_index] + math.fsum(
            map(descend, children[node_index])
        )

    return tuple(map(descend, range(len(tree.positions))))
