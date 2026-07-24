"""Pure fusion observations and folds over instance-pair contexts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from golem.kernel.engine import CompositionOperator
from golem.addressing.core import Part, parse_scope
from golem.senses.model import (
    FusionBand,
    FusionRow,
    InstanceDatum,
    PartDatum,
    immutable_mapping,
)
from golem.senses.proprio.geometry import _aabb_gap, symmetric_gap


@dataclass(frozen=True)
class FusionObservation:
    left_instance: str
    right_instance: str
    left_part: str
    right_part: str
    gap: float
    band: FusionBand | None


@dataclass(frozen=True)
class FusionResult:
    instance_gaps: Mapping[tuple[str, str], float]
    rows: Mapping[str, tuple[FusionRow, ...]]
    cross_plane: Mapping[str, float]
    adjacency: Mapping[str, frozenset[str]]
    component_count: int
    declared_pair_gaps: Mapping[tuple[str, str], float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "instance_gaps", immutable_mapping(self.instance_gaps))
        object.__setattr__(self, "rows", immutable_mapping(self.rows))
        object.__setattr__(self, "cross_plane", immutable_mapping(self.cross_plane))
        object.__setattr__(self, "adjacency", immutable_mapping(self.adjacency))
        object.__setattr__(
            self, "declared_pair_gaps", immutable_mapping(self.declared_pair_gaps)
        )


def _instance_order(
    inst_data: Mapping[str, InstanceDatum],
) -> dict[str, int]:
    return {
        instance: index
        for index, instance in enumerate(inst_data)
    }


def _incoming_blend_radius(
    left: str,
    right: str,
    inst_data: Mapping[str, InstanceDatum],
    instance_order: Mapping[str, int],
) -> float:
    """Radius of the later/incoming part's union edge.

    Engine composition folds instances in declaration order. The incoming
    instance owns both the operator and selected radius at its junction;
    crease alone suppresses smoothing reach.
    """
    incoming = inst_data[
        right
        if instance_order[right] > instance_order[left]
        else left
    ]
    return (
        0.0
        if incoming.part.get("operator")
        in (
            CompositionOperator.CREASE.value,
            CompositionOperator.LOCAL_BLEND.value,
        )
        else incoming.blend_r
    )


def compute_fusion(
    inst_data: Mapping[str, InstanceDatum],
    parts: Mapping[str, PartDatum],
    contract: Sequence[Mapping[str, object]],
) -> FusionResult:
    instance_order = _instance_order(inst_data)
    instance_pairs = tuple(combinations(inst_data, 2))
    relevant_gaps = tuple(
        result
        for result in map(
            lambda pair: _prefiltered_gap(
                pair, inst_data, instance_order
            ),
            instance_pairs,
        )
        if result is not None
    )
    instance_gaps = dict(relevant_gaps)
    referenced_pairs = frozenset(
        pair
        for clause in contract
        if clause.get("metric") in ("clearance", "gap", "neck_gap")
        for pair in _referenced_pair(clause, parts)
    )
    declared_pair_gaps = {
        pair: gap
        for pair in referenced_pairs
        if (
            gap := _declared_gap(
                pair[0], pair[1], parts, inst_data, instance_gaps
            )
        )
        is not None
    }
    observations = tuple(
        _classify_gap(
            left, right, gap, inst_data, instance_order
        )
        for (left, right), gap in relevant_gaps
    )
    significant = tuple(
        observation
        for observation in observations
        if observation.band is not None
    )
    part_ids = tuple(parts)
    adjacency = {
        part_id: frozenset(
            observation.right_part
            if observation.left_part == part_id
            else observation.left_part
            for observation in significant
            if observation.band is FusionBand.FUSED
            and part_id in (observation.left_part, observation.right_part)
        )
        for part_id in part_ids
    }
    rows = {
        part_id: tuple(
            FusionRow(
                pb=(
                    observation.right_part
                    if observation.left_part == part_id
                    else observation.left_part
                ),
                g=observation.gap,
                band=observation.band,
            )
            for observation in significant
            if observation.left_part != observation.right_part
            and part_id in (observation.left_part, observation.right_part)
            and observation.band is not None
        )
        for part_id in part_ids
    }
    cross_plane = {
        observation.left_part: observation.gap
        for observation in significant
        if observation.left_part == observation.right_part
    }
    return FusionResult(
        instance_gaps=instance_gaps,
        rows=rows,
        cross_plane=cross_plane,
        adjacency=adjacency,
        component_count=_component_count(part_ids, adjacency),
        declared_pair_gaps=declared_pair_gaps,
    )


def _prefiltered_gap(
    pair: tuple[str, str],
    inst_data: Mapping[str, InstanceDatum],
    instance_order: Mapping[str, int],
) -> tuple[tuple[str, str], float] | None:
    left, right = pair
    left_data, right_data = inst_data[left], inst_data[right]
    pair_radius = _incoming_blend_radius(
        left, right, inst_data, instance_order
    )
    return (
        None
        if _aabb_gap(
            left_data.lo,
            left_data.hi,
            right_data.lo,
            right_data.hi,
        )
        > 3.0 * pair_radius
        else (pair, symmetric_gap(left_data, right_data))
    )


def _classify_gap(
    left: str,
    right: str,
    gap: float,
    inst_data: Mapping[str, InstanceDatum],
    instance_order: Mapping[str, int],
) -> FusionObservation:
    pair_radius = _incoming_blend_radius(
        left, right, inst_data, instance_order
    )
    band = (
        FusionBand.FUSED
        if gap < 0.0
        else FusionBand.BLEND
        if gap < pair_radius
        else None
    )
    return FusionObservation(
        left_instance=left,
        right_instance=right,
        left_part=inst_data[left].pid,
        right_part=inst_data[right].pid,
        gap=gap,
        band=band,
    )


def _referenced_pair(
    clause: Mapping[str, object], parts: Mapping[str, PartDatum]
) -> tuple[tuple[str, str], ...]:
    scope = clause.get("scope", ())
    scoped = tuple(
        part_id
        for part_id in map(
            _scope_part,
            scope if isinstance(scope, (list, tuple)) else (),
        )
        if part_id in parts
    )
    return (tuple(sorted(scoped)),) if len(scoped) == 2 else ()


def _scope_part(scope: object) -> str | None:
    if not isinstance(scope, str):
        return None
    match parse_scope(scope):
        case Part(part_id):
            return part_id
        case _:
            return None


def _declared_gap(
    left_part: str,
    right_part: str,
    parts: Mapping[str, PartDatum],
    inst_data: Mapping[str, InstanceDatum],
    instance_gaps: Mapping[tuple[str, str], float],
) -> float | None:
    gaps = tuple(
        gap
        for left_instance in parts[left_part].instances
        for right_instance in parts[right_part].instances
        if left_instance != right_instance
        for gap in (
            _known_or_forced_gap(
                left_instance,
                right_instance,
                left_part,
                right_part,
                inst_data,
                instance_gaps,
            ),
        )
        if gap is not None
    )
    return min(gaps, default=None)


def _known_or_forced_gap(
    left_instance: str,
    right_instance: str,
    left_part: str,
    right_part: str,
    inst_data: Mapping[str, InstanceDatum],
    instance_gaps: Mapping[tuple[str, str], float],
) -> float | None:
    direct = instance_gaps.get((left_instance, right_instance))
    reverse = instance_gaps.get((right_instance, left_instance))
    return (
        direct
        if direct is not None
        else reverse
        if reverse is not None
        else symmetric_gap(inst_data[left_instance], inst_data[right_instance])
        if left_part != right_part
        else None
    )


def _component_count(
    part_ids: tuple[str, ...],
    adjacency: Mapping[str, frozenset[str]],
) -> int:
    if not part_ids:
        return 0
    matrix = np.asarray(
        tuple(
            tuple(float(right in adjacency[left]) for right in part_ids)
            for left in part_ids
        ),
        dtype=np.float64,
    )
    count, _labels = connected_components(
        csr_matrix(matrix), directed=False, return_labels=True
    )
    return int(count)
