"""Closed anomaly observations and pure suppression folds."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import reduce

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from golem.senses.model import Senses, immutable_mapping
from golem.senses.proprio.format import f3, s3
from golem.senses.proprio.geometry import field_at, symmetric_gap
from golem.senses.proprio.fusion import (
    _incoming_blend_radius,
    _instance_order,
)


class AnomalyKind(StrEnum):
    cross_plane_fusion = "cross_plane_fusion"
    deep_burial = "deep_burial"
    articulated_fusion = "articulated_fusion"
    unintended_fusion = "unintended_fusion"
    missing_fusion = "missing_fusion"
    blend_ambiguity = "blend_ambiguity"
    floating_contact = "floating_contact"


@dataclass(frozen=True)
class Anomaly(Mapping[str, object]):
    kind: AnomalyKind
    pair: tuple[str, ...]
    gap: float
    magnitude: float
    detail: str
    address: str
    expected: bool = False

    def __getitem__(self, key: str) -> object:
        match key:
            case "det":
                return self.kind.value
            case "pair":
                return self.pair
            case "gap":
                return self.gap
            case "mag":
                return self.magnitude
            case "detail":
                return self.detail
            case "addr":
                return self.address
            case "expected":
                return self.expected
            case _:
                raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("det", "pair", "gap", "mag", "detail", "addr", "expected"))

    def __len__(self) -> int:
        return 7


@dataclass(frozen=True)
class SuppressionResult:
    suffix_by_rule_id: Mapping[str, tuple[str, ...]]
    visible_anomalies: tuple[Anomaly, ...]
    suppressed_count: int

    def __iter__(self) -> Iterator[object]:
        return iter(
            (
                self.suffix_by_rule_id,
                self.visible_anomalies,
                self.suppressed_count,
            )
        )


def detect_anomalies(senses: Senses) -> tuple[Anomaly, ...]:
    directed_attachments = frozenset(
        (*senses.attach, *((right, left) for left, right in senses.attach))
    )
    declared_midline = frozenset(senses.midline)
    attachment_distances = _attach_distances(senses)
    expected_articulations = frozenset(
        tuple(sorted(pair)) for pair in senses.expected_articulations
    )
    instance_order = _instance_order(senses.inst_data)
    pair_candidates = tuple(
        anomaly
        for pair_gap in senses.inst_gaps.items()
        for anomaly in _gap_anomalies(
            pair_gap,
            senses,
            directed_attachments,
            attachment_distances,
            expected_articulations,
            declared_midline,
            instance_order,
        )
    )
    fused_pairs = frozenset(
        tuple(sorted((senses.inst_data[left].pid, senses.inst_data[right].pid)))
        for (left, right), gap in senses.inst_gaps.items()
        if gap < 0.0
        and senses.inst_data[left].pid != senses.inst_data[right].pid
    )
    missing_candidates = tuple(
        anomaly
        for attachment in senses.attach
        for anomaly in _missing_fusion_anomaly(
            attachment, fused_pairs, senses, instance_order
        )
    )
    floating_candidates = tuple(
        Anomaly(
            kind=AnomalyKind.floating_contact,
            pair=(part_id,),
            gap=clearance,
            magnitude=round(
                abs(clearance) / max(1.0e-6, senses.ground_tol), 3
            ),
            detail=(
                f"clearance {s3(clearance)} > tol {f3(senses.ground_tol)}"
            ),
            address=part_id,
        )
        for part_id, clearance in senses.ground.items()
        if abs(clearance) > senses.ground_tol + 1.0e-9
    )
    deduplicated = reduce(
        _prefer_more_severe,
        (*pair_candidates, *missing_candidates, *floating_candidates),
        {},
    )
    return tuple(
        sorted(
            deduplicated.values(),
            key=lambda anomaly: anomaly.magnitude,
            reverse=True,
        )
    )


def _gap_anomalies(
    pair_gap: tuple[tuple[str, str], float],
    senses: Senses,
    directed_attachments: frozenset[tuple[str, str]],
    attachment_distances: Mapping[tuple[str, str], int],
    expected_articulations: frozenset[tuple[str, str]],
    declared_midline: frozenset[str],
    instance_order: Mapping[str, int],
) -> tuple[Anomaly, ...]:
    (left, right), gap = pair_gap
    left_data, right_data = senses.inst_data[left], senses.inst_data[right]
    left_part, right_part = left_data.pid, right_data.pid
    pair_radius = _incoming_blend_radius(
        left, right, senses.inst_data, instance_order
    )
    minimum_radius = min(left_data.min_r, right_data.min_r)
    if left_part == right_part:
        return (
            (
                Anomaly(
                    kind=AnomalyKind.cross_plane_fusion,
                    pair=(left_part, left_part),
                    gap=gap,
                    magnitude=round(-gap / minimum_radius, 3),
                    detail=f"overlap {f3(-gap)} across x=0",
                    address=f"{left_part}.L~{left_part}.R",
                    # Deliberately-symmetric anatomy declared on the midline
                    # channel: the crossing is observed but expected, so it
                    # reports as intent, not accident.
                    expected=left_part in declared_midline,
                ),
            )
            if gap < 0.0
            else ()
        )
    pair = tuple(sorted((left_part, right_part)))
    declared = (left_part, right_part) in directed_attachments
    if gap < 0.0 and declared:
        burial = -gap / minimum_radius
        return (
            (
                Anomaly(
                    kind=AnomalyKind.deep_burial,
                    pair=pair,
                    gap=gap,
                    magnitude=round(burial, 3),
                    detail=(
                        f"{f3(-gap)} deep, {int(round(burial * 100))}% of "
                        f"{min_r_owner(senses.inst_data, left, right)}"
                    ),
                    address=f"{left_part}~{right_part}",
                    expected=True,
                ),
            )
            if burial > 0.5
            else ()
        )
    if gap < 0.0:
        articulated = _articulated_girdle(
            left_part,
            right_part,
            pair,
            directed_attachments,
            attachment_distances,
            expected_articulations,
            senses,
        )
        links = attachment_distances.get(pair)
        link_detail = f", {links} links apart" if links and links > 1 else ""
        return (
            Anomaly(
                kind=(
                    AnomalyKind.articulated_fusion
                    if articulated
                    else AnomalyKind.unintended_fusion
                ),
                pair=pair,
                gap=gap,
                magnitude=round(-gap / minimum_radius, 3),
                detail=(
                    f"gap {f3(gap)}, joined by fused attachment chain"
                    if articulated
                    else f"gap {f3(gap)}{link_detail}"
                ),
                address=f"{left_part}~{right_part}",
            ),
        )
    return (
        (
            Anomaly(
                kind=AnomalyKind.blend_ambiguity,
                pair=pair,
                gap=gap,
                magnitude=round((pair_radius - gap) / pair_radius, 3),
                detail=(
                    f"gap {f3(gap)} < k {f3(pair_radius)} "
                    "(no void; may bridge at mesh)"
                ),
                address=f"{left_part}~{right_part}",
            ),
        )
        if gap < pair_radius
        and not declared
        and not _carve_severs_bridge(senses, left, right)
        else ()
    )

_CARVE_NECK_SAMPLES = 5


def _carve_severs_bridge(senses: Senses, left: str, right: str) -> bool:
    """A ``blend_ambiguity`` advisory claims two close parts may bridge at mesh
    because no void separates them. An authored carve that fully occupies the
    closest-approach corridor is exactly that void: the engine's hard
    ``max(host, -carve)`` difference severs the neck, so the advisory is false.
    Consult the authoritative carve field the compiler already emitted rather
    than reconstructing cavities from part geometry."""
    carves = senses.graph.get("carves")
    if not carves:
        return False
    left_samples = senses.inst_data[left].samples
    right_samples = senses.inst_data[right].samples
    if left_samples.size == 0 or right_samples.size == 0:
        return False
    deltas = left_samples[:, None, :] - right_samples[None, :, :]
    distances = np.einsum("ijk,ijk->ij", deltas, deltas)
    nearest = int(np.argmin(distances))
    near_left = left_samples[nearest // right_samples.shape[0]]
    near_right = right_samples[nearest % right_samples.shape[0]]
    fractions = (
        np.arange(1, _CARVE_NECK_SAMPLES + 1, dtype=np.float64)
        / (_CARVE_NECK_SAMPLES + 1)
    )[:, None]
    corridor = (1.0 - fractions) * near_left + fractions * near_right
    carve_field = field_at({"parts": list(carves), "blend": 0.0}, corridor)
    return bool(np.all(carve_field <= 0.0))



def _missing_fusion_anomaly(
    attachment: tuple[str, str],
    fused_pairs: frozenset[tuple[str, str]],
    senses: Senses,
    instance_order: Mapping[str, int],
) -> tuple[Anomaly, ...]:
    left_part, right_part = attachment
    pair = tuple(sorted(attachment))
    if (
        left_part not in senses.parts
        or right_part not in senses.parts
        or pair in fused_pairs
    ):
        return ()
    gap = senses.declared_pair_gap.get(pair)
    measured_gap = (
        gap
        if gap is not None
        else _quick_declared_gap(senses, left_part, right_part)
    )
    reach = _declared_blend_reach(
        senses, left_part, right_part, instance_order
    )
    return (
        (
            Anomaly(
                kind=AnomalyKind.missing_fusion,
                pair=pair,
                gap=measured_gap,
                magnitude=round(
                    measured_gap / max(1.0e-6, senses.global_dims.k), 3
                ),
                detail=f"declared attach not fused (gap {f3(measured_gap)})",
                address=f"{left_part}~{right_part}",
            ),
        )
        if measured_gap is not None
        and measured_gap >= 0.0
        and not (
            reach is not None and measured_gap < reach
        )
        else ()
    )


def _declared_blend_reach(
    senses: Senses,
    left_part: str,
    right_part: str,
    instance_order: Mapping[str, int],
) -> float | None:
    """Blend reach at the closest approach between two parts: the blend radius
    of the instance pair attaining the minimum symmetric gap. A declared pair
    inside that reach bridges at mesh -- which is exactly what the declaration
    meant -- so it reads as declared fusion, never as missing_fusion. A pair
    declared but truly out of reach (a void the blend kernel cannot close)
    still reports."""
    approaches = (
        (
            symmetric_gap(
                senses.inst_data[left_instance],
                senses.inst_data[right_instance],
            ),
            _incoming_blend_radius(
                left_instance,
                right_instance,
                senses.inst_data,
                instance_order,
            ),
        )
        for left_instance in senses.parts[left_part].instances
        for right_instance in senses.parts[right_part].instances
        if left_instance != right_instance
    )
    closest = min(approaches, key=lambda approach: approach[0], default=None)
    return closest[1] if closest is not None else None


def _prefer_more_severe(
    candidates: dict[tuple[AnomalyKind, tuple[str, ...]], Anomaly],
    anomaly: Anomaly,
) -> dict[tuple[AnomalyKind, tuple[str, ...]], Anomaly]:
    key = (anomaly.kind, anomaly.pair)
    existing = candidates.get(key)
    return (
        candidates
        if existing is not None and existing.magnitude >= anomaly.magnitude
        else {**candidates, key: anomaly}
    )


def min_r_owner(inst_data, left: str, right: str) -> str:
    return (
        inst_data[left].pid
        if inst_data[left].min_r <= inst_data[right].min_r
        else inst_data[right].pid
    )


def _attach_distances(senses: Senses) -> Mapping[tuple[str, str], int]:
    part_ids = tuple(senses.parts)
    if not part_ids:
        return immutable_mapping({})
    attachments = frozenset(
        (*senses.attach, *((right, left) for left, right in senses.attach))
    )
    matrix = np.asarray(
        tuple(
            tuple(float((left, right) in attachments) for right in part_ids)
            for left in part_ids
        ),
        dtype=np.float64,
    )
    distances = shortest_path(csr_matrix(matrix), directed=False, unweighted=True)
    return immutable_mapping(
        {
            tuple(sorted((left, right))): int(distances[left_index, right_index])
            for left_index, left in enumerate(part_ids)
            for right_index, right in enumerate(part_ids)
            if np.isfinite(distances[left_index, right_index])
        }
    )


def _articulated_girdle(
    left_part: str,
    right_part: str,
    pair: tuple[str, ...],
    directed_attachments: frozenset[tuple[str, str]],
    attachment_distances: Mapping[tuple[str, str], int],
    expected_articulations: frozenset[tuple[str, str]],
    senses: Senses,
) -> bool:
    if pair not in expected_articulations:
        return False
    if attachment_distances.get(pair) != 2:
        return False
    intermediaries = _declared_neighbors(
        left_part, directed_attachments
    ) & _declared_neighbors(right_part, directed_attachments)
    return any(
        _fused_pair(senses, part_c, left_part)
        and _fused_pair(senses, part_c, right_part)
        for part_c in intermediaries
    )


def _declared_neighbors(
    part: str, directed_attachments: frozenset[tuple[str, str]]
) -> frozenset[str]:
    return frozenset(
        right for left, right in directed_attachments if left == part
    )


def _fused_pair(senses: Senses, first: str, second: str) -> bool:
    return second in senses.fused_adj.get(
        first, frozenset()
    ) or first in senses.fused_adj.get(second, frozenset())


def _quick_declared_gap(
    senses: Senses, left_part: str, right_part: str
) -> float | None:
    return min(
        (
            symmetric_gap(
                senses.inst_data[left_instance],
                senses.inst_data[right_instance],
            )
            for left_instance in senses.parts[left_part].instances
            for right_instance in senses.parts[right_part].instances
            if left_instance != right_instance
        ),
        default=None,
    )


_ANOMALY_CAP = 12
_ASSERT_CAP = 24
_ANOM_TAG = AnomalyKind


def _apply_suppression(
    assert_results: Sequence[Mapping[str, object]],
    anomalies: Sequence[Mapping[str, object]],
) -> SuppressionResult:
    normalized = tuple(map(_coerce_anomaly, anomalies))
    fail_pairs = tuple(
        (
            frozenset(
                part_id
                for part_id in result.get("part_scopes", ())
                if isinstance(part_id, str)
            ),
            str(result["id"]),
        )
        for result in assert_results
        if result.get("status") == "fail"
    )
    covered = tuple(
        (anomaly, _covering_assert_id(anomaly, fail_pairs))
        for anomaly in normalized
    )
    suppressed = tuple(
        (rule_id, _anomaly_suffix(anomaly))
        for anomaly, rule_id in covered
        if rule_id is not None
    )
    suffix = immutable_mapping(
        {
            rule_id: tuple(
                label
                for candidate_rule_id, label in suppressed
                if candidate_rule_id == rule_id
            )
            for rule_id in frozenset(map(lambda item: item[0], suppressed))
        }
    )
    visible = tuple(
        anomaly
        for anomaly, rule_id in covered
        if rule_id is None and not anomaly.expected
    )
    return SuppressionResult(suffix, visible, len(suppressed))


def _covering_assert_id(
    anomaly: Anomaly,
    fail_pairs: tuple[tuple[frozenset[str], str], ...],
) -> str | None:
    pair = frozenset(anomaly.pair)
    return next(
        (
            rule_id
            for part_ids, rule_id in fail_pairs
            if pair and pair <= part_ids
        ),
        None,
    )


def _coerce_anomaly(value: Mapping[str, object]) -> Anomaly:
    return (
        value
        if isinstance(value, Anomaly)
        else Anomaly(
            kind=AnomalyKind(str(value["det"])),
            pair=tuple(map(str, value["pair"])),
            gap=float(value["gap"]),
            magnitude=float(value["mag"]),
            detail=str(value["detail"]),
            address=str(value["addr"]),
            expected=bool(value.get("expected", False)),
        )
    )


def _anomaly_suffix(anomaly: Mapping[str, object]) -> str:
    kind = AnomalyKind(str(anomaly["det"]))
    magnitude = float(anomaly["mag"])
    gap = float(anomaly["gap"])
    match kind:
        case AnomalyKind.deep_burial:
            return f"burial {int(round(magnitude * 100))}%"
        case AnomalyKind.articulated_fusion:
            return f"articulated {s3(gap)}"
        case AnomalyKind.unintended_fusion | AnomalyKind.cross_plane_fusion:
            return f"{kind.value.split('_')[0]} {s3(gap)}"
        case AnomalyKind.blend_ambiguity:
            return f"blend {s3(gap)}"
        case AnomalyKind.floating_contact:
            return f"float {s3(gap)}"
        case AnomalyKind.missing_fusion:
            return kind.value
