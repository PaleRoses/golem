"""Metric descriptor registry and pure interpretation over frozen senses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, assert_never

import numpy as np

from golem.addressing.scope import Chain, Landmark, Part
from golem.contracts.model import (
    Accepted,
    CheckedClause,
    InvalidMetricDescriptorObstruction,
    InvalidScopeShapeObstruction,
    Measurement,
    MetricDescriptor,
    MetricKind,
    MissingSenseObstruction,
    Rejected,
    Result,
)
from golem.senses.model import NoSupport, PartDatum, Senses, Supported
from golem.senses.proprio.format import f3, s3

_VIEW_AXES: Mapping[str, tuple[int, int]] = MappingProxyType(
    {"front": (0, 1), "side": (2, 1)}
)


@dataclass(frozen=True)
class PartSelection:
    ids: tuple[str, ...]
    data: tuple[PartDatum, ...]


def _measurement(
    measured: float | int,
    unit: str,
    offenders: tuple[str, ...],
    part_scopes: tuple[str, ...],
    detail: str,
) -> Result[Measurement]:
    return Accepted(Measurement(measured, unit, offenders, part_scopes, detail))


def _invalid_scope(clause: CheckedClause, expected: str):
    return Rejected(
        (
            InvalidScopeShapeObstruction(
                clause.identity.authored_metric,
                expected,
            ),
        )
    )


def _invalid_metric_descriptor(clause: CheckedClause, reason: str):
    return Rejected(
        (
            InvalidMetricDescriptorObstruction(
                clause.metric.kind.value,
                reason,
            ),
        )
    )


def _part_ids(clause: CheckedClause) -> tuple[str, ...]:
    return tuple(
        scope.part_id for scope in clause.scopes if isinstance(scope, Part)
    )


def _required_part_ids(
    clause: CheckedClause, count: int | None, expected: str
) -> Result[tuple[str, ...]]:
    part_ids = _part_ids(clause)
    valid = bool(part_ids) if count is None else len(part_ids) == count
    return Accepted(part_ids) if valid else _invalid_scope(clause, expected)


def _select_parts(senses: Senses, part_ids: tuple[str, ...]) -> Result[PartSelection]:
    values = tuple(senses.parts.get(part_id) for part_id in part_ids)
    obstructions = tuple(
        MissingSenseObstruction(f"part:{part_id}")
        for part_id, datum in zip(part_ids, values, strict=True)
        if datum is None
    )
    return (
        Rejected(obstructions)
        if obstructions
        else Accepted(
            PartSelection(
                part_ids,
                tuple(datum for datum in values if datum is not None),
            )
        )
    )


def _landmark_names(
    clause: CheckedClause, count: int, expected: str
) -> Result[tuple[str, ...]]:
    names = tuple(
        scope.name for scope in clause.scopes if isinstance(scope, Landmark)
    )
    return Accepted(names) if len(names) == count else _invalid_scope(clause, expected)


def _select_landmarks(
    senses: Senses, names: tuple[str, ...]
) -> Result[tuple[np.ndarray, ...]]:
    values = tuple(senses.landmarks.get(name) for name in names)
    obstructions = tuple(
        MissingSenseObstruction(f"landmark:{name}")
        for name, value in zip(names, values, strict=True)
        if value is None
    )
    return (
        Rejected(obstructions)
        if obstructions
        else Accepted(
            tuple(
                np.asarray(value, dtype=np.float64)
                for value in values
                if value is not None
            )
        )
    )


def _m_bbox_extent(senses: Senses, clause: CheckedClause):
    part_ids_result = _required_part_ids(clause, None, "part:?")
    match part_ids_result:
        case Rejected() as rejected:
            return rejected
        case Accepted(part_ids):
            parts_result = _select_parts(senses, part_ids)
    match parts_result:
        case Rejected() as rejected:
            return rejected
        case Accepted(PartSelection(ids, data)):
            if clause.metric.axis is None:
                return _invalid_metric_descriptor(
                    clause, "bbox extent requires an axis"
                )
            lo = np.min(np.stack(tuple(part.bbox_lo for part in data)), axis=0)
            hi = np.max(np.stack(tuple(part.bbox_hi for part in data)), axis=0)
            axis = clause.metric.axis
            measured = float(hi[axis] - lo[axis])
            metric = clause.identity.authored_metric
            return _measurement(
                measured,
                clause.unit or "world_unit",
                tuple(f"part:{part_id}" for part_id in ids),
                ids,
                f"{metric} of {'+'.join(ids)} = {f3(measured)}",
            )


def _m_contact_error(senses: Senses, clause: CheckedClause):
    part_ids_result = _required_part_ids(clause, None, "part:?")
    match part_ids_result:
        case Rejected() as rejected:
            return rejected
        case Accepted((part_id, *_)):
            pass
        case _ as unreachable:
            assert_never(unreachable)
    if part_id in senses.ground:
        clearance = senses.ground[part_id]
    elif part_id in senses.near_ground:
        clearance = senses.near_ground[part_id]
    else:
        match _select_parts(senses, (part_id,)):
            case Rejected() as rejected:
                return rejected
            case Accepted(PartSelection(_, (part,))):
                clearance = round(float(part.bbox_lo[1]) - 0.02, 3)
    return _measurement(
        abs(clearance),
        "world_unit",
        (f"part:{part_id}",),
        (part_id,),
        f"{part_id} clearance {s3(clearance)}",
    )


def _m_clearance(senses: Senses, clause: CheckedClause):
    part_ids_result = _required_part_ids(clause, 2, "part:pair")
    match part_ids_result:
        case Rejected() as rejected:
            return rejected
        case Accepted((left, right)):
            key = tuple(sorted((left, right)))
            gap = senses.declared_pair_gap.get(key)
            if gap is None:
                return Rejected(
                    (MissingSenseObstruction(f"pair:{key[0]}~{key[1]}"),)
                )
            return _measurement(
                float(gap),
                "world_unit",
                (f"part:{left}", f"part:{right}"),
                (left, right),
                f"gap {left}~{right} = {s3(gap)}",
            )
        case _ as unreachable:
            assert_never(unreachable)


def _m_landmark_distance(senses: Senses, clause: CheckedClause):
    names_result = _landmark_names(clause, 2, "landmark:pair")
    match names_result:
        case Rejected() as rejected:
            return rejected
        case Accepted((left_name, right_name) as names):
            landmarks_result = _select_landmarks(senses, names)
        case _ as unreachable:
            assert_never(unreachable)
    match landmarks_result:
        case Rejected() as rejected:
            return rejected
        case Accepted((left, right)):
            measured = float(np.linalg.norm(left - right))
            return _measurement(
                measured,
                "world_unit",
                (f"landmark:{left_name}", f"landmark:{right_name}"),
                (),
                f"|{left_name}-{right_name}| = {f3(measured)}",
            )
        case _ as unreachable:
            assert_never(unreachable)


def _m_axis_offset(senses: Senses, clause: CheckedClause):
    names_result = _landmark_names(clause, 2, "landmark:pair")
    match names_result:
        case Rejected() as rejected:
            return rejected
        case Accepted((left_name, right_name) as names):
            landmarks_result = _select_landmarks(senses, names)
        case _ as unreachable:
            assert_never(unreachable)
    match landmarks_result:
        case Rejected() as rejected:
            return rejected
        case Accepted((left, right)):
            if clause.metric.axis is None:
                return _invalid_metric_descriptor(
                    clause, "axis offset requires an axis"
                )
            axis = clause.metric.axis
            measured = float(left[axis] - right[axis])
            metric = clause.identity.authored_metric
            return _measurement(
                measured,
                "world_unit",
                (f"landmark:{left_name}", f"landmark:{right_name}"),
                (),
                f"{metric} {left_name}-{right_name} = {s3(measured)}",
            )
        case _ as unreachable:
            assert_never(unreachable)


def _m_schematic_offset(senses: Senses, clause: CheckedClause):
    landmarks = tuple(
        scope for scope in clause.scopes if isinstance(scope, Landmark)
    )
    match landmarks:
        case (Landmark(name, scope_view),):
            pass
        case _:
            return _invalid_scope(clause, "landmark:<name>@<view>")
    view = scope_view or clause.view
    axes = _VIEW_AXES.get(view) if view is not None else None
    if axes is None:
        return Rejected((MissingSenseObstruction(f"view:{view}"),))
    targets = senses.schematic.get(view) or {}
    target = targets.get(name)
    if target is None:
        return Rejected((MissingSenseObstruction(f"schematic:{view}/{name}"),))
    landmark_result = _select_landmarks(senses, (name,))
    match landmark_result:
        case Rejected() as rejected:
            return rejected
        case Accepted((point,)):
            u_axis, v_axis = axes
            projected = np.array([point[u_axis], point[v_axis]], dtype=np.float64)
            target_point = np.asarray(target, dtype=np.float64)
            measured = float(np.linalg.norm(projected - target_point))
            return _measurement(
                measured,
                clause.unit or "world_unit",
                (f"landmark:{name}",),
                (),
                f"{name}@{view} proj ({f3(projected[0])},{f3(projected[1])}) vs "
                f"target ({f3(target_point[0])},{f3(target_point[1])}) off {f3(measured)}",
            )
        case _ as unreachable:
            assert_never(unreachable)


def _m_axis_from_vertical(senses: Senses, clause: CheckedClause):
    chains = tuple(
        scope for scope in clause.scopes if isinstance(scope, Chain)
    )
    match chains:
        case (chain,):
            pass
        case _:
            return _invalid_scope(clause, "chain:a..b")
    parts_result = _select_parts(senses, (chain.start, chain.end))
    match parts_result:
        case Rejected() as rejected:
            return rejected
        case Accepted(PartSelection(_, (left, right))):
            left_center = (left.bbox_lo + left.bbox_hi) / 2.0
            right_center = (right.bbox_lo + right.bbox_hi) / 2.0
            vector = right_center - left_center
            norm = np.linalg.norm(vector)
            if norm < 1e-9:
                return Rejected(
                    (MissingSenseObstruction(f"chain:{chain.start}..{chain.end}"),)
                )
            cosine = float(np.clip(vector[1] / norm, -1.0, 1.0))
            measured = math.degrees(math.acos(cosine))
            return _measurement(
                measured,
                "degree",
                (f"chain:{chain.start}..{chain.end}",),
                (chain.start, chain.end),
                f"axis {chain.start}->{chain.end} tilts {f3(measured)} deg from +y",
            )
        case _ as unreachable:
            assert_never(unreachable)


def _m_head_exposure(senses: Senses, clause: CheckedClause):
    part_ids_result = _required_part_ids(clause, 2, "part:pair")
    match part_ids_result:
        case Rejected() as rejected:
            return rejected
        case Accepted(part_ids):
            parts_result = _select_parts(senses, part_ids)
    match parts_result:
        case Rejected() as rejected:
            return rejected
        case Accepted(PartSelection((head_id, occluder_id), (head, occluder))):
            head_height = float(head.bbox_hi[1] - head.bbox_lo[1])
            if head_height < 1e-9:
                return Rejected((MissingSenseObstruction(f"part:{head_id}"),))
            exposure = float(head.bbox_hi[1] - occluder.bbox_hi[1])
            measured = exposure / head_height
            return _measurement(
                measured,
                f"{head_id}-heights",
                (f"part:{occluder_id}.top", f"part:{head_id}.top"),
                (head_id, occluder_id),
                f"{head_id} top exposed {f3(exposure)} above {occluder_id} "
                f"= {f3(measured)} {head_id}-heights",
            )
        case _ as unreachable:
            assert_never(unreachable)


def _m_heads_tall(senses: Senses, clause: CheckedClause):
    part_ids_result = _required_part_ids(clause, None, "part:head")
    match part_ids_result:
        case Rejected() as rejected:
            return rejected
        case Accepted((head_id, *_) as part_ids):
            parts_result = _select_parts(senses, (head_id,))
        case _ as unreachable:
            assert_never(unreachable)
    match parts_result:
        case Rejected() as rejected:
            return rejected
        case Accepted(PartSelection((head_id,), (head,))):
            head_height = float(head.bbox_hi[1] - head.bbox_lo[1])
            if head_height < 1e-9:
                return Rejected((MissingSenseObstruction(f"part:{head_id}"),))
            measured = senses.global_dims.H / head_height
            return _measurement(
                measured,
                "heads",
                (f"part:{head_id}",),
                part_ids,
                f"H {f3(senses.global_dims.H)} / {head_id}_h {f3(head_height)} "
                f"= {f3(measured)} heads",
            )
        case _ as unreachable:
            assert_never(unreachable)


def _m_centroid_support_margin(senses: Senses, clause: CheckedClause):
    match senses.balance:
        case NoSupport():
            return Rejected((MissingSenseObstruction("contact:support"),))
        case Supported() as balance:
            measured = float(min(balance.margin_x, balance.margin_z))
            return _measurement(
                measured,
                "world_unit",
                ("whole.centroid",),
                (),
                f"centroid margin {s3(measured)} "
                f"({'inside' if balance.inside else 'outside'})",
            )
        case _ as unreachable:
            assert_never(unreachable)


def _m_post_dust_components(senses: Senses, clause: CheckedClause):
    measured = int(senses.n_components)
    return _measurement(
        measured,
        "count",
        ("whole",),
        (),
        f"predicted {measured} component(s)",
    )


_METRICS: Mapping[MetricKind, MetricDescriptor] = MappingProxyType(
    {
        MetricKind.WIDTH: MetricDescriptor(
            MetricKind.WIDTH, _m_bbox_extent, axis=0
        ),
        MetricKind.HEIGHT: MetricDescriptor(
            MetricKind.HEIGHT, _m_bbox_extent, axis=1
        ),
        MetricKind.DEPTH: MetricDescriptor(
            MetricKind.DEPTH, _m_bbox_extent, axis=2
        ),
        MetricKind.SPAN: MetricDescriptor(
            MetricKind.SPAN,
            _m_bbox_extent,
            axis=0,
            aliases=("shoulder_span",),
        ),
        MetricKind.CONTACT_ERROR: MetricDescriptor(
            MetricKind.CONTACT_ERROR, _m_contact_error
        ),
        MetricKind.CLEARANCE: MetricDescriptor(
            MetricKind.CLEARANCE,
            _m_clearance,
            aliases=("gap", "neck_gap"),
        ),
        MetricKind.LANDMARK_DISTANCE: MetricDescriptor(
            MetricKind.LANDMARK_DISTANCE,
            _m_landmark_distance,
            aliases=("sword_length",),
        ),
        MetricKind.FORWARD_OFFSET: MetricDescriptor(
            MetricKind.FORWARD_OFFSET,
            _m_axis_offset,
            axis=2,
            aliases=("plant_offset",),
        ),
        MetricKind.VERTICAL_OFFSET: MetricDescriptor(
            MetricKind.VERTICAL_OFFSET, _m_axis_offset, axis=1
        ),
        MetricKind.SCHEMATIC_OFFSET: MetricDescriptor(
            MetricKind.SCHEMATIC_OFFSET, _m_schematic_offset
        ),
        MetricKind.AXIS_FROM_VERTICAL: MetricDescriptor(
            MetricKind.AXIS_FROM_VERTICAL, _m_axis_from_vertical
        ),
        MetricKind.HEAD_EXPOSURE: MetricDescriptor(
            MetricKind.HEAD_EXPOSURE, _m_head_exposure
        ),
        MetricKind.HEADS_TALL: MetricDescriptor(
            MetricKind.HEADS_TALL, _m_heads_tall
        ),
        MetricKind.CENTROID_SUPPORT_MARGIN: MetricDescriptor(
            MetricKind.CENTROID_SUPPORT_MARGIN,
            _m_centroid_support_margin,
        ),
        MetricKind.POST_DUST_COMPONENTS: MetricDescriptor(
            MetricKind.POST_DUST_COMPONENTS,
            _m_post_dust_components,
        ),
    }
)

_METRIC_BY_TOKEN: Mapping[str, MetricDescriptor] = MappingProxyType(
    {
        token: descriptor
        for descriptor in _METRICS.values()
        for token in (descriptor.kind.value, *descriptor.aliases)
    }
)


def resolve_metric(name: str) -> MetricDescriptor | None:
    return _METRIC_BY_TOKEN.get(name)
