"""Vocabulary obstructions derived from one checked geometry descent."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from math import ceil, isfinite
from typing import assert_never

import numpy as np

from .compile import graph_bounds_checked
from .types import (
    AbsoluteProfileDepths,
    BlobPart,
    BoxPart,
    FEATURE_MIN_CELLS,
    GencylPart,
    GeometryObstruction,
    GeometryRule,
    Part,
    RelativeProfileDepths,
    Rejected,
    decode_graph,
    read_quat,
)


_MINIMUM_POSITIVE_FLOAT = float.fromhex("0x0.0000000000001p-1022")


def _record(
    obstruction: GeometryObstruction,
    repair: str | None = None,
) -> dict:
    return {
        "part": obstruction.part_id,
        "kind": obstruction.kind,
        "rule": obstruction.rule.value,
        "detail": obstruction.detail,
        **({"repair": repair} if repair is not None else {}),
    }


def _render_world_units(value: float) -> str:
    return f"{float(value):.17g} world units"


def _minimum_resolution(
    maximum_extent: float,
    feature_size: float,
    minimum_cells: float,
) -> int:
    candidate = int(ceil(minimum_cells * maximum_extent / feature_size)) + 1
    passing_candidate = candidate + int(
        feature_size
        < minimum_cells * (
            maximum_extent / float(candidate - 1)
        )
    )
    previous_candidate = passing_candidate - 1
    return passing_candidate - int(
        previous_candidate > 1
        and feature_size
        >= minimum_cells * (
            maximum_extent / float(previous_candidate - 1)
        )
    )


def _numeric_values(value: object) -> tuple[float, ...]:
    if isinstance(value, np.ndarray):
        return _numeric_values(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(
            coordinate
            for item in value
            for coordinate in _numeric_values(item)
        )
    return (float(value),)


def _corrected_numeric(
    value: object,
    correction: Callable[[float], float],
) -> object:
    if isinstance(value, np.ndarray):
        return _corrected_numeric(value.tolist(), correction)
    if isinstance(value, (list, tuple)):
        return list(
            map(lambda item: _corrected_numeric(item, correction), value)
        )
    return correction(float(value))


def _profile_value_repair(
    raw_part: Mapping[str, object],
    parameter: str,
    correction: Callable[[float], float],
    *,
    require_positive: bool,
) -> str | None:
    profile = raw_part.get("profile")
    if not isinstance(profile, Mapping) or parameter not in profile:
        return None
    values = _numeric_values(profile[parameter])
    if not values or not all(map(isfinite, values)):
        return None
    if require_positive and min(values) <= 0.0:
        return None
    corrected = _corrected_numeric(profile[parameter], correction)
    return (
        f"set profile.{parameter} to "
        f"{json.dumps(corrected, separators=(',', ':'))}"
    )


def _raw_primitives(graph: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(graph, Mapping):
        return ()
    sections = tuple(
        section
        for section in (graph.get("parts"), graph.get("carves"))
        if isinstance(section, (list, tuple))
    )
    return tuple(
        part
        for section in sections
        for part in section
        if isinstance(part, Mapping)
    )


def _raw_part_for(
    graph: object,
    obstruction: GeometryObstruction,
) -> Mapping[str, object] | None:
    return next(
        (
            part
            for part in _raw_primitives(graph)
            if part.get("id") == obstruction.part_id
            and part.get("type") == obstruction.kind
        ),
        None,
    )


def _raw_repair(
    obstruction: GeometryObstruction,
    raw_part: Mapping[str, object] | None,
) -> str | None:
    match obstruction.rule:
        case GeometryRule.FENCE_ROT_ON_GENCYL:
            return "remove rot"
        case GeometryRule.FENCE_PROFILE_ON_NONGENCYL:
            return "remove profile"
        case GeometryRule.QUAT_NONUNIT if raw_part is not None:
            return (
                "set rot to "
                + json.dumps(
                    list(map(float, read_quat(raw_part.get("rot")))),
                    separators=(",", ":"),
                )
            )
        case GeometryRule.PROFILE_ASPECT if raw_part is not None:
            return _profile_value_repair(
                raw_part,
                "aspect",
                lambda value: min(
                    max(value, _MINIMUM_POSITIVE_FLOAT),
                    1.0,
                ),
                require_positive=False,
            )
        case GeometryRule.PROFILE_EXPONENT if raw_part is not None:
            return _profile_value_repair(
                raw_part,
                "n",
                lambda value: min(max(value, 2.0), 12.0),
                require_positive=False,
            )
        case GeometryRule.PROFILE_DEPTH if raw_part is not None:
            return _profile_value_repair(
                raw_part,
                "depth",
                lambda value: max(value, _MINIMUM_POSITIVE_FLOAT),
                require_positive=False,
            )
        case GeometryRule.BOX_NEGATIVE_ROUND:
            return "set round to 0.0 world units"
        case (
            GeometryRule.PROFILE_ANCHOR_ARITY
            | GeometryRule.PROFILE_OFFSET
            | GeometryRule.PROFILE_DEPTH_CONFLICT
            | GeometryRule.PROFILE_ROLL
            | GeometryRule.COMPOSITION_OPERATOR
            | GeometryRule.WEB_ANCHOR_ARITY
            | GeometryRule.WEB_STATION_ARITY
            | GeometryRule.WEB_HALF_EXTENT
            | GeometryRule.FEATURE_SIZE
            | GeometryRule.SHARP_EDGE
            | GeometryRule.UNKNOWN_PART_KIND
            | GeometryRule.MALFORMED_PART
            | GeometryRule.MALFORMED_GRAPH
        ):
            return None
        case _ as unreachable:
            assert_never(unreachable)


def _accepted_repair(
    obstruction: GeometryObstruction,
    part: Part,
    raw_part: Mapping[str, object],
    *,
    pitch: float,
    maximum_extent: float,
) -> str | None:
    direct = _raw_repair(obstruction, raw_part)
    if direct is not None:
        return direct
    match obstruction.rule, part:
        case GeometryRule.FEATURE_SIZE, GencylPart(
            radii=radii,
            profile=profile,
        ) if profile is not None and isinstance(
            profile.depths, RelativeProfileDepths
        ):
            thin = 2.0 * float(
                np.min(
                    np.asarray(radii, dtype=np.float64)
                    * np.clip(
                        np.asarray(profile.depths.ratios, dtype=np.float64),
                        1e-9,
                        None,
                    )
                )
            )
            required_product = FEATURE_MIN_CELLS * pitch / 2.0
            minimum_resolution = _minimum_resolution(
                maximum_extent,
                thin,
                FEATURE_MIN_CELLS,
            )
            return (
                "raise min(radius*aspect) to >= "
                f"{_render_world_units(required_product)} or res to >= "
                f"{minimum_resolution}"
            )
        case GeometryRule.FEATURE_SIZE, GencylPart(
            radii=radii,
            profile=profile,
        ) if profile is not None and isinstance(
            profile.depths, AbsoluteProfileDepths
        ):
            minimum_half_extent = float(
                np.min(
                    np.minimum(
                        np.asarray(radii, dtype=np.float64),
                        np.asarray(profile.depths.values, dtype=np.float64),
                    )
                )
            )
            minimum_resolution = _minimum_resolution(
                maximum_extent,
                2.0 * minimum_half_extent,
                FEATURE_MIN_CELLS,
            )
            return (
                "raise min(width,depth) to >= "
                f"{_render_world_units(FEATURE_MIN_CELLS * pitch / 2.0)} "
                f"or res to >= {minimum_resolution}"
            )
        case GeometryRule.FEATURE_SIZE, BoxPart(
            size=size,
            round_radius=round_radius,
        ):
            minimum_total_extent = 2.0 * (
                float(np.min(size)) + round_radius
            )
            required_half_extent = FEATURE_MIN_CELLS * pitch / 2.0
            minimum_resolution = _minimum_resolution(
                maximum_extent,
                minimum_total_extent,
                FEATURE_MIN_CELLS,
            )
            return (
                "raise min(size)+round to >= "
                f"{_render_world_units(required_half_extent)} or res to >= "
                f"{minimum_resolution}"
            )
        case GeometryRule.SHARP_EDGE, BoxPart(round_radius=round_radius):
            resolution_repair = (
                ""
                if round_radius == 0.0
                else (
                    " or res to >= "
                    f"{_minimum_resolution(maximum_extent, round_radius, 1.0)}"
                )
            )
            return (
                f"raise round to >= {_render_world_units(pitch)}"
                f"{resolution_repair}"
            )
        case (
            GeometryRule.PROFILE_ANCHOR_ARITY
            | GeometryRule.PROFILE_OFFSET
            | GeometryRule.PROFILE_ASPECT
            | GeometryRule.PROFILE_DEPTH
            | GeometryRule.PROFILE_DEPTH_CONFLICT
            | GeometryRule.PROFILE_EXPONENT
            | GeometryRule.PROFILE_ROLL
            | GeometryRule.COMPOSITION_OPERATOR
            | GeometryRule.QUAT_NONUNIT
            | GeometryRule.BOX_NEGATIVE_ROUND
            | GeometryRule.WEB_ANCHOR_ARITY
            | GeometryRule.WEB_STATION_ARITY
            | GeometryRule.WEB_HALF_EXTENT
            | GeometryRule.UNKNOWN_PART_KIND
            | GeometryRule.MALFORMED_PART
            | GeometryRule.MALFORMED_GRAPH
        ), _:
            return None
        case (
            GeometryRule.FENCE_ROT_ON_GENCYL
            | GeometryRule.FENCE_PROFILE_ON_NONGENCYL
        ), _:
            return None
        case _ as unreachable:
            assert_never(unreachable)


def _derived_obstructions(
    part: Part,
    *,
    pitch: float,
    resolution: int,
) -> tuple[GeometryObstruction, ...]:
    match part:
        case GencylPart(profile=None):
            return ()
        case GencylPart(
            part_id=part_id,
            radii=radii,
            profile=profile,
        ) if profile is not None and isinstance(
            profile.depths, RelativeProfileDepths
        ):
            thin = 2.0 * float(
                np.min(
                    np.asarray(radii, dtype=np.float64)
                    * np.clip(
                        np.asarray(profile.depths.ratios, dtype=np.float64),
                        1e-9,
                        None,
                    )
                )
            )
            return (
                ()
                if thin >= FEATURE_MIN_CELLS * pitch
                else (
                    GeometryObstruction(
                        part_id,
                        "gencyl",
                        GeometryRule.FEATURE_SIZE,
                        f"2*min(r*aspect) = {thin:.6f} < {FEATURE_MIN_CELLS}*pitch = "
                        f"{FEATURE_MIN_CELLS * pitch:.6f} at res={resolution}.",
                        False,
                    ),
                )
            )
        case GencylPart(
            part_id=part_id,
            radii=radii,
            profile=profile,
        ) if profile is not None and isinstance(
            profile.depths, AbsoluteProfileDepths
        ):
            thin = 2.0 * float(
                np.min(
                    np.minimum(
                        np.asarray(radii, dtype=np.float64),
                        np.asarray(profile.depths.values, dtype=np.float64),
                    )
                )
            )
            return (
                ()
                if thin >= FEATURE_MIN_CELLS * pitch
                else (
                    GeometryObstruction(
                        part_id,
                        "gencyl",
                        GeometryRule.FEATURE_SIZE,
                        f"2*min(width,depth) = {thin:.6f} < "
                        f"{FEATURE_MIN_CELLS}*pitch = "
                        f"{FEATURE_MIN_CELLS * pitch:.6f} at res={resolution}.",
                        False,
                    ),
                )
            )
        case BlobPart():
            return ()
        case BoxPart(
            part_id=part_id,
            size=size,
            round_radius=round_radius,
        ):
            minimum_total_extent = 2.0 * (float(np.min(size)) + round_radius)
            feature = (
                ()
                if minimum_total_extent >= FEATURE_MIN_CELLS * pitch
                else (
                    GeometryObstruction(
                        part_id,
                        "box",
                        GeometryRule.FEATURE_SIZE,
                        f"2*(min(size)+round) = {minimum_total_extent:.6f} < "
                        f"{FEATURE_MIN_CELLS}*pitch = "
                        f"{FEATURE_MIN_CELLS * pitch:.6f} at res={resolution}.",
                        False,
                    ),
                )
            )
            sharp_edge = (
                ()
                if round_radius >= pitch
                else (
                    GeometryObstruction(
                        part_id,
                        "box",
                        GeometryRule.SHARP_EDGE,
                        f"round = {round_radius:.6f} < pitch = {pitch:.6f}; sharp "
                        f"edges alias under marching cubes at res={resolution}.",
                        False,
                    ),
                )
            )
            return feature + sharp_edge
        case _ as unreachable:
            assert_never(unreachable)


def vocab_violations(graph, res=170):
    decoded = decode_graph(graph)
    if isinstance(decoded, Rejected):
        return [
            _record(
                obstruction,
                _raw_repair(
                    obstruction,
                    _raw_part_for(graph, obstruction),
                ),
            )
            for obstruction in decoded.obstructions
        ]
    lower, upper = graph_bounds_checked(decoded.value)
    maximum_extent = float(np.max(upper - lower))
    pitch = float(np.max((upper - lower) / (int(res) - 1)))
    raw_primitives = _raw_primitives(graph)
    typed_primitives = decoded.value.parts + decoded.value.carves
    contextual_obstructions = tuple(
        (obstruction, part, raw_part)
        for part, raw_part in zip(
            typed_primitives,
            raw_primitives,
            strict=True,
        )
        for obstruction in (
            part.obstructions
            + _derived_obstructions(part, pitch=pitch, resolution=int(res))
        )
    )
    return [
        _record(
            obstruction,
            _accepted_repair(
                obstruction,
                part,
                raw_part,
                pitch=pitch,
                maximum_extent=maximum_extent,
            ),
        )
        for obstruction, part, raw_part in contextual_obstructions
    ]
