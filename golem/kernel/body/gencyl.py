"""Generalized-cylinder axial station resolution + skeleton-wide validation."""

from __future__ import annotations

import math

import numpy as np

from golem.kernel.body.decode import _finite_number
from golem.kernel.body.types import (
    LoftSection,
    LoftSectionObstruction,
    LoftSectionResult,
    LoftSectionRule,
    _GencylStationObstruction,
    _GencylStationResult,
)


_LOFT_SECTION_FIELDS = frozenset(
    ("station", "width", "depth", "exponent", "roll")
)


def _decode_loft_section(
    index: int, payload: object
) -> LoftSection | LoftSectionObstruction:
    if not isinstance(payload, dict):
        return LoftSectionObstruction(
            index,
            LoftSectionRule.SECTION_OBJECT,
            observed_value=payload,
        )
    fields = frozenset(payload)
    if fields != _LOFT_SECTION_FIELDS:
        missing = tuple(sorted(_LOFT_SECTION_FIELDS - fields))
        extra = tuple(sorted(fields - _LOFT_SECTION_FIELDS))
        return LoftSectionObstruction(
            index,
            LoftSectionRule.EXACT_FIELDS,
            observed_value=tuple(sorted(fields)),
            missing_fields=missing,
            extra_fields=extra,
        )
    station, width, depth, exponent, roll = (
        payload["station"],
        payload["width"],
        payload["depth"],
        payload["exponent"],
        payload["roll"],
    )
    invalid_field = next(
        (
            name
            for name, value in (
                ("station", station),
                ("width", width),
                ("depth", depth),
                ("exponent", exponent),
                ("roll", roll),
            )
            if not _finite_number(value)
        ),
        None,
    )
    if invalid_field is not None:
        return LoftSectionObstruction(
            index,
            LoftSectionRule.FINITE_FIELD,
            field=invalid_field,
            observed_value=payload[invalid_field],
        )
    if float(width) <= 0.0:
        return LoftSectionObstruction(
            index,
            LoftSectionRule.POSITIVE_WIDTH,
            field="width",
            observed_value=width,
        )
    if float(depth) <= 0.0:
        return LoftSectionObstruction(
            index,
            LoftSectionRule.POSITIVE_DEPTH,
            field="depth",
            observed_value=depth,
        )
    if not 2.0 <= float(exponent) <= 12.0:
        return LoftSectionObstruction(
            index,
            LoftSectionRule.EXPONENT_RANGE,
            field="exponent",
            observed_value=exponent,
        )
    return LoftSection(
        station=float(station),
        width=float(width),
        depth=float(depth),
        exponent=float(exponent),
        roll=float(roll),
    )


def _loft_sections(flesh: dict) -> LoftSectionResult:
    payload = flesh.get("sections")
    if not isinstance(payload, (list, tuple)):
        return LoftSectionObstruction(
            None,
            LoftSectionRule.SECTIONS_ARRAY,
            field="sections",
            observed_value=payload,
        )
    if len(payload) < 2:
        return LoftSectionObstruction(
            None,
            LoftSectionRule.MINIMUM_ARITY,
            field="sections",
            observed_value=len(payload),
        )
    decoded = tuple(
        _decode_loft_section(index, section)
        for index, section in enumerate(payload)
    )
    obstruction = next(
        (
            result
            for result in decoded
            if isinstance(result, LoftSectionObstruction)
        ),
        None,
    )
    if obstruction is not None:
        return obstruction
    sections = tuple(
        result for result in decoded if isinstance(result, LoftSection)
    )
    unordered_index = next(
        (
            index
            for index, (left, right) in enumerate(
                zip(sections, sections[1:]), start=1
            )
            if left.station >= right.station
        ),
        None,
    )
    return (
        LoftSectionObstruction(
            unordered_index,
            LoftSectionRule.STRICT_STATION_ORDER,
            field="station",
            observed_value=(
                sections[unordered_index - 1].station,
                sections[unordered_index].station,
            ),
        )
        if unordered_index is not None
        else sections
    )


def _gencyl_stations(flesh: dict) -> _GencylStationResult:
    """Resolve optional non-uniform bone-axis stations without widening IR.

    Authored stations are normalized bone coordinates and replace only the
    old uniform sampling.  Their endpoints must agree with ``span`` so one
    flesh record cannot carry two contradictory axial extents.
    """
    radii = flesh.get("radii")
    authored = flesh.get("stations")
    if not isinstance(radii, (list, tuple)) or len(radii) < 2:
        return _GencylStationObstruction(
            "gencyl flesh requires at least two radii"
        )
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
        for value in radii
    ):
        return _GencylStationObstruction(
            "radii must contain only finite positive numbers"
        )
    span = flesh.get("span", [0.0, 1.0])
    if not (
        isinstance(span, (list, tuple))
        and len(span) == 2
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in span
        )
        and float(span[0]) < float(span[1])
    ):
        return _GencylStationObstruction(
            "span must be two finite strictly increasing numbers"
        )
    authored_profile = flesh.get("profile")
    profile = authored_profile if isinstance(authored_profile, dict) else {}
    mismatched_profile_anchor = next(
        (
            name
            for name in ("n", "aspect", "depth", "roll")
            for value in (profile.get(name),)
            if isinstance(value, (list, tuple))
            and len(value) != len(radii)
        ),
        None,
    )
    if mismatched_profile_anchor is not None:
        return _GencylStationObstruction(
            f"profile.{mismatched_profile_anchor} and radii must have identical arity"
        )
    if authored is None:
        return tuple(map(float, np.linspace(span[0], span[1], len(radii))))
    if not isinstance(authored, (list, tuple)):
        return _GencylStationObstruction("stations must be an array")
    if len(authored) != len(radii):
        return _GencylStationObstruction(
            "stations and radii must have identical arity"
        )
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in authored
    ):
        return _GencylStationObstruction(
            "stations must contain only finite numbers"
        )
    stations = tuple(map(float, authored))
    if not all(
        left < right for left, right in zip(stations, stations[1:])
    ):
        return _GencylStationObstruction(
            "stations must be strictly increasing"
        )
    if not (
        stations[0] == float(span[0])
        and stations[-1] == float(span[1])
    ):
        return _GencylStationObstruction(
            "first and last stations must equal the span endpoints"
        )
    return stations


def _gencyl_station_violations(spec: dict) -> tuple[dict, ...]:
    skeleton = spec.get("skeleton", {})
    root = skeleton.get("root", {})
    bones = (root, *tuple(skeleton.get("bones", ())))
    return tuple(
        {
            "rule": "bad_gencyl_stations",
            "address": f"skeleton/{bone.get('id', '<unknown>')}/flesh[{index}]/stations",
            "detail": result.reason,
        }
        for bone in bones
        for index, flesh in enumerate(bone.get("flesh", ()))
        if flesh.get("kind") == "gencyl"
        for result in (_gencyl_stations(flesh),)
        if isinstance(result, _GencylStationObstruction)
    )


def _loft_section_violations(spec: dict) -> tuple[dict, ...]:
    skeleton = spec.get("skeleton", {})
    root = skeleton.get("root", {})
    bones = (root, *tuple(skeleton.get("bones", ())))
    return tuple(
        {
            "rule": "bad_loft_sections",
            "address": (
                f"skeleton/{bone.get('id', '<unknown>')}/flesh[{index}]/sections"
                + (
                    f"[{result.section_index}]"
                    if result.section_index is not None
                    else ""
                )
            ),
            "detail": result.reason,
        }
        for bone in bones
        for index, flesh in enumerate(bone.get("flesh", ()))
        if flesh.get("kind") == "loft"
        for result in (_loft_sections(flesh),)
        if isinstance(result, LoftSectionObstruction)
    )
