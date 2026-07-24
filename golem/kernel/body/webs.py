"""Strict multi-anchor spanning-web authoring projected into engine primitives.

A spanning web (R10) is ONE multi-anchor record — a membrane stretched
between two or more authored bones (the wing: finger bones as anchor curves,
the membrane as the sheet between them). Anchor curves are sampled along the
FK'd bone axes after pose, so a folded wing folds the membrane for free. The
record composes its anchors into the engine's ordered anchor tuple
(``decode_web``); mirroring reflects the complete tuple as one instance,
never per-anchor. Strict decode: every malformation is a typed obstruction
naming the legal shape, and unequal anchor station counts are a rejection
naming the per-anchor counts — never a silent pad or truncate.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from golem.kernel.body.canonical import _r, _rvec
from golem.kernel.body.decode import (
    _duplicate_obstructions,
    _finite_number,
    _malformed,
    _number_array,
    _unknown_field_obstructions,
)
from golem.kernel.body.types import (
    FLESH_ROLES,
    AcceptedWebs,
    DuplicateWebIdObstruction,
    MalformedWebObstruction,
    RejectedWebs,
    UnknownWebAnchorObstruction,
    WebCompileResult,
    WebObstruction,
    WebPartCollisionObstruction,
    WebRule,
)
from golem.kernel.engine import CompositionOperator


_WEB_ID_SYNTAX = "[A-Za-z][A-Za-z0-9_.-]*"
_WEB_ID = re.compile(rf"{_WEB_ID_SYNTAX}\Z")
_WEB_FIELDS = frozenset(("id", "anchors", "mirror", "blend", "operator", "role"))
_ANCHOR_FIELDS = frozenset(("bone", "span", "stations", "radii"))
_OPERATORS = tuple(operator.value for operator in CompositionOperator)


@dataclass(frozen=True)
class _AnchorSpec:
    bone: str
    stations: tuple[float, ...]
    radii: tuple[float, ...]


@dataclass(frozen=True)
class _WebSpec:
    web_id: str
    anchors: tuple[_AnchorSpec, ...]
    mirror: bool
    blend: float | None
    operator: str | None
    role: str | None


@dataclass(frozen=True)
class _AuthoredWeb:
    address: str
    index: int
    declaration: Mapping[str, object]


def _anchor_obstructions(
    declaration: object,
    address: str,
    known_bones: frozenset[str],
) -> tuple[WebObstruction, ...]:
    if not isinstance(declaration, Mapping):
        return (
            MalformedWebObstruction(
                address=address,
                rule=WebRule.ANCHOR_OBJECT,
                authored=declaration,
                required="object",
            ),
        )
    bone = declaration.get("bone")
    span_value = declaration.get("span", (0.0, 1.0))
    radii_value = declaration.get("radii")
    stations_value = declaration.get("stations")
    span = _number_array(span_value)
    radii = _number_array(radii_value)
    stations = _number_array(stations_value) if stations_value is not None else None
    span_valid = span is not None and len(span) == 2 and span[0] < span[1]
    radii_valid = (
        radii is not None and len(radii) >= 2 and all(radius > 0.0 for radius in radii)
    )
    stations_shape_valid = (
        stations_value is None
        or (
            stations is not None
            and len(stations) >= 2
            and all(left < right for left, right in zip(stations, stations[1:]))
        )
    )
    matching_arity = (
        stations_value is None
        or not radii_valid
        or not stations_shape_valid
        or len(stations) == len(radii)
    )
    stations_match_span = (
        stations_value is None
        or not span_valid
        or not stations_shape_valid
        or (stations[0], stations[-1]) == span
    )
    return (
        *_unknown_field_obstructions(
            MalformedWebObstruction,
            WebRule.KNOWN_FIELD,
            declaration,
            _ANCHOR_FIELDS,
            address,
        ),
        *_malformed(
            MalformedWebObstruction,
            isinstance(bone, str) and bool(bone),
            f"{address}/bone",
            WebRule.IDENTIFIER,
            bone,
            "declared bone id",
        ),
        *(
            ()
            if not isinstance(bone, str) or bone in known_bones
            else (
                UnknownWebAnchorObstruction(
                    f"{address}/bone",
                    bone,
                    tuple(difflib.get_close_matches(bone, sorted(known_bones), n=3)),
                ),
            )
        ),
        *_malformed(
            MalformedWebObstruction,
            span_valid,
            f"{address}/span",
            WebRule.FINITE_STRICT_PAIR,
            span_value,
            {"arity": 2, "finite": True, "strictly_increasing": True},
        ),
        *_malformed(
            MalformedWebObstruction,
            radii_valid,
            f"{address}/radii",
            WebRule.FINITE_POSITIVE_ARRAY,
            radii_value,
            {"minimum_arity": 2, "finite": True, "exclusive_minimum": 0.0},
        ),
        *_malformed(
            MalformedWebObstruction,
            stations_shape_valid,
            f"{address}/stations",
            WebRule.FINITE_STRICT_ARRAY,
            stations_value,
            {"minimum_arity": 2, "finite": True, "strictly_increasing": True},
        ),
        *_malformed(
            MalformedWebObstruction,
            matching_arity,
            f"{address}/stations",
            WebRule.MATCHING_ARITY,
            None if stations is None else len(stations),
            None if radii is None else len(radii),
        ),
        *_malformed(
            MalformedWebObstruction,
            stations_match_span,
            f"{address}/stations",
            WebRule.STATIONS_MATCH_SPAN,
            None if stations is None else (stations[0], stations[-1]),
            span,
        ),
    )


def _station_arity_obstruction(
    declaration: Mapping[str, object], address: str
) -> tuple[MalformedWebObstruction, ...]:
    """The station-arity law: every anchor must emit the same station count.

    Rejection names each anchor's count; the compiler never silently pads or
    truncates one anchor's curve to fit another's."""
    anchors = declaration.get("anchors")
    if not isinstance(anchors, (list, tuple)):
        return ()
    counts = tuple(
        (index, anchor.get("bone"), len(radii))
        for index, anchor in enumerate(anchors)
        if isinstance(anchor, Mapping)
        for radii in (_number_array(anchor.get("radii")),)
        if radii is not None and len(radii) >= 2 and all(r > 0.0 for r in radii)
    )
    if len(counts) < 2 or len(frozenset(count for _i, _bone, count in counts)) == 1:
        return ()
    return (
        MalformedWebObstruction(
            address=f"{address}/anchors",
            rule=WebRule.STATION_ARITY,
            authored={
                f"anchors[{index}]": {"bone": bone, "stations": count}
                for index, bone, count in counts
            },
            required="identical station counts across all anchors",
        ),
    )


def _entry_obstructions(
    authored: _AuthoredWeb, known_bones: frozenset[str]
) -> tuple[WebObstruction, ...]:
    declaration = authored.declaration
    address = authored.address
    web_id = declaration.get("id")
    anchors = declaration.get("anchors")
    mirror = declaration.get("mirror", False)
    blend = declaration.get("blend")
    operator = declaration.get("operator")
    role = declaration.get("role")
    anchor_obstructions: tuple[WebObstruction, ...] = (
        tuple(
            obstruction
            for index, anchor in enumerate(anchors)
            for obstruction in _anchor_obstructions(
                anchor, f"{address}/anchors[{index}]", known_bones
            )
        )
        if isinstance(anchors, (list, tuple))
        else ()
    )
    return (
        *_unknown_field_obstructions(
            MalformedWebObstruction,
            WebRule.KNOWN_FIELD,
            declaration,
            _WEB_FIELDS,
            address,
        ),
        *_malformed(
            MalformedWebObstruction,
            isinstance(web_id, str) and _WEB_ID.fullmatch(web_id) is not None,
            f"{address}/id",
            WebRule.IDENTIFIER,
            web_id,
            _WEB_ID_SYNTAX,
        ),
        *_malformed(
            MalformedWebObstruction,
            isinstance(anchors, (list, tuple)),
            f"{address}/anchors",
            WebRule.ANCHORS_ARRAY,
            anchors,
            "array",
        ),
        *_malformed(
            MalformedWebObstruction,
            not isinstance(anchors, (list, tuple)) or len(anchors) >= 2,
            f"{address}/anchors",
            WebRule.ANCHOR_ARITY,
            None if not isinstance(anchors, (list, tuple)) else len(anchors),
            {"minimum_arity": 2},
        ),
        *anchor_obstructions,
        *_station_arity_obstruction(declaration, address),
        *_malformed(
            MalformedWebObstruction,
            isinstance(mirror, bool),
            f"{address}/mirror",
            WebRule.BOOLEAN,
            mirror,
            "boolean",
        ),
        *_malformed(
            MalformedWebObstruction,
            blend is None or (_finite_number(blend) and float(blend) >= 0.0),
            f"{address}/blend",
            WebRule.FINITE_NON_NEGATIVE_NUMBER,
            blend,
            {"finite": True, "minimum": 0.0},
        ),
        *_malformed(
            MalformedWebObstruction,
            operator is None or operator in _OPERATORS,
            f"{address}/operator",
            WebRule.OPERATOR,
            operator,
            _OPERATORS,
        ),
        *_malformed(
            MalformedWebObstruction,
            role is None or role in FLESH_ROLES,
            f"{address}/role",
            WebRule.ROLE,
            role,
            FLESH_ROLES,
        ),
    )


def _authored_entries(
    payload: object,
) -> tuple[tuple[_AuthoredWeb, ...], tuple[MalformedWebObstruction, ...]]:
    if payload is None:
        return (), ()
    if not isinstance(payload, (list, tuple)):
        return (), (
            MalformedWebObstruction(
                address="webs",
                rule=WebRule.WEBS_ARRAY,
                authored=payload,
                required="array",
            ),
        )
    entries = tuple(
        _AuthoredWeb(
            address=f"webs[{index}]",
            index=index,
            declaration=declaration,
        )
        for index, declaration in enumerate(payload)
        if isinstance(declaration, Mapping)
    )
    malformed_entries = tuple(
        MalformedWebObstruction(
            address=f"webs[{index}]",
            rule=WebRule.OBJECT,
            authored=declaration,
            required="object",
        )
        for index, declaration in enumerate(payload)
        if not isinstance(declaration, Mapping)
    )
    return entries, malformed_entries


def _decode_anchor(declaration: Mapping[str, object]) -> _AnchorSpec:
    span = tuple(map(float, declaration.get("span", (0.0, 1.0))))
    radii = tuple(map(float, declaration["radii"]))
    stations = (
        tuple(map(float, declaration["stations"]))
        if "stations" in declaration
        else tuple(map(float, np.linspace(span[0], span[1], len(radii))))
    )
    return _AnchorSpec(str(declaration["bone"]), stations, radii)


def _decode_web(declaration: Mapping[str, object]) -> _WebSpec:
    return _WebSpec(
        web_id=str(declaration["id"]),
        anchors=tuple(map(_decode_anchor, declaration["anchors"])),
        mirror=bool(declaration.get("mirror", False)),
        blend=(
            float(declaration["blend"]) if "blend" in declaration else None
        ),
        operator=(
            str(declaration["operator"]) if "operator" in declaration else None
        ),
        role=str(declaration["role"]) if "role" in declaration else None,
    )


def _place_anchor(anchor: _AnchorSpec, record: Mapping[str, object]) -> dict:
    origin = np.asarray(record["head"], dtype=np.float64)
    rotation = np.asarray(record["R"], dtype=np.float64)
    length = float(record["length"])
    return {
        "spine": [
            _rvec(origin + rotation @ np.asarray((0.0, 0.0, station * length)))
            for station in anchor.stations
        ],
        "radii": [_r(radius) for radius in anchor.radii],
    }


def _place_web(
    spec: _WebSpec, bones: Mapping[str, Mapping[str, object]]
) -> dict:
    return {
        "id": spec.web_id,
        "anchors": [
            _place_anchor(anchor, bones[anchor.bone]) for anchor in spec.anchors
        ],
        **({"mirror": True} if spec.mirror else {}),
        **({"blend": _r(spec.blend)} if spec.blend is not None else {}),
        **({"operator": spec.operator} if spec.operator is not None else {}),
        **({"role": spec.role} if spec.role is not None else {}),
    }


def compile_webs(
    payload: object,
    bones: Mapping[str, Mapping[str, object]],
    occupied_part_ids: frozenset[str],
) -> WebCompileResult:
    entries, structural = _authored_entries(payload)
    known_bones = frozenset(bones)
    obstructions: tuple[WebObstruction, ...] = (
        *structural,
        *tuple(
            obstruction
            for entry in entries
            for obstruction in _entry_obstructions(entry, known_bones)
        ),
        *_duplicate_obstructions(
            DuplicateWebIdObstruction,
            WebPartCollisionObstruction,
            entries,
            occupied_part_ids,
        ),
    )
    if obstructions:
        return RejectedWebs(obstructions)
    decoded = tuple(
        (entry, _decode_web(entry.declaration)) for entry in entries
    )
    return AcceptedWebs(
        webs=tuple(_place_web(spec, bones) for _entry, spec in decoded),
        # Carrier eligibility is the same role vocabulary as any flesh (R3):
        # a non_carrier web emits geometry but carries no carrier provenance.
        provenance=tuple(
            (
                spec.web_id,
                f"{entry.address} anchors=({', '.join(f'skeleton/{anchor.bone}' for anchor in spec.anchors)})",
            )
            for entry, spec in decoded
            if spec.role not in FLESH_ROLES
        ),
        anchors=tuple(
            (spec.web_id, tuple(anchor.bone for anchor in spec.anchors))
            for _entry, spec in decoded
        ),
        non_carrier=tuple(
            spec.web_id for _entry, spec in decoded if spec.role in FLESH_ROLES
        ),
    )
