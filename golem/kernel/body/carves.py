"""Strict bone-local carve authoring projected into engine primitives."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from golem.kernel.body.canonical import _r, _rvec
from golem.kernel.body.decode import (
    _duplicate_obstructions,
    _finite_number,
    _malformed,
    _number_array,
    _positive_vector3,
    _rotation_payload,
    _unknown_field_obstructions,
    _vector3,
)
from golem.kernel.body.geometry import placed_center
from golem.kernel.body.types import (
    CARVE_KINDS,
    AcceptedCarves,
    CarveCompileResult,
    CarveKind,
    CarveObstruction,
    CarvePartCollisionObstruction,
    CarveRule,
    DuplicateCarveIdObstruction,
    MalformedCarveObstruction,
    RejectedCarves,
    Vector3,
)


_CARVE_ID_SYNTAX = "[A-Za-z][A-Za-z0-9_.-]*"
_CARVE_ID = re.compile(rf"{_CARVE_ID_SYNTAX}\Z")
_COMMON_FIELDS = frozenset(("id", "kind"))
_CENTERED_FIELDS = _COMMON_FIELDS | frozenset(("t", "offset", "size"))
_FIELDS_BY_KIND = {
    CarveKind.GENCYL: _COMMON_FIELDS | frozenset(("span", "radii", "stations")),
    CarveKind.BLOB: _CENTERED_FIELDS,
    CarveKind.BOX: _CENTERED_FIELDS | frozenset(("round",)),
}
_ALL_FIELDS = frozenset().union(*_FIELDS_BY_KIND.values())


@dataclass(frozen=True)
class _CenteredCarve:
    carve_id: str
    kind: CarveKind
    parameter: float
    offset: Vector3
    size: Vector3
    round_radius: float


@dataclass(frozen=True)
class _GencylCarve:
    carve_id: str
    stations: tuple[float, ...]
    radii: tuple[float, ...]


type _CarveSpec = _CenteredCarve | _GencylCarve


@dataclass(frozen=True)
class _AuthoredCarve:
    address: str
    record: Mapping[str, object]
    declaration: Mapping[str, object]


def _centered_obstructions(
    declaration: Mapping[str, object],
    kind: CarveKind,
    address: str,
) -> tuple[MalformedCarveObstruction, ...]:
    parameter = declaration.get("t", 1.0)
    offset = declaration.get("offset", (0.0, 0.0, 0.0))
    size = declaration.get("size")
    round_radius = declaration.get("round", 0.0)
    return (
        *_malformed(
            MalformedCarveObstruction,
            _finite_number(parameter),
            f"{address}/t",
            CarveRule.FINITE_NUMBER,
            parameter,
            {"finite": True},
        ),
        *_malformed(
            MalformedCarveObstruction,
            _vector3(offset) is not None,
            f"{address}/offset",
            CarveRule.FINITE_VECTOR3,
            offset,
            {"arity": 3, "finite": True},
        ),
        *_malformed(
            MalformedCarveObstruction,
            _positive_vector3(size) is not None,
            f"{address}/size",
            CarveRule.FINITE_POSITIVE_VECTOR3,
            size,
            {"arity": 3, "finite": True, "exclusive_minimum": 0.0},
        ),
        *(
            _malformed(
                MalformedCarveObstruction,
                _finite_number(round_radius) and float(round_radius) >= 0.0,
                f"{address}/round",
                CarveRule.FINITE_NON_NEGATIVE_NUMBER,
                round_radius,
                {"finite": True, "minimum": 0.0},
            )
            if kind is CarveKind.BOX
            else ()
        ),
    )


def _gencyl_obstructions(
    declaration: Mapping[str, object], address: str
) -> tuple[MalformedCarveObstruction, ...]:
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
        *_malformed(
            MalformedCarveObstruction,
            span_valid,
            f"{address}/span",
            CarveRule.FINITE_STRICT_PAIR,
            span_value,
            {"arity": 2, "finite": True, "strictly_increasing": True},
        ),
        *_malformed(
            MalformedCarveObstruction,
            radii_valid,
            f"{address}/radii",
            CarveRule.FINITE_POSITIVE_ARRAY,
            radii_value,
            {"minimum_arity": 2, "finite": True, "exclusive_minimum": 0.0},
        ),
        *_malformed(
            MalformedCarveObstruction,
            stations_shape_valid,
            f"{address}/stations",
            CarveRule.FINITE_STRICT_ARRAY,
            stations_value,
            {"minimum_arity": 2, "finite": True, "strictly_increasing": True},
        ),
        *_malformed(
            MalformedCarveObstruction,
            matching_arity,
            f"{address}/stations",
            CarveRule.MATCHING_ARITY,
            None if stations is None else len(stations),
            None if radii is None else len(radii),
        ),
        *_malformed(
            MalformedCarveObstruction,
            stations_match_span,
            f"{address}/stations",
            CarveRule.STATIONS_MATCH_SPAN,
            None if stations is None else (stations[0], stations[-1]),
            span,
        ),
    )


def _entry_obstructions(
    authored: _AuthoredCarve,
) -> tuple[CarveObstruction, ...]:
    declaration = authored.declaration
    address = authored.address
    carve_id = declaration.get("id")
    kind_value = declaration.get("kind")
    kind = (
        CarveKind(kind_value)
        if isinstance(kind_value, str) and kind_value in CARVE_KINDS
        else None
    )
    return (
        *_unknown_field_obstructions(
            MalformedCarveObstruction,
            CarveRule.KNOWN_FIELD,
            declaration,
            _FIELDS_BY_KIND.get(kind, _ALL_FIELDS),
            address,
        ),
        *_malformed(
            MalformedCarveObstruction,
            isinstance(carve_id, str) and _CARVE_ID.fullmatch(carve_id) is not None,
            f"{address}/id",
            CarveRule.IDENTIFIER,
            carve_id,
            _CARVE_ID_SYNTAX,
        ),
        *_malformed(
            MalformedCarveObstruction,
            kind is not None,
            f"{address}/kind",
            CarveRule.KIND,
            kind_value,
            CARVE_KINDS,
        ),
        *(
            _centered_obstructions(declaration, kind, address)
            if kind in (CarveKind.BLOB, CarveKind.BOX)
            else _gencyl_obstructions(declaration, address)
            if kind is CarveKind.GENCYL
            else ()
        ),
    )


def _authored_entries(
    bones: Mapping[str, Mapping[str, object]], order: Sequence[str]
) -> tuple[tuple[_AuthoredCarve, ...], tuple[MalformedCarveObstruction, ...]]:
    sections = tuple(
        (bone_id, bones[bone_id], bones[bone_id].get("carves", ()))
        for bone_id in order
    )
    malformed_sections = tuple(
        MalformedCarveObstruction(
            address=f"skeleton/{bone_id}/carves",
            rule=CarveRule.CARVES_ARRAY,
            authored=payload,
            required="array",
        )
        for bone_id, _record, payload in sections
        if not isinstance(payload, (list, tuple))
    )
    entries = tuple(
        _AuthoredCarve(
            address=f"skeleton/{bone_id}/carves[{index}]",
            record=record,
            declaration=declaration,
        )
        for bone_id, record, payload in sections
        if isinstance(payload, (list, tuple))
        for index, declaration in enumerate(payload)
        if isinstance(declaration, Mapping)
    )
    malformed_entries = tuple(
        MalformedCarveObstruction(
            address=f"skeleton/{bone_id}/carves[{index}]",
            rule=CarveRule.OBJECT,
            authored=declaration,
            required="object",
        )
        for bone_id, _record, payload in sections
        if isinstance(payload, (list, tuple))
        for index, declaration in enumerate(payload)
        if not isinstance(declaration, Mapping)
    )
    return entries, (*malformed_sections, *malformed_entries)


def _decode_carve(declaration: Mapping[str, object]) -> _CarveSpec:
    kind = CarveKind(str(declaration["kind"]))
    if kind is CarveKind.GENCYL:
        span = tuple(map(float, declaration.get("span", (0.0, 1.0))))
        radii = tuple(map(float, declaration["radii"]))
        stations = (
            tuple(map(float, declaration["stations"]))
            if "stations" in declaration
            else tuple(map(float, np.linspace(span[0], span[1], len(radii))))
        )
        return _GencylCarve(str(declaration["id"]), stations, radii)
    return _CenteredCarve(
        carve_id=str(declaration["id"]),
        kind=kind,
        parameter=float(declaration.get("t", 1.0)),
        offset=tuple(map(float, declaration.get("offset", (0.0, 0.0, 0.0)))),
        size=tuple(map(float, declaration["size"])),
        round_radius=float(declaration.get("round", 0.0)),
    )


def _place_carve(spec: _CarveSpec, record: Mapping[str, object]) -> dict:
    origin = np.asarray(record["head"], dtype=np.float64)
    rotation = np.asarray(record["R"], dtype=np.float64)
    mirrored = {"mirror": True} if bool(record["mirrored"]) else {}
    if isinstance(spec, _GencylCarve):
        return {
            "id": spec.carve_id,
            "type": CarveKind.GENCYL.value,
            "spine": [
                _rvec(origin + rotation @ np.asarray((0.0, 0.0, station * float(record["length"]))))
                for station in spec.stations
            ],
            "radii": [_r(radius) for radius in spec.radii],
            **mirrored,
        }
    center = placed_center(
        {"t": spec.parameter, "offset": spec.offset},
        origin,
        rotation,
        float(record["length"]),
    )
    return {
        "id": spec.carve_id,
        "type": spec.kind.value,
        "center": _rvec(center),
        "size": [_r(value) for value in spec.size],
        **(
            {"round": _r(spec.round_radius)}
            if spec.kind is CarveKind.BOX
            else {}
        ),
        **_rotation_payload(rotation),
        **mirrored,
    }


def compile_carves(
    bones: Mapping[str, Mapping[str, object]],
    order: Sequence[str],
    occupied_part_ids: frozenset[str],
) -> CarveCompileResult:
    entries, structural = _authored_entries(bones, order)
    obstructions: tuple[CarveObstruction, ...] = (
        *structural,
        *tuple(
            obstruction
            for entry in entries
            for obstruction in _entry_obstructions(entry)
        ),
        *_duplicate_obstructions(
            DuplicateCarveIdObstruction,
            CarvePartCollisionObstruction,
            entries,
            occupied_part_ids,
        ),
    )
    if obstructions:
        return RejectedCarves(obstructions)
    placed = tuple(
        (_place_carve(_decode_carve(entry.declaration), entry.record), entry.address)
        for entry in entries
    )
    return AcceptedCarves(
        carves=tuple(carve for carve, _address in placed),
        provenance=tuple((str(carve["id"]), address) for carve, address in placed),
    )
