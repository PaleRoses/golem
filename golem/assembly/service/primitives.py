"""Applicative decode combinators over ``_DecodeResult``."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from golem.materials import AcceptedMaterial, RejectedMaterial

from golem.assembly.service.address import (
    BoneAddress,
    ContactAddress,
    ElementAddress,
    MountAddress,
    PortAddress,
    RegionAddress,
    RejectedSemanticAddress,
    SemanticAddress,
    SemanticAddressKind,
    decode_semantic_address,
)
from golem.assembly.service.model import Vector3
from golem.assembly.service.outcome import (
    InvalidSemanticAddressKindObstruction,
    MalformedServiceIntentObstruction,
    NonPositiveBudgetObstruction,
    ServiceIntentObstruction,
    UnknownPhysicalMaterialObstruction,
    UnknownSemanticAddressObstruction,
)


@dataclass(frozen=True)
class _Decoded[value]:
    value: value


type _DecodeResult[value] = _Decoded[value] | tuple[ServiceIntentObstruction, ...]


_INTERFACE_ADDRESS_TYPES = (BoneAddress, PortAddress, ContactAddress, MountAddress)
_INTERFACE_ADDRESS_KINDS = (
    SemanticAddressKind.BONE,
    SemanticAddressKind.PORT,
    SemanticAddressKind.CONTACT,
    SemanticAddressKind.MOUNT,
)
_DOMAIN_ADDRESS_TYPES = (ElementAddress, RegionAddress)
_DOMAIN_ADDRESS_KINDS = (
    SemanticAddressKind.ELEMENT,
    SemanticAddressKind.REGION,
)


def _decode_material[material](
    decoder: Callable[[object], AcceptedMaterial | RejectedMaterial],
    value: object,
    path: str,
) -> _DecodeResult[material]:
    result = decoder(value)
    return (
        _Decoded(result.material)
        if isinstance(result, AcceptedMaterial)
        else tuple(
            UnknownPhysicalMaterialObstruction(
                path,
                obstruction.material_kind,
                obstruction.identifier,
            )
            for obstruction in result.obstructions
        )
    )


def _expected_implemented[member: StrEnum](
    value: object, implemented: member, path: str, noun: str
) -> _DecodeResult[member]:
    return (
        _Decoded(implemented)
        if value == implemented.value
        else (
            MalformedServiceIntentObstruction(
                path, f"expected the implemented {noun} {implemented.value!r}"
            ),
        )
    )


def _decode_bound_address(
    value: object,
    path: str,
    known_addresses: frozenset[SemanticAddress],
    allowed_types: tuple[type[object], ...],
    allowed_kinds: tuple[SemanticAddressKind, ...],
) -> _DecodeResult[SemanticAddress]:
    decoded = decode_semantic_address(value, path)
    if isinstance(decoded, RejectedSemanticAddress):
        return decoded.obstructions
    address = decoded.address
    obstructions: tuple[ServiceIntentObstruction, ...] = (
        ()
        if isinstance(address, allowed_types)
        else (InvalidSemanticAddressKindObstruction(path, address, allowed_kinds),)
    ) + (
        ()
        if address in known_addresses
        else (UnknownSemanticAddressObstruction(path, address),)
    )
    return obstructions if obstructions else _Decoded(address)


def _record(
    value: object, path: str, expected_keys: frozenset[str]
) -> _DecodeResult[dict[str, object]]:
    if not isinstance(value, dict):
        return (MalformedServiceIntentObstruction(path, "expected an object"),)
    actual_keys = frozenset(value)
    obstructions: tuple[ServiceIntentObstruction, ...] = (
        *tuple(
            MalformedServiceIntentObstruction(
                f"{path}/{key}", "missing required field"
            )
            for key in sorted(expected_keys - actual_keys)
        ),
        *tuple(
            MalformedServiceIntentObstruction(f"{path}/{key}", "unexpected field")
            for key in sorted(actual_keys - expected_keys, key=str)
        ),
    )
    return obstructions if obstructions else _Decoded(value)


def _traverse_array[value](
    raw: object,
    path: str,
    decode_item: Callable[[object, int], _DecodeResult[value]],
    *,
    require_nonempty: bool = False,
) -> _DecodeResult[tuple[value, ...]]:
    if not isinstance(raw, list) or (require_nonempty and not raw):
        qualifier = "non-empty " if require_nonempty else ""
        return (
            MalformedServiceIntentObstruction(
                path, f"expected a {qualifier}array"
            ),
        )
    results = tuple(
        decode_item(item, index) for index, item in enumerate(raw)
    )
    obstructions = _result_obstructions(results)
    return (
        obstructions
        if obstructions
        else _Decoded(
            tuple(
                result.value
                for result in results
                if isinstance(result, _Decoded)
            )
        )
    )


def _nonempty_string(value: object, path: str) -> _DecodeResult[str]:
    return (
        _Decoded(value)
        if isinstance(value, str) and bool(value)
        else (
            MalformedServiceIntentObstruction(path, "expected a non-empty string"),
        )
    )


def _closed_value(
    value: object, vocabulary: type[StrEnum], path: str
) -> _DecodeResult[StrEnum]:
    tokens = tuple(member.value for member in vocabulary)
    return (
        _Decoded(vocabulary(value))
        if isinstance(value, str) and value in tokens
        else (
            MalformedServiceIntentObstruction(
                path, f"expected one of {tokens!r}"
            ),
        )
    )


def _vector3(value: object, path: str) -> _DecodeResult[Vector3]:
    numbers = (
        tuple(map(_finite_number, value))
        if isinstance(value, list) and len(value) == 3
        else ()
    )
    return (
        _Decoded(Vector3(*numbers))
        if len(numbers) == 3 and all(number is not None for number in numbers)
        else (
            MalformedServiceIntentObstruction(
                path, "expected three finite SI components"
            ),
        )
    )


def _positive_budget(value: object, path: str) -> _DecodeResult[float]:
    number = _finite_number(value)
    return (
        _Decoded(number)
        if number is not None and number > 0.0
        else (NonPositiveBudgetObstruction(path, value),)
    )


def _minimum_integer(value: object, path: str, minimum: int) -> _DecodeResult[int]:
    return (
        _Decoded(value)
        if isinstance(value, int) and not isinstance(value, bool) and value >= minimum
        else (
            MalformedServiceIntentObstruction(
                path, f"expected an integer greater than or equal to {minimum}"
            ),
        )
    )


def _exact_integer(value: object, path: str, expected: int) -> _DecodeResult[int]:
    return (
        _Decoded(value)
        if isinstance(value, int)
        and not isinstance(value, bool)
        and value == expected
        else (
            MalformedServiceIntentObstruction(
                path, f"expected the integer {expected}"
            ),
        )
    )


def _bounded_fraction(
    value: object, path: str, maximum: float
) -> _DecodeResult[float]:
    number = _finite_number(value)
    return (
        _Decoded(number)
        if number is not None and 0.0 < number <= maximum
        else (
            MalformedServiceIntentObstruction(
                path, f"expected a finite fraction in (0, {maximum}]"
            ),
        )
    )


def _finite_number(value: object) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        else None
    )


def _result_obstructions(
    results: tuple[_DecodeResult[object], ...],
) -> tuple[ServiceIntentObstruction, ...]:
    return tuple(
        obstruction
        for result in results
        if isinstance(result, tuple)
        for obstruction in result
    )


def _duplicate_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        value
        for index, value in enumerate(values)
        if value in values[:index]
        and value not in values[values.index(value) + 1 : index]
    )
