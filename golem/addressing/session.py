"""Immutable carriers for the session address grammar."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldName:
    name: str


@dataclass(frozen=True)
class FieldIndex:
    index: int


type FieldToken = FieldName | FieldIndex


@dataclass(frozen=True)
class DottedQualifiedToken:
    value: str


@dataclass(frozen=True)
class IndexedSegment:
    collection: str
    index: int


@dataclass(frozen=True)
class Address:
    segments: tuple[str, ...]
    fieldpath: tuple[FieldToken, ...]
    encoded_segments: frozenset[int] = frozenset()


@dataclass(frozen=True)
class AddressObstruction:
    source: str
    reason: str


@dataclass(frozen=True)
class RejectedAddress:
    obstructions: tuple[AddressObstruction, ...]


type AddressResult = Address | RejectedAddress
type DottedQualifiedTokenResult = DottedQualifiedToken | RejectedAddress
