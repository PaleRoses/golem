"""Total parser and renderer for session addresses and lexical selectors."""

from __future__ import annotations

import re
from typing import assert_never
from urllib.parse import quote, unquote

from golem.addressing.session import (
    Address,
    AddressObstruction,
    AddressResult,
    DottedQualifiedToken,
    DottedQualifiedTokenResult,
    FieldIndex,
    FieldName,
    FieldToken,
    IndexedSegment,
    RejectedAddress,
)

_FIELD_TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")
_INDEXED_SEGMENT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]")
_DOTTED_QUALIFIED_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_MALFORMED_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


def encode_address_segment(value: str) -> str:
    return quote(value, safe="-._~")


def _decode_address_segment(value: str) -> str | None:
    if _MALFORMED_PERCENT_ESCAPE.search(value) is not None:
        return None
    try:
        return unquote(value, errors="strict")
    except UnicodeDecodeError:
        return None


def parse_address(source: str) -> AddressResult:
    if not source or source.startswith("/") or source.endswith("/"):
        return RejectedAddress((AddressObstruction(source, "malformed address"),))
    segment_part, _, field_part = source.partition("@")
    encoded_segments = tuple(segment_part.split("/")) if segment_part else ()
    decoded_segments = tuple(map(_decode_address_segment, encoded_segments))
    if any(segment is None for segment in decoded_segments):
        return RejectedAddress(
            (AddressObstruction(source, "malformed percent or UTF-8 escape"),)
        )
    segments = tuple(
        segment for segment in decoded_segments if segment is not None
    )
    encoded_segments = frozenset(
        index
        for index, segment in enumerate(encoded_segments)
        if "%" in segment
    )
    fieldpath = _parse_fieldpath(field_part) if field_part else ()
    if fieldpath is None:
        return RejectedAddress(
            (AddressObstruction(source, f"malformed fieldpath in {source!r}"),)
        )
    return Address(segments, fieldpath, encoded_segments)


def _parse_fieldpath(field_part: str) -> tuple[FieldToken, ...] | None:
    def scan(
        position: int, first: bool, acc: tuple[FieldToken, ...]
    ) -> tuple[FieldToken, ...] | None:
        if position >= len(field_part):
            return acc
        start = position if first or field_part[position] == "[" else position + 1
        if not first and field_part[position] not in ".[":
            return None
        found = _FIELD_TOKEN.match(field_part, start)
        if found is None:
            return None
        token: FieldToken = (
            FieldName(found.group(1))
            if found.group(1) is not None
            else FieldIndex(int(found.group(2)))
        )
        return scan(found.end(), False, acc + (token,))

    return scan(0, True, ())


def parse_dotted_qualified_token(source: str) -> DottedQualifiedTokenResult:
    return (
        DottedQualifiedToken(source)
        if _DOTTED_QUALIFIED_TOKEN.fullmatch(source) is not None
        else RejectedAddress(
            (
                AddressObstruction(
                    source,
                    f"malformed dotted qualified token in {source!r}",
                ),
            )
        )
    )


def parse_indexed_segment(source: str) -> IndexedSegment | None:
    matched = _INDEXED_SEGMENT.fullmatch(source)
    return (
        IndexedSegment(matched.group(1), int(matched.group(2)))
        if matched is not None
        else None
    )


def render_address(address: Address) -> str:
    stem = "/".join(
        segment
        if _is_unescaped_index_selector(address, index)
        else encode_address_segment(segment)
        for index, segment in enumerate(address.segments)
    )
    fieldpath = _render_fieldpath(address.fieldpath)
    return stem if not fieldpath else f"{stem}@{fieldpath}"


def _is_unescaped_index_selector(address: Address, index: int) -> bool:
    if index in address.encoded_segments:
        return False
    segment = address.segments[index]
    indexed = parse_indexed_segment(segment)
    return isinstance(indexed, IndexedSegment) and (
        index == 2
        and len(address.segments) == 3
        and address.segments[0] == "skeleton"
        and indexed.collection in ("flesh", "arrays")
        or index == 1
        and len(address.segments) == 2
        and address.segments[0] == "mounts"
        and indexed.collection == "mounts"
    )


def _render_fieldpath(fieldpath: tuple[FieldToken, ...]) -> str:
    return "".join(
        _render_field_token(token, index == 0)
        for index, token in enumerate(fieldpath)
    )


def _render_field_token(token: FieldToken, first: bool) -> str:
    match token:
        case FieldName(name):
            return name if first else f".{name}"
        case FieldIndex(index):
            return f"[{index}]"
        case _ as unreachable:
            assert_never(unreachable)


def field_tokens(address: Address) -> tuple[str | int, ...]:
    return tuple(_raw_token(token) for token in address.fieldpath)


def _raw_token(token: FieldToken) -> str | int:
    match token:
        case FieldName(name):
            return name
        case FieldIndex(index):
            return index
        case _ as unreachable:
            assert_never(unreachable)
