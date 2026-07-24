"""Semantic-address carriers and their total grammar decode."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never


class SemanticAddressKind(StrEnum):
    ELEMENT = "element"
    REGION = "region"
    BONE = "bone"
    PORT = "port"
    CONTACT = "contact"
    MOUNT = "mount"


@dataclass(frozen=True)
class ElementAddress:
    element_id: str


@dataclass(frozen=True)
class RegionAddress:
    element_id: str
    region_id: str


@dataclass(frozen=True)
class BoneAddress:
    element_id: str
    bone_id: str


@dataclass(frozen=True)
class PortAddress:
    element_id: str
    port_id: str


@dataclass(frozen=True)
class ContactAddress:
    element_id: str
    contact_id: str


@dataclass(frozen=True)
class MountAddress:
    element_id: str


type SemanticAddress = (
    ElementAddress
    | RegionAddress
    | BoneAddress
    | PortAddress
    | ContactAddress
    | MountAddress
)


@dataclass(frozen=True)
class MalformedSemanticAddressObstruction:
    address: str
    value: object
    reason: str


@dataclass(frozen=True)
class AcceptedSemanticAddress:
    address: SemanticAddress


@dataclass(frozen=True)
class RejectedSemanticAddress:
    obstructions: tuple[MalformedSemanticAddressObstruction, ...]


type SemanticAddressResult = AcceptedSemanticAddress | RejectedSemanticAddress


def semantic_address_token(address: SemanticAddress) -> str:
    match address:
        case ElementAddress(element_id):
            return f"element:{element_id}"
        case RegionAddress(element_id, region_id):
            return f"region:{element_id}/{region_id}"
        case BoneAddress(element_id, bone_id):
            return f"bone:{element_id}/{bone_id}"
        case PortAddress(element_id, port_id):
            return f"port:{element_id}/{port_id}"
        case ContactAddress(element_id, contact_id):
            return f"contact:{element_id}/{contact_id}"
        case MountAddress(element_id):
            return f"mount:{element_id}"
        case _ as unreachable:
            assert_never(unreachable)


def semantic_address_kind(address: SemanticAddress) -> SemanticAddressKind:
    match address:
        case ElementAddress():
            return SemanticAddressKind.ELEMENT
        case RegionAddress():
            return SemanticAddressKind.REGION
        case BoneAddress():
            return SemanticAddressKind.BONE
        case PortAddress():
            return SemanticAddressKind.PORT
        case ContactAddress():
            return SemanticAddressKind.CONTACT
        case MountAddress():
            return SemanticAddressKind.MOUNT
        case _ as unreachable:
            assert_never(unreachable)


def decode_semantic_address(
    value: object, address: str = "/address"
) -> SemanticAddressResult:
    if not isinstance(value, str) or not value:
        return RejectedSemanticAddress(
            (
                MalformedSemanticAddressObstruction(
                    address, value, "expected a non-empty semantic-address token"
                ),
            )
        )
    prefix, separator, payload = value.partition(":")
    element_id, local_separator, local_id = payload.partition("/")
    malformed = (
        not separator
        or not element_id
        or (prefix in ("element", "mount") and bool(local_separator))
        or (prefix in ("region", "bone", "port", "contact") and not local_id)
    )
    if malformed:
        return RejectedSemanticAddress(
            (
                MalformedSemanticAddressObstruction(
                    address,
                    value,
                    (
                        "expected element:<id>, mount:<id>, or "
                        "<region|bone|port|contact>:<element>/<local-id>"
                    ),
                ),
            )
        )
    match prefix:
        case "element":
            return AcceptedSemanticAddress(ElementAddress(element_id))
        case "region":
            return AcceptedSemanticAddress(RegionAddress(element_id, local_id))
        case "bone":
            return AcceptedSemanticAddress(BoneAddress(element_id, local_id))
        case "port":
            return AcceptedSemanticAddress(PortAddress(element_id, local_id))
        case "contact":
            return AcceptedSemanticAddress(ContactAddress(element_id, local_id))
        case "mount":
            return AcceptedSemanticAddress(MountAddress(element_id))
        case _:
            return RejectedSemanticAddress(
                (
                    MalformedSemanticAddressObstruction(
                        address,
                        value,
                        f"unknown semantic-address kind {prefix!r}",
                    ),
                )
            )
