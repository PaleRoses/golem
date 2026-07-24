"""Immutable carriers for assertion scopes and assembly-shaped qualified tokens."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class Whole:
    pass


@dataclass(frozen=True)
class World:
    pass


@dataclass(frozen=True)
class Part:
    part_id: str


@dataclass(frozen=True)
class Bone:
    bone_id: str
    element_id: str | None = None


@dataclass(frozen=True)
class Landmark:
    name: str
    view: str | None = None


@dataclass(frozen=True)
class Chain:
    start: str
    end: str


@dataclass(frozen=True)
class Contact:
    contact_id: str
    element_id: str | None = None


@dataclass(frozen=True)
class Element:
    element_id: str


@dataclass(frozen=True)
class Port:
    element_id: str
    port_id: str


@dataclass(frozen=True)
class Mount:
    element_id: str


@dataclass(frozen=True)
class Region:
    element_id: str
    region_id: str


type AssertionScope = Whole | World | Part | Bone | Landmark | Chain | Contact
type QualifiedScope = Bone | Contact | Element | Port | Mount | Region
type Scope = AssertionScope | QualifiedScope


class ScopeKind(StrEnum):
    WHOLE = "whole"
    WORLD = "world"
    PART = "part"
    BONE = "bone"
    LANDMARK = "landmark"
    CHAIN = "chain"
    CONTACT = "contact"
    ELEMENT = "element"
    PORT = "port"
    MOUNT = "mount"
    REGION = "region"


@dataclass(frozen=True)
class ScopeObstruction:
    source: str
    reason: str


@dataclass(frozen=True)
class RejectedScope:
    obstructions: tuple[ScopeObstruction, ...]


type ScopeResult = Scope | RejectedScope
