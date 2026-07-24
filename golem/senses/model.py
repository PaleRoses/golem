"""Immutable typed carriers for the proprioceptive section."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from golem.kernel.engine.types import (
    GeometryObstruction,
    MuscleFormationObstruction,
)


class FusionBand(StrEnum):
    FUSED = "FUSED"
    BLEND = "BLEND"


@dataclass(frozen=True)
class GlobalDims:
    W: float
    H: float
    D: float
    HW: float
    centroid: NDArray[np.float64]
    k: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "centroid", immutable_array(self.centroid))


@dataclass(frozen=True)
class PartDatum:
    bbox_lo: NDArray[np.float64]
    bbox_hi: NDArray[np.float64]
    mirror: bool
    instances: tuple[str, ...]
    min_r: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox_lo", immutable_array(self.bbox_lo))
        object.__setattr__(self, "bbox_hi", immutable_array(self.bbox_hi))
        object.__setattr__(self, "instances", tuple(self.instances))


@dataclass(frozen=True)
class InstanceDatum:
    part: Mapping[str, object]
    mirrored: bool
    samples: NDArray[np.float64]
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]
    blend_r: float
    min_r: float
    pid: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "part", immutable_mapping(self.part))
        object.__setattr__(self, "samples", immutable_array(self.samples))
        object.__setattr__(self, "lo", immutable_array(self.lo))
        object.__setattr__(self, "hi", immutable_array(self.hi))


@dataclass(frozen=True)
class Band:
    ylo: float
    yhi: float
    width: float
    depth: float
    zmid: float


@dataclass(frozen=True)
class FusionRow:
    pb: str
    g: float
    band: FusionBand


@dataclass(frozen=True)
class Supported:
    hull_x: tuple[float, float]
    hull_z: tuple[float, float]
    centroid_xz: tuple[float, float]
    margin_x: float
    margin_z: float
    margin_z_fwd: float
    margin_z_aft: float
    inside: bool


@dataclass(frozen=True)
class NoSupport:
    pass


type Balance = Supported | NoSupport


type SensesObstruction = GeometryObstruction | MuscleFormationObstruction


@dataclass(frozen=True)
class RejectedSenses:
    obstructions: tuple[SensesObstruction, ...]


@dataclass(frozen=True)
class Senses:
    name: str
    txn: int
    n_parts: int
    n_mirror: int
    n_instances: int
    global_dims: GlobalDims
    raw_lo: NDArray[np.float64]
    raw_hi: NDArray[np.float64]
    parts: Mapping[str, PartDatum]
    inst_data: Mapping[str, InstanceDatum]
    inst_gaps: Mapping[tuple[str, str], float]
    declared_pair_gap: Mapping[tuple[str, str], float]
    fusion_rows: Mapping[str, tuple[FusionRow, ...]]
    cross_plane: Mapping[str, float]
    n_components: int
    fused_adj: Mapping[str, frozenset[str]]
    ground: Mapping[str, float]
    near_ground: Mapping[str, float]
    ground_decl: tuple[str, ...]
    ground_tol: float
    balance: Balance
    bands: tuple[Band, ...]
    landmarks: Mapping[str, tuple[float, ...]]
    schematic: Mapping[str, Mapping[str, tuple[float, ...]]]
    attach: tuple[tuple[str, str], ...]
    provenance: Mapping[str, str]
    anatomy: Mapping[str, object] | None
    graph: Mapping[str, object]
    expected_articulations: tuple[tuple[str, str], ...] = ()
    midline: tuple[str, ...] = ()
    roles: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_lo", immutable_array(self.raw_lo))
        object.__setattr__(self, "raw_hi", immutable_array(self.raw_hi))
        object.__setattr__(self, "parts", immutable_mapping(self.parts))
        object.__setattr__(self, "inst_data", immutable_mapping(self.inst_data))
        object.__setattr__(self, "inst_gaps", immutable_mapping(self.inst_gaps))
        object.__setattr__(
            self, "declared_pair_gap", immutable_mapping(self.declared_pair_gap)
        )
        object.__setattr__(self, "fusion_rows", immutable_mapping(self.fusion_rows))
        object.__setattr__(self, "cross_plane", immutable_mapping(self.cross_plane))
        object.__setattr__(self, "fused_adj", immutable_mapping(self.fused_adj))
        object.__setattr__(self, "ground", immutable_mapping(self.ground))
        object.__setattr__(self, "near_ground", immutable_mapping(self.near_ground))
        object.__setattr__(self, "ground_decl", tuple(self.ground_decl))
        object.__setattr__(self, "bands", tuple(self.bands))
        object.__setattr__(self, "landmarks", immutable_mapping(self.landmarks))
        object.__setattr__(self, "schematic", immutable_mapping(self.schematic))
        object.__setattr__(
            self,
            "attach",
            tuple(tuple(pair) for pair in self.attach),
        )
        object.__setattr__(self, "midline", tuple(map(str, self.midline)))
        object.__setattr__(
            self,
            "roles",
            immutable_mapping(
                {
                    str(role): tuple(map(str, declared))
                    for role, declared in self.roles.items()
                }
            ),
        )
        object.__setattr__(self, "provenance", immutable_mapping(self.provenance))
        object.__setattr__(
            self,
            "anatomy",
            immutable_mapping(self.anatomy) if self.anatomy is not None else None,
        )
        object.__setattr__(self, "graph", immutable_mapping(self.graph))


type SensesResult = Senses | RejectedSenses


def immutable_array[value: np.generic](
    value: NDArray[value],
) -> NDArray[value]:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def immutable_mapping[key, value](
    value: Mapping[key, value],
) -> Mapping[key, value]:
    return MappingProxyType(
        {
            member_key: immutable_value(member_value)
            for member_key, member_value in value.items()
        }
    )


def immutable_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return immutable_array(value)
    if isinstance(value, Mapping):
        return immutable_mapping(value)
    if isinstance(value, (tuple, list)):
        return tuple(map(immutable_value, value))
    if isinstance(value, (set, frozenset)):
        return frozenset(map(immutable_value, value))
    return value
