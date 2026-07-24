"""Typed mechanics obstructions: the closed reject vocabulary of the solver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .model import BoundaryFace, CellIndex, NodeIndex, Vector3


@dataclass(frozen=True)
class MalformedDomainObstruction:
    address: str
    reason: str


@dataclass(frozen=True)
class DuplicateCellObstruction:
    cell: CellIndex


@dataclass(frozen=True)
class MissingCellPropertiesObstruction:
    cell: CellIndex


@dataclass(frozen=True)
class UnknownCellPropertiesObstruction:
    cell: CellIndex


@dataclass(frozen=True)
class DuplicateCellPropertiesObstruction:
    cell: CellIndex


@dataclass(frozen=True)
class InvalidConstitutivePropertiesObstruction:
    address: str
    cell: CellIndex
    property_name: str
    value: float


class MaterialFractionRule(StrEnum):
    ACTIVE_CELL_RANGE = "active_cell_range"
    FULL_SOLID_VALUE = "full_solid_value"
    RESOLUTION_TYPE = "resolution_type"


@dataclass(frozen=True)
class InvalidMaterialFractionObstruction:
    address: str
    cell: CellIndex
    rule: MaterialFractionRule
    solid_fraction: float
    fraction_resolution: str


@dataclass(frozen=True)
class UnresolvedMaterialFractionObstruction:
    address: str
    cell: CellIndex
    solid_fraction: float


@dataclass(frozen=True)
class InvalidSupportObstruction:
    node: NodeIndex
    reason: str


@dataclass(frozen=True)
class ConflictingSupportObstruction:
    node: NodeIndex
    component: int


@dataclass(frozen=True)
class InvalidNodalLoadObstruction:
    node: NodeIndex
    reason: str


@dataclass(frozen=True)
class InvalidBodyForceObstruction:
    address: str
    body_acceleration: Vector3


@dataclass(frozen=True)
class InvalidTemperatureFieldObstruction:
    node: NodeIndex
    reason: str


@dataclass(frozen=True)
class InvalidPressureBoundaryObstruction:
    cell: CellIndex
    face: BoundaryFace
    reason: str


@dataclass(frozen=True)
class InvalidAcceptanceCriteriaObstruction:
    criterion: str
    value: float
    reason: str


@dataclass(frozen=True)
class RigidBodyModeObstruction:
    reason: str


@dataclass(frozen=True)
class LinearSolverFailureObstruction:
    reason: str


@dataclass(frozen=True)
class NonFiniteMechanicsResultObstruction:
    quantity: str


@dataclass(frozen=True)
class SolverResidualObstruction:
    normalized_residual: float
    tolerance: float


@dataclass(frozen=True)
class ForceBalanceObstruction:
    relative_force_imbalance: float
    relative_moment_imbalance: float
    tolerance: float


@dataclass(frozen=True)
class SmallStrainLimitObstruction:
    maximum_absolute_principal_strain: float
    limit: float


@dataclass(frozen=True)
class ExcessiveDisplacementObstruction:
    maximum_displacement: float
    limit: float


@dataclass(frozen=True)
class InsufficientYieldMarginObstruction:
    minimum_yield_safety_factor: float | None
    required: float


@dataclass(frozen=True)
class AbsentBucklingSpectrumObstruction:
    reason: str


@dataclass(frozen=True)
class SingularBucklingSpectrumObstruction:
    reason: str


@dataclass(frozen=True)
class UnresolvedBucklingSpectrumObstruction:
    reason: str


@dataclass(frozen=True)
class InvalidBucklingEigenpairObstruction:
    load_factor: float
    normalized_eigen_residual: float
    maximum_normalized_eigen_residual: float


@dataclass(frozen=True)
class InsufficientBucklingMarginObstruction:
    lowest_positive_load_factor: float
    required_load_factor: float


type MechanicsObstruction = (
    MalformedDomainObstruction
    | DuplicateCellObstruction
    | MissingCellPropertiesObstruction
    | UnknownCellPropertiesObstruction
    | DuplicateCellPropertiesObstruction
    | InvalidConstitutivePropertiesObstruction
    | InvalidMaterialFractionObstruction
    | UnresolvedMaterialFractionObstruction
    | InvalidSupportObstruction
    | ConflictingSupportObstruction
    | InvalidNodalLoadObstruction
    | InvalidBodyForceObstruction
    | InvalidTemperatureFieldObstruction
    | InvalidPressureBoundaryObstruction
    | InvalidAcceptanceCriteriaObstruction
    | RigidBodyModeObstruction
    | LinearSolverFailureObstruction
    | NonFiniteMechanicsResultObstruction
    | SolverResidualObstruction
    | ForceBalanceObstruction
    | SmallStrainLimitObstruction
    | ExcessiveDisplacementObstruction
    | InsufficientYieldMarginObstruction
    | AbsentBucklingSpectrumObstruction
    | SingularBucklingSpectrumObstruction
    | UnresolvedBucklingSpectrumObstruction
    | InvalidBucklingEigenpairObstruction
    | InsufficientBucklingMarginObstruction
)
