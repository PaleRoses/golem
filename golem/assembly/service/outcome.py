"""Service-intent obstruction ADT and decode results."""

from __future__ import annotations

from dataclasses import dataclass

from golem.materials import MaterialKind

from golem.assembly.service.address import (
    ElementAddress,
    MalformedSemanticAddressObstruction,
    RegionAddress,
    SemanticAddress,
    SemanticAddressKind,
)
from golem.assembly.service.model import (
    DecodedServiceIntent,
    SolidMaterialRole,
    SupportLaw,
)


@dataclass(frozen=True)
class MalformedServiceIntentObstruction:
    address: str
    reason: str


@dataclass(frozen=True)
class MissingUnitScaleObstruction:
    address: str = "/service/metres_per_world_unit"


@dataclass(frozen=True)
class UnknownSemanticAddressObstruction:
    address: str
    semantic_address: SemanticAddress


@dataclass(frozen=True)
class InvalidSemanticAddressKindObstruction:
    address: str
    semantic_address: SemanticAddress
    expected_kinds: tuple[SemanticAddressKind, ...]


@dataclass(frozen=True)
class ContradictorySupportObstruction:
    case_id: str
    semantic_address: SemanticAddress
    laws: tuple[SupportLaw, ...]


@dataclass(frozen=True)
class MissingHeatSinkObstruction:
    case_id: str


@dataclass(frozen=True)
class UnsupportedConstitutiveModelObstruction:
    actual_model: object


@dataclass(frozen=True)
class NonPositiveBudgetObstruction:
    address: str
    value: object


@dataclass(frozen=True)
class UnknownPhysicalMaterialObstruction:
    address: str
    material_kind: MaterialKind
    identifier: object


@dataclass(frozen=True)
class MaterialValidityObstruction:
    address: str
    material_id: str
    value: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class DuplicateServiceCaseObstruction:
    case_id: str


@dataclass(frozen=True)
class DuplicateMaterialAssignmentObstruction:
    role: SolidMaterialRole
    domain: ElementAddress | RegionAddress


type ServiceIntentObstruction = (
    MalformedServiceIntentObstruction
    | MalformedSemanticAddressObstruction
    | MissingUnitScaleObstruction
    | UnknownSemanticAddressObstruction
    | InvalidSemanticAddressKindObstruction
    | ContradictorySupportObstruction
    | MissingHeatSinkObstruction
    | UnsupportedConstitutiveModelObstruction
    | NonPositiveBudgetObstruction
    | UnknownPhysicalMaterialObstruction
    | MaterialValidityObstruction
    | DuplicateServiceCaseObstruction
    | DuplicateMaterialAssignmentObstruction
)


@dataclass(frozen=True)
class AcceptedServiceIntent:
    intent: DecodedServiceIntent


@dataclass(frozen=True)
class RejectedServiceIntent:
    obstructions: tuple[ServiceIntentObstruction, ...]


type ServiceIntentResult = AcceptedServiceIntent | RejectedServiceIntent
