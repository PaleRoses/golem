"""Assembly obstruction vocabulary and element-fit law enums (closed carriers)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING
from golem.assembly.service import (AmbientHeatTransferPolicy, ElementAddress, RegionAddress, SemanticAddress, SolidMaterialRole)
from golem.conduits.surface.types import EmissionPitchObstruction

if TYPE_CHECKING:
    from golem.materials.surface import (
        AppearancePaletteObstruction,
        SurfaceColorObstruction,
    )
    from golem.plates.core import PlatePolicyObstruction, PlateSurfaceObstruction
    from golem.kernel.body.relations import BodyRelationObstruction
    from golem.kernel.body.types import BodyObstruction
    from golem.kernel.engine.types import (
        SurfaceDetailObstruction,
        SurfaceFormationObstruction,
    )

@dataclass(frozen=True)
class MalformedAssemblyObstruction:
    address: str
    reason: str


@dataclass(frozen=True)
class UnknownElementRoleObstruction:
    element_id: str
    role: object


@dataclass(frozen=True)
class EquipmentBodyDialectObstruction:
    element_id: str


@dataclass(frozen=True)
class DuplicateElementIdObstruction:
    element_id: str


@dataclass(frozen=True)
class ElementSourceObstruction:
    element_id: str
    reason: str

@dataclass(frozen=True)
class ElementBodyObstruction:
    element_id: str
    obstructions: tuple["BodyObstruction", ...]


@dataclass(frozen=True)
class UnknownAppendageIntentObstruction:
    element_id: str
    appendage_id: str
    intent_id: str
    closest_valid_candidates: tuple[str, ...] | None = None


@dataclass(frozen=True)
class InvalidResolutionPolicyObstruction:
    policy: object
    reason: str


@dataclass(frozen=True)
class MissingPhysicalVasculatureObstruction:
    element_id: str


@dataclass(frozen=True)
class SolidIntegrityObstruction:
    element_id: str
    component_count: int
    watertight: bool
    vocabulary_violation_count: int
    violations: tuple[Mapping[str, object], ...] = ()


class ElementFitLaw(StrEnum):
    DISJOINT = "disjoint"
    SURFACE_CLEARANCE = "surface_clearance"
    BOUNDED_MOUNT_CONTACT = "bounded_mount_contact"


@dataclass(frozen=True)
class ElementFitObstruction:
    left_element_id: str
    right_element_id: str
    fit_law: ElementFitLaw
    penetrating_sample_count: int
    outside_contact_sample_count: int
    maximum_penetration_world: float
    tolerance_world: float
    contact_element_id: str | None
    contact_radius_world: float | None


@dataclass(frozen=True)
class ElementClearanceObstruction:
    left_element_id: str
    right_element_id: str
    fit_law: ElementFitLaw
    declaring_element_id: str
    minimum_clearance_world: float
    maximum_clearance_world: float
    tolerance_world: float
    fitted_surface_fraction: float | None
    required_fitted_surface_fraction: float | None


@dataclass(frozen=True)
class EmptySurfaceConduitObstruction:
    record_id: str


@dataclass(frozen=True)
class PlateElementObstruction:
    element_id: str
    obstructions: tuple["PlatePolicyObstruction | PlateSurfaceObstruction", ...]


@dataclass(frozen=True)
class SurfaceColorElementObstruction:
    element_id: str
    obstructions: tuple["SurfaceColorObstruction", ...]


@dataclass(frozen=True)
class AppearancePaletteElementObstruction:
    element_id: str
    obstructions: tuple[
        "AppearancePaletteObstruction | SurfaceColorObstruction",
        ...,
    ]


@dataclass(frozen=True)
class ElementSurfaceDetailObstruction:
    element_id: str
    obstructions: tuple["SurfaceDetailObstruction", ...]


@dataclass(frozen=True)
class MaskedIntegrityEvidence:
    """Mesh-derived integrity evidence a surface-formation rejection made
    uncomputable: no surface, no mesh, no coherence report. Names the
    withheld checks and why; never fakes their values (Wall 005)."""
    checks: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ElementSurfaceFormationObstruction:
    element_id: str
    obstructions: tuple["SurfaceFormationObstruction", ...]
    vocabulary_violations: tuple[Mapping[str, object], ...] = ()
    masked_integrity: MaskedIntegrityEvidence | None = None


@dataclass(frozen=True)
class MissingSolidMaterialAssignmentObstruction:
    element_id: str
    expected_roles: tuple[SolidMaterialRole, ...]


@dataclass(frozen=True)
class MissingVascularWallMaterialObstruction:
    element_id: str


@dataclass(frozen=True)
class PumpPowerPressureLimitObstruction:
    required_pressure_pascal: float
    allowable_pressure_pascal: float


@dataclass(frozen=True)
class EmptyPhysicalSolidDomainObstruction:
    element_id: str


@dataclass(frozen=True)
class UnsupportedAssemblyInterfaceObstruction:
    element_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class UnsupportedThermalAddressObstruction:
    address: ElementAddress | RegionAddress


@dataclass(frozen=True)
class EmptyThermalRegionObstruction:
    address: RegionAddress


@dataclass(frozen=True)
class UnderspecifiedAmbientHeatTransferObstruction:
    policy: AmbientHeatTransferPolicy
    missing_inputs: tuple[str, ...]


@dataclass(frozen=True)
class ThermalLimitObstruction:
    element_id: str
    maximum_temperature_kelvin: float
    limit_kelvin: float


@dataclass(frozen=True)
class ThermalGradientLimitObstruction:
    element_id: str
    maximum_gradient_kelvin_per_metre: float
    limit_kelvin_per_metre: float


@dataclass(frozen=True)
class CoolantTemperatureValidityObstruction:
    node_id: str
    temperature_kelvin: float
    minimum_kelvin: float
    maximum_kelvin: float


@dataclass(frozen=True)
class UnsupportedMechanicalAddressObstruction:
    address: SemanticAddress
    reason: str


@dataclass(frozen=True)
class UnsupportedStructuralRoutingMountObstruction:
    element_id: str
    reason: str


@dataclass(frozen=True)
class MissingMechanicalSupportObstruction:
    case_id: str


@dataclass(frozen=True)
class InsufficientBurstMarginObstruction:
    element_id: str
    actual_safety_factor: float
    required_safety_factor: float


@dataclass(frozen=True)
class RemainingLigamentViolationObstruction:
    element_id: str
    minimum_ligament_metres: float
    required_ligament_metres: float


@dataclass(frozen=True)
class MissingRigidPayloadInterfaceObstruction:
    element_id: str


@dataclass(frozen=True)
class RigidPayloadInterfaceGapObstruction:
    element_id: str
    gap_metres: float
    permitted_gap_metres: float


@dataclass(frozen=True)
class RigidPayloadBalanceObstruction:
    element_id: str
    force_residual_newtons: float
    moment_residual_newton_metres: float
    tolerance: float


@dataclass(frozen=True)
class ThermalCouplingNonConvergenceObstruction:
    element_id: str
    case_id: str
    iterations: int
    maximum_iterations: int
    observed_state_delta: float
    maximum_state_delta: float


@dataclass(frozen=True)
class ResolutionRefinementObstruction:
    element_id: str
    case_id: str
    quantity: str
    normalized_change: float
    permitted_change: float


@dataclass(frozen=True)
class InsufficientVascularThermalCapacityObstruction:
    element_id: str
    required_volumetric_flow_cubic_metres_per_second: float
    achieved_volumetric_flow_cubic_metres_per_second: float


type AssemblyInputObstruction = (
    MalformedAssemblyObstruction
    | UnknownElementRoleObstruction
    | EquipmentBodyDialectObstruction
    | DuplicateElementIdObstruction
    | ElementSourceObstruction
    | InvalidResolutionPolicyObstruction
)


type LocalAssemblyObstruction = (
    AssemblyInputObstruction
    | UnknownAppendageIntentObstruction
    | MissingPhysicalVasculatureObstruction
    | SolidIntegrityObstruction
    | ElementFitObstruction
    | ElementClearanceObstruction
    | EmptySurfaceConduitObstruction
    | PlateElementObstruction
    | SurfaceColorElementObstruction
    | AppearancePaletteElementObstruction
    | ElementSurfaceDetailObstruction
    | ElementSurfaceFormationObstruction
    | MissingSolidMaterialAssignmentObstruction
    | MissingVascularWallMaterialObstruction
    | PumpPowerPressureLimitObstruction
    | EmptyPhysicalSolidDomainObstruction
    | UnsupportedAssemblyInterfaceObstruction
    | UnsupportedThermalAddressObstruction
    | EmptyThermalRegionObstruction
    | UnderspecifiedAmbientHeatTransferObstruction
    | ThermalLimitObstruction
    | ThermalGradientLimitObstruction
    | CoolantTemperatureValidityObstruction
    | UnsupportedMechanicalAddressObstruction
    | UnsupportedStructuralRoutingMountObstruction
    | MissingMechanicalSupportObstruction
    | InsufficientBurstMarginObstruction
    | RemainingLigamentViolationObstruction
    | MissingRigidPayloadInterfaceObstruction
    | RigidPayloadInterfaceGapObstruction
    | RigidPayloadBalanceObstruction
    | ThermalCouplingNonConvergenceObstruction
    | ResolutionRefinementObstruction
    | InsufficientVascularThermalCapacityObstruction
    | EmissionPitchObstruction
)
