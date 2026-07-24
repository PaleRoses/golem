"""Service-intent value carriers -- enums and frozen dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from golem.materials import (
    CompressibleGas,
    IncompressibleFluid,
    IsotropicSolid,
)
from golem.kernel.sheaf import ChurchillChuIsothermalVerticalPlate

from golem.assembly.service.address import (
    BoneAddress,
    ContactAddress,
    ElementAddress,
    MountAddress,
    PortAddress,
    RegionAddress,
)


class PhysicalEvidenceStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    REQUESTED = "requested"


class SolidConstitutiveModel(StrEnum):
    ISOTROPIC_LINEAR_THERMOELASTIC = "isotropic_linear_thermoelastic"


class SolidMaterialRole(StrEnum):
    STRUCTURE = "structure"
    ARMOR = "armor"
    LUMEN_WALL = "lumen_wall"
    JOINT = "joint"
    EQUIPMENT = "equipment"


class SupportLaw(StrEnum):
    FIXED = "fixed"
    PINNED = "pinned"
    ROLLER = "roller"


class JointLaw(StrEnum):
    BONDED = "bonded"
    REVOLUTE = "revolute"
    SPHERICAL = "spherical"
    PRISMATIC = "prismatic"
    RIGID_PAYLOAD = "rigid_payload"


class AmbientHeatTransferPolicy(StrEnum):
    STILL_AIR_NATURAL_CONVECTION = "still_air_natural_convection"


class ChannelResolutionPolicy(StrEnum):
    RESOLVED_ONLY = "resolved_only"
    RESOLVED_OR_EMBEDDED_SLENDER = "resolved_or_embedded_slender"


class EvidenceObligation(StrEnum):
    MATERIAL_VALIDITY = "material_validity"
    LUMEN_RESOLUTION = "lumen_resolution"
    HYDRAULIC_RESIDUAL = "hydraulic_residual"
    THERMAL_RESIDUAL = "thermal_residual"
    MECHANICAL_RESIDUAL = "mechanical_residual"
    INTERFACE_BALANCE = "interface_balance"
    RESOLUTION_REFINEMENT = "resolution_refinement"


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float

    @property
    def values(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class SIScale:
    metres_per_world_unit: float


@dataclass(frozen=True)
class SolidMaterialAssignment:
    role: SolidMaterialRole
    domain: ElementAddress | RegionAddress
    material: IsotropicSolid


@dataclass(frozen=True)
class ForceLoad:
    address: BoneAddress | PortAddress | ContactAddress | MountAddress
    vector_newtons: Vector3


@dataclass(frozen=True)
class MomentLoad:
    address: BoneAddress | PortAddress | ContactAddress | MountAddress
    vector_newton_metres: Vector3


type AppliedLoad = ForceLoad | MomentLoad


@dataclass(frozen=True)
class Support:
    address: BoneAddress | PortAddress | ContactAddress | MountAddress
    law: SupportLaw


@dataclass(frozen=True)
class JointInterface:
    address: BoneAddress | PortAddress | ContactAddress | MountAddress
    law: JointLaw


@dataclass(frozen=True)
class HeatSource:
    address: ElementAddress | RegionAddress
    watts: float


@dataclass(frozen=True)
class AmbientCondition:
    temperature_kelvin: float
    heat_transfer_policy: AmbientHeatTransferPolicy
    gas: CompressibleGas
    pressure_pascal: float
    natural_convection_correlation: ChurchillChuIsothermalVerticalPlate


@dataclass(frozen=True)
class PressurePumpBudget:
    maximum_pressure_pascal: float


@dataclass(frozen=True)
class PowerPumpBudget:
    maximum_power_watts: float


type PumpBudget = PressurePumpBudget | PowerPumpBudget


@dataclass(frozen=True)
class CoolantService:
    material: IncompressibleFluid
    inlet_temperature_kelvin: float
    pump_budget: PumpBudget


@dataclass(frozen=True)
class ManufacturingLimits:
    minimum_lumen_diameter_metres: float
    minimum_wall_thickness_metres: float
    minimum_remaining_ligament_metres: float


@dataclass(frozen=True)
class OperatingLimits:
    maximum_temperature_kelvin: float
    maximum_temperature_gradient_kelvin_per_metre: float
    maximum_displacement_metres: float
    minimum_yield_safety_factor: float
    minimum_buckling_safety_factor: float
    minimum_burst_safety_factor: float


@dataclass(frozen=True)
class PhysicsEvidencePolicy:
    channel_resolution_policy: ChannelResolutionPolicy
    physics_resolution: int
    minimum_cells_across_lumen: int
    resolution_refinement_factor: int
    maximum_refinement_change_fraction: float

    @property
    def required_obligations(self) -> frozenset[EvidenceObligation]:
        return frozenset(EvidenceObligation)

    def maximum_physics_pitch_metres(
        self, manufacturing: ManufacturingLimits
    ) -> float:
        return (
            manufacturing.minimum_lumen_diameter_metres
            / self.minimum_cells_across_lumen
        )


@dataclass(frozen=True)
class ServiceCase:
    case_id: str
    gravity_metres_per_second_squared: Vector3
    inertial_acceleration_metres_per_second_squared: Vector3
    loads: tuple[AppliedLoad, ...]
    supports: tuple[Support, ...]
    joints: tuple[JointInterface, ...]
    heat_sources: tuple[HeatSource, ...]


@dataclass(frozen=True)
class VisualOnlyServiceIntent:
    @property
    def physical_evidence(self) -> PhysicalEvidenceStatus:
        return PhysicalEvidenceStatus.NOT_REQUESTED


@dataclass(frozen=True)
class ServiceIntent:
    scale: SIScale
    constitutive_model: SolidConstitutiveModel
    solid_materials: tuple[SolidMaterialAssignment, ...]
    cases: tuple[ServiceCase, ...]
    ambient: AmbientCondition | None
    coolant: CoolantService | None
    manufacturing: ManufacturingLimits
    limits: OperatingLimits
    evidence_policy: PhysicsEvidencePolicy

    @property
    def physical_evidence(self) -> PhysicalEvidenceStatus:
        return PhysicalEvidenceStatus.REQUESTED


type DecodedServiceIntent = VisualOnlyServiceIntent | ServiceIntent
