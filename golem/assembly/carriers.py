"""Assembly data carriers: records, receipts, accepted/rejected results, compiled elements."""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from enum import StrEnum
from golem.assembly.service import (DecodedServiceIntent, PhysicalEvidenceStatus, ServiceIntent, ServiceIntentObstruction)
from golem.kernel import engine as E
from golem.kernel.anatomy import (AcceptedPhysicalHydraulics, AcceptedVasculature, AnatomyObstruction, AnatomyResult, ClosedVascularGraph, PhysicalHydraulicsObstruction, PhysicalHydraulicsReceipt, VascularMaterialObstruction, VasculatureObstruction, VasculatureResult)
from golem.kernel.body.types import EyeGlobe
from golem.kernel.mechanics import (AcceptedEmbeddedChannelMechanics, AcceptedMechanics, EmbeddedChannelObstruction, MechanicsObstruction, MechanicsReceipt)
from golem.kernel.sheaf import (HeatTransferCorrelationKind, NaturalConvectionHeatTransferCoefficient, Obstruction as SheafObstruction, ThermalEnergyReceipt, ThermalSolution)
from golem.materials.surface import SeededAppearancePalette, SeededSurfaceColor
from typing import Mapping

from golem.assembly.freeze import (_freeze_value, _frozen_report, _read_only_array)
from golem.assembly.obstructions import (
    ElementBodyObstruction,
    ElementFitLaw,
    LocalAssemblyObstruction,
)

RES_MIN, RES_MAX = 60, 260


class ElementRole(StrEnum):
    CREATURE = "creature"
    EQUIPMENT = "equipment"


class AssemblyStratum(StrEnum):
    SOLID = "solid"
    EYE = "eye"
    CONDUIT = "conduit"
    PLATE_SEAM = "plate_seam"
    VASCULAR_SUPPLY = "vascular_supply"
    VASCULAR_RETURN = "vascular_return"
    VASCULAR_EXCHANGE = "vascular_exchange"


ROLES = tuple(ElementRole)


@dataclass(frozen=True)
class PinnedResolution:
    resolution: int


@dataclass(frozen=True)
class TargetPitch:
    pitch: float


type AssemblyResolutionPolicy = PinnedResolution | TargetPitch


type AssemblyObstruction = (
    ElementBodyObstruction
    | LocalAssemblyObstruction
    | ServiceIntentObstruction
    | AnatomyObstruction
    | VasculatureObstruction
    | VascularMaterialObstruction
    | PhysicalHydraulicsObstruction
    | MechanicsObstruction
    | EmbeddedChannelObstruction
    | SheafObstruction
)


@dataclass(frozen=True)
class AssemblyRecord:
    """One immutable derived scene stratum owned by an assembly element."""

    record_id: str
    element_id: str
    role: ElementRole
    stratum: AssemblyStratum
    appearance_material: str
    vertices: np.ndarray | None
    faces: np.ndarray | None
    resolution: int
    report: Mapping[str, object]
    violations: tuple[Mapping[str, object], ...]
    empty: bool = False
    surface_color: SeededSurfaceColor | None = None

    @classmethod
    def from_derived_geometry(
        cls,
        *,
        record_id: str,
        element_id: str,
        role: ElementRole | str,
        stratum: AssemblyStratum | str,
        appearance_material: str,
        vertices: object,
        faces: object,
        resolution: int,
        report: dict[str, object],
        violations: object = (),
        empty: bool = False,
        surface_color: SeededSurfaceColor | None = None,
    ) -> "AssemblyRecord":
        return cls(
            record_id=record_id,
            element_id=element_id,
            role=ElementRole(role),
            stratum=AssemblyStratum(stratum),
            appearance_material=appearance_material,
            vertices=_read_only_array(vertices),
            faces=_read_only_array(faces),
            resolution=resolution,
            report=_frozen_report(report),
            violations=tuple(map(_freeze_value, violations)),
            empty=empty,
            surface_color=surface_color,
        )


@dataclass(frozen=True)
class VisualAssemblyReceipt:
    physical_evidence: PhysicalEvidenceStatus = PhysicalEvidenceStatus.NOT_REQUESTED
    fit_receipts: tuple["ElementFitReceipt", ...] = ()


@dataclass(frozen=True)
class CoupledPerformanceReceipt:
    """Authoritative summaries only; derived render records never enter here."""

    physical_evidence: PhysicalEvidenceStatus
    metres_per_world_unit: float
    service_case_ids: tuple[str, ...]
    material_ids: tuple[str, ...]
    material_volumes: tuple["ElementMaterialVolumeReceipt", ...]
    vascular_sizing_receipts: tuple["VascularThermalSizingReceipt", ...]
    hydraulic_receipts: tuple[PhysicalHydraulicsReceipt, ...]
    thermal_receipts: tuple["ElementThermalReceipt", ...]
    mechanics_receipts: tuple["ElementMechanicsReceipt", ...]
    interface_receipts: tuple["RigidPayloadInterfaceReceipt", ...]
    refinement_receipts: tuple["ResolutionRefinementReceipt", ...]
    not_evaluated: tuple["CoupledUnevaluatedPhysics", ...]
    fixed_point_iterations: int
    maximum_state_delta: float
    fit_receipts: tuple["ElementFitReceipt", ...] = ()


type AssemblyReceipt = VisualAssemblyReceipt | CoupledPerformanceReceipt


@dataclass(frozen=True)
class AcceptedAssembly:
    records: tuple[AssemblyRecord, ...]
    service: DecodedServiceIntent
    receipt: AssemblyReceipt
    vasculature: tuple[AcceptedVasculature, ...] = ()
    hydraulics: tuple[AcceptedPhysicalHydraulics, ...] = ()
    thermal_solutions: tuple[ThermalSolution, ...] = ()
    mechanics_solutions: tuple[AcceptedMechanics, ...] = ()


@dataclass(frozen=True)
class ElementFitReceipt:
    left_element_id: str
    right_element_id: str
    fit_law: ElementFitLaw
    declaring_element_id: str | None
    minimum_clearance_world: float | None
    permitted_clearance_world: float | None
    maximum_penetration_world: float
    tolerance_world: float
    penetrating_sample_count: int
    fitted_surface_fraction: float | None
    required_fitted_surface_fraction: float | None


class CoupledUnevaluatedPhysics(StrEnum):
    EQUIPMENT_LOCAL_THERMAL_RESPONSE = "equipment_local_thermal_response"
    EQUIPMENT_LOCAL_DEFORMATION = "equipment_local_deformation"
    EQUIPMENT_LOCAL_STRESS = "equipment_local_stress"
    EQUIPMENT_CONTACT_STRESS = "equipment_contact_stress"
    SUBGRID_CHANNEL_GLOBAL_CONSTITUTIVE_RESPONSE = (
        "subgrid_channel_global_constitutive_response"
    )


@dataclass(frozen=True)
class ElementMaterialVolumeReceipt:
    element_id: str
    solid_fraction: float
    lumen_fraction: float
    wall_fraction: float
    minimum_ligament_metres: float | None


@dataclass(frozen=True)
class ElementThermalReceipt:
    element_id: str
    case_id: str
    maximum_temperature_kelvin: float
    mean_temperature_kelvin: float
    maximum_temperature_gradient_kelvin_per_metre: float
    coolant_outlet_temperature_kelvin: float | None
    hottest_semantic_address: str
    heat_transfer_correlations: tuple[HeatTransferCorrelationKind, ...]
    natural_convection: NaturalConvectionHeatTransferCoefficient | None
    energy: ThermalEnergyReceipt


@dataclass(frozen=True)
class ElementMechanicsReceipt:
    element_id: str
    case_id: str
    mechanics: MechanicsReceipt
    minimum_burst_safety_factor: float | None
    critical_semantic_address: str
    embedded_channel_mechanics: AcceptedEmbeddedChannelMechanics | None = None


@dataclass(frozen=True)
class RigidPayloadInterfaceReceipt:
    element_id: str
    case_id: str
    mass_kilograms: float
    center_of_mass_metres: tuple[float, float, float]
    attachment_point_metres: tuple[float, float, float]
    interface_gap_metres: float
    transmitted_force_newtons: tuple[float, float, float]
    transmitted_moment_newton_metres: tuple[float, float, float]
    force_balance_residual_newtons: float
    moment_balance_residual_newton_metres: float


@dataclass(frozen=True)
class ResolutionRefinementReceipt:
    element_id: str
    case_id: str
    base_resolution: int
    refined_resolution: int
    maximum_temperature_utilization_change: float | None
    maximum_displacement_utilization_change: float
    minimum_yield_utilization_change: float | None


@dataclass(frozen=True)
class VascularThermalSizingReceipt:
    element_id: str
    heat_load_watts: float
    maximum_total_temperature_rise_kelvin: float
    reserved_thermal_approach_kelvin: float
    allowable_bulk_coolant_rise_kelvin: float
    required_volumetric_flow_cubic_metres_per_second: float
    baseline_volumetric_flow_cubic_metres_per_second: float
    terminal_radius_scale: float
    achieved_volumetric_flow_cubic_metres_per_second: float


_SEALED_THERMAL_APPROACH_RESERVE_FRACTION = 0.10


_SEALED_VASCULAR_SIZING_REFINEMENTS = 4


@dataclass(frozen=True)
class RejectedAssembly:
    obstructions: tuple[AssemblyObstruction, ...]


type AssemblyResult = AcceptedAssembly | RejectedAssembly


@dataclass(frozen=True)
class _LoadedElement:
    entry: dict
    source_graph: dict
    anatomy: AnatomyResult | None
    eye_globes: tuple[EyeGlobe, ...] = ()


@dataclass(frozen=True)
class _CompiledElement:
    element_id: str
    entry: dict
    source_graph: dict
    graph: dict
    records: tuple[AssemblyRecord, ...]
    evaluated: E.EvaluatedMorphology
    anatomy: AnatomyResult | None
    vasculature: VasculatureResult | None
    mounted_vasculature: ClosedVascularGraph | None
    role: ElementRole = ElementRole.CREATURE
    appearance_palette: SeededAppearancePalette | None = None


def _solid_record(element: _CompiledElement) -> AssemblyRecord:
    return next(
        record for record in element.records if record.stratum is AssemblyStratum.SOLID
    )


def _addresses(address: object, element_id: str) -> bool:
    return getattr(address, "element_id", None) == element_id


def _reference_temperature_kelvin(service: ServiceIntent) -> float:
    return (
        service.coolant.inlet_temperature_kelvin
        if service.coolant is not None
        else service.ambient.temperature_kelvin
        if service.ambient is not None
        else 293.15
    )
