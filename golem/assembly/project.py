from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING

from golem.addressing.session import Address, FieldIndex, FieldName
from golem.addressing.session_grammar import parse_address, render_address
from golem.assembly.carriers import (
    RES_MAX,
    RES_MIN,
    AssemblyObstruction,
    PinnedResolution,
    TargetPitch,
)
from golem.assembly.obstructions import (
    AppearancePaletteElementObstruction,
    CoolantTemperatureValidityObstruction,
    ElementBodyObstruction,
    ElementClearanceObstruction,
    ElementFitObstruction,
    ElementSurfaceDetailObstruction,
    ElementSurfaceFormationObstruction,
    InsufficientBurstMarginObstruction,
    InsufficientVascularThermalCapacityObstruction,
    InvalidResolutionPolicyObstruction,
    PlateElementObstruction,
    PumpPowerPressureLimitObstruction,
    RemainingLigamentViolationObstruction,
    ResolutionRefinementObstruction,
    RigidPayloadBalanceObstruction,
    RigidPayloadInterfaceGapObstruction,
    SolidIntegrityObstruction,
    SurfaceColorElementObstruction,
    ThermalCouplingNonConvergenceObstruction,
    ThermalGradientLimitObstruction,
    ThermalLimitObstruction,
)
from golem.assembly.service.outcome import (
    MaterialValidityObstruction,
    NonPositiveBudgetObstruction,
)
from golem.conduits.surface.types import EmissionPitchObstruction
from golem.kernel.anatomy.graph import (
    CapsuleEscapeObstruction,
    DemandDeliveryMismatchObstruction,
    DisconnectedVascularGraphObstruction,
    EmptyPerfusionTerritoryObstruction,
    InfeasibleBifurcationObstruction,
    InsufficientTerminalSitesObstruction,
    InvalidStructuralGuidanceObstruction,
    MissingProvenanceObstruction,
    ReversedVascularFlowObstruction,
    SymmetryMismatchObstruction,
    VascularGluingObstruction,
    VascularIntersectionObstruction,
    VascularSearchBudgetObstruction,
    VascularSolverResidualObstruction,
)
from golem.kernel.anatomy.hydraulics.types import (
    EmptyPhysicalHydraulicGraphObstruction,
    InvalidDynamicViscosityObstruction,
    InvalidFluidDensityObstruction,
    InvalidMetresPerWorldUnitObstruction,
    InvalidPhysicalHydraulicEdgeGeometryObstruction,
    InvalidPhysicalHydraulicConductanceObstruction,
    InvalidPhysicalHydraulicNodePositionObstruction,
    InvalidPumpPressureBoundaryObstruction,
    NonFinitePhysicalHydraulicResultObstruction,
    NonLaminarPhysicalHydraulicFlowObstruction,
    NonPositivePhysicalHydraulicFlowObstruction,
    NonPositivePhysicalPumpFlowObstruction,
    PhysicalHydraulicBalanceObstruction,
)
from golem.kernel.anatomy.material.types import (
    ChannelInducedDisconnectionObstruction,
    UnresolvedLumenObstruction,
    UnresolvedVascularWallObstruction,
    VascularMaterialEscapeObstruction,
)
from golem.kernel.anatomy.project import (
    project_anatomy_obstruction,
    project_vascular_obstruction,
)
from golem.kernel.anatomy.vocabulary import (
    DisconnectedAnatomyRegionObstruction,
    DuplicateAnatomyRegionObstruction,
    DuplicateExchangeBedObstruction,
    DuplicateIntegumentLayerObstruction,
    DuplicateMyotendinousUnitObstruction,
    DuplicateRegionHostObstruction,
    InsufficientVascularStagesObstruction,
    InvalidMyotendinousPathObstruction,
    InvalidSkeletonAddressObstruction,
    MalformedAnatomyObstruction,
    MissingIntegumentLayerObstruction,
    MissingSkeletalCrossSectionObstruction,
    MissingTerminalRegionObstruction,
    NonterminalExchangeRegionObstruction,
    UnknownAnatomyRegionObstruction,
    UnperfusedMyotendinousUnitObstruction,
    UnreferencedAnatomyRegionObstruction,
    UnsupportedTissueCarrierObstruction,
)
from golem.kernel.body.project import project_obstruction
from golem.kernel.body.types import BodyObstruction
from golem.kernel.engine.types import (
    SkinProjectionUnsatisfied,
    SkinRelaxationObstruction,
    SkinRootMultiplicity,
    SurfaceDetailObstruction,
)
from golem.kernel.mechanics.embedded_channel.model import (
    EmbeddedChannelScaleSeparationObstruction,
    EmbeddedChannelVolumeFractionObstruction,
    InsufficientEmbeddedChannelBurstMarginObstruction,
    UnsupportedEmbeddedChannelExternalPressureObstruction,
)
from golem.kernel.mechanics.obstructions import (
    ExcessiveDisplacementObstruction,
    ForceBalanceObstruction,
    InvalidAcceptanceCriteriaObstruction,
    InvalidBodyForceObstruction,
    InvalidBucklingEigenpairObstruction,
    InvalidConstitutivePropertiesObstruction,
    InvalidMaterialFractionObstruction,
    InsufficientBucklingMarginObstruction,
    InsufficientYieldMarginObstruction,
    SmallStrainLimitObstruction,
    SolverResidualObstruction,
    MaterialFractionRule,
    UnresolvedMaterialFractionObstruction,
)
from golem.kernel.mechanics.model import MaterialFractionResolution
from golem.kernel.sheaf.result import (
    BalanceImbalanceObstruction,
    BalanceResidualObstruction,
    InvalidBalanceCellObstruction,
    InvalidBalanceCoefficientObstruction,
    InvalidBalanceSolverConfigObstruction,
    InvalidThermalParameterObstruction,
    NonFiniteBalanceValueObstruction,
    UnsupportedHeatTransferRegimeObstruction,
)
from golem.kernel.sheaf.vocabulary import BalanceSolverParameter
from golem.materials.surface import (
    AppearancePaletteObstruction,
    SurfaceColorObstruction,
)
from golem.plates.core import PlatePolicyObstruction, PlateSurfaceObstruction

if TYPE_CHECKING:
    from golem.kernel.anatomy.graph import VasculatureObstruction
    from golem.kernel.anatomy.vocabulary import AnatomyObstruction

type NestedAuthoringObstruction = (
    PlatePolicyObstruction
    | PlateSurfaceObstruction
    | SurfaceColorObstruction
    | AppearancePaletteObstruction
    | SurfaceDetailObstruction
)


def project_json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: project_json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): project_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(map(project_json_value, value))
    return value


def _evidence(
    address: str,
    predicate: str,
    required: object,
    observed: object,
) -> dict[str, object]:
    return {
        "address": address,
        "predicate": predicate,
        "required": required,
        "observed": observed,
    }





def _crossing_slack_evidence(obstruction: object) -> dict[str, object]:
    match obstruction:
        case SkinRelaxationObstruction(
            failure=SkinProjectionUnsatisfied(
                failure=SkinRootMultiplicity() as multiplicity
            )
        ):
            return _evidence(
                "surface_formation",
                "each local flesh-to-scaffold projection segment crosses "
                "the skin offset field exactly once",
                "one crossing per local projection segment",
                {
                    "multiple_crossing_segments": len(
                        multiplicity.vertex_indices
                    ),
                    "minimum_crossing_count": multiplicity.minimum_root_count,
                    "maximum_crossing_count": multiplicity.maximum_root_count,
                    "crossing_distances_world": (
                        multiplicity.root_distances_world
                    ),
                },
            )
        case _:
            return {}


def _projection_address(projected: Mapping[str, object]) -> str:
    return next(
        (
            str(projected[field])
            for field in (
                "address",
                "owner_address",
                "subject_address",
                "interface_address",
                "element_id",
                "record_id",
                "node_id",
                "edge_id",
                "region_id",
                "part_id",
            )
            if projected.get(field) is not None
        ),
        "assembly",
    )


def _sheaf_evidence(
    obstruction: AssemblyObstruction,
) -> dict[str, object] | None:
    if isinstance(obstruction, InvalidBalanceCellObstruction):
        return _evidence(
            (
                f"{obstruction.problem_id}/{obstruction.term_kind.value}:"
                f"{obstruction.term_id}/cell:{int(obstruction.cell_id)}"
            ),
            "balance cell lies within the declared domain",
            {"minimum": 0, "maximum": obstruction.cell_count - 1},
            int(obstruction.cell_id),
        )
    if isinstance(obstruction, InvalidBalanceCoefficientObstruction):
        return _evidence(
            (
                f"{obstruction.problem_id}/{obstruction.term_kind.value}:"
                f"{obstruction.term_id}"
            ),
            "balance coefficient is finite and strictly positive",
            {"finite": True, "exclusive_minimum": 0.0},
            obstruction.coefficient,
        )
    if isinstance(obstruction, NonFiniteBalanceValueObstruction):
        return _evidence(
            (
                f"{obstruction.problem_id}/{obstruction.term_kind.value}:"
                f"{obstruction.term_id}"
            ),
            "authored balance value is finite",
            {"finite": True},
            obstruction.value,
        )
    if isinstance(obstruction, InvalidBalanceSolverConfigObstruction):
        positive_parameters = (
            BalanceSolverParameter.RELATIVE_TOLERANCE,
            BalanceSolverParameter.NORMALIZATION_FLOOR,
        )
        required = (
            {"integer": True, "exclusive_minimum": 0}
            if obstruction.parameter
            is BalanceSolverParameter.MAXIMUM_ITERATION_FACTOR
            else {
                "finite": True,
                **(
                    {"exclusive_minimum": 0.0}
                    if obstruction.parameter in positive_parameters
                    else {"minimum": 0.0}
                ),
            }
        )
        return _evidence(
            f"{obstruction.problem_id}/{obstruction.parameter.value}",
            "balance solver parameter lies within its lawful numeric domain",
            required,
            obstruction.value,
        )
    if isinstance(obstruction, InvalidThermalParameterObstruction):
        return _evidence(
            (
                f"{obstruction.problem_id}/{obstruction.term_id}/"
                f"{obstruction.parameter.value}"
            ),
            "thermal parameter is finite and strictly positive",
            {"finite": True, "exclusive_minimum": 0.0},
            obstruction.value,
        )
    if isinstance(obstruction, BalanceResidualObstruction):
        return _evidence(
            f"{obstruction.problem_id}/cell:{int(obstruction.cell_id)}",
            "balance residual does not exceed tolerance",
            obstruction.tolerance,
            obstruction.magnitude,
        )
    if isinstance(obstruction, BalanceImbalanceObstruction):
        return _evidence(
            str(obstruction.problem_id),
            "balance imbalance does not exceed tolerance",
            obstruction.tolerance,
            obstruction.magnitude,
        )
    if isinstance(obstruction, UnsupportedHeatTransferRegimeObstruction):
        return _evidence(
            str(obstruction.correlation_id),
            "heat-transfer regime value lies within its validity interval",
            (obstruction.minimum, obstruction.maximum),
            obstruction.actual,
        )
    return None


def _constitutive_requirement(property_name: str) -> object:
    match property_name:
        case "young_modulus" | "yield_strength":
            return {"finite": True, "exclusive_minimum": 0.0}
        case "poisson_ratio":
            return {
                "finite": True,
                "exclusive_minimum": -1.0,
                "exclusive_maximum": 0.5,
            }
        case "density":
            return {"finite": True, "minimum": 0.0}
        case _:
            return {"finite": True}


def _semantic_projection(
    obstruction: AssemblyObstruction,
    projected: Mapping[str, object],
) -> dict[str, object]:
    if (sheaf_evidence := _sheaf_evidence(obstruction)) is not None:
        return sheaf_evidence
    if isinstance(obstruction, ElementFitObstruction):
        return _evidence(
            (
                f"assembly/{obstruction.left_element_id}"
                f"~{obstruction.right_element_id}"
            ),
            "no penetrating samples lie outside the declared contact domain",
            0,
            obstruction.outside_contact_sample_count,
        )
    if isinstance(obstruction, ElementClearanceObstruction):
        return _evidence(
            (
                f"assembly/{obstruction.left_element_id}"
                f"~{obstruction.right_element_id}"
            ),
            "clearance and fitted-surface constraints are satisfied",
            {
                "maximum_minimum_clearance_world": (
                    obstruction.maximum_clearance_world
                    + obstruction.tolerance_world
                ),
                "minimum_fitted_surface_fraction": (
                    obstruction.required_fitted_surface_fraction
                ),
            },
            {
                "minimum_clearance_world": obstruction.minimum_clearance_world,
                "fitted_surface_fraction": obstruction.fitted_surface_fraction,
            },
        )
    if isinstance(obstruction, SolidIntegrityObstruction):
        return _evidence(
            obstruction.element_id,
            "solid is one watertight component with no vocabulary violations",
            {
                "component_count": 1,
                "watertight": True,
                "vocabulary_violation_count": 0,
            },
            {
                "component_count": obstruction.component_count,
                "watertight": obstruction.watertight,
                "vocabulary_violation_count": (
                    obstruction.vocabulary_violation_count
                ),
                "violations": obstruction.violations,
            },
        )
    if isinstance(obstruction, EmissionPitchObstruction):
        return _evidence(
            obstruction.address,
            "surface-conduit displacement respects the pitch bound",
            obstruction.maximum_displacement,
            obstruction.displacement,
        )
    if isinstance(obstruction, InvalidResolutionPolicyObstruction):
        match obstruction.policy:
            case PinnedResolution(resolution):
                address = "assembly/resolution"
                required = (RES_MIN, RES_MAX)
                observed = resolution
                predicate = "pinned assembly resolution lies within its interval"
            case TargetPitch(pitch):
                address = "meta@pitch"
                required = 0.0
                observed = pitch
                predicate = "target assembly pitch is finite and strictly positive"
            case authored:
                address = "meta@pitch"
                required = "PinnedResolution or TargetPitch"
                observed = authored
                predicate = "assembly resolution policy has a supported form"
        return _evidence(
            address,
            predicate,
            required,
            observed,
        )
    if isinstance(obstruction, ThermalLimitObstruction):
        return _evidence(
            obstruction.element_id,
            "maximum element temperature does not exceed its limit",
            obstruction.limit_kelvin,
            obstruction.maximum_temperature_kelvin,
        )
    if isinstance(obstruction, ThermalGradientLimitObstruction):
        return _evidence(
            obstruction.element_id,
            "maximum thermal gradient does not exceed its limit",
            obstruction.limit_kelvin_per_metre,
            obstruction.maximum_gradient_kelvin_per_metre,
        )
    if isinstance(obstruction, ThermalCouplingNonConvergenceObstruction):
        return _evidence(
            f"assembly/{obstruction.element_id}/{obstruction.case_id}/thermal",
            "thermal coupling converges within its iteration and state-delta gates",
            {
                "converged": True,
                "maximum_iterations": obstruction.maximum_iterations,
                "maximum_state_delta": obstruction.maximum_state_delta,
            },
            {
                "converged": False,
                "iterations": obstruction.iterations,
                "state_delta": obstruction.observed_state_delta,
            },
        )
    if isinstance(obstruction, CoolantTemperatureValidityObstruction):
        return _evidence(
            obstruction.node_id,
            "coolant temperature lies within its valid interval",
            (obstruction.minimum_kelvin, obstruction.maximum_kelvin),
            obstruction.temperature_kelvin,
        )
    if isinstance(obstruction, PumpPowerPressureLimitObstruction):
        return _evidence(
            "assembly",
            "required pump pressure does not exceed allowable pressure",
            obstruction.allowable_pressure_pascal,
            obstruction.required_pressure_pascal,
        )
    if isinstance(obstruction, InsufficientBurstMarginObstruction):
        return _evidence(
            obstruction.element_id,
            "burst safety factor meets the declared minimum",
            obstruction.required_safety_factor,
            obstruction.actual_safety_factor,
        )
    if isinstance(obstruction, RemainingLigamentViolationObstruction):
        return _evidence(
            obstruction.element_id,
            "remaining ligament meets the declared minimum",
            obstruction.required_ligament_metres,
            obstruction.minimum_ligament_metres,
        )
    if isinstance(obstruction, RigidPayloadInterfaceGapObstruction):
        return _evidence(
            obstruction.element_id,
            "rigid payload interface gap does not exceed its limit",
            obstruction.permitted_gap_metres,
            obstruction.gap_metres,
        )
    if isinstance(obstruction, RigidPayloadBalanceObstruction):
        return _evidence(
            obstruction.element_id,
            "rigid payload force and moment residuals meet tolerance",
            obstruction.tolerance,
            {
                "force_residual_newtons": obstruction.force_residual_newtons,
                "moment_residual_newton_metres": (
                    obstruction.moment_residual_newton_metres
                ),
            },
        )
    if isinstance(obstruction, ResolutionRefinementObstruction):
        return _evidence(
            obstruction.element_id,
            "resolution refinement change does not exceed its limit",
            obstruction.permitted_change,
            obstruction.normalized_change,
        )
    if isinstance(obstruction, InsufficientVascularThermalCapacityObstruction):
        return _evidence(
            obstruction.element_id,
            "vascular thermal capacity meets required volumetric flow",
            obstruction.required_volumetric_flow_cubic_metres_per_second,
            obstruction.achieved_volumetric_flow_cubic_metres_per_second,
        )
    if isinstance(obstruction, MaterialValidityObstruction):
        return _evidence(
            obstruction.address,
            "physical material value lies within its valid interval",
            (obstruction.minimum, obstruction.maximum),
            obstruction.value,
        )
    if isinstance(obstruction, NonPositiveBudgetObstruction):
        return _evidence(
            obstruction.address,
            "service budget is strictly positive",
            0.0,
            obstruction.value,
        )
    if isinstance(obstruction, UnresolvedLumenObstruction):
        return _evidence(
            obstruction.edge_id,
            "vascular lumen radius resolves across the required grid cells",
            obstruction.required_cells,
            (
                obstruction.radius / obstruction.pitch
                if obstruction.pitch != 0.0
                else None
            ),
        )
    if isinstance(obstruction, UnresolvedVascularWallObstruction):
        return _evidence(
            "anatomy/vasculature/wall",
            "vascular wall thickness resolves across the required grid cells",
            obstruction.required_cells,
            (
                obstruction.wall_thickness / obstruction.pitch
                if obstruction.pitch != 0.0
                else None
            ),
        )
    if isinstance(obstruction, VascularMaterialEscapeObstruction):
        return _evidence(
            "anatomy/vasculature",
            "vascular material remains inside the body domain",
            0,
            obstruction.leaking_cell_count,
        )
    if isinstance(obstruction, ChannelInducedDisconnectionObstruction):
        return _evidence(
            "anatomy/vasculature",
            "channel carving preserves one connected load-bearing component",
            1,
            obstruction.component_count,
        )
    if isinstance(obstruction, InvalidMetresPerWorldUnitObstruction):
        return _evidence(
            "assembly/hydraulics/unit_scale",
            "metres per world unit is finite and strictly positive",
            0.0,
            obstruction.metres_per_world_unit,
        )
    if isinstance(obstruction, InvalidFluidDensityObstruction):
        return _evidence(
            "assembly/hydraulics/fluid_density",
            "fluid density is finite and strictly positive",
            0.0,
            obstruction.density_kilograms_per_cubic_metre,
        )
    if isinstance(obstruction, InvalidDynamicViscosityObstruction):
        return _evidence(
            "assembly/hydraulics/dynamic_viscosity",
            "dynamic viscosity is finite and strictly positive",
            0.0,
            obstruction.dynamic_viscosity_pascal_second,
        )
    if isinstance(obstruction, InvalidPumpPressureBoundaryObstruction):
        return _evidence(
            "assembly/pump",
            "pump pressure drop is finite and strictly positive",
            0.0,
            (
                obstruction.outlet_pressure_pascal
                - obstruction.inlet_pressure_pascal
            ),
        )
    if isinstance(obstruction, EmptyPhysicalHydraulicGraphObstruction):
        return _evidence(
            "assembly/hydraulics",
            "physical hydraulic graph has nodes and edges",
            {"node_count": 1, "edge_count": 1},
            {
                "node_count": obstruction.node_count,
                "edge_count": obstruction.edge_count,
            },
        )
    if isinstance(obstruction, InvalidPhysicalHydraulicNodePositionObstruction):
        return _evidence(
            obstruction.node_id,
            "physical hydraulic node position is finite in world and SI coordinates",
            {
                "world_coordinates": {"coordinate_count": 3, "finite": True},
                "metre_coordinates": {"coordinate_count": 3, "finite": True},
            },
            {
                "world_coordinates": obstruction.position,
                "metre_coordinates": tuple(
                    coordinate * obstruction.metres_per_world_unit
                    for coordinate in obstruction.position
                ),
                "metres_per_world_unit": obstruction.metres_per_world_unit,
            },
        )
    if isinstance(obstruction, InvalidPhysicalHydraulicConductanceObstruction):
        return _evidence(
            obstruction.edge_id,
            "physical hydraulic conductance is finite and strictly positive",
            0.0,
            obstruction.conductance_cubic_metres_per_pascal_second,
        )
    if isinstance(obstruction, InvalidPhysicalHydraulicEdgeGeometryObstruction):
        return _evidence(
            obstruction.edge_id,
            "physical hydraulic edge radius and length are strictly positive",
            {
                "radius_world_units": 0.0,
                "length_world_units": 0.0,
            },
            {
                "radius_world_units": obstruction.radius_world_units,
                "length_world_units": obstruction.length_world_units,
            },
        )
    if isinstance(obstruction, NonPositivePhysicalHydraulicFlowObstruction):
        return _evidence(
            obstruction.edge_id,
            "physical hydraulic edge flow is strictly positive",
            0.0,
            obstruction.volumetric_flow_cubic_metres_per_second,
        )
    if isinstance(obstruction, NonPositivePhysicalPumpFlowObstruction):
        return _evidence(
            "assembly/pump",
            "physical pump inlet and outlet flows are strictly positive",
            {
                "outlet_flow_cubic_metres_per_second": 0.0,
                "inlet_flow_cubic_metres_per_second": 0.0,
            },
            {
                "outlet_flow_cubic_metres_per_second": (
                    obstruction.outlet_flow_cubic_metres_per_second
                ),
                "inlet_flow_cubic_metres_per_second": (
                    obstruction.inlet_flow_cubic_metres_per_second
                ),
            },
        )
    if isinstance(obstruction, NonLaminarPhysicalHydraulicFlowObstruction):
        return _evidence(
            obstruction.edge_id,
            "physical hydraulic edge remains within the laminar regime",
            obstruction.maximum_laminar_reynolds,
            obstruction.reynolds_number,
        )
    if isinstance(obstruction, NonFinitePhysicalHydraulicResultObstruction):
        return _evidence(
            obstruction.identifier,
            f"physical hydraulic {obstruction.result_kind.value} is finite",
            {"finite": True},
            obstruction.value,
        )
    if isinstance(obstruction, InvalidConstitutivePropertiesObstruction):
        return _evidence(
            obstruction.address,
            f"constitutive property {obstruction.property_name} is lawful",
            _constitutive_requirement(obstruction.property_name),
            obstruction.value,
        )
    if isinstance(obstruction, InvalidMaterialFractionObstruction):
        match obstruction.rule:
            case MaterialFractionRule.ACTIVE_CELL_RANGE:
                return _evidence(
                    obstruction.address,
                    "active-cell material fraction is finite and lies in (0, 1]",
                    {
                        "finite": True,
                        "exclusive_minimum": 0.0,
                        "maximum": 1.0,
                    },
                    obstruction.solid_fraction,
                )
            case MaterialFractionRule.FULL_SOLID_VALUE:
                return _evidence(
                    obstruction.address,
                    "full-solid constitutive evidence has unit material fraction",
                    1.0,
                    obstruction.solid_fraction,
                )
            case MaterialFractionRule.RESOLUTION_TYPE:
                return _evidence(
                    obstruction.address,
                    "material fraction resolution belongs to the closed vocabulary",
                    tuple(
                        resolution.value
                        for resolution in MaterialFractionResolution
                    ),
                    obstruction.fraction_resolution,
                )
    if isinstance(obstruction, UnresolvedMaterialFractionObstruction):
        return _evidence(
            obstruction.address,
            "material fraction carries resolved constitutive evidence",
            {
                "resolution": (
                    MaterialFractionResolution.FULL_SOLID.value,
                    MaterialFractionResolution.HOMOGENIZED_EFFECTIVE_PROPERTIES.value,
                )
            },
            {
                "resolution": MaterialFractionResolution.UNRESOLVED.value,
                "solid_fraction": obstruction.solid_fraction,
            },
        )
    if isinstance(obstruction, InvalidBodyForceObstruction):
        return _evidence(
            obstruction.address,
            "body acceleration has three finite components",
            {"coordinate_count": 3, "finite": True},
            obstruction.body_acceleration,
        )
    if isinstance(obstruction, InvalidAcceptanceCriteriaObstruction):
        return _evidence(
            f"assembly/mechanics/{obstruction.criterion}",
            "acceptance criterion is finite and strictly positive",
            {"finite": True, "exclusive_minimum": 0.0},
            obstruction.value,
        )
    if isinstance(obstruction, InvalidBucklingEigenpairObstruction):
        return _evidence(
            "assembly/mechanics/buckling",
            "buckling eigenpair has a positive load factor and passes its residual gate",
            {
                "load_factor": {"finite": True, "exclusive_minimum": 0.0},
                "normalized_eigen_residual": {
                    "finite": True,
                    "maximum": obstruction.maximum_normalized_eigen_residual,
                },
            },
            {
                "load_factor": obstruction.load_factor,
                "normalized_eigen_residual": obstruction.normalized_eigen_residual,
            },
        )
    if isinstance(obstruction, SolverResidualObstruction):
        return _evidence(
            "assembly/mechanics",
            "mechanics solver residual does not exceed tolerance",
            obstruction.tolerance,
            obstruction.normalized_residual,
        )
    if isinstance(obstruction, ForceBalanceObstruction):
        return _evidence(
            "assembly/mechanics",
            "relative force and moment imbalances do not exceed tolerance",
            obstruction.tolerance,
            {
                "relative_force_imbalance": obstruction.relative_force_imbalance,
                "relative_moment_imbalance": (
                    obstruction.relative_moment_imbalance
                ),
            },
        )
    if isinstance(obstruction, SmallStrainLimitObstruction):
        return _evidence(
            "assembly/mechanics",
            "maximum absolute principal strain does not exceed its limit",
            obstruction.limit,
            obstruction.maximum_absolute_principal_strain,
        )
    if isinstance(obstruction, ExcessiveDisplacementObstruction):
        return _evidence(
            "assembly/mechanics",
            "maximum displacement does not exceed its limit",
            obstruction.limit,
            obstruction.maximum_displacement,
        )
    if isinstance(obstruction, InsufficientYieldMarginObstruction):
        return _evidence(
            "assembly/mechanics",
            "yield safety factor meets the declared minimum",
            obstruction.required,
            obstruction.minimum_yield_safety_factor,
        )
    if isinstance(obstruction, InsufficientBucklingMarginObstruction):
        return _evidence(
            "assembly/mechanics",
            "buckling load factor meets the declared minimum",
            obstruction.required_load_factor,
            obstruction.lowest_positive_load_factor,
        )
    if isinstance(obstruction, EmbeddedChannelScaleSeparationObstruction):
        return _evidence(
            "assembly/embedded_channels",
            "outer channel diameter to grid pitch does not exceed its limit",
            obstruction.permitted_maximum,
            obstruction.maximum_outer_diameter_to_grid_pitch,
        )
    if isinstance(obstruction, EmbeddedChannelVolumeFractionObstruction):
        return _evidence(
            "assembly/embedded_channels",
            "embedded channel volume fraction does not exceed its limit",
            obstruction.permitted_maximum,
            obstruction.volume_fraction_upper_bound,
        )
    if isinstance(
        obstruction,
        UnsupportedEmbeddedChannelExternalPressureObstruction,
    ):
        return _evidence(
            obstruction.edge_id,
            "minimum transmural pressure respects the normalization floor",
            -obstruction.normalization_floor,
            obstruction.minimum_transmural_pressure_pascal,
        )
    if isinstance(obstruction, InsufficientEmbeddedChannelBurstMarginObstruction):
        return _evidence(
            "assembly/embedded_channels",
            "embedded channel burst safety factor meets the declared minimum",
            obstruction.required_safety_factor,
            obstruction.minimum_burst_safety_factor,
        )
    return _evidence(
        _projection_address(projected),
        type(obstruction).__name__,
        projected.get("required"),
        projected.get("observed", projected.get("reason")),
    )


def _projected_obstruction_fields(obstruction: object) -> dict[str, object]:
    return {
        "obstruction": type(obstruction).__name__,
        **{
            field.name: project_json_value(getattr(obstruction, field.name))
            for field in fields(obstruction)
        },
    }


def _authoring_section_address(
    section: str,
    element_id: str,
    source: str,
) -> str:
    segments = tuple(filter(None, source.split("/")))
    local_segments = (
        segments[2:]
        if segments[:2] == ("elements", element_id)
        else segments
    )
    field_segments = (
        local_segments[1:]
        if local_segments[:1] == (section,)
        else local_segments
    )
    candidate = render_address(
        Address(
            ("meta",),
            (
                FieldName(section),
                *tuple(
                    FieldIndex(int(segment))
                    if segment.isdecimal()
                    else FieldName(segment)
                    for segment in field_segments
                ),
            ),
        )
    )
    return (
        candidate
        if isinstance(parse_address(candidate), Address)
        else f"meta@{section}"
    )


def _project_nested_authoring_obstruction(
    element_id: str,
    section: str,
    obstruction: NestedAuthoringObstruction,
) -> dict[str, object]:
    projected = {
        **_projected_obstruction_fields(obstruction),
        "element_id": element_id,
    }
    return {
        **projected,
        **_evidence(
            _authoring_section_address(
                section,
                element_id,
                str(projected.get("address", "")),
            ),
            f"{section} satisfies {obstruction.rule.value}",
            obstruction.required,
            obstruction.authored,
        ),
        "diagnostic_domain": "body",
        "contract_domain": "body",
    }


def _project_nested_body_obstruction(
    element_id: str,
    obstruction: BodyObstruction,
) -> dict[str, object]:
    projected = project_obstruction(obstruction)
    return {
        **projected,
        "element_id": element_id,
        "diagnostic_domain": "body",
        "contract_domain": projected.get("contract_domain", "body"),
    }


def _project_vascular_from_assembly(
    obstruction: VasculatureObstruction,
) -> dict[str, object]:
    return {
        **project_vascular_obstruction(obstruction),
        "diagnostic_domain": "vascular",
        "contract_domain": "vascular",
    }


def _project_anatomy_from_assembly(
    obstruction: AnatomyObstruction,
) -> dict[str, object]:
    return {
        **project_anatomy_obstruction(obstruction),
        "diagnostic_domain": "anatomy",
        "contract_domain": "anatomy",
    }


def _project_single_assembly_obstruction(
    obstruction: AssemblyObstruction,
) -> dict[str, object]:
    projected = _projected_obstruction_fields(obstruction)
    semantic = _semantic_projection(obstruction, projected)
    return {
        **projected,
        **{
            key: project_json_value(value)
            for key, value in semantic.items()
        },
    }


def project_assembly_obstructions(
    obstruction: AssemblyObstruction,
) -> tuple[dict[str, object], ...]:
    match obstruction:
        case ElementBodyObstruction(element_id, obstructions):
            return tuple(
                _project_nested_body_obstruction(element_id, nested)
                for nested in obstructions
            )
        case PlateElementObstruction(element_id, obstructions):
            return tuple(
                _project_nested_authoring_obstruction(
                    element_id,
                    "plate_policy",
                    nested,
                )
                for nested in obstructions
            )
        case SurfaceColorElementObstruction(element_id, obstructions):
            return tuple(
                _project_nested_authoring_obstruction(
                    element_id,
                    "surface_color",
                    nested,
                )
                for nested in obstructions
            )
        case AppearancePaletteElementObstruction(element_id, obstructions):
            return tuple(
                _project_nested_authoring_obstruction(
                    element_id,
                    "appearance_palette",
                    nested,
                )
                for nested in obstructions
            )
        case ElementSurfaceDetailObstruction(element_id, obstructions):
            return tuple(
                _project_nested_authoring_obstruction(
                    element_id,
                    "surface_detail",
                    nested,
                )
                for nested in obstructions
            )
        case ElementSurfaceFormationObstruction(element_id, obstructions):
            return tuple(
                {
                    **_projected_obstruction_fields(nested),
                    **_crossing_slack_evidence(nested),
                    "element_id": element_id,
                    "diagnostic_domain": "surface_formation",
                    "contract_domain": "body",
                    "vocabulary_violations": project_json_value(
                        obstruction.vocabulary_violations
                    ),
                    "masked_integrity": (
                        None
                        if obstruction.masked_integrity is None
                        else {
                            "checks": obstruction.masked_integrity.checks,
                            "reason": obstruction.masked_integrity.reason,
                        }
                    ),
                }
                for nested in obstructions
            )
        case PhysicalHydraulicBalanceObstruction(nested):
            return project_assembly_obstructions(nested)
        case (
            MissingProvenanceObstruction()
            | EmptyPerfusionTerritoryObstruction()
            | InsufficientTerminalSitesObstruction()
            | InfeasibleBifurcationObstruction()
            | VascularSearchBudgetObstruction()
            | CapsuleEscapeObstruction()
            | VascularIntersectionObstruction()
            | SymmetryMismatchObstruction()
            | VascularGluingObstruction()
            | DisconnectedVascularGraphObstruction()
            | VascularSolverResidualObstruction()
            | DemandDeliveryMismatchObstruction()
            | ReversedVascularFlowObstruction()
            | InvalidStructuralGuidanceObstruction()
        ):
            return (_project_vascular_from_assembly(obstruction),)
        case (
            MalformedAnatomyObstruction()
            | InvalidSkeletonAddressObstruction()
            | DuplicateAnatomyRegionObstruction()
            | DuplicateRegionHostObstruction()
            | DuplicateExchangeBedObstruction()
            | UnknownAnatomyRegionObstruction()
            | UnreferencedAnatomyRegionObstruction()
            | MissingTerminalRegionObstruction()
            | NonterminalExchangeRegionObstruction()
            | DuplicateMyotendinousUnitObstruction()
            | InvalidMyotendinousPathObstruction()
            | UnperfusedMyotendinousUnitObstruction()
            | UnsupportedTissueCarrierObstruction()
            | MissingSkeletalCrossSectionObstruction()
            | DuplicateIntegumentLayerObstruction()
            | MissingIntegumentLayerObstruction()
            | InsufficientVascularStagesObstruction()
            | DisconnectedAnatomyRegionObstruction()
        ):
            return (_project_anatomy_from_assembly(obstruction),)
        case _:
            return (_project_single_assembly_obstruction(obstruction),)


__all__ = ["project_assembly_obstructions", "project_json_value"]
