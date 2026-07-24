from __future__ import annotations

import json
from pathlib import Path

from types import MappingProxyType

from golem.assembly import (
    ElementClearanceObstruction,
    ElementFitLaw,
    ElementFitObstruction,
    ElementFitReceipt,
    PinnedResolution,
    RejectedAssembly,
    TargetPitch,
)
from golem.assembly.obstructions import (
    AppearancePaletteElementObstruction,
    CoolantTemperatureValidityObstruction,
    ElementBodyObstruction,
    ElementSurfaceDetailObstruction,
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
from golem.kernel.anatomy import (
    AcceptedAnatomy,
    AnatomyRegion,
    AnatomyCarrierRow,
    BodyRegionKind,
    CapillaryBed,
    CapsuleEscapeObstruction,
    ClosedVascularSystem,
    CollectingVenule,
    CirculationCircuit,
    DistributingArtery,
    InsufficientVascularStagesObstruction,
    InsufficientTerminalSitesObstruction,
    OverallAnatomy,
    PerfusedTissueKind,
    PumpOrgan,
    RejectedAnatomy,
    RejectedVasculature,
    ResistanceArteriole,
    ReturningVein,
    VascularIntersectionObstruction,
    VascularBridgeWitness,
    VascularBudgetHypothesis,
    VascularConditioningHypothesis,
    VascularConditionSectionWitness,
    VascularDemandHypothesis,
    VascularGluingInterrogation,
    VascularGluingObstruction,
    VascularResidualSectionWitness,
    VascularTopologyHypothesis,
    VascularSegmentGeometry,
)
from golem.kernel.anatomy.hydraulics.types import (
    EmptyPhysicalHydraulicGraphObstruction,
    InvalidDynamicViscosityObstruction,
    InvalidFluidDensityObstruction,
    InvalidMetresPerWorldUnitObstruction,
    InvalidPhysicalHydraulicConductanceObstruction,
    InvalidPhysicalHydraulicEdgeGeometryObstruction,
    InvalidPhysicalHydraulicNodePositionObstruction,
    InvalidPumpPressureBoundaryObstruction,
    NonFinitePhysicalHydraulicResultObstruction,
    NonLaminarPhysicalHydraulicFlowObstruction,
    NonPositivePhysicalHydraulicFlowObstruction,
    NonPositivePhysicalPumpFlowObstruction,
    PhysicalHydraulicBalanceObstruction,
    PhysicalHydraulicResultKind,
)
from golem.kernel.anatomy.material.types import (
    ChannelInducedDisconnectionObstruction,
    UnresolvedLumenObstruction,
    UnresolvedVascularWallObstruction,
    VascularMaterialEscapeObstruction,
)
from golem.kernel.body.relations import (
    BranchCycleObstruction,
    BranchSearchBudgetObstruction,
    FixedPlacementConflictObstruction,
    RelationalSolveExhaustedObstruction,
    RelationalSolverBudgetObstruction,
    UnderconstrainedRelationObstruction,
)
from golem.kernel.body.types import (
    BODY_SPEC_TOP_LEVEL_KEYS,
    DegenerateMuscleSpanObstruction,
    EyeRule,
    InvalidMuscleProfileObstruction,
    LoftSectionRule,
    MalformedEyeObstruction,
    MalformedMuscleBulkObstruction,
    MalformedMuscleDeclarationObstruction,
    MalformedMuscleSectionsObstruction,
    MuscleBulkBasis,
    MuscleBulkReference,
    MuscleBulkRule,
    MuscleDeclarationRule,
    MuscleEndpoint,
    MuscleMirrorAnchorMismatchObstruction,
    MuscleProfileRule,
    NonGlossyEyeMaterialObstruction,
    UnresolvableMuscleBulkBasisObstruction,
    UnknownBodySpecKeyObstruction,
    UnknownEyeMaterialObstruction,
)
from golem.kernel.mechanics.embedded_channel.model import (
    EmbeddedChannelScaleSeparationObstruction,
    EmbeddedChannelVolumeFractionObstruction,
    EmbeddedChannelVolumeKind,
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
    MaterialFractionRule,
    SmallStrainLimitObstruction,
    SolverResidualObstruction,
    UnresolvedMaterialFractionObstruction,
)
from golem.kernel.mechanics.model import CellIndex
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
from golem.kernel.sheaf.vocabulary import (
    BalanceSolverParameter,
    BalanceTermKind,
    HeatTransferCorrelationKind,
    HeatTransferValidityAxis,
    ThermalParameter,
)
from golem.kernel.engine.types import (
    SurfaceDetailObstruction,
    SurfaceDetailRule,
)
from golem.materials.surface import (
    AppearancePaletteObstruction,
    AppearancePaletteRule,
    SurfaceColorObstruction,
    SurfaceColorRule,
)
from golem.materials import AppearanceMaterialId
from golem.plates.core import (
    PlatePolicyObstruction,
    PlatePolicyRule,
    PlateSurfaceObstruction,
    PlateSurfaceRule,
)
from golem.session.protocol import (
    MarginDisposition,
    _anatomy_margins,
    _anatomy_repair_events,
    _anatomy_diagnostics,
    _assembly_diagnostics,
    _body_compile_diagnostics,
    _fit_margins,
    _glue_evidence,
    _margin,
    _vascular_diagnostics,
    _vascular_violation_margins,
)
from golem.session.state import BodyCompileObstruction, CompileObstructed


def _interrogated_gluing(
    region_id: str | None = "wing_tips",
    maximum_stop_residual: float = 3.0e-10,
    relative_residual: float = 3.0e-13,
) -> VascularGluingObstruction:
    return VascularGluingObstruction(
        "closed-vascular-field",
        "NonConvergentBalance @closed-vascular-balance: solver_info=40",
        (
            VascularSegmentGeometry(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                0.01,
                "edge-0",
                ("skeleton/wing",),
                ("meta@flesh[0]",),
            ),
        ),
        VascularGluingInterrogation(
            VascularTopologyHypothesis(
                1, (VascularBridgeWitness("edge-0", 1.0e-6),)
            ),
            VascularConditioningHypothesis(
                2.0e6,
                1.0e12,
                (
                    VascularConditionSectionWitness(
                        "wing-node", region_id, 1.0e6, 1.0e12
                    ),
                ),
            ),
            VascularDemandHypothesis(
                maximum_stop_residual=maximum_stop_residual,
                residual_tolerance=1.0e-10,
                relative_residual=relative_residual,
                relative_tolerance=1.0e-13,
                iteration_cap=40,
                iterations_used=40,
                exhausted=True,
                worst_sections=(
                    VascularResidualSectionWitness(
                        "wing-node", region_id, -maximum_stop_residual
                    ),
                ),
            ),
            VascularBudgetHypothesis(40, 40, 1.0e-13, True),
        ),
    )


def test_gluing_interrogation_populates_authoring_help_and_binding_constraint() -> None:
    rejection = RejectedVasculature((_interrogated_gluing(),))
    document = json.loads(
        (
            Path(__file__).parents[1] / "specs" / "cinderwake.json"
        ).read_text()
    )
    diagnostic = _vascular_diagnostics(rejection, document)[0]

    assert diagnostic.code == "vascular.FailedGluing"
    assert diagnostic.predicate == "maximum free-cell residual at solver stop"
    assert diagnostic.required == 1.0e-10
    assert diagnostic.observed == 3.0e-10
    assert diagnostic.witness["unit"] == "absolute_residual"
    demand_witness = diagnostic.witness["interrogation"]["hypotheses"][
        "demand"
    ]["witness"]
    assert demand_witness["relative_residual"] == 3.0e-13
    assert demand_witness["relative_tolerance"] == 1.0e-13
    assert demand_witness["iteration_cap"] == 40
    assert demand_witness["iterations_used"] == 40
    assert demand_witness["exhausted"] is True
    assert tuple(
        diagnostic.witness["interrogation"]["hypotheses"]
    ) == ("topology", "conditioning", "demand", "budget")
    assert diagnostic.witness["interrogation"]["suspect_addresses"] == (
        "anatomy/regions/wing_tips",
        "meta@flesh[0]",
        "skeleton/wing",
    )
    assert diagnostic.to_json()["help"]["operations"] == (
        {
            "op": "set",
            "addr": "meta@anatomy.overall.circulation.distance_decay",
            "value": 0.93,
        },
        {
            "op": "set",
            "addr": (
                "meta@anatomy.overall.circulation."
                "exchange_beds[3].demand"
            ),
            "value": 0.01,
        },
    )
    pump_region_diagnostic = _vascular_diagnostics(
        RejectedVasculature((_interrogated_gluing("axial_core"),)),
        document,
    )[0]
    assert tuple(
        operation.addr.source
        for operation in pump_region_diagnostic.help_operations
    ) == (
        "meta@anatomy.overall.circulation.distance_decay",
        "meta@anatomy.overall.circulation.exchange_beds[2].demand",
    )
    margins = _vascular_violation_margins(rejection)
    assert len(margins) == 1
    assert margins[0].margin == -2.0e-10
    assert margins[0].predicate == diagnostic.predicate
    assert margins[0].required == diagnostic.required
    assert margins[0].observed == diagnostic.observed
    assert margins[0].unit == "absolute_residual"
    assert margins[0].disposition is MarginDisposition.VIOLATED
    assert margins[0].observed > margins[0].required
    assert margins[0].authoring_addresses == (
        "anatomy/regions/wing_tips",
        "meta@flesh[0]",
        "skeleton/wing",
    )
    regionless_margins = _vascular_violation_margins(
        RejectedVasculature((_interrogated_gluing(None),))
    )
    assert regionless_margins[0].authoring_addresses == (
        "closed-vascular-field/nodes/wing-node",
        "meta@flesh[0]",
        "skeleton/wing",
    )
    evidence = _glue_evidence((diagnostic,), margins)
    assert evidence.binding_constraint is not None
    assert evidence.binding_constraint.constraint == (
        "vascular:violation:VascularGluingObstruction:0"
    )

    relative_rejection = RejectedVasculature(
        (
            _interrogated_gluing(
                maximum_stop_residual=5.0e-11,
                relative_residual=4.0e-13,
            ),
        )
    )
    relative_diagnostic = _vascular_diagnostics(
        relative_rejection, document
    )[0]
    assert relative_diagnostic.predicate == (
        "relative free-system residual at solver stop"
    )
    assert relative_diagnostic.required == 1.0e-13
    assert relative_diagnostic.observed == 4.0e-13
    assert relative_diagnostic.witness["unit"] == "relative_residual"
    relative_margins = _vascular_violation_margins(relative_rejection)
    assert len(relative_margins) == 1
    assert relative_margins[0].predicate == relative_diagnostic.predicate
    assert relative_margins[0].required == relative_diagnostic.required
    assert relative_margins[0].observed == relative_diagnostic.observed
    assert relative_margins[0].margin == -3.0e-13
    assert relative_margins[0].unit == "relative_residual"
    assert relative_margins[0].disposition is MarginDisposition.VIOLATED
    assert relative_margins[0].observed > relative_margins[0].required
    assert _vascular_violation_margins(
        RejectedVasculature(
            (
                _interrogated_gluing(
                    maximum_stop_residual=5.0e-11,
                    relative_residual=5.0e-14,
                ),
            )
        )
    ) == ()

    legacy_rejection = RejectedVasculature(
        (
            VascularGluingObstruction(
                "closed-vascular-field", "legacy reason"
            ),
        )
    )
    legacy = _vascular_diagnostics(legacy_rejection, document)[0]
    assert legacy.witness == {
        "kind": "FailedGluing",
        "address": "closed-vascular-field",
        "reason": "legacy reason",
    }
    assert legacy.help_operations == ()
    assert _vascular_violation_margins(legacy_rejection) == ()


def test_insufficient_terminal_sites_preserves_requirement_direction() -> None:
    diagnostic = _vascular_diagnostics(
        RejectedVasculature(
            (InsufficientTerminalSitesObstruction("arm", 5, 2),)
        )
    )[0]

    assert diagnostic.code == "vascular.InsufficientTerminalSites"
    assert diagnostic.address == "anatomy/regions/arm"
    assert diagnostic.required == 5
    assert diagnostic.observed == 2
    assert diagnostic.witness == {
        "kind": "InsufficientTerminalSites",
        "address": "anatomy/regions/arm",
        "predicate": "available terminal sites meet the requested count",
        "required": 5,
        "observed": 2,
        "requested": 5,
        "available": 2,
    }


def test_vascular_rejection_diagnostic_retains_localized_wound() -> None:
    failing_segment = VascularSegmentGeometry(
        source_position=(0.1, 0.2, 0.3),
        target_position=(0.4, 0.5, 0.6),
        radius=0.002,
        edge_id="supply:fore_shin:3",
        source_host_bone_addresses=("skeleton/fore_shin",),
        source_host_part_addresses=("skeleton/fore_shin/flesh[0]",),
        target_host_bone_addresses=("skeleton/ankle",),
        target_host_part_addresses=("skeleton/ankle/flesh[1]",),
    )
    diagnostic = _vascular_diagnostics(
        RejectedVasculature(
            (
                CapsuleEscapeObstruction(
                    "supply:fore_shin:3",
                    -0.01,
                    (failing_segment,),
                ),
            )
        )
    )[0]

    assert diagnostic.witness["failing_segments"] == (
        {
            "edge_id": "supply:fore_shin:3",
            "source": {
                "position": (0.1, 0.2, 0.3),
                "host_bone_addresses": ("skeleton/fore_shin",),
                "host_part_addresses": (
                    "skeleton/fore_shin/flesh[0]",
                ),
            },
            "target": {
                "position": (0.4, 0.5, 0.6),
                "host_bone_addresses": ("skeleton/ankle",),
                "host_part_addresses": ("skeleton/ankle/flesh[1]",),
            },
            "radius": 0.002,
        },
    )


def test_rejected_vascular_evidence_reports_every_violated_channel() -> None:
    failing_segment = VascularSegmentGeometry(
        source_position=(0.0, 0.0, 0.0),
        target_position=(0.0, 1.0, 0.0),
        radius=0.003,
        edge_id="supply:shin",
        source_host_bone_addresses=("skeleton/shin",),
        source_host_part_addresses=("skeleton/shin/flesh[0]",),
        target_host_bone_addresses=("skeleton/foot",),
        target_host_part_addresses=("skeleton/foot/flesh[0]",),
    )
    rejected = RejectedVasculature(
        (
            CapsuleEscapeObstruction(
                "supply:shin",
                0.0001,
                (failing_segment,),
            ),
            VascularIntersectionObstruction(
                "supply:shin",
                "return:shin",
                -0.04,
                (failing_segment,),
            ),
        )
    )
    violation_margins = _vascular_violation_margins(rejected)
    unrelated_satisfied_wall = _margin(
        "anatomy:unrelated_wall",
        "skeleton/core/flesh[0]",
        "minimum carrier radius",
        0.01,
        0.02,
        0.01,
        "world_unit",
        authoring_addresses=("skeleton/core/flesh[0]",),
    )
    evidence = _glue_evidence(
        _vascular_diagnostics(rejected),
        (*violation_margins, unrelated_satisfied_wall),
    )

    assert tuple(margin.disposition for margin in evidence.margins) == (
        MarginDisposition.VIOLATED,
        MarginDisposition.VIOLATED,
        MarginDisposition.SATISFIED,
    )
    assert tuple(margin.authoring_addresses for margin in evidence.margins) == (
        (
            "skeleton/shin/flesh[0]",
            "skeleton/shin",
            "skeleton/foot/flesh[0]",
            "skeleton/foot",
        ),
    ) * 2 + (("skeleton/core/flesh[0]",),)
    assert tuple(margin.active for margin in evidence.margins) == (
        False,
        True,
        False,
    )
    assert evidence.binding_constraint is evidence.margins[1]


def test_repaired_anatomy_margin_names_knob_and_emits_journal_event() -> None:
    pump_region = AnatomyRegion("torso", BodyRegionKind.TORSO, "root")
    exchange_region = AnatomyRegion("shin", BodyRegionKind.LIMB, "shin")
    capillary_bed = CapillaryBed(
        exchange_region,
        PerfusedTissueKind.SKELETAL_MUSCLE,
        0.5,
        None,
    )
    anatomy = AcceptedAnatomy(
        overall=OverallAnatomy(
            (pump_region, exchange_region),
            ClosedVascularSystem(
                PumpOrgan(pump_region),
                (capillary_bed,),
                1.0,
                0.0,
            ),
        ),
        circuits=(
            CirculationCircuit(
                capillary_bed,
                0.5,
                ("root", "shin"),
                (DistributingArtery("skeleton/root~shin"),),
                ResistanceArteriole("skeleton/shin@arteriole"),
                CollectingVenule("skeleton/shin@venule"),
                (ReturningVein("skeleton/shin~root"),),
            ),
        ),
        carrier_rows=(
            AnatomyCarrierRow(
                bone_id="shin",
                interface_address="anatomy/regions/shin",
                shape_address="skeleton/shin/flesh[0]@size",
                normalized_flow=0.5,
                distance_from_pump=2,
                circulation_minimum_radius=0.04,
                tissue_minimum_radius=None,
                measured_minimum_radius=0.02,
            ),
        ),
        tissue_envelopes=(),
    )
    margin = _anatomy_margins(anatomy)[0]
    repair = _anatomy_repair_events(anatomy, 7)[0]
    evidence = _glue_evidence((), (margin,), (repair,))

    assert margin.disposition is MarginDisposition.REPAIRED
    assert margin.authoring_addresses == ("skeleton/shin/flesh[0]@size",)
    assert not margin.active
    assert evidence.binding_constraint is None
    assert repair.to_json() == {
        "txn": 7,
        "kind": "auto_repair",
        "constraint": "anatomy:carrier:shin",
        "address": "skeleton/shin/flesh[0]@size",
        "predicate": "circulation_clearance",
        "authored": 0.02,
        "repaired": 0.04,
        "operation": {"operation": "scale_radius", "factor": 2.0},
        "authoring_addresses": ("skeleton/shin/flesh[0]@size",),
        "ts": 7,
    }


def test_unknown_eye_materials_batch_with_complete_legal_vocabulary() -> None:
    diagnostics = _body_compile_diagnostics(
        CompileObstructed(
            (
                BodyCompileObstruction(
                    (
                        UnknownEyeMaterialObstruction(
                            "/eyes/0/appearance_material",
                            "verdigris_glass",
                        ),
                        UnknownEyeMaterialObstruction(
                            "/eyes/1/appearance_material",
                            "moonstone",
                        ),
                    )
                ),
            )
        )
    )

    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "body.UnknownEyeMaterialObstruction",
        "body.UnknownEyeMaterialObstruction",
    )
    assert tuple(diagnostic.address for diagnostic in diagnostics) == (
        "meta@eyes[0].appearance_material",
        "meta@eyes[1].appearance_material",
    )
    assert tuple(diagnostic.observed for diagnostic in diagnostics) == (
        "verdigris_glass",
        "moonstone",
    )
    assert tuple(diagnostic.required for diagnostic in diagnostics) == (
        tuple(material.value for material in AppearanceMaterialId),
    ) * 2


def test_element_fit_preserves_pair_address_and_quantitative_evidence() -> None:
    obstruction = ElementFitObstruction(
        left_element_id="left",
        right_element_id="right",
        fit_law=ElementFitLaw.DISJOINT,
        penetrating_sample_count=7,
        outside_contact_sample_count=7,
        maximum_penetration_world=0.02,
        tolerance_world=0.001,
        contact_element_id=None,
        contact_radius_world=None,
    )
    diagnostic = _assembly_diagnostics(RejectedAssembly((obstruction,)))[0]

    assert diagnostic.code == "assembly.ElementFitObstruction"
    assert diagnostic.address == "assembly/left~right"
    assert diagnostic.predicate == (
        "no penetrating samples lie outside the declared contact domain"
    )
    assert diagnostic.required == 0
    assert diagnostic.observed == 7
    assert diagnostic.witness["penetrating_sample_count"] == 7
    assert diagnostic.witness["outside_contact_sample_count"] == 7


def test_bounded_contact_success_does_not_fabricate_penetration_slack() -> None:
    margins = _fit_margins(
        ElementFitReceipt(
            left_element_id="body",
            right_element_id="tool",
            fit_law=ElementFitLaw.BOUNDED_MOUNT_CONTACT,
            declaring_element_id="tool",
            minimum_clearance_world=0.0,
            permitted_clearance_world=0.002,
            maximum_penetration_world=0.02,
            tolerance_world=0.001,
            penetrating_sample_count=5,
            fitted_surface_fraction=None,
            required_fitted_surface_fraction=None,
        )
    )

    assert margins == ()


def test_element_clearance_preserves_both_failure_axes() -> None:
    diagnostics = _assembly_diagnostics(
        RejectedAssembly(
            (
                ElementClearanceObstruction(
                    "body",
                    "shell",
                    ElementFitLaw.SURFACE_CLEARANCE,
                    "shell",
                    0.02,
                    0.01,
                    0.001,
                    0.9,
                    0.8,
                ),
                ElementClearanceObstruction(
                    "body",
                    "shell",
                    ElementFitLaw.SURFACE_CLEARANCE,
                    "shell",
                    0.005,
                    0.01,
                    0.001,
                    0.5,
                    0.8,
                ),
            )
        )
    )

    assert tuple(
        (diagnostic.required, diagnostic.observed)
        for diagnostic in diagnostics
    ) == (
        (
            {
                "maximum_minimum_clearance_world": 0.011,
                "minimum_fitted_surface_fraction": 0.8,
            },
            {
                "minimum_clearance_world": 0.02,
                "fitted_surface_fraction": 0.9,
            },
        ),
        (
            {
                "maximum_minimum_clearance_world": 0.011,
                "minimum_fitted_surface_fraction": 0.8,
            },
            {
                "minimum_clearance_world": 0.005,
                "fitted_surface_fraction": 0.5,
            },
        ),
    )


def test_unknown_body_key_reports_the_closed_dialect_vocabulary() -> None:
    obstruction = UnknownBodySpecKeyObstruction(
        address="meta@mystery_section",
        authored_key="mystery_section",
        closest_valid_candidates=("surface_detail",),
    )
    diagnostic = _body_compile_diagnostics(
        CompileObstructed((BodyCompileObstruction((obstruction,)),))
    )[0]

    assert diagnostic.code == "body.UnknownBodySpecKeyObstruction"
    assert diagnostic.address == "meta@mystery_section"
    assert diagnostic.required == tuple(sorted(BODY_SPEC_TOP_LEVEL_KEYS))
    assert diagnostic.observed == "mystery_section"
    assert diagnostic.witness["closest_valid_candidates"] == ["surface_detail"]


def test_non_glossy_eye_preserves_upper_bound_direction() -> None:
    obstruction = NonGlossyEyeMaterialObstruction(
        address="eyes/0",
        appearance_material="iris",
        roughness=0.4,
        maximum_roughness=0.1,
    )
    diagnostic = _body_compile_diagnostics(
        CompileObstructed((BodyCompileObstruction((obstruction,)),))
    )[0]

    assert diagnostic.address == "meta@eyes[0]"
    assert diagnostic.required == 0.1
    assert diagnostic.observed == 0.4


def test_malformed_eye_preserves_typed_law_at_session_address() -> None:
    obstruction = MalformedEyeObstruction(
        address="/eyes/2/brow_ridge/size",
        rule=EyeRule.FINITE_POSITIVE_VECTOR3,
        authored=(0.2, -0.1, 0.3),
        required={
            "arity": 3,
            "finite": True,
            "exclusive_minimum": 0.0,
        },
    )
    diagnostic = _body_compile_diagnostics(
        CompileObstructed((BodyCompileObstruction((obstruction,)),))
    )[0]

    assert diagnostic.address == "meta@eyes[2].brow_ridge.size"
    assert diagnostic.predicate == (
        "eye vector has three finite strictly positive components"
    )
    assert diagnostic.required == {
        "arity": 3,
        "finite": True,
        "exclusive_minimum": 0.0,
    }
    assert diagnostic.observed == (0.2, -0.1, 0.3)


def test_degenerate_muscle_span_preserves_lower_bound_direction() -> None:
    obstruction = DegenerateMuscleSpanObstruction(
        address="muscles/0",
        muscle_id="bicep",
        origin_selector="skeleton/upper_arm/head",
        insertion_selector="skeleton/forearm/head",
        observed_span=0.01,
        minimum_span=0.05,
    )
    diagnostic = _body_compile_diagnostics(
        CompileObstructed((BodyCompileObstruction((obstruction,)),))
    )[0]

    assert diagnostic.address == "meta@muscles[0]"
    assert diagnostic.required == 0.05
    assert diagnostic.observed == 0.01


def test_muscle_authoring_obstructions_preserve_rule_evidence() -> None:
    diagnostics = _body_compile_diagnostics(
        CompileObstructed(
            (
                BodyCompileObstruction(
                    (
                        MalformedMuscleDeclarationObstruction(
                            "muscles/0/id",
                            "id",
                            MuscleDeclarationRule.IDENTIFIER,
                            3,
                        ),
                        MalformedMuscleDeclarationObstruction(
                            "muscles/0/definition",
                            "definition",
                            MuscleDeclarationRule.DEFINITION,
                            1.2,
                        ),
                        MalformedMuscleDeclarationObstruction(
                            "muscles/0/blend",
                            "blend",
                            MuscleDeclarationRule.FINITE_BLEND,
                            -0.1,
                        ),
                        MalformedMuscleBulkObstruction(
                            "muscles/0/bulk",
                            "bicep",
                            MuscleBulkRule.FINITE_POSITIVE_DIMENSION,
                            -1.0,
                        ),
                        UnresolvableMuscleBulkBasisObstruction(
                            "muscles/0/bulk/relative/to",
                            "bicep",
                            MuscleBulkReference.ORIGIN,
                            "bone:arm",
                            MuscleBulkBasis.BONE_LENGTH,
                            0.0,
                        ),
                        MalformedMuscleSectionsObstruction(
                            "muscles/0/sections/0/width",
                            "bicep",
                            0,
                            LoftSectionRule.POSITIVE_WIDTH,
                            "width",
                            0.0,
                            (),
                            (),
                        ),
                        InvalidMuscleProfileObstruction(
                            "muscles/0/sections",
                            "bicep",
                            MuscleProfileRule.MINIMUM_ARITY,
                            2,
                            0.0,
                            1.0,
                            (0.1, 0.1),
                            (0.1, 0.1),
                            None,
                            None,
                        ),
                    )
                ),
            )
        )
    )

    assert tuple(
        (diagnostic.required, diagnostic.observed)
        for diagnostic in diagnostics
    ) == (
        ("identifier", 3),
        ((0.0, 1.0), 1.2),
        ({"finite": True, "minimum": 0.0}, -0.1),
        ({"finite": True, "exclusive_minimum": 0.0}, -1.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (3, 2),
    )
    assert tuple(diagnostic.address for diagnostic in diagnostics) == (
        "meta@muscles[0].id",
        "meta@muscles[0].definition",
        "meta@muscles[0].blend",
        "meta@muscles[0].bulk",
        "meta@muscles[0].bulk.relative.to",
        "meta@muscles[0].sections[0].width",
        "meta@muscles[0].sections",
    )


def test_assembly_limits_preserve_declared_threshold_direction() -> None:
    diagnostics = _assembly_diagnostics(
        RejectedAssembly(
            (
                ThermalLimitObstruction("reactor", 450.0, 400.0),
                PumpPowerPressureLimitObstruction(120_000.0, 100_000.0),
                InsufficientBurstMarginObstruction("vessel", 1.2, 2.0),
            )
        )
    )

    assert tuple(
        (diagnostic.address, diagnostic.required, diagnostic.observed)
        for diagnostic in diagnostics
    ) == (
        ("reactor", 400.0, 450.0),
        ("assembly", 100_000.0, 120_000.0),
        ("vessel", 2.0, 1.2),
    )


def test_body_relation_and_mirror_limits_preserve_typed_semantics() -> None:
    diagnostics = _body_compile_diagnostics(
        CompileObstructed(
            (
                BodyCompileObstruction(
                    (
                        MuscleMirrorAnchorMismatchObstruction(
                            "muscles/1",
                            "left_bicep",
                            "right_bicep",
                            MuscleEndpoint.ORIGIN,
                            "bone:right/head",
                            "bone:left/head",
                            0.02,
                            0.001,
                        ),
                        FixedPlacementConflictObstruction(
                            "fixed",
                            "part:hand",
                            "part:grip",
                            0.0,
                            0.02,
                        ),
                        RelationalSolveExhaustedObstruction(
                            "skeleton/arm",
                            "local",
                            1,
                            "reach",
                            0.0,
                            0.03,
                            3.0,
                            20,
                        ),
                        UnderconstrainedRelationObstruction(
                            "skeleton/arm",
                            3,
                            2,
                            1,
                            ("skeleton/arm/yaw",),
                        ),
                        RelationalSolverBudgetObstruction(
                            "skeleton/arm",
                            200,
                            200,
                            0.2,
                        ),
                        BranchCycleObstruction(
                            "reach",
                            (("near",), ("far",)),
                            0.0,
                            0.03,
                        ),
                        BranchSearchBudgetObstruction("reach", 8, 8, 0.2),
                    )
                ),
            )
        )
    )

    assert tuple(
        (diagnostic.address, diagnostic.required, diagnostic.observed)
        for diagnostic in diagnostics
    ) == (
        ("meta@muscles[1]", 0.001, 0.02),
        ("part:hand", 0.0, 0.02),
        ("skeleton/arm", 0.0, 0.03),
        (
            "skeleton/arm",
            {"rank": 3, "free_dimension_count": 0},
            {"rank": 2, "free_dimension_count": 1},
        ),
        (
            "skeleton/arm",
            {"completed": True, "maximum_evaluations": 200},
            {"completed": False, "evaluated": 200, "best_residual": 0.2},
        ),
        ("pose/relations/reach", 0.0, 0.03),
        (
            "pose/relations/reach",
            {"completed": True, "maximum_active_sets": 8},
            {
                "completed": False,
                "evaluated_active_sets": 8,
                "best_residual": 0.2,
            },
        ),
    )
    assert tuple(
        diagnostic.contract_refs for diagnostic in diagnostics[1:]
    ) == (("relations", "receipts"),) * 6


def test_insufficient_vascular_stages_preserves_closed_stage_law() -> None:
    diagnostic = _anatomy_diagnostics(
        RejectedAnatomy((InsufficientVascularStagesObstruction("arm", 1),))
    )[0]

    assert diagnostic.address == "anatomy/regions/arm"
    assert diagnostic.required == 2
    assert diagnostic.observed == 1


def test_local_assembly_quantitative_obstructions_preserve_laws() -> None:
    integrity_violations = (
        MappingProxyType({
            "rule_id": "solid.sharp_edge",
            "part": "shell.brow",
            "applied_repair": "rounded radius to 0.022616",
        }),
    )
    diagnostics = _assembly_diagnostics(
        RejectedAssembly(
            (
                SolidIntegrityObstruction(
                    "shell",
                    2,
                    False,
                    3,
                    integrity_violations,
                ),
                EmissionPitchObstruction("coolant", "left", 0.02, 0.009, 0.01),
                InvalidResolutionPolicyObstruction(
                    PinnedResolution(20),
                    "outside range",
                ),
                InvalidResolutionPolicyObstruction(
                    TargetPitch(-0.1),
                    "not positive",
                ),
                ThermalGradientLimitObstruction("reactor", 30.0, 20.0),
                CoolantTemperatureValidityObstruction(
                    "coolant:outlet",
                    450.0,
                    273.0,
                    400.0,
                ),
                RemainingLigamentViolationObstruction("shell", 0.001, 0.003),
                RigidPayloadInterfaceGapObstruction("payload", 0.02, 0.005),
                RigidPayloadBalanceObstruction("payload", 8.0, 3.0, 1.0),
                ResolutionRefinementObstruction(
                    "shell",
                    "thermal",
                    "temperature",
                    0.2,
                    0.05,
                ),
                InsufficientVascularThermalCapacityObstruction(
                    "body",
                    0.4,
                    0.1,
                ),
            )
        )
    )

    assert tuple(
        (diagnostic.required, diagnostic.observed)
        for diagnostic in diagnostics
    ) == (
        (
            {
                "component_count": 1,
                "watertight": True,
                "vocabulary_violation_count": 0,
            },
            {
                "component_count": 2,
                "watertight": False,
                "vocabulary_violation_count": 3,
                "violations": integrity_violations,
            },
        ),
        (0.009, 0.02),
        ((60, 260), 20),
        (0.0, -0.1),
        (20.0, 30.0),
        ((273.0, 400.0), 450.0),
        (0.003, 0.001),
        (0.005, 0.02),
        (
            1.0,
            {
                "force_residual_newtons": 8.0,
                "moment_residual_newton_metres": 3.0,
            },
        ),
        (0.05, 0.2),
        (0.4, 0.1),
    )
    assert diagnostics[3].address == "meta@pitch"
    assert diagnostics[0].observed["violations"] == (
        {
            "rule_id": "solid.sharp_edge",
            "part": "shell.brow",
            "applied_repair": "rounded radius to 0.022616",
        },
    )
    assert diagnostics[0].witness["observed"]["violations"] == (
        integrity_violations
    )


def test_mechanics_obstructions_preserve_threshold_direction() -> None:
    diagnostics = _assembly_diagnostics(
        RejectedAssembly(
            (
                SolverResidualObstruction(0.02, 0.001),
                ForceBalanceObstruction(0.03, 0.04, 0.01),
                SmallStrainLimitObstruction(0.2, 0.05),
                ExcessiveDisplacementObstruction(0.03, 0.01),
                InsufficientYieldMarginObstruction(1.1, 2.0),
                InsufficientBucklingMarginObstruction(0.8, 1.5),
            )
        )
    )

    assert tuple(
        (diagnostic.address, diagnostic.required, diagnostic.observed)
        for diagnostic in diagnostics
    ) == (
        ("assembly/mechanics", 0.001, 0.02),
        (
            "assembly/mechanics",
            0.01,
            {
                "relative_force_imbalance": 0.03,
                "relative_moment_imbalance": 0.04,
            },
        ),
        ("assembly/mechanics", 0.05, 0.2),
        ("assembly/mechanics", 0.01, 0.03),
        ("assembly/mechanics", 2.0, 1.1),
        ("assembly/mechanics", 1.5, 0.8),
    )


def test_sheaf_and_embedded_channel_obstructions_preserve_laws() -> None:
    diagnostics = _assembly_diagnostics(
        RejectedAssembly(
            (
                BalanceResidualObstruction("thermal", 3, 0.02, 0.001),
                BalanceImbalanceObstruction("thermal", 0.03, 0.002),
                PhysicalHydraulicBalanceObstruction(
                    BalanceResidualObstruction("hydraulics", 2, 0.04, 0.003)
                ),
                UnsupportedHeatTransferRegimeObstruction(
                    "coolant",
                    HeatTransferCorrelationKind.DITTUS_BOELTER_SMOOTH_CIRCULAR_TUBE,
                    HeatTransferValidityAxis.REYNOLDS,
                    1500.0,
                    2300.0,
                    100_000.0,
                ),
                EmbeddedChannelScaleSeparationObstruction(0.3, 0.1),
                EmbeddedChannelVolumeFractionObstruction(
                    EmbeddedChannelVolumeKind.LUMEN,
                    0.2,
                    0.05,
                ),
                InsufficientEmbeddedChannelBurstMarginObstruction(1.2, 2.0),
            )
        )
    )

    assert tuple(
        (diagnostic.address, diagnostic.required, diagnostic.observed)
        for diagnostic in diagnostics
    ) == (
        ("thermal/cell:3", 0.001, 0.02),
        ("thermal", 0.002, 0.03),
        ("hydraulics/cell:2", 0.003, 0.04),
        ("coolant", (2300.0, 100_000.0), 1500.0),
        ("assembly/embedded_channels", 0.1, 0.3),
        ("assembly/embedded_channels", 0.05, 0.2),
        ("assembly/embedded_channels", 2.0, 1.2),
    )


def test_hydraulic_and_material_obstructions_preserve_laws() -> None:
    diagnostics = _assembly_diagnostics(
        RejectedAssembly(
            (
                MaterialValidityObstruction(
                    "/service/materials/skin",
                    "skin",
                    1200.0,
                    900.0,
                    1100.0,
                ),
                NonPositiveBudgetObstruction("/service/pump/power", 0.0),
                InsufficientTerminalSitesObstruction("arm", 5, 2),
                UnresolvedLumenObstruction("artery", 0.001, 0.001, 2.0, 3),
                UnresolvedVascularWallObstruction(0.001, 0.001, 2.0),
                VascularMaterialEscapeObstruction(4),
                ChannelInducedDisconnectionObstruction(2),
                InvalidMetresPerWorldUnitObstruction(0.0),
                InvalidFluidDensityObstruction(-1.0),
                InvalidDynamicViscosityObstruction(0.0),
                InvalidPumpPressureBoundaryObstruction(90.0, 100.0),
                EmptyPhysicalHydraulicGraphObstruction(1, 0),
                InvalidPhysicalHydraulicConductanceObstruction("artery", -0.3),
                InvalidPhysicalHydraulicEdgeGeometryObstruction(
                    "artery",
                    -0.1,
                    0.0,
                ),
                NonPositivePhysicalHydraulicFlowObstruction("artery", -0.2),
                NonPositivePhysicalPumpFlowObstruction(-0.1, 0.0),
                NonLaminarPhysicalHydraulicFlowObstruction(
                    "artery",
                    3000.0,
                    2300.0,
                ),
            )
        )
    )

    assert tuple(
        (diagnostic.address, diagnostic.required, diagnostic.observed)
        for diagnostic in diagnostics
    ) == (
        ("/service/materials/skin", (900.0, 1100.0), 1200.0),
        ("/service/pump/power", 0.0, 0.0),
        ("anatomy/regions/arm", 5, 2),
        ("artery", 2.0, 1.0),
        ("anatomy/vasculature/wall", 2.0, 1.0),
        ("anatomy/vasculature", 0, 4),
        ("anatomy/vasculature", 1, 2),
        ("assembly/hydraulics/unit_scale", 0.0, 0.0),
        ("assembly/hydraulics/fluid_density", 0.0, -1.0),
        ("assembly/hydraulics/dynamic_viscosity", 0.0, 0.0),
        ("assembly/pump", 0.0, -10.0),
        (
            "assembly/hydraulics",
            {"node_count": 1, "edge_count": 1},
            {"node_count": 1, "edge_count": 0},
        ),
        ("artery", 0.0, -0.3),
        (
            "artery",
            {"radius_world_units": 0.0, "length_world_units": 0.0},
            {"radius_world_units": -0.1, "length_world_units": 0.0},
        ),
        ("artery", 0.0, -0.2),
        (
            "assembly/pump",
            {
                "outlet_flow_cubic_metres_per_second": 0.0,
                "inlet_flow_cubic_metres_per_second": 0.0,
            },
            {
                "outlet_flow_cubic_metres_per_second": -0.1,
                "inlet_flow_cubic_metres_per_second": 0.0,
            },
        ),
        ("artery", 2300.0, 3000.0),
    )


def test_assembly_wrapper_descent_is_plural_and_preserves_native_domains() -> None:
    diagnostics = _assembly_diagnostics(
        RejectedAssembly(
            (
                ElementBodyObstruction(
                    "body",
                    (
                        MalformedEyeObstruction(
                            "/eyes/0/radius",
                            EyeRule.FINITE_POSITIVE_NUMBER,
                            -0.1,
                            {"finite": True, "exclusive_minimum": 0.0},
                        ),
                        BranchSearchBudgetObstruction("reach", 8, 8, 3.2),
                    ),
                ),
                InsufficientTerminalSitesObstruction("arm", 5, 2),
                InsufficientVascularStagesObstruction("arm", 1),
            )
        )
    )

    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "body.MalformedEyeObstruction",
        "body.BranchSearchBudgetObstruction",
        "vascular.InsufficientTerminalSites",
        "anatomy.InsufficientVascularStages",
    )
    assert tuple(diagnostic.address for diagnostic in diagnostics) == (
        "meta@eyes[0].radius",
        "pose/relations/reach",
        "anatomy/regions/arm",
        "anatomy/regions/arm",
    )
    assert "relations" in diagnostics[1].contract_refs
    assert "envelope" in diagnostics[2].contract_refs
    assert "schema" in diagnostics[3].contract_refs
    direct_vascular = _vascular_diagnostics(
        RejectedVasculature((InsufficientTerminalSitesObstruction("arm", 5, 2),))
    )[0]
    direct_anatomy = _anatomy_diagnostics(
        RejectedAnatomy((InsufficientVascularStagesObstruction("arm", 1),))
    )[0]
    assert (diagnostics[2].code, diagnostics[2].contract_refs) == (
        direct_vascular.code,
        direct_vascular.contract_refs,
    )
    assert (diagnostics[3].code, diagnostics[3].contract_refs) == (
        direct_anatomy.code,
        direct_anatomy.contract_refs,
    )


def test_nested_authoring_wrappers_recover_canonical_spec_addresses() -> None:
    diagnostics = _assembly_diagnostics(
        RejectedAssembly(
            (
                PlateElementObstruction(
                    "body",
                    (
                        PlatePolicyObstruction(
                            "/relief",
                            PlatePolicyRule.FINITE_NON_NEGATIVE,
                            -0.1,
                            {"finite": True, "minimum": 0.0},
                        ),
                        PlateSurfaceObstruction(
                            "/cell_count",
                            PlateSurfaceRule.AVAILABLE_VERTEX_COUNT,
                            8,
                            {"maximum": 4},
                        ),
                    ),
                ),
                SurfaceColorElementObstruction(
                    "body",
                    (
                        SurfaceColorObstruction(
                            "/surface_color/tint_linear_rgb/1",
                            SurfaceColorRule.BOUNDED_NUMBER,
                            1.2,
                            {
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "exclusive_minimum": False,
                            },
                        ),
                    ),
                ),
                AppearancePaletteElementObstruction(
                    "body",
                    (
                        AppearancePaletteObstruction(
                            "/appearance_palette/obsidian_warden",
                            AppearancePaletteRule.KNOWN_MATERIAL,
                            "obsidian_warden",
                            ("neutral_gray",),
                        ),
                    ),
                ),
                ElementSurfaceDetailObstruction(
                    "body",
                    (
                        SurfaceDetailObstruction(
                            "/surface_detail/scale",
                            SurfaceDetailRule.FINITE_POSITIVE_SCALE,
                            0.0,
                            {"finite": True, "exclusive_minimum": 0.0},
                        ),
                    ),
                ),
            )
        )
    )

    assert tuple(diagnostic.code for diagnostic in diagnostics) == (
        "body.PlatePolicyObstruction",
        "body.PlateSurfaceObstruction",
        "body.SurfaceColorObstruction",
        "body.AppearancePaletteObstruction",
        "body.SurfaceDetailObstruction",
    )
    assert tuple(diagnostic.address for diagnostic in diagnostics) == (
        "meta@plate_policy.relief",
        "meta@plate_policy.cell_count",
        "meta@surface_color.tint_linear_rgb[1]",
        "meta@appearance_palette.obsidian_warden",
        "meta@surface_detail.scale",
    )
    assert tuple(
        (diagnostic.required, diagnostic.observed)
        for diagnostic in diagnostics
    ) == (
        ({"finite": True, "minimum": 0.0}, -0.1),
        ({"maximum": 4}, 8),
        (
            {
                "minimum": 0.0,
                "maximum": 1.0,
                "exclusive_minimum": False,
            },
            1.2,
        ),
        (("neutral_gray",), "obsidian_warden"),
        ({"finite": True, "exclusive_minimum": 0.0}, 0.0),
    )
    assert all(
        set(diagnostic.contract_refs) == {"primer", "schema"}
        for diagnostic in diagnostics
    )


def test_sheaf_projection_names_each_quantitative_law_under_hydraulic_descent() -> None:
    diagnostics = _assembly_diagnostics(
        RejectedAssembly(
            (
                InvalidBalanceCellObstruction(
                    "thermal",
                    BalanceTermKind.CELL_SOURCE,
                    "metabolism",
                    5,
                    4,
                ),
                PhysicalHydraulicBalanceObstruction(
                    InvalidBalanceCoefficientObstruction(
                        "hydraulics",
                        BalanceTermKind.DIRECTED_TRANSPORT,
                        "artery",
                        0.0,
                    )
                ),
                NonFiniteBalanceValueObstruction(
                    "thermal",
                    BalanceTermKind.FIXED_BOUNDARY,
                    "ambient",
                    float("inf"),
                ),
                InvalidBalanceSolverConfigObstruction(
                    "thermal",
                    BalanceSolverParameter.MAXIMUM_ITERATION_FACTOR,
                    0.0,
                ),
                InvalidThermalParameterObstruction(
                    "thermal",
                    "skin",
                    ThermalParameter.SOLID_CONDUCTIVITY,
                    -1.0,
                ),
            )
        )
    )

    assert tuple(diagnostic.address for diagnostic in diagnostics) == (
        "thermal/cell_source:metabolism/cell:5",
        "hydraulics/directed_transport:artery",
        "thermal/fixed_boundary:ambient",
        "thermal/maximum_iteration_factor",
        "thermal/skin/solid_conductivity",
    )
    assert tuple(diagnostic.required for diagnostic in diagnostics) == (
        {"minimum": 0, "maximum": 3},
        {"finite": True, "exclusive_minimum": 0.0},
        {"finite": True},
        {"integer": True, "exclusive_minimum": 0},
        {"finite": True, "exclusive_minimum": 0.0},
    )
    assert diagnostics[1].code == "assembly.InvalidBalanceCoefficientObstruction"


def test_quantitative_assembly_carriers_project_without_fabricated_provenance() -> None:
    diagnostics = _assembly_diagnostics(
        RejectedAssembly(
            (
                ThermalCouplingNonConvergenceObstruction(
                    "body",
                    "hot",
                    25,
                    25,
                    0.01,
                    1.0e-8,
                ),
                InvalidPhysicalHydraulicNodePositionObstruction(
                    "pump",
                    (0.0, 1.0e308, 1.0),
                    10.0,
                ),
                NonFinitePhysicalHydraulicResultObstruction(
                    PhysicalHydraulicResultKind.EDGE_FLOW,
                    "artery",
                    float("inf"),
                ),
                InvalidConstitutivePropertiesObstruction(
                    "region:body/torso",
                    CellIndex(1, 2, 3),
                    "poisson_ratio",
                    0.8,
                ),
                InvalidMaterialFractionObstruction(
                    "region:body/torso",
                    CellIndex(1, 2, 3),
                    MaterialFractionRule.FULL_SOLID_VALUE,
                    0.5,
                    "full_solid",
                ),
                UnresolvedMaterialFractionObstruction(
                    "region:body/torso",
                    CellIndex(1, 2, 3),
                    0.5,
                ),
                InvalidBodyForceObstruction(
                    "meta@service.cases[0]",
                    (0.0, float("inf"), 0.0),
                ),
                InvalidAcceptanceCriteriaObstruction(
                    "maximum_displacement",
                    0.0,
                    "criterion must be positive",
                ),
                InvalidBucklingEigenpairObstruction(
                    0.0,
                    0.02,
                    1.0e-8,
                ),
                UnsupportedEmbeddedChannelExternalPressureObstruction(
                    "artery",
                    -0.1,
                    1.0e-6,
                ),
            )
        )
    )

    assert tuple(diagnostic.address for diagnostic in diagnostics) == (
        "assembly/body/hot/thermal",
        "pump",
        "artery",
        "region:body/torso",
        "region:body/torso",
        "region:body/torso",
        "meta@service.cases[0]",
        "assembly/mechanics/maximum_displacement",
        "assembly/mechanics/buckling",
        "artery",
    )
    assert diagnostics[0].required == {
        "converged": True,
        "maximum_iterations": 25,
        "maximum_state_delta": 1.0e-8,
    }
    assert diagnostics[1].required == {
        "world_coordinates": {"coordinate_count": 3, "finite": True},
        "metre_coordinates": {"coordinate_count": 3, "finite": True},
    }
    assert diagnostics[1].observed["metre_coordinates"][1] == float("inf")
    assert diagnostics[3].required == {
        "finite": True,
        "exclusive_minimum": -1.0,
        "exclusive_maximum": 0.5,
    }
    assert diagnostics[4].required == 1.0
    assert diagnostics[5].observed == {
        "resolution": "unresolved",
        "solid_fraction": 0.5,
    }
    assert diagnostics[8].required == {
        "load_factor": {"finite": True, "exclusive_minimum": 0.0},
        "normalized_eigen_residual": {
            "finite": True,
            "maximum": 1.0e-8,
        },
    }
    assert diagnostics[9].required == -1.0e-6
