"""Typed operating-intent boundary for coupled assembly service."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from golem.assembly.service import (
    AcceptedSemanticAddress,
    AcceptedServiceIntent,
    AmbientHeatTransferPolicy,
    BoneAddress,
    ChannelResolutionPolicy,
    ContradictorySupportObstruction,
    ElementAddress,
    EvidenceObligation,
    ForceLoad,
    InvalidSemanticAddressKindObstruction,
    JointLaw,
    MaterialValidityObstruction,
    MissingHeatSinkObstruction,
    MissingUnitScaleObstruction,
    MountAddress,
    NonPositiveBudgetObstruction,
    PhysicalEvidenceStatus,
    PortAddress,
    RegionAddress,
    RejectedSemanticAddress,
    RejectedServiceIntent,
    ServiceIntent,
    UnknownPhysicalMaterialObstruction,
    UnknownSemanticAddressObstruction,
    UnsupportedConstitutiveModelObstruction,
    VisualOnlyServiceIntent,
    decode_semantic_address,
    decode_service_intent,
    semantic_address_token,
)


KNOWN_ADDRESSES = frozenset(
    (
        ElementAddress("body"),
        ElementAddress("sword"),
        RegionAddress("body", "torso"),
        RegionAddress("body", "vascular_wall"),
        BoneAddress("body", "root"),
        BoneAddress("body", "right_hand"),
        BoneAddress("body", "right_elbow"),
        PortAddress("body", "pump_outlet"),
        MountAddress("sword"),
    )
)


def _valid_service() -> dict[str, object]:
    return {
        "metres_per_world_unit": 1.0,
        "constitutive_model": "isotropic_linear_thermoelastic",
        "solid_materials": [
            {
                "role": "structure",
                "domain": "element:body",
                "material": "stainless_steel_316l_room_temperature",
            },
            {
                "role": "lumen_wall",
                "domain": "region:body/vascular_wall",
                "material": "stainless_steel_316l_room_temperature",
            },
            {
                "role": "equipment",
                "domain": "element:sword",
                "material": "stainless_steel_316l_room_temperature",
            },
        ],
        "cases": [
            {
                "id": "greatsword_gravity",
                "gravity_m_s2": [0.0, -9.80665, 0.0],
                "inertial_acceleration_m_s2": [0.0, 0.0, 0.0],
                "loads": [
                    {
                        "kind": "force",
                        "address": "bone:body/right_hand",
                        "vector_newtons": [0.0, -180.0, 0.0],
                    }
                ],
                "supports": [
                    {"address": "bone:body/root", "law": "fixed"}
                ],
                "joints": [
                    {"address": "bone:body/right_elbow", "law": "revolute"}
                ],
                "heat_sources": [
                    {"address": "region:body/torso", "watts": 140.0}
                ],
            }
        ],
        "ambient": {
            "temperature_k": 293.15,
            "heat_transfer_policy": "still_air_natural_convection",
            "gas_material": "dry_air_101325_pa_298k",
            "pressure_pa": 101_325.0,
            "natural_convection": {
                "correlation": "churchill_chu_isothermal_vertical_plate",
                "surface_orientation": "vertical_plate",
                "characteristic_length_m": 1.8,
            },
        },
        "coolant": {
            "material": "water_0_1_mpa_293k",
            "inlet_temperature_k": 293.15,
            "pump": {
                "kind": "pressure_budget",
                "maximum_pressure_pa": 250_000.0,
            },
        },
        "manufacturing": {
            "minimum_lumen_diameter_m": 4.0e-5,
            "minimum_wall_thickness_m": 2.0e-5,
            "minimum_remaining_ligament_m": 6.0e-5,
        },
        "limits": {
            "maximum_temperature_k": 350.0,
            "maximum_temperature_gradient_k_per_m": 8000.0,
            "maximum_displacement_m": 0.004,
            "minimum_yield_safety_factor": 1.5,
            "minimum_buckling_safety_factor": 1.5,
            "minimum_burst_safety_factor": 2.0,
        },
        "evidence": {
            "channel_resolution_policy": "resolved_only",
            "physics_resolution": 24,
            "minimum_cells_across_lumen": 6,
            "resolution_refinement_factor": 2,
            "maximum_refinement_change_fraction": 0.05,
        },
    }


def _replace_service_field(
    payload: dict[str, object], field: str, value: object
) -> dict[str, object]:
    return {**payload, field: value}


def _replace_case_field(
    payload: dict[str, object], field: str, value: object
) -> dict[str, object]:
    case = payload["cases"][0]
    return {
        **payload,
        "cases": [{**case, field: value}],
    }


def test_valid_service_decodes_to_frozen_typed_intent():
    result = decode_service_intent(_valid_service(), KNOWN_ADDRESSES)
    assert isinstance(result, AcceptedServiceIntent)
    assert isinstance(result.intent, ServiceIntent)
    intent = result.intent
    assert intent.physical_evidence is PhysicalEvidenceStatus.REQUESTED
    assert isinstance(intent.cases[0].loads[0], ForceLoad)
    assert intent.cases[0].loads[0].address == BoneAddress("body", "right_hand")
    assert intent.evidence_policy.required_obligations == frozenset(EvidenceObligation)
    assert (
        intent.evidence_policy.channel_resolution_policy
        is ChannelResolutionPolicy.RESOLVED_ONLY
    )
    assert intent.evidence_policy.physics_resolution == 24
    assert intent.ambient is not None
    assert (
        intent.ambient.heat_transfer_policy
        is AmbientHeatTransferPolicy.STILL_AIR_NATURAL_CONVECTION
    )
    assert intent.ambient.gas.material_id.value == "dry_air_101325_pa_298k"
    assert intent.ambient.pressure_pascal == 101_325.0
    assert (
        intent.ambient.natural_convection_correlation.characteristic_length_metres
        == 1.8
    )
    assert intent.evidence_policy.maximum_physics_pitch_metres(
        intent.manufacturing
    ) == pytest.approx(4.0e-5 / 6.0)
    with pytest.raises(FrozenInstanceError):
        intent.scale.metres_per_world_unit = 2.0


def test_absent_service_is_explicit_visual_only_evidence():
    result = decode_service_intent(None)
    assert result == AcceptedServiceIntent(VisualOnlyServiceIntent())
    assert result.intent.physical_evidence is PhysicalEvidenceStatus.NOT_REQUESTED


@pytest.mark.parametrize(
    "address",
    (
        ElementAddress("body"),
        RegionAddress("body", "torso"),
        BoneAddress("body", "right_hand"),
        PortAddress("body", "pump_outlet"),
        MountAddress("sword"),
    ),
)
def test_semantic_address_tokens_round_trip(address):
    result = decode_semantic_address(semantic_address_token(address))
    assert result == AcceptedSemanticAddress(address)


@pytest.mark.parametrize(
    "value",
    ("", "bone:body", "element:body/extra", "solver_node:body/42", 17, []),
)
def test_semantic_address_decoder_is_total(value):
    assert isinstance(decode_semantic_address(value), RejectedSemanticAddress)


def test_missing_scale_is_a_specific_obstruction():
    service = _valid_service()
    result = decode_service_intent(
        {key: value for key, value in service.items() if key != "metres_per_world_unit"},
        KNOWN_ADDRESSES,
    )
    assert isinstance(result, RejectedServiceIntent)
    assert any(isinstance(item, MissingUnitScaleObstruction) for item in result.obstructions)


def test_unknown_address_and_physical_material_accumulate():
    service = _valid_service()
    bad_materials = [
        {
            **service["solid_materials"][0],
            "material": "neutral_gray",
        }
    ]
    bad_loads = [
        {
            **service["cases"][0]["loads"][0],
            "address": "bone:body/absent_hand",
        }
    ]
    malformed = _replace_case_field(
        _replace_service_field(service, "solid_materials", bad_materials),
        "loads",
        bad_loads,
    )
    result = decode_service_intent(malformed, KNOWN_ADDRESSES)
    assert isinstance(result, RejectedServiceIntent)
    assert any(
        isinstance(item, UnknownPhysicalMaterialObstruction)
        for item in result.obstructions
    )
    assert any(
        isinstance(item, UnknownSemanticAddressObstruction)
        for item in result.obstructions
    )


def test_contradictory_support_laws_reject_the_case():
    service = _valid_service()
    contradictory = _replace_case_field(
        service,
        "supports",
        [
            {"address": "bone:body/root", "law": "fixed"},
            {"address": "bone:body/root", "law": "roller"},
        ],
    )
    result = decode_service_intent(contradictory, KNOWN_ADDRESSES)
    assert isinstance(result, RejectedServiceIntent)
    assert any(
        isinstance(item, ContradictorySupportObstruction)
        for item in result.obstructions
    )


def test_rigid_payload_is_a_first_class_joint_law():
    result = decode_service_intent(
        _replace_case_field(
            _valid_service(),
            "joints",
            [{"address": "mount:sword", "law": "rigid_payload"}],
        ),
        KNOWN_ADDRESSES,
    )
    assert isinstance(result, AcceptedServiceIntent)
    assert isinstance(result.intent, ServiceIntent)
    assert result.intent.cases[0].joints[0].law is JointLaw.RIGID_PAYLOAD


def test_heat_generation_without_any_sink_rejects():
    service = _replace_service_field(_valid_service(), "ambient", None)
    result = decode_service_intent(
        _replace_service_field(service, "coolant", None), KNOWN_ADDRESSES
    )
    assert isinstance(result, RejectedServiceIntent)
    assert result.obstructions == (MissingHeatSinkObstruction("greatsword_gravity"),)


def test_unsupported_model_and_non_positive_pump_budget_accumulate():
    service = _valid_service()
    bad_coolant = {
        **service["coolant"],
        "pump": {"kind": "power_budget", "maximum_power_w": 0.0},
    }
    result = decode_service_intent(
        {
            **service,
            "constitutive_model": "neo_hookean",
            "coolant": bad_coolant,
        },
        KNOWN_ADDRESSES,
    )
    assert isinstance(result, RejectedServiceIntent)
    assert any(
        isinstance(item, UnsupportedConstitutiveModelObstruction)
        for item in result.obstructions
    )
    assert any(
        isinstance(item, NonPositiveBudgetObstruction)
        for item in result.obstructions
    )


@pytest.mark.parametrize(
    "evidence",
    (
        {
            "minimum_cells_across_lumen": 3,
            "resolution_refinement_factor": 2,
            "maximum_refinement_change_fraction": 0.05,
        },
        {
            "minimum_cells_across_lumen": 6,
            "resolution_refinement_factor": 3,
            "maximum_refinement_change_fraction": 0.05,
        },
        {
            "minimum_cells_across_lumen": 6,
            "resolution_refinement_factor": 2,
            "maximum_refinement_change_fraction": 0.051,
        },
    ),
)
def test_evidence_policy_cannot_weaken_resolution_gate(evidence):
    result = decode_service_intent(
        _replace_service_field(_valid_service(), "evidence", evidence),
        KNOWN_ADDRESSES,
    )
    assert isinstance(result, RejectedServiceIntent)


@pytest.mark.parametrize(
    "policy",
    (
        ChannelResolutionPolicy.RESOLVED_ONLY,
        ChannelResolutionPolicy.RESOLVED_OR_EMBEDDED_SLENDER,
    ),
)
def test_channel_resolution_policy_is_closed_and_explicit(policy):
    evidence = {
        **_valid_service()["evidence"],
        "channel_resolution_policy": policy.value,
    }
    result = decode_service_intent(
        _replace_service_field(_valid_service(), "evidence", evidence),
        KNOWN_ADDRESSES,
    )
    assert isinstance(result, AcceptedServiceIntent)
    assert isinstance(result.intent, ServiceIntent)
    assert result.intent.evidence_policy.channel_resolution_policy is policy


@pytest.mark.parametrize("physics_resolution", (11, 12.5, True))
def test_physics_resolution_is_an_explicit_integer_gate(physics_resolution):
    evidence = {
        **_valid_service()["evidence"],
        "physics_resolution": physics_resolution,
    }
    result = decode_service_intent(
        _replace_service_field(_valid_service(), "evidence", evidence),
        KNOWN_ADDRESSES,
    )
    assert isinstance(result, RejectedServiceIntent)


def test_ambient_gas_and_correlation_fail_closed_applicatively():
    ambient = {
        **_valid_service()["ambient"],
        "gas_material": "visual_gray_is_not_air",
        "natural_convection": {
            **_valid_service()["ambient"]["natural_convection"],
            "correlation": "ambient_h_equals_eight",
        },
    }
    result = decode_service_intent(
        _replace_service_field(_valid_service(), "ambient", ambient),
        KNOWN_ADDRESSES,
    )
    assert isinstance(result, RejectedServiceIntent)
    assert any(
        isinstance(item, UnknownPhysicalMaterialObstruction)
        for item in result.obstructions
    )
    assert any(
        getattr(item, "address", "")
        == "/service/ambient/natural_convection/correlation"
        for item in result.obstructions
    )


def test_address_kind_is_checked_after_address_resolution():
    result = decode_service_intent(
        _replace_case_field(
            _valid_service(),
            "heat_sources",
            [{"address": "bone:body/right_hand", "watts": 20.0}],
        ),
        KNOWN_ADDRESSES,
    )
    assert isinstance(result, RejectedServiceIntent)
    assert any(
        isinstance(item, InvalidSemanticAddressKindObstruction)
        for item in result.obstructions
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "coolant",
            {
                "material": "water_0_1_mpa_293k",
                "inlet_temperature_k": 330.0,
                "pump": {
                    "kind": "pressure_budget",
                    "maximum_pressure_pa": 250_000.0,
                },
            },
        ),
        (
            "limits",
            {
                **_valid_service()["limits"],
                "maximum_temperature_k": 400.0,
            },
        ),
        (
            "ambient",
            {
                **_valid_service()["ambient"],
                "pressure_pa": 120_000.0,
            },
        ),
    ),
)
def test_material_validity_intervals_fail_closed(field, value):
    result = decode_service_intent(
        _replace_service_field(_valid_service(), field, value), KNOWN_ADDRESSES
    )
    assert isinstance(result, RejectedServiceIntent)
    assert any(
        isinstance(item, MaterialValidityObstruction)
        for item in result.obstructions
    )


@pytest.mark.parametrize(
    "payload",
    (0, 3.14, "service", [], (), {"metres_per_world_unit": 1.0, 3: "bad"}),
)
def test_service_decoder_is_total_over_malformed_inputs(payload):
    assert isinstance(decode_service_intent(payload), RejectedServiceIntent)


def test_source_payload_is_not_mutated_by_decoding():
    service = _valid_service()
    before = deepcopy(service)
    decode_service_intent(service, KNOWN_ADDRESSES)
    assert service == before


def test_pump_pressure_budget_is_a_differential_not_absolute_pressure():
    service = _valid_service()
    coolant = {
        **service["coolant"],
        "pump": {
            "kind": "pressure_budget",
            "maximum_pressure_pa": 1.0e-7,
        },
    }
    result = decode_service_intent(
        {**service, "coolant": coolant}, KNOWN_ADDRESSES
    )
    assert isinstance(result, AcceptedServiceIntent)
