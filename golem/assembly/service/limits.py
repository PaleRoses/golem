"""Manufacturing, operating, and evidence-policy section decoders."""

from __future__ import annotations

from golem.assembly.service.model import (
    ChannelResolutionPolicy,
    ManufacturingLimits,
    OperatingLimits,
    PhysicsEvidencePolicy,
)
from golem.assembly.service.outcome import MalformedServiceIntentObstruction
from golem.assembly.service.primitives import (
    _Decoded,
    _DecodeResult,
    _bounded_fraction,
    _exact_integer,
    _minimum_integer,
    _positive_budget,
    _record,
    _result_obstructions,
)


_CHANNEL_RESOLUTION_POLICY_TOKENS = tuple(
    policy.value for policy in ChannelResolutionPolicy
)


def _decode_manufacturing(value: object) -> _DecodeResult[ManufacturingLimits]:
    path = "/service/manufacturing"
    keys = frozenset(
        (
            "minimum_lumen_diameter_m",
            "minimum_wall_thickness_m",
            "minimum_remaining_ligament_m",
        )
    )
    obj = _record(value, path, keys)
    if not isinstance(obj, _Decoded):
        return obj
    lumen = _positive_budget(
        obj.value["minimum_lumen_diameter_m"],
        f"{path}/minimum_lumen_diameter_m",
    )
    wall = _positive_budget(
        obj.value["minimum_wall_thickness_m"],
        f"{path}/minimum_wall_thickness_m",
    )
    ligament = _positive_budget(
        obj.value["minimum_remaining_ligament_m"],
        f"{path}/minimum_remaining_ligament_m",
    )
    obstructions = _result_obstructions((lumen, wall, ligament))
    return (
        obstructions
        if obstructions
        else _Decoded(ManufacturingLimits(lumen.value, wall.value, ligament.value))
    )


def _decode_limits(value: object) -> _DecodeResult[OperatingLimits]:
    path = "/service/limits"
    field_names = (
        "maximum_temperature_k",
        "maximum_temperature_gradient_k_per_m",
        "maximum_displacement_m",
        "minimum_yield_safety_factor",
        "minimum_buckling_safety_factor",
        "minimum_burst_safety_factor",
    )
    obj = _record(value, path, frozenset(field_names))
    if not isinstance(obj, _Decoded):
        return obj
    decoded = tuple(
        _positive_budget(obj.value[field_name], f"{path}/{field_name}")
        for field_name in field_names
    )
    obstructions = _result_obstructions(decoded)
    values = tuple(
        result.value for result in decoded if isinstance(result, _Decoded)
    )
    return (
        obstructions
        if obstructions
        else _Decoded(OperatingLimits(*values))
    )


def _decode_evidence_policy(value: object) -> _DecodeResult[PhysicsEvidencePolicy]:
    path = "/service/evidence"
    keys = frozenset(
        (
            "channel_resolution_policy",
            "physics_resolution",
            "minimum_cells_across_lumen",
            "resolution_refinement_factor",
            "maximum_refinement_change_fraction",
        )
    )
    obj = _record(value, path, keys)
    if not isinstance(obj, _Decoded):
        return obj
    channel_resolution_policy: _DecodeResult[ChannelResolutionPolicy] = (
        _Decoded(ChannelResolutionPolicy(obj.value["channel_resolution_policy"]))
        if isinstance(obj.value["channel_resolution_policy"], str)
        and obj.value["channel_resolution_policy"]
        in _CHANNEL_RESOLUTION_POLICY_TOKENS
        else (
            MalformedServiceIntentObstruction(
                f"{path}/channel_resolution_policy",
                (
                    "expected one of "
                    f"{_CHANNEL_RESOLUTION_POLICY_TOKENS!r}"
                ),
            ),
        )
    )
    physics_resolution = _minimum_integer(
        obj.value["physics_resolution"], f"{path}/physics_resolution", 12
    )
    minimum_cells = _minimum_integer(
        obj.value["minimum_cells_across_lumen"],
        f"{path}/minimum_cells_across_lumen",
        4,
    )
    refinement_factor = _exact_integer(
        obj.value["resolution_refinement_factor"],
        f"{path}/resolution_refinement_factor",
        2,
    )
    refinement_change = _bounded_fraction(
        obj.value["maximum_refinement_change_fraction"],
        f"{path}/maximum_refinement_change_fraction",
        0.05,
    )
    obstructions = _result_obstructions(
        (
            channel_resolution_policy,
            physics_resolution,
            minimum_cells,
            refinement_factor,
            refinement_change,
        )
    )
    return (
        obstructions
        if obstructions
        else _Decoded(
            PhysicsEvidencePolicy(
                channel_resolution_policy=channel_resolution_policy.value,
                physics_resolution=physics_resolution.value,
                minimum_cells_across_lumen=minimum_cells.value,
                resolution_refinement_factor=refinement_factor.value,
                maximum_refinement_change_fraction=refinement_change.value,
            )
        )
    )
