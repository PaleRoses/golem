"""Applicative decoder for the closed plate policy."""

from __future__ import annotations

import math

from golem.materials import AcceptedMaterial, decode_appearance_material
from golem.plates.core import (
    AcceptedPlatePolicy,
    PlateLayout,
    PlatePolicy,
    PlatePolicyObstruction,
    PlatePolicyRule,
    PlatePolicyResult,
    RejectedPlatePolicy,
)


_PLATE_LAYOUT_TOKENS = tuple(layout.value for layout in PlateLayout)


def _policy_check(
    valid: bool,
    address: str,
    rule: PlatePolicyRule,
    authored: object,
    required: object,
) -> tuple[PlatePolicyObstruction, ...]:
    return (
        ()
        if valid
        else (PlatePolicyObstruction(address, rule, authored, required),)
    )


def decode_plate_policy(payload: object) -> PlatePolicyResult:
    if not isinstance(payload, dict):
        return RejectedPlatePolicy(
            (
                PlatePolicyObstruction(
                    "/",
                    PlatePolicyRule.POLICY_OBJECT,
                    payload,
                    "object",
                ),
            )
        )
    layout_value = payload.get("layout", PlateLayout.SURFACE_VORONOI.value)
    cell_count = payload.get("cell_count")
    random_seed = payload.get("random_seed", 7)
    relief = payload.get("relief", 0.045)
    groove = payload.get("groove", 0.02)
    seam_lift = payload.get("seam_lift", 0.002)
    seam_material = payload.get("seam_material", "emissive_seam")
    axis_weight = payload.get("axis_weight", 1.0)
    circulation_emission = payload.get("circulation_emission", False)
    numeric_obstructions = tuple(
        PlatePolicyObstruction(
            address,
            PlatePolicyRule.FINITE_NON_NEGATIVE,
            value,
            {"finite": True, "minimum": 0.0},
        )
        for address, value in (
            ("/relief", relief),
            ("/groove", groove),
            ("/seam_lift", seam_lift),
        )
        if not _is_finite_non_negative_number(value)
    )
    obstructions = (
        *_policy_check(
            layout_value in _PLATE_LAYOUT_TOKENS,
            "/layout",
            PlatePolicyRule.LAYOUT,
            layout_value,
            _PLATE_LAYOUT_TOKENS,
        ),
        *_policy_check(
            isinstance(cell_count, int)
            and not isinstance(cell_count, bool)
            and 4 <= cell_count <= 512,
            "/cell_count",
            PlatePolicyRule.CELL_COUNT,
            cell_count,
            {"integer": True, "minimum": 4, "maximum": 512},
        ),
        *_policy_check(
            isinstance(random_seed, int) and not isinstance(random_seed, bool),
            "/random_seed",
            PlatePolicyRule.RANDOM_SEED,
            random_seed,
            {"integer": True},
        ),
        *numeric_obstructions,
        *_policy_check(
            isinstance(decode_appearance_material(seam_material), AcceptedMaterial),
            "/seam_material",
            PlatePolicyRule.APPEARANCE_MATERIAL,
            seam_material,
            {"registered": True},
        ),
        *_policy_check(
            _is_finite_positive_number(axis_weight),
            "/axis_weight",
            PlatePolicyRule.FINITE_POSITIVE,
            axis_weight,
            {"finite": True, "exclusive_minimum": 0.0},
        ),
        *_policy_check(
            isinstance(circulation_emission, bool),
            "/circulation_emission",
            PlatePolicyRule.BOOLEAN,
            circulation_emission,
            "boolean",
        ),
    )
    return (
        RejectedPlatePolicy(obstructions)
        if obstructions
        else AcceptedPlatePolicy(
            PlatePolicy(
                layout=PlateLayout(str(layout_value)),
                cell_count=int(cell_count),
                random_seed=int(random_seed),
                relief=float(relief),
                groove=float(groove),
                seam_lift=float(seam_lift),
                seam_material=str(seam_material),
                axis_weight=float(axis_weight),
                circulation_emission=bool(circulation_emission),
            )
        )
    )


def _is_finite_non_negative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _is_finite_positive_number(value: object) -> bool:
    return _is_finite_non_negative_number(value) and float(value) > 0.0
