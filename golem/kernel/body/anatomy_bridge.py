"""Lazy-import adapters into ``golem.kernel.anatomy`` (no eager anatomy load)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from golem.kernel.anatomy import (
        AnatomyObstruction,
        AnatomyResult,
        TissueEnvelopeSection,
    )


def _derive_anatomy_result(
    bones: dict[str, dict], order: tuple[str, ...] | list[str], payload: object
) -> "AnatomyResult":
    from golem.kernel.anatomy import derive_anatomy

    return derive_anatomy(bones, order, payload)


def _anatomy_view(result: "AnatomyResult | None") -> dict[str, object] | None:
    if result is None:
        return None
    from golem.kernel.anatomy import anatomy_to_dict

    return anatomy_to_dict(result)


def _declares_skin_formation(payload: object) -> bool:
    overall = payload.get("overall") if isinstance(payload, dict) else None
    integument = (
        overall.get("integument_layers")
        if isinstance(overall, dict)
        else None
    )
    layers = (
        tuple(integument)
        if isinstance(integument, (list, tuple))
        else (integument,)
    )
    return any(
        isinstance(layer, dict) and "formation" in layer
        for layer in layers
    )


def _skin_formation_anatomy_obstructions(
    result: "AnatomyResult | None",
    payload: object,
) -> tuple["AnatomyObstruction", ...]:
    from golem.kernel.anatomy import RejectedAnatomy

    return (
        result.obstructions
        if _declares_skin_formation(payload)
        and isinstance(result, RejectedAnatomy)
        else ()
    )


def _skin_layer_view(
    result: "AnatomyResult | None",
) -> dict[str, object] | None:
    from golem.kernel.anatomy import AcceptedAnatomy

    formed_layers = (
        tuple(
            layer
            for layer in result.overall.integument
            if layer.formation is not None
        )
        if isinstance(result, AcceptedAnatomy)
        else ()
    )
    representative = next(iter(formed_layers), None)
    formation = (
        representative.formation if representative is not None else None
    )
    return (
        {
            "formation": formation.value,
            "thickness": representative.thickness,
            "region_ids": tuple(layer.region_id for layer in formed_layers),
        }
        if representative is not None and formation is not None
        else None
    )


def _tissue_envelopes_by_address(
    result: "AnatomyResult | None",
) -> dict[str, "TissueEnvelopeSection"]:
    from golem.kernel.anatomy import AcceptedAnatomy

    return (
        {
            envelope.shape_address: envelope
            for envelope in result.tissue_envelopes
        }
        if isinstance(result, AcceptedAnatomy)
        else {}
    )
