"""Evaluate mixed-dimensional embedded-channel mechanics evidence."""

from __future__ import annotations

from math import fsum, inf, pi
from typing import TYPE_CHECKING

from ..model import (
    IsotropicConstitutiveProperties,
    MaterialFractionResolution,
    UnevaluatedPhysics,
)
from .model import (
    AcceptedEmbeddedChannelMechanics,
    EmbeddedChannelConstitutiveResponse,
    EmbeddedChannelCriteria,
    EmbeddedChannelMechanicsReceipt,
    EmbeddedChannelMechanicsResult,
    EmbeddedChannelObstruction,
    EmbeddedChannelSectionMechanics,
    FullSolidEmbeddedChannelResponse,
    HomogenizedEmbeddedChannelResponse,
    RejectedEmbeddedChannelMechanics,
    UnresolvedEmbeddedChannelConstitutiveObstruction,
    _capsule_volume,
    _vascular_edge_length_world,
)
from .validate import (
    _embedded_channel_burst_margin_obstructions,
    _embedded_channel_external_pressure_support_obstructions,
    _embedded_channel_input_obstructions,
    _embedded_channel_scale_separation_obstructions,
    _embedded_channel_volume_fraction_obstructions,
)

if TYPE_CHECKING:
    from golem.kernel.anatomy import (
        AcceptedVasculature,
        PhysicalHydraulicNodePressure,
        VascularEdge,
        VascularNode,
    )


def evaluate_embedded_channel_mechanics(
    vasculature: "AcceptedVasculature",
    node_pressures: tuple["PhysicalHydraulicNodePressure", ...],
    *,
    metres_per_world_unit: float,
    grid_pitch_metres: float,
    host_volume_cubic_metres: float,
    wall_thickness_metres: float,
    wall_allowable_stress_pascal: float,
    external_pressure_pascal: float,
    background_properties: IsotropicConstitutiveProperties,
    material_fraction_resolution: MaterialFractionResolution = (
        MaterialFractionResolution.UNRESOLVED
    ),
    homogenized_properties: IsotropicConstitutiveProperties | None = None,
    homogenization_validated_lumen_volume_fraction: float | None = None,
    criteria: EmbeddedChannelCriteria = EmbeddedChannelCriteria(),
) -> EmbeddedChannelMechanicsResult:
    """Certify subgrid geometry and pressure without inventing voxel anatomy.

    ``AcceptedVasculature`` remains the topology/geometry owner.  This function
    descends each canonical edge to an analytic capsule and thick cylinder,
    glues only additive upper-volume bounds, and keeps the Q1 background-solid
    response separate.  A nonzero channel obtains global constitutive evidence
    only from caller-supplied, already-homogenized properties with an explicit
    validated porosity range; otherwise that global claim remains a typed
    obstruction inside the accepted local receipt.
    """

    input_obstructions = _embedded_channel_input_obstructions(
        vasculature,
        node_pressures,
        metres_per_world_unit=metres_per_world_unit,
        grid_pitch_metres=grid_pitch_metres,
        host_volume_cubic_metres=host_volume_cubic_metres,
        wall_thickness_metres=wall_thickness_metres,
        wall_allowable_stress_pascal=wall_allowable_stress_pascal,
        external_pressure_pascal=external_pressure_pascal,
        background_properties=background_properties,
        material_fraction_resolution=material_fraction_resolution,
        homogenized_properties=homogenized_properties,
        homogenization_validated_lumen_volume_fraction=(
            homogenization_validated_lumen_volume_fraction
        ),
        criteria=criteria,
    )
    if input_obstructions:
        return RejectedEmbeddedChannelMechanics(input_obstructions)
    graph = vasculature.graph
    node_by_id = graph.node_by_id
    pressure_by_node_id = {
        pressure.node_id: pressure.pressure_pascal
        for pressure in node_pressures
    }
    ordered_edges = tuple(sorted(graph.edges, key=lambda edge: edge.edge_id))
    external_pressure_obstructions = (
        _embedded_channel_external_pressure_support_obstructions(
            ordered_edges,
            pressure_by_node_id,
            external_pressure_pascal=external_pressure_pascal,
            normalization_floor=criteria.normalization_floor,
        )
    )
    if external_pressure_obstructions:
        return RejectedEmbeddedChannelMechanics(
            external_pressure_obstructions
        )
    sections = _embedded_channel_sections(
        ordered_edges,
        node_by_id,
        pressure_by_node_id,
        metres_per_world_unit=metres_per_world_unit,
        wall_thickness_metres=wall_thickness_metres,
        wall_allowable_stress_pascal=wall_allowable_stress_pascal,
        external_pressure_pascal=external_pressure_pascal,
        normalization_floor=criteria.normalization_floor,
    )
    (
        lumen_volume_upper_bound,
        wall_volume_upper_bound,
        lumen_volume_fraction_upper_bound,
        channel_envelope_volume_fraction_upper_bound,
        maximum_outer_diameter_to_grid_pitch,
        minimum_burst_safety_factor,
    ) = _embedded_channel_aggregate_metrics(
        sections,
        host_volume_cubic_metres=host_volume_cubic_metres,
        grid_pitch_metres=grid_pitch_metres,
    )
    validity_obstructions: tuple[EmbeddedChannelObstruction, ...] = (
        _embedded_channel_scale_separation_obstructions(
            maximum_outer_diameter_to_grid_pitch,
            criteria.maximum_outer_diameter_to_grid_pitch,
        )
        + _embedded_channel_volume_fraction_obstructions(
            lumen_volume_fraction_upper_bound,
            channel_envelope_volume_fraction_upper_bound,
            maximum_lumen_volume_fraction=(
                criteria.maximum_lumen_volume_fraction
            ),
            maximum_channel_envelope_volume_fraction=(
                criteria.maximum_channel_envelope_volume_fraction
            ),
        )
        + _embedded_channel_burst_margin_obstructions(
            minimum_burst_safety_factor,
            criteria.minimum_burst_safety_factor,
        )
    )
    if validity_obstructions:
        return RejectedEmbeddedChannelMechanics(validity_obstructions)
    constitutive_response = _embedded_channel_constitutive_response(
        positive_radius_edge_count=sum(
            section.radius_metres > 0.0 for section in sections
        ),
        lumen_volume_fraction_upper_bound=(
            lumen_volume_fraction_upper_bound
        ),
        background_properties=background_properties,
        material_fraction_resolution=material_fraction_resolution,
        homogenized_properties=homogenized_properties,
        homogenization_validated_lumen_volume_fraction=(
            homogenization_validated_lumen_volume_fraction
        ),
        normalization_floor=criteria.normalization_floor,
    )
    not_evaluated = _embedded_channel_unevaluated_physics(
        constitutive_response
    )
    return AcceptedEmbeddedChannelMechanics(
        sections=sections,
        receipt=_embedded_channel_mechanics_receipt(
            sections,
            grid_pitch_metres=grid_pitch_metres,
            host_volume_cubic_metres=host_volume_cubic_metres,
            wall_thickness_metres=wall_thickness_metres,
            wall_allowable_stress_pascal=wall_allowable_stress_pascal,
            external_pressure_pascal=external_pressure_pascal,
            maximum_outer_diameter_to_grid_pitch=(
                maximum_outer_diameter_to_grid_pitch
            ),
            lumen_volume_upper_bound=lumen_volume_upper_bound,
            wall_volume_upper_bound=wall_volume_upper_bound,
            lumen_volume_fraction_upper_bound=(
                lumen_volume_fraction_upper_bound
            ),
            channel_envelope_volume_fraction_upper_bound=(
                channel_envelope_volume_fraction_upper_bound
            ),
            minimum_burst_safety_factor=minimum_burst_safety_factor,
            constitutive_response=constitutive_response,
            not_evaluated=not_evaluated,
        ),
    )


def _embedded_channel_sections(
    ordered_edges: tuple["VascularEdge", ...],
    node_by_id: dict[str, "VascularNode"],
    pressure_by_node_id: dict[str, float],
    *,
    metres_per_world_unit: float,
    wall_thickness_metres: float,
    wall_allowable_stress_pascal: float,
    external_pressure_pascal: float,
    normalization_floor: float,
) -> tuple[EmbeddedChannelSectionMechanics, ...]:
    return tuple(
        _embedded_channel_section_mechanics(
            edge,
            node_by_id,
            pressure_by_node_id,
            metres_per_world_unit=metres_per_world_unit,
            wall_thickness_metres=wall_thickness_metres,
            wall_allowable_stress_pascal=wall_allowable_stress_pascal,
            external_pressure_pascal=external_pressure_pascal,
            normalization_floor=normalization_floor,
        )
        for edge in ordered_edges
    )


def _embedded_channel_aggregate_metrics(
    sections: tuple[EmbeddedChannelSectionMechanics, ...],
    *,
    host_volume_cubic_metres: float,
    grid_pitch_metres: float,
) -> tuple[float, float, float, float, float, float]:
    lumen_volume_upper_bound = fsum(
        section.lumen_capsule_volume_upper_bound_cubic_metres
        for section in sections
    )
    wall_volume_upper_bound = fsum(
        section.wall_annulus_volume_upper_bound_cubic_metres
        for section in sections
    )
    lumen_volume_fraction_upper_bound = (
        lumen_volume_upper_bound / host_volume_cubic_metres
    )
    channel_envelope_volume_fraction_upper_bound = (
        lumen_volume_upper_bound + wall_volume_upper_bound
    ) / host_volume_cubic_metres
    maximum_outer_diameter_to_grid_pitch = max(
        (
            2.0 * section.outer_radius_metres / grid_pitch_metres
            for section in sections
        ),
        default=0.0,
    )
    minimum_burst_safety_factor = min(
        (section.burst_safety_factor for section in sections),
        default=inf,
    )
    return (
        lumen_volume_upper_bound,
        wall_volume_upper_bound,
        lumen_volume_fraction_upper_bound,
        channel_envelope_volume_fraction_upper_bound,
        maximum_outer_diameter_to_grid_pitch,
        minimum_burst_safety_factor,
    )


def _embedded_channel_mechanics_receipt(
    sections: tuple[EmbeddedChannelSectionMechanics, ...],
    *,
    grid_pitch_metres: float,
    host_volume_cubic_metres: float,
    wall_thickness_metres: float,
    wall_allowable_stress_pascal: float,
    external_pressure_pascal: float,
    maximum_outer_diameter_to_grid_pitch: float,
    lumen_volume_upper_bound: float,
    wall_volume_upper_bound: float,
    lumen_volume_fraction_upper_bound: float,
    channel_envelope_volume_fraction_upper_bound: float,
    minimum_burst_safety_factor: float,
    constitutive_response: EmbeddedChannelConstitutiveResponse,
    not_evaluated: tuple[UnevaluatedPhysics, ...],
) -> EmbeddedChannelMechanicsReceipt:
    return EmbeddedChannelMechanicsReceipt(
        geometry_model=(
            "canonical accepted vascular centerlines; additive capsule "
            "volumes conservatively overcount overlaps at junctions"
        ),
        pressure_containment_model=(
            "maximum endpoint transmural pressure; p*r/t thin-wall limit "
            "with Lamé inner-wall hoop-stress correction"
        ),
        background_solid_model=(
            "solve_mechanics owns the separate Q1 background-solid "
            "response; no channel radius is inflated into that grid"
        ),
        edge_count=len(sections),
        positive_radius_edge_count=sum(
            section.radius_metres > 0.0 for section in sections
        ),
        grid_pitch_metres=grid_pitch_metres,
        host_volume_cubic_metres=host_volume_cubic_metres,
        wall_thickness_metres=wall_thickness_metres,
        wall_allowable_stress_pascal=wall_allowable_stress_pascal,
        external_pressure_pascal=external_pressure_pascal,
        maximum_outer_diameter_to_grid_pitch=(
            maximum_outer_diameter_to_grid_pitch
        ),
        lumen_volume_upper_bound_cubic_metres=(
            lumen_volume_upper_bound
        ),
        wall_volume_upper_bound_cubic_metres=wall_volume_upper_bound,
        lumen_volume_fraction_upper_bound=(
            lumen_volume_fraction_upper_bound
        ),
        channel_envelope_volume_fraction_upper_bound=(
            channel_envelope_volume_fraction_upper_bound
        ),
        conservative_solid_fraction_lower_bound=max(
            0.0,
            1.0 - lumen_volume_fraction_upper_bound,
        ),
        maximum_thin_wall_hoop_stress_pascal=max(
            (
                section.thin_wall_hoop_stress_pascal
                for section in sections
            ),
            default=0.0,
        ),
        maximum_lame_inner_wall_hoop_stress_pascal=max(
            (
                section.lame_inner_wall_hoop_stress_pascal
                for section in sections
            ),
            default=0.0,
        ),
        minimum_burst_safety_factor=minimum_burst_safety_factor,
        constitutive_response=constitutive_response,
        not_evaluated=not_evaluated,
    )


def _embedded_channel_section_mechanics(
    edge: "VascularEdge",
    node_by_id: dict[str, "VascularNode"],
    pressure_by_node_id: dict[str, float],
    *,
    metres_per_world_unit: float,
    wall_thickness_metres: float,
    wall_allowable_stress_pascal: float,
    external_pressure_pascal: float,
    normalization_floor: float,
) -> EmbeddedChannelSectionMechanics:
    radius_metres = edge.radius * metres_per_world_unit
    length_metres = (
        _vascular_edge_length_world(edge, node_by_id)
        * metres_per_world_unit
    )
    positive_radius = radius_metres > 0.0
    outer_radius_metres = (
        radius_metres + wall_thickness_metres if positive_radius else 0.0
    )
    maximum_transmural_pressure = (
        max(
            pressure_by_node_id[edge.source_node_id],
            pressure_by_node_id[edge.target_node_id],
        )
        - external_pressure_pascal
        if positive_radius
        else 0.0
    )
    lumen_volume_upper_bound = (
        _capsule_volume(radius_metres, length_metres)
        if positive_radius
        else 0.0
    )
    wall_volume_upper_bound = (
        pi
        * (
            (2.0 * radius_metres * wall_thickness_metres + wall_thickness_metres**2)
            * length_metres
            + (4.0 / 3.0)
            * (
                3.0 * radius_metres**2 * wall_thickness_metres
                + 3.0 * radius_metres * wall_thickness_metres**2
                + wall_thickness_metres**3
            )
        )
        if positive_radius
        else 0.0
    )
    thin_wall_hoop_stress = (
        maximum_transmural_pressure
        * radius_metres
        / wall_thickness_metres
        if positive_radius
        else 0.0
    )
    lame_inner_wall_hoop_stress = (
        maximum_transmural_pressure
        * (outer_radius_metres**2 + radius_metres**2)
        / (outer_radius_metres**2 - radius_metres**2)
        if positive_radius
        else 0.0
    )
    return EmbeddedChannelSectionMechanics(
        edge_id=edge.edge_id,
        radius_metres=radius_metres,
        outer_radius_metres=outer_radius_metres,
        length_metres=length_metres,
        maximum_transmural_pressure_pascal=maximum_transmural_pressure,
        lumen_capsule_volume_upper_bound_cubic_metres=(
            lumen_volume_upper_bound
        ),
        wall_annulus_volume_upper_bound_cubic_metres=(
            wall_volume_upper_bound
        ),
        thin_wall_hoop_stress_pascal=thin_wall_hoop_stress,
        lame_inner_wall_hoop_stress_pascal=lame_inner_wall_hoop_stress,
        burst_safety_factor=(
            wall_allowable_stress_pascal / lame_inner_wall_hoop_stress
            if lame_inner_wall_hoop_stress > normalization_floor
            else inf
        ),
    )


def _embedded_channel_constitutive_response(
    *,
    positive_radius_edge_count: int,
    lumen_volume_fraction_upper_bound: float,
    background_properties: IsotropicConstitutiveProperties,
    material_fraction_resolution: MaterialFractionResolution,
    homogenized_properties: IsotropicConstitutiveProperties | None,
    homogenization_validated_lumen_volume_fraction: float | None,
    normalization_floor: float,
) -> EmbeddedChannelConstitutiveResponse:
    if positive_radius_edge_count == 0:
        return FullSolidEmbeddedChannelResponse(background_properties)
    unresolved_reason = (
        "nonzero subgrid channels require HOMOGENIZED_EFFECTIVE_PROPERTIES"
        if material_fraction_resolution
        is not MaterialFractionResolution.HOMOGENIZED_EFFECTIVE_PROPERTIES
        else "homogenized constitutive properties were not supplied"
        if homogenized_properties is None
        else "homogenized evidence has no validated lumen-fraction domain"
        if homogenization_validated_lumen_volume_fraction is None
        else "lumen volume-fraction upper bound exceeds homogenization evidence"
        if lumen_volume_fraction_upper_bound
        > homogenization_validated_lumen_volume_fraction + normalization_floor
        else "homogenized stiffness or strength exceeds the background solid"
        if (
            homogenized_properties.young_modulus
            > background_properties.young_modulus
            + normalization_floor
            or homogenized_properties.yield_strength
            > background_properties.yield_strength
            + normalization_floor
        )
        else None
    )
    return (
        HomogenizedEmbeddedChannelResponse(
            homogenized_properties,
            homogenization_validated_lumen_volume_fraction,
        )
        if unresolved_reason is None
        and homogenized_properties is not None
        and homogenization_validated_lumen_volume_fraction is not None
        else UnresolvedEmbeddedChannelConstitutiveObstruction(
            unresolved_reason or "homogenized constitutive evidence is incomplete",
            lumen_volume_fraction_upper_bound,
        )
    )


def _embedded_channel_unevaluated_physics(
    response: EmbeddedChannelConstitutiveResponse,
) -> tuple[UnevaluatedPhysics, ...]:
    return (
        ()
        if isinstance(response, FullSolidEmbeddedChannelResponse)
        else (
            UnevaluatedPhysics.SUBGRID_CHANNEL_STRESS_CONCENTRATION,
            UnevaluatedPhysics.SUBGRID_CHANNEL_PRESSURE_PRESTRESS,
        )
        if isinstance(response, HomogenizedEmbeddedChannelResponse)
        else (
            UnevaluatedPhysics.SUBGRID_CHANNEL_GLOBAL_STIFFNESS,
            UnevaluatedPhysics.SUBGRID_CHANNEL_GLOBAL_STRENGTH,
            UnevaluatedPhysics.SUBGRID_CHANNEL_STRESS_CONCENTRATION,
            UnevaluatedPhysics.SUBGRID_CHANNEL_PRESSURE_PRESTRESS,
        )
    )
