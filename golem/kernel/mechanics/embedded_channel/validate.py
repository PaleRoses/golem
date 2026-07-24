"""Guard embedded-channel evidence with typed input obstructions."""

from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import TYPE_CHECKING

from ..model import (
    IsotropicConstitutiveProperties,
    MaterialFractionResolution,
    _invalid_isotropic_constitutive_fields,
)
from .model import (
    EmbeddedChannelCriteria,
    EmbeddedChannelObstruction,
    EmbeddedChannelScaleSeparationObstruction,
    EmbeddedChannelVolumeFractionObstruction,
    EmbeddedChannelVolumeKind,
    InvalidEmbeddedChannelInputObstruction,
    InsufficientEmbeddedChannelBurstMarginObstruction,
    UnsupportedEmbeddedChannelExternalPressureObstruction,
    _vascular_edge_length_world,
)

if TYPE_CHECKING:
    from golem.kernel.anatomy import (
        AcceptedVasculature,
        ClosedVascularGraph,
        PhysicalHydraulicNodePressure,
        VascularEdge,
    )


def _embedded_channel_input_obstructions(
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
    material_fraction_resolution: MaterialFractionResolution,
    homogenized_properties: IsotropicConstitutiveProperties | None,
    homogenization_validated_lumen_volume_fraction: float | None,
    criteria: EmbeddedChannelCriteria,
) -> tuple[EmbeddedChannelObstruction, ...]:
    graph = vasculature.graph
    node_counts = Counter(node.node_id for node in graph.nodes)
    edge_counts = Counter(edge.edge_id for edge in graph.edges)
    pressure_counts = Counter(pressure.node_id for pressure in node_pressures)
    node_ids = frozenset(node_counts)
    positive_radius_edges = tuple(
        edge for edge in graph.edges if edge.radius > 0.0
    )
    required_pressure_node_ids = frozenset(
        node_id
        for edge in positive_radius_edges
        for node_id in (edge.source_node_id, edge.target_node_id)
    )
    return (
        _embedded_channel_scalar_input_obstructions(
            metres_per_world_unit=metres_per_world_unit,
            grid_pitch_metres=grid_pitch_metres,
            host_volume_cubic_metres=host_volume_cubic_metres,
            wall_thickness_metres=wall_thickness_metres,
            wall_allowable_stress_pascal=wall_allowable_stress_pascal,
            external_pressure_pascal=external_pressure_pascal,
            material_fraction_resolution=material_fraction_resolution,
            homogenization_validated_lumen_volume_fraction=(
                homogenization_validated_lumen_volume_fraction
            ),
            criteria=criteria,
        )
        + _embedded_channel_constitutive_input_obstructions(
            background_properties,
            homogenized_properties,
        )
        + _embedded_channel_vasculature_input_obstructions(
            graph,
            node_counts,
            edge_counts,
            node_ids,
        )
        + _embedded_channel_pressure_input_obstructions(
            node_pressures,
            pressure_counts,
            node_ids,
            required_pressure_node_ids,
        )
        + _embedded_channel_edge_length_input_obstructions(
            graph,
            positive_radius_edges,
            node_ids,
        )
    )


def _embedded_channel_scalar_input_obstructions(
    *,
    metres_per_world_unit: float,
    grid_pitch_metres: float,
    host_volume_cubic_metres: float,
    wall_thickness_metres: float,
    wall_allowable_stress_pascal: float,
    external_pressure_pascal: float,
    material_fraction_resolution: MaterialFractionResolution,
    homogenization_validated_lumen_volume_fraction: float | None,
    criteria: EmbeddedChannelCriteria,
) -> tuple[EmbeddedChannelObstruction, ...]:
    scalar_requirements = (
        (
            "metres_per_world_unit",
            isfinite(metres_per_world_unit) and metres_per_world_unit > 0.0,
            "must be finite and positive",
        ),
        (
            "grid_pitch_metres",
            isfinite(grid_pitch_metres) and grid_pitch_metres > 0.0,
            "must be finite and positive",
        ),
        (
            "host_volume_cubic_metres",
            isfinite(host_volume_cubic_metres)
            and host_volume_cubic_metres > 0.0,
            "must be finite and positive",
        ),
        (
            "wall_thickness_metres",
            isfinite(wall_thickness_metres) and wall_thickness_metres > 0.0,
            "must be finite and positive",
        ),
        (
            "wall_allowable_stress_pascal",
            isfinite(wall_allowable_stress_pascal)
            and wall_allowable_stress_pascal > 0.0,
            "must be finite and positive",
        ),
        (
            "external_pressure_pascal",
            isfinite(external_pressure_pascal)
            and external_pressure_pascal >= 0.0,
            "must be finite and non-negative",
        ),
        (
            "criteria/maximum_outer_diameter_to_grid_pitch",
            isfinite(criteria.maximum_outer_diameter_to_grid_pitch)
            and 0.0 < criteria.maximum_outer_diameter_to_grid_pitch <= 1.0,
            "must be finite and in (0, 1]",
        ),
        (
            "criteria/maximum_lumen_volume_fraction",
            isfinite(criteria.maximum_lumen_volume_fraction)
            and 0.0 < criteria.maximum_lumen_volume_fraction < 1.0,
            "must be finite and in (0, 1)",
        ),
        (
            "criteria/maximum_channel_envelope_volume_fraction",
            isfinite(criteria.maximum_channel_envelope_volume_fraction)
            and criteria.maximum_lumen_volume_fraction
            <= criteria.maximum_channel_envelope_volume_fraction
            < 1.0,
            "must be finite, below one, and no smaller than the lumen limit",
        ),
        (
            "criteria/normalization_floor",
            isfinite(criteria.normalization_floor)
            and criteria.normalization_floor > 0.0,
            "must be finite and positive",
        ),
    )
    return tuple(
        InvalidEmbeddedChannelInputObstruction(address, reason)
        for address, valid, reason in scalar_requirements
        if not valid
    ) + (
        (
            InvalidEmbeddedChannelInputObstruction(
                "criteria/minimum_burst_safety_factor",
                "optional safety factor must be finite and positive",
            ),
        )
        if criteria.minimum_burst_safety_factor is not None
        and (
            not isfinite(criteria.minimum_burst_safety_factor)
            or criteria.minimum_burst_safety_factor <= 0.0
        )
        else ()
    ) + (
        (
            InvalidEmbeddedChannelInputObstruction(
                "homogenization_validated_lumen_volume_fraction",
                "optional fraction must be finite and in (0, 1]",
            ),
        )
        if homogenization_validated_lumen_volume_fraction is not None
        and (
            not isfinite(homogenization_validated_lumen_volume_fraction)
            or not 0.0
            < homogenization_validated_lumen_volume_fraction
            <= 1.0
        )
        else ()
    ) + (
        (
            InvalidEmbeddedChannelInputObstruction(
                "material_fraction_resolution",
                "must be a MaterialFractionResolution value",
            ),
        )
        if not isinstance(
            material_fraction_resolution,
            MaterialFractionResolution,
        )
        else ()
    )


def _embedded_channel_constitutive_input_obstructions(
    background_properties: IsotropicConstitutiveProperties,
    homogenized_properties: IsotropicConstitutiveProperties | None,
) -> tuple[EmbeddedChannelObstruction, ...]:
    return tuple(
        InvalidEmbeddedChannelInputObstruction(
            f"background_properties/{name}",
            "value lies outside the isotropic constitutive domain",
        )
        for name, _value in _invalid_isotropic_constitutive_fields(
            background_properties
        )
    ) + tuple(
        InvalidEmbeddedChannelInputObstruction(
            f"homogenized_properties/{name}",
            "value lies outside the isotropic constitutive domain",
        )
        for name, _value in (
            _invalid_isotropic_constitutive_fields(homogenized_properties)
            if homogenized_properties is not None
            else ()
        )
    )


def _embedded_channel_vasculature_input_obstructions(
    graph: "ClosedVascularGraph",
    node_counts: Counter[str],
    edge_counts: Counter[str],
    node_ids: frozenset[str],
) -> tuple[EmbeddedChannelObstruction, ...]:
    return tuple(
        InvalidEmbeddedChannelInputObstruction(
            f"vasculature/nodes/{node_id}",
            "node id appears more than once",
        )
        for node_id, count in node_counts.items()
        if count > 1
    ) + tuple(
        InvalidEmbeddedChannelInputObstruction(
            f"vasculature/nodes/{node.node_id}/position",
            "position must contain three finite coordinates",
        )
        for node in graph.nodes
        if len(node.position) != 3
        or not all(isfinite(coordinate) for coordinate in node.position)
    ) + tuple(
        InvalidEmbeddedChannelInputObstruction(
            f"vasculature/edges/{edge_id}",
            "edge id appears more than once",
        )
        for edge_id, count in edge_counts.items()
        if count > 1
    ) + tuple(
        InvalidEmbeddedChannelInputObstruction(
            f"vasculature/edges/{edge.edge_id}/radius",
            "radius must be finite and non-negative",
        )
        for edge in graph.edges
        if not isfinite(edge.radius) or edge.radius < 0.0
    ) + tuple(
        InvalidEmbeddedChannelInputObstruction(
            f"vasculature/edges/{edge.edge_id}/{endpoint_name}",
            "edge endpoint does not exist in the canonical graph",
        )
        for edge in graph.edges
        for endpoint_name, node_id in (
            ("source_node_id", edge.source_node_id),
            ("target_node_id", edge.target_node_id),
        )
        if node_id not in node_ids
    )


def _embedded_channel_pressure_input_obstructions(
    node_pressures: tuple["PhysicalHydraulicNodePressure", ...],
    pressure_counts: Counter[str],
    node_ids: frozenset[str],
    required_pressure_node_ids: frozenset[str],
) -> tuple[EmbeddedChannelObstruction, ...]:
    return tuple(
        InvalidEmbeddedChannelInputObstruction(
            f"node_pressures/{node_id}",
            "node pressure appears more than once",
        )
        for node_id, count in pressure_counts.items()
        if count > 1
    ) + tuple(
        InvalidEmbeddedChannelInputObstruction(
            f"node_pressures/{pressure.node_id}",
            "pressure node does not exist in the canonical graph",
        )
        for pressure in node_pressures
        if pressure.node_id not in node_ids
    ) + tuple(
        InvalidEmbeddedChannelInputObstruction(
            f"node_pressures/{pressure.node_id}",
            "pressure must be finite",
        )
        for pressure in node_pressures
        if not isfinite(pressure.pressure_pascal)
    ) + tuple(
        InvalidEmbeddedChannelInputObstruction(
            f"node_pressures/{node_id}",
            "positive-radius edge endpoint has no pressure",
        )
        for node_id in sorted(required_pressure_node_ids - frozenset(pressure_counts))
    )


def _embedded_channel_edge_length_input_obstructions(
    graph: "ClosedVascularGraph",
    positive_radius_edges: tuple["VascularEdge", ...],
    node_ids: frozenset[str],
) -> tuple[EmbeddedChannelObstruction, ...]:
    return tuple(
        InvalidEmbeddedChannelInputObstruction(
            f"vasculature/edges/{edge.edge_id}/length",
            "positive-radius edge must have finite positive length",
        )
        for edge in positive_radius_edges
        if edge.source_node_id in node_ids
        and edge.target_node_id in node_ids
        and (
            not isfinite(_vascular_edge_length_world(edge, graph.node_by_id))
            or _vascular_edge_length_world(edge, graph.node_by_id) <= 0.0
        )
    )


def _embedded_channel_external_pressure_support_obstructions(
    ordered_edges: tuple["VascularEdge", ...],
    pressure_by_node_id: dict[str, float],
    *,
    external_pressure_pascal: float,
    normalization_floor: float,
) -> tuple[EmbeddedChannelObstruction, ...]:
    return tuple(
        UnsupportedEmbeddedChannelExternalPressureObstruction(
            edge.edge_id,
            min(
                pressure_by_node_id[edge.source_node_id],
                pressure_by_node_id[edge.target_node_id],
            )
            - external_pressure_pascal,
            normalization_floor,
        )
        for edge in ordered_edges
        if edge.radius > 0.0
        and min(
            pressure_by_node_id[edge.source_node_id],
            pressure_by_node_id[edge.target_node_id],
        )
        < external_pressure_pascal - normalization_floor
    )


def _embedded_channel_scale_separation_obstructions(
    maximum_outer_diameter_to_grid_pitch: float,
    maximum_permitted_outer_diameter_to_grid_pitch: float,
) -> tuple[EmbeddedChannelObstruction, ...]:
    return (
        (
            EmbeddedChannelScaleSeparationObstruction(
                maximum_outer_diameter_to_grid_pitch,
                maximum_permitted_outer_diameter_to_grid_pitch,
            ),
        )
        if maximum_outer_diameter_to_grid_pitch
        > maximum_permitted_outer_diameter_to_grid_pitch
        else ()
    )


def _embedded_channel_volume_fraction_obstructions(
    lumen_volume_fraction_upper_bound: float,
    channel_envelope_volume_fraction_upper_bound: float,
    *,
    maximum_lumen_volume_fraction: float,
    maximum_channel_envelope_volume_fraction: float,
) -> tuple[EmbeddedChannelObstruction, ...]:
    return (
        (
            EmbeddedChannelVolumeFractionObstruction(
                EmbeddedChannelVolumeKind.LUMEN,
                lumen_volume_fraction_upper_bound,
                maximum_lumen_volume_fraction,
            ),
        )
        if lumen_volume_fraction_upper_bound
        > maximum_lumen_volume_fraction
        else ()
    ) + (
        (
            EmbeddedChannelVolumeFractionObstruction(
                EmbeddedChannelVolumeKind.OUTER_ENVELOPE,
                channel_envelope_volume_fraction_upper_bound,
                maximum_channel_envelope_volume_fraction,
            ),
        )
        if channel_envelope_volume_fraction_upper_bound
        > maximum_channel_envelope_volume_fraction
        else ()
    )


def _embedded_channel_burst_margin_obstructions(
    minimum_burst_safety_factor: float,
    required_minimum_burst_safety_factor: float | None,
) -> tuple[EmbeddedChannelObstruction, ...]:
    return (
        (
            InsufficientEmbeddedChannelBurstMarginObstruction(
                minimum_burst_safety_factor,
                required_minimum_burst_safety_factor,
            ),
        )
        if required_minimum_burst_safety_factor is not None
        and minimum_burst_safety_factor
        < required_minimum_burst_safety_factor
        else ()
    )
