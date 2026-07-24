"""Sobol terminal siting and macro corridor construction."""

from __future__ import annotations

import math
from functools import reduce
from itertools import islice, product
from typing import TYPE_CHECKING

from golem.kernel.anatomy.geometry import (
    _certified_segment_capsule_margins,
    _point_distance,
    _union_sdf,
)
from golem.kernel.anatomy.graph import InsufficientTerminalSitesObstruction
from golem.kernel.anatomy.realize.carriers import (
    _MacroCorridorSection,
    _PairedCorridorSection,
    _SegmentCapsule,
    _TerminalPortPair,
)
from golem.kernel.anatomy.realize.cco import _point3
from golem.kernel.anatomy.realize.clearance import _capsule_sets_violate_clearance

if TYPE_CHECKING:
    import numpy as np
    from scipy.spatial import cKDTree

    from golem.kernel.anatomy.vocabulary import (
        AcceptedAnatomy,
        CirculationCircuit,
        SealedVascularConfig,
    )


def _sobol_terminal_sites(
    parts: tuple[dict, ...],
    requested: int,
    region_id: str,
    config: SealedVascularConfig,
    anchor: tuple[float, float, float] | None = None,
    corridor: tuple[tuple[float, float, float], ...] | None = None,
    forbidden_capsules: tuple[_SegmentCapsule, ...] = (),
) -> tuple[_TerminalPortPair, ...] | InsufficientTerminalSitesObstruction:
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.stats import qmc

    from golem.kernel import engine as geometry

    lo, hi = geometry.graph_bounds({"parts": list(parts)}, pad=0.0)
    candidate_count = max(2, requested * config.sobol_candidate_factor)
    exponent = int(math.ceil(math.log2(candidate_count)))
    unit = qmc.Sobol(d=3, scramble=False).random_base2(exponent)
    candidates = qmc.scale(unit, lo, hi)
    half = config.terminal_port_half_separation
    medial_points = np.asarray(corridor or (anchor,), dtype=np.float64)
    _distance, nearest = cKDTree(medial_points).query(candidates, workers=1)
    medial_z = medial_points[nearest, 2]
    radial_z = np.maximum(np.abs(candidates[:, 2] - medial_z), half)
    supply = np.column_stack((candidates[:, :2], medial_z - radial_z))
    returning = np.column_stack((candidates[:, :2], medial_z + radial_z))
    minimum_depth = config.wall_clearance + config.terminal_radius
    eligible_mask = (_union_sdf(supply, parts) <= -minimum_depth) & (
        _union_sdf(returning, parts) <= -minimum_depth
    )
    contained_pairs = tuple(
        _TerminalPortPair(
            tuple(map(float, supply_point)),
            tuple(map(float, return_point)),
        )
        for supply_point, return_point in zip(
            supply[eligible_mask], returning[eligible_mask]
        )
    )
    eligible_pairs = tuple(
        pair
        for pair in contained_pairs
        if not _capsule_sets_violate_clearance(
            (
                _SegmentCapsule(
                    pair.supply,
                    pair.returning,
                    config.terminal_radius,
                ),
            ),
            forbidden_capsules,
            config.vessel_clearance,
        )
    )
    closest = (
        None
        if anchor is None or not eligible_pairs
        else min(
            eligible_pairs,
            key=lambda pair: _point_distance(pair.midpoint, anchor),
        )
    )
    ordered_pairs = (
        eligible_pairs
        if closest is None
        else (
            closest,
            *tuple(
                pair for pair in eligible_pairs if pair != closest
            ),
        )
    )
    selected = _take_spaced_pairs(
        ordered_pairs,
        requested,
        max(
            2.0 * (config.terminal_radius + config.vessel_clearance),
            config.nonincident_centerline_separation,
        ),
    )
    return (
        selected
        if len(selected) == requested
        else InsufficientTerminalSitesObstruction(
            region_id, requested, len(selected)
        )
    )


def _take_spaced_pairs(
    candidates: tuple[_TerminalPortPair, ...],
    requested: int,
    minimum_distance: float,
    selected: tuple[_TerminalPortPair, ...] = (),
) -> tuple[_TerminalPortPair, ...]:
    if len(selected) >= requested or not candidates:
        return selected
    pair = candidates[0]
    remaining = tuple(
        candidate
        for candidate in candidates[1:]
        if _point_distance(pair.midpoint, candidate.midpoint) >= minimum_distance
    )
    return _take_spaced_pairs(
        remaining, requested, minimum_distance, (*selected, pair)
    )


def _circuit_corridor(
    circuit: CirculationCircuit,
    landmarks: dict[str, object],
    shared_pump_interfaces: tuple[tuple[float, float, float], ...] = (),
) -> _MacroCorridorSection | None:
    interface_addresses = (
        *tuple(
            segment.interface_address for segment in circuit.distributing_arteries
        ),
        circuit.resistance_arteriole.interface_address,
    )
    interface_bones = tuple(
        address.removeprefix("skeleton/").removesuffix("/joint")
        for address in interface_addresses
    )
    start = _point3(landmarks.get(f"{circuit.bones[0]}/center"))
    interface_points = tuple(
        _point3(landmarks.get(f"{bone_id}/head"))
        for bone_id in interface_bones
    )
    endpoint_pairs = tuple(
        (
            _point3(landmarks.get(f"{bone_id}/head")),
            _point3(landmarks.get(f"{bone_id}/tail")),
        )
        for bone_id in circuit.bones[:-1]
    )
    if start is None or any(point is None for point in interface_points) or any(
        head is None or tail is None for head, tail in endpoint_pairs
    ):
        return None
    first_interface = interface_points[0]
    pump_delta_y = first_interface[1] - start[1]
    pump_delta_z = first_interface[2] - start[2]
    axial_pump_branch = abs(pump_delta_z) > abs(pump_delta_y)
    common_direction = (
        pump_delta_z if axial_pump_branch else pump_delta_y
    )
    common_sign = math.copysign(1.0, common_direction)
    common = (
        start[0],
        start[1] + 0.015 * common_sign,
        start[2],
    )
    pump_turn = (
        (
            (
                common[0],
                common[1] + 0.01 * common_sign,
                common[2] + math.copysign(0.01, pump_delta_z),
            ),
            (
                common[0],
                common[1] + 0.01 * common_sign,
                common[2] + math.copysign(0.02, pump_delta_z),
            ),
        )
        if axial_pump_branch
        else ()
    )
    pump_cover_start = pump_turn[0] if pump_turn else common
    pump_cover = tuple(
        sorted(
            (
                point
                for point in shared_pump_interfaces
                if _point_lies_between(
                    pump_cover_start, first_interface, point, 0.02
                )
            ),
            key=lambda point: _point_distance(pump_cover_start, point),
        )
    )
    initial_points = tuple(
        dict.fromkeys((start, common, *pump_turn, *pump_cover))
    )
    initial = _MacroCorridorSection(
        points=initial_points,
        # Moving common rotates the glue-owned pump inlet/outlet frame.
        movable_waypoint_indices=(),
    )
    traversed = reduce(
        lambda section, transition: _extend_macro_corridor(
            section,
            transition[0][0],
            transition[0][1],
            transition[1],
        ),
        zip(endpoint_pairs, interface_points),
        initial,
    )
    final = _point3(landmarks.get(f"{circuit.bones[-1]}/center"))
    points = (*traversed.points, final)
    return None if any(point is None for point in points) else _MacroCorridorSection(
        tuple(point for point in points if point is not None),
        traversed.movable_waypoint_indices,
    )


def _shared_pump_interface_points(
    accepted: AcceptedAnatomy, landmarks: dict[str, object]
) -> tuple[tuple[float, float, float], ...]:
    points = tuple(
        _point3(
            landmarks.get(
                f"{circuit.distributing_arteries[0].interface_address.removeprefix('skeleton/').removesuffix('/joint')}/head"
            )
        )
        for circuit in accepted.circuits
    )
    return tuple(point for point in points if point is not None)


def _macro_corridor_node_count(
    circuit: CirculationCircuit,
    corridor: _MacroCorridorSection,
    landmarks: dict[str, object],
) -> int:
    interface_bone = (
        circuit.distributing_arteries[0]
        .interface_address.removeprefix("skeleton/")
        .removesuffix("/joint")
    )
    interface_point = _point3(landmarks.get(f"{interface_bone}/head"))
    return (
        1
        if interface_point is None
        else min(
            range(len(corridor.points)),
            key=lambda index: _point_distance(
                corridor.points[index], interface_point
            ),
        )
        + 1
    )


def _unit_vector(
    vector: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    magnitude = math.sqrt(math.fsum(component * component for component in vector))
    return (
        tuple(component / magnitude for component in vector)
        if magnitude > 1.0e-12
        else None
    )


def _transverse_normal(
    tangent: tuple[float, float, float],
    preferred: tuple[float, float, float] | None = None,
) -> tuple[float, float, float] | None:
    axes = (
        *((preferred,) if preferred is not None else ()),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    candidates = tuple(
        _unit_vector(
            tuple(
                axis_component
                - math.fsum(
                    local_axis * local_tangent
                    for local_axis, local_tangent in zip(axis, tangent)
                )
                * tangent_component
                for axis_component, tangent_component in zip(axis, tangent)
            )
        )
        for axis in axes
    )
    selected = next((candidate for candidate in candidates if candidate is not None), None)
    return (
        None
        if selected is None
        else tuple(
            -component
            if preferred is not None
            and math.fsum(
                local_selected * local_preferred
                for local_selected, local_preferred in zip(selected, preferred)
            )
            < 0.0
            else component
            for component in selected
        )
    )


def _corridor_transverse_normals(
    corridor: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...] | None:
    if len(corridor) < 2:
        return None
    tangents = tuple(
        _unit_vector(
            tuple(
                right_component - left_component
                for left_component, right_component in zip(
                    corridor[index - 1 if index > 0 else 0],
                    corridor[index if index > 0 else 1],
                )
            )
        )
        for index in range(len(corridor))
    )
    if any(tangent is None for tangent in tangents):
        return None

    def descend(
        remaining: tuple[tuple[float, float, float], ...],
        previous: tuple[float, float, float] | None = None,
    ) -> tuple[tuple[float, float, float], ...] | None:
        if not remaining:
            return ()
        normal = _transverse_normal(remaining[0], previous)
        if normal is None:
            return None
        suffix = descend(remaining[1:], normal)
        return None if suffix is None else (normal, *suffix)

    return descend(tuple(tangent for tangent in tangents if tangent is not None))


def _paired_corridor_section(
    corridor: tuple[tuple[float, float, float], ...],
    half_separation: float,
) -> _PairedCorridorSection | None:
    normals = _corridor_transverse_normals(corridor)
    if normals is None:
        return None
    supply = tuple(
        tuple(
            coordinate - half_separation * normal_component
            for coordinate, normal_component in zip(point, normal)
        )
        for point, normal in zip(corridor, normals)
    )
    returning = tuple(
        tuple(
            coordinate + half_separation * normal_component
            for coordinate, normal_component in zip(point, normal)
        )
        for point, normal in zip(corridor, normals)
    )
    return _PairedCorridorSection(corridor, normals, supply, returning)


def _admissible_adjusted_corridors(
    corridor: _MacroCorridorSection,
    containment_parts: tuple[dict, ...],
    maximum_tree_radius: float,
    config: SealedVascularConfig,
    *,
    include_combined: bool,
) -> tuple[_PairedCorridorSection, ...]:
    candidates = _adjusted_corridor_candidates(
        corridor, config, include_combined=include_combined
    )
    paired = tuple(
        _paired_corridor_section(candidate, config.terminal_port_half_separation)
        for candidate in candidates
    )
    return tuple(
        section
        for candidate, section in zip(candidates, paired, strict=True)
        if section is not None
        and _waypoint_order_is_preserved(corridor, candidate)
        and _paired_corridor_is_contained(
            section, containment_parts, maximum_tree_radius, config
        )
        and not _paired_corridor_violates_lane_clearance(
            section, maximum_tree_radius, config
        )
    )


def _adjusted_corridor_candidates(
    corridor: _MacroCorridorSection,
    config: SealedVascularConfig,
    *,
    include_combined: bool,
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    replacements = tuple(
        (
            waypoint_index,
            _waypoint_stencil(corridor.points, waypoint_index, config),
        )
        for waypoint_index in corridor.movable_waypoint_indices
    )
    singles = tuple(
        _replace_corridor_points(
            corridor.points, ((waypoint_index, replacement),)
        )
        for waypoint_index, stencil in replacements
        for replacement in stencil
    )
    single_candidate_budget = (
        config.corridor_candidate_budget
        if not include_combined or len(replacements) <= 1
        else config.corridor_candidate_budget // 2
    )
    bounded_singles = singles[:single_candidate_budget]
    combined_candidate_budget = max(
        0, config.corridor_candidate_budget - len(bounded_singles)
    )
    combined = (
        tuple(
            _replace_corridor_points(
                corridor.points,
                tuple(
                    (waypoint_index, replacement)
                    for (waypoint_index, _stencil), replacement in zip(
                        replacements, replacement_tuple, strict=True
                    )
                ),
            )
            for replacement_tuple in islice(
                product(
                    *(stencil for _waypoint_index, stencil in replacements)
                ),
                combined_candidate_budget,
            )
            if len(replacement_tuple) > 1
        )
        if include_combined and len(replacements) > 1
        else ()
    )
    return tuple(dict.fromkeys((*bounded_singles, *combined)))


def _waypoint_stencil(
    points: tuple[tuple[float, float, float], ...],
    waypoint_index: int,
    config: SealedVascularConfig,
) -> tuple[tuple[float, float, float], ...]:
    if waypoint_index <= 0 or waypoint_index >= len(points) - 1:
        return ()
    tangent = _unit_vector(
        tuple(
            right - left
            for left, right in zip(
                points[waypoint_index - 1], points[waypoint_index + 1]
            )
        )
    )
    normals = _corridor_transverse_normals(points)
    normal = None if normals is None else normals[waypoint_index]
    if tangent is None or normal is None or config.waypoint_stencil_step <= 0.0:
        return ()
    level_count = int(
        config.maximum_waypoint_adjustment / config.waypoint_stencil_step
    )
    offsets = tuple(
        (tangent_level, normal_level)
        for tangent_level in range(-level_count, level_count + 1)
        for normal_level in range(-level_count, level_count + 1)
        if (tangent_level, normal_level) != (0, 0)
        and math.hypot(tangent_level, normal_level) <= level_count
    )
    ordered_offsets = tuple(
        sorted(
            offsets,
            key=lambda offset: (
                offset[0] * offset[0] + offset[1] * offset[1],
                abs(offset[1]),
                offset[1],
                offset[0],
            ),
        )
    )
    waypoint = points[waypoint_index]
    return tuple(
        tuple(
            coordinate
            + config.waypoint_stencil_step
            * (
                tangent_level * tangent_component
                + normal_level * normal_component
            )
            for coordinate, tangent_component, normal_component in zip(
                waypoint, tangent, normal
            )
        )
        for tangent_level, normal_level in ordered_offsets
    )


def _replace_corridor_points(
    points: tuple[tuple[float, float, float], ...],
    replacements: tuple[
        tuple[int, tuple[float, float, float]], ...
    ],
) -> tuple[tuple[float, float, float], ...]:
    replacement_by_index = dict(replacements)
    return tuple(
        replacement_by_index.get(index, point)
        for index, point in enumerate(points)
    )


def _waypoint_order_is_preserved(
    original: _MacroCorridorSection,
    candidate: tuple[tuple[float, float, float], ...],
) -> bool:
    return all(
        _waypoint_preserves_local_order(
            original.points[index - 1],
            original.points[index + 1],
            candidate[index],
        )
        for index in original.movable_waypoint_indices
        if 0 < index < len(candidate) - 1
    )


def _waypoint_preserves_local_order(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    waypoint: tuple[float, float, float],
) -> bool:
    chord = tuple(right_value - left_value for left_value, right_value in zip(left, right))
    left_projection = math.fsum(
        (waypoint_value - left_value) * chord_value
        for left_value, waypoint_value, chord_value in zip(left, waypoint, chord)
    )
    right_projection = math.fsum(
        (right_value - waypoint_value) * chord_value
        for right_value, waypoint_value, chord_value in zip(right, waypoint, chord)
    )
    return left_projection > 0.0 and right_projection > 0.0


def _paired_corridor_is_contained(
    section: _PairedCorridorSection,
    containment_parts: tuple[dict, ...],
    maximum_tree_radius: float,
    config: SealedVascularConfig,
) -> bool:
    capsules = _paired_corridor_capsules(section, maximum_tree_radius)
    return all(
        margin >= config.wall_clearance
        for margin in _certified_segment_capsule_margins(
            capsules,
            containment_parts,
            min(3, config.capsule_samples),
            config.wall_clearance,
        )
    )


def _paired_corridor_violates_lane_clearance(
    section: _PairedCorridorSection,
    maximum_tree_radius: float,
    config: SealedVascularConfig,
) -> bool:
    supply_capsules = _corridor_capsules(
        section.supply, maximum_tree_radius
    )
    return_capsules = _corridor_capsules(
        section.returning, maximum_tree_radius
    )
    return _capsule_sets_violate_clearance(
        supply_capsules, return_capsules, config.vessel_clearance
    )


def _paired_corridor_capsules(
    section: _PairedCorridorSection,
    radius: float,
) -> tuple[_SegmentCapsule, ...]:
    return (
        *_corridor_capsules(section.supply, radius),
        *_corridor_capsules(section.returning, radius),
    )


def _corridor_capsules(
    points: tuple[tuple[float, float, float], ...],
    radius: float,
) -> tuple[_SegmentCapsule, ...]:
    return tuple(
        _SegmentCapsule(left, right, radius)
        for left, right in zip(points, points[1:])
    )


def _point_lies_between(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    point: tuple[float, float, float],
    tolerance: float,
) -> bool:
    return (
        _point_distance(left, point) <= _point_distance(left, right) + tolerance
        and _point_distance(point, right) <= _point_distance(left, right) + tolerance
        and abs(
            _point_distance(left, point)
            + _point_distance(point, right)
            - _point_distance(left, right)
        )
        <= tolerance
    )
def _extend_corridor(
    points: tuple[tuple[float, float, float], ...],
    head: tuple[float, float, float],
    tail: tuple[float, float, float],
    interface: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    return _extend_macro_corridor(
        _MacroCorridorSection(points, ()), head, tail, interface
    ).points


def _extend_macro_corridor(
    section: _MacroCorridorSection,
    head: tuple[float, float, float],
    tail: tuple[float, float, float],
    interface: tuple[float, float, float],
) -> _MacroCorridorSection:
    points = section.points
    endpoint = min((head, tail), key=lambda point: _point_distance(point, interface))
    opposite = tail if endpoint == head else head
    waypoint = tuple(
        0.98 * endpoint_value + 0.02 * opposite_value
        for endpoint_value, opposite_value in zip(endpoint, opposite)
    )
    interface_is_on_bone = _point_lies_between(
        head, tail, interface, 1.0e-5
    )
    additions = tuple(dict.fromkeys(
        point
        for point in (
            *(
                (waypoint,)
                if not interface_is_on_bone
                and _point_distance(endpoint, interface) > 0.03
                else ()
            ),
            interface,
        )
        if _point_distance(points[-1], point) > 1.0e-12
    ))
    first_addition_index = len(points)
    movable_indices = tuple(
        first_addition_index + index
        for index, point in enumerate(additions)
        if point == waypoint and not interface_is_on_bone
    )
    return _MacroCorridorSection(
        (*points, *additions),
        (*section.movable_waypoint_indices, *movable_indices),
    )
