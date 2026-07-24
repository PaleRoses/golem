"""Tissue-envelope section sampling from myotendinous units."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from golem.kernel.body.gencyl import _gencyl_stations, _loft_sections
from golem.kernel.body.types import (
    FLESH_ROLE_NON_CARRIER,
    LoftSectionObstruction,
    _GencylStationObstruction,
)
from golem.kernel.anatomy.vocabulary import (
    CarrierFleshSample,
    TissueEnvelopeSection,
    TissueEnvelopeStation,
)

if TYPE_CHECKING:
    import numpy as np

    from golem.kernel.anatomy.vocabulary import (
        IntegumentLayer,
        MyotendinousUnit,
        TissueAnchor,
    )


def _tissue_envelope_section(
    bone_id: str,
    bones: dict[str, dict],
    muscles: tuple[MyotendinousUnit, ...],
    integument: tuple[IntegumentLayer, ...],
) -> TissueEnvelopeSection | None:
    bone = bones[bone_id]
    carrier = _minimum_carrier_sample(bone.get("flesh", ()))
    if carrier is None:
        return None
    flesh = bone["flesh"][carrier.flesh_index]
    local_muscles = tuple(
        muscle
        for muscle in muscles
        if bone_id in (muscle.origin.bone_id, muscle.insertion.bone_id)
    )
    routed_muscles = tuple(
        (
            muscle,
            _anchor_world(muscle.origin, bones),
            _anchor_world(muscle.joint, bones),
            _anchor_world(muscle.insertion, bones),
        )
        for muscle in local_muscles
    )
    local_region_ids = frozenset(
        muscle.region_id for muscle in local_muscles
    )
    integument_thickness = max(
        (
            layer.thickness
            for layer in integument
            if layer.region_id in local_region_ids
            and layer.formation is None
        ),
        default=0.0,
    )
    span_start, span_end = _flesh_profile_span(flesh)
    return TissueEnvelopeSection(
        bone_id=bone_id,
        shape_address=f"skeleton/{bone_id}/flesh[{carrier.flesh_index}]",
        muscle_ids=tuple(sorted(muscle.muscle_id for muscle in local_muscles)),
        stations=tuple(
            _tissue_envelope_station(
                profile_parameter,
                min(
                    1.0,
                    max(
                        0.0,
                        span_start
                        + profile_parameter * (span_end - span_start),
                    ),
                ),
                bone,
                bones,
                routed_muscles,
                integument_thickness,
            )
            for profile_parameter in _tissue_profile_parameters(
                flesh,
                local_muscles,
                bone_id,
            )
        ),
    )


def _tissue_profile_parameters(
    flesh: dict,
    muscles: tuple[MyotendinousUnit, ...],
    bone_id: str,
) -> tuple[float, ...]:
    span_start, span_end = _flesh_profile_span(flesh)
    radii = tuple(flesh.get("radii", ()))
    authored_stations = flesh.get("stations")
    loft_result = (
        _loft_sections(flesh)
        if flesh.get("kind") == "loft"
        else None
    )
    bone_stations = (
        tuple(section.station for section in loft_result)
        if isinstance(loft_result, tuple)
        else (
            tuple(map(float, authored_stations))
            if isinstance(authored_stations, (list, tuple))
            else tuple(
                span_start
                + (span_end - span_start) * index / (len(radii) - 1)
                for index in range(len(radii))
            )
        )
    )
    muscle_parameters = tuple(
        anchor.parameter
        for muscle in muscles
        for anchor in (muscle.origin, muscle.insertion)
        if anchor.bone_id == bone_id
        and span_start <= anchor.parameter <= span_end
    )
    return tuple(
        sorted(
            frozenset(
                (
                    0.0,
                    1.0,
                    *tuple(
                        (station - span_start) / (span_end - span_start)
                        for station in (*bone_stations, *muscle_parameters)
                    ),
                )
            )
        )
    )


def _flesh_profile_span(flesh: dict) -> tuple[float, float]:
    loft_result = (
        _loft_sections(flesh)
        if flesh.get("kind") == "loft"
        else None
    )
    return (
        (loft_result[0].station, loft_result[-1].station)
        if isinstance(loft_result, tuple)
        else tuple(map(float, flesh.get("span", (0.0, 1.0))))
    )


def _tissue_envelope_station(
    profile_parameter: float,
    bone_parameter: float,
    bone: dict,
    bones: dict[str, dict],
    routed_muscles: tuple[tuple[MyotendinousUnit, object, object, object], ...],
    integument_thickness: float,
) -> TissueEnvelopeStation:
    base_radius = _interpolate_radius(bone["bone_radii"], bone_parameter)
    muscle_sections = tuple(
        section
        for routed_muscle in routed_muscles
        for section in (
            _muscle_section_at(
                routed_muscle,
                bone["id"],
                bone_parameter,
                bones,
            ),
        )
        if section is not None
    )
    negative_x = max(
        (base_radius, *(radius - offset[0] for offset, radius in muscle_sections))
    ) + integument_thickness
    positive_x = max(
        (base_radius, *(radius + offset[0] for offset, radius in muscle_sections))
    ) + integument_thickness
    negative_y = max(
        (base_radius, *(radius - offset[1] for offset, radius in muscle_sections))
    ) + integument_thickness
    positive_y = max(
        (base_radius, *(radius + offset[1] for offset, radius in muscle_sections))
    ) + integument_thickness
    return TissueEnvelopeStation(
        parameter=profile_parameter,
        offset=(
            0.5 * (positive_x - negative_x),
            0.5 * (positive_y - negative_y),
        ),
        half_size=(
            0.5 * (positive_x + negative_x),
            0.5 * (positive_y + negative_y),
        ),
    )


def _muscle_section_at(
    routed_muscle: tuple[MyotendinousUnit, object, object, object],
    bone_id: str,
    bone_parameter: float,
    bones: dict[str, dict],
) -> tuple[tuple[float, float], float] | None:
    import numpy as np

    muscle, origin_world, joint_world, insertion_world = routed_muscle
    is_origin_segment = bone_id == muscle.origin.bone_id
    is_insertion_segment = bone_id == muscle.insertion.bone_id
    active = (
        is_origin_segment and bone_parameter >= muscle.origin.parameter
    ) or (
        is_insertion_segment and bone_parameter <= muscle.insertion.parameter
    )
    if not active:
        return None
    origin_distance = (
        1.0 - muscle.origin.parameter
    ) * float(bones[muscle.origin.bone_id]["length"])
    insertion_distance = muscle.insertion.parameter * float(
        bones[muscle.insertion.bone_id]["length"]
    )
    total_distance = origin_distance + insertion_distance
    joint_parameter = origin_distance / total_distance
    segment_parameter = (
        (bone_parameter - muscle.origin.parameter)
        / (1.0 - muscle.origin.parameter)
        if is_origin_segment
        else bone_parameter / muscle.insertion.parameter
    )
    progression = (
        joint_parameter * segment_parameter
        if is_origin_segment
        else joint_parameter
        + (1.0 - joint_parameter) * segment_parameter
    )
    center_world = (
        origin_world + segment_parameter * (joint_world - origin_world)
        if is_origin_segment
        else joint_world
        + segment_parameter * (insertion_world - joint_world)
    )
    bone = bones[bone_id]
    frame = np.asarray(bone["R"], dtype=np.float64)
    axis_point = np.asarray(bone["head"], dtype=np.float64) + frame @ np.asarray(
        (0.0, 0.0, bone_parameter * float(bone["length"])),
        dtype=np.float64,
    )
    local_offset = frame.T @ (center_world - axis_point)
    radius = muscle.tendon_radius + (
        muscle.belly_radius - muscle.tendon_radius
    ) * 4.0 * progression * (1.0 - progression)
    return (float(local_offset[0]), float(local_offset[1])), radius


def _anchor_world(anchor: TissueAnchor, bones: dict[str, dict]) -> object:
    import numpy as np

    bone = bones[anchor.bone_id]
    return np.asarray(bone["head"], dtype=np.float64) + np.asarray(
        bone["R"], dtype=np.float64
    ) @ np.asarray(
        (*anchor.offset, anchor.parameter * float(bone["length"])),
        dtype=np.float64,
    )


def _interpolate_radius(radii: object, parameter: float) -> float:
    checked_radii = tuple(map(float, radii))
    scaled_parameter = parameter * (len(checked_radii) - 1)
    left_index = min(math.floor(scaled_parameter), len(checked_radii) - 2)
    local_parameter = scaled_parameter - left_index
    return checked_radii[left_index] + local_parameter * (
        checked_radii[left_index + 1] - checked_radii[left_index]
    )


def _carrier_flesh_sample(
    index: int, record: object
) -> CarrierFleshSample | None:
    if not isinstance(record, dict):
        return None
    if record.get("role") == FLESH_ROLE_NON_CARRIER:
        # Membrane role (R3): the record keeps its authored dimensions and
        # never joins carrier cross-section sampling, so a thin membrane can
        # neither deflate the measured minimum nor attract a floor-inflation
        # suggestion.
        return None
    if record.get("kind") == "gencyl":
        station_result = _gencyl_stations(record)
        return (
            None
            if isinstance(station_result, _GencylStationObstruction)
            else CarrierFleshSample(
                flesh_index=index,
                minimum_radius=min(map(float, record["radii"])),
            )
        )
    if record.get("kind") == "loft":
        section_result = _loft_sections(record)
        return (
            None
            if isinstance(section_result, LoftSectionObstruction)
            else CarrierFleshSample(
                flesh_index=index,
                minimum_radius=min(
                    min(section.width, section.depth)
                    for section in section_result
                ),
            )
        )
    return None


def _minimum_carrier_sample(flesh: object) -> CarrierFleshSample | None:
    records = flesh if isinstance(flesh, (list, tuple)) else ()
    samples = tuple(
        sample
        for index, record in enumerate(records)
        for sample in (_carrier_flesh_sample(index, record),)
        if sample is not None
    )
    return min(
        samples,
        key=lambda sample: (sample.minimum_radius, sample.flesh_index),
        default=None,
    )
