"""Anatomy overall decoding and tissue-anatomy validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from golem.kernel.anatomy.envelope import _minimum_carrier_sample
from golem.kernel.anatomy.lineage import (
    _duplicate_values,
    _is_bone_cross_section,
    _is_finite_pair,
    _is_finite_positive_number,
    _is_unit_interval_number,
    _terminal_bones,
)
from golem.kernel.anatomy.vocabulary import (
    AnatomyRegion,
    BodyRegionKind,
    CapillaryBed,
    ClosedVascularSystem,
    DIALECT,
    DuplicateAnatomyRegionObstruction,
    DuplicateExchangeBedObstruction,
    DuplicateIntegumentLayerObstruction,
    DuplicateMyotendinousUnitObstruction,
    DuplicateRegionHostObstruction,
    IncompatibleIntegumentFormationObstruction,
    IntegumentLayer,
    InvalidMyotendinousPathObstruction,
    InvalidSkeletonAddressObstruction,
    MalformedAnatomyObstruction,
    MissingIntegumentLayerObstruction,
    MissingSkeletalCrossSectionObstruction,
    MissingTerminalRegionObstruction,
    MyotendinousUnit,
    NonterminalExchangeRegionObstruction,
    OverallAnatomy,
    PerfusedTissueKind,
    PumpOrgan,
    TissueAnchor,
    UnknownAnatomyRegionObstruction,
    UnperfusedMyotendinousUnitObstruction,
    UnreferencedAnatomyRegionObstruction,
    UnsupportedTissueCarrierObstruction,
    _CLOSED_VASCULAR_KIND,
    _REGION_KIND_VALUES,
    _TISSUE_KIND_VALUES,
)
from golem.kernel.engine.types import SkinFormationKind

if TYPE_CHECKING:
    from golem.kernel.anatomy.vocabulary import AnatomyInputObstruction


def _decode_anatomy(
    payload: object,
    bones: dict[str, dict],
    ordered_bones: tuple[str, ...],
) -> OverallAnatomy | tuple[AnatomyInputObstruction, ...]:
    if not isinstance(payload, dict):
        return (MalformedAnatomyObstruction("/anatomy", "expected an object"),)
    overall_payload = payload.get("overall")
    structural_obstructions = _anatomy_shape_obstructions(
        payload, overall_payload
    )
    if structural_obstructions:
        return structural_obstructions
    raw_regions = overall_payload.get("regions")
    circulation_payload = overall_payload.get("circulation")
    raw_muscles = overall_payload.get("myotendinous_units", [])
    raw_integument = overall_payload.get("integument_layers", [])
    container_obstructions = _anatomy_container_obstructions(
        raw_regions, circulation_payload, raw_muscles, raw_integument
    )
    if container_obstructions:
        return container_obstructions
    bone_ids = frozenset(ordered_bones)
    region_results = tuple(
        _decode_region(region_payload, index, bone_ids)
        for index, region_payload in enumerate(raw_regions)
    )
    region_obstructions = tuple(
        obstruction
        for region_result in region_results
        if isinstance(region_result, tuple)
        for obstruction in region_result
    )
    regions = tuple(
        region_result
        for region_result in region_results
        if isinstance(region_result, AnatomyRegion)
    )
    duplicate_obstructions = _region_identity_obstructions(regions)
    if region_obstructions or duplicate_obstructions:
        return (*region_obstructions, *duplicate_obstructions)
    region_by_id = {region.region_id: region for region in regions}
    circulation_result = _decode_circulation(circulation_payload, region_by_id)
    if not isinstance(circulation_result, ClosedVascularSystem):
        return circulation_result
    circulation = circulation_result
    muscle_results = tuple(
        _decode_myotendinous_unit(
            muscle_payload,
            index,
            bone_ids,
            region_by_id,
        )
        for index, muscle_payload in enumerate(raw_muscles)
    )
    integument_results = tuple(
        _decode_integument_layer(layer_payload, index, region_by_id)
        for index, layer_payload in enumerate(raw_integument)
    )
    tissue_decode_obstructions = tuple(
        obstruction
        for result in (*muscle_results, *integument_results)
        if isinstance(result, tuple)
        for obstruction in result
    )
    if tissue_decode_obstructions:
        return tissue_decode_obstructions
    muscles = tuple(
        result
        for result in muscle_results
        if isinstance(result, MyotendinousUnit)
    )
    integument = tuple(
        result
        for result in integument_results
        if isinstance(result, IntegumentLayer)
    )
    terminal_bones = _terminal_bones(bones, ordered_bones)
    coverage_obstructions = _anatomy_coverage_obstructions(
        circulation, region_by_id, terminal_bones
    )
    if coverage_obstructions:
        return coverage_obstructions
    overall = OverallAnatomy(
        regions=regions,
        circulation=circulation,
        muscles=muscles,
        integument=integument,
    )
    tissue_obstructions = _validate_tissue_anatomy(
        overall,
        bones,
        bone_ids,
    )
    return tissue_obstructions if tissue_obstructions else overall


def _anatomy_shape_obstructions(
    payload: dict,
    overall_payload: object,
) -> tuple[AnatomyInputObstruction, ...]:
    return (
        ()
        if payload.get("dialect") == DIALECT
        else (
            MalformedAnatomyObstruction(
                "/anatomy/dialect", f"expected {DIALECT!r}"
            ),
        )
    ) + (
        ()
        if isinstance(overall_payload, dict)
        else (
            MalformedAnatomyObstruction(
                "/anatomy/overall", "expected an object"
            ),
        )
    )


def _anatomy_container_obstructions(
    raw_regions: object,
    circulation_payload: object,
    raw_muscles: object,
    raw_integument: object,
) -> tuple[AnatomyInputObstruction, ...]:
    return (
        ()
        if isinstance(raw_regions, list) and raw_regions
        else (
            MalformedAnatomyObstruction(
                "/anatomy/overall/regions", "expected a non-empty array"
            ),
        )
    ) + (
        ()
        if isinstance(circulation_payload, dict)
        else (
            MalformedAnatomyObstruction(
                "/anatomy/overall/circulation", "expected an object"
            ),
        )
    ) + (
        ()
        if isinstance(raw_muscles, list)
        else (
            MalformedAnatomyObstruction(
                "/anatomy/overall/myotendinous_units", "expected an array"
            ),
        )
    ) + (
        ()
        if isinstance(raw_integument, list)
        else (
            MalformedAnatomyObstruction(
                "/anatomy/overall/integument_layers", "expected an array"
            ),
        )
    )


def _region_identity_obstructions(
    regions: tuple[AnatomyRegion, ...],
) -> tuple[AnatomyInputObstruction, ...]:
    return (
        *tuple(
            map(
                DuplicateAnatomyRegionObstruction,
                _duplicate_values(tuple(map(lambda region: region.region_id, regions))),
            )
        ),
        *tuple(
            map(
                DuplicateRegionHostObstruction,
                _duplicate_values(
                    tuple(map(lambda region: region.host_bone_id, regions))
                ),
            )
        ),
    )


def _anatomy_coverage_obstructions(
    circulation: ClosedVascularSystem,
    region_by_id: dict[str, AnatomyRegion],
    terminal_bones: frozenset[str],
) -> tuple[AnatomyInputObstruction, ...]:
    exchange_hosts = frozenset(
        exchange_bed.region.host_bone_id
        for exchange_bed in circulation.exchange_beds
    )
    referenced_regions = frozenset(
        (
            circulation.pump.region.region_id,
            *tuple(
                map(
                    lambda exchange_bed: exchange_bed.region.region_id,
                    circulation.exchange_beds,
                )
            ),
        )
    )
    return (
        *tuple(
            map(MissingTerminalRegionObstruction, sorted(terminal_bones - exchange_hosts))
        ),
        *tuple(
            NonterminalExchangeRegionObstruction(
                exchange_bed.region.region_id,
                exchange_bed.region.host_bone_id,
            )
            for exchange_bed in circulation.exchange_beds
            if exchange_bed.region.host_bone_id not in terminal_bones
        ),
        *tuple(
            map(
                UnreferencedAnatomyRegionObstruction,
                sorted(frozenset(region_by_id) - referenced_regions),
            )
        ),
    )


def _decode_region(
    payload: object, index: int, bone_ids: frozenset[str]
) -> AnatomyRegion | tuple[AnatomyInputObstruction, ...]:
    address = f"/anatomy/overall/regions/{index}"
    if not isinstance(payload, dict):
        return (MalformedAnatomyObstruction(address, "expected an object"),)
    region_id = payload.get("region_id")
    kind = payload.get("kind")
    host_bone_id = payload.get("host_bone_id")
    obstructions: tuple[AnatomyInputObstruction, ...] = (
        ()
        if isinstance(region_id, str) and bool(region_id)
        else (
            MalformedAnatomyObstruction(
                f"{address}/region_id", "expected a non-empty string"
            ),
        )
    ) + (
        ()
        if kind in _REGION_KIND_VALUES
        else (
            MalformedAnatomyObstruction(
                f"{address}/kind", f"expected one of {_REGION_KIND_VALUES!r}"
            ),
        )
    ) + (
        ()
        if isinstance(host_bone_id, str) and host_bone_id in bone_ids
        else (
            InvalidSkeletonAddressObstruction(
                f"{address}/host_bone_id", host_bone_id
            ),
        )
    )
    if obstructions:
        return obstructions
    return AnatomyRegion(
        region_id=region_id,
        kind=BodyRegionKind(kind),
        host_bone_id=host_bone_id,
    )


def _decode_circulation(
    payload: dict, region_by_id: dict[str, AnatomyRegion]
) -> ClosedVascularSystem | tuple[AnatomyInputObstruction, ...]:
    address = "/anatomy/overall/circulation"
    raw_exchange_beds = payload.get("exchange_beds")
    kind = payload.get("kind")
    pump_region_id = payload.get("pump_region_id")
    carrier_radius_scale = payload.get("carrier_radius_scale")
    distance_decay = payload.get("distance_decay")
    structural_obstructions: tuple[AnatomyInputObstruction, ...] = (
        ()
        if kind == _CLOSED_VASCULAR_KIND
        else (
            MalformedAnatomyObstruction(
                f"{address}/kind",
                f"expected {_CLOSED_VASCULAR_KIND!r}",
            ),
        )
    ) + (
        ()
        if isinstance(raw_exchange_beds, list) and raw_exchange_beds
        else (
            MalformedAnatomyObstruction(
                f"{address}/exchange_beds", "expected a non-empty array"
            ),
        )
    ) + tuple(
        MalformedAnatomyObstruction(
            f"{address}/{field_name}", "expected a finite positive number"
        )
        for field_name, value in (
            ("carrier_radius_scale", carrier_radius_scale),
            ("distance_decay", distance_decay),
        )
        if not _is_finite_positive_number(value)
    )
    if structural_obstructions:
        return structural_obstructions
    exchange_results = tuple(
        _decode_exchange_bed(exchange_payload, index, region_by_id)
        for index, exchange_payload in enumerate(raw_exchange_beds)
    )
    exchange_obstructions = tuple(
        obstruction
        for exchange_result in exchange_results
        if isinstance(exchange_result, tuple)
        for obstruction in exchange_result
    )
    exchange_beds = tuple(
        exchange_result
        for exchange_result in exchange_results
        if isinstance(exchange_result, CapillaryBed)
    )
    pump_obstructions: tuple[AnatomyInputObstruction, ...] = (
        ()
        if pump_region_id in region_by_id
        else (
            UnknownAnatomyRegionObstruction(
                f"{address}/pump_region_id", pump_region_id
            ),
        )
    )
    duplicate_obstructions = tuple(
        map(
            DuplicateExchangeBedObstruction,
            _duplicate_values(
                tuple(map(lambda bed: bed.region.region_id, exchange_beds))
            ),
        )
    )
    pump_exchange_obstructions: tuple[AnatomyInputObstruction, ...] = (
        (
            MalformedAnatomyObstruction(
                f"{address}/exchange_beds",
                "the pump region cannot also be an exchange-bed region",
            ),
        )
        if pump_region_id
        in frozenset(map(lambda bed: bed.region.region_id, exchange_beds))
        else ()
    )
    obstructions = (
        *exchange_obstructions,
        *pump_obstructions,
        *duplicate_obstructions,
        *pump_exchange_obstructions,
    )
    if obstructions:
        return obstructions
    return ClosedVascularSystem(
        pump=PumpOrgan(region_by_id[pump_region_id]),
        exchange_beds=exchange_beds,
        carrier_radius_scale=float(carrier_radius_scale),
        distance_decay=float(distance_decay),
    )


def _decode_exchange_bed(
    payload: object,
    index: int,
    region_by_id: dict[str, AnatomyRegion],
) -> CapillaryBed | tuple[AnatomyInputObstruction, ...]:
    address = f"/anatomy/overall/circulation/exchange_beds/{index}"
    if not isinstance(payload, dict):
        return (MalformedAnatomyObstruction(address, "expected an object"),)
    region_id = payload.get("region_id")
    tissue = payload.get("tissue")
    demand = payload.get("demand")
    envelope = payload.get("tissue_envelope")
    obstructions: tuple[AnatomyInputObstruction, ...] = (
        ()
        if region_id in region_by_id
        else (
            UnknownAnatomyRegionObstruction(
                f"{address}/region_id", region_id
            ),
        )
    ) + (
        ()
        if tissue in _TISSUE_KIND_VALUES
        else (
            MalformedAnatomyObstruction(
                f"{address}/tissue",
                f"expected one of {_TISSUE_KIND_VALUES!r}",
            ),
        )
    ) + (
        ()
        if _is_finite_positive_number(demand)
        else (
            MalformedAnatomyObstruction(
                f"{address}/demand", "expected a finite positive number"
            ),
        )
    ) + (
        ()
        if envelope is None
        or (
            isinstance(envelope, dict)
            and _is_finite_positive_number(envelope.get("minimum_radius"))
        )
        else (
            MalformedAnatomyObstruction(
                f"{address}/tissue_envelope/minimum_radius",
                "expected a finite positive number",
            ),
        )
    )
    if obstructions:
        return obstructions
    return CapillaryBed(
        region=region_by_id[region_id],
        tissue=PerfusedTissueKind(tissue),
        demand=float(demand),
        tissue_minimum_radius=(
            float(envelope["minimum_radius"])
            if isinstance(envelope, dict)
            else None
        ),
    )


def _decode_myotendinous_unit(
    payload: object,
    index: int,
    bone_ids: frozenset[str],
    region_by_id: dict[str, AnatomyRegion],
) -> MyotendinousUnit | tuple[AnatomyInputObstruction, ...]:
    address = f"/anatomy/overall/myotendinous_units/{index}"
    if not isinstance(payload, dict):
        return (MalformedAnatomyObstruction(address, "expected an object"),)
    muscle_id = payload.get("muscle_id")
    region_id = payload.get("region_id")
    tendon_radius = payload.get("tendon_radius")
    belly_radius = payload.get("belly_radius")
    origin_result = _decode_tissue_anchor(
        payload.get("origin"), f"{address}/origin", bone_ids
    )
    insertion_result = _decode_tissue_anchor(
        payload.get("insertion"), f"{address}/insertion", bone_ids
    )
    joint_offset = payload.get("joint_offset")
    field_obstructions: tuple[AnatomyInputObstruction, ...] = (
        ()
        if isinstance(muscle_id, str) and bool(muscle_id)
        else (
            MalformedAnatomyObstruction(
                f"{address}/muscle_id", "expected a non-empty string"
            ),
        )
    ) + (
        ()
        if isinstance(region_id, str) and region_id in region_by_id
        else (
            UnknownAnatomyRegionObstruction(
                f"{address}/region_id", region_id
            ),
        )
    ) + tuple(
        obstruction
        for result in (origin_result, insertion_result)
        if isinstance(result, tuple)
        for obstruction in result
    ) + (
        ()
        if _is_finite_pair(joint_offset)
        else (
            MalformedAnatomyObstruction(
                f"{address}/joint_offset",
                "expected two finite local cross-section coordinates",
            ),
        )
    ) + (
        ()
        if _is_finite_positive_number(tendon_radius)
        else (
            MalformedAnatomyObstruction(
                f"{address}/tendon_radius",
                "expected a finite positive number",
            ),
        )
    ) + (
        ()
        if _is_finite_positive_number(belly_radius)
        and _is_finite_positive_number(tendon_radius)
        and float(belly_radius) >= float(tendon_radius)
        else (
            MalformedAnatomyObstruction(
                f"{address}/belly_radius",
                "expected a finite number not smaller than tendon_radius",
            ),
        )
    )
    if field_obstructions:
        return field_obstructions
    origin = cast(TissueAnchor, origin_result)
    insertion = cast(TissueAnchor, insertion_result)
    checked_joint_offset = cast(list[float] | tuple[float, ...], joint_offset)
    return MyotendinousUnit(
        muscle_id=muscle_id,
        region_id=region_id,
        origin=origin,
        joint=TissueAnchor(
            bone_id=insertion.bone_id,
            parameter=0.0,
            offset=tuple(map(float, checked_joint_offset)),
        ),
        insertion=insertion,
        tendon_radius=float(tendon_radius),
        belly_radius=float(belly_radius),
    )


def _decode_tissue_anchor(
    payload: object,
    address: str,
    bone_ids: frozenset[str],
) -> TissueAnchor | tuple[AnatomyInputObstruction, ...]:
    if not isinstance(payload, dict):
        return (MalformedAnatomyObstruction(address, "expected an object"),)
    bone_id = payload.get("bone_id")
    parameter = payload.get("parameter")
    offset = payload.get("offset")
    obstructions: tuple[AnatomyInputObstruction, ...] = (
        ()
        if isinstance(bone_id, str) and bone_id in bone_ids
        else (InvalidSkeletonAddressObstruction(f"{address}/bone_id", bone_id),)
    ) + (
        ()
        if _is_unit_interval_number(parameter)
        else (
            MalformedAnatomyObstruction(
                f"{address}/parameter", "expected a finite number in [0, 1]"
            ),
        )
    ) + (
        ()
        if _is_finite_pair(offset)
        else (
            MalformedAnatomyObstruction(
                f"{address}/offset",
                "expected two finite local cross-section coordinates",
            ),
        )
    )
    if obstructions:
        return obstructions
    checked_offset = cast(list[float] | tuple[float, ...], offset)
    return TissueAnchor(
        bone_id=bone_id,
        parameter=float(parameter),
        offset=tuple(map(float, checked_offset)),
    )


def _decode_integument_layer(
    payload: object,
    index: int,
    region_by_id: dict[str, AnatomyRegion],
) -> IntegumentLayer | tuple[AnatomyInputObstruction, ...]:
    address = f"/anatomy/overall/integument_layers/{index}"
    if not isinstance(payload, dict):
        return (MalformedAnatomyObstruction(address, "expected an object"),)
    region_id = payload.get("region_id")
    thickness = payload.get("thickness")
    formation_declared = "formation" in payload
    authored_formation = payload.get("formation")
    formation = next(
        (
            candidate
            for candidate in SkinFormationKind
            if authored_formation == candidate.value
        ),
        None,
    )
    obstructions: tuple[AnatomyInputObstruction, ...] = (
        ()
        if isinstance(region_id, str) and region_id in region_by_id
        else (
            UnknownAnatomyRegionObstruction(
                f"{address}/region_id", region_id
            ),
        )
    ) + (
        ()
        if _is_finite_positive_number(thickness)
        else (
            MalformedAnatomyObstruction(
                f"{address}/thickness", "expected a finite positive number"
            ),
        )
    ) + (
        ()
        if not formation_declared or formation is not None
        else (
            MalformedAnatomyObstruction(
                f"{address}/formation",
                "expected one of "
                f"{tuple(candidate.value for candidate in SkinFormationKind)!r}",
            ),
        )
    )
    return (
        obstructions
        if obstructions
        else IntegumentLayer(
            region_id=region_id,
            thickness=float(thickness),
            formation=formation,
        )
    )


def _integument_formation_obstructions(
    overall: OverallAnatomy,
) -> tuple[IncompatibleIntegumentFormationObstruction, ...]:
    formed_layers = tuple(
        layer for layer in overall.integument if layer.formation is not None
    )
    legacy_region_ids = tuple(
        layer.region_id
        for layer in overall.integument
        if layer.formation is None
    )
    formed_region_ids = tuple(layer.region_id for layer in formed_layers)
    formed_region_set = frozenset(formed_region_ids)
    uncovered_region_ids = tuple(
        region.region_id
        for region in overall.regions
        if region.region_id not in formed_region_set
    )
    formed_thicknesses = tuple(layer.thickness for layer in formed_layers)
    return (
        (
            IncompatibleIntegumentFormationObstruction(
                formed_region_ids=formed_region_ids,
                legacy_region_ids=legacy_region_ids,
                uncovered_region_ids=uncovered_region_ids,
                formed_thicknesses=formed_thicknesses,
            ),
        )
        if formed_layers
        and (
            legacy_region_ids
            or uncovered_region_ids
            or len(frozenset(formed_thicknesses)) != 1
        )
        else ()
    )


def _validate_tissue_anatomy(
    overall: OverallAnatomy,
    bones: dict[str, dict],
    bone_ids: frozenset[str],
) -> tuple[AnatomyInputObstruction, ...]:
    skeletal_muscle_regions = frozenset(
        bed.region.region_id
        for bed in overall.circulation.exchange_beds
        if bed.tissue is PerfusedTissueKind.SKELETAL_MUSCLE
    )
    integument_regions = frozenset(
        layer.region_id for layer in overall.integument
    )
    return (
        *tuple(
            map(
                DuplicateMyotendinousUnitObstruction,
                _duplicate_values(
                    tuple(muscle.muscle_id for muscle in overall.muscles)
                ),
            )
        ),
        *tuple(
            InvalidMyotendinousPathObstruction(
                muscle.muscle_id,
                muscle.origin.bone_id,
                muscle.insertion.bone_id,
            )
            for muscle in overall.muscles
            if muscle.origin.bone_id in bone_ids
            and muscle.insertion.bone_id in bone_ids
            and (
                bones[muscle.insertion.bone_id].get("parent")
                != muscle.origin.bone_id
                or muscle.joint.bone_id != muscle.insertion.bone_id
                or muscle.joint.parameter != 0.0
                or muscle.origin.parameter >= 1.0
                or muscle.insertion.parameter <= 0.0
            )
        ),
        *tuple(
            UnperfusedMyotendinousUnitObstruction(
                muscle.muscle_id, muscle.region_id
            )
            for muscle in overall.muscles
            if muscle.region_id not in skeletal_muscle_regions
        ),
        *tuple(
            UnsupportedTissueCarrierObstruction(
                muscle.muscle_id, anchor.bone_id
            )
            for muscle in overall.muscles
            for anchor in (muscle.origin, muscle.insertion)
            if _minimum_carrier_sample(
                bones[anchor.bone_id].get("flesh", ())
            )
            is None
        ),
        *tuple(
            MissingSkeletalCrossSectionObstruction(
                muscle.muscle_id, anchor.bone_id
            )
            for muscle in overall.muscles
            for anchor in (muscle.origin, muscle.insertion)
            if not _is_bone_cross_section(
                bones[anchor.bone_id].get("bone_radii")
            )
        ),
        *tuple(
            map(
                DuplicateIntegumentLayerObstruction,
                _duplicate_values(
                    tuple(layer.region_id for layer in overall.integument)
                ),
            )
        ),
        *tuple(
            MissingIntegumentLayerObstruction(
                muscle.muscle_id, muscle.region_id
            )
            for muscle in overall.muscles
            if muscle.region_id not in integument_regions
        ),
        *_integument_formation_obstructions(overall),
    )
