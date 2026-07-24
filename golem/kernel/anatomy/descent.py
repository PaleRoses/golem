"""Anatomy descent: derive the accepted anatomy and its conservation folds."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from golem.kernel.anatomy.decode import _decode_anatomy
from golem.kernel.anatomy.envelope import (
    _minimum_carrier_sample,
    _tissue_envelope_section,
)
from golem.kernel.anatomy.graph import (
    CollectingVenule,
    DistributingArtery,
    ResistanceArteriole,
    ReturningVein,
)
from golem.kernel.anatomy.lineage import _assembly_path, _interface_address
from golem.kernel.anatomy.vocabulary import (
    AcceptedAnatomy,
    AnatomyCarrierRow,
    CirculationCircuit,
    DisconnectedAnatomyRegionObstruction,
    InsufficientVascularStagesObstruction,
    OverallAnatomy,
    RejectedAnatomy,
    _MINIMUM_CARRIER_RADIUS,
    _MURRAY_BRANCH_EXPONENT,
)

if TYPE_CHECKING:
    from golem.kernel.anatomy.vocabulary import (
        AnatomyResult,
        CapillaryBed,
        ClosedVascularSystem,
    )


def derive_anatomy(
    bones: dict[str, dict],
    order: tuple[str, ...] | list[str],
    payload: object,
) -> AnatomyResult:
    ordered_bones = tuple(order)
    decoded = _decode_anatomy(payload, bones, ordered_bones)
    if not isinstance(decoded, OverallAnatomy):
        return RejectedAnatomy(decoded)
    overall = decoded
    circulation = overall.circulation
    routed_beds = tuple(
        (
            _assembly_path(
                bones,
                circulation.pump.region.host_bone_id,
                exchange_bed.region.host_bone_id,
            ),
            exchange_bed,
        )
        for exchange_bed in circulation.exchange_beds
    )
    route_obstructions = tuple(
        DisconnectedAnatomyRegionObstruction(
            exchange_bed.region.region_id,
            circulation.pump.region.host_bone_id,
            exchange_bed.region.host_bone_id,
        )
        for path_bones, exchange_bed in routed_beds
        if path_bones is None
    )
    if route_obstructions:
        return RejectedAnatomy(route_obstructions)
    circuit_results = tuple(
        _circulation_circuit(path_bones, exchange_bed, bones)
        for path_bones, exchange_bed in routed_beds
        if path_bones is not None
    )
    stage_obstructions = tuple(
        result
        for result in circuit_results
        if isinstance(result, InsufficientVascularStagesObstruction)
    )
    if stage_obstructions:
        return RejectedAnatomy(stage_obstructions)
    circuits = tuple(
        result for result in circuit_results if isinstance(result, CirculationCircuit)
    )
    total_demand = math.fsum(
        map(lambda circuit: circuit.effective_demand, circuits)
    )
    covered_bones = frozenset(
        bone_id for circuit in circuits for bone_id in circuit.bones
    )
    carrier_rows = tuple(
        _anatomy_carrier_row(
            bone_id,
            bones[bone_id],
            circuits,
            total_demand,
            circulation,
        )
        for bone_id in ordered_bones
        if bone_id in covered_bones
    )
    muscle_bones = frozenset(
        anchor.bone_id
        for muscle in overall.muscles
        for anchor in (muscle.origin, muscle.insertion)
    )
    tissue_envelopes = tuple(
        envelope
        for bone_id in ordered_bones
        if bone_id in muscle_bones
        for envelope in (
            _tissue_envelope_section(
                bone_id,
                bones,
                overall.muscles,
                overall.integument,
            ),
        )
        if envelope is not None
    )
    return AcceptedAnatomy(
        overall=overall,
        circuits=circuits,
        carrier_rows=carrier_rows,
        tissue_envelopes=tissue_envelopes,
    )


def _circulation_circuit(
    path_bones: tuple[str, ...],
    capillary_bed: CapillaryBed,
    bones: dict[str, dict],
) -> CirculationCircuit | InsufficientVascularStagesObstruction:
    interfaces = tuple(
        _interface_address(left, right, bones)
        for left, right in zip(path_bones, path_bones[1:])
    )
    match interfaces:
        case (first_distributing_interface, *middle_interfaces, final_interface):
            distributing_interfaces = (
                first_distributing_interface,
                *middle_interfaces,
            )
            return CirculationCircuit(
                capillary_bed=capillary_bed,
                effective_demand=(
                    capillary_bed.demand
                    * (
                        2.0
                        if bones[capillary_bed.region.host_bone_id]["mirrored"]
                        else 1.0
                    )
                ),
                bones=path_bones,
                distributing_arteries=tuple(
                    map(DistributingArtery, distributing_interfaces)
                ),
                resistance_arteriole=ResistanceArteriole(final_interface),
                collecting_venule=CollectingVenule(final_interface),
                returning_veins=tuple(
                    map(ReturningVein, reversed(distributing_interfaces))
                ),
            )
        case _:
            return InsufficientVascularStagesObstruction(
                capillary_bed.region.region_id, len(interfaces)
            )


def _anatomy_carrier_row(
    bone_id: str,
    bone: dict,
    circuits: tuple[CirculationCircuit, ...],
    total_demand: float,
    circulation: ClosedVascularSystem,
) -> AnatomyCarrierRow:
    local_circuits = tuple(
        circuit for circuit in circuits if bone_id in circuit.bones
    )
    flow = math.fsum(map(lambda circuit: circuit.effective_demand, local_circuits))
    distance = min(circuit.bones.index(bone_id) for circuit in local_circuits)
    circulation_target = max(
        _MINIMUM_CARRIER_RADIUS,
        circulation.carrier_radius_scale
        * (flow / total_demand) ** (1.0 / _MURRAY_BRANCH_EXPONENT)
        * circulation.distance_decay**distance,
    )
    tissue_target = next(
        map(
            lambda circuit: circuit.capillary_bed.tissue_minimum_radius,
            filter(
                lambda circuit: circuit.capillary_bed.region.host_bone_id
                == bone_id
                and circuit.capillary_bed.tissue_minimum_radius is not None,
                local_circuits,
            ),
        ),
        None,
    )
    sample = _minimum_carrier_sample(bone.get("flesh", ()))
    return AnatomyCarrierRow(
        bone_id=bone_id,
        interface_address=(
            f"skeleton/{bone_id}/joint"
            if bone.get("parent") is not None
            else f"skeleton/{bone_id}"
        ),
        shape_address=(
            f"skeleton/{bone_id}/flesh[{sample.flesh_index}]"
            if sample is not None
            else None
        ),
        normalized_flow=flow / total_demand,
        distance_from_pump=distance,
        circulation_minimum_radius=circulation_target,
        tissue_minimum_radius=tissue_target,
        measured_minimum_radius=(
            sample.minimum_radius if sample is not None else None
        ),
    )


def _conservation(
    circuits: tuple[CirculationCircuit, ...],
    rows: tuple[AnatomyCarrierRow, ...],
) -> tuple[float, float]:
    flow_by_bone = {row.bone_id: row.normalized_flow for row in rows}
    oriented_edges = frozenset(
        edge
        for circuit in circuits
        for edge in zip(circuit.bones, circuit.bones[1:])
    )
    boundary_bones = frozenset(
        (
            circuits[0].bones[0],
            *tuple(
                map(
                    lambda circuit: circuit.capillary_bed.region.host_bone_id,
                    circuits,
                )
            ),
        )
    )
    maximum_free_imbalance = max(
        (
            abs(
                flow_by_bone[bone_id]
                - math.fsum(
                    flow_by_bone[right]
                    for left, right in oriented_edges
                    if left == bone_id
                )
            )
            / max(flow_by_bone[bone_id], 1.0e-12)
            for bone_id in flow_by_bone.keys() - boundary_bones
        ),
        default=0.0,
    )
    source_flow = flow_by_bone[circuits[0].bones[0]]
    terminal_flow = math.fsum(
        flow_by_bone[circuit.capillary_bed.region.host_bone_id]
        for circuit in circuits
    )
    return (
        maximum_free_imbalance,
        abs(source_flow - terminal_flow) / max(source_flow, 1.0e-12),
    )
