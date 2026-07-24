"""Anatomy and vasculature projection to serialized dict forms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, assert_never

from golem.kernel.anatomy.graph import (
    AcceptedVasculature,
    CapsuleEscapeObstruction,
    CollectingVenule,
    DemandDeliveryMismatchObstruction,
    DisconnectedVascularGraphObstruction,
    DistributingArtery,
    EmptyPerfusionTerritoryObstruction,
    InfeasibleBifurcationObstruction,
    InsufficientTerminalSitesObstruction,
    InvalidStructuralGuidanceObstruction,
    MissingProvenanceObstruction,
    RejectedVasculature,
    ResistanceArteriole,
    ReturningVein,
    ReversedVascularFlowObstruction,
    SymmetryMismatchObstruction,
    VascularGluingInterrogation,
    VascularGluingObstruction,
    VascularIntersectionObstruction,
    VascularSearchBudgetObstruction,
    VascularSolverResidualObstruction,
)
from golem.kernel.anatomy.vocabulary import (
    AcceptedAnatomy,
    DisconnectedAnatomyRegionObstruction,
    DuplicateAnatomyRegionObstruction,
    DuplicateExchangeBedObstruction,
    DuplicateIntegumentLayerObstruction,
    DuplicateMyotendinousUnitObstruction,
    DuplicateRegionHostObstruction,
    IncompatibleIntegumentFormationObstruction,
    InsufficientVascularStagesObstruction,
    InvalidMyotendinousPathObstruction,
    InvalidSkeletonAddressObstruction,
    MalformedAnatomyObstruction,
    MissingIntegumentLayerObstruction,
    MissingSkeletalCrossSectionObstruction,
    MissingTerminalRegionObstruction,
    NonterminalExchangeRegionObstruction,
    RejectedAnatomy,
    SealedVascularConfig,
    UnknownAnatomyRegionObstruction,
    UnperfusedMyotendinousUnitObstruction,
    UnreferencedAnatomyRegionObstruction,
    UnsupportedTissueCarrierObstruction,
    _CLOSED_VASCULAR_KIND,
)

if TYPE_CHECKING:
    from golem.kernel.anatomy.graph import (
        VascularSegmentGeometry,
        VascularSegment,
        VasculatureObstruction,
        VasculatureResult,
    )
    from golem.kernel.anatomy.vocabulary import (
        AnatomyCarrierRow,
        AnatomyObstruction,
        AnatomyResult,
        CirculationCircuit,
        MyotendinousUnit,
        OverallAnatomy,
        TissueAnchor,
        TissueEnvelopeSection,
    )


def anatomy_to_dict(result: AnatomyResult) -> dict[str, object]:
    match result:
        case RejectedAnatomy(obstructions):
            return {
                "status": "rejected",
                "obstructions": tuple(
                    map(project_anatomy_obstruction, obstructions)
                ),
            }
        case AcceptedAnatomy() as accepted:
            total_effective_demand = math.fsum(
                map(lambda circuit: circuit.effective_demand, accepted.circuits)
            )
            authored_values, derived_values = accepted.authoring_counts
            maximum_free_imbalance, boundary_relative_error = (
                accepted.conservation
            )
            return {
                "status": "accepted",
                "overall": _overall_dict(accepted.overall),
                "circulation_kind": _CLOSED_VASCULAR_KIND,
                "circuits": tuple(
                    _circuit_dict(circuit, total_effective_demand)
                    for circuit in accepted.circuits
                ),
                "carrier_rows": tuple(
                    map(_carrier_row_dict, accepted.carrier_rows)
                ),
                "tissue_envelopes": tuple(
                    map(_tissue_envelope_dict, accepted.tissue_envelopes)
                ),
                "terminal_coverage": 1.0,
                "maximum_normalized_free_cell_imbalance": maximum_free_imbalance,
                "boundary_relative_error": boundary_relative_error,
                "authored_numeric_values": authored_values,
                "derived_radius_values": derived_values,
                "authoring_leverage": derived_values / authored_values,
            }
        case _ as unreachable:
            assert_never(unreachable)


def vasculature_to_dict(result: VasculatureResult) -> dict[str, object]:
    match result:
        case RejectedVasculature(obstructions):
            return {
                "status": "rejected",
                "obstructions": tuple(
                    map(project_vascular_obstruction, obstructions)
                ),
                "failing_segments": tuple(
                    _vascular_segment_geometry_dict(segment)
                    for obstruction in obstructions
                    for segment in getattr(
                        obstruction,
                        "failing_segments",
                        (),
                    )
                ),
            }
        case AcceptedVasculature(graph, receipt):
            return {
                "status": "accepted",
                "topology": {
                    "nodes": len(graph.nodes),
                    "edges": len(graph.edges),
                    "terminal_pairs": receipt.terminal_pair_budget,
                },
                "terminal_pairs_by_region": dict(receipt.terminal_pairs_by_region),
                "stage_counts": dict(receipt.stage_counts),
                "minimum_capsule_margin": receipt.minimum_capsule_margin,
                "delivered_demand_by_region": dict(
                    receipt.delivered_demand_by_region
                ),
                "pump_pressure_drop": receipt.pump_pressure_drop,
                "maximum_delivery_relative_error": (
                    receipt.maximum_delivery_relative_error
                ),
                "maximum_free_cell_residual": receipt.maximum_free_cell_residual,
                "boundary_balance_error": receipt.boundary_balance_error,
                "conservation_error": receipt.conservation_error,
                "maximum_symmetry_error": receipt.maximum_symmetry_error,
                "checked_nonincident_pair_count": (
                    receipt.checked_nonincident_pair_count
                ),
                "structural_guidance_sample_count": (
                    receipt.structural_guidance_sample_count
                ),
                "maximum_structural_cost_multiplier": (
                    receipt.maximum_structural_cost_multiplier
                ),
                **(
                    {
                        "search_effort_by_region": {
                            region_id: {
                                "deepest_stage": effort.deepest_stage.value,
                                "evaluated_state_count": (
                                    effort.evaluated_state_count
                                ),
                            }
                            for region_id, effort in (
                                receipt.search_effort_by_region
                            )
                        }
                    }
                    if receipt.search_effort_by_region
                    else {}
                ),
            }
        case _ as unreachable:
            assert_never(unreachable)


def project_vascular_obstruction(
    obstruction: VasculatureObstruction,
) -> dict[str, object]:
    match obstruction:
        case MissingProvenanceObstruction(region_id, bone_id):
            return {
                "kind": "MissingProvenance",
                "address": f"skeleton/{bone_id}",
                "reason": f"region {region_id!r} has no flesh provenance",
            }
        case EmptyPerfusionTerritoryObstruction(region_id):
            return {
                "kind": "EmptyPerfusionTerritory",
                "address": f"anatomy/regions/{region_id}",
            }
        case InsufficientTerminalSitesObstruction(region_id, requested, available):
            return {
                "kind": "InsufficientTerminalSites",
                "address": f"anatomy/regions/{region_id}",
                "predicate": "available terminal sites meet the requested count",
                "required": requested,
                "observed": available,
                "requested": requested,
                "available": available,
            }
        case InfeasibleBifurcationObstruction() as failure:
            return {
                "kind": "InfeasibleBifurcation",
                "address": (
                    f"anatomy/regions/{failure.region_id}"
                    if failure.terminal_index is None
                    else (
                        f"anatomy/regions/{failure.region_id}"
                        f"/terminals/{failure.terminal_index}"
                    )
                ),
                "region_id": failure.region_id,
                "phase": failure.phase.value,
                "lane": failure.lane.value,
                "predicate": failure.predicate.value,
                "required": failure.required,
                "observed": failure.observed,
                "terminal_index": failure.terminal_index,
                "supply_capsule": failure.supply_capsule,
                "return_capsule": failure.return_capsule,
                "split_edge": failure.split_edge,
                "candidate_point_index": failure.candidate_point_index,
                "attempted_candidate_count": (
                    failure.attempted_candidate_count
                ),
                "failing_segments": tuple(
                    map(
                        _vascular_segment_geometry_dict,
                        failure.failing_segments,
                    )
                ),
            }
        case VascularSearchBudgetObstruction() as failure:
            return {
                "kind": "VascularSearchBudget",
                "address": f"anatomy/regions/{failure.region_id}",
                "predicate": "vascular search completes within its state budget",
                "required": {
                    "completed": True,
                    "maximum_evaluated_states": failure.required_state_budget,
                },
                "observed": {
                    "completed": False,
                    "evaluated_states": failure.observed_evaluated_states,
                    "remaining_queue_size": failure.remaining_queue_size,
                    "limiting_required": failure.limiting_required,
                    "limiting_observed": failure.limiting_observed,
                },
                "region_id": failure.region_id,
                "phase": failure.phase.value,
                "lane": failure.lane.value,
                "required_state_budget": failure.required_state_budget,
                "observed_evaluated_states": (
                    failure.observed_evaluated_states
                ),
                "remaining_queue_size": failure.remaining_queue_size,
                "limiting_predicate": failure.limiting_predicate.value,
                "limiting_required": failure.limiting_required,
                "limiting_observed": failure.limiting_observed,
                "failing_segments": tuple(
                    map(
                        _vascular_segment_geometry_dict,
                        failure.failing_segments,
                    )
                ),
            }
        case CapsuleEscapeObstruction() as failure:
            return {
                "kind": "CapsuleEscape",
                "address": failure.edge_id,
                "predicate": "minimum capsule margin",
                "required": SealedVascularConfig().wall_clearance,
                "observed": failure.margin,
                "margin": failure.margin,
                "failing_segments": tuple(
                    map(
                        _vascular_segment_geometry_dict,
                        failure.failing_segments,
                    )
                ),
            }
        case VascularIntersectionObstruction() as failure:
            return {
                "kind": "Intersection",
                "address": failure.left_edge_id,
                "predicate": "minimum nonincident vessel clearance",
                "required": SealedVascularConfig().vessel_clearance,
                "observed": failure.clearance,
                "other": failure.right_edge_id,
                "clearance": failure.clearance,
                "failing_segments": tuple(
                    map(
                        _vascular_segment_geometry_dict,
                        failure.failing_segments,
                    )
                ),
            }
        case SymmetryMismatchObstruction() as failure:
            return {
                "kind": "SymmetryMismatch",
                "address": failure.witness.node_id,
                "region_id": failure.region_id,
                "predicate": failure.predicate.value,
                "required": failure.required,
                "observed": failure.observed,
                "witness": {
                    "node_id": failure.witness.node_id,
                    "expected_mirror_node_id": (
                        failure.witness.expected_mirror_node_id
                    ),
                },
            }
        case VascularGluingObstruction() as failure:
            projection: dict[str, object] = {
                "kind": "FailedGluing",
                "address": failure.interface_address,
                "reason": failure.reason,
            }
            interrogation = failure.gluing_interrogation
            if interrogation is None:
                return projection
            projected_interrogation = _gluing_interrogation_dict(
                interrogation, failure.failing_segments
            )
            stop_law = (
                {
                    "predicate": "maximum free-cell residual at solver stop",
                    "required": interrogation.demand.residual_tolerance,
                    "observed": interrogation.demand.maximum_stop_residual,
                    "unit": "absolute_residual",
                }
                if (
                    interrogation.demand.maximum_stop_residual
                    > interrogation.demand.residual_tolerance
                )
                else {
                    "predicate": "relative free-system residual at solver stop",
                    "required": interrogation.demand.relative_tolerance,
                    "observed": interrogation.demand.relative_residual,
                    "unit": "relative_residual",
                }
            )
            return {
                **projection,
                **stop_law,
                "interrogation": projected_interrogation,
                "failing_segments": tuple(
                    map(
                        _vascular_segment_geometry_dict,
                        failure.failing_segments,
                    )
                ),
            }
        case DisconnectedVascularGraphObstruction(node_id):
            return {"kind": "Disconnection", "address": node_id}
        case VascularSolverResidualObstruction(residual, tolerance):
            return {
                "kind": "SolverResidual",
                "address": "closed-vascular-field",
                "predicate": "maximum vascular solver residual",
                "required": tolerance,
                "observed": residual,
                "residual": residual,
                "tolerance": tolerance,
            }
        case DemandDeliveryMismatchObstruction(region_id, error, tolerance):
            return {
                "kind": "DemandDeliveryMismatch",
                "address": f"anatomy/regions/{region_id}",
                "predicate": "maximum demand delivery relative error",
                "required": tolerance,
                "observed": error,
                "relative_error": error,
                "tolerance": tolerance,
            }
        case ReversedVascularFlowObstruction() as failure:
            return {
                "kind": "ReversedFlow",
                "address": failure.edge_id,
                "predicate": "vascular edge flow is strictly positive",
                "required": 0.0,
                "observed": failure.solved_flow,
                "solved_flow": failure.solved_flow,
                "failing_segments": tuple(
                    map(
                        _vascular_segment_geometry_dict,
                        failure.failing_segments,
                    )
                ),
            }
        case InvalidStructuralGuidanceObstruction(address, reason):
            return {
                "kind": "InvalidStructuralGuidance",
                "address": address,
                "reason": reason,
            }
        case _ as unreachable:
            assert_never(unreachable)


def _section_address(node_id: str, region_id: str | None) -> str:
    return (
        f"anatomy/regions/{region_id}"
        if region_id is not None
        else f"closed-vascular-field/nodes/{node_id}"
    )


def _segment_authoring_addresses(
    segments: tuple[VascularSegmentGeometry, ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            address
            for segment in segments
            for address in (
                *segment.source_host_part_addresses,
                *segment.source_host_bone_addresses,
                *segment.target_host_part_addresses,
                *segment.target_host_bone_addresses,
            )
        )
    )


def _gluing_interrogation_dict(
    interrogation: VascularGluingInterrogation,
    segments: tuple[VascularSegmentGeometry, ...],
) -> dict[str, object]:
    conditioning_sections = tuple(
        {
            "node_id": section.node_id,
            "region_id": section.region_id,
            "address": _section_address(section.node_id, section.region_id),
            "diagonal": section.diagonal,
            "conductance_ratio": section.conductance_ratio,
        }
        for section in interrogation.conditioning.worst_sections
    )
    demand_sections = tuple(
        {
            "node_id": section.node_id,
            "region_id": section.region_id,
            "address": _section_address(section.node_id, section.region_id),
            "residual": section.residual,
        }
        for section in interrogation.demand.worst_sections
    )
    suspect_addresses = tuple(
        dict.fromkeys(
            (
                *(str(section["address"]) for section in conditioning_sections),
                *(str(section["address"]) for section in demand_sections),
                *_segment_authoring_addresses(segments),
            )
        )
    )
    return {
        "hypotheses": {
            "topology": {
                "predicate": "validated topology remains glued by its weakest interfaces",
                "witness": {
                    "free_component_count": (
                        interrogation.topology.free_component_count
                    ),
                    "weakest_bridges": tuple(
                        {
                            "edge_id": bridge.edge_id,
                            "address": (
                                f"closed-vascular-field/edges/{bridge.edge_id}"
                            ),
                            "conductance": bridge.conductance,
                        }
                        for bridge in interrogation.topology.weakest_bridges
                    ),
                },
            },
            "conditioning": {
                "predicate": "balance operator is sufficiently conditioned",
                "witness": {
                    "gershgorin_upper_bound": (
                        interrogation.conditioning.gershgorin_upper_bound
                    ),
                    "diagonal_spread": (
                        interrogation.conditioning.diagonal_spread
                    ),
                    "worst_sections": conditioning_sections,
                },
            },
            "demand": {
                "predicate": "local balance demands are met at solver stop",
                "witness": {
                    "maximum_stop_residual": (
                        interrogation.demand.maximum_stop_residual
                    ),
                    "residual_tolerance": (
                        interrogation.demand.residual_tolerance
                    ),
                    "relative_residual": (
                        interrogation.demand.relative_residual
                    ),
                    "relative_tolerance": (
                        interrogation.demand.relative_tolerance
                    ),
                    "iteration_cap": interrogation.demand.iteration_cap,
                    "iterations_used": interrogation.demand.iterations_used,
                    "exhausted": interrogation.demand.exhausted,
                    "worst_sections": demand_sections,
                },
            },
            "budget": {
                "predicate": "balance solve converges within the sealed budget",
                "witness": {
                    "iteration_cap": interrogation.budget.iteration_cap,
                    "iterations_used": interrogation.budget.iterations_used,
                    "relative_tolerance": (
                        interrogation.budget.relative_tolerance
                    ),
                    "exhausted": interrogation.budget.exhausted,
                },
            },
        },
        "suspect_addresses": suspect_addresses,
    }


def _vascular_segment_geometry_dict(
    segment: VascularSegmentGeometry,
) -> dict[str, object]:
    return {
        "edge_id": segment.edge_id,
        "source": {
            "position": segment.source_position,
            "host_bone_addresses": segment.source_host_bone_addresses,
            "host_part_addresses": segment.source_host_part_addresses,
        },
        "target": {
            "position": segment.target_position,
            "host_bone_addresses": segment.target_host_bone_addresses,
            "host_part_addresses": segment.target_host_part_addresses,
        },
        "radius": segment.radius,
    }


def _overall_dict(overall: OverallAnatomy) -> dict[str, object]:
    pump_region = overall.circulation.pump.region
    return {
        "regions": tuple(
            {
                "region_id": region.region_id,
                "kind": region.kind.value,
                "host_bone_id": region.host_bone_id,
            }
            for region in overall.regions
        ),
        "pump": {
            "kind": "pump_organ",
            "region_id": pump_region.region_id,
            "host_bone_id": pump_region.host_bone_id,
        },
        "myotendinous_units": tuple(
            map(_myotendinous_unit_dict, overall.muscles)
        ),
        "integument_layers": tuple(
            {
                "region_id": layer.region_id,
                "thickness": layer.thickness,
                **(
                    {"formation": layer.formation.value}
                    if layer.formation is not None
                    else {}
                ),
            }
            for layer in overall.integument
        ),
    }


def _myotendinous_unit_dict(
    muscle: MyotendinousUnit,
) -> dict[str, object]:
    return {
        "muscle_id": muscle.muscle_id,
        "region_id": muscle.region_id,
        "origin": _tissue_anchor_dict(muscle.origin),
        "joint": _tissue_anchor_dict(muscle.joint),
        "insertion": _tissue_anchor_dict(muscle.insertion),
        "tendon_radius": muscle.tendon_radius,
        "belly_radius": muscle.belly_radius,
    }


def _tissue_anchor_dict(anchor: TissueAnchor) -> dict[str, object]:
    return {
        "bone_id": anchor.bone_id,
        "parameter": anchor.parameter,
        "offset": anchor.offset,
    }


def _circuit_dict(
    circuit: CirculationCircuit, total_effective_demand: float
) -> dict[str, object]:
    capillary_bed = circuit.capillary_bed
    return {
        "exchange_region_id": capillary_bed.region.region_id,
        "host_bone_id": capillary_bed.region.host_bone_id,
        "tissue": capillary_bed.tissue.value,
        "demand": capillary_bed.demand,
        "effective_demand": circuit.effective_demand,
        "flow_fraction": circuit.effective_demand / total_effective_demand,
        "tissue_minimum_radius": capillary_bed.tissue_minimum_radius,
        "is_complete": True,
        "bones": circuit.bones,
        "supply": (
            *tuple(map(_vascular_segment_dict, circuit.distributing_arteries)),
            _vascular_segment_dict(circuit.resistance_arteriole),
        ),
        "exchange": {
            "kind": "capillary_bed",
            "region_id": capillary_bed.region.region_id,
            "host_bone_id": capillary_bed.region.host_bone_id,
            "tissue": capillary_bed.tissue.value,
        },
        "return": (
            _vascular_segment_dict(circuit.collecting_venule),
            *tuple(map(_vascular_segment_dict, circuit.returning_veins)),
        ),
    }


def _vascular_segment_dict(segment: VascularSegment) -> dict[str, str]:
    match segment:
        case DistributingArtery(interface_address):
            return {
                "kind": "distributing_artery",
                "interface_address": interface_address,
            }
        case ResistanceArteriole(interface_address):
            return {
                "kind": "resistance_arteriole",
                "interface_address": interface_address,
            }
        case CollectingVenule(interface_address):
            return {
                "kind": "collecting_venule",
                "interface_address": interface_address,
            }
        case ReturningVein(interface_address):
            return {
                "kind": "returning_vein",
                "interface_address": interface_address,
            }
        case _ as unreachable:
            assert_never(unreachable)


def _carrier_row_dict(row: AnatomyCarrierRow) -> dict[str, object]:
    return {
        "bone_id": row.bone_id,
        "interface_address": row.interface_address,
        "shape_address": row.shape_address,
        "normalized_flow": row.normalized_flow,
        "distance_from_pump": row.distance_from_pump,
        "circulation_minimum_radius": row.circulation_minimum_radius,
        "tissue_minimum_radius": row.tissue_minimum_radius,
        "controlling_constraint": row.controlling_constraint.value,
        "target_minimum_radius": row.target_minimum_radius,
        "measured_minimum_radius": row.measured_minimum_radius,
        "deficit": row.deficit,
        "suggested_scale": row.suggested_scale,
        "suggestion": (
            {"operation": "scale_radius", "factor": row.suggested_scale}
            if row.suggested_scale > 1.0
            else None
        ),
    }


def _tissue_envelope_dict(
    envelope: TissueEnvelopeSection,
) -> dict[str, object]:
    return {
        "bone_id": envelope.bone_id,
        "shape_address": envelope.shape_address,
        "muscle_ids": envelope.muscle_ids,
        "stations": tuple(
            {
                "parameter": station.parameter,
                "offset": station.offset,
                "half_size": station.half_size,
            }
            for station in envelope.stations
        ),
    }


def project_anatomy_obstruction(
    obstruction: AnatomyObstruction,
) -> dict[str, object]:
    match obstruction:
        case MalformedAnatomyObstruction(address, reason):
            return {"kind": "MalformedAnatomy", "address": address, "reason": reason}
        case InvalidSkeletonAddressObstruction(address, value):
            return {
                "kind": "InvalidSkeletonAddress",
                "address": address,
                "reason": f"unknown bone {value!r}",
            }
        case DuplicateAnatomyRegionObstruction(region_id):
            return {
                "kind": "DuplicateAnatomyRegion",
                "address": f"anatomy/regions/{region_id}",
                "reason": f"duplicate region {region_id!r}",
            }
        case DuplicateRegionHostObstruction(host_bone_id):
            return {
                "kind": "DuplicateRegionHost",
                "address": f"skeleton/{host_bone_id}",
                "reason": f"multiple anatomy regions use host {host_bone_id!r}",
            }
        case DuplicateExchangeBedObstruction(region_id):
            return {
                "kind": "DuplicateExchangeBed",
                "address": f"anatomy/regions/{region_id}",
                "reason": f"duplicate exchange bed for {region_id!r}",
            }
        case UnknownAnatomyRegionObstruction(address, region_id):
            return {
                "kind": "UnknownAnatomyRegion",
                "address": address,
                "reason": f"unknown anatomy region {region_id!r}",
            }
        case UnreferencedAnatomyRegionObstruction(region_id):
            return {
                "kind": "UnreferencedAnatomyRegion",
                "address": f"anatomy/regions/{region_id}",
                "reason": "region is neither the pump region nor an exchange region",
            }
        case MissingTerminalRegionObstruction(host_bone_id):
            return {
                "kind": "MissingTerminalRegion",
                "address": f"skeleton/{host_bone_id}",
                "reason": "terminal skeleton region has no capillary exchange bed",
            }
        case NonterminalExchangeRegionObstruction(region_id, host_bone_id):
            return {
                "kind": "NonterminalExchangeRegion",
                "address": f"anatomy/regions/{region_id}",
                "reason": f"exchange host {host_bone_id!r} is not terminal",
            }
        case InsufficientVascularStagesObstruction(region_id, interface_count):
            return {
                "kind": "InsufficientVascularStages",
                "address": f"anatomy/regions/{region_id}",
                "predicate": (
                    "closed circulation has distinct distributing and resistance stages"
                ),
                "required": 2,
                "observed": interface_count,
                "reason": (
                    "closed circulation needs distinct distributing and resistance "
                    f"segments; found {interface_count} interface(s)"
                ),
            }
        case DisconnectedAnatomyRegionObstruction(
            region_id, pump_host_bone_id, exchange_host_bone_id
        ):
            return {
                "kind": "DisconnectedAnatomyRegion",
                "address": f"anatomy/regions/{region_id}",
                "reason": (
                    f"pump host {pump_host_bone_id!r} and exchange host "
                    f"{exchange_host_bone_id!r} do not share a skeleton root"
                ),
            }
        case DuplicateMyotendinousUnitObstruction(muscle_id):
            return {
                "kind": "DuplicateMyotendinousUnit",
                "address": f"anatomy/myotendinous_units/{muscle_id}",
                "reason": f"duplicate myotendinous unit {muscle_id!r}",
            }
        case InvalidMyotendinousPathObstruction(
            muscle_id, origin_bone_id, insertion_bone_id
        ):
            return {
                "kind": "InvalidMyotendinousPath",
                "address": f"anatomy/myotendinous_units/{muscle_id}",
                "reason": (
                    f"origin {origin_bone_id!r} and insertion "
                    f"{insertion_bone_id!r} must cross one declared joint"
                ),
            }
        case UnperfusedMyotendinousUnitObstruction(muscle_id, region_id):
            return {
                "kind": "UnperfusedMyotendinousUnit",
                "address": f"anatomy/myotendinous_units/{muscle_id}",
                "reason": (
                    f"served region {region_id!r} has no skeletal-muscle "
                    "capillary bed"
                ),
            }
        case UnsupportedTissueCarrierObstruction(muscle_id, bone_id):
            return {
                "kind": "UnsupportedTissueCarrier",
                "address": f"anatomy/myotendinous_units/{muscle_id}",
                "reason": f"bone {bone_id!r} has no gencyl flesh carrier",
            }
        case MissingSkeletalCrossSectionObstruction(muscle_id, bone_id):
            return {
                "kind": "MissingSkeletalCrossSection",
                "address": f"anatomy/myotendinous_units/{muscle_id}",
                "reason": f"bone {bone_id!r} has no positive bone_radii section",
            }
        case DuplicateIntegumentLayerObstruction(region_id):
            return {
                "kind": "DuplicateIntegumentLayer",
                "address": f"anatomy/integument_layers/{region_id}",
                "reason": f"duplicate integument layer for {region_id!r}",
            }
        case MissingIntegumentLayerObstruction(muscle_id, region_id):
            return {
                "kind": "MissingIntegumentLayer",
                "address": f"anatomy/myotendinous_units/{muscle_id}",
                "reason": f"served region {region_id!r} has no integument layer",
            }
        case IncompatibleIntegumentFormationObstruction(
            formed_region_ids,
            legacy_region_ids,
            uncovered_region_ids,
            formed_thicknesses,
        ):
            return {
                "kind": "IncompatibleIntegumentFormation",
                "address": "anatomy/integument_layers",
                "predicate": (
                    "formed integument covers every anatomy region at one "
                    "shared thickness"
                ),
                "required": {
                    "legacy_region_ids": (),
                    "uncovered_region_ids": (),
                    "distinct_formed_thickness_count": 1,
                },
                "observed": {
                    "formed_region_ids": formed_region_ids,
                    "legacy_region_ids": legacy_region_ids,
                    "uncovered_region_ids": uncovered_region_ids,
                    "formed_thicknesses": formed_thicknesses,
                },
                "reason": (
                    "static skin formation requires uniform formed "
                    "integument over every anatomy region"
                ),
            }
        case _ as unreachable:
            assert_never(unreachable)
