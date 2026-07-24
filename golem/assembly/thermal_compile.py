"""Lower assembly heat intent into the authoritative kernel sheaf algebra."""

from __future__ import annotations

import math
import numpy as np

from golem.assembly.address import _domain_cell_indices_owned_by_provenance
from golem.assembly.carriers import RejectedAssembly, _CompiledElement, _addresses, _reference_temperature_kelvin
from golem.assembly.descent import collect_results
from golem.assembly.materials_binding import _solid_material_for_cell
from golem.assembly.obstructions import (EmptyThermalRegionObstruction, MissingPhysicalVasculatureObstruction, UnsupportedThermalAddressObstruction)
from golem.assembly.service import ElementAddress, RegionAddress, ServiceCase, ServiceIntent
from golem.assembly.thermal_types import _AssembledThermalProblem, _ResolvedThermalSource
from golem.assembly.voxel import (_ResolvedMaterialDomain, _WallBinding, _cell_center_metres, _surface_face_specs)
from golem.kernel.anatomy import (AcceptedAnatomy, AcceptedPhysicalHydraulics, ClosedVascularGraph, PhysicalHydraulicEdgeFlow, VascularEdge, VascularNode)
from golem.kernel.mechanics import CellIndex
from golem.kernel.sheaf import (BalanceInterfaceId, BalanceSourceId, CellAdjacency, CellId, CellularComplex, CoolantAdvectionLink, DittusBoelterSmoothCircularTube, HeatTransferCorrelation, HeatTransferCorrelationId, LaminarCircularTubeConservativeLowerBound, LaminarThermalBoundaryCondition, SolidConductionLink, ThermalSource, TurbulentThermalDirection, WallHeatExchange, CircularTubeGeometry)


def _nearest_wall_binding(
    edge: VascularEdge,
    node_by_id: dict[str, VascularNode],
    wall_cells: tuple[CellIndex, ...],
    wall_centers: np.ndarray,
) -> _WallBinding:
    start = np.asarray(node_by_id[edge.source_node_id].position, dtype=np.float64)
    end = np.asarray(node_by_id[edge.target_node_id].position, dtype=np.float64)
    midpoint = 0.5 * (start + end)
    nearest_index = int(np.argmin(np.sum(np.square(wall_centers - midpoint), axis=1)))
    return _WallBinding(edge.edge_id, wall_cells[nearest_index], edge.target_node_id)


def _wall_bindings(
    graph: ClosedVascularGraph,
    domain: _ResolvedMaterialDomain,
) -> tuple[_WallBinding, ...]:
    resolved_wall_cells = tuple(
        cell
        for cell in domain.mechanics_domain.solid_cells
        if cell in domain.wall_cells
    )
    candidate_cells = resolved_wall_cells or domain.mechanics_domain.solid_cells
    wall_center_by_cell = dict(
        zip(domain.mechanics_domain.solid_cells, domain.cell_centers_world, strict=True)
    )
    candidate_centers = np.asarray(
        tuple(wall_center_by_cell[cell] for cell in candidate_cells),
        dtype=np.float64,
    )
    return tuple(
        _nearest_wall_binding(edge, graph.node_by_id, candidate_cells, candidate_centers)
        for edge in graph.edges
    )


def _thermal_correlation(
    edge: VascularEdge,
    graph: ClosedVascularGraph,
    physical_flow: PhysicalHydraulicEdgeFlow,
    scale: float,
    prandtl: float,
) -> HeatTransferCorrelation:
    graph_length = _edge_length_world(graph, edge)
    geometry = CircularTubeGeometry(
        hydraulic_diameter_metres=2.0 * edge.radius * scale,
        upstream_development_length_metres=float(graph_length * scale),
    )
    reynolds = physical_flow.reynolds_validity.reynolds_number
    return (
        LaminarCircularTubeConservativeLowerBound(
            HeatTransferCorrelationId(f"wall:{edge.edge_id}"),
            geometry,
            reynolds,
            prandtl,
            LaminarThermalBoundaryCondition.CONSTANT_WALL_TEMPERATURE,
        )
        if reynolds < 2300.0
        else DittusBoelterSmoothCircularTube(
            HeatTransferCorrelationId(f"wall:{edge.edge_id}"),
            geometry,
            reynolds,
            prandtl,
            TurbulentThermalDirection.FLUID_HEATING,
        )
    )


def _thermal_source_cell_ids(
    element: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    address: ElementAddress | RegionAddress,
) -> tuple[int, ...] | UnsupportedThermalAddressObstruction | EmptyThermalRegionObstruction:
    if address.element_id != element.element_id:
        return UnsupportedThermalAddressObstruction(address)
    if isinstance(address, ElementAddress):
        return tuple(range(len(domain.mechanics_domain.solid_cells)))
    if not isinstance(element.anatomy, AcceptedAnatomy):
        return UnsupportedThermalAddressObstruction(address)
    region = next(
        (
            candidate
            for candidate in element.anatomy.overall.regions
            if candidate.region_id == address.region_id
        ),
        None,
    )
    if region is None:
        return UnsupportedThermalAddressObstruction(address)
    selected = _domain_cell_indices_owned_by_provenance(
        element,
        domain,
        f"skeleton/{region.host_bone_id}/",
    )
    return selected if selected else EmptyThermalRegionObstruction(address)


def _resolve_thermal_sources(
    element: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    service: ServiceIntent,
    service_case: ServiceCase,
    graph: ClosedVascularGraph | None,
    hydraulics: AcceptedPhysicalHydraulics | None,
) -> tuple[_ResolvedThermalSource, ...] | RejectedAssembly | None:
    element_heat_sources = tuple(
        source
        for source in service_case.heat_sources
        if _addresses(source.address, element.element_id)
    )
    if not element_heat_sources and service.coolant is None:
        return None
    local_results = tuple(
        (
            RejectedAssembly((cell_ids,))
            if isinstance(
                cell_ids,
                (UnsupportedThermalAddressObstruction, EmptyThermalRegionObstruction),
            )
            else (source, cell_ids)
        )
        for source in element_heat_sources
        for cell_ids in (_thermal_source_cell_ids(element, domain, source.address),)
    )
    resolved = collect_results(local_results)
    if isinstance(resolved, RejectedAssembly):
        return resolved
    if graph is None or hydraulics is None or service.coolant is None:
        return RejectedAssembly(
            (MissingPhysicalVasculatureObstruction(element.element_id),)
        )
    return resolved


def _assemble_thermal_problem(
    element: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    service: ServiceIntent,
    service_case: ServiceCase,
    graph: ClosedVascularGraph,
    hydraulics: AcceptedPhysicalHydraulics,
    source_cell_results: tuple[_ResolvedThermalSource, ...],
) -> _AssembledThermalProblem:
    fluid = service.coolant.material
    solid_cells = domain.mechanics_domain.solid_cells
    solid_count = len(solid_cells)
    cell_id_by_index = domain.cell_id_by_index
    node_index = {node.node_id: index for index, node in enumerate(graph.nodes)}
    coolant_cell_id = lambda node_id: CellId(solid_count + node_index[node_id])
    wall_bindings = _wall_bindings(graph, domain)
    binding_by_edge_id = {binding.edge_id: binding for binding in wall_bindings}
    flow_by_edge_id = {flow.edge_id: flow for flow in hydraulics.edge_flows}
    scale = service.scale.metres_per_world_unit
    prandtl = (
        fluid.specific_heat_joules_per_kilogram_kelvin
        * fluid.dynamic_viscosity_pascal_second
        / fluid.thermal_conductivity_watts_per_metre_kelvin
    )
    pump_mass_flow = (
        fluid.density_kilograms_per_cubic_metre
        * hydraulics.receipt.pump_volumetric_flow_cubic_metres_per_second
    )
    thermal_domain = CellularComplex(
        complex_id=f"thermal:{element.element_id}:{service_case.case_id}",
        cell_centers=(
            *tuple(
                _cell_center_metres(domain.mechanics_domain, cell)
                for cell in solid_cells
            ),
            *tuple(
                tuple(coordinate * scale for coordinate in node.position)
                for node in graph.nodes
            ),
        ),
        adjacencies=(
            *tuple(
                CellAdjacency(
                    CellId(cell_id_by_index[adjacency.left_cell]),
                    CellId(cell_id_by_index[adjacency.right_cell]),
                    adjacency.exchange_area_square_metres,
                )
                for adjacency in domain.adjacencies
            ),
            *tuple(
                CellAdjacency(
                    coolant_cell_id(edge.source_node_id),
                    coolant_cell_id(edge.target_node_id),
                    1.0,
                )
                for edge in graph.edges
            ),
            CellAdjacency(
                coolant_cell_id(graph.pump_inlet_node_id),
                coolant_cell_id(graph.pump_outlet_node_id),
                1.0,
            ),
            *tuple(
                CellAdjacency(
                    CellId(cell_id_by_index[binding.solid_cell]),
                    coolant_cell_id(binding.coolant_node_id),
                    1.0,
                )
                for binding in wall_bindings
            ),
        ),
    )
    conduction_links = tuple(
        SolidConductionLink(
            BalanceInterfaceId(
                f"solid:{adjacency.left_cell}:{adjacency.right_cell}"
            ),
            CellId(cell_id_by_index[adjacency.left_cell]),
            CellId(cell_id_by_index[adjacency.right_cell]),
            _harmonic_conductivity(
                _solid_material_for_cell(
                    service,
                    element.element_id,
                    adjacency.left_cell,
                    domain.wall_cells,
                ).thermal_conductivity_watts_per_metre_kelvin,
                _solid_material_for_cell(
                    service,
                    element.element_id,
                    adjacency.right_cell,
                    domain.wall_cells,
                ).thermal_conductivity_watts_per_metre_kelvin,
            ),
            adjacency.exchange_area_square_metres,
            adjacency.path_length_metres,
        )
        for adjacency in domain.adjacencies
    )
    coolant_links = (
        *tuple(
            CoolantAdvectionLink(
                BalanceInterfaceId(f"coolant:{edge.edge_id}"),
                coolant_cell_id(edge.source_node_id),
                coolant_cell_id(edge.target_node_id),
                fluid.density_kilograms_per_cubic_metre
                * flow_by_edge_id[edge.edge_id].volumetric_flow_cubic_metres_per_second,
                fluid.specific_heat_joules_per_kilogram_kelvin,
            )
            for edge in graph.edges
        ),
        CoolantAdvectionLink(
            BalanceInterfaceId("coolant:pump-reset"),
            coolant_cell_id(graph.pump_inlet_node_id),
            coolant_cell_id(graph.pump_outlet_node_id),
            pump_mass_flow,
            fluid.specific_heat_joules_per_kilogram_kelvin,
        ),
    )
    wall_links = tuple(
        WallHeatExchange(
            BalanceInterfaceId(f"wall:{edge.edge_id}"),
            CellId(cell_id_by_index[binding_by_edge_id[edge.edge_id].solid_cell]),
            coolant_cell_id(binding_by_edge_id[edge.edge_id].coolant_node_id),
            2.0 * math.pi * edge.radius * scale * _edge_length_world(graph, edge) * scale,
            _thermal_correlation(
                edge,
                graph,
                flow_by_edge_id[edge.edge_id],
                scale,
                prandtl,
            ),
            fluid.thermal_conductivity_watts_per_metre_kelvin,
        )
        for edge in graph.edges
    )
    sources = tuple(
        ThermalSource(
            BalanceSourceId(f"{service_case.case_id}:{source_index}:{cell_index}"),
            CellId(cell_index),
            source.watts / len(source_cells),
        )
        for source_index, (source, source_cells) in enumerate(source_cell_results)
        for cell_index in source_cells
    )
    surface_faces = _surface_face_specs(domain)
    return _AssembledThermalProblem(
        thermal_domain=thermal_domain,
        conduction_links=conduction_links,
        coolant_links=coolant_links,
        wall_links=wall_links,
        sources=sources,
        surface_faces=surface_faces,
        surface_cell_ids=tuple(
            dict.fromkeys(
                cell_id_by_index[cell]
                for cell, _face, _axis, _area in surface_faces
            )
        ),
        cell_id_by_index=cell_id_by_index,
        coolant_cell_id=coolant_cell_id,
        reference_temperature_kelvin=_reference_temperature_kelvin(service),
        ambient=service.ambient,
        gravity_magnitude=float(
            np.linalg.norm(
                np.asarray(
                    service_case.gravity_metres_per_second_squared.values,
                    dtype=np.float64,
                )
            )
        ),
        solid_count=solid_count,
        solid_cells=solid_cells,
        fluid=fluid,
    )


def _harmonic_conductivity(left: float, right: float) -> float:
    return 2.0 * left * right / (left + right)


def _edge_length_world(graph: ClosedVascularGraph, edge: VascularEdge) -> float:
    return float(
        np.linalg.norm(
            np.asarray(graph.node_by_id[edge.target_node_id].position)
            - np.asarray(graph.node_by_id[edge.source_node_id].position)
        )
    )
