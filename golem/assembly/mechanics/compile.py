"""Mechanics assembly lowering into the authoritative kernel solver."""

from __future__ import annotations

import math
import numpy as np

from dataclasses import replace
from itertools import groupby
from math import fsum

from golem.assembly.address import (
    _domain_cell_indices_owned_by_provenance,
    _semantic_address_for_cell,
)
from golem.assembly.carriers import RejectedAssembly, _CompiledElement, _addresses, _reference_temperature_kelvin
from golem.assembly.hydraulics import _EmbeddedVascularSection, _VascularPhysicalSection
from golem.assembly.materials_binding import (_element_material_assignment, _solid_material_for_cell)
from golem.assembly.mechanics.algebra import _finalize_element_mechanics
from golem.assembly.mechanics.coalgebra import _address_points_world
from golem.assembly.obstructions import (MissingMechanicalSupportObstruction, MissingPhysicalVasculatureObstruction, UnsupportedAssemblyInterfaceObstruction, UnsupportedMechanicalAddressObstruction)
from golem.assembly.service import (BoneAddress, ForceLoad, MomentLoad, ServiceCase, ServiceIntent, SolidMaterialRole, Support, SupportLaw)
from golem.assembly.thermal_types import _SolvedElementThermal
from golem.assembly.voxel import (_ResolvedMaterialDomain, _cell_center_metres, _cell_ijk, _cell_nodes, _domain_axes, _faces_where, _nearest_domain_cells, _nearest_domain_node, _neighbor_cell, _pressure_face_specs)
from golem.kernel.anatomy import (AcceptedPhysicalHydraulics, AcceptedVasculature, ClosedVascularGraph)
from golem.kernel.mechanics import (AcceptedEmbeddedChannelMechanics, AcceptedMechanics, CellIndex, EmbeddedChannelCriteria, InternalPressureFace, IsotropicConstitutiveProperties, MaterialFractionResolution, MechanicsCriteria, NodalForce, NodalSupport, NodalTemperatureChange, RejectedEmbeddedChannelMechanics, evaluate_embedded_channel_mechanics)
from golem.kernel.mechanics.obstructions import (
    InvalidBodyForceObstruction,
    InvalidConstitutivePropertiesObstruction,
    InvalidMaterialFractionObstruction,
    MechanicsObstruction,
    UnresolvedMaterialFractionObstruction,
)

def _case_supports(
    element: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    service_case: ServiceCase,
    graph: ClosedVascularGraph | None,
) -> tuple[NodalSupport, ...] | RejectedAssembly:
    unsupported_laws = tuple(
        UnsupportedMechanicalAddressObstruction(
            support.address,
            f"support law {support.law.value!r} lacks an authored direction/rotation law",
        )
        for support in service_case.supports
        if support.law is not SupportLaw.FIXED
    )
    if unsupported_laws:
        return RejectedAssembly(unsupported_laws)
    point_results = tuple(
        (
            support,
            _address_points_world(element, support.address, graph),
        )
        for support in service_case.supports
        if _addresses(support.address, element.element_id)
    )
    address_obstructions = tuple(
        result
        for _support, result in point_results
        if isinstance(result, UnsupportedMechanicalAddressObstruction)
    )
    if address_obstructions:
        return RejectedAssembly(address_obstructions)

    def support_cells(
        support: Support,
        points: tuple[tuple[float, float, float], ...],
    ) -> tuple[CellIndex, ...] | UnsupportedMechanicalAddressObstruction:
        match support.address:
            case BoneAddress(element_id, bone_id) if (
                element_id == element.element_id
            ):
                owned_indices = _domain_cell_indices_owned_by_provenance(
                    element, domain, f"skeleton/{bone_id}/"
                )
                return (
                    tuple(
                        domain.mechanics_domain.solid_cells[index]
                        for index in owned_indices
                    )
                    if owned_indices
                    else UnsupportedMechanicalAddressObstruction(
                        support.address,
                        "bone owns no resolved solid support section",
                    )
                )
            case _:
                return _nearest_domain_cells(domain, points)

    support_cell_results = tuple(
        support_cells(support, points)
        for support, points in point_results
        if isinstance(points, tuple)
    )
    support_cell_obstructions = tuple(
        result
        for result in support_cell_results
        if isinstance(result, UnsupportedMechanicalAddressObstruction)
    )
    if support_cell_obstructions:
        return RejectedAssembly(support_cell_obstructions)

    supports = tuple(
        dict.fromkeys(
            NodalSupport(node, (0.0, 0.0, 0.0))
            for cells in support_cell_results
            if isinstance(cells, tuple)
            for cell in cells
            for node in _cell_nodes(cell)
        )
    )
    return (
        supports
        if supports
        else RejectedAssembly((MissingMechanicalSupportObstruction(service_case.case_id),))
    )


def _case_nodal_forces(
    element: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    service: ServiceIntent,
    service_case: ServiceCase,
    graph: ClosedVascularGraph | None,
    interface_forces: tuple[NodalForce, ...] = (),
) -> tuple[NodalForce, ...] | RejectedAssembly:
    moment_obstructions = tuple(
        UnsupportedMechanicalAddressObstruction(
            load.address,
            "moment loads require a resolved interface couple",
        )
        for load in service_case.loads
        if isinstance(load, MomentLoad)
        and _addresses(load.address, element.element_id)
    )
    force_points = tuple(
        (
            load,
            _address_points_world(element, load.address, graph),
        )
        for load in service_case.loads
        if isinstance(load, ForceLoad)
        and _addresses(load.address, element.element_id)
    )
    address_obstructions = tuple(
        points
        for _load, points in force_points
        if isinstance(points, UnsupportedMechanicalAddressObstruction)
    )
    if moment_obstructions or address_obstructions:
        return RejectedAssembly((*moment_obstructions, *address_obstructions))
    scale = service.scale.metres_per_world_unit
    return (
        *tuple(
            NodalForce(
                _nearest_domain_node(domain, points[0], scale),
                load.vector_newtons.values,
            )
            for load, points in force_points
            if isinstance(points, tuple)
        ),
        *interface_forces,
    )


def _nodal_temperature_changes(
    domain: _ResolvedMaterialDomain,
    thermal: _SolvedElementThermal | None,
    reference_temperature_kelvin: float,
) -> tuple[NodalTemperatureChange, ...]:
    if thermal is None:
        return ()
    contributions = tuple(
        sorted(
            (
                (node, temperature)
                for cell, temperature in thermal.cell_temperatures
                for node in _cell_nodes(cell)
            ),
            key=lambda item: item[0],
        )
    )
    return tuple(
        NodalTemperatureChange(
            node,
            mean_temperature - reference_temperature_kelvin,
        )
        for node, group in groupby(contributions, key=lambda item: item[0])
        for grouped in (tuple(group),)
        for count in (len(grouped),)
        for mean_temperature in (
            fsum(value for _node, value in grouped) / count,
        )
    )


def _pressure_at_face(
    domain: _ResolvedMaterialDomain,
    cell: CellIndex,
    axis: int,
    direction: int,
    graph: ClosedVascularGraph,
    hydraulics: AcceptedPhysicalHydraulics,
    scale: float,
) -> float:
    center = _cell_center_metres(domain.mechanics_domain, cell)
    axes = _domain_axes(domain.mechanics_domain)
    index = _cell_ijk(cell)[axis]
    half_length = 0.5 * (axes[axis][index + 1] - axes[axis][index])
    face_center = np.asarray(center, dtype=np.float64)
    face_center[axis] += direction * half_length
    node_positions = np.asarray(
        tuple(
            tuple(coordinate * scale for coordinate in node.position)
            for node in graph.nodes
        ),
        dtype=np.float64,
    )
    nearest_index = int(
        np.argmin(np.sum(np.square(node_positions - face_center), axis=1))
    )
    pressure_by_node = {
        pressure.node_id: pressure.pressure_pascal
        for pressure in hydraulics.node_pressures
    }
    return pressure_by_node[graph.nodes[nearest_index].node_id]


def _internal_pressure_faces(
    domain: _ResolvedMaterialDomain,
    graph: ClosedVascularGraph | None,
    hydraulics: AcceptedPhysicalHydraulics | None,
    scale: float,
) -> tuple[InternalPressureFace, ...]:
    if graph is None or hydraulics is None:
        return ()
    return tuple(
        InternalPressureFace(
            cell,
            boundary_face,
            _pressure_at_face(
                domain,
                cell,
                axis,
                direction,
                graph,
                hydraulics,
                scale,
            ),
        )
        for cell, boundary_face, axis, direction in _faces_where(
            domain,
            lambda cell, axis, direction: (
                _neighbor_cell(cell, axis, direction) in domain.lumen_cells
            ),
        )
    )


def _embedded_channel_mechanics(
    section: _VascularPhysicalSection | None,
    domain: _ResolvedMaterialDomain,
    service: ServiceIntent,
    hydraulics: AcceptedPhysicalHydraulics | None,
) -> AcceptedEmbeddedChannelMechanics | RejectedAssembly | None:
    if not isinstance(section, _EmbeddedVascularSection):
        return None
    element = section.element
    vasculature = element.vasculature
    graph = element.mounted_vasculature
    wall_assignment = _element_material_assignment(
        service, element.element_id, (SolidMaterialRole.LUMEN_WALL,)
    )
    bulk_assignment = _element_material_assignment(
        service, element.element_id, (SolidMaterialRole.STRUCTURE,)
    )
    if (
        not isinstance(vasculature, AcceptedVasculature)
        or graph is None
        or hydraulics is None
        or wall_assignment is None
        or bulk_assignment is None
    ):
        return RejectedAssembly(
            (MissingPhysicalVasculatureObstruction(element.element_id),)
        )
    axes = _domain_axes(domain.mechanics_domain)
    grid_pitch_metres = max(
        float(np.max(np.diff(np.asarray(axis, dtype=np.float64))))
        for axis in axes
    )
    host_volume_cubic_metres = math.prod(
        (
            axes[0][-1] - axes[0][0],
            axes[1][-1] - axes[1][0],
            axes[2][-1] - axes[2][0],
        )
    ) * (
        len(domain.mechanics_domain.solid_cells)
        / max(
            (len(axes[0]) - 1)
            * (len(axes[1]) - 1)
            * (len(axes[2]) - 1),
            1,
        )
    )
    result = evaluate_embedded_channel_mechanics(
        replace(vasculature, graph=graph),
        hydraulics.node_pressures,
        metres_per_world_unit=service.scale.metres_per_world_unit,
        grid_pitch_metres=grid_pitch_metres,
        host_volume_cubic_metres=host_volume_cubic_metres,
        wall_thickness_metres=(
            service.manufacturing.minimum_wall_thickness_metres
        ),
        wall_allowable_stress_pascal=(
            wall_assignment.material.allowable_stress_pascal
        ),
        external_pressure_pascal=0.0,
        background_properties=(
            IsotropicConstitutiveProperties.from_isotropic_solid(
                bulk_assignment.material
            )
        ),
        material_fraction_resolution=MaterialFractionResolution.UNRESOLVED,
        criteria=EmbeddedChannelCriteria(
            minimum_burst_safety_factor=(
                service.limits.minimum_burst_safety_factor
            )
        ),
    )
    return (
        result
        if isinstance(result, AcceptedEmbeddedChannelMechanics)
        else RejectedAssembly(result.obstructions)
        if isinstance(result, RejectedEmbeddedChannelMechanics)
        else RejectedAssembly(
            (
                UnsupportedAssemblyInterfaceObstruction(
                    (element.element_id,),
                    "embedded channel mechanics returned an unknown verdict",
                ),
            )
        )
    )


def _solve_element_mechanics(
    element: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    service: ServiceIntent,
    service_case: ServiceCase,
    case_index: int,
    thermal: _SolvedElementThermal | None,
    graph: ClosedVascularGraph | None,
    hydraulics: AcceptedPhysicalHydraulics | None,
    criteria: MechanicsCriteria,
    interface_forces: tuple[NodalForce, ...] = (),
    embedded_channel_mechanics: AcceptedEmbeddedChannelMechanics | None = None,
) -> AcceptedMechanics | RejectedAssembly:
    supports = _case_supports(
        element, domain, service_case, graph
    )
    if isinstance(supports, RejectedAssembly):
        return supports
    nodal_forces = _case_nodal_forces(
        element,
        domain,
        service,
        service_case,
        graph,
        interface_forces,
    )
    if isinstance(nodal_forces, RejectedAssembly):
        return nodal_forces
    mechanics_result = _solve_mechanics_problem(
        element,
        domain,
        service,
        service_case,
        case_index,
        thermal,
        graph,
        hydraulics,
        criteria,
        supports,
        nodal_forces,
    )
    if isinstance(mechanics_result, RejectedAssembly):
        return mechanics_result
    return _finalize_element_mechanics(
        element,
        domain,
        service,
        mechanics_result,
        embedded_channel_mechanics,
    )


def _address_mechanics_obstruction(
    element: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    case_index: int,
    obstruction: MechanicsObstruction,
) -> MechanicsObstruction:
    match obstruction:
        case (
            InvalidConstitutivePropertiesObstruction(cell=cell)
            | InvalidMaterialFractionObstruction(cell=cell)
            | UnresolvedMaterialFractionObstruction(cell=cell)
        ):
            return replace(
                obstruction,
                address=_semantic_address_for_cell(element, domain, cell),
            )
        case InvalidBodyForceObstruction():
            return replace(
                obstruction,
                address=f"meta@service.cases[{case_index}]",
            )
        case _:
            return obstruction


def _solve_mechanics_problem(
    element: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    service: ServiceIntent,
    service_case: ServiceCase,
    case_index: int,
    thermal: _SolvedElementThermal | None,
    graph: ClosedVascularGraph | None,
    hydraulics: AcceptedPhysicalHydraulics | None,
    criteria: MechanicsCriteria,
    supports: tuple[NodalSupport, ...],
    nodal_forces: tuple[NodalForce, ...],
) -> AcceptedMechanics | RejectedAssembly:
    from golem.kernel.mechanics import (CellConstitutiveAssignment, MechanicsProblem, RejectedMechanics, solve_mechanics)
    reference_temperature = _reference_temperature_kelvin(service)
    cell_properties = tuple(
        CellConstitutiveAssignment(
            cell,
            IsotropicConstitutiveProperties.from_isotropic_solid(
                _solid_material_for_cell(
                    service,
                    element.element_id,
                    cell,
                    domain.wall_cells,
                )
            ),
        )
        for cell in domain.mechanics_domain.solid_cells
    )
    acceleration = tuple(
        gravity + inertia
        for gravity, inertia in zip(
            service_case.gravity_metres_per_second_squared.values,
            service_case.inertial_acceleration_metres_per_second_squared.values,
        )
    )
    mechanics_result = solve_mechanics(
        MechanicsProblem(
            domain=domain.mechanics_domain,
            cell_properties=cell_properties,
            supports=supports,
            body_acceleration=acceleration,
            nodal_forces=nodal_forces,
            nodal_temperature_changes=_nodal_temperature_changes(
                domain, thermal, reference_temperature
            ),
            internal_pressure_faces=_internal_pressure_faces(
                domain,
                graph,
                hydraulics,
                service.scale.metres_per_world_unit,
            ),
            criteria=criteria,
        )
    )
    if isinstance(mechanics_result, RejectedMechanics):
        return RejectedAssembly(
            tuple(
                _address_mechanics_obstruction(
                    element,
                    domain,
                    case_index,
                    obstruction,
                )
                for obstruction in mechanics_result.obstructions
            )
        )
    return mechanics_result
