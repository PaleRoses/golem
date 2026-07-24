"""Mechanics assembly local geometry and payload-transfer coalgebra."""

from __future__ import annotations

import math
import numpy as np

from golem.assembly.carriers import RejectedAssembly, RigidPayloadInterfaceReceipt, _CompiledElement, _addresses
from golem.assembly.descent import traverse_results
from golem.assembly.mechanics.types import _PayloadTransferGeometry, _RigidPayloadTransfer
from golem.assembly.materials_binding import _element_material_assignment
from golem.assembly.obstructions import (MissingRigidPayloadInterfaceObstruction, MissingSolidMaterialAssignmentObstruction, RigidPayloadBalanceObstruction, RigidPayloadInterfaceGapObstruction, UnsupportedMechanicalAddressObstruction)
from golem.assembly.service import (BoneAddress, ForceLoad, JointLaw, MomentLoad, MountAddress, PortAddress, SemanticAddress, ServiceCase, ServiceIntent, SolidMaterialRole)
from golem.assembly.voxel import (_ResolvedMaterialDomain, _cell_center_metres, _cell_nodes, _evaluated_solid_centers_world, _node_position_metres)
from golem.kernel import engine as E
from golem.kernel.anatomy import ClosedVascularGraph
from golem.kernel.mechanics import NodalForce

def _address_points_world(
    element: _CompiledElement,
    address: SemanticAddress,
    graph: ClosedVascularGraph | None,
) -> tuple[tuple[float, float, float], ...] | UnsupportedMechanicalAddressObstruction:
    match address:
        case BoneAddress(element_id, bone_id) if element_id == element.element_id:
            landmarks = element.graph.get("intent", {}).get("landmarks", {})
            point = landmarks.get(f"{bone_id}/center")
            if not isinstance(point, list) or len(point) != 3:
                return UnsupportedMechanicalAddressObstruction(
                    address, "bone center landmark is absent"
                )
            reflected = (-float(point[0]), float(point[1]), float(point[2]))
            original = tuple(map(float, point))
            return (
                (original, reflected)
                if any(
                    str(part.get("id", "")).endswith("_m")
                    for part in element.graph.get("parts", ())
                )
                and original != reflected
                else (original,)
            )
        case MountAddress(element_id) if element_id == element.element_id:
            mount = element.entry.get("mount")
            point = mount.get("translate") if isinstance(mount, dict) else None
            lower, upper = E.graph_bounds(element.graph)
            return (
                (tuple(map(float, point)),)
                if isinstance(point, list) and len(point) == 3
                else (
                    (
                        float(0.5 * (lower[0] + upper[0])),
                        float(lower[1]),
                        float(0.5 * (lower[2] + upper[2])),
                    ),
                )
            )
        case PortAddress(element_id, port_id) if (
            element_id == element.element_id and graph is not None
        ):
            node_id = (
                graph.pump_outlet_node_id
                if port_id == "pump_outlet"
                else graph.pump_inlet_node_id
                if port_id == "pump_inlet"
                else None
            )
            return (
                (graph.node_by_id[node_id].position,)
                if node_id is not None
                else UnsupportedMechanicalAddressObstruction(
                    address, "unknown physical vascular port"
                )
            )
        case _:
            return UnsupportedMechanicalAddressObstruction(
                address, "address does not belong to the solved solid section"
            )


def _closest_payload_attachment(
    body: _CompiledElement,
    payload: _CompiledElement,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    from scipy.spatial import cKDTree

    body_centers = _evaluated_solid_centers_world(body.evaluated)
    payload_centers = _evaluated_solid_centers_world(payload.evaluated)
    distances, body_indices = cKDTree(body_centers).query(payload_centers, k=1)
    payload_index = int(np.argmin(distances))
    gap_metres = float(distances[payload_index] * scale)
    permitted_gap_metres = float(
        1.5
        * (body.evaluated.maximum_pitch + payload.evaluated.maximum_pitch)
        * scale
    )
    return (
        scale * body_centers[int(body_indices[payload_index])],
        scale * payload_centers[payload_index],
        gap_metres,
        permitted_gap_metres,
    )


def _wrench_distribution(
    domain: _ResolvedMaterialDomain,
    attachment_metres: np.ndarray,
    force_newtons: np.ndarray,
    moment_newton_metres: np.ndarray,
) -> tuple[tuple[NodalForce, ...], float, float]:
    solid_cells = domain.mechanics_domain.solid_cells
    centers_metres = np.asarray(
        tuple(
            _cell_center_metres(domain.mechanics_domain, cell)
            for cell in solid_cells
        ),
        dtype=np.float64,
    )
    nearest_cell = solid_cells[
        int(
            np.argmin(
                np.sum(np.square(centers_metres - attachment_metres), axis=1)
            )
        )
    ]
    nodes = _cell_nodes(nearest_cell)
    offsets = tuple(
        _node_position_metres(domain, node) - attachment_metres
        for node in nodes
    )
    identity = np.eye(3, dtype=np.float64)
    skew_blocks = tuple(
        np.asarray(
            (
                (0.0, -offset[2], offset[1]),
                (offset[2], 0.0, -offset[0]),
                (-offset[1], offset[0], 0.0),
            ),
            dtype=np.float64,
        )
        for offset in offsets
    )
    operator = np.vstack(
        (
            np.hstack(tuple(identity for _node in nodes)),
            np.hstack(skew_blocks),
        )
    )
    target = np.concatenate((force_newtons, moment_newton_metres))
    distributed = np.linalg.pinv(operator, rcond=1.0e-12) @ target
    nodal_forces = tuple(
        NodalForce(
            node,
            tuple(map(float, distributed[3 * index : 3 * index + 3])),
        )
        for index, node in enumerate(nodes)
    )
    reconstructed = operator @ distributed
    return (
        nodal_forces,
        float(np.linalg.norm(reconstructed[:3] - force_newtons)),
        float(np.linalg.norm(reconstructed[3:] - moment_newton_metres)),
    )


def _payload_transfer_geometry(
    body: _CompiledElement,
    payload: _CompiledElement,
    service: ServiceIntent,
    service_case: ServiceCase,
) -> _PayloadTransferGeometry | RejectedAssembly:
    joints = tuple(
        joint
        for joint in service_case.joints
        if _addresses(joint.address, payload.element_id)
    )
    if len(joints) != 1 or joints[0].law is not JointLaw.RIGID_PAYLOAD:
        return RejectedAssembly(
            (MissingRigidPayloadInterfaceObstruction(payload.element_id),)
        )
    assignment = _element_material_assignment(
        service,
        payload.element_id,
        (SolidMaterialRole.ARMOR, SolidMaterialRole.EQUIPMENT),
    )
    if assignment is None:
        return RejectedAssembly(
            (
                MissingSolidMaterialAssignmentObstruction(
                    payload.element_id,
                    (SolidMaterialRole.ARMOR, SolidMaterialRole.EQUIPMENT),
                ),
            )
        )
    scale = service.scale.metres_per_world_unit
    centers_world = _evaluated_solid_centers_world(payload.evaluated)
    cell_volume_cubic_metres = (
        math.prod(payload.evaluated.world_pitch) * scale**3
    )
    mass_kilograms = float(
        len(centers_world)
        * cell_volume_cubic_metres
        * assignment.material.density_kilograms_per_cubic_metre
    )
    center_of_mass_metres = scale * np.mean(centers_world, axis=0)
    attachment_metres, _payload_contact, gap, permitted_gap = (
        _closest_payload_attachment(body, payload, scale)
    )
    if gap > permitted_gap:
        return RejectedAssembly(
            (
                RigidPayloadInterfaceGapObstruction(
                    payload.element_id, gap, permitted_gap
                ),
            )
        )
    return _PayloadTransferGeometry(
        scale=scale,
        mass_kilograms=mass_kilograms,
        center_of_mass_metres=center_of_mass_metres,
        attachment_metres=attachment_metres,
        gap=gap,
    )


def _payload_transfer_wrench(
    geometry: _PayloadTransferGeometry,
    payload: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    service_case: ServiceCase,
) -> _RigidPayloadTransfer | RejectedAssembly:
    scale = geometry.scale
    mass_kilograms = geometry.mass_kilograms
    center_of_mass_metres = geometry.center_of_mass_metres
    attachment_metres = geometry.attachment_metres
    gap = geometry.gap
    acceleration = np.asarray(
        tuple(
            gravity + inertia
            for gravity, inertia in zip(
                service_case.gravity_metres_per_second_squared.values,
                service_case.inertial_acceleration_metres_per_second_squared.values,
            )
        ),
        dtype=np.float64,
    )
    inertial_force = mass_kilograms * acceleration
    explicit_force_entries = tuple(
        (load, _address_points_world(payload, load.address, None))
        for load in service_case.loads
        if isinstance(load, ForceLoad)
        and _addresses(load.address, payload.element_id)
    )
    address_obstructions = tuple(
        points
        for _load, points in explicit_force_entries
        if isinstance(points, UnsupportedMechanicalAddressObstruction)
    )
    if address_obstructions:
        return RejectedAssembly(address_obstructions)
    explicit_force = np.sum(
        np.asarray(
            tuple(load.vector_newtons.values for load, _points in explicit_force_entries),
            dtype=np.float64,
        ),
        axis=0,
    ) if explicit_force_entries else np.zeros(3, dtype=np.float64)
    force_newtons = inertial_force + explicit_force
    inertial_moment = np.cross(
        center_of_mass_metres - attachment_metres, inertial_force
    )
    force_moments = tuple(
        np.cross(
            scale * np.asarray(points[0], dtype=np.float64) - attachment_metres,
            np.asarray(load.vector_newtons.values, dtype=np.float64),
        )
        for load, points in explicit_force_entries
        if isinstance(points, tuple)
    )
    authored_moments = tuple(
        np.asarray(load.vector_newton_metres.values, dtype=np.float64)
        for load in service_case.loads
        if isinstance(load, MomentLoad)
        and _addresses(load.address, payload.element_id)
    )
    moment_newton_metres = inertial_moment + np.sum(
        np.asarray((*force_moments, *authored_moments), dtype=np.float64),
        axis=0,
    ) if force_moments or authored_moments else inertial_moment
    nodal_forces, force_residual, moment_residual = _wrench_distribution(
        domain, attachment_metres, force_newtons, moment_newton_metres
    )
    tolerance = 1.0e-8
    if force_residual > tolerance or moment_residual > tolerance:
        return RejectedAssembly(
            (
                RigidPayloadBalanceObstruction(
                    payload.element_id,
                    force_residual,
                    moment_residual,
                    tolerance,
                ),
            )
        )
    return _RigidPayloadTransfer(
        receipt=RigidPayloadInterfaceReceipt(
            element_id=payload.element_id,
            case_id=service_case.case_id,
            mass_kilograms=mass_kilograms,
            center_of_mass_metres=tuple(map(float, center_of_mass_metres)),
            attachment_point_metres=tuple(map(float, attachment_metres)),
            interface_gap_metres=gap,
            transmitted_force_newtons=tuple(map(float, force_newtons)),
            transmitted_moment_newton_metres=tuple(
                map(float, moment_newton_metres)
            ),
            force_balance_residual_newtons=force_residual,
            moment_balance_residual_newton_metres=moment_residual,
        ),
        nodal_forces=nodal_forces,
    )


def _payload_interface_transfer(
    body: _CompiledElement,
    payload: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    service: ServiceIntent,
    service_case: ServiceCase,
) -> _RigidPayloadTransfer | RejectedAssembly:
    geometry = _payload_transfer_geometry(body, payload, service, service_case)
    if isinstance(geometry, RejectedAssembly):
        return geometry
    return _payload_transfer_wrench(geometry, payload, domain, service_case)


def _payload_interface_transfers_by_case(
    body: _CompiledElement,
    payloads: tuple[_CompiledElement, ...],
    domain: _ResolvedMaterialDomain,
    service: ServiceIntent,
) -> (
    tuple[tuple[str, tuple[_RigidPayloadTransfer, ...]], ...]
    | RejectedAssembly
):
    local_results = tuple(
        (
            service_case.case_id,
            tuple(
                _payload_interface_transfer(
                    body,
                    payload,
                    domain,
                    service,
                    service_case,
                )
                for payload in payloads
            ),
        )
        for service_case in service.cases
    )
    return traverse_results(
        tuple(result for _case_id, results in local_results for result in results),
        lambda _transfers: tuple(
            (
                case_id,
                tuple(
                    result
                    for result in results
                    if isinstance(result, _RigidPayloadTransfer)
                ),
            )
            for case_id, results in local_results
        ),
    )

