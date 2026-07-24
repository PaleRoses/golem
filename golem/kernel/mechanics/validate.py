"""Guard the mechanics problem with typed input obstructions."""

from __future__ import annotations

from collections import Counter
from math import isfinite

from .grid import _active_nodes, _cell_is_in_bounds, _neighbor_cell
from .model import (
    BoundaryFace,
    CellConstitutiveAssignment,
    CellIndex,
    InternalPressureFace,
    MaterialFractionResolution,
    MechanicsCriteria,
    MechanicsProblem,
    NodalForce,
    NodalSupport,
    NodalTemperatureChange,
    NodeIndex,
    StructuredHexDomain,
    Vector3,
    _invalid_isotropic_constitutive_fields,
)
from .obstructions import (
    ConflictingSupportObstruction,
    DuplicateCellObstruction,
    DuplicateCellPropertiesObstruction,
    InvalidAcceptanceCriteriaObstruction,
    InvalidBodyForceObstruction,
    InvalidConstitutivePropertiesObstruction,
    InvalidMaterialFractionObstruction,
    InvalidNodalLoadObstruction,
    InvalidPressureBoundaryObstruction,
    InvalidSupportObstruction,
    InvalidTemperatureFieldObstruction,
    MalformedDomainObstruction,
    MechanicsObstruction,
    MissingCellPropertiesObstruction,
    MaterialFractionRule,
    UnknownCellPropertiesObstruction,
    UnresolvedMaterialFractionObstruction,
)


def _validate_problem(problem: MechanicsProblem) -> tuple[MechanicsObstruction, ...]:
    domain = problem.domain
    cells = domain.solid_cells
    cell_counts = Counter(cells)
    cell_set = frozenset(cells)
    assignments = problem.cell_properties
    assignment_counts = Counter(assignment.cell for assignment in assignments)
    assigned_cells = frozenset(assignment_counts)
    active_nodes = _active_nodes(cells)
    constraint_keys = tuple(
        (support.node, component)
        for support in problem.supports
        for component, value in enumerate(support.prescribed_displacement)
        if value is not None
    )
    temperature_counts = Counter(
        temperature.node for temperature in problem.nodal_temperature_changes
    )
    pressure_counts = Counter(
        (pressure.cell, pressure.face) for pressure in problem.internal_pressure_faces
    )
    return (
        _domain_obstructions(domain, cells, cell_counts)
        + _cell_property_obstructions(
            assignments,
            cell_set,
            assigned_cells,
            assignment_counts,
        )
        + _support_obstructions(problem.supports, active_nodes, constraint_keys)
        + _nodal_load_obstructions(problem.nodal_forces, active_nodes)
        + _body_force_obstructions(problem.body_acceleration)
        + _temperature_field_obstructions(
            problem.nodal_temperature_changes,
            active_nodes,
            temperature_counts,
        )
        + _pressure_obstructions(
            problem.internal_pressure_faces,
            cell_set,
            pressure_counts,
        )
        + _criteria_obstructions(problem.criteria)
    )


def _domain_obstructions(
    domain: StructuredHexDomain,
    cells: tuple[CellIndex, ...],
    cell_counts: Counter[CellIndex],
) -> tuple[MechanicsObstruction, ...]:
    return (
        _axis_obstructions("x_coordinates", domain.x_coordinates)
        + _axis_obstructions("y_coordinates", domain.y_coordinates)
        + _axis_obstructions("z_coordinates", domain.z_coordinates)
        + (
            ()
            if cells
            else (MalformedDomainObstruction("domain/solid_cells", "empty domain"),)
        )
        + tuple(
            DuplicateCellObstruction(cell)
            for cell, count in cell_counts.items()
            if count > 1
        )
        + tuple(
            MalformedDomainObstruction(
                f"domain/solid_cells/{cell}",
                "cell index lies outside the structured axis intervals",
            )
            for cell in cells
            if not _cell_is_in_bounds(domain, cell)
        )
    )


def _cell_property_obstructions(
    assignments: tuple[CellConstitutiveAssignment, ...],
    cell_set: frozenset[CellIndex],
    assigned_cells: frozenset[CellIndex],
    assignment_counts: Counter[CellIndex],
) -> tuple[MechanicsObstruction, ...]:
    return (
        tuple(
            MissingCellPropertiesObstruction(cell)
            for cell in sorted(cell_set - assigned_cells)
        )
        + tuple(
            UnknownCellPropertiesObstruction(cell)
            for cell in sorted(assigned_cells - cell_set)
        )
        + tuple(
            DuplicateCellPropertiesObstruction(cell)
            for cell, count in assignment_counts.items()
            if count > 1
        )
        + tuple(
            obstruction
            for assignment in assignments
            for obstruction in (
                _constitutive_obstructions(assignment)
                + _material_fraction_obstructions(assignment)
            )
        )
    )


def _support_obstructions(
    supports: tuple[NodalSupport, ...],
    active_nodes: frozenset[NodeIndex],
    constraint_keys: tuple[tuple[NodeIndex, int], ...],
) -> tuple[MechanicsObstruction, ...]:
    return (
        tuple(
            InvalidSupportObstruction(support.node, "node is outside the solid domain")
            for support in supports
            if support.node not in active_nodes
        )
        + tuple(
            InvalidSupportObstruction(
                support.node, "support constrains no displacement component"
            )
            for support in supports
            if all(value is None for value in support.prescribed_displacement)
        )
        + tuple(
            InvalidSupportObstruction(
                support.node, "prescribed displacement must be finite"
            )
            for support in supports
            if any(
                value is not None and not isfinite(value)
                for value in support.prescribed_displacement
            )
        )
        + tuple(
            ConflictingSupportObstruction(node, component)
            for (node, component), count in Counter(constraint_keys).items()
            if count > 1
        )
    )


def _nodal_load_obstructions(
    forces: tuple[NodalForce, ...], active_nodes: frozenset[NodeIndex]
) -> tuple[MechanicsObstruction, ...]:
    return tuple(
        InvalidNodalLoadObstruction(force.node, "node is outside the solid domain")
        for force in forces
        if force.node not in active_nodes
    ) + tuple(
        InvalidNodalLoadObstruction(force.node, "force components must be finite")
        for force in forces
        if not all(map(isfinite, force.force))
    )


def _body_force_obstructions(
    body_acceleration: Vector3,
) -> tuple[MechanicsObstruction, ...]:
    return (
        ()
        if all(map(isfinite, body_acceleration))
        else (
            InvalidBodyForceObstruction(
                "body_acceleration",
                body_acceleration,
            ),
        )
    )


def _temperature_field_obstructions(
    temperature_changes: tuple[NodalTemperatureChange, ...],
    active_nodes: frozenset[NodeIndex],
    temperature_counts: Counter[NodeIndex],
) -> tuple[MechanicsObstruction, ...]:
    return (
        tuple(
            InvalidTemperatureFieldObstruction(
                temperature.node, "node is outside the solid domain"
            )
            for temperature in temperature_changes
            if temperature.node not in active_nodes
        )
        + tuple(
            InvalidTemperatureFieldObstruction(
                temperature.node, "temperature change must be finite"
            )
            for temperature in temperature_changes
            if not isfinite(temperature.delta_temperature)
        )
        + tuple(
            InvalidTemperatureFieldObstruction(
                node, "temperature node appears more than once"
            )
            for node, count in temperature_counts.items()
            if count > 1
        )
    )


def _pressure_obstructions(
    pressures: tuple[InternalPressureFace, ...],
    solid_cells: frozenset[CellIndex],
    counts: Counter[tuple[CellIndex, BoundaryFace]],
) -> tuple[MechanicsObstruction, ...]:
    return tuple(
        obstruction
        for pressure in pressures
        for obstruction in (
            _pressure_obstruction(pressure, solid_cells, counts),
        )
        if obstruction is not None
    )


def _axis_obstructions(
    address: str, coordinates: tuple[float, ...]
) -> tuple[MechanicsObstruction, ...]:
    return (
        (
            MalformedDomainObstruction(
                f"domain/{address}", "at least two coordinates are required"
            ),
        )
        if len(coordinates) < 2
        else ()
    ) + (
        (
            MalformedDomainObstruction(
                f"domain/{address}",
                "coordinates must be finite and strictly increasing",
            ),
        )
        if any(not isfinite(value) for value in coordinates)
        or any(right <= left for left, right in zip(coordinates, coordinates[1:]))
        else ()
    )


def _constitutive_obstructions(
    assignment: CellConstitutiveAssignment,
) -> tuple[MechanicsObstruction, ...]:
    return tuple(
        InvalidConstitutivePropertiesObstruction(
            _cell_address(assignment.cell),
            assignment.cell,
            name,
            value,
        )
        for name, value in _invalid_isotropic_constitutive_fields(
            assignment.properties
        )
    )


def _material_fraction_obstructions(
    assignment: CellConstitutiveAssignment,
) -> tuple[MechanicsObstruction, ...]:
    fraction = assignment.solid_fraction
    resolution = assignment.fraction_resolution
    address = _cell_address(assignment.cell)
    resolution_name = (
        resolution.value
        if isinstance(resolution, MaterialFractionResolution)
        else type(resolution).__name__
    )
    return (
        (
            InvalidMaterialFractionObstruction(
                address,
                assignment.cell,
                MaterialFractionRule.ACTIVE_CELL_RANGE,
                fraction,
                resolution_name,
            ),
        )
        if not isfinite(fraction) or not 0.0 < fraction <= 1.0
        else ()
    ) + (
        (
            InvalidMaterialFractionObstruction(
                address,
                assignment.cell,
                MaterialFractionRule.FULL_SOLID_VALUE,
                fraction,
                resolution_name,
            ),
        )
        if resolution is MaterialFractionResolution.FULL_SOLID and fraction != 1.0
        else ()
    ) + (
        (
            UnresolvedMaterialFractionObstruction(
                address,
                assignment.cell,
                fraction,
            ),
        )
        if resolution is MaterialFractionResolution.UNRESOLVED
        else ()
    ) + (
        (
            InvalidMaterialFractionObstruction(
                address,
                assignment.cell,
                MaterialFractionRule.RESOLUTION_TYPE,
                fraction,
                resolution_name,
            ),
        )
        if not isinstance(resolution, MaterialFractionResolution)
        else ()
    )


def _cell_address(cell: CellIndex) -> str:
    return f"domain/cell:{cell.x_index}:{cell.y_index}:{cell.z_index}"


def _criteria_obstructions(
    criteria: MechanicsCriteria,
) -> tuple[MechanicsObstruction, ...]:
    finite_positive = (
        ("linear_solver_relative_tolerance", criteria.linear_solver_relative_tolerance),
        ("equilibrium_relative_tolerance", criteria.equilibrium_relative_tolerance),
        ("maximum_principal_strain", criteria.maximum_principal_strain),
        ("normalization_floor", criteria.normalization_floor),
    )
    optional_positive = (
        ("maximum_displacement", criteria.maximum_displacement),
        ("minimum_yield_safety_factor", criteria.minimum_yield_safety_factor),
        (
            "minimum_linearized_buckling_load_factor",
            criteria.minimum_linearized_buckling_load_factor,
        ),
    )
    return tuple(
        InvalidAcceptanceCriteriaObstruction(
            name, value, "criterion must be finite and positive"
        )
        for name, value in finite_positive
        if not isfinite(value) or value <= 0.0
    ) + tuple(
        InvalidAcceptanceCriteriaObstruction(
            name, value, "optional criterion must be finite and positive"
        )
        for name, value in optional_positive
        if value is not None and (not isfinite(value) or value <= 0.0)
    )


def _pressure_obstruction(
    pressure: InternalPressureFace,
    solid_cells: frozenset[CellIndex],
    counts: Counter[tuple[CellIndex, BoundaryFace]],
) -> InvalidPressureBoundaryObstruction | None:
    neighbor = _neighbor_cell(pressure.cell, pressure.face)
    reason = (
        "pressure cell is not solid"
        if pressure.cell not in solid_cells
        else "pressure face is internal to two solid cells"
        if neighbor in solid_cells
        else "pressure must be finite and non-negative"
        if not isfinite(pressure.pressure) or pressure.pressure < 0.0
        else "pressure face appears more than once"
        if counts[(pressure.cell, pressure.face)] > 1
        else None
    )
    return (
        InvalidPressureBoundaryObstruction(pressure.cell, pressure.face, reason)
        if reason is not None
        else None
    )
