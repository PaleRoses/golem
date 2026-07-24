"""Assembly thermal orchestration and stable import surface."""

from __future__ import annotations

from golem.assembly.address import (_domain_cell_indices_owned_by_provenance, _owner_provenance_at_points, _semantic_address_for_cell)
from golem.assembly.carriers import RejectedAssembly, _CompiledElement
from golem.assembly.obstructions import MissingPhysicalVasculatureObstruction
from golem.assembly.service import ServiceCase, ServiceIntent
from golem.assembly.thermal_algebra import (_derive_thermal_receipt, _shift_thermal_solution_reference)
from golem.assembly.thermal_ambient import _solve_ambient_coupled
from golem.assembly.thermal_compile import (_assemble_thermal_problem, _edge_length_world, _harmonic_conductivity, _nearest_wall_binding, _resolve_thermal_sources, _thermal_correlation, _thermal_source_cell_ids, _wall_bindings)
from golem.assembly.thermal_types import (_AssembledThermalProblem, _SolvedElementThermal)
from golem.assembly.voxel import _ResolvedMaterialDomain
from golem.kernel.anatomy import AcceptedPhysicalHydraulics, ClosedVascularGraph


def _solve_element_thermal(
    element: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    service: ServiceIntent,
    service_case: ServiceCase,
    graph: ClosedVascularGraph | None,
    hydraulics: AcceptedPhysicalHydraulics | None,
) -> _SolvedElementThermal | RejectedAssembly | None:
    resolved = _resolve_thermal_sources(
        element,
        domain,
        service,
        service_case,
        graph,
        hydraulics,
    )
    if resolved is None or isinstance(resolved, RejectedAssembly):
        return resolved
    if graph is None or hydraulics is None:
        return RejectedAssembly(
            (MissingPhysicalVasculatureObstruction(element.element_id),)
        )
    problem = _assemble_thermal_problem(
        element,
        domain,
        service,
        service_case,
        graph,
        hydraulics,
        resolved,
    )
    solved = _solve_ambient_coupled(problem, element, service_case, graph)
    if isinstance(solved, RejectedAssembly):
        return solved
    solution, natural_convection, coupling_iterations, maximum_state_delta = solved
    return _derive_thermal_receipt(
        problem,
        solution,
        natural_convection,
        coupling_iterations,
        maximum_state_delta,
        element,
        service,
        service_case,
        domain,
        graph,
    )
