"""Vascular assembly coalgebra for derived records and thermal sizing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from golem.assembly.carriers import (AssemblyStratum, RejectedAssembly, VascularThermalSizingReceipt, _CompiledElement, _SEALED_VASCULAR_SIZING_REFINEMENTS, _solid_record)
from golem.assembly.compile import apply_appearance_palette, derive_vascular_records
from golem.assembly.descent import traverse_results
from golem.assembly.freeze import _frozen_report
from golem.assembly.hydraulics import (_assembly_hydraulics_result, _solve_element_physical_hydraulics)
from golem.assembly.mounts import mount_vascular_graph
from golem.assembly.service import ServiceIntent
from golem.kernel.anatomy import (AcceptedAnatomy, AcceptedPhysicalHydraulics, AcceptedVasculature, SealedVascularConfig, vasculature_to_dict)
from golem.kernel.mechanics import CellMechanics
from golem.materials import IncompressibleFluid

if TYPE_CHECKING:
    from golem.assembly.core import CoupledStages

def _replace_element_vasculature(
    element: _CompiledElement,
    vasculature: AcceptedVasculature,
) -> _CompiledElement:
    mounted = mount_vascular_graph(
        vasculature.graph, element.entry.get("mount")
    )
    solid_record = _solid_record(element)
    solid_report = _frozen_report(
        {
            **solid_record.report,
            "vasculature": vasculature_to_dict(vasculature),
        }
    )
    nonvascular_records = tuple(
        replace(record, report=solid_report)
        if record.stratum is AssemblyStratum.SOLID
        else record
        for record in element.records
        if not record.stratum.value.startswith("vascular_")
    )
    visual_floor = next(
        (
            float(record.report["visual_radius_floor"])
            for record in element.records
            if record.stratum.value.startswith("vascular_")
        ),
        0.075 * element.evaluated.maximum_pitch,
    )
    vascular_records = apply_appearance_palette(
        derive_vascular_records(
            element.element_id,
            element.role,
            solid_record.resolution,
            mounted,
            visual_floor / 0.075,
        ),
        element.appearance_palette,
    )
    return replace(
        element,
        records=(*nonvascular_records, *vascular_records),
        vasculature=vasculature,
        mounted_vasculature=mounted,
    )


@dataclass(frozen=True)
class _CoolantSizingContext:
    fluid: IncompressibleFluid
    maximum_total_rise: float
    reserved_thermal_approach: float
    allowable_bulk_rise: float
    radius_scaling_exponent: float


def _structurally_guided_elements(
    compiled: tuple[_CompiledElement, ...],
    service: ServiceIntent,
    structural_cells: tuple[CellMechanics, ...],
    stages: CoupledStages,
) -> (
    tuple[tuple[_CompiledElement, ...], tuple[VascularThermalSizingReceipt, ...]]
    | RejectedAssembly
):
    """Realize structural guidance with no coolant thermal resize."""
    guided_results = tuple(
        stages.realize(
            element,
            service,
            SealedVascularConfig(),
            structural_cells,
        )
        if structural_cells
        else element
        for element in compiled
    )
    return traverse_results(
        guided_results,
        lambda guided: (guided, ()),
    )


def _thermally_sized_element(
    element: _CompiledElement,
    service: ServiceIntent,
    structural_cells: tuple[CellMechanics, ...],
    context: _CoolantSizingContext,
    stages: CoupledStages,
    peak_heat_load_watts: Callable[[str, ServiceIntent], float],
    descend_terminal_radius_sizing: Callable[..., tuple[_CompiledElement, float, float] | RejectedAssembly],
) -> tuple[_CompiledElement, VascularThermalSizingReceipt | None] | RejectedAssembly:
    baseline_element_result = (
        stages.realize(
            element,
            service,
            SealedVascularConfig(),
            structural_cells,
        )
        if structural_cells
        else element
    )
    if isinstance(baseline_element_result, RejectedAssembly):
        return baseline_element_result
    baseline_element = baseline_element_result
    heat_load = peak_heat_load_watts(
        baseline_element.element_id, service
    )
    if (
        heat_load <= 0.0
        or not isinstance(baseline_element.anatomy, AcceptedAnatomy)
        or not isinstance(
            baseline_element.vasculature, AcceptedVasculature
        )
        or baseline_element.mounted_vasculature is None
    ):
        return baseline_element, None
    baseline_hydraulics = _assembly_hydraulics_result(
        _solve_element_physical_hydraulics(
            baseline_element.mounted_vasculature, service
        )
    )
    if isinstance(baseline_hydraulics, RejectedAssembly):
        return baseline_hydraulics
    required_flow = heat_load / (
        context.fluid.density_kilograms_per_cubic_metre
        * context.fluid.specific_heat_joules_per_kilogram_kelvin
        * context.allowable_bulk_rise
    )
    baseline_flow = (
        baseline_hydraulics.receipt.pump_volumetric_flow_cubic_metres_per_second
    )
    flow_ratio = max(required_flow / baseline_flow, 1.0)
    initial_terminal_radius_scale = (
        flow_ratio**context.radius_scaling_exponent
    )
    sized = descend_terminal_radius_sizing(
        initial_terminal_radius_scale,
        _SEALED_VASCULAR_SIZING_REFINEMENTS,
        baseline_element,
        baseline_hydraulics,
        required_flow,
        service,
        structural_cells,
        context.radius_scaling_exponent,
        stages,
    )
    if isinstance(sized, RejectedAssembly):
        return sized
    sized_element, terminal_radius_scale, achieved_flow = sized
    return (
        sized_element,
        VascularThermalSizingReceipt(
            element_id=baseline_element.element_id,
            heat_load_watts=heat_load,
            maximum_total_temperature_rise_kelvin=context.maximum_total_rise,
            reserved_thermal_approach_kelvin=context.reserved_thermal_approach,
            allowable_bulk_coolant_rise_kelvin=context.allowable_bulk_rise,
            required_volumetric_flow_cubic_metres_per_second=required_flow,
            baseline_volumetric_flow_cubic_metres_per_second=baseline_flow,
            terminal_radius_scale=terminal_radius_scale,
            achieved_volumetric_flow_cubic_metres_per_second=achieved_flow,
        ),
    )
