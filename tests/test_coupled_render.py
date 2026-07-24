"""Acceptance proofs for the M7 coupled-result derived renderer."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

from golem.assembly.core import (
    AcceptedAssembly,
    AssemblyRecord,
    CoupledPerformanceReceipt,
    ElementMaterialVolumeReceipt,
    ElementMechanicsReceipt,
    ElementThermalReceipt,
    RejectedAssembly,
    VisualAssemblyReceipt,
)
from golem.assembly.service import (
    ChannelResolutionPolicy,
    CoolantService,
    ElementAddress,
    ForceLoad,
    HeatSource,
    ManufacturingLimits,
    MountAddress,
    OperatingLimits,
    PhysicsEvidencePolicy,
    PortAddress,
    PressurePumpBudget,
    ServiceCase,
    ServiceIntent,
    SIScale,
    SolidConstitutiveModel,
    SolidMaterialAssignment,
    SolidMaterialRole,
    Support,
    SupportLaw,
    Vector3,
)
from golem.kernel import sheaf
from golem.kernel.mechanics import (
    AcceptedMechanics,
    CellConstitutiveAssignment,
    CellIndex,
    DomainSource,
    IsotropicConstitutiveProperties,
    MechanicsProblem,
    NodalForce,
    NodalSupport,
    NodeIndex,
    StructuredHexDomain,
    solve_mechanics,
)
from golem.materials import MATERIAL_CATALOGUE


_ROOT = Path(__file__).resolve().parents[1]
_RENDERER_PATH = _ROOT / "rehearsal" / "coupled" / "render_acceptance.py"


def _load_renderer():
    spec = importlib.util.spec_from_file_location(
        "golem_coupled_acceptance_renderer", _RENDERER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _service() -> ServiceIntent:
    solid = MATERIAL_CATALOGUE.isotropic_solids[0]
    fluid = MATERIAL_CATALOGUE.incompressible_fluids[0]
    return ServiceIntent(
        scale=SIScale(1.0),
        constitutive_model=(
            SolidConstitutiveModel.ISOTROPIC_LINEAR_THERMOELASTIC
        ),
        solid_materials=(
            SolidMaterialAssignment(
                SolidMaterialRole.STRUCTURE, ElementAddress("body"), solid
            ),
            SolidMaterialAssignment(
                SolidMaterialRole.LUMEN_WALL, ElementAddress("body"), solid
            ),
        ),
        cases=(
            ServiceCase(
                "render_case",
                Vector3(0.0, 0.0, 0.0),
                Vector3(0.0, 0.0, 0.0),
                (
                    ForceLoad(
                        PortAddress("body", "pump_inlet"),
                        Vector3(1_000.0, 0.0, 0.0),
                    ),
                ),
                (Support(MountAddress("body"), SupportLaw.FIXED),),
                (),
                (HeatSource(ElementAddress("body"), 20.0),),
            ),
        ),
        ambient=None,
        coolant=CoolantService(
            fluid, 293.15, PressurePumpBudget(200_000.0)
        ),
        manufacturing=ManufacturingLimits(0.05, 0.02, 0.02),
        limits=OperatingLimits(
            350.0,
            10_000.0,
            0.1,
            1.0,
            1.0,
            1.0,
        ),
        evidence_policy=PhysicsEvidencePolicy(
            ChannelResolutionPolicy.RESOLVED_ONLY, 4, 2, 1, 0.05
        ),
    )


def _mechanics_solution() -> AcceptedMechanics:
    domain = StructuredHexDomain(
        x_coordinates=(0.0, 0.5, 1.0),
        y_coordinates=(0.0, 1.0),
        z_coordinates=(0.0, 1.0),
        solid_cells=(CellIndex(0, 0, 0), CellIndex(1, 0, 0)),
        source=DomainSource.SDF_DERIVED,
    )
    material = IsotropicConstitutiveProperties(
        young_modulus=200.0e9,
        poisson_ratio=0.0,
        density=0.0,
        thermal_expansion_coefficient=0.0,
        yield_strength=170.0e6,
    )
    fixed_nodes = tuple(
        NodeIndex(0, y_index, z_index)
        for y_index, z_index in product(range(2), repeat=2)
    )
    loaded_nodes = tuple(
        NodeIndex(2, y_index, z_index)
        for y_index, z_index in product(range(2), repeat=2)
    )
    result = solve_mechanics(
        MechanicsProblem(
            domain=domain,
            cell_properties=tuple(
                CellConstitutiveAssignment(cell, material)
                for cell in domain.solid_cells
            ),
            supports=tuple(
                NodalSupport(node, (0.0, 0.0, 0.0)) for node in fixed_nodes
            ),
            nodal_forces=tuple(
                NodalForce(node, (250.0, 0.0, 0.0))
                for node in loaded_nodes
            ),
        )
    )
    assert isinstance(result, AcceptedMechanics), getattr(
        result, "obstructions", ()
    )
    return result


def _thermal_solution(reverse: bool = False) -> sheaf.ThermalSolution:
    temperatures = (313.15, 293.15) if reverse else (293.15, 313.15)
    result = sheaf.solve_thermal_balance(
        sheaf.ThermalBalanceProblem(
            problem_id=sheaf.BalanceProblemId("render-thermal"),
            domain=sheaf.CellularComplex(
                "render-thermal-domain",
                ((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)),
                (
                    sheaf.CellAdjacency(
                        sheaf.CellId(0), sheaf.CellId(1), 1.0
                    ),
                ),
            ),
            solid_conduction_links=(
                sheaf.SolidConductionLink(
                    sheaf.BalanceInterfaceId("body-conduction"),
                    sheaf.CellId(0),
                    sheaf.CellId(1),
                    15.0,
                    1.0,
                    0.5,
                ),
            ),
            fixed_temperatures=(
                sheaf.FixedTemperature(
                    sheaf.BoundaryId("left-temperature"),
                    sheaf.CellId(0),
                    temperatures[0],
                ),
                sheaf.FixedTemperature(
                    sheaf.BoundaryId("right-temperature"),
                    sheaf.CellId(1),
                    temperatures[1],
                ),
            ),
        )
    )
    assert isinstance(result, sheaf.Accepted), getattr(
        result, "obstructions", ()
    )
    return result.value


def _cube_record() -> AssemblyRecord:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ),
        dtype=np.int64,
    )
    return AssemblyRecord.from_derived_geometry(
        record_id="body",
        element_id="body",
        role="creature",
        stratum="solid",
        appearance_material="neutral_gray",
        vertices=vertices,
        faces=faces,
        resolution=3,
        report={"faces": len(faces)},
    )


def _vascular_record(
    stratum: str,
    appearance_material: str,
    y_coordinate: float,
) -> AssemblyRecord:
    vertices = np.asarray(
        (
            (0.10, y_coordinate - 0.03, 0.48),
            (0.90, y_coordinate, 0.52),
            (0.10, y_coordinate + 0.03, 0.48),
            (0.90, y_coordinate, 0.46),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)),
        dtype=np.int64,
    )
    return AssemblyRecord.from_derived_geometry(
        record_id=f"body__{stratum}",
        element_id="body",
        role="creature",
        stratum=stratum,
        appearance_material=appearance_material,
        vertices=vertices,
        faces=faces,
        resolution=3,
        report={"faces": len(faces)},
    )


def _accepted_assembly(reverse_temperature: bool = False) -> AcceptedAssembly:
    thermal = _thermal_solution(reverse_temperature)
    mechanics = _mechanics_solution()
    temperatures = tuple(
        value.value for value in thermal.balance.field.values
    )
    thermal_receipt = ElementThermalReceipt(
        element_id="body",
        case_id="render_case",
        maximum_temperature_kelvin=max(temperatures),
        mean_temperature_kelvin=float(np.mean(temperatures)),
        maximum_temperature_gradient_kelvin_per_metre=(
            abs(temperatures[1] - temperatures[0]) / 0.5
        ),
        coolant_outlet_temperature_kelvin=None,
        hottest_semantic_address="element:body",
        heat_transfer_correlations=(),
        natural_convection=None,
        energy=thermal.energy,
    )
    mechanics_receipt = ElementMechanicsReceipt(
        element_id="body",
        case_id="render_case",
        mechanics=mechanics.receipt,
        minimum_burst_safety_factor=None,
        critical_semantic_address="element:body",
    )
    records = (
        _cube_record(),
        _vascular_record(
            "vascular_supply", "vascular_supply_gold", 0.35
        ),
        _vascular_record(
            "vascular_return", "vascular_return_violet", 0.50
        ),
        _vascular_record(
            "vascular_exchange", "vascular_exchange_cyan", 0.65
        ),
    )
    return AcceptedAssembly(
        records=records,
        service=_service(),
        receipt=CoupledPerformanceReceipt(
            physical_evidence=_service().physical_evidence,
            metres_per_world_unit=1.0,
            service_case_ids=("render_case",),
            material_ids=(
                "stainless_steel_316l_room_temperature",
            ),
            material_volumes=(
                ElementMaterialVolumeReceipt("body", 0.9, 0.05, 0.05, 0.02),
            ),
            vascular_sizing_receipts=(),
            hydraulic_receipts=(),
            thermal_receipts=(thermal_receipt,),
            mechanics_receipts=(mechanics_receipt,),
            interface_receipts=(),
            refinement_receipts=(),
            not_evaluated=(),
            fixed_point_iterations=1,
            maximum_state_delta=0.0,
        ),
        thermal_solutions=(thermal,),
        mechanics_solutions=(mechanics,),
    )


def test_accepted_coupled_result_emits_five_deterministic_sheets(
    tmp_path: Path,
) -> None:
    renderer = _load_renderer()
    assembly = _accepted_assembly()
    first = renderer.render_coupled_acceptance(
        assembly, tmp_path / "first", image_size=96
    )
    second = renderer.render_coupled_acceptance(
        assembly, tmp_path / "second", image_size=96
    )
    assert isinstance(first, renderer.AcceptedCoupledRender)
    assert isinstance(second, renderer.AcceptedCoupledRender)
    assert tuple(path.name for path in first.paths.all_paths) == (
        "supply_return_exchange_xray_sheet.png",
        "network_only_orthographic_sheet.png",
        "temperature_sheet.png",
        "stress_sheet.png",
        "assembled_knight_sheet.png",
    )
    assert all(path.is_file() for path in first.paths.all_paths)
    assert tuple(
        Image.open(path).size for path in first.paths.all_paths
    ) == ((384, 96), (384, 96), (96, 96), (96, 96), (192, 96))
    assert tuple(
        projection.quantity for projection in first.scalar_projections
    ) == ("temperature_kelvin", "von_mises_stress_pascal")
    assert first.scalar_projections[0].source_minimum == 293.15
    assert first.scalar_projections[0].source_maximum == 313.15
    assert tuple(
        left.read_bytes() == right.read_bytes()
        for left, right in zip(
            first.paths.all_paths, second.paths.all_paths, strict=True
        )
    ) == (True,) * 5


def test_scalar_colour_is_derived_from_the_accepted_temperature_field(
    tmp_path: Path,
) -> None:
    renderer = _load_renderer()
    normal = renderer.render_coupled_acceptance(
        _accepted_assembly(), tmp_path / "normal", image_size=96
    )
    reversed_field = renderer.render_coupled_acceptance(
        _accepted_assembly(reverse_temperature=True),
        tmp_path / "reversed",
        image_size=96,
    )
    assert isinstance(normal, renderer.AcceptedCoupledRender)
    assert isinstance(reversed_field, renderer.AcceptedCoupledRender)
    assert (
        normal.paths.temperature.read_bytes()
        != reversed_field.paths.temperature.read_bytes()
    )
    assert (
        normal.paths.stress.read_bytes()
        == reversed_field.paths.stress.read_bytes()
    )


def test_rejected_or_visual_only_assembly_writes_no_images(tmp_path: Path) -> None:
    renderer = _load_renderer()
    rejected_output = tmp_path / "rejected"
    rejected = renderer.render_coupled_acceptance(
        RejectedAssembly(("unresolved lumen",)), rejected_output, image_size=96
    )
    visual_output = tmp_path / "visual"
    visual = renderer.render_coupled_acceptance(
        replace(_accepted_assembly(), receipt=VisualAssemblyReceipt()),
        visual_output,
        image_size=96,
    )
    assert isinstance(rejected, renderer.RejectedCoupledRender)
    assert isinstance(visual, renderer.RejectedCoupledRender)
    assert not rejected_output.exists()
    assert not visual_output.exists()


def test_renderer_contains_no_physics_or_assembly_solver() -> None:
    source = _RENDERER_PATH.read_text()
    assert "solve_balance" not in source
    assert "solve_thermal_balance" not in source
    assert "solve_mechanics" not in source
    assert "compile_assembly" not in source


def test_coupled_cli_is_the_only_acceptance_render_execution_surface() -> None:
    legacy = _ROOT / "rehearsal" / "vascular" / "render_acceptance.py"
    cli = (_ROOT / "rehearsal" / "coupled" / "render_cli.py").read_text()
    assert not legacy.exists()
    assert "importlib" not in cli
    assert (
        "from rehearsal.coupled.render_acceptance import" in cli
    )


def test_scalar_framing_is_local_and_the_pilot_renderer_remains_frozen() -> None:
    renderer = _load_renderer()
    assembly = _accepted_assembly()
    body = tuple(
        record for record in assembly.records if record.stratum == "solid"
    )
    panel = renderer._temperature_panel(
        body,
        assembly.thermal_solutions[0],
        "render_case",
        1.0,
        128,
    )
    pixels = np.asarray(panel.image)
    foreground = np.any(pixels != 244, axis=2)
    rows, columns = np.nonzero(foreground)
    occupancy = max(
        rows.max() - rows.min() + 1,
        columns.max() - columns.min() + 1,
    ) / panel.image.width
    pilot_source = (_ROOT / "pilots" / "render.py").read_text()
    assert 0.72 <= occupancy <= 0.90
    assert "scale = 0.44 * size" in pilot_source
    assert "FRAME_OCCUPANCY" not in pilot_source
