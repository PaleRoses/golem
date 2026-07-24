"""Analytic acceptance probes for the D25/M7 mechanics owner."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from itertools import product

import numpy as np
import pytest

from golem.assembly.mechanics.compile import _address_mechanics_obstruction
from golem.kernel.anatomy import (
    AcceptedVasculature,
    ClosedVascularGraph,
    DistributingArtery,
    PhysicalHydraulicNodePressure,
    VascularEdge,
    VascularFlowReceipt,
    VascularNode,
    VascularNodeKind,
    VascularStratum,
)
from golem.kernel.mechanics import (
    MODEL_NAME,
    AbsentBucklingSpectrumObstruction,
    AcceptedEmbeddedChannelMechanics,
    AcceptedLinearizedBuckling,
    AcceptedMechanics,
    BoundaryFace,
    BucklingNotRequested,
    CellConstitutiveAssignment,
    CellIndex,
    DomainSource,
    EmbeddedChannelCriteria,
    EmbeddedChannelScaleSeparationObstruction,
    EmbeddedChannelVolumeFractionObstruction,
    EmbeddedChannelVolumeKind,
    FullSolidEmbeddedChannelResponse,
    HomogenizedEmbeddedChannelResponse,
    InternalPressureFace,
    InvalidMaterialFractionObstruction,
    InvalidBodyForceObstruction,
    InvalidConstitutivePropertiesObstruction,
    IsotropicConstitutiveProperties,
    MaterialFractionResolution,
    MaterialFractionRule,
    MechanicsCriteria,
    MechanicsProblem,
    NodalForce,
    NodalSupport,
    NodalTemperatureChange,
    NodeIndex,
    RejectedMechanics,
    RejectedEmbeddedChannelMechanics,
    RigidBodyModeObstruction,
    SingularBucklingSpectrumObstruction,
    StructuredHexDomain,
    UnevaluatedPhysics,
    UnresolvedEmbeddedChannelConstitutiveObstruction,
    UnresolvedMaterialFractionObstruction,
    evaluate_embedded_channel_mechanics,
    solve_mechanics,
)


def _material(
    *,
    young_modulus: float = 70.0e9,
    poisson_ratio: float = 0.3,
    density: float = 2_700.0,
    thermal_expansion: float = 23.0e-6,
    yield_strength: float = 250.0e6,
) -> IsotropicConstitutiveProperties:
    return IsotropicConstitutiveProperties(
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        density=density,
        thermal_expansion_coefficient=thermal_expansion,
        yield_strength=yield_strength,
    )


def _rectangular_domain(
    x_coordinates: tuple[float, ...],
    y_coordinates: tuple[float, ...],
    z_coordinates: tuple[float, ...],
) -> StructuredHexDomain:
    return StructuredHexDomain(
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        z_coordinates=z_coordinates,
        solid_cells=tuple(
            CellIndex(x_index, y_index, z_index)
            for x_index, y_index, z_index in product(
                range(len(x_coordinates) - 1),
                range(len(y_coordinates) - 1),
                range(len(z_coordinates) - 1),
            )
        ),
        source=DomainSource.SDF_DERIVED,
    )


def _assign(
    domain: StructuredHexDomain,
    properties: IsotropicConstitutiveProperties,
) -> tuple[CellConstitutiveAssignment, ...]:
    return tuple(
        CellConstitutiveAssignment(cell, properties)
        for cell in domain.solid_cells
    )


def _accepted(result: AcceptedMechanics | RejectedMechanics) -> AcceptedMechanics:
    assert isinstance(result, AcceptedMechanics), getattr(result, "obstructions", ())
    return result


def _node(result: AcceptedMechanics, index: NodeIndex):
    node = result.node_result(index)
    assert node is not None
    return node


def _accepted_embedded_channel(
    result: AcceptedEmbeddedChannelMechanics | RejectedEmbeddedChannelMechanics,
) -> AcceptedEmbeddedChannelMechanics:
    assert isinstance(result, AcceptedEmbeddedChannelMechanics), getattr(
        result,
        "obstructions",
        (),
    )
    return result


def _single_embedded_channel(radius: float, length: float = 0.1) -> AcceptedVasculature:
    graph = ClosedVascularGraph(
        nodes=(
            VascularNode(
                "source",
                (0.0, 0.0, 0.0),
                VascularNodeKind.PUMP_OUTLET,
                None,
            ),
            VascularNode(
                "target",
                (length, 0.0, 0.0),
                VascularNodeKind.PUMP_INLET,
                None,
            ),
        ),
        edges=(
            VascularEdge(
                "channel",
                "source",
                "target",
                DistributingArtery("probe"),
                VascularStratum.SUPPLY,
                radius,
                1.0,
                1.0,
            ),
        ),
        pump_outlet_node_id="source",
        pump_inlet_node_id="target",
    )
    return AcceptedVasculature(
        graph=graph,
        receipt=VascularFlowReceipt(
            terminal_pair_budget=1,
            terminal_pairs_by_region=(),
            stage_counts=(("distributing_artery", 1),),
            minimum_capsule_margin=1.0,
            maximum_free_cell_residual=0.0,
            boundary_balance_error=0.0,
            maximum_delivery_relative_error=0.0,
            delivered_demand_by_region=(),
            pump_pressure_drop=1.0,
            conservation_error=0.0,
            maximum_symmetry_error=0.0,
            checked_nonincident_pair_count=0,
        ),
    )


def _evaluate_single_embedded_channel(
    radius: float,
    *,
    pressure: float = 250_000.0,
    wall_thickness: float = 20.0e-6,
    criteria: EmbeddedChannelCriteria = EmbeddedChannelCriteria(),
    material_fraction_resolution: MaterialFractionResolution = (
        MaterialFractionResolution.UNRESOLVED
    ),
    homogenized_properties: IsotropicConstitutiveProperties | None = None,
    homogenization_validated_lumen_volume_fraction: float | None = None,
):
    return evaluate_embedded_channel_mechanics(
        _single_embedded_channel(radius),
        (
            PhysicalHydraulicNodePressure("source", pressure),
            PhysicalHydraulicNodePressure("target", pressure),
        ),
        metres_per_world_unit=1.0,
        grid_pitch_metres=1.0e-2,
        host_volume_cubic_metres=1.0e-3,
        wall_thickness_metres=wall_thickness,
        wall_allowable_stress_pascal=100.0e6,
        external_pressure_pascal=0.0,
        background_properties=_material(
            young_modulus=1.0e9,
            density=1_000.0,
            thermal_expansion=0.0,
            yield_strength=100.0e6,
        ),
        material_fraction_resolution=material_fraction_resolution,
        homogenized_properties=homogenized_properties,
        homogenization_validated_lumen_volume_fraction=(
            homogenization_validated_lumen_volume_fraction
        ),
        criteria=criteria,
    )


def test_zero_radius_embedded_channel_recovers_full_solid_exactly() -> None:
    result = _accepted_embedded_channel(
        _evaluate_single_embedded_channel(0.0)
    )
    section = result.sections[0]
    receipt = result.receipt

    assert section.radius_metres == 0.0
    assert section.lumen_capsule_volume_upper_bound_cubic_metres == 0.0
    assert section.wall_annulus_volume_upper_bound_cubic_metres == 0.0
    assert section.thin_wall_hoop_stress_pascal == 0.0
    assert section.lame_inner_wall_hoop_stress_pascal == 0.0
    assert section.burst_safety_factor == np.inf
    assert receipt.conservative_solid_fraction_lower_bound == 1.0
    assert isinstance(
        receipt.constitutive_response,
        FullSolidEmbeddedChannelResponse,
    )
    assert receipt.not_evaluated == ()


def test_embedded_channel_porosity_and_pressure_damage_are_monotone() -> None:
    smaller = _accepted_embedded_channel(
        _evaluate_single_embedded_channel(100.0e-6, pressure=125_000.0)
    )
    larger = _accepted_embedded_channel(
        _evaluate_single_embedded_channel(200.0e-6, pressure=125_000.0)
    )
    higher_pressure = _accepted_embedded_channel(
        _evaluate_single_embedded_channel(100.0e-6, pressure=250_000.0)
    )

    assert (
        larger.receipt.lumen_volume_fraction_upper_bound
        > smaller.receipt.lumen_volume_fraction_upper_bound
    )
    assert (
        larger.receipt.conservative_solid_fraction_lower_bound
        < smaller.receipt.conservative_solid_fraction_lower_bound
    )
    assert (
        higher_pressure.receipt.maximum_lame_inner_wall_hoop_stress_pascal
        > smaller.receipt.maximum_lame_inner_wall_hoop_stress_pascal
    )
    assert (
        higher_pressure.receipt.minimum_burst_safety_factor
        < smaller.receipt.minimum_burst_safety_factor
    )
    assert isinstance(
        smaller.receipt.constitutive_response,
        UnresolvedEmbeddedChannelConstitutiveObstruction,
    )
    assert UnevaluatedPhysics.SUBGRID_CHANNEL_GLOBAL_STIFFNESS in (
        smaller.receipt.not_evaluated
    )
    assert UnevaluatedPhysics.SUBGRID_CHANNEL_GLOBAL_STRENGTH in (
        smaller.receipt.not_evaluated
    )


def test_embedded_channel_thin_wall_limit_has_exact_radius_wall_scaling() -> None:
    reference = _accepted_embedded_channel(
        _evaluate_single_embedded_channel(100.0e-6)
    )
    doubled_radius = _accepted_embedded_channel(
        _evaluate_single_embedded_channel(200.0e-6)
    )
    doubled_wall = _accepted_embedded_channel(
        _evaluate_single_embedded_channel(
            100.0e-6,
            wall_thickness=40.0e-6,
        )
    )
    reference_stress = reference.sections[0].thin_wall_hoop_stress_pascal

    assert doubled_radius.sections[0].thin_wall_hoop_stress_pascal == pytest.approx(
        2.0 * reference_stress
    )
    assert doubled_wall.sections[0].thin_wall_hoop_stress_pascal == pytest.approx(
        0.5 * reference_stress
    )
    assert reference.sections[0].lame_inner_wall_hoop_stress_pascal > (
        reference_stress
    )


def test_embedded_channel_scale_and_volume_thresholds_reject_by_type() -> None:
    scale_result = _evaluate_single_embedded_channel(600.0e-6)
    volume_result = _evaluate_single_embedded_channel(
        100.0e-6,
        criteria=EmbeddedChannelCriteria(
            maximum_lumen_volume_fraction=1.0e-6,
        ),
    )

    assert isinstance(scale_result, RejectedEmbeddedChannelMechanics)
    assert any(
        isinstance(
            obstruction,
            EmbeddedChannelScaleSeparationObstruction,
        )
        for obstruction in scale_result.obstructions
    )
    assert isinstance(volume_result, RejectedEmbeddedChannelMechanics)
    assert any(
        isinstance(obstruction, EmbeddedChannelVolumeFractionObstruction)
        and obstruction.kind is EmbeddedChannelVolumeKind.LUMEN
        for obstruction in volume_result.obstructions
    )


def test_embedded_channel_homogenization_evidence_is_explicit_and_deterministic() -> None:
    homogenized = _material(
        young_modulus=800.0e6,
        density=990.0,
        thermal_expansion=0.0,
        yield_strength=80.0e6,
    )
    evaluation = _evaluate_single_embedded_channel(
        100.0e-6,
        material_fraction_resolution=(
            MaterialFractionResolution.HOMOGENIZED_EFFECTIVE_PROPERTIES
        ),
        homogenized_properties=homogenized,
        homogenization_validated_lumen_volume_fraction=1.0e-4,
    )
    result = _accepted_embedded_channel(evaluation)

    assert _evaluate_single_embedded_channel(
        100.0e-6,
        material_fraction_resolution=(
            MaterialFractionResolution.HOMOGENIZED_EFFECTIVE_PROPERTIES
        ),
        homogenized_properties=homogenized,
        homogenization_validated_lumen_volume_fraction=1.0e-4,
    ) == result
    assert isinstance(
        result.receipt.constitutive_response,
        HomogenizedEmbeddedChannelResponse,
    )
    assert UnevaluatedPhysics.SUBGRID_CHANNEL_GLOBAL_STIFFNESS not in (
        result.receipt.not_evaluated
    )
    assert UnevaluatedPhysics.SUBGRID_CHANNEL_STRESS_CONCENTRATION in (
        result.receipt.not_evaluated
    )


def _axial_bar_problem() -> MechanicsProblem:
    length = 2.0
    domain = _rectangular_domain((0.0, length), (0.0, 1.0), (0.0, 1.0))
    fixed_nodes = tuple(
        NodeIndex(0, y_index, z_index)
        for y_index, z_index in product(range(2), repeat=2)
    )
    loaded_nodes = tuple(
        NodeIndex(1, y_index, z_index)
        for y_index, z_index in product(range(2), repeat=2)
    )
    return MechanicsProblem(
        domain=domain,
        cell_properties=_assign(
            domain,
            _material(
                young_modulus=200.0e9,
                poisson_ratio=0.0,
                density=0.0,
                thermal_expansion=0.0,
            ),
        ),
        supports=tuple(
            NodalSupport(node, (0.0, 0.0, 0.0)) for node in fixed_nodes
        ),
        nodal_forces=tuple(
            NodalForce(node, (250_000.0, 0.0, 0.0)) for node in loaded_nodes
        ),
        criteria=MechanicsCriteria(minimum_yield_safety_factor=100.0),
    )


def test_axial_bar_recovers_exact_displacement_stress_yield_and_reaction() -> None:
    problem = _axial_bar_problem()
    result = _accepted(solve_mechanics(problem))

    # P=1 MN, L=2 m, A=1 m^2, E=200 GPa.
    expected_displacement = 1.0e6 * 2.0 / 200.0e9
    expected_stress = 1.0e6
    loaded_displacements = tuple(
        _node(result, NodeIndex(1, y_index, z_index)).displacement[0]
        for y_index, z_index in product(range(2), repeat=2)
    )

    assert loaded_displacements == pytest.approx(
        (expected_displacement,) * 4, rel=1.0e-12, abs=1.0e-15
    )
    assert result.cells[0].stress.xx == pytest.approx(expected_stress, rel=1.0e-12)
    assert result.cells[0].von_mises_stress == pytest.approx(
        expected_stress, rel=1.0e-12
    )
    assert result.receipt.minimum_yield_safety_factor == pytest.approx(250.0)
    assert result.receipt.total_nodal_force == pytest.approx((1.0e6, 0.0, 0.0))
    assert result.receipt.total_reaction_force == pytest.approx(
        (-1.0e6, 0.0, 0.0), abs=1.0e-8
    )
    assert result.receipt.normalized_linear_solver_residual < 1.0e-12
    assert result.receipt.relative_force_imbalance < 1.0e-12
    assert result.receipt.constitutive_model == MODEL_NAME
    assert isinstance(result.buckling, BucklingNotRequested)


def test_cantilever_matches_euler_bernoulli_tip_deflection_at_declared_fidelity() -> None:
    length = 2.0
    width = 0.2
    height = 0.4
    element_count_x = 20
    element_count_y = 2
    element_count_z = 4
    young_modulus = 70.0e9
    load = 1_000.0
    domain = _rectangular_domain(
        tuple(np.linspace(0.0, length, element_count_x + 1)),
        tuple(np.linspace(-width / 2.0, width / 2.0, element_count_y + 1)),
        tuple(np.linspace(-height / 2.0, height / 2.0, element_count_z + 1)),
    )
    end_nodes = tuple(
        NodeIndex(element_count_x, y_index, z_index)
        for y_index, z_index in product(
            range(element_count_y + 1), range(element_count_z + 1)
        )
    )
    result = _accepted(
        solve_mechanics(
            MechanicsProblem(
                domain=domain,
                cell_properties=_assign(
                    domain,
                    _material(
                        young_modulus=young_modulus,
                        poisson_ratio=0.3,
                        thermal_expansion=0.0,
                    ),
                ),
                supports=tuple(
                    NodalSupport(
                        NodeIndex(0, y_index, z_index), (0.0, 0.0, 0.0)
                    )
                    for y_index, z_index in product(
                        range(element_count_y + 1), range(element_count_z + 1)
                    )
                ),
                nodal_forces=tuple(
                    NodalForce(node, (0.0, 0.0, -load / len(end_nodes)))
                    for node in end_nodes
                ),
                criteria=MechanicsCriteria(maximum_principal_strain=0.1),
            )
        )
    )

    mean_tip_displacement = float(
        np.mean(tuple(_node(result, node).displacement[2] for node in end_nodes))
    )
    second_moment = width * height**3 / 12.0
    analytic_tip_displacement = -(load * length**3) / (
        3.0 * young_modulus * second_moment
    )

    # The 20x2x4 full-integration Q1 grid is within 2% here; allow 3% rather
    # than laundering this benchmark into an Euler-Bernoulli beam solver.
    assert mean_tip_displacement == pytest.approx(
        analytic_tip_displacement, rel=3.0e-2
    )
    assert result.receipt.relative_moment_imbalance < 1.0e-10


def test_linearized_buckling_matches_pinned_euler_column_at_declared_mesh_fidelity() -> None:
    length = 1.0
    width = 0.1
    height = 0.1
    young_modulus = 1.0e6
    reference_compression = 1.0
    element_count_x = 30
    element_count_y = 4
    element_count_z = 4
    domain = _rectangular_domain(
        tuple(np.linspace(0.0, length, element_count_x + 1)),
        tuple(np.linspace(-width / 2.0, width / 2.0, element_count_y + 1)),
        tuple(np.linspace(-height / 2.0, height / 2.0, element_count_z + 1)),
    )
    center_cross_section_node = (
        element_count_y // 2,
        element_count_z // 2,
    )
    left_pin = tuple(
        NodalSupport(
            NodeIndex(0, y_index, z_index),
            (
                0.0
                if (y_index, z_index) == center_cross_section_node
                else None,
                0.0,
                0.0,
            ),
        )
        for y_index, z_index in product(
            range(element_count_y + 1), range(element_count_z + 1)
        )
    )
    right_pin = tuple(
        NodalSupport(
            NodeIndex(element_count_x, y_index, z_index),
            (None, 0.0, 0.0),
        )
        for y_index, z_index in product(
            range(element_count_y + 1), range(element_count_z + 1)
        )
    )
    transverse_weights = np.outer(
        np.asarray((0.5,) + (1.0,) * (element_count_y - 1) + (0.5,)),
        np.asarray((0.5,) + (1.0,) * (element_count_z - 1) + (0.5,)),
    )
    normalized_weights = transverse_weights / np.sum(transverse_weights)
    second_moment = min(width * height**3, height * width**3) / 12.0
    euler_critical_load = np.pi**2 * young_modulus * second_moment / length**2
    problem = MechanicsProblem(
        domain=domain,
        cell_properties=_assign(
            domain,
            _material(
                young_modulus=young_modulus,
                poisson_ratio=0.0,
                density=0.0,
                thermal_expansion=0.0,
                yield_strength=1.0e9,
            ),
        ),
        supports=left_pin + right_pin,
        nodal_forces=tuple(
            NodalForce(
                NodeIndex(element_count_x, y_index, z_index),
                (
                    -reference_compression
                    * float(normalized_weights[y_index, z_index]),
                    0.0,
                    0.0,
                ),
            )
            for y_index, z_index in product(
                range(element_count_y + 1),
                range(element_count_z + 1),
            )
        ),
        criteria=MechanicsCriteria(
            require_linearized_buckling=True,
            minimum_linearized_buckling_load_factor=(
                0.90 * euler_critical_load / reference_compression
            ),
        ),
    )
    result = _accepted(solve_mechanics(problem))

    assert isinstance(result.buckling, AcceptedLinearizedBuckling)
    expected_load_factor = euler_critical_load / reference_compression
    # Full-integration Q1 solids converge from above through shear locking.
    # This frozen 30x4x4 mesh is within 5%; the 6% gate states that fidelity.
    assert result.buckling.receipt.lowest_positive_load_factor == pytest.approx(
        expected_load_factor,
        rel=6.0e-2,
    )
    assert result.buckling.receipt.normalized_eigen_residual < 1.0e-8
    assert result.receipt.lowest_positive_buckling_load_factor == pytest.approx(
        result.buckling.receipt.lowest_positive_load_factor
    )
    assert UnevaluatedPhysics.LINEARIZED_BUCKLING not in result.receipt.not_evaluated
    assert max(
        np.linalg.norm(displacement)
        for displacement in result.buckling.mode_displacements
    ) == pytest.approx(1.0)
    assert _accepted(solve_mechanics(problem)).buckling == result.buckling


def test_closed_lumen_follower_pressure_conserves_and_stabilizes_buckling() -> None:
    length = 1.0
    width = 0.1
    element_count_x = 12
    element_count_y = 5
    element_count_z = 5
    pressure = 500.0
    void_cells = frozenset(
        CellIndex(x_index, 2, 2)
        for x_index in range(1, element_count_x - 1)
    )
    domain = StructuredHexDomain(
        x_coordinates=tuple(np.linspace(0.0, length, element_count_x + 1)),
        y_coordinates=tuple(
            np.linspace(-width / 2.0, width / 2.0, element_count_y + 1)
        ),
        z_coordinates=tuple(
            np.linspace(-width / 2.0, width / 2.0, element_count_z + 1)
        ),
        solid_cells=tuple(
            cell
            for indices in product(
                range(element_count_x),
                range(element_count_y),
                range(element_count_z),
            )
            for cell in (CellIndex(*indices),)
            if cell not in void_cells
        ),
    )
    center_cross_section_node = (
        element_count_y // 2,
        element_count_z // 2,
    )
    left_pin = tuple(
        NodalSupport(
            NodeIndex(0, y_index, z_index),
            (
                0.0
                if (y_index, z_index) == center_cross_section_node
                else None,
                0.0,
                0.0,
            ),
        )
        for y_index, z_index in product(
            range(element_count_y + 1),
            range(element_count_z + 1),
        )
    )
    right_pin = tuple(
        NodalSupport(
            NodeIndex(element_count_x, y_index, z_index),
            (None, 0.0, 0.0),
        )
        for y_index, z_index in product(
            range(element_count_y + 1),
            range(element_count_z + 1),
        )
    )
    transverse_weights = np.outer(
        np.asarray((0.5,) + (1.0,) * (element_count_y - 1) + (0.5,)),
        np.asarray((0.5,) + (1.0,) * (element_count_z - 1) + (0.5,)),
    )
    normalized_weights = transverse_weights / np.sum(transverse_weights)
    reference_problem = MechanicsProblem(
        domain=domain,
        cell_properties=_assign(
            domain,
            _material(
                young_modulus=1.0e6,
                poisson_ratio=0.0,
                density=0.0,
                thermal_expansion=0.0,
                yield_strength=1.0e9,
            ),
        ),
        supports=left_pin + right_pin,
        nodal_forces=tuple(
            NodalForce(
                NodeIndex(element_count_x, y_index, z_index),
                (-float(normalized_weights[y_index, z_index]), 0.0, 0.0),
            )
            for y_index, z_index in product(
                range(element_count_y + 1),
                range(element_count_z + 1),
            )
        ),
        criteria=MechanicsCriteria(require_linearized_buckling=True),
    )
    lumen_faces = (
        tuple(
            InternalPressureFace(
                CellIndex(x_index, 1, 2),
                BoundaryFace.Y_MAX,
                pressure,
            )
            for x_index in range(1, element_count_x - 1)
        )
        + tuple(
            InternalPressureFace(
                CellIndex(x_index, 3, 2),
                BoundaryFace.Y_MIN,
                pressure,
            )
            for x_index in range(1, element_count_x - 1)
        )
        + tuple(
            InternalPressureFace(
                CellIndex(x_index, 2, 1),
                BoundaryFace.Z_MAX,
                pressure,
            )
            for x_index in range(1, element_count_x - 1)
        )
        + tuple(
            InternalPressureFace(
                CellIndex(x_index, 2, 3),
                BoundaryFace.Z_MIN,
                pressure,
            )
            for x_index in range(1, element_count_x - 1)
        )
        + (
            InternalPressureFace(
                CellIndex(0, 2, 2),
                BoundaryFace.X_MAX,
                pressure,
            ),
            InternalPressureFace(
                CellIndex(element_count_x - 1, 2, 2),
                BoundaryFace.X_MIN,
                pressure,
            ),
        )
    )
    unpressurized = _accepted(solve_mechanics(reference_problem))
    pressurized = _accepted(
        solve_mechanics(
            replace(reference_problem, internal_pressure_faces=lumen_faces)
        )
    )
    reordered = _accepted(
        solve_mechanics(
            replace(
                reference_problem,
                internal_pressure_faces=tuple(reversed(lumen_faces)),
            )
        )
    )

    assert isinstance(unpressurized.buckling, AcceptedLinearizedBuckling)
    assert isinstance(pressurized.buckling, AcceptedLinearizedBuckling)
    assert isinstance(reordered.buckling, AcceptedLinearizedBuckling)
    pressure_receipt = pressurized.buckling.receipt
    assert pressure_receipt.follower_pressure_stiffness_frobenius_norm > 0.0
    assert pressure_receipt.stability_matrix_relative_asymmetry < 1.0e-10
    assert pressure_receipt.eigensolver_kind.startswith("symmetric")
    assert pressurized.receipt.total_internal_pressure_force == pytest.approx(
        (0.0, 0.0, 0.0),
        abs=1.0e-12,
    )
    assert pressure_receipt.lowest_positive_load_factor > (
        unpressurized.buckling.receipt.lowest_positive_load_factor
    )
    assert reordered.buckling.receipt.lowest_positive_load_factor == pytest.approx(
        pressure_receipt.lowest_positive_load_factor,
        rel=1.0e-10,
    )


def test_uniform_free_thermal_expansion_has_analytic_displacement_and_zero_stress() -> None:
    domain = _rectangular_domain((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    expansion_coefficient = 23.0e-6
    delta_temperature = 100.0
    result = _accepted(
        solve_mechanics(
            MechanicsProblem(
                domain=domain,
                cell_properties=_assign(
                    domain,
                    _material(thermal_expansion=expansion_coefficient),
                ),
                # Six compatible scalar constraints remove only rigid motion.
                supports=(
                    NodalSupport(NodeIndex(0, 0, 0), (0.0, 0.0, 0.0)),
                    NodalSupport(NodeIndex(1, 0, 0), (None, 0.0, 0.0)),
                    NodalSupport(NodeIndex(0, 1, 0), (None, None, 0.0)),
                ),
                nodal_temperature_changes=tuple(
                    NodalTemperatureChange(NodeIndex(*index), delta_temperature)
                    for index in product(range(2), repeat=3)
                ),
            )
        )
    )

    expansion = expansion_coefficient * delta_temperature
    assert np.asarray(tuple(node.displacement for node in result.nodes)) == pytest.approx(
        np.asarray(
            tuple(
                tuple(expansion * coordinate for coordinate in node.position)
                for node in result.nodes
            )
        ),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert result.cells[0].von_mises_stress < 1.0e-6
    assert result.cells[0].yield_safety_factor is not None
    assert result.cells[0].yield_safety_factor > 1.0e12
    assert result.receipt.relative_force_imbalance < 1.0e-12


def test_flat_pressurized_wall_patch_recovers_uniform_compression() -> None:
    thickness = 0.5
    width = 2.0
    height = 3.0
    pressure = 2.0e6
    young_modulus = 10.0e9
    domain = _rectangular_domain(
        (0.0, thickness), (0.0, width), (0.0, height)
    )
    result = _accepted(
        solve_mechanics(
            MechanicsProblem(
                domain=domain,
                cell_properties=_assign(
                    domain,
                    _material(
                        young_modulus=young_modulus,
                        poisson_ratio=0.0,
                        density=0.0,
                        thermal_expansion=0.0,
                        yield_strength=100.0e6,
                    ),
                ),
                supports=tuple(
                    NodalSupport(
                        NodeIndex(0, y_index, z_index), (0.0, 0.0, 0.0)
                    )
                    for y_index, z_index in product(range(2), repeat=2)
                ),
                internal_pressure_faces=(
                    InternalPressureFace(
                        CellIndex(0, 0, 0), BoundaryFace.X_MAX, pressure
                    ),
                ),
            )
        )
    )

    expected_displacement = -pressure * thickness / young_modulus
    assert tuple(
        node.displacement[0]
        for node in result.nodes
        if node.node.x_index == 1
    ) == pytest.approx((expected_displacement,) * 4, rel=1.0e-12)
    assert result.cells[0].stress.xx == pytest.approx(-pressure, rel=1.0e-12)
    assert result.receipt.total_internal_pressure_force == pytest.approx(
        (-pressure * width * height, 0.0, 0.0)
    )
    assert result.receipt.total_reaction_force == pytest.approx(
        (pressure * width * height, 0.0, 0.0), abs=1.0e-7
    )


def test_gravity_body_force_descends_to_support_reactions_and_force_balance() -> None:
    domain = _rectangular_domain((0.0, 2.0), (0.0, 3.0), (0.0, 4.0))
    density = 1_000.0
    gravity = -9.81
    volume = 24.0
    result = _accepted(
        solve_mechanics(
            MechanicsProblem(
                domain=domain,
                cell_properties=_assign(domain, _material(density=density)),
                supports=tuple(
                    NodalSupport(
                        NodeIndex(x_index, y_index, 0), (0.0, 0.0, 0.0)
                    )
                    for x_index, y_index in product(range(2), repeat=2)
                ),
                body_acceleration=(0.0, 0.0, gravity),
            )
        )
    )

    weight = density * volume * gravity
    assert result.receipt.total_body_force == pytest.approx((0.0, 0.0, weight))
    assert result.receipt.total_reaction_force == pytest.approx(
        (0.0, 0.0, -weight), abs=1.0e-7
    )
    assert result.receipt.relative_force_imbalance < 1.0e-12


def test_fractional_cells_require_resolved_homogenized_effective_properties() -> None:
    problem = _axial_bar_problem()
    cell = problem.domain.solid_cells[0]
    properties = problem.cell_properties[0].properties
    unresolved = replace(
        problem,
        cell_properties=(
            CellConstitutiveAssignment(
                cell,
                properties,
                solid_fraction=0.5,
                fraction_resolution=MaterialFractionResolution.UNRESOLVED,
            ),
        ),
    )
    invalid = replace(
        problem,
        cell_properties=(
            CellConstitutiveAssignment(cell, properties, solid_fraction=0.0),
        ),
    )

    unresolved_result = solve_mechanics(unresolved)
    invalid_result = solve_mechanics(invalid)
    assert isinstance(unresolved_result, RejectedMechanics)
    assert any(
        isinstance(obstruction, UnresolvedMaterialFractionObstruction)
        for obstruction in unresolved_result.obstructions
    )
    assert isinstance(invalid_result, RejectedMechanics)
    assert any(
        isinstance(obstruction, InvalidMaterialFractionObstruction)
        for obstruction in invalid_result.obstructions
    )
    invalid_fraction = next(
        obstruction
        for obstruction in invalid_result.obstructions
        if isinstance(obstruction, InvalidMaterialFractionObstruction)
    )
    assert invalid_fraction.rule is MaterialFractionRule.ACTIVE_CELL_RANGE
    assert invalid_fraction.address == "domain/cell:0:0:0"

    # The material owner may instead supply already-homogenized E/rho/yield.
    homogenized = _accepted(
        solve_mechanics(
            replace(
                problem,
                cell_properties=(
                    CellConstitutiveAssignment(
                        cell,
                        properties,
                        solid_fraction=0.5,
                        fraction_resolution=(
                            MaterialFractionResolution.HOMOGENIZED_EFFECTIVE_PROPERTIES
                        ),
                    ),
                ),
            )
        )
    )
    assert homogenized.receipt.minimum_solid_fraction == 0.5
    assert homogenized.receipt.fractional_cell_count == 1
    assert (
        UnevaluatedPhysics.SUBGRID_CHANNEL_STRESS_CONCENTRATION
        in homogenized.receipt.not_evaluated
    )


def test_assembly_mechanics_boundary_glues_semantic_and_case_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = CellIndex(1, 2, 3)
    monkeypatch.setattr(
        "golem.assembly.mechanics.compile._semantic_address_for_cell",
        lambda _element, _domain, _cell: "region:body/torso",
    )
    constitutive = InvalidConstitutivePropertiesObstruction(
        "domain/cell:1:2:3",
        cell,
        "poisson_ratio",
        0.8,
    )
    body_force = InvalidBodyForceObstruction(
        "body_acceleration",
        (0.0, float("inf"), 0.0),
    )

    addressed_constitutive = _address_mechanics_obstruction(
        object(),
        object(),
        2,
        constitutive,
    )
    addressed_body_force = _address_mechanics_obstruction(
        object(),
        object(),
        2,
        body_force,
    )

    assert addressed_constitutive.address == "region:body/torso"
    assert addressed_body_force.address == "meta@service.cases[2]"
    assert constitutive.address == "domain/cell:1:2:3"
    assert body_force.address == "body_acceleration"


def test_rigid_modes_and_nondestabilizing_buckling_spectra_are_typed_rejections() -> None:
    problem = _axial_bar_problem()
    rigid_result = solve_mechanics(replace(problem, supports=()))
    buckling_result = solve_mechanics(
        replace(
            problem,
            criteria=replace(
                problem.criteria,
                require_linearized_buckling=True,
            ),
        )
    )

    assert isinstance(rigid_result, RejectedMechanics)
    assert any(
        isinstance(obstruction, RigidBodyModeObstruction)
        for obstruction in rigid_result.obstructions
    )
    assert isinstance(buckling_result, RejectedMechanics)
    assert any(
        isinstance(obstruction, AbsentBucklingSpectrumObstruction)
        for obstruction in buckling_result.obstructions
    )


def test_singular_spectrum_rejects_and_open_follower_uses_general_pencil() -> None:
    domain = _rectangular_domain((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    properties = _assign(
        domain,
        _material(
            young_modulus=1.0e6,
            poisson_ratio=0.0,
            density=0.0,
            thermal_expansion=0.0,
        ),
    )
    fully_prescribed_compression = solve_mechanics(
        MechanicsProblem(
            domain=domain,
            cell_properties=properties,
            supports=tuple(
                NodalSupport(
                    NodeIndex(*index),
                    (-1.0e-4 * index[0], 0.0, 0.0),
                )
                for index in product(range(2), repeat=3)
            ),
            criteria=MechanicsCriteria(require_linearized_buckling=True),
        )
    )
    follower_pressure = solve_mechanics(
        MechanicsProblem(
            domain=domain,
            cell_properties=properties,
            supports=tuple(
                NodalSupport(
                    NodeIndex(0, y_index, z_index),
                    (0.0, 0.0, 0.0),
                )
                for y_index, z_index in product(range(2), repeat=2)
            ),
            internal_pressure_faces=(
                InternalPressureFace(
                    CellIndex(0, 0, 0),
                    BoundaryFace.X_MAX,
                    1_000.0,
                ),
            ),
            criteria=MechanicsCriteria(require_linearized_buckling=True),
        )
    )

    assert isinstance(fully_prescribed_compression, RejectedMechanics)
    assert any(
        isinstance(obstruction, SingularBucklingSpectrumObstruction)
        for obstruction in fully_prescribed_compression.obstructions
    )
    accepted_follower_pressure = _accepted(follower_pressure)
    assert isinstance(
        accepted_follower_pressure.buckling,
        AcceptedLinearizedBuckling,
    )
    follower_receipt = accepted_follower_pressure.buckling.receipt
    assert follower_receipt.follower_pressure_stiffness_frobenius_norm > 0.0
    assert follower_receipt.stability_matrix_relative_asymmetry > 1.0e-10
    assert follower_receipt.eigensolver_kind.startswith("nonsymmetric")
    assert follower_receipt.normalized_eigen_residual < 1.0e-8


def test_mechanics_inputs_and_results_are_frozen() -> None:
    problem = _axial_bar_problem()
    result = _accepted(solve_mechanics(problem))

    assert solve_mechanics(problem) == result
    with pytest.raises(FrozenInstanceError):
        problem.body_acceleration = (0.0, -9.81, 0.0)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.receipt.maximum_displacement = 0.0  # type: ignore[misc]
