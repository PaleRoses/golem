"""Law and analytic tests for GOLEM's shared immutable balance algebra."""

from __future__ import annotations

from dataclasses import replace

import pytest

from golem import materials
from golem.kernel import sheaf


def _balance_domain(cell_count: int) -> sheaf.CellularComplex:
    return sheaf.CellularComplex(
        complex_id=f"balance-{cell_count}",
        cell_centers=tuple((float(index), 0.0, 0.0) for index in range(cell_count)),
        adjacencies=(),
    )


def _laminar_correlation(
    correlation_id: str = "laminar-wall",
    development_length_metres: float = 0.1,
) -> sheaf.FullyDevelopedLaminarCircularTube:
    return sheaf.FullyDevelopedLaminarCircularTube(
        sheaf.HeatTransferCorrelationId(correlation_id),
        sheaf.CircularTubeGeometry(0.01, development_length_metres),
        reynolds=100.0,
        prandtl=1.0,
        boundary_condition=(
            sheaf.LaminarThermalBoundaryCondition.CONSTANT_WALL_TEMPERATURE
        ),
    )


def _dry_air() -> materials.CompressibleGas:
    result = materials.decode_compressible_gas(
        materials.CompressibleGasMaterialId.DRY_AIR_101325_PA_298K.value
    )
    assert isinstance(result, materials.AcceptedMaterial)
    return result.material


def test_balance_glues_solid_exchange_and_directed_coolant_advection() -> None:
    result = sheaf.solve_balance(
        sheaf.BalanceProblem(
            problem_id=sheaf.BalanceProblemId("solid-coolant"),
            domain=_balance_domain(4),
            symmetric_conductances=(
                sheaf.SymmetricConductance(
                    sheaf.BalanceInterfaceId("wall"),
                    sheaf.CellId(0),
                    sheaf.CellId(2),
                    10.0,
                ),
            ),
            directed_transports=(
                sheaf.DirectedTransport(
                    sheaf.BalanceInterfaceId("coolant-in"),
                    sheaf.CellId(1),
                    sheaf.CellId(2),
                    20.0,
                ),
                sheaf.DirectedTransport(
                    sheaf.BalanceInterfaceId("coolant-out"),
                    sheaf.CellId(2),
                    sheaf.CellId(3),
                    20.0,
                ),
            ),
            sources=(
                sheaf.CellSource(
                    sheaf.BalanceSourceId("solid-heat"), sheaf.CellId(0), 100.0
                ),
            ),
            fixed_boundaries=(
                sheaf.FixedBoundary(
                    sheaf.BoundaryId("coolant-inlet"), sheaf.CellId(1), 300.0
                ),
                sheaf.FixedBoundary(
                    sheaf.BoundaryId("coolant-outlet"), sheaf.CellId(3), 0.0
                ),
            ),
        )
    )
    assert isinstance(result, sheaf.Accepted)
    solution = result.value
    assert tuple(value.value for value in solution.field.values) == (
        315.0,
        300.0,
        305.0,
        0.0,
    )
    wall_flux = solution.interface_fluxes[0]
    assert isinstance(wall_flux, sheaf.SymmetricConductanceFlux)
    assert wall_flux.left_to_right_flux == 100.0
    assert tuple(
        flux.upstream_to_downstream_flux
        for flux in solution.interface_fluxes
        if isinstance(flux, sheaf.DirectedTransportFlux)
    ) == (6000.0, 6100.0)
    assert solution.maximum_normalized_residual == 0.0
    assert solution.relative_imbalance == 0.0
    assert tuple(
        supply.boundary_to_domain_flux
        for supply in solution.fixed_boundary_supplies
    ) == (6000.0, -6100.0)


def test_source_and_robin_exchange_have_the_analytic_single_cell_solution() -> None:
    result = sheaf.solve_balance(
        sheaf.BalanceProblem(
            problem_id=sheaf.BalanceProblemId("robin-cell"),
            domain=_balance_domain(1),
            sources=(
                sheaf.CellSource(
                    sheaf.BalanceSourceId("heat"), sheaf.CellId(0), 20.0
                ),
            ),
            ambient_exchanges=(
                sheaf.AmbientExchange(
                    sheaf.AmbientExchangeId("air"),
                    sheaf.CellId(0),
                    2.0,
                    10.0,
                ),
            ),
        )
    )
    assert isinstance(result, sheaf.Accepted)
    assert result.value.field.value_at(sheaf.CellId(0)) == 20.0
    assert result.value.ambient_fluxes == (
        sheaf.AmbientExchangeFlux(
            sheaf.AmbientExchangeId("air"),
            sheaf.CellId(0),
            2.0,
            10.0,
            20.0,
        ),
    )
    assert result.value.cell_balances == (
        sheaf.BalanceCell(
            sheaf.CellId(0), False, 20.0, 20.0, 20.0, 0.0, 0.0
        ),
    )
    assert result.value.relative_imbalance == 0.0


def test_balance_canonicalizes_term_and_symmetric_orientation_order() -> None:
    left_to_right = sheaf.BalanceProblem(
        problem_id=sheaf.BalanceProblemId("canonical"),
        domain=_balance_domain(2),
        symmetric_conductances=(
            sheaf.SymmetricConductance(
                sheaf.BalanceInterfaceId("pair"),
                sheaf.CellId(0),
                sheaf.CellId(1),
                4.0,
            ),
        ),
        sources=(
            sheaf.CellSource(
                sheaf.BalanceSourceId("z-source"), sheaf.CellId(1), 2.0
            ),
            sheaf.CellSource(
                sheaf.BalanceSourceId("a-source"), sheaf.CellId(1), 3.0
            ),
        ),
        fixed_boundaries=(
            sheaf.FixedBoundary(
                sheaf.BoundaryId("anchor"), sheaf.CellId(0), 1.0
            ),
        ),
    )
    right_to_left = replace(
        left_to_right,
        symmetric_conductances=(
            sheaf.SymmetricConductance(
                sheaf.BalanceInterfaceId("pair"),
                sheaf.CellId(1),
                sheaf.CellId(0),
                4.0,
            ),
        ),
        sources=tuple(reversed(left_to_right.sources)),
    )
    first = sheaf.solve_balance(left_to_right)
    second = sheaf.solve_balance(right_to_left)
    assert isinstance(first, sheaf.Accepted)
    assert isinstance(second, sheaf.Accepted)
    assert first == second
    assert first.value.field.value_at(sheaf.CellId(1)) == 2.25


def test_balance_rejects_invalid_terms_and_unglued_components_applicatively() -> None:
    invalid_result = sheaf.solve_balance(
        sheaf.BalanceProblem(
            problem_id=sheaf.BalanceProblemId("invalid"),
            domain=_balance_domain(2),
            symmetric_conductances=(
                sheaf.SymmetricConductance(
                    sheaf.BalanceInterfaceId("bad-pair"),
                    sheaf.CellId(0),
                    sheaf.CellId(2),
                    0.0,
                ),
            ),
            fixed_boundaries=(
                sheaf.FixedBoundary(
                    sheaf.BoundaryId("left"), sheaf.CellId(0), 0.0
                ),
                sheaf.FixedBoundary(
                    sheaf.BoundaryId("right"), sheaf.CellId(0), 1.0
                ),
            ),
        )
    )
    assert isinstance(invalid_result, sheaf.Rejected)
    assert invalid_result.obstructions == (
        sheaf.InvalidBalanceCellObstruction(
            sheaf.BalanceProblemId("invalid"),
            sheaf.BalanceTermKind.SYMMETRIC_CONDUCTANCE,
            "bad-pair",
            sheaf.CellId(2),
            2,
        ),
        sheaf.InvalidBalanceCoefficientObstruction(
            sheaf.BalanceProblemId("invalid"),
            sheaf.BalanceTermKind.SYMMETRIC_CONDUCTANCE,
            "bad-pair",
            0.0,
        ),
        sheaf.ConflictingFixedBalanceBoundaryObstruction(
            sheaf.BalanceProblemId("invalid"),
            sheaf.CellId(0),
            sheaf.BoundaryId("left"),
            sheaf.BoundaryId("right"),
        ),
    )
    unanchored_result = sheaf.solve_balance(
        sheaf.BalanceProblem(
            problem_id=sheaf.BalanceProblemId("unanchored"),
            domain=_balance_domain(1),
            sources=(
                sheaf.CellSource(
                    sheaf.BalanceSourceId("heat"), sheaf.CellId(0), 1.0
                ),
            ),
        )
    )
    assert isinstance(unanchored_result, sheaf.Rejected)
    assert unanchored_result.obstructions == (
        sheaf.UnanchoredBalanceComponentObstruction(
            sheaf.BalanceProblemId("unanchored"), 0
        ),
    )


def test_singular_transport_and_invalid_config_are_typed_obstructions() -> None:
    singular = sheaf.solve_balance(
        sheaf.BalanceProblem(
            problem_id=sheaf.BalanceProblemId("singular-advection"),
            domain=_balance_domain(2),
            directed_transports=(
                sheaf.DirectedTransport(
                    sheaf.BalanceInterfaceId("one-way"),
                    sheaf.CellId(0),
                    sheaf.CellId(1),
                    1.0,
                ),
            ),
            ambient_exchanges=(
                sheaf.AmbientExchange(
                    sheaf.AmbientExchangeId("upstream-anchor"),
                    sheaf.CellId(0),
                    1.0,
                    0.0,
                ),
            ),
        )
    )
    assert isinstance(singular, sheaf.Rejected)
    assert singular.obstructions == (
        sheaf.BalanceMatrixObstruction(
            sheaf.BalanceProblemId("singular-advection"),
            sheaf.BalanceMatrixFailure.LINEAR_SOLVE_FAILED,
        ),
    )
    invalid_config = sheaf.solve_balance(
        sheaf.BalanceProblem(
            problem_id=sheaf.BalanceProblemId("invalid-config"),
            domain=_balance_domain(1),
            fixed_boundaries=(
                sheaf.FixedBoundary(
                    sheaf.BoundaryId("anchor"), sheaf.CellId(0), 0.0
                ),
            ),
        ),
        replace(sheaf.SealedSolverConfig(), normalization_floor=0.0),
    )
    assert isinstance(invalid_config, sheaf.Rejected)
    assert invalid_config.obstructions == (
        sheaf.InvalidBalanceSolverConfigObstruction(
            sheaf.BalanceProblemId("invalid-config"),
            sheaf.BalanceSolverParameter.NORMALIZATION_FLOOR,
            0.0,
        ),
    )


def test_nonconvergent_balance_interrogates_topology_conditioning_demand_and_budget() -> None:
    cell_count = 40
    edges = tuple(
        (
            index,
            index + 1,
            1.0e-6 if index % 2 == 0 else 1.0e6,
        )
        for index in range(cell_count - 1)
    )
    result = sheaf.solve_balance(
        sheaf.BalanceProblem(
            problem_id=sheaf.BalanceProblemId("ill-conditioned"),
            domain=sheaf.CellularComplex(
                "ill-conditioned-path",
                tuple(
                    (float(index), 0.0, 0.0)
                    for index in range(cell_count)
                ),
                tuple(
                    sheaf.CellAdjacency(
                        sheaf.CellId(left),
                        sheaf.CellId(right),
                        conductance,
                    )
                    for left, right, conductance in edges
                ),
            ),
            symmetric_conductances=tuple(
                sheaf.SymmetricConductance(
                    sheaf.BalanceInterfaceId(f"edge-{index}"),
                    sheaf.CellId(left),
                    sheaf.CellId(right),
                    conductance,
                )
                for index, (left, right, conductance) in enumerate(edges)
            ),
            fixed_boundaries=(
                sheaf.FixedBoundary(
                    sheaf.BoundaryId("outlet"), sheaf.CellId(0), 1.0
                ),
                sheaf.FixedBoundary(
                    sheaf.BoundaryId("inlet"),
                    sheaf.CellId(cell_count - 1),
                    0.0,
                ),
            ),
        ),
        replace(
            sheaf.SealedSolverConfig(),
            maximum_iteration_factor=1,
        ),
    )

    assert isinstance(result, sheaf.Rejected)
    assert len(result.obstructions) == 1
    obstruction = result.obstructions[0]
    assert isinstance(obstruction, sheaf.NonConvergentBalanceObstruction)
    assert sheaf.render_obstruction(obstruction) == (
        "NonConvergentBalance @ill-conditioned: solver_info=40"
    )
    interrogation = obstruction.interrogation
    assert interrogation is not None
    assert interrogation.topology.free_component_count == 1
    assert len(interrogation.topology.weakest_bridges) == 3
    assert {
        bridge.conductance
        for bridge in interrogation.topology.weakest_bridges
    } == {1.0e-6}
    assert interrogation.conditioning.gershgorin_upper_bound >= 2.0e6
    assert all(
        section.conductance_ratio >= 1.0e12
        for section in interrogation.conditioning.worst_cells
    )
    assert (
        interrogation.demand.maximum_stop_residual
        > interrogation.demand.residual_tolerance
    )
    assert interrogation.demand.worst_cells
    assert interrogation.budget == sheaf.BalanceBudgetWitness(
        iteration_cap=40,
        iterations_used=40,
        relative_tolerance=1.0e-13,
        exhausted=True,
    )


def test_closed_heat_transfer_correlations_evaluate_only_in_valid_regimes() -> None:
    laminar_result = sheaf.evaluate_heat_transfer_correlation(
        _laminar_correlation(), 0.6
    )
    assert isinstance(laminar_result, sheaf.Accepted)
    assert laminar_result.value.nusselt_number == 3.66
    assert (
        abs(laminar_result.value.watt_per_square_metre_kelvin - 219.6)
        <= 1.0e-12
    )
    turbulent = sheaf.DittusBoelterSmoothCircularTube(
        sheaf.HeatTransferCorrelationId("turbulent-wall"),
        sheaf.CircularTubeGeometry(0.01, 0.2),
        reynolds=20_000.0,
        prandtl=7.0,
        thermal_direction=sheaf.TurbulentThermalDirection.FLUID_HEATING,
    )
    turbulent_result = sheaf.evaluate_heat_transfer_correlation(turbulent, 0.6)
    assert isinstance(turbulent_result, sheaf.Accepted)
    assert turbulent_result.value.nusselt_number == (
        0.023 * 20_000.0**0.8 * 7.0**0.4
    )


def test_heat_transfer_regime_failure_refuses_extrapolation_and_fallback() -> None:
    correlation = _laminar_correlation("undeveloped-wall", 0.01)
    result = sheaf.evaluate_heat_transfer_correlation(correlation, 0.6)
    assert isinstance(result, sheaf.Rejected)
    assert result.obstructions == (
        sheaf.UnsupportedHeatTransferRegimeObstruction(
            sheaf.HeatTransferCorrelationId("undeveloped-wall"),
            sheaf.HeatTransferCorrelationKind.FULLY_DEVELOPED_LAMINAR_CIRCULAR_TUBE,
            sheaf.HeatTransferValidityAxis.ENTRANCE_LENGTH_DIAMETERS,
            1.0,
            5.0,
            None,
        ),
    )
    invalid_boundary = sheaf.evaluate_heat_transfer_correlation(
        replace(
            correlation,
            geometry=sheaf.CircularTubeGeometry(0.01, 0.1),
            boundary_condition="invented-wall-law",
        ),
        0.6,
    )
    assert isinstance(invalid_boundary, sheaf.Rejected)
    assert invalid_boundary.obstructions == (
        sheaf.UnsupportedHeatTransferRegimeObstruction(
            sheaf.HeatTransferCorrelationId("undeveloped-wall"),
            sheaf.HeatTransferCorrelationKind.FULLY_DEVELOPED_LAMINAR_CIRCULAR_TUBE,
            sheaf.HeatTransferValidityAxis.BOUNDARY_CONDITION,
            "invented-wall-law",
            None,
            None,
        ),
    )


def _still_air_analytic_law(
    gas: materials.CompressibleGas,
    ambient_temperature_kelvin: float,
    surface_temperature_kelvin: float,
    ambient_pressure_pascal: float,
    gravity: float,
    length: float,
) -> tuple[float, float, float, float]:
    """Independent closed-form derivation: Pr = mu*cp/k; Ra from the film state;
    Nu from the published Churchill-Chu expression; h = Nu*k/L."""
    film_temperature = 0.5 * (
        ambient_temperature_kelvin + surface_temperature_kelvin
    )
    prandtl = (
        gas.dynamic_viscosity_pascal_second
        * gas.specific_heat_joules_per_kilogram_kelvin
        / gas.thermal_conductivity_watts_per_metre_kelvin
    )
    density = ambient_pressure_pascal / (
        gas.specific_gas_constant_joules_per_kilogram_kelvin * film_temperature
    )
    kinematic_viscosity = gas.dynamic_viscosity_pascal_second / density
    thermal_diffusivity = kinematic_viscosity / prandtl
    rayleigh = (
        gravity
        * abs(surface_temperature_kelvin - ambient_temperature_kelvin)
        * length**3
        / (film_temperature * kinematic_viscosity * thermal_diffusivity)
    )
    nusselt = (
        0.825
        + 0.387
        * rayleigh ** (1.0 / 6.0)
        / (1.0 + (0.492 / prandtl) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    ) ** 2.0
    convection = (
        nusselt * gas.thermal_conductivity_watts_per_metre_kelvin / length
    )
    return prandtl, rayleigh, nusselt, convection


def test_churchill_chu_still_air_coefficient_matches_the_analytic_law() -> None:
    gas = _dry_air()
    correlation = sheaf.ChurchillChuIsothermalVerticalPlate(
        sheaf.HeatTransferCorrelationId("ambient-air"),
        characteristic_length_metres=1.0,
        surface_orientation=sheaf.NaturalConvectionSurfaceOrientation.VERTICAL_PLATE,
    )
    result = sheaf.evaluate_natural_convection_correlation(
        correlation,
        gas,
        ambient_temperature_kelvin=293.15,
        surface_temperature_kelvin=303.15,
        ambient_pressure_pascal=101_325.0,
        gravitational_acceleration_metres_per_second_squared=9.80665,
    )
    assert isinstance(result, sheaf.Accepted)
    coefficient = result.value
    prandtl, rayleigh, nusselt, convection = _still_air_analytic_law(
        gas, 293.15, 303.15, 101_325.0, 9.80665, 1.0
    )
    assert coefficient.film_temperature_kelvin == 0.5 * (293.15 + 303.15)
    assert coefficient.prandtl_number == pytest.approx(prandtl)
    assert coefficient.rayleigh_number == pytest.approx(rayleigh)
    assert coefficient.nusselt_number == pytest.approx(nusselt)
    assert coefficient.watt_per_square_metre_kelvin == pytest.approx(convection)
    assert coefficient.validity == sheaf.NaturalConvectionValidityInterval(
        minimum_rayleigh=0.0,
        maximum_rayleigh=1.0e12,
        minimum_rayleigh_is_inclusive=True,
        maximum_rayleigh_is_inclusive=False,
        minimum_prandtl=0.0,
        minimum_prandtl_is_inclusive=False,
    )
    assert coefficient.evidence.source_uri == (
        "https://doi.org/10.1016/0017-9310(75)90243-4"
    )


def test_churchill_chu_includes_the_zero_rayleigh_conduction_limit() -> None:
    result = sheaf.evaluate_natural_convection_correlation(
        sheaf.ChurchillChuIsothermalVerticalPlate(
            sheaf.HeatTransferCorrelationId("conduction-limit"),
            1.0,
            sheaf.NaturalConvectionSurfaceOrientation.VERTICAL_PLATE,
        ),
        _dry_air(),
        ambient_temperature_kelvin=298.15,
        surface_temperature_kelvin=298.15,
        ambient_pressure_pascal=101_325.0,
        gravitational_acceleration_metres_per_second_squared=9.80665,
    )
    assert isinstance(result, sheaf.Accepted)
    assert result.value.rayleigh_number == 0.0
    assert result.value.nusselt_number == 0.825**2.0
    assert result.value.watt_per_square_metre_kelvin == pytest.approx(
        0.825**2.0 * 26.06e-3
    )


def test_natural_convection_rejects_invalid_geometry_and_film_state() -> None:
    result = sheaf.evaluate_natural_convection_correlation(
        sheaf.ChurchillChuIsothermalVerticalPlate(
            sheaf.HeatTransferCorrelationId("invalid-ambient"),
            0.0,
            "horizontal_plate",
        ),
        _dry_air(),
        ambient_temperature_kelvin=293.15,
        surface_temperature_kelvin=350.0,
        ambient_pressure_pascal=120_000.0,
        gravitational_acceleration_metres_per_second_squared=-1.0,
    )
    assert isinstance(result, sheaf.Rejected)
    assert tuple(
        obstruction.axis for obstruction in result.obstructions
    ) == (
        sheaf.HeatTransferValidityAxis.CHARACTERISTIC_LENGTH,
        sheaf.HeatTransferValidityAxis.SURFACE_ORIENTATION,
        sheaf.HeatTransferValidityAxis.FILM_TEMPERATURE,
        sheaf.HeatTransferValidityAxis.AMBIENT_PRESSURE,
        sheaf.HeatTransferValidityAxis.GRAVITATIONAL_ACCELERATION,
    )


def test_natural_convection_refuses_rayleigh_extrapolation() -> None:
    gas = _dry_air()
    result = sheaf.evaluate_natural_convection_correlation(
        sheaf.ChurchillChuIsothermalVerticalPlate(
            sheaf.HeatTransferCorrelationId("overscale-ambient"),
            100.0,
            sheaf.NaturalConvectionSurfaceOrientation.VERTICAL_PLATE,
        ),
        gas,
        ambient_temperature_kelvin=293.15,
        surface_temperature_kelvin=303.15,
        ambient_pressure_pascal=101_325.0,
        gravitational_acceleration_metres_per_second_squared=9.80665,
    )
    _, overscale_rayleigh, _, _ = _still_air_analytic_law(
        gas, 293.15, 303.15, 101_325.0, 9.80665, 100.0
    )
    assert overscale_rayleigh >= 1.0e12
    assert isinstance(result, sheaf.Rejected)
    assert result.obstructions == (
        sheaf.UnsupportedHeatTransferRegimeObstruction(
            sheaf.HeatTransferCorrelationId("overscale-ambient"),
            sheaf.HeatTransferCorrelationKind.CHURCHILL_CHU_ISOTHERMAL_VERTICAL_PLATE,
            sheaf.HeatTransferValidityAxis.RAYLEIGH,
            pytest.approx(overscale_rayleigh),
            0.0,
            1.0e12,
        ),
    )


def test_thermal_interpreter_solves_one_mixed_solid_coolant_system() -> None:
    heat_transfer_coefficient = 219.6
    problem = sheaf.ThermalBalanceProblem(
        problem_id=sheaf.BalanceProblemId("mixed-thermal"),
        domain=_balance_domain(4),
        solid_conduction_links=(
            sheaf.SolidConductionLink(
                sheaf.BalanceInterfaceId("solid-link"),
                sheaf.CellId(0),
                sheaf.CellId(1),
                thermal_conductivity_watt_per_metre_kelvin=10.0,
                exchange_area_square_metres=1.0,
                path_length_metres=1.0,
            ),
        ),
        coolant_advection_links=(
            sheaf.CoolantAdvectionLink(
                sheaf.BalanceInterfaceId("coolant-in"),
                sheaf.CellId(2),
                sheaf.CellId(3),
                mass_flow_kilograms_per_second=0.02,
                specific_heat_joule_per_kilogram_kelvin=1000.0,
            ),
            sheaf.CoolantAdvectionLink(
                sheaf.BalanceInterfaceId("coolant-return"),
                sheaf.CellId(3),
                sheaf.CellId(2),
                mass_flow_kilograms_per_second=0.02,
                specific_heat_joule_per_kilogram_kelvin=1000.0,
            ),
        ),
        wall_heat_exchanges=(
            sheaf.WallHeatExchange(
                sheaf.BalanceInterfaceId("wall-link"),
                sheaf.CellId(1),
                sheaf.CellId(3),
                exchange_area_square_metres=10.0 / heat_transfer_coefficient,
                correlation=_laminar_correlation(),
                fluid_thermal_conductivity_watt_per_metre_kelvin=0.6,
            ),
        ),
        sources=(
            sheaf.ThermalSource(
                sheaf.BalanceSourceId("authored-heat"), sheaf.CellId(0), 100.0
            ),
        ),
        fixed_temperatures=(
            sheaf.FixedTemperature(
                sheaf.BoundaryId("coolant-inlet"), sheaf.CellId(2), 300.0
            ),
        ),
        ambient_heat_exchanges=(
            sheaf.AmbientHeatExchange(
                sheaf.AmbientExchangeId("ambient"),
                sheaf.CellId(0),
                heat_transfer_coefficient_watt_per_square_metre_kelvin=2.0,
                exchange_area_square_metres=1.0,
                ambient_temperature_kelvin=300.0,
            ),
        ),
    )
    result = sheaf.solve_thermal_balance(problem)
    assert isinstance(result, sheaf.Accepted)
    solution = result.value
    expected_kelvin = (316.6666666666667, 310.0, 300.0, 303.3333333333333)
    assert all(
        abs(value.value - expected) <= 1.0e-12
        for value, expected in zip(solution.balance.field.values, expected_kelvin)
    )
    assert len(solution.wall_heat_fluxes) == 1
    wall_flux = solution.wall_heat_fluxes[0]
    assert wall_flux.reynolds_number == 100.0
    assert wall_flux.prandtl_number == 1.0
    assert wall_flux.hydraulic_diameter_metres == 0.01
    assert wall_flux.upstream_development_length_metres == 0.1
    assert (
        abs(wall_flux.solid_to_coolant_watts - 66.6666666666667)
        <= 1.0e-12
    )
    assert solution.energy.authored_source_watts == 100.0
    assert (
        abs(solution.energy.fixed_boundary_supply_watts + 66.6666666666667)
        <= 1.0e-12
    )
    assert abs(solution.energy.ambient_outflow_watts - 33.3333333333334) <= 1.0e-12
    assert solution.energy.maximum_normalized_residual <= 1.0e-14
    assert solution.energy.relative_energy_imbalance <= 1.0e-14


def test_thermal_inputs_and_wall_regimes_fail_closed() -> None:
    invalid_parameter = sheaf.solve_thermal_balance(
        sheaf.ThermalBalanceProblem(
            problem_id=sheaf.BalanceProblemId("invalid-thermal"),
            domain=_balance_domain(2),
            solid_conduction_links=(
                sheaf.SolidConductionLink(
                    sheaf.BalanceInterfaceId("solid"),
                    sheaf.CellId(0),
                    sheaf.CellId(1),
                    1.0,
                    1.0,
                    0.0,
                ),
            ),
        )
    )
    assert isinstance(invalid_parameter, sheaf.Rejected)
    assert invalid_parameter.obstructions == (
        sheaf.InvalidThermalParameterObstruction(
            sheaf.BalanceProblemId("invalid-thermal"),
            "solid",
            sheaf.ThermalParameter.SOLID_PATH_LENGTH,
            0.0,
        ),
    )
    invalid_regime = sheaf.solve_thermal_balance(
        sheaf.ThermalBalanceProblem(
            problem_id=sheaf.BalanceProblemId("invalid-wall"),
            domain=_balance_domain(2),
            wall_heat_exchanges=(
                sheaf.WallHeatExchange(
                    sheaf.BalanceInterfaceId("wall"),
                    sheaf.CellId(0),
                    sheaf.CellId(1),
                    1.0,
                    _laminar_correlation("short-wall", 0.01),
                    0.6,
                ),
            ),
        )
    )
    assert isinstance(invalid_regime, sheaf.Rejected)
    assert isinstance(
        invalid_regime.obstructions[0],
        sheaf.UnsupportedHeatTransferRegimeObstruction,
    )


def test_weaker_scalar_and_projection_surfaces_are_deleted() -> None:
    deleted_names = (
        "BoundaryCondition",
        "ScalarField",
        "FieldSolution",
        "ElementalFlow",
        "solve_field",
        "derive_elemental_flow",
        "load_problem",
        "decode_field",
        "render_elemental_flow_json",
        "main",
    )
    assert not tuple(name for name in deleted_names if hasattr(sheaf, name))
