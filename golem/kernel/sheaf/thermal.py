"""The reference interpreter: physical thermal links (conduction, advection, wall
convection, sources, boundaries, ambient exchange) lowered into one balance problem."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum

from golem.kernel.sheaf.arrays import _positive_finite
from golem.kernel.sheaf.correlation import (
    HeatTransferCoefficient,
    HeatTransferCorrelation,
    HeatTransferValidityInterval,
)
from golem.kernel.sheaf.interpreter import (
    Lowering,
    LoweredProblem,
    LoweringRejected,
    interpret_balance,
)
from golem.kernel.sheaf.nusselt import evaluate_heat_transfer_correlation
from golem.kernel.sheaf.problem import (
    AmbientExchange,
    BalanceProblem,
    CellSource,
    CellularComplex,
    DirectedTransport,
    FixedBoundary,
    SealedSolverConfig,
    SymmetricConductance,
)
from golem.kernel.sheaf.result import (
    Accepted,
    InvalidThermalParameterObstruction,
    Obstruction,
    Rejected,
    Result,
)
from golem.kernel.sheaf.section import BalanceSolution
from golem.kernel.sheaf.vocabulary import (
    AmbientExchangeId,
    BalanceInterfaceId,
    BalanceProblemId,
    BalanceSourceId,
    BoundaryId,
    CellId,
    HeatTransferCorrelationId,
    HeatTransferCorrelationKind,
    ThermalParameter,
)


@dataclass(frozen=True)
class SolidConductionLink:
    interface_id: BalanceInterfaceId
    left_cell: CellId
    right_cell: CellId
    thermal_conductivity_watt_per_metre_kelvin: float
    exchange_area_square_metres: float
    path_length_metres: float


@dataclass(frozen=True)
class CoolantAdvectionLink:
    interface_id: BalanceInterfaceId
    upstream_cell: CellId
    downstream_cell: CellId
    mass_flow_kilograms_per_second: float
    specific_heat_joule_per_kilogram_kelvin: float


@dataclass(frozen=True)
class WallHeatExchange:
    interface_id: BalanceInterfaceId
    solid_cell: CellId
    coolant_cell: CellId
    exchange_area_square_metres: float
    correlation: HeatTransferCorrelation
    fluid_thermal_conductivity_watt_per_metre_kelvin: float


@dataclass(frozen=True)
class ThermalSource:
    source_id: BalanceSourceId
    cell_id: CellId
    watts: float


@dataclass(frozen=True)
class FixedTemperature:
    boundary_id: BoundaryId
    cell_id: CellId
    kelvin: float


@dataclass(frozen=True)
class AmbientHeatExchange:
    exchange_id: AmbientExchangeId
    cell_id: CellId
    heat_transfer_coefficient_watt_per_square_metre_kelvin: float
    exchange_area_square_metres: float
    ambient_temperature_kelvin: float


@dataclass(frozen=True)
class ThermalBalanceProblem:
    problem_id: BalanceProblemId
    domain: CellularComplex
    solid_conduction_links: tuple[SolidConductionLink, ...] = ()
    coolant_advection_links: tuple[CoolantAdvectionLink, ...] = ()
    wall_heat_exchanges: tuple[WallHeatExchange, ...] = ()
    sources: tuple[ThermalSource, ...] = ()
    fixed_temperatures: tuple[FixedTemperature, ...] = ()
    ambient_heat_exchanges: tuple[AmbientHeatExchange, ...] = ()


@dataclass(frozen=True)
class WallHeatFlux:
    interface_id: BalanceInterfaceId
    solid_cell: CellId
    coolant_cell: CellId
    correlation_id: HeatTransferCorrelationId
    correlation_kind: HeatTransferCorrelationKind
    nusselt_number: float
    heat_transfer_coefficient_watt_per_square_metre_kelvin: float
    validity: HeatTransferValidityInterval
    reynolds_number: float
    prandtl_number: float
    hydraulic_diameter_metres: float
    upstream_development_length_metres: float
    solid_to_coolant_watts: float


@dataclass(frozen=True)
class ThermalEnergyReceipt:
    authored_source_watts: float
    fixed_boundary_supply_watts: float
    ambient_outflow_watts: float
    maximum_normalized_residual: float
    relative_energy_imbalance: float


@dataclass(frozen=True)
class ThermalSolution:
    balance: BalanceSolution
    wall_heat_fluxes: tuple[WallHeatFlux, ...]
    energy: ThermalEnergyReceipt


@dataclass(frozen=True)
class _ThermalCarrier:
    problem: ThermalBalanceProblem
    resolved_walls: tuple[
        tuple[WallHeatExchange, HeatTransferCoefficient], ...
    ]


@dataclass(frozen=True)
class ThermalInterpreter:
    """Interpret physical thermal links as one balance problem."""

    config: SealedSolverConfig = SealedSolverConfig()

    def lower(
        self, problem: ThermalBalanceProblem
    ) -> Lowering[_ThermalCarrier, Result[ThermalSolution]]:
        parameter_obstructions = _thermal_parameter_obstructions(problem)
        wall_coefficients = _resolve_wall_coefficients(problem)
        correlation_obstructions = _wall_correlation_obstructions(
            wall_coefficients
        )
        if parameter_obstructions or correlation_obstructions:
            return LoweringRejected(
                Rejected((*parameter_obstructions, *correlation_obstructions))
            )
        resolved_walls = tuple(
            (wall, result.value)
            for wall, result in wall_coefficients
            if isinstance(result, Accepted)
        )
        return LoweredProblem(
            _lowered_balance_problem(problem, resolved_walls),
            self.config,
            _ThermalCarrier(problem, resolved_walls),
        )

    def lift(
        self,
        carrier: _ThermalCarrier,
        balance: Result[BalanceSolution],
    ) -> Result[ThermalSolution]:
        if isinstance(balance, Rejected):
            return balance
        solution = balance.value
        return Accepted(
            ThermalSolution(
                solution,
                _wall_heat_fluxes(carrier, solution),
                _thermal_energy_receipt(carrier, solution),
            )
        )


def solve_thermal_balance(
    problem: ThermalBalanceProblem,
    config: SealedSolverConfig = SealedSolverConfig(),
) -> Result[ThermalSolution]:
    """Interpret physical thermal links as one monolithic balance problem."""
    return interpret_balance(ThermalInterpreter(config), problem)


def _resolve_wall_coefficients(
    problem: ThermalBalanceProblem,
) -> tuple[tuple[WallHeatExchange, Result[HeatTransferCoefficient]], ...]:
    return tuple(
        (
            wall,
            evaluate_heat_transfer_correlation(
                wall.correlation,
                wall.fluid_thermal_conductivity_watt_per_metre_kelvin,
            ),
        )
        for wall in problem.wall_heat_exchanges
    )


def _wall_correlation_obstructions(
    wall_coefficients: tuple[
        tuple[WallHeatExchange, Result[HeatTransferCoefficient]], ...
    ],
) -> tuple[Obstruction, ...]:
    return tuple(
        obstruction
        for _wall, result in wall_coefficients
        if isinstance(result, Rejected)
        for obstruction in result.obstructions
    )


def _solid_conduction_conductances(
    problem: ThermalBalanceProblem,
) -> tuple[SymmetricConductance, ...]:
    return tuple(
        SymmetricConductance(
            link.interface_id,
            link.left_cell,
            link.right_cell,
            link.thermal_conductivity_watt_per_metre_kelvin
            * link.exchange_area_square_metres
            / link.path_length_metres,
        )
        for link in problem.solid_conduction_links
    )


def _wall_conductances(
    resolved_walls: tuple[tuple[WallHeatExchange, HeatTransferCoefficient], ...],
) -> tuple[SymmetricConductance, ...]:
    return tuple(
        SymmetricConductance(
            wall.interface_id,
            wall.solid_cell,
            wall.coolant_cell,
            coefficient.watt_per_square_metre_kelvin
            * wall.exchange_area_square_metres,
        )
        for wall, coefficient in resolved_walls
    )


def _coolant_advection_transports(
    problem: ThermalBalanceProblem,
) -> tuple[DirectedTransport, ...]:
    return tuple(
        DirectedTransport(
            link.interface_id,
            link.upstream_cell,
            link.downstream_cell,
            link.mass_flow_kilograms_per_second
            * link.specific_heat_joule_per_kilogram_kelvin,
        )
        for link in problem.coolant_advection_links
    )


def _thermal_cell_sources(
    problem: ThermalBalanceProblem,
) -> tuple[CellSource, ...]:
    return tuple(
        CellSource(source.source_id, source.cell_id, source.watts)
        for source in problem.sources
    )


def _fixed_temperature_boundaries(
    problem: ThermalBalanceProblem,
) -> tuple[FixedBoundary, ...]:
    return tuple(
        FixedBoundary(boundary.boundary_id, boundary.cell_id, boundary.kelvin)
        for boundary in problem.fixed_temperatures
    )


def _ambient_heat_exchange_terms(
    problem: ThermalBalanceProblem,
) -> tuple[AmbientExchange, ...]:
    return tuple(
        AmbientExchange(
            exchange.exchange_id,
            exchange.cell_id,
            exchange.heat_transfer_coefficient_watt_per_square_metre_kelvin
            * exchange.exchange_area_square_metres,
            exchange.ambient_temperature_kelvin,
        )
        for exchange in problem.ambient_heat_exchanges
    )


def _lowered_balance_problem(
    problem: ThermalBalanceProblem,
    resolved_walls: tuple[tuple[WallHeatExchange, HeatTransferCoefficient], ...],
) -> BalanceProblem:
    return BalanceProblem(
        problem.problem_id,
        problem.domain,
        symmetric_conductances=(
            *_solid_conduction_conductances(problem),
            *_wall_conductances(resolved_walls),
        ),
        directed_transports=_coolant_advection_transports(problem),
        sources=_thermal_cell_sources(problem),
        fixed_boundaries=_fixed_temperature_boundaries(problem),
        ambient_exchanges=_ambient_heat_exchange_terms(problem),
    )


def _wall_heat_fluxes(
    carrier: _ThermalCarrier, solution: BalanceSolution
) -> tuple[WallHeatFlux, ...]:
    return tuple(
        WallHeatFlux(
            wall.interface_id,
            wall.solid_cell,
            wall.coolant_cell,
            coefficient.correlation_id,
            coefficient.correlation_kind,
            coefficient.nusselt_number,
            coefficient.watt_per_square_metre_kelvin,
            coefficient.validity,
            wall.correlation.reynolds,
            wall.correlation.prandtl,
            wall.correlation.geometry.hydraulic_diameter_metres,
            wall.correlation.geometry.upstream_development_length_metres,
            coefficient.watt_per_square_metre_kelvin
            * wall.exchange_area_square_metres
            * (
                solution.field.value_at(wall.solid_cell)
                - solution.field.value_at(wall.coolant_cell)
            ),
        )
        for wall, coefficient in carrier.resolved_walls
    )


def _thermal_energy_receipt(
    carrier: _ThermalCarrier, solution: BalanceSolution
) -> ThermalEnergyReceipt:
    return ThermalEnergyReceipt(
        fsum(source.watts for source in carrier.problem.sources),
        fsum(
            supply.boundary_to_domain_flux
            for supply in solution.fixed_boundary_supplies
        ),
        fsum(
            flux.cell_to_ambient_flux for flux in solution.ambient_fluxes
        ),
        solution.maximum_normalized_residual,
        solution.relative_imbalance,
    )


def _thermal_parameter_obstructions(
    problem: ThermalBalanceProblem,
) -> tuple[Obstruction, ...]:
    checks = (
        *(
            (
                str(link.interface_id),
                ThermalParameter.SOLID_CONDUCTIVITY,
                link.thermal_conductivity_watt_per_metre_kelvin,
            )
            for link in problem.solid_conduction_links
        ),
        *(
            (
                str(link.interface_id),
                ThermalParameter.SOLID_EXCHANGE_AREA,
                link.exchange_area_square_metres,
            )
            for link in problem.solid_conduction_links
        ),
        *(
            (
                str(link.interface_id),
                ThermalParameter.SOLID_PATH_LENGTH,
                link.path_length_metres,
            )
            for link in problem.solid_conduction_links
        ),
        *(
            (
                str(link.interface_id),
                ThermalParameter.COOLANT_MASS_FLOW,
                link.mass_flow_kilograms_per_second,
            )
            for link in problem.coolant_advection_links
        ),
        *(
            (
                str(link.interface_id),
                ThermalParameter.COOLANT_SPECIFIC_HEAT,
                link.specific_heat_joule_per_kilogram_kelvin,
            )
            for link in problem.coolant_advection_links
        ),
        *(
            (
                str(exchange.interface_id),
                ThermalParameter.WALL_EXCHANGE_AREA,
                exchange.exchange_area_square_metres,
            )
            for exchange in problem.wall_heat_exchanges
        ),
        *(
            (
                str(exchange.exchange_id),
                ThermalParameter.AMBIENT_HEAT_TRANSFER_COEFFICIENT,
                exchange.heat_transfer_coefficient_watt_per_square_metre_kelvin,
            )
            for exchange in problem.ambient_heat_exchanges
        ),
        *(
            (
                str(exchange.exchange_id),
                ThermalParameter.AMBIENT_EXCHANGE_AREA,
                exchange.exchange_area_square_metres,
            )
            for exchange in problem.ambient_heat_exchanges
        ),
    )
    return tuple(
        InvalidThermalParameterObstruction(
            problem.problem_id, term_id, parameter, value
        )
        for term_id, parameter, value in checks
        if not _positive_finite(value)
    )
