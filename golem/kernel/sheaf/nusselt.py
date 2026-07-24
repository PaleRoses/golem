"""Nusselt-number algebra: evaluate a closed heat-transfer correlation strictly
inside its declared regime, emitting typed regime obstructions otherwise."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import assert_never

from golem.materials import CompressibleGas, GasEquationOfState
from golem.kernel.sheaf.arrays import _nonnegative_finite, _positive_finite
from golem.kernel.sheaf.correlation import (
    ChurchillChuIsothermalVerticalPlate,
    CircularTubeGeometry,
    DittusBoelterSmoothCircularTube,
    FullyDevelopedLaminarCircularTube,
    HeatTransferCoefficient,
    HeatTransferCorrelation,
    LaminarCircularTubeConservativeLowerBound,
    NaturalConvectionHeatTransferCoefficient,
)
from golem.kernel.sheaf.result import (
    Accepted,
    Obstruction,
    Rejected,
    Result,
    UnsupportedHeatTransferRegimeObstruction,
)
from golem.kernel.sheaf.vocabulary import (
    HeatTransferCorrelationId,
    HeatTransferCorrelationKind,
    HeatTransferValidityAxis,
    LaminarThermalBoundaryCondition,
    NaturalConvectionSurfaceOrientation,
    TurbulentThermalDirection,
)


type _RegimeCheck = tuple[
    HeatTransferValidityAxis, float | str, float | None, float | None, bool
]


@dataclass(frozen=True)
class _NaturalConvectionFilmState:
    film_temperature_kelvin: float
    density: float
    volumetric_thermal_expansion: float
    kinematic_viscosity: float
    thermal_diffusivity: float
    prandtl_number: float
    rayleigh_number: float


def evaluate_heat_transfer_correlation(
    correlation: HeatTransferCorrelation,
    fluid_thermal_conductivity_watt_per_metre_kelvin: float,
) -> Result[HeatTransferCoefficient]:
    """Evaluate one closed correlation only inside its declared regime."""
    obstructions = _heat_transfer_correlation_obstructions(
        correlation, fluid_thermal_conductivity_watt_per_metre_kelvin
    )
    if obstructions:
        return Rejected(obstructions)
    nusselt_number = _circular_tube_nusselt_number(correlation)
    return Accepted(
        HeatTransferCoefficient(
            correlation.correlation_id,
            correlation.correlation_kind,
            nusselt_number,
            nusselt_number
            * fluid_thermal_conductivity_watt_per_metre_kelvin
            / correlation.geometry.hydraulic_diameter_metres,
            correlation.validity,
        )
    )


def _circular_tube_nusselt_number(
    correlation: HeatTransferCorrelation,
) -> float:
    match correlation:
        case (
            FullyDevelopedLaminarCircularTube(boundary_condition=condition)
            | LaminarCircularTubeConservativeLowerBound(
                boundary_condition=condition
            )
        ):
            match condition:
                case LaminarThermalBoundaryCondition.CONSTANT_WALL_TEMPERATURE:
                    return 3.66
                case LaminarThermalBoundaryCondition.CONSTANT_HEAT_FLUX:
                    return 4.36
                case _ as unreachable_laminar_condition:
                    assert_never(unreachable_laminar_condition)
        case DittusBoelterSmoothCircularTube(thermal_direction=direction):
            match direction:
                case TurbulentThermalDirection.FLUID_HEATING:
                    prandtl_exponent = 0.4
                case TurbulentThermalDirection.FLUID_COOLING:
                    prandtl_exponent = 0.3
                case _ as unreachable_turbulent_direction:
                    assert_never(unreachable_turbulent_direction)
            return (
                0.023
                * correlation.reynolds**0.8
                * correlation.prandtl**prandtl_exponent
            )
        case _ as unreachable_correlation:
            assert_never(unreachable_correlation)


def evaluate_natural_convection_correlation(
    correlation: ChurchillChuIsothermalVerticalPlate,
    gas: CompressibleGas,
    ambient_temperature_kelvin: float,
    surface_temperature_kelvin: float,
    ambient_pressure_pascal: float,
    gravitational_acceleration_metres_per_second_squared: float,
) -> Result[NaturalConvectionHeatTransferCoefficient]:
    """Evaluate the closed still-gas vertical-plate law at one film state."""
    local_obstructions = _natural_convection_input_obstructions(
        correlation,
        gas,
        ambient_temperature_kelvin,
        surface_temperature_kelvin,
        ambient_pressure_pascal,
        gravitational_acceleration_metres_per_second_squared,
    )
    if local_obstructions:
        return Rejected(local_obstructions)
    film = _natural_convection_film_state(
        correlation,
        gas,
        ambient_temperature_kelvin,
        surface_temperature_kelvin,
        ambient_pressure_pascal,
        gravitational_acceleration_metres_per_second_squared,
    )
    dimensionless_obstructions = _natural_convection_dimensionless_obstructions(
        correlation, film
    )
    if dimensionless_obstructions:
        return Rejected(dimensionless_obstructions)
    nusselt_number = _churchill_chu_nusselt_number(
        film.rayleigh_number, film.prandtl_number
    )
    return Accepted(
        NaturalConvectionHeatTransferCoefficient(
            correlation.correlation_id,
            correlation.correlation_kind,
            gas.material_id,
            correlation.surface_orientation,
            correlation.characteristic_length_metres,
            film.film_temperature_kelvin,
            film.density,
            film.volumetric_thermal_expansion,
            film.kinematic_viscosity,
            film.thermal_diffusivity,
            film.prandtl_number,
            film.rayleigh_number,
            nusselt_number,
            nusselt_number
            * gas.thermal_conductivity_watts_per_metre_kelvin
            / correlation.characteristic_length_metres,
            correlation.validity,
            correlation.evidence,
        )
    )


def _natural_convection_film_state(
    correlation: ChurchillChuIsothermalVerticalPlate,
    gas: CompressibleGas,
    ambient_temperature_kelvin: float,
    surface_temperature_kelvin: float,
    ambient_pressure_pascal: float,
    gravitational_acceleration_metres_per_second_squared: float,
) -> _NaturalConvectionFilmState:
    film_temperature_kelvin = 0.5 * (
        ambient_temperature_kelvin + surface_temperature_kelvin
    )
    match gas.equation_of_state:
        case GasEquationOfState.IDEAL_GAS:
            density = ambient_pressure_pascal / (
                gas.specific_gas_constant_joules_per_kilogram_kelvin
                * film_temperature_kelvin
            )
            volumetric_thermal_expansion = 1.0 / film_temperature_kelvin
        case _ as unreachable_equation_of_state:
            assert_never(unreachable_equation_of_state)
    kinematic_viscosity = gas.dynamic_viscosity_pascal_second / density
    thermal_diffusivity = (
        gas.thermal_conductivity_watts_per_metre_kelvin
        / (density * gas.specific_heat_joules_per_kilogram_kelvin)
    )
    prandtl_number = kinematic_viscosity / thermal_diffusivity
    rayleigh_number = (
        gravitational_acceleration_metres_per_second_squared
        * volumetric_thermal_expansion
        * abs(surface_temperature_kelvin - ambient_temperature_kelvin)
        * correlation.characteristic_length_metres
        * correlation.characteristic_length_metres
        * correlation.characteristic_length_metres
        / (kinematic_viscosity * thermal_diffusivity)
    )
    return _NaturalConvectionFilmState(
        film_temperature_kelvin,
        density,
        volumetric_thermal_expansion,
        kinematic_viscosity,
        thermal_diffusivity,
        prandtl_number,
        rayleigh_number,
    )


def _natural_convection_dimensionless_obstructions(
    correlation: ChurchillChuIsothermalVerticalPlate,
    film: _NaturalConvectionFilmState,
) -> tuple[Obstruction, ...]:
    validity = correlation.validity
    return (
        (
            UnsupportedHeatTransferRegimeObstruction(
                correlation.correlation_id,
                correlation.correlation_kind,
                HeatTransferValidityAxis.PRANDTL,
                film.prandtl_number,
                validity.minimum_prandtl,
                None,
            ),
        )
        if not isfinite(film.prandtl_number)
        or film.prandtl_number <= validity.minimum_prandtl
        else ()
    ) + (
        (
            UnsupportedHeatTransferRegimeObstruction(
                correlation.correlation_id,
                correlation.correlation_kind,
                HeatTransferValidityAxis.RAYLEIGH,
                film.rayleigh_number,
                validity.minimum_rayleigh,
                validity.maximum_rayleigh,
            ),
        )
        if not isfinite(film.rayleigh_number)
        or not validity.minimum_rayleigh <= film.rayleigh_number
        < validity.maximum_rayleigh
        else ()
    )


def _churchill_chu_nusselt_number(
    rayleigh_number: float, prandtl_number: float
) -> float:
    return (
        0.825
        + 0.387
        * rayleigh_number ** (1.0 / 6.0)
        / (1.0 + (0.492 / prandtl_number) ** (9.0 / 16.0))
        ** (8.0 / 27.0)
    ) ** 2.0


def _regime_obstructions(
    correlation_id: HeatTransferCorrelationId,
    correlation_kind: HeatTransferCorrelationKind,
    checks: tuple[_RegimeCheck, ...],
) -> tuple[Obstruction, ...]:
    return tuple(
        UnsupportedHeatTransferRegimeObstruction(
            correlation_id,
            correlation_kind,
            axis,
            actual,
            minimum,
            maximum,
        )
        for axis, actual, minimum, maximum, is_valid in checks
        if not is_valid
    )


def _natural_convection_correlation_checks(
    correlation: ChurchillChuIsothermalVerticalPlate,
) -> tuple[_RegimeCheck, ...]:
    return (
        (
            HeatTransferValidityAxis.CORRELATION_ID,
            str(correlation.correlation_id),
            None,
            None,
            bool(correlation.correlation_id),
        ),
        (
            HeatTransferValidityAxis.CHARACTERISTIC_LENGTH,
            correlation.characteristic_length_metres,
            0.0,
            None,
            _positive_finite(correlation.characteristic_length_metres),
        ),
        (
            HeatTransferValidityAxis.SURFACE_ORIENTATION,
            str(correlation.surface_orientation),
            None,
            None,
            correlation.surface_orientation
            is NaturalConvectionSurfaceOrientation.VERTICAL_PLATE,
        ),
    )


def _natural_convection_temperature_checks(
    gas: CompressibleGas,
    ambient_temperature_kelvin: float,
    surface_temperature_kelvin: float,
) -> tuple[_RegimeCheck, ...]:
    valid_ambient_temperature = _positive_finite(ambient_temperature_kelvin)
    valid_surface_temperature = _positive_finite(surface_temperature_kelvin)
    film_temperature_kelvin = (
        0.5 * (ambient_temperature_kelvin + surface_temperature_kelvin)
        if valid_ambient_temperature and valid_surface_temperature
        else float("nan")
    )
    return (
        (
            HeatTransferValidityAxis.AMBIENT_TEMPERATURE,
            ambient_temperature_kelvin,
            0.0,
            None,
            valid_ambient_temperature,
        ),
        (
            HeatTransferValidityAxis.SURFACE_TEMPERATURE,
            surface_temperature_kelvin,
            0.0,
            None,
            valid_surface_temperature,
        ),
        (
            HeatTransferValidityAxis.FILM_TEMPERATURE,
            film_temperature_kelvin,
            gas.operating_temperature.minimum_kelvin,
            gas.operating_temperature.maximum_kelvin,
            isfinite(film_temperature_kelvin)
            and gas.operating_temperature.contains(film_temperature_kelvin),
        ),
    )


def _natural_convection_environment_checks(
    gas: CompressibleGas,
    ambient_pressure_pascal: float,
    gravitational_acceleration_metres_per_second_squared: float,
) -> tuple[_RegimeCheck, ...]:
    return (
        (
            HeatTransferValidityAxis.AMBIENT_PRESSURE,
            ambient_pressure_pascal,
            gas.allowable_pressure.minimum_pascal,
            gas.allowable_pressure.maximum_pascal,
            isfinite(ambient_pressure_pascal)
            and gas.allowable_pressure.contains(ambient_pressure_pascal),
        ),
        (
            HeatTransferValidityAxis.GRAVITATIONAL_ACCELERATION,
            gravitational_acceleration_metres_per_second_squared,
            0.0,
            None,
            _nonnegative_finite(
                gravitational_acceleration_metres_per_second_squared
            ),
        ),
    )


def _natural_convection_gas_property_checks(
    gas: CompressibleGas,
) -> tuple[_RegimeCheck, ...]:
    return (
        (
            HeatTransferValidityAxis.GAS_EQUATION_OF_STATE,
            str(gas.equation_of_state),
            None,
            None,
            gas.equation_of_state is GasEquationOfState.IDEAL_GAS,
        ),
        (
            HeatTransferValidityAxis.GAS_CONSTANT,
            gas.specific_gas_constant_joules_per_kilogram_kelvin,
            0.0,
            None,
            _positive_finite(gas.specific_gas_constant_joules_per_kilogram_kelvin),
        ),
        (
            HeatTransferValidityAxis.DYNAMIC_VISCOSITY,
            gas.dynamic_viscosity_pascal_second,
            0.0,
            None,
            _positive_finite(gas.dynamic_viscosity_pascal_second),
        ),
        (
            HeatTransferValidityAxis.SPECIFIC_HEAT,
            gas.specific_heat_joules_per_kilogram_kelvin,
            0.0,
            None,
            _positive_finite(gas.specific_heat_joules_per_kilogram_kelvin),
        ),
        (
            HeatTransferValidityAxis.THERMAL_CONDUCTIVITY,
            gas.thermal_conductivity_watts_per_metre_kelvin,
            0.0,
            None,
            _positive_finite(gas.thermal_conductivity_watts_per_metre_kelvin),
        ),
    )


def _natural_convection_input_obstructions(
    correlation: ChurchillChuIsothermalVerticalPlate,
    gas: CompressibleGas,
    ambient_temperature_kelvin: float,
    surface_temperature_kelvin: float,
    ambient_pressure_pascal: float,
    gravitational_acceleration_metres_per_second_squared: float,
) -> tuple[Obstruction, ...]:
    checks = (
        *_natural_convection_correlation_checks(correlation),
        *_natural_convection_temperature_checks(
            gas, ambient_temperature_kelvin, surface_temperature_kelvin
        ),
        *_natural_convection_environment_checks(
            gas,
            ambient_pressure_pascal,
            gravitational_acceleration_metres_per_second_squared,
        ),
        *_natural_convection_gas_property_checks(gas),
    )
    return _regime_obstructions(
        correlation.correlation_id, correlation.correlation_kind, checks
    )


def _entrance_length_diameters(
    geometry: CircularTubeGeometry,
    valid_diameter: bool,
    valid_development_length: bool,
) -> float:
    return (
        geometry.upstream_development_length_metres
        / geometry.hydraulic_diameter_metres
        if valid_diameter and valid_development_length
        else float("nan")
    )


def _reynolds_in_range(
    reynolds: float,
    minimum_reynolds: float,
    maximum_reynolds: float,
    maximum_reynolds_is_exclusive: bool,
) -> bool:
    return (
        isfinite(reynolds)
        and reynolds > minimum_reynolds
        and (
            reynolds < maximum_reynolds
            if maximum_reynolds_is_exclusive
            else reynolds <= maximum_reynolds
        )
    )


def _prandtl_in_range(
    prandtl: float, minimum_prandtl: float, maximum_prandtl: float
) -> bool:
    return (
        isfinite(prandtl)
        and minimum_prandtl <= prandtl <= maximum_prandtl
    )


def _correlation_boundary_condition(
    correlation: HeatTransferCorrelation,
) -> tuple[LaminarThermalBoundaryCondition | TurbulentThermalDirection, bool]:
    return (
        (
            correlation.boundary_condition,
            isinstance(
                correlation.boundary_condition,
                LaminarThermalBoundaryCondition,
            ),
        )
        if isinstance(
            correlation,
            (
                FullyDevelopedLaminarCircularTube,
                LaminarCircularTubeConservativeLowerBound,
            ),
        )
        else (
            correlation.thermal_direction,
            isinstance(
                correlation.thermal_direction,
                TurbulentThermalDirection,
            ),
        )
    )


def _heat_transfer_correlation_obstructions(
    correlation: HeatTransferCorrelation,
    fluid_thermal_conductivity_watt_per_metre_kelvin: float,
) -> tuple[Obstruction, ...]:
    validity = correlation.validity
    geometry = correlation.geometry
    valid_diameter = _positive_finite(geometry.hydraulic_diameter_metres)
    valid_development_length = _positive_finite(
        geometry.upstream_development_length_metres
    )
    entrance_length_diameters = _entrance_length_diameters(
        geometry, valid_diameter, valid_development_length
    )
    reynolds_in_range = _reynolds_in_range(
        correlation.reynolds,
        validity.minimum_reynolds,
        validity.maximum_reynolds,
        validity.maximum_reynolds_is_exclusive,
    )
    prandtl_in_range = _prandtl_in_range(
        correlation.prandtl,
        validity.minimum_prandtl,
        validity.maximum_prandtl,
    )
    boundary_condition, boundary_condition_is_valid = (
        _correlation_boundary_condition(correlation)
    )
    checks = (
        (
            HeatTransferValidityAxis.CORRELATION_ID,
            str(correlation.correlation_id),
            None,
            None,
            bool(correlation.correlation_id),
        ),
        (
            HeatTransferValidityAxis.HYDRAULIC_DIAMETER,
            geometry.hydraulic_diameter_metres,
            0.0,
            None,
            valid_diameter,
        ),
        (
            HeatTransferValidityAxis.UPSTREAM_DEVELOPMENT_LENGTH,
            geometry.upstream_development_length_metres,
            0.0,
            None,
            valid_development_length,
        ),
        (
            HeatTransferValidityAxis.REYNOLDS,
            correlation.reynolds,
            validity.minimum_reynolds,
            validity.maximum_reynolds,
            reynolds_in_range,
        ),
        (
            HeatTransferValidityAxis.PRANDTL,
            correlation.prandtl,
            validity.minimum_prandtl,
            validity.maximum_prandtl,
            prandtl_in_range,
        ),
        (
            HeatTransferValidityAxis.ENTRANCE_LENGTH_DIAMETERS,
            entrance_length_diameters,
            validity.minimum_entrance_length_diameters,
            None,
            not (
                valid_diameter
                and valid_development_length
                and reynolds_in_range
                and prandtl_in_range
            )
            or entrance_length_diameters
            >= validity.minimum_entrance_length_diameters,
        ),
        (
            HeatTransferValidityAxis.BOUNDARY_CONDITION,
            str(boundary_condition),
            None,
            None,
            boundary_condition_is_valid,
        ),
        (
            HeatTransferValidityAxis.FLUID_THERMAL_CONDUCTIVITY,
            fluid_thermal_conductivity_watt_per_metre_kelvin,
            0.0,
            None,
            _positive_finite(fluid_thermal_conductivity_watt_per_metre_kelvin),
        ),
    )
    return _regime_obstructions(
        correlation.correlation_id, correlation.correlation_kind, checks
    )
