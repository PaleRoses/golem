"""Closed vocabulary for the balance algebra: solver DOF identity plus the term,
solver, and heat-transfer regime enums every interpreter names."""

from __future__ import annotations

from enum import StrEnum
from typing import NewType


# Solver DOF ordinal: a dense index into ``solid_cells`` (``values[int(id)]``),
# NOT a geometric grid position. Never unify with the addressing module's CellId.
CellId = NewType("CellId", int)
BoundaryId = NewType("BoundaryId", str)
BalanceProblemId = NewType("BalanceProblemId", str)
BalanceInterfaceId = NewType("BalanceInterfaceId", str)
BalanceSourceId = NewType("BalanceSourceId", str)
AmbientExchangeId = NewType("AmbientExchangeId", str)
HeatTransferCorrelationId = NewType("HeatTransferCorrelationId", str)


class BalanceTermKind(StrEnum):
    SYMMETRIC_CONDUCTANCE = "symmetric_conductance"
    DIRECTED_TRANSPORT = "directed_transport"
    CELL_SOURCE = "cell_source"
    FIXED_BOUNDARY = "fixed_boundary"
    AMBIENT_EXCHANGE = "ambient_exchange"


class BalanceMatrixFailure(StrEnum):
    LINEAR_SOLVE_FAILED = "linear_solve_failed"
    NONFINITE_SOLUTION = "nonfinite_solution"


class BalanceSolverParameter(StrEnum):
    RELATIVE_TOLERANCE = "relative_tolerance"
    RESIDUAL_TOLERANCE = "residual_tolerance"
    IMBALANCE_TOLERANCE = "imbalance_tolerance"
    NORMALIZATION_FLOOR = "normalization_floor"
    MAXIMUM_ITERATION_FACTOR = "maximum_iteration_factor"


class HeatTransferCorrelationKind(StrEnum):
    FULLY_DEVELOPED_LAMINAR_CIRCULAR_TUBE = (
        "fully_developed_laminar_circular_tube"
    )
    LAMINAR_CIRCULAR_TUBE_CONSERVATIVE_LOWER_BOUND = (
        "laminar_circular_tube_conservative_lower_bound"
    )
    DITTUS_BOELTER_SMOOTH_CIRCULAR_TUBE = (
        "dittus_boelter_smooth_circular_tube"
    )
    CHURCHILL_CHU_ISOTHERMAL_VERTICAL_PLATE = (
        "churchill_chu_isothermal_vertical_plate"
    )


class LaminarThermalBoundaryCondition(StrEnum):
    CONSTANT_WALL_TEMPERATURE = "constant_wall_temperature"
    CONSTANT_HEAT_FLUX = "constant_heat_flux"


class TurbulentThermalDirection(StrEnum):
    FLUID_HEATING = "fluid_heating"
    FLUID_COOLING = "fluid_cooling"


class NaturalConvectionSurfaceOrientation(StrEnum):
    VERTICAL_PLATE = "vertical_plate"


class HeatTransferValidityAxis(StrEnum):
    CORRELATION_ID = "correlation_id"
    HYDRAULIC_DIAMETER = "hydraulic_diameter"
    UPSTREAM_DEVELOPMENT_LENGTH = "upstream_development_length"
    REYNOLDS = "reynolds"
    PRANDTL = "prandtl"
    ENTRANCE_LENGTH_DIAMETERS = "entrance_length_diameters"
    BOUNDARY_CONDITION = "boundary_condition"
    FLUID_THERMAL_CONDUCTIVITY = "fluid_thermal_conductivity"
    CHARACTERISTIC_LENGTH = "characteristic_length"
    SURFACE_ORIENTATION = "surface_orientation"
    AMBIENT_TEMPERATURE = "ambient_temperature"
    SURFACE_TEMPERATURE = "surface_temperature"
    FILM_TEMPERATURE = "film_temperature"
    AMBIENT_PRESSURE = "ambient_pressure"
    GRAVITATIONAL_ACCELERATION = "gravitational_acceleration"
    GAS_EQUATION_OF_STATE = "gas_equation_of_state"
    GAS_CONSTANT = "gas_constant"
    DYNAMIC_VISCOSITY = "dynamic_viscosity"
    SPECIFIC_HEAT = "specific_heat"
    THERMAL_CONDUCTIVITY = "thermal_conductivity"
    RAYLEIGH = "rayleigh"


class ThermalParameter(StrEnum):
    SOLID_CONDUCTIVITY = "solid_conductivity"
    SOLID_EXCHANGE_AREA = "solid_exchange_area"
    SOLID_PATH_LENGTH = "solid_path_length"
    COOLANT_MASS_FLOW = "coolant_mass_flow"
    COOLANT_SPECIFIC_HEAT = "coolant_specific_heat"
    WALL_EXCHANGE_AREA = "wall_exchange_area"
    AMBIENT_HEAT_TRANSFER_COEFFICIENT = "ambient_heat_transfer_coefficient"
    AMBIENT_EXCHANGE_AREA = "ambient_exchange_area"
