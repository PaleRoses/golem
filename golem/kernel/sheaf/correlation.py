"""Closed heat-transfer correlations: the circular-tube Nusselt laws and the
Churchill-Chu natural-convection law, each carrying its own validity regime."""

from __future__ import annotations

from dataclasses import dataclass

from golem.materials import CompressibleGasMaterialId
from golem.kernel.sheaf.vocabulary import (
    HeatTransferCorrelationId,
    HeatTransferCorrelationKind,
    LaminarThermalBoundaryCondition,
    NaturalConvectionSurfaceOrientation,
    TurbulentThermalDirection,
)


@dataclass(frozen=True)
class CircularTubeGeometry:
    hydraulic_diameter_metres: float
    upstream_development_length_metres: float


@dataclass(frozen=True)
class HeatTransferValidityInterval:
    minimum_reynolds: float
    maximum_reynolds: float
    maximum_reynolds_is_exclusive: bool
    minimum_prandtl: float
    maximum_prandtl: float
    minimum_entrance_length_diameters: float


@dataclass(frozen=True)
class FullyDevelopedLaminarCircularTube:
    correlation_id: HeatTransferCorrelationId
    geometry: CircularTubeGeometry
    reynolds: float
    prandtl: float
    boundary_condition: LaminarThermalBoundaryCondition

    @property
    def correlation_kind(self) -> HeatTransferCorrelationKind:
        return HeatTransferCorrelationKind.FULLY_DEVELOPED_LAMINAR_CIRCULAR_TUBE

    @property
    def validity(self) -> HeatTransferValidityInterval:
        return HeatTransferValidityInterval(
            minimum_reynolds=0.0,
            maximum_reynolds=2300.0,
            maximum_reynolds_is_exclusive=True,
            minimum_prandtl=0.6,
            maximum_prandtl=1000.0,
            minimum_entrance_length_diameters=0.05
            * self.reynolds
            * max(self.prandtl, 1.0),
        )


@dataclass(frozen=True)
class LaminarCircularTubeConservativeLowerBound:
    """Asymptotic Nu lower bound; developing-flow enhancement is ignored."""

    correlation_id: HeatTransferCorrelationId
    geometry: CircularTubeGeometry
    reynolds: float
    prandtl: float
    boundary_condition: LaminarThermalBoundaryCondition

    @property
    def correlation_kind(self) -> HeatTransferCorrelationKind:
        return (
            HeatTransferCorrelationKind.LAMINAR_CIRCULAR_TUBE_CONSERVATIVE_LOWER_BOUND
        )

    @property
    def validity(self) -> HeatTransferValidityInterval:
        return HeatTransferValidityInterval(
            minimum_reynolds=0.0,
            maximum_reynolds=2300.0,
            maximum_reynolds_is_exclusive=True,
            minimum_prandtl=0.6,
            maximum_prandtl=1000.0,
            minimum_entrance_length_diameters=0.0,
        )


@dataclass(frozen=True)
class DittusBoelterSmoothCircularTube:
    correlation_id: HeatTransferCorrelationId
    geometry: CircularTubeGeometry
    reynolds: float
    prandtl: float
    thermal_direction: TurbulentThermalDirection

    @property
    def correlation_kind(self) -> HeatTransferCorrelationKind:
        return HeatTransferCorrelationKind.DITTUS_BOELTER_SMOOTH_CIRCULAR_TUBE

    @property
    def validity(self) -> HeatTransferValidityInterval:
        return HeatTransferValidityInterval(
            minimum_reynolds=10_000.0,
            maximum_reynolds=120_000.0,
            maximum_reynolds_is_exclusive=False,
            minimum_prandtl=0.7,
            maximum_prandtl=160.0,
            minimum_entrance_length_diameters=10.0,
        )


type HeatTransferCorrelation = (
    FullyDevelopedLaminarCircularTube
    | LaminarCircularTubeConservativeLowerBound
    | DittusBoelterSmoothCircularTube
)


@dataclass(frozen=True)
class NaturalConvectionValidityInterval:
    minimum_rayleigh: float
    maximum_rayleigh: float
    minimum_rayleigh_is_inclusive: bool
    maximum_rayleigh_is_inclusive: bool
    minimum_prandtl: float
    minimum_prandtl_is_inclusive: bool


@dataclass(frozen=True)
class HeatTransferCorrelationEvidence:
    source_title: str
    source_uri: str
    note: str


@dataclass(frozen=True)
class ChurchillChuIsothermalVerticalPlate:
    correlation_id: HeatTransferCorrelationId
    characteristic_length_metres: float
    surface_orientation: NaturalConvectionSurfaceOrientation

    @property
    def correlation_kind(self) -> HeatTransferCorrelationKind:
        return HeatTransferCorrelationKind.CHURCHILL_CHU_ISOTHERMAL_VERTICAL_PLATE

    @property
    def validity(self) -> NaturalConvectionValidityInterval:
        return NaturalConvectionValidityInterval(
            minimum_rayleigh=0.0,
            maximum_rayleigh=1.0e12,
            minimum_rayleigh_is_inclusive=True,
            maximum_rayleigh_is_inclusive=False,
            minimum_prandtl=0.0,
            minimum_prandtl_is_inclusive=False,
        )

    @property
    def evidence(self) -> HeatTransferCorrelationEvidence:
        return HeatTransferCorrelationEvidence(
            source_title=(
                "Churchill and Chu, correlating equations for laminar and "
                "turbulent free convection from a vertical plate"
            ),
            source_uri="https://doi.org/10.1016/0017-9310(75)90243-4",
            note=(
                "space-mean isothermal vertical-plate correlation; accepted "
                "for 0 <= Ra_L < 1e12 and Pr > 0"
            ),
        )


@dataclass(frozen=True)
class HeatTransferCoefficient:
    correlation_id: HeatTransferCorrelationId
    correlation_kind: HeatTransferCorrelationKind
    nusselt_number: float
    watt_per_square_metre_kelvin: float
    validity: HeatTransferValidityInterval


@dataclass(frozen=True)
class NaturalConvectionHeatTransferCoefficient:
    correlation_id: HeatTransferCorrelationId
    correlation_kind: HeatTransferCorrelationKind
    gas_material_id: CompressibleGasMaterialId
    surface_orientation: NaturalConvectionSurfaceOrientation
    characteristic_length_metres: float
    film_temperature_kelvin: float
    gas_density_kilograms_per_cubic_metre: float
    volumetric_thermal_expansion_per_kelvin: float
    kinematic_viscosity_square_metres_per_second: float
    thermal_diffusivity_square_metres_per_second: float
    prandtl_number: float
    rayleigh_number: float
    nusselt_number: float
    watt_per_square_metre_kelvin: float
    validity: NaturalConvectionValidityInterval
    evidence: HeatTransferCorrelationEvidence
