"""The single typed catalogue for visual and physical material evidence.

Appearance coatings are annotations and may resolve to the neutral visual
fallback.  Solids, liquids, and gases are constitutive evidence: their
decoders fail closed and never borrow that fallback.  Catalogue records are
immutable; exchange JSON is a derived boundary view.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


type LinearRgb = tuple[float, float, float]


class MaterialKind(StrEnum):
    APPEARANCE = "appearance"
    ISOTROPIC_SOLID = "isotropic_solid"
    INCOMPRESSIBLE_FLUID = "incompressible_fluid"
    COMPRESSIBLE_GAS = "compressible_gas"


class AppearanceMaterialId(StrEnum):
    OBSIDIAN_WARDEN = "obsidian_warden"
    OBSIDIAN_DEEP = "obsidian_deep"
    STEEL_VIOLET = "steel_violet"
    EMISSIVE_SEAM = "emissive_seam"
    EYE_GLOSS = "eye_gloss"
    VASCULAR_SUPPLY_GOLD = "vascular_supply_gold"
    VASCULAR_RETURN_VIOLET = "vascular_return_violet"
    VASCULAR_EXCHANGE_CYAN = "vascular_exchange_cyan"
    GAMBESON_DARK = "gambeson_dark"
    NEUTRAL_GRAY = "neutral_gray"


class IsotropicSolidMaterialId(StrEnum):
    STAINLESS_STEEL_316L_ROOM_TEMPERATURE = (
        "stainless_steel_316l_room_temperature"
    )


class IncompressibleFluidMaterialId(StrEnum):
    WATER_0_1_MPA_293K = "water_0_1_mpa_293k"


class CompressibleGasMaterialId(StrEnum):
    DRY_AIR_101325_PA_298K = "dry_air_101325_pa_298k"


class GasEquationOfState(StrEnum):
    IDEAL_GAS = "ideal_gas"


class EvidenceKind(StrEnum):
    MANUFACTURER_DATASHEET = "manufacturer_datasheet"
    REFERENCE_CORRELATION = "reference_correlation"
    DERIVED_POLICY = "derived_policy"


@dataclass(frozen=True)
class TemperatureInterval:
    minimum_kelvin: float
    maximum_kelvin: float

    def contains(self, temperature_kelvin: float) -> bool:
        return self.minimum_kelvin <= temperature_kelvin <= self.maximum_kelvin


@dataclass(frozen=True)
class PressureInterval:
    minimum_pascal: float
    maximum_pascal: float

    def contains(self, pressure_pascal: float) -> bool:
        return self.minimum_pascal <= pressure_pascal <= self.maximum_pascal


@dataclass(frozen=True)
class FluidPhaseLimits:
    freezing_temperature_kelvin: float
    normal_boiling_temperature_kelvin: float
    critical_temperature_kelvin: float
    critical_pressure_pascal: float


@dataclass(frozen=True)
class MaterialEvidence:
    kind: EvidenceKind
    source_title: str
    source_uri: str
    note: str


@dataclass(frozen=True)
class AppearanceMaterial:
    material_id: AppearanceMaterialId
    base_color: LinearRgb
    roughness: float
    metallic: float
    emissive_color: LinearRgb
    emissive_strength: float
    texture_set: str | None = None

    @property
    def is_emissive(self) -> bool:
        return self.emissive_strength > 0.0


@dataclass(frozen=True)
class IsotropicSolid:
    material_id: IsotropicSolidMaterialId
    density_kilograms_per_cubic_metre: float
    thermal_conductivity_watts_per_metre_kelvin: float
    specific_heat_joules_per_kilogram_kelvin: float
    elastic_modulus_pascal: float
    poisson_ratio: float
    thermal_expansion_per_kelvin: float
    yield_stress_pascal: float
    allowable_stress_pascal: float
    operating_temperature: TemperatureInterval
    evidence: tuple[MaterialEvidence, ...]


@dataclass(frozen=True)
class IncompressibleFluid:
    material_id: IncompressibleFluidMaterialId
    density_kilograms_per_cubic_metre: float
    dynamic_viscosity_pascal_second: float
    specific_heat_joules_per_kilogram_kelvin: float
    thermal_conductivity_watts_per_metre_kelvin: float
    operating_temperature: TemperatureInterval
    allowable_pressure: PressureInterval
    phase_limits: FluidPhaseLimits
    evidence: tuple[MaterialEvidence, ...]


@dataclass(frozen=True)
class CompressibleGas:
    """Constant transport properties plus an explicit density law.

    The narrow temperature interval is the validity domain for the reference
    viscosity, heat capacity, and conductivity.  Density and volumetric
    expansion remain derived from the declared equation of state at the
    evaluated film state.
    """

    material_id: CompressibleGasMaterialId
    equation_of_state: GasEquationOfState
    specific_gas_constant_joules_per_kilogram_kelvin: float
    dynamic_viscosity_pascal_second: float
    specific_heat_joules_per_kilogram_kelvin: float
    thermal_conductivity_watts_per_metre_kelvin: float
    reference_temperature_kelvin: float
    reference_pressure_pascal: float
    operating_temperature: TemperatureInterval
    allowable_pressure: PressureInterval
    evidence: tuple[MaterialEvidence, ...]


@dataclass(frozen=True)
class MaterialCatalogue:
    appearance_materials: tuple[AppearanceMaterial, ...]
    isotropic_solids: tuple[IsotropicSolid, ...]
    incompressible_fluids: tuple[IncompressibleFluid, ...]
    compressible_gases: tuple[CompressibleGas, ...]


@dataclass(frozen=True)
class UnknownMaterialObstruction:
    material_kind: MaterialKind
    identifier: object


@dataclass(frozen=True)
class AcceptedMaterial[material]:
    material: material


@dataclass(frozen=True)
class RejectedMaterial:
    obstructions: tuple[UnknownMaterialObstruction, ...]


type MaterialDecodeResult[material] = AcceptedMaterial[material] | RejectedMaterial


_NEUTRAL_GRAY = AppearanceMaterial(
    material_id=AppearanceMaterialId.NEUTRAL_GRAY,
    base_color=(0.40, 0.41, 0.44),
    roughness=0.6,
    metallic=0.0,
    emissive_color=(0.0, 0.0, 0.0),
    emissive_strength=0.0,
)


MATERIAL_CATALOGUE = MaterialCatalogue(
    appearance_materials=(
        AppearanceMaterial(
            AppearanceMaterialId.OBSIDIAN_WARDEN,
            (0.034, 0.033, 0.048),
            0.24,
            0.05,
            (0.0, 0.0, 0.0),
            0.0,
        ),
        AppearanceMaterial(
            AppearanceMaterialId.OBSIDIAN_DEEP,
            (0.030, 0.029, 0.044),
            0.30,
            0.05,
            (0.0, 0.0, 0.0),
            0.0,
        ),
        AppearanceMaterial(
            AppearanceMaterialId.STEEL_VIOLET,
            (0.060, 0.060, 0.086),
            0.38,
            0.75,
            (0.0, 0.0, 0.0),
            0.0,
        ),
        AppearanceMaterial(
            AppearanceMaterialId.EMISSIVE_SEAM,
            (0.10, 0.05, 0.15),
            0.5,
            0.0,
            (0.66, 0.33, 0.97),
            2.5,
        ),
        AppearanceMaterial(
            AppearanceMaterialId.EYE_GLOSS,
            (0.035, 0.045, 0.055),
            0.08,
            0.0,
            (0.0, 0.0, 0.0),
            0.0,
        ),
        AppearanceMaterial(
            AppearanceMaterialId.VASCULAR_SUPPLY_GOLD,
            (0.72, 0.43, 0.08),
            0.32,
            0.18,
            (1.0, 0.52, 0.10),
            1.8,
        ),
        AppearanceMaterial(
            AppearanceMaterialId.VASCULAR_RETURN_VIOLET,
            (0.16, 0.10, 0.44),
            0.38,
            0.08,
            (0.28, 0.18, 0.92),
            1.7,
        ),
        AppearanceMaterial(
            AppearanceMaterialId.VASCULAR_EXCHANGE_CYAN,
            (0.36, 0.78, 0.82),
            0.45,
            0.0,
            (0.56, 0.96, 1.0),
            1.45,
        ),
        AppearanceMaterial(
            AppearanceMaterialId.GAMBESON_DARK,
            (0.048, 0.042, 0.046),
            0.88,
            0.0,
            (0.0, 0.0, 0.0),
            0.0,
        ),
        _NEUTRAL_GRAY,
    ),
    isotropic_solids=(
        IsotropicSolid(
            material_id=(
                IsotropicSolidMaterialId.STAINLESS_STEEL_316L_ROOM_TEMPERATURE
            ),
            density_kilograms_per_cubic_metre=8000.0,
            thermal_conductivity_watts_per_metre_kelvin=15.0,
            specific_heat_joules_per_kilogram_kelvin=500.0,
            elastic_modulus_pascal=200.0e9,
            poisson_ratio=0.30,
            thermal_expansion_per_kelvin=16.0e-6,
            yield_stress_pascal=170.0e6,
            allowable_stress_pascal=170.0e6 / 1.5,
            operating_temperature=TemperatureInterval(293.15, 373.15),
            evidence=(
                MaterialEvidence(
                    EvidenceKind.MANUFACTURER_DATASHEET,
                    "Outokumpu Supra 316L/4404 range datasheet",
                    (
                        "https://otke-cdn.outokumpu.com/-/media/files/products/"
                        "supra/outokumpu-supra-range-datasheet.pdf"
                    ),
                    (
                        "ASTM minimum yield and EN room-temperature density, "
                        "elastic, expansion, conductivity, and heat-capacity data"
                    ),
                ),
                MaterialEvidence(
                    EvidenceKind.DERIVED_POLICY,
                    "GOLEM M7 first-cut allowable-stress policy",
                    "repo:docs/implementation/plans/creature/golem-v05-appearance-ladder.md",
                    "allowable stress is minimum yield divided by 1.5",
                ),
            ),
        ),
    ),
    incompressible_fluids=(
        IncompressibleFluid(
            material_id=IncompressibleFluidMaterialId.WATER_0_1_MPA_293K,
            density_kilograms_per_cubic_metre=998.2,
            dynamic_viscosity_pascal_second=1.0016e-3,
            specific_heat_joules_per_kilogram_kelvin=4182.0,
            thermal_conductivity_watts_per_metre_kelvin=0.598,
            operating_temperature=TemperatureInterval(288.15, 313.15),
            allowable_pressure=PressureInterval(90_000.0, 500_000.0),
            phase_limits=FluidPhaseLimits(
                freezing_temperature_kelvin=273.15,
                normal_boiling_temperature_kelvin=373.124,
                critical_temperature_kelvin=647.096,
                critical_pressure_pascal=22.064e6,
            ),
            evidence=(
                MaterialEvidence(
                    EvidenceKind.REFERENCE_CORRELATION,
                    "IAPWS SR6-08(2011), liquid water at 0.1 MPa",
                    "https://iapws.org/documents/release/LiquidWater",
                    (
                        "constant-property 293 K first cut inside the published "
                        "correlation range; viscosity uses the ISO reference value"
                    ),
                ),
                MaterialEvidence(
                    EvidenceKind.DERIVED_POLICY,
                    "GOLEM M7 liquid-water constant-property interval",
                    (
                        "repo:docs/implementation/plans/creature/"
                        "golem-v05-appearance-ladder.md"
                    ),
                    (
                        "293.15 K reference transport properties are accepted "
                        "only for liquid states within 288.15..313.15 K"
                    ),
                ),
            ),
        ),
    ),
    compressible_gases=(
        CompressibleGas(
            material_id=CompressibleGasMaterialId.DRY_AIR_101325_PA_298K,
            equation_of_state=GasEquationOfState.IDEAL_GAS,
            specific_gas_constant_joules_per_kilogram_kelvin=287.05,
            dynamic_viscosity_pascal_second=18.468e-6,
            specific_heat_joules_per_kilogram_kelvin=1006.0,
            thermal_conductivity_watts_per_metre_kelvin=26.06e-3,
            reference_temperature_kelvin=298.15,
            reference_pressure_pascal=101_325.0,
            operating_temperature=TemperatureInterval(288.15, 308.15),
            allowable_pressure=PressureInterval(90_000.0, 110_000.0),
            evidence=(
                MaterialEvidence(
                    EvidenceKind.REFERENCE_CORRELATION,
                    "NIST viscosity and thermal conductivity of dry air",
                    "https://doi.org/10.1063/1.555744",
                    (
                        "room-temperature dry-air viscosity and conductivity "
                        "at 101325 Pa"
                    ),
                ),
                MaterialEvidence(
                    EvidenceKind.REFERENCE_CORRELATION,
                    "NIST thermodynamic properties of air",
                    "https://doi.org/10.1063/1.1285884",
                    (
                        "dry-air gas constant and isobaric heat capacity near "
                        "298.15 K"
                    ),
                ),
                MaterialEvidence(
                    EvidenceKind.DERIVED_POLICY,
                    "GOLEM M7 constant-property film-state policy",
                    (
                        "repo:docs/implementation/plans/creature/"
                        "golem-v05-appearance-ladder.md"
                    ),
                    (
                        "reference transport properties are accepted only for "
                        "film temperatures within 288.15..308.15 K"
                    ),
                ),
            ),
        ),
    ),
)


DEFAULT_APPEARANCE_MATERIAL_ID = AppearanceMaterialId.NEUTRAL_GRAY
