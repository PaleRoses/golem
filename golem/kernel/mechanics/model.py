"""Immutable mechanics inputs: domain, constitutive projection, and load case.

These are the pure carriers the small-strain thermoelastic solver consumes.
``CellIndex`` is a geometric structured-grid triple with no flat-id projection;
it is deliberately distinct from ``addressing.CellId`` and ``sheaf.CellId`` and
must never be unified with either.  The sole addressing coupling is
``StructuredHexDomain.cell_dims`` yielding a :class:`GridDims`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import TYPE_CHECKING

from golem.addressing.core import GridDims

if TYPE_CHECKING:
    from golem.materials.core import IsotropicSolid


MODEL_NAME = (
    "small-strain isotropic linear thermoelasticity on a structured "
    "volumetric/SDF-derived domain"
)
DISCRETIZATION_NAME = (
    "conforming trilinear eight-node hexahedra with 2x2x2 Gauss integration"
)

type Vector3 = tuple[float, float, float]


class DomainSource(StrEnum):
    """Provenance of the derived analysis cells, never a geometry authority."""

    STRUCTURED_VOLUMETRIC = "structured_volumetric"
    SDF_DERIVED = "sdf_derived"


class BoundaryFace(StrEnum):
    X_MIN = "x_min"
    X_MAX = "x_max"
    Y_MIN = "y_min"
    Y_MAX = "y_max"
    Z_MIN = "z_min"
    Z_MAX = "z_max"


class UnevaluatedPhysics(StrEnum):
    LINEARIZED_BUCKLING = "linearized_buckling"
    FATIGUE = "fatigue"
    FRACTURE = "fracture"
    LARGE_DEFORMATION = "large_deformation"
    CONTACT_PLASTICITY = "contact_plasticity"
    ANISOTROPIC_COMPOSITE_FAILURE = "anisotropic_composite_failure"
    SUBGRID_CHANNEL_STRESS_CONCENTRATION = "subgrid_channel_stress_concentration"
    SUBGRID_CHANNEL_GLOBAL_STIFFNESS = "subgrid_channel_global_stiffness"
    SUBGRID_CHANNEL_GLOBAL_STRENGTH = "subgrid_channel_global_strength"
    SUBGRID_CHANNEL_PRESSURE_PRESTRESS = "subgrid_channel_pressure_prestress"


class MaterialFractionResolution(StrEnum):
    """Evidence attached to a cell fraction before mechanics may consume it."""

    FULL_SOLID = "full_solid"
    HOMOGENIZED_EFFECTIVE_PROPERTIES = "homogenized_effective_properties"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, order=True)
class NodeIndex:
    x_index: int
    y_index: int
    z_index: int


@dataclass(frozen=True, order=True)
class CellIndex:
    x_index: int
    y_index: int
    z_index: int


@dataclass(frozen=True)
class StructuredHexDomain:
    """A derived structured analysis domain.

    ``solid_cells`` indexes intervals between successive axis coordinates.
    It is intentionally only a discretized view of an upstream morphology.
    """

    x_coordinates: tuple[float, ...]
    y_coordinates: tuple[float, ...]
    z_coordinates: tuple[float, ...]
    solid_cells: tuple[CellIndex, ...]
    source: DomainSource = DomainSource.SDF_DERIVED

    @property
    def cell_dims(self) -> GridDims:
        return GridDims(
            len(self.x_coordinates) - 1,
            len(self.y_coordinates) - 1,
            len(self.z_coordinates) - 1,
        )


@dataclass(frozen=True)
class IsotropicConstitutiveProperties:
    """Mechanics projection of properties resolved by the material owner."""

    young_modulus: float
    poisson_ratio: float
    density: float
    thermal_expansion_coefficient: float
    yield_strength: float

    @classmethod
    def from_isotropic_solid(
        cls, material: "IsotropicSolid"
    ) -> "IsotropicConstitutiveProperties":
        """Project the canonical material record without importing its catalogue."""

        return cls(
            young_modulus=material.elastic_modulus_pascal,
            poisson_ratio=material.poisson_ratio,
            density=material.density_kilograms_per_cubic_metre,
            thermal_expansion_coefficient=material.thermal_expansion_per_kelvin,
            yield_strength=material.yield_stress_pascal,
        )


@dataclass(frozen=True)
class CellConstitutiveAssignment:
    cell: CellIndex
    properties: IsotropicConstitutiveProperties
    solid_fraction: float = 1.0
    fraction_resolution: MaterialFractionResolution = (
        MaterialFractionResolution.FULL_SOLID
    )


@dataclass(frozen=True)
class NodalSupport:
    """Prescribed displacement by component; ``None`` leaves it free."""

    node: NodeIndex
    prescribed_displacement: tuple[float | None, float | None, float | None]


@dataclass(frozen=True)
class NodalForce:
    node: NodeIndex
    force: Vector3


@dataclass(frozen=True)
class NodalTemperatureChange:
    node: NodeIndex
    delta_temperature: float


@dataclass(frozen=True)
class InternalPressureFace:
    """Physical pressure whose traction follows the displaced Q1 face."""

    cell: CellIndex
    face: BoundaryFace
    pressure: float


@dataclass(frozen=True)
class MechanicsCriteria:
    linear_solver_relative_tolerance: float = 1.0e-9
    equilibrium_relative_tolerance: float = 1.0e-8
    maximum_principal_strain: float = 2.0e-2
    maximum_displacement: float | None = None
    minimum_yield_safety_factor: float | None = None
    require_linearized_buckling: bool = False
    minimum_linearized_buckling_load_factor: float | None = None
    normalization_floor: float = 1.0e-14


@dataclass(frozen=True)
class MechanicsProblem:
    domain: StructuredHexDomain
    cell_properties: tuple[CellConstitutiveAssignment, ...]
    supports: tuple[NodalSupport, ...]
    body_acceleration: Vector3 = (0.0, 0.0, 0.0)
    nodal_forces: tuple[NodalForce, ...] = ()
    nodal_temperature_changes: tuple[NodalTemperatureChange, ...] = ()
    internal_pressure_faces: tuple[InternalPressureFace, ...] = ()
    criteria: MechanicsCriteria = MechanicsCriteria()


def _invalid_isotropic_constitutive_fields(
    properties: IsotropicConstitutiveProperties,
) -> tuple[tuple[str, float], ...]:
    fields = (
        ("young_modulus", properties.young_modulus, properties.young_modulus > 0.0),
        (
            "poisson_ratio",
            properties.poisson_ratio,
            -1.0 < properties.poisson_ratio < 0.5,
        ),
        ("density", properties.density, properties.density >= 0.0),
        (
            "thermal_expansion_coefficient",
            properties.thermal_expansion_coefficient,
            True,
        ),
        ("yield_strength", properties.yield_strength, properties.yield_strength > 0.0),
    )
    return tuple(
        (name, value)
        for name, value, valid_range in fields
        if not isfinite(value) or not valid_range
    )
