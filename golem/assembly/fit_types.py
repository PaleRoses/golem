"""Element-fit role carriers."""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass

from golem.assembly.obstructions import ElementFitLaw


@dataclass(frozen=True)
class _BoundedElementContact:
    declaring_element_id: str
    counterpart_element_id: str
    center_world: tuple[float, float, float]
    radius_world: float


@dataclass(frozen=True)
class _ElementSurfaceClearance:
    declaring_element_id: str
    counterpart_element_id: str
    maximum_clearance_world: float


type _ElementFitDeclaration = _BoundedElementContact | _ElementSurfaceClearance


@dataclass(frozen=True, eq=False)
class _ElementPairFitGeometry:
    tolerance: float
    left_vertices: np.ndarray
    right_vertices: np.ndarray
    left_field: np.ndarray
    right_field: np.ndarray
    left_penetrating: np.ndarray
    right_penetrating: np.ndarray
    penetrating_count: int
    declaration: _ElementFitDeclaration | None
    fit_law: ElementFitLaw
    fitted_surface_fraction: float | None
    required_fitted_surface_fraction: float | None
    maximum_penetration: float

