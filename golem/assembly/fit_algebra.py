"""Element-fit verdict and result-gluing algebra."""

from __future__ import annotations

import numpy as np

from dataclasses import replace
from itertools import combinations

from golem.assembly.carriers import AcceptedAssembly, AssemblyResult, ElementFitReceipt, _CompiledElement
from golem.assembly.fit_compile import (_MINIMUM_FITTED_SURFACE_FRACTION, _element_pair_fit_geometry, _fit_receipt, _minimum_declared_clearance)
from golem.assembly.fit_types import (_BoundedElementContact, _ElementPairFitGeometry)
from golem.assembly.obstructions import (
    ElementClearanceObstruction,
    ElementFitObstruction,
    ElementSurfaceFormationObstruction,
)

def _zero_penetration_verdict(
    geometry: _ElementPairFitGeometry,
    left: _CompiledElement,
    right: _CompiledElement,
) -> ElementFitReceipt | ElementClearanceObstruction:
    tolerance = geometry.tolerance
    left_field = geometry.left_field
    right_field = geometry.right_field
    penetrating_count = geometry.penetrating_count
    declaration = geometry.declaration
    fit_law = geometry.fit_law
    fitted_surface_fraction = geometry.fitted_surface_fraction
    required_fitted_surface_fraction = geometry.required_fitted_surface_fraction
    maximum_penetration = geometry.maximum_penetration
    if declaration is None:
        return _fit_receipt(
            left,
            right,
            fit_law,
            declaration,
            left_field,
            right_field,
            maximum_penetration,
            tolerance,
            penetrating_count,
            fitted_surface_fraction,
            required_fitted_surface_fraction,
        )
    minimum_clearance = _minimum_declared_clearance(
        declaration, left, left_field, right_field
    )
    maximum_clearance = (
        2.0 * tolerance
        if isinstance(declaration, _BoundedElementContact)
        else declaration.maximum_clearance_world
    )
    return (
        _fit_receipt(
            left,
            right,
            fit_law,
            declaration,
            left_field,
            right_field,
            maximum_penetration,
            tolerance,
            penetrating_count,
            fitted_surface_fraction,
            required_fitted_surface_fraction,
        )
        if minimum_clearance <= maximum_clearance + tolerance
        and (
            fitted_surface_fraction is None
            or fitted_surface_fraction
            >= _MINIMUM_FITTED_SURFACE_FRACTION
        )
        else ElementClearanceObstruction(
            left_element_id=left.element_id,
            right_element_id=right.element_id,
            fit_law=fit_law,
            declaring_element_id=declaration.declaring_element_id,
            minimum_clearance_world=minimum_clearance,
            maximum_clearance_world=maximum_clearance,
            tolerance_world=tolerance,
            fitted_surface_fraction=fitted_surface_fraction,
            required_fitted_surface_fraction=(
                required_fitted_surface_fraction
            ),
        )
    )


def _penetrating_verdict(
    geometry: _ElementPairFitGeometry,
    left: _CompiledElement,
    right: _CompiledElement,
) -> ElementFitReceipt | ElementFitObstruction:
    tolerance = geometry.tolerance
    left_vertices = geometry.left_vertices
    right_vertices = geometry.right_vertices
    left_field = geometry.left_field
    right_field = geometry.right_field
    left_penetrating = geometry.left_penetrating
    right_penetrating = geometry.right_penetrating
    penetrating_count = geometry.penetrating_count
    declaration = geometry.declaration
    fit_law = geometry.fit_law
    fitted_surface_fraction = geometry.fitted_surface_fraction
    required_fitted_surface_fraction = geometry.required_fitted_surface_fraction
    maximum_penetration = geometry.maximum_penetration
    penetrating_points = np.concatenate(
        (
            left_vertices[left_penetrating],
            right_vertices[right_penetrating],
        ),
        axis=0,
    )
    contact = (
        declaration
        if isinstance(declaration, _BoundedElementContact)
        else None
    )
    outside_contact_count = (
        penetrating_count
        if contact is None
        else int(
            np.count_nonzero(
                np.linalg.norm(
                    penetrating_points
                    - np.asarray(contact.center_world, dtype=np.float64),
                    axis=1,
                )
                > contact.radius_world + tolerance
            )
        )
    )
    return (
        _fit_receipt(
            left,
            right,
            fit_law,
            declaration,
            left_field,
            right_field,
            maximum_penetration,
            tolerance,
            penetrating_count,
            fitted_surface_fraction,
            required_fitted_surface_fraction,
        )
        if outside_contact_count == 0
        else ElementFitObstruction(
            left_element_id=left.element_id,
            right_element_id=right.element_id,
            fit_law=fit_law,
            penetrating_sample_count=penetrating_count,
            outside_contact_sample_count=outside_contact_count,
            maximum_penetration_world=maximum_penetration,
            tolerance_world=tolerance,
            contact_element_id=(
                contact.declaring_element_id
                if contact is not None
                else None
            ),
            contact_radius_world=(
                contact.radius_world if contact is not None else None
            ),
        )
    )


def _element_pair_fit_descent(
    left: _CompiledElement,
    right: _CompiledElement,
) -> (
    ElementFitReceipt
    | ElementFitObstruction
    | ElementClearanceObstruction
    | ElementSurfaceFormationObstruction
):
    geometry = _element_pair_fit_geometry(left, right)
    if isinstance(geometry, ElementSurfaceFormationObstruction):
        return geometry
    if geometry.penetrating_count == 0:
        return _zero_penetration_verdict(geometry, left, right)
    return _penetrating_verdict(geometry, left, right)


def _compiled_fit_descent(
    compiled: tuple[_CompiledElement, ...],
) -> tuple[
    ElementFitReceipt
    | ElementFitObstruction
    | ElementClearanceObstruction
    | ElementSurfaceFormationObstruction,
    ...,
]:
    return tuple(
        _element_pair_fit_descent(left, right)
        for left, right in combinations(compiled, 2)
    )


def _attach_fit_receipts(
    result: AssemblyResult,
    fit_receipts: tuple[ElementFitReceipt, ...],
) -> AssemblyResult:
    return (
        replace(
            result,
            receipt=replace(result.receipt, fit_receipts=fit_receipts),
        )
        if isinstance(result, AcceptedAssembly)
        else result
    )
