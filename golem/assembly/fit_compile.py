"""Element-fit input decoding and pair-geometry coalgebra."""

from __future__ import annotations

import math
import numpy as np

from golem.assembly.carriers import ElementFitReceipt, _CompiledElement, _solid_record
from golem.assembly.fit_types import (_BoundedElementContact, _ElementFitDeclaration, _ElementPairFitGeometry, _ElementSurfaceClearance)
from golem.assembly.obstructions import (
    ElementFitLaw,
    ElementSurfaceFormationObstruction,
    MalformedAssemblyObstruction,
)
from golem.kernel import engine as E


_MINIMUM_FITTED_SURFACE_FRACTION = 0.05

def _element_fit_input_obstructions(
    entries: tuple[dict, ...],
) -> tuple[MalformedAssemblyObstruction, ...]:
    element_ids = frozenset(
        identifier
        for entry in entries
        for identifier in (entry.get("id"),)
        if isinstance(identifier, str)
    )
    local_obstructions = tuple(
        obstruction
        for index, entry in enumerate(entries)
        for obstruction in _single_element_fit_input_obstructions(
            index, entry, element_ids
        )
    )
    reciprocal_obstructions = tuple(
        MalformedAssemblyObstruction(
            f"/elements/{index}/fit",
            "a solid pair may declare a non-default fit from only one element",
        )
        for index, entry in enumerate(entries)
        for payload in (entry.get("fit"),)
        if isinstance(payload, dict)
        and isinstance(payload.get("with"), str)
        and any(
            isinstance(other.get("fit"), dict)
            and other.get("id") == payload.get("with")
            and other["fit"].get("with") == entry.get("id")
            for other in entries[:index]
        )
    )
    return (*local_obstructions, *reciprocal_obstructions)


def _single_element_fit_input_obstructions(
    index: int,
    entry: dict,
    element_ids: frozenset[str],
) -> tuple[MalformedAssemblyObstruction, ...]:
    payload = entry.get("fit")
    address = f"/elements/{index}/fit"
    if payload is None:
        return ()
    if not isinstance(payload, dict):
        return (MalformedAssemblyObstruction(address, "expected an object"),)
    law = payload.get("law")
    expected_keys = (
        frozenset(("law", "with", "radius"))
        if law == ElementFitLaw.BOUNDED_MOUNT_CONTACT.value
        else frozenset(("law", "with", "maximum_gap"))
        if law == ElementFitLaw.SURFACE_CLEARANCE.value
        else frozenset()
    )
    if not expected_keys:
        return (
            MalformedAssemblyObstruction(
                f"{address}/law",
                "expected surface_clearance or bounded_mount_contact; omission means disjoint",
            ),
        )
    if frozenset(payload) != expected_keys:
        return (
            MalformedAssemblyObstruction(
                address,
                f"expected exactly {', '.join(sorted(expected_keys))}",
            ),
        )
    counterpart = payload.get("with")
    if (
        not isinstance(counterpart, str)
        or not counterpart
        or counterpart not in element_ids
        or counterpart == entry.get("id")
    ):
        return (
            MalformedAssemblyObstruction(
                f"{address}/with",
                "expected another declared element id",
            ),
        )
    distance_key = (
        "radius"
        if law == ElementFitLaw.BOUNDED_MOUNT_CONTACT.value
        else "maximum_gap"
    )
    distance = payload.get(distance_key)
    if (
        isinstance(distance, bool)
        or not isinstance(distance, (int, float))
        or not math.isfinite(float(distance))
        or float(distance) <= 0.0
    ):
        return (
            MalformedAssemblyObstruction(
                f"{address}/{distance_key}",
                "expected a finite positive world-space distance",
            ),
        )
    if law == ElementFitLaw.SURFACE_CLEARANCE.value:
        return ()
    mount = entry.get("mount")
    translation = mount.get("translate") if isinstance(mount, dict) else None
    if (
        not isinstance(translation, list)
        or len(translation) != 3
        or any(
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(float(coordinate))
            for coordinate in translation
        )
    ):
        return (
            MalformedAssemblyObstruction(
                f"/elements/{index}/mount/translate",
                "bounded mount contact requires a finite three-coordinate translation",
            ),
        )
    return ()


def _decoded_element_fit(
    element: _CompiledElement,
) -> _ElementFitDeclaration | None:
    payload = element.entry.get("fit")
    if not isinstance(payload, dict):
        return None
    if payload["law"] == ElementFitLaw.SURFACE_CLEARANCE.value:
        return _ElementSurfaceClearance(
            declaring_element_id=element.element_id,
            counterpart_element_id=str(payload["with"]),
            maximum_clearance_world=float(payload["maximum_gap"]),
        )
    mount = element.entry["mount"]
    return _BoundedElementContact(
        declaring_element_id=element.element_id,
        counterpart_element_id=str(payload["with"]),
        center_world=tuple(map(float, mount["translate"])),
        radius_world=float(payload["radius"]),
    )


def _solid_vertices(element: _CompiledElement) -> np.ndarray:
    return np.asarray(_solid_record(element).vertices, dtype=np.float64)


def _vertices_in_graph_bounds(
    vertices: np.ndarray,
    graph: dict,
    tolerance: float,
) -> np.ndarray:
    lower, upper = E.graph_bounds(graph)
    mask = np.all(
        (vertices >= lower - tolerance) & (vertices <= upper + tolerance),
        axis=1,
    )
    return vertices[mask]


def _declared_fit_for_pair(
    left: _CompiledElement,
    right: _CompiledElement,
) -> _ElementFitDeclaration | None:
    declarations = tuple(
        declaration
        for element, counterpart in ((left, right), (right, left))
        for declaration in (_decoded_element_fit(element),)
        if declaration is not None
        and declaration.counterpart_element_id == counterpart.element_id
    )
    return declarations[0] if declarations else None


def _minimum_declared_clearance(
    declaration: _ElementFitDeclaration,
    left: _CompiledElement,
    left_field: np.ndarray,
    right_field: np.ndarray,
) -> float:
    declaring_field = (
        left_field
        if declaration.declaring_element_id == left.element_id
        else right_field
    )
    return float(np.min(np.maximum(declaring_field, 0.0), initial=math.inf))


def _surface_clearance_area_fraction(
    declaration: _ElementSurfaceClearance,
    left: _CompiledElement,
    right: _CompiledElement,
    tolerance: float,
) -> float | ElementSurfaceFormationObstruction:
    declaring, counterpart = (
        (left, right)
        if declaration.declaring_element_id == left.element_id
        else (right, left)
    )
    record = _solid_record(declaring)
    vertices = np.asarray(record.vertices, dtype=np.float64)
    triangles = vertices[np.asarray(record.faces, dtype=np.int64)]
    centroids = np.mean(triangles, axis=1)
    areas = 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    field = E.sample_graph_field(counterpart.graph, centroids)
    if isinstance(field, E.RejectedSurfaceFormation):
        return ElementSurfaceFormationObstruction(
            counterpart.element_id,
            field.obstructions,
        )
    fitted_area = float(
        np.sum(
            areas[
                field
                <= declaration.maximum_clearance_world + tolerance
            ]
        )
    )
    total_area = float(np.sum(areas))
    return fitted_area / total_area if total_area > 0.0 else 0.0


def _fit_receipt(
    left: _CompiledElement,
    right: _CompiledElement,
    fit_law: ElementFitLaw,
    declaration: _ElementFitDeclaration | None,
    left_field: np.ndarray,
    right_field: np.ndarray,
    maximum_penetration: float,
    tolerance: float,
    penetrating_count: int,
    fitted_surface_fraction: float | None = None,
    required_fitted_surface_fraction: float | None = None,
) -> ElementFitReceipt:
    sampled_field = np.concatenate((left_field, right_field), axis=0)
    minimum_clearance = (
        _minimum_declared_clearance(
            declaration, left, left_field, right_field
        )
        if declaration is not None
        else float(
            np.min(np.maximum(sampled_field, 0.0), initial=math.inf)
        )
    )
    return ElementFitReceipt(
        left_element_id=left.element_id,
        right_element_id=right.element_id,
        fit_law=fit_law,
        declaring_element_id=(
            declaration.declaring_element_id
            if declaration is not None
            else None
        ),
        minimum_clearance_world=(
            minimum_clearance if math.isfinite(minimum_clearance) else None
        ),
        permitted_clearance_world=(
            None
            if declaration is None
            else 2.0 * tolerance
            if isinstance(declaration, _BoundedElementContact)
            else declaration.maximum_clearance_world
        ),
        maximum_penetration_world=maximum_penetration,
        tolerance_world=tolerance,
        penetrating_sample_count=penetrating_count,
        fitted_surface_fraction=fitted_surface_fraction,
        required_fitted_surface_fraction=required_fitted_surface_fraction,
    )


def _element_pair_fit_geometry(
    left: _CompiledElement,
    right: _CompiledElement,
) -> _ElementPairFitGeometry | ElementSurfaceFormationObstruction:
    tolerance = 0.25 * min(
        left.evaluated.maximum_pitch,
        right.evaluated.maximum_pitch,
    )
    left_vertices = _vertices_in_graph_bounds(
        _solid_vertices(left), right.graph, tolerance
    )
    right_vertices = _vertices_in_graph_bounds(
        _solid_vertices(right), left.graph, tolerance
    )
    left_field = E.sample_graph_field(right.graph, left_vertices)
    if isinstance(left_field, E.RejectedSurfaceFormation):
        return ElementSurfaceFormationObstruction(
            right.element_id,
            left_field.obstructions,
        )
    right_field = E.sample_graph_field(left.graph, right_vertices)
    if isinstance(right_field, E.RejectedSurfaceFormation):
        return ElementSurfaceFormationObstruction(
            left.element_id,
            right_field.obstructions,
        )
    left_penetrating = left_field < -tolerance
    right_penetrating = right_field < -tolerance
    penetrating_count = int(
        np.count_nonzero(left_penetrating)
        + np.count_nonzero(right_penetrating)
    )
    declaration = _declared_fit_for_pair(left, right)
    fit_law = (
        ElementFitLaw.DISJOINT
        if declaration is None
        else ElementFitLaw.BOUNDED_MOUNT_CONTACT
        if isinstance(declaration, _BoundedElementContact)
        else ElementFitLaw.SURFACE_CLEARANCE
    )
    fitted_surface_fraction = (
        _surface_clearance_area_fraction(
            declaration, left, right, tolerance
        )
        if isinstance(declaration, _ElementSurfaceClearance)
        else None
    )
    if isinstance(
        fitted_surface_fraction,
        ElementSurfaceFormationObstruction,
    ):
        return fitted_surface_fraction
    required_fitted_surface_fraction = (
        _MINIMUM_FITTED_SURFACE_FRACTION
        if isinstance(declaration, _ElementSurfaceClearance)
        else None
    )
    maximum_penetration = float(
        max(
            np.max(-left_field, initial=0.0),
            np.max(-right_field, initial=0.0),
            0.0,
        )
    )
    return _ElementPairFitGeometry(
        tolerance=tolerance,
        left_vertices=left_vertices,
        right_vertices=right_vertices,
        left_field=left_field,
        right_field=right_field,
        left_penetrating=left_penetrating,
        right_penetrating=right_penetrating,
        penetrating_count=penetrating_count,
        declaration=declaration,
        fit_law=fit_law,
        fitted_surface_fraction=fitted_surface_fraction,
        required_fitted_surface_fraction=required_fitted_surface_fraction,
        maximum_penetration=maximum_penetration,
    )
