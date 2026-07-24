"""Vascular-material mask carving over the evaluated morphology."""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

from golem.kernel.anatomy.material.types import (
    AcceptedVascularMaterial,
    ChannelInducedDisconnectionObstruction,
    EmptyVascularMaterialObstruction,
    RejectedVascularMaterial,
    UnresolvedLumenObstruction,
    UnresolvedVascularWallObstruction,
    VascularMaterialEscapeObstruction,
    VascularMaterialMasks,
)

if TYPE_CHECKING:
    import numpy as np

    from golem.kernel.engine import EvaluatedMorphology

    from golem.kernel.anatomy.graph import ClosedVascularGraph
    from golem.kernel.anatomy.material.types import VascularMaterialResult


def derive_vascular_material_masks(
    evaluated: "EvaluatedMorphology",
    graph: ClosedVascularGraph,
    *,
    wall_thickness: float,
    minimum_cells_across_radius: float = 2.0,
) -> VascularMaterialResult:
    """Carve a resolved vascular lumen and wall from the morphology field.

    Tiny channels are rejected before rasterization.  Inflating them to the
    grid would manufacture load-bearing geometry and counterfeit the channel's
    structural penalty, so a multiscale model must be selected explicitly
    rather than smuggled through a visual-radius floor.
    """
    import numpy as np
    from scipy import ndimage

    pitch = evaluated.maximum_pitch
    unresolved = tuple(
        edge
        for edge in graph.edges
        if edge.radius < minimum_cells_across_radius * pitch
    )
    if unresolved:
        limiting = min(unresolved, key=lambda edge: (edge.radius, edge.edge_id))
        return RejectedVascularMaterial(
            (
                UnresolvedLumenObstruction(
                    edge_id=limiting.edge_id,
                    radius=limiting.radius,
                    pitch=pitch,
                    required_cells=minimum_cells_across_radius,
                    unresolved_edge_count=len(unresolved),
                ),
            )
        )
    if wall_thickness < minimum_cells_across_radius * pitch:
        return RejectedVascularMaterial(
            (
                UnresolvedVascularWallObstruction(
                    wall_thickness=wall_thickness,
                    pitch=pitch,
                    required_cells=minimum_cells_across_radius,
                ),
            )
        )
    if not graph.edges:
        return RejectedVascularMaterial(
            (EmptyVascularMaterialObstruction("lumen"),)
        )

    axes = tuple(
        np.linspace(lower, upper, evaluated.resolution)
        for lower, upper in zip(evaluated.lower, evaluated.upper)
    )
    coordinate_fields = np.meshgrid(*axes, indexing="ij")
    points = np.stack(
        tuple(coordinate.ravel() for coordinate in coordinate_fields), axis=1
    )
    nodes = graph.node_by_id
    vascular_distance = reduce(
        np.minimum,
        map(
            lambda edge: _vascular_capsule_distance(
                points,
                np.asarray(nodes[edge.source_node_id].position, dtype=np.float64),
                np.asarray(nodes[edge.target_node_id].position, dtype=np.float64),
                edge.radius,
            ),
            graph.edges,
        ),
    ).reshape(evaluated.field.shape)
    body = evaluated.field <= 0.0
    raw_lumen = vascular_distance <= 0.0
    leaking_cells = int(np.count_nonzero(raw_lumen & ~body))
    if leaking_cells:
        return RejectedVascularMaterial(
            (VascularMaterialEscapeObstruction(leaking_cells),)
        )
    lumen = body & raw_lumen
    wall = body & ~lumen & (vascular_distance <= wall_thickness)
    empty_strata = tuple(
        name
        for name, mask in (("lumen", lumen), ("wall", wall))
        if not bool(np.any(mask))
    )
    if empty_strata:
        return RejectedVascularMaterial(
            tuple(map(EmptyVascularMaterialObstruction, empty_strata))
        )
    load_bearing_solid = body & ~lumen
    _labels, component_count = ndimage.label(
        load_bearing_solid, ndimage.generate_binary_structure(3, 1)
    )
    if component_count != 1:
        return RejectedVascularMaterial(
            (ChannelInducedDisconnectionObstruction(int(component_count)),)
        )
    sealed = tuple(
        _seal_material_mask(mask)
        for mask in (lumen, wall, load_bearing_solid)
    )
    return AcceptedVascularMaterial(
        VascularMaterialMasks(
            lumen=sealed[0],
            wall=sealed[1],
            load_bearing_solid=sealed[2],
            minimum_radius_to_pitch=min(edge.radius for edge in graph.edges)
            / pitch,
            minimum_wall_to_pitch=wall_thickness / pitch,
        )
    )


def _vascular_capsule_distance(points, start, end, radius: float):
    import numpy as np

    direction = end - start
    squared_length = float(direction @ direction)
    interpolation = (
        np.clip(((points - start) @ direction) / squared_length, 0.0, 1.0)
        if squared_length > 0.0
        else np.zeros(points.shape[0], dtype=np.float64)
    )
    closest = start + interpolation[:, None] * direction
    return np.linalg.norm(points - closest, axis=1) - radius


def _seal_material_mask(mask):
    mask.setflags(write=False)
    return mask
