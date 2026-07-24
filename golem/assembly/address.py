"""Assembly semantic-address projection over morphology provenance."""

from __future__ import annotations

import numpy as np

from golem.assembly.carriers import _CompiledElement
from golem.assembly.service import ElementAddress, RegionAddress, semantic_address_token
from golem.assembly.voxel import _ResolvedMaterialDomain
from golem.kernel import engine as E
from golem.kernel.anatomy import AcceptedAnatomy
from golem.kernel.mechanics import CellIndex


def _owner_provenance_at_points(
    element: _CompiledElement,
    points: np.ndarray,
) -> tuple[str, ...]:
    provenance = element.graph.get("intent", {}).get("provenance", {})
    instances = tuple(E.instances(element.graph))
    if not instances or not isinstance(provenance, dict):
        return ()
    instance_provenance = tuple(
        str(provenance.get(str(part.get("id", "")).removesuffix("_m"), ""))
        for part, _mirrored in instances
    )
    ownership = np.argmin(
        np.column_stack(
            tuple(
                E.part_sdf(points, part, mirrored=mirrored)
                for part, mirrored in instances
            )
        ),
        axis=1,
    )
    return tuple(
        instance_provenance[int(owner_index)] for owner_index in ownership
    )


def _domain_cell_indices_owned_by_provenance(
    element: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    provenance_fragment: str,
) -> tuple[int, ...]:
    owner_provenance = _owner_provenance_at_points(
        element, np.asarray(domain.cell_centers_world, dtype=np.float64)
    )
    return tuple(
        index
        for index, provenance in enumerate(owner_provenance)
        if provenance_fragment in provenance
    )


def _semantic_address_for_cell(
    element: _CompiledElement,
    domain: _ResolvedMaterialDomain,
    cell: CellIndex,
) -> str:
    fallback = semantic_address_token(ElementAddress(element.element_id))
    if not isinstance(element.anatomy, AcceptedAnatomy):
        return fallback
    cell_index = domain.mechanics_domain.solid_cells.index(cell)
    center = np.asarray((domain.cell_centers_world[cell_index],), dtype=np.float64)
    owner_provenance = _owner_provenance_at_points(element, center)
    if not owner_provenance:
        return fallback
    region = next(
        (
            candidate
            for candidate in element.anatomy.overall.regions
            if f"skeleton/{candidate.host_bone_id}/" in owner_provenance[0]
        ),
        None,
    )
    return (
        semantic_address_token(RegionAddress(element.element_id, region.region_id))
        if region is not None
        else fallback
    )
