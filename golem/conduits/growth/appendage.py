"""Compile appendage declarations directly into authoritative part sequences."""

from __future__ import annotations

from itertools import chain as _chain
import numpy as np

from golem.conduits.growth.grow import (
    appendage_envelope_graph,
    consolidate_appendage_chains,
    grow_appendage_network,
)
from golem.conduits.surface.embed import part_station


def grow_appendage(
    intent_parts: list[dict] | tuple[dict, ...],
    root: np.ndarray,
    declaration: dict,
) -> tuple[dict, ...]:
    network = grow_appendage_network(
        intent_parts,
        root,
        count=int(declaration.get("count", 70)),
        seed=int(declaration.get("seed", 7)),
        step=float(declaration.get("step", 0.045)),
        influence=float(declaration.get("influence", 0.28)),
        kill=float(declaration.get("kill", 0.05)),
        min_radius=float(declaration.get("min_radius", 0.008)),
        root_radius=float(declaration.get("root_radius", 0.022)),
    )
    chains = consolidate_appendage_chains(
        network.nodes,
        network.parents,
        network.radii,
        simplify_tol=float(declaration.get("simplify_tol", 0.012)),
    )
    compiled_graph = appendage_envelope_graph(
        chains,
        envelope=float(declaration.get("envelope", 1.7)),
        min_radius=float(declaration.get("env_min_radius", 0.013)),
        tip_taper=float(declaration.get("tip_taper", 0.5)),
        smooth_iters=int(declaration.get("smooth_iters", 2)),
        envelope_tip=declaration.get("envelope_tip"),
    )
    return tuple(compiled_graph["parts"])


def _expanded_declaration_parts(
    declaration: dict, parts_by_id: dict[str, dict]
) -> tuple[dict, ...]:
    anchor = parts_by_id.get(declaration["anchor_part"])
    if anchor is None:
        raise ValueError(
            f"appendage {declaration.get('id')!r}: unknown anchor part "
            f"{declaration['anchor_part']!r}"
        )
    root, _tangent, _radius = part_station(
        anchor, float(declaration.get("anchor_t", 1.0))
    )
    offset_root = np.asarray(root, dtype=np.float64) + np.asarray(
        declaration.get("root_offset", (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    mirror = bool(declaration.get("mirror", False))
    blend = declaration.get("blend", 0.02)
    return tuple(
        {
            **part,
            "id": f"{declaration['id']}_{part['id']}",
            **({"mirror": True} if mirror else {}),
            **({"blend": float(blend)} if blend is not None else {}),
        }
        for part in grow_appendage(
            declaration["intent"], offset_root, declaration
        )
    )


def expand_appendages(graph: dict) -> dict:
    declarations = tuple(graph.get("appendages", ()))
    if not declarations:
        return graph
    parts_by_id = {part["id"]: part for part in graph["parts"]}
    expanded_parts = tuple(
        _chain.from_iterable(
            _expanded_declaration_parts(declaration, parts_by_id)
            for declaration in declarations
        )
    )
    return {
        **{
            key: value
            for key, value in graph.items()
            if key != "appendages"
        },
        "parts": [*graph["parts"], *expanded_parts],
    }


__all__ = ["expand_appendages", "grow_appendage"]
