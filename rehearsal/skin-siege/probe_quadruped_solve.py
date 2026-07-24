# Fast separated-limb quadruped probe for the Wall 004 cure. A torso blob
# with four leg gencyls dropping from the hips: adjacent legs leave a surface
# gap far wider than twice the skin thickness, so outward rays from a leg's
# inner face re-enter the torso or a neighbour exactly as Wall 004 requires,
# while the flesh stays one connected component. Prints the full-solve verdict
# at a modest authoring resolution. Not a test; a tuning oracle.
from __future__ import annotations

import sys
import time

from golem.kernel import engine

def quadruped_graph(thickness: float, blend: float) -> dict:
    return {
        "name": "skin-siege-quadruped",
        "blend": blend,
        "parts": [
            {
                "id": "torso",
                "type": "blob",
                "center": (0.0, 0.5, 0.0),
                "size": (0.45, 0.12, 0.22),
            },
            {
                "id": "leg_fl",
                "type": "gencyl",
                "spine": ((-0.35, 0.5, 0.15), (-0.35, 0.05, 0.15)),
                "radii": (0.05, 0.05),
            },
            {
                "id": "leg_fr",
                "type": "gencyl",
                "spine": ((-0.35, 0.5, -0.15), (-0.35, 0.05, -0.15)),
                "radii": (0.05, 0.05),
            },
            {
                "id": "leg_bl",
                "type": "gencyl",
                "spine": ((0.35, 0.5, 0.15), (0.35, 0.05, 0.15)),
                "radii": (0.05, 0.05),
            },
            {
                "id": "leg_br",
                "type": "gencyl",
                "spine": ((0.35, 0.5, -0.15), (0.35, 0.05, -0.15)),
                "radii": (0.05, 0.05),
            },
        ],
        "skin": {
            "formation": "static_implicit_relaxation",
            "thickness": thickness,
            "region_ids": ("host",),
        },
    }


def main() -> None:
    res = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    thickness = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    blend = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    graph = quadruped_graph(thickness, blend)
    start = time.perf_counter()
    evaluated = engine.evaluate(graph, res=res)
    seconds = time.perf_counter() - start
    print(f"res={res} thickness={thickness} blend={blend} solve={seconds:.2f}s")
    if isinstance(evaluated, engine.RejectedSurfaceFormation):
        for obstruction in evaluated.obstructions:
            failure = getattr(obstruction, "failure", obstruction)
            inner = getattr(failure, "failure", failure)
            print("REJECTED", type(inner).__name__)
            for name in (
                "vertex_indices",
                "maximum_displacement",
                "locality_radius",
                "minimum_root_count",
                "maximum_root_count",
                "maximum_residual",
                "tolerance",
                "minimum_ratio",
                "maximum_ratio",
            ):
                if hasattr(inner, name):
                    value = getattr(inner, name)
                    if name == "vertex_indices":
                        print(f"  {name}: {len(value)}")
                    else:
                        print(f"  {name}: {value}")
        return
    ev = evaluated
    print(
        "ACCEPTED",
        f"vertices={len(ev.vertices)}",
        f"faces={len(ev.faces)}",
    )
    e = ev.skin_evidence
    print(
        "  min/max stretch",
        round(e.minimum_edge_stretch, 4),
        round(e.maximum_edge_stretch, 4),
    )
    print("  min orientation", round(e.minimum_face_orientation, 5))
    print("  min area ratio", round(e.minimum_face_area_ratio, 5))
    print("  max offset residual", e.maximum_offset_residual)
    print("  fixed point residual", e.fixed_point_residual)
    print("  max projection displacement", e.maximum_projection_displacement)
    print("  root multiplicity", e.minimum_root_multiplicity, e.maximum_root_multiplicity)


if __name__ == "__main__":
    main()
