"""Identical judging procedure for both conditions.

Occupancy is computed on a voxelization of the *solid* (surface voxels +
hole fill), so intersecting-but-unmerged shells still count as connected.
This deliberately does NOT penalize the control for shell count.
"""

import numpy as np
import trimesh
from scipy import ndimage

PITCH = 0.035


def occupancy(mesh):
    vg = mesh.voxelized(pitch=PITCH)
    grid = np.asarray(vg.matrix, dtype=bool)
    solid = ndimage.binary_fill_holes(grid)
    return solid, vg


def report(mesh, name):
    solid, vg = occupancy(mesh)
    labels, n = ndimage.label(solid)
    sizes = np.sort(np.bincount(labels.ravel())[1:])[::-1]

    # symmetry: IoU of occupied voxel centers vs their x-mirror
    pts = vg.points[ndimage.binary_fill_holes(np.asarray(vg.matrix, dtype=bool))[
        tuple(np.round((vg.points - vg.translation) / PITCH).astype(int).T)]]
    occ = {tuple(p) for p in np.round(pts / PITCH).astype(int)}
    mir = {(-x, y, z) for (x, y, z) in occ}
    sym_iou = len(occ & mir) / len(occ | mir)

    shells = mesh.split(only_watertight=False)
    real_shells = [s for s in shells if len(s.faces) >= 10]

    print(f"== {name}")
    print(f"   solid connected components: {n}  (sizes: {sizes[:5].tolist()})")
    print(f"   mesh shells: {len(real_shells)}")
    print(f"   watertight single solid: {len(real_shells) == 1 and real_shells[0].is_watertight}")
    print(f"   bilateral symmetry IoU: {sym_iou:.3f}")


if __name__ == "__main__":
    import json
    from engine import evaluate
    from control_creature import build

    graph = json.load(open("golem_v4.json"))
    v, f, *_ = evaluate(graph, res=190)
    constrained = trimesh.Trimesh(vertices=v, faces=f, process=True)
    control = build()

    report(constrained, "CONSTRAINED (part graph + solver, golem_v4)")
    report(control, "CONTROL (free primitives code, round 4)")
