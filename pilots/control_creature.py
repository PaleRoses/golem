"""CONTROL CONDITION: unconstrained code-with-primitives (LL3M/Blender-python style).

Rules of the condition: free Python + trimesh primitives and transforms.
NO typed part graph, NO SDF smooth-min solver, NO mirror flag, NO port constraints.
The model places geometry by explicit numbers and matrices, like Blender scripting.
"""

import numpy as np
import trimesh
from trimesh.creation import icosphere, capsule
from trimesh.geometry import align_vectors


def seg(a, b, r1, r2):
    """Place a capsule from a to b. trimesh capsules are uniform-radius,
    so taper must be faked with the mean radius (a limitation of the substrate)."""
    a, b = np.array(a, float), np.array(b, float)
    v = b - a
    L = np.linalg.norm(v)
    r = (r1 + r2) / 2.0
    c = capsule(height=max(L - 2 * r, 0.01), radius=r)
    T = align_vectors([0, 0, 1], v / L)
    T[:3, 3] = (a + b) / 2
    c.apply_transform(T)
    return c


def ball(center, size):
    s = icosphere(subdivisions=3, radius=1.0)
    s.apply_scale(size)
    s.apply_translation(center)
    return s


def build():
    parts = []

    # torso: multiple overlapping balls to fake a tapered mass (round 2 workaround)
    parts.append(ball([0, 0.95, -0.7], [0.55, 0.52, 0.6]))
    parts.append(ball([0, 1.3, -0.25], [0.68, 0.65, 0.7]))
    parts.append(ball([0, 1.6, 0.15], [0.75, 0.7, 0.72]))
    # face dome, sunk deeper into torso + filler to hide the crease
    parts.append(ball([0, 1.62, 0.35], [0.68, 0.66, 0.5]))
    parts.append(ball([0, 1.6, 0.55], [0.58, 0.58, 0.4]))

    # left arm (+x): filler balls at elbow/wrist to hide capsule seams
    parts.append(ball([0.95, 1.75, 0.28], [0.52, 0.52, 0.52]))
    parts.append(seg([1.0, 1.7, 0.35], [1.5, 1.05, 0.45], 0.4, 0.4))
    parts.append(ball([1.5, 1.05, 0.45], [0.46, 0.46, 0.46]))
    parts.append(seg([1.5, 1.05, 0.45], [1.28, 0.45, 0.95], 0.55, 0.55))
    parts.append(ball([1.28, 0.5, 0.9], [0.55, 0.52, 0.55]))
    parts.append(ball([1.18, 0.3, 1.25], [0.5, 0.34, 0.55]))
    # claws: two segments each to fake taper (round 3)
    parts.append(seg([0.9, 0.28, 1.5], [0.82, 0.14, 1.75], 0.19, 0.19))
    parts.append(seg([0.85, 0.19, 1.66], [0.75, 0.0, 2.0], 0.11, 0.11))
    parts.append(seg([1.18, 0.3, 1.55], [1.18, 0.15, 1.85], 0.21, 0.21))
    parts.append(seg([1.18, 0.2, 1.74], [1.18, 0.0, 2.15], 0.12, 0.12))
    parts.append(seg([1.46, 0.28, 1.45], [1.54, 0.14, 1.68], 0.19, 0.19))
    parts.append(seg([1.51, 0.19, 1.6], [1.62, 0.0, 1.9], 0.11, 0.11))

    # right arm (-x), authored by hand
    parts.append(ball([-0.95, 1.75, 0.28], [0.52, 0.52, 0.52]))
    parts.append(seg([-1.0, 1.7, 0.35], [-1.5, 1.05, 0.45], 0.4, 0.4))
    parts.append(ball([-1.5, 1.05, 0.45], [0.46, 0.46, 0.46]))
    parts.append(seg([-1.5, 1.05, 0.45], [-1.28, 0.45, 0.95], 0.55, 0.55))
    parts.append(ball([-1.28, 0.5, 0.9], [0.55, 0.52, 0.55]))
    parts.append(ball([-1.18, 0.3, 1.25], [0.5, 0.34, 0.55]))
    parts.append(seg([-0.9, 0.28, 1.5], [-0.82, 0.14, 1.75], 0.19, 0.19))
    parts.append(seg([-0.85, 0.19, 1.66], [-0.75, 0.0, 2.0], 0.11, 0.11))
    parts.append(seg([-1.18, 0.3, 1.55], [-1.18, 0.15, 1.85], 0.21, 0.21))
    parts.append(seg([-1.18, 0.2, 1.74], [-1.18, 0.0, 2.15], 0.12, 0.12))
    parts.append(seg([-1.46, 0.28, 1.45], [-1.54, 0.14, 1.68], 0.19, 0.19))
    parts.append(seg([-1.51, 0.19, 1.6], [-1.62, 0.0, 1.9], 0.11, 0.11))

    # rear legs
    parts.append(ball([0.42, 0.6, -0.85], [0.4, 0.38, 0.44]))
    parts.append(seg([0.48, 0.55, -0.85], [0.6, 0.3, -0.45], 0.28, 0.2))
    parts.append(ball([0.62, 0.14, -0.5], [0.24, 0.14, 0.36]))
    parts.append(ball([-0.42, 0.6, -0.85], [0.4, 0.38, 0.44]))
    parts.append(seg([-0.48, 0.55, -0.85], [-0.6, 0.3, -0.45], 0.28, 0.2))
    parts.append(ball([-0.62, 0.14, -0.5], [0.24, 0.14, 0.36]))

    return trimesh.util.concatenate(parts)


if __name__ == "__main__":
    import sys
    from render import render_view, contact_sheet
    tag = sys.argv[1]
    mesh = build()
    views = [render_view(mesh.vertices, mesh.faces, az) for az in (0, 40, 90, 180)]
    contact_sheet(views, ["front (0°)", "3/4 (40°)", "side (90°)", "back (180°)"],
                  f"out/{tag}_views.png")
    mesh.export(f"out/{tag}.glb")
    print("shells:", len(mesh.split(only_watertight=False)))
