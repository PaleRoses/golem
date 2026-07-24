"""
Minimal LLM-native part-graph creature engine.

DSL (JSON):
{
  "name": "...",
  "blend": 0.06,              # global smooth-min radius (fraction of bbox diag)
  "parts": [
    {
      "id": "torso",
      "type": "gencyl",        # generalized cylinder: spine polyline + radius profile
      "spine": [[x,y,z], ...], # 2+ control points (world units, y-up, z-forward)
      "radii": [r0, r1, ...],  # one radius per spine point (linearly interpolated)
      "mirror": false,         # if true, also instanced with x -> -x
      "blend": 0.08            # optional per-part override of blend into the union
    },
    {
      "id": "shoulder_L",
      "type": "blob",          # ellipsoid
      "center": [x,y,z],
      "size": [sx,sy,sz],      # semi-axes
      "mirror": true
    }
  ]
}

Convention: +x = creature's LEFT, +y = up, +z = forward (facing +z).
mirror:true parts are authored on the +x side and auto-instanced on -x.

Everything continuous (blending, surfacing, normals) is solved, not generated.
"""

import json
import numpy as np
from skimage import measure


# ---------- SDF primitives ----------

def sdf_gencyl(pts, spine, radii):
    """Distance to a tapered capsule chain (generalized cylinder)."""
    d = np.full(pts.shape[0], np.inf)
    spine = np.asarray(spine, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    for i in range(len(spine) - 1):
        a, b = spine[i], spine[i + 1]
        ra, rb = radii[i], radii[i + 1]
        ab = b - a
        denom = float(ab @ ab)
        if denom < 1e-12:
            t = np.zeros(pts.shape[0])
        else:
            t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
        closest = a + t[:, None] * ab
        r = ra + (rb - ra) * t
        d = np.minimum(d, np.linalg.norm(pts - closest, axis=1) - r)
    return d


def sdf_blob(pts, center, size):
    """Approximate ellipsoid SDF (scaled-space distance, good enough for blending)."""
    c = np.asarray(center, dtype=np.float64)
    s = np.asarray(size, dtype=np.float64)
    q = (pts - c) / s
    k = np.linalg.norm(q, axis=1)
    return (k - 1.0) * float(np.min(s))


def part_sdf(pts, part, mirrored=False):
    if part["type"] == "gencyl":
        spine = np.array(part["spine"], dtype=np.float64)
        if mirrored:
            spine = spine * np.array([-1.0, 1.0, 1.0])
        return sdf_gencyl(pts, spine, part["radii"])
    if part["type"] == "blob":
        c = np.array(part["center"], dtype=np.float64)
        if mirrored:
            c = c * np.array([-1.0, 1.0, 1.0])
        return sdf_blob(pts, c, part["size"])
    raise ValueError(f"unknown part type {part['type']}")


def smin(a, b, k):
    """Polynomial smooth min."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


# ---------- graph -> field -> mesh ----------

def instances(graph):
    for part in graph["parts"]:
        yield part, False
        if part.get("mirror"):
            yield part, True


def graph_bounds(graph, pad=0.15):
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for part, mirrored in instances(graph):
        if part["type"] == "gencyl":
            spine = np.array(part["spine"], dtype=np.float64)
            if mirrored:
                spine = spine * np.array([-1, 1, 1])
            r = np.max(part["radii"])
            lo = np.minimum(lo, spine.min(axis=0) - r)
            hi = np.maximum(hi, spine.max(axis=0) + r)
        else:
            c = np.array(part["center"], dtype=np.float64)
            if mirrored:
                c = c * np.array([-1, 1, 1])
            s = np.array(part["size"], dtype=np.float64)
            lo = np.minimum(lo, c - s)
            hi = np.maximum(hi, c + s)
    span = hi - lo
    return lo - pad * span, hi + pad * span


def evaluate(graph, res=170):
    lo, hi = graph_bounds(graph)
    diag = float(np.linalg.norm(hi - lo))
    k_global = graph.get("blend", 0.05) * diag

    axes = [np.linspace(lo[i], hi[i], res) for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    field = None
    for part, mirrored in instances(graph):
        d = part_sdf(pts, part, mirrored)
        k = part.get("blend", None)
        k = k * diag if k is not None else k_global
        field = d if field is None else smin(field, d, k)

    F = field.reshape(res, res, res)
    spacing = [(hi[i] - lo[i]) / (res - 1) for i in range(3)]
    verts, faces, normals, _ = measure.marching_cubes(F, level=0.0, spacing=spacing)
    verts = verts + lo
    return verts, faces, normals, F, lo, hi


def coherence_report(verts, faces):
    """Connected components = the cheap global-coherence metric."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    comps = mesh.split(only_watertight=False)
    real = [c for c in comps if len(c.faces) >= 10]
    dust = len(comps) - len(real)
    vols = sorted((abs(c.volume) for c in real), reverse=True)
    return {
        "components": len(real),
        "grid_dust_slivers": dust,
        "watertight_main": real[0].is_watertight if real else False,
        "largest_component_volume_share": (vols[0] / sum(vols)) if vols else 0.0,
        "faces": len(faces),
    }


if __name__ == "__main__":
    import sys
    graph = json.load(open(sys.argv[1]))
    v, f, n, F, lo, hi = evaluate(graph)
    print(json.dumps(coherence_report(v, f), indent=2))
