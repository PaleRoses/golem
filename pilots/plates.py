"""Plate-tessellation layer: the first surface cell complex.

Decomposes the golem's surface into Voronoi plate cells, displaces plate
interiors outward (relief) and seam bands inward (grooves), and colors
seams emissive violet. The plate adjacency graph — cells, shared seam
edges, per-cell data — is exactly the base complex the appearance
sheaves live on; it is exported alongside the mesh.
"""

import json
import numpy as np
import trimesh
from engine import evaluate

RNG_SEED = 7
N_PLATES = 110
RELIEF = 0.045      # outward displacement of plate interiors
GROOVE = 0.02       # inward displacement of seam band
SEAM_RINGS = 0     # boundary edges only: thin seams


def farthest_point_seeds(verts, k, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(len(verts)))]
    d = np.linalg.norm(verts - verts[seeds[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        seeds.append(nxt)
        d = np.minimum(d, np.linalg.norm(verts - verts[nxt], axis=1))
    return np.array(seeds)


def main():
    graph = json.load(open("golem_v4.json"))
    v, f, *_ = evaluate(graph, res=190)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh = max(mesh.split(only_watertight=False), key=lambda c: len(c.faces))
    verts = mesh.vertices.copy()
    faces = mesh.faces

    # --- cells: Voronoi plate assignment per vertex (Euclidean approx) ---
    seed_idx = farthest_point_seeds(verts, N_PLATES)
    seed_pts = verts[seed_idx]
    # chunked nearest-seed
    label = np.empty(len(verts), dtype=np.int32)
    for i in range(0, len(verts), 20000):
        d = np.linalg.norm(verts[i:i + 20000, None, :] - seed_pts[None], axis=2)
        label[i:i + 20000] = np.argmin(d, axis=1)

    # --- seam detection: vertices adjacent to a different label ---
    edges = mesh.edges_unique
    diff = label[edges[:, 0]] != label[edges[:, 1]]
    seam = np.zeros(len(verts), dtype=bool)
    seam[edges[diff].ravel()] = True
    # widen seam band by SEAM_RINGS
    adj = {i: set() for i in range(len(verts))}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    band = seam.copy()
    frontier = set(np.nonzero(seam)[0])
    for _ in range(SEAM_RINGS):
        nxt = set()
        for i in frontier:
            nxt |= adj[i]
        newly = nxt - set(np.nonzero(band)[0])
        band[list(nxt)] = True
        frontier = newly

    # --- relief displacement along vertex normals ---
    normals = mesh.vertex_normals
    disp = np.where(band, -GROOVE, RELIEF)
    verts_out = verts + normals * disp[:, None]

    plated = trimesh.Trimesh(vertices=verts_out, faces=faces, process=False)

    # --- coloring: stone plates (per-plate value jitter) + emissive seams ---
    rng = np.random.default_rng(RNG_SEED)
    plate_val = 0.75 + 0.25 * rng.random(N_PLATES)          # per-plate brightness
    base = np.array([72, 74, 86], dtype=np.float64)          # dark obsidian
    seam_col = np.array([168, 85, 247], dtype=np.float64)    # emissive violet

    face_label = label[faces[:, 0]]
    face_seam = band[faces].any(axis=1)
    fc = base[None, :] * plate_val[face_label][:, None]
    fc[face_seam] = seam_col
    face_colors = np.hstack([fc, np.full((len(faces), 1), 255.0)]).astype(np.uint8)
    plated.visual.face_colors = face_colors

    plated.export("out/golem_plated.glb")

    # --- the cell complex itself: plates, adjacency, seam sizes ---
    pairs = {}
    for (a, b) in edges[diff]:
        key = tuple(sorted((int(label[a]), int(label[b]))))
        pairs[key] = pairs.get(key, 0) + 1
    complex_out = {
        "plates": int(N_PLATES),
        "plate_centers": np.round(seed_pts, 3).tolist(),
        "adjacency": [{"plates": list(k), "seam_edges": c} for k, c in sorted(pairs.items())],
    }
    json.dump(complex_out, open("out/plate_complex.json", "w"))
    print("plates:", N_PLATES, "| seam pairs:", len(pairs),
          "| seam faces:", int(face_seam.sum()), "/", len(faces))
    return plated, face_colors


if __name__ == "__main__":
    main()
