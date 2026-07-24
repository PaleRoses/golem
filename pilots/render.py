"""Software renderer: lambert-shaded painter's-algorithm views + turntable GIF.

No GPU dependency; deterministic. Feedback surface for the agent loop.
"""

import json
import sys
import numpy as np
from PIL import Image

from engine import evaluate, coherence_report

LIGHT = np.array([0.45, 0.8, 0.55])
LIGHT2 = np.array([-0.6, 0.2, -0.5])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def render_view(verts, faces, azim_deg, elev_deg=12, size=560, tint=(168, 172, 182),
                face_colors=None):
    R = rot_x(np.radians(elev_deg)) @ rot_y(np.radians(azim_deg))
    v = verts @ R.T

    center = (v.max(axis=0) + v.min(axis=0)) / 2
    v = v - center
    scale = 0.44 * size / max(np.ptp(v[:, 0]), np.ptp(v[:, 1]))
    px = v[:, 0] * scale + size / 2
    py = size / 2 - v[:, 1] * scale
    depth = v[:, 2]

    tri = faces
    tv = np.stack([px[tri], py[tri]], axis=2)          # (F,3,2)
    tz = depth[tri].mean(axis=1)

    # face normals in view space
    a = v[tri[:, 1]] - v[tri[:, 0]]
    b = v[tri[:, 2]] - v[tri[:, 0]]
    fn = np.cross(a, b)
    nl = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(nl, 1e-12)

    l1 = LIGHT / np.linalg.norm(LIGHT)
    l2 = LIGHT2 / np.linalg.norm(LIGHT2)
    lam = 0.72 * np.abs(fn @ l1) + 0.22 * np.abs(fn @ l2) + 0.14
    lam = np.clip(lam, 0, 1)

    order = np.argsort(tz)  # back to front

    img = np.full((size, size, 3), 244, dtype=np.uint8)
    from PIL import ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    if face_colors is None:
        base = np.tile(np.array(tint, dtype=np.float64), (len(tri), 1))
    else:
        base = np.asarray(face_colors, dtype=np.float64)[:, :3]
    shaded = np.clip(base * lam[:, None], 0, 255).astype(int)
    for i in order:
        dr.polygon([tuple(p) for p in tv[i]], fill=tuple(shaded[i]))
    return im


def contact_sheet(views, labels, path):
    w, h = views[0].size
    sheet = Image.new("RGB", (w * len(views), h + 26), (244, 244, 244))
    from PIL import ImageDraw
    dr = ImageDraw.Draw(sheet)
    for i, (v, lab) in enumerate(zip(views, labels)):
        sheet.paste(v, (i * w, 26))
        dr.text((i * w + 10, 6), lab, fill=(40, 40, 40))
    sheet.save(path)


def turntable(verts, faces, path, frames=24, size=420):
    imgs = [render_view(verts, faces, az, size=size)
            for az in np.linspace(0, 360, frames, endpoint=False)]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=110, loop=0)


if __name__ == "__main__":
    graph_path, tag = sys.argv[1], sys.argv[2]
    graph = json.load(open(graph_path))
    verts, faces, *_ = evaluate(graph, res=int(sys.argv[3]) if len(sys.argv) > 3 else 170)
    views = [render_view(verts, faces, az) for az in (0, 40, 90, 180)]
    contact_sheet(views, ["front (0°)", "3/4 (40°)", "side (90°)", "back (180°)"],
                  f"out/{tag}_views.png")
    print(json.dumps(coherence_report(verts, faces), indent=2))
    if len(sys.argv) > 4 and sys.argv[4] == "gif":
        turntable(verts, faces, f"out/{tag}_turntable.gif")
        print("gif written")
