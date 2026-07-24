#!/usr/bin/env python3
"""GOLEM native presentation backend -- THE authoritative judged sense (D16).

Consumes an exchange GLB scene + its ``.materials.json`` sidecar ONLY (per
plan D3 it imports nothing from ``golem``), shades each element from its
material record (roughness -> highlight tightness, metallic -> tinted
specular and dimmed diffuse, emissive -> glow), and renders the CONTRACT
views with a painter's-algorithm rasterizer plus fresnel rim and filmic-ish
tonemapping on the contract's dark stage.

This is GOLEM rendering GOLEM: deterministic, self-contained, CI-runnable. The
Blender backend (``blender_contract.py``) is the same CONTRACT with a real
renderer used only as an optional external cross-check (D16: GOLEM judges through its own sense).

Usage:
    python3 presentation/native.py <scene.glb> <out_prefix>
        [--views hero|contract] [--turntable] [--size N]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import CONTRACT  # noqa: E402


def _norm(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def load_scene(glb_path: str):
    """-> list of (name, verts, faces, record)."""
    import trimesh

    sidecar_path = Path(str(glb_path) + ".materials.json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {"elements": {}, "records": {}}
    scene = trimesh.load(glb_path)
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene({"model": scene})
    neutral = {
        "base_color": (0.40, 0.41, 0.44),
        "roughness": 0.6,
        "metallic": 0.0,
        "emissive": False,
        "emissive_color": (0, 0, 0),
        "emissive_strength": 0.0,
    }
    out = []
    for name, geom in scene.geometry.items():
        tag = sidecar["elements"].get(name)
        rec = sidecar["records"].get(tag, neutral) if tag else neutral
        out.append((name, np.asarray(geom.vertices), np.asarray(geom.faces), rec))
    return out


def stage_background(size: int) -> Image.Image:
    """Return the pinned native stage without scene geometry."""
    grad = np.linspace(0, 1, size)[:, None]
    bg = (
        (1 - grad) * np.asarray(CONTRACT["bg_top"])
        + grad * np.asarray(CONTRACT["bg_bottom"])
    )
    return Image.fromarray(
        np.repeat(bg[:, None, :], size, axis=1).astype(np.uint8)
    )


def _shade(elements, azim_deg, size, *, frame_elements=None):
    C = CONTRACT
    az, el = np.radians(azim_deg), np.radians(C["elev_deg"])
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    R = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]]) @ np.array(
        [[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]]
    )
    V = np.array([0.0, 0.0, 1.0])
    L1, L2 = _norm(C["key_dir"]), _norm(C["fill_dir"])
    H = _norm(L1 + V)
    all_tris, all_cols = [], []
    for _name, verts, faces, rec in elements:
        v = verts @ R.T
        tris = v[faces]
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        base = np.asarray(rec["base_color"], dtype=np.float64)
        rough = float(rec["roughness"])
        metal = float(rec["metallic"])
        spec_p = float(np.clip(2.0 / (rough * rough + 1e-3), 8, 240))
        spec_k = 0.18 + 0.55 * (1.0 - rough) + 0.25 * metal
        spec_tint = (1.0 - metal) * np.ones(3) + metal * (base / max(base.max(), 1e-3))
        ndv = np.abs(n @ V)
        diff = (
            C["key_energy"] * np.abs(n @ L1)
            + C["fill_energy"] * np.abs(n @ L2)
            + C["ambient"]
        ) * (1.0 - 0.7 * metal)
        spec = spec_k * np.clip(np.abs(n @ H), 0, 1) ** spec_p
        rim = C["rim_k"] * (1.0 - ndv) ** 3.2
        col = base * diff[:, None] + spec[:, None] * spec_tint
        col = col + rim[:, None] * np.asarray(C["rim_color"])
        if rec.get("emissive"):
            col = col + np.asarray(rec["emissive_color"]) * (
                0.35 * float(rec["emissive_strength"])
            )
        col = col / (1.0 + col)
        col = np.clip(col, 0, 1) ** (1 / 2.2)
        all_tris.append(tris)
        all_cols.append(col)
    tris = np.vstack(all_tris)
    col = np.vstack(all_cols)
    framing_triangles = (
        tris
        if frame_elements is None
        else np.vstack(
            tuple(
                (np.asarray(verts) @ R.T)[np.asarray(faces)]
                for _name, verts, faces, _record in frame_elements
            )
        )
    )
    lo = framing_triangles.reshape(-1, 3).min(axis=0)
    hi = framing_triangles.reshape(-1, 3).max(axis=0)
    c = (lo + hi) / 2
    scale = 0.86 * size / max(hi[0] - lo[0], hi[1] - lo[1])
    xy = (tris[:, :, :2] - c[:2]) * scale
    xy[:, :, 0] += size / 2
    xy[:, :, 1] = size / 2 - xy[:, :, 1]
    img = stage_background(size)
    dr = ImageDraw.Draw(img, "RGBA")
    foot_y = size / 2 - (lo[1] - c[1]) * scale
    w = (hi[0] - lo[0]) * scale * 0.55
    for i, a in ((3.2, 40), (2.2, 55), (1.35, 80)):
        dr.ellipse(
            [size / 2 - w / i, foot_y - w / (i * 7), size / 2 + w / i, foot_y + w / (i * 7)],
            fill=(0, 0, 0, a),
        )
    order = np.argsort(tris[:, :, 2].mean(axis=1))
    cols8 = (col * 255).astype(np.uint8)
    for fi in order:
        p = xy[fi]
        dr.polygon([tuple(p[0]), tuple(p[1]), tuple(p[2])], fill=tuple(cols8[fi]))
    return img


def _label(img, text):
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    dr.text((24, 18), text, fill=(225, 225, 235), font=font)
    return img


def render(glb_path: str, out_prefix: str, views: str = "hero",
           turntable: bool = False, size: int | None = None) -> list[str]:
    elements = load_scene(glb_path)
    size = size or CONTRACT["size"]
    azims = CONTRACT["hero_views_azim_deg"] if views == "hero" else CONTRACT["views_azim_deg"]
    paths = []
    imgs = []
    for az in azims:
        img = _shade(elements, az, size)
        p = f"{out_prefix}_az{az:03d}.png"
        img.save(p)
        paths.append(p)
        imgs.append(img)
    sheet = Image.new("RGB", (size * len(imgs), size))
    for i, im in enumerate(imgs):
        sheet.paste(im, (size * i, 0))
    _label(sheet, f"{Path(glb_path).stem} -- presentation/{views} ({CONTRACT['version']}, fallback backend)")
    sp = f"{out_prefix}_sheet.png"
    sheet.save(sp)
    paths.append(sp)
    if turntable:
        n, ts = CONTRACT["turntable_frames"], CONTRACT["turntable_size"]
        frames = [_shade(elements, 360.0 * i / n, ts) for i in range(n)]
        tp = f"{out_prefix}_turntable.gif"
        frames[0].save(tp, save_all=True, append_images=frames[1:], duration=90, loop=0)
        paths.append(tp)
    return paths


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    views = "contract" if "--views" in argv and argv[argv.index("--views") + 1] == "contract" else "hero"
    size = int(argv[argv.index("--size") + 1]) if "--size" in argv else None
    paths = render(argv[0], argv[1], views=views, turntable="--turntable" in argv, size=size)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
