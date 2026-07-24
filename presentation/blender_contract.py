"""Blender presentation backend -- OPTIONAL external cross-check (D16: the native backend is authoritative).

STATUS HONESTY: written 2026-07-18 against the Blender 4.x Python API but not
yet executed (no Blender exists in the authoring sandbox). First run belongs
to a machine with Blender >= 4.0 on PATH; treat the first successful sheet as
this file's acceptance, and record any API repair in the plan's Surprises.

Same CONTRACT as the fallback backend; consumes an exchange GLB + its
``.materials.json`` sidecar only.

Usage:
    blender --background --python presentation/blender_contract.py -- \
        --in <scene.glb> --out <dir> [--views hero|contract] [--turntable]
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import CONTRACT  # noqa: E402

try:
    import bpy
    from mathutils import Vector
except ImportError:  # pragma: no cover
    sys.exit("This script runs inside Blender: blender --background --python presentation/blender_contract.py -- ...")


def _args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    def val(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default
    return {
        "in": val("--in"),
        "out": val("--out", "presentation/out"),
        "views": val("--views", "contract"),
        "turntable": "--turntable" in argv,
    }


def _set_principled(mat, rec):
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    def set_input(names, value):
        for name in names:
            if name in bsdf.inputs:
                bsdf.inputs[name].default_value = value
                return
    set_input(["Base Color"], list(rec["base_color"]) + [1.0])
    set_input(["Roughness"], float(rec["roughness"]))
    set_input(["Metallic"], float(rec["metallic"]))
    if rec.get("emissive"):
        set_input(["Emission Color", "Emission"], list(rec["emissive_color"]) + [1.0])
        set_input(["Emission Strength"], float(rec["emissive_strength"]))


def main():
    a = _args()
    out_dir = Path(a["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = Path(a["in"] + ".materials.json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {"elements": {}, "records": {}}

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    b = CONTRACT["blender"]
    scn.render.engine = b["engine"]
    scn.cycles.samples = b["samples"]
    scn.view_settings.view_transform = b["view_transform"]
    scn.render.resolution_x = scn.render.resolution_y = CONTRACT["size"]
    scn.render.film_transparent = False
    world = bpy.data.worlds.new("stage")
    scn.world = world
    world.use_nodes = True
    bgc = [c / 255.0 for c in CONTRACT["bg_bottom"]]
    world.node_tree.nodes["Background"].inputs[0].default_value = bgc + [1.0]

    bpy.ops.import_scene.gltf(filepath=a["in"])
    meshes = [o for o in scn.objects if o.type == "MESH"]
    for obj in meshes:
        tag = sidecar["elements"].get(obj.name)
        rec = sidecar["records"].get(tag) if tag else None
        if rec:
            mat = bpy.data.materials.new(f"golem_{tag}_{obj.name}")
            _set_principled(mat, rec)
            obj.data.materials.clear()
            obj.data.materials.append(mat)

    lo = Vector((min(min(v.co[i] for v in o.data.vertices) + o.location[i] for o in meshes) for i in range(3)))
    hi = Vector((max(max(v.co[i] for v in o.data.vertices) + o.location[i] for o in meshes) for i in range(3)))
    center, span = (lo + hi) / 2, max((hi - lo)) * b["camera_fit_margin"]

    def add_sun(direction, energy):
        light = bpy.data.lights.new("sun", type="SUN")
        light.energy = energy * 3.0
        obj = bpy.data.objects.new("sun", light)
        scn.collection.objects.link(obj)
        d = Vector(direction).normalized()
        obj.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
        return obj

    add_sun(CONTRACT["key_dir"], CONTRACT["key_energy"])
    add_sun(CONTRACT["fill_dir"], CONTRACT["fill_energy"])

    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = span
    cam = bpy.data.objects.new("cam", cam_data)
    scn.collection.objects.link(cam)
    scn.camera = cam
    elev = math.radians(CONTRACT["elev_deg"])
    dist = span * 3.0

    def place(azim_deg):
        az = math.radians(azim_deg)
        offset = Vector((math.sin(az) * math.cos(elev), -math.cos(az) * math.cos(elev), math.sin(elev)))
        # glTF import is +Z up in Blender; exchange meshes are y-up, importer rotates.
        cam.location = center + offset * dist
        look = center - cam.location
        cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()

    azims = CONTRACT["hero_views_azim_deg"] if a["views"] == "hero" else CONTRACT["views_azim_deg"]
    for az in azims:
        place(az)
        scn.render.filepath = str(out_dir / f"az{az:03d}.png")
        bpy.ops.render.render(write_still=True)
    if a["turntable"]:
        scn.render.resolution_x = scn.render.resolution_y = CONTRACT["turntable_size"]
        n = CONTRACT["turntable_frames"]
        for i in range(n):
            place(360.0 * i / n)
            scn.render.filepath = str(out_dir / f"turn{i:03d}.png")
            bpy.ops.render.render(write_still=True)
    print("presentation/blender_contract: DONE ->", out_dir)


if __name__ == "__main__":
    main()
