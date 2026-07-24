"""Procedural climber builder — Obsidian Warden (reach-specialized).

Coordinate contract: spec is authored in G-coords (y-up, z-forward, bilateral
about x=0, surface radii are TRUE surface distances). Mapping to Blender
(z-up): G(x, y, z) -> B(x, -z, y). glTF export with +Y up round-trips exactly
back to G-coords.

Metaball calibration (threshold 0.6, stiffness 3.0, resolution 0.05):
  BALL surface semi-extent   = BALL_K * element.radius
  ELLIPSOID surface semi-axis = ELL_K * element.size_i
So element values = desired surface value / K.
"""
import bpy, json, math, os
from mathutils import Vector

BALL_K = 0.6407   # measured in-session at res 0.05 (lab value 0.6434)
ELL_K  = 0.6420   # measured in-session, PER UNIT element.radius (must set radius=1.0)
RES    = 0.05
THRESH = 0.6
STIFF  = 3.0
ARENA  = '/Users/bluerose/Developer/pale-meridian/potentialimprovements/golem-kernel/rehearsal/arena/blender'
MASTER = '/Users/bluerose/Developer/pale-meridian/potentialimprovements/golem-kernel/pilots/golem_hands.json'

HAND_REGION_ZG_MIN = 0.9   # G z beyond this -> hand cluster
FEET_REGION_ZG_MAX = -0.2  # G z below this -> hindquarters/feet


def g2b(p):
    return Vector((p[0], -p[2], p[1]))


def wipe_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    for mb in list(bpy.data.metaballs):
        bpy.data.metaballs.remove(mb)
    for me in list(bpy.data.meshes):
        if me.users == 0:
            bpy.data.meshes.remove(me)


def make_spec_initial():
    """Adapt the family master toward the climber brief:
    longest forearms, largest hands, compact torso."""
    m = json.load(open(MASTER))
    mp = {p['id']: p for p in m['parts']}
    mpalm = Vector(mp['hand_palm']['center'])
    S = 1.10                      # hand enlargement vs family master
    palm_off = Vector((0.0, -0.15, 0.15))  # palm center rel wrist (hand local frame)
    hand_parts = []
    hp = mp['hand_palm']
    hand_parts.append({'id': 'hand_palm', 'type': 'blob',
                       'center_local': list(palm_off + (Vector(hp['center']) - mpalm) * S),
                       'size': [v * S for v in hp['size']]})
    for did in ['hand_index', 'hand_middle', 'hand_ring', 'hand_pinky', 'hand_thumb']:
        d = mp[did]
        hand_parts.append({'id': did, 'type': 'gencyl',
                           'spine_local': [list(palm_off + (Vector(pt) - mpalm) * S) for pt in d['spine']],
                           'radii': [r * S for r in d['radii']]})
    spec = {
        'name': 'climber', 'stage': 'initial',
        'parts': [
            {'id': 'torso', 'type': 'gencyl', 'mirror': False,
             'spine': [[0, 0.95, -0.65], [0, 1.50, -0.20], [0, 1.78, 0.25]],
             'radii': [0.46, 0.64, 0.68]},
            {'id': 'face_plate', 'type': 'blob', 'mirror': False,
             'center': [0, 1.66, 0.62], 'size': [0.56, 0.60, 0.28]},
            {'id': 'shoulder', 'type': 'blob', 'mirror': True,
             'center': [0.95, 1.78, 0.25], 'size': [0.50, 0.50, 0.50]},
            {'id': 'upper_arm', 'type': 'gencyl', 'mirror': True,
             'spine': [[1.00, 1.72, 0.30], [1.45, 1.18, 0.50]], 'radii': [0.34, 0.30]},
            {'id': 'forearm', 'type': 'gencyl', 'mirror': True,
             'spine': [[1.45, 1.18, 0.50], [1.30, 0.75, 1.28]], 'radii': [0.30, 0.26]},
            {'id': 'hip', 'type': 'blob', 'mirror': True,
             'center': [0.40, 0.62, -0.72], 'size': [0.38, 0.36, 0.42]},
            {'id': 'thigh', 'type': 'gencyl', 'mirror': True,
             'spine': [[0.44, 0.55, -0.72], [0.58, 0.28, -0.50]], 'radii': [0.26, 0.19]},
            {'id': 'foot', 'type': 'blob', 'mirror': True,
             'center': [0.58, 0.16, -0.52], 'size': [0.24, 0.14, 0.36]},
        ],
        'hand': {'wrist': [1.30, 0.75, 1.28], 'scale2': 1.0, 'parts': hand_parts},
    }
    return spec


def expand_parts(spec):
    """Spec -> flat list of world-G parts (hand locals instantiated at wrist)."""
    parts = []
    W = Vector(spec['hand']['wrist'])
    s2 = spec['hand'].get('scale2', 1.0)
    for p in spec['parts']:
        q = json.loads(json.dumps(p))
        if q['id'] == 'forearm':
            q['spine'][-1] = list(W)  # forearm distal end == wrist junction, always
        parts.append(q)
    for hp in spec['hand']['parts']:
        q = {'id': hp['id'], 'type': hp['type'], 'mirror': True}
        if hp['type'] == 'blob':
            q['center'] = list(W + Vector(hp['center_local']) * s2)
            q['size'] = [v * s2 for v in hp['size']]
        else:
            q['spine'] = [list(W + Vector(pt) * s2) for pt in hp['spine_local']]
            q['radii'] = [r * s2 for r in hp['radii']]
        parts.append(q)
    return parts


def build_controls(parts):
    """Named procedural control groups: one collection per part id, empties per
    spine point / blob center, surface radii stored as custom props."""
    root = bpy.data.collections.new('CLIMBER_CTRL')
    bpy.context.scene.collection.children.link(root)
    for p in parts:
        c = bpy.data.collections.new('CTRL_' + p['id'])
        root.children.link(c)
        c['mirror'] = bool(p.get('mirror', False))
        c['part_type'] = p['type']
        if p['type'] == 'blob':
            e = bpy.data.objects.new('CTRL_%s_c' % p['id'], None)
            e.empty_display_type = 'SPHERE'
            e.location = g2b(p['center'])
            e['size'] = list(p['size'])
            e.empty_display_size = max(p['size'])
            c.objects.link(e)
        else:
            for i, (pt, r) in enumerate(zip(p['spine'], p['radii'])):
                e = bpy.data.objects.new('CTRL_%s_p%02d' % (p['id'], i), None)
                e.empty_display_type = 'SPHERE'
                e.location = g2b(pt)
                e['surf_r'] = float(r)
                e.empty_display_size = r
                c.objects.link(e)
    return root


def sample_chain(pts, rs):
    out = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ra, rb = rs[i], rs[i + 1]
        d = (b - a).length
        step = max(0.03, 0.45 * min(ra, rb))
        n = max(2, int(math.ceil(d / step)) + 1)
        k0 = 0 if i == 0 else 1
        for k in range(k0, n):
            t = k / (n - 1)
            out.append((a.lerp(b, t), ra + (rb - ra) * t))
    return out


def gen_metaball_from_controls(root):
    """Generate the implicit surface FROM the control empties (controls drive
    the build). Elements in world coords, object at identity."""
    mb = bpy.data.metaballs.new('climber_mball')
    mb.resolution = RES
    mb.render_resolution = RES
    mb.threshold = THRESH
    ob = bpy.data.objects.new('climber_mball', mb)
    geo = bpy.data.collections.new('CLIMBER_GEO')
    bpy.context.scene.collection.children.link(geo)
    geo.objects.link(ob)
    n_elem = 0
    for c in root.children:
        signs = [1.0, -1.0] if c.get('mirror', False) else [1.0]
        if c['part_type'] == 'blob':
            e = c.objects[0]
            loc = e.location.copy()
            size = list(e['size'])  # G-order (sx, sy, sz)
            for s in signs:
                el = mb.elements.new(type='ELLIPSOID')
                el.co = Vector((loc.x * s, loc.y, loc.z))
                el.radius = 1.0               # ellipsoid semi = ELL_K * radius * size_i
                el.size_x = size[0] / ELL_K   # G x -> B x
                el.size_y = size[2] / ELL_K   # G z -> B y
                el.size_z = size[1] / ELL_K   # G y -> B z
                el.stiffness = STIFF
                n_elem += 1
        else:
            es = sorted(c.objects, key=lambda o: o.name)
            pts = [o.location.copy() for o in es]
            rs = [float(o['surf_r']) for o in es]
            for s in signs:
                for (pt, r) in sample_chain(pts, rs):
                    el = mb.elements.new(type='BALL')
                    el.co = Vector((pt.x * s, pt.y, pt.z))
                    el.radius = r / BALL_K
                    el.stiffness = STIFF
                    n_elem += 1
    return ob, n_elem


def to_mesh_and_measure(mb_ob):
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(mb_ob.evaluated_get(deps))
    me.name = 'climber_mesh'
    mesh_ob = bpy.data.objects.new('climber_mesh', me)
    bpy.data.collections['CLIMBER_GEO'].objects.link(mesh_ob)
    xs = [v.co.x for v in me.vertices]
    ys = [v.co.y for v in me.vertices]
    zs = [v.co.z for v in me.vertices]
    hand_min = min((v.co.z for v in me.vertices if v.co.y < -HAND_REGION_ZG_MIN), default=None)
    feet_min = min((v.co.z for v in me.vertices if v.co.y > -FEET_REGION_ZG_MAX), default=None)
    stats = {
        'verts': len(me.vertices), 'polys': len(me.polygons),
        'min_yG': min(zs), 'max_yG': max(zs),
        'x_range': [min(xs), max(xs)], 'x_asym': abs(min(xs) + max(xs)),
        'zG_range': [-max(ys), -min(ys)],
        'hand_min_yG': hand_min, 'feet_min_yG': feet_min,
    }
    return mesh_ob, stats


def build_stage(spec, stage):
    wipe_scene()
    spec['stage'] = stage
    parts = expand_parts(spec)
    root = build_controls(parts)
    mb_ob, n = gen_metaball_from_controls(root)
    mesh_ob, stats = to_mesh_and_measure(mb_ob)
    txt = bpy.data.texts.get('climber_spec') or bpy.data.texts.new('climber_spec')
    txt.clear()
    txt.write(json.dumps(spec, indent=1))
    os.makedirs(ARENA, exist_ok=True)
    json.dump(spec, open(os.path.join(ARENA, 'climber_spec_%s.json' % stage), 'w'), indent=1)
    stats['n_elements'] = n
    return stats


def shift_spec_y(spec, dy):
    """Global vertical shift baked into all spec coords (G y)."""
    for p in spec['parts']:
        if p['type'] == 'blob':
            p['center'][1] += dy
        else:
            for pt in p['spine']:
                pt[1] += dy
    spec['hand']['wrist'][1] += dy


def load_spec(stage):
    return json.load(open(os.path.join(ARENA, 'climber_spec_%s.json' % stage)))


def export_stage(stage):
    mesh_ob = bpy.data.objects['climber_mesh']
    for ob in bpy.data.objects:
        ob.select_set(False)
    mesh_ob.select_set(True)
    bpy.context.view_layer.objects.active = mesh_ob
    path = os.path.join(ARENA, 'climber_%s.glb' % stage)
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB',
                              use_selection=True, export_apply=True,
                              export_yup=True)
    return path
