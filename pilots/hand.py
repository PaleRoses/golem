"""Hand sub-grammar: spec -> forward kinematics -> part-graph compile.

Local frame: palm center at origin, fingers extend +z, back of hand +y
(palm faces -y). Knuckle line near z = palm.length/2.

A hand spec is discrete and diffable, like the body graph:
{
  "palm": {"width": W, "length": L, "thickness": T},
  "fingers": [
    {"id": "index", "knuckle_x": -0.3, "splay_deg": -4,
     "phalanges": [l1, l2, l3], "radii": [r0, r1, r2, r3]},  # n+1 radii
    ...
  ],
  "thumb": {"root": [x, y, z], "yaw_deg": 55, "elev_deg": -15,
            "opposition_deg": 40, "phalanges": [l1, l2], "radii": [r0, r1, r2]}
}

A pose assigns curl angles per joint; the compiler emits the same JSON
part-graph dialect the body engine consumes. All continuous surfacing is
still solved (smooth-min SDF), never generated.
"""

import json
import numpy as np

UP = np.array([0.0, 1.0, 0.0])       # back of hand
FWD = np.array([0.0, 0.0, 1.0])      # finger direction at rest


def rodrigues(v, axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    th = np.radians(deg)
    v = np.asarray(v, float)
    return (v * np.cos(th) + np.cross(a, v) * np.sin(th)
            + a * (a @ v) * (1 - np.cos(th)))


def finger_chain(finger, palm, curls, splay_scale=1.0):
    """Forward kinematics: list of joint points + per-joint radii."""
    L = palm["length"]
    x = finger["knuckle_x"]
    root = np.array([x, 0.0, L / 2 * 0.92])
    d = rodrigues(FWD, UP, finger["splay_deg"] * splay_scale)
    curl_axis = np.cross(UP, d)                     # rotating d toward palm (-y)
    pts = [root]
    total = 0.0
    for length, ang in zip(finger["phalanges"], curls):
        total += ang
        di = rodrigues(d, curl_axis, -total)         # negative: curl toward -y
        pts.append(pts[-1] + di * length)
    return np.array(pts), list(finger["radii"]), curl_axis


def thumb_chain(thumb, curls):
    root = np.array(thumb["root"], float)
    d = rodrigues(FWD, UP, thumb["yaw_deg"])
    d = rodrigues(d, np.cross(UP, d), -thumb["elev_deg"])
    # opposition tilts the curl plane toward the palm center
    curl_axis = rodrigues(np.cross(UP, d), d, thumb["opposition_deg"])
    pts = [root]
    total = 0.0
    for length, ang in zip(thumb["phalanges"], curls):
        total += ang
        di = rodrigues(d, curl_axis, -total)
        pts.append(pts[-1] + di * length)
    return np.array(pts), list(thumb["radii"]), curl_axis


def pose_chains(spec, pose):
    """All chains for a pose: {id: (points, radii, curl_axis)}."""
    chains = {}
    ss = pose.get("splay_scale", 1.0)
    for f in spec["fingers"]:
        chains[f["id"]] = finger_chain(f, spec["palm"], pose["curls"][f["id"]], ss)
    chains["thumb"] = thumb_chain(spec["thumb"], pose["curls"]["thumb"])
    return chains


def compile_graph(spec, pose, blend=0.028, wrist_stub=True):
    """Emit the body engine's part-graph JSON for a posed hand."""
    palm = spec["palm"]
    parts = [{
        "id": "palm", "type": "blob", "center": [0, 0, 0],
        "size": [palm["width"] / 2, palm["thickness"] / 2, palm["length"] / 2],
    }]
    if wrist_stub:
        parts.append({"id": "wrist", "type": "gencyl",
                      "spine": [[0, 0, -palm["length"] / 2 - 0.35],
                                [0, 0, -palm["length"] / 2 + 0.1]],
                      "radii": [palm["width"] * 0.30, palm["width"] * 0.33]})
    for fid, (pts, radii, _) in pose_chains(spec, pose).items():
        parts.append({"id": fid, "type": "gencyl",
                      "spine": [p.tolist() for p in pts], "radii": radii})
    return {"name": "hand", "blend": blend, "parts": parts}


POSES = {
    "open":  {"splay_scale": 1.0,
              "curls": {"index": [4, 6, 4], "middle": [4, 6, 4],
                        "ring": [4, 6, 4], "pinky": [4, 6, 4],
                        "thumb": [8, 10]}},
    "fist":  {"splay_scale": 0.25,
              "curls": {"index": [80, 100, 45], "middle": [80, 100, 45],
                        "ring": [80, 100, 45], "pinky": [82, 100, 45],
                        "thumb": [40, 50]}},
    "point": {"splay_scale": 0.4,
              "curls": {"index": [4, 4, 4], "middle": [80, 100, 45],
                        "ring": [82, 100, 45], "pinky": [84, 100, 45],
                        "thumb": [35, 45]}},
    "knuckle_walk": {"splay_scale": 0.4,
              "curls": {"index": [55, 92, 30], "middle": [55, 92, 30],
                        "ring": [55, 92, 30], "pinky": [57, 92, 30],
                        "thumb": [58, 68]}},
}


if __name__ == "__main__":
    spec = json.load(open("hand_v1.json"))
    g = compile_graph(spec, POSES["open"])
    print(json.dumps(g, indent=1)[:400])
