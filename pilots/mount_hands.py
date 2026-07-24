"""Port composition: mount the pack-passing hand sub-graph on the golem.

The wrist port carries the frame: fingers continue the forearm direction,
back of hand as vertical as the forearm allows, knuckle-walk pose, ground
contact solved by measurement (lift so knuckles kiss y=0). The hand parts
are authored on +x and instanced to both sides by the engine's mirror flag —
symmetry stays guaranteed by construction through composition.
"""

import json
import numpy as np
from hand import compile_graph, POSES

SCALE = 1.3
ELBOW = np.array([1.55, 1.05, 0.45])
WRIST = np.array([1.28, 0.55, 1.05])


def wrist_frame():
    """Knuckle-walk port frame: hand plants at ~45 degrees (like the
    reference's forearms-into-claws), back of hand facing forward-up."""
    z = np.array([0.0, -0.68, 0.73])          # local +z (finger rest dir)
    z = z / np.linalg.norm(z)
    fwd = np.array([0.0, 0.2, 1.0])
    y = fwd - (fwd @ z) * z                    # local +y (back of hand)
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)   # columns: local x,y,z in world


def transform_parts(parts, R, t, scale):
    out = []
    for p in parts:
        q = dict(p)
        if p["type"] == "gencyl":
            spine = np.array(p["spine"], float) * scale
            q["spine"] = (spine @ R.T + t).tolist()
            q["radii"] = [r * scale for r in p["radii"]]
        else:
            c = np.array(p["center"], float) * scale
            q["center"] = (R @ c + t).tolist()
            q["size"] = [s * scale for s in p["size"]]
        q["mirror"] = True
        q["id"] = "hand_" + p["id"]
        out.append(q)
    return out


def main():
    spec = json.load(open("hand_v2.json"))
    hand_graph = compile_graph(spec, POSES["knuckle_walk"], wrist_stub=False)
    R = wrist_frame()

    # provisional placement: palm center below-forward of the wrist
    t0 = WRIST + np.array([0.0, -0.25, 0.25])
    parts = transform_parts(hand_graph["parts"], R, t0, SCALE)

    # solve ground contact: lowest point (spine minus radius) should touch y≈0.02
    low = np.inf
    for p in parts:
        if p["type"] == "gencyl":
            for pt, r in zip(p["spine"], p["radii"]):
                low = min(low, pt[1] - r)
        else:
            low = min(low, p["center"][1] - p["size"][1])
    lift = 0.02 - low
    for p in parts:
        if p["type"] == "gencyl":
            for pt in p["spine"]:
                pt[1] += lift
        else:
            p["center"][1] += lift
    t_final = t0 + np.array([0.0, lift, 0.0])
    print(f"ground solve: lowest point {low:.3f}, lifted by {lift:.3f}")

    # bidirectional port: re-aim the forearm's distal end at the hand's
    # wrist junction so the arm meets the hand, not the other way round
    spec_palm_len = spec["palm"]["length"]
    junction = t_final - R[:, 2] * (spec_palm_len / 2 * SCALE + 0.12)
    golem = json.load(open("golem_v4.json"))
    golem["parts"] = [p for p in golem["parts"]
                      if p["id"] not in ("hand", "claw_inner", "claw_mid", "claw_outer")]
    for p in golem["parts"]:
        if p["id"] == "forearm":
            p["spine"][1] = np.round(junction, 3).tolist()
    print(f"forearm re-aimed at junction {np.round(junction, 2).tolist()}")
    golem["parts"] += parts
    golem["name"] = "obsidian-golem-with-hands"
    json.dump(golem, open("golem_hands.json", "w"), indent=1)
    print("parts:", len(golem["parts"]))


if __name__ == "__main__":
    main()
