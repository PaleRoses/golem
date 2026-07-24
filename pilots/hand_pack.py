"""The hand constraint pack: anatomical predicates + behavioral pose tests.

Every check is a pure function on the spec/pose — no meshing, no rendering.
Violations are named, local, and quantified: the feedback surface the LLM
edits against. Norms report; they do not reject (tiered constraints:
invariants / norms / briefs — this pack is norms + behavioral tests).
"""

import numpy as np
from hand import pose_chains, POSES
from engine import sdf_blob

MIN_TIP_RADIUS = 0.05        # invariant: grid feature-size lesson from the claws


def _violation(rule, where, msg, value=None):
    return {"rule": rule, "where": where, "msg": msg, "value": value}


# ---------- anatomical predicates (norms) ----------

def check_anatomy(spec):
    v = []
    fingers = {f["id"]: f for f in spec["fingers"]}
    order = ["index", "middle", "ring", "pinky"]

    if set(fingers) != set(order):
        v.append(_violation("finger_count", "hand",
                            f"expected fingers {order}, got {sorted(fingers)}"))
        return v
    thumb = spec["thumb"]

    # 1. phalanx ratios ~2/3, each segment shorter than the last
    for fid in order:
        ph = fingers[fid]["phalanges"]
        for i in range(1, len(ph)):
            r = ph[i] / ph[i - 1]
            if not (0.55 <= r <= 0.90):
                v.append(_violation("phalanx_ratio", f"{fid}/seg{i}",
                                    f"ratio {r:.2f} outside [0.55, 0.90]", r))

    # 2. taper: radii non-increasing, tip above grid feature size
    for fid in order + ["thumb"]:
        radii = (fingers[fid]["radii"] if fid != "thumb" else thumb["radii"])
        for i in range(1, len(radii)):
            if radii[i] > radii[i - 1] + 1e-9:
                v.append(_violation("taper_monotone", f"{fid}/r{i}",
                                    f"radius grows {radii[i-1]:.3f}->{radii[i]:.3f}"))
        if radii[-1] < MIN_TIP_RADIUS:
            v.append(_violation("min_feature", f"{fid}/tip",
                                f"tip radius {radii[-1]:.3f} < {MIN_TIP_RADIUS}",
                                radii[-1]))

    # 3. length profile: middle longest; index≈ring; pinky clearly shortest
    tot = {fid: sum(fingers[fid]["phalanges"]) for fid in order}
    if not (tot["middle"] >= tot["index"] and tot["middle"] >= tot["ring"]
            and tot["middle"] > tot["pinky"]):
        v.append(_violation("length_profile", "hand",
                            f"middle must be longest: {dict((k, round(x,2)) for k,x in tot.items())}"))
    if abs(tot["index"] - tot["ring"]) > 0.15 * tot["middle"]:
        v.append(_violation("length_profile", "index/ring",
                            f"index and ring should be near-equal "
                            f"({tot['index']:.2f} vs {tot['ring']:.2f})"))
    if tot["pinky"] > 0.85 * tot["middle"]:
        v.append(_violation("length_profile", "pinky",
                            f"pinky {tot['pinky']:.2f} too long vs middle {tot['middle']:.2f}"))

    # 4. proportion: middle finger ~ palm length
    ratio = tot["middle"] / spec["palm"]["length"]
    if not (0.85 <= ratio <= 1.2):
        v.append(_violation("palm_proportion", "middle",
                            f"middle/palm {ratio:.2f} outside [0.85, 1.2]", ratio))

    # 5. splay fan: monotone across the knuckle line, no crossings, bounded
    xs = sorted(order, key=lambda fid: fingers[fid]["knuckle_x"])
    splays = [fingers[fid]["splay_deg"] for fid in xs]
    for a, b, fa, fb in zip(splays, splays[1:], xs, xs[1:]):
        if b - a < 1.5:
            v.append(_violation("splay_fan", f"{fa}->{fb}",
                                f"splay must open along knuckle line ({a:.1f} -> {b:.1f})"))
    if max(abs(s) for s in splays) > 22:
        v.append(_violation("splay_fan", "hand", "splay exceeds 22 deg"))

    # 6. curl-plane parallelism across fingers (open pose axes)
    chains = pose_chains(spec, POSES["open"])
    axes = [chains[fid][2] for fid in order]
    for i in range(len(axes) - 1):
        ang = np.degrees(np.arccos(np.clip(axes[i] @ axes[i + 1], -1, 1)))
        if ang > 20:
            v.append(_violation("curl_parallel", f"{order[i]}/{order[i+1]}",
                                f"curl axes diverge {ang:.0f} deg > 20"))

    # 7. thumb: divergent base direction + opposed curl plane
    if not (35 <= thumb["yaw_deg"] <= 80):
        v.append(_violation("thumb_divergence", "thumb",
                            f"yaw {thumb['yaw_deg']} outside [35, 80]"))
    if not (25 <= thumb["opposition_deg"] <= 70):
        v.append(_violation("thumb_opposition", "thumb",
                            f"opposition {thumb['opposition_deg']} outside [25, 70]"))

    return v


# ---------- behavioral pose tests ----------

def _seg_dist(p1, q1, p2, q2):
    """Min distance between segments p1q1 and p2q2."""
    d1, d2 = q1 - p1, q2 - p2
    r = p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c = d1 @ r
    b = d1 @ d2
    denom = a * e - b * b
    s = np.clip((b * f - c * e) / denom, 0, 1) if denom > 1e-12 else 0.0
    t = (b * s + f) / e if e > 1e-12 else 0.0
    t = np.clip(t, 0, 1)
    s = np.clip((b * t - c) / a, 0, 1) if a > 1e-12 else 0.0
    c1, c2 = p1 + d1 * s, p2 + d2 * t
    return float(np.linalg.norm(c1 - c2))


def _capsules(chains):
    caps = []
    for fid, (pts, radii, _) in chains.items():
        for i in range(len(pts) - 1):
            caps.append((fid, i, pts[i], pts[i + 1],
                         (radii[i] + radii[i + 1]) / 2))
    return caps


def _palm_sdf(spec, pts):
    palm = spec["palm"]
    size = np.array([palm["width"] / 2, palm["thickness"] / 2, palm["length"] / 2])
    return sdf_blob(np.atleast_2d(pts), [0, 0, 0], size)


def check_pose(spec, pose_name):
    v = []
    chains = pose_chains(spec, POSES[pose_name])
    caps = _capsules(chains)

    # (a) inter-finger interpenetration (blend tolerance 30%)
    for i in range(len(caps)):
        for j in range(i + 1, len(caps)):
            fa, ia, pa, qa, ra = caps[i]
            fb, ib, pb, qb, rb = caps[j]
            if fa == fb:
                continue
            d = _seg_dist(pa, qa, pb, qb)
            pen = (ra + rb) - d
            if pen > 0.30 * (ra + rb):
                v.append(_violation("interpenetration", f"{pose_name}:{fa}{ia}/{fb}{ib}",
                                    f"overlap {pen:.3f} (radii {ra:.2f}+{rb:.2f})", pen))

    # (b) fingers must not bury into the palm (root segments exempt)
    for fid, i, p, q, r in caps:
        if i == 0:
            continue
        mid_sdf = float(_palm_sdf(spec, (p + q) / 2)[0])
        if mid_sdf < -0.4 * r:
            v.append(_violation("palm_burial", f"{pose_name}:{fid}{i}",
                                f"segment buried in palm (sdf {mid_sdf:.3f})"))

    # (c) pose-specific contacts
    tips = {fid: (pts[-1], radii[-1]) for fid, (pts, radii, _) in chains.items()}
    if pose_name == "fist":
        for fid in ["index", "middle", "ring", "pinky"]:
            tip, r = tips[fid]
            gap = float(_palm_sdf(spec, tip)[0]) - r
            if gap > 0.12:
                v.append(_violation("fist_contact", f"fist:{fid}",
                                    f"fingertip {gap:.3f} short of palm", gap))
            if gap < -0.6 * r:
                v.append(_violation("fist_contact", f"fist:{fid}",
                                    f"fingertip buried in palm ({gap:.3f})", gap))
    if pose_name == "point":
        # index straight; curled fingers near palm
        for fid in ["middle", "ring", "pinky"]:
            tip, r = tips[fid]
            gap = float(_palm_sdf(spec, tip)[0]) - r
            if gap > 0.2:
                v.append(_violation("point_curl", f"point:{fid}",
                                    f"tip {gap:.3f} from palm", gap))
    if pose_name == "open":
        # clearance: no contacts at all between fingers
        pass  # covered by (a)

    return v


def run_pack(spec, poses=("open", "fist", "point")):
    report = {"anatomy": check_anatomy(spec)}
    for p in poses:
        report[p] = check_pose(spec, p)
    total = sum(len(x) for x in report.values())
    return report, total


def print_report(report):
    total = 0
    for section, violations in report.items():
        status = "PASS" if not violations else f"{len(violations)} violations"
        print(f"[{section}] {status}")
        for x in violations:
            print(f"   - {x['rule']} @ {x['where']}: {x['msg']}")
        total += len(violations)
    print(f"TOTAL: {total}")
    return total


if __name__ == "__main__":
    import json, sys
    spec = json.load(open(sys.argv[1]))
    report, _ = run_pack(spec)
    print_report(report)
