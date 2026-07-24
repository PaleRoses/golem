"""Mounted reflection (D17): a single-sided equipment piece mounted with
``mirror_x`` must equal the exact geometric reflection of the unmounted piece
-- verified at the SDF level, exactly, on an asymmetric profiled part."""

import numpy as np

from golem.assembly import mount_graph
from golem.kernel import engine as E

_PIECE = {
    "name": "asym_arm",
    "blend": 0.02,
    "parts": [
        {
            "id": "vambrace",
            "type": "gencyl",
            "spine": [[0.325, 0.88, 0.07], [0.10, 0.60, 0.20]],
            "radii": [0.068, 0.052],
            "profile": {"n": 3.5, "aspect": 0.7, "up": [0, 0, 1]},
        },
        {
            "id": "gauntlet",
            "type": "blob",
            "center": [0.066, 0.558, 0.235],
            "size": [0.085, 0.075, 0.095],
            "rot": [0.9238795325112867, 0.0, 0.0, 0.3826834323650898],
        },
    ],
}


def test_mount_mirror_x_is_exact_reflection():
    mirrored = mount_graph(_PIECE, {"mirror_x": True})
    rng = np.random.RandomState(11)
    pts = rng.uniform(-0.6, 1.2, size=(4000, 3))
    for orig, refl in zip(_PIECE["parts"], mirrored["parts"]):
        d_refl = E.part_sdf(pts, refl)
        d_orig = E.part_sdf(pts * np.array([-1.0, 1.0, 1.0]), orig)
        assert float(np.max(np.abs(d_refl - d_orig))) == 0.0


def test_mount_mirror_x_composes_with_translate():
    t = [0.0, 0.5, -0.2]
    mirrored = mount_graph(_PIECE, {"mirror_x": True, "translate": t})
    rng = np.random.RandomState(12)
    pts = rng.uniform(-0.6, 1.4, size=(3000, 3))
    for orig, moved in zip(_PIECE["parts"], mirrored["parts"]):
        d_moved = E.part_sdf(pts, moved)
        d_ref = E.part_sdf((pts - np.array(t)) * np.array([-1.0, 1.0, 1.0]), orig)
        assert float(np.max(np.abs(d_moved - d_ref))) < 1e-12
