"""Contact-pattern matching, identity-quaternion test, inline mount-pose normalization."""

from __future__ import annotations

import numpy as np


def _is_identity_quat(q: np.ndarray) -> bool:
    return abs(q[0] - 1.0) < 1e-9 and float(np.linalg.norm(q[1:])) < 1e-9


def _matches(pid: str, bid: str, pat: str) -> bool:
    """A contact pattern matches an emitted part. Patterns are bone names,
    exact part ids, or ``prefix/*`` globs (e.g. ``greatsword/*``)."""
    if pat == pid or pat == bid:
        return True
    if pat.endswith("/*") and pid.startswith(pat[:-1]):
        return True
    if "/" not in pat and pid.startswith(pat + "."):  # bone.flesh<i>
        return True
    return False


def _mount_pose(pose: dict) -> dict:
    """Normalize an inline mount pose ({splay_scale?, curls{...}}) for hand.py."""
    curls = pose.get("curls", {})
    return {"splay_scale": float(pose.get("splay_scale", 1.0)), "curls": curls}
