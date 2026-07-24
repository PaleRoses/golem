"""Small deterministic linear algebra (float64; no randomness anywhere)."""

from __future__ import annotations

import math

import numpy as np

from golem.kernel.body.types import _PARALLEL_COS


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def _rodrigues(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    """Rotate ``v`` about unit-ish ``axis`` by ``deg`` degrees."""
    a = _normalize(axis)
    th = math.radians(deg)
    v = np.asarray(v, dtype=np.float64)
    return (v * math.cos(th) + np.cross(a, v) * math.sin(th)
            + a * (a @ v) * (1.0 - math.cos(th)))


def _rot_axis(deg: float, axis: int) -> np.ndarray:
    """Elementary rotation matrix about world axis 0/1/2 by ``deg`` degrees."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    R = np.eye(3)
    if axis == 0:  # x
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    elif axis == 1:  # y
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    else:  # z
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return R


def _pose_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """R_pose = Ry(yaw) Rx(pitch) Rz(roll), degrees, in the rest-local frame."""
    return _rot_axis(yaw, 1) @ _rot_axis(pitch, 0) @ _rot_axis(roll, 2)


def _frame_from_axis(z_world: np.ndarray, up_hint_world: np.ndarray,
                     fallback_world: np.ndarray, twist_deg: float) -> np.ndarray:
    """Build a right-handed frame (columns x,y,z) whose +z is ``z_world`` and
    whose +y derives from ``up_hint_world`` projected orthogonal to +z (or
    ``fallback_world`` when the hint is within 1 degree of the axis), then
    twisted about +z. Reproduces ``mount_hands.wrist_frame``'s construction."""
    z = _normalize(z_world)
    hint = _normalize(up_hint_world)
    if abs(float(hint @ z)) > _PARALLEL_COS:
        hint = _normalize(fallback_world)
    y = hint - (hint @ z) * z
    y = _normalize(y)
    if twist_deg:
        y = _normalize(_rodrigues(y, z, twist_deg))
    x = np.cross(y, z)
    x = _normalize(x)
    y = np.cross(z, x)  # re-orthogonalize
    return np.stack([x, y, z], axis=1)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Unit quaternion [w,x,y,z], canonical w>=0, of a rotation matrix."""
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    q = q / float(np.linalg.norm(q))
    if q[0] < 0.0:
        q = -q
    return q


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle in degrees between two vectors."""
    cu, cv = _normalize(u), _normalize(v)
    return math.degrees(math.acos(float(np.clip(cu @ cv, -1.0, 1.0))))
