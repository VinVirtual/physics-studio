"""Scripted inverse kinematics: damped least squares on the end-effector site, with joint limits.

Used only to plan joint targets before the run; the run itself is pure physics with motors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray            # arm joint values
    pos_error: float         # metres
    rot_error: float         # radians (approximate)


def rotation_error(R: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """Small-angle rotation vector taking R towards R_target (both 3x3, columns are axes)."""
    return 0.5 * sum(np.cross(R[:, i], R_target[:, i]) for i in range(3))


def down_frame(finger_axis) -> np.ndarray:
    """Target orientation: tool z axis straight down, finger opening axis along finger_axis (horizontal)."""
    x = np.array([finger_axis[0], finger_axis[1], 0.0], float)
    n = np.linalg.norm(x)
    if n < 1e-9:
        raise ValueError("finger axis must have a horizontal component")
    x /= n
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def solve(model, data, site_id: int, qadr: list[int], dofadr: list[int], limits: list[tuple | None],
          target_pos, target_rot: np.ndarray | None, q0, iters: int = 300, damping: float = 0.02,
          rot_weight: float = 0.35) -> IKResult:
    """Solve on a scratch MjData (data is overwritten). Deterministic: same input, same output."""
    import mujoco

    q = np.array(q0, float)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    target_pos = np.asarray(target_pos, float)
    for _ in range(iters):
        for a, v in zip(qadr, q):
            data.qpos[a] = v
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        pos = data.site_xpos[site_id].copy()
        R = data.site_xmat[site_id].reshape(3, 3)
        e_pos = target_pos - pos
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = jacp[:, dofadr]
        e = e_pos
        if target_rot is not None:
            e = np.concatenate([e_pos, rot_weight * rotation_error(R, target_rot)])
            J = np.vstack([J, rot_weight * jacr[:, dofadr]])
        if np.linalg.norm(e) < 1e-6:
            break
        JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
        dq = J.T @ np.linalg.solve(JJt, e)
        step = np.linalg.norm(dq)
        if step > 0.2:
            dq *= 0.2 / step
        q = q + dq
        for i, lim in enumerate(limits):
            if lim is not None:
                q[i] = min(max(q[i], lim[0]), lim[1])
    for a, v in zip(qadr, q):
        data.qpos[a] = v
    mujoco.mj_kinematics(model, data)
    pos = data.site_xpos[site_id]
    R = data.site_xmat[site_id].reshape(3, 3)
    rot_err = float(np.linalg.norm(rotation_error(R, target_rot))) if target_rot is not None else 0.0
    return IKResult(q=q, pos_error=float(np.linalg.norm(target_pos - pos)), rot_error=rot_err)


def min_jerk(s: float) -> float:
    """0..1 -> 0..1 with zero velocity and acceleration at both ends."""
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def min_jerk_rate(s: float) -> float:
    """d(min_jerk)/ds."""
    s = min(max(s, 0.0), 1.0)
    return 30 * s * s * (1 - s) ** 2
