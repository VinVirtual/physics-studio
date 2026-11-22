"""Controllers for each task. Each one is planned once before the run and then queried every step.

- pick_place / push: joint targets from scripted IK, followed by a PD controller plus gravity and
  Coriolis compensation (the motor torques shown in the video are what the motors really apply).
- cartpole: LQR around upright, from MuJoCo's own linearisation (finite differences).
- pendulum / drop: no control.
Plan events (reach, close, lift, ...) are reported with the time they start.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import RecipeError
from .ik import down_frame, min_jerk, min_jerk_rate, solve


def full_mass_matrix(m, d, out: np.ndarray) -> None:
    """Dense joint-space inertia matrix; the call changed signature across MuJoCo versions."""
    import mujoco
    try:
        mujoco.mj_fullM(m, d, out)
    except TypeError:
        mujoco.mj_fullM(m, out, d.qM)


@dataclass
class Segment:
    t0: float
    t1: float
    knots: list[np.ndarray]     # arm joint values along the segment (2 for a joint-space move)
    g0: np.ndarray
    g1: np.ndarray
    label: str


@dataclass
class Plan:
    segments: list[Segment] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    unreachable: list[dict] = field(default_factory=list)
    place_target: list[float] | None = None
    push_target: list[float] | None = None

    def end(self) -> float:
        return self.segments[-1].t1 if self.segments else 0.0


def sample_segment(seg: Segment, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Desired arm q, arm qdot and gripper q at time t inside (or clamped to) the segment."""
    dur = max(seg.t1 - seg.t0, 1e-9)
    u = (t - seg.t0) / dur
    s = min_jerk(u)
    ds = min_jerk_rate(u) / dur if 0 <= u <= 1 else 0.0
    n = len(seg.knots) - 1
    x = s * n
    i = min(int(x), n - 1)
    f = x - i
    q = seg.knots[i] * (1 - f) + seg.knots[i + 1] * f
    qd = (seg.knots[i + 1] - seg.knots[i]) * n * ds
    g = seg.g0 * (1 - s) + seg.g1 * s
    return q, qd, g


class ArmController:
    def __init__(self, m, d, recipe: dict, scene, world) -> None:
        import mujoco
        self.m = m
        self.recipe = recipe
        r = recipe["robot"]
        self.kp, self.kd = r["kp"], r["kd"]
        self.arm_q = [world.qadr(j) for j in scene.arm]
        self.arm_v = [world.dofadr(j) for j in scene.arm]
        self.grip_q = [world.qadr(j) for j in scene.gripper]
        self.grip_v = [world.dofadr(j) for j in scene.gripper]
        self.arm_act = [world.actuator(f"ps_act_{j}") for j in scene.arm]
        self.grip_act = [world.actuator(f"ps_act_{j}") for j in scene.gripper]
        self.limits = [world.joint_range(j) for j in scene.arm]
        g = r.get("gripper")
        self.g_open = np.array(g["open"]) if g else np.zeros(0)
        self.g_closed = np.array(g["closed"]) if g else np.zeros(0)
        self.grip_force = g["force"] if g else 0.0
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ps_ee")
        for j, v in (r.get("home") or {}).items():
            d.qpos[world.qadr(j)] = v
        for a, v in zip(self.grip_q, self.g_open):
            d.qpos[a] = v
        mujoco.mj_forward(m, d)
        self.M = np.zeros((m.nv, m.nv))
        self.scratch = mujoco.MjData(m)
        self.scratch.qpos[:] = d.qpos
        self.q_home = np.array([d.qpos[a] for a in self.arm_q])
        self.base = np.array(d.xpos[world.body("ps_robot")])
        self.plan = Plan()
        if recipe["task"] == "pick_place":
            self._plan_pick(world, scene)
        else:
            self._plan_push(world, scene)

    # --- planning -----------------------------------------------------------------------------
    def _ik(self, pos, rot, q0, label: str) -> np.ndarray:
        res = solve(self.m, self.scratch, self.site, self.arm_q, self.arm_v, self.limits, pos, rot, q0)
        if res.pos_error > 0.01 or res.rot_error > 0.15:
            self.plan.unreachable.append({"waypoint": label, "pos_error_m": round(res.pos_error, 4),
                                          "rot_error_rad": round(res.rot_error, 3)})
        return res.q

    def _finger_axis(self, obj_xy, obj_yaw_deg: float, along=None) -> np.ndarray:
        radial = np.array(obj_xy, float) - self.base[:2]
        radial = radial / (np.linalg.norm(radial) or 1.0)
        tangent = np.array([-radial[1], radial[0]])
        if along is not None:      # pushing: fingers across the push direction
            a = np.array([-along[1], along[0]])
            return np.array([*a, 0.0]) * (1 if a @ tangent >= 0 else -1)
        yaw = math.radians(obj_yaw_deg)
        cands = [np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])]
        best = max(cands, key=lambda c: abs(c @ tangent))
        if best @ tangent < 0:
            best = -best
        return np.array([*best, 0.0])

    def _add(self, dur: float, knots, g0, g1, label: str) -> None:
        t0 = self.plan.end() if self.plan.segments else 0.0
        self.plan.segments.append(Segment(t0, t0 + dur, [np.array(k) for k in knots], np.array(g0), np.array(g1), label))
        self.plan.events.append({"t": round(t0, 4), "type": label})

    def _plan_pick(self, world, scene) -> None:
        goal = self.recipe["goal"]
        obj = next(o for o in self.recipe["objects"] if o["id"] == goal["object"])
        c = np.array(scene.object_start[obj["id"]])
        hh = c[2] - scene.surface[obj["id"]]
        px, py = goal["place"]
        place_surface = world.surface_at(px, py)
        place_c = np.array([px, py, place_surface + hh])
        self.plan.place_target = [round(px, 6), round(py, 6), round(float(place_c[2]), 6)]
        lift = goal["lift"]
        R1 = down_frame(self._finger_axis(c[:2], obj["yaw_deg"]))
        R2 = down_frame(self._finger_axis(place_c[:2], obj["yaw_deg"]))
        go, gc = self.g_open, self.g_closed
        q = self.q_home
        q_pre = self._ik(c + [0, 0, lift + 0.05], R1, q, "above object")
        q_grasp = self._ik(c, R1, q_pre, "grasp")
        q_lift = self._ik(c + [0, 0, lift], R1, q_grasp, "lift")
        q_over = self._ik(place_c + [0, 0, lift + 0.004], R2, q_lift, "above target")
        q_down = self._ik(place_c + [0, 0, 0.004], R2, q_over, "set down")
        q_up = self._ik(place_c + [0, 0, lift + 0.05], R2, q_down, "retreat")
        self._add(0.3, [q, q], go, go, "settle")
        self._add(1.3, [q, q_pre], go, go, "reach")
        self._add(0.8, [q_pre, q_grasp], go, go, "descend")
        self._add(0.6, [q_grasp, q_grasp], go, gc, "close")
        self._add(0.9, [q_grasp, q_lift], gc, gc, "lift")
        self._add(1.3, [q_lift, q_over], gc, gc, "carry")
        self._add(0.8, [q_over, q_down], gc, gc, "lower")
        self._add(0.5, [q_down, q_down], gc, go, "open")
        self._add(0.8, [q_down, q_up], go, go, "retreat")

    def _plan_push(self, world, scene) -> None:
        goal = self.recipe["goal"]
        obj = next(o for o in self.recipe["objects"] if o["id"] == goal["object"])
        c = np.array(scene.object_start[obj["id"]])
        a = math.radians(goal["direction_deg"])
        d = np.array([math.cos(a), math.sin(a), 0.0])
        s = obj["size"]
        reach = extent_along(obj, d[:2])
        pad = 0.02
        target = c + d * goal["distance"]
        self.plan.push_target = [round(float(target[0]), 6), round(float(target[1]), 6)]
        R = down_frame(self._finger_axis(c[:2], 0, along=d[:2]))
        z = np.array([0, 0, 0.004])
        start = c - d * (reach + pad + 0.03) + z
        end = c + d * goal["distance"] - d * (reach + pad) + z
        g = self.g_closed if len(self.g_closed) else np.zeros(0)
        q = self.q_home
        q_pre = self._ik(start + [0, 0, 0.12], R, q, "above start")
        q_start = self._ik(start, R, q_pre, "push start")
        knots, qk = [q_start], q_start
        for i in range(1, 13):
            qk = self._ik(start + (end - start) * i / 12, R, qk, f"push {i}/12")
            knots.append(qk)
        q_back = self._ik(end - d * 0.05 + [0, 0, 0.1], R, qk, "retreat")
        go = self.g_open if len(self.g_open) else g
        self._add(0.3, [q, q], go, g, "settle")
        self._add(1.3, [q, q_pre], g, g, "reach")
        self._add(0.8, [q_pre, q_start], g, g, "descend")
        self._add(max(1.2, goal["distance"] / 0.08), knots, g, g, "push")
        self._add(0.9, [qk, q_back], g, g, "retreat")

    # --- control ------------------------------------------------------------------------------
    def desired(self, t: float):
        segs = self.plan.segments
        seg = next((s for s in segs if s.t0 <= t < s.t1), segs[-1])
        q, qd, g = sample_segment(seg, t)
        if t >= segs[-1].t1:
            qd = qd * 0
        return q, qd, g

    def control(self, m, d, t: float) -> None:
        import mujoco
        q, qd, g = self.desired(t)
        full_mass_matrix(m, d, self.M)
        Ma = self.M[np.ix_(self.arm_v, self.arm_v)]
        e = np.array([q[i] - d.qpos[a] for i, a in enumerate(self.arm_q)])
        ed = np.array([qd[i] - d.qvel[v] for i, v in enumerate(self.arm_v)])
        taus = Ma @ (self.kp * e + self.kd * ed)      # computed torque: gains act on acceleration
        for i, (v, act) in enumerate(zip(self.arm_v, self.arm_act)):
            tau = taus[i] + d.qfrc_bias[v]
            lo, hi = m.actuator_ctrlrange[act]
            d.ctrl[act] = min(max(tau, lo), hi)
        for i, (a, v, act) in enumerate(zip(self.grip_q, self.grip_v, self.grip_act)):
            f = 2000.0 * (g[i] - d.qpos[a]) - 30.0 * d.qvel[v]
            f = min(max(f, -self.grip_force), self.grip_force) + d.qfrc_bias[v]
            lo, hi = m.actuator_ctrlrange[act]
            d.ctrl[act] = min(max(f, lo), hi)


def extent_along(obj: dict, direction) -> float:
    """Half-width of an object's footprint along a horizontal unit direction (support function)."""
    s = obj["size"]
    if obj["shape"] != "box":
        return float(s[0])
    yaw = math.radians(obj["yaw_deg"])
    ex = np.array([math.cos(yaw), math.sin(yaw)])
    ey = np.array([-math.sin(yaw), math.cos(yaw)])
    d = np.asarray(direction, float)
    return float(s[0] * abs(d @ ex) + s[1] * abs(d @ ey))


def dare(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, iters: int = 5000) -> np.ndarray:
    """Discrete algebraic Riccati equation by fixed-point iteration (no SciPy needed)."""
    P = Q.copy()
    for _ in range(iters):
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        Pn = Q + A.T @ P @ (A - B @ K)
        if np.max(np.abs(Pn - P)) < 1e-10 * max(1.0, np.max(np.abs(P))):
            P = Pn
            break
        P = Pn
    return P


def lqr_gain(A, B, Q, R) -> np.ndarray:
    P = dare(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


class CartpoleController:
    def __init__(self, m, d, recipe: dict, scene, world) -> None:
        import mujoco
        c = recipe["cartpole"]
        self.cfg = c
        self.act = world.actuator("ps_act_slide")
        self.hinge_q = world.qadr("hinge")
        self.plan = Plan()
        self.K = None
        if c["controller"] == "lqr":
            d0 = mujoco.MjData(m)
            mujoco.mj_forward(m, d0)
            nx = 2 * m.nv
            A = np.zeros((nx, nx))
            B = np.zeros((nx, m.nu))
            mujoco.mjd_transitionFD(m, d0, 1e-6, True, A, B, None, None)
            Q = np.diag([1.0, 40.0, 0.5, 2.0])
            R = np.array([[0.02]])
            self.K = lqr_gain(A, B, Q, R)
        d.qpos[self.hinge_q] = math.radians(c["angle_deg"])
        mujoco.mj_forward(m, d)
        self.plan.events.append({"t": 0.0, "type": "start"})
        p = c.get("push")
        self.push = p
        self.cart_body = world.body("cart")
        if p:
            self.plan.events.append({"t": round(p["at"], 4), "type": "push"})

    def control(self, m, d, t: float) -> None:
        if self.K is None:
            d.ctrl[self.act] = 0.0
        else:
            x = np.concatenate([d.qpos, d.qvel])
            u = float(-(self.K @ x)[0])
            lim = self.cfg["force_limit"]
            d.ctrl[self.act] = min(max(u, -lim), lim)
        d.xfrc_applied[self.cart_body] = 0
        p = self.push
        if p and p["at"] <= t < p["at"] + 0.1:        # the impulse is spread over 0.1 s
            d.xfrc_applied[self.cart_body, 0] = p["impulse"] / 0.1


class PassiveController:
    def __init__(self, m, d, recipe: dict, scene, world) -> None:
        import mujoco
        self.plan = Plan()
        self.plan.events.append({"t": 0.0, "type": "release"})
        if recipe["task"] == "pendulum":
            p = recipe["pendulum"]
            d.qpos[world.qadr("swing1")] = math.radians(p["angle_deg"])
        mujoco.mj_forward(m, d)

    def control(self, m, d, t: float) -> None:
        return None


def make_controller(m, d, recipe: dict, scene, world):
    task = recipe["task"]
    if task in ("pick_place", "push"):
        if not scene.arm:
            raise RecipeError("this task needs a robot arm")
        return ArmController(m, d, recipe, scene, world)
    if task == "cartpole":
        return CartpoleController(m, d, recipe, scene, world)
    return PassiveController(m, d, recipe, scene, world)
