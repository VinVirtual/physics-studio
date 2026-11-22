"""Run the simulation with a fixed timestep, record what the video needs at each frame, and write a
log of what actually happened (contacts, peak forces, motor effort, events, task result).

Deterministic: the same recipe and seed give the same log, byte for byte, on the same machine and
MuJoCo version (single thread, no wall-clock input, seeded jitter only).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from .scene import Scene, surface_z
from .tasks import make_controller

REST_HOLD = 0.25        # seconds an object must stay slow to count as resting
CONTACT_EVENT_N = 0.5   # a contact counts as "touching" above this normal force (newtons)


def r6(x: float) -> float:
    v = round(float(x), 6)
    return 0.0 if v == 0 else v


class World:
    """Name lookups on a compiled model."""

    def __init__(self, m, recipe: dict, scene: Scene) -> None:
        import mujoco
        self.mj = mujoco
        self.m = m
        self.recipe = recipe
        self.scene = scene

    def _id(self, kind, name: str) -> int:
        i = self.mj.mj_name2id(self.m, kind, name)
        if i < 0:
            raise KeyError(name)
        return i

    def joint(self, name: str) -> int:
        return self._id(self.mj.mjtObj.mjOBJ_JOINT, name)

    def qadr(self, name: str) -> int:
        return int(self.m.jnt_qposadr[self.joint(name)])

    def dofadr(self, name: str) -> int:
        return int(self.m.jnt_dofadr[self.joint(name)])

    def body(self, name: str) -> int:
        return self._id(self.mj.mjtObj.mjOBJ_BODY, name)

    def actuator(self, name: str) -> int:
        return self._id(self.mj.mjtObj.mjOBJ_ACTUATOR, name)

    def joint_range(self, name: str):
        j = self.joint(name)
        return (float(self.m.jnt_range[j][0]), float(self.m.jnt_range[j][1])) if self.m.jnt_limited[j] else None

    def surface_at(self, x: float, y: float) -> float:
        return surface_z(self.recipe, x, y)

    def name(self, kind, i: int) -> str:
        return self.mj.mj_id2name(self.m, kind, i) or f"#{i}"


@dataclass
class FrameRecord:
    t: float
    step: int
    qpos: np.ndarray
    forces: list[dict]                 # {"pos": [3], "f": [3], "n": float, "label": str}
    coms: list[dict]                   # {"pos": [3], "label": str}
    torques: list[dict]                # {"name", "value", "limit"}
    status: str = ""


@dataclass
class RunResult:
    log: dict
    frames: list[FrameRecord] = field(default_factory=list)


def compile_scene(scene: Scene):
    import mujoco

    from . import RecipeError
    try:
        return mujoco.MjModel.from_xml_string(scene.xml, assets=scene.files)
    except ValueError as e:
        raise RecipeError(f"MuJoCo could not compile the scene (a robot name may clash with an object id or a "
                          f"ps_ name): {str(e).strip()[:400]}") from None


def _body_root_label(world: World, body: int) -> str:
    """The name shown for a body: an object id, 'robot', 'table' or 'floor'."""
    m = world.m
    mj = world.mj
    if body == 0:
        return "floor"
    b = body
    while m.body_parentid[b] != 0:
        b = int(m.body_parentid[b])
    name = world.name(mj.mjtObj.mjOBJ_BODY, b)
    if name == "ps_robot":
        return "robot"
    if name == "ps_table":
        return "table"
    return name


def frame_times(duration: float, fps: int, speed: float) -> list[float]:
    """Simulation time shown in each video frame (video time * speed)."""
    n = int(math.floor(duration / speed * fps + 1e-9)) + 1
    return [k / fps * speed for k in range(n)]


def _stepped(t: float, dt: float) -> int:
    return int(round(t / dt))


class Recorder:
    def __init__(self, world: World, m, recipe: dict, scene: Scene) -> None:
        mj = world.mj
        self.world = world
        self.m = m
        self.recipe = recipe
        self.scene = scene
        self.dynamic = {}          # body id -> label for objects and robot bodies
        for i in range(m.nbody):
            lbl = _body_root_label(world, i)
            if lbl not in ("floor", "table"):
                self.dynamic[i] = lbl
        self.objects = {oid: world.body(oid) for oid in scene.object_ids}
        self.pairs: dict[tuple[str, str], dict] = {}
        self.peak = {"n": 0.0, "t": 0.0, "pair": None}
        self.act_names = [world.name(mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        self.act_max = np.zeros(m.nu)
        self.act_sat = np.zeros(m.nu, dtype=int)
        self.events: list[dict] = []
        self.flags: dict = {}
        self.slow_since: dict[str, float | None] = {o: None for o in self.objects}
        self.max_height = {o: -1e9 for o in self.objects}
        self.max_speed = {o: 0.0 for o in self.objects}
        self.f6 = np.zeros(6)
        self.vel6 = np.zeros(6)
        self.robot_bodies = {i for i, lbl in self.dynamic.items() if lbl == "robot"}
        self.finger_bodies = set()
        for j in scene.gripper:
            self.finger_bodies.add(int(m.jnt_bodyid[world.joint(j)]))

    def event(self, t: float, kind: str, **detail) -> None:
        e = {"t": r6(t), "type": kind}
        e.update(detail)
        self.events.append(e)

    def contacts(self, d) -> list[dict]:
        """Every active contact: bodies, world position, force on the 'subject' body (newtons)."""
        m, mj = self.m, self.world.mj
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.efc_address < 0:
                continue
            mj.mj_contactForce(m, d, i, self.f6)
            fw = c.frame.reshape(3, 3).T @ self.f6[:3]      # force on geom2's body, world frame
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            # draw the force on the object if one is involved, else on whichever body moves
            if b1 in self.objects.values() and b2 not in self.objects.values():
                subj, other, f = b1, b2, -fw
            elif b2 in self.dynamic or b1 not in self.dynamic:
                subj, other, f = b2, b1, fw
            else:
                subj, other, f = b1, b2, -fw
            out.append({"subject": subj, "other": other, "pos": c.pos.copy(), "f": f, "n": float(self.f6[0])})
        return out

    def observe(self, d, t: float) -> None:
        m = self.m
        touching: dict[str, set] = {o: set() for o in self.objects}
        step_sum: dict[tuple[str, str], float] = {}
        for c in self.contacts(d):
            a = _body_root_label(self.world, c["subject"])
            b = _body_root_label(self.world, c["other"])
            key = tuple(sorted((a, b)))
            step_sum[key] = step_sum.get(key, 0.0) + c["n"]
            for o, bid in self.objects.items():
                if bid not in (c["subject"], c["other"]):
                    continue
                partner = c["other"] if c["subject"] == bid else c["subject"]
                if partner in self.finger_bodies:
                    if c["n"] > CONTACT_EVENT_N:
                        touching[o].add("finger")
                elif c["n"] > 1e-4:
                    touching[o].add(b if c["subject"] == bid else a)
        for key, n in step_sum.items():    # per pair, the normal forces of all its contact points added up
            rec = self.pairs.get(key)
            if rec is None:
                rec = self.pairs[key] = {"bodies": list(key), "first_t": r6(t), "max_normal_N": 0.0, "max_t": r6(t)}
            if n > rec["max_normal_N"]:
                rec["max_normal_N"], rec["max_t"] = n, r6(t)
            if n > self.peak["n"]:
                self.peak = {"n": n, "t": t, "pair": list(key)}
        f = np.abs(d.actuator_force[:m.nu])
        self.act_max = np.maximum(self.act_max, f)
        for i in range(m.nu):
            lo, hi = m.actuator_ctrlrange[i]
            if m.actuator_ctrllimited[i] and abs(d.ctrl[i]) >= 0.999 * max(abs(lo), abs(hi)):
                self.act_sat[i] += 1
        self._object_events(d, t, touching)

    def _object_events(self, d, t: float, touching: dict) -> None:
        rest_speed = (self.recipe.get("goal") or {}).get("rest_speed", 0.02)
        for o, bid in self.objects.items():
            z = float(d.xpos[bid][2])
            self.world.mj.mj_objectVelocity(self.m, d, self.world.mj.mjtObj.mjOBJ_BODY, bid, self.vel6, 0)
            v = float(np.linalg.norm(self.vel6[3:]))
            self.max_height[o] = max(self.max_height[o], z)
            self.max_speed[o] = max(self.max_speed[o], v)
            fl = self.flags.setdefault(o, {"touched": False, "grasped": False, "lifted": False, "rest": False, "moved": False})
            surf = touching[o] - {"finger", "robot"}
            if v > rest_speed * 3:
                fl["moved"] = True
            if surf and not fl["touched"]:
                fl["touched"] = True
                if t > 0:
                    self.event(t, "impact", object=o, on=sorted(surf)[0], speed_m_s=r6(self.max_speed[o]))
            if self.recipe["task"] != "pick_place":
                if ("finger" in touching[o] or "robot" in touching[o]) and not fl.get("contact"):
                    fl["contact"] = True
                    self.event(t, "contact", object=o)
                two_fingers = False
            else:
                two_fingers = "finger" in touching[o] and self._fingers_on(d, bid) >= 2
            if two_fingers and not fl["grasped"]:
                fl["grasped"] = True
                self.event(t, "grasp", object=o)
            if fl["grasped"] and not fl["lifted"] and not surf and z > self.scene.object_start[o][2] + 0.01:
                fl["lifted"] = True
                self.event(t, "lift_off", object=o)
            if fl["lifted"] and fl["grasped"] and "finger" not in touching[o] and not fl.get("released"):
                fl["released"] = True
                self.event(t, "let_go", object=o, height_m=r6(z - self.world.surface_at(*d.xpos[bid][:2])))
            if v < rest_speed:
                if self.slow_since[o] is None:
                    self.slow_since[o] = t
                if not fl["rest"] and t - self.slow_since[o] >= REST_HOLD and surf:
                    fl["rest"] = True
                    if fl["moved"]:            # an object resting from the start gets no event, but is at rest
                        self.event(self.slow_since[o], "rest", object=o)
            else:
                self.slow_since[o] = None
                if fl["rest"] and v > rest_speed * 3:
                    fl["rest"] = False

    def _fingers_on(self, d, bid: int) -> int:
        bodies = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            if b1 == bid and b2 in self.finger_bodies:
                bodies.add(b2)
            if b2 == bid and b1 in self.finger_bodies:
                bodies.add(b1)
        return len(bodies)

    def frame(self, d, t: float, step: int) -> FrameRecord:
        m, mj = self.m, self.world.mj
        groups: dict[tuple[int, int], dict] = {}
        for c in self.contacts(d):
            g = groups.setdefault((c["subject"], c["other"]), {"f": np.zeros(3), "pw": np.zeros(3), "n": 0.0})
            g["f"] += c["f"]
            g["pw"] += c["pos"] * max(c["n"], 1e-9)
            g["n"] += max(c["n"], 1e-9)
        forces = [{"pos": (g["pw"] / g["n"]).tolist(), "f": g["f"].tolist(), "n": float(np.linalg.norm(g["f"])),
                   "label": f"{_body_root_label(self.world, s)}<{_body_root_label(self.world, o)}"}
                  for (s, o), g in sorted(groups.items())]
        for b in range(1, m.nbody):        # forces applied from outside (the cart-pole shove)
            fx = d.xfrc_applied[b, :3]
            n = float(np.linalg.norm(fx))
            if n > 0:
                forces.append({"pos": d.xipos[b].tolist(), "f": fx.tolist(), "n": n, "label": "push", "applied": True})
        coms = []
        if self.robot_bodies:
            coms.append({"pos": d.subtree_com[self.world.body("ps_robot")].tolist(), "label": "robot CoM"})
        for o, bid in self.objects.items():
            coms.append({"pos": d.xipos[bid].tolist(), "label": o})
        for nm in ("link1", "cart"):
            try:
                coms.append({"pos": d.subtree_com[self.world.body(nm)].tolist(), "label": "system CoM"})
            except KeyError:
                pass
        torques = []
        for i in range(m.nu):
            lo, hi = m.actuator_ctrlrange[i]
            torques.append({"name": self.act_names[i].removeprefix("ps_act_"), "value": float(d.actuator_force[i]),
                            "limit": float(max(abs(lo), abs(hi))) if m.actuator_ctrllimited[i] else 0.0,
                            "unit": "N" if m.jnt_type[m.actuator_trnid[i][0]] == mj.mjtJoint.mjJNT_SLIDE else "N·m"})
        return FrameRecord(t=t, step=step, qpos=d.qpos.copy(), forces=forces, coms=coms, torques=torques)


def run(recipe: dict, scene: Scene, record_frames: bool = True, speed: float | None = None) -> RunResult:
    """speed: simulation seconds per video second for the recorded frames (needed when recording)."""
    import mujoco
    m = compile_scene(scene)
    d = mujoco.MjData(m)
    world = World(m, recipe, scene)
    ctrl = make_controller(m, d, recipe, scene, world)
    rec = Recorder(world, m, recipe, scene)
    dt = m.opt.timestep
    steps = int(round(recipe["duration"] / dt))
    v = recipe["video"]
    want = {}
    if record_frames:
        for t in frame_times(recipe["duration"], v["fps"], speed or 1.0):
            want.setdefault(min(_stepped(t, dt), steps), t)
    frames: list[FrameRecord] = []
    traj = hashlib.sha256()
    energy0 = None
    if recipe["task"] == "pendulum":
        m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        mujoco.mj_forward(m, d)
        energy0 = swing_energy(m, d)
    swing = []          # (t, angle) for the pendulum period
    hinge = []          # cart-pole angle samples
    unstable = None
    for k in range(steps + 1):
        t = k * dt
        if k > 0:
            ctrl.control(m, d, t)
            mujoco.mj_step(m, d)
        bad = [w for w in (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                           mujoco.mjtWarning.mjWARN_BADQACC) if d.warning[int(w)].number > 0]
        if bad or not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
            # MuJoCo resets a diverging state and carries on, so its warnings are the signal
            unstable = t
            rec.event(t, "unstable", detail=bad[0].name if bad else "the state became NaN or infinite")
            break
        rec.observe(d, t)
        if k % 10 == 0:
            traj.update(np.round(d.qpos, 9).tobytes())
        if recipe["task"] == "pendulum":
            swing.append((t, float(d.qpos[world.qadr("swing1")])))
        if recipe["task"] == "cartpole":
            hinge.append((t, float(d.qpos[world.qadr("hinge")]), float(d.qpos[world.qadr("slide")])))
        if k in want:
            frames.append(rec.frame(d, want[k], k))
    log = _log(recipe, scene, m, d, world, rec, ctrl, steps, unstable, traj.hexdigest(), swing, hinge, energy0)
    return RunResult(log=log, frames=frames)


def swing_energy(m, d) -> float:
    """Kinetic plus potential energy of the moving bodies, measured from the hanging-down rest pose,
    so 0 means at rest at the bottom (static bodies such as the stand do not count)."""
    import mujoco
    rest = mujoco.MjData(m)
    mujoco.mj_kinematics(m, rest)
    g = -m.opt.gravity[2]
    moving = [b for b in range(1, m.nbody) if m.body_jntnum[b] > 0 or _has_moving_parent(m, b)]
    pe = sum(m.body_mass[b] * g * (d.xipos[b][2] - rest.xipos[b][2]) for b in moving)
    mujoco.mj_energyVel(m, d)
    return float(pe + d.energy[1])


def _has_moving_parent(m, b: int) -> bool:
    while b != 0:
        if m.body_jntnum[b] > 0:
            return True
        b = int(m.body_parentid[b])
    return False


def pendulum_period(samples: list[tuple[float, float]]) -> float | None:
    """Mean time between upward zero crossings (linear interpolation between steps)."""
    ups = []
    for (t0, a0), (t1, a1) in zip(samples, samples[1:]):
        if a0 < 0 <= a1 and a1 != a0:
            ups.append(t0 + (t1 - t0) * (-a0) / (a1 - a0))
    if len(ups) < 2:
        return None
    return float((ups[-1] - ups[0]) / (len(ups) - 1))


def _task_result(recipe, scene, d, world, rec: Recorder, ctrl, unstable, swing, hinge, energy0, m) -> dict:
    import mujoco
    task = recipe["task"]
    res: dict = {"name": task}
    if unstable is not None:
        return {**res, "success": False, "reason": f"the simulation became unstable at {unstable:.3f} s"}
    if task == "pick_place":
        g = recipe["goal"]
        o = g["object"]
        bid = world.body(o)
        pos = d.xpos[bid]
        tgt = ctrl.plan.place_target
        err = float(np.hypot(pos[0] - tgt[0], pos[1] - tgt[1]))
        lifted = rec.max_height[o] - scene.object_start[o][2]
        fl = rec.flags.get(o, {})
        ok_lift = lifted >= 0.8 * g["lift"]
        ok_place = err <= g["tolerance"]
        resting = bool(fl.get("rest"))
        res.update({"object": o, "target_xy_m": tgt[:2], "final_xy_m": [r6(pos[0]), r6(pos[1])],
                    "final_error_m": r6(err), "final_error_cm": round(err * 100, 1), "tolerance_m": g["tolerance"],
                    "max_lift_m": r6(lifted), "max_lift_cm": round(lifted * 100, 1), "lift_required_m": g["lift"],
                    "grasped": bool(fl.get("grasped")), "resting_at_end": resting})
        why = []
        if not fl.get("grasped"):
            why.append("the fingers never held the object")
        if not ok_lift:
            why.append(f"it was lifted {lifted * 100:.1f} cm, less than 80% of the {g['lift'] * 100:.0f} cm asked for")
        if not ok_place:
            why.append(f"it ended {err * 100:.1f} cm from the target (tolerance {g['tolerance'] * 100:.1f} cm)")
        if not resting:
            why.append("it was still moving at the end")
        res["success"] = not why
        res["reason"] = "; ".join(why) if why else f"placed {err * 100:.1f} cm from the target"
    elif task == "push":
        g = recipe["goal"]
        o = g["object"]
        pos = d.xpos[world.body(o)]
        tgt = ctrl.plan.push_target
        start = scene.object_start[o]
        moved = float(np.hypot(pos[0] - start[0], pos[1] - start[1]))
        err = float(np.hypot(pos[0] - tgt[0], pos[1] - tgt[1]))
        res.update({"object": o, "target_xy_m": tgt, "final_xy_m": [r6(pos[0]), r6(pos[1])], "moved_m": r6(moved),
                    "moved_cm": round(moved * 100, 1), "final_error_m": r6(err), "final_error_cm": round(err * 100, 1),
                    "tolerance_m": g["tolerance"]})
        touched = bool(rec.flags.get(o, {}).get("contact"))
        enough = moved >= 0.5 * g["distance"]
        res["success"] = err <= g["tolerance"] and touched and enough
        res["reason"] = f"pushed {moved * 100:.1f} cm, ending {err * 100:.1f} cm from the target"
        if not touched:
            res["reason"] = "the arm never touched the object; " + res["reason"]
        elif not enough:
            res["reason"] += f" (less than half the {g['distance'] * 100:.0f} cm asked for)"
        elif not res["success"]:
            res["reason"] += f" (tolerance {g['tolerance'] * 100:.1f} cm)"
    elif task == "drop":
        at_rest = [o for o in rec.objects if rec.flags.get(o, {}).get("rest")]
        res.update({"objects": len(rec.objects), "at_rest": at_rest,
                    "max_speed_m_s": {o: r6(v) for o, v in rec.max_speed.items()}})
        res["success"] = len(at_rest) == len(rec.objects)
        res["reason"] = (f"all {len(rec.objects)} objects came to rest" if res["success"]
                         else f"{len(rec.objects) - len(at_rest)} of {len(rec.objects)} objects were still moving at the end")
    elif task == "pendulum":
        p = recipe["pendulum"]
        per = pendulum_period(swing) if p["links"] == 1 else None
        small = 2 * math.pi * math.sqrt(p["length"] / 9.81)
        e1 = swing_energy(m, d)
        drift = (e1 - energy0) / energy0 * 100 if energy0 else 0.0
        res.update({"links": p["links"], "measured_period_s": r6(per) if per else None,
                    "small_angle_period_s": r6(small), "release_angle_deg": p["angle_deg"],
                    "energy_start_J": r6(energy0 or 0), "energy_end_J": r6(e1), "energy_change_pct": round(drift, 2) or 0.0,
                    "max_angle_deg": round(max(abs(a) for _, a in swing) * 180 / math.pi, 1)})
        res["success"] = True
        res["reason"] = ("measured the period" if per else "ran to the end") + " (a pendulum has no pass or fail)"
    elif task == "cartpole":
        c = recipe["cartpole"]
        angles = [abs(a) for _, a, _ in hinge]
        fell = next((t for t, a, _ in hinge if abs(a) > math.radians(60)), None)
        final = abs(hinge[-1][1]) * 180 / math.pi
        res.update({"controller": c["controller"], "start_angle_deg": c["angle_deg"],
                    "max_angle_deg": round(max(angles) * 180 / math.pi, 2), "final_angle_deg": round(final, 2),
                    "max_cart_travel_m": r6(max(abs(x) for _, _, x in hinge)), "fell_at_s": r6(fell) if fell else None})
        settle = None
        p = c.get("push")
        after = p["at"] if p else 0.0
        for t, a, _ in hinge:
            if t > after and abs(a) < math.radians(1.0):
                if settle is None:
                    settle = t
            else:
                if t > after:
                    settle = None
        res["settled_at_s"] = r6(settle) if settle is not None else None
        res["success"] = fell is None and final < 2.0
        res["reason"] = ("the pole stayed up" + (f" and settled within 1 degree by {settle:.2f} s" if settle else "")
                         if res["success"] else
                         (f"the pole fell past 60 degrees at {fell:.2f} s" if fell else f"the pole ended {final:.1f} degrees from upright"))
    return res


def _log(recipe, scene, m, d, world, rec: Recorder, ctrl, steps, unstable, traj, swing, hinge, energy0) -> dict:
    import mujoco
    mj = mujoco
    pairs = sorted(rec.pairs.values(), key=lambda p: (-p["max_normal_N"], p["bodies"]))
    for p in pairs:
        p["max_normal_N"] = r6(p["max_normal_N"])
    acts = []
    for i in range(m.nu):
        lo, hi = m.actuator_ctrlrange[i]
        lim = float(max(abs(lo), abs(hi)))
        acts.append({"name": rec.act_names[i].removeprefix("ps_act_"), "max_abs": r6(rec.act_max[i]), "limit": r6(lim),
                     "saturated_pct": round(100 * rec.act_sat[i] / max(steps, 1), 2)})
    end_t = steps * m.opt.timestep if unstable is None else unstable
    # planned phases that would start after the run ended never happened: leave them out
    events = sorted([e for e in ctrl.plan.events if e["t"] <= end_t] + rec.events, key=lambda e: (e["t"], e["type"]))
    warnings = []
    for w in range(len(d.warning)):
        if d.warning[w].number:
            warnings.append({"kind": mj.mjtWarning(w).name, "count": int(d.warning[w].number)})
    joints = [world.name(mj.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    total_mass = float(sum(m.body_mass))
    robot_mass = None
    if scene.arm:
        rb = world.body("ps_robot")
        robot_mass = r6(sum(m.body_mass[b] for b in range(m.nbody) if _is_under(m, b, rb)))
    return {
        "physics_studio": 1,
        "recipe": recipe["name"], "title": recipe["title"], "task_type": recipe["task"], "seed": recipe["seed"],
        "mujoco_version": mujoco.__version__,
        "timestep_s": r6(m.opt.timestep), "steps": steps if unstable is None else int(round(unstable / m.opt.timestep)),
        "duration_s": r6(recipe["duration"]),
        "model": {"bodies": int(m.nbody), "joints": int(m.njnt), "actuators": int(m.nu), "dof": int(m.nv),
                  "joint_names": joints, "total_mass_kg": r6(total_mass), "robot_mass_kg": robot_mass,
                  "arm_joints": len(scene.arm), "gripper_joints": len(scene.gripper),
                  "robot_source": scene.robot_source or None},
        "objects": [{"id": o["id"], "shape": o["shape"], "size_m": o["size"], "mass_kg": o["mass"],
                     "start_m": scene.object_start[o["id"]],
                     "end_m": [r6(x) for x in d.xpos[world.body(o["id"])]],
                     "max_height_m": r6(rec.max_height[o["id"]])} for o in recipe["objects"]],
        "events": events,
        "contacts": {"pairs": pairs, "max_normal_N": r6(rec.peak["n"]), "max_pair": rec.peak["pair"],
                     "max_t": r6(rec.peak["t"])},
        "actuators": acts,
        "plan": {"unreachable": ctrl.plan.unreachable, "ends_at_s": r6(ctrl.plan.end()),
                 "cut_short": ctrl.plan.end() > end_t + 1e-9},
        "warnings": warnings,
        "task": _task_result(recipe, scene, d, world, rec, ctrl, unstable, swing, hinge, energy0, m),
        "trajectory_sha256": traj,
    }


def _is_under(m, b: int, root: int) -> bool:
    while b != 0:
        if b == root:
            return True
        b = int(m.body_parentid[b])
    return False


def log_json(log: dict) -> str:
    return json.dumps(log, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
