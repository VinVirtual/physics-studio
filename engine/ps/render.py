"""Render recorded states with MuJoCo's offscreen renderer and draw the overlays on top.

Rendering replays the recorded joint positions (mj_forward for the pose only); the forces, centre of
mass and motor values drawn come from the run itself, not from the replay.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import overlay as ov
from .scene import Scene, surface_z
from .simulate import FrameRecord
from .timing import cue_at

EVENT_LABELS = {"reach": "reach", "descend": "descend", "close": "fingers close", "grasp": "grasp",
                "lift": "lift", "lift_off": "lift-off", "carry": "carry", "lower": "lower", "open": "fingers open",
                "let_go": "released", "rest": "at rest", "retreat": "retreat", "impact": "impact",
                "push": "push", "contact": "contact", "release": "release", "start": "start", "unstable": "UNSTABLE"}
SHOWN_EVENTS = ("reach", "close", "grasp", "lift_off", "carry", "lower", "open", "rest", "retreat", "impact",
                "push", "contact", "unstable")


def gl_backend() -> str:
    os.environ.setdefault("MUJOCO_GL", "egl")
    return os.environ["MUJOCO_GL"]


@dataclass
class Composer:
    recipe: dict
    scene: Scene
    log: dict
    cues: list[dict]
    total: float          # video duration (s)
    run_end: float        # video time at which the run ends (s)

    def __post_init__(self) -> None:
        self.events = [e for e in self.log["events"] if e["type"] in SHOWN_EVENTS]
        seen, uniq = set(), []
        for e in self.events:      # one tick per kind and object
            k = (e["type"], e.get("object"))
            if k not in seen:
                seen.add(k)
                uniq.append(e)
        self.events = uniq
        self.speed = self.recipe["video"]["speed"]
        task = self.log["task"]
        self.success = task.get("success") if self.recipe["task"] not in ("pendulum",) else None
        self.headline = ("TASK PASSED" if task["success"] else "TASK FAILED") if self.success is not None else "RUN COMPLETE"
        self.detail = task["reason"][0].upper() + task["reason"][1:] if task.get("reason") else ""
        if self.recipe["task"] == "pendulum" and task.get("measured_period_s"):
            self.detail = (f"Measured period {task['measured_period_s']:.3f} s · small-angle formula "
                           f"{task['small_angle_period_s']:.3f} s")
        self.engine = f"MuJoCo {self.log['mujoco_version']}"

    def force_legend(self) -> str | None:
        if not self.recipe["overlays"]["forces"]:
            return None
        task = self.recipe["task"]
        if task == "pendulum":
            return None
        if task == "cartpole":
            return "applied force" if self.recipe["cartpole"].get("push") else None
        return "contact force"

    def current_event(self, t_sim: float) -> str | None:
        cur = None
        for e in self.events:
            if e["t"] <= t_sim and (t_sim - e["t"]) / self.speed < 1.5:
                cur = EVENT_LABELS.get(e["type"], e["type"])
        return cur

    def compose(self, img: np.ndarray, rec: FrameRecord, t_video: float) -> Image.Image:
        o = self.recipe["overlays"]
        t_sim = rec.t
        p = ov.Painter(self.scene.width, self.scene.height)
        events_v = [{"t": e["t"] / self.speed} for e in self.events]
        if o["com"]:
            ov.draw_com(p, self.scene.camera, rec.coms, lambda x, y: surface_z(self.recipe, x, y))
        if o["forces"]:
            ov.draw_forces(p, self.scene.camera, rec.forces, o["force_scale"], o["min_force"], max_len=0.12)
        if o["hud"]:
            ov.draw_hud(p, t_sim, rec.step, self.recipe["timestep"], self.speed, self.engine,
                        legend_forces=self.force_legend(), legend_com=o["com"])
            ov.draw_title(p, self.recipe["title"], _fade(t_video, 0.2, 3.6))
            ov.draw_timeline(p, t_video, self.total, events_v,
                             self.current_event(t_sim) if t_video <= self.run_end + 0.05 else None)
        if o["torques"]:
            ov.draw_effort(p, rec.torques)
        if t_video > self.run_end:
            ov.draw_result(p, self.success, self.headline, self.detail, min(1.0, (t_video - self.run_end) / 0.35))
        if o["captions"]:
            c = cue_at(self.cues, t_video)
            if c:
                ov.draw_caption(p, c["words"], t_video)
        return p.finish(Image.fromarray(img))


def _fade(t: float, t_in: float, t_out: float, ramp: float = 0.35) -> float:
    if t < t_in or t > t_out + ramp:
        return 0.0
    return min(1.0, (t - t_in) / ramp, (t_out + ramp - t) / ramp)


class SceneRenderer:
    def __init__(self, model, scene: Scene) -> None:
        import mujoco
        gl_backend()
        self.mj = mujoco
        self.m = model
        self.d = mujoco.MjData(model)
        self.r = mujoco.Renderer(model, scene.height, scene.width)
        self.cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ps_cam")

    def render(self, qpos: np.ndarray) -> np.ndarray:
        self.d.qpos[:] = qpos
        self.mj.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, camera=self.cam)
        return self.r.render().copy()

    def close(self) -> None:
        self.r.close()


def frames_for_video(frames: list[FrameRecord], total: float, fps: int):
    """(index into frames, video time) for every video frame; the last state is held at the end."""
    n = int(math.ceil(total * fps - 1e-9))
    for k in range(n):
        yield min(k, len(frames) - 1), k / fps


def contact_sheet(images: list[tuple[Image.Image, str]], cols: int = 3, width: int = 1500) -> Image.Image:
    w0, h0 = images[0][0].size
    cw = width // cols
    ch = round(cw * h0 / w0)
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cw * cols, (ch + 26) * rows), ov.PAPER)
    from PIL import ImageDraw
    dr = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(images):
        x, y = (i % cols) * cw, (i // cols) * (ch + 26)
        sheet.paste(im.resize((cw, ch), Image.LANCZOS), (x, y + 26))
        dr.text((x + 8, y + 5), label, font=ov.font("mono", 15), fill=ov.INK)
    return sheet
