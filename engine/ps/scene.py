"""Build the MJCF scene from a validated recipe: studio (floor, lights, sky, camera), table, objects,
the robot (if any) with one motor per controlled joint, or a built-in pendulum / cart-pole.

Heights in a recipe are measured from the surface below: an object's pos [x, y, h] puts its bottom h
metres above the table top (if x, y is over the table) or above the floor. The same goes for the robot
base. The built scene is self-contained apart from the robot's mesh files, which are passed as bytes.
"""
from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import RecipeError
from .camera import Camera, fit_distance, make_camera, xyaxes_attr
from .recipe import PALETTE, parse_color
from .robot import RobotPart, load_robot

FRAME = {"landscape": (1280, 720), "vertical": (720, 1280), "square": (960, 960)}
TABLE_TOP = 0.03
FINGER_COLOR = "#d9d2c3"


@dataclass
class Scene:
    xml: str
    files: dict[str, bytes]
    camera: Camera
    width: int
    height: int
    object_ids: list[str]
    controlled: list[str] = field(default_factory=list)   # joint names with a ps_act_ motor, in actuator order
    gripper: list[str] = field(default_factory=list)
    arm: list[str] = field(default_factory=list)
    object_start: dict[str, list[float]] = field(default_factory=dict)  # world centre of each object at t=0
    surface: dict[str, float] = field(default_factory=dict)              # z of the surface under each object
    robot_source: str = ""


def f6(v: float) -> str:
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def vec(v) -> str:
    return " ".join(f6(float(x)) for x in v)


def rgba(color, alpha: float = 1.0) -> str:
    c = parse_color(color) or PALETTE["ochre"]
    return vec((*c, alpha))


def yaw_quat(deg: float) -> str:
    h = math.radians(deg) / 2
    return vec((math.cos(h), 0, 0, math.sin(h)))


def surface_z(recipe: dict, x: float, y: float) -> float:
    t = recipe.get("table")
    if t:
        cx, cy = t["pos"]
        sx, sy = t["size"]
        if abs(x - cx) <= sx and abs(y - cy) <= sy:
            return t["height"]
    return 0.0


def half_height(o: dict) -> float:
    s = o["size"]
    return s[0] if o["shape"] == "sphere" else (s[1] if o["shape"] == "cylinder" else s[2])


def object_starts(recipe: dict) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Centre of every object at t = 0, with the seeded jitter applied (same seed, same result)."""
    rng = np.random.default_rng(recipe["seed"])
    starts, surf = {}, {}
    for o in recipe["objects"]:
        jx, jy = rng.uniform(-1, 1, 2) * o["jitter"]
        x, y, h = o["pos"]
        x, y = x + jx, y + jy
        z0 = surface_z(recipe, x, y)
        starts[o["id"]] = [round(x, 9), round(y, 9), round(z0 + h + half_height(o), 9)]
        surf[o["id"]] = z0
    return starts, surf


def default_camera(recipe: dict) -> dict:
    task = recipe["task"]
    if task == "pendulum":
        p = recipe["pendulum"]
        reach = p["length"] * p["links"]
        return {"lookat": [0, 0, p["pivot_height"] - reach * 0.55], "distance": 0.9 + 1.8 * reach,
                "azimuth": 90, "elevation": -6, "fovy": 40}
    if task == "cartpole":
        c = recipe["cartpole"]
        return {"lookat": [0, 0, 0.45 + c["pole_length"] * 0.45], "distance": 1.1 + 1.6 * c["pole_length"],
                "azimuth": 90, "elevation": -8, "fovy": 40}
    t = recipe.get("table")
    top = t["height"] if t else 0.0
    cx, cy = (t["pos"] if t else [0.0, 0.0])
    if recipe["objects"] and not recipe.get("robot"):
        xs = [o["pos"][0] for o in recipe["objects"]]
        ys = [o["pos"][1] for o in recipe["objects"]]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
    if recipe.get("robot"):
        return {"lookat": [cx * 0.7, cy, top + 0.16], "distance": 1.05, "azimuth": 60, "elevation": -24, "fovy": 40}
    high = max((o["pos"][2] for o in recipe["objects"]), default=0.0)   # keep the drop heights in view
    return {"lookat": [cx, cy, top + 0.08 + 0.4 * high], "distance": 0.9 + 0.9 * high, "azimuth": 120,
            "elevation": -16, "fovy": 40}


def camera_for(recipe: dict, fmt: str) -> Camera:
    w, h = FRAME[fmt]
    cam = default_camera(recipe)
    user = recipe.get("camera") or {}
    auto_fit = user.get(fmt, {}).get("distance") is None
    for src in (user, user.get(fmt) or {}):
        for k in ("lookat", "distance", "azimuth", "elevation", "fovy"):
            if src.get(k) is not None:
                cam[k] = src[k]
    dist = fit_distance(cam["distance"], w, h) if auto_fit else cam["distance"]
    lookat = list(cam["lookat"])
    if fmt == "vertical" and user.get(fmt, {}).get("lookat") is None:
        lookat[2] += 0.16 * dist          # aim higher so the scene sits below the panels at the top
    return make_camera(lookat, dist, cam["azimuth"], cam["elevation"], cam["fovy"], w, h)


def _studio(root: ET.Element, asset: ET.Element, wb: ET.Element, cam: Camera, w: int, h: int) -> None:
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", {"offwidth": str(w), "offheight": str(h), "fovy": f6(cam.fovy)})
    ET.SubElement(vis, "quality", {"shadowsize": "4096", "offsamples": "8"})
    ET.SubElement(vis, "headlight", {"ambient": "0.42 0.42 0.42", "diffuse": "0.38 0.38 0.38", "specular": "0.05 0.05 0.05"})
    ET.SubElement(vis, "map", {"znear": "0.01", "zfar": "60", "haze": "0.25", "shadowclip": "3"})
    ET.SubElement(vis, "rgba", {"haze": "0.93 0.91 0.86 1"})
    ET.SubElement(asset, "texture", {"name": "ps_sky", "type": "skybox", "builtin": "gradient",
                                     "rgb1": "0.96 0.95 0.92", "rgb2": "0.84 0.82 0.77", "width": "256", "height": "256"})
    ET.SubElement(asset, "texture", {"name": "ps_grid", "type": "2d", "builtin": "flat", "width": "512", "height": "512",
                                     "rgb1": "0.92 0.90 0.85", "mark": "edge", "markrgb": "0.78 0.75 0.68"})
    ET.SubElement(asset, "material", {"name": "ps_floor", "texture": "ps_grid", "texrepeat": "10 10", "reflectance": "0.04"})
    ET.SubElement(asset, "material", {"name": "ps_metal", "rgba": "0.28 0.30 0.33 1", "specular": "0.4", "shininess": "0.5"})
    ET.SubElement(wb, "light", {"name": "ps_key", "pos": "1.2 -1.6 3.2", "dir": "-0.35 0.45 -1", "directional": "true",
                                "castshadow": "true", "diffuse": "0.55 0.53 0.5", "specular": "0.15 0.15 0.15"})
    ET.SubElement(wb, "light", {"name": "ps_fill", "pos": "-2 2 2.5", "dir": "0.6 -0.6 -0.6", "directional": "true",
                                "castshadow": "false", "diffuse": "0.18 0.19 0.22"})
    ET.SubElement(wb, "geom", {"name": "ps_floor", "type": "plane", "size": "6 6 0.05", "material": "ps_floor",
                               "friction": "1 0.005 0.0001"})
    ET.SubElement(wb, "camera", {"name": "ps_cam", "pos": vec(cam.pos), "xyaxes": xyaxes_attr(cam), "fovy": f6(cam.fovy)})


def _table(wb: ET.Element, t: dict) -> None:
    (cx, cy), (sx, sy), ht = t["pos"], t["size"], t["height"]
    body = ET.SubElement(wb, "body", {"name": "ps_table", "pos": vec((cx, cy, 0))})
    ET.SubElement(body, "geom", {"name": "ps_table_top", "type": "box", "size": vec((sx, sy, TABLE_TOP / 2)),
                                 "pos": vec((0, 0, ht - TABLE_TOP / 2)), "rgba": rgba(t["color"]),
                                 "friction": "1 0.005 0.0001"})
    leg = 0.022
    for i, (dx, dy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        ET.SubElement(body, "geom", {"name": f"ps_table_leg{i}", "type": "box", "size": vec((leg, leg, (ht - TABLE_TOP) / 2)),
                                     "pos": vec((dx * (sx - 0.05), dy * (sy - 0.05), (ht - TABLE_TOP) / 2)),
                                     "material": "ps_metal", "contype": "0", "conaffinity": "0"})


def _objects(wb: ET.Element, recipe: dict, starts: dict) -> None:
    for o in recipe["objects"]:
        body = ET.SubElement(wb, "body", {"name": o["id"], "pos": vec(starts[o["id"]]), "quat": yaw_quat(o["yaw_deg"])})
        ET.SubElement(body, "freejoint", {"name": f"{o['id']}_free"})
        size = o["size"]
        ET.SubElement(body, "geom", {"name": o["id"], "type": o["shape"], "size": vec(size), "mass": f6(o["mass"]),
                                     "rgba": rgba(o["color"]), "friction": f"{f6(o['friction'])} 0.01 0.001",
                                     "condim": "4", "solref": "0.004 1", "priority": "1"})


def _pendulum(wb: ET.Element, p: dict) -> None:
    pivot = p["pivot_height"]
    stand = ET.SubElement(wb, "body", {"name": "ps_stand", "pos": vec((0, 0, 0))})
    ET.SubElement(stand, "geom", {"type": "box", "size": vec((0.03, 0.03, pivot / 2)), "pos": vec((0, 0.09, pivot / 2)),
                                  "material": "ps_metal", "contype": "0", "conaffinity": "0"})
    ET.SubElement(stand, "geom", {"type": "cylinder", "size": "0.012 0.06", "pos": vec((0, 0.04, pivot)),
                                  "quat": "0.7071068 0.7071068 0 0", "material": "ps_metal", "contype": "0", "conaffinity": "0"})
    parent = ET.SubElement(wb, "body", {"name": "link1", "pos": vec((0, 0, pivot))})
    L = p["length"]
    for i in range(p["links"]):
        body = parent if i == 0 else ET.SubElement(parent, "body", {"name": f"link{i + 1}", "pos": vec((0, 0, -L))})
        ET.SubElement(body, "joint", {"name": f"swing{i + 1}", "type": "hinge", "axis": "0 1 0",
                                      "damping": f6(p["damping"])})
        ET.SubElement(body, "geom", {"type": "capsule", "fromto": vec((0, 0, 0, 0, 0, -L)), "size": "0.008",
                                     "mass": f6(p["mass"] * 0.01), "rgba": "0.23 0.25 0.28 1", "contype": "0", "conaffinity": "0"})
        ET.SubElement(body, "geom", {"name": f"bob{i + 1}", "type": "sphere", "size": f6(0.035 + 0.01 * min(p["mass"], 4)),
                                     "pos": vec((0, 0, -L)), "mass": f6(p["mass"]),
                                     "rgba": rgba("ochre" if i == 0 else "cobalt"), "contype": "0", "conaffinity": "0"})
        parent = body


def _cartpole(wb: ET.Element, act: ET.Element, c: dict) -> None:
    rail_z = 0.42
    ET.SubElement(wb, "geom", {"name": "ps_rail", "type": "box", "size": "1.6 0.012 0.012", "pos": vec((0, 0, rail_z)),
                               "material": "ps_metal", "contype": "0", "conaffinity": "0"})
    for x in (-1.6, 1.6):
        ET.SubElement(wb, "geom", {"type": "box", "size": vec((0.02, 0.02, rail_z / 2)), "pos": vec((x, 0, rail_z / 2)),
                                   "material": "ps_metal", "contype": "0", "conaffinity": "0"})
    cart = ET.SubElement(wb, "body", {"name": "cart", "pos": vec((0, 0, rail_z))})
    ET.SubElement(cart, "joint", {"name": "slide", "type": "slide", "axis": "1 0 0", "limited": "true",
                                  "range": "-1.5 1.5", "damping": "0.05"})
    ET.SubElement(cart, "geom", {"name": "cart", "type": "box", "size": "0.11 0.07 0.045", "mass": f6(c["cart_mass"]),
                                 "rgba": rgba("graphite"), "contype": "0", "conaffinity": "0"})
    pole = ET.SubElement(cart, "body", {"name": "pole", "pos": vec((0, 0, 0.045))})
    ET.SubElement(pole, "joint", {"name": "hinge", "type": "hinge", "axis": "0 1 0", "damping": "0.002"})
    ET.SubElement(pole, "geom", {"name": "pole", "type": "capsule", "fromto": vec((0, 0, 0, 0, 0, c["pole_length"])),
                                 "size": "0.018", "mass": f6(c["pole_mass"]), "rgba": rgba("ochre"),
                                 "contype": "0", "conaffinity": "0"})
    ET.SubElement(act, "motor", {"name": "ps_act_slide", "joint": "slide", "gear": "1", "ctrllimited": "true",
                                 "ctrlrange": vec((-c["force_limit"], c["force_limit"]))})


def _color_robot(bodies: list[ET.Element], asset: ET.Element, base_color, gripper: set[str]) -> None:
    c = np.array(parse_color(base_color) or PALETTE["graphite"])
    light = np.clip(c + (1 - c) * 0.45, 0, 1)
    ET.SubElement(asset, "material", {"name": "ps_link_a", "rgba": vec((*c, 1)), "specular": "0.35", "shininess": "0.4"})
    ET.SubElement(asset, "material", {"name": "ps_link_b", "rgba": vec((*light, 1)), "specular": "0.25", "shininess": "0.3"})
    ET.SubElement(asset, "material", {"name": "ps_finger", "rgba": rgba(FINGER_COLOR), "specular": "0.2"})

    def walk(el: ET.Element, depth: int, in_gripper: bool) -> None:
        if el.tag == "body":
            js = {j.get("name") for j in el.findall("joint")}
            in_gripper = in_gripper or bool(js & gripper)
            depth += 1
        for ch in el:
            if ch.tag == "geom" and ch.get("rgba") is None and ch.get("material") is None and ch.get("class") is None:
                ch.set("material", "ps_finger" if in_gripper else ("ps_link_a" if depth % 2 == 0 else "ps_link_b"))
            walk(ch, depth, in_gripper)
    for b in bodies:
        walk(b, 0, False)


def _probe_joints(robot: RobotPart) -> list[dict]:
    """Compile the robot alone to read its joints (type, range, effort) as MuJoCo sees them."""
    import mujoco
    root = ET.Element("mujoco", {"model": "probe"})
    ET.SubElement(root, "compiler", robot.compiler)
    _robot_sections(root, robot)
    wb = ET.SubElement(root, "worldbody")
    holder = ET.SubElement(wb, "body", {"name": "ps_robot", "childclass": "ps_robot_main"} if robot.defaults else {"name": "ps_robot"})
    for b in robot.bodies:
        holder.append(b)
    for s, els in robot.sections.items():
        if els:
            sec = ET.SubElement(root, s)
            sec.extend(els)
    try:
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets=robot.files)
    except (ValueError, RuntimeError) as e:
        raise RecipeError(f"MuJoCo could not compile the robot: {str(e).strip()[:400]}") from None
    out = []
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        jt = int(m.jnt_type[j])
        lim = (float(m.jnt_range[j][0]), float(m.jnt_range[j][1])) if m.jnt_limited[j] else None
        frc = float(m.jnt_actfrcrange[j][1]) if m.jnt_actfrclimited[j] else None
        out.append({"name": name, "type": {0: "free", 1: "ball", 2: "slide", 3: "hinge"}[jt], "range": lim, "effort": frc})
    return out


def _robot_sections(root: ET.Element, robot: RobotPart) -> None:
    if robot.assets:
        a = root.find("asset")
        if a is None:
            a = ET.SubElement(root, "asset")
        a.extend(robot.assets)
    if robot.defaults:
        d = root.find("default")
        if d is None:
            d = ET.SubElement(root, "default")
        cls = ET.SubElement(d, "default", {"class": "ps_robot_main"})
        cls.extend(robot.defaults)


def robot_joints(recipe: dict, recipe_dir: Path) -> tuple[RobotPart, list[dict]]:
    robot = load_robot(recipe_dir, recipe["robot"])
    return robot, _probe_joints(copy.deepcopy(robot))     # the probe moves elements; keep the original intact


def build_scene(recipe: dict, recipe_dir: Path, fmt: str | None = None) -> Scene:
    fmt = fmt or recipe["video"]["format"]
    w, h = FRAME[fmt]
    cam = camera_for(recipe, fmt)
    root = ET.Element("mujoco", {"model": f"physics-studio {recipe['name']}"})
    robot: RobotPart | None = None
    joints: list[dict] = []
    if recipe.get("robot"):
        robot, joints = robot_joints(recipe, recipe_dir)
        ET.SubElement(root, "compiler", robot.compiler)
    else:
        ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(root, "option", {"timestep": f6(recipe["timestep"]), "integrator": "implicitfast",
                                   "cone": "elliptic", "impratio": "10", "gravity": "0 0 -9.81"})
    ET.SubElement(root, "size", {"memory": "64M"})
    asset = ET.SubElement(root, "asset")
    wb = ET.SubElement(root, "worldbody")
    act = ET.Element("actuator")
    _studio(root, asset, wb, cam, w, h)
    # <visual> must come before <asset> for readability only; MuJoCo accepts any order
    if recipe.get("table"):
        _table(wb, recipe["table"])
    starts, surf = object_starts(recipe)
    _objects(wb, recipe, starts)
    scene = Scene(xml="", files={}, camera=cam, width=w, height=h, object_ids=[o["id"] for o in recipe["objects"]],
                  object_start=starts, surface=surf)
    if recipe["task"] == "pendulum":
        _pendulum(wb, recipe["pendulum"])
    elif recipe["task"] == "cartpole":
        _cartpole(wb, act, recipe["cartpole"])
        scene.controlled = ["slide"]
    if robot is not None:
        _add_robot(root, wb, act, recipe, robot, joints, scene)
    if len(act):
        root.append(act)
    ET.indent(root)
    scene.xml = ET.tostring(root, encoding="unicode")
    return scene


def _add_robot(root, wb, act, recipe: dict, robot: RobotPart, joints: list[dict], scene: Scene) -> None:
    cfg = recipe["robot"]
    names = {j["name"]: j for j in joints}
    grip = list(cfg["gripper"]["joints"]) if cfg.get("gripper") else []
    for j in grip + list(cfg.get("arm_joints") or []) + list(cfg.get("home") or {}):
        if j not in names:
            raise RecipeError(f'robot joint "{j}" not found; the robot has: {", ".join(n for n in names if n) or "no named joints"}')
    for j in grip:
        if names[j]["type"] not in ("hinge", "slide"):
            raise RecipeError(f'gripper joint "{j}" must be a hinge or slide joint')
    arm = list(cfg["arm_joints"]) if cfg.get("arm_joints") else [n for n, j in names.items() if j["type"] in ("hinge", "slide") and n not in grip and n]
    if not arm:
        raise RecipeError("the robot has no hinge or slide joints to move")
    ee = cfg["end_effector"]
    bodies_named = {b.get("name") for top in robot.bodies for b in top.iter("body")}
    if ee["body"] not in bodies_named:
        raise RecipeError(f'end_effector body "{ee["body"]}" not found; bodies: {", ".join(sorted(n for n in bodies_named if n))[:400]}')
    _color_robot(robot.bodies, root.find("asset"), cfg["color"], set(grip))
    for top in robot.bodies:
        for b in top.iter("body"):
            if b.get("name") == ee["body"]:
                ET.SubElement(b, "site", {"name": "ps_ee", "pos": vec(ee["offset"]), "size": "0.006", "rgba": "0 0 0 0"})
            for jel in b.findall("joint"):
                if jel.get("armature") is None:
                    jel.set("armature", f6(cfg["armature"]))
    _robot_sections(root, robot)
    x, y, hgt = cfg["base"]
    z = surface_z(recipe, x, y) + hgt
    attrs = {"name": "ps_robot", "pos": vec((x, y, z)), "quat": yaw_quat(cfg["yaw_deg"])}
    if robot.defaults:
        attrs["childclass"] = "ps_robot_main"
    holder = ET.SubElement(wb, "body", attrs)
    for b in robot.bodies:
        holder.append(b)
    for s, els in robot.sections.items():
        if els:
            sec = ET.SubElement(root, s)
            sec.extend(els)
    for j in arm + grip:
        eff = names[j]["effort"] or (10.0 if j in grip else 30.0)
        act.append(ET.Element("motor", {"name": f"ps_act_{j}", "joint": j, "gear": "1", "ctrllimited": "true",
                                        "ctrlrange": vec((-eff, eff))}))
    scene.controlled = arm + grip
    scene.arm = arm
    scene.gripper = grip
    scene.files = dict(robot.files)
    scene.robot_source = robot.source


def write_scene(scene: Scene, out_xml: Path) -> None:
    """Save the scene as a standalone MJCF file with its mesh files next to it in <name>.assets/."""
    from .files import safe_dir, write_atomic
    root = ET.fromstring(scene.xml)
    if scene.files:
        adir = out_xml.with_suffix("").name + ".assets"
        safe_dir(out_xml.parent / adir)
        for n, data in scene.files.items():
            write_atomic(out_xml.parent / adir / n, data)
        comp = root.find("compiler")
        comp.set("meshdir", adir)
        comp.set("texturedir", adir)
    ET.indent(root)
    write_atomic(out_xml, "<!-- generated by physics-studio; edit the recipe, not this file -->\n"
                 + ET.tostring(root, encoding="unicode") + "\n")
