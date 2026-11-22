"""Load and validate a scene recipe (JSON). Every value is checked for type and range, unknown keys
are refused, and defaults are filled in. The result is a new plain dict; the input is never changed.

A recipe is untrusted input: it may come from anywhere. Nothing in it is executed, paths are only
checked here (see safety.py for where files may be read from), and sizes and counts are capped so a
recipe cannot ask for an hour-long simulation or a million objects.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from . import RecipeError

MAX_RECIPE_BYTES = 256 * 1024
MAX_OBJECTS = 12
MAX_NARRATION_LINES = 16
MAX_LINE_CHARS = 240
MAX_STEPS = 60_000
TASKS = ("drop", "push", "pick_place", "pendulum", "cartpole")
FORMATS = ("landscape", "vertical", "square")
SHAPES = ("box", "sphere", "cylinder")
FPS = (24, 25, 30, 50, 60)
ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,39}$")
VOICE_RE = re.compile(r"^[a-z]{2}_[a-z]{2,20}$")
HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
PLACEHOLDER_RE = re.compile(r"\{([^{}]*)\}")
ANCHOR_RE = re.compile(r"(start|end|event:[a-z_]{1,30})")

# named colours of the house palette (RGB 0..1); objects may also use "#rrggbb"
PALETTE = {
    "ochre": (0.949, 0.647, 0.255),
    "cobalt": (0.184, 0.365, 0.659),
    "teal": (0.165, 0.616, 0.561),
    "plum": (0.486, 0.290, 0.478),
    "sage": (0.553, 0.663, 0.545),
    "graphite": (0.227, 0.247, 0.278),
    "bone": (0.910, 0.890, 0.835),
    "brick": (0.682, 0.302, 0.247),
}

DEFAULT_DURATION = {"drop": 4.0, "push": 6.0, "pick_place": 8.0, "pendulum": 6.0, "cartpole": 8.0}
ROBOT_TASKS = ("push", "pick_place")

_MISSING = object()


class _Checker:
    """Collects every problem in one pass so the user can fix them all at once."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def err(self, path: str, msg: str) -> None:
        self.errors.append(f"{path}: {msg}")

    def keys(self, obj: Any, path: str, allowed: tuple[str, ...]) -> bool:
        if not isinstance(obj, dict):
            self.err(path, f"must be an object, got {type(obj).__name__}")
            return False
        for k in obj:
            if k not in allowed:
                self.err(path, f'unknown key "{k}" (allowed: {", ".join(allowed)})')
        return True

    def num(self, obj: dict, key: str, path: str, lo: float, hi: float, default: Any = _MISSING,
            integer: bool = False) -> Any:
        v = obj.get(key, _MISSING)
        if v is _MISSING:
            if default is _MISSING:
                self.err(f"{path}.{key}", "is required")
            return None if default is _MISSING else default
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and not isinstance(v, int)):
            self.err(f"{path}.{key}", f"must be {'a whole number' if integer else 'a number'}, got {json.dumps(v)[:40]}")
            return default if default is not _MISSING else None
        if not math.isfinite(v) or v < lo or v > hi:
            self.err(f"{path}.{key}", f"must be between {lo} and {hi}, got {v}")
            return default if default is not _MISSING else None
        return v

    def vec(self, obj: dict, key: str, path: str, n: int | tuple[int, ...], lo: float, hi: float,
            default: Any = _MISSING) -> list[float] | None:
        v = obj.get(key, _MISSING)
        if v is _MISSING:
            if default is _MISSING:
                self.err(f"{path}.{key}", "is required")
                return None
            return list(default) if default is not None else None
        sizes = (n,) if isinstance(n, int) else n
        if not isinstance(v, list) or len(v) not in sizes:
            self.err(f"{path}.{key}", f"must be a list of {' or '.join(map(str, sizes))} numbers")
            return None
        out = []
        for i, x in enumerate(v):
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < lo or x > hi:
                self.err(f"{path}.{key}[{i}]", f"must be a number between {lo} and {hi}, got {json.dumps(x)[:40]}")
                return None
            out.append(float(x))
        return out

    def boolean(self, obj: dict, key: str, path: str, default: bool) -> bool:
        v = obj.get(key, default)
        if not isinstance(v, bool):
            self.err(f"{path}.{key}", "must be true or false")
            return default
        return v

    def text(self, obj: dict, key: str, path: str, max_len: int, default: Any = _MISSING,
             pattern: re.Pattern | None = None) -> Any:
        v = obj.get(key, _MISSING)
        if v is _MISSING:
            if default is _MISSING:
                self.err(f"{path}.{key}", "is required")
                return None
            return default
        if not isinstance(v, str):
            self.err(f"{path}.{key}", "must be text")
            return default if default is not _MISSING else None
        if len(v) > max_len:
            self.err(f"{path}.{key}", f"is longer than {max_len} characters")
            return default if default is not _MISSING else None
        if any(ord(c) < 32 and c not in "\t" for c in v):
            self.err(f"{path}.{key}", "contains control characters")
            return default if default is not _MISSING else None
        if pattern is not None and not pattern.match(v):
            self.err(f"{path}.{key}", f"{json.dumps(v)[:50]} does not match {pattern.pattern}")
            return default if default is not _MISSING else None
        return v

    def choice(self, obj: dict, key: str, path: str, options: tuple, default: Any = _MISSING) -> Any:
        v = obj.get(key, _MISSING)
        if v is _MISSING:
            if default is _MISSING:
                self.err(f"{path}.{key}", f"is required (one of {', '.join(map(str, options))})")
                return None
            return default
        if v not in options or isinstance(v, bool):
            self.err(f"{path}.{key}", f"must be one of {', '.join(map(str, options))}, got {json.dumps(v)[:40]}")
            return default if default is not _MISSING else None
        return v


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise RecipeError(f'duplicate key "{k}" in the recipe JSON')
        out[k] = v
    return out


def _no_constants(name: str) -> None:
    raise RecipeError(f"{name} is not allowed in a recipe (numbers must be finite)")


def parse_json(text: str) -> Any:
    """Strict JSON: no NaN or Infinity, no duplicate keys, size-capped."""
    if len(text.encode("utf-8")) > MAX_RECIPE_BYTES:
        raise RecipeError(f"recipe is larger than {MAX_RECIPE_BYTES // 1024} KB")
    try:
        return json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_no_constants)
    except RecursionError:
        raise RecipeError("recipe JSON is nested too deeply") from None
    except json.JSONDecodeError as e:
        raise RecipeError(f"recipe is not valid JSON: {e}") from None


def load_recipe(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise RecipeError(f"recipe file not found: {p}")
    if p.stat().st_size > MAX_RECIPE_BYTES:
        raise RecipeError(f"recipe is larger than {MAX_RECIPE_BYTES // 1024} KB")
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise RecipeError("recipe is not UTF-8 text") from None
    return validate(parse_json(text))


def parse_color(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, str) and value in PALETTE:
        return PALETTE[value]
    if isinstance(value, str) and HEX_RE.match(value):
        return tuple(int(value[i:i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]
    return None


def _robot(c: _Checker, r: Any) -> dict | None:
    path = "robot"
    if not c.keys(r, path, ("file", "base", "yaw_deg", "end_effector", "gripper", "arm_joints", "home",
                           "kp", "kd", "armature", "color")):
        return None
    out: dict = {"file": c.text(r, "file", path, 200)}
    f = out["file"]
    if isinstance(f, str):
        low = f.lower()
        if not (low.endswith(".urdf") or low.endswith(".xml")):
            c.err(f"{path}.file", "must be a .urdf or an MJCF .xml file")
        if f.startswith(("/", "\\", "~")) or re.match(r"^[A-Za-z]:", f) or "://" in f:
            c.err(f"{path}.file", "must be a path relative to the recipe's folder")
        if any(part == ".." for part in re.split(r"[\\/]", f)):
            c.err(f"{path}.file", 'may not climb out of the recipe folder with ".."')
    out["base"] = c.vec(r, "base", path, 3, -5, 5, default=[0.0, 0.0, 0.0])
    out["yaw_deg"] = c.num(r, "yaw_deg", path, -360, 360, default=0.0)
    ee = r.get("end_effector", _MISSING)
    if ee is _MISSING:
        c.err(f"{path}.end_effector", 'is required: {"body": "<link name>", "offset": [x, y, z]}')
        out["end_effector"] = None
    elif c.keys(ee, f"{path}.end_effector", ("body", "offset")):
        out["end_effector"] = {
            "body": c.text(ee, "body", f"{path}.end_effector", 80),
            "offset": c.vec(ee, "offset", f"{path}.end_effector", 3, -1, 1, default=[0.0, 0.0, 0.0]),
        }
    g = r.get("gripper", None)
    out["gripper"] = None
    if g is not None and c.keys(g, f"{path}.gripper", ("joints", "open", "closed", "force")):
        joints = g.get("joints")
        if not isinstance(joints, list) or not 1 <= len(joints) <= 4 or not all(isinstance(j, str) and 0 < len(j) <= 80 for j in joints):
            c.err(f"{path}.gripper.joints", "must be a list of 1 to 4 joint names")
            joints = []
        n = max(1, len(joints))
        out["gripper"] = {
            "joints": joints,
            "open": c.vec(g, "open", f"{path}.gripper", n, -3.2, 3.2),
            "closed": c.vec(g, "closed", f"{path}.gripper", n, -3.2, 3.2),
            "force": c.num(g, "force", f"{path}.gripper", 0.1, 200, default=6.0),
        }
    aj = r.get("arm_joints", None)
    if aj is not None and (not isinstance(aj, list) or not 1 <= len(aj) <= 12 or not all(isinstance(j, str) and 0 < len(j) <= 80 for j in aj)):
        c.err(f"{path}.arm_joints", "must be a list of 1 to 12 joint names")
        aj = None
    out["arm_joints"] = aj
    home = r.get("home", {})
    out["home"] = {}
    if not isinstance(home, dict) or len(home) > 16:
        c.err(f"{path}.home", "must be an object of at most 16 joint name -> angle (radians)")
    else:
        for k, v in home.items():
            if len(k) > 80 or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 6.3:
                c.err(f"{path}.home.{k[:40]}", "must be a number between -6.3 and 6.3")
            else:
                out["home"][k] = float(v)
    out["kp"] = c.num(r, "kp", path, 1, 5000, default=400.0)
    out["kd"] = c.num(r, "kd", path, 0, 500, default=40.0)
    out["armature"] = c.num(r, "armature", path, 0, 1, default=0.01)
    col = r.get("color", "graphite")
    if parse_color(col) is None:
        c.err(f"{path}.color", f"must be one of {', '.join(PALETTE)} or #rrggbb")
        col = "graphite"
    out["color"] = col
    return out


def _objects(c: _Checker, objs: Any) -> list[dict]:
    if objs is None:
        return []
    if not isinstance(objs, list):
        c.err("objects", "must be a list")
        return []
    if len(objs) > MAX_OBJECTS:
        c.err("objects", f"at most {MAX_OBJECTS} objects, got {len(objs)}")
        return []
    out, seen = [], set()
    for i, o in enumerate(objs):
        p = f"objects[{i}]"
        if not c.keys(o, p, ("id", "shape", "size", "mass", "pos", "yaw_deg", "color", "friction", "jitter")):
            continue
        oid = c.text(o, "id", p, 40, pattern=ID_RE)
        if oid in seen:
            c.err(f"{p}.id", f'"{oid}" is used twice')
        seen.add(oid)
        shape = c.choice(o, "shape", p, SHAPES, default="box")
        n = {"box": 3, "sphere": 1, "cylinder": 2}.get(shape, 3)
        col = o.get("color", "ochre")
        if parse_color(col) is None:
            c.err(f"{p}.color", f"must be one of {', '.join(PALETTE)} or #rrggbb")
            col = "ochre"
        out.append({
            "id": oid, "shape": shape,
            "size": c.vec(o, "size", p, n, 0.005, 1.0),
            "mass": c.num(o, "mass", p, 0.001, 100, default=0.1),
            "pos": c.vec(o, "pos", p, 3, -10, 10),
            "yaw_deg": c.num(o, "yaw_deg", p, -360, 360, default=0.0),
            "color": col,
            "friction": c.num(o, "friction", p, 0.05, 3.0, default=1.0),
            "jitter": c.num(o, "jitter", p, 0, 0.05, default=0.0),
        })
    return out


def _table(c: _Checker, t: Any, task: str) -> dict | None:
    if t is False or (t is None and task in ("pendulum", "cartpole")):
        return None
    t = {} if t is None or t is True else t
    if not c.keys(t, "table", ("size", "height", "pos", "color")):
        return None
    col = t.get("color", "#c9a66b")
    if parse_color(col) is None:
        c.err("table.color", "must be a palette name or #rrggbb")
        col = "#c9a66b"
    return {
        "size": c.vec(t, "size", "table", 2, 0.1, 4, default=[0.45, 0.35]),
        "height": c.num(t, "height", "table", 0.05, 1.5, default=0.4),
        "pos": c.vec(t, "pos", "table", 2, -5, 5, default=[0.25, 0.0]),
        "color": col,
    }


def _goal(c: _Checker, g: Any, task: str, object_ids: set) -> dict:
    g = {} if g is None else g
    path = "goal"
    if task == "pick_place":
        if not c.keys(g, path, ("object", "place", "tolerance", "lift")):
            return {}
        out = {"object": c.text(g, "object", path, 40), "place": c.vec(g, "place", path, 2, -5, 5),
               "tolerance": c.num(g, "tolerance", path, 0.002, 0.5, default=0.03),
               "lift": c.num(g, "lift", path, 0.01, 0.5, default=0.08)}
    elif task == "push":
        if not c.keys(g, path, ("object", "direction_deg", "distance", "tolerance")):
            return {}
        out = {"object": c.text(g, "object", path, 40),
               "direction_deg": c.num(g, "direction_deg", path, -360, 360, default=90.0),
               "distance": c.num(g, "distance", path, 0.01, 1.0, default=0.12),
               "tolerance": c.num(g, "tolerance", path, 0.002, 0.5, default=0.03)}
    elif task == "drop":
        if not c.keys(g, path, ("rest_speed",)):
            return {}
        out = {"rest_speed": c.num(g, "rest_speed", path, 0.001, 1.0, default=0.02)}
    else:
        if not c.keys(g, path, ()):
            return {}
        out = {}
    obj = out.get("object")
    if obj is not None and obj not in object_ids:
        c.err(f"{path}.object", f'"{obj}" is not one of the objects ({", ".join(sorted(object_ids)) or "none"})')
    return out


def _pendulum(c: _Checker, p: Any) -> dict:
    p = {} if p is None else p
    if not c.keys(p, "pendulum", ("links", "length", "mass", "angle_deg", "damping", "pivot_height")):
        return {}
    return {"links": c.num(p, "links", "pendulum", 1, 2, default=1, integer=True),
            "length": c.num(p, "length", "pendulum", 0.05, 3.0, default=0.6),
            "mass": c.num(p, "mass", "pendulum", 0.01, 50, default=1.0),
            "angle_deg": c.num(p, "angle_deg", "pendulum", -179, 179, default=30.0),
            "damping": c.num(p, "damping", "pendulum", 0, 10, default=0.0),
            "pivot_height": c.num(p, "pivot_height", "pendulum", 0.2, 5, default=1.3)}


def _cartpole(c: _Checker, p: Any) -> dict:
    p = {} if p is None else p
    if not c.keys(p, "cartpole", ("pole_length", "pole_mass", "cart_mass", "angle_deg", "controller",
                                  "force_limit", "push")):
        return {}
    push = p.get("push", None)
    push_out = None
    if push is not None and c.keys(push, "cartpole.push", ("at", "impulse")):
        push_out = {"at": c.num(push, "at", "cartpole.push", 0, 60),
                    "impulse": c.num(push, "impulse", "cartpole.push", -20, 20)}
    return {"pole_length": c.num(p, "pole_length", "cartpole", 0.1, 3, default=0.6),
            "pole_mass": c.num(p, "pole_mass", "cartpole", 0.01, 20, default=0.2),
            "cart_mass": c.num(p, "cart_mass", "cartpole", 0.05, 50, default=1.0),
            "angle_deg": c.num(p, "angle_deg", "cartpole", -60, 60, default=8.0),
            "controller": c.choice(p, "controller", "cartpole", ("lqr", "none"), default="lqr"),
            "force_limit": c.num(p, "force_limit", "cartpole", 0.1, 500, default=20.0),
            "push": push_out}


def _camera(c: _Checker, cam: Any, path: str = "camera", allow_formats: bool = True) -> dict:
    cam = {} if cam is None else cam
    allowed = ("lookat", "distance", "azimuth", "elevation", "fovy") + (FORMATS if allow_formats else ())
    if not c.keys(cam, path, allowed):
        return {}
    out = {
        "lookat": c.vec(cam, "lookat", path, 3, -10, 10, default=None),
        "distance": c.num(cam, "distance", path, 0.1, 20, default=None),
        "azimuth": c.num(cam, "azimuth", path, -360, 360, default=None),
        "elevation": c.num(cam, "elevation", path, -89, 89, default=None),
        "fovy": c.num(cam, "fovy", path, 10, 120, default=None),
    }
    if allow_formats:
        for f in FORMATS:
            if f in cam:
                out[f] = _camera(c, cam[f], f"{path}.{f}", allow_formats=False)
    return out


def _narration(c: _Checker, n: Any) -> Any:
    if n is None or n == "auto":
        return "auto"
    if not isinstance(n, list) or not 1 <= len(n) <= MAX_NARRATION_LINES:
        c.err("narration", f'must be "auto" or a list of 1 to {MAX_NARRATION_LINES} lines')
        return "auto"
    out = []
    for i, line in enumerate(n):
        p = f"narration[{i}]"
        if not c.keys(line, p, ("at", "say")):
            continue
        at = line.get("at", "start")
        if isinstance(at, str):
            if not ANCHOR_RE.fullmatch(at):
                c.err(f"{p}.at", 'must be "start", "end", "event:<type>" or a time in seconds')
        elif isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at) or not 0 <= at <= 60:
            c.err(f"{p}.at", "a time must be between 0 and 60 seconds")
        say = c.text(line, "say", p, MAX_LINE_CHARS)
        if isinstance(say, str):
            if not say.strip():
                c.err(f"{p}.say", "is empty")
            for ph in PLACEHOLDER_RE.findall(say):
                if not re.fullmatch(r"[a-z_][A-Za-z0-9_.]{0,60}(:[0-9]{1})?", ph):
                    c.err(f"{p}.say", f'placeholder "{{{ph[:30]}}}" must look like {{task.final_error_cm}} or {{task.final_error_cm:1}}')
        out.append({"at": at, "say": say})
    return out


def validate(raw: Any) -> dict:
    """Return a new, complete recipe dict, or raise RecipeError listing every problem."""
    c = _Checker()
    top = ("version", "name", "title", "task", "seed", "timestep", "duration", "robot", "table", "objects",
           "goal", "pendulum", "cartpole", "camera", "overlays", "video", "voice", "narration")
    if not c.keys(raw, "recipe", top):
        raise RecipeError("; ".join(c.errors))
    if raw.get("version", 1) != 1:
        c.err("version", "only version 1 is understood")
    task = c.choice(raw, "task", "recipe", TASKS)
    r: dict = {
        "version": 1,
        "name": c.text(raw, "name", "recipe", 40, pattern=ID_RE),
        "title": c.text(raw, "title", "recipe", 80, default=""),
        "task": task,
        "seed": c.num(raw, "seed", "recipe", 0, 2**31 - 1, default=0, integer=True),
        "timestep": c.num(raw, "timestep", "recipe", 0.0005, 0.01, default=0.002),
        "duration": c.num(raw, "duration", "recipe", 0.5, 60, default=DEFAULT_DURATION.get(task, 6.0)),
    }
    if r["timestep"] and r["duration"] and r["duration"] / r["timestep"] > MAX_STEPS:
        c.err("recipe.duration", f"duration / timestep is more than {MAX_STEPS} steps; use a larger timestep or a shorter run")
    robot = raw.get("robot")
    if task in ROBOT_TASKS and robot is None:
        c.err("robot", f'is required for the "{task}" task')
    if task in ("pendulum", "cartpole") and robot is not None:
        c.err("robot", f'is not used by the "{task}" task; remove it')
        robot = None
    r["robot"] = _robot(c, robot) if robot is not None else None
    if r["robot"] and task in ROBOT_TASKS and r["robot"].get("gripper") is None and task == "pick_place":
        c.err("robot.gripper", "is required for pick_place")
    r["table"] = _table(c, raw.get("table"), task or "")
    r["objects"] = _objects(c, raw.get("objects"))
    if task in ("drop", "push", "pick_place") and not r["objects"]:
        c.err("objects", f'the "{task}" task needs at least one object')
    if task in ("pendulum", "cartpole") and r["objects"]:
        c.err("objects", f'the "{task}" task builds its own bodies; remove "objects"')
    r["goal"] = _goal(c, raw.get("goal"), task or "", {o["id"] for o in r["objects"]})
    r["pendulum"] = _pendulum(c, raw.get("pendulum")) if task == "pendulum" else None
    r["cartpole"] = _cartpole(c, raw.get("cartpole")) if task == "cartpole" else None
    for k in ("pendulum", "cartpole"):
        if k in raw and task != k:
            c.err(k, f'is only used by the "{k}" task')
    if task == "cartpole" and r["cartpole"] and r["cartpole"].get("push") and r["duration"]:
        at = r["cartpole"]["push"].get("at")
        if isinstance(at, (int, float)) and at >= r["duration"]:
            c.err("cartpole.push.at", "is after the end of the run")
    r["camera"] = _camera(c, raw.get("camera"))
    ov = raw.get("overlays", {})
    r["overlays"] = {}
    if c.keys(ov, "overlays", ("forces", "com", "torques", "hud", "captions", "force_scale", "min_force")):
        r["overlays"] = {k: c.boolean(ov, k, "overlays", True) for k in ("forces", "com", "torques", "hud", "captions")}
        r["overlays"]["force_scale"] = c.num(ov, "force_scale", "overlays", 0.0005, 1.0, default=0.04)
        r["overlays"]["min_force"] = c.num(ov, "min_force", "overlays", 0, 1000, default=0.05)
    v = raw.get("video", {})
    r["video"] = {}
    if c.keys(v, "video", ("format", "fps", "speed", "hold_end")):
        r["video"] = {"format": c.choice(v, "format", "video", FORMATS, default="landscape"),
                      "fps": c.choice(v, "fps", "video", FPS, default=30),
                      "speed": "auto" if v.get("speed", "auto") == "auto" else c.num(v, "speed", "video", 0.1, 4, default=1.0),
                      "hold_end": c.num(v, "hold_end", "video", 0, 10, default=1.2)}
    vo = raw.get("voice", {})
    r["voice"] = {}
    if c.keys(vo, "voice", ("on", "voice", "speed", "lang")):
        r["voice"] = {"on": c.boolean(vo, "on", "voice", True),
                      "voice": c.text(vo, "voice", "voice", 24, default="af_heart", pattern=VOICE_RE),
                      "speed": c.num(vo, "speed", "voice", 0.5, 2.0, default=1.0),
                      "lang": c.choice(vo, "lang", "voice", ("en-us", "en-gb"), default="en-us")}
    r["narration"] = _narration(c, raw.get("narration"))
    if c.errors:
        raise RecipeError("recipe has problems:\n  " + "\n  ".join(c.errors))
    return r
