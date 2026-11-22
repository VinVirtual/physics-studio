"""The narration script, written from the run log so it only says what the simulation actually did.

Auto scripts branch on what happened (a grasp that never came is reported as such). Custom lines in a
recipe may use placeholders such as {task.final_error_cm} that are filled from the log; an unknown
placeholder, or one whose value is missing, is an error, and a line anchored to an event that did not
happen is left out (and listed in the script file) rather than spoken.
"""
from __future__ import annotations

import re
from typing import Any

from . import RecipeError

PH_RE = re.compile(r"\{([a-z_][A-Za-z0-9_.]*)(?::([0-9]))?\}")
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
                9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def facts(log: dict) -> dict[str, Any]:
    """Every value a narration line may use: the log flattened, plus first event times as events.<type>."""
    f = {k: v for k, v in flatten({k: v for k, v in log.items() if k not in ("events", "objects")}).items()
         if not isinstance(v, list)}
    for e in log.get("events", []):
        f.setdefault(f"events.{e['type']}", e["t"])
    for o in log.get("objects", []):
        for k, v in o.items():
            if not isinstance(v, list):
                f[f"objects.{o['id']}.{k}"] = v
    return f


def say_number(v: float, decimals: int | None = None) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if decimals is None:
        decimals = 0 if float(v).is_integer() else (1 if abs(v) >= 10 else 2)
    s = f"{v:.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def fill(text: str, f: dict[str, Any]) -> str:
    def rep(m: re.Match) -> str:
        key, dec = m.group(1), m.group(2)
        if key not in f:
            near = sorted(k for k in f if k.split(".")[0] == key.split(".")[0])[:12]
            raise RecipeError(f"narration placeholder {{{key}}} is not in the run log"
                              + (f"; similar: {', '.join(near)}" if near else ""))
        v = f[key]
        if v is None:
            raise RecipeError(f"narration placeholder {{{key}}} has no value in this run")
        if isinstance(v, (int, float)):
            return say_number(v, int(dec) if dec is not None else None)
        return str(v)
    return PH_RE.sub(rep, text)


def anchor_time(at: Any, log: dict) -> float | None:
    """Simulation time a line is tied to, or None if its event never happened."""
    if isinstance(at, (int, float)) and not isinstance(at, bool):
        return float(at)
    if at == "start":
        return 0.0
    if at == "end":
        return float(log["duration_s"])
    if isinstance(at, str) and at.startswith("event:"):
        kind = at[6:]
        times = [e["t"] for e in log.get("events", []) if e["type"] == kind]
        return float(times[0]) if times else None
    return None


def _n(k: int) -> str:
    return NUMBER_WORDS.get(k, str(k))


def _cm(m: float) -> str:
    v = round(m * 100, 1 if m * 100 < 10 else 0)
    return say_number(v, 0 if float(v).is_integer() else 1)


def _event(log: dict, kind: str, obj: str | None = None) -> dict | None:
    for e in log.get("events", []):
        if e["type"] == kind and (obj is None or e.get("object") == obj):
            return e
    return None


def _pair_max(log: dict, a: str, b: str) -> float | None:
    for p in log["contacts"]["pairs"]:
        if sorted(p["bodies"]) == sorted([a, b]):
            return p["max_normal_N"]
    return None


def _result_line(task: dict) -> str:
    if task["success"]:
        return "Task passed."
    return f"Task failed: {task['reason']}."


def _shape_word(o: dict) -> str:
    return {"box": "block" if len(set(o["size_m"])) > 1 else "cube", "sphere": "ball", "cylinder": "cylinder"}[o["shape"]]


def auto_script(recipe: dict, log: dict) -> list[dict]:
    task = log["task"]
    kind = recipe["task"]
    if any(e["type"] == "unstable" for e in log.get("events", [])):
        return [{"at": "start", "say": "This simulation became unstable."},
                {"at": "event:unstable", "say": f"The solver diverged at {say_number(anchor_time('event:unstable', log), 2)} "
                                                "seconds, so the run was stopped. Nothing after this point is shown."}]
    lines: list[dict] = []
    if kind in ("pick_place", "push"):
        obj = next(o for o in log["objects"] if o["id"] == task["object"])
        noun = _shape_word(obj)
        arm_n = log["model"]["arm_joints"]
        size = _cm(obj["size_m"][0] * 2) if obj["shape"] != "cylinder" else _cm(obj["size_m"][0] * 2)
        if kind == "pick_place":
            lines.append({"at": "start", "say": f"A {_n(arm_n)}-joint arm reaches for a {size} centimetre {noun}."})
        else:
            lines.append({"at": "start", "say": f"A {_n(arm_n)}-joint arm pushes a {size} centimetre {noun} "
                                                f"{_cm(recipe['goal']['distance'])} centimetres across the table."})
        if kind == "pick_place":
            g = _event(log, "grasp", obj["id"])
            if g:
                squeeze = _pair_max(log, obj["id"], "robot")
                text = f"Grip at {say_number(g['t'], 1)} seconds" + (
                    f", with up to {say_number(squeeze, 1)} newtons of finger force." if squeeze else ".")
                if _event(log, "lift_off", obj["id"]):
                    text += f" The {noun} rises {_cm(task['max_lift_m'])} centimetres."
                lines.append({"at": "event:grasp", "say": text})
            else:
                lines.append({"at": "event:close", "say": f"The fingers close, but never get a grip on the {noun}."})
            if g and _event(log, "let_go", obj["id"]):      # only what was really carried can be set down
                lines.append({"at": "event:let_go",
                              "say": f"It is set down {_cm(task['final_error_m'])} centimetres from the target."})
        else:
            if _event(log, "contact", obj["id"]):
                lines.append({"at": "event:contact", "say": f"The {noun} slides {_cm(task['moved_m'])} centimetres "
                                                            f"and stops {_cm(task['final_error_m'])} centimetres from the target."})
            else:
                lines.append({"at": "event:push" if _event(log, "push") else "end",
                              "say": f"The arm never touches the {noun}."})
        lines.append({"at": "end", "say": _result_line(task)})
    elif kind == "drop":
        n = len(log["objects"])
        what = log["objects"][0]["shape"] if n == 1 else "objects"
        lines.append({"at": "start", "say": f"{_n(n).capitalize()} {what if n > 1 else 'object'} "
                                            f"{'fall' if n > 1 else 'falls'} under gravity, 9.81 metres per second squared."})
        imp = _event(log, "impact")
        if imp:
            peak = log["contacts"]["max_normal_N"]
            lines.append({"at": "event:impact", "say": f"First impact at {say_number(imp['t'], 2)} seconds, "
                                                       f"at {say_number(imp.get('speed_m_s', 0), 1)} metres per second."
                                                       + (f" Peak contact force: {say_number(peak, 1)} newtons." if peak else "")})
        lines.append({"at": "end", "say": "All objects came to rest." if task["success"] else _result_line(task)})
    elif kind == "pendulum":
        p = recipe["pendulum"]
        what = (f"A {say_number(p['length'], 2)} metre pendulum" if p["links"] == 1
                else f"A double pendulum, {say_number(p['length'], 2)} metres per link,")
        lines.append({"at": "start", "say": f"{what} is released from {say_number(p['angle_deg'])} degrees."})
        if task.get("measured_period_s"):
            lines.append({"at": "event:period",     # spoken once two full swings have been timed
                          "say": f"Measured period: {say_number(task['measured_period_s'], 2)} seconds. "
                                 f"The small-angle formula gives {say_number(task['small_angle_period_s'], 2)} seconds."})
        else:
            lines.append({"at": log["duration_s"] * 0.35, "say": f"The links swing up to {say_number(task['max_angle_deg'])} degrees."})
        pct = task["energy_change_pct"]
        lines.append({"at": "end", "say": f"Over {say_number(log['duration_s'])} seconds, the total energy changed by "
                                          f"{say_number(abs(pct), 2)} percent" + (" with damping on." if p["damping"] > 0 else ".")})
    elif kind == "cartpole":
        c = recipe["cartpole"]
        if c["controller"] == "lqr":
            verb = "balances" if task["success"] else "tries to balance"
            lines.append({"at": "start", "say": f"An LQR controller {verb} a {say_number(c['pole_length'], 1)} metre pole on a cart, "
                                                f"starting {say_number(c['angle_deg'])} degrees from upright."})
        else:
            lines.append({"at": "start", "say": f"With no controller, a pole starts {say_number(c['angle_deg'])} degrees from upright."})
        if c.get("push") and _event(log, "push"):
            lines.append({"at": "event:push", "say": f"At {say_number(c['push']['at'], 1)} seconds, the cart is shoved with "
                                                     f"{say_number(abs(c['push']['impulse']), 1)} newton seconds."})
        lines.append({"at": "end", "say": f"Largest lean: {say_number(task['max_angle_deg'], 1)} degrees. " + _result_line(task)})
    return lines


def build_script(recipe: dict, log: dict) -> tuple[list[dict], list[dict]]:
    """(lines to speak with their anchor time, lines left out and why)."""
    src = auto_script(recipe, log) if recipe["narration"] == "auto" else recipe["narration"]
    if any(e["type"] == "unstable" for e in log.get("events", [])) and recipe["narration"] != "auto":
        src = auto_script(recipe, log)      # custom lines may describe things that never happened
    f = facts(log)
    out, skipped = [], []
    for i, line in enumerate(src):
        t = anchor_time(line["at"], log)
        if t is None:
            skipped.append({"line": i, "at": line["at"], "why": "that event did not happen in this run"})
            continue
        text = fill(line["say"], f).strip()
        out.append({"id": f"l{i:02d}", "at": line["at"], "anchor_s": round(min(t, log["duration_s"]), 4), "text": text})
    out.sort(key=lambda ln: ln["anchor_s"])     # spoken in the order things happened (stable for ties)
    return out, skipped
