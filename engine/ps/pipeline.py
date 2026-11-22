"""The whole run: recipe -> scene -> simulation -> script -> voice -> captions -> frames -> MP4."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import RecipeError
from .narrate import build_script
from .recipe import load_recipe
from .scene import FRAME, build_scene, write_scene
from .simulate import compile_scene, log_json, run
from .files import remove, safe_dir, save_image, write_atomic
from .timing import MAX_VIDEO_S, choose_speed, chunks, schedule, silent_timing, vtt

VOICE_TIMEOUT_S = 900

ENGINE = Path(__file__).resolve().parents[1]


def verdict(log: dict) -> str:
    """PASS or FAIL, or DONE for runs that have no pass or fail (a pendulum)."""
    if log.get("task_type") == "pendulum":
        return "DONE"
    return "PASS" if log["task"]["success"] else "FAIL"


def kokoro_python() -> str:
    if os.environ.get("PHYSICS_STUDIO_VOICE_PYTHON"):
        return os.environ["PHYSICS_STUDIO_VOICE_PYTHON"]
    home = Path(os.environ.get("KOKORO_HOME") or Path.home() / "kokoro")
    cand = home / ".venv" / "bin" / "python"
    return str(cand) if cand.is_file() else "python3"


def say(msg: str) -> None:
    print(msg, flush=True)


def outputs(out_dir: Path, name: str, fmt_suffix: str = "") -> dict[str, Path]:
    b = out_dir / f"{name}{fmt_suffix}"
    return {"mp4": b.with_suffix(".mp4"), "scene": b.with_suffix(".scene.xml"), "log": out_dir / f"{name}.log.json",
            "vtt": b.with_suffix(".vtt"), "script": b.with_suffix(".script.json"),
            "sheet": out_dir / f"{name}{fmt_suffix}-sheet.png", "poster": b.with_suffix(".jpg")}


def simulate_only(recipe_path: Path, fmt: str | None = None) -> tuple[dict, object, object]:
    recipe = load_recipe(recipe_path)
    scene = build_scene(recipe, recipe_path.parent, fmt)
    res = run(recipe, scene, record_frames=False)
    return recipe, scene, res


def voice_lines(lines: list[dict], recipe: dict, work: Path, use_voice: bool) -> tuple[dict, Path | None]:
    v = recipe["voice"]
    if not use_voice or not v["on"] or not lines:
        return {ln["id"]: silent_timing(ln["text"], v["speed"]) for ln in lines}, None
    vdir = safe_dir(work / "voice")
    safe_dir(vdir / ".cache")
    job = vdir / "job.json"
    write_atomic(job, json.dumps({"voice": v["voice"], "speed": v["speed"], "lang": v["lang"],
                                  "lines": [{"id": ln["id"], "text": ln["text"]} for ln in lines]}))
    for ln in lines:                       # the voice step writes these; never through a planted link
        remove(vdir / f"{ln['id']}.wav")
    remove(vdir / "timings.json")
    try:
        r = subprocess.run([kokoro_python(), str(ENGINE / "voice.py"), str(job), str(vdir)], capture_output=True,
                           text=True, timeout=VOICE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise RecipeError(f"the voice step took longer than {VOICE_TIMEOUT_S} s and was stopped") from None
    if r.returncode != 0:
        raise RecipeError("the voice step failed: " + (r.stderr or r.stdout).strip()[-600:]
                          + "\n  (run ./check_env.sh, or use --no-voice for captions without a voice)")
    timings = json.loads((vdir / "timings.json").read_text(encoding="utf-8"))
    return timings, vdir


def build(recipe_path: Path, out_dir: Path | None = None, fmt: str | None = None, use_voice: bool = True,
          sheet_only: bool = False) -> dict[str, Path]:
    from PIL import Image

    from .media import Encoder, mix, write_wav
    from .render import Composer, SceneRenderer, contact_sheet, frames_for_video

    t0 = time.time()
    recipe_path = recipe_path.resolve()
    recipe = load_recipe(recipe_path)
    fmt = fmt or recipe["video"]["format"]
    out_dir = safe_dir(Path(os.path.abspath(out_dir or recipe_path.parent / "out")))
    work = safe_dir(recipe_path.parent / "build")
    suffix = "" if fmt == recipe["video"]["format"] else f"-{fmt}"
    paths = outputs(out_dir, recipe["name"], suffix)
    for key in ("poster", "sheet", "mp4"):       # a stale file from an earlier build must not survive this one
        remove(paths[key])
    say(f"scene    {recipe['task']}, {fmt} {FRAME[fmt][0]}x{FRAME[fmt][1]}")
    scene = build_scene(recipe, recipe_path.parent, fmt)
    write_scene(scene, paths["scene"])
    say(f"simulate {recipe['duration']} s at {recipe['timestep'] * 1000:g} ms steps")
    log = run(recipe, scene, record_frames=False).log
    write_atomic(paths["log"], log_json(log))
    say(f"         result: {verdict(log)} - {log['task']['reason']}")
    lines, skipped = build_script(recipe, log)
    say(f"voice    {len(lines)} lines" + ("" if use_voice and recipe['voice']['on'] else " (no voice: captions only)"))
    timings, wav_dir = voice_lines(lines, recipe, work, use_voice)
    v = dict(recipe["video"])
    v["speed"] = choose_speed(lines, timings, recipe["duration"], v["hold_end"], v["speed"])
    recipe = {**recipe, "video": v}
    if v["speed"] != 1:
        why = "so the voice keeps up with the events" if wav_dir is not None else "so the captions keep up with the events"
        say(f"         playback at {v['speed']:g}x {why}")
    res = run(recipe, scene, speed=v["speed"])
    if log_json(res.log) != log_json(log):
        raise RecipeError("the second run differs from the first: the simulation is not deterministic here")
    sched = schedule(lines, timings, recipe["duration"], v["speed"], v["hold_end"])
    if sched["duration"] > MAX_VIDEO_S:
        raise RecipeError(f"the video would be {sched['duration']:.0f} s long; the limit is {MAX_VIDEO_S} s "
                          "(shorten the run or the narration, or raise video.speed)")
    cues = chunks(sched["lines"])
    write_atomic(paths["vtt"], vtt(cues))
    write_atomic(paths["script"], json.dumps({"lines": sched["lines"], "skipped": skipped, "late": sched["late"],
                                           "duration_s": sched["duration"], "run_end_s": sched["run_end"],
                                           "voiced": wav_dir is not None, "speed": v["speed"]}, indent=1) + "\n")
    comp = Composer(recipe, scene, log, cues, sched["duration"], sched["run_end"])
    model = compile_scene(scene)
    rend = SceneRenderer(model, scene)
    try:
        sheet_times = _sheet_times(log, sched, v["speed"])
        sheet = []
        for tv in sheet_times:
            i = min(int(round(tv * v["fps"])), len(res.frames) - 1)
            img = comp.compose(rend.render(res.frames[i].qpos), res.frames[i], tv)
            sheet.append((img, f"t_video {tv:5.2f} s   t_sim {res.frames[i].t:5.2f} s"))
        save_image(contact_sheet(sheet, cols=3 if fmt != "vertical" else 4), paths["sheet"])
        say(f"sheet    {paths['sheet']}")
        if sheet_only:
            return paths
        audio = work / f"{recipe['name']}{suffix}.wav"
        write_wav(audio, mix(sched["lines"], wav_dir, sched["duration"]))
        enc = Encoder(paths["mp4"], scene.width, scene.height, v["fps"], audio, sched["duration"])
        last_i, last_img = -1, None
        n = 0
        poster_t = _poster_time(log, v["speed"])
        try:
            for i, tv in frames_for_video(res.frames, sched["duration"], v["fps"]):
                if i != last_i:
                    last_img, last_i = rend.render(res.frames[i].qpos), i
                frame = comp.compose(last_img, res.frames[i], tv)
                if abs(tv - poster_t) < 0.5 / v["fps"]:
                    save_image(frame, paths["poster"], quality=88)
                enc.write(frame.tobytes())
                n += 1
        finally:
            enc.close()
        if not paths["poster"].exists():
            save_image(frame, paths["poster"], quality=88)
        say(f"video    {paths['mp4']}  ({n} frames, {sched['duration']:.2f} s, built in {time.time() - t0:.0f} s)")
    finally:
        rend.close()
    return paths


def _sheet_times(log: dict, sched: dict, speed: float) -> list[float]:
    run_end = sched["run_end"]
    marks = [0.0]
    for kind in ("grasp", "lift_off", "carry", "impact", "push", "open", "rest"):
        e = next((e for e in log["events"] if e["type"] == kind), None)
        if e:
            marks.append(e["t"] / speed + 0.15)
    marks += [run_end * f for f in (0.25, 0.5, 0.75)]
    marks.append(min(sched["duration"] - 0.05, run_end + 0.6))
    marks = sorted({round(m, 2) for m in marks if m < sched["duration"]})
    while len(marks) > 6:   # keep the first, the last and an even spread between
        gaps = [(marks[i + 1] - marks[i - 1], i) for i in range(1, len(marks) - 1)]
        marks.pop(min(gaps)[1])
    return marks


def _poster_time(log: dict, speed: float) -> float:
    for kind in ("lift_off", "grasp", "impact", "push"):
        e = next((e for e in log["events"] if e["type"] == kind), None)
        if e:
            return round(e["t"] / speed + 0.4, 2)
    return round(log["duration_s"] / speed * 0.4, 2)


