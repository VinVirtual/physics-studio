"""physics-studio command line. Run through the ./physics-studio launcher."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from ps import RecipeError  # noqa: E402

HELP = """physics-studio <command> ...

  check [--no-voice]            check MuJoCo, offscreen rendering, ffmpeg and the Kokoro voice
  new <dir> [--task T]          start a project: recipe.json (+ robot/tern.urdf for arm tasks)
                                tasks: pick_place, push, drop, pendulum, cartpole
  validate <recipe.json>        check a recipe (and its robot file) without running it
  inspect <robot.urdf|xml>      joints, limits and bodies of a robot file   [--json]
  scene <recipe.json>           write out/<name>.scene.xml (the generated MJCF) only
  sim <recipe.json>             run the simulation, write the scene and out/<name>.log.json
  sheet <recipe.json>           simulation + a contact sheet PNG of overlaid frames, no video
  build <recipe.json>           everything: scene, run log, voice, captions (VTT), MP4, sheet, poster
      --format landscape|vertical|square   --out <dir>   --no-voice
  test                          run the test suite
"""


def cmd_check(a) -> int:
    ok = True

    def line(good: bool, what: str, fix: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"{'ok  ' if good else 'MISS'} {what}" + (f"\n       {fix}" if fix and not good else ""))

    try:
        import mujoco
        line(True, f"mujoco {mujoco.__version__}")
    except ImportError:
        line(False, "mujoco (Python package)", "uv pip install --python .venv/bin/python mujoco numpy pillow imageio-ffmpeg pytest")
        return 1
    found = None
    probe = ("import mujoco;m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\".1\"/></worldbody></mujoco>');"
             "r=mujoco.Renderer(m,32,32);d=mujoco.MjData(m);mujoco.mj_forward(m,d);r.update_scene(d);print(r.render().mean())")
    for gl in ([os.environ["PHYSICS_STUDIO_GL"]] if os.environ.get("PHYSICS_STUDIO_GL") else ["egl", "osmesa"]):
        try:
            r = subprocess.run([sys.executable, "-I", "-c", probe], env={**os.environ, "MUJOCO_GL": gl},
                               capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            continue
        if r.returncode == 0:
            found = gl
            break
    line(found is not None, f"offscreen rendering (MUJOCO_GL={found})" if found else "offscreen rendering",
         "install EGL (apt install libegl1 libgl1-mesa-dri) or OSMesa (apt install libosmesa6); then set MUJOCO_GL=egl or osmesa")
    if found and found != os.environ.get("MUJOCO_GL"):
        print(f"       note: export MUJOCO_GL={found} before running physics-studio")
    try:
        from ps.media import ffmpeg
        line(True, f"ffmpeg ({ffmpeg()})")
    except RecipeError as e:
        line(False, "ffmpeg", str(e))
    if a.no_voice:
        print("all good (voice not checked: --no-voice)" if ok else "something is missing: fix the MISS lines")
        return 0 if ok else 1
    from ps.pipeline import kokoro_python
    py = kokoro_python()
    r = subprocess.run([py, "-I", "-c", "import kokoro_onnx, soundfile, numpy"], capture_output=True, timeout=120)
    line(r.returncode == 0, f"kokoro-onnx voice ({py})",
         "python3 -m venv ~/kokoro/.venv && ~/kokoro/.venv/bin/pip install kokoro-onnx soundfile numpy "
         "(or set PHYSICS_STUDIO_VOICE_PYTHON); without it, build with --no-voice")
    home = Path(os.environ.get("KOKORO_HOME") or Path.home() / "kokoro")
    have = all((home / f).is_file() for f in ("kokoro-v1.0.onnx", "voices-v1.0.bin"))
    line(have, f"kokoro model files ({home})",
         "download kokoro-v1.0.onnx and voices-v1.0.bin from https://github.com/thewh1teagle/kokoro-onnx/releases "
         "(model-files-v1.0) into ~/kokoro, or set KOKORO_HOME")
    print("all good" if ok else "something is missing: fix the MISS lines (a video without a voice still works with --no-voice)")
    return 0 if ok else 1


TEMPLATES = {"pick_place": "pick-cube.json", "push": "push-block.json", "drop": "drop-blocks.json",
             "pendulum": "pendulum.json", "cartpole": "cartpole.json"}


def cmd_new(a) -> int:
    d = Path(a.dir)
    if (d / "recipe.json").exists():
        print(f"physics-studio: {d}/recipe.json already exists; not overwriting", file=sys.stderr)
        return 1
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "templates" / "recipes" / TEMPLATES[a.task], d / "recipe.json")
    if a.task in ("pick_place", "push"):
        (d / "robot").mkdir(exist_ok=True)
        shutil.copy(ROOT / "templates" / "robots" / "tern.urdf", d / "robot" / "tern.urdf")
    print(f"wrote {d}/recipe.json" + (" and robot/tern.urdf" if a.task in ("pick_place", "push") else ""))
    print(f"next: edit it, then ./physics-studio validate {d}/recipe.json and ./physics-studio build {d}/recipe.json")
    return 0


def cmd_validate(a) -> int:
    from ps.recipe import load_recipe
    from ps.scene import build_scene
    p = Path(a.recipe)
    r = load_recipe(p)
    s = build_scene(r, p.parent.resolve())
    from ps.simulate import compile_scene
    m = compile_scene(s)
    print(f"ok: {r['name']} ({r['task']}), {m.nbody} bodies, {m.njnt} joints, {m.nu} motors, "
          f"{int(round(r['duration'] / r['timestep']))} steps")
    if s.arm:
        print(f"   arm joints: {', '.join(s.arm)}; gripper: {', '.join(s.gripper) or 'none'}")
    return 0


def cmd_inspect(a) -> int:
    from ps.recipe import validate
    from ps.scene import robot_joints
    p = Path(a.robot).resolve()
    recipe = validate({"name": "inspect", "task": "drop", "objects": [{"id": "x", "size": [0.01, 0.01, 0.01], "pos": [5, 5, 0]}],
                       "robot": {"file": p.name, "end_effector": {"body": "_"}}})
    robot, joints = robot_joints(recipe, p.parent)
    bodies = sorted({b.get("name") for top in robot.bodies for b in top.iter("body") if b.get("name")})
    if a.json:
        print(json.dumps({"joints": joints, "bodies": bodies, "source": robot.source, "mesh_files": len(robot.files)}, indent=1))
        return 0
    print(f"{p.name} ({robot.source}): {len(joints)} joints, {len(bodies)} bodies, {len(robot.files)} mesh files")
    for j in joints:
        rng = f"{j['range'][0]:.3f} .. {j['range'][1]:.3f}" if j["range"] else "unlimited"
        print(f"  {j['name']:<22} {j['type']:<6} range {rng:<20} effort {j['effort'] if j['effort'] is not None else '-'}")
    print("  bodies: " + ", ".join(bodies))
    return 0


def cmd_scene(a) -> int:
    from ps.recipe import load_recipe
    from ps.scene import build_scene, write_scene
    p = Path(a.recipe).resolve()
    r = load_recipe(p)
    s = build_scene(r, p.parent, a.format)
    from ps.files import safe_dir
    out = safe_dir(Path(os.path.abspath(a.out or p.parent / "out")))
    write_scene(s, out / f"{r['name']}.scene.xml")
    print(f"wrote {out / (r['name'] + '.scene.xml')}")
    return 0


def cmd_sim(a) -> int:
    from ps.pipeline import outputs, simulate_only, verdict
    from ps.scene import write_scene
    from ps.simulate import log_json
    p = Path(a.recipe).resolve()
    r, s, res = simulate_only(p, a.format)
    from ps.files import safe_dir, write_atomic
    out = safe_dir(Path(os.path.abspath(a.out or p.parent / "out")))
    paths = outputs(out, r["name"])
    write_scene(s, paths["scene"])
    write_atomic(paths["log"], log_json(res.log))
    t = res.log["task"]
    print(f"{verdict(res.log)}: {t['reason']}")
    print(f"wrote {paths['log']} and {paths['scene']}")
    return 0


def cmd_build(a, sheet_only: bool = False) -> int:
    from ps.pipeline import build
    build(Path(a.recipe), Path(a.out) if a.out else None, a.format, use_voice=not a.no_voice, sheet_only=sheet_only)
    return 0


def cmd_test(_a) -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "-q", "-rs", str(ROOT / "tests")], cwd=ROOT)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("help", "-h", "--help"):
        print(HELP)
        return 0
    ap = argparse.ArgumentParser(prog="physics-studio", add_help=False)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("check").add_argument("--no-voice", action="store_true")
    n = sub.add_parser("new")
    n.add_argument("dir")
    n.add_argument("--task", default="pick_place", choices=sorted(TEMPLATES))
    for name in ("validate",):
        sub.add_parser(name).add_argument("recipe")
    i = sub.add_parser("inspect")
    i.add_argument("robot")
    i.add_argument("--json", action="store_true")
    for name in ("scene", "sim", "sheet", "build"):
        s = sub.add_parser(name)
        s.add_argument("recipe")
        s.add_argument("--format", choices=["landscape", "vertical", "square"])
        s.add_argument("--out")
        s.add_argument("--no-voice", action="store_true")
    sub.add_parser("test")
    a = ap.parse_args(argv)
    table = {"check": cmd_check, "new": cmd_new, "validate": cmd_validate, "inspect": cmd_inspect, "scene": cmd_scene,
             "sim": cmd_sim, "sheet": lambda x: cmd_build(x, True), "build": cmd_build, "test": cmd_test}
    try:
        return table[a.cmd](a)
    except RecipeError as e:
        print(f"physics-studio: {e}", file=sys.stderr)
        return 2
    except RecursionError:
        print("physics-studio: an input is nested too deeply to read", file=sys.stderr)
        return 2
    except OSError as e:          # a missing or dangling file, a permission problem, a full disk
        print(f"physics-studio: {e.strerror or e}: {e.filename or ''}".rstrip(": "), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
