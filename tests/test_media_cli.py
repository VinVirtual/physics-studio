"""Audio mix, WAV files and the command line (errors are messages, not tracebacks)."""
import json
import subprocess
import sys

import numpy as np
import pytest
from conftest import ROOT, recipe_template

from ps.media import RATE, mix, read_wav, write_wav

CLI = [sys.executable, str(ROOT / "engine" / "cli.py")]


def test_wav_roundtrip(tmp_path):
    y = (np.sin(np.linspace(0, 100, RATE)) * 0.5).astype(np.float32)
    write_wav(tmp_path / "a.wav", y)
    back = read_wav(tmp_path / "a.wav")
    assert len(back) == RATE and np.max(np.abs(back - y)) < 1e-3


def test_mix_places_lines_and_levels(tmp_path):
    tone = (np.sin(np.arange(RATE // 2) / 5) * 0.9).astype(np.float32)
    write_wav(tmp_path / "l00.wav", tone)
    write_wav(tmp_path / "l01.wav", tone)
    out = mix([{"id": "l00", "start": 0.5}, {"id": "l01", "start": 2.0}], tmp_path, 3.0)
    assert len(out) == 3 * RATE + 1
    loud = np.nonzero(np.abs(out) > 0.01)[0]
    assert loud[0] == pytest.approx(0.5 * RATE, abs=10) and loud[-1] < 2.5 * RATE + 10
    assert np.max(np.abs(out)) <= 10 ** (-1 / 20) + 1e-6
    assert np.all(out[int(1.1 * RATE):int(1.9 * RATE)] == 0)


def test_mix_without_voice_is_silent():
    assert not np.any(mix([{"id": "l00", "start": 0}], None, 1.0))


def run_cli(*args, cwd=None):
    return subprocess.run(CLI + list(args), capture_output=True, text=True, cwd=cwd, timeout=300)


def test_help():
    r = run_cli("help")
    assert r.returncode == 0 and "build <recipe.json>" in r.stdout


def test_validate_reports_problems_without_traceback(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"name": "x", "task": "pendulum", "duration": 999, "rocket": 1}))
    r = run_cli("validate", str(p))
    assert r.returncode == 2 and "Traceback" not in r.stderr
    assert "duration" in r.stderr and 'unknown key "rocket"' in r.stderr


def test_new_writes_template_and_refuses_overwrite(tmp_path):
    r = run_cli("new", str(tmp_path / "proj"), "--task", "pick_place")
    assert r.returncode == 0
    assert (tmp_path / "proj" / "recipe.json").is_file() and (tmp_path / "proj" / "robot" / "tern.urdf").is_file()
    assert run_cli("new", str(tmp_path / "proj")).returncode == 1


def test_sim_twice_writes_identical_logs(tmp_path):
    pytest.importorskip("mujoco")
    p = tmp_path / "recipe.json"
    p.write_text(json.dumps(recipe_template("cartpole.json")))
    assert run_cli("sim", str(p)).returncode == 0
    first = (tmp_path / "out" / "cartpole.log.json").read_bytes()
    scene1 = (tmp_path / "out" / "cartpole.scene.xml").read_bytes()
    assert run_cli("sim", str(p)).returncode == 0
    assert (tmp_path / "out" / "cartpole.log.json").read_bytes() == first
    assert (tmp_path / "out" / "cartpole.scene.xml").read_bytes() == scene1


def test_inspect_lists_joints():
    pytest.importorskip("mujoco")
    r = run_cli("inspect", str(ROOT / "templates" / "robots" / "tern.urdf"), "--json")
    assert r.returncode == 0, r.stderr
    info = json.loads(r.stdout)
    names = [j["name"] for j in info["joints"]]
    assert names == ["base_yaw", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "finger_left", "finger_right"]
    assert "palm" in info["bodies"]


def test_end_to_end_build_without_voice(tmp_path):
    """Recipe to MP4 with captions and no voice (needs offscreen OpenGL and ffmpeg)."""
    pytest.importorskip("mujoco")
    from conftest import GL
    if not GL:
        pytest.skip("no offscreen OpenGL (EGL or OSMesa) on this machine")
    from ps.media import probe_duration
    raw = {**recipe_template("pendulum.json"), "duration": 1.5, "video": {"format": "square", "fps": 24, "speed": 1.0}}
    p = tmp_path / "recipe.json"
    p.write_text(json.dumps(raw))
    r = run_cli("build", str(p), "--no-voice")
    assert r.returncode == 0, r.stderr
    out = tmp_path / "out"
    script = json.loads((out / "pendulum.script.json").read_text())
    assert script["voiced"] is False and script["lines"]
    dur = probe_duration(out / "pendulum.mp4")
    assert dur == pytest.approx(script["duration_s"], abs=0.1)
    assert (out / "pendulum.vtt").read_text().startswith("WEBVTT")
    from PIL import Image
    assert Image.open(out / "pendulum-sheet.png").size[0] == 1500
    assert Image.open(out / "pendulum.jpg").size == (960, 960)
