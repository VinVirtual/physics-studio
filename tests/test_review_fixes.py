"""Regression tests for the second review: resource budgets, isolated probes, output links,
narration gating, card layout, video caps, numbers in custom lines, odd inputs, VTT escaping."""
import json
import os
import re
import struct
import subprocess
import sys

import pytest
from conftest import ROOT, deep, recipe_template

from ps import RecipeError
from ps.recipe import parse_json, validate
from ps.safety import MAX_MESH_FACES, AssetStore, check_mjcf_elements, parse_xml_bytes, png_texels

CLI = [sys.executable, str(ROOT / "engine" / "cli.py")]


def png_header(w: int, h: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", w, h) + b"\x08\x02\x00\x00\x00" + b"\0" * 4


def binary_stl(faces: int) -> bytes:
    return b" " * 80 + struct.pack("<I", faces) + b"\0" * (50 * faces)


# --- H1: budgets over the whole robot ---------------------------------------------------------

def test_png_dimensions_are_read_before_decoding(tmp_path):
    (tmp_path / "huge.png").write_bytes(png_header(16000, 16000))
    with pytest.raises(RecipeError, match="16000x16000"):
        AssetStore(tmp_path).add("huge.png", (".png",), "texture")
    assert png_texels(png_header(64, 32)) == 2048
    with pytest.raises(RecipeError, match="not a PNG"):
        png_texels(b"GIF89a" + b"\0" * 30)


def test_generated_textures_have_a_total_budget():
    tex = "".join(f'<texture name="t{i}" type="2d" builtin="flat" width="4096" height="4096"/>' for i in range(3))
    with pytest.raises(RecipeError, match="generated textures add up"):
        check_mjcf_elements(parse_xml_bytes(f"<mujoco><asset>{tex}</asset></mujoco>".encode()))


def test_texture_and_mesh_element_counts_are_capped():
    tex = "".join(f'<texture name="t{i}" type="2d" builtin="flat" width="8" height="8"/>' for i in range(31))
    with pytest.raises(RecipeError, match="more than 30 <texture>"):
        check_mjcf_elements(parse_xml_bytes(f"<mujoco><asset>{tex}</asset></mujoco>".encode()))


def test_height_fields_have_a_total_budget():
    hf = "".join(f'<hfield name="h{i}" nrow="1024" ncol="1024" size="1 1 1 1"/>' for i in range(2))
    with pytest.raises(RecipeError, match="height fields add up"):
        check_mjcf_elements(parse_xml_bytes(f"<mujoco><asset>{hf}</asset></mujoco>".encode()))


def test_mesh_faces_count_every_reference(tmp_path):
    (tmp_path / "m.stl").write_bytes(binary_stl(40000))
    store = AssetStore(tmp_path)
    refs = MAX_MESH_FACES // 40000
    for _ in range(refs):
        store.add("m.stl", (".stl",), "mesh")         # the same file, used again and again
    assert len(store.files) == 1
    with pytest.raises(RecipeError, match="faces"):
        store.add("m.stl", (".stl",), "mesh")


def test_mujoco_runtime_errors_become_recipe_errors(monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    from ps.scene import Scene
    from ps.simulate import compile_scene

    def boom(*_a, **_k):
        raise RuntimeError("qhull failed")
    monkeypatch.setattr(mujoco.MjModel, "from_xml_string", boom)
    with pytest.raises(RecipeError, match="qhull"):
        compile_scene(Scene(xml="<mujoco/>", files={}, camera=None, width=1, height=1, object_ids=[]))


# --- M1: probes run isolated --------------------------------------------------------------------

def test_python_probes_use_isolated_mode():
    for f in ("engine/cli.py", "tests/conftest.py"):
        for call in re.findall(r'\[(?:sys\.executable|py), "(-[^"]*)", "-c"', (ROOT / f).read_text()):
            assert call == "-I", f
    assert '-I -c "import mujoco' in (ROOT / "check_env.sh").read_text()


# --- M2: outputs never follow links -----------------------------------------------------------

def test_linked_output_file_is_refused_and_target_untouched(tmp_path):
    pytest.importorskip("mujoco")
    p = tmp_path / "recipe.json"
    p.write_text(json.dumps(recipe_template("pendulum.json")))
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    (tmp_path / "out").mkdir()
    os.symlink(victim, tmp_path / "out" / "pendulum.log.json")
    r = subprocess.run(CLI + ["sim", str(p)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 2 and "symbolic link" in r.stderr
    assert victim.read_text() == "keep me"


def test_linked_output_folder_is_refused(tmp_path):
    from ps.files import safe_dir, write_atomic
    real = tmp_path / "real"
    real.mkdir()
    os.symlink(real, tmp_path / "out")
    with pytest.raises(RecipeError, match="symbolic link"):
        safe_dir(tmp_path / "out")
    write_atomic(real / "a.txt", "x")
    assert (real / "a.txt").read_text() == "x" and not [f for f in real.iterdir() if f.name.startswith(".")]


# --- M3: narration only for what happened -----------------------------------------------------

def test_no_set_down_line_without_a_grasp():
    from test_narrate import PICK_LOG
    from ps.narrate import auto_script
    log = deep(PICK_LOG)
    log["events"] = [e for e in log["events"] if e["type"] not in ("grasp", "lift_off", "let_go")]
    text = " ".join(ln["say"] for ln in auto_script(validate(recipe_template("pick-cube.json")), log))
    assert "set down" not in text


def test_push_without_contact_says_so():
    from ps.narrate import auto_script
    log = {"duration_s": 6.0, "model": {"arm_joints": 5, "gripper_joints": 2},
           "objects": [{"id": "block", "shape": "box", "size_m": [0.025, 0.04, 0.02]}],
           "events": [{"t": 2.4, "type": "push"}], "contacts": {"pairs": [], "max_normal_N": 0, "max_t": 0},
           "task": {"success": False, "reason": "the arm never touched the object", "object": "block",
                    "moved_m": 0.0, "final_error_m": 0.15}}
    text = " ".join(ln["say"] for ln in auto_script(validate(recipe_template("push-block.json")), log))
    assert "never touches" in text and "slides" not in text


# --- M4: cards fit the frame ------------------------------------------------------------------

def test_result_card_wraps_inside_a_vertical_frame():
    from ps.overlay import Painter, result_lines
    p = Painter(720, 1280)
    reason = "It was lifted 1.2 cm, less than 80% of the 8 cm asked for; it ended 14.3 cm from the target " * 2
    lines, w = result_lines(p, "TASK FAILED", reason)
    assert len(lines) > 1 and w <= 720 * 0.86 + 1e-6


def test_long_title_is_clamped():
    from ps.overlay import SS, Painter, fit_text
    p = Painter(720, 1280)
    f = p.f("sans", 22, "SemiBold")
    t = fit_text("W" * 80, f, 600 * SS)
    assert t.endswith("…") and f.getlength(t) <= 600 * SS


# --- M5: video length is capped ---------------------------------------------------------------

def test_fixed_speed_video_length_is_capped():
    with pytest.raises(RecipeError, match="longer than 120 s"):
        validate({**recipe_template("pendulum.json"), "duration": 20, "video": {"speed": 0.1}})


def test_auto_speed_never_exceeds_the_cap():
    from ps.timing import MAX_VIDEO_S, choose_speed, estimate_words
    lines = [{"id": "l00", "anchor_s": 0.0, "text": "x"}, {"id": "l01", "anchor_s": 0.1, "text": "y"}]
    tim = {i: {"duration": 20.0, "words": estimate_words("x", 20.0)} for i in ("l00", "l01")}
    s = choose_speed(lines, tim, 50.0, 1.0, "auto")
    assert 50.0 / s + 1.0 <= MAX_VIDEO_S


# --- M6: custom lines take numbers only from the log ------------------------------------------

def test_digits_in_custom_lines_are_refused():
    with pytest.raises(RecipeError, match="numbers may only come from the run log"):
        validate({**recipe_template("pendulum.json"), "narration": [{"at": "start", "say": "It lifts 50 kg."}]})
    ok = validate({**recipe_template("pendulum.json"), "narration": [{"at": "start", "say": "Period {task.measured_period_s:2} s."}]})
    assert ok["narration"][0]["say"].startswith("Period")


# --- L1: odd inputs give clean errors ---------------------------------------------------------

def test_huge_integers_are_clean_errors():
    with pytest.raises(RecipeError, match="duration"):
        validate({**recipe_template("pendulum.json"), "duration": 10 ** 400})
    with pytest.raises(RecipeError, match="cannot be read"):
        parse_json('{"seed": ' + "9" * 5000 + "}")


def test_deep_nesting_in_a_robot_is_refused():
    xml = "<mujoco><worldbody>" + "<frame>" * 100 + "</frame>" * 100 + "</worldbody></mujoco>"
    with pytest.raises(RecipeError, match="nested"):
        check_mjcf_elements(parse_xml_bytes(xml.encode()))


def test_dangling_robot_link_is_a_clean_error(tmp_path):
    raw = recipe_template("pick-cube.json")
    (tmp_path / "robot").mkdir()
    os.symlink(tmp_path / "gone.urdf", tmp_path / "robot" / "tern.urdf")
    p = tmp_path / "recipe.json"
    p.write_text(json.dumps(raw))
    r = subprocess.run(CLI + ["validate", str(p)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "Traceback" not in r.stderr


# --- L2: VTT text is escaped --------------------------------------------------------------------

def test_vtt_escapes_markup():
    from ps.timing import vtt
    out = vtt([{"start": 0, "end": 1, "words": [{"w": "<b>A&B</b>", "t": 0}, {"w": "-->", "t": 0.5}]}])
    body = out.split("\n", 4)[-1]
    assert "&lt;b&gt;A&amp;B&lt;/b&gt;" in body and "-->" not in body


# --- L6: the period line comes after two full swings ------------------------------------------

def test_period_event_marks_the_second_upward_crossing(tmp_path):
    pytest.importorskip("mujoco")
    from ps.narrate import build_script
    from ps.scene import build_scene
    from ps.simulate import run, upward_crossings
    r = validate(recipe_template("pendulum.json"))
    log = run(r, build_scene(r, tmp_path), record_frames=False).log
    ev = next(e for e in log["events"] if e["type"] == "period")
    assert ev["t"] == pytest.approx(0.75 * log["task"]["measured_period_s"] + log["task"]["measured_period_s"], rel=0.02)
    line = next(ln for ln in build_script(r, log)[0] if "Measured period" in ln["text"])
    assert line["anchor_s"] == pytest.approx(ev["t"], abs=1e-4)
    assert upward_crossings([(0, -1.0), (1, 1.0)]) == [0.5]


# --- L7: a failing ffmpeg leaves nothing behind -------------------------------------------------

def test_failed_encode_raises_and_cleans_up(tmp_path):
    from ps.media import Encoder
    out = tmp_path / "x.mp4"
    enc = Encoder(out, 64, 64, 24, tmp_path / "missing.wav", 1.0)
    try:
        enc.write(b"\0" * 64 * 64 * 3)
    except RecipeError:
        pass
    with pytest.raises(RecipeError, match="ffmpeg failed"):
        enc.close()
    assert not out.exists() and list(tmp_path.iterdir()) == []
