"""Narration: only facts from the run log, placeholders checked, missing events never spoken."""
import pytest
from conftest import deep, recipe_template

from ps import RecipeError
from ps.narrate import anchor_time, auto_script, build_script, facts, fill, say_number
from ps.recipe import validate

PICK_LOG = {
    "duration_s": 8.0, "mujoco_version": "x",
    "model": {"arm_joints": 5, "gripper_joints": 2},
    "objects": [{"id": "cube", "shape": "box", "size_m": [0.02, 0.02, 0.02]}],
    "events": [{"t": 0.3, "type": "reach"}, {"t": 2.4, "type": "close"}, {"t": 2.72, "type": "grasp", "object": "cube"},
               {"t": 3.28, "type": "lift_off", "object": "cube"}, {"t": 6.0, "type": "open"},
               {"t": 6.24, "type": "let_go", "object": "cube"}],
    "contacts": {"pairs": [{"bodies": ["cube", "robot"], "max_normal_N": 16.01}], "max_normal_N": 16.01, "max_t": 2.75},
    "task": {"success": True, "reason": "placed 0.2 cm from the target", "final_error_m": 0.0017, "max_lift_m": 0.081,
             "object": "cube", "final_error_cm": 0.2},
}


def test_say_number():
    assert say_number(16.0, 1) == "16" and say_number(2.718, 1) == "2.7" and say_number(1.55, 2) == "1.55"
    assert say_number(-0.0001, 2) == "0" and say_number(True) == "yes"


def test_fill_placeholders():
    f = facts(PICK_LOG)
    assert fill("Error {task.final_error_cm} cm, grip at {events.grasp:1} s.", f) == "Error 0.2 cm, grip at 2.7 s."
    with pytest.raises(RecipeError, match="not in the run log"):
        fill("{task.made_up}", f)
    with pytest.raises(RecipeError, match="no value"):
        fill("{x}", {"x": None})


def test_anchor_times():
    assert anchor_time("start", PICK_LOG) == 0 and anchor_time("end", PICK_LOG) == 8.0
    assert anchor_time("event:grasp", PICK_LOG) == 2.72 and anchor_time(1.5, PICK_LOG) == 1.5
    assert anchor_time("event:slip", PICK_LOG) is None


def test_auto_script_uses_the_log():
    r = validate(recipe_template("pick-cube.json"))
    text = " ".join(ln["say"] for ln in auto_script(r, PICK_LOG))
    assert "five-joint" in text and "4 centimetre cube" in text and "16 newtons" in text
    assert "8.1 centimetres" in text and "0.2 centimetres" in text and text.endswith("Task passed.")


def test_auto_script_reports_a_failed_grasp():
    r = validate(recipe_template("pick-cube.json"))
    log = deep(PICK_LOG)
    log["events"] = [e for e in log["events"] if e["type"] not in ("grasp", "lift_off")]
    log["task"] = {**log["task"], "success": False, "reason": "the fingers never held the object"}
    text = " ".join(ln["say"] for ln in auto_script(r, log))
    assert "never get a grip" in text and "rises" not in text
    assert text.endswith("Task failed: the fingers never held the object.")


def test_custom_lines_for_missing_events_are_skipped():
    r = validate({**recipe_template("pick-cube.json"),
                  "narration": [{"at": "start", "say": "Watch."}, {"at": "event:slip", "say": "It slipped!"},
                                {"at": "event:grasp", "say": "Grip: {contacts.max_normal_N:0} newtons."}]})
    lines, skipped = build_script(r, PICK_LOG)
    assert [ln["text"] for ln in lines] == ["Watch.", "Grip: 16 newtons."]
    assert skipped == [{"line": 1, "at": "event:slip", "why": "that event did not happen in this run"}]
    assert lines[1]["anchor_s"] == 2.72


def test_custom_line_with_unknown_fact_fails():
    r = validate({**recipe_template("pick-cube.json"), "narration": [{"at": "start", "say": "Top speed {task.top_speed}."}]})
    with pytest.raises(RecipeError):
        build_script(r, PICK_LOG)


@pytest.mark.parametrize("name", ["pendulum.json", "cartpole.json", "drop-blocks.json"])
def test_auto_scripts_from_real_runs(tmp_path, name):
    pytest.importorskip("mujoco")
    from ps.scene import build_scene
    from ps.simulate import run
    r = validate(recipe_template(name))
    log = run(r, build_scene(r, tmp_path), record_frames=False).log
    lines, skipped = build_script(r, log)
    assert lines and not skipped
    text = " ".join(ln["text"] for ln in lines)
    assert "{" not in text
    if name == "pendulum.json":
        assert f"{log['task']['measured_period_s']:.2f}".rstrip("0") in text
    if name == "cartpole.json":
        assert "Task passed." in text and "3.5 seconds" in text


def test_custom_lines_are_spoken_in_event_order():
    r = validate({**recipe_template("pick-cube.json"),
                  "narration": [{"at": "event:open", "say": "Open."}, {"at": "start", "say": "Start."}]})
    lines, _ = build_script(r, PICK_LOG)
    assert [ln["text"] for ln in lines] == ["Start.", "Open."]


def test_unstable_run_gets_an_honest_script():
    r = validate(recipe_template("pick-cube.json"))
    log = {**deep(PICK_LOG), "events": [{"t": 1.234, "type": "unstable"}],
           "task": {"success": False, "reason": "the simulation became unstable at 1.234 s"}}
    lines, _ = build_script(r, log)
    assert "unstable" in lines[0]["text"] and "1.23 seconds" in lines[1]["text"] and len(lines) == 2


def test_anchor_with_trailing_newline_refused():
    with pytest.raises(RecipeError):
        validate({**recipe_template("pendulum.json"), "narration": [{"at": "start\n", "say": "x"}]})
