"""Recipe validation: good recipes get defaults, bad and hostile ones are refused with a clear message."""
import json

import pytest
from conftest import deep, recipe_template

from ps import RecipeError
from ps.recipe import MAX_OBJECTS, load_recipe, parse_color, parse_json, validate

PICK = recipe_template("pick-cube.json")
PEND = recipe_template("pendulum.json")
CART = recipe_template("cartpole.json")
DROP = recipe_template("drop-blocks.json")
PUSH = recipe_template("push-block.json")


def bad(raw, *needles):
    with pytest.raises(RecipeError) as e:
        validate(raw)
    for n in needles:
        assert n in str(e.value), str(e.value)
    return str(e.value)


@pytest.mark.parametrize("raw", [PICK, PEND, CART, DROP, PUSH])
def test_templates_are_valid(raw):
    r = validate(deep(raw))
    assert r["name"] == raw["name"] and r["task"] == raw["task"]


def test_defaults_are_filled():
    r = validate({"name": "p", "task": "pendulum"})
    assert r["timestep"] == 0.002 and r["duration"] == 6.0 and r["seed"] == 0
    assert r["video"] == {"format": "landscape", "fps": 30, "speed": "auto", "hold_end": 1.2}
    assert r["voice"]["voice"] == "af_heart" and r["narration"] == "auto"
    assert r["pendulum"]["links"] == 1 and r["table"] is None
    assert all(r["overlays"][k] for k in ("forces", "com", "torques", "hud", "captions"))


def test_input_is_not_mutated():
    raw = deep(PICK)
    before = json.dumps(raw, sort_keys=True)
    validate(raw)
    assert json.dumps(raw, sort_keys=True) == before


def test_not_an_object():
    bad([1, 2], "must be an object")
    bad("pick", "must be an object")


def test_unknown_top_level_key():
    bad({**deep(PEND), "exec": "rm -rf /"}, 'unknown key "exec"')


def test_unknown_nested_key():
    raw = deep(PICK)
    raw["objects"][0]["script"] = "x"
    bad(raw, 'objects[0]: unknown key "script"')


def test_task_required_and_checked():
    bad({"name": "x"}, "task: is required")
    bad({"name": "x", "task": "fly"}, "must be one of")


def test_name_pattern():
    bad({**deep(PEND), "name": "../evil"}, "name")
    bad({**deep(PEND), "name": "a" * 41}, "longer than 40")


def test_bool_is_not_a_number():
    bad({**deep(PEND), "seed": True}, "seed: must be a whole number")
    bad({**deep(PEND), "duration": False}, "duration: must be a number")


def test_number_ranges():
    bad({**deep(PEND), "timestep": 0.5}, "timestep: must be between")
    bad({**deep(PEND), "duration": -1}, "duration")
    bad({**deep(PEND), "seed": 2**40}, "seed")


def test_step_cap():
    bad({**deep(PEND), "timestep": 0.0005, "duration": 60}, "steps")


def test_nan_and_infinity_refused_by_parser():
    for txt in ('{"name":"p","task":"pendulum","duration":NaN}', '{"name":"p","task":"pendulum","duration":Infinity}',
                '{"name":"p","task":"pendulum","duration":-Infinity}'):
        with pytest.raises(RecipeError, match="not allowed"):
            parse_json(txt)


def test_duplicate_keys_refused():
    with pytest.raises(RecipeError, match="duplicate key"):
        parse_json('{"name":"p","task":"pendulum","task":"drop"}')


def test_deep_nesting_refused():
    with pytest.raises(RecipeError):
        parse_json("[" * 100000 + "]" * 100000)


def test_huge_recipe_refused(tmp_path):
    p = tmp_path / "r.json"
    p.write_text('{"name": "p", "task": "pendulum", "title": "' + "x" * 300000 + '"}')
    with pytest.raises(RecipeError, match="larger than"):
        load_recipe(p)


def test_not_utf8(tmp_path):
    p = tmp_path / "r.json"
    p.write_bytes(b'{"name": "\xff\xfe"}')
    with pytest.raises(RecipeError, match="UTF-8"):
        load_recipe(p)


def test_missing_file(tmp_path):
    with pytest.raises(RecipeError, match="not found"):
        load_recipe(tmp_path / "nope.json")


def test_invalid_json(tmp_path):
    p = tmp_path / "r.json"
    p.write_text("{name: 1}")
    with pytest.raises(RecipeError, match="not valid JSON"):
        load_recipe(p)


@pytest.mark.parametrize("path", ["/etc/passwd.urdf", "../robot.urdf", "robot/../../x.urdf", "~/r.urdf",
                                  "file:///r.urdf", "C:/r.urdf", "http://x/r.urdf"])
def test_robot_path_must_stay_inside(path):
    raw = deep(PICK)
    raw["robot"]["file"] = path
    bad(raw, "robot.file")


def test_robot_file_extension():
    raw = deep(PICK)
    raw["robot"]["file"] = "robot/tern.py"
    bad(raw, ".urdf or an MJCF .xml")


def test_robot_required_for_arm_tasks():
    raw = deep(PICK)
    del raw["robot"]
    bad(raw, 'robot: is required for the "pick_place" task')


def test_robot_refused_for_pendulum():
    raw = {**deep(PEND), "robot": deep(PICK["robot"])}
    bad(raw, "is not used")


def test_gripper_required_for_pick():
    raw = deep(PICK)
    del raw["robot"]["gripper"]
    bad(raw, "gripper")


def test_gripper_lengths_must_match():
    raw = deep(PICK)
    raw["robot"]["gripper"]["open"] = [0.04]
    bad(raw, "gripper.open")


def test_object_cap():
    raw = deep(DROP)
    raw["objects"] = [{"id": f"o{i}", "size": [0.01, 0.01, 0.01], "pos": [0, 0, 0.1]} for i in range(MAX_OBJECTS + 1)]
    bad(raw, f"at most {MAX_OBJECTS}")


def test_object_size_by_shape():
    raw = deep(DROP)
    raw["objects"][1]["size"] = [0.03, 0.03]       # a sphere takes one radius
    bad(raw, "objects[1].size")


def test_object_ids_unique_and_safe():
    raw = deep(DROP)
    raw["objects"][1]["id"] = "cube"
    bad(raw, "used twice")
    raw = deep(DROP)
    raw["objects"][0]["id"] = "a b"
    bad(raw, "objects[0].id")


def test_goal_object_must_exist():
    raw = deep(PICK)
    raw["goal"]["object"] = "sphere"
    bad(raw, '"sphere" is not one of the objects')


def test_pendulum_objects_refused():
    bad({**deep(PEND), "objects": deep(DROP["objects"])}, "builds its own bodies")


def test_section_for_other_task_refused():
    bad({**deep(PEND), "cartpole": {}}, 'only used by the "cartpole" task')


def test_push_after_end_refused():
    raw = deep(CART)
    raw["cartpole"]["push"]["at"] = 99
    bad(raw, "cartpole.push.at")


def test_colours():
    assert parse_color("ochre") and parse_color("#ff0000") == (1.0, 0.0, 0.0)
    assert parse_color("red") is None and parse_color("#ff00") is None and parse_color(3) is None
    raw = deep(DROP)
    raw["objects"][0]["color"] = "url(javascript:x)"
    bad(raw, "objects[0].color")


def test_control_characters_refused():
    bad({**deep(PEND), "title": "hi\x1b[31m"}, "control characters")


def test_video_options():
    bad({**deep(PEND), "video": {"fps": 29}}, "video.fps")
    bad({**deep(PEND), "video": {"format": "cinema"}}, "video.format")
    bad({**deep(PEND), "video": {"speed": 100}}, "video.speed")
    assert validate({**deep(PEND), "video": {"speed": 0.5}})["video"]["speed"] == 0.5


def test_voice_name_pattern():
    bad({**deep(PEND), "voice": {"voice": "../../etc"}}, "voice.voice")


def test_narration_lines():
    ok = validate({**deep(PEND), "narration": [{"at": "start", "say": "Period {task.measured_period_s:2} s."},
                                                {"at": 2.5, "say": "Hi."}, {"at": "event:release", "say": "Go."}]})
    assert len(ok["narration"]) == 3
    bad({**deep(PEND), "narration": [{"at": "later", "say": "x"}]}, "narration[0].at")
    bad({**deep(PEND), "narration": [{"at": "start", "say": "{__import__('os')}"}]}, "placeholder")
    bad({**deep(PEND), "narration": [{"at": "start", "say": "   "}]}, "is empty")
    bad({**deep(PEND), "narration": [{"at": "start", "say": "x" * 500}]}, "longer than")
    bad({**deep(PEND), "narration": []}, "narration")


def test_all_problems_reported_together():
    msg = bad({"name": "bad name", "task": "drop", "seed": -1, "objects": "many"}, "name", "seed", "objects")
    assert msg.count("\n") >= 3
