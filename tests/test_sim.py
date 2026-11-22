"""Simulation: determinism, physics sanity against theory, task outcomes and the run log."""
import math

import numpy as np
import pytest
from conftest import deep, recipe_template

mujoco = pytest.importorskip("mujoco")

from ps.recipe import validate  # noqa: E402
from ps.scene import build_scene  # noqa: E402
from ps.simulate import frame_times, log_json, pendulum_period, run  # noqa: E402


def sim(raw, folder, record=False, speed=1.0):
    r = validate(raw)
    return run(r, build_scene(r, folder), record_frames=record, speed=speed)


def test_same_seed_same_log(tmp_project):
    p = tmp_project(deep(recipe_template("pick-cube.json")))
    a = sim(recipe_template("pick-cube.json"), p.parent).log
    b = sim(recipe_template("pick-cube.json"), p.parent).log
    assert log_json(a) == log_json(b)


def test_different_seed_different_trajectory(tmp_path):
    raw = recipe_template("drop-blocks.json")
    a = sim(raw, tmp_path).log
    b = sim({**raw, "seed": raw["seed"] + 1}, tmp_path).log
    assert a["trajectory_sha256"] != b["trajectory_sha256"]


def test_recording_frames_does_not_change_the_run(tmp_path):
    raw = recipe_template("cartpole.json")
    a = sim(raw, tmp_path, record=False).log
    b = sim(raw, tmp_path, record=True, speed=0.5)
    assert log_json(a) == log_json(b.log)
    assert len(b.frames) == len(frame_times(raw["duration"], 30, 0.5))


def test_pick_and_place_succeeds(tmp_project):
    p = tmp_project(recipe_template("pick-cube.json"))
    log = sim(recipe_template("pick-cube.json"), p.parent).log
    t = log["task"]
    assert t["success"], t["reason"]
    kinds = [e["type"] for e in log["events"]]
    for k in ("reach", "close", "grasp", "lift_off", "let_go", "rest"):
        assert k in kinds
    assert kinds.index("grasp") < kinds.index("lift_off") < kinds.index("let_go")
    assert t["final_error_m"] <= t["tolerance_m"] and t["max_lift_m"] >= 0.8 * t["lift_required_m"]
    assert log["plan"]["unreachable"] == [] and log["warnings"] == []
    pair = next(q for q in log["contacts"]["pairs"] if sorted(q["bodies"]) == ["cube", "robot"])
    assert 10 < pair["max_normal_N"] < 40          # two fingers squeezing at 8 N each, plus the set-down


def test_pick_fails_honestly_when_out_of_reach(tmp_project):
    raw = recipe_template("pick-cube.json")
    raw["goal"]["place"] = [0.65, 0.3]
    raw["table"]["size"] = [0.6, 0.4]
    p = tmp_project(raw)
    log = sim(raw, p.parent).log
    assert log["plan"]["unreachable"], "far waypoints are reported"
    assert not log["task"]["success"] and "target" in log["task"]["reason"]


def test_push_moves_the_block(tmp_project):
    raw = recipe_template("push-block.json")
    p = tmp_project(raw)
    log = sim(raw, p.parent).log
    assert log["task"]["success"], log["task"]["reason"]
    assert log["task"]["moved_m"] == pytest.approx(raw["goal"]["distance"], abs=0.03)
    assert "grasp" not in [e["type"] for e in log["events"]] and "contact" in [e["type"] for e in log["events"]]


def test_resting_box_feels_its_weight(tmp_path):
    raw = {"name": "rest", "task": "drop", "duration": 1.0, "table": False,
           "objects": [{"id": "b", "size": [0.05, 0.05, 0.05], "mass": 2.0, "pos": [0, 0, 0]}]}
    r = validate(raw)
    res = run(r, build_scene(r, tmp_path), record_frames=True, speed=1.0)
    last = res.frames[-1]
    up = sum(np.array(f["f"]) for f in last.forces if f["label"] == "b<floor")
    assert up[2] == pytest.approx(2.0 * 9.81, rel=0.02)      # support force on the box points up and equals m g
    assert abs(up[0]) < 0.05 and abs(up[1]) < 0.05


def test_drop_comes_to_rest_and_impact_speed_matches_free_fall(tmp_path):
    raw = recipe_template("drop-blocks.json")
    log = sim(raw, tmp_path).log
    assert log["task"]["success"]
    imp = next(e for e in log["events"] if e["type"] == "impact" and e["object"] == "ball")
    h = raw["objects"][1]["pos"][2]
    assert imp["speed_m_s"] == pytest.approx(math.sqrt(2 * 9.81 * h), rel=0.03)
    assert imp["t"] == pytest.approx(math.sqrt(2 * h / 9.81), abs=0.01)


def test_pendulum_period_matches_theory(tmp_path):
    raw = recipe_template("pendulum.json")
    t = sim(raw, tmp_path).log["task"]
    L = raw["pendulum"]["length"]
    th = math.radians(raw["pendulum"]["angle_deg"])
    # large-amplitude series: T = T0 (1 + th^2/16 + 11 th^4/3072); the bob is a sphere, the rod is 1% of the mass
    expect = 2 * math.pi * math.sqrt(L / 9.81) * (1 + th ** 2 / 16 + 11 * th ** 4 / 3072)
    assert t["measured_period_s"] == pytest.approx(expect, rel=0.01)
    assert t["small_angle_period_s"] == pytest.approx(2 * math.pi * math.sqrt(L / 9.81), rel=1e-6)


def test_undamped_pendulum_keeps_its_energy(tmp_path):
    t = sim(recipe_template("pendulum.json"), tmp_path).log["task"]
    assert abs(t["energy_change_pct"]) < 0.5
    assert t["energy_start_J"] == pytest.approx(1.01 * 9.81 * 0.6 * (1 - math.cos(math.radians(40))), rel=0.03)


def test_damped_pendulum_loses_energy(tmp_path):
    raw = recipe_template("pendulum.json")
    raw["pendulum"]["damping"] = 0.2
    assert sim(raw, tmp_path).log["task"]["energy_change_pct"] < -20


def test_cartpole_lqr_recovers_from_push(tmp_path):
    log = sim(recipe_template("cartpole.json"), tmp_path).log
    t = log["task"]
    assert t["success"] and t["fell_at_s"] is None and t["final_angle_deg"] < 1
    assert t["settled_at_s"] > recipe_template("cartpole.json")["cartpole"]["push"]["at"]


def test_cartpole_without_controller_falls(tmp_path):
    raw = recipe_template("cartpole.json")
    raw["cartpole"]["controller"] = "none"
    t = sim(raw, tmp_path).log["task"]
    assert not t["success"] and t["fell_at_s"] is not None


def test_period_from_samples():
    samples = [(k * 0.001, math.sin(2 * math.pi * k * 0.001 / 1.25)) for k in range(5000)]
    assert pendulum_period(samples) == pytest.approx(1.25, abs=1e-3)
    assert pendulum_period([(0, 1.0), (1, 0.5)]) is None


def test_frame_times():
    assert frame_times(1.0, 30, 1.0)[-1] == pytest.approx(1.0) and len(frame_times(1.0, 30, 1.0)) == 31
    assert len(frame_times(1.0, 30, 0.5)) == 61


def test_log_is_json_and_complete(tmp_path):
    import json
    log = sim(recipe_template("drop-blocks.json"), tmp_path).log
    back = json.loads(log_json(log))
    for k in ("recipe", "task", "task_type", "seed", "mujoco_version", "timestep_s", "steps", "events", "contacts", "actuators",
              "warnings", "trajectory_sha256", "objects", "model"):
        assert k in back
    assert back["contacts"]["max_normal_N"] == max(p["max_normal_N"] for p in back["contacts"]["pairs"])


def test_short_run_does_not_claim_later_phases(tmp_project):
    raw = recipe_template("pick-cube.json")
    raw["duration"] = 3.0
    p = tmp_project(raw)
    log = sim(raw, p.parent).log
    kinds = [e["type"] for e in log["events"]]
    assert "open" not in kinds and "carry" not in kinds and log["plan"]["cut_short"]
    assert not log["task"]["success"]
    from ps.narrate import build_script
    text = " ".join(ln["text"] for ln in build_script(validate(raw), log)[0])
    assert "set down" not in text and "Task failed" in text


def test_object_resting_from_the_start_counts_as_resting(tmp_path):
    raw = {"name": "still", "task": "drop", "duration": 1.0,
           "objects": [{"id": "b", "size": [0.03, 0.03, 0.03], "pos": [0.25, 0, 0]}]}
    log = sim(raw, tmp_path).log
    assert log["task"]["success"] and not [e for e in log["events"] if e["type"] == "rest"]
