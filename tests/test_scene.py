"""MJCF generation: the scene compiles, things are where the recipe says, robots are sanitised."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from conftest import deep, recipe_template

mujoco = pytest.importorskip("mujoco")

from ps import RecipeError  # noqa: E402
from ps.recipe import validate  # noqa: E402
from ps.scene import FRAME, build_scene, object_starts, write_scene  # noqa: E402

PICK = recipe_template("pick-cube.json")
DROP = recipe_template("drop-blocks.json")

def tetra_stl() -> bytes:
    """A small binary STL tetrahedron (MuJoCo reads binary STL only)."""
    import struct
    v = [(0, 0, 0), (0.02, 0, 0), (0, 0.02, 0), (0, 0, 0.02)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    out = bytearray(b"physics-studio test".ljust(80, b" ")) + struct.pack("<I", len(faces))
    for f in faces:
        out += struct.pack("<3f", 0, 0, 0)
        for i in f:
            out += struct.pack("<3f", *v[i])
        out += b"\0\0"
    return bytes(out)


MJCF_ARM = """<mujoco model="twolink">
  <compiler angle="degree" meshdir="meshes"/>
  <default><joint damping="0.1"/><default class="tip"><geom rgba="1 0 0 1"/></default></default>
  <asset><mesh name="tip" file="tip.stl"/></asset>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="cylinder" size="0.05 0.02"/>
    <body name="l1" pos="0 0 0.04">
      <joint name="j1" axis="0 0 1" range="-170 170" actuatorfrcrange="-2 2"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <body name="l2" pos="0.2 0 0">
        <joint name="j2" axis="0 1 0" range="-90 90"/>
        <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.015"/>
        <geom class="tip" type="mesh" mesh="tip" pos="0.15 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="j1"/></actuator>
</mujoco>"""


def mjcf_project(tmp_path, xml=MJCF_ARM):
    (tmp_path / "robot" / "meshes").mkdir(parents=True)
    (tmp_path / "robot" / "meshes" / "tip.stl").write_bytes(tetra_stl())
    (tmp_path / "robot" / "arm.xml").write_text(xml)
    raw = {"name": "m", "task": "drop", "robot": {"file": "robot/arm.xml", "end_effector": {"body": "l2", "offset": [0.15, 0, 0]}},
           "objects": [{"id": "cube", "size": [0.02, 0.02, 0.02], "pos": [0.3, 0.1, 0.1]}]}
    return validate(raw)


def compile_(scene):
    return mujoco.MjModel.from_xml_string(scene.xml, assets=scene.files)


def test_pick_scene_compiles_with_motors(tmp_project):
    p = tmp_project(deep(PICK))
    s = build_scene(validate(deep(PICK)), p.parent)
    m = compile_(s)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    assert names == [f"ps_act_{j}" for j in s.arm + s.gripper]
    assert s.arm == ["base_yaw", "shoulder", "elbow", "wrist_pitch", "wrist_roll"]
    shoulder = names.index("ps_act_shoulder")
    assert tuple(m.actuator_ctrlrange[shoulder]) == (-6.0, 6.0)      # the URDF effort becomes the motor limit
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ps_ee") >= 0
    assert (m.vis.global_.offwidth, m.vis.global_.offheight) == FRAME["landscape"]


def test_objects_sit_on_the_surface_below(tmp_project):
    r = validate(deep(PICK))
    starts, surf = object_starts(r)
    cube = r["objects"][0]
    assert surf["cube"] == r["table"]["height"]
    assert starts["cube"][2] == pytest.approx(r["table"]["height"] + cube["pos"][2] + cube["size"][2])
    off = deep(PICK)
    off["objects"][0]["pos"] = [3.0, 3.0, 0.0]    # not over the table: on the floor
    s2, f2 = object_starts(validate(off))
    assert f2["cube"] == 0.0 and s2["cube"][2] == pytest.approx(0.02)


def test_jitter_is_seeded():
    a = validate(deep(PICK))
    b = validate(deep(PICK))
    c = validate({**deep(PICK), "seed": 8})
    assert object_starts(a) == object_starts(b)
    assert object_starts(a)[0]["cube"] != object_starts(c)[0]["cube"]
    j = a["objects"][0]["jitter"]
    d = np.array(object_starts(a)[0]["cube"][:2]) - np.array(a["objects"][0]["pos"][:2])
    assert np.all(np.abs(d) <= j + 1e-12)


def test_camera_element_matches_projection_camera(tmp_project):
    p = tmp_project(deep(PICK))
    s = build_scene(validate(deep(PICK)), p.parent)
    m = compile_(s)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ps_cam")
    assert np.allclose(d.cam_xpos[cid], s.camera.pos, atol=1e-5)
    R = d.cam_xmat[cid].reshape(3, 3)
    assert np.allclose(R[:, 0], s.camera.xaxis, atol=1e-5) and np.allclose(R[:, 1], s.camera.yaxis, atol=1e-5)


@pytest.mark.parametrize("fmt", ["landscape", "vertical", "square"])
def test_formats_change_frame_size(tmp_project, fmt):
    p = tmp_project(deep(DROP))
    s = build_scene(validate(deep(DROP)), p.parent, fmt)
    assert (s.width, s.height) == FRAME[fmt]
    assert f'offwidth="{FRAME[fmt][0]}"' in s.xml


def test_unknown_gripper_joint(tmp_project):
    raw = deep(PICK)
    raw["robot"]["gripper"]["joints"] = ["finger_left", "thumb"]
    p = tmp_project(raw)
    with pytest.raises(RecipeError, match='"thumb" not found'):
        build_scene(validate(raw), p.parent)


def test_unknown_end_effector_body(tmp_project):
    raw = deep(PICK)
    raw["robot"]["end_effector"]["body"] = "hand"
    p = tmp_project(raw)
    with pytest.raises(RecipeError, match='end_effector body "hand" not found'):
        build_scene(validate(raw), p.parent)


def test_missing_robot_file(tmp_project):
    raw = deep(PICK)
    raw["robot"]["file"] = "robot/none.urdf"
    p = tmp_project(raw)
    with pytest.raises(RecipeError, match="not found"):
        build_scene(validate(raw), p.parent)


def test_urdf_with_doctype_refused(tmp_project):
    p = tmp_project(deep(PICK))
    f = p.parent / "robot" / "tern.urdf"
    f.write_text('<!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/hostname">]>' + f.read_text().split("?>", 1)[1])
    with pytest.raises(RecipeError, match="DOCTYPE"):
        build_scene(validate(deep(PICK)), p.parent)


def test_urdf_mesh_outside_folder_refused(tmp_project):
    p = tmp_project(deep(PICK))
    f = p.parent / "robot" / "tern.urdf"
    f.write_text(f.read_text().replace('<cylinder radius="0.075" length="0.06"/>', '<mesh filename="file:///etc/passwd"/>', 1))
    with pytest.raises(RecipeError, match="file://"):
        build_scene(validate(deep(PICK)), p.parent)


def test_mjcf_robot_with_mesh_defaults_and_degrees(tmp_path):
    r = mjcf_project(tmp_path)
    s = build_scene(r, tmp_path)
    m = compile_(s)
    j1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "j1")
    assert m.jnt_range[j1][1] == pytest.approx(np.radians(170))      # degrees kept as degrees
    assert m.dof_damping[m.jnt_dofadr[j1]] == pytest.approx(0.1)      # the robot's defaults still apply
    assert list(s.files) == ["a000_tip.stl"]
    assert m.nu == 2 and s.arm == ["j1", "j2"]                         # the robot's own actuator was replaced
    assert "meshdir" not in s.xml and "/" not in json.dumps(list(s.files))
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ps_floor")
    assert m.geom_contype[floor] == 1                                  # robot defaults do not leak into the studio


def test_mjcf_unnamed_mesh_keeps_its_name(tmp_path):
    xml = MJCF_ARM.replace('<mesh name="tip" file="tip.stl"/>', '<mesh file="tip.stl"/>')
    m = compile_(build_scene(mjcf_project(tmp_path, xml), tmp_path))
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "tip") >= 0


def test_meshes_may_sit_beside_the_robot_folder(tmp_path):
    """The common urdf/ + meshes/ layout: meshes outside the robot file's folder but inside the recipe folder."""
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes" / "tip.stl").write_bytes(tetra_stl())
    (tmp_path / "urdf").mkdir()
    (tmp_path / "urdf" / "arm.xml").write_text(MJCF_ARM.replace('meshdir="meshes"', 'meshdir="../meshes"'))
    r = validate({"name": "m", "task": "drop", "robot": {"file": "urdf/arm.xml", "end_effector": {"body": "l2"}},
                  "objects": [{"id": "cube", "size": [0.02, 0.02, 0.02], "pos": [0.3, 0.1, 0.1]}]})
    assert compile_(build_scene(r, tmp_path)).nmesh == 1


def test_mjcf_include_refused(tmp_path):
    with pytest.raises(RecipeError, match="<include>"):
        build_scene(mjcf_project(tmp_path, MJCF_ARM.replace("<worldbody>", '<include file="other.xml"/><worldbody>')), tmp_path)


def test_mjcf_mesh_path_escape_refused(tmp_path):
    with pytest.raises(RecipeError, match="'..'"):
        build_scene(mjcf_project(tmp_path, MJCF_ARM.replace('file="tip.stl"', 'file="../../../tip.stl"')), tmp_path)


def test_mjcf_other_file_attributes_refused(tmp_path):
    xml = MJCF_ARM.replace("<asset>", '<asset><hfield name="h" size="1 1 1 1" file="tip.png"/>')
    with pytest.raises(RecipeError):
        build_scene(mjcf_project(tmp_path, xml), tmp_path)


def test_write_scene_is_standalone(tmp_path):
    r = mjcf_project(tmp_path)
    s = build_scene(r, tmp_path)
    out = tmp_path / "out" / "m.scene.xml"
    out.parent.mkdir()
    write_scene(s, out)
    assert (tmp_path / "out" / "m.scene.assets" / "a000_tip.stl").is_file()
    m = mujoco.MjModel.from_xml_path(str(out))                       # loads by itself, meshes found next to it
    assert m.nmesh == 1
    root = ET.parse(out).getroot()
    assert root.find("compiler").get("meshdir") == "m.scene.assets"
    assert str(tmp_path) not in out.read_text()


def test_pendulum_and_cartpole_scenes(tmp_path):
    for name in ("pendulum.json", "cartpole.json"):
        r = validate(recipe_template(name))
        m = compile_(build_scene(r, tmp_path))
        assert m.nu == (1 if r["task"] == "cartpole" else 0)
    r = validate({**recipe_template("pendulum.json"), "pendulum": {"links": 2}})
    m = compile_(build_scene(r, tmp_path))
    assert m.njnt == 2
