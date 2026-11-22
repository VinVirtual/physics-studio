import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
os.environ.setdefault("MUJOCO_GL", "egl")

TEMPLATES = ROOT / "templates"


def recipe_template(name: str) -> dict:
    return json.loads((TEMPLATES / "recipes" / name).read_text())


@pytest.fixture
def tmp_project(tmp_path):
    """A folder with the Tern URDF in robot/ and a writer for recipe.json."""
    (tmp_path / "robot").mkdir()
    shutil.copy(TEMPLATES / "robots" / "tern.urdf", tmp_path / "robot" / "tern.urdf")

    def write(recipe: dict) -> Path:
        p = tmp_path / "recipe.json"
        p.write_text(json.dumps(recipe))
        return p
    write.dir = tmp_path
    return write


def _gl_works() -> bool:
    code = ("import mujoco;m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\".1\"/></worldbody></mujoco>');"
            "r=mujoco.Renderer(m,16,16);d=mujoco.MjData(m);mujoco.mj_forward(m,d);r.update_scene(d);r.render()")
    try:
        return subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=60).returncode == 0
    except Exception:
        return False


GL = _gl_works()
needs_gl = pytest.mark.skipif(not GL, reason="no offscreen OpenGL (EGL or OSMesa) on this machine")


def deep(d):
    return copy.deepcopy(d)
