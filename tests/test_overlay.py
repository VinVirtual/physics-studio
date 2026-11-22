"""Overlay maths: camera pose and projection, force arrows, effort bars, text layout, drawing."""
import math

import numpy as np
import pytest
from conftest import needs_gl, recipe_template

from ps.camera import fit_distance, forward, make_camera, project
from ps.overlay import (FORCE, Painter, arrow_polygon, bar_fraction, draw_caption, draw_effort, draw_forces,
                        draw_hud, fmt_force, font, force_tip, layout, visible, wrap_words)

CAM = make_camera([0.2, 0.0, 0.5], 1.2, 60, -25, 40, 1280, 720)


def test_camera_axes_are_orthonormal_and_right_handed():
    x, y, z = np.array(CAM.xaxis), np.array(CAM.yaxis), CAM.zaxis
    for a in (x, y, z):
        assert np.linalg.norm(a) == pytest.approx(1)
    assert x @ y == pytest.approx(0, abs=1e-9) and np.cross(x, y) @ z == pytest.approx(1)
    assert x[2] == pytest.approx(0, abs=1e-9)          # no roll: the horizon stays level
    assert -z @ forward(60, -25) == pytest.approx(1)   # the camera looks along its forward vector


def test_lookat_projects_to_centre():
    u, v, dep = project(CAM, [0.2, 0.0, 0.5])[0]
    assert (u, v) == pytest.approx((640, 360)) and dep == pytest.approx(1.2)


def test_right_and_up_on_screen():
    c = np.array([0.2, 0.0, 0.5])
    right = project(CAM, c + 0.1 * np.array(CAM.xaxis))[0]
    up = project(CAM, c + 0.1 * np.array(CAM.yaxis))[0]
    assert right[0] > 640 and right[1] == pytest.approx(360)
    assert up[1] < 360 and up[0] == pytest.approx(640)


def test_vertical_fov_reaches_the_frame_edge():
    c = np.array([0.2, 0.0, 0.5])
    half = math.tan(math.radians(20)) * 1.2
    top = project(CAM, c + half * np.array(CAM.yaxis))[0]
    assert top[1] == pytest.approx(0, abs=1e-6)


def test_points_behind_the_camera_are_not_visible():
    behind = np.array(CAM.pos) + 0.5 * CAM.zaxis
    uvd = project(CAM, behind)[0]
    assert uvd[2] < 0 and not visible(uvd, 1280, 720)


def test_project_many_points_at_once():
    assert project(CAM, np.zeros((5, 3))).shape == (5, 3)


def test_elevation_limits():
    with pytest.raises(ValueError):
        make_camera([0, 0, 0], 1, 0, -90, 40, 100, 100)


def test_fit_distance_only_moves_back_for_narrow_frames():
    assert fit_distance(1.0, 1280, 720) == 1.0
    assert fit_distance(1.0, 720, 1280) > fit_distance(1.0, 960, 960) > 1.0


def test_force_tip_scales_and_clips():
    tip, clipped = force_tip([0, 0, 0], [0, 0, 10], 0.01, 0.5)
    assert np.allclose(tip, [0, 0, 0.1]) and not clipped
    tip, clipped = force_tip([1, 1, 1], [30, 40, 0], 0.01, 0.2)
    assert clipped and np.linalg.norm(tip - [1, 1, 1]) == pytest.approx(0.2)
    assert np.allclose((tip - [1, 1, 1]) / 0.2, [0.6, 0.8, 0])   # direction is kept
    tip, clipped = force_tip([1, 2, 3], [0, 0, 0], 0.01, 0.2)
    assert np.allclose(tip, [1, 2, 3]) and not clipped


def test_arrow_polygon_shape():
    poly = arrow_polygon(0, 0, 100, 0, 4, 15)
    assert len(poly) == 7 and poly[3] == (100, 0)
    ys = sorted(p[1] for p in poly)
    assert ys[0] == pytest.approx(-ys[-1])               # symmetric about the shaft
    assert max(p[0] for p in poly) == 100 and min(p[0] for p in poly) == 0
    assert arrow_polygon(5, 5, 5, 5, 4, 15) == []
    short = arrow_polygon(0, 0, 10, 0, 4, 15)
    assert short[1][0] == pytest.approx(4)               # the head takes at most 60% of a short arrow


def test_bar_fraction():
    assert bar_fraction(3, 6) == 0.5 and bar_fraction(-9, 6) == -1 and bar_fraction(9, 6) == 1
    assert bar_fraction(1, 0) == 0 and bar_fraction(float("nan"), 5) == 0


def test_fmt_force():
    assert fmt_force(8.04) == "8.0 N" and fmt_force(401.9) == "402 N"


def test_wrap_words_fits_width():
    f = font("sans", 30)
    words = "the quick brown fox jumps over the lazy dog again and again".split()
    lines = wrap_words(words, f, 260)
    assert [i for ln in lines for i in ln] == list(range(len(words)))
    for ln in lines:
        w = sum(f.getlength(words[i]) for i in ln) + f.getlength(" ") * (len(ln) - 1)
        assert w <= 260 or len(ln) == 1


def test_layouts_keep_panels_inside_the_frame():
    for w, h in ((1280, 720), (720, 1280), (960, 960)):
        L = layout(w, h)
        x, y, pw = L.effort
        assert 0 <= x and x + pw <= w and 0 <= y < h and L.caption_y < h


def test_drawing_produces_marks():
    from PIL import Image
    base = Image.new("RGB", (1280, 720), (200, 200, 200))
    p = Painter(1280, 720)
    draw_forces(p, CAM, [{"pos": [0.2, 0.0, 0.5], "f": [0, 0, 5], "n": 5.0, "label": "a<b"}], 0.04, 0.05, 0.12)
    draw_hud(p, 1.234, 617, 0.002, 1.0, "MuJoCo")
    draw_effort(p, [{"name": "shoulder", "value": 3.0, "limit": 6.0}])
    draw_caption(p, [{"w": "Hello", "t": 0.0}, {"w": "world.", "t": 0.5}], 0.2)
    out = np.asarray(p.finish(base))
    assert out.shape == (720, 1280, 3)
    red = (abs(out[..., 0].astype(int) - FORCE[0]) < 30) & (out[..., 1] < 110) & (out[..., 2] < 100)
    ys, xs = np.nonzero(red[:, 500:780])
    assert len(ys) > 50 and ys.min() < 360               # an arrow pointing up from the centre


@needs_gl
def test_projection_matches_mujoco_rendering(tmp_path):
    """A red ball rendered by MuJoCo lands where our projection says it should."""
    import mujoco

    from ps.recipe import validate
    from ps.render import SceneRenderer
    from ps.scene import build_scene
    r = validate({"name": "p", "task": "drop", "table": False,
                  "objects": [{"id": "ball", "shape": "sphere", "size": [0.03], "pos": [0.15, -0.1, 0.0], "color": "#ff0000"}],
                  "camera": {"lookat": [0, 0, 0.1], "distance": 1.0, "azimuth": 110, "elevation": -30}})
    s = build_scene(r, tmp_path)
    m = mujoco.MjModel.from_xml_string(s.xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rend = SceneRenderer(m, s)
    img = rend.render(d.qpos).astype(int)
    rend.close()
    mask = (img[..., 0] > 150) & (img[..., 1] < 90) & (img[..., 2] < 90)
    ys, xs = np.nonzero(mask)
    u, v, _ = project(s.camera, s.object_start["ball"])[0]
    assert len(xs) > 30 and abs(xs.mean() - u) < 3 and abs(ys.mean() - v) < 3
