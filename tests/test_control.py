"""Planning and control maths: minimum-jerk profiles, IK frames, the Riccati solver, footprints."""
import math

import numpy as np
import pytest

from ps.ik import down_frame, min_jerk, min_jerk_rate, rotation_error
from ps.tasks import Segment, dare, extent_along, lqr_gain, sample_segment


def test_min_jerk_profile():
    assert min_jerk(0) == 0 and min_jerk(1) == 1 and min_jerk(0.5) == pytest.approx(0.5)
    assert min_jerk(-1) == 0 and min_jerk(2) == 1
    assert min_jerk_rate(0) == 0 and min_jerk_rate(1) == 0 and min_jerk_rate(0.5) == pytest.approx(1.875)
    s = np.linspace(0, 1, 101)
    num = np.gradient([min_jerk(x) for x in s], s)
    assert np.allclose(num[5:-5], [min_jerk_rate(x) for x in s][5:-5], atol=1e-2)


def test_sample_segment_ends_and_velocity():
    seg = Segment(1.0, 3.0, [np.array([0.0, 0.0]), np.array([1.0, -2.0])], np.array([0.04]), np.array([0.01]), "move")
    q, qd, g = sample_segment(seg, 1.0)
    assert np.allclose(q, [0, 0]) and np.allclose(qd, 0) and g[0] == pytest.approx(0.04)
    q, qd, g = sample_segment(seg, 3.0)
    assert np.allclose(q, [1, -2]) and np.allclose(qd, 0) and g[0] == pytest.approx(0.01)
    q, qd, _ = sample_segment(seg, 2.0)
    assert np.allclose(q, [0.5, -1]) and np.allclose(qd, np.array([1, -2]) * 1.875 / 2)


def test_multi_knot_segment_passes_through_knots():
    knots = [np.array([0.0]), np.array([1.0]), np.array([3.0])]
    seg = Segment(0, 1, knots, np.zeros(0), np.zeros(0), "push")
    s_half = 0.5   # min_jerk(0.5) = 0.5 -> exactly the middle knot
    q, _, _ = sample_segment(seg, s_half)
    assert q[0] == pytest.approx(1.0)


def test_down_frame():
    R = down_frame([0, 1, 0])
    assert np.allclose(R[:, 2], [0, 0, -1]) and np.allclose(R[:, 0], [0, 1, 0])
    assert np.allclose(R.T @ R, np.eye(3)) and np.linalg.det(R) == pytest.approx(1)
    with pytest.raises(ValueError):
        down_frame([0, 0, 1])


def test_rotation_error_is_small_angle_axis():
    a = 0.01
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    e = rotation_error(np.eye(3), Rz)
    assert np.allclose(e, [0, 0, a], atol=1e-6)
    assert np.allclose(rotation_error(Rz, Rz), 0)


def test_dare_matches_scalar_solution():
    # x+ = a x + b u, cost q x^2 + r u^2: P = q + a^2 P - (a b P)^2 / (r + b^2 P)
    a, b, q, r = 1.1, 0.5, 2.0, 1.0
    P = dare(np.array([[a]]), np.array([[b]]), np.array([[q]]), np.array([[r]]))[0, 0]
    assert P == pytest.approx(q + a * a * P - (a * b * P) ** 2 / (r + b * b * P), rel=1e-8)
    K = lqr_gain(np.array([[a]]), np.array([[b]]), np.array([[q]]), np.array([[r]]))
    assert abs(a - b * K[0, 0]) < 1        # the closed loop is stable


def test_lqr_stabilises_a_double_integrator():
    dt = 0.01
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0], [dt]])
    K = lqr_gain(A, B, np.eye(2), np.array([[0.1]]))
    assert np.all(np.abs(np.linalg.eigvals(A - B @ K)) < 1)


def test_extent_along():
    box = {"shape": "box", "size": [0.02, 0.04, 0.01], "yaw_deg": 0}
    assert extent_along(box, [1, 0]) == pytest.approx(0.02) and extent_along(box, [0, 1]) == pytest.approx(0.04)
    turned = {**box, "yaw_deg": 90}
    assert extent_along(turned, [1, 0]) == pytest.approx(0.04)
    diag = extent_along(box, [math.sqrt(0.5), math.sqrt(0.5)])
    assert diag == pytest.approx((0.02 + 0.04) * math.sqrt(0.5))
    assert extent_along({"shape": "sphere", "size": [0.03], "yaw_deg": 0}, [0, 1]) == 0.03
