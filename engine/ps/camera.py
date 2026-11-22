"""Camera placement and projection. Pure numpy, no MuJoCo, so it is easy to test.

The scene camera is a fixed MJCF <camera> built from lookat / distance / azimuth / elevation (the same
convention as MuJoCo's free camera: azimuth in degrees around +z, elevation negative when looking
down). Overlays are drawn by projecting world points with the very same pose and field of view.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    pos: tuple[float, float, float]
    xaxis: tuple[float, float, float]     # camera right
    yaxis: tuple[float, float, float]     # camera up
    fovy: float                           # vertical field of view, degrees
    width: int
    height: int

    @property
    def zaxis(self) -> np.ndarray:        # MuJoCo cameras look along -z
        return np.cross(np.array(self.xaxis), np.array(self.yaxis))

    @property
    def focal(self) -> float:
        return (self.height / 2) / math.tan(math.radians(self.fovy) / 2)


def forward(azimuth: float, elevation: float) -> np.ndarray:
    a, e = math.radians(azimuth), math.radians(elevation)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def make_camera(lookat, distance: float, azimuth: float, elevation: float, fovy: float,
                width: int, height: int) -> Camera:
    if not -89.5 <= elevation <= 89.5:
        raise ValueError("elevation must be within -89.5..89.5 degrees")
    f = forward(azimuth, elevation)
    pos = np.asarray(lookat, float) - distance * f
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return Camera(tuple(pos.round(9)), tuple(x.round(9)), tuple(y.round(9)), float(fovy), int(width), int(height))


def project(cam: Camera, points) -> np.ndarray:
    """World points (N,3) -> pixels (N,3): u, v, depth. Depth <= 0 means behind the camera."""
    p = np.atleast_2d(np.asarray(points, float)) - np.asarray(cam.pos)
    x, y, z = np.asarray(cam.xaxis), np.asarray(cam.yaxis), cam.zaxis
    xc, yc, zc = p @ x, p @ y, p @ z
    depth = -zc
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.width / 2 + cam.focal * xc / depth
        v = cam.height / 2 - cam.focal * yc / depth
    return np.stack([u, v, depth], axis=1)


def fit_distance(distance: float, width: int, height: int, ref_aspect: float = 16 / 9) -> float:
    """A narrower frame needs the camera further back to keep the same width of scene in view."""
    aspect = width / height
    if aspect >= ref_aspect:
        return distance
    return distance * (ref_aspect / aspect) ** 0.55


def xyaxes_attr(cam: Camera) -> str:
    return " ".join(f"{v:.6f}" for v in (*cam.xaxis, *cam.yaxis))
