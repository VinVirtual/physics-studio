"""Turn a robot file (URDF or MJCF) into a sanitised MJCF fragment that scene.py places in a scene.

URDF is converted by MuJoCo's own URDF importer (fed from a string, with meshes passed as bytes),
then saved as MJCF and read back. Both paths end in the same RobotPart.
"""
from __future__ import annotations

import copy
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from . import RecipeError
from .safety import IMAGE_EXT, MESH_EXT, AssetStore, check_mjcf_elements, inside, read_xml

COMPILER_KEEP = ("angle", "eulerseq", "autolimits", "inertiafromgeom", "boundmass", "boundinertia",
                 "balanceinertia", "settotalmass", "inertiagrouprange", "discardvisual", "fitaabb")
SECTIONS_KEEP = ("contact", "equality", "tendon")


@dataclass
class RobotPart:
    name: str
    bodies: list[ET.Element]                    # children of the robot's <worldbody>
    assets: list[ET.Element]                    # children of <asset>, file names rewritten
    defaults: list[ET.Element]                  # children of the top <default>
    sections: dict[str, list[ET.Element]]       # contact / equality / tendon children
    compiler: dict[str, str]
    files: dict[str, bytes] = field(default_factory=dict)
    source: str = "mjcf"


def _urdf_mesh_rel(filename: str, urdf_dir: Path, base: Path) -> str:
    """package://pkg/meshes/a.stl -> a path relative to base that exists, tried in a fixed order."""
    f = filename.strip()
    candidates = []
    if f.startswith("package://"):
        rest = f[len("package://"):]
        parts = rest.split("/", 1)
        candidates += [rest, parts[1] if len(parts) > 1 else rest]
    elif f.startswith("file://"):
        raise RecipeError(f"mesh {filename!r}: file:// paths are refused; use a path inside the robot folder")
    else:
        rel_to_urdf = os.path.relpath(urdf_dir / f, base) if not f.startswith(("/", "\\")) else f
        candidates += [rel_to_urdf, f]
    for cnd in candidates:
        try:
            p = inside(base, cnd, "mesh")
        except RecipeError:
            continue
        if p.is_file():
            return cnd
    return candidates[0]


def _load_urdf(path: Path, base: Path) -> tuple[ET.Element, AssetStore]:
    import mujoco  # imported here so recipe validation works without MuJoCo

    root = read_xml(path)
    if root.tag != "robot":
        raise RecipeError(f"{path.name}: a URDF must start with <robot>, found <{root.tag}>")
    for ext in list(root.findall("mujoco")):   # a URDF may carry MuJoCo settings (meshdir...); we set our own
        root.remove(ext)
    if sum(1 for _ in root.iter("link")) > 400:
        raise RecipeError(f"{path.name} has more than 400 links")
    for mat in root.iter("material"):          # image textures are not used; never let MuJoCo look for them
        for tex in list(mat.findall("texture")):
            mat.remove(tex)
    store = AssetStore(base)
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn is None:
            continue
        rel = _urdf_mesh_rel(fn, path.parent, base)
        mesh.set("filename", store.add(rel, MESH_EXT, "mesh"))
    ext = ET.SubElement(root, "mujoco")
    ET.SubElement(ext, "compiler", {"strippath": "false", "fusestatic": "false", "discardvisual": "true"})
    text = ET.tostring(root, encoding="unicode")
    try:
        model = mujoco.MjModel.from_xml_string(text, assets=store.files)
    except ValueError as e:
        raise RecipeError(f"MuJoCo could not read {path.name}: {str(e).strip()[:400]}") from None
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "robot.xml"
        mujoco.mj_saveLastXML(str(out), model)
        mjcf = ET.fromstring(out.read_bytes())
    return mjcf, store


def _rewrite_mjcf_files(root: ET.Element, base: Path, store: AssetStore, xml_dir: Path) -> None:
    comp = root.find("compiler")
    dirs = {"mesh": "", "texture": "", "hfield": ""}
    if comp is not None:
        asset_dir = comp.get("assetdir", "")
        dirs["mesh"] = comp.get("meshdir", asset_dir)
        dirs["texture"] = comp.get("texturedir", asset_dir)
        dirs["hfield"] = asset_dir
    for tag, kinds in (("mesh", MESH_EXT), ("texture", IMAGE_EXT), ("hfield", IMAGE_EXT)):
        for el in root.iter(tag):
            for attr in el.attrib:     # cube-map faces and any other file-like attribute: refused
                if attr.startswith("file") and attr != "file":
                    raise RecipeError(f"<{tag} {attr}=...> is not supported; use one file per {tag}")
            fn = el.get("file")
            if fn is None:
                continue
            if el.get("name") is None:  # MuJoCo names an unnamed asset after its file; keep that name
                el.set("name", Path(fn.replace("\\", "/")).stem)
            rel = os.path.relpath(xml_dir / dirs[tag] / fn, base) if not fn.startswith(("/", "\\")) else fn
            el.set("file", store.add(rel, kinds, tag))


def load_robot(recipe_dir: Path, cfg: dict) -> RobotPart:
    """cfg is the validated recipe["robot"]. Files are read only from inside the recipe folder."""
    path = inside(recipe_dir, cfg["file"], "robot file")
    base = recipe_dir.resolve()      # meshes may sit anywhere inside the recipe's folder (urdf/ + meshes/ works)
    if path.suffix.lower() == ".urdf":
        mjcf, store = _load_urdf(path, base)
        source = "urdf"
    else:
        mjcf = read_xml(path)
        if mjcf.tag != "mujoco":
            raise RecipeError(f"{path.name}: an MJCF file must start with <mujoco>, found <{mjcf.tag}>")
        check_mjcf_elements(mjcf)
        store = AssetStore(base)
        _rewrite_mjcf_files(mjcf, base, store, path.parent)
        source = "mjcf"
    check_mjcf_elements(mjcf)
    compiler: dict[str, str] = {"angle": "radian" if source == "urdf" else "degree"}
    for comp in mjcf.findall("compiler"):
        for k, v in comp.attrib.items():
            if k in COMPILER_KEEP:
                compiler[k] = v
    wb = mjcf.find("worldbody")
    bodies = [copy.deepcopy(e) for e in (wb if wb is not None else [])]
    bodies = [b for b in bodies if b.tag not in ("light", "camera")]   # the scene brings its own
    assets: list[ET.Element] = []
    for a in mjcf.findall("asset"):
        assets += [copy.deepcopy(e) for e in a]
    defaults: list[ET.Element] = []
    for d in mjcf.findall("default"):
        defaults += [copy.deepcopy(e) for e in d]
    sections = {s: [copy.deepcopy(e) for el in mjcf.findall(s) for e in el] for s in SECTIONS_KEEP}
    name = re.sub(r"[^A-Za-z0-9_-]", "_", mjcf.get("model", path.stem))[:40] or "robot"
    return RobotPart(name=name, bodies=bodies, assets=assets, defaults=defaults, sections=sections,
                     compiler=compiler, files=dict(store.files), source=source)


def joints_of(bodies: list[ET.Element]) -> list[ET.Element]:
    out = []
    for b in bodies:
        out += [j for j in b.iter("joint")]
    return out
