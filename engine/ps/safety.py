"""What a robot file is allowed to reach. Robot files (URDF, MJCF) from someone else are untrusted:

- they are parsed by Python's XML parser with DOCTYPE and ENTITY declarations refused, never handed
  to MuJoCo by path, so MuJoCo cannot open any file on its own;
- every mesh, texture or height field they name is looked up inside the robot's folder only (after
  following symbolic links), must have an expected extension and size, and is passed to MuJoCo as
  bytes under a new flat name;
- MJCF elements that load code or other files (<include>, <plugin>, <extension>, <model>, skins)
  are refused.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from . import RecipeError

MAX_XML_BYTES = 5 * 1024 * 1024
MAX_ASSET_BYTES = 40 * 1024 * 1024
MAX_ASSETS_TOTAL = 120 * 1024 * 1024
MAX_ASSETS = 200
MESH_EXT = (".stl", ".obj", ".msh")
IMAGE_EXT = (".png",)
REFUSED_MJCF = ("include", "plugin", "extension", "model", "skin", "instance", "flexcomp", "replicate",
                "composite", "attach", "flex")
# the only MJCF elements that may reach MuJoCo (inside <worldbody>, <asset>, <default>, <contact>,
# <equality>, <tendon>); everything else in those sections is refused
ALLOWED_MJCF = {
    "worldbody", "asset", "default", "contact", "equality", "tendon", "body", "frame", "inertial", "joint",
    "freejoint", "geom", "site", "camera", "light", "texture", "material", "mesh", "hfield", "layer", "pair",
    "exclude", "connect", "weld", "distance", "spatial", "fixed", "pulley", "position", "velocity",
    "motor", "general", "intvelocity", "damper", "cylinder", "muscle", "adhesion",
}
LIMITS = {("texture", "width"): 4096, ("texture", "height"): 4096, ("hfield", "nrow"): 1024, ("hfield", "ncol"): 1024}
MAX_BODIES = 400
MAX_ELEMENTS = 20000
_DECL = re.compile(r"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)
_ENCODING = re.compile(r"^\s*<\?xml[^>]*encoding\s*=\s*[\"']([^\"']+)", re.IGNORECASE)


def inside(base: Path, rel: str, what: str = "file") -> Path:
    """Resolve rel under base. Refuse absolute paths, URL schemes, '..' and links that lead outside."""
    if not isinstance(rel, str) or not rel or "\x00" in rel:
        raise RecipeError(f"{what}: empty or invalid path")
    if rel.startswith(("/", "\\", "~")) or re.match(r"^[A-Za-z]:", rel) or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", rel):
        raise RecipeError(f"{what} {rel!r}: absolute paths and URLs are refused; use a path inside {base.name}/")
    if any(part == ".." for part in re.split(r"[\\/]", rel)):
        raise RecipeError(f"{what} {rel!r}: '..' is refused; keep files inside {base.name}/")
    root = base.resolve()
    p = (root / rel).resolve()
    if p != root and root not in p.parents:
        raise RecipeError(f"{what} {rel!r} leads outside {base.name}/ (through a symbolic link?)")
    return p


def read_xml(path: Path) -> ET.Element:
    if not path.is_file():
        raise RecipeError(f"robot file not found: {path.name}")
    if path.stat().st_size > MAX_XML_BYTES:
        raise RecipeError(f"robot file {path.name} is larger than {MAX_XML_BYTES // (1024 * 1024)} MB")
    return parse_xml_bytes(path.read_bytes(), path.name)


def parse_xml_bytes(data: bytes, name: str = "robot file") -> ET.Element:
    if len(data) > MAX_XML_BYTES:
        raise RecipeError(f"{name} is larger than {MAX_XML_BYTES // (1024 * 1024)} MB")
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise RecipeError(f"{name} must be UTF-8 text") from None
    if "\x00" in text:
        raise RecipeError(f"{name} must be UTF-8 text (it contains NUL bytes)")
    enc = _ENCODING.match(text)
    if enc and enc.group(1).lower().replace("_", "-") not in ("utf-8", "utf8", "us-ascii", "ascii"):
        raise RecipeError(f"{name}: only UTF-8 files are read (it declares {enc.group(1)})")
    if _DECL.search(text):
        raise RecipeError(f"{name}: DOCTYPE and ENTITY declarations are refused")
    from xml.parsers import expat

    def refuse(*_a) -> None:
        raise RecipeError(f"{name}: DOCTYPE and ENTITY declarations are refused")
    scan = expat.ParserCreate()                   # a second line of defence after the text check
    scan.StartDoctypeDeclHandler = refuse
    scan.EntityDeclHandler = refuse
    try:
        scan.Parse(data, True)
        return ET.fromstring(data)
    except (ET.ParseError, expat.ExpatError) as e:
        raise RecipeError(f"{name} is not valid XML: {e}") from None


class AssetStore:
    """Files a robot may use, read once, renamed to flat safe names, with size and count caps."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.files: dict[str, bytes] = {}
        self._by_path: dict[Path, str] = {}
        self.total = 0

    def add(self, rel: str, kinds: tuple[str, ...], what: str) -> str:
        p = inside(self.base, rel, what)
        if p in self._by_path:
            return self._by_path[p]
        if p.suffix.lower() not in kinds:
            raise RecipeError(f"{what} {rel!r}: only {', '.join(kinds)} files are read")
        if not p.is_file():
            raise RecipeError(f"{what} {rel!r} not found inside {self.base.name}/")
        size = p.stat().st_size
        if size > MAX_ASSET_BYTES:
            raise RecipeError(f"{what} {rel!r} is larger than {MAX_ASSET_BYTES // (1024 * 1024)} MB")
        if len(self.files) >= MAX_ASSETS or self.total + size > MAX_ASSETS_TOTAL:
            raise RecipeError(f"the robot uses too many or too large files (limit {MAX_ASSETS} files, "
                              f"{MAX_ASSETS_TOTAL // (1024 * 1024)} MB)")
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", p.stem)[:40] or "asset"
        name = f"a{len(self.files):03d}_{stem}{p.suffix.lower()}"
        self.files[name] = p.read_bytes()
        self.total += size
        self._by_path[p] = name
        return name


def check_mjcf_elements(root: ET.Element) -> None:
    """Refuse elements that load files or code or multiply the model, and cap sizes."""
    n = 0
    for el in root.iter():
        n += 1
        tag = el.tag if isinstance(el.tag, str) else ""
        if tag in REFUSED_MJCF:
            raise RecipeError(f"MJCF <{tag}> is refused (it can load other files or code, or multiply the model); "
                              "inline what you need")
        for attr in el.attrib:
            if attr.startswith("file") and not (attr == "file" and tag in ("mesh", "texture", "hfield")):
                raise RecipeError(f"<{tag} {attr}=...> is refused; only meshes, textures and height fields may "
                                  "name one file each")
        for (t, a), cap in LIMITS.items():
            if tag == t and el.get(a) is not None:
                try:
                    v = int(float(el.get(a)))
                except ValueError:
                    raise RecipeError(f"<{t} {a}=...> is not a number") from None
                if v > cap:
                    raise RecipeError(f"<{t} {a}={v}> is larger than {cap}")
    if n > MAX_ELEMENTS:
        raise RecipeError(f"the robot file has more than {MAX_ELEMENTS} elements")
    if sum(1 for _ in root.iter("body")) > MAX_BODIES:
        raise RecipeError(f"the robot has more than {MAX_BODIES} bodies")
    for section in ("worldbody", "asset", "default", "contact", "equality", "tendon"):
        for top in root.findall(section):
            for el in top.iter():
                if el.tag not in ALLOWED_MJCF:
                    raise RecipeError(f"MJCF <{el.tag}> inside <{section}> is not supported")
