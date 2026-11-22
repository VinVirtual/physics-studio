"""Writing outputs without following symbolic links: a link planted at an output path (or an output
folder that is a link) is refused, and files are written to a temporary name and renamed into place,
so a run never writes through a link to somewhere else."""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

from . import RecipeError


def safe_dir(path: Path) -> Path:
    if path.is_symlink():
        raise RecipeError(f"output folder {path} is a symbolic link; refusing to write through it")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise RecipeError(f"{path} is not a folder")
    return path


def _check_target(path: Path) -> None:
    if path.parent.is_symlink():
        raise RecipeError(f"output folder {path.parent} is a symbolic link; refusing to write through it")
    if path.is_symlink():
        raise RecipeError(f"output {path} is a symbolic link; refusing to write through it")
    if path.exists() and not path.is_file():
        raise RecipeError(f"output {path} exists and is not a file")


def temp_beside(path: Path, suffix: str = "") -> Path:
    """A new empty file next to path (same folder, so os.replace is atomic)."""
    _check_target(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    os.close(fd)
    return Path(tmp)


def commit(tmp: Path, path: Path) -> None:
    _check_target(path)
    os.replace(tmp, path)


def write_atomic(path: Path, data: bytes | str) -> None:
    tmp = temp_beside(path)
    try:
        tmp.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
        commit(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_image(img, path: Path, **kw) -> None:
    buf = io.BytesIO()
    img.save(buf, format="JPEG" if path.suffix.lower() in (".jpg", ".jpeg") else "PNG", **kw)
    write_atomic(path, buf.getvalue())


def remove(path: Path) -> None:
    """Remove a stale output (a link is removed, never followed)."""
    if path.is_symlink() or path.is_file():
        path.unlink()
