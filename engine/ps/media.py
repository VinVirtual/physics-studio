"""ffmpeg discovery, the voice mix and the MP4 encode."""
from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from . import RecipeError

RATE = 24000


def ffmpeg() -> str:
    env = os.environ.get("FF")
    if env and Path(env).is_file():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - any failure means "not available"
        raise RecipeError("ffmpeg not found: install ffmpeg, or pip install imageio-ffmpeg into this venv, or set FF") from None


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise RecipeError(f"{path.name}: expected 16-bit mono audio")
        if w.getframerate() != RATE:
            raise RecipeError(f"{path.name}: expected {RATE} Hz audio")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return data.astype(np.float32) / 32768.0


def write_wav(path: Path, y: np.ndarray) -> None:
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def mix(placed: list[dict], wav_dir: Path | None, duration: float) -> np.ndarray:
    """Every line's audio at its start time, then levelled: voiced RMS near -19 dBFS, peaks below -1 dBFS."""
    out = np.zeros(int(round(duration * RATE)) + 1, np.float32)
    if wav_dir is None:
        return out
    for ln in placed:
        y = read_wav(wav_dir / f"{ln['id']}.wav")
        a = int(round(ln["start"] * RATE))
        b = min(len(out), a + len(y))
        out[a:b] += y[: b - a]
    voiced = out[np.abs(out) > 0.01]
    if len(voiced):
        rms = float(np.sqrt(np.mean(voiced ** 2)))
        gain = 10 ** (-19 / 20) / max(rms, 1e-6)
        peak = float(np.max(np.abs(out))) * gain
        if peak > 10 ** (-1 / 20):
            gain *= 10 ** (-1 / 20) / peak
        out *= gain
    return out


class Encoder:
    """Raw RGB frames in through a pipe, H.264 + AAC MP4 out."""

    def __init__(self, out: Path, width: int, height: int, fps: int, audio: Path, duration: float, crf: int = 22) -> None:
        cmd = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
               "-i", str(audio), "-map", "0:v", "-map", "1:a",
               "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-t", f"{duration:.3f}",
               "-movflags", "+faststart", str(out)]
        import tempfile
        self.err = tempfile.TemporaryFile()        # a file, not a pipe, so ffmpeg can never block on it
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self.err)

    def _failed(self) -> RecipeError:
        self.p.wait()
        self.err.seek(0)
        return RecipeError(f"ffmpeg failed: {self.err.read().decode(errors='replace').strip()[:600]}")

    def write(self, rgb: bytes) -> None:
        try:
            self.p.stdin.write(rgb)
        except BrokenPipeError:
            raise self._failed() from None

    def close(self) -> None:
        try:
            self.p.stdin.close()
        except BrokenPipeError:
            pass
        if self.p.wait() != 0:
            raise self._failed()


def probe_duration(path: Path) -> float | None:
    r = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    import re
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
