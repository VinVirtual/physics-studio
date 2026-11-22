"""physics-studio voice step. Runs with the Python that has kokoro-onnx (default ~/kokoro/.venv/bin/python).

usage: python voice.py <job.json> <out_dir>
  job.json: {"voice": "af_heart", "speed": 1.0, "lang": "en-us", "lines": [{"id": "l00", "text": "..."}]}
writes <out_dir>/<id>.wav (24 kHz mono, 16-bit) per line and <out_dir>/timings.json:
  {"<id>": {"duration": s, "words": [{"w": word, "t": seconds from line start}]}}

Each line is spoken phrase by phrase (split at punctuation), so phrase boundaries are exact; word
times inside a phrase are estimated from syllable counts. Spoken phrases are cached on disk.
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

RATE = 24000


def die(msg):
    sys.exit("voice: " + msg)


def syllables(word):
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def phrases(words):
    out, cur = [], []
    for w in words:
        cur.append(w)
        if re.search(r"[,.;:?!]$", w):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def spoken(text):
    """Read numbers and units the way a person would say them."""
    text = re.sub(r"(\d+)\.(\d+)", lambda m: m.group(1) + " point " + " ".join(m.group(2)), text)
    text = text.replace("LQR", "L Q R").replace("MuJoCo", "Mu Jo Co")
    return text


def main():
    if len(sys.argv) != 3:
        die("usage: python voice.py <job.json> <out_dir>")
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    voice = job.get("voice", "af_heart")
    if not re.fullmatch(r"[a-z]{2}_[a-z]{2,20}", voice):
        die(f"bad voice name {voice!r}")
    speed = float(job.get("speed", 1.0))
    lang = job.get("lang", "en-us")
    try:
        import numpy as np
        import soundfile as sf
        from kokoro_onnx import Kokoro
    except ImportError as e:
        die(f"{e}. Install into this Python: pip install kokoro-onnx soundfile numpy")
    home = Path(os.environ.get("KOKORO_HOME") or Path.home() / "kokoro")
    model = Path(os.environ.get("KOKORO_MODEL") or home / "kokoro-v1.0.onnx")
    voices = Path(os.environ.get("KOKORO_VOICES") or home / "voices-v1.0.bin")
    for f in (model, voices):
        if not f.is_file():
            die(f"Kokoro model file not found: {f} (see README.md, or set KOKORO_HOME)")
    cache = out / ".cache"
    cache.mkdir(exist_ok=True)
    engine = None

    def speak(text):
        nonlocal engine
        key = hashlib.sha1(json.dumps([text, voice, speed, lang, model.name]).encode()).hexdigest()
        hit = cache / f"{key}.npy"
        if hit.is_file():
            return np.load(hit)
        if engine is None:
            engine = Kokoro(str(model), str(voices))
        audio, _ = engine.create(text, voice=voice, speed=speed, lang=lang)
        y = np.asarray(audio, dtype=np.float32)
        np.save(hit, y)
        return y

    def trim(y, thr=0.008, pad=0.03):
        env = np.convolve(np.abs(y), np.ones(240) / 240, "same")
        idx = np.where(env > thr)[0]
        if len(idx) == 0:
            return y
        return y[max(0, idx[0] - int(pad * RATE)):min(len(y), idx[-1] + int(pad * RATE))]

    timings = {}
    for line in job["lines"]:
        lid = line["id"]
        if not re.fullmatch(r"l\d{2}", lid):
            die(f"bad line id {lid!r}")
        words = line["text"].split()
        parts, wt, pos = [], [], 0.0
        ph = phrases(words)
        for i, p in enumerate(ph):
            y = trim(speak(spoken(" ".join(p))))
            dur = len(y) / RATE
            tot = sum(syllables(w) + (2 if re.search(r"\d", w) else 0) for w in p)
            acc = 0
            for w in p:
                wt.append({"w": w, "t": round(pos + acc / tot * dur * 0.96, 3)})
                acc += syllables(w) + (2 if re.search(r"\d", w) else 0)
            parts.append(y)
            pos += dur
            if i < len(ph) - 1:
                gap = 0.24 if re.search(r"[.?!]$", p[-1]) else 0.1
                parts.append(np.zeros(int(gap * RATE), np.float32))
                pos += gap
        y = np.concatenate(parts) if parts else np.zeros(RATE // 2, np.float32)
        sf.write(out / f"{lid}.wav", y, RATE, subtype="PCM_16")
        timings[lid] = {"duration": round(len(y) / RATE, 3), "words": wt}
    (out / "timings.json").write_text(json.dumps(timings, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
