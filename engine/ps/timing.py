"""Where each spoken line sits on the video timeline, word timings, caption chunks and WebVTT.

Video time = simulation time / speed. A line starts at its anchor (the event it describes) or right
after the previous line, whichever is later; the video is as long as the run plus a short hold, or
longer if the voice needs it (the last frame is held).
"""
from __future__ import annotations

import math
import re

GAP = 0.3            # silence between lines (s)
LEAD = 0.35          # nothing is spoken before this (s)
TAIL = 0.8           # after the last word (s)
LATE_WARN = 1.5      # a line starting this much after its anchor is reported
WORDS_PER_S = 2.6    # used when there is no voice


def syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1 if re.search(r"\d", word) else 1
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def estimate_words(text: str, duration: float) -> list[dict]:
    """Word start times inside a phrase of known duration, weighted by syllables."""
    words = text.split()
    if not words:
        return []
    weights = [syllables(w) + (2 if re.search(r"\d", w) else 0) for w in words]
    total = sum(weights)
    out, acc = [], 0
    for w, k in zip(words, weights):
        out.append({"w": w, "t": round(acc / total * duration * 0.97, 3)})
        acc += k
    return out


def silent_timing(text: str, speed: float = 1.0) -> dict:
    words = text.split()
    dur = max(0.8, len(words) / (WORDS_PER_S * speed))
    return {"duration": round(dur, 3), "words": estimate_words(text, dur)}


def schedule(lines: list[dict], timings: dict[str, dict], sim_duration: float, speed: float,
             hold_end: float) -> dict:
    """lines: [{id, anchor_s, text}], timings: id -> {duration, words[{w, t}]} (t relative to the line).
    Returns the placed lines (with absolute word times), the video duration and the run's end on video."""
    run_end = sim_duration / speed
    placed, cursor = [], LEAD
    for ln in lines:
        tm = timings[ln["id"]]
        want = ln["anchor_s"] / speed
        start = max(want, cursor)
        end = start + tm["duration"]
        placed.append({**ln, "start": round(start, 3), "end": round(end, 3), "late_s": round(start - want, 3),
                       "words": [{"w": w["w"], "t": round(start + w["t"], 3)} for w in tm["words"]]})
        cursor = end + GAP
    last = placed[-1]["end"] + TAIL if placed else 0.0
    duration = max(run_end + hold_end, last)
    return {"lines": placed, "duration": round(duration, 3), "run_end": round(run_end, 3),
            "late": [p["id"] for p in placed if p["late_s"] > LATE_WARN]}


AUTO_SPEEDS = (1.0, 0.8, 0.67, 0.5, 0.4, 0.33, 0.25)


def choose_speed(lines: list[dict], timings: dict[str, dict], sim_duration: float, hold_end: float,
                 requested) -> float:
    """A fixed speed is used as given. "auto" takes the fastest speed from AUTO_SPEEDS at which no line
    starts more than LATE_WARN seconds after the moment it describes (the slowest if none fits)."""
    if requested != "auto":
        return float(requested)
    for s in AUTO_SPEEDS:
        if not schedule(lines, timings, sim_duration, s, hold_end)["late"]:
            return s
    return AUTO_SPEEDS[-1]


def chunks(placed: list[dict], max_words: int = 7) -> list[dict]:
    """Caption cues: each line split at punctuation, then into pieces of at most max_words words,
    shown from the first word until the next cue (or the end of the line plus a little)."""
    cues = []
    for ln in placed:
        words = ln["words"]
        groups, cur = [], []
        for i, w in enumerate(words):
            cur.append(w)
            if re.search(r"[,.;:?!]$", w["w"]) and len(cur) >= 3 or len(cur) >= max_words:
                groups.append(cur)
                cur = []
        if cur:
            if groups and len(cur) <= 2 and len(groups[-1]) + len(cur) <= max_words + 2:
                groups[-1] = groups[-1] + cur
            else:
                groups.append(cur)
        for gi, g in enumerate(groups):
            start = g[0]["t"]
            end = groups[gi + 1][0]["t"] if gi + 1 < len(groups) else ln["end"] + 0.25
            cues.append({"start": round(start, 3), "end": round(end, 3), "words": g})
    for a, b in zip(cues, cues[1:]):          # never overlap
        if a["end"] > b["start"]:
            a["end"] = b["start"]
    return cues


def cue_at(cues: list[dict], t: float) -> dict | None:
    for c in cues:
        if c["start"] <= t < c["end"]:
            return c
    return None


def vtt_time(t: float) -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def vtt(cues: list[dict]) -> str:
    out = ["WEBVTT", ""]
    for i, c in enumerate(cues, 1):
        text = " ".join(w["w"] for w in c["words"]).replace("-->", "->")
        out += [str(i), f"{vtt_time(c['start'])} --> {vtt_time(c['end'])}", text, ""]
    return "\n".join(out)


def video_to_sim(t_video: float, speed: float, sim_duration: float) -> float:
    return min(t_video * speed, sim_duration)


def frame_count(duration: float, fps: int) -> int:
    return int(math.ceil(duration * fps - 1e-9))
