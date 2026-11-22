"""Timing: where lines sit, word times, caption cues, WebVTT and the automatic playback speed."""
import re

import pytest

from ps.timing import (GAP, LEAD, LATE_WARN, choose_speed, chunks, cue_at, estimate_words, frame_count, schedule,
                       silent_timing, syllables, video_to_sim, vtt, vtt_time)

LINES = [{"id": "l00", "anchor_s": 0.0, "text": "A five-joint arm reaches for a cube."},
         {"id": "l01", "anchor_s": 2.7, "text": "Grip at 2.7 seconds."},
         {"id": "l02", "anchor_s": 8.0, "text": "Task passed."}]
TIM = {"l00": {"duration": 3.0, "words": estimate_words(LINES[0]["text"], 3.0)},
       "l01": {"duration": 2.0, "words": estimate_words(LINES[1]["text"], 2.0)},
       "l02": {"duration": 1.0, "words": estimate_words(LINES[2]["text"], 1.0)}}


def test_syllables():
    assert syllables("cat") == 1 and syllables("centimetre") >= 3 and syllables("table") == 2 and syllables("2.7") == 1


def test_estimate_words_monotonic_within_duration():
    w = estimate_words("Measured period one point six zero seconds.", 2.5)
    ts = [x["t"] for x in w]
    assert ts[0] == 0 and ts == sorted(ts) and ts[-1] < 2.5 and len(w) == 7
    assert estimate_words("", 1.0) == []


def test_schedule_respects_anchors_and_never_overlaps():
    s = schedule(LINES, TIM, 8.0, 1.0, 1.0)
    L = s["lines"]
    assert L[0]["start"] == LEAD
    assert L[1]["start"] == pytest.approx(max(2.7, L[0]["end"] + GAP))
    assert L[2]["start"] == 8.0
    for a, b in zip(L, L[1:]):
        assert b["start"] >= a["end"] + GAP - 1e-9
    assert s["duration"] == pytest.approx(max(8.0 + 1.0, L[-1]["end"] + 0.8))
    assert L[1]["words"][0]["t"] == L[1]["start"]


def test_slow_motion_moves_anchors():
    s = schedule(LINES, TIM, 8.0, 0.5, 1.0)
    assert s["run_end"] == 16.0 and s["lines"][1]["start"] == pytest.approx(5.4)


def test_late_lines_are_reported():
    tight = {**TIM, "l00": {"duration": 9.0, "words": estimate_words(LINES[0]["text"], 9.0)}}
    s = schedule(LINES, tight, 8.0, 1.0, 1.0)
    assert "l01" in s["late"] and s["lines"][1]["late_s"] > LATE_WARN


def test_choose_speed():
    assert choose_speed(LINES, TIM, 8.0, 1.0, 0.75) == 0.75          # a fixed speed is kept
    assert choose_speed(LINES, TIM, 8.0, 1.0, "auto") == 1.0
    tight = {**TIM, "l00": {"duration": 5.5, "words": estimate_words(LINES[0]["text"], 5.5)}}
    sp = choose_speed(LINES, tight, 8.0, 1.0, "auto")
    assert sp < 1.0 and not schedule(LINES, tight, 8.0, sp, 1.0)["late"]


def test_chunks_cover_every_word_once_without_overlap():
    s = schedule(LINES, TIM, 8.0, 1.0, 1.0)
    cues = chunks(s["lines"], max_words=4)
    words = [w["w"] for c in cues for w in c["words"]]
    assert words == " ".join(ln["text"] for ln in LINES).split()
    for a, b in zip(cues, cues[1:]):
        assert a["end"] <= b["start"] and a["start"] < a["end"]
    assert all(len(c["words"]) <= 6 for c in cues)
    assert cue_at(cues, cues[0]["start"] + 0.01) is cues[0] and cue_at(cues, 1000) is None


def test_vtt():
    s = schedule(LINES, TIM, 8.0, 1.0, 1.0)
    text = vtt(chunks(s["lines"]))
    assert text.startswith("WEBVTT\n\n1\n")
    stamps = re.findall(r"(\d\d:\d\d:\d\d\.\d\d\d) --> (\d\d:\d\d:\d\d\.\d\d\d)", text)
    assert stamps and all(a < b for a, b in stamps)
    assert vtt_time(3725.5) == "01:02:05.500" and vtt_time(-1) == "00:00:00.000"


def test_vtt_cannot_be_broken_by_text():
    cues = [{"start": 0, "end": 1, "words": [{"w": "a-->b", "t": 0}]}]
    assert "-->b" not in vtt(cues).split("\n", 4)[-1]


def test_silent_timing_and_helpers():
    t = silent_timing("one two three four five six", 1.0)
    assert t["duration"] == pytest.approx(6 / 2.6, abs=0.01) and len(t["words"]) == 6
    assert video_to_sim(10, 0.5, 4.0) == 4.0 and video_to_sim(2, 0.5, 4.0) == 1.0
    assert frame_count(1.0, 30) == 30 and frame_count(1.01, 30) == 31
