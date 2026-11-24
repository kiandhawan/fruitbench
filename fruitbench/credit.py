"""Which Moshi frames get which turn's advantage (GRPO credit assignment), and audio talk-over.

Game time -> aligned training frame: Moshi's text token for aligned frame t is heard at game time
(t + 2) * 0.08 s (see fruitbench/game.py), so time tau maps to frame ceil(tau / 0.08 - 2).

credit_windows (full run; fixes the smoke 1-4 windows): the window of reply k is
[start of user line k, start of user line k+1) - the same window the scorer uses to assign
utterances to replies, so a fruit Moshi starts saying before the user finishes line k is trained with
reply k's (overlap-capped) reward. If an utterance of reply k-1 is still going at line k's start, its
frames up to its last word + 0.5 s stay in window k-1 (and window k starts there).
legacy_windows: the smoke 1-4 rule [end of user line k, end of user line k+1) - an early "Mango."
said over the end of line k fell into window k-1 and got the previous turn's advantage.
"""

from __future__ import annotations

import math

FRAME = 0.08
CARRY = 0.5  # same gap that ends an utterance in the scorer


def frame_of(t: float) -> int:
    return max(0, math.ceil(t / FRAME - 2 - 1e-6))


def credit_windows(summary: dict, score: dict, T: int) -> list[tuple[int, int, int]]:
    """(scored turn index j, first frame, end frame) per scored reply, new rule."""
    ut = summary["user_turns"]
    reps = score["replies"]
    n = len(ut)
    lo = [ut[k]["start"] for k in range(n)]
    hi = [ut[k + 1]["start"] if k + 1 < n else summary["end_time"] + 1.0 for k in range(n)]
    for k in range(1, n):
        prev = reps[k - 1].get("word_t") or []
        if prev and prev[-1] >= lo[k]:  # reply k-1 still talking when line k starts
            cut = prev[-1] + CARRY
            hi[k - 1] = max(hi[k - 1], cut)
            lo[k] = max(lo[k], cut)
    out, j = [], 0
    for k, r in enumerate(reps):
        if not r["scored"]:
            continue
        a, b = frame_of(lo[k]), min(T, frame_of(hi[k]))
        out.append((j, a, max(a, b)))
        j += 1
    return out


def legacy_windows(summary: dict, score: dict, T: int) -> list[tuple[int, int, int]]:
    """Smoke 1-4 rule: [speech_end of line k, speech_end of line k+1)."""
    ut = summary["user_turns"]
    out, j = [], 0
    for r in score["replies"]:
        if not r["scored"]:
            continue
        k = r["turn"]
        a = ut[k]["speech_end"]
        b = ut[k + 1]["speech_end"] if k + 1 < len(ut) else summary["end_time"] + 1.0
        out.append((j, frame_of(a), min(T, frame_of(b))))
        j += 1
    return out


def talkover_flags(summary: dict, score: dict, min_frames: int = 2) -> list[bool]:
    """Audio-based talk-over per scored reply k: >= min_frames of Moshi speech (level above the game's
    speech_dbfs, -45 dBFS) inside user line k's clip [start, speech_end)."""
    lv = summary.get("moshi_dbfs") or []
    thr = summary.get("speech_dbfs", -45.0)
    ut = summary["user_turns"]
    out = []
    for r in score["replies"]:
        if not r["scored"]:
            continue
        t = ut[r["turn"]]
        a, b = math.ceil(t["start"] / FRAME - 1e-6), math.floor(t["speech_end"] / FRAME + 1e-6)
        out.append(sum(1 for j in range(a, min(b, len(lv))) if lv[j] > thr) >= min_frames)
    return out
