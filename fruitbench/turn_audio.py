"""Cut Moshi's audio (left channel) for each scored reply of a saved game, for audio-side checks
(speech quality, padded Whisper recall). Pure numpy + soundfile, usable from any of the venvs.

Segment of reply k: inside the scorer's reply window [start of user line k, start of user line k+1),
from the first to the last frame where Moshi's audio is speech (per-frame level > the game's
speech_dbfs, i.e. -45 dBFS), plus 0.1 s margins. Replies with no speech frame get no segment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .score import score_game

FRAME = 0.08
SR = 24000


def turn_segments(game_json: Path, rules: str = "v3", margin: float = 0.1) -> list[dict]:
    g = json.loads(Path(game_json).read_text())
    sc = score_game(g, rules)
    lv = g.get("moshi_dbfs") or []
    thr = g.get("speech_dbfs", -45.0)
    ut = g["user_turns"]
    out = []
    pos = 0
    for r in sc["replies"]:
        if r["kind"] == "fruit":
            pos += 1
        if not r["scored"]:
            continue
        k = r["turn"]
        lo = ut[k]["start"]
        hi = ut[k + 1]["start"] if k + 1 < len(ut) else g["end_time"]
        js = [j for j in range(int(lo / FRAME), min(len(lv), int(np.ceil(hi / FRAME)))) if lv[j] > thr]
        seg = None
        if js:
            seg = (max(0.0, js[0] * FRAME - margin), min(g["end_time"], (js[-1] + 1) * FRAME + margin))
        out.append({"game": g["game_id"], "turn": k, "pos": pos if r["kind"] == "fruit" else 0, "cls": r["cls"],
                    "text": r["text"], "fruits": r["fruits"], "fruit": r["fruit"], "echo": r["echo"],
                    "user_text": r["user_text"], "user_end": r["user_end"], "seg": seg})
    return out


def load_stereo(wav: Path) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    assert sr == SR, sr
    return x.T  # (2, n): 0 = Moshi, 1 = user


def cut(x: np.ndarray, seg: tuple[float, float]) -> np.ndarray:
    a, b = int(seg[0] * SR), int(seg[1] * SR)
    return x[..., a:b]
