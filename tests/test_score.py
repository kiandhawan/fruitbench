"""Scorer check on the real base-Moshi game from the task description (Whisper times, seconds).
Run: python tests/test_score.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fruitbench.score import score_game


def toks(t, text):
    out = []
    for i, w in enumerate(text.split()):
        out.append({"t": round(t + 0.25 * i, 2), "piece": "▁" + w})
    return out


def ut(kind, text, start, end, fruit=None):
    return {"kind": kind, "text": text, "start": start, "speech_end": end, "fruit": fruit}


game = {
    "game_id": "real_failure", "starter": "user", "end_time": 28.0,
    "user_turns": [ut("instructions", "Let's take turns naming fruits as fast as we can.", 3.0, 5.6),
                   ut("fruit", "Orange.", 8.8, 9.3, "orange"), ut("fruit", "Grape.", 11.2, 11.7, "grape"),
                   ut("fruit", "Banana.", 13.7, 14.2, "banana"),
                   ut("fruit", "Cantaloupe.", 20.9, 21.4, "cantaloupe"), ut("fruit", "Guava.", 22.9, 23.4, "guava")],
    "moshi_tokens": toks(1.1, "Hey, what's up?") + toks(6.1, "Sure, go ahead.") + toks(9.7, "Okay, I'm ready.")
                    + toks(14.7, "Apple.") + toks(18.8, "Okay, I'm ready.") + toks(21.6, "Okay, I'm ready.")
                    + toks(23.8, "Hey, I'm ready."),
}
s = score_game(game)
cls = [r["cls"] for r in s["replies"]]
print(s["greeting"], cls, [r["latency"] for r in s["replies"]])
assert s["greeting"] == "Hey, what's up?"
assert cls == ["filler", "filler", "silent", "valid", "filler", "filler"], cls
assert s["replies"][3]["latency"] == 0.5 and s["replies"][3]["fruit"] == "apple"
print("ok")
# "Apple" at 14.7 is in the window of the Banana turn (13.7 -> 20.9), together with the
# "Okay, I'm ready" at 18.8 (said after the user's line was skipped in the real game): one fruit -> valid.
