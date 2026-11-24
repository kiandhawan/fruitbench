"""Credit-window unit tests (full-run rule, fruitbench/credit.py).

Game: instructions 0-3 s; user "Apple." 4.0-4.6 s; "Kiwi." 10.0-10.7 s; "Pear." 13.0-13.6 s.
1. Early "Mango." starting 0.5 s before "Kiwi." ends (10.2 s): the scorer assigns it to the reply to
   "Kiwi." (overlap -> capped reward). Its text token must lie in that reply's credit window (new
   rule), whereas the legacy smoke 1-4 rule put it in the window of the reply to "Apple.".
2. Carry-over: a reply to "Apple." that keeps talking across the start of "Kiwi." keeps those frames
   (to its last word + 0.5 s) in its own window.
  python tests/test_credit.py [configs/grpo_full.yaml]
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fruitbench.credit import credit_windows, frame_of, legacy_windows  # noqa: E402
from fruitbench.rewards import game_rewards  # noqa: E402
from fruitbench.score import score_game  # noqa: E402

CFG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "grpo_full.yaml"
W = yaml.safe_load(CFG.read_text())["reward"]
T = 400


def game(moshi: list[tuple[float, str]]) -> dict:
    toks = [{"t": t, "id": 1, "piece": "▁" + w} for t, w in moshi]
    turns = [{"kind": "instructions", "text": "Let's name fruits.", "fruit": None, "start": 0.0, "speech_end": 3.0},
             {"kind": "fruit", "text": "Apple.", "fruit": "apple", "start": 4.0, "speech_end": 4.6},
             {"kind": "fruit", "text": "Kiwi.", "fruit": "kiwi", "start": 10.0, "speech_end": 10.7},
             {"kind": "fruit", "text": "Pear.", "fruit": "pear", "start": 13.0, "speech_end": 13.6}]
    return {"game_id": "unit", "starter": "user", "end_time": 16.0, "user_turns": turns, "moshi_tokens": toks}


def which(windows, f):
    return [j for j, a, b in windows if a <= f < b]


def main() -> None:
    # 1. early Mango over the end of "Kiwi." (+ a normal "Banana." reply to "Apple.")
    g = game([(5.0, "Banana."), (10.2, "Mango.")])
    sc = score_game(g, W.get("scorer", "v3"))
    rw = game_rewards(sc, W)
    f = frame_of(10.2)
    new, old = credit_windows(g, sc, T), legacy_windows(g, sc, T)
    kiwi_j = 1  # scored replies: 0 = reply to Apple, 1 = reply to Kiwi, 2 = reply to Pear
    print(f"test 1: 'Mango.' at 10.20 s -> frame {f}; reply to Kiwi: {sc['replies'][2]['cls']}, reward {rw[kiwi_j]:+.2f}")
    print(f"  new windows {new}\n  old windows {old}")
    print(f"  frame {f} in new window of scored reply {which(new, f)}, legacy window {which(old, f)}")
    assert which(new, f) == [kiwi_j], "new rule: early fruit must be credited to its own reply"
    assert which(old, f) == [0], "legacy rule put it in the previous reply's window (the bug)"
    assert rw[kiwi_j] <= W["valid_overlap_cap"] + 1e-9
    # 2. carry-over: reply to Apple talks from 9.5 s to 10.3 s across the start of "Kiwi." (10.0 s)
    g2 = game([(9.5, "I"), (9.7, "like"), (9.9, "that"), (10.1, "one"), (10.3, "too."), (11.0, "Mango.")])
    sc2 = score_game(g2, W.get("scorer", "v3"))
    new2 = credit_windows(g2, sc2, T)
    print(f"test 2: reply to Apple = {sc2['replies'][1]['text']!r}, reply to Kiwi = {sc2['replies'][2]['text']!r}")
    print(f"  windows {new2}; frames of 'one' (10.1 s) {frame_of(10.1)}, 'too.' (10.3 s) {frame_of(10.3)}, "
          f"'Mango.' (11.0 s) {frame_of(11.0)}")
    assert which(new2, frame_of(10.3)) == [0] and which(new2, frame_of(10.3 + 0.4)) == [0] and which(new2, frame_of(10.3 + 0.5)) == [1]
    assert which(new2, frame_of(11.0)) == [1]
    print("ok")


if __name__ == "__main__":
    main()
