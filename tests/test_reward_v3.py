"""Reward v3 unit table: each case is a tiny game run through the real scorer (rules v3) and reward
(configs/grpo_smoke3.yaml). Prints the table and writes results/reward_v3_table.md.

Game: instructions 0-3 s; user "Apple." 4.0-4.6 s (so apple is already said); user "Kiwi." 10.0-10.7 s
(the turn under test); user "Pear." 13.0-13.6 s. Moshi's test reply starts at the given time, words
0.15 s apart; latency = first fruit word start - 10.7 s.

  python tests/test_reward_v3.py
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fruitbench.rewards import counted_overlap, turn_reward  # noqa: E402
from fruitbench.score import score_game  # noqa: E402

W = yaml.safe_load((ROOT / "configs" / "grpo_smoke3.yaml").read_text())["reward"]
KIWI_END = 10.7

CASES = [  # (label, Moshi reply text, start time)
    ('"Mango." fast (0.3 s)', "Mango.", 11.0),
    ('"Mango." slow (1.5 s)', "Mango.", 12.2),
    ('"Mango." in the grace window (starts 0.1 s before user ends)', "Mango.", 10.6),
    ('"Mango." over the user (starts 0.5 s before user ends)', "Mango.", 10.2),
    ('"A mango!"', "A mango!", 11.0),
    ('"Mangoes?"', "Mangoes?", 11.0),
    ('"Dragon fruit."', "Dragon fruit.", 11.0),
    ('"I\'ll go with mango."', "I'll go with mango.", 11.0),
    ('"The sixth one is a mango."', "The sixth one is a mango.", 11.0),
    ('"Um, mango."', "Um, mango.", 11.0),
    ("13-word fruit fact", "Mangoes are a tropical fruit that grow on tall trees in warm places.", 10.9),
    ('"Mango, kiwi, plum."', "Mango, kiwi, plum.", 11.0),
    ('"Okay."', "Okay.", 11.0),
    ("silence", "", 11.0),
    ('"Carrot."', "Carrot.", 11.0),
    ('echo: "Kiwi."', "Kiwi.", 11.0),
    ('repeat: "Apple." (user said it earlier)', "Apple.", 11.0),
    ('echo over the user: "Kiwi." starting 0.5 s before user ends', "Kiwi.", 10.2),
]


def game(text: str, t0: float) -> dict:
    toks = [{"t": round(t0 + 0.15 * i, 3), "id": 1, "piece": "▁" + w} for i, w in enumerate(text.split())]
    turns = [{"kind": "instructions", "text": "Let's name fruits.", "fruit": None, "start": 0.0, "speech_end": 3.0},
             {"kind": "fruit", "text": "Apple.", "fruit": "apple", "start": 4.0, "speech_end": 4.6},
             {"kind": "fruit", "text": "Kiwi.", "fruit": "kiwi", "start": 10.0, "speech_end": KIWI_END},
             {"kind": "fruit", "text": "Pear.", "fruit": "pear", "start": 13.0, "speech_end": 13.6}]
    return {"game_id": "unit", "starter": "user", "end_time": 16.0, "user_turns": turns, "moshi_tokens": toks}


def main() -> None:
    rows = ["| case | Moshi reply | class (v3) | latency | counted overlap | words | reward v3 |",
            "|---|---|---|---|---|---|---|"]
    for label, text, t0 in CASES:
        sc = score_game(game(text, t0), "v3")
        r = sc["replies"][2]  # the reply to "Kiwi."
        lat = f"{r['latency']:+.2f}" if r["latency"] is not None else ""
        rows.append(f"| {label} | {text or '(nothing)'} | {r['cls']}{' (echo)' if r['echo'] else ''} | {lat} | "
                    f"{'yes' if counted_overlap(r, W) else 'no'} | {len(r['word_t'])} | **{turn_reward(r, W):+.2f}** |")
    table = "\n".join(rows)
    print(table)
    (ROOT / "results" / "reward_v3_table.md").write_text(
        "# Reward v3 unit table (`tests/test_reward_v3.py`, `configs/grpo_smoke3.yaml`)\n\n"
        "Reply to the user's \"Kiwi.\" (user speech ends at 10.7 s; \"Apple.\" was said earlier).\n\n" + table + "\n")


if __name__ == "__main__":
    main()
