"""Ordinary (non-game) conversations for the "does Moshi still talk normally?" check at validation.

20 short scripted conversations (3 user lines each, rendered with the 4 train voices, kind "chat" in
data/clips.jsonl). Nothing in them mentions fruit or words that are also fruit names (date, orange,
lime, ...). Metric: % of Moshi's replies to the chat lines that contain a fruit (v3 list); flag > 10%.
"""
from __future__ import annotations

import json
import random

CONVERSATIONS: list[list[str]] = [
    ["Hi, how are you?", "What have you been up to today?", "That sounds nice. Anything planned for the weekend?"],
    ["Can you help me plan a trip?", "I'm thinking about a week in Portugal in the spring.", "What should I pack?"],
    ["I just got a new puppy.", "Do you have any tips for house training?", "How long does that usually take?"],
    ["What's a good way to start running?", "I can only run for about five minutes right now.", "How often should I go out?"],
    ["Can you recommend a good book?", "I like mysteries and detective stories.", "Is that one suitable for a teenager?"],
    ["I have a job interview tomorrow.", "I'm really nervous about it.", "What questions should I prepare for?"],
    ["How do I fix a slow laptop?", "It takes forever to start up.", "Should I buy a new one instead?"],
    ["Tell me something interesting about space.", "How far away is the moon?", "Could people live on Mars one day?"],
    ["I want to learn to play the guitar.", "Should I start with an acoustic or an electric one?", "How many hours a week should I practice?"],
    ["What's the best way to save money?", "I always spend too much on takeaway.", "Do budgeting apps actually help?"],
    ["Can you explain how rainbows form?", "Why are they curved?", "Can you ever reach the end of one?"],
    ["I can't sleep well lately.", "I usually look at my phone in bed.", "What else could I try?"],
    ["Let's talk about movies.", "What's a good film for a family night?", "Something animated would be great."],
    ["How do I make my plants grow better?", "They are on a windowsill that faces north.", "How often should I water them?"],
    ["I'm learning Spanish.", "What's the hardest part for English speakers?", "Can you say good morning in Spanish?"],
    ["My car is making a strange noise.", "It's a squeaking sound when I brake.", "Is it safe to keep driving?"],
    ["What's a good hobby to pick up in winter?", "Something I can do indoors.", "How much would it cost to start?"],
    ["I want to be more productive at work.", "I get distracted by emails all the time.", "How long should my breaks be?"],
    ["Can you tell me a fun fact about the ocean?", "How deep is the deepest part?", "What lives down there?"],
    ["I'm hosting a dinner party next week.", "There will be eight people.", "How should I arrange the seating?"],
]


def chat_specs(seed: int = 0) -> list:
    """GameSpecs for the 20 conversations (train voices, rotating), user speaks after Moshi's greeting."""
    from .game import GameSpec
    voices = [v["title"] for v in json.loads((__import__("pathlib").Path(__file__).resolve().parent.parent
                                              / "data" / "voices.json").read_text())["voices"] if v["split"] != "test"]
    rng = random.Random(seed)
    out = []
    for i, conv in enumerate(CONVERSATIONS):
        out.append(GameSpec(game_id=f"chat_{i:02d}", split="chat", seed=seed, voice=voices[i % len(voices)],
                            starter="user", phrasing="", fruit_order=[], gaps=[round(rng.uniform(0.3, 0.8), 3) for _ in range(6)],
                            n_user_fruits=len(conv), script=[f"chat{i:02d}_{j}" for j in range(len(conv))]))
    return out


def chat_fruit_rate(games: list[dict]) -> dict:
    """% of Moshi replies to chat lines that mention a fruit (v3 list)."""
    from .score import ALIASES_BY_RULES, find_fruits, tokens_to_words, utterances
    n = hit = 0
    examples = []
    for g in games:
        words = tokens_to_words(g["moshi_tokens"])
        utts = utterances(words)
        ut = g["user_turns"]
        for k, t in enumerate(ut):
            lo, hi = t["start"], ut[k + 1]["start"] if k + 1 < len(ut) else g["end_time"] + 1e9
            ws = [w for u in utts if lo <= u[0]["t"] < hi for w in u]
            fr = [m["fruit"] for m in find_fruits(ws, ALIASES_BY_RULES["v3"])]
            n += 1
            if fr:
                hit += 1
                examples.append({"game": g["game_id"], "user": t["text"], "moshi": " ".join(w["text"] for w in ws)[:160]})
    return {"replies": n, "with_fruit": hit, "rate": round(hit / max(n, 1), 4), "examples": examples[:10]}
