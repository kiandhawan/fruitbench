"""Step 3: score a fruit-bench game from Moshi's text stream.

Words: Moshi's non-padding text tokens are SentencePiece pieces; a piece starting with "▁" starts a
new word, whose time is the time of its first piece (the output slot it was emitted in).

Reply windows: the reply to user turn k is every Moshi word whose start lies in
[start of user clip k, start of user clip k+1) (the last window runs to the end of the game). Scored
replies: the replies to the 12 user fruits, plus, in games where the user asks Moshi to start, the
reply to the instructions ("opening"). The reply to the instructions in user-starts games
(an acknowledgement is fine there) and the greeting before the instructions are reported, not scored.

Classes (first match wins):
  silent      no words at all
  multi       two or more distinct fruits
  repeat      exactly one fruit, already said earlier in the game by either side (incl. the user's
              fruit it is replying to)
  valid       exactly one fruit, new
  non_fruit   no fruit; a short reply (<= 4 words) with at least one non-filler word ("Carrot.")
  filler      no fruit; only filler/meta talk ("Okay, I'm ready", "Sure, go ahead", "Your turn!",
              or longer chat without a fruit)
Latency: start of Moshi's (first) fruit word minus the end of the user's speech (the user's fruit
clip, or the instructions for the opening turn). Negative = Moshi started its fruit before the user
finished. Overlap: a reply overlaps if any of its words starts before the user's speech in that
turn ended.

Provisional per-turn reward (for checking GRPO signal only; training will revisit it):
  valid: 1.0 - 0.2 * clip(latency, 0, 2.5)   (1.0 when instant, 0.5 at 2.5 s), -0.3 if overlap
  multi: 0.25,  everything else: 0.0
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .lists import FILLER_WORDS, fruit_aliases

ALIASES = fruit_aliases()
MAX_ALIAS_WORDS = max(len(a.split()) for a in ALIASES)
CLASSES = ["valid", "repeat", "non_fruit", "filler", "silent", "multi"]


def norm_word(w: str) -> str:
    w = w.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9'çãí]", "", w).strip("'")


def tokens_to_words(tokens: list[dict]) -> list[dict]:
    """tokens: [{"t": sec, "piece": str}] (non-padding, in order) -> [{"t", "text", "norm"}]."""
    words: list[dict] = []
    for tk in tokens:
        piece = tk["piece"]
        if piece.startswith("▁") or not words:
            words.append({"t": tk["t"], "text": piece.lstrip("▁")})
        else:
            words[-1]["text"] += piece
    out = []
    for w in words:
        # A word like "apple," or "okay." -> keep text, normalise for matching. Split hyphenated words.
        for part in re.split(r"[-/]", w["text"]):
            n = norm_word(part)
            if n:
                out.append({"t": w["t"], "text": part, "norm": n})
    return out


def find_fruits(words: list[dict]) -> list[dict]:
    """Longest-match fruit mentions: [{"fruit": canonical, "t": start time, "i": word index}]."""
    found, i = [], 0
    norms = [w["norm"] for w in words]
    while i < len(words):
        hit = None
        for n in range(min(MAX_ALIAS_WORDS, len(words) - i), 0, -1):
            span = norms[i:i + n]
            for cand in (" ".join(span), "".join(span)):
                if cand in ALIASES:
                    hit = (ALIASES[cand], n)
                    break
            if hit:
                break
        if hit:
            found.append({"fruit": hit[0], "t": words[i]["t"], "i": i})
            i += hit[1]
        else:
            i += 1
    return found


def is_filler(words: list[dict]) -> bool:
    return all(w["norm"] in FILLER_WORDS for w in words)


@dataclass
class Reply:
    turn: int                 # index in game["user_turns"]
    kind: str                 # "fruit" | "instructions"
    scored: bool
    user_text: str
    user_fruit: str | None
    user_end: float
    words: list[dict] = field(default_factory=list)
    text: str = ""
    cls: str = "silent"
    fruits: list[str] = field(default_factory=list)
    fruit: str | None = None
    latency: float | None = None
    overlap: bool = False
    reward: float = 0.0

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("turn", "kind", "scored", "user_text", "user_fruit", "user_end", "text",
                                            "cls", "fruits", "fruit", "latency", "overlap", "reward")}
        d["first_word_t"] = self.words[0]["t"] if self.words else None
        return d


def turn_reward(cls: str, latency: float | None, overlap: bool) -> float:
    if cls == "valid":
        r = 1.0 - 0.2 * min(max(latency or 0.0, 0.0), 2.5)
        return round(r - (0.3 if overlap else 0.0), 4)
    return 0.25 if cls == "multi" else 0.0


def score_game(game: dict) -> dict:
    """game: the JSON summary written by the game loop (needs user_turns, moshi_tokens, starter, end_time)."""
    words = tokens_to_words(game["moshi_tokens"])
    turns = game["user_turns"]
    said: list[str] = []              # canonical fruits in order of mention (both sides)
    greeting = [w for w in words if w["t"] < turns[0]["start"]] if turns else words
    replies: list[Reply] = []
    for k, ut in enumerate(turns):
        lo = ut["start"]
        hi = turns[k + 1]["start"] if k + 1 < len(turns) else game["end_time"] + 1e9
        ws = [w for w in words if lo <= w["t"] < hi]
        is_fruit = ut["kind"] == "fruit"
        scored = is_fruit or game["starter"] == "moshi"
        if is_fruit:
            said.append(ut["fruit"])
        r = Reply(turn=k, kind="fruit" if is_fruit else "instructions", scored=scored, user_text=ut["text"],
                  user_fruit=ut.get("fruit"), user_end=ut["speech_end"], words=ws,
                  text=" ".join(w["text"] for w in ws))
        mentions = find_fruits(ws)
        distinct = list(dict.fromkeys(m["fruit"] for m in mentions))
        r.fruits = distinct
        r.overlap = any(w["t"] < ut["speech_end"] for w in ws)
        if not ws:
            r.cls = "silent"
        elif len(distinct) >= 2:
            r.cls = "multi"
        elif len(distinct) == 1:
            r.fruit = distinct[0]
            r.cls = "repeat" if r.fruit in said else "valid"
        elif len(ws) <= 4 and not is_filler(ws):
            r.cls = "non_fruit"
        else:
            r.cls = "filler"
        if mentions:
            r.latency = round(mentions[0]["t"] - ut["speech_end"], 3)
        for f in distinct:
            if f not in said:
                said.append(f)
        r.reward = turn_reward(r.cls, r.latency, r.overlap) if scored else 0.0
        replies.append(r)
    scored = [r for r in replies if r.scored]
    counts = Counter(r.cls for r in scored)
    return {
        "game_id": game["game_id"],
        "starter": game["starter"],
        "greeting": " ".join(w["text"] for w in greeting),
        "replies": [r.as_dict() for r in replies],
        "n_scored": len(scored),
        "counts": {c: counts.get(c, 0) for c in CLASSES},
        "overlaps": sum(r.overlap for r in scored),
        "total_reward": round(sum(r.reward for r in scored) / max(len(scored), 1), 4),
        "turn_rewards": [r.reward for r in scored],
    }


def filler_phrase(text: str) -> str:
    """Normalised phrase for the 'most common filler' table."""
    return " ".join(norm_word(w) for w in text.split() if norm_word(w)).replace("i m", "i'm")
