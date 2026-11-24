"""Step 3: score a fruit-bench game from Moshi's text stream.

Words: Moshi's non-padding text tokens are SentencePiece pieces; a piece starting with "▁" starts a
new word, whose time is the time of its first piece (the output slot it was emitted in).

Utterances: consecutive words less than 0.5 s apart (the same pause that ends Moshi's turn in the
game loop). Reply windows: the reply to user turn k is every Moshi utterance whose first word starts
in [start of user clip k, start of user clip k+1) (the last window runs to the end of the game), so
an utterance that runs past the next user clip's start stays with the turn it began in. Scored
replies: the replies to the 12 user fruits, plus, in games where the user asks Moshi to start, the
reply to the instructions ("opening"). The reply to the instructions in user-starts games
(an acknowledgement is fine there) and the greeting before the instructions are reported, not scored.

Classes (first match wins):
  silent      no words at all
  multi       two or more distinct fruits
  repeat      exactly one fruit, already said earlier in the game by either side (incl. the user's
              fruit it is replying to; `echo` = it is that fruit). Moshi's text often misspells a rare
              fruit it echoes ("Lichie", "samumberry", "kolamansee"), so words of 5+ letters that
              are not filler are also fuzzy-matched (difflib ratio >= 0.6, and consonant skeletons
              with the same first letter matching >= 0.75) against the fruits already
              said in the game (only those: fuzzy matching all fruits turns "speech" into "peach").
  valid       exactly one fruit, new
  non_fruit   no fruit, but Moshi names something else: "go with / I'll say / how about X" with X not
              a fruit ("I'm going to go with a lobster"), or a short reply (<= 4 words) with at most
              2 non-filler words ("Carrot.", "Potato chips!")
  filler      no fruit; filler/meta talk ("Okay, I'm ready", "Sure, go ahead", "I'm out of ideas",
              or longer chat without a fruit)
Latency: start of Moshi's (first) fruit word minus the end of the user's speech (the user's fruit
clip, or the instructions for the opening turn). Negative = Moshi started its fruit before the user
finished. Overlap: a reply overlaps if any of its words falls inside the user's speech, in that
turn's clip or the next one (Moshi talking over the user).

Provisional per-turn reward (for checking GRPO signal only; training will revisit it):
  valid: 1.0 - 0.2 * clip(latency, 0, 2.5)   (1.0 when instant, 0.5 at 2.5 s), -0.3 if overlap
  multi: 0.25 if it contains at least one new fruit (else 0),  everything else: 0.0
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .lists import FILLER_WORDS, fruit_aliases

ALIASES = fruit_aliases("v1")
ALIASES_BY_RULES = {"v1": ALIASES, "v3": fruit_aliases("v3")}
MAX_ALIAS_WORDS = max(len(a.split()) for m in ALIASES_BY_RULES.values() for a in m)
CLASSES = ["valid", "sentence", "repeat", "non_fruit", "filler", "silent", "multi"]
ARTICLES = {"a", "an", "some"}


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


def find_fruits(words: list[dict], aliases: dict | None = None) -> list[dict]:
    """Longest-match fruit mentions: [{"fruit": canonical, "t": start time, "i": word index, "n": words}]."""
    ALIASES = aliases if aliases is not None else ALIASES_BY_RULES["v1"]  # noqa: N806
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
            found.append({"fruit": hit[0], "t": words[i]["t"], "i": i, "n": hit[1]})
            i += hit[1]
        else:
            i += 1
    return found


UTTERANCE_GAP = 0.5
NAMING = re.compile(r"\b(?:go(?:ing)? with|i(?:'ll| will)? say|how about|what about|i choose|i pick|"
                    r"mine is|my (?:fruit|pick|answer) is)\s+(?:an? |the |some )?([a-z']+)")


def utterances(words: list[dict]) -> list[list[dict]]:
    out: list[list[dict]] = []
    for w in words:
        if out and w["t"] - out[-1][-1]["t"] < UTTERANCE_GAP:
            out[-1].append(w)
        else:
            out.append([w])
    return out


FUZZY_RATIO = 0.6
FUZZY_MIN_LEN = 5


def skeleton(w: str) -> str:
    """Consonant skeleton (c/k/q merged, y as a vowel): 'lichie'/'lychee' -> 'lch'."""
    w = re.sub(r"[ck]+|q", "k", w)
    return re.sub(r"[aeiouy']", "", w)


def fuzzy_said(words: list[dict], said: list[str], skip: set[int], aliases: dict | None = None,
               same_first_letter: bool = False) -> list[dict]:
    """Misspelled mentions of fruits already said in the game (canonical names in `said`)."""
    import difflib
    ALIASES = aliases if aliases is not None else ALIASES_BY_RULES["v1"]  # noqa: N806
    forms = {}
    for a, c in ALIASES.items():
        if c in said:
            forms[a.replace(" ", "")] = c
    out = []
    for i, w in enumerate(words):
        if i in skip or len(w["norm"]) < FUZZY_MIN_LEN or w["norm"] in FILLER_WORDS:
            continue
        cands = [w["norm"]]
        if i + 1 < len(words) and i + 1 not in skip:
            cands.append(w["norm"] + words[i + 1]["norm"])
        # Spelling ratio >= 0.6, and the consonant skeletons must start alike and match >= 0.75
        # ("benefits" vs "breadfruits" is 0.6 by spelling but bnfts/brdfrts 0.67; "speech" vs "peach"
        # starts differently; "lichie" vs "lychee": skeletons identical).
        def sim(c, f):
            if same_first_letter and c[:1].replace("c", "k") != f[:1].replace("c", "k"):
                return 0.0  # v3: "places" is not a misspelled "apples"
            sc, sf = skeleton(c), skeleton(f)
            if not sc or not sf or sc[0] != sf[0] or difflib.SequenceMatcher(None, sc, sf).ratio() < 0.75:
                return 0.0
            return difflib.SequenceMatcher(None, c, f).ratio()
        best = max(((sim(c, f), f) for c in cands for f in forms), default=(0, None))
        if best[0] >= FUZZY_RATIO:
            out.append({"fruit": forms[best[1]], "t": w["t"], "i": i, "fuzzy": w["text"]})
    return out


def is_filler(words: list[dict]) -> bool:
    return all(w["norm"] in FILLER_WORDS for w in words)


def names_non_fruit(words: list[dict]) -> bool:
    content = [w for w in words if w["norm"] not in FILLER_WORDS]
    if content and len(words) <= 4 and len(content) <= 2:
        return True
    m = NAMING.search(" ".join(w["norm"] for w in words))
    return bool(m) and m.group(1) not in FILLER_WORDS


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
    echo: bool = False
    fuzzy: list[str] = field(default_factory=list)
    reward: float = 0.0
    multi_new: bool = False   # multi reply that contains at least one new fruit
    user_spans: list = field(default_factory=list)  # (start, speech_end) of this and the next user line

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("turn", "kind", "scored", "user_text", "user_fruit", "user_end", "text",
                                            "cls", "fruits", "fruit", "latency", "overlap", "echo", "fuzzy", "reward", "multi_new")}
        d["first_word_t"] = self.words[0]["t"] if self.words else None
        d["word_t"] = [w["t"] for w in self.words]          # for reward terms (word count, overlap grace)
        d["user_spans"] = [list(x) for x in self.user_spans]
        return d


def turn_reward(cls: str, latency: float | None, overlap: bool, multi_new: bool = True) -> float:
    if cls == "valid":
        r = 1.0 - 0.2 * min(max(latency or 0.0, 0.0), 2.5)
        return round(r - (0.3 if overlap else 0.0), 4)
    return 0.25 if cls == "multi" and multi_new else 0.0


def is_bare(ws: list[dict], mentions: list[dict]) -> bool:
    """v3 'valid' form: the whole reply is one fruit name (any number of words, plural allowed),
    optionally preceded by a single "a" / "an" / "some"; punctuation is ignored."""
    if len(mentions) != 1 or "fuzzy" in mentions[0]:
        return False
    m = mentions[0]
    return m["i"] + m["n"] == len(ws) and m["i"] <= 1 and all(w["norm"] in ARTICLES for w in ws[:m["i"]])


def score_game(game: dict, rules: str = "v1") -> dict:
    """game: the JSON summary written by the game loop (needs user_turns, moshi_tokens, starter, end_time).
    rules="v1": base benchmark rules. rules="v3": common-sense fruit list, and a reply with one new
    fruit is "valid" only if it is the bare fruit name (see is_bare), otherwise "sentence"."""
    aliases = ALIASES_BY_RULES[rules]
    words = tokens_to_words(game["moshi_tokens"])
    utts = utterances(words)
    turns = game["user_turns"]
    said: list[str] = []              # canonical fruits in order of mention (both sides)
    first_start = turns[0]["start"] if turns else 1e9
    greeting = [w for u in utts if u[0]["t"] < first_start for w in u]
    speech = [(t["start"], t["speech_end"]) for t in turns]
    replies: list[Reply] = []
    for k, ut in enumerate(turns):
        lo = ut["start"]
        hi = turns[k + 1]["start"] if k + 1 < len(turns) else game["end_time"] + 1e9
        ws = [w for u in utts if lo <= u[0]["t"] < hi for w in u]
        is_fruit = ut["kind"] == "fruit"
        scored = is_fruit or game["starter"] == "moshi"
        if is_fruit:
            said.append(ut["fruit"])
        r = Reply(turn=k, kind="fruit" if is_fruit else "instructions", scored=scored, user_text=ut["text"],
                  user_fruit=ut.get("fruit"), user_end=ut["speech_end"], words=ws,
                  text=" ".join(w["text"] for w in ws))
        mentions = find_fruits(ws, aliases)
        exact_idx = {j for m in mentions for j in range(m["i"], m["i"] + m["n"])}
        fuzzy = fuzzy_said(ws, said, exact_idx, aliases, same_first_letter=rules == "v3")
        r.fuzzy = [f"{m['fuzzy']}->{m['fruit']}" for m in fuzzy]
        mentions = sorted(mentions + fuzzy, key=lambda m: m["i"])
        distinct = list(dict.fromkeys(m["fruit"] for m in mentions))
        r.fruits = distinct
        talk_over = speech[k:k + 2]
        r.user_spans = talk_over
        r.overlap = any(a <= w["t"] < b for w in ws for a, b in talk_over)
        if not ws:
            r.cls = "silent"
        elif len(distinct) >= 2:
            r.cls = "multi"
        elif len(distinct) == 1:
            r.fruit = distinct[0]
            r.cls = "repeat" if r.fruit in said else "valid"
            if rules == "v3" and r.cls == "valid" and not is_bare(ws, mentions):
                r.cls = "sentence"
            r.echo = is_fruit and r.fruit == ut["fruit"]
        elif names_non_fruit(ws):
            r.cls = "non_fruit"
        else:
            r.cls = "filler"
        if mentions:
            r.latency = round(mentions[0]["t"] - ut["speech_end"], 3)
        multi_new = any(f not in said for f in distinct)
        r.multi_new = r.cls == "multi" and multi_new
        for f in distinct:
            if f not in said:
                said.append(f)
        r.reward = turn_reward(r.cls, r.latency, r.overlap, multi_new) if scored else 0.0
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
