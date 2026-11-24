"""Print (and save) a readable timeline for fruit-bench games.

  python scripts/timeline.py runs/smoke            # every game in the folder -> <dir>/timelines.txt
  python scripts/timeline.py runs/smoke/<id>.json  # one game

Lines: USER clips with speech span and why they were inserted; MOSHI utterances (words grouped
when < 0.4 s apart) with per-word times; Moshi turn ends with the rule that fired; and the scorer's
class for each reply.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fruitbench.score import score_game, tokens_to_words  # noqa: E402

GROUP_GAP = 0.4


def timeline(game: dict) -> str:
    sc = score_game(game)
    words = tokens_to_words(game["moshi_tokens"])
    rows = []
    utt: list[dict] = []

    def flush():
        if utt:
            txt = " ".join(f"{w['text']}@{w['t']:.2f}" for w in utt)
            rows.append((utt[0]["t"], 1, f"MOSHI  {utt[0]['t']:6.2f}-{utt[-1]['t']:6.2f}  {txt}"))
            utt.clear()

    for w in words:
        if utt and w["t"] - utt[-1]["t"] > GROUP_GAP:
            flush()
        utt.append(w)
    flush()
    for k, t in enumerate(game["user_turns"]):
        rows.append((t["start"], 0, f"USER   {t['start']:6.2f}-{t['speech_end']:6.2f}  {t['text']!r:<58} [{t['reason']}]"))
    for e in game["events"]:
        if e["event"] != "user_clip":
            rows.append((e["t"], 2, f"  --   {e['t']:6.2f}         {e['event']}: {e['why']}"))
    for r in sc["replies"]:
        ut = game["user_turns"][r["turn"]]
        tag = "scored" if r["scored"] else "not scored"
        lat = f", latency {r['latency']:+.2f}s" if r["latency"] is not None else ""
        rows.append((ut["speech_end"], 3, f"  >>   reply to {ut['text'][:30]!r}: {r['cls'].upper()}"
                     f"{' (' + r['fruit'] + ')' if r['fruit'] else ''}{' ECHO' if r['echo'] else ''}"
                     f"{' fuzzy ' + str(r['fuzzy']) if r['fuzzy'] else ''}{lat}{', OVERLAP' if r['overlap'] else ''} [{tag}]"))
    # Turn-taking check: at each user clip start, is Moshi mid-sentence? Flag if Moshi's audio was
    # speech in the 0.3 s before the clip, or its last word before the clip has no end punctuation
    # and its next word follows within 1.0 s after the clip starts (the sentence continues).
    lv, thr = game.get("moshi_dbfs"), game.get("speech_dbfs", -45.0)
    flags = []
    for t in game["user_turns"]:
        s0 = t["start"]
        why = []
        if lv:
            j = int(round(s0 / 0.08))
            if any(x > thr for x in lv[max(0, j - 4):j]):
                why.append("Moshi audio still speech in last 0.3 s")
        before = [w for w in words if w["t"] < s0]
        after = [w for w in words if s0 <= w["t"] < s0 + 1.0]
        if before and after and not before[-1]["text"].rstrip().endswith((".", "?", "!")):
            why.append(f"sentence continues: ...{before[-1]['text']!r}@{before[-1]['t']:.2f} -> {after[0]['text']!r}@{after[0]['t']:.2f}")
        if why:
            flags.append(f"{s0:.2f}s: " + "; ".join(why))
            rows.append((s0, 0.5, "  !!   user started while Moshi mid-sentence: " + "; ".join(why)))
    game["_midsentence_flags"] = flags
    # Show each reply verdict right after that turn's Moshi turn-end line.
    ends = [e["t"] for e in game["events"] if e["event"] in ("moshi_turn_end", "game_cap")]
    fixed = []
    for t, pri, s in rows:
        if pri == 3:
            later = [x for x in ends if x >= t]
            t = later[0] if later else t
        fixed.append((t, pri, s))
    fixed.sort(key=lambda x: (x[0], x[1]))
    sp = game["spec"]
    head = (f"=== {game['game_id']}  starter={game['starter']}  voice={sp['voice']}  length={game['end_time']:.1f}s\n"
            f"    instructions: {sp['phrasing']!r}\n    greeting: {sc['greeting']!r}\n"
            f"    counts: {sc['counts']}  overlaps={sc['overlaps']}  reward={sc['total_reward']}\n"
            f"    user lines started while Moshi mid-sentence: {len(flags)} of {len(game['user_turns'])}")
    return head + "\n" + "\n".join(s for _, _, s in fixed) + "\n"


def main() -> None:
    p = Path(sys.argv[1])
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    games = [json.loads(f.read_text()) for f in files]
    games = [g for g in games if isinstance(g, dict) and "user_turns" in g]
    text = "\n".join(timeline(g) for g in games)
    print(text)
    if p.is_dir():
        (p / "timelines.txt").write_text(text)


if __name__ == "__main__":
    main()
