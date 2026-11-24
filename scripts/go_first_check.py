"""Moshi turns that announce "I'll go first" / "let me start" / "me first": did Moshi go on to say a
fruit in that turn, or was it cut off by the next user line (Moshi's audio, > -45 dBFS, resumes
within 1 s after the user line starts)?

  python scripts/go_first_check.py results/base_games/test_s1  -> results/go_first_check.json
"""
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fruitbench.score import find_fruits, tokens_to_words, utterances  # noqa: E402

PAT = re.compile(r"\b(i'll go first|i will go first|let me go first|let me start|me first)\b")


def main() -> None:
    d = Path(sys.argv[1])
    rows = []
    for f in sorted(glob.glob(str(d / "*_[0-9][0-9][0-9][0-9].json"))):
        g = json.load(open(f))
        words, lv, T = tokens_to_words(g["moshi_tokens"]), g["moshi_dbfs"], g["user_turns"]
        thr = g.get("speech_dbfs", -45.0)
        utts = utterances(words)
        # Moshi turn windows: greeting [0, first user line), then after user line k until line k+1.
        bounds = [(0.0, T[0]["start"], -1)] + [(T[k]["start"], T[k + 1]["start"] if k + 1 < len(T) else 1e9, k)
                                               for k in range(len(T))]
        for lo, hi, k in bounds:
            ws = [w for u in utts if lo <= u[0]["t"] < hi for w in u]
            txt = " ".join(w["norm"] for w in ws)
            m = PAT.search(txt)
            if not m:
                continue
            pre = txt[:m.end()].count(" ") + 1
            fruits = [x["fruit"] for x in find_fruits(ws[pre:])]
            r = {"game": g["game_id"], "starter": g["starter"], "after_user_line": k, "phrase": m.group(1),
                 "text": " ".join(w["text"] for w in ws), "fruit_in_turn": fruits, "cut_off": None}
            if k + 1 < len(T):
                s = T[k + 1]["start"]
                j = int(round(s / 0.08))
                res = [i * 0.08 - s for i in range(j, min(len(lv), j + 13)) if lv[i] > thr]
                after = [w for w in words if s <= w["t"] < s + 3]
                r.update(cut_off=bool(res), resume_after_sec=round(res[0], 2) if res else None,
                         next_user_line=T[k + 1]["text"], next_user_speech_end=round(T[k + 1]["speech_end"] - s, 2),
                         moshi_after=" ".join(w["text"] for w in after),
                         fruit_after=[x["fruit"] for x in find_fruits(after)])
            rows.append(r)
    n = len(rows)
    out = {"turns": n, "games": len({r["game"] for r in rows}),
           "fruit_in_turn": sum(bool(r["fruit_in_turn"]) for r in rows),
           "cut_off": sum(bool(r["cut_off"]) for r in rows),
           "cut_off_without_fruit_in_turn": sum(bool(r["cut_off"]) and not r["fruit_in_turn"] for r in rows),
           "neither": sum(not r["fruit_in_turn"] and not r["cut_off"] for r in rows), "rows": rows}
    (ROOT / "results" / "go_first_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))


if __name__ == "__main__":
    main()
