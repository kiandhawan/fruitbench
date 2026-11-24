"""Step 3 check: does Moshi's AUDIO say what its text stream says?

Transcribes Moshi's channel (left) of N games with faster-whisper large-v3 (word timestamps), turns
the Whisper words into a pseudo text stream (one "word token" at each word's start time), scores it
with the same scorer, and compares the per-turn class with the text-stream classification.

Runs in the separate ASR venv (faster-whisper + CUDA libs), so the Moshi venv stays untouched:
  .venv-asr/bin/python scripts/asr_check.py runs/base_test_s1 --n 20 --gpu 0
Writes results/asr_check.json (per turn: both classes, both texts) and prints the agreement.
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def ensure_cuda_libs() -> None:
    """ctranslate2 needs the pip-installed cuBLAS/cuDNN on LD_LIBRARY_PATH at process start."""
    import site
    libs = []
    for sp in site.getsitepackages():
        libs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if libs and not all(l in cur for l in libs):
        os.environ["LD_LIBRARY_PATH"] = ":".join(libs + ([cur] if cur else []))
        os.execv(sys.executable, [sys.executable] + sys.argv)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "asr_check.json")
    p.add_argument("--rules", default="v1", choices=["v1", "v3"])
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    ensure_cuda_libs()
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
    from fruitbench.score import score_game

    files = sorted(f for f in args.dir.glob("*.json") if f.with_suffix(".wav").exists())
    files = sorted(random.Random(0).sample(files, min(args.n, len(files))))
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    turns, games = [], []
    for f in files:
        g = json.loads(f.read_text())
        audio, sr = sf.read(str(f.with_suffix(".wav")), dtype="float32")
        left = audio[:, 0]
        if sr != 16000:  # faster-whisper wants 16 kHz float input
            n = int(len(left) * 16000 / sr)
            left = np.interp(np.linspace(0, len(left) - 1, n), np.arange(len(left)), left).astype(np.float32)
        segs, _ = model.transcribe(left, language="en", word_timestamps=True, vad_filter=True,
                                   condition_on_previous_text=False, beam_size=5)
        words = [w for s in segs for w in (s.words or [])]
        pseudo = dict(g, moshi_tokens=[{"t": round(w.start, 3), "id": -1, "piece": "▁" + w.word.strip()}
                                       for w in words if w.word.strip()])
        st, sa = score_game(g, args.rules), score_game(pseudo, args.rules)
        games.append({"game": g["game_id"], "asr_text": " ".join(w.word.strip() for w in words),
                      "moshi_text": g["moshi_text"]})
        for rt, ra in zip(st["replies"], sa["replies"]):
            if not rt["scored"]:
                continue
            turns.append({"game": g["game_id"], "turn": rt["turn"], "user": rt["user_text"],
                          "text_cls": rt["cls"], "asr_cls": ra["cls"], "text_fruit": rt["fruit"],
                          "asr_fruit": ra["fruit"], "text": rt["text"], "asr": ra["text"]})
        print(g["game_id"], "done", flush=True)
    agree = [t["text_cls"] == t["asr_cls"] for t in turns]
    # Same class AND (for one-fruit classes) the same fruit.
    strict = [t["text_cls"] == t["asr_cls"] and t["text_fruit"] == t["asr_fruit"] for t in turns]
    conf = Counter(f"{t['text_cls']}->{t['asr_cls']}" for t in turns if t["text_cls"] != t["asr_cls"])
    per_cls = {}
    for c in sorted({t["text_cls"] for t in turns}):
        ts = [t for t in turns if t["text_cls"] == c]
        per_cls[c] = {"n": len(ts), "agree": round(sum(t["asr_cls"] == c for t in ts) / len(ts), 3)}
    # Timing-free check: fruit mentions per game (multiset), text stream vs Whisper transcript.
    from fruitbench.score import find_fruits, norm_word

    def fr(t):
        from fruitbench.score import ALIASES_BY_RULES
        return Counter(m["fruit"] for m in find_fruits(
            [{"t": 0, "text": w, "norm": norm_word(w)} for w in t.split() if norm_word(w)], ALIASES_BY_RULES[args.rules]))
    inter = tot_t = tot_a = 0
    for gg in games:
        ct, ca = fr(gg["moshi_text"]), fr(gg["asr_text"])
        inter += sum((ct & ca).values())
        tot_t += sum(ct.values())
        tot_a += sum(ca.values())
    mentions = {"fruit_mentions_text": tot_t, "fruit_mentions_asr": tot_a, "matched": inter,
                "text_mentions_heard_in_audio": round(inter / max(tot_t, 1), 3),
                "asr_mentions_in_text": round(inter / max(tot_a, 1), 3)}
    out = {"games": len(files), "fruit_mentions_game_level": mentions, "scored_turns": len(turns), "agree_class": round(sum(agree) / len(turns), 3),
           "agree_class_and_fruit": round(sum(strict) / len(turns), 3), "per_text_class": per_cls,
           "disagreements": dict(conf.most_common()), "turns": turns, "transcripts": games,
           "whisper": {"model": "large-v3", "vad_filter": True, "beam_size": 5, "language": "en"}}
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("games", "fruit_mentions_game_level", "scored_turns", "agree_class", "agree_class_and_fruit",
                                          "per_text_class", "disagreements")}, indent=1))


if __name__ == "__main__":
    main()
