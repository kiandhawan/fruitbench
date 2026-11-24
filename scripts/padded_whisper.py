"""Padded Whisper recall: does Moshi's AUDIO say the fruit its text stream says, turn by turn?

Each Moshi turn's speech segment (fruitbench/turn_audio.py) is cut out on its own, padded with 0.5 s
of silence on both sides, and transcribed by faster-whisper large-v3 with a fruit-list
initial_prompt (so short isolated fruit words are not dropped or turned into other words).
  recall:          turns whose text stream has a fruit -> share where Whisper's transcript has that fruit
  false positives: turns whose text has no fruit (filler / non_fruit) -> share where Whisper hears one
Calibration on our own TTS fruit clips gives the ceiling for clean single words. Run in .venv-asr.

  .venv-asr/bin/python scripts/padded_whisper.py --calib 100 --out results/quality/whisper_tts_calibration.json
  .venv-asr/bin/python scripts/padded_whisper.py --games runs/grpo/grpo_smoke3/val/step_020 --rules v3 \
      --out results/quality/whisper_smoke3_step020.json
"""

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PAD = 0.5


def ensure_cuda_libs() -> None:
    import site
    libs = []
    for sp in site.getsitepackages():
        libs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if libs and not all(l in cur for l in libs):
        os.environ["LD_LIBRARY_PATH"] = ":".join(libs + ([cur] if cur else []))
        os.execv(sys.executable, [sys.executable] + sys.argv)


def resample16(x: np.ndarray, sr: int) -> np.ndarray:
    n = int(len(x) * 16000 / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=Path)
    ap.add_argument("--rules", default="v3")
    ap.add_argument("--calib", type=int)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    ensure_cuda_libs()
    from faster_whisper import WhisperModel
    from fruitbench.lists import COMMON_FRUITS
    from fruitbench.score import ALIASES_BY_RULES, find_fruits, norm_word
    from fruitbench.turn_audio import SR, cut, load_stereo, turn_segments
    aliases = ALIASES_BY_RULES[a.rules]
    prompt = "Fruit names: " + ", ".join(COMMON_FRUITS) + "."
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    pad = np.zeros(int(PAD * 16000), dtype=np.float32)

    def hear(x24: np.ndarray) -> tuple[str, list[str]]:
        x = np.concatenate([pad, resample16(x24, SR), pad])
        segs, _ = model.transcribe(x, language="en", beam_size=5, vad_filter=False, initial_prompt=prompt,
                                   condition_on_previous_text=False, temperature=0.0)
        txt = " ".join(s.text.strip() for s in segs)
        ws = [{"t": 0, "text": w, "norm": norm_word(w)} for w in txt.replace("-", " ").split() if norm_word(w)]
        return txt, [m["fruit"] for m in find_fruits(ws, aliases)]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.calib:
        import soundfile as sf
        rows = [json.loads(l) for l in (ROOT / "data" / "clips.jsonl").read_text().splitlines()]
        rows = [r for r in rows if r["kind"] == "fruit" and r["key"] in aliases.values()]
        pick = random.Random(0).sample(rows, a.calib)
        per = []
        for r in pick:
            x, sr = sf.read(str(ROOT / r["path"]), dtype="float32")
            txt, fr = hear(x)
            per.append({"clip": r["path"], "voice": r["voice"], "fruit": r["key"], "asr": txt, "heard": r["key"] in fr})
        rec = sum(p["heard"] for p in per) / len(per)
        by_voice = {v: round(np.mean([p["heard"] for p in per if p["voice"] == v]), 3) for v in sorted({p["voice"] for p in per})}
        res = {"what": f"{a.calib} TTS fruit clips (seed 0), padded {PAD}s, fruit prompt", "recall": round(rec, 4),
               "recall_by_voice": by_voice, "misses": [p for p in per if not p["heard"]], "clips": per}
        a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(json.dumps({k: v for k, v in res.items() if k != "clips"}, ensure_ascii=False))
        return
    turns = []
    for f in sorted(a.games.glob("*.json")):
        if f.name.endswith(".score.json") or not f.name[-9:-5].isdigit():
            continue
        x = load_stereo(f.with_suffix(".wav"))
        for t in turn_segments(f, a.rules):
            if t["seg"] is None or not (t["fruits"] or t["cls"] in ("filler", "non_fruit")):
                continue
            txt, fr = hear(cut(x[0], t["seg"]))
            t.update(asr=txt, asr_fruits=fr, heard=bool(t["fruits"]) and all(f_ in fr for f_ in t["fruits"]),
                     heard_any=bool(set(t["fruits"]) & set(fr)))
            turns.append(t)
    with_f = [t for t in turns if t["fruits"]]
    valid = [t for t in turns if t["cls"] == "valid"]
    neg = [t for t in turns if not t["fruits"]]
    res = {"games_dir": str(a.games), "rules": a.rules, "pad_sec": PAD,
           "recall_all_fruit_turns": round(np.mean([t["heard"] for t in with_f]), 4) if with_f else None,
           "n_fruit_turns": len(with_f),
           "recall_bare_valid": round(np.mean([t["heard"] for t in valid]), 4) if valid else None,
           "n_bare_valid": len(valid),
           "recall_by_class": {c: round(np.mean([t["heard"] for t in with_f if t["cls"] == c]), 4)
                               for c in sorted({t["cls"] for t in with_f})},
           "false_positive_rate_no_fruit_turns": round(np.mean([bool(t["asr_fruits"]) for t in neg]), 4) if neg else None,
           "n_no_fruit_turns": len(neg),
           "valid_misses": [{k: t[k] for k in ("game", "turn", "text", "asr", "seg")} for t in valid if not t["heard"]][:40],
           "turns": [{k: t[k] for k in ("game", "turn", "pos", "cls", "text", "fruits", "asr", "asr_fruits", "heard", "seg")}
                     for t in turns]}
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k not in ("turns", "valid_misses")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
