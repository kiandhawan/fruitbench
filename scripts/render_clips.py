"""Step 1: render every fruit and every instruction phrasing with each Fish voice (24 kHz mono).

Audio goes to data/audio/clips/<voice_title>/<kind>/<slug>.wav (git-ignored); the manifest
data/clips.jsonl (tracked) lists text, voice, duration and path. Uses moshi-finetune's
scripts/render_lib.py (Fish s2-pro TTS, header fix, silence trim, retries, cache) unchanged.
Clips are loudness-normalised to -21 dBFS RMS over speech (the user level of the old project).

FISH_API_KEY must be set in this process's environment only:
  FISH_API_KEY=$(cat ~/.fish_key) python scripts/render_clips.py
Cost: Fish credit before/after (account API) and UTF-8 bytes sent, appended to results/fish_cost.json.
"""

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sphn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/root/moshi-finetune/scripts")
import render_lib  # noqa: E402

sys.path.insert(0, str(ROOT))
from fruitbench.lists import PHRASINGS, make_splits  # noqa: E402

SR = 24000
TARGET_DBFS = -21.0
USD_PER_MILLION_BYTES = 15.0  # Fish TTS price used in the old project's billing check (results/LOG.md there)
CLIPS = ROOT / "data" / "audio" / "clips"
render_lib.CACHE = ROOT / "data" / "audio" / "tts_cache"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def fruit_text(f: str) -> str:
    return f[0].upper() + f[1:] + "."


def normalise(x: np.ndarray) -> np.ndarray:
    hop = SR // 100
    n = len(x) // hop
    rms = np.sqrt((x[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1))
    act = rms > rms.max() * 10 ** (-30 / 20)
    level = np.sqrt((rms[act] ** 2).mean()) if act.any() else 1e-4
    y = x * (10 ** (TARGET_DBFS / 20) / level)
    peak = np.abs(y).max()
    return (y * (0.97 / peak) if peak > 0.97 else y).astype(np.float32)


def main() -> None:
    from fishaudio import FishAudio
    voices = json.loads((ROOT / "data" / "voices.json").read_text())["voices"]
    splits = make_splits()
    fruits = sorted(f for split in splits["fruits"].values() for f in split)
    jobs = []
    for v in voices:
        for f in fruits:
            jobs.append({"kind": "fruit", "key": f, "text": fruit_text(f), "voice": v["title"], "voice_id": v["id"]})
        for who, lst in PHRASINGS.items():
            for t in lst:
                jobs.append({"kind": "phrasing", "key": t, "starter": who, "text": t, "voice": v["title"],
                             "voice_id": v["id"]})
    for j in jobs:
        j["path"] = str(Path("data/audio/clips") / j["voice"] / j["kind"] / f"{slug(j['key'])}.wav")
    todo = [j for j in jobs if not (ROOT / j["path"]).exists()]
    billed_bytes = sum(len(j["text"].encode()) for j in todo
                       if not (render_lib.CACHE / f"{render_lib.cache_key(j['text'], j['voice_id'])}.wav").exists())
    print(f"{len(jobs)} clips, {len(todo)} to render, ~{billed_bytes} new bytes "
          f"(~${billed_bytes * USD_PER_MILLION_BYTES / 1e6:.3f})", flush=True)

    client = FishAudio()
    credit_before = float(client.account.get_credits().credit)
    t0 = time.time()

    def work(j):
        x, _ = render_lib.synth(client, j["text"], j["voice_id"], align=False)
        x = normalise(x)
        out = ROOT / j["path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        sphn.write_wav(str(out), x[None], sample_rate=SR)
        return j

    done = 0
    with ThreadPoolExecutor(8) as ex:
        for _ in ex.map(work, todo):
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for j in jobs:
        x = sphn.read(str(ROOT / j["path"]), sample_rate=SR)[0][0]
        rows.append({k: j[k] for k in ("kind", "key", "text", "voice", "voice_id", "path")} |
                    ({"starter": j["starter"]} if "starter" in j else {}) | {"duration": round(len(x) / SR, 3)})
    (ROOT / "data" / "clips.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    time.sleep(5)
    credit_after = float(client.account.get_credits().credit)
    cost = {"date": time.strftime("%Y-%m-%d %H:%M"), "clips_rendered": len(todo), "bytes_sent": billed_bytes,
            "estimated_usd": round(billed_bytes * USD_PER_MILLION_BYTES / 1e6, 4),
            "credit_before": credit_before, "credit_after_5s": credit_after,
            "credit_delta_5s": round(credit_before - credit_after, 4), "wall_sec": round(time.time() - t0, 1),
            "note": "Fish bills with a delay of minutes; re-check the credit later (scripts/fish_credit.py)."}
    log = ROOT / "results" / "fish_cost.json"
    hist = json.loads(log.read_text()) if log.exists() else []
    log.write_text(json.dumps(hist + [cost], indent=1))
    print(json.dumps(cost, indent=1))


if __name__ == "__main__":
    main()
