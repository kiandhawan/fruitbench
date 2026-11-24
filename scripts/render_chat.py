"""Render the ordinary-conversation lines (fruitbench/chat.py) with Fish, append kind "chat" rows to
data/clips.jsonl. FISH_API_KEY only in this process: FISH_API_KEY=$(cat ~/.fish_key) python scripts/render_chat.py"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sphn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import render_clips as rc  # noqa: E402  (normalise, render_lib with the fruitbench cache)
from fruitbench.chat import CONVERSATIONS, chat_specs  # noqa: E402


def main() -> None:
    from fishaudio import FishAudio
    voices = {v["title"]: v["id"] for v in json.loads((ROOT / "data" / "voices.json").read_text())["voices"]}
    jobs = []
    for spec, conv in zip(chat_specs(), CONVERSATIONS):
        for key, text in zip(spec.script, conv):
            jobs.append({"kind": "chat", "key": key, "text": text, "voice": spec.voice, "voice_id": voices[spec.voice],
                         "path": f"data/audio/clips/{spec.voice}/chat/{key}.wav"})
    nbytes = sum(len(j["text"].encode()) for j in jobs)
    client = FishAudio()
    before = float(client.account.get_credits().credit)

    def work(j):
        x, _ = rc.render_lib.synth(client, j["text"], j["voice_id"], align=False)
        out = ROOT / j["path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        sphn.write_wav(str(out), rc.normalise(x)[None], sample_rate=rc.SR)
        return j

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(work, jobs))
    rows = [json.loads(l) for l in (ROOT / "data" / "clips.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["kind"] != "chat"]
    for j in jobs:
        x = sphn.read(str(ROOT / j["path"]), sample_rate=rc.SR)[0][0]
        rows.append({k: j[k] for k in ("kind", "key", "text", "voice", "voice_id", "path")} | {"duration": round(len(x) / rc.SR, 3)})
    (ROOT / "data" / "clips.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    time.sleep(5)
    after = float(client.account.get_credits().credit)
    cost = {"date": time.strftime("%Y-%m-%d %H:%M"), "what": "ordinary-conversation lines", "clips_rendered": len(jobs),
            "bytes_sent": nbytes, "estimated_usd": round(nbytes * rc.USD_PER_MILLION_BYTES / 1e6, 4),
            "credit_before": before, "credit_after_5s": after}
    log = ROOT / "results" / "fish_cost.json"
    log.write_text(json.dumps(json.loads(log.read_text()) + [cost], indent=1))
    print(json.dumps(cost))


if __name__ == "__main__":
    main()
