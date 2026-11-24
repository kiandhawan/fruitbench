"""Check the pre-encoded user tokens against a live streaming Mimi encode of the same user audio.

For each game in DIR: rebuild the user token stream the game fed Moshi (steady silence code +
each clip's pre-encoded tokens from its start frame), stream-encode the saved right channel frame by
frame exactly like moshi.server does (including the reset after the first frame), and report the
agreement per codebook, over all frames and over frames inside user speech.

  python scripts/check_tokens.py runs/smoke [--gpu 0]
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    import numpy as np
    import sphn
    import torch
    torch.set_grad_enabled(False)
    from moshi.models import loaders
    info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
    mimi = info.get_mimi(device="cuda")
    fs = 1920
    bank = torch.load(ROOT / "data" / "tokens" / "clips.pt")
    sil = bank["silence"].long().numpy()
    mimi.streaming_forever(1)
    rows = []
    for f in sorted(args.dir.glob("*.json")):
        g = json.loads(f.read_text())
        if not isinstance(g, dict) or "user_turns" not in g:
            continue
        user = sphn.read(str(f.with_suffix(".wav")), sample_rate=24000)[0][1]
        n = len(user) // fs
        fed = np.tile(sil[:, None], (1, n))
        speech = np.zeros(n, bool)
        for t in g["user_turns"]:
            tk = bank["clips"][t["clip"]].long().numpy()
            a = t["frame"]
            m = min(tk.shape[1], n - a)
            fed[:, a:a + m] = tk[:, :m]
            speech[a:min(n, int(np.ceil(t["speech_end"] / 0.08)))] = True
        mimi.reset_streaming()
        live = []
        for i in range(n):
            c = mimi.encode(torch.from_numpy(user[i * fs:(i + 1) * fs]).cuda()[None, None])
            if i == 0:
                mimi.reset_streaming()
            live.append(c[0, :, 0].cpu().numpy())
        live = np.stack(live, 1)
        saved = f.with_name(f.stem + ".user_codes.npy")
        if saved.exists():  # what the game actually fed Moshi (live mode: batched streaming encode)
            fed = np.load(saved).astype(np.int64)[:, :n]
            n = min(n, fed.shape[1])
            fed, live, speech = fed[:, :n], live[:, :n], speech[:n]
        eq = fed == live
        rows.append({"game": g["game_id"], "compared": "saved fed codes" if saved.exists() else "spliced tokens",
                     "frames": n, "speech_frames": int(speech.sum()),
                     "agree_all_cb0": round(float(eq[0].mean()), 3), "agree_all_mean": round(float(eq.mean()), 3),
                     "agree_speech_cb0": round(float(eq[0, speech].mean()), 3),
                     "agree_speech_mean": round(float(eq[:, speech].mean()), 3),
                     "agree_speech_per_cb": [round(float(x), 3) for x in eq[:, speech].mean(1)],
                     "agree_silence_mean": round(float(eq[:, ~speech].mean()), 3)})
        print(json.dumps(rows[-1]))
    (args.dir / "token_check.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
