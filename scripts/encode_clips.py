"""Step 1b: pre-encode every rendered clip (data/clips.jsonl) to Mimi tokens, once.

Each clip is encoded with LEAD_FRAMES of digital silence before it (so the encoder state looks like
"user was quiet", as it is in the game), and TAIL_FRAMES of silence after the last audio frame. We
keep the frames from the clip's first frame to the end of the tail: in the game those tokens are fed
starting at the clip's start frame, and steady-state silence tokens before/after.

Output (git-ignored): data/tokens/clips.pt = {"silence": (8,) long, "clips": {path: (8, n) int16},
"lead_frames", "tail_frames", "frame_sec"}.

Usage: python scripts/encode_clips.py [--gpu 0]
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEAD_FRAMES = 12   # 0.96 s
TAIL_FRAMES = 4    # 0.32 s; afterwards the game feeds the steady silence code


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
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
    fs = int(mimi.sample_rate / mimi.frame_rate)
    rows = [json.loads(l) for l in (ROOT / "data" / "clips.jsonl").read_text().splitlines()]

    # Digital silence does not encode to one constant frame: after ~1 s each codebook flips between
    # one or two near-tied values. Use the per-codebook mode over 2-8 s of silence.
    sil = mimi.encode(torch.zeros(1, 1, fs * 100, device="cuda"))[0, :, 25:].cpu()  # (8, 75)
    silence = torch.mode(sil, dim=1).values

    out = {}
    for j in range(0, len(rows), args.batch):
        group = rows[j:j + args.batch]
        pcms = [sphn.read(str(ROOT / r["path"]), sample_rate=mimi.sample_rate)[0][0] for r in group]
        nfr = [-(-len(x) // fs) for x in pcms]
        T = LEAD_FRAMES + max(nfr) + TAIL_FRAMES
        batch = np.zeros((len(group), 1, T * fs), dtype=np.float32)
        for b, x in enumerate(pcms):
            batch[b, 0, LEAD_FRAMES * fs:LEAD_FRAMES * fs + len(x)] = x
        codes = mimi.encode(torch.from_numpy(batch).cuda()).cpu()  # (B, 8, T)
        for b, r in enumerate(group):
            out[r["path"]] = codes[b, :, LEAD_FRAMES:LEAD_FRAMES + nfr[b] + TAIL_FRAMES].to(torch.int16).clone()
        print(f"{j + len(group)}/{len(rows)}", flush=True)

    dst = ROOT / "data" / "tokens" / "clips.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"silence": silence, "clips": out, "lead_frames": LEAD_FRAMES, "tail_frames": TAIL_FRAMES,
                "frame_sec": 1 / mimi.frame_rate}, dst)
    print(f"wrote {dst} ({len(out)} clips); silence code {silence.tolist()}")


if __name__ == "__main__":
    main()
