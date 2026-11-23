"""Checkpoint A: load base Moshi on one GPU and stream ~8 s of input, recording what it says.

There is no rendered user audio yet, so the user channel is silence. Moshi usually greets the user
on its own ("Hey, what's up?"), which is enough to show the model loads and generates audio + text.

Usage: python scripts/check_load.py --gpu 0 --out runs/check_load
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # same workaround as moshi-finetune/scripts/run_inference.py


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("runs/check_load"))
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import numpy as np
    import sphn
    import torch
    torch.set_grad_enabled(False)
    from moshi.models import LMGen, loaders
    from moshi.run_inference import seed_all

    t0 = time.perf_counter()
    info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
    mimi = info.get_mimi(device="cuda")
    tok = info.get_text_tokenizer()
    lm = info.get_moshi(device="cuda", dtype=torch.bfloat16)
    lm_gen = LMGen(lm, **info.lm_gen_config)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s on {torch.cuda.get_device_name(0)}", flush=True)

    frame = int(mimi.sample_rate / mimi.frame_rate)
    n = int(args.seconds * mimi.frame_rate)
    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    seed_all(args.seed)
    moshi_pcm = np.zeros(n * frame, dtype=np.float32)
    words = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i in range(n):
        codes = mimi.encode(torch.zeros(1, 1, frame, device="cuda"))
        if i == 0:
            mimi.reset_streaming()
        tokens = lm_gen.step(codes)
        if tokens is None:
            continue
        slot = i + 1
        if slot < n:
            moshi_pcm[slot * frame:(slot + 1) * frame] = mimi.decode(tokens[:, 1:])[0, 0].float().cpu().numpy()
            tid = int(tokens[0, 0, 0])
            if tid not in (0, 3):
                words.append({"t": round(slot / mimi.frame_rate, 2), "piece": tok.id_to_piece(tid)})
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    args.out.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(str(args.out / "check_load.wav"), np.stack([moshi_pcm, np.zeros_like(moshi_pcm)]),
                   sample_rate=mimi.sample_rate)
    text = "".join(w["piece"] for w in words).replace("▁", " ").strip()
    rms = float(np.sqrt((moshi_pcm ** 2).mean()))
    summary = {"seconds": args.seconds, "wall_sec": round(wall, 2), "realtime_factor": round(wall / args.seconds, 3),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "moshi_rms": round(rms, 4),
               "text": text, "tokens": words}
    (args.out / "check_load.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "tokens"}, indent=1, ensure_ascii=False))
    for w in words:
        print(f"  {w['t']:5.2f}s {w['piece']}")


if __name__ == "__main__":
    main()
