"""Run fruit-bench games with base Moshi, sharded over GPUs (one worker process per GPU).

  # 200 test games, seed 1, on 4 GPUs
  python scripts/run_games.py --split test --n 200 --seed 1 --gpus 0 1 2 3 --out runs/base_test_s1
  # GRPO readiness: 25 train starts x 16 games
  python scripts/run_games.py --split train --n 25 --repeat 16 --seed 1 --gpus 0 1 2 3 --out runs/grpo_check
  # checkpoint B smoke test: 2 user-starts + 2 Moshi-starts games
  python scripts/run_games.py --split val --smoke --seed 1 --gpus 0 --batch 4 --out runs/smoke

Each game writes <out>/<game_id>.json (game summary) and .score.json; audio <game_id>.wav (left =
Moshi, right = user) unless --no-audio. When all workers succeed, the per-game JSONs (summaries
with Moshi's timed text tokens, and per-turn scores; no audio or tokens) are archived to
results/games/<out name>/ (or a .tar.gz if > 20 MB; see fruitbench/archive.py) for committing;
--archive-dir overrides the destination, --no-archive skips it. Sampling seed of batch j of worker w: seed*100000 + w*1000 + j.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def build_specs(args):
    from fruitbench.game import GameSpec, make_specs
    if args.smoke:
        pool = make_specs(args.split, 12, args.seed, prefix=f"smoke_{args.split}")
        user = [s for s in pool if s.starter == "user"][:2]
        moshi = [s for s in pool if s.starter == "moshi"][:2]
        return user + moshi
    specs = make_specs(args.split, args.n, args.seed)
    if args.repeat > 1:
        specs = [GameSpec(**(s.to_dict() | {"game_id": f"{s.game_id}_r{r:02d}"})) for s in specs for r in range(args.repeat)]
    return specs


def worker(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.worker_gpu)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    from fruitbench.game import GameRunner, GameSpec, save_game
    from fruitbench.score import score_game
    specs = [GameSpec(**d) for d in json.loads(Path(args.worker_specs).read_text())]
    runner = GameRunner(args.batch, decode_audio=not args.no_audio, user_audio=args.user_audio, lora=args.lora)
    out = Path(args.out)
    timing = []
    for j in range(0, len(specs), args.batch):
        chunk = specs[j:j + args.batch]
        if not args.overwrite and all((out / f"{s.game_id}.score.json").exists() for s in chunk):
            continue
        seed = args.seed * 100000 + args.worker_index * 1000 + j // args.batch
        t0 = time.perf_counter()
        results = runner.run(chunk, seed)
        for res in results:
            save_game(res, out)
            sc = score_game(res["summary"])
            (out / f"{res['summary']['game_id']}.score.json").write_text(json.dumps(sc, indent=1, ensure_ascii=False))
        s0 = results[0]["summary"]
        timing.append({"games": len(chunk), "batch": args.batch, "wall_sec": round(time.perf_counter() - t0, 2),
                       "loop_sec": s0["batch_wall_sec"], "frames": s0["batch_frames"],
                       "game_sec": round(s0["batch_frames"] * 0.08, 1), "peak_mem_gb": s0["peak_mem_gb"]})
        print(f"[gpu{args.worker_gpu}] {j + len(chunk)}/{len(specs)} {timing[-1]}", flush=True)
    (out / f"timing_worker{args.worker_index}.json").write_text(json.dumps(timing, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--repeat", type=int, default=1, help="Play each start this many times (GRPO groups).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--user-audio", default="live", choices=["live", "tokens"],
                   help="live: stream-encode the user PCM with Mimi each frame (default, exact); "
                        "tokens: splice pre-encoded clip tokens (faster, inexact).")
    p.add_argument("--archive-dir", type=Path, help="Where to archive per-game JSONs (default results/games/<out name>).")
    p.add_argument("--no-archive", action="store_true")
    p.add_argument("--lora", help="LoRA safetensors (from grpo_train.py) to evaluate instead of base Moshi")
    p.add_argument("--worker-specs")
    p.add_argument("--worker-gpu", type=int)
    p.add_argument("--worker-index", type=int)
    args = p.parse_args()
    if args.worker_specs:
        return worker(args)

    specs = build_specs(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "specs.json").write_text(json.dumps([s.to_dict() for s in specs], indent=1))
    # Contiguous shards keep GRPO groups (consecutive repeats) inside one batch.
    n_w = len(args.gpus)
    per = -(-len(specs) // n_w)
    per = -(-per // args.batch) * args.batch if len(specs) > args.batch else per
    procs = []
    t0 = time.time()
    for w, gpu in enumerate(args.gpus):
        shard = specs[w * per:(w + 1) * per]
        if not shard:
            continue
        sp = args.out / f"shard{w}.json"
        sp.write_text(json.dumps([s.to_dict() for s in shard]))
        cmd = [sys.executable, __file__, "--worker-specs", str(sp), "--worker-gpu", str(gpu), "--worker-index", str(w),
               "--seed", str(args.seed), "--batch", str(args.batch), "--out", str(args.out)]
        cmd += ["--no-audio"] if args.no_audio else []
        cmd += ["--overwrite"] if args.overwrite else []
        cmd += ["--user-audio", args.user_audio]
        cmd += ["--lora", args.lora] if args.lora else []
        procs.append(subprocess.Popen(cmd))
    codes = [pr.wait() for pr in procs]
    print(f"done in {time.time() - t0:.0f}s; worker exit codes {codes}")
    if any(codes):
        sys.exit(1)
    if not args.no_archive:
        from fruitbench.archive import archive_run
        archive_run(args.out, args.archive_dir or ROOT / "results" / "games" / args.out.name)


if __name__ == "__main__":
    main()
