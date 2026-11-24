"""GRPO for Moshi on fruit-bench (LoRA on base moshiko, DDP over the GPUs).

  torchrun --nproc_per_node 4 scripts/grpo_train.py --config configs/grpo_smoke.yaml

Each step: every rank plays one start (group) x group_size games of the TRAIN split with the current
policy (same game loop and scorer as the base results; 6 user turns), scores them, computes per-turn
GRPO advantages within the group, and then all ranks do one pass over their games (split into
`minibatches` optimizer steps) with a PPO-clipped objective + k3 KL to the reference (LoRA disabled).
The advantage of Moshi turn k (reply to user line k) is applied to Moshi's text and audio tokens in
the frames from the end of user line k to the end of user line k+1; no loss elsewhere and never on
user tokens (LMModel only predicts Moshi's streams).

Outputs in git: results/grpo/<run>/{config.yaml, metrics.jsonl (one row per training step),
val.jsonl, val_step_XXX/ (validation per-game JSONs), val_step_XXX_asr.json}, results/plots/grpo_*.png.
On disk only (runs/, not in git): ckpt/step_XXX/lora.safetensors, rollouts/step_XXX.tar.gz (that
step's training games), val/step_XXX/ (validation audio).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/root/moshi-finetune")  # reuse finetune/mixed_precision.py (read-only)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from fruitbench.game import FRAME_SEC, GameRunner, GameSpec, make_specs, save_game  # noqa: E402
from fruitbench.chat import chat_fruit_rate, chat_specs  # noqa: E402
from fruitbench.credit import credit_windows, talkover_flags  # noqa: E402
from fruitbench.rewards import counted_overlap, game_rewards  # noqa: E402
from fruitbench.score import CLASSES, score_game  # noqa: E402

ASR_PY = ROOT / ".venv-asr" / "bin" / "python"
RULES = "v1"


# ----------------------------------------------------------------------------- setup
def log0(*a):
    if dist.get_rank() == 0:
        print(*a, flush=True)


def load_policy(cfg):
    from moshi.models import loaders
    from moshi.modules.lora import LoRALinear
    from finetune.mixed_precision import prepare_mixed_precision
    m = cfg["model"]
    info = loaders.CheckpointInfo.from_hf_repo(m["hf_repo"])
    dev = f"cuda:{torch.cuda.current_device()}"
    mimi = info.get_mimi(device=dev)
    tok = info.get_text_tokenizer()
    torch.manual_seed(cfg["seed"])  # identical LoRA init on every rank (DDP also broadcasts rank 0's)
    lm = info.get_moshi(device=dev, dtype=torch.bfloat16, lm_kwargs_overrides={
        "gradient_checkpointing": m["gradient_checkpointing"], "lora": True,
        "lora_rank": m["lora_rank"], "lora_scaling": m["lora_scaling"]})
    n_lora = 0
    for mod in lm.modules():
        if isinstance(mod, LoRALinear):
            torch.nn.init.zeros_(mod.lora_B.weight)  # fresh LoRA: the policy starts exactly at base Moshi
            n_lora += 1
    for name, p in lm.named_parameters():
        p.requires_grad = "lora_" in name
    prepare_mixed_precision(lm.parameters(), torch.bfloat16, torch.float32)
    n_train = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in lm.parameters())
    import inspect
    from moshi.models import LMGen
    sig = inspect.signature(LMGen.__init__).parameters
    gen_cfg = {k: sig[k].default for k in ("use_sampling", "temp", "temp_text", "top_k", "top_k_text")}
    gen_cfg.update(info.lm_gen_config or {})
    for k in ("temp", "temp_text"):
        if cfg["rollout"].get(k) is not None:
            gen_cfg[k] = cfg["rollout"][k]
    return mimi, tok, lm, gen_cfg, {"lora_layers": n_lora, "trainable": n_train, "total": n_all}


def set_checkpointing(lm, on: bool):
    from moshi.modules.transformer import StreamingTransformer
    for mod in lm.modules():
        if isinstance(mod, StreamingTransformer):
            mod.checkpointing = on


class lora_disabled:
    """Reference policy = same model with every LoRA branch scaled to 0."""

    def __init__(self, lm):
        from moshi.modules.lora import LoRALinear
        self.mods = [m for m in lm.modules() if isinstance(m, LoRALinear)]

    def __enter__(self):
        self.saved = [m.scaling for m in self.mods]
        for m in self.mods:
            m.scaling = 0.0

    def __exit__(self, *exc):
        for m, s in zip(self.mods, self.saved):
            m.scaling = s


# ----------------------------------------------------------------------------- rollouts
def turn_windows(summary: dict, score: dict, T: int) -> list[tuple[int, int, int]]:
    """(scored turn index j, first aligned frame, end frame) for each scored Moshi turn.
    Turn k (reply to user line k) covers game time [speech_end_k, speech_end_{k+1}); Moshi's aligned
    frame t is heard at game time (t + 2) * FRAME_SEC."""
    ut = summary["user_turns"]
    out, j = [], 0
    for r in score["replies"]:
        if not r["scored"]:
            continue
        k = r["turn"]
        a = ut[k]["speech_end"]
        b = ut[k + 1]["speech_end"] if k + 1 < len(ut) else summary["end_time"] + 1.0
        fa = max(0, math.ceil(a / FRAME_SEC - 2 - 1e-6))
        fb = min(T, math.ceil(b / FRAME_SEC - 2 - 1e-6))
        out.append((j, fa, fb))
        j += 1
    return out


def check_alignment(summary: dict, codes: np.ndarray) -> int:
    """Moshi text tokens in the game JSON must sit at aligned frame round(t / FRAME_SEC) - 2."""
    bad = 0
    for tk in summary["moshi_tokens"]:
        i = int(round(tk["t"] / FRAME_SEC)) - 2
        if 0 <= i < codes.shape[1] and int(codes[0, i]) != tk["id"]:
            bad += 1
    return bad


def group_advantages(rewards: list[list[float]], eps: float, std_floor: float | None = None) -> tuple[list[list[float]], dict]:
    """Per turn index j: (r - mean_j) / (std_j + eps), or / max(std_j, std_floor) when std_floor is set."""
    L = max(len(r) for r in rewards)
    adv = [[0.0] * len(r) for r in rewards]
    vary_pos = 0
    for j in range(L):
        vals = [(g, r[j]) for g, r in enumerate(rewards) if j < len(r)]
        x = np.array([v for _, v in vals])
        mu, sd = x.mean(), x.std()
        vary_pos += sd > 0
        for g, v in vals:
            adv[g][j] = float((v - mu) / (max(sd, std_floor) if std_floor else sd + eps))
    tot = [sum(r) for r in rewards]
    return adv, {"group_spread": float(np.std(tot)), "group_varies": bool(np.std(tot) > 0),
                 "turn_pos_vary": vary_pos / L}


# ----------------------------------------------------------------------------- log-probs
def token_logps(model, codes: torch.Tensor, temps: tuple[float, float]):
    """Teacher-forced log-probs of Moshi's text (B,1,T) and audio (B,8,T) tokens + validity masks."""
    out = model(codes=codes)
    tt, ta = temps
    txt_logits = torch.nan_to_num(out.text_logits.float(), nan=0.0) / tt
    aud_logits = torch.nan_to_num(out.logits.float(), nan=0.0) / ta
    lm = model.module if hasattr(model, "module") else model
    tgt_t = codes[:, :1].clamp(min=0)
    tgt_a = codes[:, lm.audio_offset:lm.audio_offset + lm.dep_q].clamp(min=0)
    lp_t = torch.log_softmax(txt_logits, -1).gather(-1, tgt_t[..., None])[..., 0]
    lp_a = torch.log_softmax(aud_logits, -1).gather(-1, tgt_a[..., None])[..., 0]
    return lp_t, lp_a, out.text_mask, out.mask, txt_logits, aud_logits


def batch_codes(items: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    T = max(it["codes"].shape[1] for it in items)
    codes = torch.full((len(items), items[0]["codes"].shape[0], T), -1, dtype=torch.long)
    adv = torch.full((len(items), T), float("nan"))
    for i, it in enumerate(items):
        t = it["codes"].shape[1]
        codes[i, :, :t] = torch.from_numpy(it["codes"].astype(np.int64))
        adv[i, :t] = torch.from_numpy(it["frame_adv"])
    return codes.cuda(), adv.cuda()


def tar_step(d: Path) -> Path:
    import tarfile
    out = d.with_name(d.name + ".tar.gz")
    with tarfile.open(out, "w:gz") as tar:
        tar.add(d, arcname=d.name)
    shutil.rmtree(d)
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--steps", type=int, help="override config steps")
    ap.add_argument("--no-val", action="store_true", help="skip validation (quick trials)")
    ap.add_argument("--run-name", help="override run_name")
    ap.add_argument("--resume", type=Path, help="checkpoint dir runs/grpo/<run>/ckpt/step_XXX: LoRA + optimizer + step")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.steps:
        cfg["steps"] = args.steps
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.no_val:
        cfg["val"]["every"] = 10 ** 9
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local}"))
    rank, world = dist.get_rank(), dist.get_world_size()
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    global RULES
    RULES = cfg["reward"].get("scorer", "v1")  # v3: bare-fruit 'valid', 'sentence' class, common-sense fruit list
    run = cfg["run_name"]
    res_dir = ROOT / "results" / "grpo" / run
    run_dir = ROOT / "runs" / "grpo" / run
    if rank == 0:
        res_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        if not args.resume:  # a resumed run appends to its logs
            for f in ("metrics.jsonl", "val.jsonl"):
                (res_dir / f).unlink(missing_ok=True)
    dist.barrier()

    from finetune.mixed_precision import downcast_mixed_precision, upcast_mixed_precision
    from fruitbench.archive import archive_run
    mimi, tok, lm, gen_cfg, nparams = load_policy(cfg)
    log0(f"[setup] world={world} LoRA layers={nparams['lora_layers']} trainable={nparams['trainable']:,} "
         f"of {nparams['total']:,} params; sampling {gen_cfg}")
    params = [p for p in lm.parameters() if p.requires_grad]
    u = cfg["update"]
    opt = torch.optim.AdamW(params, lr=u["lr"], betas=tuple(u["betas"]), weight_decay=u["weight_decay"])
    start_step = 1
    if args.resume:
        from safetensors.torch import load_file
        sd = load_file(str(args.resume / "lora.safetensors"), device=f"cuda:{local}")
        with torch.no_grad():
            for name, p in lm.named_parameters():
                if p.requires_grad:
                    p._mp_param.copy_(sd[name])
                    p.data.copy_(sd[name].to(p.dtype))
        # Optimizer.load_state_dict casts state to each param's dtype: load it while params point at the
        # fp32 master copies, so Adam moments stay fp32.
        saved = [p.data for p in params]
        for p in params:
            p.data = p._mp_param
        opt.load_state_dict(torch.load(args.resume / "optimizer.pt", map_location=f"cuda:{local}"))
        for p, d in zip(params, saved):
            p.data = d
        start_step = int(args.resume.name.split("_")[-1]) + 1
        log0(f"[resume] loaded {args.resume} -> continuing at step {start_step}")
    model = torch.nn.parallel.DistributedDataParallel(lm, device_ids=[local], find_unused_parameters=False)
    runner = GameRunner(cfg["rollout"]["group_size"], decode_audio=True, models=(mimi, tok, lm, gen_cfg),
                        record_codes=True)
    temps = (gen_cfg["temp_text"], gen_cfg["temp"]) if u["logprob_temperature"] else (1.0, 1.0)
    ro = cfg["rollout"]
    assert ro["starts_per_step"] == world, "one start (group) per rank"
    pool = make_specs(ro["split"], ro["starts_per_step"] * cfg["steps"], cfg["seed"], prefix="grpo")

    # ------------------------------------------------------------------ validation
    def validate(step: int) -> None:
        v = cfg["val"]
        lm.eval()
        set_checkpointing(lm, False)
        specs = make_specs("val", v["n_games"], v["seed"])
        per = math.ceil(len(specs) / world)
        mine = [dataclasses.replace(s, n_user_fruits=v["n_user_fruits"]) for s in specs[rank * per:(rank + 1) * per]]
        vdir = run_dir / "val" / f"step_{step:03d}"
        t0 = time.time()
        rows = []
        if mine:
            results = runner.run(mine, v["seed"] * 1000 + rank)  # same specs & sampling seeds every eval
            for res in results:
                sc = score_game(res["summary"], RULES)
                res.pop("codes", None)
                save_game(res, vdir)
                (vdir / f"{res['summary']['game_id']}.score.json").write_text(json.dumps(sc, ensure_ascii=False))
                sr = [r for r in sc["replies"] if r["scored"]]
                fr = [r for r in sc["replies"] if r["kind"] == "fruit"]  # user fruit turns 1..12 in order
                rows.append({"talkover": sum(talkover_flags(res["summary"], sc)),
                             "counts": sc["counts"], "rewards": game_rewards(sc, cfg["reward"]),
                             "lat": [r["latency"] for r in sr if r["cls"] == "valid"],
                             "lat_all": [r["latency"] for r in sr if r["latency"] is not None],
                             "overlaps": sc["overlaps"], "n": sc["n_scored"],
                             "counted_overlaps": sum(counted_overlap(r, cfg["reward"]) for r in sr),
                             "echo": sum(r["echo"] for r in sr), "words": [len(r["word_t"]) for r in sr],
                             "caps": sum(1 for e in res["summary"]["events"] if "turn cap" in e.get("why", "")),
                             "pos": [(i + 1, r["cls"], bool(r["overlap"])) for i, r in enumerate(fr)]})
        chat_games = []
        if v.get("chat", False):  # (8) ordinary conversations: does Moshi still talk normally?
            cs = chat_specs()
            cper = math.ceil(len(cs) / world)
            cmine = cs[rank * cper:(rank + 1) * cper]
            if cmine:
                cdir = run_dir / "val" / f"step_{step:03d}_chat"
                for res in runner.run(cmine, v["seed"] * 1000 + 500 + rank):
                    res.pop("codes", None)
                    save_game(res, cdir)
                    chat_games.append(res["summary"])
        allchat = [None] * world
        dist.all_gather_object(allchat, chat_games)
        allrows = [None] * world
        dist.all_gather_object(allrows, rows)
        if rank == 0:
            rows = [r for rr in allrows for r in rr]
            n = sum(r["n"] for r in rows)
            cnt = {c: sum(r["counts"][c] for r in rows) for c in CLASSES}
            lat = [x for r in rows for x in r["lat"]]
            rec = {"step": step, "games": len(rows), "turns": n,
                   "mean_reward": round(float(np.mean([x for r in rows for x in r["rewards"]])), 4),
                   **{f"{c}_rate": round(cnt[c] / n, 4) for c in CLASSES},
                   "echo_rate": round(sum(r["echo"] for r in rows) / n, 4),
                   "overlap_rate": round(sum(r["overlaps"] for r in rows) / n, 4),
                   "latency_median": round(float(np.median(lat)), 3) if lat else None,
                   "latency_mean": round(float(np.mean(lat)), 3) if lat else None,
                   "counted_overlap_rate": round(sum(r["counted_overlaps"] for r in rows) / n, 4),
                   "words_per_turn_median": float(np.median([x for r in rows for x in r["words"]])),
                   "words_per_turn_mean": round(float(np.mean([x for r in rows for x in r["words"]])), 2),
                   "turn_cap_per_game": round(sum(r["caps"] for r in rows) / len(rows), 3),
                   "games_with_turn_cap": round(sum(r["caps"] > 0 for r in rows) / len(rows), 3),
                   "talkover_audio_rate": round(sum(r["talkover"] for r in rows) / n, 4),
                   "gen_sec": round(time.time() - t0, 1)}
            if v.get("chat", False):
                rec["chat"] = chat_fruit_rate([gm for cc in allchat for gm in cc])
                archive_run(run_dir / "val" / f"step_{step:03d}_chat", res_dir / f"val_step_{step:03d}_chat")
            lat_all = [x for r in rows for x in r["lat_all"]]
            rec["latency_dist"] = {
                "valid_n": len(lat), "valid_before_user_ends": round(float(np.mean([x < 0 for x in lat])), 4) if lat else None,
                "valid_p10_p50_p90": [round(float(q), 3) for q in np.percentile(lat, [10, 50, 90])] if lat else None,
                "any_fruit_n": len(lat_all),
                "any_fruit_before_user_ends": round(float(np.mean([x < 0 for x in lat_all])), 4) if lat_all else None,
                "any_fruit_p10_p50_p90": [round(float(q), 3) for q in np.percentile(lat_all, [10, 50, 90])] if lat_all else None}
            for name, lo, hi in (("turns_1_6", 1, 6), ("turns_7_12", 7, 12)):
                ps = [(c, o) for r in rows for (i, c, o) in r["pos"] if lo <= i <= hi]
                rec[name] = {"n": len(ps), "valid": round(sum(c == "valid" for c, _ in ps) / len(ps), 4),
                             "sentence": round(sum(c == "sentence" for c, _ in ps) / len(ps), 4),
                             "repeat": round(sum(c == "repeat" for c, _ in ps) / len(ps), 4),
                             "overlap": round(sum(o for _, o in ps) / len(ps), 4)}
            archive_run(vdir, res_dir / f"val_step_{step:03d}")
            asr_out = res_dir / f"val_step_{step:03d}_asr.json"
            try:
                subprocess.run([str(ASR_PY), str(ROOT / "scripts" / "asr_check.py"), str(vdir), "--n",
                                str(v["whisper_games"]), "--gpu", str(local), "--out", str(asr_out), "--rules", RULES],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                a = json.loads(asr_out.read_text())
                agree = a["fruit_mentions_game_level"]["text_mentions_heard_in_audio"]
                rec.update(asr_fruit_agreement=agree, asr_class_agreement=a["agree_class"],
                           asr_flag=agree < v["whisper_min_agreement"])
            except Exception as e:  # noqa: BLE001
                rec.update(asr_error=str(e)[:200])
            if v.get("padded_whisper", False):
                # padded Whisper recall per turn (.venv-asr, GPU): is the text stream's fruit in the audio?
                qd = res_dir / f"val_step_{step:03d}_quality"
                qd.mkdir(exist_ok=True)
                try:
                    subprocess.run([str(ASR_PY), str(ROOT / "scripts" / "padded_whisper.py"), "--games", str(vdir),
                                    "--rules", RULES, "--gpu", str(local), "--out", str(qd / "padded_whisper.json")],
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    w = json.loads((qd / "padded_whisper.json").read_text())
                    rec["padded_whisper"] = {k: w[k] for k in ("recall_all_fruit_turns", "recall_bare_valid", "n_bare_valid",
                                                               "recall_by_class", "false_positive_rate_no_fruit_turns")}
                except Exception as e:  # noqa: BLE001
                    rec["padded_whisper_error"] = str(e)[:200]
            with open(res_dir / "val.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            log0(f"[val step {step}] " + json.dumps(rec))
            if rec.get("asr_flag"):
                log0(f"[val step {step}] FLAG: Whisper heard only {rec['asr_fruit_agreement']:.0%} of the text "
                     f"stream's fruits (< {v['whisper_min_agreement']:.0%}): possible reward hacking")
            if "chat" in rec:
                log0(f"[val step {step}] ordinary conversations: {rec['chat']['with_fruit']}/{rec['chat']['replies']} "
                     f"Moshi replies mention a fruit ({rec['chat']['rate']:.1%})")
        dist.barrier()

    def save_ckpt(step: int) -> None:
        if rank == 0:
            from safetensors.torch import save_file
            d = run_dir / "ckpt" / f"step_{step:03d}"
            d.mkdir(parents=True, exist_ok=True)
            sd = {n: p._mp_param.detach().clone().contiguous().cpu() for n, p in lm.named_parameters() if p.requires_grad}
            save_file(sd, str(d / "lora.safetensors"))
            torch.save(opt.state_dict(), d / "optimizer.pt")
            (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            log0(f"[ckpt] saved {d}")
        dist.barrier()

    def val_history() -> list[dict]:
        f = res_dir / "val.jsonl"
        return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []

    def check_stop_after_val(step: int) -> str | None:
        sc_ = cfg.get("stop", {})
        h = [x for x in val_history() if x["step"] <= step]
        last = h[-1]
        if step == 20 and "val_valid_min_at_step_20" in sc_ and last["valid_rate"] < sc_["val_valid_min_at_step_20"]:
            return f"validation bare valid {last['valid_rate']:.1%} at step 20 < {sc_['val_valid_min_at_step_20']:.1%}"
        if step == 30 and "val_silent_max_at_step_30" in sc_ and last["silent_rate"] > sc_["val_silent_max_at_step_30"]:
            return f"validation silence {last['silent_rate']:.1%} at step 30 > {sc_['val_silent_max_at_step_30']:.0%}"
        n = sc_.get("val_valid_drops_in_a_row")
        if n and len(h) >= n + 1:
            v_ = [x["valid_rate"] for x in h[-(n + 1):]]
            if all(v_[i + 1] < v_[i] for i in range(n)):
                return f"validation bare valid dropped {n} validations in a row: {' -> '.join(f'{x:.1%}' for x in v_)}"
        if "chat_fruit_max" in sc_ and "chat" in last and last["chat"]["rate"] > sc_["chat_fruit_max"]:
            return f"ordinary conversations: {last['chat']['rate']:.1%} of replies mention a fruit (> {sc_['chat_fruit_max']:.0%})"
        return None

    upload_proc = [None]

    def after_validation(step: int) -> None:
        """rank 0: best checkpoint, async upload to the draft release, commit + push the run's results."""
        h = val_history()
        bm = cfg["val"].get("best_metric", "mean_reward")
        cands = [x for x in h if x["step"] > 0 and (run_dir / "ckpt" / f"step_{x['step']:03d}" / "lora.safetensors").exists()]
        if cands:
            best = max(cands, key=lambda x: (x[bm], x["valid_rate"]))
            (res_dir / "best.json").write_text(json.dumps({"best_metric": bm, "step": best["step"], bm: best[bm],
                                                           "valid_rate": best["valid_rate"]}, indent=1))
            up = cfg.get("upload", {})
            if up.get("every_validation") and up.get("release"):
                if upload_proc[0] is not None:
                    upload_proc[0].wait()
                ud = run_dir / "upload"
                shutil.rmtree(ud, ignore_errors=True)
                ud.mkdir(parents=True)
                os.link(run_dir / "ckpt" / f"step_{step:03d}" / "lora.safetensors", ud / "lora_latest.safetensors")
                os.link(run_dir / "ckpt" / f"step_{best['step']:03d}" / "lora.safetensors", ud / "lora_best.safetensors")
                (ud / "checkpoints.json").write_text(json.dumps(
                    {"run": run, "latest_step": step, "best_step": best["step"], "best_metric": bm,
                     "best_value": best[bm], "best_valid_rate": best["valid_rate"],
                     "format": "fp32 LoRA (rank 64) safetensors for kyutai/moshiko-pytorch-bf16"}, indent=1))
                upload_proc[0] = subprocess.Popen(
                    ["gh", "release", "upload", up["release"], str(ud / "lora_latest.safetensors"),
                     str(ud / "lora_best.safetensors"), str(ud / "checkpoints.json"), "--clobber"],
                    cwd=ROOT, stdout=open(run_dir / "upload.log", "a"), stderr=subprocess.STDOUT)
        if cfg.get("commit_after_validation"):
            last = h[-1]
            paths = [str(res_dir.relative_to(ROOT)), "results/plots"]
            subprocess.run(["git", "add", *paths], cwd=ROOT, check=False)
            staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, capture_output=True,
                                    text=True).stdout.split()
            bad = [f for f in staged if f.endswith((".wav", ".mp3", ".npy", ".pt", ".safetensors", ".tar.gz"))
                   or "key" in f.lower() or ".env" in f]
            if bad:
                subprocess.run(["git", "reset", "-q", *bad], cwd=ROOT)
                log0(f"[commit] refused to commit {bad}")
            msg = (f"{run}: validation step {step} (bare valid {last['valid_rate']:.1%}, mean reward "
                   f"{last['mean_reward']:+.3f})\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
            r = subprocess.run(["git", "-c", "user.name=kiandhawan", "-c", "user.email=kiandhawan@gmail.com",
                                "commit", "-q", "-m", msg], cwd=ROOT, capture_output=True, text=True)
            if r.returncode == 0:
                subprocess.run(["git", "push", "-q"], cwd=ROOT, capture_output=True)
                log0(f"[commit] pushed validation step {step}")

    def stop_all(reason: str | None) -> bool:
        flag = [reason]
        dist.broadcast_object_list(flag, src=0)
        if flag[0]:
            log0(f"[STOP] {flag[0]}")
        return bool(flag[0])

    if not args.no_val and start_step == 1:
        validate(0)
        if rank == 0:
            after_validation(0)
    stopped = False
    for step in range(start_step, cfg["steps"] + 1):
        t_step = time.time()
        torch.cuda.reset_peak_memory_stats()
        # ---------------- rollouts (this rank's group)
        lm.eval()
        set_checkpointing(lm, False)
        base = pool[(step - 1) * world + rank]
        specs = [dataclasses.replace(base, game_id=f"s{step:03d}_{base.game_id}_r{j:02d}", n_user_fruits=ro["n_user_fruits"])
                 for j in range(ro["group_size"])]
        t0 = time.time()
        results = runner.run(specs, cfg["seed"] * 1_000_000 + step * 100 + rank)
        gen_sec = time.time() - t0
        scores = [score_game(r["summary"], RULES) for r in results]
        rewards = [game_rewards(s, cfg["reward"]) for s in scores]
        adv, gstats = group_advantages(rewards, u["adv_eps"], u.get("adv_std_floor"))
        win_fn = credit_windows if u.get("credit_windows") == "v2" else turn_windows
        # (6) (group, turn position) slots with at least one bare valid fruit among the 16 games
        cls_by_pos = [[r["cls"] for r in sc["replies"] if r["scored"]] for sc in scores]
        L = max(len(c) for c in cls_by_pos)
        slots_valid = sum(any(j < len(c) and c[j] == "valid" for c in cls_by_pos) for j in range(L))
        items, misaligned = [], 0
        for res, sc, a in zip(results, scores, adv):
            codes = res["codes"]
            misaligned += check_alignment(res["summary"], codes)
            fa = np.full(codes.shape[1], np.nan, dtype=np.float32)
            for j, lo, hi in win_fn(res["summary"], sc, codes.shape[1]):
                fa[lo:hi] = a[j]
            items.append({"codes": codes, "frame_adv": fa})
        save_audio = ro["save_audio_every"] and step % ro["save_audio_every"] == 0
        rdir = run_dir / "rollouts" / f"step_{step:03d}"
        for res, sc, rw, a in zip(results, scores, rewards, adv):
            res.pop("codes", None)
            res.pop("user_codes", None)
            if not save_audio:
                res["audio"] = None
            save_game(res, rdir)
            sc["grpo_rewards"], sc["grpo_advantages"] = rw, [round(x, 4) for x in a]
            (rdir / f"{res['summary']['game_id']}.score.json").write_text(json.dumps(sc, ensure_ascii=False))

        # ---------------- old / reference log-probs (no grad)
        t1 = time.time()
        set_checkpointing(lm, cfg["model"]["gradient_checkpointing"])
        lm.train()
        mb = u["micro_batch"]
        with torch.no_grad():
            for i in range(0, len(items), mb):
                chunk = items[i:i + mb]
                codes, _ = batch_codes(chunk)
                lp_t, lp_a, *_ = token_logps(lm, codes, temps)
                with lora_disabled(lm):
                    rf_t, rf_a, _, _, _, rf_al = token_logps(lm, codes, temps)
                rf_full = torch.log_softmax(rf_al, -1).to(torch.bfloat16) if u.get("kl_audio_exact") else None
                for k, it in enumerate(chunk):
                    t_own = it["codes"].shape[1]
                    it["old_t"], it["old_a"] = lp_t[k][:, :t_own], lp_a[k][:, :t_own]
                    it["ref_t"], it["ref_a"] = rf_t[k][:, :t_own], rf_a[k][:, :t_own]
                    if rf_full is not None:
                        it["ref_a_full"] = rf_full[k][:, :t_own]  # (8, T, 2048) reference log-probs
                del rf_full
        # sanity: sampled tokens must be inside the sampler's top-k under teacher forcing (step 1 only)
        topk_ok = None
        if step == 1:
            with torch.no_grad():
                codes, advt = batch_codes(items[:2])
                _, _, tm, am, tl, al = token_logps(lm, codes, temps)
                ttgt = codes[:, :1].clamp(min=0)
                atgt = codes[:, lm.audio_offset:lm.audio_offset + lm.dep_q].clamp(min=0)
                rk_t = (tl > tl.gather(-1, ttgt[..., None])).sum(-1)
                rk_a = (al > al.gather(-1, atgt[..., None])).sum(-1)
                topk_ok = {"text_in_top_k": round(float((rk_t[tm] < gen_cfg["top_k_text"]).float().mean()), 4),
                           "audio_in_top_k": round(float((rk_a[am] < gen_cfg["top_k"]).float().mean()), 4)}
        logp_sec = time.time() - t1

        # ---------------- update: one pass, `minibatches` optimizer steps
        t2 = time.time()
        order = np.random.default_rng(step * 7919 + rank).permutation(len(items))
        n_mb = u["minibatches"]
        per_mb = math.ceil(len(items) / n_mb)
        st = {"kl_t": 0.0, "kl_a": 0.0, "n_t": 0, "n_a": 0, "clip_t": 0.0, "clip_a": 0.0, "loss": 0.0, "gnorm": []}
        for m in range(n_mb):
            idx = order[m * per_mb:(m + 1) * per_mb]
            mb_items = [items[i] for i in idx]
            n_t = sum(int(np.isfinite(it["frame_adv"]).sum()) for it in mb_items)
            n_a = n_t * lm.dep_q
            micro = [mb_items[i:i + mb] for i in range(0, len(mb_items), mb)]
            for k, ch in enumerate(micro):
                codes, advt = batch_codes(ch)
                Tm = codes.shape[-1]
                old_t = torch.stack([F.pad(it["old_t"], (0, Tm - it["old_t"].shape[-1])) for it in ch])
                old_a = torch.stack([F.pad(it["old_a"], (0, Tm - it["old_a"].shape[-1])) for it in ch])
                ref_t = torch.stack([F.pad(it["ref_t"], (0, Tm - it["ref_t"].shape[-1])) for it in ch])
                ref_a = torch.stack([F.pad(it["ref_a"], (0, Tm - it["ref_a"].shape[-1])) for it in ch])
                ref_full = (torch.stack([F.pad(it["ref_a_full"], (0, 0, 0, Tm - it["ref_a_full"].shape[1]))
                                         for it in ch]).float() if u.get("kl_audio_exact") else None)
                sync = k == len(micro) - 1
                ctx = model.no_sync() if not sync else torch.enable_grad()
                with ctx:
                    lp_t, lp_a, tm, am, _, al = token_logps(model, codes, temps)
                    fmask = torch.isfinite(advt)
                    A = torch.nan_to_num(advt, nan=0.0)
                    mt = tm & fmask[:, None]
                    ma = am & fmask[:, None]

                    def part(lp, old, ref, mask, pg_w, beta, kl_exact=None):
                        ratio = torch.exp(lp - old)
                        Ae = A[:, None].expand_as(lp)
                        pg = -torch.minimum(ratio * Ae, ratio.clamp(1 - u["clip"], 1 + u["clip"]) * Ae)
                        if kl_exact is None:  # k3 estimate on the sampled token
                            d = ref - lp
                            kl = torch.exp(d) - d - 1
                        else:                 # exact KL(policy || reference) over the full vocabulary
                            kl = kl_exact
                        # the KL term is added whatever pg_w is: with audio_weight = 0 the audio tokens
                        # still get beta_audio * KL (and no policy gradient)
                        tok = (pg_w * pg + beta * kl) * mask
                        clipped = ((ratio - 1).abs() > u["clip"]) & mask
                        return tok.sum(), (kl * mask).sum().detach(), clipped.sum().detach()

                    # text: text_weight * PG + kl_beta_text * KL; audio: audio_weight * PG + kl_beta_audio * KL
                    # (configs without kl_beta_text/audio: kl_beta for both, i.e. the smoke 1-3 loss)
                    bt = u.get("kl_beta_text", u["kl_beta"])
                    ba = u.get("kl_beta_audio", u["kl_beta"])
                    lt, klt, ct = part(lp_t, old_t, ref_t, mt, u["text_weight"], bt)
                    kl_ex = None
                    if ref_full is not None:
                        lpf = torch.log_softmax(al, -1)                       # (B, 8, T, 2048) policy
                        kl_ex = (lpf.exp() * (lpf - ref_full)).sum(-1)        # (B, 8, T) exact KL per token
                    la, kla, ca = part(lp_a, old_a, ref_a, ma, u["audio_weight"], ba, kl_ex)
                    loss = lt / max(n_t, 1) + la / max(n_a, 1)
                    loss.backward()
                st["kl_t"] += float(klt)
                st["kl_a"] += float(kla)
                st["clip_t"] += float(ct)
                st["clip_a"] += float(ca)
                st["n_t"] += int(mt.sum())
                st["n_a"] += int(ma.sum())
                st["loss"] += float(loss.detach())
            upcast_mixed_precision(params, torch.float32)
            gn = torch.nn.utils.clip_grad_norm_(params, u["grad_clip"])
            opt.step()
            downcast_mixed_precision(params, torch.bfloat16)
            opt.zero_grad(set_to_none=True)
            st["gnorm"].append(float(gn))
        train_sec = time.time() - t2

        # ---------------- metrics (all ranks -> rank 0)
        reps = [r for sc in scores for r in sc["replies"] if r["scored"]]
        caps = sum(1 for r in results for e in r["summary"]["events"] if "turn cap" in e.get("why", ""))
        words = [len(r["word_t"]) for r in reps]
        talk = sum(sum(talkover_flags(r["summary"], sc)) for r, sc in zip(results, scores))
        mine = {"n": len(reps), "cnt": {c: sum(r["cls"] == c for r in reps) for c in CLASSES},
                "talkover": talk, "slots_valid": slots_valid, "slots": L,
                "echo": sum(r["echo"] for r in reps), "overlap": sum(r["overlap"] for r in reps),
                "counted_overlap": sum(counted_overlap(r, cfg["reward"]) for r in reps),
                "lat": [r["latency"] for r in reps if r["cls"] == "valid"], "rew": [x for rw in rewards for x in rw],
                "fruits_per_turn": float(np.mean([len(r["fruits"]) for r in reps])), "words": words,
                "caps": caps, "game_sec": float(np.mean([r["summary"]["end_time"] for r in results])),
                "misaligned": misaligned, **gstats, **{k: v for k, v in st.items()},
                "gen_sec": gen_sec, "logp_sec": logp_sec, "train_sec": train_sec,
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "topk": topk_ok}
        allm = [None] * world
        dist.all_gather_object(allm, mine)
        if rank == 0:
            n = sum(m["n"] for m in allm)
            cnt = {c: sum(m["cnt"][c] for m in allm) for c in CLASSES}
            lat = [x for m in allm for x in m["lat"]]
            nt, na = sum(m["n_t"] for m in allm), sum(m["n_a"] for m in allm)
            rec = {"step": step, "turns": n, "mean_reward": round(float(np.mean([x for m in allm for x in m["rew"]])), 4),
                   **{f"{c}_rate": round(cnt[c] / n, 4) for c in CLASSES},
                   "echo_rate": round(sum(m["echo"] for m in allm) / n, 4),
                   "overlap_rate": round(sum(m["overlap"] for m in allm) / n, 4),
                   "counted_overlap_rate": round(sum(m["counted_overlap"] for m in allm) / n, 4),
                   "talkover_audio_rate": round(sum(m["talkover"] for m in allm) / n, 4),
                   "slots_with_valid": round(sum(m["slots_valid"] for m in allm) / max(sum(m["slots"] for m in allm), 1), 4),
                   "latency_median": round(float(np.median(lat)), 3) if lat else None,
                   "latency_mean": round(float(np.mean(lat)), 3) if lat else None,
                   "words_per_turn_median": float(np.median([w for m in allm for w in m["words"]])),
                   "kl_text": round(sum(m["kl_t"] for m in allm) / max(nt, 1), 6),
                   "kl_audio": round(sum(m["kl_a"] for m in allm) / max(na, 1), 6),
                   "clipfrac_text": round(sum(m["clip_t"] for m in allm) / max(nt, 1), 4),
                   "clipfrac_audio": round(sum(m["clip_a"] for m in allm) / max(na, 1), 4),
                   "grad_norm": round(float(np.mean([g for m in allm for g in m["gnorm"]])), 4),
                   "groups_with_spread": round(float(np.mean([m["group_varies"] for m in allm])), 3),
                   "turn_positions_with_spread": round(float(np.mean([m["turn_pos_vary"] for m in allm])), 3),
                   "mean_group_std": round(float(np.mean([m["group_spread"] for m in allm])), 4),
                   "fruits_per_turn": round(float(np.mean([m["fruits_per_turn"] for m in allm])), 3),
                   "words_per_turn": round(float(np.mean([w for m in allm for w in m["words"]])), 2),
                   "turn_cap_hits": sum(m["caps"] for m in allm),
                   "game_sec": round(float(np.mean([m["game_sec"] for m in allm])), 1),
                   "misaligned_tokens": sum(m["misaligned"] for m in allm),
                   "gen_sec": round(max(m["gen_sec"] for m in allm), 1),
                   "logp_sec": round(max(m["logp_sec"] for m in allm), 1),
                   "train_sec": round(max(m["train_sec"] for m in allm), 1),
                   "step_sec": round(time.time() - t_step, 1),
                   "peak_mem_gb": round(max(m["peak_gb"] for m in allm), 1)}
            if allm[0]["topk"]:
                rec["alignment_topk_check"] = allm[0]["topk"]
            with open(res_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[step {step}] reward {rec['mean_reward']:+.3f} valid {rec['valid_rate']:.1%} filler "
                  f"{rec['filler_rate']:.1%} silent {rec['silent_rate']:.1%} repeat {rec['repeat_rate']:.1%} "
                  f"multi {rec['multi_rate']:.1%} sentence {rec['sentence_rate']:.1%} overlap {rec['overlap_rate']:.1%} "
                  f"(counted {rec['counted_overlap_rate']:.1%}, audio talk-over {rec['talkover_audio_rate']:.1%}) "
                  f"valid-slots {rec['slots_with_valid']:.0%} KL t/a {rec['kl_text']:.4f}/"
                  f"{rec['kl_audio']:.4f} spread {rec['groups_with_spread']:.0%} step {rec['step_sec']}s "
                  f"(gen {rec['gen_sec']} logp {rec['logp_sec']} train {rec['train_sec']}) mem {rec['peak_mem_gb']}GB"
                  + (f" topk {rec['alignment_topk_check']}" if "alignment_topk_check" in rec else ""), flush=True)
            subprocess.run([str(ASR_PY), str(ROOT / "scripts" / "plot_grpo.py"), str(res_dir)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if rank == 0:
            # Training rollouts stay out of git: the step's games (JSONs, scores with rewards and
            # advantages, WAVs on audio steps) become runs/grpo/<run>/rollouts/step_XXX.tar.gz on disk;
            # only the step's metrics row (metrics.jsonl) is committed.
            tar_step(rdir)
        dist.barrier()
        kl_reason = None
        if rank == 0 and "audio_kl_max" in cfg.get("stop", {}) and rec["kl_audio"] > cfg["stop"]["audio_kl_max"]:
            kl_reason = f"training audio KL {rec['kl_audio']:.4f} > {cfg['stop']['audio_kl_max']}"
        if step % cfg["checkpoint_every"] == 0 or kl_reason:
            save_ckpt(step)
        if stop_all(kl_reason):
            stopped = True
            break
        if step % cfg["val"]["every"] == 0:
            validate(step)
            reason = None
            if rank == 0:
                subprocess.run([str(ASR_PY), str(ROOT / "scripts" / "plot_grpo.py"), str(res_dir)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                after_validation(step)
                reason = check_stop_after_val(step)
            if stop_all(reason):
                stopped = True
                break
    if rank == 0:
        if upload_proc[0] is not None:
            upload_proc[0].wait()
        log0("[stopped]" if stopped else "[done]")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
