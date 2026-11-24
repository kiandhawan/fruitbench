"""Finish the full GRPO run unattended (run in its own tmux window; it waits for the trainer to exit).

  python scripts/finalize_full.py --config configs/grpo_full.yaml
  python scripts/finalize_full.py --config configs/grpo_smoke4.yaml --dry-run --n-test 8   # test on old data

1. Final checkpoint = best validation mean reward among checkpoints that passed the ordinary-
   conversation check (<= 10% of replies with a fruit) with training audio KL < 0.03 at that step.
   The overall best (no filters) is reported too, and whether it differs.
2. The 200 TEST games (seed 1, same specs as the base benchmark) once with the final LoRA
   (runs/grpo_final_test; per-game JSONs archived to results/games/grpo_final_test).
3. Base vs final on the same test games: scorer v3 + the run's reward, and v1 (base benchmark rules).
4. Release "grpo-final": before/after MP3s (base vs final) of 6 test games + timelines.
5. Draft release "checkpoints": lora_final.safetensors and lora_best.safetensors (+ final.json).
6. results/full_run_summary.md; commit + push. The pod is left running.
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = "/root/moshi-finetune/.venv/bin/python"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="no waiting, no uploads/releases/commits; outputs in scratch")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--out-root", type=Path, default=ROOT)
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / a.config).read_text())
    run = cfg["run_name"]
    res_dir, run_dir = ROOT / "results" / "grpo" / run, ROOT / "runs" / "grpo" / run
    out = a.out_root
    logf = (ROOT / "runs" / "grpo" / f"{run}_finalize.log") if not a.dry_run else out / "finalize.log"
    logf.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[finalize {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logf, "a") as f:
            f.write(line + "\n")

    def sh(cmd: list[str], check: bool = True, **kw) -> subprocess.CompletedProcess:
        log("$ " + " ".join(cmd)[:300])
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)
        if r.returncode != 0:
            log(f"  exit {r.returncode}: {(r.stderr or r.stdout)[-600:]}")
            if check:
                raise RuntimeError(cmd[0])
        return r

    # ------------------------------------------------------------------ 0. wait for the trainer
    if not a.dry_run:
        log(f"waiting for the trainer of {run} to exit")
        while subprocess.run(["pgrep", "-f", f"grpo_train.py --config {a.config}"], capture_output=True).stdout.strip():
            time.sleep(60)
        time.sleep(90)  # let the trainer's last commit/upload and the watchdog finish
    tlog = (ROOT / "runs" / "grpo" / f"{run}.log").read_text(errors="ignore") if (ROOT / "runs" / "grpo" / f"{run}.log").exists() else ""
    es_file = res_dir / "early_stop.json"
    ended = ("ended by early stopping (watchdog): " + json.loads(es_file.read_text())["reason"] if es_file.exists() else
             "finished all steps" if "[done]" in tlog[-5000:] else
             "stopped: " + (re.findall(r"\[STOP\] (.*)", tlog) or ["?"])[-1] if "[STOP]" in tlog else
             "exited without [done] (crash?)")
    log(f"trainer {ended}")
    flags = sorted(set(re.findall(r"\[FLAG\] (.*?) \(logged only", tlog)))
    flags += sorted(set(re.findall(r"\] (FLAG: Whisper heard only .*)", tlog)))

    # ------------------------------------------------------------------ 1. choose the checkpoint
    val = jsonl(res_dir / "val.jsonl")
    met = {m["step"]: m for m in jsonl(res_dir / "metrics.jsonl")}
    chat_max = cfg.get("stop", {}).get("chat_fruit_max", 0.10)
    kl_max = cfg.get("stop", {}).get("audio_kl_max", 0.03)
    rows = []
    for v in val:
        ck = run_dir / "ckpt" / f"step_{v['step']:03d}" / "lora.safetensors"
        kl_a = met.get(v["step"], {}).get("kl_audio")
        chat = v.get("chat", {}).get("rate")
        rows.append({**v, "ckpt": ck if ck.exists() else None, "kl_audio_train": kl_a, "chat_rate": chat,
                     "eligible": bool(ck.exists() and v["step"] > 0 and (chat is None or chat <= chat_max)
                                      and (kl_a is not None and kl_a < kl_max))})
    with_ck = [r for r in rows if r["ckpt"]]
    if not with_ck:
        log("no checkpoint with a validation: nothing to evaluate")
        final = best = None
    else:
        key = lambda r: (r["mean_reward"], r["valid_rate"])  # noqa: E731
        best = max(with_ck, key=key)
        elig = [r for r in with_ck if r["eligible"]]
        final = max(elig, key=key) if elig else best
        log(f"overall best: step {best['step']} (mean reward {best['mean_reward']:+.3f}); final: step {final['step']}"
            + ("" if elig else " (no checkpoint passed the chat and audio-KL filters; using the overall best)"))

    # ------------------------------------------------------------------ 2. test games with the final LoRA
    test_dir = out / "runs" / "grpo_final_test" if not a.dry_run else out / "grpo_final_test"
    test_res = None
    if final:
        cmd = [PY, "scripts/run_games.py", "--split", "test", "--n", str(a.n_test), "--seed", "1", "--gpus", "0", "1", "2", "3",
               "--batch", "50" if a.n_test >= 200 else str(max(1, a.n_test // 4)), "--lora", str(final["ckpt"]),
               "--out", str(test_dir)]
        if a.dry_run:
            cmd += ["--no-archive"]
        else:
            cmd += ["--archive-dir", str(ROOT / "results" / "games" / "grpo_final_test")]
        sh(cmd, timeout=4 * 3600)
        # ------------------------------------------------------------------ 3. base vs final (same games)
        ids = sorted(f.stem for f in test_dir.glob("test_s1_*.json") if not f.name.endswith(".score.json"))
        base_src = ROOT / "results" / "base_games" / "test_s1"
        if a.n_test < 200:  # dry run: only the games that were played
            sub = out / "base_subset"
            shutil.rmtree(sub, ignore_errors=True)
            sub.mkdir(parents=True)
            for i in ids:
                shutil.copy2(base_src / f"{i}.json", sub / f"{i}.json")
            base_src = sub
        rdir = res_dir.parent if not a.dry_run else out
        sh([PY, "scripts/rescore_val.py", "--config", a.config, "--out", str(rdir / f"{run}_test_v3"),
            f"base={base_src.relative_to(ROOT) if base_src.is_relative_to(ROOT) else base_src}",
            f"final_step{final['step']}={test_dir.relative_to(ROOT) if test_dir.is_relative_to(ROOT) else test_dir}"])
        sh([PY, "scripts/rescore_val.py", "--config", "configs/grpo_smoke.yaml", "--out", str(rdir / f"{run}_test_v1"),
            f"base={base_src.relative_to(ROOT) if base_src.is_relative_to(ROOT) else base_src}",
            f"final_step{final['step']}={test_dir.relative_to(ROOT) if test_dir.is_relative_to(ROOT) else test_dir}"])
        test_res = {"v3": json.loads((rdir / f"{run}_test_v3.json").read_text())["runs"],
                    "v1": json.loads((rdir / f"{run}_test_v1.json").read_text())["runs"],
                    "v3_md": (rdir / f"{run}_test_v3.md").read_text()}

        # ------------------------------------------------------------------ 4. before/after MP3s
        from timeline import timeline
        mp = out / "runs" / "grpo_final_mp3" if not a.dry_run else out / "mp3"
        shutil.rmtree(mp, ignore_errors=True)
        mp.mkdir(parents=True)
        pick = sorted(random.Random(0).sample(ids, min(6, len(ids))))
        base_wavs = ROOT / "runs" / "base_test_s1"
        lines = []
        for gid in pick:
            for tag, d in (("base", base_wavs), (f"final_step{final['step']}", test_dir)):
                if (d / f"{gid}.wav").exists():
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(d / f"{gid}.wav"), "-codec:a",
                                    "libmp3lame", "-q:a", "2", str(mp / f"{gid}_{tag}.mp3")], check=True)
                g = json.loads(((base_src if tag == "base" else test_dir) / f"{gid}.json").read_text())
                lines.append(f"##### {gid} - {tag}\n" + timeline(g, "v3"))
        (mp / "timelines.txt").write_text("\n".join(lines))
        log(f"MP3s: {sorted(p.name for p in mp.glob('*.mp3'))}")
        if not a.dry_run:
            sh(["gh", "release", "delete", "grpo-final", "--yes", "--cleanup-tag"], check=False)
            sh(["gh", "release", "create", "grpo-final", *map(str, sorted(mp.glob("*.mp3"))), str(mp / "timelines.txt"),
                "--title", "grpo-final", "--target", "main", "--notes",
                f"Base Moshi vs the final GRPO LoRA ({run}, step {final['step']}) on the same 6 TEST games (seed 1). "
                "Stereo MP3: left = Moshi, right = simulated user. Timelines (scorer v3) in timelines.txt. "
                "Summary: results/full_run_summary.md."])

    # ------------------------------------------------------------------ 5. LoRA uploads (draft release)
    rel = cfg.get("upload", {}).get("release")
    if final and rel and not a.dry_run:
        ud = run_dir / "upload_final"
        shutil.rmtree(ud, ignore_errors=True)
        ud.mkdir(parents=True)
        os.link(final["ckpt"], ud / "lora_final.safetensors")
        os.link(best["ckpt"], ud / "lora_best.safetensors")
        (ud / "final.json").write_text(json.dumps({"run": run, "final_step": final["step"], "best_step": best["step"],
                                                   "rule": "best validation mean reward among checkpoints with chat <= "
                                                   f"{chat_max:.0%} and training audio KL < {kl_max}"}, indent=1))
        sh(["gh", "release", "upload", rel, str(ud / "lora_final.safetensors"), str(ud / "lora_best.safetensors"),
            str(ud / "final.json"), "--clobber"], check=False, timeout=3600)

    # ------------------------------------------------------------------ 6. summary
    def pc(x):
        return "-" if x is None else f"{x:.1%}"

    md = [f"# Full GRPO run summary ({run})", "",
          f"Config `{a.config}`; trainer {ended}; {len(met)} training steps logged, {len(val)} validations.",
          "Validation: 100 val games (12 user turns, scorer v3, the run's reward) every 10 steps; training curves:",
          f"`results/plots/{run}/grpo_*.png`.", "", "## Validation curve", "",
          "| step | bare valid | sentence | repeat | silent | filler | multi | overlap counted | talk-over (audio) | "
          "chat replies w/ fruit | train audio KL | train text KL | mean reward | Whisper fruits | eligible |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mt = met.get(r["step"], {})
        md.append(f"| {r['step']} | {pc(r['valid_rate'])} | {pc(r.get('sentence_rate'))} | {pc(r['repeat_rate'])} | "
                  f"{pc(r['silent_rate'])} | {pc(r['filler_rate'])} | {pc(r['multi_rate'])} | {pc(r.get('counted_overlap_rate'))} | "
                  f"{pc(r.get('talkover_audio_rate'))} | {pc(r['chat_rate'])} | "
                  f"{'-' if mt.get('kl_audio') is None else format(mt['kl_audio'], '.4f')} | "
                  f"{'-' if mt.get('kl_text') is None else format(mt['kl_text'], '.4f')} | {r['mean_reward']:+.3f} | "
                  f"{pc(r.get('asr_fruit_agreement'))} | {'yes' if r['eligible'] else 'no'} |")
    md += ["", "## Chosen checkpoint", ""]
    if final:
        md.append(f"- Rule: highest validation mean reward among checkpoints with ordinary-conversation fruit rate <= "
                  f"{chat_max:.0%} and training audio KL < {kl_max} at that step.")
        md.append(f"- **Final: step {final['step']}** (mean reward {final['mean_reward']:+.3f}, bare valid {pc(final['valid_rate'])}).")
        md.append(f"- Overall best on validation mean reward: step {best['step']} ({best['mean_reward']:+.3f})"
                  + (" - the same checkpoint." if best["step"] == final["step"] else " - **differs from the final checkpoint**."))
        md.append(f"- LoRAs: draft release `{rel}` (lora_final.safetensors, lora_best.safetensors).")
    else:
        md.append("- No checkpoint with a validation was available.")
    if test_res:
        b, f_ = test_res["v3"]["base"], test_res["v3"][f"final_step{final['step']}"]
        b1, f1 = test_res["v1"]["base"], test_res["v1"][f"final_step{final['step']}"]
        md += ["", f"## Test: 200 test games (seed 1), base Moshi vs final (step {final['step']}), same games", "",
               f"- Bare valid (v3): base {pc(b['valid_rate'])} -> final **{pc(f_['valid_rate'])}**; valid under the base "
               f"benchmark's v1 rules: {pc(b1['valid_rate'])} -> {pc(f1['valid_rate'])}.",
               f"- Mean reward per turn (run's reward): {b['mean_reward']:+.3f} -> {f_['mean_reward']:+.3f}.",
               "- Before/after MP3s of 6 test games: release `grpo-final`.", "", "Full table (scorer v3, run's reward):", "",
               test_res["v3_md"].split("\n", 2)[2]]
    md += ["", "## Flags (logged, never stopped the run)", ""] + ([f"- {x}" for x in flags] or ["- none"])
    md += ["", "Validation flags for Whisper are expected around 81-88% even for base Moshi (see results/LOG.md)."]
    summ = (ROOT / "results" / "full_run_summary.md") if not a.dry_run else out / "full_run_summary.md"
    summ.write_text("\n".join(md) + "\n")
    log(f"wrote {summ}")

    # ------------------------------------------------------------------ 7. commit + push
    if not a.dry_run:
        subprocess.run([PY, "-c", "print()"])
        sh([str(ROOT / ".venv-asr" / "bin" / "python"), "scripts/plot_grpo.py", str(res_dir)], check=False)
        paths = ["results/full_run_summary.md", str(res_dir.relative_to(ROOT)), f"results/grpo/{run}_test_v3.md",
                 f"results/grpo/{run}_test_v3.json", f"results/grpo/{run}_test_v1.md", f"results/grpo/{run}_test_v1.json",
                 "results/games/grpo_final_test", "results/plots"]
        sh(["git", "add", *[p for p in paths if (ROOT / p).exists()]], check=False)
        staged = sh(["git", "diff", "--cached", "--name-only"], check=False).stdout.split()
        bad = [x for x in staged if x.endswith((".wav", ".mp3", ".npy", ".pt", ".safetensors", ".tar.gz"))
               or "key" in x.lower() or ".env" in x]
        if bad:
            sh(["git", "reset", "-q", *bad], check=False)
            log(f"refused to commit {bad}")
        sh(["git", "-c", "user.name=kiandhawan", "-c", "user.email=kiandhawan@gmail.com", "commit", "-q", "-m",
            f"{run}: final checkpoint, 200 test games vs base, summary\n\n"
            "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"], check=False)
        sh(["git", "push", "-q"], check=False)
    log("all done; the pod is left running")


if __name__ == "__main__":
    main()
