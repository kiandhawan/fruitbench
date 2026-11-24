"""Watchdog for a GRPO run, independent of the trainer and of any Claude session (run in its own tmux window).

  python scripts/watchdog.py --config configs/grpo_full.yaml

Every 30 s it reads results/grpo/<run>/{val.jsonl, metrics.jsonl} and checks the config's `stop:`
conditions (the same rules the trainer checks itself):
  - validation bare valid at step 20 below val_valid_min_at_step_20
  - validation silence at step 30 above val_silent_max_at_step_30
  - validation bare valid dropping val_valid_drops_in_a_row validations in a row
  - ordinary conversations: share of Moshi replies with a fruit above chat_fruit_max
  - training audio KL above audio_kl_max
With `stop_mode: flag_only` (night run) a condition is only logged once as "FLAG: <reason>".
Otherwise: if a condition holds and the trainer is still running GRACE seconds later (it should have stopped by
itself), the watchdog writes "[watchdog STOP] <reason>" to the run log and stops the trainer (SIGINT,
then SIGTERM). Whenever the trainer is no longer running (stopped, crashed or done), it uploads the
latest checkpoint (and the best one, from best.json) to the draft release, writes a line to the log
and exits.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
GRACE = 300  # s: time the trainer gets to stop by itself after a condition first holds
POLL = 30


def log(logfile: Path, msg: str) -> None:
    line = f"[watchdog {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(logfile, "a") as f:
        f.write(line + "\n")


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def trainer_pids(config: str) -> list[int]:
    out = subprocess.run(["pgrep", "-f", f"grpo_train.py --config {config}"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def check(cfg: dict, val: list[dict], met: list[dict]) -> str | None:
    s = cfg.get("stop", {})
    if met and "audio_kl_max" in s and met[-1]["kl_audio"] > s["audio_kl_max"]:
        return f"training audio KL {met[-1]['kl_audio']:.4f} at step {met[-1]['step']} > {s['audio_kl_max']}"
    if not val:
        return None
    last = val[-1]
    if last["step"] == 20 and "val_valid_min_at_step_20" in s and last["valid_rate"] < s["val_valid_min_at_step_20"]:
        return f"validation bare valid {last['valid_rate']:.1%} at step 20 < {s['val_valid_min_at_step_20']:.1%}"
    if last["step"] == 30 and "val_silent_max_at_step_30" in s and last["silent_rate"] > s["val_silent_max_at_step_30"]:
        return f"validation silence {last['silent_rate']:.1%} at step 30 > {s['val_silent_max_at_step_30']:.0%}"
    n = s.get("val_valid_drops_in_a_row")
    if n and len(val) >= n + 1:
        v = [x["valid_rate"] for x in val[-(n + 1):]]
        if all(v[i + 1] < v[i] for i in range(n)):
            return f"validation bare valid dropped {n} validations in a row: {' -> '.join(f'{x:.1%}' for x in v)}"
    if "chat_fruit_max" in s and "chat" in last and last["chat"]["rate"] > s["chat_fruit_max"]:
        return f"ordinary conversations: {last['chat']['rate']:.1%} of Moshi replies mention a fruit (> {s['chat_fruit_max']:.0%})"
    return None


def early_stop_reason(cfg: dict, val: list[dict]) -> str | None:
    """Best validation mean reward (trained checkpoints, step > 0) not improved for `patience` validations."""
    pat = cfg.get("early_stop", {}).get("patience_validations")
    h = [x for x in val if x["step"] > 0]
    if not pat or len(h) <= pat:
        return None
    best_i = max(range(len(h)), key=lambda i: h[i]["mean_reward"])  # first index of the best value
    since = len(h) - 1 - best_i
    if since >= pat:
        b = h[best_i]
        return (f"best validation mean reward {b['mean_reward']:+.3f} (step {b['step']}) not improved for {since} "
                f"validations (last: step {h[-1]['step']}, {h[-1]['mean_reward']:+.3f})")
    return None


def stop_trainer(config: str) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        for pid in trainer_pids(config):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        for _ in range(12):
            if not trainer_pids(config):
                return
            time.sleep(5)


def upload_final(cfg: dict, run_dir: Path, res_dir: Path, logfile: Path) -> None:
    rel = cfg.get("upload", {}).get("release")
    ck = sorted((run_dir / "ckpt").glob("step_*/lora.safetensors"))
    if not rel or not ck:
        log(logfile, "no checkpoint or no release configured; nothing to upload")
        return
    latest = ck[-1]
    ud = run_dir / "upload_watchdog"
    shutil.rmtree(ud, ignore_errors=True)
    ud.mkdir(parents=True)
    os.link(latest, ud / "lora_latest.safetensors")
    files = [ud / "lora_latest.safetensors"]
    info = {"run": cfg["run_name"], "latest_step": int(latest.parent.name.split("_")[1]), "uploaded_by": "watchdog"}
    bj = res_dir / "best.json"
    if bj.exists():
        b = json.loads(bj.read_text())
        bp = run_dir / "ckpt" / f"step_{b['step']:03d}" / "lora.safetensors"
        if bp.exists():
            os.link(bp, ud / "lora_best.safetensors")
            files.append(ud / "lora_best.safetensors")
            info.update(best_step=b["step"], best=b)
    (ud / "checkpoints.json").write_text(json.dumps(info, indent=1))
    files.append(ud / "checkpoints.json")
    r = subprocess.run(["gh", "release", "upload", rel, *map(str, files), "--clobber"], cwd=ROOT,
                       capture_output=True, text=True)
    log(logfile, f"uploaded {[f.name for f in files]} (latest = {latest.parent.name}) to draft release "
                 f"'{rel}': {'ok' if r.returncode == 0 else 'FAILED ' + r.stderr[-200:]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / a.config).read_text())
    run = cfg["run_name"]
    res_dir, run_dir = ROOT / "results" / "grpo" / run, ROOT / "runs" / "grpo" / run
    # own log file: the trainer log is written by tee at its own offset and would overwrite appended lines
    logfile = ROOT / "runs" / "grpo" / f"{run}_watchdog.log"
    trainer_log = ROOT / "runs" / "grpo" / f"{run}.log"
    log(logfile, f"started for {run}: checks every {POLL}s, grace {GRACE}s, stop rules {cfg.get('stop', {})}")
    seen_running = False
    first_hit: dict[str, float] = {}
    early = None
    while True:
        pids = trainer_pids(a.config)
        if pids:
            seen_running = True
            reason = check(cfg, jsonl(res_dir / "val.jsonl"), jsonl(res_dir / "metrics.jsonl"))
            if reason and reason not in first_hit:
                first_hit[reason] = time.time()
                if cfg.get("stop_mode") == "flag_only":
                    log(logfile, f"FLAG: {reason} (logged only; the run continues)")
            val = jsonl(res_dir / "val.jsonl")
            es = early_stop_reason(cfg, val)
            if es and early is None:
                # end the run normally once this validation's commit/push is done (checkpoint already saved)
                step = val[-1]["step"]
                deadline = time.time() + 900
                while time.time() < deadline and f"[commit] pushed validation step {step}" not in \
                        (trainer_log.read_text(errors="ignore")[-50000:] if trainer_log.exists() else ""):
                    time.sleep(10)
                early = es
                log(logfile, f"EARLY STOP: {es}; ending the run (checkpoints kept; finalization runs next)")
                with open(res_dir / "early_stop.json", "w") as f:
                    json.dump({"reason": es, "at_validation_step": step, "time": time.strftime("%Y-%m-%d %H:%M")}, f)
                stop_trainer(a.config)
                continue
            if reason and cfg.get("stop_mode") != "flag_only" and time.time() - first_hit[reason] > GRACE:
                log(logfile, f"STOP condition holds and the trainer is still running: {reason}; stopping it")
                for sig in (signal.SIGINT, signal.SIGTERM):
                    for pid in trainer_pids(a.config):
                        try:
                            os.kill(pid, sig)
                        except ProcessLookupError:
                            pass
                    time.sleep(60)
        elif seen_running:
            text = trainer_log.read_text(errors="ignore")[-20000:] if trainer_log.exists() else ""
            how = (f"early stop: {early}" if early else
                   "stopped by a stop condition" if "[STOP]" in text
                   else "finished" if "[done]" in text else "exited without [done] (crash?)")
            log(logfile, f"trainer is no longer running ({how})")
            upload_final(cfg, run_dir, res_dir, logfile)
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
