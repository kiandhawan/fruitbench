"""Step 4 report: base fruit-bench numbers, GRPO readiness, timing, best/worst games.

  python scripts/report.py --test runs/base_test_s1 --grpo runs/grpo_check --timing runs/timing32 \
      --asr results/asr_check.json
Writes results/base_fruitbench.md, results/base_fruitbench.json, and results/best_worst/ (transcripts +
timelines of the 5 best and 5 worst test games); their MP3s go to runs/listen/ (gitignored).
"""

import argparse
import json
import re
import shutil
import statistics as st
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from fruitbench.score import CLASSES, filler_phrase, score_game  # noqa: E402
from timeline import timeline  # noqa: E402


def load(d: Path) -> list[tuple[dict, dict]]:
    out = []
    for f in sorted(d.glob("*.score.json")):
        g = json.loads(f.with_name(f.name.replace(".score.json", ".json")).read_text())
        out.append((g, score_game(g)))  # rescore with the current scorer
    return out


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def bench(games: list[tuple[dict, dict]]) -> dict:
    reps = [r for _, s in games for r in s["replies"] if r["scored"]]
    n = len(reps)
    cls = Counter(r["cls"] for r in reps)
    lat_valid = [r["latency"] for r in reps if r["cls"] == "valid"]
    lat_fruit = [r["latency"] for r in reps if r["latency"] is not None]
    fill = Counter(filler_phrase(r["text"]) for r in reps if r["cls"] == "filler")
    by_starter = {}
    for who in ("user", "moshi"):
        rs = [r for g, s in games if g["starter"] == who for r in s["replies"] if r["scored"]]
        by_starter[who] = {"games": sum(g["starter"] == who for g, _ in games), "turns": len(rs),
                           "valid": round(sum(r["cls"] == "valid" for r in rs) / max(len(rs), 1), 4)}
    opening = Counter(r["cls"] for _, s in games for r in s["replies"] if r["scored"] and r["kind"] == "instructions")
    greet = Counter(filler_phrase(s["greeting"]) for _, s in games)
    return {
        "games": len(games), "scored_turns": n,
        "rates": {c: round(cls.get(c, 0) / n, 4) for c in CLASSES},
        "counts": {c: cls.get(c, 0) for c in CLASSES},
        "overlap_rate": round(sum(r["overlap"] for r in reps) / n, 4),
        "echo_rate": round(sum(r["echo"] for r in reps) / n, 4),
        "latency_valid": {"n": len(lat_valid),
                          "median": round(st.median(lat_valid), 3) if lat_valid else None,
                          "mean": round(st.mean(lat_valid), 3) if lat_valid else None},
        "latency_any_fruit": {"n": len(lat_fruit),
                              "median": round(st.median(lat_fruit), 3) if lat_fruit else None,
                              "mean": round(st.mean(lat_fruit), 3) if lat_fruit else None},
        "mean_reward": round(st.mean(s["total_reward"] for _, s in games), 4),
        "games_with_any_valid": round(sum(s["counts"]["valid"] > 0 for _, s in games) / len(games), 4),
        "by_starter": by_starter,
        "opening_turn_moshi_starts": dict(opening),
        "top_filler": fill.most_common(10),
        "top_greetings": greet.most_common(5),
        "game_len_sec": {"median": round(st.median(g["end_time"] for g, _ in games), 1),
                         "max": round(max(g["end_time"] for g, _ in games), 1)},
    }


def grpo(games: list[tuple[dict, dict]]) -> dict:
    groups = defaultdict(list)
    for g, s in games:
        groups[re.sub(r"_r\d+$", "", g["game_id"])].append(s)
    tot_var, turn_var, pos_var, pos_n, sizes, stds = 0, 0, 0, 0, [], []
    for k, ss in groups.items():
        sizes.append(len(ss))
        tots = [s["total_reward"] for s in ss]
        tot_var += len(set(tots)) > 1
        stds.append(st.pstdev(tots))
        vecs = [tuple(s["turn_rewards"]) for s in ss]
        turn_var += len(set(vecs)) > 1
        L = min(len(v) for v in vecs)
        for j in range(L):
            pos_n += 1
            pos_var += len({v[j] for v in vecs}) > 1
    n = len(groups)
    return {"groups": n, "group_sizes": sorted(set(sizes)),
            "frac_groups_total_reward_varies": round(tot_var / n, 4),
            "frac_groups_turn_rewards_vary": round(turn_var / n, 4),
            "frac_group_turn_positions_vary": round(pos_var / max(pos_n, 1), 4),
            "mean_within_group_std_total": round(st.mean(stds), 4),
            "bench": bench(games)}


def timing(d: Path) -> dict:
    rows = json.loads((d / "timing_worker0.json").read_text())
    r = rows[0]
    return {"games": r["games"], "batch": r["batch"], "wall_sec": r["wall_sec"], "loop_sec": r["loop_sec"],
            "game_sec_longest": r["game_sec"], "realtime_factor": round(r["loop_sec"] / r["game_sec"], 3),
            "peak_mem_gb": r["peak_mem_gb"], "gpu": "1x H200"}


def export(games, out_md: Path, out_audio: Path) -> dict:
    ranked = sorted(games, key=lambda gs: (-gs[1]["total_reward"], gs[0]["game_id"]))
    best = ranked[:5]
    # Worst: lowest reward; among ties prefer games where Moshi talked but failed (filler/repeat/
    # non-fruit) over pure silence, as they are more informative to listen to.
    worst = sorted(games, key=lambda gs: (gs[1]["total_reward"], -(gs[1]["counts"]["filler"] + gs[1]["counts"]["repeat"]
                                                                    + gs[1]["counts"]["non_fruit"]), gs[0]["game_id"]))[:5]
    out_md.mkdir(parents=True, exist_ok=True)
    out_audio.mkdir(parents=True, exist_ok=True)
    picked = {}
    for tag, lst in (("best", best), ("worst", worst)):
        for i, (g, s) in enumerate(lst, 1):
            name = f"{tag}{i}_{g['game_id']}"
            (out_md / f"{name}.txt").write_text(timeline(g))
            wav = Path(g["_dir"]) / f"{g['game_id']}.wav"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame",
                            "-q:a", "2", str(out_audio / f"{name}.mp3")], check=True)
            picked[name] = {"reward": s["total_reward"], "counts": s["counts"]}
    return picked


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--grpo", type=Path)
    p.add_argument("--timing", type=Path)
    p.add_argument("--asr", type=Path)
    args = p.parse_args()
    test = load(args.test)
    for g, _ in test:
        g["_dir"] = str(args.test)
    res = {"test": bench(test)}
    if args.grpo:
        res["grpo"] = grpo(load(args.grpo))
    if args.timing:
        res["timing"] = timing(args.timing)
    if args.asr and args.asr.exists():
        a = json.loads(args.asr.read_text())
        res["asr"] = {k: a[k] for k in ("games", "fruit_mentions_game_level", "scored_turns", "agree_class", "agree_class_and_fruit",
                                        "per_text_class", "disagreements")}
    res["listen"] = export(test, ROOT / "results" / "best_worst", ROOT / "runs" / "listen")
    (ROOT / "results" / "base_fruitbench.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
