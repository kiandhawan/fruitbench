"""Rescore saved validation games under one scorer/reward config so runs are comparable.

  python scripts/rescore_val.py --config configs/grpo_smoke3.yaml \
      base=results/grpo/grpo_smoke/val_step_000 smoke1_s20=results/grpo/grpo_smoke/val_step_020 \
      smoke2_s20=results/grpo/grpo_smoke2/val_step_020 --out results/grpo/rescore_v3
Writes <out>.json and <out>.md (same metric definitions as grpo_train.py's validation).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fruitbench.rewards import counted_overlap, game_rewards  # noqa: E402
from fruitbench.score import CLASSES, score_game  # noqa: E402


def stats(d: Path, w: dict, rules: str) -> dict:
    files = sorted(f for f in d.glob("*.json") if not f.name.endswith(".score.json") and f.name[-9:-5].isdigit())
    reps, rew, caps, pos = [], [], [], []
    for f in files:
        g = json.loads(f.read_text())
        sc = score_game(g, rules)
        sr = [r for r in sc["replies"] if r["scored"]]
        reps += sr
        rew += game_rewards(sc, w)
        caps.append(sum(1 for e in g["events"] if "turn cap" in e.get("why", "")))
        pos += [(i + 1, r) for i, r in enumerate(r for r in sc["replies"] if r["kind"] == "fruit")]
    n = len(reps)
    lat = [r["latency"] for r in reps if r["cls"] == "valid"]
    words = [len(r["word_t"]) for r in reps]
    out = {"games": len(files), "turns": n, "mean_reward": round(float(np.mean(rew)), 4),
           **{f"{c}_rate": round(sum(r["cls"] == c for r in reps) / n, 4) for c in CLASSES},
           "echo_rate": round(sum(r["echo"] for r in reps) / n, 4),
           "overlap_rate": round(sum(r["overlap"] for r in reps) / n, 4),
           "counted_overlap_rate": round(sum(counted_overlap(r, w) for r in reps) / n, 4),
           "latency_median": round(float(np.median(lat)), 3) if lat else None,
           "words_per_turn_median": float(np.median(words)), "words_per_turn_mean": round(float(np.mean(words)), 2),
           "turn_cap_per_game": round(sum(caps) / len(files), 3)}
    for name, lo, hi in (("turns_1_6", 1, 6), ("turns_7_12", 7, 12)):
        ps = [r for i, r in pos if lo <= i <= hi]
        out[name] = {k: round(sum(r["cls"] == k for r in ps) / len(ps), 4) for k in ("valid", "sentence", "repeat")}
        out[name]["overlap"] = round(sum(r["overlap"] for r in ps) / len(ps), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("runs", nargs="+", help="label=path/to/val_step_dir")
    a = ap.parse_args()
    w = yaml.safe_load(a.config.read_text())["reward"]
    rules = w.get("scorer", "v1")
    res = {lab: stats(ROOT / p, w, rules) for lab, p in (x.split("=", 1) for x in a.runs)}
    a.out.with_suffix(".json").write_text(json.dumps({"config": str(a.config), "rules": rules, "runs": res}, indent=1))
    labs = list(res)
    rows = [("valid (bare fruit)", "valid_rate"), ("sentence (new fruit in other words)", "sentence_rate"),
            ("repeat", "repeat_rate"), ("echo", "echo_rate"), ("multi", "multi_rate"), ("filler", "filler_rate"),
            ("silent", "silent_rate"), ("non_fruit", "non_fruit_rate"), ("overlap (raw)", "overlap_rate"),
            ("overlap (counted, 0.2 s grace)", "counted_overlap_rate")]
    md = [f"# Validation rescored under {rules} (`{a.config}`)\n", "| metric | " + " | ".join(labs) + " |",
          "|---|" + "---|" * len(labs)]
    md.append("| mean reward per turn | " + " | ".join(f"{res[l]['mean_reward']:+.3f}" for l in labs) + " |")
    for t, k in rows:
        md.append(f"| {t} | " + " | ".join(f"{res[l][k]:.1%}" for l in labs) + " |")
    md.append("| valid latency median (s) | " + " | ".join(str(res[l]["latency_median"]) for l in labs) + " |")
    md.append("| words per turn median (mean) | " + " | ".join(f"{res[l]['words_per_turn_median']:.0f} ({res[l]['words_per_turn_mean']})" for l in labs) + " |")
    md.append("| 10 s turn cap per game | " + " | ".join(str(res[l]["turn_cap_per_game"]) for l in labs) + " |")
    for part in ("turns_1_6", "turns_7_12"):
        for k in ("valid", "sentence", "repeat", "overlap"):
            md.append(f"| {part.replace('_', ' ').replace('turns', 'turns').replace(' 1 6', ' 1-6').replace(' 7 12', ' 7-12')}: {k} | "
                      + " | ".join(f"{res[l][part][k]:.1%}" for l in labs) + " |")
    a.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
