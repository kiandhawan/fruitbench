"""Bare-valid rate (scorer v3) by turn number, base Moshi vs the final GRPO checkpoint, on the 200 test
games (seed 1). Reads the archived per-game summaries and rescores them with rules v3.
  .venv-asr/bin/python scripts/plot_turn_valid.py  ->  results/plots/grpo_full/valid_by_turn.png (+ .json)"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fruitbench.score import score_game  # noqa: E402

RUNS = {"base Moshi": ROOT / "results" / "base_games" / "test_s1",
        "GRPO (step 120)": ROOT / "results" / "games" / "grpo_final_test"}
COLORS = {"base Moshi": "#eb6834", "GRPO (step 120)": "#2a78d6"}
OUT = ROOT / "results" / "plots" / "grpo_full" / "valid_by_turn"


def by_turn(d: Path) -> tuple[list[float], list[int], float]:
    hit, n = {}, {}
    files = sorted(f for f in d.glob("test_s1_*.json") if not f.name.endswith(".score.json"))
    for f in files:
        reps = [r for r in score_game(json.loads(f.read_text()), "v3")["replies"] if r["scored"]]
        for i, r in enumerate(reps, 1):
            n[i] = n.get(i, 0) + 1
            hit[i] = hit.get(i, 0) + (r["cls"] == "valid")
    turns = sorted(n)
    return [hit[t] / n[t] for t in turns], [n[t] for t in turns], sum(hit.values()) / sum(n.values()), len(files)


def main():
    res = {}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, d in RUNS.items():
        rate, n, overall, games = by_turn(d)
        res[name] = {"games": games, "overall": overall, "rate_by_turn": rate, "n_by_turn": n}
        x = list(range(1, len(rate) + 1))
        ax.plot(x, [100 * r for r in rate], "-o", color=COLORS[name], lw=2, ms=6, label=f"{name} ({100 * overall:.1f}% overall)")
        ax.annotate(f"{100 * rate[-1]:.0f}%", (x[-1], 100 * rate[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#333")
        print(name, games, "games", f"overall {overall:.1%}", " ".join(f"{100 * r:.0f}" for r in rate), "n", n)
    ax.set_xticks(range(1, 14))
    ax.set_xlim(0.5, 13.9)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Moshi's turn number (turn 13: only the 67 games Moshi starts)")
    ax.set_ylabel("bare valid new fruit (%)")
    ax.set_title("Valid rate by turn: 200 test games, base vs GRPO")
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".png"), dpi=130)
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
