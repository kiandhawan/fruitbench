"""Compare GRPO runs side by side: training curves + step-0 vs final validation.
  .venv-asr/bin/python scripts/plot_grpo_compare.py results/grpo/grpo_smoke results/grpo/grpo_smoke2
  -> results/plots/grpo_compare_<run1>_vs_<run2>.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load(p):
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def main():
    runs = [Path(a) for a in sys.argv[1:]]
    data = {r.name: (load(r / "metrics.jsonl"), load(r / "val.jsonl")) for r in runs}
    keys = [("valid_rate", "% valid new fruit"), ("silent_rate", "% silent"), ("multi_rate", "% multi-fruit"),
            ("overlap_rate", "% overlap (raw, scorer)"), ("words_per_turn", "mean words per turn"),
            ("mean_reward", "mean reward per turn (own reward)"), ("kl_text", "KL to base, text tokens"),
            ("kl_audio", "KL to base, audio tokens")]
    f, axes = plt.subplots(2, 4, figsize=(20, 7.5))
    for ax, (k, title) in zip(axes.flat, keys):
        for name, (tr, _) in data.items():
            pts = [(r["step"], r[k]) for r in tr if r.get(k) is not None]
            if pts:
                ax.plot(*zip(*pts), "-o", ms=3, label=name)
        ax.set_title(f"train: {title}")
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    f.tight_layout()
    out = ROOT / "results" / "plots" / f"grpo_compare_{'_vs_'.join(data)}.png"
    f.savefig(out, dpi=110)
    plt.close(f)
    # validation: first vs last eval of each run
    vk = [("valid_rate", "valid"), ("repeat_rate", "repeat"), ("filler_rate", "filler"), ("silent_rate", "silent"),
          ("multi_rate", "multi"), ("overlap_rate", "overlap (raw)"), ("counted_overlap_rate", "overlap (counted)")]
    bars = []
    for name, (_, va) in data.items():
        if va:
            bars.append((f"{name} step {va[0]['step']}", va[0]))
            if len(va) > 1:
                bars.append((f"{name} step {va[-1]['step']}", va[-1]))
    if bars:
        f, ax = plt.subplots(figsize=(13, 4.5))
        x = np.arange(len(vk))
        wd = 0.8 / len(bars)
        for i, (label, rec) in enumerate(bars):
            ys = [rec.get(k) or 0 for k, _ in vk]
            ax.bar(x + i * wd, ys, wd, label=label)
        ax.set_xticks(x + wd * (len(bars) - 1) / 2, [t for _, t in vk])
        ax.set_ylabel("fraction of scored turns")
        ax.set_title("validation (100 val games, 12 user turns)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        f.tight_layout()
        f.savefig(out.with_name(out.stem + "_val.png"), dpi=110)
        plt.close(f)
    print(out)


if __name__ == "__main__":
    main()
