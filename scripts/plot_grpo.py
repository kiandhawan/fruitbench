"""Plot GRPO curves: results/grpo/<run>/{metrics,val}.jsonl -> results/plots/<run>/grpo_*.png.
Runs in .venv-asr (has matplotlib):  .venv-asr/bin/python scripts/plot_grpo.py results/grpo/<run>"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load(p):
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def main():
    d = Path(sys.argv[1])
    drop = {int(x) for x in sys.argv[2].split(",")} if len(sys.argv) > 2 else set()  # steps left out of the KL plot
    tr, va = load(d / "metrics.jsonl"), load(d / "val.jsonl")
    out = ROOT / "results" / "plots" / d.name  # one folder per run, e.g. results/plots/grpo_smoke2/
    out.mkdir(parents=True, exist_ok=True)
    s = [r["step"] for r in tr]
    vs = [r["step"] for r in va]

    def fig(name, panels):
        f, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.6), squeeze=False)
        for ax, (title, series) in zip(axes[0], panels):
            for label, x, y, style in series:
                pts = [(a, b) for a, b in zip(x, y) if b is not None]
                if pts:
                    ax.plot(*zip(*pts), style, label=label, ms=4)
            ax.set_title(title)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        f.tight_layout()
        f.savefig(out / f"grpo_{name}.png", dpi=110)
        plt.close(f)

    g = lambda rows, k: [r.get(k) for r in rows]  # noqa: E731
    fig("reward", [("mean reward per turn", [("train", s, g(tr, "mean_reward"), "-o"), ("val", vs, g(va, "mean_reward"), "s--")]),
                   ("% valid new fruit", [("train", s, g(tr, "valid_rate"), "-o"), ("val", vs, g(va, "valid_rate"), "s--")])])
    cls = ["sentence", "filler", "silent", "repeat", "echo", "multi", "non_fruit", "overlap"]
    fig("rates", [("train turn rates", [(c, s, g(tr, f"{c}_rate"), "-") for c in cls]),
                  ("val turn rates", [(c, vs, g(va, f"{c}_rate"), "s-") for c in cls])])
    fig("latency", [("valid-fruit latency (s)", [("train median", s, g(tr, "latency_median"), "-o"),
                                                 ("train mean", s, g(tr, "latency_mean"), "-"),
                                                 ("val median", vs, g(va, "latency_median"), "s--")])])
    fig("kl", [("KL to reference (k3, per token)", [("text", s, [None if r["step"] in drop else r.get("kl_text") for r in tr], "-o"),
                                                       ("audio", s, g(tr, "kl_audio"), "-o")]),
               ("clip fraction", [("text", s, g(tr, "clipfrac_text"), "-"), ("audio", s, g(tr, "clipfrac_audio"), "-")])])
    fig("spread", [("groups with nonzero spread", [("groups", s, g(tr, "groups_with_spread"), "-o"),
                                                   ("turn positions", s, g(tr, "turn_positions_with_spread"), "-")]),
                   ("hack watch", [("fruits/turn", s, g(tr, "fruits_per_turn"), "-"), ("words/turn/10", s,
                                   [x / 10 if x is not None else None for x in g(tr, "words_per_turn")], "-"),
                                   ("turn-cap hits/64 games", s, [x / 64 for x in g(tr, "turn_cap_hits")], "-")])])
    fig("time", [("seconds per step", [("step", s, g(tr, "step_sec"), "-o"), ("generation", s, g(tr, "gen_sec"), "-"),
                                       ("log-probs", s, g(tr, "logp_sec"), "-"), ("update", s, g(tr, "train_sec"), "-")]),
                 ("peak memory (GB)", [("peak", s, g(tr, "peak_mem_gb"), "-o")])])
    if va:
        fig("asr", [("val Whisper fruit agreement", [("audio/text", vs, g(va, "asr_fruit_agreement"), "s-"),
                                                     ("class agreement", vs, g(va, "asr_class_agreement"), "s-")])])


if __name__ == "__main__":
    main()
