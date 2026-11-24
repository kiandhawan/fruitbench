# Base fruit-bench per-game outputs (no audio, no tokens)

Each game: `<id>.json` (spec, user turns with times and why they were inserted, Moshi's text tokens
with frame times, events/turn ends, Moshi's per-frame audio level in dBFS) and `<id>.score.json`
(per-turn class, fruit, latency, overlap, reward). Also `specs.json` and `timing_worker*.json`.

- `test_s1/`: the 200 test-split games, seed 1 (numbers in `../base_fruitbench.md`)
- `grpo_check/`: 25 train-split starts x 16 games (`_rNN` = repeat), seed 1
- `timing32/`: 32 test games, seed 2, one batch on one GPU (timing run)

Audio: GitHub releases `smoke-test` and `base-best-worst` (best/worst 5 only). Rescore with
`python -c "from fruitbench.score import score_game"` or `scripts/report.py --test results/base_games/test_s1`.
Future runs of `scripts/run_games.py` archive automatically to `results/games/<out name>/`.
