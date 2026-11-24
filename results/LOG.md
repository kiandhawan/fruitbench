# LOG (newest first)

## 2026-11-24 — Step 1: lists, splits, voices, rendering, Mimi tokens
- **Fruits** (`fruitbench/lists.py`): 150 fruits the simulated user may say, each with accepted
  spellings/aliases (e.g. "kiwifruit", "dragonfruit"/"pitaya", "litchi") plus generated plurals; 32 more
  names are only recognised in Moshi's replies (generic "melon"/"berry", nuts, apple varieties, ...).
  Botanical edge cases (tomato, olive, pumpkin, nuts, rhubarb, ...) count as fruits when Moshi says them
  but the user never says them. Varieties (fuji apple, navel orange) are separate fruits, so "apple"
  after "fuji apple" is new; this keeps the matcher simple and is rare.
- **Phrasings**: 27 instructions, 16 "user starts" and 11 "Moshi starts" ("Fruit game: you go first").
- **Splits** (`data/splits.json`, seed 0): fruits 105 train / 15 val / 30 test; phrasings: a third of
  each starter type for test (5 + 4 = 9), 3 val, 15 train, so every split has both starter types.
- **Voices** (`data/voices.json`; Fish Official stock voices from the old project's list, no clones):
  train/val: Sheila (f, US, young, f3a2b900...), Megan (f, London Estuary, late 20s, 70569022...),
  Gareth (m, Australian, mid-50s, deep, e58a6b7e...), Rohan (m, Indian English, young, brisk, 2e8a3bd7...);
  test only: Cathal (m, Irish, mid-40s, warm, 3fdb457d...). 2 f + 3 m, 5 accents, young to mid-50s.
- **Rendering**: Fish s2-pro via moshi-finetune `render_lib.synth` (unchanged, imported), 24 kHz mono,
  silence-trimmed, -21 dBFS RMS. 885 clips (5 voices x (150 fruits + 27 phrasings)), 53 s wall.
  Fruit clips 0.32-1.82 s (median 0.78), phrasings 1.6-5.8 s (median 4.1).
- **Fish cost of step 1: $0.227** (15,130 UTF-8 bytes at $15/M; credit $59.0768 -> $58.8498, matching).
- **Mimi tokens**: every clip encoded once (`scripts/encode_clips.py`) with 0.96 s silence before it and
  0.32 s after, so its tokens start from the "user was quiet" encoder state they will have in the game.
  Digital silence is *not* one constant Mimi frame: after ~1 s each codebook flips between one or two
  near-tied values, so the game feeds the per-codebook mode as the silence frame. Splicing pre-encoded
  clips into silence is an approximation of a live encoder; checked against a live streaming encode of
  the smoke-test games (step 2).
- Scorer unit test (`tests/test_score.py`) on the real failure game from the task: greeting "Hey,
  what's up?", replies filler / filler / silent / valid "apple" (0.5 s) / filler / filler.

## 2026-11-23 — Step 0: setup
- Cloned kiandhawan/moshi -> /root/moshi-finetune (read-only reference; not modified) and the empty
  kiandhawan/fruitbench -> /root/fruitbench.
- Environment: reuse /root/moshi-finetune/.venv (`uv sync`) instead of a second venv: same pinned
  torch 2.6 / moshi, saves disk and install time.
- Secrets: FISH_API_KEY is read from ~/.fish_key only in the processes that need it; never logged.
- .gitignore excludes audio, Mimi token caches (*.npy, data/tokens), checkpoints, runs/, .venv, .env, key files.
- Checkpoint A: base Moshi loads in 4.9 s on one H200 and greets unprompted on a silent user channel
  ("Hey, how can I help you?" starting at 0.56 s); peak 17.7 GB. Unbatched x1.49 real time on this first
  run (includes kernel/graph warm-up); batched throughput is measured in step 4.
- Hardware: 4x NVIDIA H200 (143,771 MiB each), 2 TB RAM, 250 GB free disk (HF cache on the same
  non-persistent overlay).
