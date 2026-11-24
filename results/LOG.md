# LOG (newest first)

## 2026-11-24 — Step 2: streaming game loop; checkpoint B smoke test
- **Loop** (`fruitbench/game.py`, `scripts/run_games.py`): B games in lockstep on one GPU, one 80 ms
  frame per step, one worker process per GPU. Output-frame convention from moshi-finetune
  `run_inference.py`: the frame LMGen returns after input frame i goes in slot i+1, so the simulated user
  only reacts to Moshi text it could have heard. Game: Moshi's greeting turn from t=0 (base Moshi greets
  unprompted), instructions, Moshi's turn, then 12 user fruits each followed by Moshi's turn; 1/3 of games
  "Moshi starts". Turn end: 0.5 s with no text after Moshi spoke, 2.5 s of silence, or a 10 s cap; then a
  0.3-0.8 s gap and the next fruit in the game's shuffled order that neither side has said.
- **Turn-end fix**: at first, text emitted *during* the user's clip started the 0.5 s clock. In a
  Moshi-starts smoke game the tail of the greeting ("what's going on?") ran into the instructions, so
  Moshi's turn ended 0.5 s after the instructions, before it could answer. Now only text after the
  user's speech ends counts toward ending the turn; talk over the user is scored as overlap.
- **User audio is encoded live** (default `--user-audio live`): the right-channel PCM (clips placed at
  their frames, zeros elsewhere) goes through streaming Mimi each frame, in the batch, like
  moshi.server. Splicing the pre-encoded clip tokens into a constant silence code (the step-1 plan)
  agreed poorly with a live encode of the same audio (`scripts/check_tokens.py` on the smoke games:
  ~27% of silence frames and ~58% of speech-frame tokens, codebook 0 85%), because Mimi's encoder has
  seconds of context: silence after speech does not encode like steady silence. Moshi would have heard
  a user channel no real conversation produces. Live encoding costs no measurable time (4 games:
  13.8 s either way) and the fed tokens match a single-stream server-style encode 97.5-98.9%
  (codebook 0: 99.6-100%; the rest are batch-numerics near-ties). The pre-encoded tokens remain
  available (`--user-audio tokens`) but are not used for results. Behaviour changed noticeably: with
  spliced tokens Moshi chatted ("I've heard of cloudberries, but I've never had them") and once named 6
  fruits; with live audio it behaves like the real failure (listener mode: "Mm hmm", "Now it's your
  turn", "Yes, that's correct", echoes the user's fruit, silence).
- **Scorer changes from reading the smoke timelines** (`fruitbench/score.py`):
  - Words are grouped into utterances (gaps < 0.5 s) and whole utterances are assigned to the turn in
    which they start; per-word windows split "I like blood | oranges" across two turns.
  - Generic "berry", "citrus", "nut" are no longer fruit names ("not a berry" was valid).
  - Echoes of rare fruits are misspelled in Moshi's text ("Lichie", "samumberry", "kolamansee",
    "muubell"); fuzzy match (ratio >= 0.6, words of 5+ letters, non-filler) against fruits *already
    said* turns them into repeats (`echo` flag when it is the user's fruit). Not against all fruits:
    "speech"/"peach" would be false hits.
  - non_fruit = names something else ("go with / I'll say / how about X", or <= 4 words with <= 2
    content words); more meta words count as filler ("correct", "out of ideas", "stuck", ...).
- **Smoke test** (val split, seed 1, batch 4, GPU 0): 4 games (2 user-starts, 2 Moshi-starts), 43-45 s
  each, 13.8 s wall for the batch (x0.30 real time), 22.3 GB peak. 0 valid fruits in 49 scored turns;
  filler 27, silent 12, repeat 11 (8 of them echoes). Timelines: `results/smoke/timelines.txt`.
- Open point for review: the 0.5 s rule on *text* tokens ends Moshi's greeting mid-sentence in 3 of 4
  games ("Hi there, | what's going on?": text tokens arrive one per word with pauses of 0.5-0.8 s inside
  a sentence), so the instructions start while Moshi is still talking.

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
