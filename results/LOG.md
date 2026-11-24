# LOG (newest first)

## 2026-11-24 — GRPO implementation and 20-step smoke run (`configs/grpo_smoke.yaml`)
- **Code**: `scripts/grpo_train.py` (torchrun, 4 ranks, DDP), `fruitbench/rewards.py` (per-turn reward
  from the unchanged scorer classes + config weights), `scripts/plot_grpo.py` -> `results/plots/grpo_*.png`.
  Game loop: `GameRunner(models=..., record_codes=True)` reuses the same loop with the training model
  (streaming entered per run) and records the delayed-token column LMGen reads each step; aligned
  codes `codes[k][t] = D[t+1+d_k][k]` are what `LMModel.forward` re-delays. Checks: 0 misaligned text
  tokens vs the game JSON; sampled tokens are inside the sampler's top-k under teacher forcing (text
  100%, audio 99.98%); the evaluation path still reproduces the saved smoke games token for token.
- **Setup**: fresh LoRA r64 (scaling 2.0, all 337 linear layers, 194M trainable of 7.9B; B zeroed so
  step 0 = base), fp32 master weights for the LoRA (moshi-finetune `mixed_precision.py`), bf16 compute,
  gradient checkpointing. Per step 4 train starts x 16 games (one group per GPU), 6 user turns,
  default sampling (no temperature change needed: 100% of groups had spread every step).
  Advantage per turn index within the group; applied to Moshi's text + 8 audio tokens of the frames
  from the end of user line k to the end of user line k+1; loss = mean over text tokens + mean over
  audio tokens of [PPO-clip(0.2) + 0.04 * k3 KL to LoRA-off]. Interpretation choice: "one optimizer
  pass per batch" = one pass over the step's 64 games split into 4 minibatches (4 AdamW steps per
  GRPO step), so the clip is active after the first minibatch (clip fraction ~1-3%); old log-probs
  recomputed with the rollout policy; log-probs use the sampling temperatures (text 0.7, audio 0.8).
- **Smoke results** (20 steps; reward multi_new = +0.25 here):
  - Train: valid new fruit 4.8% (step 1) -> 37.0% (step 20),
    peak 54.8%; silent 35% -> 6%;
    mean reward -0.31 -> +0.19; KL text/audio 0.037/0.030.
  - **Validation (100 val games, 12 turns)**: valid 3.2% -> **24.7%**, mean reward
    -0.35 -> -0.04, silent 36% -> 2%, filler
    33% -> 14%, repeat 16% -> 25% (echo
    12% -> 17%), multi 6% -> 29%, overlap
    26% -> 53%; valid latency median 0.83 -> 0.41 s.
    Whisper fruit agreement 90.3% -> 94.9% (no flag).
  - **Hack watch** (val step 20 vs 0): talking over the user doubles (overlap 26% -> 53%; 95% of
    overlapping replies start before the user's fruit ends); fruit lists (multi 5.5% -> 29%, median 3
    fruits, max 11); longer turns (median 1 -> 6 words, p90 17 -> 29; 10 s turn cap 66 -> 188 hits per
    100 games, e.g. "Sweet flavors, sugary treats, a taste of heaven."). Garbled audio: not seen
    (Whisper agreement 95%). The multi_new 0.0 change for the full run addresses the lists; overlap
    and long turns are not yet penalised beyond the -0.3 overlap term.
  - Time: 35.9-57.5 s per step (mean 51; generation ~70%), peak 45 GB per GPU.
- Training rollouts are not in git: per step `runs/grpo/<run>/rollouts/step_XXX.tar.gz` on disk; git
  gets `metrics.jsonl` (one row per step), `val.jsonl`, validation game archives and plots. Samples:
  release `grpo-smoke` (best/median/worst of the 64 step-20 training games).

## 2026-11-24 — Reward change before the full GRPO run: multi-fruit turn with a new fruit 0.25 -> 0.0
- `configs/grpo_full.yaml`: `reward.multi_new` = **0.0** (was +0.25 in `configs/grpo_smoke.yaml`, the
  value the 20-step smoke run trains with; that run's saved config keeps 0.25). Requested by the user:
  a multi-fruit turn that adds a new fruit should stay above filler/silent (-0.3) but far below a
  single new fruit (+1.0 / +1.2), so GRPO is not pulled toward rapid fruit lists ("Blueberry, plum,
  grape, orange, lemon, banana, tomato." got +0.25 in smoke step 1). A multi-fruit turn with no new
  fruit stays -0.3; the overlap penalty (-0.3) still applies on top. Nothing else differs between
  the two configs except run name and steps (200).

## 2026-11-24 — Check before GRPO: "I'll go first" turns (`scripts/go_first_check.py`)
- In the 200 base test games, 7 Moshi turns (7 games, 0.3% of turns) contain "I'll go first" (no
  "let me start"/"me first"). Moshi said a fruit later in that turn in 2/7 (0123 "...is an apple",
  0192 "oranges are round", where the user waited through 3 cancelled gaps). By the agreed
  definition (Moshi's audio resumes within 1 s after the next user line starts) **4/7 are cut-offs**,
  above the 10% threshold, so GRPO is on hold for the user's decision.
- Detail: in all 4 the audio resumes 0.64-0.88 s after the user line starts, i.e. when the user's
  0.6-0.8 s fruit clip is ending, and what Moshi says reacts to the user ("Yes, that's correct",
  "That's correct", "Okay. I'll name the fruit", "apple."). Before each of these user lines Moshi had
  been silent >= 0.8 s + the 0.3-0.8 s gap (1.1-1.6 s). Only 0136 ("Okay, I'll go first." then, after
  the user's "Jackfruit", "apple.") looks like Moshi's own fruit arriving late.

## 2026-11-24 — Per-game outputs are kept in git
- The base runs' per-game JSONs (summaries with Moshi's timed text tokens, per-turn scores; no audio,
  no .npy tokens) are in `results/base_games/{test_s1,grpo_check,timing32}` (7 + 15 + 1 MB).
- From now on `scripts/run_games.py` archives every run automatically to `results/games/<out name>/`
  (`fruitbench/archive.py`; one .tar.gz instead if the kept files exceed 20 MB) once all workers
  exit 0, so every evaluation can be rescored later without re-running Moshi. Commit it with the run.

## 2026-11-24 — Step 4/5: base fruit-bench, GRPO readiness, timing (no training)
- Runs: 200 test games seed 1 (4 GPUs, batch 50, 118 s); GRPO check 25 train starts x 16 (`--repeat
  16`, batch 50, 214 s); timing 32 test games seed 2 on 1 GPU at batch 32. Report:
  `scripts/report.py` -> `results/base_fruitbench.md/.json` (it rescores every game with the current
  scorer). Batch 50 instead of 32 for the big runs so each GPU gets whole batches (200 = 4 x 50).
- **Base: 6.0% valid new fruit** of 2467 scored turns; filler 34.4%, silent 28.6%, repeat 18.9%
  (echo 14.9%), multi 7.5%, non-fruit 4.6%; overlap 12.4%. Valid latency median 0.13 s, mean 0.78 s.
  Top filler "okay" (151). Listener mode reproduced, as in the real failure game.
- **GRPO readiness**: 25/25 groups have non-identical total rewards and non-identical per-turn reward
  vectors; 79.9% of (group, turn) positions vary; mean within-group std of the total 0.072. Signal
  exists everywhere but is thin; reward shaping (e.g. partial credit for answering at all, or a
  penalty for silence/filler) can be considered when training starts.
- **Timing**: 32 games / 1 H200: 74.9 s wall (longest game 134.5 s; 0.55x real time), peak 67.1 GB;
  batch 50 peaks at 95.9 GB.
- Turn-taking at scale: 2.2% of user lines start while Moshi's audio is speech (mostly 10 s caps);
  646 gaps cancelled because Moshi resumed. The timeline's text heuristic over-flags comma lists
  ("Apple, | banana,"), which are one fruit per turn and end with a real audio pause.
- Best/worst 5 games: transcripts in `results/best_worst/`, MP3s in GitHub release `base-best-worst`
  (not in git). Stopping here as instructed: no training started.

## 2026-11-24 — Step 3: text-stream scorer vs Whisper on Moshi's audio
- `scripts/asr_check.py` (separate `.venv-asr`: faster-whisper 1.1.1, large-v3, fp16, VAD on, beam 5,
  English) transcribes Moshi's channel of 20 randomly chosen (seed 0) base test games, turns Whisper's
  words into a pseudo text stream (one word token at each word's start) and scores it with the same
  scorer. Output: `results/asr_check.json`.
- **Per-turn class agreement: 85.8%** of 247 scored turns (85.0% same class and same fruit). By text
  class: silent 94%, repeat 84%, filler 78%, valid 78%, multi 78%, non_fruit 45%.
- **Timing-free check: 95.6%** of the 183 fruit mentions in the text stream are also in the Whisper
  transcript of the same game (96.7% the other way round).
- Reading the disagreements, most are not text/audio mismatches: (a) Whisper's word timestamps sit
  later than the text tokens, so a word near a turn boundary moves into the next reply window and
  shifts a chain of turns (text "Grapefruit | Jackfruit | Lemon" vs ASR "Jackfruit | Lemon | Orange");
  (b) Whisper drops or mishears short isolated words ("Vec", "zing", "Kiwi." -> nothing, "Prickly pear"
  -> "High quit pair"); (c) a few genuine text/audio differences ("A comfrey fruit" vs audio "A kumquat
  fruit"; "durian" vs "gruyere"). non_fruit is the weakest class because it is mostly one-word
  nonsense replies that Whisper hears differently. Decision: keep the text stream as the reward source
  (frame-exact, no ASR noise); re-check agreement on trained checkpoints, since training could push the
  text and audio streams apart.

## 2026-11-24 — Step 2 revision (user feedback after checkpoint B)
- **Turn end now listens to Moshi's audio, for every Moshi turn.** The 0.5 s text rule cut Moshi off
  mid-sentence (text comes ~1 token per word with 0.5-0.8 s gaps inside sentences): with the
  mid-sentence check below, 16 of 52 user lines in the old smoke games started while Moshi was
  mid-sentence. New rule: a frame is Moshi speech when the RMS of its decoded 80 ms audio is above
  **-45 dBFS**; the turn ends after **0.8 s** of consecutive non-speech frames once Moshi has produced
  >= 2 speech frames after the user stopped, after 2.5 s with no speech, or at the 10 s cap.
  Calibration (smoke audio, 2187 frames): silence floor -74 dBFS (most frames), a low-noise hump at
  -65..-50, speech frames (within 0.3 s after a text token) median -29 dBFS, 25th pct -39; the valley is
  at -50..-45. At -45 every loud frame had a text token within 1 s (no false speech).
- **The user waits if Moshi resumes during the 0.3-0.8 s gap.** With only the audio rule, 9 of 52 user
  lines still started mid-sentence: Moshi paused > 0.8 s inside a sentence and resumed during the
  gap ("my favorite fruit, the lemon, | has high sugar content"). Now speech during the gap cancels
  it and Moshi's turn continues (still capped at 10 s after the user's speech). Result on the re-run
  smoke test: **1 of 52** user lines starts mid-sentence, and that is the 10 s cap (Moshi monologue;
  the user barges in by design). Check: `scripts/timeline.py` flags a user line if Moshi's audio was
  speech in the 0.3 s before it, or Moshi's last word before it has no end punctuation and its next
  word follows within 1 s. Caveat: the audio decision for slot i+1 is made after step i, i.e. up to
  80 ms of look-ahead vs a strictly causal listener; the decision only gates gap timing.
- Moshi's audio is now always decoded (the rule needs it); `--no-audio` only skips saving WAVs. Per
  game the JSON stores Moshi's per-frame level (`moshi_dbfs`).
- **The simulated user says only 80 common fruits** (`COMMON_FRUITS`, hand-ranked by familiarity;
  all already rendered, no new Fish cost), split **48 train / 12 val / 20 test** (deterministic, seed
  0). Val needed >= 12 (one game has 12 user fruits); a first 52/8/20 split ran out in a val game. If
  Moshi says so many pool fruits that the split's pool runs out, the user falls back to a shuffled
  backup pool of the other 70 rendered fruits that are in no split's pool (no train/test leak). The
  scorer still accepts any real fruit (FRUITS + EXTRA_RECOGNISED, 188 canonical names).
- **Scorer**: the fuzzy echo matcher turned "benefits" into "breadfruit"; fuzzy matches now also need
  consonant skeletons with the same first letter matching >= 0.75 (still catches lichie/lychee,
  samumberry/salmonberry, kumkwat/kumquat, pomegranite/pomegranate). Provisional reward: a multi-fruit
  reply gets 0.25 only if it contains a new fruit (base Moshi played shopkeeper: "I'll take 10
  apricots, 4 figs, 2 blood oranges", repeating the user's fruits).
- Smoke re-run (val, seed 1, 4 games, batch 4, 1 GPU): 56-73 s games, 22.2 s wall, 22.3 GB peak.
  Scored turns: valid 1, repeat 2, non_fruit 2, filler 9, silent 31, multi 5 (of 50). Base Moshi is
  even more often silent with the audio rule (it is no longer interrupted mid-reply, and often says
  nothing after a fruit). MP3s: GitHub release `smoke-test`.

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
