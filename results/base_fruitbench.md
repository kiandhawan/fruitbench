# Base fruit-bench (kyutai/moshiko-pytorch-bf16, no training)

Setup: 200 test-split games, seed 1 (test fruits = 20 of the 80 common fruits, 9 held-out
phrasings, held-out voice Cathal); 1/3 of games the user asks Moshi to start. Moshi's default
sampling (temp 0.8 audio / 0.7 text, top-k 250 / 25). User audio is live-encoded with streaming Mimi.
Turn end from Moshi's audio (0.8 s below -45 dBFS after speaking, 2.5 s without speech, 10 s cap;
the user waits if Moshi resumes during its 0.3-0.8 s gap). Scored turns: replies to the 12 user
fruits, plus the opening reply in Moshi-starts games. Full numbers: `base_fruitbench.json`.

## Headline (2467 scored turns)

| metric | value |
|---|---|
| **valid new fruit** | **6.0%** (149 turns) |
| latency of valid fruits, median / mean | **0.13 s / 0.78 s** (p10 -0.03, p25 0.06, p75 1.12, p90 2.73; 11% negative = fruit starts before the user finished) |
| repeat (already said) | 18.9% (echo of the user's fruit: 14.9% of all turns) |
| non-fruit | 4.6% |
| filler / meta talk only | 34.4% |
| silent | 28.6% |
| more than one fruit | 7.5% |
| overlap with user speech | 12.4% of turns (19.5% of valid turns) |
| games with at least one valid fruit | 43.5% |
| mean provisional reward per turn | 0.061 |

By starter: user starts 6.8% valid (133 games), Moshi starts 4.7% (67 games). The opening reply when
Moshi is asked to start: filler 38, silent 15, valid 10, multi 4 (of 67).

Latency is bimodal: when Moshi does name a new fruit, it is usually immediately (median 0.13 s,
often a comma list "Apple, | banana, | grape," one item per turn); the rest come after chat.

## 10 most common filler replies

| reply | count |
|---|---|
| okay | 151 |
| yeah | 28 |
| yes | 22 |
| oh okay | 16 |
| oh | 15 |
| oh i know | 11 |
| yes that's correct | 11 |
| no | 11 |
| oh that's a good one | 10 |
| yeah yeah yeah | 9 |

(848 filler replies in total; the rest are a long tail of acknowledgements and chat.) Most common
greetings (before the instructions): "Hi, how is it going?" 20, "Hello, how is it going?" 16,
"Hi there, how can I help you?" 15. As in the real failure game, Moshi is in listener mode: it
acknowledges ("Okay", "Yes, that's correct"), echoes the user's fruit, or stays silent.

## GRPO readiness (25 train starts x 16 games, seed 1)

| metric | value |
|---|---|
| groups whose 16 total rewards are not all identical | **25 / 25 (100%)** |
| groups whose per-turn reward vectors are not all identical | **25 / 25 (100%)** |
| (group, turn position) pairs where the 16 rewards differ | 79.9% |
| mean within-group std of total reward | 0.072 |
| valid rate on these train games | 8.0% |

Every group has reward variance, so GRPO advantages are non-zero from the start, but the signal is
thin (6-8% valid turns; std 0.07).

## Timing and memory (1x H200)

- 32 games in one batch: **74.9 s wall** for a batch whose longest game ran 134.5 s (the batch runs
  until its longest game ends) = 0.55x real time; **peak 67.1 GB**.
- Batch 50: 92-98 s wall per batch, peak 95.9 GB. 200 test games on 4 GPUs: 118 s end to end.
- Moshi's audio is always decoded (the turn-end rule listens to it).

## Text stream vs Whisper (step 3, 20 test games)

Per-turn class agreement 85.8% of 247 turns (85.0% with the same fruit); 95.6% of fruit mentions in
the text stream are in the Whisper transcript of the same game. Most disagreements are Whisper
timestamps moving a word into the neighbouring turn, or misheard short words (`results/LOG.md`).

## Turn-taking check

2.2% of the 2600 user lines start while Moshi's audio was speech in the previous 0.3 s; these are
mostly the 87 turns that hit the 10 s cap (Moshi monologue). 646 gaps were cancelled because Moshi
resumed speaking.

## Games to listen to

Transcripts/timelines: `results/best_worst/`; MP3s (left = Moshi, right = user): GitHub release
`base-best-worst`.

| game | reward | valid / repeat / filler / silent / multi / non-fruit |
|---|---|---|
| best1 test_s1_0195 | 0.49 | 6 / 3 / 0 / 3 / 0 / 0 |
| best2 test_s1_0048 | 0.47 | 6 / 6 / 0 / 0 / 0 / 0 |
| best3 test_s1_0136 | 0.41 | 5 / 5 / 1 / 0 / 1 / 0 |
| best4 test_s1_0121 | 0.33 | 4 / 6 / 1 / 0 / 0 / 1 |
| best5 test_s1_0129 | 0.28 | 4 / 3 / 2 / 3 / 0 / 0 |
| worst1 test_s1_0001 | 0 | 0 / 0 / 13 / 0 / 0 / 0 |
| worst2 test_s1_0036 | 0 | 0 / 0 / 13 / 0 / 0 / 0 |
| worst3 test_s1_0041 | 0 | 0 / 0 / 13 / 0 / 0 / 0 |
| worst4 test_s1_0088 | 0 | 0 / 12 / 1 / 0 / 0 / 0 |
| worst5 test_s1_0094 | 0 | 0 / 2 / 9 / 0 / 0 / 2 |

(Worst = reward 0; among the many zero-reward games, those where Moshi talked but failed are shown,
as they are more informative than silence.)
