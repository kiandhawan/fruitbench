# Full GRPO run summary (grpo_full)

Config `configs/grpo_full.yaml`; trainer ended by early stopping (watchdog): best validation mean reward +1.020 (step 120) not improved for 4 validations (last: step 160, +1.015); 160 training steps logged, 17 validations.
Validation: 100 val games (12 user turns, scorer v3, the run's reward) every 10 steps; training curves:
`results/plots/grpo_full/grpo_*.png`.

## Validation curve

| step | bare valid | sentence | repeat | silent | filler | multi | overlap counted | talk-over (audio) | chat replies w/ fruit | train audio KL | train text KL | mean reward | Whisper fruits | eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.2% | 2.7% | 16.0% | 36.4% | 33.6% | 5.3% | 12.5% | 13.4% | 0.0% | - | - | -0.566 | 88.4% | no |
| 10 | 18.8% | 1.4% | 20.8% | 35.7% | 9.5% | 9.1% | 4.1% | 5.2% | 0.0% | 0.0038 | 0.0203 | -0.133 | 98.1% | yes |
| 20 | 74.1% | 1.1% | 12.2% | 1.9% | 2.2% | 3.2% | 2.3% | 3.4% | 0.0% | 0.0066 | 0.0544 | +0.732 | 91.8% | yes |
| 30 | 78.1% | 1.6% | 10.5% | 1.9% | 0.2% | 3.6% | 1.5% | 2.8% | 0.0% | 0.0093 | 0.0791 | +0.813 | 94.7% | yes |
| 40 | 83.9% | 0.9% | 9.1% | 1.6% | 0.6% | 1.9% | 1.3% | 2.7% | 0.0% | 0.0074 | 0.0857 | +0.911 | 85.6% | yes |
| 50 | 85.2% | 0.3% | 9.4% | 0.6% | 1.3% | 2.1% | 1.5% | 2.2% | 0.0% | 0.0088 | 0.0908 | +0.926 | 93.4% | yes |
| 60 | 85.8% | 0.3% | 9.8% | 1.8% | 0.2% | 1.2% | 1.5% | 3.2% | 0.0% | 0.0078 | 0.1133 | +0.940 | 93.4% | yes |
| 70 | 86.9% | 0.2% | 9.1% | 0.7% | 0.2% | 2.2% | 1.6% | 3.4% | 0.0% | 0.0083 | 0.0860 | +0.957 | 89.0% | yes |
| 80 | 83.3% | 0.4% | 12.5% | 0.4% | 0.1% | 2.7% | 1.5% | 2.8% | 0.0% | 0.0074 | 0.0910 | +0.905 | 92.8% | yes |
| 90 | 87.0% | 0.5% | 9.8% | 0.7% | 0.2% | 0.8% | 0.6% | 2.9% | 0.0% | 0.0081 | 0.1040 | +0.966 | 89.5% | yes |
| 100 | 86.3% | 0.5% | 10.9% | 0.7% | 0.0% | 1.4% | 1.1% | 3.6% | 0.0% | 0.0074 | 0.1083 | +0.957 | 83.7% | yes |
| 110 | 86.5% | 1.1% | 9.5% | 0.3% | 0.2% | 2.3% | 1.6% | 3.6% | 0.0% | 0.0070 | 0.1022 | +0.958 | 92.9% | yes |
| 120 | 89.9% | 0.4% | 8.0% | 0.2% | 0.1% | 0.9% | 0.6% | 2.7% | 0.0% | 0.0070 | 0.0999 | +1.020 | 91.5% | yes |
| 130 | 89.0% | 0.2% | 8.8% | 0.4% | 0.0% | 1.1% | 0.7% | 2.4% | 0.0% | 0.0072 | 0.1055 | +1.006 | 86.5% | yes |
| 140 | 88.3% | 0.1% | 8.0% | 0.1% | 0.0% | 2.2% | 1.3% | 2.8% | 0.0% | 0.0060 | 0.0844 | +0.992 | 90.6% | yes |
| 150 | 89.0% | 0.2% | 8.2% | 0.2% | 0.0% | 0.9% | 0.7% | 2.0% | 0.0% | 0.0061 | 0.1065 | +1.005 | 91.2% | yes |
| 160 | 89.8% | 0.5% | 7.3% | 0.4% | 0.1% | 0.6% | 0.8% | 2.8% | 0.0% | 0.0064 | 0.1159 | +1.015 | 89.0% | yes |

## Chosen checkpoint

- Rule: highest validation mean reward among checkpoints with ordinary-conversation fruit rate <= 10% and training audio KL < 0.03 at that step.
- **Final: step 120** (mean reward +1.020, bare valid 89.9%).
- Overall best on validation mean reward: step 120 (+1.020) - the same checkpoint.
- LoRAs: draft release `checkpoints` (lora_final.safetensors, lora_best.safetensors).

## Test: 200 test games (seed 1), base Moshi vs final (step 120), same games

- Bare valid (v3): base 2.5% -> final **88.2%**; valid under the base benchmark's v1 rules: 6.0% -> 88.3%.
- Mean reward per turn (run's reward): -0.471 -> +0.997.
- Before/after MP3s of 6 test games: release `grpo-final`.

Full table (scorer v3, run's reward):

| metric | base | final_step120 |
|---|---|---|
| mean reward per turn | -0.471 | +0.997 |
| valid (bare fruit) | 2.5% | 88.2% |
| sentence (new fruit in other words) | 3.2% | 0.0% |
| repeat | 18.9% | 9.5% |
| echo | 14.9% | 1.1% |
| multi | 7.4% | 0.8% |
| filler | 34.6% | 0.0% |
| silent | 28.6% | 0.1% |
| non_fruit | 4.8% | 1.3% |
| overlap (raw) | 12.4% | 1.4% |
| overlap (counted, 0.16 s grace) | 8.3% | 0.7% |
| valid latency median (s) | 0.08 | 0.119 |
| words per turn median (mean) | 1 (4.55) | 1 (1.02) |
| valid fruits starting before the user finishes | 16.1% | 0.9% |
| any fruit turn starting before the user finishes | 11.9% | 1.4% |
| valid latency p10 / p50 / p90 (s) | -0.10 / +0.08 / +0.53 | +0.06 / +0.12 / +0.21 |
| 10 s turn cap per game | 0.435 | 0.005 |
| turns 1-6: valid | 3.3% | 96.7% |
| turns 1-6: sentence | 2.6% | 0.0% |
| turns 1-6: repeat | 15.4% | 2.2% |
| turns 1-6: overlap | 13.8% | 1.6% |
| turns 7-12: valid | 1.8% | 80.2% |
| turns 7-12: sentence | 3.2% | 0.0% |
| turns 7-12: repeat | 23.3% | 17.3% |
| turns 7-12: overlap | 9.6% | 0.2% |


## Flags (logged, never stopped the run)

- validation bare valid dropped 2 validations in a row: 89.9% -> 89.0% -> 88.3%
- FLAG: Whisper heard only 84% of the text stream's fruits (< 90%): possible reward hacking
- FLAG: Whisper heard only 86% of the text stream's fruits (< 90%): possible reward hacking
- FLAG: Whisper heard only 88% of the text stream's fruits (< 90%): possible reward hacking
- FLAG: Whisper heard only 89% of the text stream's fruits (< 90%): possible reward hacking
- FLAG: Whisper heard only 90% of the text stream's fruits (< 90%): possible reward hacking

Validation flags for Whisper are expected around 81-88% even for base Moshi (see results/LOG.md).
