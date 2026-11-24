# Validation rescored with scorer v3 and the rewards of `configs/grpo_full.yaml`

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
