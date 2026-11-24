# Validation rescored under v3 (`configs/grpo_smoke3.yaml`)

| metric | base | smoke1_step20 | smoke2_step20 | smoke3_step20 |
|---|---|---|---|---|
| mean reward per turn | -0.538 | -0.618 | -0.078 | +0.348 |
| valid (bare fruit) | 0.2% | 5.9% | 26.1% | 52.7% |
| sentence (new fruit in other words) | 2.7% | 17.7% | 7.6% | 4.5% |
| repeat | 16.0% | 25.9% | 28.8% | 19.6% |
| echo | 11.6% | 16.9% | 12.2% | 8.3% |
| multi | 5.3% | 27.7% | 10.4% | 9.7% |
| filler | 33.6% | 15.4% | 10.0% | 5.1% |
| silent | 36.4% | 2.3% | 7.1% | 3.4% |
| non_fruit | 5.8% | 5.2% | 10.1% | 5.0% |
| overlap (raw) | 26.4% | 53.0% | 41.9% | 40.4% |
| overlap (counted, 0.2 s grace) | 11.3% | 22.1% | 9.6% | 9.8% |
| valid latency median (s) | 0.0 | 0.03 | 0.02 | 0.03 |
| words per turn median (mean) | 1 (5.21) | 6 (11.62) | 1 (2.64) | 1 (1.55) |
| 10 s turn cap per game | 0.66 | 1.88 | 0.57 | 0.03 |
| turns 1-6: valid | 0.2% | 8.3% | 34.0% | 58.5% |
| turns 1-6: sentence | 3.5% | 21.7% | 9.3% | 5.2% |
| turns 1-6: repeat | 13.2% | 19.7% | 18.7% | 12.2% |
| turns 1-6: overlap | 27.5% | 54.5% | 44.2% | 44.0% |
| turns 7-12: valid | 0.3% | 3.8% | 19.7% | 49.2% |
| turns 7-12: sentence | 1.2% | 14.5% | 5.8% | 3.3% |
| turns 7-12: repeat | 19.7% | 33.5% | 40.5% | 28.2% |
| turns 7-12: overlap | 25.7% | 49.5% | 38.3% | 35.5% |
