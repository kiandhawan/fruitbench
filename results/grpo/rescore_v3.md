# Validation rescored under v3 (`configs/grpo_smoke3.yaml`)

| metric | base | smoke1_step20 | smoke2_step20 |
|---|---|---|---|
| mean reward per turn | -0.538 | -0.618 | -0.078 |
| valid (bare fruit) | 0.2% | 5.9% | 26.1% |
| sentence (new fruit in other words) | 2.7% | 17.7% | 7.6% |
| repeat | 16.0% | 25.9% | 28.8% |
| echo | 11.6% | 16.9% | 12.2% |
| multi | 5.3% | 27.7% | 10.4% |
| filler | 33.6% | 15.4% | 10.0% |
| silent | 36.4% | 2.3% | 7.1% |
| non_fruit | 5.8% | 5.2% | 10.1% |
| overlap (raw) | 26.4% | 53.0% | 41.9% |
| overlap (counted, 0.2 s grace) | 11.3% | 22.1% | 9.6% |
| valid latency median (s) | 0.0 | 0.03 | 0.02 |
| words per turn median (mean) | 1 (5.21) | 6 (11.62) | 1 (2.64) |
| 10 s turn cap per game | 0.66 | 1.88 | 0.57 |
| turns 1-6: valid | 0.2% | 8.3% | 34.0% |
| turns 1-6: sentence | 3.5% | 21.7% | 9.3% |
| turns 1-6: repeat | 13.2% | 19.7% | 18.7% |
| turns 1-6: overlap | 27.5% | 54.5% | 44.2% |
| turns 7-12: valid | 0.3% | 3.8% | 19.7% |
| turns 7-12: sentence | 1.2% | 14.5% | 5.8% |
| turns 7-12: repeat | 19.7% | 33.5% | 40.5% |
| turns 7-12: overlap | 25.7% | 49.5% | 38.3% |
