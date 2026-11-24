# Validation rescored with scorer v1 and the rewards of `configs/grpo_smoke.yaml`

| metric | base | final_step120 |
|---|---|---|
| mean reward per turn | -0.263 | +1.005 |
| valid (bare fruit) | 6.0% | 88.3% |
| sentence (new fruit in other words) | 0.0% | 0.0% |
| repeat | 18.9% | 9.4% |
| echo | 14.9% | 1.1% |
| multi | 7.5% | 0.8% |
| filler | 34.4% | 0.0% |
| silent | 28.6% | 0.1% |
| non_fruit | 4.6% | 1.3% |
| overlap (raw) | 12.4% | 1.4% |
| overlap (counted, 0 s grace) | 12.4% | 1.4% |
| valid latency median (s) | 0.132 | 0.119 |
| words per turn median (mean) | 1 (4.55) | 1 (1.02) |
| valid fruits starting before the user finishes | 11.4% | 0.9% |
| any fruit turn starting before the user finishes | 12.1% | 1.4% |
| valid latency p10 / p50 / p90 (s) | -0.03 / +0.13 / +2.73 | +0.06 / +0.12 / +0.21 |
| 10 s turn cap per game | 0.435 | 0.005 |
| turns 1-6: valid | 6.2% | 96.8% |
| turns 1-6: sentence | 0.0% | 0.0% |
| turns 1-6: repeat | 15.5% | 2.1% |
| turns 1-6: overlap | 13.8% | 1.6% |
| turns 7-12: valid | 5.3% | 80.2% |
| turns 7-12: sentence | 0.0% | 0.0% |
| turns 7-12: repeat | 23.2% | 17.3% |
| turns 7-12: overlap | 9.6% | 0.2% |
