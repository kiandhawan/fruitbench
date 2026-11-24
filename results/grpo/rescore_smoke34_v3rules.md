# Validation rescored with scorer v3 and the rewards of `configs/grpo_smoke3.yaml`

| metric | s3_step0 | s3_step20 | s4_step0 | s4_step10 | s4_step20 |
|---|---|---|---|---|---|
| mean reward per turn | -0.538 | +0.348 | -0.538 | -0.245 | +0.207 |
| valid (bare fruit) | 0.2% | 52.7% | 0.2% | 7.9% | 42.9% |
| sentence (new fruit in other words) | 2.7% | 4.5% | 2.7% | 1.1% | 3.1% |
| repeat | 16.0% | 19.6% | 16.0% | 7.5% | 22.1% |
| echo | 11.6% | 8.3% | 11.6% | 3.6% | 10.1% |
| multi | 5.3% | 9.7% | 5.3% | 4.5% | 9.8% |
| filler | 33.6% | 5.1% | 33.6% | 5.0% | 7.1% |
| silent | 36.4% | 3.4% | 36.4% | 69.7% | 6.2% |
| non_fruit | 5.8% | 5.0% | 5.8% | 4.4% | 8.8% |
| overlap (raw) | 26.4% | 40.4% | 26.4% | 10.0% | 33.1% |
| overlap (counted, 0.2 s grace) | 11.3% | 9.8% | 11.3% | 4.7% | 10.1% |
| valid latency median (s) | 0.0 | 0.03 | 0.0 | 0.06 | 0.06 |
| words per turn median (mean) | 1 (5.21) | 1 (1.55) | 1 (5.21) | 0 (0.85) | 1 (1.47) |
| valid fruits starting before the user finishes | 33.3% | 38.0% | 33.3% | 29.9% | 29.7% |
| any fruit turn starting before the user finishes | 19.8% | 39.1% | 19.8% | 27.9% | 31.9% |
| valid latency p10 / p50 / p90 (s) | -0.03 / +0.00 / +0.05 | -0.13 / +0.03 / +0.13 | -0.03 / +0.00 / +0.05 | -0.27 / +0.06 / +0.57 | -0.11 / +0.06 / +0.16 |
| 10 s turn cap per game | 0.66 | 0.03 | 0.66 | 0.06 | 0.03 |
| turns 1-6: valid | 0.2% | 58.5% | 0.2% | 9.3% | 49.8% |
| turns 1-6: sentence | 3.5% | 5.2% | 3.5% | 1.2% | 3.5% |
| turns 1-6: repeat | 13.2% | 12.2% | 13.2% | 7.3% | 13.7% |
| turns 1-6: overlap | 27.5% | 44.0% | 27.5% | 13.7% | 34.8% |
| turns 7-12: valid | 0.3% | 49.2% | 0.3% | 6.8% | 38.3% |
| turns 7-12: sentence | 1.2% | 3.3% | 1.2% | 0.7% | 2.5% |
| turns 7-12: repeat | 19.7% | 28.2% | 19.7% | 8.0% | 31.8% |
| turns 7-12: overlap | 25.7% | 35.5% | 25.7% | 6.5% | 29.8% |
