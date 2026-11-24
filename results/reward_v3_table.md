# Reward v3 unit table (`tests/test_reward_v3.py`, `configs/grpo_smoke3.yaml`)

Reply to the user's "Kiwi." (user speech ends at 10.7 s; "Apple." was said earlier).

| case | Moshi reply | class (v3) | latency | counted overlap | words | reward v3 |
|---|---|---|---|---|---|---|
| "Mango." fast (0.3 s) | Mango. | valid | +0.30 | no | 1 | **+1.20** |
| "Mango." slow (1.5 s) | Mango. | valid | +1.50 | no | 1 | **+1.00** |
| "Mango." in the grace window (starts 0.1 s before user ends) | Mango. | valid | -0.10 | no | 1 | **+1.00** |
| "Mango." over the user (starts 0.5 s before user ends) | Mango. | valid | -0.50 | yes | 1 | **+0.20** |
| "A mango!" | A mango! | valid | +0.45 | no | 2 | **+1.20** |
| "Mangoes?" | Mangoes? | valid | +0.30 | no | 1 | **+1.20** |
| "Dragon fruit." | Dragon fruit. | valid | +0.30 | no | 2 | **+1.20** |
| "I'll go with mango." | I'll go with mango. | sentence | +0.75 | no | 4 | **-0.15** |
| "The sixth one is a mango." | The sixth one is a mango. | sentence | +1.05 | no | 6 | **-0.25** |
| "Um, mango." | Um, mango. | sentence | +0.45 | no | 2 | **-0.10** |
| 13-word fruit fact | Mangoes are a tropical fruit that grow on tall trees in warm places. | sentence | +0.20 | no | 13 | **-0.60** |
| "Mango, kiwi, plum." | Mango, kiwi, plum. | multi | +0.30 | no | 3 | **-0.10** |
| "Okay." | Okay. | filler |  | no | 1 | **-0.30** |
| silence | (nothing) | silent |  | no | 0 | **-0.30** |
| "Carrot." | Carrot. | non_fruit |  | no | 1 | **-0.30** |
| echo: "Kiwi." | Kiwi. | repeat (echo) | +0.30 | no | 1 | **-0.50** |
| repeat: "Apple." (user said it earlier) | Apple. | repeat | +0.30 | no | 1 | **-0.50** |
| echo over the user: "Kiwi." starting 0.5 s before user ends | Kiwi. | repeat (echo) | -0.50 | yes | 1 | **-1.30** |
