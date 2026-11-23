# LOG (newest first)

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
