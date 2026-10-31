# - WSL and Docker data on E:, since C: had 35 GB free.
- Code lives in ~/code inside WSL, not /mnt/c or /mnt/e: cross-system file access is slow.
- Python 3.12 via uv, one environment per repo.
- VS Code for these repos: its WSL support is the most reliable.
- NVIDIA free API first; Together ($15) only if NVIDIA can't cover tool calling or Week 3 load tests.
- RAM upgraded to 16 GB (2 × 8 GB); WSL memory raised to 8 GB, so Weeks 4–5 run locally without a cloud VM.
- Cost per query = input_tokens × input_price/1M + output_tokens × output_price/1M, using each model's real token counts (dataset has no cost column to check against).
- Score metric = mean of `performance` (squad keeps partial F1 credit); always labelled "avg score", never "accuracy".
- Oracle = per query, the model with the highest performance; ties broken by lowest cost. No correctness threshold.
- Queries no model answers: all tie at 0, so the oracle picks the cheapest model (follows from the rule above).
