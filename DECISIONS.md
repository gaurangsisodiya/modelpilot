# - WSL and Docker data on E:, since C: had 35 GB free.
- Code lives in ~/code inside WSL, not /mnt/c or /mnt/e: cross-system file access is slow.
- Python 3.12 via uv, one environment per repo.
- VS Code for these repos: its WSL support is the most reliable.
- NVIDIA free API first; Together ($15) only if NVIDIA can't cover tool calling or Week 3 load tests.
- RAM upgraded to 16 GB (2 × 8 GB); WSL memory raised to 8 GB, so Weeks 4–5 run locally without a cloud VM.