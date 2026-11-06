# ## 2026-10-23 · setup
Did: WSL2 + Ubuntu on E:, Docker on E:, VS Code connected, SSH key, three repos, modelpilot on Python 3.12, NVIDIA key in .env, RunPod account.
Blocked: Antigravity skipped for VS Code; Together needs $15 deposit, deferred.

## 2026-11-06 · LLMRouter comparison (Phase 7, in progress)
Did: ran LLMRouter's benchmark_pipeline on the same split with MiniLM embeddings. Found that its default labels give tied queries to whichever model is listed first (qwen2.5-7b, 58.6% of labels), which explains its weak pure-performance scores; fixing the tie-break turns knn/svm into "always cogito". At the cheap end its cost-aware knn/svm (α=0.8) tie my λ=10 on score (bootstrap CI spans 0) for 10–24% less cost, so I'm not ahead there.
Blocked: Qwen3 embeddings cancelled (~4 h on CPU); LLMRouter α sweep still running (~3 h), so the final table and frontier check are pending.
