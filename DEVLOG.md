# ## 2026-10-23 · setup
Did: WSL2 + Ubuntu on E:, Docker on E:, VS Code connected, SSH key, three repos, modelpilot on Python 3.12, NVIDIA key in .env, RunPod account.
Blocked: Antigravity skipped for VS Code; Together needs $15 deposit, deferred.

## 2026-11-06 · LLMRouter comparison
Did: cloned LLMRouter to ~/code/LLMRouter, own uv env on 3.10, ran its benchmark pipeline on same split with MiniLM. Their routers mostly pick qwen2.5-7b, ties go to first model in the file (58.6% of labels). Fixed tie-break for knn/svm, they become always cogito, 0.81. Cheap end: their knn/svm at α=0.8 same score as my λ=10 (bootstrap, no real diff) but cheaper. Pushed.
Blocked: Qwen3 embeddings too slow on CPU (~4 h), cancelled. α sweep still running, final table pending.

## 2026-11-06 · learned router
Did: held out 20% of train as validation, tuned everything there (k=20, λ=1, MLP 256). On validation MLP lost to kNN so kNN stays default. Ran test once and MLP was better at every λ, +0.014 at λ=1 but 20% more cost. Didn't expect validation and test to disagree like that, 897 queries is probably too small. Kept kNN, didn't touch it after seeing test.
