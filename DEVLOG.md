# ## 2026-10-23 · setup
Did: WSL2 + Ubuntu on E:, Docker on E:, VS Code connected, SSH key, three repos, modelpilot on Python 3.12, NVIDIA key in .env, RunPod account.
Blocked: Antigravity skipped for VS Code; Together needs $15 deposit, deferred.

## 2026-11-06 · LLMRouter comparison
Did: cloned LLMRouter to ~/code/LLMRouter, own uv env on 3.10, ran its benchmark pipeline on same split with MiniLM. Their routers mostly pick qwen2.5-7b, ties go to first model in the file (58.6% of labels). Fixed tie-break for knn/svm, they become always cogito, 0.81. Cheap end: their knn/svm at α=0.8 same score as my λ=10 (bootstrap, no real diff) but cheaper. Pushed.
Blocked: Qwen3 embeddings too slow on CPU (~4 h), cancelled. α sweep still running, final table pending.

## 2026-11-06 · learned router
Did: held out 20% of train as validation, tuned everything there (k=20, λ=1, MLP 256). On validation MLP lost to kNN so kNN stays default. Ran test once and MLP was better at every λ, +0.014 at λ=1 but 20% more cost. Didn't expect validation and test to disagree like that, 897 queries is probably too small. Kept kNN, didn't touch it after seeing test.

## 2026-11-09 · gateway
Did: OpenAI-compatible gateway, model auto/small/large, streaming passthrough, fake backend with fixed delay, JSON log per request with latency, tokens, cost. Tests run the fake in-process, no ports. Checked Together before relying on it, Llama-3.3-70B Turbo got removed from serverless on 10-22, switched large to gpt-oss-120b. Surprise: by dataset price gpt-oss-120b is cheaper than qwen-7b on 61% of queries, so router sends ~70% to large. Real trade-off is going to be GPU hour cost for qwen, not tokens.
Blocked: no Together key yet, couldn't hit /v1/models myself.

## 2026-11-09 · GPU guide
Did: wrote docs/gpu_session.md for RunPod + vLLM (qwen 7B, 24 GB card, vllm v0.32.0 pinned). Went with RunPod https proxy url not ssh tunnel, vllm image has no sshd. API key is the main thing, url is public so without --api-key anyone can use my GPU. Added checks: key set in .env before deploy, 401 without key / 200 with key before gateway, find Terminate button before starting, check balance after. Terminate not stop, no volume disk.
Blocked: SMALL_API_KEY not in .env yet, haven't deployed.

## 2026-11-09 · live benchmark scripts
Did: guidellm 0.8.1 scripts for all-small / all-large / routed through the gateway, real test prompts, max_tokens 512, concurrency 1/8/32 x 60s. Docs were vague so installed it and tested the CLI against the fake backend first. Two cost columns, token price vs what I actually pay (Together + GPU $/hr for the window). Preflight hits each backend once so a bad .env fails before the GPU minutes burn. All-large runs before deploy, GPU only for small + auto, ~20-25 min.
Surprise: routing itself is the bottleneck on my laptop, embedding caps at ~23 req/s, torch thread limits didn't help. Added a route ms column instead of fixing it.
Blocked: no keys in .env yet, live numbers wait for the pod session.

## 2026-11-09 · all-large live run
Did: ran all-large through the gateway, conc 1/8/32 x 60s, 636 ok. 7.9 req/s at 32, ttft median ~0.45s at 1-8, ~1.1s at 32.
Finding: the 4 errors at conc 8 were all HTTP 503 from Together, passed through by the gateway, all in the same 1.8s at the start of the run when 8 requests hit at once. None after that, none at 32. Gateway doesn't save the error body so I can't tell if it was a rate limit or capacity, should log it.
Finding: gpt-oss-120b reasons before it answers. 111/636 replies (17.5%) hit the 512 cap, and 84 of those (13.2% of all) came back with an empty answer because reasoning used the whole budget. Every empty answer was a capped one. So the 512 cap costs score on large, and the offline expected avg score won't match live for it.

## 2026-11-09 · GPU session
Did: ran the GPU session through the RunPod API, no browser. qwen 7B on vLLM, 3090 at $0.50/h, ready in ~6 min, 401/200 key check passed, small + routed at 1/8/32, pod deleted 13s after the bench finished. Total GPU $0.24.
Mistake: first pod (4090, $0.89/h) got killed before benchmarking. .env had no newline at the end, so appending SMALL_BASE_URL glued it onto RUNPOD_API_KEY, the first delete came back 401. Restored .env from the backup, deleted, redeployed. Cost $0.16 for nothing. Now I check the last byte before appending to .env.
Finding: live, gpt-oss-120b costs 3.3x what xRouteBench says on the same queries (qwen 1.1x). Reasoning tokens bill as output, ~222 out vs 56 in the dataset. So the router picked large at λ=1 with costs ~3x too low, that's probably why ~75% went to large. Under load routed still beats all-large, $0.136 vs $0.164 per 1k at conc 32, same expected score. At conc 1 the idle GPU makes routed the most expensive.
