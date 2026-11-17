# ModelPilot terminal demo, about 90 s. Render from the repo root with:  vhs demo/modelpilot.tape
#
# Makes NO paid calls: the gateway runs against the keyless fake backend (README "Gateway without
# a GPU", both URLs set on the command line), and every table is printed from committed results/
# files by demo/show.py. Needs vhs, ttyd and ffmpeg on PATH, `uv sync` done, the embedding cache
# (cache/, built by the first pipeline run) for a fast gateway start, and ports 8000 and 8001 free.
# The gateway's log goes to a temp file, not logs/gateway.jsonl.
#
# If VHS's bundled Chromium fails to start (missing libnss3/libnspr4/libasound), point it at
# locally unpacked copies of those libraries, e.g.
#   LD_LIBRARY_PATH=~/.local/lib/chromium-deps/usr/lib/x86_64-linux-gnu vhs demo/modelpilot.tape

Output demo/modelpilot.gif
Output demo/modelpilot.mp4

Require uv
Require curl
Require python3

Set Shell "bash"
Set Theme "Catppuccin Mocha"
Set FontSize 16
Set Width 1200
Set Height 700
Set Padding 20
Set Framerate 10
Set TypingSpeed 12ms

Hide
Type "export PS1='$ ' GATEWAY_LOG=$(mktemp -d)/gateway.jsonl && cd $(git rev-parse --show-toplevel) && clear" Enter
Sleep 500ms
Show

# a. What it is
Type "# ModelPilot: an LLM router. For each query, pick the cheapest model that is likely to answer it well." Enter
Type "# kNN over MiniLM embeddings of xRouteBench (18 models, ~8,200 scored queries), plus a routing gateway." Enter
Sleep 2s

# b. The keyless gateway: fake backend for both models, one routed request each way
Type "# Gateway with a fake backend for both models: no keys, no GPU, no API calls" Enter
Type "FAKE_DELAY_SECONDS=0.5 uv run uvicorn modelpilot.fake_backend:app --port 8001 >/dev/null 2>&1 &" Enter
Type `SMALL_BASE_URL=http://localhost:8001/v1 LARGE_BASE_URL=http://localhost:8001/v1 \` Enter
Type "  uv run uvicorn modelpilot.gateway:app --port 8000 >/dev/null 2>&1 &" Enter
Type "# (startup loads the router and its embeddings, about 15 s: cut)" Enter
Hide
Type `until curl -sf localhost:8000/health >/dev/null; do sleep 1; done; printf '\e[1A\e[2K'` Enter
Wait+Line@180s /^\$$/
Show
Type `curl -si localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d \` Enter
Type `  '{"model":"auto","messages":[{"role":"user","content":"What is 17 * 23?"}]}' \` Enter
Type `  | grep backend` Enter
Wait+Line@30s /^\$$/
Type `curl -si localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d \` Enter
Type `  '{"model":"auto","messages":[{"role":"user","content":"Write a Python palindrome checker."}]}' \` Enter
Type `  | grep backend` Enter
Wait+Line@30s /^\$$/
Type "tail -2 $GATEWAY_LOG     # one JSON line per request: backend, λ, route time, tokens, cost" Enter
Sleep 5s
Type "kill %1 %2; wait     # stop both servers" Enter
Wait+Line@30s /^\$$/
Sleep 1500ms

# c. Offline router results
Hide
Type "clear" Enter
Show
Type "# Offline, 3,729 test queries (committed results): the final router (kNN, k = 20) at λ = 1 scores" Enter
Type "# 0.732 for $0.124, against 0.805 for $1.707 for the best single model: 93% cheaper for 0.07 less." Enter
Type "python3 demo/show.py results/baselines.csv" Enter
Wait+Line@30s /^\$$/
Type "python3 demo/show.py results/v6_learned/test_comparison.md" Enter
Wait+Line@30s /^\$$/
Sleep 9s

# d. Live GPU benchmark
Hide
Type "clear" Enter
Show
Type "# Live: GuideLLM through the gateway. Small = Qwen2.5-7B on vLLM (RTX 3090), large = gpt-oss-120b (Together)" Enter
Type `python3 demo/show.py results/v5_live/live_results.md --table \` Enter
Type `  --columns 'Setup,Conc.,Req/s,Route ms (median),Share large,Live cost / 1k ($),Expected avg score,Errors'` Enter
Wait+Line@30s /^\$$/
Sleep 6s
Type "# Under load, routing pays: at concurrency 32 it keeps all-large's expected score (0.743 vs 0.744) at 17%" Enter
Type "# lower live cost. At concurrency 1 the idle GPU makes routed the most expensive setup ($0.704 per 1k)." Enter
Sleep 5s

# e. The surprise
Type "# Surprise: gpt-oss-120b cost 3.3x the dataset's prices live (reasoning tokens bill as output), so the router" Enter
Type "# treats large as ~3x cheaper than it is, and 13% of its replies were empty at a 512-token cap." Enter
Sleep 6s
