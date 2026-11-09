#!/usr/bin/env bash
# Benchmark the gateway with GuideLLM, one run per setup given (small, large, auto).
#
#   scripts/live_benchmark.sh large        # before deploying the pod: needs only TOGETHER_API_KEY
#   scripts/live_benchmark.sh small auto   # during the GPU session
#
# Starts its own gateway (logging to results/v5_live/raw/gateway.jsonl), sends one request to each
# backend it needs and stops if any fails, then runs concurrency 1, 8, 32 for 60 s each.
# Override with STREAMS=1,4 SECONDS_PER_LEVEL=10 MAX_TOKENS=512 LIVE_OUT=... GATEWAY_PORT=...
set -euo pipefail
cd "$(dirname "$0")/.."

[ $# -gt 0 ] || { echo "usage: $0 small|large|auto [...]"; exit 1; }
OUT=${LIVE_OUT:-results/v5_live/raw}
PROMPTS=cache/live_prompts.jsonl
PORT=${GATEWAY_PORT:-8000}
STREAMS=${STREAMS:-1,8,32}
SECONDS_PER_LEVEL=${SECONDS_PER_LEVEL:-60}
MAX_TOKENS=${MAX_TOKENS:-512}

needed=()
for setup in "$@"; do
  case $setup in
    small|large) needed+=("$setup") ;;
    auto) needed+=(small large) ;;
    *) echo "unknown setup '$setup' (use small, large or auto)"; exit 1 ;;
  esac
done
command -v guidellm >/dev/null || { echo "guidellm not found: uv tool install guidellm==0.8.1"; exit 1; }
mkdir -p "$OUT"
[ -f "$PROMPTS" ] || uv run python scripts/live_prompts.py

export GATEWAY_LOG="$OUT/gateway.jsonl"
uv run uvicorn modelpilot.gateway:app --port "$PORT" --log-level warning >> "$OUT/gateway_stdout.log" 2>&1 &
GATEWAY_PID=$!
trap 'kill $GATEWAY_PID 2>/dev/null || true' EXIT
for _ in $(seq 120); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 1; done
curl -sf "localhost:$PORT/health" >/dev/null || { echo "gateway did not start, see $OUT/gateway_stdout.log"; exit 1; }

# Preflight: a broken .env should fail in seconds, not after minutes of paid GPU time.
for backend in $(printf '%s\n' "${needed[@]}" | sort -u); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "localhost:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$backend\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":5}")
  [ "$code" = 200 ] || { echo "preflight: $backend backend returned HTTP $code; check .env (see tail $GATEWAY_LOG)"; exit 1; }
  echo "preflight: $backend ok"
done

for setup in "$@"; do
  echo "== $setup: concurrency $STREAMS, ${SECONDS_PER_LEVEL}s each, max_tokens $MAX_TOKENS  ($(date +%T))"
  guidellm run \
    --backend "{\"kind\":\"openai_http\",\"target\":\"http://localhost:$PORT\",\"model\":\"$setup\",\"request_format\":\"/v1/chat/completions\",\"extras\":{\"body\":{\"max_tokens\":$MAX_TOKENS}}}" \
    --profile kind=concurrent --override profile.streams "$STREAMS" \
    --constraint "kind=max_duration,seconds=$SECONDS_PER_LEVEL" \
    --data "kind=json_file,path=$PROMPTS" \
    --output "kind=json,path=$OUT/guidellm_$setup.json" \
    --disable-console-interactive
done
echo "done ($(date +%T)). Summarise with: uv run python scripts/summarize_live.py --gpu-price-per-hour <\$/hr>"
