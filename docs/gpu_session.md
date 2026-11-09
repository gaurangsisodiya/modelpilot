# GPU session: Qwen2.5-7B-Instruct on RunPod with vLLM

This is how to run the gateway's **small** backend for real: `Qwen/Qwen2.5-7B-Instruct` served by vLLM on a rented 24 GB GPU.
You run these steps yourself; nothing here is automated. The clock is running (and billing) from **Deploy** until **Terminate**, so have the Phase 11 benchmark ready before you start.

Facts checked against the vLLM and RunPod docs on 2026-11-09. RunPod's web UI changes often; if a button has moved, look for the same setting nearby.

## 0. Before you deploy (no cost yet)

- [ ] `uv run pytest -q` passes, and the gateway works with the fake backend (see the README).
- [ ] `TOGETHER_API_KEY` is in `.env` if you want the routed and all-large runs (Phase 11).
- [ ] Generate a key that protects the vLLM server. The RunPod URL is public, so without a key anyone could use your GPU:
  ```bash
  openssl rand -hex 24
  ```
  Put it in `.env` as `SMALL_API_KEY=<that value>`. Never commit `.env`.
- [ ] Some RunPod credit loaded.

## 1. Deploy the pod

In the RunPod console: **Pods → Deploy**.

| Setting | Value |
|---|---|
| GPU | any **24 GB** card: RTX 4090, RTX 3090, RTX A5000, L4 or A10. Pick the cheapest one available. |
| GPU count | 1 |
| Template | start from any template, then **Edit Template** (or **Change Template → custom**) to set the fields below |
| Container image | `vllm/vllm-openai:v0.32.0` (released 2026-10-18; pin a version, don't use `latest`) |
| Container start command / Docker command | see below |
| Expose HTTP ports | `8000` |
| Expose TCP ports | none |
| Container disk | **50 GB** (the model download is ~15 GB) |
| Volume disk | **0 GB**, so nothing is left to bill after you terminate |
| Environment variables | optional `HF_TOKEN=<your Hugging Face token>` for faster downloads. Qwen2.5 is not gated, so it isn't required. |

**Container start command.** The vLLM image already starts the OpenAI server, so give **only the arguments**:

```
--model Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.90 --api-key <SMALL_API_KEY value>
```

What each flag does:

- `--model`: the Hugging Face model to download and serve. The served model name will be `Qwen/Qwen2.5-7B-Instruct`, which matches the gateway's default `SMALL_MODEL_ID`.
- `--max-model-len 8192`: the context limit (prompt plus output). The 7B weights take ~15 GB, so a smaller context leaves more of the 24 GB for the KV cache, i.e. more requests in parallel.
- `--gpu-memory-utilization 0.90`: use 90% of the GPU memory.
- `--api-key`: requests under `/v1` must send `Authorization: Bearer <key>`. `/health` stays open, so the health check needs no key.

📝 **Write down the price per hour** shown on the deploy screen (e.g. `$0.69/hr`) and the GPU type. Phase 11 needs both for the cost per 1k requests.

Click **Deploy**.

## 2. Wait for the server

1. Open the pod and go to **Logs**. You will see the model download, then the weights load, then the server start. Expect 3–10 minutes. "Running" in the pod list only means the container started, not that vLLM is ready.
2. It's ready when the logs show the API server listening on port 8000 (`Uvicorn running on http://0.0.0.0:8000` or `Application startup complete`).
3. If the logs say `unrecognized arguments` or show a usage message, the image expected a full command. Change the start command to `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 ...` (same flags) and restart the pod.

## 3. Connect: exposed HTTP port

Your server URL is

```
https://<POD_ID>-8000.proxy.runpod.net
```

where `<POD_ID>` is the ID shown on the pod card. RunPod puts it behind HTTPS for you.

Why not an SSH tunnel: the vLLM image has no SSH server, and RunPod's basic SSH proxy (`ssh.runpod.io`) does not document port forwarding. The HTTPS port plus `--api-key` is simpler, and nothing extra is installed.

Limit to know about: RunPod's proxy (Cloudflare) closes a connection with a **524 error if no response arrives within 100 seconds**. Streaming requests are fine because tokens keep flowing. A very long non-streaming answer under heavy load could hit it.

## 4. Health check (from your laptop)

```bash
POD=https://<POD_ID>-8000.proxy.runpod.net
KEY=<SMALL_API_KEY value>

curl -s -o /dev/null -w "%{http_code}\n" $POD/health                 # 200 = ready
curl -s $POD/v1/models -H "Authorization: Bearer $KEY"               # lists Qwen/Qwen2.5-7B-Instruct
curl -s $POD/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Say hi in 3 words."}],"max_tokens":10}'
```

`/v1/models` without the key should return **401**. That confirms the key works.

## 5. Point the gateway at the pod

In `.env`:

```
SMALL_BASE_URL=https://<POD_ID>-8000.proxy.runpod.net/v1
SMALL_API_KEY=<the same key>
TOGETHER_API_KEY=<your Together key>
```

Then:

```bash
uv run uvicorn modelpilot.gateway:app --port 8000
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"small","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":10}'
tail -1 logs/gateway.jsonl          # backend "small", status 200, tokens, latency
```

Now run the Phase 11 benchmarks in this same session.

## 6. STOP: terminate the pod

Do this as soon as the benchmarks finish. An idle pod bills exactly like a busy one.

1. RunPod console → **Pods** → your pod → **Stop**, then **Terminate** (the trash-can icon).
   - **Stop** releases the GPU, but anything on a volume disk keeps billing. Container disk data is wiped anyway.
   - **Terminate** deletes the pod and everything on it, and billing ends. With no volume disk there is nothing worth keeping, so **terminate**.
2. Or from the command line, if you use `runpodctl` (set up once with `runpodctl config --apiKey <RUNPOD_API_KEY>`):
   ```bash
   runpodctl pod list            # find the pod ID
   runpodctl pod stop <POD_ID>
   runpodctl pod delete <POD_ID>
   runpodctl pod list --all      # the pod should be gone
   ```
3. Check: the pod is gone from **Pods**, and the billing page shows no running pods.
4. In `.env`, set `SMALL_BASE_URL` back to `http://localhost:8001/v1` (the fake backend), so the gateway doesn't point at a dead URL.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `/health` returns 502 or 404 for a while | vLLM is still downloading or loading. Watch **Logs**. |
| CUDA out of memory in the logs | Lower `--max-model-len` to 4096, or `--gpu-memory-utilization` to 0.85. |
| 401 from `/v1/...` | The key in `.env` differs from `--api-key`, or the `Bearer ` prefix is missing. |
| 524 from the proxy | No response within 100 s: use `"stream": true`, or lower the concurrency. |
| Gateway logs `status 502` for small | `SMALL_BASE_URL` is wrong or the pod is stopped. It must end in `/v1`. |
