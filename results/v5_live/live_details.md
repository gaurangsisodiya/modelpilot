# Live run details (results/v5_live)

## Errored requests

| Setup | Conc. | HTTP | Error (GuideLLM) | Failed after (s) |
|---|---:|---:|---|---:|
| all-small | 32 | 502 | Server error '502 Bad Gateway' | 0.17 |
| all-large | 8 | 503 | Server error '503 Service Unavailable' | 1.83 |
| all-large | 8 | 503 | Server error '503 Service Unavailable' | 1.83 |
| all-large | 8 | 503 | Server error '503 Service Unavailable' | 1.84 |
| all-large | 8 | 503 | Server error '503 Service Unavailable' | 1.84 |

Notes on errors: the gateway passes a backend's non-200 status straight through, and makes its own 502 only
when it can't reach the backend. Neither GuideLLM nor the gateway log saved the response body, so the 502
could be the gateway's 'backend unreachable' or a 502 from RunPod's proxy; the log can't tell them apart.

## Empty replies (successful requests whose answer content is empty)

| Setup | Conc. | Successful | Empty | Empty share | Empty and at 512-token cap |
|---|---:|---:|---:|---:|---:|
| all-small | 1 | 38 | 0 | 0.0% | 0 |
| all-small | 8 | 300 | 0 | 0.0% | 0 |
| all-small | 32 | 920 | 0 | 0.0% | 0 |
| all-large | 1 | 17 | 4 | 23.5% | 4 |
| all-large | 8 | 145 | 17 | 11.7% | 17 |
| all-large | 32 | 474 | 63 | 13.3% | 63 |
| routed | 1 | 13 | 2 | 15.4% | 2 |
| routed | 8 | 134 | 11 | 8.2% | 11 |
| routed | 32 | 480 | 47 | 9.8% | 47 |

## Live token cost vs offline cost per 1k requests, per backend

Both use xRouteBench prices; only the token counts differ. Live = gateway log (status 200, all setups,
max_tokens 512). Offline = xRouteBench test split. "Same queries" = offline cost of exactly the prompts
sent in that backend's single-backend run (all-small / all-large), matched by prompt text.

| Backend | Model | Live requests | Live in / out tok (mean) | Live $/1k | Offline in / out tok (mean, full test) | Offline $/1k (full test) | Offline $/1k (same queries) | Live / offline (same queries) |
|---|---|---:|---|---:|---|---:|---:|---:|
| small | qwen2.5-7b-instruct | 1462 | 155 / 61 | 0.043 | 128 / 68 | 0.039 | 0.038 (1258/1258 matched) | 1.14x |
| large | gpt-oss-120b | 1185 | 190 / 222 | 0.162 | 128 / 56 | 0.053 | 0.049 (636/636 matched) | 3.32x |
