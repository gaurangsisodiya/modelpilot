Live benchmark through the gateway (RTX 3090 at $0.5/h). Expected avg score: offline, full test split, two models only.
Token cost = xRouteBench prices for both models; live cost = Together tokens + GPU time for the window.

| Setup | Conc. | Req/s | Out tok/s | TTFT med / p95 (ms) | Latency med / p95 (s) | Route ms (median) | Share large | Token cost / 1k ($) | Live cost / 1k ($) | Expected avg score | Errors |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| all-small | 1 | 0.63 | 38 | 284 / 935 | 0.40 / 7.63 | 0 | 0% | 0.046 | 0.219 | 0.575 | 0 |
| all-small | 8 | 5.00 | 251 | 375 / 1001 | 0.51 / 8.08 | 0 | 0% | 0.043 | 0.027 | 0.575 | 0 |
| all-small | 32 | 15.33 | 703 | 582 / 1290 | 0.79 / 9.51 | 0 | 0% | 0.041 | 0.009 | 0.575 | 1 |
| all-large | 1 | 0.27 | 80 | 451 / 4081 | 3.31 / 7.15 | 0 | 100% | 0.200 | 0.200 | 0.744 | 0 |
| all-large | 8 | 2.40 | 622 | 437 / 1452 | 2.84 / 6.34 | 0 | 100% | 0.176 | 0.176 | 0.744 | 4 |
| all-large | 32 | 7.90 | 1914 | 1083 / 2169 | 3.23 / 7.52 | 0 | 100% | 0.164 | 0.164 | 0.744 | 0 |
| routed λ=1 | 1 | 0.22 | 50 | 604 / 4717 | 3.67 / 11.10 | 67 | 71% | 0.131 | 0.704 | 0.743 | 0 |
| routed λ=1 | 8 | 2.23 | 451 | 488 / 1098 | 2.60 / 7.63 | 71 | 73% | 0.127 | 0.170 | 0.743 | 0 |
| routed λ=1 | 32 | 8.00 | 1657 | 805 / 1609 | 2.91 / 8.03 | 143 | 77% | 0.132 | 0.136 | 0.743 | 0 |

## GPU session cost (RunPod, 2026-11-09)

| Pod | GPU | $/h | Running (deploy to delete, UTC) | Hours | Cost ($) |
|---|---|---:|---|---:|---:|
| 1 (aborted before benchmarking, my .env edit broke a key) | RTX 4090 | 0.89 | 18:00:05 to 18:10:58 | 0.181 | 0.161 |
| 2 (the runs above) | RTX 3090 | 0.50 | 18:16:24 to 18:26:02 | 0.161 | 0.080 |
| **Total** | | | | **0.342** | **0.242** |

Pod 2 alone (the session that produced these numbers): 0.161 h x $0.50 = $0.080. Live cost / 1k above counts only each 60 s window, not model load time.
