Paired bootstrap, 1000 resamples over 3729 test queries. Difference = A − B in avg score.

| A | B | Avg score A | Avg score B | Difference (95% CI) | Cost A ($) | Cost B ($) |
|---|---|---:|---:|---|---:|---:|
| LLMRouter knnrouter α=0.8 | ModelPilot kNN λ=10 | 0.712 | 0.710 | +0.0014 [-0.0072, +0.0098] | 0.052 | 0.058 |
| LLMRouter knnrouter α=0.8 | always-cheapest (gpt-oss-20b) | 0.712 | 0.698 | +0.0143 [+0.0073, +0.0214] | 0.052 | 0.054 |
| LLMRouter svmrouter α=0.8 | ModelPilot kNN λ=10 | 0.713 | 0.710 | +0.0027 [-0.0066, +0.0117] | 0.044 | 0.058 |
| LLMRouter svmrouter α=0.8 | always-cheapest (gpt-oss-20b) | 0.713 | 0.698 | +0.0156 [+0.0099, +0.0215] | 0.044 | 0.054 |
