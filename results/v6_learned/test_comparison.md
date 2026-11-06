Test split (3729 queries), run once with settings fixed on validation; default λ = 1.

| λ | kNN k=20 avg score | kNN k=20 cost ($) | MLP hidden=256 avg score | MLP hidden=256 cost ($) |
|---:|---:|---:|---:|---:|
| 0 | 0.748 | 0.806 | 0.781 | 1.137 |
| 0.1 | 0.749 | 0.663 | 0.757 | 0.473 |
| 0.3 | 0.740 | 0.316 | 0.752 | 0.215 |
| 1 | 0.732 | 0.124 | 0.746 | 0.149 |
| 3 | 0.725 | 0.085 | 0.739 | 0.082 |
| 10 | 0.710 | 0.054 | 0.719 | 0.063 |

Paired bootstrap at λ = 1 (MLP − kNN avg score, 1,000 resamples): +0.0141 [+0.0034, +0.0243]; MLP cost / kNN cost = 1.20.
Default router chosen on validation: knn.
