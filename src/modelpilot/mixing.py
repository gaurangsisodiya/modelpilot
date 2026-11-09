"""Random-mixing baseline: send a random share of queries to the best model, the rest to the cheapest.

A router is only useful if it beats this: mixing two fixed models at random already buys any
total cost between always-cheapest and always-best, with no knowledge of the queries.
"""

import numpy as np


def mix_fraction(target_cost: float, cheap_cost: float, best_cost: float) -> float:
    """Share p of queries sent to the best model so the mix's total cost equals `target_cost`.

    Mix cost = (1 - p) * cheap_cost + p * best_cost, all as totals over the same queries.
    Clipped to [0, 1]: a target below always-cheapest gives 0 (the mix then costs more than
    the target), a target above always-best gives 1.
    """
    if best_cost <= cheap_cost:
        raise ValueError("best_cost must be higher than cheap_cost")
    return float(np.clip((target_cost - cheap_cost) / (best_cost - cheap_cost), 0.0, 1.0))


def mixed_scores(cheap_scores: np.ndarray, best_scores: np.ndarray, p: float) -> np.ndarray:
    """Per-query expected score of the random mix: (1 - p) * cheap + p * best.

    This is the average over every random assignment, so the comparison has no extra
    randomness from one particular draw; it pairs with the router query by query.
    """
    cheap, best = np.asarray(cheap_scores, dtype=float), np.asarray(best_scores, dtype=float)
    if cheap.shape != best.shape:
        raise ValueError("cheap_scores and best_scores must have the same length")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")
    return (1.0 - p) * cheap + p * best
